from baselines.common import Dataset, explained_variance, zipsame
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from baselines.ppo1.gcn_policy import discriminator_net, GCNPolicy
import copy


def trajectory_segment_generator(args, pi, env, horizon, stochastic, d_step_func, d_final_func):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_env = 0
    cur_ep_ret_d_step = 0
    cur_ep_ret_d_final = 0
    cur_ep_len = 0 # len of current episode
    cur_ep_len_valid = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_d_step = []
    ep_rets_d_final = []
    ep_rets_env = []
    ep_lens = [] # lengths of ...
    ep_lens_valid = [] # lengths of ...
    ep_rew_final = []
    ep_rew_final_stat = []

    # Initialize history arrays
    ob_adjs = np.array([ob_adj for _ in range(horizon)])
    ob_nodes = np.array([ob_node for _ in range(horizon)])
    ob_adjs_final = []
    ob_nodes_final = []
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, debug = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob_adj" : ob_adjs,
                   "ob_node" : ob_nodes,
                   "ob_adj_final" : np.array(ob_adjs_final),
                   "ob_node_final" : np.array(ob_nodes_final),
                   "rew" : rews,
                   "vpred" : vpreds,
                   "new" : news,
                   "ac" : acs,
                   "prevac" : prevacs,
                   "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets,
                   "ep_lens" : ep_lens,
                   "ep_lens_valid" : ep_lens_valid,
                   "ep_final_rew":ep_rew_final,
                   "ep_final_rew_stat":ep_rew_final_stat,
                   "ep_rets_env" : ep_rets_env,
                   "ep_rets_d_step" : ep_rets_d_step,
                   "ep_rets_d_final" : ep_rets_d_final}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_lens_valid = []
            ep_rew_final = []
            ep_rew_final_stat = []
            ep_rets_d_step = []
            ep_rets_d_final = []
            ep_rets_env = []
            ob_adjs_final = []
            ob_nodes_final = []

        i = t % horizon
        ob_adjs[i] = ob['adj']
        ob_nodes[i] = ob['node']
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew_env, new, info = env.step(ac)
        rew_d_step = 0 # default
        if rew_env>0: # if action valid
            cur_ep_len_valid += 1
            # add stepwise discriminator reward
            if args['has_d_step']==1:
                if args['gan_type']=='normal':
                    rew_d_step = args['gan_step_ratio'] * (
                        d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])) / env.max_atom
                elif args['gan_type'] == 'recommend':
                    rew_d_step = args['gan_step_ratio'] * (
                        max(1-d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),-2)) / env.max_atom
        rew_d_final = 0 # default
        if new:
            if args['has_d_final']==1:
                if args['gan_type'] == 'normal':
                    rew_d_final = args['gan_final_ratio'] * (
                        d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]))
                elif args['gan_type'] == 'recommend':
                    rew_d_final = args['gan_final_ratio'] * (
                        max(1 - d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),
                            -2))

        rews[i] = rew_d_step + rew_env + rew_d_final

        cur_ep_ret += rews[i]
        cur_ep_ret_d_step += rew_d_step
        cur_ep_ret_d_final += rew_d_final
        cur_ep_ret_env += rew_env
        cur_ep_len += 1

        if new:
            ob_adjs_final.append(ob['adj'])
            ob_nodes_final.append(ob['node'])
            ep_rets.append(cur_ep_ret)
            ep_rets_env.append(cur_ep_ret_env)
            ep_rets_d_step.append(cur_ep_ret_d_step)
            ep_rets_d_final.append(cur_ep_ret_d_final)
            ep_lens.append(cur_ep_len)
            ep_lens_valid.append(cur_ep_len_valid)
            ep_rew_final.append(rew_env)
            ep_rew_final_stat.append(info['final_stat'])
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_len_valid = 0
            cur_ep_ret_d_step = 0
            cur_ep_ret_d_final = 0
            cur_ep_ret_env = 0
            ob = env.reset()

        t += 1

def traj_final_generator(pi, env, batch_size, stochastic):
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']
    ob_adjs = np.array([ob_adj for _ in range(batch_size)])
    ob_nodes = np.array([ob_node for _ in range(batch_size)])
    for i in range(batch_size):
        ob = env.reset()
        while True:
            ac, vpred, debug = pi.act(stochastic, ob)
            ob, rew_env, new, info = env.step(ac)
            np.set_printoptions(precision=2, linewidth=200)
            if new:
                ob_adjs[i]=ob['adj']
                ob_nodes[i]=ob['node']
                break
    return ob_adjs,ob_nodes

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(args, env, evaluator, horizon, max_time_steps=0,
          max_episodes=0, max_iters=0, max_seconds=0,
          init_lr=0.001, clip_param=0.2, entropy_coef=0.01, optim_epochs=8,
          optim_batchsize=32, gamma=1, lam=0.95, adam_epsilon=1e-5,
          schedule='linear', writer=None):
    """
    Parameters
    ----------
    args : dict
    env : MoleculeEnv
    evaluator : Evaluator
    max_time_steps : int
    horizon : int
    max_episodes : int
    max_iters : int
    max_seconds : int
    init_lr : float
        Initial learning rate
    clip_param : float
        Hyperparameter for PPO objective
    entropy_coef : float
        Weight for entropy objective
    optim_epochs : int
    gamma : int
    lam : float
    adam_epislon : float
        Hyperparameter for Adam to stabilize the learning
    schedule : str
    writer : tensorboardX.writer.SummaryWriter
    """
    ob_space = env.observation_space
    ac_space = env.action_space

    # Initialize models
    pi = GCNPolicy(name="pi",
                   ob_space=ob_space,
                   ac_space=ac_space,
                   atom_type_num=env.atom_type_num,
                   args=args)

    if args['rl']:
        old_pi = GCNPolicy(name="old_pi",
                           ob_space=ob_space,
                           ac_space=ac_space,
                           atom_type_num=env.atom_type_num,
                           args=args)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = {}
    ob['adj'] = U.get_placeholder_cached(name="adj")
    ob['node'] = U.get_placeholder_cached(name="node")

    if args['has_d_step'] or args['has_d_final']:
        ob_gen = {}
        ob_gen['adj'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,
                                          name='adj_gen')
        ob_gen['node'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,
                                           name='node_gen')

        ob_real = {}
        ob_real['adj'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,
                                           name='adj_real')
        ob_real['node'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,
                                            name='node_real')

    ac = tf.placeholder(dtype=tf.int64, shape=[None,4],name='ac_real')
    pi_logp = pi.pd.logp(ac)

    ## PPO loss
    if args['rl']:
        clip_param = clip_param * lrmult  # Annealed cliping parameter epislon
        kloldnew = old_pi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entropy_coef) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - old_pi.pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]

    ## Expert loss
    loss_expert = - tf.reduce_mean(pi_logp)

    if args['has_d_step']:
        step_pred_real, step_logit_real = discriminator_net(ob_real, args, name='d_step')
        step_pred_gen, step_logit_gen = discriminator_net(ob_gen, args, name='d_step')
        loss_d_step_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real, labels=tf.ones_like(step_logit_real)*0.9))
        loss_d_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
        loss_d_step = loss_d_step_real+loss_d_step_gen

        if args['gan_type'] == 'normal':
            loss_g_step_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
        elif args['gan_type'] == 'recommend':
            loss_g_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen,
                                                                                     labels=tf.ones_like(
                                                                                         step_logit_gen) * 0.9))

    if args['has_d_final']:
        final_pred_real, final_logit_real = discriminator_net(ob_real, args, name='d_final')
        final_pred_gen, final_logit_gen = discriminator_net(ob_gen, args, name='d_final')
        loss_d_final_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_real,
                                                                                   labels=tf.ones_like(
                                                                                       final_logit_real) * 0.9))
        loss_d_final_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
        loss_d_final = loss_d_final_real + loss_d_final_gen

        if args['gan_type'] == 'normal':
            loss_g_final_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
        elif args['gan_type'] == 'recommend':
            loss_g_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen,
                                                                                      labels=tf.ones_like(
                                                                                          final_logit_gen) * 0.9))

    var_list_pi = pi.get_trainable_variables()

    if args['has_d_step']:
        var_list_d_step = [var for var in tf.global_variables() if 'd_step' in var.name]
    if args['has_d_final']:
        var_list_d_final = [var for var in tf.global_variables() if 'd_final' in var.name]

    ## loss update function
    if args['rl']:
        lossandgrad_ppo = U.function([ob['adj'], ob['node'], ac, pi.ac_real, old_pi.ac_real, atarg, ret, lrmult],
                                     losses + [U.flatgrad(total_loss, var_list_pi)])

    lossandgrad_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi)])

    if args['has_d_step']:
        lossandgrad_d_step = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']], [loss_d_step, U.flatgrad(loss_d_step, var_list_d_step)])
        loss_g_gen_step_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_step_gen)
    if args['has_d_final']:
        lossandgrad_d_final = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']],
                                         [loss_d_final, U.flatgrad(loss_d_final, var_list_d_final)])
        loss_g_gen_final_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_final_gen)

    adam_pi = MpiAdam(var_list_pi, epsilon=adam_epsilon)
    if args['has_d_step']:
        adam_d_step = MpiAdam(var_list_d_step, epsilon=adam_epsilon)
    if args['has_d_final']:
        adam_d_final = MpiAdam(var_list_d_final, epsilon=adam_epsilon)

    if args['rl']:
        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(old_pi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob['adj'], ob['node'], ac, pi.ac_real, old_pi.ac_real, atarg, ret, lrmult], losses)

    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_step = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_final = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final_stat = deque(maxlen=100) # rolling buffer for episode rewardsn

    if args['rl']:
        seg_gen = trajectory_segment_generator(args, pi, env, horizon, True, loss_g_gen_step_func, loss_g_gen_final_func)

    assert sum([max_iters>0, max_time_steps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    if args['load'] == 1:
        try:
            fname = './ckpt/' + args['name_full_load']
            sess = tf.get_default_session()
            saver = tf.train.Saver(var_list_pi)
            saver.restore(sess, fname)
            iters_so_far = int(fname.split('_')[-1])+1
            print('model restored!', fname, 'iters_so_far:', iters_so_far)
        except:
            print(fname,'ckpt not found, start with iters 0')

    U.initialize()
    adam_pi.sync()
    adam_d_step.sync()
    adam_d_final.sync()

    checkpoint_path = './ckpt/' + args['name_full']
    def checkpoint():
        saver = tf.train.Saver(var_list_pi)
        saver.save(tf.get_default_session(), checkpoint_path)
        print('model saved!', checkpoint_path)

    if evaluator is not None:
        checkpoint()
        evaluator(pi, n_samples=1024, checkpoint_path=checkpoint_path)

    best_loss = float('inf')
    n_patient_rounds = 0
    last_evaluation_time = time.time()
    level = 0

    if args['rl']:
        check_interval = 1800
    else:
        check_interval = 300

    ## start training
    while True:
        if max_time_steps and timesteps_so_far >= max_time_steps:
            return pi, var_list_pi
        elif max_episodes and episodes_so_far >= max_episodes:
            return pi, var_list_pi
        elif max_iters and iters_so_far >= max_iters:
            return pi, var_list_pi
        elif max_seconds and time.time() - tstart >= max_seconds:
            return pi, var_list_pi

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            if args['rl']:
                cur_lrmult = max(1.0 - float(timesteps_so_far) / max_time_steps, 0)
            else:
                cur_lrmult =  max(1.0 - float(iters_so_far) / max_time_steps, 0)
        else:
            raise NotImplementedError

        if args['rl']:
            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            ob_adj, ob_node, ac, atarg, tdlamret = seg["ob_adj"], seg["ob_node"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node, ac=ac, atarg=atarg, vtarg=tdlamret),
                        shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob_adj.shape[0]

        # inner training loop, train policy
        all_pocliy_loss = []
        all_teacher_forcing_loss = []

        if args['rl']:
            all_total_rl_loss = []
            all_ppo_surrogate = []
            all_entropy = []
            all_vpred_loss = []
            if args['has_d_step']:
                all_loss_d_step = []
            if args['has_d_final']:
                all_loss_d_final = []

        for i_optim in range(optim_epochs):
            policy_loss = 0
            loss_expert = 0
            g_expert = 0

            if args['rl']:
                g_ppo = 0
                if args['has_d_step']:
                    loss_d_step = 0
                    g_d_step = 0
                if args['has_d_final']:
                    loss_d_final = 0
                    g_d_final = 0

            pretrain_shift = 5
            ## Expert
            if iters_so_far >= args['expert_start'] and iters_so_far <= args['expert_end'] + pretrain_shift:
                ## Expert train
                # # # learn how to stop
                ob_expert, ac_expert = env.get_expert(optim_batchsize)
                loss_expert, g_expert = lossandgrad_expert(ob_expert['adj'], ob_expert['node'], ac_expert, ac_expert)
                loss_expert = np.mean(loss_expert)
                policy_loss += loss_expert
                all_teacher_forcing_loss.append(loss_expert)

            if args['rl']:
                if iters_so_far >= args['rl_start'] and iters_so_far <= args['rl_end']:
                    assign_old_eq_new()  # set old parameter values to new parameter values
                    batch = d.next_batch(optim_batchsize)
                    # ppo
                    if iters_so_far >= args['rl_start'] + pretrain_shift:  # start generator after discriminator trained a well..
                        *newlosses, g_ppo = lossandgrad_ppo(batch["ob_adj"], batch["ob_node"], batch["ac"], batch["ac"],
                                                            batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                        pol_surr, pol_entpen, vf_loss, meankl, meanent = newlosses
                        total_loss = pol_surr + pol_entpen + vf_loss
                        policy_loss += total_loss
                        all_total_rl_loss.append(total_loss)
                        all_ppo_surrogate.append(pol_surr)
                        all_entropy.append(meanent)
                        all_vpred_loss.append(vf_loss)

                    if args['has_d_step'] == 1 and i_optim >= optim_epochs // 2:
                        # update step discriminator
                        ob_expert, _ = env.get_expert(optim_batchsize, curriculum=args['curriculum'],
                                                      level_total=args['curriculum_num'], level=level)
                        loss_d_step, g_d_step = lossandgrad_d_step(ob_expert["adj"], ob_expert["node"], batch["ob_adj"],
                                                                   batch["ob_node"])
                        adam_d_step.update(g_d_step, init_lr * cur_lrmult)
                        loss_d_step = np.mean(loss_d_step)
                        all_loss_d_step.append(loss_d_step)

                    if args['has_d_final'] == 1 and i_optim >= optim_epochs // 4 * 3:
                        # update final discriminator
                        ob_expert, _ = env.get_expert(optim_batchsize, is_final=True, curriculum=args['curriculum'],
                                                      level_total=args['curriculum_num'], level=level)
                        seg_final_adj, seg_final_node = traj_final_generator(pi, copy.deepcopy(env), optim_batchsize,
                                                                             True)
                        # update final discriminator
                        loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"],
                                                                      seg_final_adj, seg_final_node)
                        adam_d_final.update(g_d_final, init_lr * cur_lrmult)
                        loss_d_final = np.mean(loss_d_final)
                        all_loss_d_final.append(loss_d_final)

            # update generator
            if args['rl']:
                adam_pi.update(0.2 * g_ppo + 0.05 * g_expert, init_lr * cur_lrmult)
            else:
                adam_pi.update(0.25 * g_expert, init_lr * cur_lrmult)
            all_pocliy_loss.append(policy_loss)
        mean_policy_loss = np.mean(all_pocliy_loss)
        mean_expert_loss = np.mean(all_teacher_forcing_loss)
        mean_total_rl_loss = np.mean(all_total_rl_loss)
        mean_ppo_surrogate = np.mean(all_ppo_surrogate)
        mean_entropy = np.mean(all_entropy)
        mean_vf_loss = np.mean(all_vpred_loss)
        mean_d_step_loss = np.mean(all_loss_d_step)
        mean_d_final_loss = np.mean(all_loss_d_final)

        if writer is not None:
            writer.add_scalar("loss_teacher_forcing",  mean_expert_loss, iters_so_far)
            writer.add_scalar("policy_loss", mean_policy_loss, iters_so_far)
            writer.add_scalar('lr', init_lr * cur_lrmult, iters_so_far)
            writer.add_scalar("TimeElapsed", time.time() - tstart, iters_so_far)
            writer.add_scalar("level", level, iters_so_far)

        if args['rl']:
            lrlocal = (seg["ep_lens"], seg["ep_lens_valid"], seg["ep_rets"], seg["ep_rets_env"], seg["ep_rets_d_step"],
                       seg["ep_rets_d_final"], seg["ep_final_rew"], seg["ep_final_rew_stat"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, lens_valid, rews, rews_env, rews_d_step, rews_d_final, rews_final, rews_final_stat = map(
                flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            lenbuffer_valid.extend(lens_valid)
            rewbuffer.extend(rews)
            rewbuffer_d_step.extend(rews_d_step)
            rewbuffer_d_final.extend(rews_d_final)
            rewbuffer_env.extend(rews_env)
            rewbuffer_final.extend(rews_final)
            rewbuffer_final_stat.extend(rews_final_stat)

            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)

            if writer is not None:
                writer.add_scalar("total_rl_loss", mean_total_rl_loss, iters_so_far)
                writer.add_scalar("ppo surrogate", mean_ppo_surrogate, iters_so_far)
                writer.add_scalar("entropy", mean_entropy, iters_so_far)
                writer.add_scalar("value function loss", vf_loss, iters_so_far)
                writer.add_scalar("ev_tdlam_before", explained_variance(vpredbefore, tdlamret), iters_so_far)

                writer.add_scalar("EpThisIter", len(lens), iters_so_far)
                writer.add_scalar("EpisodesSoFar", episodes_so_far, iters_so_far)
                writer.add_scalar("TimestepsSoFar", timesteps_so_far, iters_so_far)
                writer.add_scalar("EpLenMean", np.mean(lenbuffer), iters_so_far)
                writer.add_scalar("EpLenValidMean", np.mean(lenbuffer_valid), iters_so_far)
                writer.add_scalar("EpRewMean", np.mean(rewbuffer), iters_so_far)
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), iters_so_far)
                writer.add_scalar("EpRewFinalMean", np.mean(rewbuffer_final), iters_so_far)
                writer.add_scalar("EpRewFinalStatMean", np.mean(rewbuffer_final_stat), iters_so_far)

                if args['has_d_step']:
                    writer.add_scalar("loss_d_step", mean_d_step_loss, iters_so_far)
                    writer.add_scalar("EpRewDStepMean", np.mean(rewbuffer_d_step), iters_so_far)
                if args['has_d_final']:
                    writer.add_scalar("loss_d_final", mean_d_final_loss, iters_so_far)
                    writer.add_scalar("EpRewDFinalMean", np.mean(rewbuffer_d_final), iters_so_far)

        if mean_policy_loss < best_loss:
            if MPI.COMM_WORLD.Get_rank() == 0:
                checkpoint()
            best_loss = mean_policy_loss
            n_patient_rounds = 0
            current_time = time.time()

            if (evaluator is not None) and ((current_time - last_evaluation_time) > check_interval):
                evaluator(pi, n_samples=1024, checkpoint_path=checkpoint_path)
                last_evaluation_time = time.time()
        else:
            n_patient_rounds += 1

        if writer is not None:
            writer.add_scalar('n_patient_rounds', n_patient_rounds, iters_so_far)

        if (n_patient_rounds == args['patience']) and (level == (args['curriculum_num'] - 1)):
            print('Early stop!')
            return pi, var_list_pi

        iters_so_far += 1
        if (iters_so_far%args['curriculum_step'] == 0) and \
                (iters_so_far//args['curriculum_step'] < args['curriculum_num']):
            level += 1
            best_loss = float('inf')

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
