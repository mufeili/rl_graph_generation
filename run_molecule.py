#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
from tensorboardX import SummaryWriter
import os
import tensorflow as tf
import time

import gym
from eval import Evaluator


def train(args, seed, writer=None):
    from baselines.ppo1 import pposgd_simple_gcn, gcn_policy
    import baselines.common.tf_util as U
    import gym_molecule

    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    if args['env']=='molecule':
        env = gym.make('molecule-v0')
        env.init(data_type=args['dataset'],logp_ratio=args['logp_ratio'],qed_ratio=args['qed_ratio'],
                 sa_ratio=args['sa_ratio'], reward_step_total=args['reward_step_total'],
                 is_normalize=args['normalize_adj'], reward_type=args['reward_type'],
                 reward_target=args['reward_target'], has_feature=bool(args['has_feature']),
                 is_conditional=bool(args['is_conditional']), conditional=args['conditional'],
                 max_action=args['max_action'], min_action=args['min_action']) # remember call this after gym.make!!
    else:
        raise ValueError
    print(env.observation_space)
    env.seed(workerseed)

    if rank == 0:
        evaluator = Evaluator('molecule_gen/', 'ZINC250K', env, writer=writer)
    else:
        evaluator = None

    pi, var_list_pi = pposgd_simple_gcn.learn(args, env, evaluator,
                                              max_time_steps=args['num_steps'],
                                              horizon=256, clip_param=0.2, entropy_coef=0.01,
                                              optim_epochs=8, init_lr=args['lr'], optim_batchsize=32,
                                              gamma=1, lam=0.95, schedule='linear', writer=writer)

    if evaluator is not None:
        evaluator(pi, n_samples=1024, final=True, checkpoint_path='./ckpt/' + args['name_full'])

    env.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()
    parser.add_argument('-rl', '--rl', default=True)
    parser.add_argument('--env', type=str, help='environment name: molecule; graph',
                        default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(1e6))
    parser.add_argument('--name', type=str, default='test_conditional')
    parser.add_argument('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    parser.add_argument('--reward_type', type=str, default='logp_pen',help='logppen;logp_target;qed;qedsa;mw_target;gan')
    parser.add_argument('--reward_target', type=float, default=0.5,help='target reward value')
    parser.add_argument('--logp_ratio', type=float, default=1)
    parser.add_argument('--qed_ratio', type=float, default=0)
    parser.add_argument('--sa_ratio', type=float, default=0)
    parser.add_argument('--gan_step_ratio', type=float, default=1)
    parser.add_argument('--gan_final_ratio', type=float, default=1)
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--has_d_step', type=int, default=1)
    parser.add_argument('--has_d_final', type=int, default=1)
    parser.add_argument('--rl_start', type=int, default=250)
    parser.add_argument('--rl_end', type=int, default=int(1e6))
    parser.add_argument('--expert_start', type=int, default=0)
    parser.add_argument('--expert_end', type=int, default=int(1e6))
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--curriculum_num', type=int, default=6)
    parser.add_argument('--curriculum_step', type=int, default=200)
    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--layer_num_g', type=int, default=3)
    parser.add_argument('--layer_num_d', type=int, default=3)
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--stop_shift', type=int, default=-3)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=128) # default 64
    parser.add_argument('--gcn_aggregate', type=str, default='mean')# sum, mean, concat
    parser.add_argument('--gan_type', type=str, default='recommend')# normal, recommend
    parser.add_argument('--gate_sum_d', type=int, default=0)
    parser.add_argument('--mask_null', type=int, default=0)
    parser.add_argument('--is_conditional', type=int, default=0) # default 0
    parser.add_argument('--conditional', type=str, default='low') # default 0
    parser.add_argument('--max_action', type=int, default=128) # default 0
    parser.add_argument('--min_action', type=int, default=20) # default 0
    parser.add_argument('--bn', type=int, default=1)
    parser.add_argument('-pa', '--patience', type=int, default=1000)

    return parser

def log_time(time):
    day = time // (24 * 3600)
    time %= (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))

def main():
    args = molecule_arg_parser().parse_args().__dict__
    print(args)
    args['name_full'] = args['env'] + '_' + args['dataset'] + '_' + args['name']
    args['name_full_load'] = args['env'] + '_' + args['dataset']
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    # only keep first worker result in tensorboard
    if MPI.COMM_WORLD.Get_rank() == 0:
        writer = SummaryWriter(comment='_' + args['dataset'] + '_' + args['name'])
    else:
        writer = None

    t_start = time.time()
    train(args, seed=args['seed'], writer=writer)
    t_end = time.time()
    log_time(t_end - t_start)

if __name__ == '__main__':
    main()
