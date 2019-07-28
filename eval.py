import numpy as np
import os
import tensorflow as tf
import time
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from gym_molecule.utils import get_penalized_logP
from gym_molecule.sascorer import calculateScore
from utils import mkdir_p, get_valid_smiles, get_unique_smiles, \
    get_novel_smiles, get_FCD_distance, continuous_kldiv

__all__ = ['Evaluator']

rdBase.DisableLog('rdApp.error')

class Evaluator(object):
    def __init__(self, root_dir, data, env, node_feat_name='h', writer=None, topk=3):
        """
        Parameters
        ----------
        root_dir : str
        data : str
        env : gym.Env
        node_feat_name : str
            The field name for storing node features
        writer : Tensorboard writer or None
        topk : int
            We will track the top topk values for each property
        """
        self.root_dir = root_dir
        self.env = env
        self.writer = writer
        # Number of evaluation trials
        self.n_eval = 0
        self.topk = topk

        self.node_feat_name = node_feat_name

        assert data == 'ZINC250K', 'Other dataset has not been supported yet.'
        if data == 'ZINC250K':
            summary_file_name = 'dataset_summary/250k_rndm_zinc_drugs_clean_sorted/summary.npz'
            if not os.path.isfile(summary_file_name):
                from dataset_summary import summarize_dataset
                summarize_dataset(dataset='250k_rndm_zinc_drugs_clean_sorted',
                                  file_name='250k_rndm_zinc_drugs_clean_sorted.smi')
            self.data_info = np.load(summary_file_name)
            assert len(self.data_info['valid_smiles']) > 1, 'If the set contains 1 sample only, ' \
                                                            'a function will raise an Error'

        self.data_logP = (
            float(self.data_info['logP'].mean()),
            float(self.data_info['logP'].std()),
            self.data_info['logP'][:topk]
        )
        self.data_penalized_logP = (
            float(self.data_info['penalized_logP'].mean()),
            float(self.data_info['penalized_logP'].std()),
            self.data_info['penalized_logP'][:topk]
        )
        self.data_QED = (
            float(self.data_info['QED'].mean()),
            float(self.data_info['QED'].std()),
            self.data_info['QED'][:topk]
        )
        self.data_MW = (
            float(self.data_info['MW'].mean()),
            float(self.data_info['MW'].std()),
            self.data_info['MW'][:topk]
        )
        self.data_SAS = (
            float(self.data_info['SAS'].mean()),
            float(self.data_info['SAS'].std()),
            self.data_info['SAS'][:topk]
        )
        self.data_properties = {
            'logP': self.data_logP,
            'penalized_logP': self.data_penalized_logP,
            'QED': self.data_QED,
            'MW': self.data_MW,
            'SAS': self.data_SAS
        }

    def __call__(self, policy, checkpoint_path, n_samples=None, final=False):
        path = os.path.join(self.root_dir, 'generation_evaluation', str(self.n_eval))
        mkdir_p(path)
        reference_smiles = self.data_info['valid_smiles'].tolist()

        if n_samples is None:
            # Follow the fashion of GuacaMol
            n_samples = 10000
        assert n_samples > 1, 'If the set contains 1 sample only, a function will raise an Error'

        smiles = []

        with tf.Session() as sess:
            var_list_pi = policy.get_trainable_variables()
            saver = tf.train.Saver(var_list_pi)
            saver.restore(sess, checkpoint_path)

            while len(smiles) < n_samples:
                print('Generating the molecule {:d}/{:d}'.format(len(smiles) + 1, n_samples))
                ob_t = self.env.reset()
                stop = False
                while not stop:
                    a_t, _, _ = policy.act(True, ob_t)
                    ob_t, _, stop, info_t = self.env.step(a_t)
                smiles.append(info_t['smile'])
        valid_smiles = get_valid_smiles(smiles)
        unique_smiles = get_unique_smiles(valid_smiles)
        novel_smiles = get_novel_smiles(unique_smiles, reference_smiles)

        # Get discriptors among valid smiles generated
        valid_logP = []
        valid_penalized_logP = []
        valid_QED = []
        valid_MW = []
        valid_SAS = []
        for mol_s in valid_smiles:
            mol = Chem.MolFromSmiles(mol_s)
            valid_logP.append(MolLogP(mol))
            valid_penalized_logP.append(get_penalized_logP(mol))
            valid_QED.append(qed(mol))
            valid_MW.append(CalcExactMolWt(mol))
            valid_SAS.append(calculateScore(mol))

        # Save smiles
        with open(os.path.join(path, 'valid_molecules.txt'), 'w') as f:
            for mol_s in valid_smiles:
                f.write('{}\n'.format(mol_s))

        # Evaluation report
        with open(os.path.join(path, 'report.txt'), 'w') as f:
            f.write('n_samples \t {:d} \n'.format(n_samples))
            f.write('\n')

            # Validity
            n_valid = len(valid_smiles)
            ratio_valid = n_valid / n_samples
            f.write('n_valid \t {:d} \t ratio_valid {:.4f}\n'.format(n_valid, ratio_valid))
            if self.writer is not None:
                self.writer.add_scalar('ratio_valid', ratio_valid, self.n_eval)

            # Uniqueness
            n_unique = len(unique_smiles)
            ratio_unique_valid = n_unique / n_valid
            f.write('n_unique \t {:d} \t ratio_unique_valid {:.4f}\n'.format(n_unique, ratio_unique_valid))
            if self.writer is not None:
                self.writer.add_scalar('ratio_unique_among_valid', ratio_unique_valid, self.n_eval)

            # Novelty
            n_novel = len(novel_smiles)
            ratio_novel_unique = n_novel / n_unique
            f.write('n_novel \t {:d} \t ratio_novel_unique {:.4f}\n'.format(n_novel, ratio_novel_unique))
            if self.writer is not None:
                self.writer.add_scalar('ratio_novel_among_unique_valid', ratio_novel_unique, self.n_eval)

            if final:
                # Frechet ChemNet Distance

                # FCD distance is quite expensive to compute so we compute it only
                # when the training is finished.
                fcd_score = get_FCD_distance(valid_smiles, reference_smiles)
                f.write('\n')
                f.write('FCD score \t {:.4f}\n'.format(fcd_score))
                if self.writer is not None:
                    # Since we only evaluate FCD score when the training is completed,
                    # we use a timestep of 0.
                    self.writer.add_scalar('FCD_score', fcd_score, 0)

            def property_report(name, values):
                mean = float(np.mean(values))
                std = float(np.std(values))
                values.sort(reverse=True)
                topk = values[:self.topk]
                values = np.array(values)
                KL = continuous_kldiv(self.data_info[name], values)
                mean_ref, std_ref, topk_ref = self.data_properties[name]

                f.write('\n')
                f.write('{}\n'.format(name))
                f.write('--------------------------------------------------------------------------------------\n')
                f.write('Dataset mean {:.4f} \t std {:.4f} \t'.format(mean_ref, std_ref) +
                        ' \t'.join([str(v) for v in topk_ref.tolist()]) + '\n')
                f.write('Samples mean {:.4f} \t std {:.4f} \t'.format(mean, std) +
                        ' \t'.join([str(v) for v in topk]) + '\n')
                f.write('KL(Dataset || Samples) \t {:.4f}\n'.format(KL))
                f.write('--------------------------------------------------------------------------------------\n')

                if self.writer is not None:
                    self.writer.add_scalar('{}_mean'.format(name), mean, self.n_eval)
                    self.writer.add_scalar('{}_std'.format(name), std, self.n_eval)
                    for i in range(self.topk):
                        self.writer.add_scalar('{}_{:d}'.format(name, i + 1), topk[i], self.n_eval)
                    self.writer.add_scalar('{}_KL_dataset_samples'.format(name), KL, self.n_eval)

            property_report('logP', valid_logP)
            property_report('penalized_logP', valid_penalized_logP)
            property_report('QED', valid_QED)
            property_report('MW', valid_MW)
            property_report('SAS', valid_SAS)

        self.n_eval += 1

if __name__ == '__main__':
    import gym

    from baselines.ppo1.gcn_policy import GCNPolicy

    dataset = 'ZINC250K'
    task = 'optimize'
    property = 'penalized_logP'
    min_n_actions = 20
    max_n_actions = 128
    additional_features = None
    reward_step_total = 0.5

    env = gym.make('molecule-v0')
    env.seed(0)
    env.init()
    ob_space = env.observation_space
    ac_space = env.action_space

    evaluator = Evaluator('./', dataset, env)
    policy_args = {
        'emb_size': 128,
        'bn': 1,
        'gcn_aggregate': 'mean',
        'layer_num_g': 3,
        'stop_shift': -3
    }
    policy = GCNPolicy(name="pi",
                       ob_space=ob_space,
                       ac_space=ac_space,
                       atom_type_num=env.atom_type_num,
                       args=policy_args)

    t1 = time.time()
    evaluator(policy, n_samples=1024)
    t2 = time.time()
    print(t2 - t1)
