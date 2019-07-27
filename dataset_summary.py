"""
Summarizes the information of a molecule dataset, including:
- The number of valid molecules, the number of unique samples and their frequency
- The number of atom types and their frequency
- Top 3, mean and std of the following molecule properties:
  - penalized logP
  - QED
  - MW
  - SAS
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from gym_molecule.utils import get_penalized_logP
from gym_molecule.sascorer import calculateScore
from utils import mkdir_p

def make_hist_plot(name, values, root_path):
    x = pd.Series(values, name=name)
    ax = sns.distplot(x, kde=False)
    fig = ax.get_figure()
    fig.savefig(os.path.join(root_path, '{}.png'.format(name)))
    plt.clf()

def summarize_dataset(dataset, file_name):
    root_path = os.path.dirname(__file__)
    file_path = os.path.join(root_path, 'gym_molecule', 'dataset', file_name)
    summary_path = os.path.join(root_path, 'dataset_summary', dataset)
    mkdir_p(summary_path)

    data = pd.read_csv(file_path, header=None, names=['smiles'])

    to_dump = {
        'n_samples': len(data),
        'n_valid_samples': 0,
        'molecule_counting': defaultdict(int),
        'atom_type_counting': defaultdict(int),
        'atom_degree_counting': defaultdict(int),
        'bond_type_counting': defaultdict(int),
        'num_rings_counting': defaultdict(int), # Count number of rings in a molecule
        'ring_size_counting': defaultdict(int)  # Count the size of rings in a molecule
    }
    valid_smiles = []
    logP_values = []
    penalized_logP_values = []
    QED_values = []
    MW_values = []
    SAS_values = []

    for i in range(to_dump['n_samples']):
        print('Processing smiles {:d}/{:d}'.format(i + 1, to_dump['n_samples']))
        smiles_i = data['smiles'][i].strip()
        mol_i = Chem.MolFromSmiles(smiles_i)
        if mol_i is None:
            continue
        valid_smiles.append(smiles_i)

        to_dump['n_valid_samples'] += 1
        to_dump['molecule_counting'][smiles_i] += 1

        logP_values.append(MolLogP(mol_i))
        penalized_logP_values.append(get_penalized_logP(mol_i))
        QED_values.append(qed(mol_i))
        MW_values.append(CalcExactMolWt(mol_i))
        SAS_values.append(calculateScore(mol_i))
        n_rings = mol_i.GetRingInfo().NumRings()
        if n_rings > 0:
            to_dump['num_rings_counting'][n_rings] += 1
            ring_sizes = [len(r) for r in mol_i.GetRingInfo().AtomRings()]
            for size in ring_sizes:
                to_dump['ring_size_counting'][size] += 1

        Chem.SanitizeMol(mol_i, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        for atom in mol_i.GetAtoms():
            to_dump['atom_type_counting'][atom.GetSymbol()] += 1
            to_dump['atom_degree_counting'][atom.GetDegree()] += 1

        for bond in mol_i.GetBonds():
            to_dump['bond_type_counting'][str(bond.GetBondType())] += 1

    logP_values.sort(reverse=True)
    penalized_logP_values.sort(reverse=True)
    QED_values.sort(reverse=True)
    MW_values.sort(reverse=True)
    SAS_values.sort(reverse=True)

    with open(os.path.join(summary_path, 'summary.txt'), 'w') as f:
        f.write('n_samples\t{:d}\n'.format(to_dump['n_samples']))
        f.write('n_valid_samples\t{:d}\n'.format(to_dump['n_valid_samples']))
        f.write('\n')

        def _dump_count_info(key, item):
            f.write('{} counting\n'.format(item))
            f.write('================================================\n')
            f.write('{}\tcount\n'.format(item))
            ascending_keys = sorted(
                to_dump[key].keys(), key=lambda k:to_dump[key][k])
            for type in ascending_keys:
                f.write('{}\t{:d}\n'.format(type, to_dump[key][type]))
            f.write('================================================\n')
            f.write('\n')

        _dump_count_info('atom_type_counting', 'atom type')
        _dump_count_info('atom_degree_counting', 'atom degree')
        _dump_count_info('bond_type_counting', 'bond type')
        _dump_count_info('num_rings_counting', 'num rings in a molecule')
        _dump_count_info('ring_size_counting', 'ring size')

        def _dump_value_info(item, values):
            f.write('{} \t mean {:.4f} \t std {:.4f} \t top 3 \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
                item, np.mean(values), np.std(values),
                values[0], values[1], values[2]))

        _dump_value_info('logP', logP_values)
        _dump_value_info('penalized logP', penalized_logP_values)
        _dump_value_info('QED', QED_values)
        _dump_value_info('MW', MW_values)
        _dump_value_info('SAS', SAS_values)

    make_hist_plot('logP', logP_values, summary_path)
    make_hist_plot('penalized logP', penalized_logP_values, summary_path)
    make_hist_plot('QED', QED_values, summary_path)
    make_hist_plot('MW', MW_values, summary_path)
    make_hist_plot('SAS', SAS_values, summary_path)

    dataset_info = {
        'valid_smiles': valid_smiles,
        'logP': logP_values,
        'penalized_logP': penalized_logP_values,
        'QED': QED_values,
        'MW': MW_values,
        'SAS': SAS_values
    }
    np.savez(
        os.path.join(summary_path, 'summary.npz'),
        **dataset_info
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GCPN-DGL')
    parser.add_argument('-d', '--dataset', type=str,
                        default='250k_rndm_zinc_drugs_clean_sorted')
    parser.add_argument('-f', '--file-name', type=str,
                        default='250k_rndm_zinc_drugs_clean_sorted.smi')
    args = parser.parse_args().__dict__
    summarize_dataset(args['dataset'], args['file_name'])
