__author__ = "Bowen Liu"
__copyright__ = "Copyright 2018, Stanford University"

import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.Descriptors import MolLogP

from .sascorer import calculateScore

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def load_dataset(path):
  """
  Loads gdb13 dataset from path to pandas dataframe
  :param path:
  :return:
  """
  df = pd.read_csv(path, header=None, names=['smiles'])
  return df

def sort_dataset(in_path, out_path):
    """
    Sorts the dataset of smiles from input path by molecular complexity as
    proxied by the BertzCT index, and outputs the new sorted dataset
    :param in_path:
    :param out_path:
    :return:
    """
    def _calc_bertz_ct(smiles):
        return GraphDescriptors.BertzCT(Chem.MolFromSmiles(smiles))

    in_df = load_dataset(in_path)
    in_df['BertzCT'] = in_df.smiles.apply(_calc_bertz_ct)
    sorted_df = in_df.sort_values(by=['BertzCT'])
    sorted_df['smiles'].to_csv(out_path, index=False)

class gdb_dataset:
  """
  Simple object to contain the gdb13 dataset
  """
  def __init__(self, path):
    self.df = load_dataset(path)

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, item):
    """
    Returns an rdkit mol object
    :param item:
    :return:
    """
    smiles = self.df['smiles'][item]
    mol = Chem.MolFromSmiles(smiles)
    return mol

def get_penalized_logP(mol):
    """
    The function computes logP penalized by SAS and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on
    the statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
    """
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = -3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    sa = - calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = - cycle_length

    normalized_log_p = (log_p - logp_mean) / logp_std
    normalized_SA = (sa - sa_mean) / sa_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle
