import fcd
import numpy as np
import os
import pkgutil
import tempfile
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import entropy, gaussian_kde

__all__ = ['tanimoto_similarity', 'get_valid_smiles', 'get_unique_smiles',
           'get_novel_smiles', 'get_FCD_distance', 'continuous_kldiv']

def tanimoto_similarity(mol_a, mol_b):
    """
    Computes the Tanimoto similarity between two molecules.

    The code is based on the released implementation of graph2graph
    by Jin et al. 2019.

    Given two molecules mol_a, mol_b, denote their fingerprint by
    fp_a and fp_b, the Tanimoto similarity is defined as
    c / (a + b - c), where:
    - c bits are same for fp_a and fp_b
    - fp_a and fp_b separately has a and b bits

    Parameters
    ----------
    a : SMILES or Chem.rdchem.Mol
    b : SMILES or Chem.rdchem.Mol

    Returns
    -------
    float
        According to the Wikipedia, two structures are usually considered
        similar if T > 0.85 (for Daylight fingerprints). However, this does
        not necessarily suggest similar bioactivities in general.
    """
    if isinstance(mol_a, str):
        mol_a = Chem.MolFromSmiles(mol_a)
    assert isinstance(mol_a, Chem.rdchem.Mol)

    if isinstance(mol_b, str):
        mol_b = Chem.MolFromSmiles(mol_b)
    assert isinstance(mol_b, Chem.rdchem.Mol)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048, useChirality=False)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_valid_smiles(smiles, include_stereocenters=True):
    """Given a list of smiles, return a list consisting of valid ones only.

    Parameters
    ----------
    smiles : list of str
    include_stereocenters: bool
        whether to keep the stereochemical information in the canonical SMILES strings

    Return
    ------
    list of str
        List of valid smiles only
    """
    valid_smiles = []
    for mol_s in smiles:
        mol = Chem.MolFromSmiles(mol_s)
        if mol is None:
            continue
        valid_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters))

    return valid_smiles

def get_unique_smiles(smiles):
    """Given a list of smiles, return a list consisting of unique elements in it.

    Parameters
    ----------
    smiles : list of str

    Returns
    -------
    list of str
        Sublist where each smiles occurs exactly once
    """
    unique_set = set()
    for mol_s in smiles:
        if mol_s not in unique_set:
            unique_set.add(mol_s)

    return list(unique_set)

def get_novel_smiles(new_smiles, reference_smiles):
    """Get a list of novel smiles.

    Identify the smiles in new_smiles that are not in reference_smiles

    Parameters
    ----------
    new_smiles : list of str
    reference_smiles : list of str

    Returns
    -------
    novel_smiles : list of str
    """
    # Here we follow the approach of GuacaMol and does not include stereocenters.
    new_smiles = get_valid_smiles(new_smiles, include_stereocenters=False)
    new_smiles = set(get_unique_smiles(new_smiles))

    reference_smiles = get_valid_smiles(reference_smiles, include_stereocenters=False)
    reference_smiles = set(get_unique_smiles(reference_smiles))
    novel_smiles = new_smiles.difference(reference_smiles)

    return novel_smiles

def load_chemnet(model_file_name='ChemNet_v0.13_pretrained.h5'):
    """
    Load the ChemNet model.

    This file lives inside a package but to use it, it must always be an actual file.
    The safest way to proceed is therefore:
    1. read the file with pkgutil
    2. save it to a temporary file
    3. load the model from the temporary file

    Parameters
    ----------
    model_file_name : str
        Name for the model file to save

    Returns
    -------
    keras.engine.sequential.Sequential
        A pre-trained model for predicting drug activities, which will later be used
        for FCD computation
    """
    model_bytes = pkgutil.get_data('fcd', model_file_name)
    tmpdir = tempfile.gettempdir()
    model_path = os.path.join(tmpdir, model_file_name)

    with open(model_path, 'wb') as f:
        f.write(model_bytes)
    print('Saved ChemNet model to {}'.format(model_path))

    return fcd.load_ref_model(model_path)

def get_distribution_stats(model, smiles):
    """
    1. Fetch the activations of a list of smiles from the penultimate layer of the ChemNet
    2. Calculate the mean and covariance of the activations

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        pre-trained ChemNet model
    smiles : list of str
    """
    canonical_smiles = fcd.canoncial_smiles(smiles)
    mol_activations = fcd.get_predictions(model, canonical_smiles)

    mu = np.mean(mol_activations, axis=0)
    cov = np.cov(mol_activations.T)

    return mu, cov

def get_FCD_distance(new_smiles, reference_smiles):
    """Compute the Frechet ChemNet distance (FCD) between two collection of smiles.

    Identify the smiles in new_smiles that are not in reference_smiles

    Parameters
    ----------
    new_smiles : list of str
    reference_smiles : list of str

    Returns
    -------
    fcd_score : float
    """
    chemnet = load_chemnet()
    mu, cov = get_distribution_stats(chemnet, new_smiles)
    mu_reference, cov_reference = get_distribution_stats(chemnet, reference_smiles)

    fcd_score_ = fcd.calculate_frechet_distance(mu1=mu_reference, mu2=mu,
                                                sigma1=cov_reference, sigma2=cov)
    fcd_score = np.exp(-0.2 * fcd_score_)

    return float(fcd_score)

def continuous_kldiv(X_baseline, X_sampled):
    """
    Parameters
    ----------
    X_baseline : float64 numpy.ndarray of shape (A, )
    X_sampled : float64 numpy.ndarray of shape (B, )

    Returns
    -------
    float
    """
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)

    X_all = np.hstack([X_baseline, X_sampled])
    x_eval = np.linspace(X_all.min(), X_all.max(), num=1000)
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return float(entropy(P, Q))

if __name__ == '__main__':
    s_a = 'CCO'
    s_b = 'C'

    mol_a = Chem.MolFromSmiles(s_a)
    mol_b = Chem.MolFromSmiles(s_b)

    print(type(tanimoto_similarity(s_a, mol_b)))
    print(tanimoto_similarity(s_a, s_b))
    print(tanimoto_similarity(mol_a, s_b))
    print(tanimoto_similarity(mol_a, mol_b))

    smiles_1 = ['N#[PH]1(F)C(=I(Br)(Br)OBr)P=C(Cl)P1Cl',
                'N#COBr',
                'N#S(F)=C=I(=O)N=NBr']
    smiles_2 = ['O=C1I2#S#CI12=O',
                'FC(Br)C1(Br)C#I1Br',
                'N=NP1#CI#1I(=N)(Br)NP=I']
    print(get_FCD_distance(smiles_1, smiles_2))
