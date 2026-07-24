"""Pipeline tests on standard amino acids. Run with: pytest tests/"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qmode.feature_extraction import extract_features, mol_from_smiles
from qmode.site_selection import topological_order
from qmode.pdb_reader import mol_from_amino_acid
from qmode.abraham_hbond import assign_abraham_hb_intensities

def _embed(mol):
    """AddHs + 3D conformer (ETKDG) -- required by topological_order."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    return mol


def test_mol_from_smiles():
    mol = mol_from_smiles("CC(N)C(=O)O")
    assert mol is not None

def test_mol_from_smiles_invalid():
    with pytest.raises(ValueError):
        mol_from_smiles("INVALID_SMILES!!!")

def test_extract_features_returns_list():
    mol = mol_from_smiles("CC(N)C(=O)O")
    features = extract_features(mol, embed_3d=True)
    assert isinstance(features, list)
    assert len(features) > 0

def test_features_have_3d_coords():
    mol = mol_from_smiles("OC(=O)C(N)Cc1ccccc1")  # PHE
    features = extract_features(mol, embed_3d=True)
    for f in features:
        assert f.coords.shape == (3,), f"coords shape attesa (3,), trovata {f.coords.shape}"


def test_topological_order_returns_all_sites():
    mol = _embed(mol_from_smiles("OC(=O)C(N)Cc1c[nH]c2ccccc12"))  # TRP
    features = extract_features(mol, embed_3d=True)
    sites = topological_order(features, mol=mol)
    assert len(sites) == len(features)

def test_topological_order_single_site():
    mol = _embed(mol_from_smiles("CC(N)C(=O)O"))
    features = extract_features(mol, embed_3d=True)[:1]
    sites = topological_order(features, mol=mol)
    assert sites == features


# end-to-end on a standalone amino acid, same steps as run_pipeline.py:
# extraction + h/hb + topological ordering
def _map_amino_acid(name):
    mol = mol_from_amino_acid(name)
    features = extract_features(mol, embed_3d=True)
    assign_abraham_hb_intensities(features, res_name=name, mol=mol)
    return topological_order(features, mol=mol)

STANDARD_20_NAMES = ["ALA", "GLY", "TRP", "PHE", "SER"]

@pytest.mark.parametrize("aa", STANDARD_20_NAMES)
def test_pipeline_all_standard_aa(aa):
    sites = _map_amino_acid(aa)
    assert len(sites) >= 1

def test_pipeline_chain_format():
    sites = _map_amino_acid("ALA")
    for s in sites:
        assert isinstance(s.feature_type, str)
        assert isinstance(s.intensity, float)

def test_pipeline_batch():
    results = [_map_amino_acid(aa) for aa in STANDARD_20_NAMES]
    assert len(results) == len(STANDARD_20_NAMES)
