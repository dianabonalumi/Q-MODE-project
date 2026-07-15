"""
Test della pipeline su amminoacidi standard.
Eseguire con: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from amino_lattice.feature_extraction import extract_features, mol_from_smiles
from amino_lattice.site_selection import choose_k, select_representative_sites, topological_order
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice
from amino_lattice.labeling import label_sites, encode_chain
from amino_lattice.pdb_reader import mol_from_amino_acid
from amino_lattice.abraham_hbond import assign_abraham_hb_intensities

def _embed(mol):
    """AddHs + conformero 3D (ETKDG) — necessario per topological_order."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    return mol


# ─── Feature extraction ───────────────────────────────────────────────────────

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


# ─── Site selection ───────────────────────────────────────────────────────────

def test_choose_k_active_features():
    mol = mol_from_smiles("CC(N)C(=O)O")
    features = extract_features(mol)
    k = choose_k(features, strategy="active_features", max_k=8)
    assert 1 <= k <= 8

def test_choose_k_fixed():
    mol = mol_from_smiles("CC(N)C(=O)O")
    features = extract_features(mol)
    k = choose_k(features, strategy="fixed", fixed_k=5)
    assert k == 5

def test_select_sites_returns_k():
    mol = _embed(mol_from_smiles("OC(=O)C(N)Cc1c[nH]c2ccccc12"))  # TRP
    features = extract_features(mol, embed_3d=True)
    k = 4
    sites = select_representative_sites(features, k, mol=mol)
    assert len(sites) == k


# ─── Fitting geometrico ───────────────────────────────────────────────────────

def test_fit_pca_shape():
    mol = _embed(mol_from_smiles("OC(=O)C(N)Cc1ccccc1"))
    features = extract_features(mol, embed_3d=True)
    sites = select_representative_sites(features, 4, mol=mol)
    coords_2d = fit_to_lattice_2d(sites, method="pca")
    assert coords_2d.shape == (4, 2)

def test_fit_mds_shape():
    mol = _embed(mol_from_smiles("OC(=O)C(N)Cc1ccccc1"))
    features = extract_features(mol, embed_3d=True)
    sites = select_representative_sites(features, 4, mol=mol)
    coords_2d = fit_to_lattice_2d(sites, method="mds")
    assert coords_2d.shape == (4, 2)


# ─── Snapping ─────────────────────────────────────────────────────────────────

def test_snap_round_returns_ints():
    coords = np.array([[1.3, -0.7], [2.8, 1.1]])
    nodes = snap_to_lattice(coords, strategy="round")
    assert len(nodes) == 2
    for i, j in nodes:
        assert isinstance(i, int)
        assert isinstance(j, int)

def test_snap_hungarian_no_collisions():
    coords = np.array([[0.1, 0.1], [0.2, 0.2], [3.0, 3.0]])
    nodes = snap_to_lattice(coords, strategy="hungarian")
    assert len(nodes) == len(set(nodes)), "Collisione rilevata nei nodi"


# ─── Labeling ─────────────────────────────────────────────────────────────────

def test_label_one_hot_shape():
    mol = _embed(mol_from_smiles("CC(N)C(=O)O"))
    features = extract_features(mol, embed_3d=True)
    sites = select_representative_sites(features, 3, mol=mol)
    coords_2d = fit_to_lattice_2d(sites)
    nodes = snap_to_lattice(coords_2d)
    labeled = label_sites(sites, nodes, mode="one_hot")
    assert len(labeled) == 3
    for ls in labeled:
        assert ls.label.shape == (6,)  # 6 tipi farmacofori

def test_encode_chain_shape():
    mol = _embed(mol_from_smiles("CC(N)C(=O)O"))
    features = extract_features(mol, embed_3d=True)
    sites = select_representative_sites(features, 3, mol=mol)
    coords_2d = fit_to_lattice_2d(sites)
    nodes = snap_to_lattice(coords_2d)
    labeled = label_sites(sites, nodes, mode="one_hot")
    mat = encode_chain(labeled)
    assert mat.shape == (3, 8)  # 2 coord + 6 one-hot


# ─── Pipeline end-to-end (amminoacido isolato, stessa metodologia di
#     scripts/run_pipeline.py: estrazione + h/hb + ordinamento topologico) ────

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
