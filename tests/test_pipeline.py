"""
Test della pipeline su amminoacidi standard.
Eseguire con: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from rdkit import Chem

from amino_lattice import AminoLatticePipeline
from amino_lattice.feature_extraction import extract_features, mol_from_smiles
from amino_lattice.site_selection import choose_k, select_representative_sites
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice
from amino_lattice.labeling import label_sites, encode_chain

# SMILES dei 20 AA standard
STANDARD_AA = {
    "ALA": "CC(N)C(=O)O",
    "GLY": "NCC(=O)O",
    "TRP": "OC(=O)C(N)Cc1c[nH]c2ccccc12",
    "PHE": "OC(=O)C(N)Cc1ccccc1",
    "SER": "OCC(N)C(=O)O",
}


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
    mol = mol_from_smiles("OC(=O)C(N)Cc1c[nH]c2ccccc12")  # TRP
    features = extract_features(mol)
    k = 4
    sites = select_representative_sites(features, k)
    assert len(sites) == k


# ─── Fitting geometrico ───────────────────────────────────────────────────────

def test_fit_pca_shape():
    mol = mol_from_smiles("OC(=O)C(N)Cc1ccccc1")
    features = extract_features(mol)
    sites = select_representative_sites(features, 4)
    coords_2d = fit_to_lattice_2d(sites, method="pca")
    assert coords_2d.shape == (4, 2)

def test_fit_mds_shape():
    mol = mol_from_smiles("OC(=O)C(N)Cc1ccccc1")
    features = extract_features(mol)
    sites = select_representative_sites(features, 4)
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
    mol = mol_from_smiles("CC(N)C(=O)O")
    features = extract_features(mol)
    sites = select_representative_sites(features, 3)
    coords_2d = fit_to_lattice_2d(sites)
    nodes = snap_to_lattice(coords_2d)
    labeled = label_sites(sites, nodes, mode="one_hot")
    assert len(labeled) == 3
    for ls in labeled:
        assert ls.label.shape == (6,)  # 6 tipi farmacofori

def test_encode_chain_shape():
    mol = mol_from_smiles("CC(N)C(=O)O")
    features = extract_features(mol)
    sites = select_representative_sites(features, 3)
    coords_2d = fit_to_lattice_2d(sites)
    nodes = snap_to_lattice(coords_2d)
    labeled = label_sites(sites, nodes, mode="one_hot")
    mat = encode_chain(labeled)
    assert mat.shape == (3, 8)  # 2 coord + 6 one-hot


# ─── Pipeline end-to-end ──────────────────────────────────────────────────────

@pytest.mark.parametrize("aa,smiles", STANDARD_AA.items())
def test_pipeline_all_standard_aa(aa, smiles):
    pipeline = AminoLatticePipeline(max_k=8, embed_3d=True)
    result = pipeline.run(smiles=smiles, name=aa)
    assert result.k >= 1
    assert len(result.chain) == result.k
    assert result.stress >= 0.0

def test_pipeline_chain_format():
    pipeline = AminoLatticePipeline()
    result = pipeline.run("CC(N)C(=O)O", name="ALA")
    for i, j, t in result.chain:
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(t, str)

def test_pipeline_batch():
    pipeline = AminoLatticePipeline(max_k=6)
    records = [{"smiles": s, "name": n} for n, s in STANDARD_AA.items()]
    results = pipeline.run_batch(records)
    assert len(results) == len(STANDARD_AA)
