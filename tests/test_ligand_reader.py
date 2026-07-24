import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qmode.ligand_reader as ligand_reader
from qmode.ligand_reader import load_ligand_from_pdb, ligand_to_mol, LigandRecord


# Synthetic PDB block: water (excluded), a chelating CA ion (excluded, below
# min_heavy_atoms), a standard amino acid tagged as HETATM (excluded, already
# covered by AMINO_SMILES), and a 6-atom ligand (BNZ, realistic hexagonal
# geometry) that should be the chosen group.
PDB_TEXT = """\
HETATM    1  O   HOH A 900      10.000  10.000  10.000  1.00  0.00           O
HETATM    2 CA    CA A 901      20.000  20.000  20.000  1.00  0.00          CA
HETATM    3  N   ALA A 902       0.000   0.000   0.000  1.00  0.00           N
HETATM    4  CA  ALA A 902       1.458   0.000   0.000  1.00  0.00           C
HETATM    5  C   ALA A 902       2.000   1.400   0.000  1.00  0.00           C
HETATM    6  C1  BNZ A 903       1.390   0.000   0.000  1.00  0.00           C
HETATM    7  C2  BNZ A 903       0.695   1.204   0.000  1.00  0.00           C
HETATM    8  C3  BNZ A 903      -0.695   1.204   0.000  1.00  0.00           C
HETATM    9  C4  BNZ A 903      -1.390   0.000   0.000  1.00  0.00           C
HETATM   10  C5  BNZ A 903      -0.695  -1.204   0.000  1.00  0.00           C
HETATM   11  C6  BNZ A 903       0.695  -1.204   0.000  1.00  0.00           C
END
"""


def _write_pdb(tmp_path):
    p = tmp_path / "synthetic.pdb"
    p.write_text(PDB_TEXT)
    return str(p)


def test_load_ligand_picks_largest_group_and_excludes_water_ion_amino(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_reader, "fetch_ccd_smiles", lambda code, timeout=5.0: None)
    pdb_path = _write_pdb(tmp_path)
    rec = load_ligand_from_pdb(pdb_path)
    assert rec is not None
    assert rec.res_name == "BNZ"
    assert len(rec.atoms) == 6


def test_load_ligand_respects_explicit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_reader, "fetch_ccd_smiles", lambda code, timeout=5.0: None)
    pdb_path = _write_pdb(tmp_path)
    rec = load_ligand_from_pdb(pdb_path, ligand_code="bnz")
    assert rec is not None
    assert rec.res_name == "BNZ"


def test_load_ligand_min_heavy_atoms_filters_ion(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_reader, "fetch_ccd_smiles", lambda code, timeout=5.0: None)
    pdb_path = _write_pdb(tmp_path)
    rec = load_ligand_from_pdb(pdb_path, ligand_code="CA")
    assert rec is None


def test_ligand_to_mol_falls_back_to_geometric_perception(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_reader, "fetch_ccd_smiles", lambda code, timeout=5.0: None)
    pdb_path = _write_pdb(tmp_path)
    rec = load_ligand_from_pdb(pdb_path, ligand_code="BNZ")
    assert rec is not None
    assert rec.mol is not None
    assert rec.mol.GetNumAtoms() >= 6


def test_ligand_to_mol_uses_ccd_template_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_reader, "fetch_ccd_smiles", lambda code, timeout=5.0: "c1ccccc1")
    pdb_path = _write_pdb(tmp_path)
    rec = load_ligand_from_pdb(pdb_path, ligand_code="BNZ")
    assert rec is not None
    assert rec.mol is not None
    ring_info = rec.mol.GetRingInfo()
    assert ring_info.NumRings() >= 1
