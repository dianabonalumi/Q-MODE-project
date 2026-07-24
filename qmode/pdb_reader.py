"""
Reads a PDB file (full protein or pocket) and returns residues as RDKit Mol
objects ready for feature_extraction.

  - load_residues_from_pdb()  -> list of ResidueRecord
  - residue_to_mol()          -> RDKit Mol with real 3D coordinates from the PDB
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


AMINO_SMILES = {
    "ALA": "CC(N)C(=O)O",
    "ARG": "N=C(N)NCCCC(N)C(=O)O",
    "ASN": "NC(=O)CC(N)C(=O)O",
    "ASP": "OC(=O)CC(N)C(=O)O",
    "CYS": "SCC(N)C(=O)O",
    "GLN": "NC(=O)CCC(N)C(=O)O",
    "GLU": "OC(=O)CCC(N)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "OC(=O)C(N)Cc1c[nH]cn1",
    "ILE": "CCC(C)C(N)C(=O)O",
    "LEU": "CC(C)CC(N)C(=O)O",
    "LYS": "NCCCCC(N)C(=O)O",
    "MET": "CSCCC(N)C(=O)O",
    "PHE": "OC(=O)C(N)Cc1ccccc1",
    "PRO": "OC(=O)C1CCCN1",
    "SER": "OCC(N)C(=O)O",
    "THR": "CC(O)C(N)C(=O)O",
    "TRP": "OC(=O)C(N)Cc1c[nH]c2ccccc12",
    "TYR": "OC(=O)C(N)Cc1ccc(O)cc1",
    "VAL": "CC(C)C(N)C(=O)O",
    "HSD": "OC(=O)C(N)Cc1c[nH]cn1",
    "HSE": "OC(=O)C(N)Cc1cnc[nH]1",
    # HSP/HIP: doubly protonated histidine (imidazolium, +1 charge).
    # The previous "OC(=O)C(N)Cc1c[nH+]cn1" wouldn't kekulize (MolFromSmiles
    # returned None) -- explicit Kekule form fixes it:
    "HSP": "OC(=O)C(N)CC1=C[NH+]=CN1",
    "HIE": "OC(=O)C(N)Cc1cnc[nH]1",
    "HID": "OC(=O)C(N)Cc1c[nH]cn1",
    "HIP": "OC(=O)C(N)CC1=C[NH+]=CN1",
    "CYX": "SCC(N)C(=O)O",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "H", "HA", "HN1", "HN2", "HN3"}

# Canonical PDB atom names in the same order they appear in AMINO_SMILES
# (derived from connectivity, not by eye -- verified atom-by-atom with
# RDKit). Used to build a "PDB-like" molecule (names + PDBResidueInfo) even
# without a real PDB file, e.g. for the standalone-amino-acid path.
AMINO_ATOM_NAMES = {
    "ALA": ["CB", "CA", "N", "C", "O", "OXT"],
    "ARG": ["NH2", "CZ", "NH1", "NE", "CD", "CG", "CB", "CA", "N", "C", "O", "OXT"],
    "ASN": ["ND2", "CG", "OD1", "CB", "CA", "N", "C", "O", "OXT"],
    "ASP": ["OD2", "CG", "OD1", "CB", "CA", "N", "C", "O", "OXT"],
    "CYS": ["SG", "CB", "CA", "N", "C", "O", "OXT"],
    "GLN": ["NE2", "CD", "OE1", "CG", "CB", "CA", "N", "C", "O", "OXT"],
    "GLU": ["OE2", "CD", "OE1", "CG", "CB", "CA", "N", "C", "O", "OXT"],
    "GLY": ["N", "CA", "C", "O", "OXT"],
    "HIS": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "ILE": ["CD1", "CG1", "CB", "CG2", "CA", "N", "C", "O", "OXT"],
    "LEU": ["CD1", "CG", "CD2", "CB", "CA", "N", "C", "O", "OXT"],
    "LYS": ["NZ", "CE", "CD", "CG", "CB", "CA", "N", "C", "O", "OXT"],
    "MET": ["CE", "SD", "CG", "CB", "CA", "N", "C", "O", "OXT"],
    "PHE": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "PRO": ["OXT", "C", "O", "CA", "CB", "CG", "CD", "N"],
    "SER": ["OG", "CB", "CA", "N", "C", "O", "OXT"],
    "THR": ["CG2", "CB", "OG1", "CA", "N", "C", "O", "OXT"],
    "TRP": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "TYR": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2"],
    "VAL": ["CG1", "CB", "CG2", "CA", "N", "C", "O", "OXT"],
    "HSD": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "HSE": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "HSP": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "HIE": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "HID": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "HIP": ["OXT", "C", "O", "CA", "N", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
    "CYX": ["SG", "CB", "CA", "N", "C", "O", "OXT"],
}


@dataclass
class ResidueRecord:
    res_name: str
    res_seq: int
    chain_id: str
    atoms: List[dict]
    mol: Optional[object] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.chain_id}{self.res_seq}_{self.res_name}"


def load_residues_from_pdb(
    pdb_path: str,
    skip_water: bool = True,
    chains: Optional[List[str]] = None,
) -> List[ResidueRecord]:
    residues: dict = {}

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            alt_loc   = line[16].strip()
            res_name  = line[17:20].strip()
            chain_id  = line[21].strip()
            res_seq   = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) > 76 else atom_name[0]

            # altLoc: keep only the primary conformation (blank or "A")
            if alt_loc and alt_loc != "A":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            if skip_water and res_name in ("HOH", "WAT", "TIP3"):
                continue
            if chains and chain_id not in chains:
                continue
            if res_name not in AMINO_SMILES:
                continue

            key = (chain_id, res_seq, res_name)
            if key not in residues:
                residues[key] = ResidueRecord(
                    res_name=res_name,
                    res_seq=res_seq,
                    chain_id=chain_id,
                    atoms=[],
                )
            residues[key].atoms.append({
                "name": atom_name, "x": x, "y": y, "z": z, "element": element
            })

    sorted_keys = sorted(residues.keys(), key=lambda k: (k[0], k[1]))
    records = [residues[k] for k in sorted_keys]

    for rec in records:
        rec.mol = residue_to_mol(rec)

    return records


# Free amino acid backbone H2N-CHR-C(=O)OH. NX3 (not NX3;H2) to also match
# proline (secondary ring nitrogen).
_BACKBONE_FREE_ACID_SMARTS = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)[OX2H1]")

_FREE_TEMPLATE_CACHE: dict = {}
_INTERNAL_TEMPLATE_CACHE: dict = {}


def _get_free_template(res_name: str):
    """Mol from the free amino acid's SMILES (full -COOH carboxyl)."""
    if res_name not in _FREE_TEMPLATE_CACHE:
        _FREE_TEMPLATE_CACHE[res_name] = Chem.MolFromSmiles(AMINO_SMILES[res_name])
    return _FREE_TEMPLATE_CACHE[res_name]


def _get_internal_template(res_name: str):
    """Template for a chain-internal residue: the free SMILES minus the
    backbone carboxyl's terminal hydroxyl."""
    if res_name not in _INTERNAL_TEMPLATE_CACHE:
        free = _get_free_template(res_name)
        match = free.GetSubstructMatch(_BACKBONE_FREE_ACID_SMARTS)
        if not match:
            _INTERNAL_TEMPLATE_CACHE[res_name] = free
        else:
            oh_idx = match[-1]  # last atom of the pattern = terminal -OH
            rw = Chem.RWMol(free)
            rw.RemoveAtom(oh_idx)
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            _INTERNAL_TEMPLATE_CACHE[res_name] = m
    return _INTERNAL_TEMPLATE_CACHE[res_name]


def _is_hydrogen(atom_dict: dict) -> bool:
    elem = atom_dict.get("element", "").strip().upper()
    if elem:
        return elem == "H"
    return atom_dict["name"].strip().upper().startswith("H")


def _atoms_to_pdb_block(rec: "ResidueRecord") -> str:
    """ATOM lines for the residue's real heavy atoms (original names/
    coordinates). Any hydrogens present are excluded here and regenerated
    later with AddHs(addCoords=True)."""
    lines = []
    i = 0
    for a in rec.atoms:
        if _is_hydrogen(a):
            continue
        i += 1
        name = a["name"].strip()
        elem = a.get("element", "").strip() or (name[0] if not name[0].isdigit() else name[1])
        lines.append(
            f"ATOM  {i:5d} {name:<4s} {rec.res_name:<3s} {rec.chain_id:1s}{rec.res_seq:4d}    "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00          {elem:>2s}"
        )
    return "\n".join(lines) + "\nEND\n"


def residue_to_mol(rec: ResidueRecord) -> Optional[object]:
    """Builds the RDKit molecule directly from real PDB atoms (coordinates/
    names correct by construction); bond order and aromaticity assigned by
    comparison against a known template."""
    if rec.res_name not in AMINO_SMILES:
        warnings.warn(f"Residuo sconosciuto: {rec.res_name}")
        return None

    has_oxt = any(a["name"].strip() == "OXT" for a in rec.atoms)
    template = _get_free_template(rec.res_name) if has_oxt else _get_internal_template(rec.res_name)

    pdb_block = _atoms_to_pdb_block(rec)
    mol_from_pdb = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol_from_pdb is None:
        warnings.warn(f"{rec.label}: impossibile costruire la molecola dagli atomi PDB")
        return None

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol_from_pdb)
        Chem.SanitizeMol(mol)
    except Exception as e:
        warnings.warn(
            f"{rec.label}: assegnazione ordini di legame fallita ({e}) — "
            f"probabile residuo con atomi mancanti o non standard, saltato"
        )
        return None

    # hydrogens missing from the X-ray PDB, estimated from the real geometry
    mol = Chem.AddHs(mol, addCoords=True)

    return mol


def mol_from_amino_acid(res_name: str, chain_id: str = "A", res_seq: int = 1):
    """Builds the RDKit molecule for a standalone amino acid (no real PDB):
    free SMILES embedded in 3D (ETKDG) + canonical PDB names
    (AMINO_ATOM_NAMES) attached via PDBResidueInfo. Lets topological_order()
    and assign_abraham_hb_intensities() work the same as with a real PDB,
    without crystallographic coordinates."""
    if res_name not in AMINO_SMILES:
        warnings.warn(f"Residuo sconosciuto: {res_name}")
        return None

    mol = Chem.MolFromSmiles(AMINO_SMILES[res_name])
    if mol is None:
        warnings.warn(f"{res_name}: SMILES non valido")
        return None

    names = AMINO_ATOM_NAMES[res_name]
    if mol.GetNumAtoms() != len(names):
        warnings.warn(f"{res_name}: AMINO_ATOM_NAMES disallineato con lo SMILES")
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    else:
        AllChem.MMFFOptimizeMolecule(mol)

    for idx, name in enumerate(names):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f" {name:<3s}"[:4])
        info.SetResidueName(res_name)
        info.SetChainId(chain_id)
        info.SetResidueNumber(res_seq)
        mol.GetAtomWithIdx(idx).SetMonomerInfo(info)

    return mol


def compute_pocket_centroid(residues: List[ResidueRecord]) -> np.ndarray:
    ca_coords = []
    for rec in residues:
        for atom in rec.atoms:
            if atom["name"] == "CA":
                ca_coords.append([atom["x"], atom["y"], atom["z"]])
                break
    if not ca_coords:
        all_coords = [[a["x"], a["y"], a["z"]] for r in residues for a in r.atoms]
        return np.array(all_coords).mean(axis=0)
    return np.array(ca_coords).mean(axis=0)


def sort_residues_by_distance(
    residues: List[ResidueRecord],
    centroid: np.ndarray,
) -> List[ResidueRecord]:
    def ca_distance(rec: ResidueRecord) -> float:
        for atom in rec.atoms:
            if atom["name"] == "CA":
                ca = np.array([atom["x"], atom["y"], atom["z"]])
                return float(np.linalg.norm(ca - centroid))
        return float("inf")

    return sorted(residues, key=ca_distance)