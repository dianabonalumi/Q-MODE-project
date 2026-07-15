"""
PDB Reader — da file PDB a residui per la pipeline
====================================================
Legge un file PDB (proteina intera o tasca) e restituisce
i residui come oggetti RDKit Mol, pronti per feature_extraction.

Funzioni principali:
  - load_residues_from_pdb()  → lista di ResidueRecord
  - residue_to_mol()          → RDKit Mol con coordinate 3D reali dal PDB
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
    "HSP": "OC(=O)C(N)Cc1c[nH+]cn1",
    "HIE": "OC(=O)C(N)Cc1cnc[nH]1",
    "HID": "OC(=O)C(N)Cc1c[nH]cn1",
    "HIP": "OC(=O)C(N)Cc1c[nH+]cn1",
    "CYX": "SCC(N)C(=O)O",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "H", "HA", "HN1", "HN2", "HN3"}


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

            # altLoc: tieni solo la conformazione primaria (vuota o "A")
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


# Backbone dell'amminoacido libero H2N-CHR-C(=O)OH. NX3 (non NX3;H2) per
# includere anche la Prolina (azoto secondario nell'anello).
_BACKBONE_FREE_ACID_SMARTS = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)[OX2H1]")

_FREE_TEMPLATE_CACHE: dict = {}
_INTERNAL_TEMPLATE_CACHE: dict = {}


def _get_free_template(res_name: str):
    """Mol dallo SMILES dell'amminoacido libero (con carbossile -COOH intero)."""
    if res_name not in _FREE_TEMPLATE_CACHE:
        _FREE_TEMPLATE_CACHE[res_name] = Chem.MolFromSmiles(AMINO_SMILES[res_name])
    return _FREE_TEMPLATE_CACHE[res_name]


def _get_internal_template(res_name: str):
    """Template per un residuo interno alla catena: SMILES libero meno
    l'ossidrile terminale del carbossile di backbone."""
    if res_name not in _INTERNAL_TEMPLATE_CACHE:
        free = _get_free_template(res_name)
        match = free.GetSubstructMatch(_BACKBONE_FREE_ACID_SMARTS)
        if not match:
            _INTERNAL_TEMPLATE_CACHE[res_name] = free
        else:
            oh_idx = match[-1]  # ultimo atomo del pattern = -OH terminale
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
    """Righe ATOM con gli atomi pesanti reali del residuo (nomi/coordinate
    originali). Gli idrogeni, se presenti, vengono esclusi qui e rigenerati
    dopo con AddHs(addCoords=True)."""
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
    """Costruisce la molecola RDKit direttamente dagli atomi reali del PDB
    (coordinate/nomi corretti per costruzione); ordine dei legami e
    aromaticità assegnati per confronto con un template noto."""
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

    # idrogeni assenti nel PDB a raggi X, stimati dalla geometria reale
    mol = Chem.AddHs(mol, addCoords=True)

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