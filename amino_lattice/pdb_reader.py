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

            res_name  = line[17:20].strip()
            chain_id  = line[21].strip()
            res_seq   = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) > 76 else atom_name[0]

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


def residue_to_mol(rec: ResidueRecord) -> Optional[object]:
    if rec.res_name not in AMINO_SMILES:
        warnings.warn(f"Residuo sconosciuto: {rec.res_name}")
        return None

    smiles = AMINO_SMILES[rec.res_name]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    pdb_coords = {a["name"]: np.array([a["x"], a["y"], a["z"]]) for a in rec.atoms}

    conf = mol.GetConformer()
    _overlay_coords_by_element(mol, conf, pdb_coords, rec.res_name, rec.atoms)

    return mol


def _overlay_coords_by_element(mol, conf, pdb_coords: dict, res_name: str, atoms_pdb: list):
    """
    Mappa le coordinate PDB sugli atomi RDKit.
    Usa prima il nome canonico (CA, CB, N, O...), poi fallback per elemento
    nel caso in cui un atomo non venga mappato correttamente.
    """
    NAME_TO_ELEMENT = {
        "N": "N", "CA": "C", "C": "C", "O": "O",
        "CB": "C", "CG": "C", "CG1": "C", "CG2": "C",
        "CD": "C", "CD1": "C", "CD2": "C", "CE": "C",
        "CE1": "C", "CE2": "C", "CE3": "C", "CZ": "C",
        "CZ2": "C", "CZ3": "C", "CH2": "C",
        "OG": "O", "OG1": "O", "OD1": "O", "OD2": "O",
        "OE1": "O", "OE2": "O", "OH": "O", "OXT": "O",
        "ND1": "N", "ND2": "N", "NE": "N", "NE1": "N",
        "NE2": "N", "NH1": "N", "NH2": "N", "NZ": "N",
        "SD": "S", "SG": "S",
    }

    used = set()
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]

    pdb_heavy = {k: v for k, v in pdb_coords.items()}

    pdb_order = [name for name in [
        "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2",
        "CD", "CD1", "CD2", "CE", "CE1", "CE2",
        "CZ", "NZ", "OG", "OG1", "OD1", "OD2",
        "OE1", "OE2", "OH", "ND1", "ND2",
        "NE", "NE1", "NE2", "NH1", "NH2",
        "SD", "SG", "CH2", "CZ2", "CZ3", "CE3", "NE1"
    ] if name in pdb_heavy]

    for name in pdb_heavy:
        if name not in pdb_order:
            pdb_order.append(name)

    # Prima passata: assegna per nome canonico
    for i, atom in enumerate(heavy_atoms):
        if i < len(pdb_order):
            name = pdb_order[i]
            if name in pdb_coords:
                coord = pdb_coords[name]
                conf.SetAtomPosition(
                    atom.GetIdx(),
                    (float(coord[0]), float(coord[1]), float(coord[2]))
                )
                used.add(name)

    # Seconda passata: fallback per atomi non mappati
    # Se la coordinata risultante è troppo lontana dal centroide del residuo
    # (indica che è rimasta quella di ETKDGv3), usa l'atomo PDB più vicino
    # per simbolo chimico.
    if atoms_pdb:
        pdb_positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_pdb])
        centroid = pdb_positions.mean(axis=0)

        for atom in heavy_atoms:
            pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
            if np.linalg.norm(pos - centroid) > 15.0:
                symbol = atom.GetSymbol()
                candidates = [a for a in atoms_pdb if a.get("element", "")== symbol]
                if not candidates:
                    # fallback sul primo carattere del nome atomo
                    candidates = [a for a in atoms_pdb if a["name"].strip()[0] == symbol]
                if candidates:
                    dists = [
                        np.linalg.norm(np.array([a["x"], a["y"], a["z"]]) - centroid)
                        for a in candidates
                    ]
                    best = candidates[int(np.argmin(dists))]
                    conf.SetAtomPosition(
                        atom.GetIdx(),
                        (best["x"], best["y"], best["z"])
                    )


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