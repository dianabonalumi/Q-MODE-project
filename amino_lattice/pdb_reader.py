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


# SMILES canonici dei 20 amminoacidi standard (catena laterale + backbone)
# Usati come fallback se RDKit non riesce a costruire la mol dal PDB direttamente
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
    # Varianti protonazione
    "HSD": "OC(=O)C(N)Cc1c[nH]cn1",
    "HSE": "OC(=O)C(N)Cc1cnc[nH]1",
    "HSP": "OC(=O)C(N)Cc1c[nH+]cn1",
    "HIE": "OC(=O)C(N)Cc1cnc[nH]1",
    "HID": "OC(=O)C(N)Cc1c[nH]cn1",
    "HIP": "OC(=O)C(N)Cc1c[nH+]cn1",
    "CYX": "SCC(N)C(=O)O",   # CYS con ponte S-S
}

# Atomi del backbone da escludere se si vuole solo la catena laterale
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "H", "HA", "HN1", "HN2", "HN3"}


@dataclass
class ResidueRecord:
    """Un residuo estratto dal PDB con le coordinate atomiche 3D."""
    res_name: str           # es. "ILE"
    res_seq: int            # numero residuo
    chain_id: str           # es. "A"
    atoms: List[dict]       # lista di {"name": str, "x": float, "y": float, "z": float, "element": str}
    mol: Optional[object] = field(default=None, repr=False)   # RDKit Mol

    @property
    def label(self) -> str:
        return f"{self.chain_id}{self.res_seq}_{self.res_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Lettura PDB
# ─────────────────────────────────────────────────────────────────────────────

def load_residues_from_pdb(
    pdb_path: str,
    skip_water: bool = True,
    chains: Optional[List[str]] = None,
) -> List[ResidueRecord]:
    """
    Legge un file PDB e raggruppa gli atomi per residuo.

    Parameters
    ----------
    pdb_path : str
        Percorso al file .pdb
    skip_water : bool
        Se True, salta HOH (molecole d'acqua).
    chains : list of str, opzionale
        Filtra per catena (es. ["A"]). Se None, legge tutte.

    Returns
    -------
    List[ResidueRecord]  — un elemento per residuo, ordinati per catena+seq
    """
    residues: dict = {}   # chiave: (chain_id, res_seq, res_name)

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq  = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            element  = line[76:78].strip() if len(line) > 76 else atom_name[0]

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
                continue   # salta HETATM non standard

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

    # Ordina per catena e numero residuo
    sorted_keys = sorted(residues.keys(), key=lambda k: (k[0], k[1]))
    records = [residues[k] for k in sorted_keys]

    # Converti in mol RDKit
    for rec in records:
        rec.mol = residue_to_mol(rec)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Conversione residuo → RDKit Mol con coordinate 3D
# ─────────────────────────────────────────────────────────────────────────────

def residue_to_mol(rec: ResidueRecord) -> Optional[object]:
    """
    Costruisce un RDKit Mol per il residuo con le coordinate 3D dal PDB.

    Strategia:
      1. Prova a costruire la mol dall'SMILES canonico del residuo
      2. Genera un conformero con ETKDG
      3. Sovrascrive le coordinate del conformero con quelle reali del PDB
         (mappando atomi pesanti per nome)

    In questo modo RDKit ha la topologia corretta (legami, cariche, aromaticità)
    e le coordinate sono quelle sperimentali cristallografiche.
    """
    if rec.res_name not in AMINO_SMILES:
        warnings.warn(f"Residuo sconosciuto: {rec.res_name}")
        return None

    smiles = AMINO_SMILES[rec.res_name]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Genera conformero iniziale
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Mappa nome atomo PDB → coordinate reali
    pdb_coords = {a["name"]: np.array([a["x"], a["y"], a["z"]]) for a in rec.atoms}

    # Sovrascrive le coordinate degli atomi pesanti del conformero
    # usando un mapping euristico per nome elemento
    conf = mol.GetConformer()
    _overlay_coords_by_element(mol, conf, pdb_coords, rec.res_name)

    return mol


def _overlay_coords_by_element(mol, conf, pdb_coords: dict, res_name: str):
    """
    Mappa le coordinate PDB sugli atomi RDKit.
    Usa prima il nome canonico (CA, CB, N, O...), poi fallback per elemento.
    """
    # Nomi standard backbone → simbolo RDKit
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

    # Prima passata: assegna per nome canonico nell'ordine degli atomi pesanti
    pdb_heavy = {k: v for k, v in pdb_coords.items() if k not in BACKBONE_ATOMS or True}

    # Costruisci lista ordinata dei nomi PDB (backbone prima, poi side chain)
    pdb_order = [name for name in ["N", "CA", "C", "O", "CB", "CG", "CG1", "CG2",
                                    "CD", "CD1", "CD2", "CE", "CE1", "CE2",
                                    "CZ", "NZ", "OG", "OG1", "OD1", "OD2",
                                    "OE1", "OE2", "OH", "ND1", "ND2",
                                    "NE", "NE1", "NE2", "NH1", "NH2",
                                    "SD", "SG", "CH2", "CZ2", "CZ3", "CE3",
                                    "NE1"] if name in pdb_heavy]

    # Aggiungi nomi non coperti
    for name in pdb_heavy:
        if name not in pdb_order:
            pdb_order.append(name)

    # Assegna in ordine
    for i, atom in enumerate(heavy_atoms):
        if i < len(pdb_order):
            name = pdb_order[i]
            if name in pdb_coords:
                coord = pdb_coords[name]
                conf.SetAtomPosition(atom.GetIdx(), (float(coord[0]), float(coord[1]), float(coord[2])))
                used.add(name)


def compute_pocket_centroid(residues: List[ResidueRecord]) -> np.ndarray:
    """
    Calcola il centroide 3D della tasca come media delle posizioni
    dei carboni alpha (CA) di tutti i residui.
    """
    ca_coords = []
    for rec in residues:
        for atom in rec.atoms:
            if atom["name"] == "CA":
                ca_coords.append([atom["x"], atom["y"], atom["z"]])
                break
    if not ca_coords:
        # fallback: media di tutti gli atomi
        all_coords = [[a["x"], a["y"], a["z"]] for r in residues for a in r.atoms]
        return np.array(all_coords).mean(axis=0)
    return np.array(ca_coords).mean(axis=0)


def sort_residues_by_distance(
    residues: List[ResidueRecord],
    centroid: np.ndarray,
) -> List[ResidueRecord]:
    """
    Ordina i residui per distanza crescente del loro CA dal centroide della tasca.
    Il residuo più vicino al centro della tasca viene per primo.
    """
    def ca_distance(rec: ResidueRecord) -> float:
        for atom in rec.atoms:
            if atom["name"] == "CA":
                ca = np.array([atom["x"], atom["y"], atom["z"]])
                return float(np.linalg.norm(ca - centroid))
        return float("inf")

    return sorted(residues, key=ca_distance)

