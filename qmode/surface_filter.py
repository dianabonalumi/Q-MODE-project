"""
Filters pharmacophore features by solvent accessibility (SASA): buried
features aren't physically reachable by a ligand and just add noise. SASA
computed via BioPython's Shrake & Rupley implementation; an atom counts as
exposed above the threshold, and a feature is kept if any of its atoms is.
Default threshold 1.0 A^2 -- inclusive, but drops fully buried atoms
(literature range: 1.0 A^2 conservative to 5.0 A^2 for significant exposure).
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np

from .feature_extraction import AtomFeature

try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.SASA import ShrakeRupley
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False


def compute_atom_sasa(pdb_path: str) -> Dict[tuple, float]:
    """Per-atom SASA for the PDB structure (Shrake & Rupley, probe radius
    1.4 A). Returns {(chain_id, res_seq, atom_name): sasa_value_angstrom2}."""
    if not _HAS_BIOPYTHON:
        raise ImportError(
            "BioPython non installato. Esegui: pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    sr = ShrakeRupley()
    sr.compute(structure, level="A")   # level="A" -> per atom

    sasa_map = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_seq = residue.get_id()[1]
                chain_id = chain.get_id()
                for atom in residue:
                    key = (chain_id, res_seq, atom.get_name().strip())
                    sasa_map[key] = atom.sasa

    return sasa_map


def filter_surface_features(
    features: List[AtomFeature],
    mol,
    chain_id: str,
    res_seq: int,
    sasa_map: Dict[tuple, float],
    sasa_threshold: float = 1.0,
) -> List[AtomFeature]:
    """Keeps only features with at least one solvent-exposed atom
    (SASA > sasa_threshold). `features` are for a single residue;
    `sasa_map` comes from compute_atom_sasa() run on the full structure."""
    from rdkit.Chem import RemoveHs

    mol_no_h = RemoveHs(mol)
    atoms = list(mol_no_h.GetAtoms())

    filtered = []
    for feat in features:
        exposed = False
        for atom_idx in feat.atom_indices:
            if atom_idx >= len(atoms):
                continue
            atom_name = atoms[atom_idx].GetMonomerInfo()
            if atom_name is not None:
                name = atom_name.GetName().strip()
            else:
                name = atoms[atom_idx].GetSymbol()   # fallback: element symbol

            key = (chain_id, res_seq, name)
            sasa_val = sasa_map.get(key, 0.0)
            if sasa_val > sasa_threshold:
                exposed = True
                break

        if exposed:
            filtered.append(feat)

    return filtered


def filter_surface_features_by_coords(
    features: List[AtomFeature],
    sasa_map: Dict[tuple, float],
    chain_id: str,
    res_seq: int,
    all_atom_coords: List[dict],
    sasa_threshold: float = 1.0,
) -> List[AtomFeature]:
    """Matches features to SASA by nearest PDB atom coordinates instead of
    RDKit atom indices -- more robust when the index-to-name mapping is
    uncertain. `all_atom_coords` is the residue's {name, x, y, z} list
    (ResidueRecord.atoms)."""
    if not all_atom_coords:
        return features

    atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in all_atom_coords])
    atom_names = [a["name"].strip() for a in all_atom_coords]

    filtered = []
    for feat in features:
        # nearest PDB atom to the feature's centroid
        dists = np.linalg.norm(atom_positions - feat.coords, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_name = atom_names[nearest_idx]

        key = (chain_id, res_seq, nearest_name)
        sasa_val = sasa_map.get(key, 0.0)

        if sasa_val > sasa_threshold:
            filtered.append(feat)

    return filtered