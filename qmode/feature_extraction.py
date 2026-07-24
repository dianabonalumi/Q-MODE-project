"""
Extracts pharmacophore features from a molecule using RDKit's chemical
feature factory (HBD, HBA, Hydrophobe, Aromatic, PosIonizable, NegIonizable).
Each feature carries the 3D coordinates of its atom (or group centroid).
Hydrophobic intensity comes from per-atom Crippen LogP contributions; H-bond
intensity is filled in later by abraham_hbond.assign_abraham_hb_intensities().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.rdchem import Mol

# RDKit's modern feature factory API. The old `MolChemicalFeatures` is gone
# in recent RDKit versions -- use `ChemicalFeatures` instead.
try:
    from rdkit.Chem import ChemicalFeatures
    _HAS_FACTORY = True
except ImportError:
    _HAS_FACTORY = False

import os
from rdkit import RDConfig
FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")


# ─────────────────────────────────────────────────────────────────────────────
# Struttura dati per una singola feature
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_TYPES = [
    "HBondDonor",
    "HBondAcceptor",
    "Hydrophobe",
    "Aromatic",
    "PosIonizable",
    "NegIonizable",
]

# Indice intero per ogni tipo (usato nel labeling one-hot)
FEATURE_INDEX = {ft: i for i, ft in enumerate(FEATURE_TYPES)}




# RDKit's families (BaseFeatures.fdef) use different names than ours --
# "Donor"/"Acceptor" instead of "HBondDonor"/"HBondAcceptor", plus a
# "LumpedHydrophobe" family for aggregated hydrophobic groups. Map to ours.
FAMILY_MAP = {
    "Donor": "HBondDonor",
    "Acceptor": "HBondAcceptor",
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "Aromatic": "Aromatic",
    "PosIonizable": "PosIonizable",
    "NegIonizable": "NegIonizable",
}


@dataclass
class AtomFeature:
    """A pharmacophore feature localized in 3D space."""
    feature_type: str          # e.g. "HBondDonor"
    coords: np.ndarray         # 3D coordinates (x, y, z)
    atom_indices: List[int]    # indices of the atoms making up the feature
    intensity: float = 1.0     # e.g. partial LogP for Hydrophobe

    def type_index(self) -> int:
        return FEATURE_INDEX.get(self.feature_type, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Funzione principale
# ─────────────────────────────────────────────────────────────────────────────

# HBond features start at neutral intensity 1.0; the real value (Abraham
# scales) is filled in downstream by abraham_hbond.assign_abraham_hb_intensities().
_HB_NEUTRAL_INTENSITY = 1.0

def extract_features(mol: Mol, embed_3d: bool = True) -> List[AtomFeature]:
    """Extracts the list of AtomFeature from an RDKit Mol (with or without
    3D coordinates). If embed_3d and the molecule has none yet, generates an
    ETKDG conformer."""
    mol = Chem.AddHs(mol)

    if embed_3d:
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                # fallback: 2D coordinates lifted to z=0
                rdDepictor.Compute2DCoords(mol)
                conf = mol.GetConformer()
                positions = conf.GetPositions()
            else:
                AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()  # shape (N_atoms, 3)
    else:
        # reuse the existing conformer; fall back to 2D only if there's none at all
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()

    features: List[AtomFeature] = []

    # Crippen LogP contributions per heavy atom (indices line up with `mol`
    # since AddHs appends Hs at the end). Used for a continuous hydrophobic
    # intensity, not to create new features (avoids flooding the molecule
    # with Hydrophobe sites).
    crippen_logp = {}
    try:
        mol_no_h = Chem.RemoveHs(mol)
        for atom_idx, (logp, _mr) in enumerate(rdMolDescriptors._CalcCrippenContribs(mol_no_h)):
            crippen_logp[atom_idx] = float(logp)
    except Exception:
        pass

    def _hydrophobic_intensity(atom_ids):
        vals = [crippen_logp.get(i, 0.0) for i in atom_ids]
        s = sum(v for v in vals if v > 0)
        return s if s > 0 else 1.0

    if _HAS_FACTORY:
        factory = ChemicalFeatures.BuildFeatureFactory(FDEF_PATH)
        rdkit_feats = factory.GetFeaturesForMol(mol)
        for f in rdkit_feats:
            fname = FAMILY_MAP.get(f.GetFamily())
            if fname is None:
                continue
            atom_ids = list(f.GetAtomIds())
            centroid = positions[atom_ids].mean(axis=0)

            if fname in ("HBondDonor", "HBondAcceptor"):
                intensity = _HB_NEUTRAL_INTENSITY   # real strength assigned downstream
            elif fname == "Hydrophobe":
                intensity = _hydrophobic_intensity(atom_ids)  # Crippen LogP
            else:
                intensity = 1.0                      # Aromatic / Pos / Neg Ionizable

            features.append(AtomFeature(
                feature_type=fname,
                coords=centroid,
                atom_indices=atom_ids,
                intensity=intensity,
            ))
    else:
        # fallback without the factory: manual H-bond detection + per-atom Crippen
        features.extend(_manual_hbond_features(mol, positions))
        for atom_idx, logp in crippen_logp.items():
            if logp > 0.1:
                coord = positions[atom_idx] if atom_idx < len(positions) else positions[0]
                features.append(AtomFeature("Hydrophobe", coord, [atom_idx], intensity=logp))

    return features


def _manual_hbond_features(mol: Mol, positions: np.ndarray) -> List[AtomFeature]:
    """Manual H-bond donor/acceptor extraction, used when the feature factory is unavailable."""
    features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        coord = positions[idx]

        # donor: N or O with at least one H
        if symbol in ("N", "O") and atom.GetTotalNumHs() > 0:
            features.append(AtomFeature("HBondDonor", coord, [idx], intensity=_HB_NEUTRAL_INTENSITY))

        # acceptor: N or O with a lone pair (approximation: all N and O)
        if symbol in ("N", "O", "F"):
            features.append(AtomFeature("HBondAcceptor", coord, [idx], intensity=_HB_NEUTRAL_INTENSITY))

    return features


def mol_from_smiles(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES non valido: {smiles}")
    return mol
