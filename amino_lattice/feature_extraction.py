"""
Step 2 — Feature Extraction
============================
Estrae feature farmacofore da un amminoacido usando RDKit:
  - Farmacofori (MolChemicalFeatures): HBD, HBA, Hydrophobe, Aromatic, PosIonizable, NegIonizable
  - H-bond donor / acceptor (Lipinski)
  - Idrofobicità (LogP per atomo tramite Crippen contributions)

Ogni feature è associata alle coordinate 3D dell'atomo (o centroide del gruppo).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.rdchem import Mol

# Feature factories built-in di RDKit
try:
    from rdkit.Chem import MolChemicalFeatures
    from rdkit.Chem.Features.FeatDirUtilsRD import GetIonizable
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


@dataclass
class AtomFeature:
    """Rappresenta una feature farmacofora localizzata nello spazio 3D."""
    feature_type: str          # es. "HBondDonor"
    coords: np.ndarray         # coordinate 3D (x, y, z)
    atom_indices: List[int]    # indici degli atomi che la costituiscono
    intensity: float = 1.0    # es. LogP parziale per Hydrophobe

    def type_index(self) -> int:
        return FEATURE_INDEX.get(self.feature_type, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Funzione principale
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(mol: Mol, embed_3d: bool = True) -> List[AtomFeature]:
    """
    Dato un oggetto RDKit Mol (con o senza coordinate 3D), restituisce
    la lista di AtomFeature estratte.

    Parameters
    ----------
    mol : Mol
        Molecola RDKit. Può essere ottenuta da SMILES o da file SDF/PDB.
    embed_3d : bool
        Se True e la molecola non ha già coordinate 3D, genera un conformero
        con ETKDG. Necessario per coordinate spaziali accurate.

    Returns
    -------
    List[AtomFeature]
    """
    mol = Chem.AddHs(mol)

    # ── Generazione coordinate 3D ──────────────────────────────────────────
    if embed_3d:
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                # fallback: coordinate 2D alzate a z=0
                rdDepictor.Compute2DCoords(mol)
                conf = mol.GetConformer()
                positions = conf.GetPositions()
            else:
                AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()  # shape (N_atoms, 3)
    else:
        # coordinate 2D come surrogate
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()

    features: List[AtomFeature] = []

    # ── Farmacofori via Feature Factory ───────────────────────────────────
    if _HAS_FACTORY:
        factory = MolChemicalFeatures.BuildFeatureFactory(FDEF_PATH)
        rdkit_feats = factory.GetFeaturesForMol(mol)
        for f in rdkit_feats:
            fname = f.GetFamily()
            if fname not in FEATURE_TYPES:
                continue
            atom_ids = list(f.GetAtomIds())
            centroid = positions[atom_ids].mean(axis=0)
            features.append(AtomFeature(
                feature_type=fname,
                coords=centroid,
                atom_indices=atom_ids,
            ))
    else:
        # fallback manuale se la factory non è disponibile
        features.extend(_manual_hbond_features(mol, positions))

    # ── Idrofobicità (Crippen per atomo) ──────────────────────────────────
    mol_no_h = Chem.RemoveHs(mol)
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol_no_h)
        for atom_idx, (logp, _mr) in enumerate(contribs):
            if logp > 0.1:  # soglia per considerare l'atomo idrofobico
                atom = mol_no_h.GetAtomWithIdx(atom_idx)
                # mappa indice back to mol con H (approssimato: stesso idx se atomo pesante)
                coord = positions[atom_idx] if atom_idx < len(positions) else positions[0]
                features.append(AtomFeature(
                    feature_type="Hydrophobe",
                    coords=coord,
                    atom_indices=[atom_idx],
                    intensity=float(logp),
                ))
    except Exception:
        pass  # alcuni amminoacidi molto semplici possono non avere contributi

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Fallback manuale H-bond
# ─────────────────────────────────────────────────────────────────────────────

def _manual_hbond_features(mol: Mol, positions: np.ndarray) -> List[AtomFeature]:
    """Estrazione manuale di H-bond donor/acceptor senza feature factory."""
    features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        coord = positions[idx]

        # Donor: N o O con almeno un H
        if symbol in ("N", "O") and atom.GetTotalNumHs() > 0:
            features.append(AtomFeature("HBondDonor", coord, [idx]))

        # Acceptor: N o O con lone pair (approssimazione: tutti N e O)
        if symbol in ("N", "O", "F"):
            features.append(AtomFeature("HBondAcceptor", coord, [idx]))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Utility: crea mol da SMILES
# ─────────────────────────────────────────────────────────────────────────────

def mol_from_smiles(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES non valido: {smiles}")
    return mol
