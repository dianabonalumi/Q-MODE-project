"""
Step 2 — Feature Extraction
============================
Estrae feature farmacofore da un amminoacido usando RDKit:
  - Farmacofori (MolChemicalFeatures): HBD, HBA, Hydrophobe, Aromatic, PosIonizable, NegIonizable
  - H-bond donor / acceptor (Lipinski)
    - Idrofobicità (LogP per atomo tramite Crippen contributions)
    - Legami idrogeno (intensità calcolata in modo continuo, simulando un'energia)

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

# Feature factory di RDKit (API moderna). La vecchia `MolChemicalFeatures`
# non esiste più nelle versioni recenti di RDKit: usare `ChemicalFeatures`.
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

# Specificità farmacoforica: quanto un tipo è informativo/raro. Quando più tipi
# coincidono nello stesso cluster, il tipo PIÙ SPECIFICO deve prevalere — un
# anello aromatico o un gruppo carico non va mascherato dai numerosi atomi
# idrofobici che lo circondano (che sono generici).
FEATURE_SPECIFICITY = {
    "Aromatic":      5,
    "HBondDonor":    4,
    "HBondAcceptor": 4,
    "NegIonizable":  3,
    "PosIonizable":  3,
    "Hydrophobe":    1,
}

# Le famiglie di RDKit (BaseFeatures.fdef) usano nomi diversi dai nostri:
# "Donor"/"Acceptor" invece di "HBondDonor"/"HBondAcceptor", e una famiglia
# "LumpedHydrophobe" per gruppi idrofobici aggregati. Mappiamo sui nostri tipi.
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

# NOTA: l'intensità dei legami idrogeno NON viene più calcolata qui con un
# placeholder. Le feature HBond escono con intensità neutra 1.0; la forza
# geometrica reale (distanza + angolo D–H···A, tra residui diversi) viene
# assegnata a valle da `hbond_geometry.assign_feature_hbond_intensities()`
# quando è disponibile il contesto della tasca (vedi scripts/run_pocket.py).
_HB_NEUTRAL_INTENSITY = 1.0

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

    # Contributi di Crippen al LogP, per atomo pesante (indici allineati a `mol`
    # perché AddHs aggiunge gli H in coda, preservando gli indici degli atomi
    # pesanti). Usati per assegnare un'INTENSITÀ idrofobica continua, non per
    # creare nuove feature (evita di inondare la molecola di siti Hydrophobe).
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

    # ── Farmacofori via Feature Factory ───────────────────────────────────
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
                intensity = _HB_NEUTRAL_INTENSITY   # forza geometrica assegnata a valle
            elif fname == "Hydrophobe":
                intensity = _hydrophobic_intensity(atom_ids)  # LogP di Crippen
            else:
                intensity = 1.0                      # Aromatic / Pos / Neg Ionizable

            features.append(AtomFeature(
                feature_type=fname,
                coords=centroid,
                atom_indices=atom_ids,
                intensity=intensity,
            ))
    else:
        # ── Fallback senza factory: H-bond manuali + Crippen per atomo ─────
        features.extend(_manual_hbond_features(mol, positions))
        for atom_idx, logp in crippen_logp.items():
            if logp > 0.1:
                coord = positions[atom_idx] if atom_idx < len(positions) else positions[0]
                features.append(AtomFeature("Hydrophobe", coord, [atom_idx], intensity=logp))

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
            features.append(AtomFeature("HBondDonor", coord, [idx], intensity=_HB_NEUTRAL_INTENSITY))

        # Acceptor: N o O con lone pair (approssimazione: tutti N e O)
        if symbol in ("N", "O", "F"):
            features.append(AtomFeature("HBondAcceptor", coord, [idx], intensity=_HB_NEUTRAL_INTENSITY))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Utility: crea mol da SMILES
# ─────────────────────────────────────────────────────────────────────────────

def mol_from_smiles(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES non valido: {smiles}")
    return mol
