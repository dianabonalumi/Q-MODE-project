"""
Extracts pharmacophore features from a molecule using RDKit's chemical
feature factory (HBD, HBA, Hydrophobe, Aromatic, PosIonizable, NegIonizable).
Each feature carries the 3D coordinates of its atom (or group centroid).
Hydrophobic intensity comes from per-atom Crippen LogP contributions; H-bond
intensity is filled in later by abraham_hbond.assign_abraham_hb_intensities().

Backbone amide nitrogens are excluded from PosIonizable -- see
_is_backbone_amide_n() for why. Hydrophobic groups are emitted once, as the
aggregate site -- see the LumpedHydrophobe note on FAMILY_MAP.
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

# Standard residue names, reused as the single source of truth from pdb_reader
# (same constant ligand_reader imports). Used to tell a protein backbone atom
# from a ligand atom that happens to be called "N".
from .pdb_reader import AMINO_SMILES as _AMINO_SMILES
STANDARD_RESIDUES = frozenset(_AMINO_SMILES)


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
#
# Both hydrophobe families map to "Hydrophobe", but they are two alternative
# descriptions of the SAME group, not two groups: RDKit reports a valine's
# isopropyl as three atomic Hydrophobe features (CB, CG1, CG2) *and* as one
# LumpedHydrophobe spanning all three. Emitting both double-counts the group
# and, since the aggregate centroid sits among the atoms it lumps, guarantees
# sites closer than the dedup threshold. We keep one site per hydrophobic
# group -- the aggregate -- and drop the atomic features it already covers;
# atomic hydrophobes outside any lumped group are kept as they are.
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

def _drop_subsumed(features: List[AtomFeature]) -> List[AtomFeature]:
    """Drops features whose atoms are already covered by a larger feature of the
    same type, keeping the aggregate.

    RDKit's factory deliberately reports some groups twice, atom-by-atom and as
    a whole: a valine isopropyl comes back as three Hydrophobe/ThreeWayAttach
    plus one LumpedHydrophobe/iPropyl, and an arginine guanidinium as one
    PosN plus one Guanidine. FAMILY_MAP collapses each pair onto a single
    feature type, so without this the group is counted twice -- and since the
    aggregate's centroid sits among the atoms it aggregates, the copies are
    guaranteed to land within the dedup threshold of each other.

    Keeping the aggregate models one pharmacophoric site per functional group,
    which is the granularity the docking model works at. To switch to the
    atom-level view instead, invert the comparison here rather than editing
    FAMILY_MAP -- the two families must stay distinguishable at this point.
    """
    sets = [frozenset(f.atom_indices) for f in features]
    keep = []
    for i, f in enumerate(features):
        if any(j != i
               and features[j].feature_type == f.feature_type
               and sets[i] < sets[j]                     # strict subset
               for j in range(len(features))):
            continue
        keep.append(f)
    return keep


# A carboxylate's two oxygens are equivalent by resonance and both accept
# H-bonds, but RDKit's Acceptor definition only matches the anionic one.
_CARBOXYLATE = Chem.MolFromSmarts("[CX3](=[OX1])[OX1-]")


def _carboxylate_acceptors(mol: Mol, positions, features: List[AtomFeature]) -> List[AtomFeature]:
    """Acceptor features for carboxylate oxygens RDKit's factory skipped.

    Since pdb_reader switched Asp/Glu to their physiological -COO- form, the
    factory reports a single Acceptor (on the anionic oxygen) where the neutral
    -COOH reported two. That drop is an artifact of the SMARTS, not chemistry:
    the negative charge is delocalized over both oxygens and both accept. This
    is generic -- carboxylate ligands get the same treatment.
    """
    have = {tuple(f.atom_indices) for f in features if f.feature_type == "HBondAcceptor"}
    added = []
    for match in mol.GetSubstructMatches(_CARBOXYLATE):
        for o_idx in match[1:]:          # skip the carboxyl carbon
            if (o_idx,) in have:
                continue
            added.append(AtomFeature(
                feature_type="HBondAcceptor",
                coords=positions[o_idx],
                atom_indices=[o_idx],
                intensity=_HB_NEUTRAL_INTENSITY,
            ))
    return added


def _is_backbone_amide_n(mol: Mol, atom_ids: List[int]) -> bool:
    """True if the feature sits entirely on a standard residue's backbone N.

    pdb_reader builds each residue from the *free* amino acid SMILES, where the
    backbone nitrogen is a primary amine (-NH2). _get_internal_template() caps
    the C side (it strips the carboxyl -OH, leaving the peptide-bond acyl
    carbon) but nothing caps the N side, so RDKit's BasicGroup pattern matches
    that free amine and every residue picks up a phantom PosIonizable.

    In the real protein that nitrogen is a peptide-bond amide: its lone pair is
    delocalized onto the carbonyl and it is not protonated at physiological pH
    (pKa ~ -1, vs ~9.5 for a free amine). The H-bond *donor* on the same atom is
    genuine and is deliberately left alone -- only the ionizable character is
    wrong.

    Not handled on purpose: a chain's true N-terminus really is an ammonium and
    gets dropped here too. There is no reliable way to spot it -- the C side has
    OXT, but the N side would need H1/H2/H3, which X-ray structures almost never
    contain, and "lowest res_seq" is meaningless on a cropped pocket. That is at
    most one site per chain against ~1000 phantoms removed.
    """
    if len(atom_ids) != 1:
        return False   # guanidinium / imidazole span several atoms
    info = mol.GetAtomWithIdx(int(atom_ids[0])).GetPDBResidueInfo()
    if info is None:
        return False   # no PDB provenance (SMILES-built mol, ligand): leave it
    return (info.GetName().strip() == "N"
            and info.GetResidueName().strip() in STANDARD_RESIDUES)


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
        rdkit_feats = list(factory.GetFeaturesForMol(mol))

        # atoms already described by an aggregate hydrophobic site
        lumped_atoms = set()
        for f in rdkit_feats:
            if f.GetFamily() == "LumpedHydrophobe":
                lumped_atoms.update(f.GetAtomIds())

        for f in rdkit_feats:
            fname = FAMILY_MAP.get(f.GetFamily())
            if fname is None:
                continue
            atom_ids = list(f.GetAtomIds())

            # phantom PosIonizable on the peptide-bond nitrogen
            if fname == "PosIonizable" and _is_backbone_amide_n(mol, atom_ids):
                continue

            # atomic hydrophobe already covered by its aggregate site
            if (f.GetFamily() == "Hydrophobe"
                    and lumped_atoms.issuperset(atom_ids)):
                continue

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

        features.extend(_carboxylate_acceptors(mol, positions, features))
        features = _drop_subsumed(features)
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
