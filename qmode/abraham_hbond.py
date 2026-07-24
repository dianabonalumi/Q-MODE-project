"""
H-bond intensity (hb) from Abraham LSER scales, intrinsic to each site
(alpha2H donor, beta2H acceptor) rather than to a site-partner pair.
Values verified against the UFZ-LSER Database (https://www.ufz.de/lserd),
not transcribed from Abraham & Platts (2001).

Ionized groups (Asp/Glu carboxylate, Lys ammonium, Arg guanidinium): Abraham
only covers neutral solutes, so we proxy with the nearest neutral analog
(underestimates the real strength) -- flagged on each affected entry.

Ligands: the table is keyed by (res_name, pdb_atom_name), which a ligand
never matches. _heuristic_group_key() covers that case by classifying the
atom's local environment (symbol/valence/neighbors), limited to carboxyl,
amine, phenol and thiol -- cheaper than a general SMARTS matcher but misses
anything else, which stays at the neutral default.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_extraction import AtomFeature


@dataclass(frozen=True)
class AbrahamGroup:
    alpha2H: Optional[float]   # donor strength; None if the role doesn't apply
    beta2H: Optional[float]    # acceptor strength; None if the role doesn't apply
    model_compound: str
    reference: str
    note: str = ""


# Key: (res_name, pdb_atom_name). "*" as res_name covers backbone atoms,
# shared by all standard amino acids.
ABRAHAM_TABLE: Dict[Tuple[str, str], AbrahamGroup] = {

    # Backbone (all amino acids)
    ("*", "N"): AbrahamGroup(
        alpha2H=0.40, beta2H=None,
        model_compound="N-methylacetamide", reference="Abraham (1994)"),
    ("*", "O"): AbrahamGroup(
        alpha2H=None, beta2H=0.72,
        model_compound="N-methylacetamide", reference="Abraham (1994)"),
    ("*", "OXT"): AbrahamGroup(
        alpha2H=0.62, beta2H=None,
        model_compound="acetic acid (free C-terminal carboxyl)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)"),

    # Asn / Gln: primary amide side chain
    ("ASN", "ND2"): AbrahamGroup(
        alpha2H=0.55, beta2H=None,
        model_compound="acetamide", reference="Abraham (2010/2012)"),
    ("ASN", "OD1"): AbrahamGroup(
        alpha2H=None, beta2H=0.69,
        model_compound="acetamide", reference="Abraham (2010/2012)"),
    ("GLN", "NE2"): AbrahamGroup(
        alpha2H=0.55, beta2H=None,
        model_compound="acetamide", reference="Abraham (2010/2012)"),
    ("GLN", "OE1"): AbrahamGroup(
        alpha2H=None, beta2H=0.69,
        model_compound="acetamide", reference="Abraham (2010/2012)"),

    # Asp / Glu: carboxylate (acceptor only)
    ("ASP", "OD1"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acetic acid (neutral proxy for -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: neutral carbonyl, not the real carboxylate anion"),
    ("ASP", "OD2"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acetic acid (neutral proxy for -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: neutral carbonyl, not the real carboxylate anion"),
    ("GLU", "OE1"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acetic acid (neutral proxy for -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: neutral carbonyl, not the real carboxylate anion"),
    ("GLU", "OE2"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acetic acid (neutral proxy for -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: neutral carbonyl, not the real carboxylate anion"),

    # Lys: ammonium (donor only)
    ("LYS", "NZ"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propylamine (neutral proxy for -NH3+)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: neutral amine, not the real ammonium cation"),

    # Arg: guanidinium (donor only) -- no Abraham data for guanidine/guanidinium
    # (checked against UFZ-LSER: no entry), reuses the amine proxy.
    ("ARG", "NE"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propylamine (proxy, no guanidinium data)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: no direct Abraham value for guanidinium"),
    ("ARG", "NH1"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propylamine (proxy, no guanidinium data)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: no direct Abraham value for guanidinium"),
    ("ARG", "NH2"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propylamine (proxy, no guanidinium data)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: no direct Abraham value for guanidinium"),

    # Ser / Thr: aliphatic hydroxyl (both roles)
    ("SER", "OG"): AbrahamGroup(
        alpha2H=0.37, beta2H=0.48,
        model_compound="1-propanol", reference="Abraham (1994/2006/2012)"),
    ("THR", "OG1"): AbrahamGroup(
        alpha2H=0.37, beta2H=0.48,
        model_compound="1-propanol", reference="Abraham (1994/2006/2012)"),

    # Tyr: phenolic hydroxyl (both roles)
    ("TYR", "OH"): AbrahamGroup(
        alpha2H=0.60, beta2H=0.30,
        model_compound="phenol", reference="Abraham (1994/2000/2004/2010)"),

    # His: imidazole ring (ambiguous, both roles)
    ("HIS", "ND1"): AbrahamGroup(
        alpha2H=0.42, beta2H=0.78,
        model_compound="imidazole", reference="Abraham (1993/1994)",
        note="Ambiguous (tautomer-dependent): using the whole-molecule value"),
    ("HIS", "NE2"): AbrahamGroup(
        alpha2H=0.42, beta2H=0.78,
        model_compound="imidazole", reference="Abraham (1993/1994)",
        note="Ambiguous (tautomer-dependent): using the whole-molecule value"),

    # Trp: indole N-H (donor only -- the lone pair is delocalized into the
    # ring, so it isn't a real acceptor)
    ("TRP", "NE1"): AbrahamGroup(
        alpha2H=0.44, beta2H=None,
        model_compound="indole", reference="Abraham (1993/1994)"),

    # Cys: thiol (both roles, very weak -- not in the technical note's
    # Table A.3, added for completeness)
    ("CYS", "SG"): AbrahamGroup(
        alpha2H=0.00, beta2H=0.12,
        model_compound="methanethiol", reference="Abraham (2007)"),
}

# Histidine protonation variants used by CHARMM/AMBER force fields (see
# pdb_reader.AMINO_SMILES): same imidazole ring as HIS.
for _his_variant in ("HSD", "HSE", "HSP", "HIE", "HID", "HIP"):
    ABRAHAM_TABLE[(_his_variant, "ND1")] = ABRAHAM_TABLE[("HIS", "ND1")]
    ABRAHAM_TABLE[(_his_variant, "NE2")] = ABRAHAM_TABLE[("HIS", "NE2")]


# Pos/NegIonizable count as H-bond donors/acceptors (protonated Lys/Arg,
# deprotonated Asp/Glu).
_ROLE_FOR_FEATURE_TYPE = {
    "HBondDonor": "alpha2H",
    "HBondAcceptor": "beta2H",
    "PosIonizable": "alpha2H",
    "NegIonizable": "beta2H",
}


def _atom_names_for_feature(feat: AtomFeature, mol) -> List[str]:
    """Real PDB atom names for a feature's atoms (via PDBResidueInfo).
    A feature can span multiple atoms (e.g. NegIonizable on -COO-: C+O+O);
    each name is tried until one is found in the table."""
    names = []
    for idx in feat.atom_indices:
        if idx >= mol.GetNumAtoms():
            continue
        info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if info is not None:
            names.append(info.GetName().strip())
    return names


def _nearest_atom_name(coords: np.ndarray, atom_records: List[dict]) -> Optional[str]:
    """Fallback: name of the closest PDB atom to a feature's coordinates,
    for atoms without a PDBResidueInfo."""
    if not atom_records:
        return None
    positions = np.array([[a["x"], a["y"], a["z"]] for a in atom_records])
    dists = np.linalg.norm(positions - coords, axis=1)
    return atom_records[int(np.argmin(dists))]["name"].strip()


def _heuristic_group_key(feat: AtomFeature, mol) -> Optional[Tuple[str, str]]:
    """Fallback for sites whose (res_name, atom_name) isn't in ABRAHAM_TABLE
    (typically ligand atoms, e.g. "BNZ"+"C1"): classify the atom's local
    chemical group and reuse the closest protein proxy already in the table."""
    for idx in feat.atom_indices:
        if idx >= mol.GetNumAtoms():
            continue
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]

        if symbol == "O" and len(heavy_neighbors) == 1 and heavy_neighbors[0].GetSymbol() == "C":
            carbon = heavy_neighbors[0]
            if carbon.GetIsAromatic():
                return ("TYR", "OH")  # phenolic hydroxyl
            other_o = [n for n in carbon.GetNeighbors()
                       if n.GetSymbol() == "O" and n.GetIdx() != idx]
            if other_o:
                return ("ASP", "OD1")  # carboxyl / carboxylate

        elif symbol == "N" and not atom.GetIsAromatic() and len(heavy_neighbors) <= 3:
            is_amide = any(
                n.GetSymbol() == "C" and any(
                    b.GetSymbol() == "O"
                    and mol.GetBondBetweenAtoms(n.GetIdx(), b.GetIdx()).GetBondTypeAsDouble() == 2.0
                    for b in n.GetNeighbors()
                )
                for n in heavy_neighbors
            )
            if not is_amide:
                return ("LYS", "NZ")  # amine

        elif symbol == "S" and len(heavy_neighbors) == 1 and heavy_neighbors[0].GetSymbol() == "C":
            return ("CYS", "SG")  # thiol

    return None


def assign_abraham_hb_intensities(
    features: List[AtomFeature],
    res_name: str,
    mol,
    atom_records: Optional[List[dict]] = None,
) -> List[AtomFeature]:
    """Assign hb intensity (alpha2H for donors/PosIonizable, beta2H for
    acceptors/NegIonizable) from the Abraham table. Mutates in place; sites
    with no table match keep the neutral intensity from extract_features()."""
    for feat in features:
        role = _ROLE_FOR_FEATURE_TYPE.get(feat.feature_type)
        if role is None:
            continue

        candidate_names = _atom_names_for_feature(feat, mol)
        if not candidate_names and atom_records:
            nearest = _nearest_atom_name(feat.coords, atom_records)
            if nearest:
                candidate_names = [nearest]

        matched = False
        for atom_name in candidate_names:
            group = ABRAHAM_TABLE.get((res_name, atom_name)) or ABRAHAM_TABLE.get(("*", atom_name))
            if group is None:
                continue
            value = getattr(group, role)
            if value is not None:
                feat.intensity = value
                matched = True
                break

        if not matched:
            group_key = _heuristic_group_key(feat, mol)
            group = ABRAHAM_TABLE.get(group_key) if group_key else None
            if group is not None:
                value = getattr(group, role)
                if value is not None:
                    feat.intensity = value

    return features
