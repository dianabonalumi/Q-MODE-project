"""
Intensità di legame idrogeno (hb) — scale di Abraham.
Sostituisce il calcolo geometrico di hbond_geometry.py con valori
INTRINSECI (alpha2H donatore, beta2H accettore), proprietà del sito e non
della coppia sito-partner. Valori verificati su UFZ-LSER Database
(https://www.ufz.de/lserd), non trascritti da Abraham & Platts (2001).

Gruppi ionizzati (carbossilato Asp/Glu, ammonio Lys, guanidinio Arg): la
scala di Abraham copre solo soluti neutri, quindi si usa come proxy
l'analogo neutro più vicino (sottostima la forza reale) — segnalato in
ogni voce interessata.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_extraction import AtomFeature


@dataclass(frozen=True)
class AbrahamGroup:
    alpha2H: Optional[float]   # forza come donatore; None se il ruolo non si applica
    beta2H: Optional[float]    # forza come accettore; None se il ruolo non si applica
    model_compound: str
    reference: str
    note: str = ""


# Chiave: (res_name, nome_atomo_PDB). "*" come res_name copre il backbone,
# comune a tutti gli amminoacidi standard.
ABRAHAM_TABLE: Dict[Tuple[str, str], AbrahamGroup] = {

    # ── Backbone (tutti gli amminoacidi) ──────────────────────────────
    ("*", "N"): AbrahamGroup(
        alpha2H=0.40, beta2H=None,
        model_compound="N-metilacetamide", reference="Abraham (1994)"),
    ("*", "O"): AbrahamGroup(
        alpha2H=None, beta2H=0.72,
        model_compound="N-metilacetamide", reference="Abraham (1994)"),

    # ── Asn / Gln: ammide primaria laterale ────────────────────────────
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

    # ── Asp / Glu: carbossilato (SEMPRE accettore) ─────────────────────
    ("ASP", "OD1"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acido acetico (proxy neutro di -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: carbonile neutro, non l'anione carbossilato reale"),
    ("ASP", "OD2"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acido acetico (proxy neutro di -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: carbonile neutro, non l'anione carbossilato reale"),
    ("GLU", "OE1"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acido acetico (proxy neutro di -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: carbonile neutro, non l'anione carbossilato reale"),
    ("GLU", "OE2"): AbrahamGroup(
        alpha2H=None, beta2H=0.44,
        model_compound="acido acetico (proxy neutro di -COO-)",
        reference="Abraham (2003/2005/2006/2007/2010/2012)",
        note="PROXY: carbonile neutro, non l'anione carbossilato reale"),

    # ── Lys: ammonio (SEMPRE donatore) ─────────────────────────────────
    ("LYS", "NZ"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propilammina (proxy neutro di -NH3+)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: ammina neutra, non il catione ammonio reale"),

    # ── Arg: guanidinio (SEMPRE donatore) ──────────────────────────────
    # Nessun dato Abraham per guanidina/guanidinio (verificato sulla
    # UFZ-LSER Database: nessuna voce); si riusa il proxy amminico.
    ("ARG", "NE"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propilammina (proxy, nessun dato per guanidinio)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: nessun valore Abraham diretto per il guanidinio"),
    ("ARG", "NH1"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propilammina (proxy, nessun dato per guanidinio)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: nessun valore Abraham diretto per il guanidinio"),
    ("ARG", "NH2"): AbrahamGroup(
        alpha2H=0.16, beta2H=None,
        model_compound="propilammina (proxy, nessun dato per guanidinio)",
        reference="Abraham (1994/2007/2009/2010)",
        note="PROXY: nessun valore Abraham diretto per il guanidinio"),

    # ── Ser / Thr: ossidrile alifatico (entrambi i ruoli) ──────────────
    ("SER", "OG"): AbrahamGroup(
        alpha2H=0.37, beta2H=0.48,
        model_compound="1-propanolo", reference="Abraham (1994/2006/2012)"),
    ("THR", "OG1"): AbrahamGroup(
        alpha2H=0.37, beta2H=0.48,
        model_compound="1-propanolo", reference="Abraham (1994/2006/2012)"),

    # ── Tyr: ossidrile fenolico (entrambi i ruoli) ─────────────────────
    ("TYR", "OH"): AbrahamGroup(
        alpha2H=0.60, beta2H=0.30,
        model_compound="fenolo", reference="Abraham (1994/2000/2004/2010)"),

    # ── His: anello imidazolico (ambiguo, entrambi i ruoli) ────────────
    ("HIS", "ND1"): AbrahamGroup(
        alpha2H=0.42, beta2H=0.78,
        model_compound="imidazolo", reference="Abraham (1993/1994)",
        note="Ambiguo (dipende dal tautomero): usato il valore della molecola intera"),
    ("HIS", "NE2"): AbrahamGroup(
        alpha2H=0.42, beta2H=0.78,
        model_compound="imidazolo", reference="Abraham (1993/1994)",
        note="Ambiguo (dipende dal tautomero): usato il valore della molecola intera"),

    # ── Trp: N-H indolico (solo donatore: il lone pair è delocalizzato
    #        nell'anello, non è un accettore reale) ─────────────────────
    ("TRP", "NE1"): AbrahamGroup(
        alpha2H=0.44, beta2H=None,
        model_compound="indolo", reference="Abraham (1993/1994)"),

    # ── Cys: tiolo (entrambi i ruoli, molto deboli — non in Tabella A.3
    #        della nota tecnica, aggiunto per completezza) ──────────────
    ("CYS", "SG"): AbrahamGroup(
        alpha2H=0.00, beta2H=0.12,
        model_compound="metantiolo", reference="Abraham (2007)"),
}

# Varianti di protonazione dell'istidina usate nei force field CHARMM/AMBER
# (vedi pdb_reader.AMINO_SMILES): stesso anello imidazolico di HIS.
for _his_variant in ("HSD", "HSE", "HSP", "HIE", "HID", "HIP"):
    ABRAHAM_TABLE[(_his_variant, "ND1")] = ABRAHAM_TABLE[("HIS", "ND1")]
    ABRAHAM_TABLE[(_his_variant, "NE2")] = ABRAHAM_TABLE[("HIS", "NE2")]


# Pos/NegIonizable contano come donatori/accettori H-bond a tutti gli
# effetti (Lys/Arg protonati, Asp/Glu deprotonati).
_ROLE_FOR_FEATURE_TYPE = {
    "HBondDonor": "alpha2H",
    "HBondAcceptor": "beta2H",
    "PosIonizable": "alpha2H",
    "NegIonizable": "beta2H",
}


def _atom_names_for_feature(feat: AtomFeature, mol) -> List[str]:
    """Nomi PDB reali degli atomi di una feature (via PDBResidueInfo).
    Una feature può coprire più atomi (es. NegIonizable su -COO-: C+O+O);
    si prova ciascun nome finché non se ne trova uno in tabella."""
    names = []
    for idx in feat.atom_indices:
        if idx >= mol.GetNumAtoms():
            continue
        info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if info is not None:
            names.append(info.GetName().strip())
    return names


def _nearest_atom_name(coords: np.ndarray, atom_records: List[dict]) -> Optional[str]:
    """Fallback: nome dell'atomo PDB più vicino alle coordinate di una
    feature, per i casi in cui l'atomo non abbia un PDBResidueInfo."""
    if not atom_records:
        return None
    positions = np.array([[a["x"], a["y"], a["z"]] for a in atom_records])
    dists = np.linalg.norm(positions - coords, axis=1)
    return atom_records[int(np.argmin(dists))]["name"].strip()


def assign_abraham_hb_intensities(
    features: List[AtomFeature],
    res_name: str,
    mol,
    atom_records: Optional[List[dict]] = None,
) -> List[AtomFeature]:
    """Assegna l'intensità hb (alpha2H donatori/PosIonizable, beta2H
    accettori/NegIonizable) leggendo il valore dalla tabella di Abraham.
    Mutazione in-place; se il gruppo non è in tabella l'intensità resta
    quella neutra di extract_features()."""
    for feat in features:
        role = _ROLE_FOR_FEATURE_TYPE.get(feat.feature_type)
        if role is None:
            continue

        candidate_names = _atom_names_for_feature(feat, mol)
        if not candidate_names and atom_records:
            nearest = _nearest_atom_name(feat.coords, atom_records)
            if nearest:
                candidate_names = [nearest]

        for atom_name in candidate_names:
            group = ABRAHAM_TABLE.get((res_name, atom_name)) or ABRAHAM_TABLE.get(("*", atom_name))
            if group is None:
                continue
            value = getattr(group, role)
            if value is not None:
                feat.intensity = value
                break

    return features
