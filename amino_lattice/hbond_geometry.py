"""
Geometria dei legami idrogeno (sostituisce l'intensità HB placeholder)
======================================================================
Calcola l'intensità di un legame idrogeno in modo FISICO, a partire dalla
geometria della tasca:

    intensità = termine_distanza(d)  ×  termine_angolare(D–H···A)

dove
  - d           = distanza donatore-pesante ··· accettore-pesante (Å)
  - D–H···A     = angolo al donatore (idrogeno tra D e A); ottimale ≈ 180°

Modello (forma standard usata in molti scoring di docking):
  termine_distanza(d) = exp( -(d - d0)² / (2 σ²) )      con d0 = 2.9 Å, σ = 0.45
  termine_angolare    = max(0, -cos(∠ D–H···A))          (1 a 180°, 0 a ≤ 90°)

I legami idrogeno vengono cercati SOLO TRA RESIDUI DIVERSI (un H-bond
intramolecolare dentro un singolo amminoacido non è un'interazione reale di
tasca). Se la struttura non contiene idrogeni espliciti, il termine angolare
viene sostituito da un valore di default morbido (solo distanza).

Razionale scientifico
---------------------
La forza di un legame idrogeno dipende fortemente dalla geometria: cala
rapidamente se la distanza si allontana dall'optimum (~2.9 Å) o se la
disposizione D–H···A si discosta dalla linearità. Codificare questa dipendenza
rende l'asse `hb` dell'encoding quantistico una grandezza fisicamente fondata
invece di un placeholder.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


# Parametri geometrici del modello
D0 = 2.9          # distanza ottimale donatore···accettore (Å)
SIGMA = 0.45      # ampiezza della gaussiana sulla distanza (Å)
D_MIN = 2.2       # sotto questa distanza: clash, non è un H-bond
D_MAX = 3.6       # oltre questa distanza: troppo lontano
NO_H_ANGLE_FACTOR = 0.85   # default angolare se la struttura è priva di H
INTENSITY_SCALE = 3.0      # scala per ottenere magnitudini leggibili
INTENSITY_FLOOR = 0.05     # valore minimo per evitare il collasso del range hb
H_BOND_LEN = 1.3           # soglia per identificare un H legato (Å)


# Atomi (per nome PDB) che possono donare / accettare legami idrogeno
DONOR_NAMES = {
    "N",                                  # ammide del backbone
    "ND1", "ND2", "NE", "NE1", "NE2",     # azoti di sidechain (His, Asn, Arg, Trp, Gln)
    "NH1", "NH2", "NZ",                   # Arg, Lys
    "OG", "OG1", "OH",                    # ossidrili (Ser, Thr, Tyr)
}
ACCEPTOR_NAMES = {
    "O", "OXT",                           # carbonile del backbone
    "OD1", "OD2", "OE1", "OE2",           # carbossili/ammidi (Asp, Glu, Asn, Gln)
    "OG", "OG1", "OH",                    # ossidrili (anche accettori)
    "ND1", "NE2",                         # azoti dell'anello istidinico
}


@dataclass
class HBAtom:
    """Un atomo donatore o accettore nella tasca."""
    res_label: str
    name: str
    pos: np.ndarray
    role: str                    # "donor" | "acceptor"
    h_positions: List[np.ndarray]  # idrogeni legati (vuoto se assenti)
    strength: float = 0.0        # intensità geometrica calcolata


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione donatori / accettori
# ─────────────────────────────────────────────────────────────────────────────

def extract_donor_acceptor_atoms(residues) -> List[HBAtom]:
    """
    Costruisce la lista di atomi donatori/accettori della tasca a partire dai
    ResidueRecord (usa i dict atomici grezzi: name, element, x, y, z).
    Per ogni donatore cerca gli idrogeni legati nello stesso residuo.
    """
    atoms: List[HBAtom] = []
    for rec in residues:
        # idrogeni del residuo, per cercare quelli legati a un donatore
        hydrogens = [
            np.array([a["x"], a["y"], a["z"]])
            for a in rec.atoms
            if a.get("element", "").upper() == "H" or a["name"].startswith("H")
        ]
        for a in rec.atoms:
            pos = np.array([a["x"], a["y"], a["z"]])
            name = a["name"]
            if name in DONOR_NAMES:
                h_pos = [h for h in hydrogens if np.linalg.norm(h - pos) < H_BOND_LEN]
                atoms.append(HBAtom(rec.label, name, pos, "donor", h_pos))
            if name in ACCEPTOR_NAMES:
                atoms.append(HBAtom(rec.label, name, pos, "acceptor", []))
    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# Score geometrico di una coppia donatore-accettore
# ─────────────────────────────────────────────────────────────────────────────

def hbond_pair_score(donor: HBAtom, acceptor: HBAtom) -> float:
    """
    Intensità geometrica del legame idrogeno donor···acceptor.
    Combina termine di distanza (gaussiano attorno a D0) e termine angolare
    (linearità D–H···A). Ritorna 0 se fuori dal range di distanza.
    """
    d = float(np.linalg.norm(donor.pos - acceptor.pos))
    if d < D_MIN or d > D_MAX:
        return 0.0

    dist_term = np.exp(-((d - D0) ** 2) / (2.0 * SIGMA ** 2))

    if donor.h_positions:
        # termine angolare: ottimale quando D, H, A sono collineari (180°)
        best_ang = 0.0
        for H in donor.h_positions:
            u = donor.pos - H          # da H verso il donatore
            v = acceptor.pos - H       # da H verso l'accettore
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-6 or nv < 1e-6:
                continue
            cos_ang = float(np.dot(u, v) / (nu * nv))
            best_ang = max(best_ang, max(0.0, -cos_ang))  # 1 a 180°, 0 a ≤90°
        ang_term = best_ang
    else:
        # struttura senza idrogeni espliciti → solo distanza
        ang_term = NO_H_ANGLE_FACTOR

    return INTENSITY_SCALE * dist_term * ang_term


# ─────────────────────────────────────────────────────────────────────────────
# Calcolo delle intensità a livello di tasca
# ─────────────────────────────────────────────────────────────────────────────

def compute_pocket_hbond_strengths(residues) -> Dict[str, List[HBAtom]]:
    """
    Per ogni atomo donatore/accettore della tasca calcola l'intensità del
    MIGLIOR legame idrogeno che può formare con un atomo complementare di un
    RESIDUO DIVERSO. Ritorna un dizionario {res_label: [HBAtom, ...]}.
    """
    atoms = extract_donor_acceptor_atoms(residues)
    donors = [a for a in atoms if a.role == "donor"]
    acceptors = [a for a in atoms if a.role == "acceptor"]

    for don in donors:
        best = 0.0
        for acc in acceptors:
            if acc.res_label == don.res_label:
                continue   # niente H-bond intramolecolari
            best = max(best, hbond_pair_score(don, acc))
        don.strength = best

    for acc in acceptors:
        best = 0.0
        for don in donors:
            if don.res_label == acc.res_label:
                continue
            best = max(best, hbond_pair_score(don, acc))
        acc.strength = best

    grouped: Dict[str, List[HBAtom]] = {}
    for a in atoms:
        grouped.setdefault(a.res_label, []).append(a)
    return grouped


def assign_feature_hbond_intensities(features, residue_atoms: Optional[List[HBAtom]]):
    """
    Sovrascrive l'intensità delle feature HBondDonor/HBondAcceptor di un residuo
    con la forza geometrica reale, abbinando ciascuna feature all'atomo
    donatore/accettore più vicino (dello stesso ruolo) nel residuo.

    Mutazione in-place delle AtomFeature. Se non c'è contesto geometrico, le
    intensità restano invariate.
    """
    if not residue_atoms:
        return features

    donors = [a for a in residue_atoms if a.role == "donor"]
    acceptors = [a for a in residue_atoms if a.role == "acceptor"]
    role_atoms = {
        "HBondDonor":    donors,
        "HBondAcceptor": acceptors,
        # I gruppi carichi sono forti partecipanti a legami idrogeno /
        # interazioni elettrostatiche: ricevono la stessa intensità geometrica.
        "PosIonizable":  donors,     # ammine cariche (Lys, Arg) → donatori
        "NegIonizable":  acceptors,  # carbossilati (Asp, Glu) → accettori
    }

    for f in features:
        candidates = role_atoms.get(f.feature_type)
        if not candidates:
            continue
        # abbina la feature all'atomo del ruolo giusto più vicino al suo centroide
        nearest = min(candidates, key=lambda a: np.linalg.norm(a.pos - f.coords))
        f.intensity = max(INTENSITY_FLOOR, nearest.strength)

    return features
