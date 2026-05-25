"""
Qubit Chain
===========
Converte la sequenza flat di siti farmacofori in una catena di qubit binari.

Logica:
  1. La flat_chain (ordinata per distanza dal centroide) viene suddivisa
     in segmenti di N residui contigui.
  2. Per ogni segmento si contano i siti di tipo HBondDonor o HBondAcceptor.
  3. Se il conteggio >= soglia → qubit = |1⟩  (sito di interazione attivo)
     altrimenti              → qubit = |0⟩  (sito inattivo)

Parametri default (modificabili):
  - residues_per_segment = 2
  - threshold            = 2  (HBD + HBA >= 2 → |1⟩)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


POLAR_TYPES = {"HBondDonor", "HBondAcceptor"}


@dataclass
class QubitSegment:
    """Un segmento = un qubit."""
    segment_idx: int           # indice del segmento (0-based)
    residues: List[str]        # nomi dei residui nel segmento (es. ["A188_CYS", "A204_HIS"])
    n_sites: int               # numero totale di siti nel segmento
    n_polar: int               # numero di siti HBD + HBA nel segmento
    qubit: int                 # 0 o 1

    def __repr__(self):
        state = "|1⟩" if self.qubit == 1 else "|0⟩"
        return (f"Seg{self.segment_idx:02d} {state}  "
                f"residui={self.residues}  "
                f"polar={self.n_polar}/{self.n_sites}")


def build_qubit_chain(
    flat_chain: List[dict],
    residues_per_segment: int = 2,
    threshold: int = 2,
) -> List[QubitSegment]:
    """
    Costruisce la catena di qubit dalla flat_chain.

    Parameters
    ----------
    flat_chain : List[dict]
        Output di run_pocket — lista di siti con chiavi
        "residue", "type", "i", "j", "intensity".
    residues_per_segment : int
        Numero di residui contigui che formano un segmento.
    threshold : int
        Numero minimo di siti HBD+HBA nel segmento per avere qubit=1.

    Returns
    -------
    List[QubitSegment]
    """
    # ── Raggruppa i siti per residuo (mantenendo l'ordine spaziale) ───────
    residue_order = []   # lista ordinata di nomi residuo (senza duplicati)
    sites_by_residue = {}

    for site in flat_chain:
        res = site["residue"]
        if res not in sites_by_residue:
            sites_by_residue[res] = []
            residue_order.append(res)
        sites_by_residue[res].append(site)

    # ── Suddividi i residui in segmenti di dimensione fissa ───────────────
    n_residues = len(residue_order)
    segments: List[QubitSegment] = []
    seg_idx = 0

    for start in range(0, n_residues, residues_per_segment):
        end = min(start + residues_per_segment, n_residues)
        seg_residues = residue_order[start:end]

        # Raccogli tutti i siti del segmento
        seg_sites = []
        for res in seg_residues:
            seg_sites.extend(sites_by_residue[res])

        n_sites = len(seg_sites)
        n_polar = sum(1 for s in seg_sites if s["type"] in POLAR_TYPES)
        qubit = 1 if n_polar >= threshold else 0

        segments.append(QubitSegment(
            segment_idx=seg_idx,
            residues=seg_residues,
            n_sites=n_sites,
            n_polar=n_polar,
            qubit=qubit,
        ))
        seg_idx += 1

    return segments


def qubit_chain_to_bitstring(segments: List[QubitSegment]) -> str:
    """Restituisce la catena come stringa binaria, es. '10110100...'"""
    return "".join(str(s.qubit) for s in segments)


def print_qubit_chain(segments: List[QubitSegment]):
    """Stampa la catena di qubit in modo leggibile."""
    print(f"\n  {'Seg':5s}  {'Qubit':6s}  {'Polar/Tot':10s}  Residui")
    print(f"  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*30}")
    for s in segments:
        state = "|1⟩" if s.qubit == 1 else "|0⟩"
        res_str = ", ".join(s.residues)
        print(f"  {s.segment_idx:5d}  {state:6s}  "
              f"{s.n_polar:3d}/{s.n_sites:<6d}  {res_str}")

    bitstring = qubit_chain_to_bitstring(segments)
    n1 = bitstring.count("1")
    n0 = bitstring.count("0")
    print(f"\n  Bitstring : {bitstring}")
    print(f"  Lunghezza : {len(bitstring)} qubit")
    print(f"  |1⟩       : {n1}  ({100*n1/len(bitstring):.1f}%)")
    print(f"  |0⟩       : {n0}  ({100*n0/len(bitstring):.1f}%)")
