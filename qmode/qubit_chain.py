"""
Splits the flat site sequence into ligand-sized sliding-window segments and
applies the first/second quantum encoding to each.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from qmode.quantum_encoding import first_encoding, second_encoding


@dataclass
class QubitSegment:
    segment_idx: int
    sites: List[dict]
    first_encoding_state: str           # e.g. "100111"
    second_encoding_amplitudes: List[Dict[str, float]]  # {a,b,c,d} per site


def get_h_hb_intensities(site: dict) -> tuple[float, float]:
    """Reduce a site to its two interaction channels.

      h  (hydrophobic/apolar): Hydrophobe, Aromatic
      hb (polar/charged):      HBondDonor, HBondAcceptor, PosIonizable, NegIonizable

    Ionizable groups feed into hb with their geometric intensity rather than
    being dropped to (0,0), so every site lands in at least one channel.
    """
    t = site["type"]
    intensity = site.get("intensity", 1.0)

    if t in ["Hydrophobe", "Aromatic"]:
        return intensity, 0.0
    elif t in ["HBondDonor", "HBondAcceptor", "PosIonizable", "NegIonizable"]:
        return 0.0, intensity
    else:
        return 0.0, 0.0


def compute_h_hb_thresholds(flat_chain: List[dict], h_min: float = 0.0, h_max: float = 1.0,
                             hb_min: float = 0.0, hb_max: float = 1.0) -> tuple[float, float]:
    """Binarization thresholds = median of each channel's active values.
    The median splits sites ~50/50, avoiding the all-zero chain the range
    mean would produce (see quantum_encoding.first_encoding)."""
    import statistics

    h_active = [get_h_hb_intensities(s)[0] for s in flat_chain if get_h_hb_intensities(s)[0] > 0]
    hb_active = [get_h_hb_intensities(s)[1] for s in flat_chain if get_h_hb_intensities(s)[1] > 0]
    h_thr = statistics.median(h_active) if h_active else (h_min + h_max) / 2.0
    hb_thr = statistics.median(hb_active) if hb_active else (hb_min + hb_max) / 2.0
    return h_thr, hb_thr


def build_qubit_chain(
    flat_chain: List[dict],
    ligand_size: int,
    h_min: float,
    h_max: float,
    hb_min: float,
    hb_max: float
) -> List[QubitSegment]:
    """Sliding-window segmentation ("protein shift" in the paper) + encoding."""
    segments = []
    n_sites = len(flat_chain)

    if n_sites < ligand_size:
        return []

    h_thr, hb_thr = compute_h_hb_thresholds(flat_chain, h_min, h_max, hb_min, hb_max)

    for i in range(n_sites - ligand_size + 1):
        segment_sites = flat_chain[i:i+ligand_size]

        first_enc_str = ""
        second_enc_amps = []

        for site in segment_sites:
            h, hb = get_h_hb_intensities(site)

            q_str = first_encoding(h, hb, h_thr, hb_thr)   # 2 qubits per site
            first_enc_str += q_str

            amps = second_encoding(h, hb, h_min, h_max, hb_min, hb_max)  # 4 amplitudes per site
            second_enc_amps.append(amps)

        segments.append(QubitSegment(
            segment_idx=i,
            sites=segment_sites,
            first_encoding_state=first_enc_str,
            second_encoding_amplitudes=second_enc_amps
        ))

    return segments


def print_qubit_chain(segments: List[QubitSegment]):
    """Stampa i segmenti codificati quantisticamente."""
    if not segments:
        print("Nessun segmento trovato (ligand_size > siti totali?).")
        return

    print(f"\n  {'Seg':5s}  {'First Encoding':20s}  Residui Inclusi")
    print(f"  {'─'*5}  {'─'*20}  {'─'*30}")
    for s in segments:
        residues = []
        for site in s.sites:
            res = site["residue"].split("_")[0]
            if res not in residues:
                residues.append(res)
        res_str = ", ".join(residues)
        print(f"  {s.segment_idx:5d}  |{s.first_encoding_state}⟩{' '*max(0, 18-len(s.first_encoding_state))}  {res_str}")

    print(f"\n  Totale segmenti (shift): {len(segments)}")
