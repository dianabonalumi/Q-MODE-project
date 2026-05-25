"""
Qubit Chain (Quantum Encoding)
==============================
Suddivide la sequenza flat di siti in segmenti (sliding window) della stessa
dimensione del ligando e applica il First e Second Encoding quantistico.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from amino_lattice.quantum_encoding import first_encoding, second_encoding


@dataclass
class QubitSegment:
    """Un segmento della proteina di dimensione pari al ligando."""
    segment_idx: int
    sites: List[dict]
    first_encoding_state: str           # es. "100111"
    second_encoding_amplitudes: List[Dict[str, float]] # lista di {a,b,c,d} per ogni sito


def get_h_hb_intensities(site: dict) -> tuple[float, float]:
    """Ritorna (h, hb) per un sito."""
    t = site["type"]
    intensity = site.get("intensity", 1.0)
    
    if t in ["Hydrophobe", "Aromatic"]:
        return intensity, 0.0
    elif t in ["HBondDonor", "HBondAcceptor"]:
        return 0.0, intensity
    else:
        return 0.0, 0.0


def build_qubit_chain(
    flat_chain: List[dict],
    ligand_size: int,
    h_min: float,
    h_max: float,
    hb_min: float,
    hb_max: float
) -> List[QubitSegment]:
    """
    Crea la catena di segmenti e codifica gli stati quantistici.
    Implementa la logica di 'protein shift' (sliding window) del paper.
    """
    segments = []
    n_sites = len(flat_chain)
    
    if n_sites < ligand_size:
        return []
        
    for i in range(n_sites - ligand_size + 1):
        segment_sites = flat_chain[i:i+ligand_size]
        
        first_enc_str = ""
        second_enc_amps = []
        
        for site in segment_sites:
            h, hb = get_h_hb_intensities(site)
            
            # First encoding (Grover Search) - 2 qubit per sito
            q_str = first_encoding(h, hb, h_min, h_max, hb_min, hb_max)
            first_enc_str += q_str
            
            # Second encoding (Euclidean distance) - 4 ampiezze per sito
            amps = second_encoding(h, hb, h_min, h_max, hb_min, hb_max)
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
