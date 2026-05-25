"""
Modulo per l'encoding quantistico descritto nel paper:
"Quantum algorithm for protein-ligand docking sites identification in the interaction space"

Implementa:
- First Encoding: Binarizzazione per Grover Search
- Second Encoding: Amplitude Encoding per il calcolo della distanza euclidea
"""

import numpy as np
from typing import Tuple, Dict

def first_encoding(h: float, hb: float, h_min: float, h_max: float, hb_min: float, hb_max: float) -> str:
    """
    First encoding: converte i valori continui in stati binari (qubit di base |0> o |1>).
    h: idrofobicità
    hb: legame idrogeno
    Ritorna una stringa di 2 bit, es. "10", dove il primo bit è per h e il secondo per hb.
    """
    # Soglia per idrofobicità
    h_threshold = (h_min + h_max) / 2.0
    q_h = "0" if h < h_threshold else "1"
    
    # Soglia per legame idrogeno
    hb_threshold = (hb_min + hb_max) / 2.0
    q_hb = "0" if hb < hb_threshold else "1"
    
    return f"{q_h}{q_hb}"


def second_encoding(h: float, hb: float, h_min: float, h_max: float, hb_min: float, hb_max: float) -> Dict[str, float]:
    """
    Second encoding: calcola le ampiezze di probabilità a, b, c, d
    (Equazioni 16-19 del paper).
    
    Ritorna un dizionario con i coefficienti { 'a': a, 'b': b, 'c': c, 'd': d }.
    """
    # Gestione di casi limite per evitare divisioni per zero
    eps = 1e-9
    
    # --- Interazione idrofobica (a, b) ---
    num_a = h_max - h
    num_b = h - h_min
    den_h_sq = (abs(h_max - h)**2 + abs(h - h_min)**2)
    if den_h_sq < eps:
        a, b = 0.7071, 0.7071 # 1/sqrt(2) se min == max == h
    else:
        a = np.sqrt(num_a / den_h_sq) if num_a > 0 else 0.0
        b = np.sqrt(num_b / den_h_sq) if num_b > 0 else 0.0
        
    # --- Interazione legame idrogeno (c, d) ---
    num_c = hb_max - hb
    num_d = hb - hb_min
    den_hb_sq = (abs(hb_max - hb)**2 + abs(hb - hb_min)**2)
    if den_hb_sq < eps:
        c, d = 0.7071, 0.7071
    else:
        c = np.sqrt(num_c / den_hb_sq) if num_c > 0 else 0.0
        d = np.sqrt(num_d / den_hb_sq) if num_d > 0 else 0.0
        
    return {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
