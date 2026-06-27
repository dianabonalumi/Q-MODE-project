"""
Modulo per l'encoding quantistico descritto nel paper:
"Quantum algorithm for protein-ligand docking sites identification in the interaction space"

Implementa:
- First Encoding: Binarizzazione per Grover Search
- Second Encoding: Amplitude Encoding per il calcolo della distanza euclidea
"""

import numpy as np
from typing import Tuple, Dict

def first_encoding(h: float, hb: float, h_thr: float, hb_thr: float) -> str:
    """
    First encoding: converte i valori continui in stati binari (qubit di base |0> o |1>).
    h: idrofobicità ;  hb: legame idrogeno
    h_thr, hb_thr: soglie di binarizzazione (vedi nota sotto).
    Ritorna una stringa di 2 bit, es. "10" (primo bit = h, secondo = hb).

    Nota sulla soglia
    -----------------
    Il paper usa la media del range, (min+max)/2. Su dati reali la distribuzione
    delle intensità è molto asimmetrica (molti siti vicini al floor, pochi forti):
    la media viene trascinata in alto dai massimi e quasi nessun sito la supera,
    producendo una catena dominata da zeri. Usiamo quindi una soglia (tipicamente
    la MEDIANA del canale, calcolata a monte) che separa i siti ~50/50 e rende la
    binarizzazione informativa per la ricerca di Grover.
    """
    q_h = "0" if h < h_thr else "1"
    q_hb = "0" if hb < hb_thr else "1"
    return f"{q_h}{q_hb}"


def second_encoding(h: float, hb: float, h_min: float, h_max: float, hb_min: float, hb_max: float) -> Dict[str, float]:
    """
    Second encoding: calcola le ampiezze di probabilità a, b, c, d
    (Equazioni 16-19 del paper).

    Ritorna un dizionario con i coefficienti { 'a': a, 'b': b, 'c': c, 'd': d }.

    Le ampiezze sono NORMALIZZATE come uno stato quantistico valido:
        a² + b² = 1   e   c² + d² = 1
    con
        a = (v_max − v) / √[(v_max − v)² + (v − v_min)²]
        b = (v − v_min) / √[(v_max − v)² + (v − v_min)²]
    (forma coseno/seno: a e b sono le proiezioni dell'angolo che codifica v nel
    range [v_min, v_max]). Il valore v viene prima riportato (clamp) entro il
    range del proprio canale, così i siti fuori canale (es. h = 0 per un sito
    polare) finiscono all'estremo inferiore invece di rompere la normalizzazione.
    """
    eps = 1e-9

    def _amplitudes(v, v_min, v_max):
        v = min(max(v, v_min), v_max)          # clamp nel range del canale
        p = v_max - v                          # "quanto manca al massimo"
        q = v - v_min                          # "quanto supera il minimo"
        den = np.sqrt(p * p + q * q)
        if den < eps:                          # v_min == v_max → stato uniforme
            return 0.7071067811865476, 0.7071067811865476
        return p / den, q / den

    a, b = _amplitudes(h, h_min, h_max)
    c, d = _amplitudes(hb, hb_min, hb_max)

    return {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
