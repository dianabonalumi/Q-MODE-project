"""
Quantum encoding from "Quantum algorithm for protein-ligand docking sites
identification in the interaction space": first encoding (binarization for
Grover search) and second encoding (amplitude encoding for Euclidean distance).
"""

import numpy as np
from typing import Tuple, Dict

def first_encoding(h: float, hb: float, h_thr: float, hb_thr: float) -> str:
    """Binarize (h, hb) into a 2-bit basis state, e.g. "10" (first bit = h,
    second = hb).

    The paper thresholds at the range midpoint (min+max)/2, but on real data
    the intensity distribution is skewed (most sites near the floor, a few
    strong outliers) so the mean gets dragged up and almost nothing crosses
    it, producing an all-zero chain. We use the channel median instead
    (computed upstream), which splits sites ~50/50 and keeps the
    binarization useful for Grover search.
    """
    q_h = "0" if h < h_thr else "1"
    q_hb = "0" if hb < hb_thr else "1"
    return f"{q_h}{q_hb}"


def second_encoding(h: float, hb: float, h_min: float, h_max: float, hb_min: float, hb_max: float) -> Dict[str, float]:
    """Probability amplitudes a, b, c, d (paper Eqs. 16-19).

    Normalized as a valid quantum state: a^2 + b^2 = 1 and c^2 + d^2 = 1,
    with a = (v_max-v)/sqrt((v_max-v)^2+(v-v_min)^2), b analogous (cosine/
    sine form: a, b are the projections of the angle encoding v within
    [v_min, v_max]). v is clamped to its channel's range first so sites
    outside that channel (e.g. h=0 for a polar site) land at the lower
    extreme instead of breaking normalization.
    """
    eps = 1e-9

    def _amplitudes(v, v_min, v_max):
        v = min(max(v, v_min), v_max)
        p = v_max - v
        q = v - v_min
        den = np.sqrt(p * p + q * q)
        if den < eps:                          # v_min == v_max -> uniform state
            return 0.7071067811865476, 0.7071067811865476
        return p / den, q / den

    a, b = _amplitudes(h, h_min, h_max)
    c, d = _amplitudes(hb, hb_min, hb_max)

    return {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
