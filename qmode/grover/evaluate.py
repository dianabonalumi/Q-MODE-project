"""Distance-based validation of Grover's candidate windows: how far each
candidate's site centroid sits from the true ligand centroid. Ground truth
only exists when the ligand's real pose is known (benchmark mode via
--ligand-pdb); this is a classical stand-in for the paper's SWAP-test-based
ranking, which is not implemented here."""

from __future__ import annotations
from typing import List

import numpy as np


def compute_window_centroid(flat_chain: List[dict], window_start: int, ligand_size: int) -> np.ndarray:
    window_sites = flat_chain[window_start:window_start + ligand_size]
    coords = [s["coords"] for s in window_sites]
    return np.array(coords).mean(axis=0)


def evaluate_candidates(
    candidates: List[dict],
    flat_chain: List[dict],
    ligand_centroids: List[np.ndarray],
    ligand_size: int,
) -> List[dict]:
    """Adds `window_centroid` and `distance_to_ligand_A` to each candidate
    (Euclidean distance, Å) and returns them sorted ascending by distance —
    the closest candidate to the true ligand position first.

    `ligand_centroids` takes one centroid per bound copy of the ligand: if
    it's bound at several sites in the same structure, distance is measured
    to the nearest copy, not to one arbitrarily-picked copy -- otherwise a
    candidate near a real (but not the auto-picked) copy would score as a
    huge miss."""
    evaluated = []
    for c in candidates:
        centroid = compute_window_centroid(flat_chain, c["window_start_index"], ligand_size)
        dists = [float(np.linalg.norm(centroid - lc)) for lc in ligand_centroids]
        nearest_idx = int(np.argmin(dists))
        evaluated.append({
            **c,
            "window_centroid": [round(float(x), 3) for x in centroid],
            "distance_to_ligand_A": round(dists[nearest_idx], 3),
            "nearest_ligand_copy": nearest_idx,
        })

    evaluated.sort(key=lambda c: c["distance_to_ligand_A"])
    return evaluated


def all_window_distances(
    flat_chain: List[dict],
    ligand_centroids: List[np.ndarray],
    ligand_size: int,
) -> np.ndarray:
    """Distance to the nearest ligand copy for EVERY step-1 window of the
    chain — the population a random control draws from."""
    cs = np.array(ligand_centroids)
    out = []
    for i in range(len(flat_chain) - ligand_size + 1):
        c = np.mean([s["coords"] for s in flat_chain[i:i + ligand_size]], axis=0)
        out.append(float(np.min(np.linalg.norm(cs - c, axis=1))))
    return np.array(out)


def _best_of_n_survival(dists: np.ndarray, n: int, d: float) -> float:
    """P(all n windows drawn without replacement are strictly worse than d).
    Closed form: C(M, n) / C(N, n) with M = #{dist > d}, expanded as a product
    so nothing overflows and no sampling is involved."""
    n_win = len(dists)
    worse = int((dists > d).sum())
    p = 1.0
    for i in range(n):
        if worse - i <= 0:
            return 0.0
        p *= (worse - i) / (n_win - i)
    return p


def random_control(
    flat_chain: List[dict],
    ligand_centroids: List[np.ndarray],
    ligand_size: int,
    grover_best: float,
    n: int = 3,
    seed: int = 0,
) -> dict:
    """Control for the Grover candidates: how good are `n` windows picked at
    random from the same chain?

    Returns one seeded draw of `n` windows (so the run is reproducible and the
    table can be shown side by side with the candidates) plus the exact
    probability that `n` random windows do at least as well as Grover's best.
    That probability is computed in closed form from the full distribution of
    window distances, not estimated by sampling, so it does not move between
    runs and does not depend on the seed — the seed only picks which three
    windows get displayed.

    Read it as a p-value: large means Grover's candidates are worth no more
    than picking the same number of windows blindly."""
    dists = all_window_distances(flat_chain, ligand_centroids, ligand_size)
    n_win = len(dists)
    n = min(n, n_win)
    rng = np.random.default_rng(seed)
    shown = sorted(int(i) for i in rng.choice(n_win, size=n, replace=False))

    # mediana della distribuzione "migliore di n a caso", dalla stessa forma chiusa
    median_best = float(dists.max())
    for d in np.sort(dists):
        if 1.0 - _best_of_n_survival(dists, n, float(d)) >= 0.5:
            median_best = float(d)
            break

    return {
        "n_windows": n_win,
        "n": n,
        "shown": shown,
        "shown_dists": [round(float(dists[i]), 3) for i in shown],
        "grover_best": round(float(grover_best), 3),
        "median_best_of_n": round(median_best, 3),
        "p_value": round(1.0 - _best_of_n_survival(dists, n, float(grover_best)), 4),
    }
