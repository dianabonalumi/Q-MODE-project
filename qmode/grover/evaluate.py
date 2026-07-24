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
