"""
Step 5 — Snapping sui Nodi del Reticolo
=========================================
Converte le coordinate 2D continue (float) in coordinate intere (i, j)
del reticolo discreto.

Sfide:
  - Due siti proiettati possono finire sullo stesso nodo → risoluzione collisioni
  - Il reticolo può essere quadrato o esagonale

Strategie di snapping:
  1. "round"  — arrotondamento semplice al nodo intero più vicino
  2. "hungarian" — assegnazione ottima (minimizza distanza totale) tramite
                   algoritmo ungherese; garantisce nodi distinti
"""

from __future__ import annotations
from typing import List, Literal, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


SnapStrategy = Literal["round", "hungarian"]


def snap_to_lattice(
    coords_2d: np.ndarray,
    strategy: SnapStrategy = "hungarian",
) -> List[Tuple[int, int]]:
    """
    Arrotonda le coordinate 2D continue ai nodi interi (i, j) del reticolo.

    Parameters
    ----------
    coords_2d : np.ndarray shape (K, 2)
        Coordinate continue in unità reticolo.
    strategy : "round" | "hungarian"
        Strategia di snapping. "hungarian" evita collisioni.

    Returns
    -------
    List[Tuple[int, int]]
        Lista di K coppie (i, j) — coordinate intere sul reticolo.
    """
    if strategy == "round":
        return _snap_round(coords_2d)
    elif strategy == "hungarian":
        return _snap_hungarian(coords_2d)
    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Implementazioni
# ─────────────────────────────────────────────────────────────────────────────

def _snap_round(coords_2d: np.ndarray) -> List[Tuple[int, int]]:
    """Arrotondamento diretto — può produrre collisioni."""
    nodes = []
    for x, y in coords_2d:
        nodes.append((int(np.round(x)), int(np.round(y))))
    return nodes


def _snap_hungarian(coords_2d: np.ndarray) -> List[Tuple[int, int]]:
    """
    Assegnazione ottima senza collisioni tramite algoritmo ungherese.

    Genera un insieme di nodi candidati (griglia intorno ai siti proiettati),
    poi trova l'assegnazione 1-a-1 che minimizza la distanza euclidea totale.
    """
    k = len(coords_2d)

    # Genera candidati: per ogni sito i nodi interi entro raggio 2
    candidate_set = set()
    for x, y in coords_2d:
        for di in range(-2, 3):
            for dj in range(-2, 3):
                candidate_set.add((int(np.round(x)) + di, int(np.round(y)) + dj))

    candidates = list(candidate_set)
    n_cand = len(candidates)

    # Matrice dei costi: distanza euclidea sito → candidato
    cost = np.zeros((k, n_cand))
    for i, (x, y) in enumerate(coords_2d):
        for j, (ci, cj) in enumerate(candidates):
            cost[i, j] = (x - ci) ** 2 + (y - cj) ** 2

    # Algoritmo ungherese (scipy) — richiede matrice quadrata o rettangolare
    row_ind, col_ind = linear_sum_assignment(cost)

    result = [(0, 0)] * k
    for r, c in zip(row_ind, col_ind):
        result[r] = candidates[c]

    return result
