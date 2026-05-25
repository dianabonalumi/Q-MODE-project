"""
Step 5 — Snapping sui Nodi del Reticolo
=========================================
Converte le coordinate 2D continue (float) in coordinate intere (i, j)
del reticolo discreto.

Sfide:
  - Due siti proiettati possono finire sullo stesso nodo → risoluzione collisioni
  - Il reticolo può essere quadrato o esagonale

Strategie di snapping:
  1. "round"     — arrotondamento semplice al nodo intero più vicino
  2. "hungarian" — assegnazione ottima (minimizza distanza totale) tramite
                   algoritmo ungherese; garantisce nodi distinti

Fix applicati rispetto alla versione precedente:
  - Il raggio di generazione candidati è adattivo: parte da 2 e viene
    incrementato automaticamente finché n_cand >= k (evita il bug silenzioso
    in cui result[r] rimaneva al default (0,0)).
  - Viene sollevata una ValueError esplicita se anche con raggio_max=10
    non si riesce a raccogliere abbastanza candidati distinti.
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

    Il raggio di ricerca parte da 2 e viene espanso automaticamente finché
    n_cand >= k, garantendo che l'algoritmo ungherese abbia sempre abbastanza
    nodi distinti da assegnare (fix al bug precedente con (0,0) silenzioso).
    """
    k = len(coords_2d)
    if k == 0:
        return []

    # Raggio adattivo: espande finché ci sono abbastanza candidati distinti
    MAX_RADIUS = 10
    candidate_set: set = set()
    for radius in range(2, MAX_RADIUS + 1):
        candidate_set = set()
        for x, y in coords_2d:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    candidate_set.add((int(np.round(x)) + di, int(np.round(y)) + dj))
        if len(candidate_set) >= k:
            break
    else:
        raise ValueError(
            f"Impossibile trovare {k} nodi distinti entro raggio {MAX_RADIUS}. "
            f"Candidati disponibili: {len(candidate_set)}. "
            "Aumenta lattice_spacing o riduci max_k."
        )

    candidates = list(candidate_set)
    n_cand = len(candidates)

    # Matrice dei costi: distanza euclidea² sito → candidato
    cost = np.zeros((k, n_cand))
    for i, (x, y) in enumerate(coords_2d):
        for j, (ci, cj) in enumerate(candidates):
            cost[i, j] = (x - ci) ** 2 + (y - cj) ** 2

    # Algoritmo ungherese — matrice rettangolare (k × n_cand, k ≤ n_cand)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Verifica post-assegnazione: tutti i k siti devono essere assegnati
    assert len(row_ind) == k, (
        f"linear_sum_assignment ha assegnato solo {len(row_ind)}/{k} siti."
    )

    result: List[Tuple[int, int]] = [(0, 0)] * k
    for r, c in zip(row_ind, col_ind):
        result[r] = candidates[c]

    # Sanity check: nessuna collisione
    assigned = [result[r] for r in row_ind]
    assert len(set(assigned)) == k, (
        f"Collisione residua dopo algoritmo ungherese: {assigned}"
    )

    return result
