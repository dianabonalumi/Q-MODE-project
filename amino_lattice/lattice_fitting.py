"""
Step 4 — Fitting Geometrico sul Reticolo 2D
=============================================
Proietta le coordinate 3D dei K siti su un piano 2D continuo usando:

  - PCA  (default): ruota il sistema di riferimento lungo i primi 2 PC.
           Veloce, deterministico, preserva la struttura globale.
  - MDS  : Multi-Dimensional Scaling metrico. Preserva meglio le distanze
           a coppie quando la struttura non è planare.

L'output sono coordinate reali (float) in 2D, normalizzate in modo che
il sito più vicino all'origine sia in (0, 0) e la scala rifletta le
distanze in Angstrom divise per la spaziatura del reticolo.
"""

from __future__ import annotations
from typing import List, Literal, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from .feature_extraction import AtomFeature


Method = Literal["pca", "mds"]


def fit_to_lattice_2d(
    sites: List[AtomFeature],
    method: Method = "pca",
    lattice_spacing: float = 1.5,   # Angstrom per passo reticolo
    center: bool = True,
) -> np.ndarray:
    """
    Proietta le coordinate 3D dei siti su un piano 2D.

    Parameters
    ----------
    sites : List[AtomFeature]
        I K siti (output di site_selection.select_representative_sites).
    method : "pca" | "mds"
    lattice_spacing : float
        Distanza in Angstrom che corrisponde a un passo unitario del reticolo.
        Usata per scalare le coordinate 2D → unità reticolo.
    center : bool
        Se True, trasla le coordinate in modo che il baricentro sia in (0,0).

    Returns
    -------
    np.ndarray  shape (K, 2)
        Coordinate 2D continue (float) in unità reticolo.
    """
    if not sites:
        raise ValueError("Lista di siti vuota")

    coords_3d = np.array([s.coords for s in sites])  # (K, 3)

    if len(sites) == 1:
        return np.array([[0.0, 0.0]])

    if method == "pca":
        coords_2d = _project_pca(coords_3d)
    elif method == "mds":
        coords_2d = _project_mds(coords_3d)
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    # Scala in unità reticolo
    coords_2d = coords_2d / lattice_spacing

    # Centra sul baricentro
    if center:
        coords_2d -= coords_2d.mean(axis=0)

    return coords_2d   # (K, 2) float


# ─────────────────────────────────────────────────────────────────────────────
# Proiezioni
# ─────────────────────────────────────────────────────────────────────────────

def _project_pca(coords_3d: np.ndarray) -> np.ndarray:
    """Proiezione PCA: mantieni i primi 2 componenti principali."""
    pca = PCA(n_components=2)
    return pca.fit_transform(coords_3d)


def _project_mds(coords_3d: np.ndarray) -> np.ndarray:
    """MDS metrico: minimizza la differenza tra distanze 3D e 2D."""
    mds = MDS(
        n_components=2,
        metric=True,
        random_state=42,
        dissimilarity="euclidean",
        n_init=4,
        max_iter=300,
    )
    return mds.fit_transform(coords_3d)


# ─────────────────────────────────────────────────────────────────────────────
# Utility diagnostica
# ─────────────────────────────────────────────────────────────────────────────

def projection_stress(
    sites: List[AtomFeature],
    coords_2d: np.ndarray,
) -> float:
    """
    Calcola lo stress della proiezione:
      stress = sqrt( sum((d_3d - d_2d)^2) / sum(d_3d^2) )

    Valori < 0.1 sono considerati buoni.
    """
    coords_3d = np.array([s.coords for s in sites])
    k = len(sites)
    d3_sq, d2_sq = 0.0, 0.0

    for i in range(k):
        for j in range(i + 1, k):
            d3 = np.linalg.norm(coords_3d[i] - coords_3d[j])
            d2 = np.linalg.norm(coords_2d[i] - coords_2d[j])
            d3_sq += d3 ** 2
            d2_sq += (d3 - d2) ** 2

    return float(np.sqrt(d2_sq / d3_sq)) if d3_sq > 0 else 0.0
