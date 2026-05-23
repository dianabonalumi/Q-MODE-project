"""
Step 6 — Assegnazione Label ai Siti
=====================================
Converte il tipo farmacoforo di ciascun sito in una rappresentazione numerica:

  1. "one_hot"   — vettore binario di lunghezza |FEATURE_TYPES|
  2. "index"     — intero (indice nel vocabolario)
  3. "embedding" — vettore denso apprendibile (placeholder: random normal
                   con seed fisso; da sostituire con pesi appresi)

Output finale per ogni sito:
  LabeledSite(i, j, feature_type, label_vector)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

from .feature_extraction import FEATURE_TYPES, FEATURE_INDEX, AtomFeature


LabelMode = Literal["one_hot", "index", "embedding"]

N_FEATURE_TYPES = len(FEATURE_TYPES)   # 6


@dataclass
class LabeledSite:
    """Un sito sul reticolo con coordinate intere e label."""
    i: int                    # coordinata riga reticolo
    j: int                    # coordinata colonna reticolo
    feature_type: str         # es. "HBondDonor"
    label: np.ndarray         # vettore label (shape dipende da mode)
    intensity: float = 1.0

    def to_tuple(self) -> Tuple:
        return (self.i, self.j, self.feature_type)

    def __repr__(self) -> str:
        return f"Site({self.i},{self.j}, {self.feature_type})"


# ─────────────────────────────────────────────────────────────────────────────
# Funzione principale
# ─────────────────────────────────────────────────────────────────────────────

def label_sites(
    sites: List[AtomFeature],
    lattice_nodes: List[Tuple[int, int]],
    mode: LabelMode = "one_hot",
    embedding_dim: int = 16,
    embedding_matrix: np.ndarray | None = None,
) -> List[LabeledSite]:
    """
    Assegna label numeriche ai siti.

    Parameters
    ----------
    sites : List[AtomFeature]
        I K siti (tipo + intensità).
    lattice_nodes : List[Tuple[int, int]]
        Le coordinate intere (i, j) di ciascun sito (output dello snapping).
    mode : "one_hot" | "index" | "embedding"
    embedding_dim : int
        Dimensione dell'embedding denso (usato solo se mode="embedding").
    embedding_matrix : np.ndarray shape (N_FEATURE_TYPES, embedding_dim), opzionale
        Matrice di embedding pre-addestrata. Se None, usa un'inizializzazione
        casuale riproducibile.

    Returns
    -------
    List[LabeledSite]
    """
    assert len(sites) == len(lattice_nodes), \
        f"Mismatch: {len(sites)} siti ma {len(lattice_nodes)} nodi"

    if mode == "embedding" and embedding_matrix is None:
        rng = np.random.default_rng(42)
        embedding_matrix = rng.standard_normal((N_FEATURE_TYPES, embedding_dim))

    labeled = []
    for site, (i, j) in zip(sites, lattice_nodes):
        label = _encode(site.feature_type, mode, embedding_dim, embedding_matrix)
        labeled.append(LabeledSite(
            i=i, j=j,
            feature_type=site.feature_type,
            label=label,
            intensity=site.intensity,
        ))

    return labeled


def encode_chain(labeled_sites: List[LabeledSite]) -> np.ndarray:
    """
    Converte la lista di LabeledSite in una matrice (K, 2 + label_dim)
    dove le prime 2 colonne sono (i, j) e le restanti sono il label vettore.

    Utile come input diretto a modelli ML.
    """
    rows = []
    for ls in labeled_sites:
        row = np.concatenate([[ls.i, ls.j], ls.label])
        rows.append(row)
    return np.array(rows)  # (K, 2 + label_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode(
    feature_type: str,
    mode: LabelMode,
    embedding_dim: int,
    embedding_matrix: np.ndarray | None,
) -> np.ndarray:
    idx = FEATURE_INDEX.get(feature_type, 0)  # default 0 se tipo sconosciuto

    if mode == "index":
        return np.array([idx], dtype=np.int32)

    elif mode == "one_hot":
        vec = np.zeros(N_FEATURE_TYPES, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    elif mode == "embedding":
        return embedding_matrix[idx].astype(np.float32)

    else:
        raise ValueError(f"Mode sconosciuta: {mode}")
