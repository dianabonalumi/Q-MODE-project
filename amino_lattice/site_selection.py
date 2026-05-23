"""
Step 3 — Scelta di K (numero di interaction sites)
====================================================
Strategie implementate:

  1. "heavy_atoms"  — K = numero atomi pesanti (senza H), cappato a max_k
  2. "active_features" — K = numero feature attive estratte allo step 2
  3. "fixed"  — K costante per tutti gli amminoacidi (utile per confronti)
  4. "groups"  — raggruppa feature dello stesso tipo vicine (clustering)

La strategia "active_features" è quella più fedele allo spirito del paper:
ogni feature farmacofora distinta diventa un sito.
"""

from __future__ import annotations
from typing import List, Literal

import numpy as np
from sklearn.cluster import KMeans

from .feature_extraction import AtomFeature


Strategy = Literal["heavy_atoms", "active_features", "fixed", "groups"]


def choose_k(
    features: List[AtomFeature],
    mol=None,
    strategy: Strategy = "active_features",
    max_k: int = 10,
    min_k: int = 1,
    fixed_k: int = 5,
) -> int:
    """
    Determina il numero K di siti per un amminoacido.

    Parameters
    ----------
    features : list di AtomFeature
        Output di feature_extraction.extract_features().
    mol : RDKit Mol, opzionale
        Necessario solo per strategy="heavy_atoms".
    strategy : str
        Una delle strategie definite sopra.
    max_k : int
        Numero massimo di siti (cap).
    min_k : int
        Numero minimo di siti.
    fixed_k : int
        Valore usato quando strategy="fixed".

    Returns
    -------
    int
        K scelto.
    """
    if strategy == "fixed":
        return fixed_k

    elif strategy == "heavy_atoms":
        if mol is None:
            raise ValueError("strategy='heavy_atoms' richiede il parametro mol")
        from rdkit.Chem import rdMolDescriptors
        k = rdMolDescriptors.CalcNumHeavyAtoms(mol)

    elif strategy == "active_features":
        # Conta feature uniche per tipo e posizione (deduplicazione greedy)
        k = _count_distinct_features(features)

    elif strategy == "groups":
        # Raggruppa feature spazialmente vicine → K = num cluster ottimale
        k = _cluster_features(features, max_k)

    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")

    return int(np.clip(k, min_k, max_k))


# ─────────────────────────────────────────────────────────────────────────────
# Helper privati
# ─────────────────────────────────────────────────────────────────────────────

def _count_distinct_features(features: List[AtomFeature], distance_threshold: float = 1.5) -> int:
    """
    Conta feature distinte: due feature dello stesso tipo a distanza < threshold
    vengono considerate la stessa.
    """
    if not features:
        return 1

    distinct = []
    for f in features:
        merged = False
        for d in distinct:
            if d.feature_type == f.feature_type:
                dist = np.linalg.norm(f.coords - d.coords)
                if dist < distance_threshold:
                    merged = True
                    break
        if not merged:
            distinct.append(f)

    return max(1, len(distinct))


def _cluster_features(features: List[AtomFeature], max_k: int) -> int:
    """
    Usa l'inertia (elbow method semplificato) per scegliere K ottimale
    tramite clustering spaziale delle coordinate 3D delle feature.
    """
    if len(features) <= 2:
        return len(features)

    coords = np.array([f.coords for f in features])
    n = len(coords)
    max_k = min(max_k, n)

    best_k = 2
    best_score = float("inf")

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(coords)
        inertia = km.inertia_
        # penalizza k grandi (BIC-like)
        score = inertia + k * np.log(n)
        if score < best_score:
            best_score = score
            best_k = k

    return best_k


def select_representative_sites(
    features: List[AtomFeature],
    k: int,
) -> List[AtomFeature]:
    """
    Data la lista di feature e il K scelto, seleziona/aggrega K siti
    rappresentativi tramite K-Means sulle coordinate 3D.

    Ogni cluster → un sito con:
      - coordinate = centroide del cluster
      - tipo = tipo di feature più frequente nel cluster
      - intensità = somma intensità del cluster

    Returns
    -------
    List[AtomFeature]  — lunghezza esattamente K
    """
    if not features:
        raise ValueError("Nessuna feature estratta: impossibile selezionare siti")

    coords = np.array([f.coords for f in features])
    k = min(k, len(features))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    from collections import Counter
    from .feature_extraction import AtomFeature

    sites: List[AtomFeature] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_features = [f for f, m in zip(features, mask) if m]

        centroid = coords[mask].mean(axis=0)

        # tipo dominante nel cluster
        type_counts = Counter(f.feature_type for f in cluster_features)
        dominant_type = type_counts.most_common(1)[0][0]

        total_intensity = sum(f.intensity for f in cluster_features)
        all_atom_ids = [idx for f in cluster_features for idx in f.atom_indices]

        sites.append(AtomFeature(
            feature_type=dominant_type,
            coords=centroid,
            atom_indices=all_atom_ids,
            intensity=total_intensity,
        ))

    return sites
