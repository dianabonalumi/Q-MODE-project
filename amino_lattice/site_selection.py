"""
Step 3 — Scelta di K (numero di interaction sites)
====================================================
Strategie implementate:

  1. "heavy_atoms"     — K = numero atomi pesanti (senza H), cappato a max_k
  2. "active_features" — K = numero feature attive estratte allo step 2
  3. "fixed"           — K costante per tutti gli amminoacidi (confronti uniformi)
  4. "groups"          — raggruppa feature dello stesso tipo vicine (clustering)

La strategia "active_features" è quella più fedele allo spirito del paper:
ogni feature farmacofora distinta diventa un sito.

Topologia covalente
-------------------
Dopo la selezione dei K centroidi tramite KMeans, i siti vengono riordinati
in modo da riflettere la topologia del grafo dei legami della molecola.
L'algoritmo:
  1. Costruisce il grafo dei legami da RDKit (se mol è disponibile).
  2. Mappa ogni centroide all'atomo pesante più vicino.
  3. Visita il grafo con una BFS/DFS partendo dall'atomo con indice minore
     (convenzionalmente il backbone N-terminale).
  4. I siti vengono restituiti nell'ordine in cui i loro atomi rappresentativi
     vengono incontrati dalla visita → la chain finale segue la connettività
     chimica della molecola.
Se mol non è disponibile o il grafo non è connesso, l'ordine di fallback
è quello spaziale lungo il primo asse PCA (invariante per rotazione).
"""

from __future__ import annotations
from typing import List, Literal, Optional

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
        k = _count_distinct_features(features)

    elif strategy == "groups":
        k = _cluster_features(features, max_k)

    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")

    return int(np.clip(k, min_k, max_k))


# ─────────────────────────────────────────────────────────────────────────────
# Helper privati — scelta K
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
        score = inertia + k * np.log(n)
        if score < best_score:
            best_score = score
            best_k = k

    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# Selezione siti + ordinamento topologico
# ─────────────────────────────────────────────────────────────────────────────

def select_representative_sites(
    features: List[AtomFeature],
    k: int,
    mol=None,
) -> List[AtomFeature]:
    """
    Data la lista di feature e il K scelto, seleziona/aggrega K siti
    rappresentativi tramite K-Means sulle coordinate 3D, poi li riordina
    secondo la topologia covalente della molecola (se mol è disponibile).

    Ogni cluster → un sito con:
      - coordinate = centroide del cluster
      - tipo       = tipo di feature più frequente nel cluster
      - intensità  = somma intensità del cluster

    Ordinamento topologico
    ----------------------
    Se mol è fornita, i siti vengono riordinati tramite BFS sul grafo dei
    legami RDKit, partendo dall'atomo pesante con indice minore.
    Ogni sito viene mappato all'atomo pesante più vicino al suo centroide;
    i siti seguono l'ordine di visita BFS di quegli atomi.
    Se mol=None, l'ordinamento è spaziale lungo il primo asse PCA.

    Returns
    -------
    List[AtomFeature]  — lunghezza esattamente K, ordinata topologicamente
    """
    if not features:
        raise ValueError("Nessuna feature estratta: impossibile selezionare siti")

    from collections import Counter, deque
    from .feature_extraction import AtomFeature

    coords = np.array([f.coords for f in features])
    k = min(k, len(features))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    # Costruisci i K siti (centroidi + tipo dominante)
    sites: List[AtomFeature] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_features = [f for f, m in zip(features, mask) if m]
        if not cluster_features:
            # KMeans può lasciare un cluster vuoto se ci sono coordinate
            # duplicate (es. Hydrophobe della factory + Crippen sullo stesso
            # atomo): in tal caso non c'è un sito da costruire, lo saltiamo.
            continue
        centroid = coords[mask].mean(axis=0)
        type_counts = Counter(f.feature_type for f in cluster_features)
        # Tipo dominante = il più SPECIFICO presente (Aromatic/Ionizable >
        # HBond > Hydrophobe); a parità di specificità, il più frequente.
        from .feature_extraction import FEATURE_SPECIFICITY
        dominant_type = max(
            type_counts,
            key=lambda t: (FEATURE_SPECIFICITY.get(t, 0), type_counts[t]),
        )
        total_intensity = sum(f.intensity for f in cluster_features)
        all_atom_ids = [idx for f in cluster_features for idx in f.atom_indices]
        sites.append(AtomFeature(
            feature_type=dominant_type,
            coords=centroid,
            atom_indices=all_atom_ids,
            intensity=total_intensity,
        ))

    # ── Ordinamento topologico ────────────────────────────────────────────
    ordered = topological_order(sites, mol)
    return ordered


def topological_order(
    sites: List[AtomFeature],
    mol=None,
) -> List[AtomFeature]:
    """
    Riordina i siti secondo la topologia covalente della molecola.

    Strategia
    ---------
    1. Se mol è disponibile:
       a. Raccoglie le coordinate degli atomi pesanti dal conformatore RDKit.
       b. Mappa ogni sito al suo atomo rappresentativo più vicino (NN search).
       c. Esegue una BFS sul grafo dei legami partendo dall'atomo con idx più
          basso (convenzione: azoto N-terminale del backbone).
       d. I siti vengono restituiti nell'ordine di primo incontro BFS.
    2. Se mol=None o la molecola non ha un conformatore:
       Ordina i siti lungo il primo asse PCA delle loro coordinate 3D
       (approssima l'ordine lungo la catena principale).
    """
    from collections import deque

    if mol is not None:
        try:
            from rdkit import Chem
            mol_no_h = Chem.RemoveHs(mol)
            if mol_no_h.GetNumConformers() == 0:
                raise RuntimeError("no conformer")

            conf = mol_no_h.GetConformer()
            heavy_coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol_no_h.GetNumAtoms())
            ])  # shape (N_heavy, 3)

            # Mappa ogni sito → indice atomo pesante più vicino
            site_to_atom: List[int] = []
            for site in sites:
                dists = np.linalg.norm(heavy_coords - site.coords, axis=1)
                site_to_atom.append(int(np.argmin(dists)))

            # Costruisce il grafo di adiacenza (solo atomi pesanti)
            adj: dict[int, list[int]] = {i: [] for i in range(mol_no_h.GetNumAtoms())}
            for bond in mol_no_h.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                adj[a].append(b)
                adj[b].append(a)

            # BFS partendo dall'atomo rappresentativo con indice minore
            start = min(site_to_atom)
            visited: dict[int, int] = {}   # atom_idx → ordine di visita
            queue: deque = deque([start])
            order = 0
            seen = set([start])
            while queue:
                node = queue.popleft()
                visited[node] = order
                order += 1
                for nb in sorted(adj[node]):   # sorted → determinismo
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)

            # Ordina i siti per ordine BFS del loro atomo rappresentativo
            bfs_order = [visited.get(atom_idx, len(sites) + i)
                         for i, atom_idx in enumerate(site_to_atom)]
            ordered = [s for _, s in sorted(zip(bfs_order, sites), key=lambda x: x[0])]
            return ordered

        except Exception:
            pass   # fallback spaziale

    # Fallback: ordina lungo il primo asse PCA
    site_coords = np.array([s.coords for s in sites])
    if len(sites) > 1:
        mean = site_coords.mean(axis=0)
        centered = site_coords - mean
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, -1]                     # primo componente principale
        projections = centered @ pc1
        order = np.argsort(projections)
        return [sites[i] for i in order]

    return sites
