"""
Step 3 alternativo — Metodo 3: deduplicazione con centroide locale.

Feature dello stesso tipo a distanza < DIST_THRESHOLD → fuse in un
solo sito con coordinate = media delle coordinate del gruppo.
Feature dello stesso tipo ma lontane → tenute come siti distinti.

A differenza del Metodo 2, non forza un unico sito per tipo:
se due HBondDonor sono lontani rimangono due siti separati.
"""

from __future__ import annotations
from typing import List
import numpy as np
from collections import defaultdict, deque

from .feature_extraction import AtomFeature

DIST_THRESHOLD = 1.5  # Å


def select_dedup_centroid(
    features: List[AtomFeature],
    mol,
    max_sites: int = 6,
) -> List[AtomFeature]:
    """
    Deduplicazione con centroide locale per feature vicine.

    Parameters
    ----------
    features : List[AtomFeature]
        Feature estratte dallo Step 2 per UN residuo.
    mol : RDKit Mol
        Molecola RDKit per ordinamento topologico.
    max_sites : int
        Numero massimo di siti (default 6).

    Returns
    -------
    List[AtomFeature]
        Siti distinti con centroide calcolato sui vicini, ordinati
        topologicamente, al massimo max_sites.
    """
    distinct = _dedup_with_centroid(features)
    distinct = distinct[:max_sites]
    return _topological_order(distinct, mol)


def _dedup_with_centroid(features: List[AtomFeature]) -> List[AtomFeature]:
    """
    Per ogni tipo farmacoforo raggruppa le feature vicine (< DIST_THRESHOLD)
    e calcola il centroide del gruppo. Feature lontane restano separate.
    """
    by_type = defaultdict(list)
    for feat in features:
        by_type[feat.feature_type].append(feat)

    distinct = []
    for ftype, feats in by_type.items():
        # Clustering greedy per posizione
        groups = []
        for feat in feats:
            placed = False
            for group in groups:
                for member in group:
                    if np.linalg.norm(feat.coords - member.coords) < DIST_THRESHOLD:
                        group.append(feat)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([feat])

        # Calcola centroide per ogni gruppo
        for group in groups:
            centroid = np.mean([f.coords for f in group], axis=0)
            intensity = sum(f.intensity for f in group)
            atom_indices = []
            for f in group:
                atom_indices.extend(f.atom_indices)
            distinct.append(AtomFeature(
                feature_type=ftype,
                coords=centroid,
                atom_indices=atom_indices,
                intensity=intensity,
            ))

    return distinct


def _topological_order(sites, mol):
    if len(sites) <= 1:
        return sites
    try:
        from rdkit.Chem import RemoveHs
        mol_no_h = RemoveHs(mol)
        conf = mol_no_h.GetConformer()
        heavy_coords = conf.GetPositions()

        site_to_atom = []
        for site in sites:
            dists = np.linalg.norm(heavy_coords - site.coords, axis=1)
            site_to_atom.append(int(np.argmin(dists)))

        adj = {i: [] for i in range(mol_no_h.GetNumAtoms())}
        for bond in mol_no_h.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            adj[a].append(b)
            adj[b].append(a)

        start = min(site_to_atom)
        visited = {}
        queue = deque([start])
        seen = {start}
        order = 0
        while queue:
            node = queue.popleft()
            visited[node] = order
            order += 1
            for nb in sorted(adj[node]):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

        indexed = sorted(
            enumerate(sites),
            key=lambda x: visited.get(site_to_atom[x[0]], float("inf"))
        )
        return [s for _, s in indexed]

    except Exception:
        if len(sites) < 2:
            return sites
        coords = np.array([s.coords for s in sites])
        centered = coords - coords.mean(axis=0)
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, -1]
        projections = centered @ axis
        return [sites[i] for i in np.argsort(projections)]