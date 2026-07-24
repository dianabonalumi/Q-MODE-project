"""
Reorders a residue's pharmacophore sites by covalent-bond topology: build
the bond graph, map each site to its nearest heavy atom, BFS from the
backbone N, return sites in visit order.
"""

from __future__ import annotations
from typing import List

import numpy as np

from .feature_extraction import AtomFeature


def topological_order(
    sites: List[AtomFeature],
    mol,
) -> List[AtomFeature]:
    """mol must have a valid 3D conformer -- no spatial fallback."""
    from collections import deque
    from rdkit import Chem

    if len(sites) <= 1:
        return sites

    mol_no_h = Chem.RemoveHs(mol)
    if mol_no_h.GetNumConformers() == 0:
        raise ValueError("topological_order: mol non ha un conformero 3D")

    conf = mol_no_h.GetConformer()
    heavy_coords = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol_no_h.GetNumAtoms())
    ])

    # map each site to its nearest heavy atom
    site_to_atom: List[int] = []
    for site in sites:
        dists = np.linalg.norm(heavy_coords - site.coords, axis=1)
        site_to_atom.append(int(np.argmin(dists)))

    adj: dict[int, list[int]] = {i: [] for i in range(mol_no_h.GetNumAtoms())}
    for bond in mol_no_h.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    # BFS from the lowest-index representative atom (conventionally the
    # N-terminal backbone nitrogen)
    start = min(site_to_atom)
    visited: dict[int, int] = {}
    queue: deque = deque([start])
    order = 0
    seen = set([start])
    while queue:
        node = queue.popleft()
        visited[node] = order
        order += 1
        for nb in sorted(adj[node]):   # sorted for determinism
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)

    bfs_order = [visited.get(atom_idx, len(sites) + i)
                 for i, atom_idx in enumerate(site_to_atom)]
    return [s for _, s in sorted(zip(bfs_order, sites), key=lambda x: x[0])]
