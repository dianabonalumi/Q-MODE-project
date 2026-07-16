"""
Ordinamento topologico dei siti farmacoforici
================================================
Dato l'insieme di siti (feature) di un residuo, li riordina secondo la
topologia del grafo dei legami della molecola:
  1. Costruisce il grafo dei legami da RDKit.
  2. Mappa ogni sito all'atomo pesante più vicino.
  3. Visita il grafo con una BFS partendo dall'atomo con indice minore
     (convenzionalmente il backbone N-terminale).
  4. I siti vengono restituiti nell'ordine in cui i loro atomi
     rappresentativi vengono incontrati dalla visita → la catena finale
     segue la connettività chimica della molecola.
"""

from __future__ import annotations
from typing import List

import numpy as np

from .feature_extraction import AtomFeature


def topological_order(
    sites: List[AtomFeature],
    mol,
) -> List[AtomFeature]:
    """
    Riordina i siti secondo la topologia covalente della molecola:
      1. Raccoglie le coordinate degli atomi pesanti dal conformatore RDKit.
      2. Mappa ogni sito al suo atomo rappresentativo più vicino (NN search).
      3. Esegue una BFS sul grafo dei legami partendo dall'atomo con idx più
         basso (convenzione: azoto N-terminale del backbone).
      4. I siti vengono restituiti nell'ordine di primo incontro BFS.

    mol deve avere un conformero 3D valido — nessun fallback spaziale.
    """
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
    ])  # shape (N_heavy, 3)

    # Mappa ogni sito → indice atomo pesante più vicino
    site_to_atom: List[int] = []
    for site in sites:
        dists = np.linalg.norm(heavy_coords - site.coords, axis=1)
        site_to_atom.append(int(np.argmin(dists)))

    # Grafo di adiacenza (solo atomi pesanti)
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
    return [s for _, s in sorted(zip(bfs_order, sites), key=lambda x: x[0])]
