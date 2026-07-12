"""
Step 3 alternativo — Metodo 2: centroide per tipo farmacoforo.

Per ogni tipo presente nel residuo calcola la media di TUTTE
le coordinate delle feature di quel tipo -> un solo sito per tipo.
Massimo 6 siti (uno per tipo).

Quando due tipi farmacofori diversi hanno le stesse coordinate 3D
(stesso atomo fisico con piu' proprieta'), viene tenuto solo quello
con specificita' piu' alta per il docking secondo la scala:
Aromatic > HBondDonor > HBondAcceptor > NegIonizable > PosIonizable > Hydrophobe
"""

from __future__ import annotations
from typing import List
import numpy as np
from collections import defaultdict, deque

from .feature_extraction import AtomFeature, FEATURE_SPECIFICITY


def select_centroid_by_type(
    features: List[AtomFeature],
    mol,
) -> List[AtomFeature]:
    """
    Calcola un centroide per ogni tipo farmacoforo presente.
    Rimuove i siti con coordinate duplicate tenendo il tipo
    piu' specifico per il docking.

    Parameters
    ----------
    features : List[AtomFeature]
        Feature estratte dallo Step 2 per UN residuo.
    mol : RDKit Mol
        Molecola RDKit per ordinamento topologico.

    Returns
    -------
    List[AtomFeature]
        Al massimo 6 siti (uno per tipo), senza coordinate duplicate,
        ordinati topologicamente.
    """
    # raggruppa per tipo
    by_type = defaultdict(list)
    for feat in features:
        by_type[feat.feature_type].append(feat)

    # calcola centroide per tipo
    sites = []
    for ftype, feats in by_type.items():
        centroid = np.mean([f.coords for f in feats], axis=0)
        intensity = sum(f.intensity for f in feats)
        atom_indices = []
        for f in feats:
            atom_indices.extend(f.atom_indices)
        sites.append(AtomFeature(
            feature_type=ftype,
            coords=centroid,
            atom_indices=atom_indices,
            intensity=intensity,
        ))

    # rimuovi duplicati di coordinate tenendo il tipo piu' specifico
    sites = _remove_duplicate_coords(sites)

    # ordina topologicamente
    sites = _topological_order(sites, mol)

    return sites


def _remove_duplicate_coords(sites: List[AtomFeature]) -> List[AtomFeature]:
    """
    Se due siti hanno le stesse coordinate 3D (arrotondate a 2 decimali),
    tiene solo quello con specificita' farmacofora piu' alta per il docking.

    Scala di specificita':
        Aromatic(4) > HBondDonor(2) = HBondAcceptor(2) >
        NegIonizable(3) > PosIonizable(3) > Hydrophobe(1)

    Nota: NegIonizable e PosIonizable hanno specificita' 3 nel dizionario
    originale, quindi prevalgono su HBondDonor/HBondAcceptor solo se
    il dizionario lo indica cosi'. Adatta FEATURE_SPECIFICITY se necessario.
    """
    seen = {}   # coord_key -> sito tenuto
    for site in sites:
        key = tuple(np.round(site.coords, 2))
        if key not in seen:
            seen[key] = site
        else:
            current_spec = FEATURE_SPECIFICITY.get(seen[key].feature_type, 0)
            new_spec     = FEATURE_SPECIFICITY.get(site.feature_type, 0)
            if new_spec > current_spec:
                seen[key] = site
    return list(seen.values())


def _topological_order(sites: List[AtomFeature], mol) -> List[AtomFeature]:
    """
    Riordina i siti seguendo la connettivita' covalente del residuo
    tramite BFS sul grafo degli atomi pesanti.
    Fallback: proiezione sul primo asse PCA.
    """
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
        seen_nodes = {start}
        order = 0
        while queue:
            node = queue.popleft()
            visited[node] = order
            order += 1
            for nb in sorted(adj[node]):
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
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