from __future__ import annotations
from typing import List
import numpy as np
from collections import defaultdict

from qmode.feature_extraction import AtomFeature

DIST_THRESHOLD = 1.5

def select_dedup_centroid(features, mol, max_sites=12):
    distinct = _dedup_with_centroid(features)
    return distinct[:max_sites]

def _dedup_with_centroid(features):
    by_type = defaultdict(list)
    for feat in features:
        by_type[feat.feature_type].append(feat)
    distinct = []
    for ftype, feats in by_type.items():
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
        for group in groups:
            centroid = np.mean([f.coords for f in group], axis=0)
            intensity = sum(f.intensity for f in group)
            atom_indices = []
            for f in group:
                atom_indices.extend(f.atom_indices)
            distinct.append(AtomFeature(feature_type=ftype, coords=centroid, atom_indices=atom_indices, intensity=intensity))
    return distinct
