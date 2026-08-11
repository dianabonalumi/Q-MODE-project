"""
Validazione Step 4: deduplicazione con centroide locale.
Verifica che:
  1. Nessuna coppia di siti dello stesso tipo rimanga a distanza < 1.5 A
  2. I centroidi calcolati siano geometricamente sensati
  3. Le intensità aggregate siano corrette (somma conservata)
"""
import sys
import os
import warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from qmode.pdb_reader import load_residues_from_pdb
from qmode.feature_extraction import extract_features
from qmode.surface_filter import compute_atom_sasa, filter_surface_features_by_coords
from qmode.site_dedup_centroid import select_dedup_centroid, DIST_THRESHOLD

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()

print(f"\nValidazione Step 4 — Site Deduplication")
print(f"PDB: {args.pdb}")
print(f"Soglia deduplicazione: {DIST_THRESHOLD} A")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    residues = load_residues_from_pdb(args.pdb)

sasa_map = compute_atom_sasa(args.pdb)

# contatori globali
total_before = 0
total_after = 0
test1_violations = []   # coppie rimaste a distanza < threshold
test2_violations = []   # centroidi fuori dalla molecola
test3_violations = []   # intensità non conservata

MAX_CENTROID_DIST = 3.0  # Å — distanza massima accettabile tra centroide e atomo PDB

for rec in residues:
    if rec.mol is None:
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(rec.mol, embed_3d=True)

    features = filter_surface_features_by_coords(
        features=features, sasa_map=sasa_map,
        chain_id=rec.chain_id, res_seq=rec.res_seq,
        all_atom_coords=rec.atoms, sasa_threshold=1.0)
    if not features:
        continue

    total_intensity_before = sum(f.intensity for f in features)
    total_before += len(features)

    deduped = select_dedup_centroid(features, mol=rec.mol, max_sites=12)
    total_after += len(deduped)

    total_intensity_after = sum(f.intensity for f in deduped)

    # ── Test 1: nessuna coppia rimasta a distanza < threshold ────
    by_type = defaultdict(list)
    for site in deduped:
        by_type[site.feature_type].append(site)

    for ftype, sites in by_type.items():
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                dist = np.linalg.norm(sites[i].coords - sites[j].coords)
                if dist < DIST_THRESHOLD:
                    test1_violations.append(
                        (rec.label, ftype, dist))

    # ── Test 2: centroidi geometricamente sensati ────────────────
    if rec.atoms:
        atom_positions = np.array(
            [[a["x"], a["y"], a["z"]] for a in rec.atoms])
        for site in deduped:
            dists = np.linalg.norm(atom_positions - site.coords, axis=1)
            min_dist = float(np.min(dists))
            if min_dist > MAX_CENTROID_DIST:
                test2_violations.append(
                    (rec.label, site.feature_type,
                     min_dist, site.coords.tolist()))

    # ── Test 3: intensità conservata ────────────────────────────
    # tolleranza floating point
    if abs(total_intensity_before - total_intensity_after) > 0.001:
        test3_violations.append(
            (rec.label,
             total_intensity_before,
             total_intensity_after,
             total_intensity_before - total_intensity_after))

    if args.verbose:
        print(f"  {rec.label:15s} "
              f"before={len(features):3d} after={len(deduped):3d} "
              f"intensity: {total_intensity_before:.3f} -> "
              f"{total_intensity_after:.3f}")

# ── Risultati ────────────────────────────────────────────────────
reduction = (total_before - total_after) / total_before * 100 \
    if total_before > 0 else 0

print(f"\nSiti prima della deduplicazione: {total_before}")
print(f"Siti dopo la deduplicazione:     {total_after}")
print(f"Riduzione:                       {total_before - total_after} "
      f"({reduction:.1f}%)")

print(f"\n{'─'*60}")
print(f"[Test 1] Coppie stesso tipo a distanza < {DIST_THRESHOLD} A: "
      f"{len(test1_violations)}")
if test1_violations:
    for label, ftype, dist in test1_violations[:5]:
        print(f"  {label:15s} {ftype:20s} dist={dist:.3f} A")
else:
    print("  ✓ Nessuna coppia rimasta sotto soglia")

print(f"\n[Test 2] Centroidi fuori dalla molecola (> {MAX_CENTROID_DIST} A): "
      f"{len(test2_violations)}")
if test2_violations:
    for label, ftype, dist, coords in test2_violations[:5]:
        print(f"  {label:15s} {ftype:20s} min_dist={dist:.3f} A")
else:
    print("  ✓ Tutti i centroidi sono geometricamente sensati")

print(f"\n[Test 3] Intensità non conservata: {len(test3_violations)}")
if test3_violations:
    for label, before, after, diff in test3_violations[:5]:
        print(f"  {label:15s} before={before:.3f} after={after:.3f} "
              f"diff={diff:.4f}")
else:
    print("  ✓ Intensità conservata per tutti i residui")

print(f"\n{'='*60}")
print("RIEPILOGO")
print(f"{'='*60}")
print(f"Test 1 — Nessun duplicato rimasto: "
      f"{'✓ PASS' if not test1_violations else f'✗ FAIL ({len(test1_violations)} violazioni)'}")
print(f"Test 2 — Centroidi sensati:        "
      f"{'✓ PASS' if not test2_violations else f'✗ FAIL ({len(test2_violations)} violazioni)'}")
print(f"Test 3 — Intensità conservata:     "
      f"{'✓ PASS' if not test3_violations else f'✗ FAIL ({len(test3_violations)} violazioni)'}")