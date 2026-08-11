"""
Validazione Step 5: ordinamento topologico BFS.
Verifica che:
  1. L'ordine sia deterministico (due esecuzioni producono lo stesso risultato)
  2. Il primo sito di ogni residuo corrisponda sempre al backbone
  3. L'ordine segua la connettività chimica attesa
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
from qmode.site_dedup_centroid import select_dedup_centroid
from qmode.site_selection import topological_order

# Atomi del backbone — il primo sito dovrebbe essere associato a uno di questi
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()

print(f"\nValidazione Step 5 — Topological Ordering")
print(f"PDB: {args.pdb}")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    residues = load_residues_from_pdb(args.pdb)

sasa_map = compute_atom_sasa(args.pdb)

# ── Test 1: Determinismo ─────────────────────────────────────────
print("\n[Test 1] Determinismo — due esecuzioni producono lo stesso ordine")

non_deterministic = []
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

    features = select_dedup_centroid(features, mol=rec.mol, max_sites=12)
    if not features:
        continue

    # esegue due volte e confronta
    order1 = [s.feature_type for s in topological_order(features, mol=rec.mol)]
    order2 = [s.feature_type for s in topological_order(features, mol=rec.mol)]

    if order1 != order2:
        non_deterministic.append(rec.label)

print(f"  Residui non deterministici: {len(non_deterministic)}")
if non_deterministic:
    for r in non_deterministic:
        print(f"    {r}")
else:
    print("  ✓ Ordine deterministico per tutti i residui")

# ── Test 2: Backbone prima ───────────────────────────────────────
print("\n[Test 2] Backbone — il primo sito è associato al backbone")

backbone_first = 0
backbone_not_first = []
total_checked = 0

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

    features = select_dedup_centroid(features, mol=rec.mol, max_sites=12)
    if len(features) < 2:
        continue

    ordered = topological_order(features, mol=rec.mol)
    total_checked += 1

    # trova l'atomo PDB più vicino al primo sito
    atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in rec.atoms])
    atom_names = [a["name"].strip() for a in rec.atoms]
    first_site = ordered[0]
    dists = np.linalg.norm(atom_positions - first_site.coords, axis=1)
    nearest_atom = atom_names[int(np.argmin(dists))]

    is_backbone = nearest_atom in BACKBONE_ATOMS
    if is_backbone:
        backbone_first += 1
    else:
        backbone_not_first.append((rec.label, nearest_atom,
                                   first_site.feature_type))

    if args.verbose:
        status = "✓" if is_backbone else "✗"
        print(f"  {status} {rec.label:15s} first_atom={nearest_atom:6s} "
              f"type={first_site.feature_type}")

pct = backbone_first / total_checked * 100 if total_checked > 0 else 0
print(f"  Residui con backbone come primo sito: "
      f"{backbone_first}/{total_checked} ({pct:.1f}%)")

if backbone_not_first:
    print(f"  Residui con primo sito non backbone:")
    for label, atom, ftype in backbone_not_first[:10]:
        print(f"    {label:15s} first_atom={atom:6s} type={ftype}")

# ── Test 3: Coerenza ordine per tipo di residuo ──────────────────
print("\n[Test 3] Coerenza ordine per tipo di residuo")
print("  (confronta ordine dei tipi tra residui dello stesso tipo)")

order_by_resname = defaultdict(list)
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

    features = select_dedup_centroid(features, mol=rec.mol, max_sites=12)
    if not features:
        continue

    ordered = topological_order(features, mol=rec.mol)
    order_by_resname[rec.res_name].append(
        tuple(s.feature_type for s in ordered))

print(f"\n  {'Residuo':8s} {'N istanze':>10s} {'Ordini distinti':>16s} "
      f"{'Coerente':>10s}")
print(f"  {'─'*50}")
inconsistent = []
for res_name, orders in sorted(order_by_resname.items()):
    unique_orders = set(orders)
    consistent = len(unique_orders) == 1
    if not consistent:
        inconsistent.append(res_name)
    status = "✓" if consistent else "✗"
    print(f"  {status} {res_name:8s} {len(orders):10d} "
          f"{len(unique_orders):16d}")

# ── Riepilogo ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RIEPILOGO")
print(f"{'='*60}")
print(f"Test 1 — Determinismo:        "
      f"{'✓ PASS' if not non_deterministic else '✗ FAIL'}")
print(f"Test 2 — Backbone primo:      "
      f"{'✓ PASS' if pct >= 80 else '✗ FAIL'} ({pct:.1f}%)")
print(f"Test 3 — Coerenza per tipo:   "
      f"{'✓ PASS' if not inconsistent else f'✗ {len(inconsistent)} tipi inconsistenti'}")