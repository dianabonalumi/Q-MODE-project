"""
Validazione Step 3: verifica range e coerenza delle intensità
farmacofore assegnate da Crippen (Hydrophobe) e Abraham (HBond).
"""
import sys
import os
import warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from qmode.pdb_reader import load_residues_from_pdb
from qmode.feature_extraction import extract_features, FEATURE_TYPES
from qmode.abraham_hbond import assign_abraham_hb_intensities

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
args = parser.parse_args()

print(f"\nValidazione Step 3 — Intensity Assignment")
print(f"PDB: {args.pdb}")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    residues = load_residues_from_pdb(args.pdb)

# raccoglie intensità per tipo
intensities_by_type = defaultdict(list)
negative_intensities = []
zero_intensities = []

for rec in residues:
    if rec.mol is None:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(rec.mol, embed_3d=True)

    # assegna intensità Abraham
    assign_abraham_hb_intensities(
        features, res_name=rec.res_name,
        mol=rec.mol, atom_records=rec.atoms
    )

    for feat in features:
        intensities_by_type[feat.feature_type].append(feat.intensity)
        if feat.intensity < 0:
            negative_intensities.append(
                (rec.label, feat.feature_type, feat.intensity))
        if feat.intensity == 0.0:
            zero_intensities.append(
                (rec.label, feat.feature_type))

# statistiche per tipo
print(f"\nStatistiche intensità per tipo farmacoforo:")
print(f"{'─'*65}")
print(f"{'Tipo':20s} {'N':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print(f"{'─'*65}")
for ft in FEATURE_TYPES:
    vals = intensities_by_type.get(ft, [])
    if not vals:
        continue
    arr = np.array(vals)
    print(f"  {ft:18s} {len(vals):5d} {arr.min():8.3f} {arr.max():8.3f} "
          f"{arr.mean():8.3f} {arr.std():8.3f}")

# problemi
print(f"\n{'─'*65}")
print(f"Intensità negative: {len(negative_intensities)}")
if negative_intensities:
    for label, ft, val in negative_intensities[:10]:
        print(f"  {label:15s} {ft:20s} {val:.4f}")

print(f"Intensità zero:     {len(zero_intensities)}")

# verifica ranking biologico
print(f"\n{'─'*65}")
print("Verifica ranking biologico (HBondDonor per residuo):")
print("Atteso: ARG, LYS > GLY, ALA")

hbd_by_res = defaultdict(list)
for rec in residues:
    if rec.mol is None:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(rec.mol, embed_3d=True)
    assign_abraham_hb_intensities(
        features, res_name=rec.res_name,
        mol=rec.mol, atom_records=rec.atoms
    )
    hbd = [f.intensity for f in features if f.feature_type == "HBondDonor"]
    if hbd:
        hbd_by_res[rec.res_name].append(np.mean(hbd))

# stampa media per tipo di residuo
print(f"\n  {'Residuo':8s} {'Media HBD intensity':>20s} {'N campioni':>12s}")
print(f"  {'─'*45}")
for res_name in sorted(hbd_by_res.keys(),
                        key=lambda r: -np.mean(hbd_by_res[r])):
    vals = hbd_by_res[res_name]
    print(f"  {res_name:8s} {np.mean(vals):20.4f} {len(vals):12d}")

# riepilogo
print(f"\n{'='*60}")
print("RIEPILOGO")
print(f"{'='*60}")
total = sum(len(v) for v in intensities_by_type.values())
print(f"Feature totali analizzate: {total}")
print(f"Intensità negative:        {len(negative_intensities)} "
      f"({'OK' if len(negative_intensities) == 0 else 'PROBLEMA'})")
print(f"Intensità zero:            {len(zero_intensities)}")

if len(negative_intensities) == 0:
    print("\n✓ Nessuna intensità negativa — range valori corretto.")
else:
    print("\n✗ Intensità negative trovate — verificare Abraham parameters.")
    