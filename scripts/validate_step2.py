"""
Validazione Step 2: verifica che i tipi farmacofori estratti per ogni
amminoacido siano coerenti con le proprietà chimiche attese.
"""
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from collections import defaultdict
from qmode.pdb_reader import load_residues_from_pdb
from qmode.feature_extraction import extract_features

# Tipi farmacofori ATTESI per ogni amminoacido
# Ogni residuo deve avere ALMENO questi tipi
EXPECTED_TYPES = {
    "ALA": {"HBondDonor", "HBondAcceptor"},
    "ARG": {"HBondDonor", "HBondAcceptor", "PosIonizable"},
    "ASN": {"HBondDonor", "HBondAcceptor"},
    "ASP": {"HBondDonor", "HBondAcceptor", "NegIonizable"},
    "CYS": {"HBondDonor", "HBondAcceptor"},
    "GLN": {"HBondDonor", "HBondAcceptor"},
    "GLU": {"HBondDonor", "HBondAcceptor", "NegIonizable"},
    "GLY": {"HBondDonor", "HBondAcceptor"},
    "HIS": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "ILE": {"HBondDonor", "HBondAcceptor", "Hydrophobe"},
    "LEU": {"HBondDonor", "HBondAcceptor", "Hydrophobe"},
    "LYS": {"HBondDonor", "HBondAcceptor", "PosIonizable"},
    "MET": {"HBondDonor", "HBondAcceptor", "Hydrophobe"},
    "PHE": {"HBondDonor", "HBondAcceptor", "Aromatic", "Hydrophobe"},
    "PRO": {"HBondAcceptor", "Hydrophobe"},
    "SER": {"HBondDonor", "HBondAcceptor"},
    "THR": {"HBondDonor", "HBondAcceptor"},
    "TRP": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "TYR": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "VAL": {"HBondDonor", "HBondAcceptor", "Hydrophobe"},
    # varianti istidina
    "HSD": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "HSE": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "HSP": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "HIE": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "HID": {"HBondDonor", "HBondAcceptor", "Aromatic"},
    "HIP": {"HBondDonor", "HBondAcceptor", "Aromatic", "PosIonizable"},
    "CYX": {"HBondDonor", "HBondAcceptor"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--verbose", action="store_true", default=False,
                    help="Mostra dettaglio per ogni residuo")
args = parser.parse_args()

print(f"\nValidazione Step 2 — Feature Extraction")
print(f"PDB: {args.pdb}")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    residues = load_residues_from_pdb(args.pdb)

# statistiche globali
type_counts = defaultdict(int)   # quante volte appare ogni tipo
missing_types = []               # residui con tipi mancanti
unexpected = []                  # residui senza nessuna feature
total_features = 0

for rec in residues:
    if rec.mol is None:
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(rec.mol, embed_3d=True)

    total_features += len(features)
    found_types = {f.feature_type for f in features}

    for ft in found_types:
        type_counts[ft] += 1

    # verifica tipi attesi
    expected = EXPECTED_TYPES.get(rec.res_name, set())
    missing = expected - found_types

    if not features:
        unexpected.append(rec.label)
    elif missing:
        missing_types.append((rec.label, missing, found_types))

    if args.verbose:
        status = "✓" if not missing else "✗"
        print(f"  {status} {rec.label:15s} found={sorted(found_types)}")
        if missing:
            print(f"      MISSING: {missing}")

# riepilogo distribuzione tipi
print(f"\nDistribuzione tipi farmacofori estratti:")
print(f"{'─'*40}")
for ft, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    pct = count / len(residues) * 100
    print(f"  {ft:20s}  {count:4d} residui  ({pct:.1f}%)")

print(f"\nFeature totali estratte: {total_features}")
print(f"Media per residuo:       {total_features/len(residues):.1f}")

# errori
print(f"\n{'─'*60}")
print(f"Residui senza nessuna feature: {len(unexpected)}")
if unexpected:
    for r in unexpected:
        print(f"  {r}")

print(f"\nResidue con tipi attesi mancanti: {len(missing_types)}")
if missing_types:
    for label, missing, found in missing_types[:20]:
        print(f"  {label:15s} mancanti={missing}  trovati={found}")
    if len(missing_types) > 20:
        print(f"  ... e altri {len(missing_types)-20}")

# riepilogo finale
print(f"\n{'='*60}")
print("RIEPILOGO")
print(f"{'='*60}")
print(f"Residui processati:              {len(residues)}")
print(f"Residui con tutti i tipi attesi: "
      f"{len(residues)-len(missing_types)-len(unexpected)} "
      f"({(len(residues)-len(missing_types)-len(unexpected))/len(residues)*100:.1f}%)")
print(f"Residui con tipi mancanti:       {len(missing_types)} "
      f"({len(missing_types)/len(residues)*100:.1f}%)")
print(f"Residui senza nessuna feature:   {len(unexpected)} "
      f"({len(unexpected)/len(residues)*100:.1f}%)")

if len(missing_types) == 0 and len(unexpected) == 0:
    print("\n✓ Feature extraction corretta per tutti i residui.")
else:
    print("\n⚠ Alcuni residui hanno tipi farmacofori mancanti.")
    print("  Questo può essere normale per residui terminali o")
    print("  con conformazioni particolari nel cristallo.")