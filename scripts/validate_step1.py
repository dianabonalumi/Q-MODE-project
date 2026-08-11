"""
Validazione Step 1: verifica che il template matching produca
topologie molecolari corrette per ogni residuo.
Controlla:
  1. Quanti residui vengono saltati vs processati
  2. Che ogni residuo abbia il numero corretto di atomi pesanti
  3. Che i legami aromatici siano assegnati correttamente
  4. Che le cariche formali siano corrette per residui ionizzabili
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
from rdkit import Chem
from qmode.pdb_reader import load_residues_from_pdb

# Numero atteso di atomi pesanti per ogni amminoacido standard
EXPECTED_HEAVY_ATOMS = {
    "ALA": 5,  "ARG": 11, "ASN": 8,  "ASP": 8,
    "CYS": 6,  "GLN": 9,  "GLU": 9,  "GLY": 4,
    "HIS": 10, "ILE": 8,  "LEU": 8,  "LYS": 9,
    "MET": 8,  "PHE": 11, "PRO": 7,  "SER": 6,
    "THR": 7,  "TRP": 14, "TYR": 12, "VAL": 7,
    # varianti istidina
    "HSD": 10, "HSE": 10, "HSP": 10,
    "HIE": 10, "HID": 10, "HIP": 10,
    "CYX": 6,
}

# Residui che devono avere aromaticità
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS", "HSD", "HSE", "HSP",
                     "HIE", "HID", "HIP"}

# Residui con carica formale attesa
EXPECTED_CHARGE = {
    "ARG": 1, "LYS": 1,
    "ASP": -1, "GLU": -1,
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
args = parser.parse_args()

print(f"\nValidazione Step 1 — {args.pdb}")
print("=" * 60)

# Sopprime i warning di RDKit durante il caricamento
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    residues = load_residues_from_pdb(args.pdb)

# Conta warning
skipped = [w for w in caught_warnings if "saltato" in str(w.message).lower()
           or "fallita" in str(w.message).lower()]

print(f"\nResidue trovati nel PDB:    {len(residues) + len(skipped)}")
print(f"Residui processati:         {len(residues)}")
print(f"Residui saltati (warning):  {len(skipped)}")
if skipped:
    print("  Residui saltati:")
    for w in skipped:
        print(f"    {w.message}")

# Verifica topologia per ogni residuo processato
print(f"\n{'─'*60}")
print("Verifica topologia molecolare:")
print(f"{'─'*60}")

errors = []
warnings_list = []

for rec in residues:
    if rec.mol is None:
        errors.append(f"  {rec.label}: mol is None")
        continue

    mol = rec.mol
    mol_no_h = Chem.RemoveHs(mol)
    n_heavy = mol_no_h.GetNumAtoms()

    # 1. Numero atomi pesanti
    expected = EXPECTED_HEAVY_ATOMS.get(rec.res_name)
    if expected is not None:
        # tolleranza ±1 per varianti terminali
        if abs(n_heavy - expected) > 1:
            warnings_list.append(
                f"  {rec.label}: heavy atoms = {n_heavy}, expected ~{expected}"
            )

    # 2. Aromaticità
    if rec.res_name in AROMATIC_RESIDUES:
        aromatic_atoms = [a for a in mol_no_h.GetAtoms() if a.GetIsAromatic()]
        if len(aromatic_atoms) == 0:
            errors.append(
                f"  {rec.label}: should have aromatic atoms but found none"
            )

    # 3. Carica formale
    expected_charge = EXPECTED_CHARGE.get(rec.res_name)
    if expected_charge is not None:
        actual_charge = Chem.GetFormalCharge(mol_no_h)
        if actual_charge != expected_charge:
            warnings_list.append(
                f"  {rec.label}: formal charge = {actual_charge}, "
                f"expected {expected_charge}"
            )

    # 4. Conformero 3D presente
    if mol.GetNumConformers() == 0:
        errors.append(f"  {rec.label}: no 3D conformer")

print(f"\nErrori critici: {len(errors)}")
for e in errors:
    print(e)

print(f"\nWarning (valori inattesi): {len(warnings_list)}")
for w in warnings_list:
    print(w)

# Riepilogo finale
print(f"\n{'='*60}")
print("RIEPILOGO")
print(f"{'='*60}")
total = len(residues) + len(skipped)
print(f"Residui totali nel PDB:     {total}")
print(f"Residui processati:         {len(residues)} "
      f"({len(residues)/total*100:.1f}%)")
print(f"Residui saltati:            {len(skipped)} "
      f"({len(skipped)/total*100:.1f}%)")
print(f"Errori critici topologia:   {len(errors)}")
print(f"Warning topologia:          {len(warnings_list)}")

if len(errors) == 0 and len(warnings_list) == 0:
    print("\n✓ Topologia corretta per tutti i residui processati.")
elif len(errors) == 0:
    print("\n✓ Nessun errore critico. Alcuni warning da verificare.")
else:
    print("\n✗ Errori critici trovati — verificare i residui elencati.")