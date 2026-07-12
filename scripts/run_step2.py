"""
Esegue solo Step 1 + Step 2 e stampa le feature farmacofore estratte.
Supporta il confronto tra estrazione originale e filtrata per SASA.
"""
import argparse
import numpy as np
from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--surface-filter", action="store_true", default=False)
parser.add_argument("--sasa-threshold", type=float, default=1.0)
args = parser.parse_args()

if args.surface_filter:
    from amino_lattice.surface_filter import compute_atom_sasa
    sasa_map = compute_atom_sasa(args.pdb)

residues = load_residues_from_pdb(args.pdb)

for rec in residues:
    # Step 2 con embed_3d=True per avere coordinate 3D reali
    features = extract_features(rec.mol, embed_3d=True)

    if args.surface_filter:
        # Costruisce mappa nome_atomo → SASA per questo residuo
        atom_sasa = {}
        for atom in rec.atoms:
            key = (rec.chain_id, rec.res_seq, atom["name"].strip())
            atom_sasa[atom["name"].strip()] = sasa_map.get(key, 0.0)

        # Costruisce array posizioni 3D reali degli atomi del PDB
        atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in rec.atoms])
        atom_names = [a["name"].strip() for a in rec.atoms]

        filtered = []
        for feat in features:
            # Trova l'atomo PDB più vicino al centroide della feature
            # usando le coordinate 3D reali (non quelle RDKit locali)
            dists = np.linalg.norm(atom_positions - feat.coords, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_name = atom_names[nearest_idx]
            key = (rec.chain_id, rec.res_seq, nearest_name)
            sasa_val = sasa_map.get(key, 0.0)
            if sasa_val > args.sasa_threshold:
                filtered.append(feat)
        features = filtered

    print(f"\n{'='*50}")
    print(f"Residuo: {rec.res_name} {rec.res_seq} (chain {rec.chain_id})")
    print(f"Feature estratte: {len(features)}")
    for f in features:
        print(f"  {f.feature_type:20s}  coords=({f.coords[0]:.2f}, {f.coords[1]:.2f}, {f.coords[2]:.2f})  intensity={f.intensity:.3f}")