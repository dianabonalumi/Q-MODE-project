"""
Confronto Step 3 — tre metodi:
  1. K-Means
  2. Centroide per tipo (un sito per tipo farmacoforo)
  3. Deduplicazione con centroide locale (vicini fusi, lontani separati)

Input: Step 2 filtrato con SASA.
"""
import argparse
import numpy as np
from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features
from amino_lattice.site_selection import choose_k, select_representative_sites
from amino_lattice.site_centroid_by_type import select_centroid_by_type
from amino_lattice.site_dedup_centroid import select_dedup_centroid

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--surface-filter", action="store_true", default=False)
parser.add_argument("--sasa-threshold", type=float, default=0.5)
parser.add_argument("--max-k", type=int, default=6)
args = parser.parse_args()

if args.surface_filter:
    from amino_lattice.surface_filter import compute_atom_sasa
    sasa_map = compute_atom_sasa(args.pdb)

residues = load_residues_from_pdb(args.pdb)

total_kmeans = 0
total_type   = 0
total_dedup  = 0

for rec in residues:
    features = extract_features(rec.mol, embed_3d=True)

    if args.surface_filter:
        atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in rec.atoms])
        atom_names = [a["name"].strip() for a in rec.atoms]
        filtered = []
        for feat in features:
            dists = np.linalg.norm(atom_positions - feat.coords, axis=1)
            nearest_name = atom_names[int(np.argmin(dists))]
            key = (rec.chain_id, rec.res_seq, nearest_name)
            if sasa_map.get(key, 0.0) > args.sasa_threshold:
                filtered.append(feat)
        features = filtered

    if not features:
        continue

    # Metodo 1 — K-Means
    k = choose_k(features, mol=rec.mol, strategy="active_features",
                 max_k=args.max_k)
    sites_kmeans = select_representative_sites(features, k, mol=rec.mol)

    # Metodo 2 — Centroide per tipo
    sites_type = select_centroid_by_type(features, mol=rec.mol)

    # Metodo 3 — Deduplicazione con centroide locale
    sites_dedup = select_dedup_centroid(features, mol=rec.mol,
                                        max_sites=args.max_k)

    total_kmeans += len(sites_kmeans)
    total_type   += len(sites_type)
    total_dedup  += len(sites_dedup)

    print(f"\n{'='*60}")
    print(f"Residuo: {rec.res_name} {rec.res_seq} (chain {rec.chain_id})")
    print(f"  Feature Step 2 filtrate: {len(features)}")

    print(f"  [M1 K-Means]        → {len(sites_kmeans)} siti")
    for s in sites_kmeans:
        print(f"    {s.feature_type:20s}  coords=({s.coords[0]:.1f},{s.coords[1]:.1f},{s.coords[2]:.1f})  int={s.intensity:.2f}")

    print(f"  [M2 Centroide/tipo] → {len(sites_type)} siti")
    for s in sites_type:
        print(f"    {s.feature_type:20s}  coords=({s.coords[0]:.1f},{s.coords[1]:.1f},{s.coords[2]:.1f})  int={s.intensity:.2f}")

    print(f"  [M3 Dedup+centroid] → {len(sites_dedup)} siti")
    for s in sites_dedup:
        print(f"    {s.feature_type:20s}  coords=({s.coords[0]:.1f},{s.coords[1]:.1f},{s.coords[2]:.1f})  int={s.intensity:.2f}")

print(f"\n{'='*60}")
print(f"TOTALE siti K-Means:              {total_kmeans}")
print(f"TOTALE siti centroide per tipo:   {total_type}")
print(f"TOTALE siti dedup+centroid:       {total_dedup}")