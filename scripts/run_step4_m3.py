"""
Esegue Step 1 -> Step 2 (SASA) -> Step 3 (M3 dedup+centroid locale)
-> Step 4 (PCA). Rimuove duplicati di coordinate per specificita'.
"""
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances

from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features, FEATURE_SPECIFICITY
from amino_lattice.site_dedup_centroid import select_dedup_centroid
from amino_lattice.lattice_fitting import fit_to_lattice_2d

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--surface-filter", action="store_true", default=False)
parser.add_argument("--sasa-threshold", type=float, default=0.5)
parser.add_argument("--max-sites", type=int, default=6)
parser.add_argument("--method", default="pca", choices=["pca", "mds"])
args = parser.parse_args()

if args.surface_filter:
    from amino_lattice.surface_filter import compute_atom_sasa
    sasa_map = compute_atom_sasa(args.pdb)

residues = load_residues_from_pdb(args.pdb)

total_sites = 0
high_stress = []
duplicate_residues = []

def remove_duplicate_coords(sites):
    """Rimuove siti con coordinate duplicate tenendo quello
    con specificita' farmacofora piu' alta."""
    seen = {}
    for site in sites:
        key = tuple(np.round(site.coords, 2))
        if key not in seen:
            seen[key] = site
        else:
            current_spec = FEATURE_SPECIFICITY.get(seen[key].feature_type, 0)
            new_spec = FEATURE_SPECIFICITY.get(site.feature_type, 0)
            if new_spec > current_spec:
                seen[key] = site
    return list(seen.values())

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

    print(f"\n{'='*55}")
    print(f"Residuo: {rec.res_name} {rec.res_seq} (chain {rec.chain_id})")
    print(f"  Feature dopo Step 2 + SASA: {len(features)}")

    if not features:
        print(f"  skip — nessuna feature dopo filtro SASA")
        continue

    # Step 3 M3 — deduplicazione con centroide locale
    sites = select_dedup_centroid(features, mol=rec.mol,
                                  max_sites=args.max_sites)

    # rimuovi duplicati di coordinate per specificita'
    n_before = len(sites)
    sites = remove_duplicate_coords(sites)
    n_after = len(sites)
    if n_before != n_after:
        duplicate_residues.append((rec.res_name, rec.res_seq,
                                   n_before - n_after))

    print(f"  Siti M3 (dopo dedup coords): {len(sites)}")
    for s in sites:
        print(f"    {s.feature_type:20s}  "
              f"coords=({s.coords[0]:.2f},{s.coords[1]:.2f},{s.coords[2]:.2f})")

    if len(sites) < 2:
        print(f"  skip PCA — serve almeno 2 siti")
        continue

    coords_3d = np.array([s.coords for s in sites])
    coords_2d = fit_to_lattice_2d(sites, method=args.method,
                                   lattice_spacing=1.5)

    d3d = pairwise_distances(coords_3d)
    d2d = pairwise_distances(coords_2d)
    num = np.sum((d3d - d2d) ** 2)
    den = np.sum(d3d ** 2)
    stress = float(np.sqrt(num / den)) if den > 0 else 0.0

    total_sites += len(sites)

    print(f"  Stress PCA: {stress:.4f}  {'OK' if stress < 0.1 else '*** ALTO ***'}")
    print(f"  Coordinate 2D:")
    for i, (site, coord2d) in enumerate(zip(sites, coords_2d)):
        print(f"    [{i}] {site.feature_type:20s}  "
              f"3D=({site.coords[0]:.1f},{site.coords[1]:.1f},"
              f"{site.coords[2]:.1f})  "
              f"->  2D=({coord2d[0]:.3f}, {coord2d[1]:.3f})")

    if stress >= 0.1:
        high_stress.append((rec.res_name, rec.res_seq, stress))

print(f"\n{'='*55}")
print(f"TOTALE siti proiettati: {total_sites}")
print(f"Residui con duplicati rimossi: {len(duplicate_residues)}")
for res_name, res_seq, n_removed in duplicate_residues:
    print(f"  {res_name} {res_seq} — rimossi {n_removed} duplicati")
print(f"Residui con stress > 0.1: {len(high_stress)}")
if not high_stress:
    print("  Nessuno.")