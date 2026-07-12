"""
Esegue la pipeline completa Step 1 -> Step 5 e stampa la flat chain finale.
Step 2: filtro SASA
Step 3: M3 dedup+centroid con rimozione duplicati per specificita'
Step 4: PCA
Step 5: Snapping ungherese
"""
import argparse
import numpy as np

from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features, FEATURE_SPECIFICITY
from amino_lattice.site_dedup_centroid import select_dedup_centroid
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice

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

def remove_duplicate_coords(sites):
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

# flat chain finale
flat_chain = []

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

    # Step 3 M3
    sites = select_dedup_centroid(features, mol=rec.mol,
                                  max_sites=args.max_sites)
    sites = remove_duplicate_coords(sites)

    if len(sites) < 2:
        continue

    # Step 4 PCA
    coords_2d = fit_to_lattice_2d(sites, method=args.method,
                                   lattice_spacing=1.5)

    # Step 5 Snapping
    lattice_nodes = snap_to_lattice(coords_2d, strategy="hungarian")

    # aggiungi alla flat chain
    for site, (i, j) in zip(sites, lattice_nodes):
        flat_chain.append({
            "residue":      f"{rec.res_name} {rec.res_seq}",
            "chain":        rec.chain_id,
            "feature_type": site.feature_type,
            "intensity":    site.intensity,
            "i":            i,
            "j":            j,
        })

# stampa flat chain
print(f"\n{'='*70}")
print(f"FLAT CHAIN — {len(flat_chain)} siti totali")
print(f"{'='*70}")
print(f"{'#':>4}  {'Residuo':12}  {'Tipo':20}  {'(i,j)':12}  Intensity")
print(f"{'-'*70}")
for idx, entry in enumerate(flat_chain):
    print(f"{idx:>4}  "
          f"{entry['residue']:12}  "
          f"{entry['feature_type']:20}  "
          f"({entry['i']:3d},{entry['j']:3d})     "
          f"{entry['intensity']:.3f}")

print(f"\n{'='*70}")
print(f"Totale siti nella flat chain: {len(flat_chain)}")
print(f"Residui rappresentati: {len(set(e['residue'] for e in flat_chain))}")

# distribuzione per tipo farmacoforo
from collections import Counter
type_counts = Counter(e['feature_type'] for e in flat_chain)
print(f"\nDistribuzione per tipo farmacoforo:")
for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    pct = count / len(flat_chain) * 100
    print(f"  {ftype:20s}  {count:4d}  ({pct:.1f}%)")