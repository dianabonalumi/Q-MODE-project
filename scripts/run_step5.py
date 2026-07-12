"""
Esegue Step 1 -> Step 2 (SASA) -> Step 3 (M3 dedup+centroid)
-> Step 4 (PCA) -> Step 5 (Snapping ungherese)
Misura la perdita di informazione tra coordinate 2D continue e discrete.
"""
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

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

total_sites = 0
snap_errors = []   # distanza euclidea tra coord continua e discreta per ogni sito
high_snap = []     # residui con errore di snapping alto

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

    total_sites += len(sites)

    # misura perdita di informazione snapping
    # distanza euclidea tra coordinata continua e nodo intero assegnato
    max_err = 0.0
    for i, ((ci, cj), (xi, xj)) in enumerate(
            zip(lattice_nodes, coords_2d)):
        err = np.sqrt((ci - xi)**2 + (cj - xj)**2)
        snap_errors.append(err)
        max_err = max(max_err, err)

    # distanze relative preservate dopo snapping
    coords_cont  = np.array([[x, y] for x, y in coords_2d])
    coords_disc  = np.array([[i, j] for i, j in lattice_nodes],
                             dtype=float)
    d_cont = pairwise_distances(coords_cont)
    d_disc = pairwise_distances(coords_disc)
    num = np.sum((d_cont - d_disc)**2)
    den = np.sum(d_cont**2)
    snap_stress = float(np.sqrt(num / den)) if den > 0 else 0.0

    if snap_stress > 0.3:
        high_snap.append((rec.res_name, rec.res_seq, snap_stress))

    print(f"\n{'='*55}")
    print(f"Residuo: {rec.res_name} {rec.res_seq} (chain {rec.chain_id})")
    print(f"  Siti: {len(sites)}  "
          f"Snap stress: {snap_stress:.4f}  "
          f"Max err sito: {max_err:.3f} unita reticolo")
    print(f"  {'Tipo':20s}  {'2D continuo':20s}  ->  {'2D discreto':15s}  err")
    for site, (xi, xj), (ci, cj) in zip(sites, coords_2d, lattice_nodes):
        err = np.sqrt((ci - xi)**2 + (cj - xj)**2)
        print(f"  {site.feature_type:20s}  "
              f"({xi:6.3f}, {xj:6.3f})        ->  "
              f"({ci:3d}, {cj:3d})          {err:.3f}")

print(f"\n{'='*55}")
print(f"TOTALE siti snappati: {total_sites}")
print(f"Errore snapping medio:  {np.mean(snap_errors):.4f} unita reticolo")
print(f"Errore snapping max:    {np.max(snap_errors):.4f} unita reticolo")
print(f"Errore snapping mediano:{np.median(snap_errors):.4f} unita reticolo")
print(f"Residui con snap stress > 0.3: {len(high_snap)}")
if high_snap:
    for res_name, res_seq, s in high_snap:
        print(f"  {res_name} {res_seq}  stress={s:.4f}")
else:
    print("  Nessuno.")