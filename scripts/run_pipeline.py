"""
Pipeline completa Step 1-5 con le modifiche implementate:
- Step 2: filtro SASA (Shrake & Rupley 1973)
- Step 3: M3 dedup+centroid con rimozione duplicati per specificita'
- Step 4: PCA 3D -> 2D
- Step 5: Snapping ungherese

Output: flat_chain.json — input per l'encoding quantistico (Step 6-8)
"""
import argparse
import json
import numpy as np

from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features, FEATURE_SPECIFICITY
from amino_lattice.site_dedup_centroid import select_dedup_centroid
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True,
                    help="File PDB della tasca")
parser.add_argument("--output", default="data/processed/flat_chain.json",
                    help="File JSON di output (default: data/processed/flat_chain.json)")
parser.add_argument("--surface-filter", action="store_true", default=False,
                    help="Applica filtro SASA")
parser.add_argument("--sasa-threshold", type=float, default=0.5,
                    help="Soglia SASA in A² (default: 0.5)")
parser.add_argument("--max-sites", type=int, default=6,
                    help="Numero massimo siti per residuo (default: 6)")
parser.add_argument("--method", default="pca", choices=["pca", "mds"],
                    help="Metodo proiezione 2D (default: pca)")
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

flat_chain = []

for rec in residues:
    # Step 2
    features = extract_features(rec.mol, embed_3d=True)

    # Filtro SASA
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
            "residue_name":  rec.res_name,
            "residue_seq":   rec.res_seq,
            "chain_id":      rec.chain_id,
            "feature_type":  site.feature_type,
            "intensity":     float(site.intensity),
            "coords_3d":     [float(site.coords[0]),
                              float(site.coords[1]),
                              float(site.coords[2])],
            "lattice_i":     int(i),
            "lattice_j":     int(j),
        })

# salva JSON
import os
os.makedirs(os.path.dirname(args.output), exist_ok=True)

output_data = {
    "metadata": {
        "pdb":            args.pdb,
        "surface_filter": args.surface_filter,
        "sasa_threshold": args.sasa_threshold,
        "max_sites":      args.max_sites,
        "method":         args.method,
        "total_sites":    len(flat_chain),
        "total_residues": len(set(
            f"{e['residue_name']}_{e['residue_seq']}"
            for e in flat_chain
        )),
    },
    "flat_chain": flat_chain,
}

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Flat chain salvata in: {args.output}")
print(f"Siti totali:    {len(flat_chain)}")
print(f"Residui:        {output_data['metadata']['total_residues']}")