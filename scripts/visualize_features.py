"""
Visualizzazione dei punti farmacofori prima e dopo il filtro SASA.
Usa la proteina completa come contesto e mostra la superficie molecolare.
"""
import argparse
import numpy as np

from qmode.pdb_reader import load_residues_from_pdb
from qmode.feature_extraction import extract_features

COLORS = {
    "HBondDonor":    "blue",
    "HBondAcceptor": "red",
    "Hydrophobe":    "yellow",
    "Aromatic":      "orange",
    "PosIonizable":  "cyan",
    "NegIonizable":  "magenta",
}

parser = argparse.ArgumentParser()
parser.add_argument("--pdb-pocket", required=True,
                    help="PDB della tasca (per estrarre feature)")
parser.add_argument("--pdb-protein", required=True,
                    help="PDB della proteina completa (per visualizzazione)")
parser.add_argument("--sasa-threshold", type=float, default=0.5)
parser.add_argument("--output", default="features_visualization.html")
args = parser.parse_args()

from qmode.surface_filter import compute_atom_sasa
sasa_map = compute_atom_sasa(args.pdb_pocket)

residues = load_residues_from_pdb(args.pdb_pocket)

all_features = []
filtered_features = []

for rec in residues:
    features = extract_features(rec.mol, embed_3d=True)
    atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in rec.atoms])
    atom_names = [a["name"].strip() for a in rec.atoms]

    for feat in features:
        all_features.append(feat)
        dists = np.linalg.norm(atom_positions - feat.coords, axis=1)
        nearest_name = atom_names[int(np.argmin(dists))]
        key = (rec.chain_id, rec.res_seq, nearest_name)
        if sasa_map.get(key, 0.0) > args.sasa_threshold:
            filtered_features.append(feat)

with open(args.pdb_protein, encoding="utf-8") as f:
    protein_str = f.read()

with open(args.pdb_pocket, encoding="utf-8") as f:
    pocket_str = f.read()

def make_spheres_js(features, viewer_var):
    lines = []
    for feat in features:
        x, y, z = feat.coords
        color = COLORS.get(feat.feature_type, "white")
        lines.append(
            f'{viewer_var}.addSphere({{center:{{x:{x:.2f},y:{y:.2f},z:{z:.2f}}},'
            f'radius:0.5,color:"{color}",opacity:0.9}});'
        )
    return "\n".join(lines)

spheres_all      = make_spheres_js(all_features, "viewer1")
spheres_filtered = make_spheres_js(filtered_features, "viewer2")

legend_html = "".join([
    f'<span style="color:{c};margin-right:12px;">&#9679; {t}</span>'
    for t, c in COLORS.items()
])

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pharmacophore Features</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: white; margin: 0; padding: 10px; }}
    h2 {{ text-align: center; margin-bottom: 4px; }}
    .legend {{ text-align: center; padding: 6px; font-size: 13px; }}
    .container {{ display: flex; justify-content: space-around; gap: 10px; }}
    .panel {{ width: 49%; }}
    h3 {{ text-align: center; margin: 6px 0; }}
    .stats {{ text-align: center; font-size: 12px; color: #aaa; margin-bottom: 4px; }}
    .viewer {{ width: 100%; height: 520px; position: relative; border: 1px solid #333; border-radius: 6px; }}
    .controls {{ text-align: center; font-size: 11px; color: #666; margin-top: 4px; }}
  </style>
</head>
<body>
  <h2>Pharmacophore Feature Visualization</h2>
  <div class="legend">{legend_html}</div>
  <div class="container">
    <div class="panel">
      <h3>Before SASA filter &mdash; {len(all_features)} features</h3>
      <div class="stats">All features including buried atoms</div>
      <div id="viewer1" class="viewer"></div>
      <div class="controls">Scroll to zoom &middot; Click+drag to rotate &middot; Right-click+drag to translate</div>
    </div>
    <div class="panel">
      <h3>After SASA filter &mdash; {len(filtered_features)} features</h3>
      <div class="stats">Surface-exposed only (SASA threshold = {args.sasa_threshold} A²)</div>
      <div id="viewer2" class="viewer"></div>
      <div class="controls">Scroll to zoom &middot; Click+drag to rotate &middot; Right-click+drag to translate</div>
    </div>
  </div>
  <script>
    var proteinStr = {repr(protein_str)};
    var pocketStr  = {repr(pocket_str)};

    var viewer1 = $3Dmol.createViewer("viewer1", {{backgroundColor:"#0d1117"}});
    viewer1.addModel(proteinStr, "pdb");
    viewer1.setStyle({{}}, {{surface:{{opacity:0.15, color:"#4a90d9"}}}});
    viewer1.addModel(pocketStr, "pdb");
    viewer1.setStyle({{}}, {{cartoon:{{color:"#7ab8f5", opacity:0.6}}}});
    {spheres_all}
    viewer1.zoomTo();
    viewer1.render();

    var viewer2 = $3Dmol.createViewer("viewer2", {{backgroundColor:"#0d1117"}});
    viewer2.addModel(proteinStr, "pdb");
    viewer2.setStyle({{}}, {{surface:{{opacity:0.15, color:"#4a90d9"}}}});
    viewer2.addModel(pocketStr, "pdb");
    viewer2.setStyle({{}}, {{cartoon:{{color:"#7ab8f5", opacity:0.6}}}});
    {spheres_filtered}
    viewer2.zoomTo();
    viewer2.render();
  </script>
</body>
</html>"""

with open(args.output, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved: {args.output}")
print(f"Features before: {len(all_features)}")
print(f"Features after:  {len(filtered_features)}")
print(f"Reduction: {len(all_features)-len(filtered_features)} ({(1-len(filtered_features)/len(all_features))*100:.1f}%)")