"""
Visualizza la flat chain 3D per uno o più residui o per tutta la proteina.
Con --residue mostra solo un residuo, con --residues mostra più residui,
mantenendo sempre l'indice globale nella flat chain completa.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmode.pdb_reader import load_residues_from_pdb
from qmode.feature_extraction import extract_features
from qmode.surface_filter import compute_atom_sasa, filter_surface_features_by_coords
from qmode.site_selection import topological_order
from qmode.site_dedup_centroid import select_dedup_centroid

COLORS = {
    "HBondDonor":    "blue",
    "HBondAcceptor": "red",
    "Hydrophobe":    "orange",
    "Aromatic":      "purple",
    "PosIonizable":  "cyan",
    "NegIonizable":  "magenta",
}

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True)
parser.add_argument("--sasa-threshold", type=float, default=0.5)
parser.add_argument("--output", default="chain_visualization.html")
parser.add_argument("--residue", default=None,
                    help="Mostra solo questo residuo es. A159_ARG")
parser.add_argument("--residues", nargs="+", default=None,
                    help="Mostra più residui es. A158_ARG A159_ARG A160_GLU")
args = parser.parse_args()

# Calcola SASA
sasa_map = compute_atom_sasa(args.pdb)

# Carica residui
residues = load_residues_from_pdb(args.pdb)

# Costruisce PRIMA la flat chain completa per avere gli indici globali
full_chain = []

for rec in residues:
    if rec.mol is None:
        continue

    features = extract_features(rec.mol, embed_3d=True)
    if not features:
        continue

    features = filter_surface_features_by_coords(
        features=features,
        sasa_map=sasa_map,
        chain_id=rec.chain_id,
        res_seq=rec.res_seq,
        all_atom_coords=rec.atoms,
        sasa_threshold=args.sasa_threshold,
    )
    if not features:
        continue

    features = select_dedup_centroid(features, mol=rec.mol, max_sites=12)
    if not features:
        continue

    sites = topological_order(features, mol=rec.mol)

    for site in sites:
        full_chain.append({
            "residue": rec.label,
            "type":    site.feature_type,
            "x":       float(site.coords[0]),
            "y":       float(site.coords[1]),
            "z":       float(site.coords[2]),
        })

# Determina il filtro residui
selected = set()
if args.residues:
    selected = set(args.residues)
elif args.residue:
    selected = {args.residue}

# Filtra mantenendo l'indice globale
flat_chain = [
    {"global_idx": i + 1, **s}
    for i, s in enumerate(full_chain)
    if not selected or s["residue"] in selected
]

print(f"Totale siti nella catena completa: {len(full_chain)}")
print(f"Siti selezionati:                  {len(flat_chain)}")
for s in flat_chain:
    print(f"  Posizione {s['global_idx']:3d} — {s['type']:20s} "
          f"({s['x']:.1f}, {s['y']:.1f}, {s['z']:.1f})")

# Legge la proteina per il contesto 3D
with open(args.pdb, encoding="utf-8") as f:
    protein_str = f.read()


def make_chain_js(flat_chain, viewer_var):
    lines = []
    for site in flat_chain:
        color = COLORS.get(site["type"], "gray")
        lines.append(
            f'{viewer_var}.addSphere({{center:{{x:{site["x"]:.2f},'
            f'y:{site["y"]:.2f},z:{site["z"]:.2f}}},'
            f'radius:0.8,color:"{color}",opacity:0.95}});'
        )
    for i in range(len(flat_chain) - 1):
        s1 = flat_chain[i]
        s2 = flat_chain[i + 1]
        same_res = s1["residue"] == s2["residue"]
        width = 0.12 if same_res else 0.04
        color = '"#333333"' if same_res else '"#aaaaaa"'
        lines.append(
            f'{viewer_var}.addCylinder({{'
            f'start:{{x:{s1["x"]:.2f},y:{s1["y"]:.2f},z:{s1["z"]:.2f}}},'
            f'end:{{x:{s2["x"]:.2f},y:{s2["y"]:.2f},z:{s2["z"]:.2f}}},'
            f'radius:{width},color:{color},opacity:0.7}});'
        )
    return "\n".join(lines)


chain_js = make_chain_js(flat_chain, "viewer")

legend_html = "".join([
    f'<span style="color:{c};margin-right:12px;font-weight:bold;">&#9679; {t}</span>'
    for t, c in COLORS.items()
])

table_rows = ""
prev_res = None
for site in flat_chain:
    if site["residue"] != prev_res:
        table_rows += (
            f'<tr style="background:#f0f0f0">'
            f'<td colspan="3"><b>{site["residue"]}</b></td>'
            f'</tr>'
        )
        prev_res = site["residue"]
    color = COLORS.get(site["type"], "gray")
    table_rows += (
        f'<tr>'
        f'<td style="font-weight:bold;color:#2c5f8a;text-align:center">'
        f'{site["global_idx"]}</td>'
        f'<td><span style="color:{color};font-weight:bold;">&#9679;</span> '
        f'{site["type"]}</td>'
        f'<td style="font-size:11px;color:#555">'
        f'({site["x"]:.1f}, {site["y"]:.1f}, {site["z"]:.1f})</td>'
        f'</tr>'
    )

if args.residues:
    title = " — ".join(args.residues)
elif args.residue:
    title = f"Residuo {args.residue}"
else:
    title = "Flat Chain — proteina completa"

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Chain Visualization</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f5f5; color: #1a1a2e;
            margin: 0; padding: 10px; display: flex; flex-direction: column; }}
    h2 {{ text-align: center; margin-bottom: 4px; }}
    .legend {{ text-align: center; padding: 8px; font-size: 13px; background: white;
               border-radius: 6px; margin-bottom: 8px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .main {{ display: flex; gap: 12px; height: 580px; }}
    #viewer {{ flex: 1; border: 1px solid #ccc; border-radius: 6px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
    .table-panel {{ width: 360px; overflow-y: auto; background: white;
                    border: 1px solid #ccc; border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ background: #e8e8e8; padding: 6px 8px; text-align: left; }}
    td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
    .stats {{ text-align: center; font-size: 12px; color: #555;
              margin: 4px 0 8px 0; }}
    .controls {{ text-align: center; font-size: 11px; color: #888;
                 margin-top: 4px; }}
  </style>
</head>
<body>
  <h2>Pharmacophore Sites — {title}</h2>
  <div class="legend">{legend_html}</div>
  <div class="stats">
    {len(flat_chain)} siti selezionati &mdash;
    catena completa: {len(full_chain)} siti totali &mdash;
    linee spesse = stesso residuo &mdash;
    linee sottili = cambio residuo
  </div>
  <div class="main">
    <div id="viewer"></div>
    <div class="table-panel">
      <table>
        <thead>
          <tr>
            <th>Pos.</th>
            <th>Tipo farmacoforo</th>
            <th>Coords 3D</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="controls">
    Scroll to zoom &middot; Click+drag to rotate &middot; Right-click+drag to translate
  </div>
  <script>
    var proteinStr = {repr(protein_str)};
    var viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"white"}});
    viewer.addModel(proteinStr, "pdb");
    viewer.setStyle({{}}, {{cartoon:{{color:"#aac4dd", opacity:0.4}}}});
    {chain_js}
    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>"""

with open(args.output, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved: {args.output}")