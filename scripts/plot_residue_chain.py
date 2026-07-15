"""
Visualizza un amminoacido: struttura 3D con i siti farmacoforici
superficiali, e la loro disposizione nella catena 1D (stesso ordinamento
topologico di run_pocket.py).

Uso:
    python scripts/plot_residue_chain.py --pdb data/raw/4dfr_pocket.pdb --chain A --resseq 30
    python scripts/plot_residue_chain.py --pdb data/raw/4dfr_pocket.pdb --chain A --resseq 30 \
        --surface-filter --save-plot out/A30_TRP.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amino_lattice.pdb_reader import load_residues_from_pdb
from amino_lattice.feature_extraction import extract_features
from amino_lattice.site_selection import topological_order
from amino_lattice.surface_filter import compute_atom_sasa, filter_surface_features_by_coords
from amino_lattice.abraham_hbond import assign_abraham_hb_intensities


COLOR_MAP = {
    "HBondDonor":    "#2196F3",
    "HBondAcceptor": "#4CAF50",
    "Hydrophobe":    "#FF9800",
    "Aromatic":      "#9C27B0",
    "PosIonizable":  "#F44336",
    "NegIonizable":  "#00BCD4",
}
TYPE_SHORT = {
    "HBondDonor": "HBD", "HBondAcceptor": "HBA",
    "Hydrophobe": "Hyd", "Aromatic": "Aro",
    "PosIonizable": "Pos", "NegIonizable": "Neg",
}
ELEMENT_COLOR = {
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "S": "#FFC832", "H": "#FFFFFF",
}


# ─────────────────────────────────────────────────────────────────────────────

def find_residue(residues, chain_id: str, res_seq: int):
    for rec in residues:
        if rec.chain_id == chain_id and rec.res_seq == res_seq:
            return rec
    return None


def get_ordered_sites(rec, surface_filter=False, sasa_map=None, sasa_threshold=1.0):
    """Stesso Step 2+3 di run_pocket.py: feature, intensità hb (Abraham),
    filtro SASA opzionale, ordinamento topologico (no K-Means)."""
    features = extract_features(rec.mol, embed_3d=True)
    if not features:
        return []

    assign_abraham_hb_intensities(features, res_name=rec.res_name, mol=rec.mol, atom_records=rec.atoms)

    if surface_filter and sasa_map is not None:
        features = filter_surface_features_by_coords(
            features=features,
            sasa_map=sasa_map,
            chain_id=rec.chain_id,
            res_seq=rec.res_seq,
            all_atom_coords=rec.atoms,
            sasa_threshold=sasa_threshold,
        )

    return topological_order(features, mol=rec.mol)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_molecule_3d(ax, mol):
    """Disegna atomi pesanti + legami dal conformatore 3D reale (coordinate PDB)."""
    from rdkit.Chem import RemoveHs

    mol_no_h = RemoveHs(mol)
    conf = mol_no_h.GetConformer()
    coords = conf.GetPositions()

    for bond in mol_no_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1, p2 = coords[i], coords[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="#999999", lw=1.5, zorder=1)

    for atom in mol_no_h.GetAtoms():
        pos = coords[atom.GetIdx()]
        color = ELEMENT_COLOR.get(atom.GetSymbol(), "#CCCCCC")
        ax.scatter(*pos, color=color, s=70, edgecolors="black",
                   linewidths=0.5, zorder=2)


def plot_sites_3d(ax, sites):
    """Sovrappone i siti farmacoforici (centroidi) e la linea che segue
    l'ordine della catena 1D, proiettata nello spazio 3D reale."""
    if not sites:
        return

    xs = [s.coords[0] for s in sites]
    ys = [s.coords[1] for s in sites]
    zs = [s.coords[2] for s in sites]

    if len(sites) > 1:
        ax.plot(xs, ys, zs, "--", color="#333333", lw=1.3, zorder=3)

    for idx, s in enumerate(sites):
        color = COLOR_MAP.get(s.feature_type, "#607D8B")
        ax.scatter(*s.coords, color=color, s=240, edgecolors="white",
                   linewidths=1.8, zorder=4)
        ax.text(*s.coords, str(idx), fontsize=8, color="black",
                fontweight="bold", zorder=5, ha="center", va="center")


def plot_1d_chain(ax, sites):
    """Disposizione linearizzata: un sito per posizione, nell'ordine
    topologico usato per costruire la flat chain."""
    n = len(sites)
    xs = np.arange(n)

    if n > 1:
        ax.plot(xs, np.zeros(n), "-", color="#bbbbbb", lw=1.5, zorder=1)

    for idx, s in enumerate(sites):
        color = COLOR_MAP.get(s.feature_type, "#607D8B")
        ax.scatter(idx, 0, s=500, color=color, zorder=3,
                   edgecolors="white", linewidths=1.8)
        ax.text(idx, 0, TYPE_SHORT.get(s.feature_type, "?"),
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)
        ax.text(idx, 0.32, str(idx), ha="center", va="center", fontsize=8)

    ax.set_xlim(-1, max(n, 1))
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xlabel("Posizione nella catena 1D (ordine topologico)")
    ax.spines[["top", "right", "left"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plotta un amminoacido in 3D con i siti farmacoforici "
                     "superficiali e la loro disposizione nella catena 1D."
    )
    parser.add_argument("--pdb", required=True, help="File PDB (proteina o tasca)")
    parser.add_argument("--chain", required=True, help="Chain ID del residuo (es. A)")
    parser.add_argument("--resseq", type=int, required=True, help="Numero di sequenza del residuo (es. 30)")
    parser.add_argument("--surface-filter", action="store_true", default=False,
                         help="Mostra solo i siti esposti al solvente (SASA)")
    parser.add_argument("--sasa-threshold", type=float, default=1.0,
                         help="Soglia SASA in Å² (default 1.0)")
    parser.add_argument("--save-plot", default=None, help="Salva il plot come PNG invece di mostrarlo")
    args = parser.parse_args()

    residues = load_residues_from_pdb(args.pdb, skip_water=True)
    rec = find_residue(residues, args.chain, args.resseq)
    if rec is None:
        raise SystemExit(
            f"Residuo {args.chain}{args.resseq} non trovato in {args.pdb}. "
            f"Residui disponibili: {', '.join(r.label for r in residues[:10])}..."
        )
    if rec.mol is None:
        raise SystemExit(f"{rec.label}: molecola RDKit non disponibile")

    sasa_map = None
    if args.surface_filter:
        sasa_map = compute_atom_sasa(args.pdb)

    sites = get_ordered_sites(
        rec,
        surface_filter=args.surface_filter,
        sasa_map=sasa_map,
        sasa_threshold=args.sasa_threshold,
    )

    if not sites:
        raise SystemExit(
            f"{rec.label}: nessun sito trovato"
            + (" con --surface-filter attivo (prova una soglia più bassa)." if args.surface_filter else ".")
        )

    label = "siti superficiali" if args.surface_filter else "siti totali"
    print(f"\n{rec.label} — {len(sites)} {label}")
    for idx, s in enumerate(sites):
        print(f"  [{idx:2d}] {s.feature_type:15s}  "
              f"coords=({s.coords[0]:6.2f}, {s.coords[1]:6.2f}, {s.coords[2]:6.2f})  "
              f"intensity={s.intensity:.3f}")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(13, 6))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    plot_molecule_3d(ax3d, rec.mol)
    plot_sites_3d(ax3d, sites)
    ax3d.set_title(f"{rec.label} — struttura 3D + {label}")
    ax3d.set_xlabel("x (Å)")
    ax3d.set_ylabel("y (Å)")
    ax3d.set_zlabel("z (Å)")

    ax1d = fig.add_subplot(1, 2, 2)
    plot_1d_chain(ax1d, sites)
    ax1d.set_title(f"{rec.label} — disposizione nella catena 1D")

    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="#dddddd")

    fig.tight_layout()

    if args.save_plot:
        os.makedirs(os.path.dirname(args.save_plot) or ".", exist_ok=True)
        fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot salvato in: {args.save_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
