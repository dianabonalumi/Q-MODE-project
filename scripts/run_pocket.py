"""
Script principale: PDB tasca → catena di siti di interazione
=============================================================

Uso:
    python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb
    python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --plot
    python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --output data/processed/

Cosa fa:
  1. Legge il PDB della tasca → estrae i residui
  2. Per ogni residuo esegue la pipeline (feature → K siti → reticolo 2D)
  3. Aggrega tutti i siti in una catena unica della tasca
  4. Stampa e visualizza la catena
"""

import argparse
import os
import sys
import json
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amino_lattice import AminoLatticePipeline
from amino_lattice.pdb_reader import load_residues_from_pdb, AMINO_SMILES
from amino_lattice.feature_extraction import extract_features
from amino_lattice.site_selection import choose_k, select_representative_sites
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice
from amino_lattice.labeling import label_sites, LabeledSite


def run_pocket(
    pdb_path: str,
    k_strategy: str = "active_features",
    projection: str = "pca",
    label_mode: str = "one_hot",
    max_k: int = 6,
    output_dir: str = None,
    plot: bool = False,
    save_plot: str = None,
):
    print(f"\n{'='*60}")
    print(f"  Pocket Mapping: {os.path.basename(pdb_path)}")
    print(f"{'='*60}")

    # ── Step 1: Lettura PDB ───────────────────────────────────────────────
    print("\n[1/4] Lettura residui dal PDB...")
    residues = load_residues_from_pdb(pdb_path, skip_water=True)
    print(f"      Trovati {len(residues)} residui: "
          + ", ".join(f"{r.chain_id}{r.res_seq}({r.res_name})" for r in residues))

    # ── Step 2-6: Pipeline per ogni residuo ──────────────────────────────
    print(f"\n[2/4] Estrazione feature e mapping per ogni residuo...")
    all_sites = []       # AtomFeature aggregati di tutti i residui
    residue_labels = []  # per tracciare a quale residuo appartiene ogni sito

    pipeline = AminoLatticePipeline(
        k_strategy=k_strategy,
        projection_method=projection,
        label_mode=label_mode,
        max_k=max_k,
        embed_3d=False,   # usiamo le coordinate 3D già dal PDB
    )

    residue_results = []
    for rec in residues:
        if rec.mol is None:
            warnings.warn(f"  Skipped {rec.label}: mol non costruita")
            continue

        smiles = AMINO_SMILES.get(rec.res_name)
        if not smiles:
            continue

        try:
            # Estrai feature dalla mol con coordinate PDB reali
            features = extract_features(rec.mol, embed_3d=False)
            if not features:
                warnings.warn(f"  {rec.label}: nessuna feature, skip")
                continue

            k = choose_k(features, mol=rec.mol, strategy=k_strategy,
                         max_k=max_k, min_k=1)
            sites = select_representative_sites(features, k)

            residue_results.append({
                "record": rec,
                "features": features,
                "sites": sites,
            })

            for s in sites:
                all_sites.append(s)
                residue_labels.append(rec.label)

            print(f"      {rec.label:20s} → {k} siti  "
                  f"({', '.join(set(s.feature_type[:3] for s in sites))})")

        except Exception as e:
            warnings.warn(f"  {rec.label}: errore — {e}")
            continue

    if not all_sites:
        print("ERRORE: nessun sito estratto. Controlla il file PDB.")
        return

    # ── Fitting globale sul reticolo 2D ──────────────────────────────────
    print(f"\n[3/4] Fitting globale sul reticolo 2D ({projection.upper()})...")
    total_k = len(all_sites)

    coords_2d = fit_to_lattice_2d(all_sites, method=projection, lattice_spacing=2.0)
    lattice_nodes = snap_to_lattice(coords_2d, strategy="hungarian")
    labeled = label_sites(all_sites, lattice_nodes, mode=label_mode)

    # ── Output ────────────────────────────────────────────────────────────
    print(f"\n[4/4] Risultati")
    print(f"{'─'*60}")
    print(f"  Residui processati : {len(residue_results)}")
    print(f"  Siti totali (K)    : {total_k}")
    print(f"\n  Catena di siti della tasca:")
    print(f"  {'Sito':5s}  {'Residuo':20s}  {'(i, j)':10s}  Tipo")
    print(f"  {'─'*5}  {'─'*20}  {'─'*10}  {'─'*15}")
    for idx, (ls, res_label) in enumerate(zip(labeled, residue_labels)):
        print(f"  {idx+1:5d}  {res_label:20s}  ({ls.i:3d},{ls.j:3d})    {ls.feature_type}")

    chain = [(ls.i, ls.j, ls.feature_type) for ls in labeled]

    # ── Salva output ──────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # JSON
        output = {
            "pdb": os.path.basename(pdb_path),
            "n_residues": len(residue_results),
            "n_sites": total_k,
            "chain": [
                {"site_idx": i, "residue": res_label, "i": ls.i, "j": ls.j,
                 "type": ls.feature_type, "intensity": round(ls.intensity, 3)}
                for i, (ls, res_label) in enumerate(zip(labeled, residue_labels))
            ]
        }
        json_path = os.path.join(output_dir, "pocket_chain.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  JSON salvato in: {json_path}")

        # CSV
        import pandas as pd
        df = pd.DataFrame([
            {"site_idx": i, "residue": res_label, "i": ls.i, "j": ls.j,
             "type": ls.feature_type, "intensity": round(ls.intensity, 3)}
            for i, (ls, res_label) in enumerate(zip(labeled, residue_labels))
        ])
        csv_path = os.path.join(output_dir, "pocket_chain.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV salvato in:  {csv_path}")

    # ── Visualizzazione ───────────────────────────────────────────────────
    if plot or save_plot:
        _plot_pocket(labeled, residue_labels, pdb_path, save_plot)

    return labeled, residue_labels


def _plot_pocket(labeled, residue_labels, pdb_path, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLOR_MAP = {
        "HBondDonor":    "#2196F3",
        "HBondAcceptor": "#4CAF50",
        "Hydrophobe":    "#FF9800",
        "Aromatic":      "#9C27B0",
        "PosIonizable":  "#F44336",
        "NegIonizable":  "#00BCD4",
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    # Griglia di sfondo
    all_i = [ls.i for ls in labeled]
    all_j = [ls.j for ls in labeled]
    margin = 2
    for gi in range(min(all_i) - margin, max(all_i) + margin + 1):
        for gj in range(min(all_j) - margin, max(all_j) + margin + 1):
            ax.plot(gj, gi, ".", color="#eeeeee", markersize=3, zorder=0)

    # Connessioni (catena)
    for k in range(len(labeled) - 1):
        i1, j1 = labeled[k].i, labeled[k].j
        i2, j2 = labeled[k+1].i, labeled[k+1].j
        ax.plot([j1, j2], [i1, i2], "-", color="#cccccc", lw=0.8, zorder=1, alpha=0.5)

    # Raggruppa per residuo per colorare il bordo
    unique_residues = list(dict.fromkeys(residue_labels))
    res_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_residues)))
    res_color_map = {r: res_colors[i] for i, r in enumerate(unique_residues)}

    # Siti
    for ls, res_label in zip(labeled, residue_labels):
        fcolor = COLOR_MAP.get(ls.feature_type, "#607D8B")
        ecolor = res_color_map[res_label]
        ax.scatter(ls.j, ls.i, s=250, color=fcolor, zorder=3,
                   edgecolors=ecolor, linewidths=3)
        ax.annotate(
            ls.feature_type[:3],
            (ls.j, ls.i), fontsize=6, ha="center", va="center",
            color="white", fontweight="bold", zorder=4,
        )

    # Legenda tipi farmacofori
    feat_patches = [mpatches.Patch(color=c, label=ft)
                    for ft, c in COLOR_MAP.items()]
    legend1 = ax.legend(handles=feat_patches, title="Feature type",
                        loc="upper left", fontsize=8)
    ax.add_artist(legend1)

    ax.set_title(f"Tasca di interazione — {os.path.basename(pdb_path)}\n"
                 f"{len(unique_residues)} residui, {len(labeled)} siti sul reticolo 2D",
                 fontsize=12)
    ax.set_xlabel("j (colonna reticolo)")
    ax.set_ylabel("i (riga reticolo)")
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot salvato in: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mappa la tasca di una proteina (PDB) sul reticolo 2D"
    )
    parser.add_argument("--pdb", type=str, required=True,
                        help="Percorso al file PDB della tasca (es. data/raw/1a08_pocket.pdb)")
    parser.add_argument("--k-strategy", type=str, default="active_features",
                        choices=["active_features", "heavy_atoms", "fixed", "groups"])
    parser.add_argument("--projection", type=str, default="pca", choices=["pca", "mds"])
    parser.add_argument("--label-mode", type=str, default="one_hot",
                        choices=["one_hot", "index", "embedding"])
    parser.add_argument("--max-k", type=int, default=6,
                        help="Max siti per residuo (default 6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory dove salvare JSON e CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Mostra la visualizzazione interattiva")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Salva il plot in un file PNG (es. output/pocket.png)")
    args = parser.parse_args()

    run_pocket(
        pdb_path=args.pdb,
        k_strategy=args.k_strategy,
        projection=args.projection,
        label_mode=args.label_mode,
        max_k=args.max_k,
        output_dir=args.output,
        plot=args.plot,
        save_plot=args.save_plot,
    )
