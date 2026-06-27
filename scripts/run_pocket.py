"""
Script principale: PDB tasca → sequenza flat di siti di interazione
====================================================================
Ogni residuo ha il suo reticolo locale indipendente.
I residui vengono ordinati per distanza dal centroide della tasca.
L'output è una sequenza flat: [(i,j,tipo), ...] per tutti i siti.

Uso:
    python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --plot
    python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --output data/processed/
"""

import argparse
import os
import sys
import json
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amino_lattice.pdb_reader import (
    load_residues_from_pdb,
    compute_pocket_centroid,
    sort_residues_by_distance,
    AMINO_SMILES,
)
from amino_lattice.feature_extraction import extract_features
from amino_lattice.site_selection import choose_k, select_representative_sites
from amino_lattice.lattice_fitting import fit_to_lattice_2d
from amino_lattice.snapping import snap_to_lattice
from amino_lattice.labeling import label_sites, LabeledSite
from amino_lattice.hbond_geometry import (
    compute_pocket_hbond_strengths,
    assign_feature_hbond_intensities,
)


# ─────────────────────────────────────────────────────────────────────────────

def process_residue(rec, k_strategy, max_k, hb_atoms=None):
    """
    Esegue la pipeline locale su un singolo residuo.
    Restituisce la lista di LabeledSite con coordinate (i,j) locali,
    oppure None se il residuo non può essere processato.

    hb_atoms : lista di HBAtom del residuo (con forze geometriche pre-calcolate
    a livello di tasca). Se fornita, sovrascrive le intensità delle feature HBond
    con i valori geometrici reali (distanza + angolo D–H···A).
    """
    if rec.mol is None:
        return None

    try:
        # Step 2: feature extraction (usa coord 3D reali dal PDB)
        features = extract_features(rec.mol, embed_3d=False)
        if not features:
            warnings.warn(f"  {rec.label}: nessuna feature estratta, skip")
            return None

        # Intensità HB geometrica (sostituisce il placeholder)
        if hb_atoms is not None:
            assign_feature_hbond_intensities(features, hb_atoms)

        # Step 3: scelta K e selezione siti
        k = choose_k(features, mol=rec.mol, strategy=k_strategy,
                     max_k=max_k, min_k=1)
        sites = select_representative_sites(features, k)

        # Step 4: fitting geometrico LOCALE (PCA sul solo residuo)
        coords_2d = fit_to_lattice_2d(sites, method="pca", lattice_spacing=1.5)

        # Step 5: snapping al reticolo locale
        lattice_nodes = snap_to_lattice(coords_2d, strategy="hungarian")

        # Step 6: labeling
        labeled = label_sites(sites, lattice_nodes, mode="one_hot")

        return labeled

    except Exception as e:
        warnings.warn(f"  {rec.label}: errore — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────

def run_pocket(
    pdb_path,
    k_strategy="active_features",
    max_k=6,
    output_dir=None,
    plot=False,
    save_plot=None,
    args_ligand_size=3,
):
    print(f"\n{'='*60}")
    print(f"  Pocket Mapping: {os.path.basename(pdb_path)}")
    print(f"{'='*60}")

    # ── Step 1: lettura PDB ───────────────────────────────────────────────
    print("\n[1/4] Lettura residui dal PDB...")
    residues = load_residues_from_pdb(pdb_path, skip_water=True)
    print(f"      {len(residues)} residui trovati")

    # ── Step 2: ordinamento spaziale ──────────────────────────────────────
    print("\n[2/4] Ordinamento per distanza dal centroide della tasca...")
    centroid = compute_pocket_centroid(residues)
    residues = sort_residues_by_distance(residues, centroid)
    print(f"      Centroide tasca: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) Å")
    print(f"      Ordine: " + " → ".join(r.label for r in residues[:5]) + " → ...")

    # ── Intensità HB geometrica a livello di tasca ────────────────────────
    # Legami idrogeno reali TRA residui diversi: distanza + angolo D–H···A.
    hb_strengths = compute_pocket_hbond_strengths(residues)

    # ── Step 3: pipeline locale per ogni residuo ──────────────────────────
    print(f"\n[3/4] Pipeline locale per ogni residuo (reticolo indipendente)...")

    flat_chain = []          # sequenza flat finale: lista di dict
    per_residue = []         # per visualizzazione e debug

    for rec in residues:
        labeled = process_residue(rec, k_strategy, max_k,
                                  hb_atoms=hb_strengths.get(rec.label))
        if labeled is None:
            continue

        print(f"      {rec.label:20s} → {len(labeled)} siti: "
              + "  ".join(f"({ls.i:2d},{ls.j:2d}) {ls.feature_type[:3]}"
                          for ls in labeled))

        for ls in labeled:
            flat_chain.append({
                "residue":   rec.label,
                "res_name":  rec.res_name,
                "res_seq":   rec.res_seq,
                "chain_id":  rec.chain_id,
                "i":         ls.i,
                "j":         ls.j,
                "type":      ls.feature_type,
                "intensity": round(ls.intensity, 3),
            })

        per_residue.append({"record": rec, "labeled": labeled})

    # ── Step 4: output ────────────────────────────────────────────────────
    print(f"\n[4/4] Sequenza flat finale")
    print(f"{'─'*60}")
    print(f"  Residui processati : {len(per_residue)}")
    print(f"  Siti totali        : {len(flat_chain)}")
    print(f"\n  idx  {'Residuo':20s}  (i, j)     Tipo")
    print(f"  {'─'*4}  {'─'*20}  {'─'*8}  {'─'*15}")
    for idx, s in enumerate(flat_chain):
        print(f"  {idx+1:4d}  {s['residue']:20s}  ({s['i']:3d},{s['j']:3d})  {s['type']}")

    # ── Salvataggio ───────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import pandas as pd

        # JSON
        result = {
            "pdb": os.path.basename(pdb_path),
            "pocket_centroid": centroid.tolist(),
            "n_residues": len(per_residue),
            "n_sites_total": len(flat_chain),
            "ordering": "spatial_distance_from_centroid",
            "flat_chain": flat_chain,
        }
        json_path = os.path.join(output_dir, "pocket_chain.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  JSON salvato in: {json_path}")

        # CSV
        df = pd.DataFrame(flat_chain)
        csv_path = os.path.join(output_dir, "pocket_chain.csv")
        df.to_csv(csv_path, index=True, index_label="site_idx")
        print(f"  CSV salvato in:  {csv_path}")

    # ── Visualizzazione ───────────────────────────────────────────────────
    if plot or save_plot:
        _plot_sequence(per_residue, flat_chain, pdb_path, save_plot)

    # ── Catena di qubit (Quantum Encoding) ────────────────────────────────
    from amino_lattice.qubit_chain import build_qubit_chain, print_qubit_chain

    # 1. Range globale di idrofobicità e H-Bond calcolato SOLO sui valori
    #    attivi (>0) di ciascun canale. Includere gli zeri strutturali (siti
    #    dell'altro canale) abbasserebbe il minimo a 0 e spingerebbe la soglia
    #    a metà del massimo: quasi nessun sito la supererebbe → catena di zeri.
    from amino_lattice.qubit_chain import get_h_hb_intensities
    h_pos, hb_pos = [], []
    for s in flat_chain:
        h, hb = get_h_hb_intensities(s)
        if h > 0:
            h_pos.append(h)
        if hb > 0:
            hb_pos.append(hb)

    h_min = min(h_pos) if h_pos else 0.0
    h_max = max(h_pos) if h_pos else 1.0
    hb_min = min(hb_pos) if hb_pos else 0.0
    hb_max = max(hb_pos) if hb_pos else 1.0

    print(f"\n{'─'*60}")
    print(f"  Quantum Encoding  (Sliding window ligand size = {args_ligand_size})")
    print(f"  h_range: [{h_min:.2f}, {h_max:.2f}], hb_range: [{hb_min:.2f}, {hb_max:.2f}]")
    print(f"{'─'*60}")

    segments = build_qubit_chain(
        flat_chain,
        ligand_size=args_ligand_size,
        h_min=h_min, h_max=h_max, hb_min=hb_min, hb_max=hb_max
    )
    
    print_qubit_chain(segments)

    if output_dir:
        # Salva qubit chain come JSON
        qubit_data = {
            "ligand_size": args_ligand_size,
            "h_range": [h_min, h_max],
            "hb_range": [hb_min, hb_max],
            "n_segments": len(segments),
            "segments": [
                {
                    "segment_idx": s.segment_idx,
                    "first_encoding_state": s.first_encoding_state,
                    "second_encoding_amplitudes": s.second_encoding_amplitudes,
                    "residues": list(set([site["residue"] for site in s.sites])),
                }
                for s in segments
            ],
        }
        qjson_path = os.path.join(output_dir, "quantum_chain.json")
        with open(qjson_path, "w") as f:
            json.dump(qubit_data, f, indent=2)
        print(f"\n  Quantum JSON salvato in: {qjson_path}")

    return flat_chain, per_residue, segments


# ─────────────────────────────────────────────────────────────────────────────
# Visualizzazione
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sequence(per_residue, flat_chain, pdb_path, save_path=None):
    """
    Tre figure separate, ciascuna salvata/mostrata indipendentemente:
      Fig 1 — Griglia di reticoli locali (6 per riga), uno per residuo
      Fig 2 — Sequenza flat: heatmap tipo × posizione
      Fig 3 — Barchart composizione farmacofori per residuo
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from collections import Counter

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
    FEAT_TYPES = list(COLOR_MAP.keys())
    fname = os.path.splitext(os.path.basename(pdb_path))[0]

    # ═════════════════════════════════════════════════════════════════════
    # FIGURA 1 — Griglia di reticoli locali (N_COLS per riga)
    # ═════════════════════════════════════════════════════════════════════
    N_COLS = 6
    n_res  = len(per_residue)
    n_rows = (n_res + N_COLS - 1) // N_COLS

    fig1, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 3.2, n_rows * 3.2),
        squeeze=False,
    )
    fig1.suptitle(
        f"Reticoli locali — {fname}\n"
        f"(ordinati per distanza dal centroide, sinistra-alto = più vicino)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for idx, entry in enumerate(per_residue):
        row, col = divmod(idx, N_COLS)
        ax = axes[row][col]
        rec     = entry["record"]
        labeled = entry["labeled"]

        all_i = [ls.i for ls in labeled]
        all_j = [ls.j for ls in labeled]
        margin = 1
        i_min, i_max = min(all_i) - margin, max(all_i) + margin
        j_min, j_max = min(all_j) - margin, max(all_j) + margin

        # Griglia di sfondo
        for gi in range(i_min, i_max + 1):
            for gj in range(j_min, j_max + 1):
                ax.plot(gj, gi, ".", color="#e8e8e8", markersize=5, zorder=0)

        # Connessioni intra-residuo
        for k in range(len(labeled) - 1):
            ax.plot(
                [labeled[k].j, labeled[k+1].j],
                [labeled[k].i, labeled[k+1].i],
                "-", color="#cccccc", lw=1.5, zorder=1,
            )

        # Siti
        for ls in labeled:
            color = COLOR_MAP.get(ls.feature_type, "#607D8B")
            ax.scatter(ls.j, ls.i, s=320, color=color,
                       zorder=3, edgecolors="white", linewidths=1.8)
            ax.text(ls.j, ls.i, TYPE_SHORT.get(ls.feature_type, "?"),
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=4)

        ax.set_title(f"{rec.label}", fontsize=8, fontweight="bold", pad=4)
        ax.set_xlim(j_min - 0.8, j_max + 0.8)
        ax.set_ylim(i_min - 0.8, i_max + 0.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top","right","left","bottom"]].set_visible(False)

    # Nascondi assi vuoti nell'ultima riga
    for idx in range(n_res, n_rows * N_COLS):
        row, col = divmod(idx, N_COLS)
        axes[row][col].set_visible(False)

    # Legenda comune in basso
    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()]
    fig1.legend(handles=patches, loc="lower center", ncol=6,
                fontsize=9, bbox_to_anchor=(0.5, -0.03),
                frameon=True, edgecolor="#dddddd")

    fig1.tight_layout()
    if save_path:
        p = save_path.replace(".png", "_lattices.png")
        fig1.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Plot 1 (reticoli) salvato in: {p}")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURA 2 — Heatmap sequenza flat: righe=tipi, colonne=indice sito
    # ═════════════════════════════════════════════════════════════════════
    n_sites = len(flat_chain)
    n_types = len(FEAT_TYPES)
    matrix  = np.zeros((n_types, n_sites))

    for col_idx, s in enumerate(flat_chain):
        row_idx = FEAT_TYPES.index(s["type"]) if s["type"] in FEAT_TYPES else 0
        matrix[row_idx, col_idx] = 1.0

    fig2, ax2 = plt.subplots(figsize=(max(14, n_sites * 0.22), 4))

    # Disegna ogni cella colorata manualmente per usare i colori farmacofori
    for r, ft in enumerate(FEAT_TYPES):
        for c in range(n_sites):
            if matrix[r, c] > 0:
                ax2.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color=COLOR_MAP[ft], alpha=0.85, zorder=2,
                ))

    # Linee verticali di separazione tra residui
    current_res = None
    for c, s in enumerate(flat_chain):
        if s["residue"] != current_res:
            if current_res is not None:
                ax2.axvline(c - 0.5, color="#888888", lw=0.8, ls="--", zorder=3)
            current_res = s["residue"]

    # Etichette residui (ogni primo sito del residuo)
    seen = set()
    for c, s in enumerate(flat_chain):
        if s["residue"] not in seen:
            ax2.text(c, n_types - 0.1, s["residue"].split("_")[0],
                     ha="left", va="bottom", fontsize=6,
                     rotation=60, color="#333333")
            seen.add(s["residue"])

    ax2.set_xlim(-0.5, n_sites - 0.5)
    ax2.set_ylim(-0.5, n_types + 0.5)
    ax2.set_yticks(range(n_types))
    ax2.set_yticklabels(FEAT_TYPES, fontsize=9)
    ax2.set_xlabel("Indice sito nella sequenza flat", fontsize=10)
    ax2.set_title(
        f"Sequenza flat dei siti — {fname}  ({n_sites} siti totali)",
        fontsize=12, fontweight="bold",
    )
    ax2.grid(axis="x", alpha=0.15)
    fig2.tight_layout()

    if save_path:
        p = save_path.replace(".png", "_sequence.png")
        fig2.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Plot 2 (sequenza flat) salvato in: {p}")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURA 3 — Barchart impilato: composizione per residuo
    # ═════════════════════════════════════════════════════════════════════
    # Conta tipi per residuo
    res_order = [e["record"].label for e in per_residue]
    counts    = {res: Counter() for res in res_order}
    for s in flat_chain:
        counts[s["residue"]][s["type"]] += 1

    fig3, ax3 = plt.subplots(figsize=(max(14, n_res * 0.55), 5))

    bottoms = np.zeros(n_res)
    for ft in FEAT_TYPES:
        vals = np.array([counts[r][ft] for r in res_order], dtype=float)
        bars = ax3.bar(range(n_res), vals, bottom=bottoms,
                       color=COLOR_MAP[ft], label=ft, width=0.75)
        bottoms += vals

    ax3.set_xticks(range(n_res))
    ax3.set_xticklabels(
        [r.split("_")[0] for r in res_order],
        rotation=60, ha="right", fontsize=8,
    )
    ax3.set_ylabel("Numero di siti", fontsize=10)
    ax3.set_title(
        f"Composizione farmacofori per residuo — {fname}\n"
        f"(ordinati per distanza dal centroide della tasca)",
        fontsize=12, fontweight="bold",
    )
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()

    if save_path:
        p = save_path.replace(".png", "_composition.png")
        fig3.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Plot 3 (composizione) salvato in: {p}")


    if not save_path:
        plt.show()



# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mappa la tasca PDB in una sequenza flat di siti sul reticolo"
    )
    parser.add_argument("--pdb", required=True,
                        help="File PDB della tasca (es. data/raw/1a08_pocket.pdb)")
    parser.add_argument("--k-strategy", default="active_features",
                        choices=["active_features", "heavy_atoms", "fixed", "groups"])
    parser.add_argument("--max-k", type=int, default=6,
                        help="Max siti per residuo (default 6)")
    parser.add_argument("--output", default=None,
                        help="Directory dove salvare JSON e CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Mostra visualizzazione interattiva")
    parser.add_argument("--save-plot", default=None,
                        help="Salva il plot come PNG")
    parser.add_argument("--ligand-size", type=int, default=3,
                        help="Dimensione del ligando in siti (default 3)")
    args = parser.parse_args()

    run_pocket(
        pdb_path=args.pdb,
        k_strategy=args.k_strategy,
        max_k=args.max_k,
        output_dir=args.output,
        plot=args.plot,
        save_plot=args.save_plot,
        args_ligand_size=args.ligand_size,
    )
