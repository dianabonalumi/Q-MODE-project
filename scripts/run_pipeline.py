"""
Script principale: PDB tasca -> sequenza flat di siti di interazione
====================================================================
I residui vengono ordinati seguendo la sequenza di catena (chain_id, res_seq);
dentro ciascun residuo i siti sono ordinati per topologia covalente.
L'output è una sequenza flat: [(tipo, intensity, coords), ...] per tutti i siti.

Uso:
    python scripts/run_pipeline.py --pdb data/raw/1a08_pocket.pdb --plot
    python scripts/run_pipeline.py --pdb data/raw/1a08_pocket.pdb --output data/processed/
"""

import argparse
import os
import sys
import json
import warnings

import numpy as np

# Assicurati che i moduli custom siano importabili
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmode.surface_filter import compute_atom_sasa, filter_surface_features_by_coords
from qmode.pdb_reader import (
    load_residues_from_pdb,
    compute_pocket_centroid,
    AMINO_SMILES,
)
from qmode.feature_extraction import extract_features
from qmode.site_selection import topological_order
from qmode.abraham_hbond import assign_abraham_hb_intensities


# ─────────────────────────────────────────────────────────────────────────────

def process_residue(rec, surface_filter=True, sasa_threshold=1.0, sasa_map=None):
    """
    Esegue la pipeline locale su un singolo residuo.
    Restituisce la lista di AtomFeature (siti, coordinate 3D reali,
    ordinati per topologia covalente), oppure None se il residuo non può
    essere processato.
    """
    if rec.mol is None:
        return None

    try:
        # Step 2: feature extraction (usa coord 3D reali dal PDB)
        features = extract_features(rec.mol, embed_3d=True)
        if not features:
            warnings.warn(f"  {rec.label}: nessuna feature estratta, skip")
            return None

        # Intensità HB intrinseca (scale di Abraham)
        assign_abraham_hb_intensities(features, res_name=rec.res_name, mol=rec.mol, atom_records=rec.atoms)

        # ── Filtro superficie (opzionale) ────────────────────────────────────
        if surface_filter and sasa_map is not None:
            features = filter_surface_features_by_coords(
                features=features,
                sasa_map=sasa_map,
                chain_id=rec.chain_id,
                res_seq=rec.res_seq,
                all_atom_coords=rec.atoms,
                sasa_threshold=sasa_threshold,
            )
            if not features:
                return None

        # Step 3: tutti i siti, ordinati per topologia covalente (no K-Means)
        sites = topological_order(features, mol=rec.mol)

        return sites

    except Exception as e:
        warnings.warn(f"  {rec.label}: errore — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdb_path,
    output_dir=None,
    plot=False,
    save_plot=None,
    args_ligand_size=3,
    surface_filter=True,
    sasa_threshold=1.0
):
    print(f"\n{'='*60}")
    print(f"  Pipeline Mapping: {os.path.basename(pdb_path)}")
    print(f"{'='*60}")

    # ── Step 1: lettura PDB ───────────────────────────────────────────────
    print("\n[1/4] Lettura residui dal PDB...")
    residues = load_residues_from_pdb(pdb_path, skip_water=True)
    print(f"      {len(residues)} residui trovati")

    # ── Step 2: ordinamento per catena ────────────────────────────────────
    print("\n[2/4] Ordinamento per sequenza di catena (chain_id, res_seq)...")
    centroid = compute_pocket_centroid(residues)
    print(f"      Centroide tasca: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) Å")
    print(f"      Ordine: " + " -> ".join(r.label for r in residues[:5]) + " -> ...")

    # ── Calcolo SASA (se richiesto) ───────────────────────────────────────
    sasa_map = None
    if surface_filter:
        print("\n[!] Calcolo mappa SASA per il filtro di superficie...")
        sasa_map = compute_atom_sasa(pdb_path)

    # ── Step 3: pipeline locale per ogni residuo ──────────────────────────
    print(f"\n[3/4] Pipeline locale per ogni residuo (reticolo indipendente)...")

    flat_chain = []          # sequenza flat finale: lista di dict
    per_residue = []         # per visualizzazione e debug

    for rec in residues:
        sites = process_residue(
            rec,
            surface_filter=surface_filter,
            sasa_threshold=sasa_threshold,
            sasa_map=sasa_map
        )
        if sites is None:
            continue

        print(f"      {rec.label:20s} -> {len(sites)} siti: "
              + "  ".join(s.feature_type[:3] for s in sites))

        for s in sites:
            flat_chain.append({
                "residue":   rec.label,
                "res_name":  rec.res_name,
                "res_seq":   rec.res_seq,
                "chain_id":  rec.chain_id,
                "coords":    [round(float(c), 3) for c in s.coords],
                "type":      s.feature_type,
                "intensity": round(s.intensity, 3),
            })

        per_residue.append({"record": rec, "sites": sites})

    # ── Step 4: output ────────────────────────────────────────────────────
    print(f"\n[4/4] Sequenza flat finale")
    print(f"{'-'*60}")
    print(f"  Residui processati : {len(per_residue)}")
    print(f"  Siti totali        : {len(flat_chain)}")
    print(f"\n  idx  {'Residuo':20s}  Tipo                 Intensity")
    print(f"  {'-'*4}  {'-'*20}  {'-'*15}  {'-'*9}")
    for idx, s in enumerate(flat_chain):
        print(f"  {idx+1:4d}  {s['residue']:20s}  {s['type']:15s}  {s['intensity']:.3f}")

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
            "ordering": "chain_sequence",
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
    from qmode.qubit_chain import build_qubit_chain, print_qubit_chain
    from qmode.qubit_chain import get_h_hb_intensities
    
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

    print(f"\n{'-'*60}")
    print(f"  Quantum Encoding  (Sliding window ligand size = {args_ligand_size})")
    print(f"  h_range: [{h_min:.2f}, {h_max:.2f}], hb_range: [{hb_min:.2f}, {hb_max:.2f}]")
    print(f"{'-'*60}")

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
    Due figure separate, ciascuna salvata/mostrata indipendentemente:
      Fig 1 — Sequenza flat: heatmap tipo × posizione
      Fig 2 — Barchart composizione farmacofori per residuo
    """
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
    FEAT_TYPES = list(COLOR_MAP.keys())
    fname = os.path.splitext(os.path.basename(pdb_path))[0]
    n_res = len(per_residue)

    # ═════════════════════════════════════════════════════════════════════
    # FIGURA 1 — Heatmap sequenza flat
    # ═════════════════════════════════════════════════════════════════════
    n_sites = len(flat_chain)
    n_types = len(FEAT_TYPES)
    matrix  = np.zeros((n_types, n_sites))

    for col_idx, s in enumerate(flat_chain):
        row_idx = FEAT_TYPES.index(s["type"]) if s["type"] in FEAT_TYPES else 0
        matrix[row_idx, col_idx] = 1.0

    fig2, ax2 = plt.subplots(figsize=(max(14, n_sites * 0.22), 4))

    for r, ft in enumerate(FEAT_TYPES):
        for c in range(n_sites):
            if matrix[r, c] > 0:
                ax2.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color=COLOR_MAP[ft], alpha=0.85, zorder=2,
                ))

    current_res = None
    for c, s in enumerate(flat_chain):
        if s["residue"] != current_res:
            if current_res is not None:
                ax2.axvline(c - 0.5, color="#888888", lw=0.8, ls="--", zorder=3)
            current_res = s["residue"]

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
        print(f"  Plot 1 (sequenza flat) salvato in: {p}")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURA 2 — Barchart impilato
    # ═════════════════════════════════════════════════════════════════════
    from collections import Counter
    res_order = [e["record"].label for e in per_residue]
    counts    = {res: Counter() for res in res_order}
    for s in flat_chain:
        counts[s["residue"]][s["type"]] += 1

    fig3, ax3 = plt.subplots(figsize=(max(14, n_res * 0.55), 5))

    bottoms = np.zeros(n_res)
    for ft in FEAT_TYPES:
        vals = np.array([counts[r][ft] for r in res_order], dtype=float)
        ax3.bar(range(n_res), vals, bottom=bottoms,
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
        f"(ordinati per sequenza di catena)",
        fontsize=12, fontweight="bold",
    )
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()

    if save_path:
        p = save_path.replace(".png", "_composition.png")
        fig3.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Plot 2 (composizione) salvato in: {p}")


    if not save_path:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mappa la tasca PDB in una sequenza flat di siti sul reticolo"
    )
    parser.add_argument(
        "--no-surface-filter",
        dest="surface_filter",
        action="store_false",
        default=True,
        help="Disattiva il filtro di accessibilità al solvente (SASA), attivo "
             "di default. Con il filtro attivo si mantengono solo le feature "
             "esposte verso la tasca.")

    parser.add_argument(
        "--sasa-threshold",
        type=float,
        default=1.0,
        help="Soglia SASA in Å² per considerare un atomo esposto (default: 1.0).")

    parser.add_argument("--pdb", required=True,
                        help="File PDB della tasca (es. data/raw/1a08_pocket.pdb)")
    parser.add_argument("--output", default=None,
                        help="Directory dove salvare JSON e CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Mostra visualizzazione interattiva")
    parser.add_argument("--save-plot", default=None,
                        help="Salva il plot come PNG")
    parser.add_argument("--ligand-size", type=int, default=3,
                        help="Dimensione del ligando in siti (default 3)")
    
    args = parser.parse_args()

    # Chiamata pulita alla funzione principale, passando gli argomenti esplicitamente
    run_pipeline(
        pdb_path=args.pdb,
        output_dir=args.output,
        plot=args.plot,
        save_plot=args.save_plot,
        args_ligand_size=args.ligand_size,
        surface_filter=args.surface_filter,
        sasa_threshold=args.sasa_threshold
    )
