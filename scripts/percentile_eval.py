"""
Ricalcola le statistiche spaziali del benchmark: soffitto/caso per pocket,
percentili dei candidati di Grover e sweep sulla dimensione della finestra.

Riproduce le tabelle "Reachability of the 5 A criterion" e "Effect of the
sliding-window size" del report. Le distanze assolute del top-1 sono quelle
di scripts/fast_eval.py: i candidati vengono costruiti con lo stesso
tile_offset usato da run_pipeline.py.

Uso:
    python scripts/percentile_eval.py 3 4 5

Il percentile di una finestra e' la frazione di TUTTE le finestre a passo 1
della stessa pocket che sono piu' vicine di essa al centroide del ligando:
0% e' la finestra migliore che la rappresentazione rende disponibile, 50% e'
il caso. Il baseline casuale e' E[min su n estrazioni] = 1/(n+1), valutato
con il numero di candidati di ciascuna pocket e poi mediato.
"""
import os, sys, statistics, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from qmode.pdb_reader import load_residues_from_pdb
from qmode.surface_filter import compute_atom_sasa
from scripts.run_pipeline import process_residue, process_ligand
from qmode.qubit_chain import get_h_hb_intensities, compute_h_hb_thresholds
from qmode.quantum_encoding import first_encoding
from qmode.ligand_reader import load_ligand_from_pdb, compute_all_ligand_centroids
from qmode.grover.search import window_interactivity, tile_offset
from scripts.make_pockets import LIGAND_CODES

PDBS = ["1GM8","1GPK","1HNN","1HQ2","1HSG","1IG3","1K3U","1L2S","1MMV","1N2V","1N46",
        "1OPK","1OYT","1Q1G","1R58","1R9O","1S19","1STP","1T46","1TZ8","1U1C","1XM6",
        "1Y6B","1YGC","1YV3","1YWR","2BSM","3PTB","4DFR"]


def build_chain(pocket_pdb):
    residues = load_residues_from_pdb(pocket_pdb, skip_water=True)
    sasa_map = compute_atom_sasa(pocket_pdb)
    chain = []
    for rec in residues:
        for s in process_residue(rec, surface_filter=True, sasa_threshold=1.0,
                                 sasa_map=sasa_map) or []:
            chain.append({"residue": rec.label, "coords": s.coords,
                          "type": s.feature_type, "intensity": s.intensity})
    return chain


def analyse(code, k):
    pocket = f"data/raw/{code.lower()}_pocket.pdb"
    full = f"data/raw/{code}.pdb"
    chain = build_chain(pocket)
    lig = load_ligand_from_pdb(full, ligand_code=LIGAND_CODES.get(code.lower()))
    if not lig:
        return None
    lig_sites = process_ligand(lig, max_sites=k)
    if not lig_sites or len(chain) < len(lig_sites):
        return None
    size = len(lig_sites)

    # soglie di binarizzazione dalla catena
    h_thr, hb_thr = compute_h_hb_thresholds(chain, 0.0, 1.0, 0.0, 1.0)
    lig_bits = "".join(first_encoding(*get_h_hb_intensities(
        {"type": s.feature_type, "intensity": s.intensity}), h_thr, hb_thr)
        for s in lig_sites)

    centroids = compute_all_ligand_centroids(full, lig.res_name)
    if not centroids:
        return None
    cs = np.array(centroids)

    # tutte le finestre a passo 1: distanza e bitstring
    dists, bits = [], []
    for i in range(len(chain) - size + 1):
        win = chain[i:i + size]
        c = np.mean([w["coords"] for w in win], axis=0)
        dists.append(float(np.min(np.linalg.norm(cs - c, axis=1))))
        bits.append("".join(first_encoding(*get_h_hb_intensities(w), h_thr, hb_thr)
                            for w in win))
    dists = np.array(dists)

    def pct(d):
        return 100.0 * float((dists < d).mean())

    match = [i for i, b in enumerate(bits) if b == lig_bits]
    out = {"code": code, "n_windows": len(dists), "size": size,
           "ceiling": float(dists.min()), "chance": float(np.median(dists)),
           "n_match": len(match)}
    if match:
        md = dists[match]
        out["match_pct_median"] = float(np.median([pct(d) for d in md]))
        out["match_best_pct"] = pct(md.min())
        out["match_best_dist"] = float(md.min())

    # candidati veri della pipeline: uno per shift offset, come run_pipeline/fast_eval
    cand = []
    for off in range(size):
        _, best_position = tile_offset(chain, size, off, h_thr, hb_thr)
        if lig_bits in best_position:
            cand.append(best_position[lig_bits])
    out["n_cand"] = len(cand)
    if cand:
        top = max(cand, key=lambda i: window_interactivity(chain[i:i + size]))
        out["top1_dist"] = float(dists[top])
        out["top1_pct"] = pct(dists[top])
        out["best_cand_dist"] = float(min(dists[i] for i in cand))
        out["best_cand_pct"] = pct(min(dists[i] for i in cand))
    return out


def report(k):
    rows = [r for r in (analyse(c, k) for c in PDBS) if r]
    got = [r for r in rows if r["n_match"]]
    cg  = [r for r in rows if r.get("n_cand")]
    print(f"\n{'='*78}\nk = {k}   ({len(rows)} pocket analizzate, {len(got)} con almeno un match)\n{'='*78}")
    print(f"{'target':8} {'#win':>5} {'soffitto':>9} {'caso':>7} {'top-1':>7} {'pct top-1':>10} {'#match':>7} {'pct set':>8} {'best set':>9}")
    for r in rows:
        if r.get("n_cand"):
            print(f"{r['code']:8} {r['n_windows']:5d} {r['ceiling']:9.2f} {r['chance']:7.2f} "
                  f"{r['top1_dist']:7.2f} {r['top1_pct']:9.1f}% {r['n_match']:7d} "
                  f"{r['match_pct_median']:7.1f}% {r['match_best_pct']:8.1f}%")
        else:
            print(f"{r['code']:8} {r['n_windows']:5d} {r['ceiling']:9.2f} {r['chance']:7.2f} "
                  f"{'--':>7} {'--':>10} {0:7d} {'--':>8} {'--':>9}")

    med = statistics.median
    print(f"\n-- riepilogo k={k} --")
    print(f"  copertura (pocket con match)        : {len(got)}/{len(rows)}")
    print(f"  soffitto  mediano / range           : {med([r['ceiling'] for r in rows]):.2f} A  "
          f"({min(r['ceiling'] for r in rows):.2f}-{max(r['ceiling'] for r in rows):.2f})")
    print(f"  caso      mediano / range           : {med([r['chance'] for r in rows]):.2f} A  "
          f"({min(r['chance'] for r in rows):.2f}-{max(r['chance'] for r in rows):.2f})")
    if got:
        print(f"  top-1     mediano / range           : {med([r['top1_dist'] for r in got]):.2f} A  "
              f"({min(r['top1_dist'] for r in got):.2f}-{max(r['top1_dist'] for r in got):.2f})")
        print(f"  pct mediano del set che matcha      : {med([r['match_pct_median'] for r in got]):.1f}%")
        print(f"  pct del migliore del set (mediana)  : {med([r['match_best_pct'] for r in got]):.1f}%")
        print(f"  pct top-1: media {statistics.mean([r['top1_pct'] for r in got]):.1f}%  "
              f"mediana {med([r['top1_pct'] for r in got]):.1f}%")
        print(f"  -- candidati veri della pipeline (tile_offset) --")
        print(f"  pocket con candidati                : {len(cg)}/{len(rows)}")
        print(f"  candidati per pocket (media)        : {statistics.mean([r['n_cand'] for r in cg]):.2f}")
        print(f"  top-1 distanza mediana              : {med([r['top1_dist'] for r in cg]):.2f} A  "
              f"({min(r['top1_dist'] for r in cg):.2f}-{max(r['top1_dist'] for r in cg):.2f})")
        print(f"  top-1 percentile  media / mediana   : {statistics.mean([r['top1_pct'] for r in cg]):.1f}% / "
              f"{med([r['top1_pct'] for r in cg]):.1f}%")
        print(f"  best-of-cand percentile media/med   : {statistics.mean([r['best_cand_pct'] for r in cg]):.1f}% / "
              f"{med([r['best_cand_pct'] for r in cg]):.1f}%")
        print(f"  baseline sui candidati 1/(n+1)      : "
              f"{statistics.mean([1.0/(r['n_cand']+1) for r in cg])*100:.1f}%")
        print(f"  hit <= 5 A del top-1 (candidati)    : {len([r for r in cg if r['top1_dist']<=5.0])}/{len(cg)}")
        print(f"  -- set che matcha (tutte le finestre) --")
        base = statistics.mean([1.0 / (r["n_match"] + 1) for r in got]) * 100
        print(f"  baseline casuale E[min_n]=1/(n+1)   : {base:.1f}%   "
              f"(n mediano del set = {med([r['n_match'] for r in got]):.0f})")
        reach = [r for r in rows if r["ceiling"] <= 5.0]
        hit = [r for r in got if r["top1_dist"] <= 5.0]
        print(f"  pocket con una finestra <= 5 A      : {len(reach)}/{len(rows)}")
        print(f"  hit <= 5 A del top-1                : {len(hit)}/{len(got)}  "
              f"(su quelle raggiungibili: {len([r for r in got if r['ceiling']<=5.0 and r['top1_dist']<=5.0])}/"
              f"{len([r for r in got if r['ceiling']<=5.0])})")


for k in (int(a) for a in (sys.argv[1:] or ["3"])):
    report(k)
