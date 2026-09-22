"""Effetto di --candidates-per-offset sul benchmark: quanto migliora il
miglior candidato quando ogni shift offset ne restituisce n invece di 1.

Per ogni tasca, per ogni n:
  - i candidati sono le prime n finestre che matchano il bitstring del
    ligando in ciascun offset, ordinate per interattivita' (tile_offset_ranked);
  - "best of set" e' la distanza minima fra loro e il centroide del ligando;
  - il controllo pesca ESATTAMENTE lo stesso numero di finestre a caso fra
    tutte quelle a passo 1 della catena, senza reinserimento. Appaiare il
    conteggio e' cio' che rende confrontabili le due colonne: prendere un
    minimo su piu' estrazioni abbassa la distanza comunque.

Niente Qiskit e niente campionamento, quindi i numeri sono deterministici e
non c'e' nessun seed: il gate della soglia equivale all'appartenenza del
bitstring alla piastrellatura (una iterazione porta il marcato a ~9/N contro
una soglia di 1/N), e media, mediana e p del controllo sono in forma chiusa
sulla distribuzione completa delle distanze:

    P(min di n estrazioni senza reinserimento >= d_(i)) = C(N-i, n) / C(N, n)

Uso:
    python scripts/peroffset_eval.py            # n = 1 2 3
    python scripts/peroffset_eval.py 1 3 5
"""
import os, sys, statistics, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from qmode.pdb_reader import load_residues_from_pdb
from qmode.surface_filter import compute_atom_sasa
from scripts.run_pipeline import process_residue, process_ligand
from qmode.qubit_chain import get_h_hb_intensities, compute_h_hb_thresholds
from qmode.quantum_encoding import first_encoding
from qmode.ligand_reader import load_ligand_from_pdb, compute_all_ligand_centroids
from qmode.grover.search import tile_offset_ranked
from qmode.grover.evaluate import all_window_distances
from scripts.make_pockets import LIGAND_CODES

PDBS = ["1GM8","1GPK","1HNN","1HQ2","1HSG","1IG3","1K3U","1L2S","1MMV","1N2V","1N46",
        "1OPK","1OYT","1Q1G","1R58","1R9O","1S19","1STP","1T46","1TZ8","1U1C","1XM6",
        "1Y6B","1YGC","1YV3","1YWR","2BSM","3PTB","4DFR"]
HIT_A = 5.0


def _survival(n_windows, n_draws, i):
    """P(il migliore di n_draws estrazioni sia >= la i-esima distanza ordinata)."""
    if n_windows - i < n_draws:
        return 0.0
    p = 1.0
    for j in range(n_draws):
        p *= (n_windows - i - j) / (n_windows - j)
    return p


def random_best_stats(dists, n_draws):
    """Media e mediana esatte del 'migliore di n_draws finestre a caso'."""
    sd = np.sort(dists)
    n_windows = len(sd)
    n_draws = min(n_draws, n_windows)
    mean, median, prev = 0.0, float(sd[-1]), 1.0
    for i in range(n_windows):
        nxt = _survival(n_windows, n_draws, i + 1)
        mean += float(sd[i]) * (prev - nxt)
        if prev >= 0.5 > nxt:
            median = float(sd[i])
        prev = nxt
    return mean, median


def p_at_least_as_good(dists, n_draws, d):
    """P(n_draws finestre a caso facciano almeno quanto d) = 1 - C(M,n)/C(N,n)."""
    n_windows = len(dists)
    worse = int((dists > d).sum())
    p = 1.0
    for j in range(n_draws):
        if worse - j <= 0:
            return 1.0
        p *= (worse - j) / (n_windows - j)
    return 1.0 - p


def build(code):
    """Catena piatta, bitstring del ligando e distanze di TUTTE le finestre."""
    pocket = os.path.join("data", "raw", f"{code.lower()}_pocket.pdb")
    full = os.path.join("data", "raw", f"{code}.pdb")
    sasa_map = compute_atom_sasa(pocket)
    chain = []
    for rec in load_residues_from_pdb(pocket, skip_water=True):
        for s in process_residue(rec, surface_filter=True, sasa_threshold=1.0,
                                 sasa_map=sasa_map) or []:
            chain.append({"residue": rec.label, "coords": s.coords,
                          "type": s.feature_type, "intensity": s.intensity})

    lig = load_ligand_from_pdb(full, ligand_code=LIGAND_CODES.get(code.lower()))
    if not lig:
        return None, "no ligand"
    lig_sites = process_ligand(lig, max_sites=3)
    if not lig_sites or len(chain) < len(lig_sites):
        return None, "no ligand sites"

    hbs = [get_h_hb_intensities({"type": s.feature_type, "intensity": s.intensity})
           for s in lig_sites]
    h_pos = [get_h_hb_intensities(s)[0] for s in chain if get_h_hb_intensities(s)[0] > 0]
    hb_pos = [get_h_hb_intensities(s)[1] for s in chain if get_h_hb_intensities(s)[1] > 0]
    h_thr, hb_thr = compute_h_hb_thresholds(
        chain, min(h_pos or [0.0]), max(h_pos or [1.0]),
        min(hb_pos or [0.0]), max(hb_pos or [1.0]))

    bits = "".join(first_encoding(h, hb, h_thr, hb_thr) for h, hb in hbs)
    dists = all_window_distances(
        chain, compute_all_ligand_centroids(full, lig.res_name), len(hbs))
    return (chain, len(hbs), h_thr, hb_thr, bits, dists), None


def main(per_offsets):
    rows = {n: [] for n in per_offsets}
    skipped = []

    for code in PDBS:
        built, why = build(code)
        if built is None:
            skipped.append(f"{code} [{why}]")
            continue
        chain, size, h_thr, hb_thr, bits, dists = built

        ranked = {}
        for off in range(size):
            _, positions = tile_offset_ranked(chain, size, off, h_thr, hb_thr)
            if bits in positions:
                ranked[off] = positions[bits]
        if not ranked:
            skipped.append(f"{code} [no match]")
            continue

        for n in per_offsets:
            starts = [s for off in sorted(ranked) for s in ranked[off][:n]]
            best = min(float(dists[s]) for s in starts)
            r_mean, r_med = random_best_stats(dists, len(starts))
            rows[n].append({
                "code": code, "n_cand": len(starts), "best": best,
                "pct": 100.0 * float((dists < best).sum()) / len(dists),
                "rand_mean": r_mean, "rand_med": r_med,
                "p": p_at_least_as_good(dists, len(starts), best),
                "hit": best <= HIT_A,
            })

    ref = per_offsets[0]
    print(f"\nTasche con candidati: {len(rows[ref])}/{len(PDBS)}"
          f"   (escluse: {', '.join(skipped)})\n")

    hdr = (f"{'n/offset':>9} {'cand/tasca':>11} {'best media':>11} {'best med.':>10} "
           f"{'pct media':>10} {'pct med.':>9} {'caso media':>11} {'caso med.':>10} "
           f"{'p mediano':>10} {'hit<=5A':>8}")
    print(hdr)
    print("-" * len(hdr))
    for n in per_offsets:
        R = rows[n]
        col = lambda k: [r[k] for r in R]
        print(f"{n:>9} {statistics.mean(col('n_cand')):>11.2f} "
              f"{statistics.mean(col('best')):>10.2f}A {statistics.median(col('best')):>9.2f}A "
              f"{statistics.mean(col('pct')):>9.1f}% {statistics.median(col('pct')):>8.1f}% "
              f"{statistics.mean(col('rand_mean')):>10.2f}A {statistics.median(col('rand_med')):>9.2f}A "
              f"{statistics.median(col('p')):>10.3f} {sum(col('hit')):>5}/{len(R)}")

    last = per_offsets[-1]
    print(f"\nPer tasca — best (A) a n = {', '.join(map(str, per_offsets))}, "
          f"caso appaiato e p a n = {last}:")
    head = f"{'target':>7} {'cand':>5} " + "".join(f"{'n=' + str(n):>8}" for n in per_offsets)
    print(head + f" {'caso':>8} {'p':>7}")
    for i, r in enumerate(rows[ref]):
        line = f"{r['code']:>7} {rows[last][i]['n_cand']:>5} "
        line += "".join(f"{rows[n][i]['best']:>8.2f}" for n in per_offsets)
        print(line + f" {rows[last][i]['rand_med']:>8.2f} {rows[last][i]['p']:>7.3f}")

    improved = sum(1 for i, r in enumerate(rows[ref])
                   if rows[last][i]["best"] < r["best"] - 1e-9)
    print(f"\nTasche in cui n = {last} migliora il best: {improved}/{len(rows[ref])}")
    print("Deterministico: nessun campionamento, nessun seed.")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    main(args)
