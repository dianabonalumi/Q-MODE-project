"""
Correlazione fra distanza nello spazio delle ampiezze (second encoding) e
distanza 3D reale dal centroide del ligando, su tutte le finestre di ogni pocket.

E' la misura decisiva del report: se il descrittore portasse informazione sulla
posizione del ligando, finestre che assomigliano all'encoding del ligando
sarebbero anche piu' vicine a esso nello spazio.

Uso:
    python scripts/diagnose_descriptor.py 3 5

Spearman e' calcolato sui ranghi con numpy (nessuna dipendenza da scipy).
rho positivo = segno corretto (piu' lontano nell'encoding, piu' lontano nello spazio).
"""
import os, sys, statistics, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from qmode.pdb_reader import load_residues_from_pdb
from qmode.surface_filter import compute_atom_sasa
from scripts.run_pipeline import process_residue, process_ligand
from qmode.qubit_chain import get_h_hb_intensities
from qmode.quantum_encoding import second_encoding
from qmode.ligand_reader import load_ligand_from_pdb, compute_all_ligand_centroids
from scripts.make_pockets import LIGAND_CODES

PDBS = ["1GM8","1GPK","1HNN","1HQ2","1HSG","1IG3","1K3U","1L2S","1MMV","1N2V","1N46",
        "1OPK","1OYT","1Q1G","1R58","1R9O","1S19","1STP","1T46","1TZ8","1U1C","1XM6",
        "1Y6B","1YGC","1YV3","1YWR","2BSM","3PTB","4DFR"]


def spearman(x, y):
    """rho di Spearman = Pearson sui ranghi (ranghi medi in caso di pari merito)."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # media dei ranghi per i valori uguali
        vs = v[order]
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(np.arange(i, j + 1, dtype=float))
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def amplitudes(sites, bounds):
    h_min, h_max, hb_min, hb_max = bounds
    out = []
    for s in sites:
        h, hb = s if isinstance(s, tuple) else get_h_hb_intensities(s)
        a = second_encoding(h, hb, h_min, h_max, hb_min, hb_max)
        out.extend([a["a"], a["b"], a["c"], a["d"]])
    return np.array(out)


def analyse(code, k):
    pocket, full = f"data/raw/{code.lower()}_pocket.pdb", f"data/raw/{code}.pdb"
    sasa_map = compute_atom_sasa(pocket)
    chain = []
    for rec in load_residues_from_pdb(pocket, skip_water=True):
        for s in process_residue(rec, surface_filter=True, sasa_threshold=1.0,
                                 sasa_map=sasa_map) or []:
            chain.append({"coords": s.coords, "type": s.feature_type,
                          "intensity": s.intensity})

    lig = load_ligand_from_pdb(full, ligand_code=LIGAND_CODES.get(code.lower()))
    if not lig:
        return None
    lig_sites = process_ligand(lig, max_sites=k)
    if not lig_sites:
        return None
    size = len(lig_sites)
    if len(chain) < size:
        return None

    h_pos = [get_h_hb_intensities(s)[0] for s in chain if get_h_hb_intensities(s)[0] > 0]
    hb_pos = [get_h_hb_intensities(s)[1] for s in chain if get_h_hb_intensities(s)[1] > 0]
    bounds = ((min(h_pos), max(h_pos)) if h_pos else (0.0, 1.0)) + \
             ((min(hb_pos), max(hb_pos)) if hb_pos else (0.0, 1.0))

    lig_vec = amplitudes([get_h_hb_intensities(
        {"type": s.feature_type, "intensity": s.intensity}) for s in lig_sites], bounds)

    centroids = compute_all_ligand_centroids(full, lig.res_name)
    if not centroids:
        return None
    cs = np.array(centroids)

    enc_d, real_d = [], []
    for i in range(len(chain) - size + 1):
        win = chain[i:i + size]
        enc_d.append(float(np.linalg.norm(amplitudes(win, bounds) - lig_vec)))
        c = np.mean([w["coords"] for w in win], axis=0)
        real_d.append(float(np.min(np.linalg.norm(cs - c, axis=1))))

    return {"code": code, "n": len(enc_d), "size": size,
            "rho": spearman(enc_d, real_d)}


def report(k):
    rows = [r for r in (analyse(c, k) for c in PDBS) if r and not np.isnan(r["rho"])]
    print(f"\n{'='*56}\nk = {k}   ({len(rows)} pocket con un rho definito)\n{'='*56}")
    for r in rows:
        print(f"  {r['code']:8} finestre {r['n']:5d}   rho = {r['rho']:+.4f}")
    rhos = [r["rho"] for r in rows]
    n = len(rows)
    print(f"\n  rho medio                    : {statistics.mean(rhos):+.4f}")
    print(f"  rho mediano                  : {statistics.median(rhos):+.4f}")
    print(f"  pocket con |rho| > 0.3       : {sum(1 for r in rhos if abs(r) > 0.3)}/{n}")
    print(f"  pocket con il segno corretto : {sum(1 for r in rhos if r > 0)}/{n}")


for k in (int(a) for a in (sys.argv[1:] or ["3"])):
    report(k)
