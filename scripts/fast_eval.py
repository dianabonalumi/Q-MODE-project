import os
import sys
import warnings

# Suppress RDKit/SASA warnings for clean output
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qmode.pdb_reader import load_residues_from_pdb
from qmode.surface_filter import compute_atom_sasa
from scripts.run_pipeline import process_residue, process_ligand
from qmode.qubit_chain import get_h_hb_intensities, compute_h_hb_thresholds
from qmode.quantum_encoding import first_encoding
from qmode.ligand_reader import load_ligand_from_pdb, compute_all_ligand_centroids
from qmode.grover.search import tile_offset, window_interactivity
from qmode.grover.evaluate import evaluate_candidates
from scripts.make_pockets import LIGAND_CODES

pdbs = ["1GM8", "1GPK", "1HNN", "1HQ2", "1HSG", "1IG3", "1K3U", "1L2S", "1MMV", "1N2V", "1N46", "1OPK", "1OYT", "1Q1G", "1R58", "1R9O", "1S19", "1STP", "1T46", "1TZ8", "1U1C", "1XM6", "1Y6B", "1YGC", "1YV3", "1YWR", "2BSM", "3PTB", "4DFR"]

results = []

for code in pdbs:
    pocket_pdb = os.path.join(os.path.dirname(__file__), f"../data/raw/{code.lower()}_pocket.pdb")
    ligand_pdb = os.path.join(os.path.dirname(__file__), f"../data/raw/{code}.pdb")
    
    # 1. Load pocket
    residues = load_residues_from_pdb(pocket_pdb, skip_water=True)
    sasa_map = compute_atom_sasa(pocket_pdb)
    
    flat_chain = []
    for rec in residues:
        sites = process_residue(rec, surface_filter=True, sasa_threshold=1.0, sasa_map=sasa_map)
        if not sites: continue
        for s in sites:
            flat_chain.append({
                "residue": rec.label,
                "res_name": rec.res_name,
                "res_seq": rec.res_seq,
                "chain_id": rec.chain_id,
                "coords": s.coords,
                "type": s.feature_type,
                "intensity": s.intensity
            })
            
    # 2. Load Ligand
    ligand_rec = load_ligand_from_pdb(ligand_pdb, ligand_code=LIGAND_CODES.get(code.lower()))
    if not ligand_rec:
        results.append((code, "Failed Ligand Parse"))
        continue
        
    ligand_sites = process_ligand(ligand_rec, max_sites=3)
    if not ligand_sites:
        results.append((code, "No Ligand Sites"))
        continue
        
    ligand_hbs = [get_h_hb_intensities({"type": s.feature_type, "intensity": s.intensity}) for s in ligand_sites]
    grover_ligand_size = len(ligand_hbs)
    
    # 3. Thresholds
    h_pos = [get_h_hb_intensities(s)[0] for s in flat_chain if get_h_hb_intensities(s)[0] > 0]
    hb_pos = [get_h_hb_intensities(s)[1] for s in flat_chain if get_h_hb_intensities(s)[1] > 0]
    h_min, h_max = (min(h_pos), max(h_pos)) if h_pos else (0.0, 1.0)
    hb_min, hb_max = (min(hb_pos), max(hb_pos)) if hb_pos else (0.0, 1.0)
    
    h_thr, hb_thr = compute_h_hb_thresholds(flat_chain, h_min, h_max, hb_min, hb_max)
    
    ligand_bitstring = "".join(first_encoding(h, hb, h_thr, hb_thr) for h, hb in ligand_hbs)
    
    # 4. Fast Classical Grover Search
    candidates = []
    for offset in range(grover_ligand_size):
        unique_bitstrings, best_position = tile_offset(flat_chain, grover_ligand_size, offset, h_thr, hb_thr)
        if ligand_bitstring in best_position:
            window_start = best_position[ligand_bitstring]
            window_sites = flat_chain[window_start:window_start + grover_ligand_size]
            candidates.append({
                "shift_offset": offset,
                "window_start_index": window_start,
                "interactivity_score": window_interactivity(window_sites),
                "residues": list(dict.fromkeys(s["residue"] for s in window_sites)),
            })
            
    if not candidates:
        results.append((code, "No Match (Threshold Failed)"))
        continue
        
    # 5. Evaluate
    candidates.sort(key=lambda c: c["interactivity_score"], reverse=True)
    ligand_centroids = compute_all_ligand_centroids(ligand_pdb, ligand_rec.res_name)
    evaluated = evaluate_candidates(candidates, flat_chain, ligand_centroids, grover_ligand_size)
    
    top_candidate_start = candidates[0]["window_start_index"]
    dist_of_top = next(e["distance_to_ligand_A"] for e in evaluated if e["window_start_index"] == top_candidate_start)
    
    results.append((code, f"{dist_of_top:.2f} A"))

print("Target | Top-1 Distance")
print("---|---")
for r in results:
    print(f"{r[0]} | {r[1]}")
