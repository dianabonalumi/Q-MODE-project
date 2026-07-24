# Q-MODE: Amino Acid Lattice Mapping for Binding Pocket Representation

> A pipeline for translating protein binding pockets into discrete 2D lattice interaction chains, then encoding them as qubit-ready quantum states.

---

## Overview

Q-MODE takes a protein binding pocket, supplied as a `.pdb` file, and turns it into a structured, spatially-consistent sequence of pharmacophoric interaction sites on a 2D integer lattice. Each site carries a pharmacophore type (hydrophobic, aromatic, H-bond donor/acceptor, ionizable, ...) and an intensity derived from real geometry, not placeholder values.

On top of the lattice chain, the pipeline implements a **quantum encoding stage**, inspired by *"Quantum algorithm for protein-ligand docking sites identification in the interaction space"*, which converts sliding-window segments of the chain into qubit-ready binary states (first encoding) and probability amplitudes (second encoding), so the pocket representation can feed into Grover-search-style or amplitude-based quantum docking algorithms.

Typical downstream uses: docking-score prediction, pocket similarity search, and quantum-inspired optimization models.

---

## Pipeline Stages

1. **Residue extraction** — parse residues and real 3D coordinates directly from PDB atoms; bond orders assigned by structural comparison against a known template, not by positional overlay (`pdb_reader.py`).
2. **Pharmacophoric feature computation** — RDKit-based atom feature extraction (`feature_extraction.py`).
3. **Intensity assignment** — every site gets an intrinsic h/hb value: hydrophobic sites use Crippen atomic LogP contributions (`feature_extraction.py`); H-bond donor/acceptor/ionizable sites use Abraham solute descriptors, looked up by functional group and real PDB atom name (`abraham_hbond.py`).
4. **Surface filter** (on by default) — keeps only solvent-exposed sites via SASA (`surface_filter.py`).
5. **Topological ordering** — within each residue, sites are ordered by a BFS over the covalent bond graph starting from the backbone N atom (`site_selection.py`).
6. **Chain assembly** — residues are concatenated in protein-chain order (`chain_id`, `res_seq`); each residue's sites inherit that position plus their own topological order, producing one flat, chain-consistent sequence.
7. **Quantum encoding** — splits the flat chain into ligand-sized sliding-window segments and applies:
   - **First encoding**: binarizes h/hb intensity into 2-bit qubit basis states for Grover search.
   - **Second encoding**: computes probability amplitudes `(a, b, c, d)` for amplitude-based distance calculations.
   (`quantum_encoding.py`, `qubit_chain.py`)
8. **Ligand extraction** (optional, via `--ligand-pdb`) — reads the ligand's HETATM group from a PDB file (auto-picks the largest non-water, non-standard-amino-acid group, or a specific one via `--ligand-code`) and assigns its bond orders from the PDB Chemical Component Dictionary (`qmode/ligand_reader.py`), falling back to geometric bond-order perception (`rdDetermineBonds`) if the ligand has no CCD entry or the network is unavailable. The ligand then goes through the same feature-extraction/dedup/ordering steps as protein residues.
9. **Grover search** (optional, via `--ligand-pdb`) — tiles the protein into non-overlapping ligand-sized windows for each shift offset, builds the protein superposition state over the unique first-encoding basis states, and runs the modified Grover oracle + diffusion operator on a Qiskit simulator to identify which windows match the extracted ligand's (h, hb) profile above the `1/N` threshold (`qmode/grover/search.py`). When several windows collapse onto the same first-encoding bitstring, the one with the highest aggregate h/hb intensity ("most interactive") is kept, and matching candidates are ranked by that same score.
10. **Distance validation** (optional, requires `--ligand-pdb`) — a classical stand-in for the paper's SWAP-test-based ranking: each Grover candidate's site centroid is compared, by Euclidean distance, to the real ligand's heavy-atom centroid, giving a ground-truth accuracy check per candidate (`qmode/grover/evaluate.py`). Only meaningful in benchmark mode, where the true ligand pose is known.

---

## Repository Structure

```
Q-MODE-project/
├── qmode/
│   ├── pdb_reader.py           # PDB parsing → residues with real 3D coordinates
│   ├── ligand_reader.py        # Ligand HETATM parsing → mol via PDB CCD template (or geometric fallback)
│   ├── feature_extraction.py   # RDKit-based pharmacophore feature extraction + Crippen h
│   ├── abraham_hbond.py        # Abraham hb intensity lookup by functional group
│   ├── surface_filter.py       # SASA-based solvent-exposure filter
│   ├── site_selection.py       # Topological (BFS) site ordering
│   ├── site_dedup_centroid.py  # Merges near-duplicate same-type sites into one centroid site
│   ├── quantum_encoding.py     # First/second quantum encoding (Grover / amplitude)
│   ├── qubit_chain.py          # Sliding-window segmentation + qubit chain assembly
│   ├── grover/                 # Modified Grover search (oracle + diffusion + shift)
│   │   ├── __init__.py         # Re-exports the public search API
│   │   ├── search.py           # tile_offset, oracle/diffusion, circuit execution, search_docking_sites
│   │   └── evaluate.py         # Distance-to-ligand validation of candidate windows
│   ├── lattice_fitting.py      # 3D → 2D projection (PCA) — utility, not used by the main pipeline
│   ├── snapping.py             # 2D coords → integer lattice nodes — utility, not used by the main pipeline
│   └── labeling.py             # One-hot pharmacophore labeling — utility, not used by the main pipeline
├── scripts/
│   ├── run_pipeline.py         # Main CLI entry point (whole protein or cropped pocket PDB)
│   ├── plot_residue_chain.py   # 3D structure + 1D chain visualization for a single residue
│   └── make_pockets.py         # Downloads sample PDB structures and crops binding pockets
├── data/
│   └── raw/                    # Input / generated pocket PDB files
├── tests/                      # Unit tests
├── Report/                     # Project report (PDF)
├── requirements.txt
└── setup.py
```

---

## Installation

**Requirements:** Python ≥ 3.9, pip.

```bash
# Clone the repository
git clone https://github.com/dianabonalumi/Q-MODE-project.git
cd Q-MODE-project

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install the package and its dependencies
pip install -e .
```

### Dependencies

| Package | Minimum Version |
|---|---|
| `rdkit` | 2023.3.1 |
| `numpy` | 1.24 |
| `pandas` | 2.0 |
| `scikit-learn` | 1.3 |
| `scipy` | 1.11 |
| `matplotlib` | 3.7 |
| `tqdm` | 4.65 |

---

## Usage

### 1. Get a pocket to work with

Either drop your own pocket PDB file into `data/raw/`, or generate sample pockets from real PDB structures (trypsin, HIV-1 protease, DHFR, streptavidin) by cropping around their bound ligand:

```bash
python scripts/make_pockets.py
```

### 2. Run the pipeline on a pocket

```bash
python scripts/run_pipeline.py --pdb data/raw/3PTB_pocket.pdb --plot
```

```bash
# Save JSON/CSV outputs and static plot images
python scripts/run_pipeline.py --pdb data/raw/3PTB_pocket.pdb \
    --output data/processed/ \
    --save-plot data/processed/pocket.png
```

```bash
# Run Grover search against a real ligand extracted from a full PDB structure
python scripts/run_pipeline.py --pdb data/raw/4dfr_pocket.pdb --ligand-pdb data/raw/4DFR.pdb
```

```bash
# Visualize a single residue: 3D structure + its 1D site chain
python scripts/plot_residue_chain.py --pdb data/raw/3PTB_pocket.pdb --chain A --resseq 189
```

### Command-line Options (`run_pipeline.py`)

| Option | Default | Description |
|---|---|---|
| `--pdb` | *(required)* | Path to the PDB file (whole protein or cropped pocket) |
| `--output` | `None` | Directory for JSON/CSV output files |
| `--plot` | `False` | Show an interactive visualization |
| `--save-plot` | `None` | Save plots as PNG images |
| `--no-surface-filter` | *(filter on by default)* | Disable the SASA solvent-exposure filter (keep buried sites too) |
| `--sasa-threshold` | `1.0` | SASA threshold (Å²) for considering an atom solvent-exposed |
| `--ligand-size` | `3` | Sliding-window size (in sites) used for the classical quantum-chain segmentation (Step 7) — unrelated to the real ligand used by Grover |
| `--ligand-pdb` | `None` | Path to a PDB file containing the ligand's HETATM records (e.g. the full structure downloaded from PDB). When set, runs Grover search with the extracted ligand |
| `--ligand-code` | `None` | 3-letter ligand code to disambiguate when `--ligand-pdb` has multiple non-water HETATM groups. Default: auto-picks the group with the most heavy atoms |
| `--ligand-max-sites` | `3` | Max number of ligand pharmacophore sites used by Grover (6 qubits). Raising this past ~5 sites (10+ qubits) makes oracle/diffusion synthesis very slow in Qiskit |

---

## Output Format

**`pocket_chain.json`** — full flat lattice chain with per-site metadata:

```json
{
  "pdb": "3PTB_pocket.pdb",
  "pocket_centroid": [12.4, 8.1, -3.2],
  "n_residues": 18,
  "n_sites_total": 41,
  "ordering": "chain_sequence",
  "flat_chain": [
    {"residue": "A156_ILE", "coords": [11.2, 7.4, -2.1], "type": "Hydrophobe", "intensity": 0.812}
  ]
}
```

**`pocket_chain.csv`** — equivalent tabular format, suitable for direct ingestion into ML pipelines.

**`quantum_chain.json`** — sliding-window qubit segments with first/second quantum encodings:

```json
{
  "ligand_size": 3,
  "h_range": [0.10, 0.95],
  "hb_range": [0.05, 0.88],
  "n_segments": 14,
  "segments": [
    {
      "segment_idx": 0,
      "first_encoding_state": "1011",
      "second_encoding_amplitudes": [{"a": 0.71, "b": 0.35, "c": 0.51, "d": 0.33}],
      "residues": ["A156_ILE", "A157_VAL"]
    }
  ]
}
```

**`grover_search.json`** — Grover candidates, ranked by descending interactivity score, each with a distance-to-ligand validation:

```json
{
  "ligand": "A1_BEN",
  "ligand_hbs": [[0.55, 0.0], [0.0, 0.81], [0.42, 0.0]],
  "ligand_size": 3,
  "candidates": [
    {
      "shift_offset": 2,
      "window_start_index": 44,
      "interactivity_score": 1.8,
      "residues": ["A215_TRP", "A216_GLY"],
      "ligand_bitstring": "100000",
      "matching_probability": 0.558,
      "threshold": 0.0769,
      "n_unique_states": 13,
      "window_centroid": [2.313, 16.52, 14.168],
      "distance_to_ligand_A": 5.327
    }
  ]
}
```

---

## Running Tests

```bash
pytest tests/
```

---

## Scientific Background

The quantum-encoding stage follows the two-step scheme from *"Quantum algorithm for protein-ligand docking sites identification in the interaction space"*: a **first encoding** that binarizes hydrophobicity/H-bond intensity into qubit basis states for Grover search, and a **second encoding** that computes probability amplitudes for amplitude-based Euclidean distance estimation. See `Report/Q-MODE_Report_EN.pdf` for the full derivation and results.

The Grover search itself (`qmode/grover/`) is implemented and unit-tested: protein superposition state, oracle, and diffusion operator (Eqs. 5-8 of the paper), run per shift offset on a Qiskit simulator. It is wired into `run_pipeline.py` via `--ligand-pdb`, with the ligand's own (h, hb) profile extracted from a real PDB (`qmode/ligand_reader.py`) rather than hand-typed values.

Two known limitations: the ligand's H-bond intensity has no Abraham data (the table is indexed by amino-acid residue/atom name), so it stays at the neutral default — same open question as the protein-side Abraham assumptions above. And Qiskit's `UnitaryGate` synthesis for the oracle/diffusion operators doesn't scale past ~10 qubits (tens of seconds to minutes per shift offset), which is why `--ligand-max-sites` defaults to 3 (6 qubits).

Not yet implemented: the paper's own SWAP-test-based (amplitude/quantum) ranking of candidate docking sites. `qmode/grover/evaluate.py` covers the same evaluation *goal* — ranking candidates by distance to the ligand — with a classical Euclidean distance between each candidate's site centroid and the ligand's real heavy-atom centroid, rather than a quantum SWAP test on the second encoding's amplitudes. It only applies in benchmark mode (`--ligand-pdb` with a known bound ligand), not prospective screening.

