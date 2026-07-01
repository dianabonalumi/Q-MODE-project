# Q-MODE: Amino Acid Lattice Mapping for Binding Pocket Representation

> A pipeline for translating protein binding pockets into discrete 2D lattice interaction chains, then encoding them as qubit-ready quantum states.

---

## Overview

Q-MODE takes a protein binding pocket, supplied as a `.pdb` file, and turns it into a structured, spatially-consistent sequence of pharmacophoric interaction sites on a 2D integer lattice. Each site carries a pharmacophore type (hydrophobic, aromatic, H-bond donor/acceptor, ionizable, ...) and an intensity derived from real geometry, not placeholder values.

On top of the lattice chain, the pipeline implements a **quantum encoding stage**, inspired by *"Quantum algorithm for protein-ligand docking sites identification in the interaction space"*, which converts sliding-window segments of the chain into qubit-ready binary states (first encoding) and probability amplitudes (second encoding), so the pocket representation can feed into Grover-search-style or amplitude-based quantum docking algorithms.

Typical downstream uses: docking-score prediction, pocket similarity search, and quantum-inspired optimization models.

---

## Pipeline Stages

1. **Residue extraction** — parse residues and 3D coordinates from a PDB file (`pdb_reader.py`).
2. **Pharmacophoric feature computation** — RDKit-based atom feature extraction (`feature_extraction.py`).
3. **Hydrogen-bond geometry** — physically grounded H-bond intensity from donor–acceptor distance and D–H···A angle, computed only between different residues (`hbond_geometry.py`).
4. **Representative site selection** — chooses K sites per residue, with a configurable strategy (`site_selection.py`).
5. **Lattice fitting** — projects 3D coordinates to 2D via PCA (`lattice_fitting.py`).
6. **Snapping** — maps continuous 2D coordinates to integer lattice nodes `(i, j)` (`snapping.py`).
7. **Labeling** — encodes each site's pharmacophore type and intensity (`labeling.py`).
8. **Quantum encoding** — splits the flat chain into ligand-sized sliding-window segments and applies:
   - **First encoding**: binarizes hydrophobic/H-bond intensity into 2-bit qubit basis states for Grover search.
   - **Second encoding**: computes probability amplitudes `(a, b, c, d)` for amplitude-based distance calculations.
   (`quantum_encoding.py`, `qubit_chain.py`)

---

## Repository Structure

```
Q-MODE-project/
├── amino_lattice/
│   ├── pdb_reader.py           # PDB parsing → residues with 3D coordinates
│   ├── feature_extraction.py   # RDKit-based pharmacophore feature extraction
│   ├── hbond_geometry.py       # Distance/angle-based hydrogen-bond intensity model
│   ├── site_selection.py       # K representative sites per residue
│   ├── lattice_fitting.py      # 3D → 2D projection (PCA)
│   ├── snapping.py             # Continuous 2D coordinates → integer lattice nodes (i, j)
│   ├── labeling.py             # Pharmacophore type/intensity labeling
│   ├── quantum_encoding.py     # First/second quantum encoding (Grover / amplitude)
│   ├── qubit_chain.py          # Sliding-window segmentation + qubit chain assembly
│   └── pipeline.py             # AminoLatticePipeline: SMILES, batch, and PDB entry points
├── scripts/
│   ├── run_pocket.py           # Main CLI entry point for pocket PDB files
│   ├── make_pockets.py         # Downloads sample PDB structures and crops binding pockets
│   └── validate.py             # Chemical sanity checks, real-pocket runs, determinism checks
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
python scripts/run_pocket.py --pdb data/raw/3PTB_pocket.pdb --plot
```

```bash
# Save JSON/CSV outputs and a static plot image
python scripts/run_pocket.py --pdb data/raw/3PTB_pocket.pdb \
    --output data/processed/ \
    --save-plot data/processed/pocket.png
```

### Command-line Options

| Option | Default | Description |
|---|---|---|
| `--pdb` | *(required)* | Path to the pocket PDB file |
| `--output` | `None` | Directory for JSON/CSV output files |
| `--plot` | `False` | Show an interactive 2D lattice visualization |
| `--save-plot` | `None` | Save the lattice plot as a PNG image |
| `--k-strategy` | `active_features` | Site-count selection strategy: `active_features`, `heavy_atoms`, `fixed`, or `groups` |
| `--max-k` | `6` | Maximum number of interaction sites per residue |
| `--ligand-size` | `3` | Sliding-window size (in sites) used for quantum-chain segmentation |

---

## Output Format

**`pocket_chain.json`** — full flat lattice chain with per-site metadata:

```json
{
  "pdb": "3PTB_pocket.pdb",
  "pocket_centroid": [12.4, 8.1, -3.2],
  "n_residues": 18,
  "n_sites_total": 41,
  "ordering": "spatial_distance_from_centroid",
  "flat_chain": [
    {"residue": "A156_ILE", "i": 2, "j": -3, "type": "Hydrophobe", "intensity": 0.812}
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

---

## Validation

`scripts/validate.py` runs three independent checks:

- **Chemical sanity** — the 20 standard amino acids produce the pharmacophore types expected from their known chemistry (e.g., hydrophobic side chains → `Hydrophobe`, acidic residues → `NegIonizable`).
- **Real pockets** — the pipeline runs on the 5 heterogeneous pockets generated by `make_pockets.py` and reports pharmacophore composition and H-bond statistics.
- **Determinism** — two runs on the same pocket with fixed seeds produce identical output.

```bash
python scripts/make_pockets.py   # once, to download and crop sample pockets
python scripts/validate.py
```

---

## Running Tests

```bash
pytest tests/
```

---

## Scientific Background

The quantum-encoding stage follows the two-step scheme from *"Quantum algorithm for protein-ligand docking sites identification in the interaction space"*: a **first encoding** that binarizes hydrophobicity/H-bond intensity into qubit basis states for Grover search, and a **second encoding** that computes probability amplitudes for amplitude-based Euclidean distance estimation. See `Report/Q-MODE_Report_EN.pdf` for the full derivation and results.

