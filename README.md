# Q-MODE: Amino Acid Lattice Mapping for Binding Pocket Representation

> A pre-processing pipeline for translating protein binding pockets into discrete 2D lattice interaction chains, implementing the pharmacophoric site-selection algorithm omitted in Lilliapoulos *et al.*

---

## Overview

This repository provides an implementation of the pre-processing steps required to encode a protein binding pocket (supplied as a `.pdb` file) into a structured sequence of pharmacophoric interaction sites mapped onto a 2D discrete lattice. The pipeline operationalises the *inner lattice* construction introduced at a theoretical level by Lilliapoulos *et al.*, for which no reference implementation was previously available.

The resulting lattice chain encodes per-residue pharmacophoric features — hydrophobicity, hydrogen-bond donor/acceptor capacity, charge, and related descriptors — in a spatially consistent representation suitable for downstream machine-learning tasks such as molecular docking score prediction, pocket similarity search, and quantum-inspired optimisation models.

---

## Scientific Background

Protein–ligand interaction modelling frequently requires a compact yet information-rich encoding of the binding site geometry. Lilliapoulos *et al.* proposed a discrete lattice abstraction (*inner lattice*) in which each residue of the pocket is represented by one or more interaction sites projected onto a 2D integer grid. However, the original work did not supply the algorithmic pipeline connecting raw structural data to the lattice encoding. This repository fills that gap by implementing:

1. **Residue extraction** from PDB coordinate files.
2. **Pharmacophoric feature computation** via RDKit.
3. **Representative site selection** per residue (configurable K-selection strategy).
4. **Dimensionality reduction** of 3D coordinates to a 2D projection plane (PCA or MDS).
5. **Lattice snapping** — mapping continuous 2D coordinates to integer grid nodes (i, j).
6. **Site labelling** — encoding pharmacophore type as one-hot vectors, integer indices, or learned embeddings.

---

## Repository Structure

```
Q-MODE-project/
├── amino_lattice/
│   ├── pdb_reader.py           # PDB parsing → residues with 3D coordinates
│   ├── feature_extraction.py   # RDKit-based pharmacophore and H-bond feature extraction
│   ├── site_selection.py       # K representative sites per residue
│   ├── lattice_fitting.py      # 3D → 2D projection (PCA / MDS)
│   ├── snapping.py             # Continuous 2D coordinates → integer lattice nodes (i, j)
│   ├── labeling.py             # Pharmacophore type → one-hot / index / embedding vector
│   └── pipeline.py             # Orchestration entry point for SMILES-based input
├── scripts/
│   └── run_pocket.py           # Main entry point for PDB file input
├── data/
│   └── raw/                    # Place input PDB files here
├── idrofobicity/               # Hydrophobicity scale definitions and utilities
├── tests/                      # Unit tests
├── Report/                     # Project report and supplementary material
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
| `seaborn` | 0.12 |
| `tqdm` | 4.65 |

---

## Usage

Place the target PDB file in `data/raw/`, then invoke the main script:

```bash
# Print lattice chain to stdout and open an interactive plot
python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --plot

# Save JSON, CSV, and a static plot image to an output directory
python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb \
    --output data/processed/ \
    --save-plot data/processed/pocket.png
```

### Command-line Options

| Option | Default | Description |
|---|---|---|
| `--pdb` | *(required)* | Path to the PDB file of the binding pocket or full protein |
| `--output` | `None` | Directory for JSON and CSV output files |
| `--plot` | `False` | Open an interactive 2D lattice visualisation |
| `--save-plot` | `None` | Save the lattice plot as a PNG image |
| `--projection` | `pca` | 3D → 2D projection method: `pca` or `mds` |
| `--max-k` | `6` | Maximum number of interaction sites per residue |
| `--k-strategy` | `active_features` | Site-count selection strategy: `active_features`, `heavy_atoms`, `fixed`, or `groups` |
| `--label-mode` | `one_hot` | Pharmacophore encoding: `one_hot`, `index`, or `embedding` |

---

## Output Format

**Standard output:** a tabular summary listing each interaction site with its index, source residue identifier, integer lattice coordinates (i, j), and pharmacophore type.

**`pocket_chain.json`** — full chain with per-site metadata:

```json
[
  {"site_idx": 0, "residue": "A156_ILE", "i": 2, "j": -3, "type": "Hydrophobe"},
  ...
]
```

**`pocket_chain.csv`** — equivalent tabular format, suitable for direct ingestion into ML pipelines.

---

## Running Tests

```bash
pytest tests/
```

---

## Citation

If you use this software in academic work, please cite the original theoretical framework:

> Lilliapoulos *et al.* *(full citation to be added — see `Report/` for reference details)*

---

## License

This project is released for academic and research use. See `LICENSE` for details (if applicable).
