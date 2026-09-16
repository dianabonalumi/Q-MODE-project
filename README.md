# Q-MODE: Amino Acid Lattice Mapping for Binding Pocket Representation

> A pipeline that turns a protein binding pocket into an ordered 1D chain of pharmacophoric
> interaction sites, then encodes that chain as qubit-ready quantum states and searches it
> with a modified Grover algorithm.

---

## Overview

Q-MODE takes a protein binding pocket, supplied as a `.pdb` file, and turns it into a
discrete, ordered sequence of pharmacophoric interaction sites — a flat 1D chain,
analogous to a sequence of residues. Each site carries a pharmacophore type (hydrophobic,
aromatic, H-bond donor/acceptor, ionizable) and an intensity derived from real geometry
and measured physical scales, not placeholder values.

On top of the flat chain, the pipeline implements a **quantum encoding stage**, following
*"Quantum algorithm for protein-ligand docking sites identification in the interaction
space"*, which converts sliding-window segments of the chain into qubit-ready binary
states (first encoding) and probability amplitudes (second encoding), and a **modified
Grover search** that scans the chain for windows matching a ligand's own encoded profile.

**The end-to-end result is negative.** The search does not localize the binding site: over
29 crystallographic complexes the windows it returns are no closer to the true ligand than
windows drawn at random from the same pocket. The cause is the descriptor, not the quantum
stage — see [Benchmark Results](#benchmark-results) below and `report/Q-MODE_Report.pdf`
for the full analysis.

---

## Pipeline Stages

The eleven stages below are the same ones numbered in Section 2 of the report.

1. **Residue extraction** — parse residues and real 3D coordinates directly from PDB atoms;
   bond orders assigned by matching each residue against a canonical SMILES template
   (`pdb_reader.py`).
2. **Pharmacophoric feature extraction** — RDKit feature factory over `BaseFeatures.fdef`,
   six interaction families, each feature placed at the centroid of its constituent atoms
   (`feature_extraction.py`).
3. **Intensity assignment** — hydrophobic sites use Crippen atomic LogP contributions
   (`feature_extraction.py`); H-bond donor/acceptor/ionizable sites use Abraham solute
   descriptors, looked up by residue and PDB atom name (`abraham_hbond.py`).
4. **Surface filtering** (on by default) — keeps only solvent-exposed sites via SASA,
   computed on the ligand-free structure (`surface_filter.py`).
5. **Site deduplication** — same-type features within 1.5 Å are merged into a single site
   at the group centroid, intensities summed (`site_dedup_centroid.py`).
6. **Topological ordering** — within each residue, sites are ordered by a BFS over the
   covalent bond graph, backbone first, then the side chain outward (`site_selection.py`).
7. **Chain assembly** — residues are concatenated in protein-chain order
   (`chain_id`, `res_seq`), producing one flat, chain-consistent sequence.
8. **Quantum encoding** — splits the flat chain into ligand-sized sliding windows and
   applies:
   - **First encoding**: binarizes the h/hb intensities into 2-bit basis states per site,
     for Grover search.
   - **Second encoding**: computes probability amplitudes `(a, b, c, d)` per site, for
     amplitude-based distance estimation.

   (`quantum_encoding.py`, `qubit_chain.py`)
9. **Ligand extraction** (optional, via `--ligand-pdb`) — reads the ligand's HETATM group
   from a PDB file (auto-picks the largest non-water, non-standard-amino-acid group, or a
   specific one via `--ligand-code`) and assigns its bond orders from the PDB Chemical
   Component Dictionary, falling back to geometric perception (`rdDetermineBonds`) if the
   ligand has no CCD entry or the network is unavailable. The ligand then goes through the
   same feature-extraction/intensity/dedup/ordering steps as a protein residue
   (`ligand_reader.py`).
10. **Grover search** (optional, via `--ligand-pdb`) — tiles the flat chain into
    non-overlapping ligand-sized windows for each shift offset, builds the superposition
    over the unique first-encoding basis states, and runs the oracle + diffusion operator
    on a Qiskit simulator to identify which windows match the ligand's `(h, hb)` profile
    above the `1/N` threshold. When several windows collapse onto the same bitstring, the
    one with the highest aggregate h/hb intensity is kept, and matching candidates are
    ranked by that same score (`qmode/grover/search.py`).
11. **Distance-to-ligand validation** (optional, requires `--ligand-pdb`) — a classical
    stand-in for the paper's SWAP-test-based ranking: each candidate's site centroid is
    compared, by Euclidean distance, to the real ligand's heavy-atom centroid
    (`qmode/grover/evaluate.py`). Only meaningful in benchmark mode, where the true ligand
    pose is known.

---

## Repository Structure

```
Q-MODE-project/
├── qmode/
│   ├── pdb_reader.py           # PDB parsing → residues with real 3D coordinates
│   ├── ligand_reader.py        # Ligand HETATM parsing → mol via PDB CCD template (or geometric fallback)
│   ├── feature_extraction.py   # RDKit pharmacophore feature extraction + Crippen h
│   ├── abraham_hbond.py        # Abraham hb intensity lookup
│   ├── surface_filter.py       # SASA-based solvent-exposure filter (Shrake & Rupley)
│   ├── site_dedup_centroid.py  # Merges near-duplicate same-type sites into one centroid site
│   ├── site_selection.py       # Topological (BFS) site ordering
│   ├── quantum_encoding.py     # First/second quantum encoding (Grover / amplitude)
│   ├── qubit_chain.py          # Sliding-window segmentation + qubit chain assembly
│   └── grover/                 # Modified Grover search
│       ├── __init__.py         # Re-exports the public search API
│       ├── search.py           # tile_offset, oracle/diffusion, circuit execution, search_docking_sites
│       └── evaluate.py         # Distance-to-ligand validation of candidate windows
├── scripts/
│   ├── run_pipeline.py         # Main CLI entry point (whole protein or cropped pocket PDB)
│   ├── make_pockets.py         # Downloads the benchmark PDB structures and crops binding pockets
│   ├── plot_residue_chain.py   # 3D structure + 1D chain visualization for a single residue
│   ├── visualize_features.py   # Interactive 3D page: pharmacophore sites before/after the SASA filter
│   ├── visualize_chain.py      # Interactive 3D page: the flat chain, plus the site table
│   ├── percentile_eval.py      # Benchmark tables: ceiling / chance / top-1 percentile, k sweep
│   ├── diagnose_descriptor.py  # Spearman rho: amplitude-space similarity vs true 3D distance
│   ├── fast_eval.py            # Benchmark driver over the 29 evaluated pockets
│   ├── make_demo_figure.py     # Closing figure of the demo, built from the percentile table
│   └── validate_step1.py … validate_step5.py
│                               # Per-stage validation: template matching, type coverage,
│                               # intensity ranges, deduplication, BFS determinism
├── data/
│   ├── raw/                    # Pocket PDB files (tracked) + full structures (gitignored)
│   └── processed/              # JSON/CSV pipeline outputs (gitignored)
├── tests/                      # Unit tests
├── report/
│   ├── Q-MODE_Report.pdf       # Project report
│   ├── demo_runbook.md         # Step-by-step script for the live demo
│   └── figures/                # Generated figures (PNGs are gitignored)
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
| `matplotlib` | 3.7 |
| `biopython` | 1.79 |
| `qiskit` | 2.2 |
| `qiskit-aer` | 0.17 |

`biopython` provides the Shrake & Rupley SASA implementation; `qiskit` and `qiskit-aer`
run the Grover circuit. The same list is in `requirements.txt` and `setup.py`.

---

## Usage

### 1. Get a pocket to work with

The cropped pockets are tracked in `data/raw/` (31 of them). The full structures they came
from are **not** — they are regenerable, so they are gitignored. To fetch them, and to
regenerate the pockets from scratch:

```bash
python scripts/make_pockets.py
```

This downloads each structure from RCSB (**network required**) and crops, for each, the
protein residues with at least one atom within 5 Å of a ligand atom. The set is four
hand-picked complexes — trypsin, HIV-1 protease, DHFR, streptavidin — plus the Astex
Diverse Set (Hartshorn et al.).

Pocket files are lowercase (`1hsg_pocket.pdb`), full structures uppercase (`1HSG.pdb`).

### 2. Run the pipeline on a pocket

```bash
python scripts/run_pipeline.py --pdb data/raw/3ptb_pocket.pdb --plot
```

```bash
# Save JSON/CSV outputs and static plot images
python scripts/run_pipeline.py --pdb data/raw/3ptb_pocket.pdb \
    --output data/processed/ \
    --save-plot data/processed/pocket.png
```

```bash
# Run Grover search against a real ligand extracted from a full PDB structure
# (needs data/raw/4DFR.pdb — run make_pockets.py first)
python scripts/run_pipeline.py --pdb data/raw/4dfr_pocket.pdb --ligand-pdb data/raw/4DFR.pdb
```

```bash
# Visualize a single residue: 3D structure + its 1D site chain
python scripts/plot_residue_chain.py --pdb data/raw/3ptb_pocket.pdb --chain A --resseq 189
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
| `--ligand-size` | `3` | Sliding-window size (in sites) used for the classical quantum-chain segmentation (Stage 8) — unrelated to the real ligand used by Grover |
| `--ligand-pdb` | `None` | Path to a PDB file containing the ligand's HETATM records (e.g. the full structure downloaded from PDB). When set, runs Grover search with the extracted ligand |
| `--ligand-code` | `None` | 3-letter ligand code to disambiguate when `--ligand-pdb` has multiple non-water HETATM groups. Default: auto-picks the group with the most heavy atoms |
| `--ligand-max-sites` | `3` | Max number of ligand pharmacophore sites used by Grover (6 qubits). Raising this past ~5 sites (10+ qubits) makes oracle/diffusion synthesis very slow in Qiskit |
| `--max-rows` | `None` | Truncate the two long tables (flat sequence and quantum encoding) to their first N rows. On a typical pocket each is 120+ rows; meant for presenting on a projector. Default prints everything |

---

## Demo: running the whole thing on one target

A four-minute walkthrough on **1HSG** (HIV-1 protease bound to the inhibitor MK1) that
takes a pocket from PDB file to Grover candidates. The computation itself takes about six
seconds and needs no quantum hardware — Grover runs on the Qiskit simulator. The
*computation* is offline; the *setup* below downloads from RCSB, so do it before you need
it, not in front of the audience.

`report/demo_runbook.md` is the full script: what to say at each step, the numbers to have
ready, and the fallback if the projector misbehaves. The short version follows.

### Setup

The demo needs both the cropped pocket and the full structure it came from. The full
structures are gitignored (they are regenerable), so fetch them first — **this step needs
network access**:

```bash
python scripts/make_pockets.py
```

Then build the two interactive pages:

```bash
python scripts/visualize_features.py --pdb-pocket data/raw/1HSG.pdb --pdb-protein data/raw/1HSG.pdb --output features_1hsg_protein.html
```

```bash
python scripts/visualize_chain.py --pdb data/raw/1hsg_pocket.pdb --output chain_1hsg.html
```

Open both in a browser window **at least 1280 px wide** — below that the 3D panel stays
narrow and the molecule spills out of its frame.

### The four steps

**1. What the pipeline starts from.** `features_1hsg_protein.html` shows the 966
pharmacophore sites RDKit finds on the protein, next to the 559 that survive the SASA
filter (−42%). Both panels are rendered at the same scale, so the difference reads as
"fewer sites", not "a smaller protein".

**2. How the pocket becomes a sequence.** `chain_1hsg.html` shows the 126 sites of the
pocket as one ordered chain in 3D — BFS over the bond graph within each residue, residues
in chain order. Thick lines are hops inside a residue, thin lines are residue changes.
Rotating the molecule shows the 1D chain jumping back and forth in space.

**3. The run.**

```bash
python scripts/run_pipeline.py --pdb data/raw/1hsg_pocket.pdb --ligand-pdb data/raw/1HSG.pdb --max-rows 12 2>/dev/null
```

`--max-rows 12` keeps the flat-sequence and encoding tables (126 and 124 rows) on one
screen; `2>/dev/null` hides RDKit's warnings. Without either flag the output is the usual
complete one.

The run ends on three Grover candidates, validated against the real ligand position. The
top two are `A25_ASP` and `B25_ASP` — the catalytic aspartates of HIV protease, the active
site every inhibitor targets. The top-1 sits 6.48 Å from MK1's centroid.

Grover samples at finite shots, so the printed probabilities move by a few thousandths
between runs. The candidates, their order and the distances do not.

**4. What the result is actually worth.** This is the part not to skip.

```bash
python scripts/percentile_eval.py 3 2>/dev/null > report/figures/percentile_k3.txt
python scripts/make_demo_figure.py
```

This writes `report/figures/demo_ceiling_chance.png`: one row per pocket, a segment from
the *ceiling* (the best window the representation makes available) to *chance* (the
average window), with Grover's top-1 plotted on it. The dots are scattered across the
whole segment with no pull towards the ceiling; nine of 23 land beyond chance. Over the
29 pockets the top-1 sits at the 47.2nd percentile on average, against 50 for chance.

1HSG is one of the better cases — 13.7th percentile — which is exactly why the demo should
not end at step 3. See [Benchmark Results](#benchmark-results) for the full picture.

### Trying other targets

Any of the 29 benchmark pockets works, with the pocket in lowercase and the full structure
in uppercase:

```bash
python scripts/run_pipeline.py --pdb data/raw/1ig3_pocket.pdb --ligand-pdb data/raw/1IG3.pdb --max-rows 12 2>/dev/null
```

Six targets (1HNN, 1N2V, 1R58, 1STP, 1TZ8, 1XM6) print `No candidate site found for the
given ligand`: no window clears the 1/N threshold. That is a result, not a crash. The two
pockets without a full structure in `data/raw/` — 1a08 and 1ddm — run up to the flat chain
but cannot run Grover.

---

## Output Format

**`pocket_chain.json`** — the full flat chain with per-site metadata:

```json
{
  "pdb": "3ptb_pocket.pdb",
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

The quantum-encoding stage follows the two-step scheme from *"Quantum algorithm for
protein-ligand docking sites identification in the interaction space"*: a **first
encoding** that binarizes hydrophobicity/H-bond intensity into qubit basis states for
Grover search, and a **second encoding** that computes probability amplitudes for
amplitude-based Euclidean distance estimation. See `report/Q-MODE_Report.pdf` for the full
derivation and results.

The Grover search itself (`qmode/grover/`) is implemented and unit-tested: protein
superposition state, oracle, and diffusion operator, run per shift offset on a Qiskit
simulator. It is wired into `run_pipeline.py` via `--ligand-pdb`, with the ligand's own
`(h, hb)` profile extracted from a real PDB (`qmode/ligand_reader.py`) rather than
hand-typed values.

Two known limitations of the implementation. The Abraham intensity table is indexed by
residue and PDB atom name, so it covers protein residues by construction and not ligand
atoms; the heuristic fallback recognises only four functional groups, and anything outside
them stays at the neutral default. And Qiskit's `UnitaryGate` synthesis for the
oracle/diffusion operators does not scale past ~10 qubits (tens of seconds to minutes per
shift offset), which is why `--ligand-max-sites` defaults to 3 (6 qubits).

**Not implemented, and on the evidence below not worth implementing as it stands:** the
paper's own SWAP-test-based ranking of candidate docking sites. `qmode/grover/evaluate.py`
covers the same evaluation *goal* — ranking candidates by distance to the ligand — with a
classical Euclidean distance between each candidate's site centroid and the ligand's real
heavy-atom centroid, rather than a quantum SWAP test on the second encoding's amplitudes.
A classical stand-in for the amplitude ranking was measured on the benchmark set and
carries no localization signal; see below. It only applies in benchmark mode
(`--ligand-pdb` with a known bound ligand), not prospective screening.

---

## Benchmark Results

Measured 2026-09-15 on the 29 pockets in `data/raw/` that have a matching full
structure (`1a08` and `1ddm` do not). The aggregate tables below are reproduced by:

```bash
python scripts/percentile_eval.py 3 4 5
```

and a single target by:

```bash
python scripts/run_pipeline.py --pdb data/raw/<x>_pocket.pdb --ligand-pdb data/raw/<X>.pdb
```

> Earlier versions of these numbers (21/29 pockets, 8.22 Å median) were measured before
> a ligand-selection fix: on five targets `load_ligand_from_pdb` was auto-picking the
> largest HETATM group, which is a cofactor bound elsewhere rather than the ligand the
> pocket was cropped around. `make_pockets.LIGAND_CODES` now pins the right code per
> structure. The conclusion did not change; the numbers did.

### Headline: the search does not localize the binding site

At the default `--ligand-max-sites 3`, **23 of 29** pockets produce candidates, 2.43 per
pocket on average. The other 6 produce none because no window clears the `1/N`
probability threshold. Median distance from the top-ranked candidate's window centroid to
the true ligand centroid is **7.75 Å** (range 4.26–22.18); only **2 of 23** land within
5 Å.

Absolute distances depend on pocket size, so the meaningful measure is where a candidate
falls in the distribution of *all* possible windows of that pocket (0% = closest window in
the protein, 50% = indistinguishable from chance):

| | mean percentile | median | random baseline |
|---|---|---|---|
| Top-1 by `interactivity_score` | 47.2% | 42.0% | 50.0% |
| Best of the candidates returned | 32.2% | 26.2% | 30.8% |

The random baseline for "best of *n* candidates" is `E[min] = 1/(n+1)` over the observed
candidate counts. **Grover's candidate set is not better than picking the same number of
windows at random**, and ranking within it by `interactivity_score` is not better than
picking one of them at random.

The 5 Å criterion is not out of reach for the representation: the closest window available
in each pocket (the *ceiling*) has a median distance of **3.76 Å**, and 25 of 29 pockets
contain at least one window within 5 Å. The average window (*chance*) sits at **7.88 Å** —
and the search's top-1, at 7.75 Å, lands next to it rather than next to the ceiling.
`scripts/make_demo_figure.py` draws this one row per pocket.

### The bottleneck is the descriptor, not the ranking

Ranking criteria and window size were both tested and neither is the limiting factor.
Because Grover searches for an exact bitstring, the set of windows matching the ligand is
determined classically, so its quality can be measured directly:

| `--ligand-max-sites` | pockets with a match | median percentile of the matching set | best of the set | random baseline |
|---|---|---|---|---|
| 3 | 23/29 | 48.2% | 16.8% | 18.4% |
| 4 | 12/29 | 49.7% | 11.7% | 23.9% |
| 5 | 7/29 | 31.1% | 9.8% | 39.9% |

At the default `k=3` the matching set sits at the 48.2nd percentile — essentially chance —
so no ranking criterion can help: there is nothing to rank. Larger `k` does buy real signal
(at `k=5` the best of the matching set is at 9.8% against 39.9% expected by chance), but
coverage collapses to 7 pockets, because the bitstring has `4^k` possible values and an
exact match becomes vanishingly rare. And the signal does not survive the pipeline's own
selection: at `k=5` the actual top-1 lands at the 54.7th percentile on average, i.e. worse
than chance.

Replacing the exact bitstring match with a distance on the second encoding's amplitudes
restores coverage (29/29 pockets at `k=5`) but not accuracy: top-1 by Euclidean amplitude
distance lands at the 44.8th percentile on average (42.1st median), and best-of-3 at 27.8%
against 25% expected from three random draws. *(Measured separately; no script in the repo
reproduces this regime yet.)*

The reason shows up in the Spearman correlation between amplitude-space similarity to the
ligand and true 3D distance to it, over every window of every pocket — reproduce with
`python scripts/diagnose_descriptor.py 3 5`:

```
k=3   rho mean -0.008   median +0.013   |rho| > 0.3 in 2/29   correct sign in 16/29
k=5   rho mean +0.020   median +0.069   |rho| > 0.3 in 4/29   correct sign in 17/29
```

Zero. A window's `(h, hb)` profile carries no information about where the ligand binds.

The encoding is also badly degenerate. Across the 29 benchmark ligands, the 6-qubit first
encoding takes only **9 distinct values out of 64 possible**, and a single state,
`|010101⟩`, covers 11 of them — MK1 (an HIV protease inhibitor) and FSN (a thrombin
inhibitor) are the same object as far as the oracle is concerned. That ligand-side
distribution carries **2.74 bits of the 6 available** (the figure quoted in the report).
Measured instead over the *windows* of a pocket, the effective entropy is 4.10 of 6 bits
at `k=3` and 5.59 of 10 at `k=5` — the two are different populations, not two estimates of
the same quantity. Within a pocket, windows that share a bitstring have centroids 17.21 Å
apart on average, against 16.66 Å for random pairs: never closer than chance.

### What was ruled out

> The three checks below are reported here only; the report's Section 6.1 argues the same
> eliminations from the benchmark measurements instead.

**It is not the known defects.** Removing the spurious `PosIonizable` sites (keeping them
only on ARG/LYS/HIS) moves the correlation from -0.0119 to -0.0122 and leaves the AUC
unchanged at 0.654.

**It is not the loss of 3D geometry.** This was the obvious hypothesis — the per-site
`coords` survive in the flat chain but never enter the encoding — and it was tested and
rejected. Median per-pocket AUC, cross-validated within each pocket: amplitudes alone
0.654, amplitudes plus the pairwise distances between a window's sites 0.658, six channels
with donors and acceptors separated 0.612, six channels plus distances 0.615. Under a
scale-free label (the closest 10% of windows, all 29 pockets) every variant falls between
0.47 and 0.55. **Making the descriptor geometry-aware does not recover signal.**

**The windows are not geometrically absurd, but they are not surface patches either.**
Across windows of 5 consecutive sites the median maximum internal distance is 7.63 Å and
the median radius of gyration 3.09 Å, which looks like a legitimate local patch. But the
chain is ordered residue by residue, and within a residue by BFS over the bond graph, so a
short window is almost always a single sidechain rather than a patch of surface. The
windows are compact because they are one residue, not because they trace a pocket.

**What is actually missing is the empty space.** Binding is complementarity, bulk and
burial, and none of the three is expressible as a list of *k* sites with intensities —
with or without their pairwise distances. What distinguishes a pocket is the cavity
*around* the sites. The SASA filter measures global solvent accessibility, not membership
in a cavity, so it keeps every exposed site on the protein surface and has no way to
prefer the concave ones. Replacing it with a cavity detector (fpocket, CASTp) is the
change that would have to come before any further work on the quantum ranking.

