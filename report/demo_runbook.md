# Q-MODE demo — runbook (target: 1HSG, HIV-1 protease + MK1)

Runtime: ~4 minutes, of which ~6 seconds is computation. Everything runs locally, no
network needed.

Prerequisites: the `venv` in place and `data/raw/1HSG.pdb` present (it is gitignored
because it is regenerable: `python scripts/make_pockets.py` downloads it again).

---

## Step 0 — before you go up

Open these in three browser tabs, already loaded:

1. `features_1hsg_protein.html` — SASA filter on the whole protein (966 → 559 sites)
2. `chain_1hsg.html` — the 3D flat chain of the pocket (126 sites)
3. the closing figure, `report/figures/demo_ceiling_chance.png` (Step 4)

In the terminal: large font, full-screen window, `cd` into the project root.

⚠️ The HTML pages need a window at least **1280 px** wide: below that the 3D panel stays
narrow and the molecule spills out of its frame.

---

## Step 1 — "what we start from" (30 s)

Show `features_1hsg_protein.html`, left panel.

> This is HIV-1 protease. RDKit finds 966 pharmacophore sites on it — donors, acceptors,
> hydrophobes, ionizables. Most of them are buried inside the protein, where a ligand
> never reaches.

Move to the right panel: **559 sites**, −42%. Both panels are at the same scale, so what
you see is genuinely "fewer sites", not "a smaller protein".

---

## Step 2 — "how we linearize it" (45 s)

Show `chain_1hsg.html`.

> Within each residue the sites are ordered by a BFS over the covalent bond graph;
> residues are concatenated in chain order. The result is **one sequence**: 126 sites,
> each with a type and an intensity. Thick lines are hops inside the same residue, thin
> ones are residue changes.

Rotate the molecule once: you can see the 1D chain jumping back and forth in 3D space.
That is the key point of the representation — and also its limit (Step 4).

---

## Step 3 — the run (90 s)

```bash
venv/bin/python scripts/run_pipeline.py --pdb data/raw/1hsg_pocket.pdb --ligand-pdb data/raw/1HSG.pdb --max-rows 12 2>/dev/null
```

`--max-rows 12` truncates the two long tables (flat sequence and quantum encoding), which
are otherwise 126 and 124 rows and scroll off the screen. `2>/dev/null` drops RDKit's
warnings. Without the flag the output is the usual complete one.

What to point at, in the order it appears:

| Line | What to say |
|---|---|
| `33 residues found` | the pocket is an 8 Å crop around the ligand |
| `[3/4] ... A8_ARG -> 8 sites` | the local pipeline, residue by residue |
| `Total sites : 126` | the flat chain of Step 2, now as a table |
| `Quantum Encoding` | every 3-site window → one **6-qubit** state, `\|001001⟩` |
| `Total segments: 124` | the search space |
| `Ligand: B902_MK1` | the real ligand, extracted and run through the same pipeline |
| `Shift/Site/Prob.` | Grover: 3 windows above the 1/N threshold |
| `Validation` | real distance from the ligand |

Probabilities wobble by a few thousandths between runs (Grover samples at finite shots):
the candidates, their order and the distances stay identical.

The ending tells itself: the top two candidates are **A25_ASP and B25_ASP**, the two
catalytic aspartates of HIV protease — the active site, the one every inhibitor targets.

---

## Step 4 — the honest slide (60-90 s)

**Do not close on Step 3.** The candidate is 6.48 Å from MK1's centroid: it got the right
region, not the right spot. And above all, 1HSG is a **lucky** case.

The slide is a single figure: `report/figures/demo_ceiling_chance.png`.

Regenerate it with:

```bash
venv/bin/python scripts/percentile_eval.py 3 2>/dev/null > report/figures/percentile_k3.txt
venv/bin/python scripts/make_demo_figure.py
```

### How to read it, out loud

One row per pocket. The **segment** is the space the method could have landed in:

- the left end (blue) is the **ceiling** — the best window the representation makes
  available, the best a perfect ranking could possibly pick;
- the right end (grey) is **chance** — the average window, i.e. where you land drawing
  at random.

The **dot** is Grover's top-1. If the method worked, the dots would hug the left end.

Three things to point out, in this order:

1. **The dots are scattered across the whole segment.** No pull towards the ceiling.
2. **Nine dots out of 23 sit beyond chance** (red dotted line): on those pockets the
   search does *worse* than drawing at random. 1YGC ends at 22.18 Å, with a ceiling
   at 4.94.
3. **There are two green ones**, 1IG3 and 1Q1G. They are the only two top-1 under 5 Å in
   the whole benchmark. 1OYT grazes the criterion but sits at 5.000475 Å: above, not below.

Then the 1HSG row, highlighted in blue on the axis: ceiling 2.96 — top-1 **6.48** —
chance 9.57. It is one of the better cases, and it still lands halfway.

### The numbers, if anyone asks

| 1HSG | | Benchmark (29 pockets, k=3) | |
|---|---|---|---|
| windows | 124 | coverage | 23/29 |
| ceiling | 2.96 Å | candidates per pocket | 2.43 |
| chance | 9.57 Å | top-1, median distance | **7.75 Å** (chance: 7.88) |
| top-1 | **6.48 Å** | top-1, percentile | mean **47.2%**, median 42.0% |
| percentile | **13.7%** | best-of-candidates | 32.2% (random baseline 30.8%) |
| | | hits ≤ 5 Å | 2/23 (2/19 among the reachable ones) |

### The closing line

> On 1HSG the method lands in the 14th percentile. Averaged over the 29 pockets it lands
> in the 47th, and chance is 50. The pipeline builds the representation and the search
> runs, but **the ranking does not beat drawing at random**. The reason is back at Step 2:
> a list of k sites with their intensities does not say where the empty space is, and what
> makes a pocket a pocket is precisely the cavity around the sites.

This opens the "known limitations" slide and the discussion section of the report.

---

## Fallback

If the laptop, the projector or the venv misbehave: screenshots of the same four steps on
the slides. To be redone every time the HTML pages are regenerated.

To capture:
1. `features_1hsg_protein.html` — the two panels side by side
2. `chain_1hsg.html` — the 3D chain + the first rows of the table
3. the output of the Step 3 command, in two screenfuls (flat chain + encoding; Grover +
   validation)
4. `report/figures/demo_ceiling_chance.png`
