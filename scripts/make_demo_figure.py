"""Closing figure of the demo (Step 4 of report/demo_runbook.md): for every
target, the ceiling->chance segment with Grover's top-1 plotted on it.

Reads the table produced by scripts/percentile_eval.py. To regenerate the data:

    venv/bin/python scripts/percentile_eval.py 3 2>/dev/null > report/figures/percentile_k3.txt
    venv/bin/python scripts/make_demo_figure.py

Usage: python scripts/make_demo_figure.py [table] [output.png]
"""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = sys.argv[1] if len(sys.argv) > 1 else "report/figures/percentile_k3.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report/figures/demo_ceiling_chance.png"

rows = []
CODE = re.compile(r"^[0-9][A-Z0-9]{3}$")
for line in open(SRC, encoding="utf-8"):
    f = line.split()
    # target #win soffitto caso top-1 pct#match pct-set best-set
    if len(f) < 6 or not CODE.match(f[0]):
        continue
    rows.append({
        "code": f[0], "ceiling": float(f[2]), "chance": float(f[3]),
        "top1": None if f[4] == "--" else float(f[4]),
        "pct": None if f[5] == "--" else float(f[5].rstrip("%")),
    })

# The source table is rounded to 2 decimals and 1OYT shows up there as "5.00",
# but the true value is 5.000475: above the criterion, not below. Without this
# the hits become 3 instead of 2 — the same error that is in the report PDF.
for r in rows:
    if r["code"] == "1OYT":
        r["top1"] = 5.000475

assert len(rows) == 29, len(rows)
got = [r for r in rows if r["top1"] is not None]
assert len(got) == 23, len(got)

# sorted by ceiling: a quantity independent of the result, so the scatter of
# the dots reads for what it is
rows.sort(key=lambda r: r["ceiling"])

fig, ax = plt.subplots(figsize=(9.2, 7.6))
ys = range(len(rows))

for y, r in zip(ys, rows):
    ax.plot([r["ceiling"], r["chance"]], [y, y],
            color="#bfc7d1", lw=3.2, solid_capstyle="round", zorder=1)
    ax.plot(r["ceiling"], y, "|", color="#2c5f8a", ms=11, mew=2.0, zorder=3)
    ax.plot(r["chance"], y, "|", color="#8a8f98", ms=11, mew=2.0, zorder=3)
    if r["top1"] is None:
        ax.text(r["chance"] + 0.45, y, "no candidate", va="center",
                fontsize=7.5, color="#9aa0a8", style="italic")
        continue
    hit = r["top1"] <= 5.0
    ax.plot(r["top1"], y, "o", ms=7.2,
            color="#1a7f4b" if hit else "#c0392b",
            markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    if r["top1"] > r["chance"]:
        ax.plot([r["chance"], r["top1"]], [y, y],
                color="#e3b4ae", lw=1.2, ls=":", zorder=2)

ax.axvline(5.0, color="#1a7f4b", lw=1.0, ls="--", alpha=0.55, zorder=0)
ax.text(5.0, -1.25, " 5 Å criterion", color="#1a7f4b",
        fontsize=8.5, va="bottom", ha="left")

ax.set_yticks(list(ys))
ax.set_yticklabels([r["code"] for r in rows], fontsize=8.5, family="monospace")
for lbl, r in zip(ax.get_yticklabels(), rows):
    if r["code"] == "1HSG":
        lbl.set_color("#2c5f8a")
        lbl.set_fontweight("bold")

ax.set_xlabel("distance from the ligand centroid (Å)", fontsize=10)
ax.set_xlim(0, 24)
ax.set_ylim(-1.35, len(rows) - 0.2)
ax.grid(axis="x", color="#eceef1", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)

ax.set_title("Where the search lands, between the best possible and chance",
             fontsize=13, pad=44, loc="left", fontweight="bold")
ax.text(0, 1.012,
        "k = 3, 29 pockets.  The segment runs from the best window the representation makes "
        "available\n(ceiling) to the average window (chance).  The dot is Grover's top-1.",
        transform=ax.transAxes, fontsize=9, color="#555", va="bottom")

handles = [
    Line2D([], [], color="#2c5f8a", marker="|", ls="none", ms=11, mew=2, label="ceiling"),
    Line2D([], [], color="#8a8f98", marker="|", ls="none", ms=11, mew=2, label="chance"),
    Line2D([], [], color="#1a7f4b", marker="o", ls="none", ms=7, label="top-1 ≤ 5 Å (2/23)"),
    Line2D([], [], color="#c0392b", marker="o", ls="none", ms=7, label="top-1 > 5 Å (21/23)"),
]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, ncol=2)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("written", OUT)

med = lambda v: sorted(v)[len(v)//2] if len(v) % 2 else sum(sorted(v)[len(v)//2-1:len(v)//2+1])/2
print("checks:")
print("  median ceiling      ", round(med([r['ceiling'] for r in rows]), 2))
print("  median chance       ", round(med([r['chance'] for r in rows]), 2))
print("  median top-1        ", round(med([r['top1'] for r in got]), 2))
print("  mean percentile     ", round(sum(r['pct'] for r in got)/len(got), 1))
print("  hits <= 5 A         ", len([r for r in got if r['top1'] <= 5.0]), "/", len(got))
print("  top-1 beyond chance ", len([r for r in got if r['top1'] > r['chance']]), "/", len(got))
