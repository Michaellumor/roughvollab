"""Two per-cell figures of P5's Stage 1 and Stage 2 results, from their committed output.

Inputs (read only; by default from ../output relative to this file):
  p5_stage2_cells.csv  Stage 2's cells, one row per (calibrator, seed q, pair): 12,000
                       rows, 6,000 per calibrator (ensemble, nre), five seeds each (D109).
                       Read here: calibrator, q, pair, s, bin, okA, okB, failure, abs_dA, E.
  p5_stage1.csv        Stage 1's output, one row per accepted pair: 1,200 rows (D106).
                       Read here: pair, s, H (true H), c_A (the exact posterior's
                       contraction of H under prior A), bin. Joined to the cells on pair.

Outputs (to --out, ../output by default; PDF and 300-dpi PNG of each):
  p5_fig_fidelity.pdf / .png   Figure 1: fidelity against contraction, per cell.
  p5_fig_excess.pdf / .png     Figure 2: excess prior dependence against true H, per
                               cell, at s = 0.005, labelled exploratory.

Figure 1 plots, for each calibrator (the deep ensemble, then the neural ratio
estimator), |d| = |H_hat_A - exact posterior mean| on a shared log axis against c under
prior A, for every Identified and Partial cell at all three noise levels: (842 + 328)
pairs x 5 seeds = 5,850 cells per panel, coloured by noise level. The dashed lines are
H-a's expected bound, 0.01, and its refutation threshold, 0.03; the vertical line is
the bin edge, c = 0.9. Any |d| below the axis floor is drawn at the floor, and the
count is printed and written on the panel. The Flat bin (30 pairs, H-d untestable) is
not plotted; H-a is read on the Identified bin alone, so the Partial cells are context.

Figure 2 plots, for each calibrator, E = (d_B - d_A) / 0.20 on a shared linear axis
against true H, for every Identified and Partial cell at s = 0.005, the one level at
which both calibrators have networks under both priors: (139 + 231) x 5 = 1,850 cells
per panel. The bins differ by colour and by marker. The dashed lines are prior B's first
percentile, H = 0.1875, and its mean, 0.32; the thin line is E = 0. H-b's and H-c's
thresholds are not drawn: they apply to medians over a bin, not to single cells.

Neither figure is pre-specified by the frozen draft, and neither decides anything.
Figure 2 is labelled exploratory: its view by true H comes close to line 135's first
exploratory item, as D109 says of its own cuts by true H.

Honesty guards (each refuses to plot, with a message, rather than draw a misleading
figure): the Stage 1 bin counts and the counts at s = 0.005 must be D106's; each bin
must agree with its c_A; the cells must hold both calibrators, five seeds each, every
pair once per seed, with bin and s agreeing with Stage 1; no plotted cell may carry a
failure or a missing value; the plotted counts must be exactly 5,850 and 1,850 per
calibrator; and every plotted c and true H must lie inside the drawn axis ranges.

Pure plotting: no result is recomputed. Run:
  py -3 -X utf8 analysis/p5_paper_figures.py
  py -3 -X utf8 analysis/p5_paper_figures.py --out DIR --cells CELLS.csv --stage1 STAGE1.csv
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

# Facts asserted against the data (D106, D109).
CALIBRATORS = ("ensemble", "nre")
TITLES = {"ensemble": "Deep ensemble", "nre": "Neural ratio estimator"}
SEEDS = 5
LEVELS = (0.001, 0.003, 0.005)
BINS_ALL = {"Identified": 842, "Partial": 328, "Flat": 30}          # 1,200 pairs
BINS_005 = {"Identified": 139, "Partial": 231, "Flat": 30}          # the 400 at s = 0.005
FIG1_CELLS = (BINS_ALL["Identified"] + BINS_ALL["Partial"]) * SEEDS  # 5,850 per calibrator
FIG2_CELLS = (BINS_005["Identified"] + BINS_005["Partial"]) * SEEDS  # 1,850 per calibrator

# Axis ranges and reference lines, from the frozen draft: the box (line 39), prior B's mean
# (line 66) and first percentile (line 126), the bin edges (line 89), and H-a's bound and
# refutation threshold (line 121).
C_RANGE = (0.5, 1.0)           # Partial and Identified bins
C_EDGE = 0.9                   # Identified: c >= 0.9; Partial: 0.5 <= c < 0.9
HA_BOUND, HA_REFUTE = 0.01, 0.03
H_BOX = (0.02, 0.48)
B_P01, B_MEAN = 0.1875, 0.32
FLOOR = 1e-5                   # Figure 1's log-axis floor; cells below it are drawn at it

# Colour. Noise level is an ordered magnitude: one hue (blue) in three steps, light to dark.
NOISE_HUE = {0.001: "#3987e5", 0.003: "#1c5cab", 0.005: "#0d366b"}
# The two bins are categories: two distinct hues, plus a marker shape each so the figure
# survives greyscale print.
BIN_HUE = {"Identified": "#2a78d6", "Partial": "#eb6834"}
BIN_MARKER = {"Identified": "o", "Partial": "^"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"   # primary, secondary, muted text
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
DASH = (0, (4, 2.5))


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def floats(rows, key):
    """A column as float; an empty field becomes NaN (how the cells CSV writes a missing value)."""
    return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])


def refuse(msg):
    raise ValueError(f"refusing to plot: {msg}")


def check_stage1(stage1):
    pairs = [r["pair"] for r in stage1]
    if len(pairs) != sum(BINS_ALL.values()) or len(set(pairs)) != len(pairs):
        refuse(f"Stage 1 has {len(pairs)} rows ({len(set(pairs))} distinct pairs); "
               f"D106 records {sum(BINS_ALL.values())}")
    got = {b: sum(r["bin"] == b for r in stage1) for b in BINS_ALL}
    if got != BINS_ALL:
        refuse(f"Stage 1 bins {got} differ from D106's {BINS_ALL}")
    got = {b: sum(r["bin"] == b and float(r["s"]) == 0.005 for r in stage1) for b in BINS_005}
    if got != BINS_005:
        refuse(f"Stage 1 bins at s = 0.005 {got} differ from D106's {BINS_005}")
    for r in stage1:
        c = float(r["c_A"])
        want = "Identified" if c >= C_EDGE else "Partial" if c >= C_RANGE[0] else "Flat"
        if r["bin"] != want:
            refuse(f"pair {r['pair']} is binned {r['bin']} but c_A = {c} says {want}")


def check_cells(cells, by_pair):
    if len(cells) != 2 * SEEDS * len(by_pair):
        refuse(f"the cells file has {len(cells)} rows, not {2 * SEEDS * len(by_pair)}")
    if set(r["calibrator"] for r in cells) != set(CALIBRATORS):
        refuse(f"calibrators {sorted(set(r['calibrator'] for r in cells))} are not {CALIBRATORS}")
    for cal in CALIBRATORS:
        seeds = sorted(set(r["q"] for r in cells if r["calibrator"] == cal))
        if len(seeds) != SEEDS:
            refuse(f"{cal} has seeds {seeds}, not {SEEDS} of them")
        for q in seeds:
            got = [r["pair"] for r in cells if r["calibrator"] == cal and r["q"] == q]
            if sorted(got) != sorted(by_pair):
                refuse(f"{cal} seed {q} does not hold every pair exactly once")
    for r in cells:
        s1 = by_pair[r["pair"]]
        if r["bin"] != s1["bin"] or float(r["s"]) != float(s1["s"]):
            refuse(f"cell ({r['calibrator']}, q {r['q']}, {r['pair']}) has bin {r['bin']} and "
                   f"s {r['s']}; Stage 1 has {s1['bin']} and {s1['s']}")
        if r["failure"] != "":
            refuse(f"cell ({r['calibrator']}, q {r['q']}, {r['pair']}) carries failure "
                   f"{r['failure']!r}; D109 records none")


def style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.labelsize": 8, "axes.labelcolor": INK, "text.color": INK,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "legend.fontsize": 7, "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "none", "legend.facecolor": SURFACE, "legend.handletextpad": 0.5,
        "legend.borderpad": 0.4, "legend.labelspacing": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def proxy(color, marker, ms=4.5):
    from matplotlib.lines import Line2D
    return Line2D([], [], ls="", marker=marker, color=color, ms=ms)


def save(fig, out_dir, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.normpath(os.path.join(out_dir, f"{stem}.{ext}"))
        fig.savefig(p)
        paths.append(p)
    print(f"figure -> {paths[0]}")
    print(f"figure -> {paths[1]}")


def figure_fidelity(cells, cal_col, s_col, bin_col, c_A, abs_dA, okA, out_dir, plt):
    """Figure 1: |d| under prior A against c, per cell, Identified and Partial, all levels."""
    import matplotlib.transforms as mtx
    keep = bin_col != "Flat"
    if not np.all(okA[keep]):
        refuse(f"{np.sum(~okA[keep])} Identified/Partial cells have okA False")
    if not np.all(np.isfinite(abs_dA[keep])) or np.any(abs_dA[keep] < 0):
        refuse("abs_dA is missing or negative on an Identified/Partial cell")
    if (not np.all(np.isfinite(c_A[keep])) or np.any(c_A[keep] < C_RANGE[0])
            or np.any(c_A[keep] > C_RANGE[1])):
        refuse("an Identified/Partial cell has a missing c or one outside the drawn range "
               f"[{C_RANGE[0]}, {C_RANGE[1]}]: {np.nanmin(c_A[keep])} to {np.nanmax(c_A[keep])}")
    ymax = float(abs_dA[keep].max())
    top = 10 ** np.ceil(np.log10(ymax))         # a decade boundary at or above the largest |d|
    if top / ymax < 1.6:
        top *= 2.0                                # headroom for the bin names
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08)
    for ax, cal in zip(axes, CALIBRATORS):
        sel = keep & (cal_col == cal)
        if int(sel.sum()) != FIG1_CELLS:
            refuse(f"{cal}: {int(sel.sum())} Identified/Partial cells, not {FIG1_CELLS}")
        y = abs_dA[sel]
        n_floor = int(np.sum(y < FLOOR))
        y_draw = np.maximum(y, FLOOR)
        per_level = {}
        for s in LEVELS:
            m = s_col[sel] == s
            per_level[s] = int(m.sum())
            ax.scatter(c_A[sel][m], y_draw[m], s=4, c=NOISE_HUE[s], alpha=0.35,
                       linewidths=0, zorder=2)
        print(f"Figure 1, {TITLES[cal]}: {int(sel.sum()):,} cells plotted "
              f"(s = 0.001: {per_level[0.001]:,}; 0.003: {per_level[0.003]:,}; "
              f"0.005: {per_level[0.005]:,}); {n_floor} below the floor {FLOOR:g}, drawn at it; "
              f"largest |d| {y.max():.4f}")
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(*C_RANGE)
        ax.set_ylim(FLOOR, top)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, which="major")
        ax.grid(False, which="minor")
        ax.tick_params(which="minor", length=0)
        # H-a's bound and refutation threshold, labelled directly, in neutral ink.
        # The labels sit on a light surface box so they read where cells are dense.
        for yv, lab in ((HA_BOUND, f"H-a (Identified): median expected below {HA_BOUND}"),
                        (HA_REFUTE, f"H-a (Identified): median refuted above {HA_REFUTE}")):
            ax.axhline(yv, ls=DASH, lw=0.8, color=INK2, zorder=3)
            ax.text(C_RANGE[0] + 0.006, yv * 1.18, lab, ha="left", va="bottom",
                    fontsize=6.5, color=INK2, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
        # The bin edge, with the bins named.
        ax.axvline(C_EDGE, lw=0.8, color=INK2, zorder=3)
        tr = mtx.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(C_EDGE - 0.008, 0.985, "Partial", transform=tr, ha="right", va="top",
                fontsize=7, color=INK2, zorder=4)
        ax.text(C_EDGE + 0.008, 0.985, "Identified", transform=tr, ha="left", va="top",
                fontsize=7, color=INK2, zorder=4)
        if n_floor:
            exp = int(round(np.log10(FLOOR)))
            ax.text(C_RANGE[0] + 0.006, FLOOR * 1.25,
                    f"{n_floor} cells below $10^{{{exp}}}$ are drawn at the floor",
                    ha="left", va="bottom", fontsize=6.5, color=INK2, zorder=4)
        ax.set_title(TITLES[cal])
        ax.set_xlabel("contraction $c$ of $H$ under prior A")
        # The legend sits below the panel, so it covers no cell.
        ax.legend([proxy(NOISE_HUE[s], "o") for s in LEVELS],
                  [f"$s$ = {s}" for s in LEVELS], loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3, title="noise level",
                  title_fontsize=7, columnspacing=1.2)
    axes[0].set_ylabel("$|d|$ = $|\\hat{H}_A - $ exact posterior mean under A$|$")
    save(fig, out_dir, "p5_fig_fidelity")
    plt.close(fig)


def figure_excess(cells, cal_col, s_col, bin_col, H, E, okA, okB, out_dir, plt):
    """Figure 2: E against true H, per cell, Identified and Partial, s = 0.005 (exploratory)."""
    import matplotlib.transforms as mtx
    keep = (bin_col != "Flat") & (s_col == 0.005)
    if not (np.all(okA[keep]) and np.all(okB[keep])):
        refuse(f"{np.sum(~(okA[keep] & okB[keep]))} Identified/Partial cells at s = 0.005 "
               "have okA or okB False")
    if not np.all(np.isfinite(E[keep])):
        refuse(f"E is missing on {int(np.sum(~np.isfinite(E[keep])))} Identified/Partial cells "
               "at s = 0.005")
    if (not np.all(np.isfinite(H[keep])) or np.any(H[keep] < H_BOX[0])
            or np.any(H[keep] > H_BOX[1])):
        refuse(f"a plotted cell has a missing true H or one outside the box {H_BOX}: "
               f"{np.nanmin(H[keep])} to {np.nanmax(H[keep])}")
    for cal in CALIBRATORS:
        for s in LEVELS[:2]:
            m = (cal_col == cal) & (s_col == s)
            n_missing = int(np.sum(~np.isfinite(E[m])))
            print(f"note: {TITLES[cal]} at s = {s}: E missing on {n_missing:,} of {int(m.sum()):,} "
                  "cells (not plotted; Figure 2 is at s = 0.005)")
    lo, hi = float(E[keep].min()), float(E[keep].max())
    span = hi - lo
    ylim = (lo - 0.05 * span, hi + 0.16 * span)       # the top band holds the line labels
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.7), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08)
    for ax, cal in zip(axes, CALIBRATORS):
        sel = keep & (cal_col == cal)
        if int(sel.sum()) != FIG2_CELLS:
            refuse(f"{cal}: {int(sel.sum())} Identified/Partial cells at s = 0.005, not {FIG2_CELLS}")
        counts = {}
        for b in ("Identified", "Partial"):
            m = sel & (bin_col == b)
            counts[b] = int(m.sum())
            ax.scatter(H[m], E[m], s=9, c=BIN_HUE[b], marker=BIN_MARKER[b], alpha=0.5,
                       linewidths=0, zorder=2)
        print(f"Figure 2, {TITLES[cal]}: {int(sel.sum()):,} cells plotted "
              f"(Identified {counts['Identified']:,}; Partial {counts['Partial']:,}); "
              f"E from {E[sel].min():.4f} to {E[sel].max():.4f}")
        ax.axhline(0, lw=0.7, color=MUTED, zorder=1)
        tr = mtx.blended_transform_factory(ax.transData, ax.transAxes)
        for xv, lab, ha, dx in ((B_P01, f"B's 1st percentile\n$H$ = {B_P01}", "right", -0.006),
                                (B_MEAN, f"B's mean\n$H$ = {B_MEAN}", "left", 0.006)):
            ax.axvline(xv, ls=DASH, lw=0.8, color=INK2, zorder=3)
            ax.text(xv + dx, 0.985, lab, transform=tr, ha=ha, va="top", fontsize=6.5,
                    color=INK2, linespacing=1.15, zorder=4)
        ax.set_xlim(H_BOX[0] - 0.005, H_BOX[1] + 0.005)
        ax.set_xticks([H_BOX[0], 0.1, 0.2, 0.3, 0.4, H_BOX[1]])
        ax.set_ylim(*ylim)
        ax.set_title(TITLES[cal])
        ax.set_xlabel("true $H$")
        # The legend sits below the panel, so it covers no cell.
        ax.legend([proxy(BIN_HUE[b], BIN_MARKER[b]) for b in ("Identified", "Partial")],
                  [f"Identified ($c \\geq {C_EDGE}$)", f"Partial (${C_RANGE[0]} \\leq c < {C_EDGE}$)"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, title="bin",
                  title_fontsize=7, columnspacing=1.2)
    axes[0].set_ylabel("excess prior dependence $E = (d_B - d_A)\\,/\\,0.20$")
    fig.suptitle("Exploratory: per-cell $E$ at $s$ = 0.005; H-b's and H-c's thresholds apply to bin "
                 "medians, not cells",
                 fontsize=7.5, color=INK2, x=0.02, ha="left")
    save(fig, out_dir, "p5_fig_excess")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, help="output folder (default ../output)")
    ap.add_argument("--cells", default=os.path.join(OUT, "p5_stage2_cells.csv"),
                    help="Stage 2's cells CSV (default ../output/p5_stage2_cells.csv)")
    ap.add_argument("--stage1", default=os.path.join(OUT, "p5_stage1.csv"),
                    help="Stage 1's CSV (default ../output/p5_stage1.csv)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    stage1 = read_rows(args.stage1)
    cells = read_rows(args.cells)
    check_stage1(stage1)
    by_pair = {r["pair"]: r for r in stage1}
    check_cells(cells, by_pair)

    cal_col = np.array([r["calibrator"] for r in cells])
    s_col = floats(cells, "s")
    bin_col = np.array([r["bin"] for r in cells])
    okA = np.array([r["okA"] == "True" for r in cells])
    okB = np.array([r["okB"] == "True" for r in cells])
    abs_dA = floats(cells, "abs_dA")
    E = floats(cells, "E")
    c_A = np.array([float(by_pair[r["pair"]]["c_A"]) for r in cells])
    H = np.array([float(by_pair[r["pair"]]["H"]) for r in cells])

    os.makedirs(args.out, exist_ok=True)
    print(f"Stage 1: {len(stage1):,} pairs; bins {BINS_ALL}; at s = 0.005 {BINS_005}; "
          f"{SEEDS} seeds per calibrator; Flat cells not plotted "
          f"({BINS_ALL['Flat'] * SEEDS} per calibrator; H-d untestable)")
    figure_fidelity(cells, cal_col, s_col, bin_col, c_A, abs_dA, okA, args.out, plt)
    figure_excess(cells, cal_col, s_col, bin_col, H, E, okA, okB, args.out, plt)
    print("palette: noise level (one hue, light to dark) "
          + ", ".join(f"s = {s}: {NOISE_HUE[s]}" for s in LEVELS)
          + "; bins " + ", ".join(f"{b}: {BIN_HUE[b]} ({BIN_MARKER[b]})" for b in BIN_HUE)
          + f"; ink {INK} / {INK2} / {MUTED}; grid {GRID}; axis {AXIS}")


if __name__ == "__main__":
    main()
