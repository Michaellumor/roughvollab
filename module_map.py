#!/usr/bin/env python3
"""RoughVolLab x BSc Mathematics (Salford) -- module-to-layer map.
Regenerable flagship: encodes the CURRENT build statuses (all layers built --
Layer 3 = D40 deep hedging, Layer 4 = D31-D39 convergence + calibration).
Regenerate after status changes: `python module_map.py` -> roughvollab_module_map.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

# ---- palette (RoughVolLab brand; year colour-coding) ----
GREEN  = "#1D9E75"   # Year 1
PURPLE = "#7F77DD"   # Year 2
ORANGE = "#D85A30"   # Year 3
GRAY   = "#8C8B86"   # Beyond syllabus
YCOL = {"y1": GREEN, "y2": PURPLE, "y3": ORANGE, "g": GRAY}

# status pill tints (bg, fg)
ST = {
    "done": ("#d3ece0", "#15795a"),
    "prog": ("#f4e6c6", "#8a6212"),
    "spec": ("#e6e6e3", "#6f6e6a"),
}
ROW_BORDER = "#d9d9d6"
BANNER_BG  = "#efeefa"
OUT_BG     = "#ead7b6"
OUT_BORDER = "#d8c096"
INK        = "#222222"
SUB        = "#6b6b67"

# ---- canvas ----
W, H = 1650, 1509
FIGW_IN = 16.5
fig = plt.figure(figsize=(FIGW_IN, FIGW_IN*H/W), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.canvas.draw()
RND = fig.canvas.get_renderer()
DATA_PER_PT = (W / FIGW_IN) / 72.0   # data units per typographic point

def tw(s, fs, weight="normal"):
    """measured text width in data units"""
    t = ax.text(0, -1000, s, fontsize=fs, fontweight=weight)
    bb = t.get_window_extent(renderer=RND)
    inv = ax.transData.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0)); x1, _ = inv.transform((bb.x1, bb.y1))
    t.remove()
    return abs(x1 - x0)

def pill(x, ycen, label, fill, fg, fs=12.5, weight="bold", padx=12, pady=7.5, gap=11):
    w = tw(label, fs, weight)
    px = padx * DATA_PER_PT * 0 + padx      # padx already in data units
    pw = w + 2 * padx
    ph = (fs * 1.4) * DATA_PER_PT + 2 * pady
    box = FancyBboxPatch((x, ycen - ph/2), pw, ph,
                         boxstyle=f"round,pad=0,rounding_size={ph/2}",
                         linewidth=0, facecolor=fill, zorder=4,
                         mutation_aspect=1)
    box.set_mutation_scale(1.0/DATA_PER_PT)  # make rounding_size honour data units
    ax.add_patch(box)
    ax.text(x + pw/2, ycen, label, fontsize=fs, color=fg, fontweight=weight,
            ha="center", va="center", zorder=5)
    return x + pw + gap

def module_lines(mods, x0, xlim, fs=12.5, gap=11, lh=44):
    """simulate wrap; return list of (label,color,x,row) and n_rows"""
    placed = []; x = x0; row = 0
    for label, ck in mods:
        w = tw(label, fs, "bold") + 2*12
        if x + w > xlim and x > x0:
            row += 1; x = x0
        placed.append((label, YCOL[ck], x, row)); x += w + gap
    return placed, row + 1

# ================= CONTENT (current / pre-update) =================
TITLE = "RoughVolLab × BSc Mathematics (Salford) — module-to-layer map"
LEGEND = [("Year 1 — complete", GREEN), ("Year 2 — from Sept 2026", PURPLE),
          ("Year 3", ORANGE), ("Beyond syllabus", GRAY)]
BANNER = ("Year 2 · Business & Industrial Mathematics spans every layer — "
          "RoughVolLab is the industrial case study (reports + presentations)")

LAYERS = [
    ("roughvol_core.py — tested foundation",
     "κ=0 Volterra engine + 18 pytest guards (shared by all layers)",
     "validated", "done",
     [("Probability","y1"),("Numerical Analysis","y2"),("Software testing","g")]),
    ("Layer 1 — Simulation core",
     "fBm, hybrid scheme, rough Bergomi/Heston paths, Hurst estimation",
     "complete", "done",
     [("Probability","y1"),("Analysis","y1"),("Linear Algebra","y1"),
      ("Numerical Analysis","y2"),("Statistics","y1")]),
    ("Layer 1b — MLMC pricing",
     "coupled paths, Giles rates, conditional + κ=1 estimators (P2 concluded)",
     "v0.1 + P2 done", "done",
     [("Probability","y1"),("Numerical Analysis","y2"),("Statistics","y2"),
      ("Mathematical Statistics","y3")]),
    ("Layer 1c — Roughness-estimator audit",
     "GJR + Cont–Das + MF-DFA, full corruption ladder, identifiability map (Phase B done)",
     "complete, tested", "done",
     [("Statistics","y2"),("Mathematical Statistics","y3"),("Numerical Analysis","y2"),
      ("Programming & Optimisation","y3")]),
    ("Layer 2 — Market frictions / execution",
     "Almgren–Chriss execution, causal vol-timing probe — no edge (kill-switch fired)",
     "execution built — negative", "done",
     [("Mathematical Methods 2","y2"),("Dynamical Systems","y2"),
      ("Operational Research","y3"),("Programming & Optimisation","y3")]),
    ("Layer 3 — Deep-hedging engine",
     "Buehler-style direct policy opt + CVaR + signatures — no roughness edge beyond frictions",
     "built — D40", "done",
     [("Programming & Optimisation","y3"),("Operational Research","y3"),
      ("Mathematical Statistics","y3"),("ML + rough paths (self-study)","g")]),
    ("Layer 4 — Convergence & calibration",
     "Markovian lift (O(N·n)), high-ν pricing, weak-order study, calibration → live Deribit BTC",
     "built — D31–D39", "done",
     [("Mathematical Methods 3","y3"),("Mathematical Statistics","y3"),
      ("Statistics","y2"),("Final-year Project","y3")]),
]
OUTPUTS = [
    ("Tested, validated code", "252 pytest tests + gate-checks"),
    ("arXiv notes (P2, P3)", "+ Zenodo DOI at v0.2"),
    ("JOSS software paper", "peer-reviewed, citable"),
    ("PhD applications", "Oxford · Imperial · Manchester"),
]

# ================= DRAW =================
LM, RM = 22, W - 22
y = 30

# title
ax.text(LM+8, y+8, TITLE, fontsize=23, fontweight="bold", color=INK, va="center")
y += 44

# legend
lx = LM + 8
for label, col in LEGEND:
    ax.add_patch(FancyBboxPatch((lx, y-9), 18, 18, boxstyle="round,pad=0,rounding_size=4",
                                facecolor=col, linewidth=0))
    ax.text(lx+26, y, label, fontsize=12, color=INK, va="center")
    lx += 26 + tw(label, 12) + 40
y += 28

# banner
bh = 40
ax.add_patch(FancyBboxPatch((LM, y-bh/2+6), RM-LM, bh, boxstyle="round,pad=0,rounding_size=8",
                            facecolor=BANNER_BG, linewidth=0))
ax.text(LM+18, y+6, BANNER, fontsize=12.5, color="#3a3a55", va="center")
y += 44

MODX = 535          # module-pill column start
MODXLIM = RM - 8

for title, desc, status, skind, mods in LAYERS:
    placed, nrows = module_lines(mods, MODX, MODXLIM)
    left_h = 96
    mod_h = 16 + nrows*44 + 8
    row_h = max(left_h, mod_h)
    # row box
    ax.add_patch(FancyBboxPatch((LM, y), RM-LM, row_h, boxstyle="round,pad=0,rounding_size=10",
                                facecolor="white", edgecolor=ROW_BORDER, linewidth=1.4))
    # left text
    ax.text(LM+24, y+26, title, fontsize=16, fontweight="bold", color=INK, va="center")
    ax.text(LM+24, y+50, desc, fontsize=12, color=SUB, va="center")
    bg, fg = ST[skind]
    pill(LM+24, y+78, status, bg, fg, fs=12, padx=12, pady=6)
    # module pills
    for label, col, mx, mrow in placed:
        pill(mx, y + 26 + mrow*44, label, col, "white", fs=12.5)
    y += row_h + 14

# outputs pathway
y += 8
ax.text(LM+8, y+6, "Outputs pathway", fontsize=15, fontweight="bold", color=INK, va="center")
y += 30
n = len(OUTPUTS); gap = 46
bw = (RM - LM - (n-1)*gap) / n
bh2 = 80
for i, (t1, t2) in enumerate(OUTPUTS):
    bx = LM + i*(bw+gap)
    ax.add_patch(FancyBboxPatch((bx, y), bw, bh2, boxstyle="round,pad=0,rounding_size=9",
                                facecolor=OUT_BG, edgecolor=OUT_BORDER, linewidth=1.3))
    ax.text(bx+bw/2, y+30, t1, fontsize=13, fontweight="bold", color="#5a4a2a", ha="center", va="center")
    ax.text(bx+bw/2, y+54, t2, fontsize=11, color="#6a5a3a", ha="center", va="center")
    if i < n-1:
        ax.annotate("", xy=(bx+bw+gap-8, y+bh2/2), xytext=(bx+bw+8, y+bh2/2),
                    arrowprops=dict(arrowstyle="->", color="#9a8a6a", lw=1.6))

fig.savefig("roughvollab_module_map.png", dpi=150, facecolor="white", bbox_inches=None)
print("saved", flush=True)
