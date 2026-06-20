"""
RoughVolLab — identifiability_map.py
====================================
Layer 1c capstone / publication seed P3.

Where is the roughness exponent H *recoverable at all* from realized-variance
data, and where do the major asset classes actually sit on that map?

Why this module exists
----------------------
Phase B produced a *point* result: at data-calibrated vol-of-vol η, the
roughness reading for BTC / ETH / SPX is NON-IDENTIFIED. A point invites the
rebuttal "you picked bad settings". This module answers the rebuttal by
characterising the whole identifiable *region* of the inverse problem
true_H -> E[observed Ĥ], as a function of (η, proxy window Δ), for each of the
three audited estimators — then leaves a hook to drop a real asset onto it.

That reframes the fact-or-artefact debate as one ill-posed inverse problem:
Gatheral–Jaisson–Rosenbaum ("rough", H ≈ 0.1) and Cont–Das ("artefact") become
two readings of a map whose identifiable region the data may simply fall
outside of. The map is the contribution; the point estimate was the seed.

This is a *new file only* — it reuses, rather than re-implements, the audited
primitives in interpret_h (`build_bias_curve`, `_is_monotone`, `_local_slope`,
`invert`, `_classify_inversion`) so the map and the Phase-B interpreter share a
single definition of monotonicity, local slope and inversion. Nothing tested is
touched.

Public API
----------
    cell_status(true_grid, mean, std, i, ...)      step 1: identifiability of one true H
    build_identifiability_map(...)                 step 2: factorial (η × Δ) sweep
    plot_identifiability_map(imap, ...)            step 3: the phase diagram
    locate_observed(observed_H, curve, name, ...)  step 4 hook: place a real asset

CLI:  python identifiability_map.py [--quick] [--no-show] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Headless-safe before pyplot is imported (sandbox / CI).
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Audited primitives — the single source of truth, reused deliberately.
from interpret_h import (
    ESTIMATORS,
    BiasCurve,
    build_bias_curve,
    _classify_inversion,
    _is_monotone,
    _local_slope,
    _FLAT_SLOPE,
)

__all__ = [
    "IDENTIFIED",
    "DEBIASABLE",
    "NON_IDENTIFIED",
    "BELOW_FLOOR",
    "ABOVE_CEILING",
    "UNCALIBRATED",
    "cell_status",
    "IdentifiabilityMap",
    "build_identifiability_map",
    "plot_identifiability_map",
    "locate_observed",
]

# ── project palette ────────────────────────────────────────────────────────
TEAL, PURPLE, CORAL, GRAY, AMBER = "#1D9E75", "#7F77DD", "#D85A30", "#888780", "#BA7517"

# ── status vocabulary (compatible with interpret_h._classify_inversion) ─────
# 'ok' there splits here into IDENTIFIED vs DEBIASABLE on the smooth-exclusion
# test; 'multivalued' maps to NON_IDENTIFIED. The rest carry through unchanged.
IDENTIFIED = "identified"          # invertible AND CI excludes the smooth boundary
DEBIASABLE = "de-biasable"         # point-invertible, but CI cannot exclude H = 1/2
NON_IDENTIFIED = "non-identified"  # multivalued (hump) or collapsed (flat slope)
BELOW_FLOOR = "below-floor"        # observed rougher than the model makes at any H
ABOVE_CEILING = "above-ceiling"    # observed smoother than the model's ceiling
UNCALIBRATED = "uncalibrated"      # estimator returned NaN

# Plot ordering: best → worst, so the colour bar reads as a quality scale.
_STATUS_ORDER = [IDENTIFIED, DEBIASABLE, NON_IDENTIFIED, BELOW_FLOOR,
                 ABOVE_CEILING, UNCALIBRATED]
_STATUS_COLOR = {IDENTIFIED: TEAL, DEBIASABLE: AMBER, NON_IDENTIFIED: CORAL,
                 BELOW_FLOOR: "#6E2C12", ABOVE_CEILING: PURPLE, UNCALIBRATED: GRAY}
_STATUS_CODE = {s: i for i, s in enumerate(_STATUS_ORDER)}

# ── default sweep grids ─────────────────────────────────────────────────────
# True-H grid: dense in the rough zone where the debate sits (mirrors interpret_h),
# with 0.60 included so the smooth null E[Ĥ | H = 1/2] is on the curve (interpolable).
MAP_TRUE_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60])
# Vol-of-vol: spans the assumed-small (0.5) regime up through the data-calibrated
# crypto regime (η ≳ 1.5 from Phase B). The whole verdict turns on this axis.
DEFAULT_ETA_GRID = np.array([0.5, 1.0, 1.5, 2.0])
# Proxy window Δ = RV observations per estimate (≈ 30m, 15m, 5m, 1m at daily obs).
DEFAULT_WINDOW_GRID = np.array([48, 96, 288, 1440])

_SMOOTH = 0.5  # the smooth (semimartingale) boundary H = 1/2


# ───────────────────────────────────────────────────────────────────────────
# step 1 — operational identifiability of a single true H
# ───────────────────────────────────────────────────────────────────────────
def cell_status(true_grid: Sequence[float],
                mean_curve: Sequence[float],
                std_curve: Sequence[float],
                i: int,
                *,
                z: float = 1.96,
                slope_floor: float = _FLAT_SLOPE,
                smooth: float = _SMOOTH) -> str:
    """Is true_H = true_grid[i] identifiable under this estimator + measurement config?

    The bias curve true_H -> E[observed Ĥ] (with single-sample spread σ) defines
    an inverse problem. true_grid[i] is:

    NON_IDENTIFIED  if the curve is non-monotone (a rough and a smooth true H give
                    the same observed Ĥ — observational equivalence), or if its
                    local slope is below `slope_floor` (a flat segment maps many
                    true H onto one observed Ĥ — the collapse);
    IDENTIFIED      if monotone and well-posed AND this true H is rough (< smooth)
                    AND its observed Ĥ differs from the smooth null's expected Ĥ
                    (= E[Ĥ | H = 1/2], read off the same curve) by more than the
                    ±zσ single-sample band — so the data could conclude "rough"
                    with confidence;
    DEBIASABLE      a non-rough control cell (H ≥ 1/2), or a rough cell whose band
                    cannot separate it from the smooth null (estimable, but the
                    smooth null is not excluded);
    UNCALIBRATED    if the estimator returned NaN here.

    Reuses `_is_monotone` and `_local_slope` so this definition matches the one
    the Phase-B interpreter applies to real data.
    """
    g = np.asarray(true_grid, float)
    m = np.asarray(mean_curve, float)
    s = np.asarray(std_curve, float)
    if i >= m.size or not np.isfinite(m[i]):
        return UNCALIBRATED

    # (1) global injectivity — a hump curve is non-identified *everywhere* on it
    if not _is_monotone(m):
        return NON_IDENTIFIED

    # (2) local well-posedness — flat slope ⇒ ill-posed inverse (the collapse)
    slope = _local_slope(g, m, g[i])
    if not np.isfinite(slope) or abs(slope) < slope_floor:
        return NON_IDENTIFIED

    # (3) can the data conclude "rough" (H < 1/2) here? Compare this cell's
    #     observed Ĥ to the smooth null's expected Ĥ = E[Ĥ | true H = 1/2],
    #     read off the SAME curve. If they differ by more than the ±zσ
    #     single-sample band, the smooth null is excluded → roughness identified.
    if g[i] >= smooth:
        return DEBIASABLE                    # not a rough truth — control cell
    fin = np.isfinite(g) & np.isfinite(m)
    if fin.sum() < 2:
        return DEBIASABLE
    m_smooth = float(np.interp(smooth, g[fin], m[fin]))   # grid reaches H = 1/2
    sigma = float(s[i]) if (i < s.size and np.isfinite(s[i])) else 0.0
    return IDENTIFIED if abs(m[i] - m_smooth) > z * sigma else DEBIASABLE


# ───────────────────────────────────────────────────────────────────────────
# step 2 — factorial sweep over (η, Δ)
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class IdentifiabilityMap:
    """Result of the (η × Δ) sweep.

    `status[name]` is an (n_eta, n_window) nested list; each entry is a list of
    status strings, one per true_H in `true_grid`. `curves[(a, b)]` keeps the
    underlying BiasCurve so a real asset can later be located against it.
    """
    true_grid: np.ndarray
    eta_grid: np.ndarray
    window_grid: np.ndarray
    status: dict
    curves: dict = field(repr=False)
    meta: dict = field(default_factory=dict)


def build_identifiability_map(eta_grid: Sequence[float] = DEFAULT_ETA_GRID,
                              window_grid: Sequence[float] = DEFAULT_WINDOW_GRID,
                              true_grid: Sequence[float] = MAP_TRUE_GRID,
                              n_obs: int = 2500,
                              n_mc: int = 40,
                              *,
                              z: float = 1.96,
                              slope_floor: float = _FLAT_SLOPE,
                              smooth: float = _SMOOTH,
                              seed: int = 505,
                              progress: bool = True) -> IdentifiabilityMap:
    """Build the identifiability phase map by sweeping the bias curve over (η, Δ).

    For each (η, window) we simulate the rough-Bergomi bias curve over `true_grid`
    (single-path Ĥ spread, exactly as Phase B does), then classify every true_H
    for every estimator with `cell_status`. Heavy: n_eta·n_window bias curves,
    each n_mc single-path simulations per grid point. Use --quick / small grids
    in the sandbox; run the full grid on the workstation.
    """
    eta_grid = np.asarray(eta_grid, float)
    window_grid = np.asarray(window_grid, int)
    true_grid = np.asarray(true_grid, float)

    status = {name: [[None] * len(window_grid) for _ in eta_grid]
              for name in ESTIMATORS}
    curves: dict = {}

    total = len(eta_grid) * len(window_grid)
    k = 0
    for a, eta in enumerate(eta_grid):
        for b, win in enumerate(window_grid):
            k += 1
            if progress:
                print(f"  [{k:>2}/{total}] η={eta:<4} window={int(win):<5} "
                      f"(n_obs={n_obs}, n_mc={n_mc}) …", flush=True)
            curve = build_bias_curve(true_grid, n_obs=n_obs, window=int(win),
                                     n_mc=n_mc, eta=float(eta), seed=seed)
            curves[(a, b)] = curve
            for name in ESTIMATORS:
                status[name][a][b] = [
                    cell_status(curve.true_grid, curve.mean[name],
                                curve.std[name], i, z=z,
                                slope_floor=slope_floor, smooth=smooth)
                    for i in range(true_grid.size)
                ]

    return IdentifiabilityMap(
        true_grid=true_grid, eta_grid=eta_grid, window_grid=window_grid,
        status=status, curves=curves,
        meta=dict(n_obs=n_obs, n_mc=n_mc, z=z, slope_floor=slope_floor,
                  smooth=smooth, seed=seed),
    )


# ───────────────────────────────────────────────────────────────────────────
# step 3 — the phase diagram
# ───────────────────────────────────────────────────────────────────────────
def plot_identifiability_map(imap: IdentifiabilityMap,
                             out: Optional[str] = "output/identifiability_map.png",
                             show: bool = True):
    """Render the map: rows = estimators, cols = Δ; each panel x = true H, y = η,
    cell colour = identifiability status. The teal region is where roughness is
    actually recoverable; coral is where rough ≡ smooth."""
    names = list(ESTIMATORS)
    nrow, ncol = len(names), imap.window_grid.size
    cmap = ListedColormap([_STATUS_COLOR[s] for s in _STATUS_ORDER])

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol + 1.2, 2.7 * nrow + 0.6),
                             squeeze=False)
    H_lab = [f"{h:g}" for h in imap.true_grid]
    eta_lab = [f"{e:g}" for e in imap.eta_grid]

    for r, name in enumerate(names):
        for c in range(ncol):
            ax = axes[r][c]
            grid = np.array([[_STATUS_CODE[imap.status[name][a][c][i]]
                              for i in range(imap.true_grid.size)]
                             for a in range(imap.eta_grid.size)])
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(_STATUS_ORDER) - 1,
                      aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xticks(range(imap.true_grid.size)); ax.set_xticklabels(H_lab, fontsize=8)
            ax.set_yticks(range(imap.eta_grid.size)); ax.set_yticklabels(eta_lab, fontsize=8)
            if r == 0:
                ax.set_title(f"Δ = {int(imap.window_grid[c])} obs", fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel("true H", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\nη (vol-of-vol)", fontsize=9)

    present = sorted({imap.status[n][a][b][i]
                      for n in names for a in range(imap.eta_grid.size)
                      for b in range(ncol) for i in range(imap.true_grid.size)},
                     key=lambda s: _STATUS_CODE.get(s, 99))
    handles = [Patch(facecolor=_STATUS_COLOR[s], label=s) for s in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Identifiability map of the roughness exponent "
                 "(rough-Bergomi ground truth)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  figure → {out}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ───────────────────────────────────────────────────────────────────────────
# step 4 hook — locate a real asset on the map (entry point, not the full leg)
# ───────────────────────────────────────────────────────────────────────────
def locate_observed(observed_H: float, curve: BiasCurve, name: str):
    """Where does a real asset's observed Ĥ land against one config's curve?

    Returns (status, candidate_true_H_list) using the audited classifier. This
    is the bridge to step 4 (calibrate η per asset, then drop BTC/ETH/SPX onto
    the panel): feed the asset's Ĥ and the curve at its calibrated (η, Δ).
    """
    inv_status, sols = _classify_inversion(observed_H, curve.true_grid,
                                           curve.mean[name])
    mapping = {"ok": IDENTIFIED, "multivalued": NON_IDENTIFIED,
               "below_floor": BELOW_FLOOR, "above_ceiling": ABOVE_CEILING,
               "uncalibrated": UNCALIBRATED}
    return mapping.get(inv_status, inv_status), sols


# ───────────────────────────────────────────────────────────────────────────
# text summary + CLI
# ───────────────────────────────────────────────────────────────────────────
def _summarise(imap: IdentifiabilityMap, stream=sys.stdout) -> None:
    def line(s: str = "") -> None:
        print(s, file=stream)

    cells = imap.eta_grid.size * imap.window_grid.size * imap.true_grid.size
    line()
    line("Identifiability summary (fraction of η × Δ × H cells)")
    line("─" * 56)
    for name in ESTIMATORS:
        flat = [imap.status[name][a][b][i]
                for a in range(imap.eta_grid.size)
                for b in range(imap.window_grid.size)
                for i in range(imap.true_grid.size)]
        frac = {s: flat.count(s) / cells for s in set(flat)}
        parts = ", ".join(f"{s} {frac[s]:.0%}"
                          for s in sorted(frac, key=lambda x: -frac[x]))
        line(f"  {name:<9} {parts}")
    line("─" * 56)
    line("Teal/identified = roughness recoverable; coral/non-identified "
         "= rough ≡ smooth.")
    line("Next (step 4): calibrate η per asset, then locate_observed() drops "
         "BTC/ETH/SPX onto the panel.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Identifiability phase map of the "
                                            "roughness exponent (Layer 1c / P3).")
    p.add_argument("--quick", action="store_true",
                   help="tiny grid + few MC paths for a fast smoke run")
    p.add_argument("--no-show", action="store_true", help="save figure, don't display")
    p.add_argument("--out", default="output/identifiability_map.png")
    p.add_argument("--eta", help="comma list of η, e.g. 0.5,1.0,1.5,2.0")
    p.add_argument("--windows", help="comma list of Δ windows, e.g. 48,96,288")
    p.add_argument("--n-obs", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    args = p.parse_args(argv)

    if args.quick:
        eta_grid, window_grid = np.array([0.5, 1.7]), np.array([24, 96])
        true_grid = np.array([0.05, 0.10, 0.20, 0.45, 0.60])
        n_obs, n_mc = args.n_obs or 250, args.n_mc or 8
    else:
        eta_grid, window_grid, true_grid = DEFAULT_ETA_GRID, DEFAULT_WINDOW_GRID, MAP_TRUE_GRID
        n_obs, n_mc = args.n_obs or 2500, args.n_mc or 40
    if args.eta:
        eta_grid = np.array([float(x) for x in args.eta.split(",")])
    if args.windows:
        window_grid = np.array([int(x) for x in args.windows.split(",")])

    print(f"Building identifiability map: η={eta_grid.tolist()}  "
          f"Δ={window_grid.tolist()}  |H|={true_grid.size}  "
          f"(n_obs={n_obs}, n_mc={n_mc})")
    imap = build_identifiability_map(eta_grid, window_grid, true_grid,
                                     n_obs=n_obs, n_mc=n_mc)
    _summarise(imap)
    plot_identifiability_map(imap, out=args.out, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
