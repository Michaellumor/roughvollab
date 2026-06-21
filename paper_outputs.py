r"""
RoughVolLab — paper_outputs.py
==============================
Regenerate every figure and number for the P3 paper from one command, so the
manuscript's quantitative claims are never hand-typed and always trace to a run.

What it produces (into --outdir, default ./output):
  fig1_bias_curves.png        estimator bias curves vs true H, with the smooth null
  fig2_map_with_assets.png    the identifiability map with BTC/ETH overlaid
  a PAPER NUMBERS block printed to the terminal (paste it back verbatim)

Usage (Windows PowerShell, from the repo root):
  python paper_outputs.py `
    --btc data\processed\btc_rv.csv --eth data\processed\eth_rv.csv `
    --spx data\processed\spx_rv.csv --window 288 --spx-window 1 `
    --eta 0.5,1.5,2.5,3.5 --map-windows 48,96,288 --n-obs 2500 --n-mc 40

Notes
-----
- The map build is the slow part (one bias curve per η×Δ cell). 48,96,288 is a
  tractable default; add 1440 (1-minute) only if you can spare the time.
- SPX uses the Garman–Klass *range* proxy, which is not an intraday-RV sum, so
  its placement is reported separately with a caveat — it is suggestive, not a
  clean point on the crypto map.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

from interpret_h import ESTIMATORS, build_bias_curve
from estimate_h import load_log_rv_csv
from identifiability_map import (
    build_identifiability_map, plot_identifiability_map, place_asset,
    _SMOOTH, TEAL, PURPLE, CORAL, GRAY,
)

_CURVE_COLOR = {"GJR": TEAL, "Cont-Das": PURPLE, "MF-DFA": CORAL}


def fig1_bias_curves(out: str, *, eta: float, window: int, n_obs: int,
                     n_mc: int) -> None:
    """Figure 1: E[Ĥ] vs true H for the three estimators, with the smooth null."""
    grid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60])
    curve = build_bias_curve(grid, n_obs=n_obs, window=window, n_mc=n_mc, eta=eta)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(grid, grid, "--", color=GRAY, lw=1.0, label="unbiased (Ĥ = H)")
    for name in ESTIMATORS:
        m = np.asarray(curve.mean[name]); s = np.asarray(curve.std[name])
        ax.plot(grid, m, "-o", color=_CURVE_COLOR[name], ms=3, lw=1.5, label=name)
        ax.fill_between(grid, m - s, m + s, color=_CURVE_COLOR[name], alpha=0.12)
    ax.axhline(_SMOOTH, color="black", lw=0.8, ls=":", label="smooth null (H = ½)")
    ax.set_xlabel("true H"); ax.set_ylabel("estimated Ĥ")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  figure → {out}")


def _map_fractions(imap) -> dict:
    cells = imap.eta_grid.size * imap.window_grid.size * imap.true_grid.size
    out = {}
    for name in ESTIMATORS:
        flat = [imap.status[name][a][b][i]
                for a in range(imap.eta_grid.size)
                for b in range(imap.window_grid.size)
                for i in range(imap.true_grid.size)]
        out[name] = {s: flat.count(s) / cells for s in set(flat)}
    return out


def _n_obs_of(path: str):
    try:
        return int(load_log_rv_csv(path)[0].size)
    except Exception:
        return None


def _place(label, path, window, n_mc):
    if not Path(path).exists():
        print(f"  ! {label}: {path} not found — skipping")
        return None
    try:
        return place_asset(label, path, window, n_mc=n_mc)
    except Exception as e:  # never let one asset abort the whole run
        print(f"  ! {label}: placement failed ({type(e).__name__}: {e})")
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Regenerate P3 figures + numbers.")
    p.add_argument("--btc", default="data/processed/btc_rv.csv")
    p.add_argument("--eth", default="data/processed/eth_rv.csv")
    p.add_argument("--spx", default="data/processed/spx_rv.csv")
    p.add_argument("--window", type=int, default=288, help="crypto proxy window")
    p.add_argument("--spx-window", type=int, default=1, help="equity range proxy ≈ 1")
    p.add_argument("--eta", default="0.5,1.5,2.5,3.5")
    p.add_argument("--map-windows", default="48,96,288")
    p.add_argument("--n-obs", type=int, default=2500)
    p.add_argument("--n-mc", type=int, default=40)
    p.add_argument("--outdir", default="output")
    args = p.parse_args(argv)

    eta = np.array([float(x) for x in args.eta.split(",")])
    mwin = np.array([int(x) for x in args.map_windows.split(",")])
    od = Path(args.outdir)

    print("Figure 1 — estimator bias curves")
    fig1_bias_curves(str(od / "fig1_bias_curves.png"), eta=1.5,
                     window=args.window, n_obs=args.n_obs, n_mc=args.n_mc)

    print("\nFigure 2 — identifiability map (this is the slow step)")
    imap = build_identifiability_map(eta, mwin, n_obs=args.n_obs, n_mc=args.n_mc)

    crypto = [pl for pl in (_place("BTC", args.btc, args.window, args.n_mc),
                            _place("ETH", args.eth, args.window, args.n_mc)) if pl]
    eta_ref = float(np.mean([pl.eta_hat for pl in crypto])) if crypto else None
    plot_identifiability_map(
        imap, out=str(od / "fig2_map_with_assets.png"), show=False,
        eta_reference=eta_ref,
        eta_reference_label=(f"crypto η̂ ≈ {eta_ref:.2f} → non-identified"
                             if eta_ref is not None else None),
        title=None)

    spx = _place("SPX", args.spx, args.spx_window, args.n_mc)

    # ───────────────────────── numbers block ─────────────────────────
    paths = {"BTC": args.btc, "ETH": args.eth, "SPX": args.spx}
    print("\n" + "=" * 66)
    print("PAPER NUMBERS  —  paste this whole block back")
    print("=" * 66)
    print(f"map grid : eta={eta.tolist()}  Delta={mwin.tolist()}  "
          f"n_obs={args.n_obs}  n_mc={args.n_mc}")
    print("\n[Map identifiable fractions]")
    fr = _map_fractions(imap)
    for name in ESTIMATORS:
        print(f"  {name:<9} " + ", ".join(
            f"{s} {fr[name][s]:.0%}" for s in sorted(fr[name], key=lambda x: -fr[name][x])))
    print("\n[Per-asset]")
    for pl in crypto + ([spx] if spx else []):
        n = _n_obs_of(paths[pl.label])
        oh = ", ".join(f"{k}={pl.observed_H[k]:.3f}"
                       if np.isfinite(pl.observed_H[k]) else f"{k}=nan"
                       for k in ESTIMATORS)
        st = ", ".join(f"{k}={pl.status[k]}" for k in ESTIMATORS)
        print(f"  {pl.label}: n={n}, window={pl.window}, eta_hat={pl.eta_hat:.2f}")
        print(f"        observed H : {oh}")
        print(f"        status     : {st}")
    print("=" * 66)
    print("(SPX is the range-proxy caveat case — report separately.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
