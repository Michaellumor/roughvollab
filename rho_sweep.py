"""
rho_sweep.py — sweep the rough-Bergomi correlation rho and track the G-C4
headline cost ratio cond-stdMC / cond-MLMC as rho -> -1.

For each rho it deep-copies PARAMS, sets p["rho"], and re-runs the (now
p-parameterised) conditional-MC gate p2.gate_c4(quick, p=p). The eps=0.10 row
(L* from the naive bias test) supplies the four matched-accuracy costs; we take
the seed-averaged conditional standard-MC cost c_cs and conditional-MLMC cost
c_cm, their ratio, and the single-level / level-diff variance-reduction factors
from p2.variance_factors.

The question: does conditional standard-MC keep beating conditional MLMC
(ratio < 1) across the realistic rho in [-0.9, -0.7], or does the ratio cross 1
somewhere as the geometric control variate captures ever more of the variance?

Run:  python rho_sweep.py            (full)
      python rho_sweep.py --quick
"""
import argparse
import contextlib
import copy
import csv
import io
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layer1b_mlmc_asian import PARAMS
import p2_conditional_verify as p2

RHOS = [-0.5, -0.6, -0.7, -0.8, -0.9, -0.95, -0.99]
EPS_INDEX = 0          # rows[0] = eps=0.10 (the G-C4 headline regime)
REALISTIC = (-0.9, -0.7)


def run_one(rho, quick):
    """One rho: seed-averaged c_cs, c_cm, ratio + deep-level variance factors."""
    p = copy.deepcopy(PARAMS)
    p["rho"] = rho
    rows, res, seeds, kappa = p2.gate_c4(quick=quick, p=p)
    row = rows[EPS_INDEX]
    per = row["per_seed"]
    c_cs = float(np.mean([ps["c_cs"] for ps in per]))     # conditional standard MC
    c_cm = float(np.mean([ps["c_cm"] for ps in per]))     # conditional MLMC
    ratio = c_cs / c_cm
    sl, ld = p2.variance_factors(res, seeds)              # single-level / level-diff
    sl_deep = float(sl[-3:].mean())
    ld_deep = float(ld[-3:].mean())
    winner = "cond-stdMC" if ratio < 1.0 else "cond-MLMC"
    return dict(rho=rho, eps=row["eps"], Lstar=row["Lstar"], c_cs=c_cs, c_cm=c_cm,
                ratio=ratio, sl=sl_deep, ld=ld_deep, kappa=kappa, winner=winner)


def crossing(rhos, ratios):
    """Linear-interpolate the rho where ratio crosses 1, if it does. rhos are in
    sweep order (increasing |rho|); returns the first crossing or None."""
    for i in range(len(rhos) - 1):
        a, b = ratios[i] - 1.0, ratios[i + 1] - 1.0
        if a == 0.0:
            return rhos[i]
        if a * b < 0.0:                                   # sign change in [i, i+1]
            t = a / (a - b)                               # fraction to the root
            return rhos[i] + t * (rhos[i + 1] - rhos[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("\n" + "=" * 78)
    print(f"  RHO SWEEP — conditional std-MC vs MLMC across correlation  "
          f"[{'QUICK' if args.quick else 'FULL'}]")
    print(f"  rho in {RHOS}")
    print("=" * 78)

    results = []
    for rho in RHOS:
        # p2.gate_c4 / variance_factors print a lot; keep the sweep readable by
        # capturing their stdout, but surface it if the call raises.
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = run_one(rho, args.quick)
        except Exception:
            print(buf.getvalue())
            raise
        results.append(r)
        print(f"  rho={rho:6.2f}  ratio={r['ratio']:.3f}  "
              f"(c_cs={r['c_cs']:.3g}, c_cm={r['c_cm']:.3g}, "
              f"sl={r['sl']:.2f}, ld={r['ld']:.2f})  ->  {r['winner']}",
              flush=True)

    eps = results[0]["eps"]
    lstars = sorted(set(r["Lstar"] for r in results))
    lstar_str = (str(lstars[0]) if len(lstars) == 1
                 else "{" + ",".join(map(str, lstars)) + "} (per-rho)")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  SUMMARY  (eps={eps:g}, L*={lstar_str}, ratio = cond-stdMC / cond-MLMC, "
          f"seed-averaged)")
    print("=" * 78)
    hdr = (f"  {'rho':>6} | {'L*':>2} | {'c_cs':>10} | {'c_cm':>10} | {'ratio':>6} | "
           f"{'sl':>5} | {'ld':>5} | winner")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['rho']:6.2f} | {r['Lstar']:>2d} | {r['c_cs']:10.4g} | "
              f"{r['c_cm']:10.4g} | {r['ratio']:6.3f} | {r['sl']:5.2f} | "
              f"{r['ld']:5.2f} | {r['winner']}")

    # ── crossing + verdict ───────────────────────────────────────────────────
    rhos = [r["rho"] for r in results]
    ratios = [r["ratio"] for r in results]
    xc = crossing(rhos, ratios)
    band = [r["ratio"] for r in results if REALISTIC[0] <= r["rho"] <= REALISTIC[1]]
    print("\n" + "-" * 78)
    if xc is None:
        if all(x < 1.0 for x in ratios):
            print(f"  VERDICT: cond-stdMC wins (ratio < 1) at EVERY rho in "
                  f"[{rhos[0]:g}, {rhos[-1]:g}] — conditional MLMC never pays, and "
                  f"there is no 1-crossing.")
            if band:
                print(f"  Across the realistic band rho in "
                      f"[{REALISTIC[0]:g}, {REALISTIC[1]:g}] the ratio is "
                      f"{min(band):.3f}-{max(band):.3f}, so the G-C4 conclusion is "
                      f"robust to correlation.")
        elif all(x > 1.0 for x in ratios):
            print("  VERDICT: cond-MLMC wins (ratio > 1) at every rho — no crossing.")
        else:
            print("  VERDICT: the ratio is non-monotone with no clean 1-crossing; "
                  "see the table.")
    else:
        print(f"  VERDICT: the ratio crosses 1 at rho ~= {xc:.3f} (linear interp) "
              f"— cond-stdMC wins for |rho| below it, cond-MLMC above.")
    print("-" * 78)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.axhline(1.0, color="0.4", lw=1.2, ls="--", label="ratio = 1 (break-even)")
    ax.axvspan(REALISTIC[0], REALISTIC[1], color="tab:blue", alpha=0.12,
               label=f"realistic rho in [{REALISTIC[0]:g}, {REALISTIC[1]:g}]")
    ax.plot(rhos, ratios, "o-", color="tab:red", lw=2, ms=6,
            label="cond-stdMC / cond-MLMC")
    if xc is not None:
        ax.plot([xc], [1.0], "k*", ms=14, label=f"crossing rho = {xc:.3f}")
    ax.set_xlabel(r"correlation  $\rho$")
    ax.set_ylabel("cost ratio  cond-stdMC / cond-MLMC")
    ax.set_title(f"Conditional std-MC vs MLMC across $\\rho$  "
                 f"(eps={eps:g}, per-$\\rho$ $L^*$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("rho_sweep.png", dpi=140)
    print("  saved rho_sweep.png")

    # ── csv ──────────────────────────────────────────────────────────────────
    with open("rho_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "eps", "Lstar", "c_cs", "c_cm", "ratio",
                    "sl_deep", "ld_deep", "kappa", "winner"])
        for r in results:
            w.writerow([r["rho"], r["eps"], r["Lstar"], r["c_cs"], r["c_cm"],
                        r["ratio"], r["sl"], r["ld"], r["kappa"], r["winner"]])
    print("  saved rho_sweep.csv")
    print(f"\n  total wall {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
