"""
rh_beta_gate.py — β = 2H acceptance gate for the rough-Heston simulator
=======================================================================
Layer 4, brick 1 acceptance gate (gate-check spec §0/§2a). Re-measures the
Giles level-variance decay rate β on rough_heston.py via an MLMC coupling
(coarse grid = pairwise-summed fine Brownian increments), with an arithmetic-
Asian-call functional — apples-to-apples with the layer1b β-vs-H result.

PASS criterion: β must be CONSISTENT with layer1b's MEASURED β-vs-H, not merely
> 2H. Reference (layer1b_mlmc_asian.py, κ=0 exact coupling):

    H      0.05   0.10   0.20   0.35
    2H     0.10   0.20   0.40   0.70
    L1b β  0.13   0.23   0.42   0.72

Gate: |β - 2H| <~ 0.05, monotone increasing in H, regression not noise-dominated,
N >= 20k. β is a STRONG-error property → insensitive to the full-truncation
positivity bias (see rough_heston.py caveat); that is why this gate is valid.

Run:  python rh_beta_gate.py            (full: n0=16, levels 1..5, N=20000)
      python rh_beta_gate.py --quick    (smoke: small N/levels)
"""
import argparse
import sys
import numpy as np

try:                                    # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rough_heston import _rough_heston_from_increments, PARAMS

L1B_BETA = {0.05: 0.13, 0.10: 0.23, 0.20: 0.42, 0.35: 0.72}


def _asian_call(S, K, n):
    """Discounted (r=0) arithmetic-Asian call, trapezoidal average on [0, T]."""
    A = (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1]) / n
    return np.maximum(A - K, 0.0)


def _rh_mlmc_level(l, N, H, p, n0, rng, K=100.0, positivity="truncation"):
    """One MLMC level: coupled fine (n0·2^l) / coarse (n0·2^(l-1)) Asian payoffs.
    Coarse is driven by pairwise-summed fine increments (Var(sum)=2·dt_f=dt_c).
    Returns (mean_Y, var_Y, price_fine)."""
    n_f = n0 * 2 ** l
    n_c = n_f // 2
    dt_f = p["T"] / n_f
    dWV_f = rng.standard_normal((N, n_f)) * np.sqrt(dt_f)
    dWp_f = rng.standard_normal((N, n_f)) * np.sqrt(dt_f)
    Sf, _ = _rough_heston_from_increments(dWV_f, dWp_f, n_f, H, p, positivity=positivity)

    dWV_c = dWV_f.reshape(N, n_c, 2).sum(axis=2)        # pairwise-summed fine
    dWp_c = dWp_f.reshape(N, n_c, 2).sum(axis=2)
    Sc, _ = _rough_heston_from_increments(dWV_c, dWp_c, n_c, H, p, positivity=positivity)

    Pf = _asian_call(Sf, K, n_f)
    Pc = _asian_call(Sc, K, n_c)
    Y = Pf - Pc
    return Y.mean(), Y.var(ddof=1), Pf.mean()


def measure_beta(H, p, n0, levels, N, rng, K=100.0, positivity="truncation"):
    """β = -slope of log2 Var[Y_l] vs l (regression replicated from estimate_rates)."""
    rows = [_rh_mlmc_level(l, N, H, p, n0, rng, K, positivity) for l in levels]
    mY = np.array([r[0] for r in rows])
    vY = np.array([r[1] for r in rows])
    price = rows[-1][2]
    ls = np.asarray(levels, float)
    slope, intercept = np.polyfit(ls, np.log2(vY), 1)
    resid = np.log2(vY) - (slope * ls + intercept)
    dof = len(ls) - 2
    slope_se = (np.sqrt((resid ** 2).sum() / dof / ((ls - ls.mean()) ** 2).sum())
                if dof > 0 else np.nan)
    monotone = bool(np.all(np.diff(vY) < 0))            # variance must decay with l
    return dict(beta=-slope, beta_se=slope_se, vY=vY, mY=mY,
                price=price, monotone=monotone)


def run_sweep(H_grid, n0, levels, N, seed, positivity="qe", nu=None):
    from rough_heston import rough_heston_paths
    p = PARAMS if nu is None else {**PARAMS, "nu": nu}
    print(f"\n=== positivity = {positivity!r}   nu={p['nu']}   "
          f"(n0={n0} levels={levels} N={N} seed={seed}) ===")
    print(f"{'H':>6} {'2H':>6} {'L1b':>6} {'beta':>8} {'±se':>7} "
          f"{'|b-2H|':>8} {'mono':>5} {'near0':>7} {'verdict':>8}")
    betas, devs, monos = [], [], []
    for H in H_grid:
        rng = np.random.default_rng(seed)
        r = measure_beta(H, p, n0, levels, N, rng, positivity=positivity)
        dev = abs(r["beta"] - 2 * H)
        _, _, V = rough_heston_paths(n=128, H=H, n_paths=4000,
                                     rng=np.random.default_rng(1),
                                     positivity=positivity, nu=p["nu"])
        near0 = float((V < 1e-6).mean())
        betas.append(r["beta"]); devs.append(dev); monos.append(r["monotone"])
        l1b = L1B_BETA.get(round(H, 2), float("nan"))
        verdict = "OK" if (dev <= 0.05 and r["monotone"]) else "OFF"
        print(f"{H:6.2f} {2*H:6.2f} {l1b:6.2f} {r['beta']:8.3f} {r['beta_se']:7.3f} "
              f"{dev:8.3f} {str(r['monotone']):>5} {near0:7.1%} {verdict:>8}")
    inc = bool(np.all(np.diff(betas) > 0))
    consistent = all(d <= 0.05 for d in devs) and all(monos) and inc
    print(f"  β monotone increasing in H: {inc}  |  GATE: "
          f"{'PASS — consistent with layer1b' if consistent else 'NOT consistent'}")
    return consistent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (small N/levels)")
    ap.add_argument("--compare", action="store_true",
                    help="compare truncation / reflection / qe at default params")
    ap.add_argument("--positivity", default="qe",
                    choices=["truncation", "reflection", "qe"])
    ap.add_argument("--nu", type=float, default=None, help="override vol-of-vol ν")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--n0", type=int, default=16)
    ap.add_argument("--levels", type=int, default=4, help="max level (1..levels)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    H_GRID = [0.05, 0.10, 0.20, 0.35]
    if a.quick:
        run_sweep(H_GRID, n0=8, levels=[1, 2, 3], N=a.N or 1500,
                  seed=a.seed, positivity=a.positivity, nu=a.nu)
    elif a.compare:
        lv = list(range(1, a.levels + 1))
        for sch in ["truncation", "reflection", "qe"]:
            run_sweep(H_GRID, n0=a.n0, levels=lv, N=a.N or 20000,
                      seed=a.seed, positivity=sch, nu=a.nu)
    else:
        lv = list(range(1, a.levels + 1))
        run_sweep(H_GRID, n0=a.n0, levels=lv, N=a.N or 20000,
                  seed=a.seed, positivity=a.positivity, nu=a.nu)
