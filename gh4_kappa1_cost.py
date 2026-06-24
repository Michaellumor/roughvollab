"""
gh4_kappa1_cost.py — STEP 2b: matched-accuracy cost, kappa=0 vs kappa=1, for the
conditional standard-MC path.  Runs only because 2a's gate passed (robustly, all
seeds): kappa=1 clears |bias|<=eps/sqrt(2) one grid coarser
   eps=0.05 : n*_k0=32 -> n*_k1=16
   eps=0.025: n*_k0=64 -> n*_k1=32
and Var_k1/Var_k0 ~ 1.13 < 1.6.

Cost = (2/eps^2) * Var(P_cond) * c(n*), with c(n*) the per-path wall cost MEASURED
empirically for each scheme at its own n* (kappa=1 overhead not assumed).  Seed-
averaged over {5,11,23}; if the win flips sign across seeds it is inside the
measurement noise.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import numpy as np

from layer1b_mlmc_asian import PARAMS, _cond_asian_payoff

NSTAR = {0.05: (32, 16), 0.025: (64, 32)}     # (n*_k0, n*_k1) from 2a, robust
SEEDS = [5, 11, 23]


def per_path_cost(n, p, kappa, batch=20_000, reps=12):
    """Wall seconds per path for the conditional payoff at grid n, scheme kappa."""
    dt = p["T"] / n
    rng = np.random.default_rng(0)
    dW1 = rng.standard_normal((batch, n)) * np.sqrt(dt)
    dW2 = rng.standard_normal((batch, n)) * np.sqrt(dt)
    Z = rng.standard_normal((batch, n)) if kappa == 1 else None
    for _ in range(2):                                   # warm up
        _cond_asian_payoff(dW1, dW2, n, p, kappa=kappa, Z=Z)
    t0 = time.perf_counter()
    for _ in range(reps):
        _cond_asian_payoff(dW1, dW2, n, p, kappa=kappa, Z=Z)
    return (time.perf_counter() - t0) / (reps * batch)


def cond_var(n, p, kappa, seed, N=200_000, B=40_000):
    dt = p["T"] / n
    rng = np.random.default_rng(seed)
    s = ss = 0.0
    done = 0
    while done < N:
        nb = min(B, N - done)
        dW1 = rng.standard_normal((nb, n)) * np.sqrt(dt)
        dW2 = rng.standard_normal((nb, n)) * np.sqrt(dt)
        Z = rng.standard_normal((nb, n)) if kappa == 1 else None
        q = _cond_asian_payoff(dW1, dW2, n, p, kappa=kappa, Z=Z)
        s += q.sum(); ss += (q**2).sum()
        done += nb
    m = s / N
    return ss / N - m**2


def main():
    t0 = time.time()
    print("#" * 78)
    print("  G-H4 step 2b — matched-accuracy cost, kappa=0 vs kappa=1 (conditional)")
    print("#" * 78)
    p = dict(PARAMS)

    # per-path cost at the relevant grids, both schemes (median of 3 timings)
    print("\n  per-path cost c(n) [microseconds], measured:")
    grids = sorted({n for pr in NSTAR.values() for n in pr})
    cost = {}
    for n in grids:
        c0 = np.median([per_path_cost(n, p, 0) for _ in range(3)])
        c1 = np.median([per_path_cost(n, p, 1) for _ in range(3)])
        cost[(0, n)] = c0; cost[(1, n)] = c1
        print(f"    n={n:>4d} : kappa0 {c0*1e6:6.3f}   kappa1 {c1*1e6:6.3f}   "
              f"overhead {c1/c0:.2f}x")

    print("\n  matched-accuracy cost (seed-avg {5,11,23}):")
    print("   eps    | n*_k0 Var_k0 c_k0 | n*_k1 Var_k1 c_k1 | cost_k0   cost_k1   k1/k0")
    for eps, (n0, n1) in NSTAR.items():
        v0 = np.mean([cond_var(n0, p, 0, s) for s in SEEDS])
        ratios = []
        for s in SEEDS:
            vv0 = cond_var(n0, p, 0, s)
            vv1 = cond_var(n1, p, 1, s)
            ck0 = 2.0 / eps**2 * vv0 * cost[(0, n0)]
            ck1 = 2.0 / eps**2 * vv1 * cost[(1, n1)]
            ratios.append(ck1 / ck0)
        v1 = np.mean([cond_var(n1, p, 1, s) for s in SEEDS])
        ck0 = 2.0 / eps**2 * v0 * cost[(0, n0)]
        ck1 = 2.0 / eps**2 * v1 * cost[(1, n1)]
        r = np.array(ratios)
        flip = (r.min() < 1) != (r.max() < 1)
        print(f"   {eps:<6g} | {n0:>4d}  {v0:5.2f} {cost[(0,n0)]*1e6:5.2f} | "
              f"{n1:>4d}  {v1:5.2f} {cost[(1,n1)]*1e6:5.2f} | {ck0:.3g}  {ck1:.3g}  "
              f"{ck1/ck0:.3f}")
        print(f"          per-seed k1/k0 = [{', '.join(f'{x:.3f}' for x in r)}]  "
              f"{'SIGN FLIPS -> inside noise' if flip else 'stable'}")

    print("\n" + "=" * 78)
    print("  VERDICT (predicted: grid halves; net cost ratio k1/k0 ~0.6-0.75,")
    print("  i.e. kappa=1 ~1.3-1.7x cheaper for the conditional std-MC path):")
    print("=" * 78)
    print(f"  total wall {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
