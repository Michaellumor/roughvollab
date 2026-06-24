"""
gh4_kappa1_conditional.py — STEP 2a: does kappa=1 pay for the conditional
standard-MC path?  Single-grid (NO MLMC levels), via _cond_asian_payoff(kappa,Z).
kappa=0 stays default; the coarse coupler is NOT used.

Bias is measured WITHOUT a noisy independent proxy (a naive n=2048 proxy has
s.e. ~0.003, comparable to the biases, and flips the gate).  Instead, anchored
to truth = kappa=1 @ n=2048 via low-variance coupled/paired differences:
    bias_k0(n) = E[P0(n) - P0(2048)] - d2048          (k0 grid-refinement, coupled)
    bias_k1(n) = E[P0(n) - P0(2048)] + E[P1(n)-P0(n)] - d2048
    d2048      = E[P1(2048) - P0(2048)]               (same-grid k1-k0 at 2048)
P0(n) uses increments = block-sums of the n=2048 fine increments (kappa=0
refinement, no coupler); P1(n)-P0(n) is a same-grid pair (shared dW, fresh Z).
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import numpy as np

from layer1b_mlmc_asian import PARAMS, _cond_asian_payoff, _paths_from_increments


def unbiased_check(p):
    n, N = 128, 300_000
    dt = p["T"] / n
    rng = np.random.default_rng(1)
    sa = sc = sd2 = 0.0
    done, B = 0, 40_000
    while done < N:
        nb = min(B, N - done)
        dW1 = rng.standard_normal((nb, n)) * np.sqrt(dt)
        dW2 = rng.standard_normal((nb, n)) * np.sqrt(dt)
        Z = rng.standard_normal((nb, n))
        ar = _paths_from_increments(dW1, dW2, n, p, "asian", kappa=1, Z=Z)
        pc = _cond_asian_payoff(dW1, dW2, n, p, kappa=1, Z=Z)
        sa += ar.sum(); sc += pc.sum(); sd2 += ((ar - pc)**2).sum()
        done += nb
    ma, mc = sa / N, sc / N
    se = np.sqrt(sd2 / N) / np.sqrt(N)
    z = abs(ma - mc) / se
    print(f"  [unbiasedness @n=128] kappa=1 cond {mc:.4f} vs naive arith {ma:.4f}"
          f"  |diff|={abs(ma-mc):.5f} z={z:.2f}  "
          f"{'OK' if z < 3 else 'CHECK'}")


def measure(seed, p, grids, n_fine=2048, N=150_000, B=2000):
    """Clean coupled/paired biases + single-grid variances, one seed."""
    rng = np.random.default_rng(seed)
    dt_f = p["T"] / n_fine
    acc = {n: dict(bc=0.0, dn=0.0, s0=0.0, ss0=0.0, s1=0.0, ss1=0.0) for n in grids}
    sd2048 = 0.0
    done = 0
    while done < N:
        nb = min(B, N - done)
        dW1f = rng.standard_normal((nb, n_fine)) * np.sqrt(dt_f)
        dW2f = rng.standard_normal((nb, n_fine)) * np.sqrt(dt_f)
        Zf = rng.standard_normal((nb, n_fine))
        P0f = _cond_asian_payoff(dW1f, dW2f, n_fine, p, kappa=0)
        P1f = _cond_asian_payoff(dW1f, dW2f, n_fine, p, kappa=1, Z=Zf)
        sd2048 += (P1f - P0f).sum()
        for n in grids:
            bf = n_fine // n
            dW1n = dW1f.reshape(nb, n, bf).sum(2)
            dW2n = dW2f.reshape(nb, n, bf).sum(2)
            Zn = rng.standard_normal((nb, n))
            P0n = _cond_asian_payoff(dW1n, dW2n, n, p, kappa=0)
            P1n = _cond_asian_payoff(dW1n, dW2n, n, p, kappa=1, Z=Zn)
            a = acc[n]
            a["bc"] += (P0n - P0f).sum()           # coupled k0 refinement
            a["dn"] += (P1n - P0n).sum()           # same-grid k1-k0 gap
            a["s0"] += P0n.sum(); a["ss0"] += (P0n**2).sum()
            a["s1"] += P1n.sum(); a["ss1"] += (P1n**2).sum()
        done += nb
    d2048 = sd2048 / N
    out = {}
    for n in grids:
        a = acc[n]
        bc, dn = a["bc"] / N, a["dn"] / N
        m0, m1 = a["s0"] / N, a["s1"] / N
        v0 = a["ss0"] / N - m0**2
        v1 = a["ss1"] / N - m1**2
        out[n] = dict(bias_k0=bc - d2048, bias_k1=bc + dn - d2048, vratio=v1 / v0)
    return out


def n_star(bias_by_n, grids, thr):
    for n in grids:
        if abs(bias_by_n[n]) <= thr:
            return n
    return None


def main():
    t0 = time.time()
    print("#" * 78)
    print("  G-H4 step 2a — does kappa=1 pay for the conditional standard-MC path?")
    print("  (clean coupled/paired bias; truth = kappa=1 @ n=2048)")
    print("#" * 78)
    p = dict(PARAMS)
    print()
    unbiased_check(p)

    grids = [16, 32, 64, 128]
    seeds = [5, 11, 23]
    per_seed = {s: measure(s, p, grids) for s in seeds}

    # seed-averaged bias + variance ratio
    b0 = {n: np.mean([per_seed[s][n]["bias_k0"] for s in seeds]) for n in grids}
    b1 = {n: np.mean([per_seed[s][n]["bias_k1"] for s in seeds]) for n in grids}
    b0sd = {n: np.std([per_seed[s][n]["bias_k0"] for s in seeds]) for n in grids}
    b1sd = {n: np.std([per_seed[s][n]["bias_k1"] for s in seeds]) for n in grids}
    vr = {n: np.mean([per_seed[s][n]["vratio"] for s in seeds]) for n in grids}

    print("\n  2a RESULTS (H=0.10, seed-avg {5,11,23}, vs kappa=1 n=2048):")
    print("    n   | bias_k0 (sd)     bias_k1 (sd)     ratio | Var_k1/Var_k0")
    for n in grids:
        ratio = b1[n] / b0[n] if abs(b0[n]) > 1e-5 else float('nan')
        print(f"   {n:>4d} | {b0[n]:+.4f}({b0sd[n]:.4f})  {b1[n]:+.4f}({b1sd[n]:.4f})"
              f"  {ratio:5.2f}x | {vr[n]:.3f}x")
    bias_ratio = np.mean([b1[n]/b0[n] for n in grids if abs(b0[n]) > 1e-5])
    var_ratio = float(np.mean(list(vr.values())))
    print(f"\n    => bias ratio (avg)  ~ {bias_ratio:.2f}x   (predicted ~0.65x)")
    print(f"    => variance ratio    ~ {var_ratio:.2f}x   (predicted ~1.10x)")

    # ── GATE, with per-seed stability ────────────────────────────────────────
    print("\n  GATE DECISION (per-seed n* stability shown):")
    gate_i = {}
    for eps in (0.05, 0.025):
        thr = eps / np.sqrt(2.0)
        n0a = n_star(b0, grids, thr); n1a = n_star(b1, grids, thr)
        coarser_avg = (n0a and n1a and n1a <= n0a // 2)
        # per-seed
        flips = []
        for s in seeds:
            bs0 = {n: per_seed[s][n]["bias_k0"] for n in grids}
            bs1 = {n: per_seed[s][n]["bias_k1"] for n in grids}
            ns0, ns1 = n_star(bs0, grids, thr), n_star(bs1, grids, thr)
            flips.append((ns0, ns1, bool(ns0 and ns1 and ns1 <= ns0 // 2)))
        stable = all(f[2] == coarser_avg for f in flips)
        gate_i[eps] = coarser_avg and stable
        print(f"    eps={eps:<6g}(|b|<= {thr:.4f}): avg n*_k0={n0a} n*_k1={n1a} "
              f"-> coarser? {'YES' if coarser_avg else 'no'} | per-seed "
              f"{[ (f[0],f[1]) for f in flips]} {'STABLE' if stable else 'FLIPS->noise'}")
    cond_i = any(gate_i.values())
    cond_ii = var_ratio < 1.6
    print(f"\n    (i)  grid reduced >=1 level AND stable across seeds: "
          f"{'PASS' if cond_i else 'FAIL'}")
    print(f"    (ii) Var_k1/Var_k0 < 1.6: {var_ratio:.2f}  "
          f"{'PASS' if cond_ii else 'FAIL'}")
    run2b = cond_i and cond_ii
    print("\n" + "=" * 78)
    if run2b:
        print(f"  GATE PASSES -> run STEP 2b (cost) for eps "
              f"{[e for e in (0.05,0.025) if gate_i[e]]}.")
    else:
        print("  GATE FAILS -> STOP.  kappa=1 does not buy a coarser bias-grid")
        print("  (robustly, across seeds) for the conditional standard-MC path.")
        print("  Adoption question closes as a clean NEGATIVE: no production role.")
    print("=" * 78)
    print(f"  total wall {time.time()-t0:.0f}s\n")
    return run2b


if __name__ == "__main__":
    main()
