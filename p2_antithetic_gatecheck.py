"""
p2_antithetic_gatecheck.py — run the four antithetic gates (G-A1..G-A4)
THROUGH the production functions in layer1b_mlmc_asian.py (estimate_rates and
mlmc_run, both now carrying the `antithetic` flag).  This both produces the
naive-vs-antithetic comparison table and end-to-end-checks the integrated flag.

Seeds: 7 (estimate_rates / G-A1), 23 (sweep / G-A2,G-A3), 11 (adaptive / G-A4),
matching the baseline.  Gates reconstructed from p2_antithetic_build_and_verify.md
(the referenced p2_coupling_gate_check.md is not in the repo).

G-A4 is reported two ways: the literal free-running adaptive driver (as the doc
asks) AND the honest matched-finest-level cost — because the free-running driver
selects different L for naive vs antithetic (identical bias => L must match),
which manufactures phantom cost ratios.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.signal import fftconvolve

from layer1b_mlmc_asian import (PARAMS, estimate_rates, mlmc_run,
                                volterra_weights, _level_cost_coef)


# ── G-A1 ─────────────────────────────────────────────────────────────────────
def gate_a1():
    print("\n" + "=" * 72)
    print("  G-A1  bias-free:  E[V_t]=xi0 (engine) + telescoping consistency")
    print("=" * 72)
    p = dict(PARAMS)
    # forward variance E[V_t]=xi0 — a property of the variance path, identical
    # for naive and antithetic (the swap does not touch the Volterra law)
    N, n = 60_000, 256
    rng = np.random.default_rng(1)
    dW1 = rng.standard_normal((N, n)) * np.sqrt(p["T"] / n)
    g, v = volterra_weights(n, p["H"], p["T"])
    W = np.sqrt(2 * p["H"]) * fftconvolve(dW1, g[None, :], axes=1)[:, :n]
    V = p["xi0"] * np.exp(p["eta"] * W - 0.5 * p["eta"]**2 * v[None, :])
    fwd_err = np.abs(V.mean(axis=0) / p["xi0"] - 1.0).max()
    noise = 3 * np.exp(p["eta"]**2 * v[-1] / 2) / np.sqrt(N)
    print(f"  E[V_t]=xi0:  max_t |E[V_t]/xi0 - 1| = {fwd_err:.4f}  "
          f"(MC noise ~ {noise:.4f})  {'OK' if fwd_err < noise else 'CHECK'}")

    rn = estimate_rates(L=5, N=16_000, seed=7, antithetic=False, verbose=False)
    ra = estimate_rates(L=5, N=16_000, seed=7, antithetic=True, verbose=False)
    print(f"  telescoping consistency (gate < 0.51):  naive {rn['consistency']:.3f}"
          f"   antithetic {ra['consistency']:.3f}   "
          f"{'OK' if ra['consistency'] < 0.51 else 'CHECK'}")
    return ra['consistency'] < 0.51 and fwd_err < noise


# ── G-A2 + G-A3 ──────────────────────────────────────────────────────────────
def gate_a2_a3():
    print("\n" + "=" * 72)
    print("  G-A2 rate (beta vs 2H)  +  G-A3 variance factor  (sweep seed 23)")
    print("=" * 72)
    H_list = [0.05, 0.10, 0.20, 0.35]
    rows, detail = [], {}
    for H in H_list:
        p = dict(PARAMS, H=H)
        rn = estimate_rates(L=5, N=12_000, p=p, seed=23, antithetic=False, verbose=False)
        ra = estimate_rates(L=5, N=12_000, p=p, seed=23, antithetic=True, verbose=False)
        ratios = rn['v'][1:] / ra['v'][1:]            # per-level Var ratio (l>=1)
        rows.append((H, rn['beta'], ra['beta'], 2 * H, ratios.mean()))
        detail[H] = ratios
    print("\n   H      beta_naive  beta_anti   2H     mean Var(Yn)/Var(Ya)")
    for H, bn, ba, twoH, rm in rows:
        print(f"  {H:<6g}  {bn:9.3f}  {ba:9.3f}  {twoH:5.2f}     {rm:.3f}")
    print("\n   G-A3 per-level Var(Y_naive)/Var(Y_anti)  (levels 1..5):")
    for H in H_list:
        print(f"     H={H:<5g}  " + "  ".join(f"{r:.3f}" for r in detail[H]))
    return rows, detail


# ── G-A4 ─────────────────────────────────────────────────────────────────────
def _choose_L(m, alpha, eps, Lmax=9):
    Lp = len(m) - 1
    m_at = lambda l: abs(m[l]) if l <= Lp else abs(m[Lp]) * 2.0**(-alpha*(l-Lp))
    for Lc in range(2, Lmax + 1):
        offs = np.arange(min(3, Lc))
        tail = np.array([m_at(Lc - o) * 2.0**(-alpha * o) for o in offs])
        if tail.max() / (2**alpha - 1) <= eps / np.sqrt(2.0):
            return Lc
    return Lmax


def _giles(v, c, eps, Ls):
    return 2.0 / eps**2 * np.sqrt(v[:Ls+1] * c[:Ls+1]).sum()**2


def gate_a4():
    print("\n" + "=" * 72)
    print("  G-A4 cost: adaptive MLMC naive vs antithetic (seed 11)")
    print("=" * 72)
    n0 = PARAMS["n0"]
    eps_list = [0.10, 0.05, 0.025]

    # (i) LITERAL free-running adaptive driver (what the doc asks for)
    print("\n  [free-running adaptive driver — the literal request]")
    print("   eps     | naive: L  cost      stdMC/MLMC | anti: L  cost      stdMC/MLMC | anti/naive")
    free = []
    for eps in eps_list:
        rn = mlmc_run(eps, seed=11, antithetic=False, verbose=False)
        ra = mlmc_run(eps, seed=11, antithetic=True, verbose=False)
        mc_n = 2.0 * rn['Vl'][0] / eps**2 * n0 * 2**rn['L']
        mc_a = 2.0 * ra['Vl'][0] / eps**2 * n0 * 2**ra['L']
        free.append((eps, rn, ra, mc_n/rn['cost'], mc_a/ra['cost']))
        print(f"   {eps:<7g} | {rn['L']:>2d}  {rn['cost']:.3g}  {mc_n/rn['cost']:>6.2f}x   "
              f"| {ra['L']:>2d}  {ra['cost']:.3g}  {mc_a/ra['cost']:>6.2f}x   "
              f"| {ra['cost']/rn['cost']:.3f}x")
    print("   ^ NOTE: naive and anti pick DIFFERENT L (identical bias => this is a"
          "\n     stopping-rule artifact, not a real cost difference). Honest below.")

    # (ii) HONEST matched finest level (pilot per-level V_l, C_l from production
    #      estimate_rates; production cost model already charges anti 2.5x)
    print("\n  [matched finest level L — honest comparison]")
    pn = estimate_rates(L=6, N=12_000, seed=11, antithetic=False, verbose=False)
    pa = estimate_rates(L=6, N=12_000, seed=11, antithetic=True, verbose=False)
    alpha = max(0.5, -np.polyfit(np.arange(1, 7), np.log2(np.abs(pn['m'][1:])), 1)[0])
    print("   eps     | L* | naive cost   anti cost    stdMC/MLMC n->a   | anti/naive  eff")
    matched = []
    for eps in eps_list:
        Ls = _choose_L(pn['m'], alpha, eps, Lmax=6)
        cn = _giles(pn['v'], pn['c'], eps, Ls)
        ca = _giles(pa['v'], pa['c'], eps, Ls)
        mc = 2.0 * pn['v'][0] / eps**2 * n0 * 2**Ls
        avn = ca / cn
        matched.append((eps, Ls, avn))
        print(f"   {eps:<7g} | {Ls:>2d} | {cn:.3g}    {ca:.3g}    "
              f"{mc/cn:>5.2f}x -> {mc/ca:>5.2f}x  | {avn:.3f}x     {1/avn:.3f}x")
    return free, matched


# ── verdicts ─────────────────────────────────────────────────────────────────
def verdicts(a2, a4):
    rows, detail = a2
    free, matched = a4
    print("\n" + "#" * 72)
    print("  VERDICTS vs the doc's predictions")
    print("#" * 72)

    p1 = all(abs(ba - bn) < 0.05 and abs(ba - 2*H) < 0.6*(2*H)+0.05
             for H, bn, ba, _, _ in rows)
    print("\n  [P1] beta_anti ~ 2H, no rate change (refute if ~4H):")
    for H, bn, ba, twoH, _ in rows:
        print(f"        H={H:<5g}  beta_anti={ba:.3f}  vs 2H={twoH:.2f}  "
              f"vs naive={bn:.3f}")
    print(f"        => {'HOLDS' if p1 else 'REFUTED'}")

    allr = np.concatenate([detail[H] for H in detail])
    p2 = 1.30 <= allr.mean() <= 1.60
    print(f"\n  [P2] variance ratio ~1.45 (linearised guess said 2.0): "
          f"measured mean {allr.mean():.3f} (range {allr.min():.3f}-{allr.max():.3f})")
    print(f"        => {'HOLDS' if p2 else 'DIFFERS'}")

    effs = [1/avn for _, _, avn in matched]
    p3 = np.mean(effs) <= 1.0
    print(f"\n  [P3] antithetic net WORSE on cost (efficiency ~0.87x; ratio should"
          f" NOT improve):")
    for eps, Ls, avn in matched:
        print(f"        eps={eps:<6g} L*={Ls}  anti/naive cost={avn:.3f}x  "
              f"efficiency={1/avn:.3f}x")
    print(f"        => matched-L mean efficiency {np.mean(effs):.3f}x <= 1: "
          f"{'HOLDS' if p3 else 'REFUTED'}  "
          f"(free-running driver's apparent win is the L-selection artifact)")

    print("\n" + "-" * 72)
    print(f"  SUMMARY: P1 {'HOLDS' if p1 else 'REFUTED'} | "
          f"P2 {'HOLDS' if p2 else 'DIFFERS'} | "
          f"P3 {'HOLDS' if p3 else 'REFUTED'}")
    print("-" * 72)


def main():
    import time
    t0 = time.time()
    print("#" * 72)
    print("  ANTITHETIC GATE-CHECK via production layer1b_mlmc_asian.py")
    print("#" * 72)
    gate_a1()
    a2 = gate_a2_a3()
    a4 = gate_a4()
    verdicts(a2, a4)
    print(f"\n  total wall {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
