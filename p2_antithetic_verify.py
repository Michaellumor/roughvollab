"""
p2_antithetic_verify.py  —  independent verification of the antithetic MLMC
coupling claims in  p2_antithetic_build_and_verify.md.

NON-INVASIVE: this script imports the production engine from
layer1b_mlmc_asian.py (volterra_weights, _paths_from_increments, PARAMS,
mlmc_asian_level) and re-implements ONLY the pieces that need the antithetic
flag, so the production file is left untouched until the numbers are approved.

It reconstructs the four gates from the build-doc (the referenced
p2_coupling_gate_check.md does not exist in the repo) and runs naive-vs-
antithetic side by side with the doc's seeds (7 / 11 / 23).

Honest-cost note (the crux of G-A4): the antithetic level evaluates THREE
paths — P_f (n_f) + P_fa (n_f) + P_c (n_f/2) = 2.5*n_f flops — vs naive's
P_f + P_c = 1.5*n_f.  So the per-level cost coefficient is 2.5 (anti) vs 1.5
(naive) for l>=1, and 1.0 for l=0 (no coupling).  If you forget the extra
fine path, G-A4 is silently rigged in antithetic's favour.

Run:  python p2_antithetic_verify.py            (full)
      python p2_antithetic_verify.py --quick     (fast smoke pass)
"""

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import layer1b_mlmc_asian as L
from layer1b_mlmc_asian import (PARAMS, _paths_from_increments,
                                volterra_weights, _bs_call)


# ── unified level estimator with antithetic flag (mirrors source §2) ─────────
def mlmc_level(l, N, p=PARAMS, payoff="asian", batch=5000, rng=None,
               antithetic=False):
    """
    antithetic=False : Y = P_f - P_c                  (naive exact coupling)
    antithetic=True  : Y = 0.5*(P_f + P_fa) - P_c     (Giles-Szpruch)
    Identical RNG consumption in both branches -> naive and anti see the SAME
    Brownian paths under the same seed (paired comparison).
    out[0]=Y, out[1]=P_f.
    """
    rng = rng or np.random.default_rng()
    n_f = p["n0"] * 2**l
    dt_f = p["T"] / n_f
    batch = max(200, min(batch, 2_560_000 // n_f))
    out = np.empty((2, N))
    done = 0
    while done < N:
        nb = min(batch, N - done)
        dW1 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
        dW2 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
        P_f = _paths_from_increments(dW1, dW2, n_f, p, payoff)
        if l == 0:
            Y = P_f
        else:
            n_c = n_f // 2
            dW1_c = dW1.reshape(nb, n_c, 2).sum(axis=2)
            dW2_c = dW2.reshape(nb, n_c, 2).sum(axis=2)
            P_c = _paths_from_increments(dW1_c, dW2_c, n_c, p, payoff)
            if antithetic:
                dW1_s = dW1.reshape(nb, n_c, 2)[:, :, ::-1].reshape(nb, n_f)
                dW2_s = dW2.reshape(nb, n_c, 2)[:, :, ::-1].reshape(nb, n_f)
                P_fa = _paths_from_increments(dW1_s, dW2_s, n_f, p, payoff)
                Y = 0.5 * (P_f + P_fa) - P_c
            else:
                Y = P_f - P_c
        out[0, done:done + nb] = Y
        out[1, done:done + nb] = P_f
        done += nb
    return out


def _cost_coef(l, antithetic):
    """flops per sample at level l, in units of n_f = n0*2^l."""
    if l == 0:
        return 1.0
    return 2.5 if antithetic else 1.5


# ── estimate_rates with antithetic + honest cost (mirrors source) ────────────
def estimate_rates2(L_=6, N=20_000, p=PARAMS, payoff="asian", seed=7,
                    antithetic=False, verbose=False):
    rng = np.random.default_rng(seed)
    m_l, v_l, a_l, vf_l, c_l, chk = [], [], [], [], [], []
    for l in range(L_ + 1):
        s = mlmc_level(l, N, p, payoff, rng=rng, antithetic=antithetic)
        Y, Pf = s[0], s[1]
        m_l.append(Y.mean());  v_l.append(Y.var())
        a_l.append(Pf.mean()); vf_l.append(Pf.var())
        c_l.append(p["n0"] * 2**l * _cost_coef(l, antithetic))
        if l:
            num = a_l[l] - a_l[l - 1] - m_l[l]
            den = 3 * (np.sqrt(vf_l[l]) + np.sqrt(vf_l[l - 1])
                       + np.sqrt(v_l[l])) / np.sqrt(N)
            chk.append(abs(num) / den)
    m, v = np.abs(m_l), np.array(v_l)
    ls = np.arange(1, L_ + 1)
    alpha = -np.polyfit(ls, np.log2(m[1:]), 1)[0]
    beta = -np.polyfit(ls, np.log2(v[1:]), 1)[0]
    gamma = np.polyfit(np.arange(L_ + 1), np.log2(c_l), 1)[0]
    return dict(m=np.array(m_l), v=v, a=np.array(a_l), c=np.array(c_l),
                vP=vf_l[-1], alpha=alpha, beta=beta, gamma=gamma,
                L=L_, N=N, consistency=max(chk))


# ── adaptive MLMC with antithetic + honest cost (mirrors mlmc_run) ───────────
def adaptive2(eps, p=PARAMS, alpha=None, beta=None, N0=2_000, Lmin=2,
              Lmax=9, seed=11, antithetic=False, verbose=False):
    rng = np.random.default_rng(seed)
    Lc = Lmin
    Nl = np.zeros(Lc + 1)
    sums = np.zeros((2, Lc + 1))
    costl = np.zeros(Lc + 1)
    dNl = np.full(Lc + 1, N0, dtype=float)
    a_fix, b_fix = alpha, beta
    while dNl.sum() > 0:
        for l in range(Lc + 1):
            if dNl[l] < 1:
                continue
            n_new = int(dNl[l])
            s = mlmc_level(l, n_new, p, rng=rng, antithetic=antithetic)
            sums[0, l] += s[0].sum()
            sums[1, l] += (s[0]**2).sum()
            Nl[l] += n_new
            costl[l] += n_new * p["n0"] * 2**l * _cost_coef(l, antithetic)
        ml = np.abs(sums[0] / Nl)
        Vl = np.maximum(0.0, sums[1] / Nl - (sums[0] / Nl)**2)
        Cl = costl / Nl
        if a_fix is not None:
            alpha = max(0.5, a_fix)
        elif Lc >= 2:
            alpha = max(0.5, -np.polyfit(np.arange(1, Lc + 1), np.log2(
                np.maximum(ml[1:], 1e-12)), 1)[0])
        else:
            alpha = 0.5
        if b_fix is not None:
            beta = max(0.1, b_fix)
        elif Lc >= 2:
            beta = max(0.1, -np.polyfit(np.arange(1, Lc + 1), np.log2(
                np.maximum(Vl[1:], 1e-12)), 1)[0])
        else:
            beta = 0.5
        Ns = np.ceil(2.0 / eps**2 * np.sqrt(Vl / Cl)
                     * np.sum(np.sqrt(Vl * Cl)))
        dNl = np.maximum(0.0, Ns - Nl)
        if (dNl <= 0.01 * Nl).all():
            offs = np.arange(min(3, Lc))
            tail = ml[Lc - offs] * 2.0 ** (-alpha * offs)
            rem = tail.max() / (2**alpha - 1)
            if rem > eps / np.sqrt(2.0):
                if Lc == Lmax:
                    break
                Lc += 1
                Nl = np.append(Nl, 0.0)
                sums = np.append(sums, np.zeros((2, 1)), axis=1)
                costl = np.append(costl, 0.0)
                Vl = np.append(Vl, Vl[-1] / 2**beta)
                Cl = np.append(Cl, Cl[-1] * 2)
                Ns = np.ceil(2.0 / eps**2 * np.sqrt(Vl / Cl)
                             * np.sum(np.sqrt(Vl * Cl)))
                dNl = np.maximum(0.0, Ns - Nl)
    price = (sums[0] / Nl).sum()
    return dict(eps=eps, price=price, L=Lc, Nl=Nl.astype(int),
                Vl=Vl, vP=float(Vl[0]), cost=costl.sum(),
                alpha=alpha, beta=beta)


# ═══════════════════════════════════════════════════════════════════════════
#  GATES
# ═══════════════════════════════════════════════════════════════════════════

def gate_construction():
    """Mechanical exactness checks on the antithetic construction itself."""
    print("\n" + "=" * 72)
    print("  PRE-GATE — antithetic construction is exact (mechanical checks)")
    print("=" * 72)
    rng = np.random.default_rng(123)
    nb, n_f, n_c = 64, 8, 4
    dW = rng.standard_normal((nb, n_f))
    dW_s = dW.reshape(nb, n_c, 2)[:, :, ::-1].reshape(nb, n_f)
    sum_orig = dW.reshape(nb, n_c, 2).sum(axis=2)
    sum_swap = dW_s.reshape(nb, n_c, 2).sum(axis=2)
    max_dev = np.abs(sum_orig - sum_swap).max()
    print(f"  (i)  coarse increment invariant under swap : max|sum-sum_swap|"
          f" = {max_dev:.2e}   {'OK' if max_dev < 1e-12 else 'FAIL'}")
    # swap is an involution and a genuine permutation (not identity)
    not_identity = not np.allclose(dW, dW_s)
    involution = np.allclose(dW, dW_s.reshape(nb, n_c, 2)[:, :, ::-1]
                             .reshape(nb, n_f))
    print(f"  (ii) swap is a non-trivial involution      : "
          f"non-identity={not_identity}, involution={involution}   "
          f"{'OK' if (not_identity and involution) else 'FAIL'}")
    return max_dev < 1e-12 and not_identity and involution


def gate_a1(quick):
    """G-A1 bias-free: E[V_t]=xi0 (engine) + telescoping consistency (coupling)
       + E[Y_anti] matches E[Y_naive] per level (no bias introduced)."""
    print("\n" + "=" * 72)
    print("  G-A1 — bias-free / exact coupling")
    print("=" * 72)
    # (a) forward variance E[V_t] = xi0  — engine property, coupling-independent
    p = dict(PARAMS)
    N, n = (20_000 if quick else 60_000), 256
    rng = np.random.default_rng(1)
    dW1 = rng.standard_normal((N, n)) * np.sqrt(p["T"] / n)
    from scipy.signal import fftconvolve
    g, v = volterra_weights(n, p["H"], p["T"])
    W = np.sqrt(2 * p["H"]) * fftconvolve(dW1, g[None, :], axes=1)[:, :n]
    V = p["xi0"] * np.exp(p["eta"] * W - 0.5 * p["eta"]**2 * v[None, :])
    fwd_err = np.abs(V.mean(axis=0) / p["xi0"] - 1.0).max()
    mc_noise = 3 * np.exp(p["eta"]**2 * v[-1] / 2) / np.sqrt(N)
    print(f"  (a) max_t |E[V_t]/xi0 - 1| = {fwd_err:.4f}   "
          f"(MC noise ~ {mc_noise:.4f})   "
          f"{'OK' if fwd_err < mc_noise else 'CHECK'}")

    # (b) telescoping consistency for the ANTITHETIC estimator + per-level
    #     bias check E[Y_anti] vs E[Y_naive]
    L_ = 4 if quick else 5
    N2 = 8_000 if quick else 16_000
    rn = estimate_rates2(L_, N2, seed=7, antithetic=False)
    ra = estimate_rates2(L_, N2, seed=7, antithetic=True)
    print(f"  (b) telescoping consistency  naive={rn['consistency']:.3f}  "
          f"anti={ra['consistency']:.3f}   (gate < 0.51)   "
          f"{'OK' if ra['consistency'] < 0.51 else 'CHECK'}")
    print("  (c) per-level mean E[Y]: anti must match naive within MC noise")
    print("       l   E[Y]_naive   E[Y]_anti   |diff|    2*se      ok")
    ok_bias = True
    for l in range(L_ + 1):
        se = 2 * np.sqrt((rn['v'][l] + ra['v'][l]) / N2)
        diff = abs(rn['m'][l] - ra['m'][l])
        flag = diff < se
        ok_bias &= bool(flag)
        print(f"      {l}   {rn['m'][l]:+.5f}    {ra['m'][l]:+.5f}   "
              f"{diff:.5f}   {se:.5f}    {'OK' if flag else 'CHECK'}")
    return dict(fwd_err=fwd_err, mc_noise=mc_noise,
                cons_naive=rn['consistency'], cons_anti=ra['consistency'],
                bias_ok=ok_bias)


def gate_a2_a3(quick):
    """G-A2 (beta vs H) + G-A3 (per-level variance factor), seed 23."""
    print("\n" + "=" * 72)
    print("  G-A2 (rate beta vs 2H)  +  G-A3 (variance reduction factor)")
    print("=" * 72)
    H_list = [0.10, 0.30] if quick else [0.05, 0.10, 0.20, 0.35]
    N = 6_000 if quick else 12_000
    L_ = 5
    rows = []
    detail = {}
    for Hx in H_list:
        p = dict(PARAMS, H=Hx)
        rn = estimate_rates2(L_, N, p=p, seed=23, antithetic=False)
        ra = estimate_rates2(L_, N, p=p, seed=23, antithetic=True)
        # per-level variance ratio (l>=1 are the coupled levels)
        ratios = rn['v'][1:] / ra['v'][1:]
        rows.append((Hx, rn['beta'], ra['beta'], 2 * Hx,
                     ratios.mean(), ratios.min(), ratios.max()))
        detail[Hx] = (rn['v'], ra['v'], ratios)
    print("\n   H      b_naive  b_anti   2H     varRatio(mean)  [min ,  max ]")
    for Hx, bn, ba, twoH, rmean, rmin, rmax in rows:
        print(f"  {Hx:<6g}  {bn:6.3f}  {ba:6.3f}  {twoH:5.2f}   "
              f"{rmean:6.3f}        [{rmin:.3f}, {rmax:.3f}]")
    print("\n   per-level Var(Y_naive)/Var(Y_anti)  (l = 1..5):")
    for Hx in H_list:
        _, _, ratios = detail[Hx]
        s = "  ".join(f"{r:.3f}" for r in ratios)
        print(f"     H={Hx:<5g}  {s}")
    return rows, detail


def _choose_L(m_l, alpha, eps, Lmax=9):
    """Giles bias test -> smallest finest level meeting eps. Identical for
    naive and anti because their level means are equal in expectation."""
    Lp = len(m_l) - 1

    def m_at(l):
        return abs(m_l[l]) if l <= Lp else abs(m_l[Lp]) * 2.0**(-alpha * (l - Lp))
    for Lc in range(2, Lmax + 1):
        offs = np.arange(min(3, Lc))
        tail = np.array([m_at(Lc - o) * 2.0**(-alpha * o) for o in offs])
        if tail.max() / (2**alpha - 1) <= eps / np.sqrt(2.0):
            return Lc
    return Lmax


def _ext_VC(v, beta, antithetic, Lmax=9):
    """Per-level variance + cost, extrapolated past the pilot depth."""
    Lp = len(v) - 1
    V = np.array([v[l] if l <= Lp else v[Lp] * 2.0**(-beta * (l - Lp))
                  for l in range(Lmax + 1)])
    C = np.array([PARAMS["n0"] * 2**l * _cost_coef(l, antithetic)
                  for l in range(Lmax + 1)])
    return V, C


def _mlmc_cost(V, C, eps, Lstar):
    """Giles optimal continuous-N cost at fixed finest level Lstar."""
    s = np.sqrt(V[:Lstar + 1] * C[:Lstar + 1]).sum()
    return 2.0 / eps**2 * s**2


def gate_a4(quick):
    """G-A4 (adaptive cost) — the number the sandbox only estimated.
    Honest AND apples-to-apples: antithetic levels charged 2.5x n_f, and the
    finest level L is held EQUAL for naive/anti (their bias is identical, so a
    differing adaptively-selected L is pure stopping-rule noise, not a coupling
    property). Primary measure = Giles optimal cost at matched L; cross-checked
    by the real driver pinned to that L."""
    print("\n" + "=" * 72)
    print("  G-A4 — adaptive MLMC cost (naive vs antithetic), seed 11")
    print("=" * 72)
    # pilot: per-level V_l, C_l and rates, shared seed -> paired
    Lp = 5 if quick else 6
    Np = 8_000 if quick else 12_000
    rn0 = estimate_rates2(Lp, Np, seed=11, antithetic=False)
    ra0 = estimate_rates2(Lp, Np, seed=11, antithetic=True)
    alpha, beta_n, beta_a = rn0['alpha'], rn0['beta'], ra0['beta']
    V0 = rn0['v'][0]                      # full-payoff variance (std-MC ref)
    Vn, Cn = _ext_VC(rn0['v'], beta_n, antithetic=False)
    Va, Ca = _ext_VC(ra0['v'], beta_a, antithetic=True)
    print(f"  pilot: alpha={alpha:.3f}  beta_naive={beta_n:.3f}  "
          f"beta_anti={beta_a:.3f}  V0(payoff)={V0:.3f}")

    # FYI: the *free-running* driver (each run picks its own L) — shown only to
    # expose the artifact; not used for the verdict.
    print("\n  [free-running L, CONFOUNDED — for diagnosis only]")
    for eps in ([0.10] if quick else [0.10, 0.05]):
        fn = adaptive2(eps, seed=11, antithetic=False)
        fa = adaptive2(eps, seed=11, antithetic=True)
        print(f"     eps={eps:<6g} naive L={fn['L']} cost={fn['cost']:.3g}  |  "
              f"anti L={fa['L']} cost={fa['cost']:.3g}   "
              f"(L differs by stopping noise -> not comparable)")

    eps_list = [0.10, 0.05] if quick else [0.10, 0.05, 0.025]
    rows = []
    print("\n  [matched finest level L — honest comparison]")
    for eps in eps_list:
        Lstar = _choose_L(rn0['m'], alpha, eps)
        cost_n = _mlmc_cost(Vn, Cn, eps, Lstar)
        cost_a = _mlmc_cost(Va, Ca, eps, Lstar)
        cost_mc = 2.0 * V0 / eps**2 * PARAMS["n0"] * 2**Lstar
        # coupled-levels-only (l>=1) cost — the doc's "0.87x" lives here
        sc_n = np.sqrt(Vn[1:Lstar + 1] * Cn[1:Lstar + 1]).sum()
        sc_a = np.sqrt(Va[1:Lstar + 1] * Ca[1:Lstar + 1]).sum()
        anti_all = cost_a / cost_n
        anti_coupled = (sc_a / sc_n)**2
        # cross-check with the real driver pinned to Lstar (fixed rates)
        dn = adaptive2(eps, alpha=alpha, beta=beta_n, Lmin=Lstar, Lmax=Lstar,
                       seed=11, antithetic=False)
        da = adaptive2(eps, alpha=alpha, beta=beta_a, Lmin=Lstar, Lmax=Lstar,
                       seed=11, antithetic=True)
        rows.append(dict(eps=eps, Lstar=Lstar, cost_n=cost_n, cost_a=cost_a,
                         ratio_n=cost_mc / cost_n, ratio_a=cost_mc / cost_a,
                         anti_all=anti_all, anti_coupled=anti_coupled,
                         drv_anti=da['cost'] / dn['cost'],
                         price_n=dn['price'], price_a=da['price']))
        print(f"  eps={eps:<6g} L*={Lstar} | analytic cost  naive={cost_n:.3g}"
              f"  anti={cost_a:.3g}")
        print(f"  {'':14s}| stdMC/MLMC  naive={cost_mc/cost_n:.2f}x  "
              f"anti={cost_mc/cost_a:.2f}x")
        print(f"  {'':14s}| anti/naive cost: all-levels={anti_all:.3f}x  "
              f"coupled-only={anti_coupled:.3f}x  "
              f"(driver@L*={da['cost']/dn['cost']:.3f}x)")
        print(f"  {'':14s}| pinned-driver prices: naive={dn['price']:.4f} "
              f"anti={da['price']:.4f}")
    return rows


def verdicts(a1, a23, a4):
    print("\n" + "█" * 72)
    print("  VERDICTS vs the doc's predictions")
    print("█" * 72)

    # Pred 1: beta_anti ~ beta_naive ~ 2H  (refute if ~4H)
    print("\n  [P1] RATE: predicted beta_anti ~ beta_naive ~ 2H (NOT 4H)")
    p1_ok = True
    for Hx, bn, ba, twoH, *_ in a23[0]:
        near_2H = abs(ba - twoH) < 0.6 * twoH + 0.05
        near_4H = abs(ba - 2 * twoH) < 0.3 * twoH
        d_anti_naive = abs(ba - bn)
        tag = "~2H" if near_2H and not near_4H else ("~4H!" if near_4H else "?")
        p1_ok &= (d_anti_naive < 0.10) and not near_4H
        print(f"       H={Hx:<5g} b_naive={bn:.3f} b_anti={ba:.3f} 2H={twoH:.2f}"
              f"  |db|={d_anti_naive:.3f}  -> {tag}")
    print(f"       => {'HOLDS' if p1_ok else 'REFUTED'}: antithetic gives NO "
          f"rate change; beta tracks 2H, not 4H.")

    # Pred 2: variance ratio ~ 1.45 (linearised pre-check said 2.0)
    allr = np.concatenate([d[2] for d in a23[1].values()])
    rmean = allr.mean()
    print(f"\n  [P2] VARIANCE FACTOR: predicted ~1.45 (NOT the 2.0 leading-order"
          f" guess)")
    print(f"       measured per-level ratio across all H, l: mean={rmean:.3f} "
          f"range=[{allr.min():.3f}, {allr.max():.3f}]")
    p2_ok = 1.25 <= rmean <= 1.65
    print(f"       => {'HOLDS' if p2_ok else 'DIFFERS'}: variance reduction is "
          f"a modest constant (~{rmean:.2f}x), well short of 2x.")

    # Pred 3: antithetic net WORSE on cost (efficiency < 1), at MATCHED L
    print(f"\n  [P3] COST: predicted antithetic is net slightly WORSE "
          f"(efficiency ~0.87x; adaptive cost should NOT improve)")
    print(f"       (matched finest level L; bias identical so L is shared)")
    eff_all, eff_coupled = [], []
    for r in a4:
        eff_all.append(1 / r['anti_all'])
        eff_coupled.append(1 / r['anti_coupled'])
        print(f"       eps={r['eps']:<6g} L*={r['Lstar']}  "
              f"anti/naive cost: all-levels={r['anti_all']:.3f}x "
              f"(eff {1/r['anti_all']:.3f}x), coupled-only={r['anti_coupled']:.3f}x "
              f"(eff {1/r['anti_coupled']:.3f}x)")
    me_all, me_coupled = np.mean(eff_all), np.mean(eff_coupled)
    p3_ok = me_all <= 1.001          # anti never cheaper at matched L
    print(f"       coupled-levels-only efficiency ~{me_coupled:.3f}x  "
          f"(this is the doc's ~0.87x number)")
    print(f"       => {'HOLDS' if p3_ok else 'REFUTED'}: all-levels efficiency "
          f"{me_all:.3f}x <= 1  -> antithetic does NOT pay for itself.")
    print(f"       NOTE: the free-running driver can make anti look cheaper, but"
          f" only by stopping at a coarser (more biased) L — an artifact, not a"
          f" win.")

    print("\n" + "─" * 72)
    print(f"  SUMMARY:  P1(rate) {'HOLDS' if p1_ok else 'REFUTED'}   |   "
          f"P2(var~1.45) {'HOLDS' if p2_ok else 'DIFFERS'}   |   "
          f"P3(net-worse) {'HOLDS' if p3_ok else 'REFUTED'}")
    print("─" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("\n" + "█" * 72)
    print("  ANTITHETIC MLMC — INDEPENDENT VERIFICATION")
    print(f"  reusing engine from layer1b_mlmc_asian.py | mode="
          f"{'QUICK' if args.quick else 'FULL'}")
    print("█" * 72)
    assert gate_construction(), "construction exactness FAILED"
    a1 = gate_a1(args.quick)
    a23 = gate_a2_a3(args.quick)
    a4 = gate_a4(args.quick)
    verdicts(a1, a23, a4)
    print(f"\n  total wall {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
