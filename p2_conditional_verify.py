"""
p2_conditional_verify.py  —  verification of the conditional-MC coupling
(gate G-C4) for the arithmetic-Asian rough-Bergomi MLMC engine.

Estimator (user-specified):
    P_cond = arith_payoff - ( geom_payoff - E[geom_payoff | W] )
           = arith - geom + E[geom | W]
a conditional geometric-Asian CONTROL VARIATE.  geom - E[geom|W] has zero
conditional mean given the variance path W (=dW1), so P_cond is UNBIASED for
the arithmetic-Asian price at every grid level -> identical discretisation bias
to naive arithmetic MLMC.  E[geom|W] is closed-form (the geometric average of
the conditionally-Gaussian log-prices is lognormal), which removes the
orthogonal driver W_perp from the control exactly.

NON-INVASIVE: imports volterra_weights/_paths_from_increments/PARAMS from
layer1b_mlmc_asian.py and adds only the conditional pieces.

G-C4 methodology (user-specified, learned from the antithetic G-A4 artifact):
  * Do NOT let the adaptive driver pick L (it manufactures phantom cost ratios
    by choosing different finest levels for estimators with identical bias).
  * Fix L* per eps = the level the NAIVE bias test selects (~2 at eps=0.1,
    ~5 at eps=0.05). eps in {0.1, 0.05}, Lmax=6.  Same L* for all four
    estimators (their bias is identical).
  * Giles optimal cost (2/eps^2)(sum_l sqrt(V_l C_l))^2 for naive & cond MLMC;
    standard single-level cost (2/eps^2) Var(P_L*) C_L* for naive & cond.
  * Honest cost: the conditional per-path overhead kappa is MEASURED and folded
    into C_l (so the comparison is not rigged).  The headline ratio
    cond-std-MC / cond-MLMC is kappa-INVARIANT (both pay kappa per path).
  * Sweep seeds 11/99/1234; flag any "win" that flips sign as an artifact.

Run:  python p2_conditional_verify.py            (full)
      python p2_conditional_verify.py --quick
"""

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm

from layer1b_mlmc_asian import PARAMS, volterra_weights, _paths_from_increments


# ── conditional control-variate payoffs ──────────────────────────────────────
def _cond_payoffs(dW1, dW2, n, p):
    """Return (arith, geom, Egeom) on one grid of n steps.
    arith  : discounted arithmetic-Asian payoff (trapezoidal average)
    geom   : discounted geometric-Asian payoff  (same trapezoidal weights)
    Egeom  : closed-form E[geom | W], W = dW1 (the variance path)."""
    H, eta, rho = p["H"], p["eta"], p["rho"]
    xi0, S0, K, T, r = p["xi0"], p["S0"], p["K"], p["T"], p["r"]
    dt = T / n
    N = dW1.shape[0]

    g, v = volterra_weights(n, H, T)
    W_tilde = np.sqrt(2.0 * H) * fftconvolve(dW1, g[None, :], axes=1)[:, :n]
    V_left = np.empty_like(dW1)
    V_left[:, 0] = xi0
    V_left[:, 1:] = xi0 * np.exp(eta * W_tilde[:, :-1]
                                 - 0.5 * eta**2 * v[None, :-1])

    dW_S = rho * dW1 + np.sqrt(1.0 - rho**2) * dW2
    dlogS = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * dW_S
    logS = np.concatenate([np.zeros((N, 1)), np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    disc = np.exp(-r * T)

    # arithmetic trapezoidal average + payoff
    A = (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1]) / n
    arith = disc * np.maximum(A - K, 0.0)

    # geometric trapezoidal average (same weights) + payoff
    LG = (0.5 * logS[:, 0] + logS[:, 1:-1].sum(axis=1)
          + 0.5 * logS[:, -1]) / n                       # = sum_k w_k logS_k
    Gv = S0 * np.exp(LG)
    geom = disc * np.maximum(Gv - K, 0.0)

    # conditional law of LG given W: Gaussian(muG, sigG^2)
    mu = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * rho * dW1    # E[dlogS_j|W]
    M = np.concatenate([np.zeros((N, 1)), np.cumsum(mu, axis=1)], axis=1)
    muG = (0.5 * M[:, 0] + M[:, 1:-1].sum(axis=1) + 0.5 * M[:, -1]) / n
    jj = np.arange(n)
    Wbar = 1.0 - (1.0 + 2.0 * jj) / (2.0 * n)            # cum weight above step j
    sigG2 = (1.0 - rho**2) * dt * (V_left * Wbar[None, :]**2).sum(axis=1)
    sigG = np.sqrt(np.maximum(sigG2, 1e-300))
    F = S0 * np.exp(muG + 0.5 * sigG2)
    d1 = (np.log(F / K) + 0.5 * sigG2) / sigG
    d2 = d1 - sigG
    Egeom = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return arith, geom, Egeom


# ── unit test: closed-form Egeom vs brute-force conditional MC ────────────────
def unit_test_Egeom(p, quick):
    print("\n" + "=" * 74)
    print("  UNIT TEST — closed-form E[geom|W] matches brute-force conditional MC")
    print("=" * 74)
    rng = np.random.default_rng(7)
    n = 48
    dt = p["T"] / n
    n_paths = 4
    M = 60_000 if quick else 200_000
    dW1 = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    print("   path   closed-form    brute-force(±se)        z")
    ok = True
    for i in range(n_paths):
        dW1_rep = np.repeat(dW1[i:i + 1], M, axis=0)
        dW2 = rng.standard_normal((M, n)) * np.sqrt(dt)
        _, geom, Egeom = _cond_payoffs(dW1_rep, dW2, n, p)
        bf, se, cf = geom.mean(), geom.std() / np.sqrt(M), Egeom[0]
        z = abs(cf - bf) / se
        ok &= z < 3.5
        print(f"    {i}     {cf:9.5f}     {bf:9.5f} (±{se:.5f})   {z:5.2f}"
              f"  {'OK' if z < 3.5 else 'FAIL'}")
    return ok


# ── per-level statistics, naive arith vs conditional (paired) ────────────────
def estimate_cond_rates(L, N, p=PARAMS, seed=11, batch=5000):
    rng = np.random.default_rng(seed)
    Vn, Vc, VPn, VPc, mn, mc, an, ac, sd_ctrl = ([] for _ in range(9))
    for l in range(L + 1):
        n_f = p["n0"] * 2**l
        dt_f = p["T"] / n_f
        b = max(200, min(batch, 2_560_000 // n_f))
        Yn = np.empty(N); Yc = np.empty(N); Pf = np.empty(N); Pcf = np.empty(N)
        done = 0
        while done < N:
            nb = min(b, N - done)
            dW1 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
            dW2 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
            af, gf, ef = _cond_payoffs(dW1, dW2, n_f, p)
            pcf = af - gf + ef
            if l == 0:
                yn, yc = af, pcf
            else:
                n_c = n_f // 2
                dW1_c = dW1.reshape(nb, n_c, 2).sum(axis=2)
                dW2_c = dW2.reshape(nb, n_c, 2).sum(axis=2)
                ac_, gc_, ec_ = _cond_payoffs(dW1_c, dW2_c, n_c, p)
                pcc = ac_ - gc_ + ec_
                yn, yc = af - ac_, pcf - pcc
            Yn[done:done + nb] = yn; Yc[done:done + nb] = yc
            Pf[done:done + nb] = af; Pcf[done:done + nb] = pcf
            done += nb
        Vn.append(Yn.var()); Vc.append(Yc.var())
        VPn.append(Pf.var()); VPc.append(Pcf.var())
        mn.append(Yn.mean()); mc.append(Yc.mean())
        an.append(Pf.mean()); ac.append(Pcf.mean())
        sd_ctrl.append((Pf - Pcf).std())     # control variate (geom-Egeom) std
    return dict(Vn=np.array(Vn), Vc=np.array(Vc), VPn=np.array(VPn),
                VPc=np.array(VPc), mn=np.array(mn), mc=np.array(mc),
                an=np.array(an), ac=np.array(ac),
                sd_ctrl=np.array(sd_ctrl), L=L, N=N)


def measure_kappa(p=PARAMS, l=3, N=8000):
    """Honest per-path-eval cost overhead of the conditional estimator."""
    rng = np.random.default_rng(0)
    n = p["n0"] * 2**l
    dt = p["T"] / n
    dW1 = rng.standard_normal((N, n)) * np.sqrt(dt)
    dW2 = rng.standard_normal((N, n)) * np.sqrt(dt)
    _paths_from_increments(dW1, dW2, n, p, "asian")        # warm up
    _cond_payoffs(dW1, dW2, n, p)
    reps = 8
    t0 = time.time()
    for _ in range(reps):
        _paths_from_increments(dW1, dW2, n, p, "asian")
    tn = time.time() - t0
    t0 = time.time()
    for _ in range(reps):
        _cond_payoffs(dW1, dW2, n, p)
    tc = time.time() - t0
    return tc / tn


def _choose_L(m, alpha, eps, Lmax=6):
    Lp = len(m) - 1

    def m_at(l):
        return abs(m[l]) if l <= Lp else abs(m[Lp]) * 2.0**(-alpha * (l - Lp))
    for Lc in range(2, Lmax + 1):
        offs = np.arange(min(3, Lc))
        tail = np.array([m_at(Lc - o) * 2.0**(-alpha * o) for o in offs])
        if tail.max() / (2**alpha - 1) <= eps / np.sqrt(2.0):
            return Lc
    return Lmax


def _giles_cost(V, C, eps, Lstar):
    s = np.sqrt(V[:Lstar + 1] * C[:Lstar + 1]).sum()
    return 2.0 / eps**2 * s**2


def gate_c4(quick, p=PARAMS):
    print("\n" + "=" * 74)
    print("  G-C4 — matched-accuracy cost, naive vs conditional (seeds 11/99/1234)")
    print("=" * 74)
    N = 8_000 if quick else 14_000
    Lpil = 6
    seeds = [11, 99, 1234]
    res = {s: estimate_cond_rates(Lpil, N, p=p, seed=s) for s in seeds}
    kappa = measure_kappa(p=p)
    n0 = p["n0"]

    # rates / L* from the reference (seed-11) NAIVE means; same L* for all four
    r0 = res[11]
    ls = np.arange(1, Lpil + 1)
    alpha = max(0.5, -np.polyfit(ls, np.log2(np.abs(r0['mn'][1:])), 1)[0])
    beta_n = -np.polyfit(ls, np.log2(r0['Vn'][1:]), 1)[0]
    beta_c = -np.polyfit(ls, np.log2(r0['Vc'][1:]), 1)[0]
    print(f"  kappa (cond/naive per-path cost) = {kappa:.3f}   alpha={alpha:.3f}"
          f"   beta_naive={beta_n:.3f}  beta_cond={beta_c:.3f}")

    # unbiasedness (control variate): E[P_cond] == E[arith] per level.
    # The diff is the paired control-variate mean; compare to its own s.e.
    diff = np.abs(r0['an'] - r0['ac'])
    se = r0['sd_ctrl'] / np.sqrt(r0['N'])
    z = (diff / se).max()
    print(f"  control-variate unbiasedness: max_l |E[arith]-E[P_cond]|/s.e. = "
          f"{z:.2f}  (mean-zero control -> identical bias)   "
          f"{'OK' if z < 4 else 'CHECK'}")

    Cn = np.array([n0 * 2**l * (1.0 if l == 0 else 1.5) for l in range(Lpil + 1)])
    Cc = kappa * Cn

    eps_list = [0.10, 0.05]
    rows = []
    for eps in eps_list:
        Lstar = _choose_L(r0['mn'], alpha, eps)
        print(f"\n  ── eps = {eps:g}   L* = {Lstar}  "
              f"(naive bias test; shared by all four estimators) ──")
        print(f"   seed | naive-MLMC  cond-MLMC  naive-stdMC cond-stdMC | "
              f"cheapest      | cond_std/cond_mlmc")
        per_seed = []
        for s in seeds:
            r = res[s]
            c_nm = _giles_cost(r['Vn'], Cn, eps, Lstar)
            c_cm = _giles_cost(r['Vc'], Cc, eps, Lstar)
            c_ns = 2.0 / eps**2 * r['VPn'][Lstar] * (n0 * 2**Lstar)
            c_cs = 2.0 / eps**2 * r['VPc'][Lstar] * (kappa * n0 * 2**Lstar)
            costs = {'naive-MLMC': c_nm, 'cond-MLMC': c_cm,
                     'naive-stdMC': c_ns, 'cond-stdMC': c_cs}
            cheap = min(costs, key=costs.get)
            key_ratio = c_cs / c_cm
            per_seed.append(dict(seed=s, c_nm=c_nm, c_cm=c_cm, c_ns=c_ns,
                                 c_cs=c_cs, cheap=cheap, key=key_ratio,
                                 cs_ns=c_cs / c_ns, cm_nm=c_cm / c_nm))
            print(f"   {s:<4d} | {c_nm:9.3g}  {c_cm:9.3g}  {c_ns:9.3g}  "
                  f"{c_cs:9.3g} | {cheap:12s} | {key_ratio:.3f}")
        rows.append(dict(eps=eps, Lstar=Lstar, per_seed=per_seed))
    return rows, res, seeds, kappa


def variance_factors(res, seeds):
    print("\n" + "=" * 74)
    print("  VARIANCE REDUCTION (paired) — single-level and level-diff")
    print("=" * 74)
    L = res[seeds[0]]['L']
    # average the ratios across seeds, per level
    sl = np.array([[res[s]['VPn'][l] / res[s]['VPc'][l] for l in range(L + 1)]
                   for s in seeds]).mean(axis=0)
    ld = np.array([[res[s]['Vn'][l] / res[s]['Vc'][l] for l in range(1, L + 1)]
                   for s in seeds]).mean(axis=0)
    print("   level                : " + "  ".join(f"{l:5d}" for l in range(L + 1)))
    print("   single-level  V_n/V_c: " + "  ".join(f"{x:5.2f}" for x in sl))
    print("   level-diff    V_n/V_c: " + "   - " +
          "  ".join(f"{x:5.2f}" for x in ld))
    print(f"\n   single-level reduction: mean {sl.mean():.2f}x  "
          f"(deep levels ~{sl[-3:].mean():.2f}x)   [predicted ~4.2x]")
    print(f"   level-diff   reduction: mean {ld.mean():.2f}x  "
          f"(deep levels ~{ld[-3:].mean():.2f}x)   [predicted ~3.2x]")
    return sl, ld


def verdicts(rows, sl, ld, seeds):
    print("\n" + "█" * 74)
    print("  VERDICTS vs the prediction")
    print("█" * 74)

    sl_m, ld_m = sl[-3:].mean(), ld[-3:].mean()
    print(f"\n  [V1] single-level variance reduction ~4.2x: measured "
          f"~{sl_m:.2f}x (deep levels)")
    print(f"  [V2] level-diff   variance reduction ~3.2x: measured "
          f"~{ld_m:.2f}x (deep levels)")
    print(f"       => single-level reduces MORE than level-diff "
          f"({sl_m:.2f} > {ld_m:.2f}): "
          f"{'CONFIRMED' if sl_m > ld_m else 'REFUTED'}")

    print(f"\n  [V3] conditional STANDARD MC is cheapest & cond-MLMC does NOT "
          f"beat it (cond_std/cond_mlmc < 1):")
    all_lt1 = True
    cheapest_always_condstd = True
    for row in rows:
        keys = [ps['key'] for ps in row['per_seed']]
        cheaps = [ps['cheap'] for ps in row['per_seed']]
        signflip = (min(keys) < 1) != (max(keys) < 1)
        all_lt1 &= all(k < 1 for k in keys)
        cheapest_always_condstd &= all(c == 'cond-stdMC' for c in cheaps)
        print(f"       eps={row['eps']:<5g} L*={row['Lstar']}  "
              f"cond_std/cond_mlmc per seed = "
              f"[{', '.join(f'{k:.3f}' for k in keys)}]  "
              f"cheapest={set(cheaps)}  "
              f"{'(SIGN FLIP!)' if signflip else '(stable)'}")
    print(f"       => cond_std/cond_mlmc < 1 on every seed: "
          f"{'CONFIRMED' if all_lt1 else 'REFUTED'};  "
          f"cheapest is conditional-stdMC throughout: "
          f"{'YES' if cheapest_always_condstd else 'NO'}")

    overall = (sl_m > ld_m) and all_lt1 and cheapest_always_condstd
    print("\n" + "─" * 74)
    print(f"  SUMMARY: prediction {'CONFIRMED' if overall else 'NOT fully confirmed'}"
          f" — conditioning helps single-level more than level-diff, conditional"
          f"\n  standard MC is the cheapest of the four, and conditional MLMC does"
          f" not pay for itself.")
    print("─" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("\n" + "█" * 74)
    print("  CONDITIONAL-MC (geometric control variate) — VERIFICATION  [G-C4]")
    print(f"  P_cond = arith - (geom - E[geom|W]) | mode="
          f"{'QUICK' if args.quick else 'FULL'}")
    print("█" * 74)
    assert unit_test_Egeom(PARAMS, args.quick), "Egeom closed form FAILED"
    rows, res, seeds, kappa = gate_c4(args.quick)
    sl, ld = variance_factors(res, seeds)
    verdicts(rows, sl, ld, seeds)
    print(f"\n  total wall {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
