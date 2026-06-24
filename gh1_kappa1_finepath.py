"""
gh1_kappa1_finepath.py — G-H1 gate-check for the kappa=1 hybrid FINE path now
built into layer1b_mlmc_asian.py (opt-in kappa flag; kappa=0 stays the default).
Runs kappa=0 vs kappa=1 SIDE BY SIDE.

  G-H1a  variance gap (headline): empirical Var(W~_T) vs continuum T^{2H},
         swept over H in {0.05,0.10,0.20}.  PASS = kappa=1 within 1% of continuum.
  G-H1b  forward variance: max_t |E[V_t]/xi0 - 1| with the kappa=1 compensator
         (and the WRONG kappa=0 compensator, to show the silent-bias trap).
  G-H1c  BS anchor: eta=0 European price vs Black-Scholes.  PASS = z < 2.
  G-H1d  near-cell law: empirical Var(W_{i,1}), Cov(W_{i,1},dW_i) vs closed forms.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import numpy as np

from layer1b_mlmc_asian import (PARAMS, volterra_weights, volterra_weights_kappa1,
                                _volterra, _paths_from_increments, _bs_call)


def gh1a():
    print("\n" + "=" * 76)
    print("  G-H1a  variance gap: Var(W~_T)/T^{2H}  (>=1e5 paths)   PASS = k1 within 1%")
    print("=" * 76)
    n, N = 256, 120_000
    print("    H    | continuum | kappa0 emp (analytic) | kappa1 emp (analytic) | k1<1%?")
    ok = True
    headline = {}
    for H in (0.05, 0.10, 0.20):
        p = dict(PARAMS, H=H)
        dt = p["T"] / n
        rng = np.random.default_rng(100)
        dW1 = rng.standard_normal((N, n)) * np.sqrt(dt)
        Z = rng.standard_normal((N, n))
        W0, v0 = _volterra(dW1, n, p, kappa=0)
        W1, v1 = _volterra(dW1, n, p, kappa=1, Z=Z)
        cont = p["T"]**(2 * H)
        r0e, r1e = W0[:, -1].var() / cont, W1[:, -1].var() / cont
        r0a, r1a = v0[-1] / cont, v1[-1] / cont
        within = abs(r1e - 1.0) < 0.01
        ok &= within
        headline[H] = (r1e, r0e)
        print(f"   {H:<5g} |  {cont:.4f}   |  {r0e:.4f} ({r0a:.4f})    |  "
              f"{r1e:.4f} ({r1a:.4f})    | {'yes' if within else 'NO'}")
    r1, r0 = headline[0.10]
    print(f"\n   headline (n=256, H=0.10): kappa1 = {r1:.4f} (pred ~0.9996), "
          f"kappa0 = {r0:.4f} (pred ~0.8529)")
    print(f"   => does Var(W~_T) clear 0.99 under kappa=1?  "
          f"{'YES' if r1 > 0.99 else 'NO'}")
    return ok, headline


def gh1b():
    print("\n" + "=" * 76)
    print("  G-H1b  forward variance: max_t |E[V_t]/xi0 - 1|  (correct vs WRONG compensator)")
    print("=" * 76)
    p = dict(PARAMS)
    H, eta, n = p["H"], p["eta"], 256
    Ntot, B, dt = 200_000, 40_000, p["T"] / n
    g, v_k0 = volterra_weights(n, H, p["T"])
    _, v_k1, _, _ = volterra_weights_kappa1(n, H, p["T"])
    acc = {k: [np.zeros(n), np.zeros(n)] for k in ("k0", "k1c", "k1w")}
    rng = np.random.default_rng(7)
    done = 0
    while done < Ntot:
        nb = min(B, Ntot - done)
        dW1 = rng.standard_normal((nb, n)) * np.sqrt(dt)
        Z = rng.standard_normal((nb, n))
        W0, _ = _volterra(dW1, n, p, kappa=0)
        W1, _ = _volterra(dW1, n, p, kappa=1, Z=Z)
        V = {"k0":  p["xi0"] * np.exp(eta * W0 - 0.5 * eta**2 * v_k0[None, :]),
             "k1c": p["xi0"] * np.exp(eta * W1 - 0.5 * eta**2 * v_k1[None, :]),
             "k1w": p["xi0"] * np.exp(eta * W1 - 0.5 * eta**2 * v_k0[None, :])}
        for k in acc:
            acc[k][0] += V[k].sum(0); acc[k][1] += (V[k]**2).sum(0)
        done += nb
    print("    scheme                     | max_t|E[V_t]/xi0-1| | max per-point z")
    res = {}
    for k, label in [("k0", "kappa0 (baseline)"),
                     ("k1c", "kappa1 CORRECT comp."),
                     ("k1w", "kappa1 WRONG comp. (v_k0)")]:
        m = acc[k][0] / Ntot
        se = np.sqrt(np.maximum(acc[k][1]/Ntot - m**2, 0) / Ntot)
        err = np.abs(m / p["xi0"] - 1.0).max()
        z = (np.abs(m - p["xi0"]) / se).max()
        res[k] = (err, z)
        print(f"    {label:<26s} |      {err:.4f}        |   {z:.1f}")
    # PASS: correct compensator is noise-level (z like baseline), wrong is a
    # large systematic bias (huge z) -> the trap is caught.
    passb = res["k1c"][1] < 2 * res["k0"][1] + 5 and res["k1w"][1] > 4 * res["k1c"][1]
    print(f"   => kappa1 correct compensator unbiased (z~baseline), wrong one a "
          f"systematic bias: {'PASS' if passb else 'CHECK'}")
    return passb


def gh1c():
    print("\n" + "=" * 76)
    print("  G-H1c  BS anchor: eta=0 European price vs Black-Scholes   PASS = z < 2")
    print("=" * 76)
    p0 = dict(PARAMS, eta=0.0)
    n, N = 128, 120_000
    dt = p0["T"] / n
    rng = np.random.default_rng(2)
    dW1 = rng.standard_normal((N, n)) * np.sqrt(dt)
    dW2 = rng.standard_normal((N, n)) * np.sqrt(dt)
    Z = rng.standard_normal((N, n))
    bs = _bs_call(p0["S0"], p0["K"], p0["T"], p0["r"], np.sqrt(p0["xi0"]))
    pay = {"kappa0": _paths_from_increments(dW1, dW2, n, p0, "european", 0),
           "kappa1": _paths_from_increments(dW1, dW2, n, p0, "european", 1, Z)}
    okc = True
    for name, q in pay.items():
        mc, se = q.mean(), q.std() / np.sqrt(N)
        z = abs(mc - bs) / se
        okc &= z < 2 if name == "kappa1" else True
        print(f"    {name}: MC {mc:.4f} ± {se:.4f} | BS {bs:.4f} | z = {z:.2f}  "
              f"{'OK' if z < 2 else 'CHECK'}")
    return okc


def gh1d():
    print("\n" + "=" * 76)
    print("  G-H1d  near-cell law: Var(W_{i,1}), Cov(W_{i,1},dW_i) vs closed forms")
    print("=" * 76)
    p = dict(PARAMS)
    H, n, N = p["H"], 128, 200_000
    dt = p["T"] / n
    _, _, c_near, sig_perp = volterra_weights_kappa1(n, H, p["T"])
    rng = np.random.default_rng(11)
    dW1 = rng.standard_normal((N, n)) * np.sqrt(dt)
    Z = rng.standard_normal((N, n))
    W_near = c_near * dW1 + sig_perp * Z            # the exact nearest-cell integral
    var_cf = dt**(2 * H) / (2 * H)                  # closed-form Var(W_{i,1})
    cov_cf = dt**(H + 0.5) / (H + 0.5)              # closed-form Cov(W_{i,1}, dW_i)
    print(f"    closed form:  Var(W_i1) = {var_cf:.4e}   Cov(W_i1,dW_i) = {cov_cf:.4e}")
    print("     i   | emp Var      (rel.err) | emp Cov       (rel.err)")
    okd = True
    for i in (1, 32, 64, 127):
        ev = W_near[:, i].var()
        ec = np.cov(W_near[:, i], dW1[:, i])[0, 1]
        re_v, re_c = abs(ev/var_cf - 1), abs(ec/cov_cf - 1)
        okd &= (re_v < 0.02) and (re_c < 0.03)
        print(f"    {i:>4d} | {ev:.4e} ({re_v:.3%}) | {ec:.4e} ({re_c:.3%})")
    print(f"   => near-cell law matches closed forms within MC noise: "
          f"{'PASS' if okd else 'CHECK'}")
    return okd


def main():
    t0 = time.time()
    print("#" * 76)
    print("  G-H1  kappa=1 FINE-PATH gate  (production layer1b_mlmc_asian.py)")
    print("#" * 76)
    a_ok, headline = gh1a()
    b_ok = gh1b()
    c_ok = gh1c()
    d_ok = gh1d()
    print("\n" + "#" * 76)
    res = [("G-H1a variance gap", a_ok), ("G-H1b forward variance", b_ok),
           ("G-H1c BS anchor", c_ok), ("G-H1d near-cell law", d_ok)]
    for name, ok in res:
        print(f"   {'PASS' if ok else 'CHECK'}   {name}")
    allg = all(ok for _, ok in res)
    r1 = headline[0.10][0]
    print(f"\n  headline: Var(W~_T) under kappa=1 = {r1:.4f} "
          f"({'clears' if r1 > 0.99 else 'does NOT clear'} 0.99)")
    print(f"  {'ALL GREEN' if allg else 'NOT all green'}")
    print("#" * 76)
    print(f"  total wall {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
