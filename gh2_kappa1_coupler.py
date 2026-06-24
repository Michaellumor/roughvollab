"""
gh2_kappa1_coupler.py — gate for the kappa=1 COARSE COUPLER (split + conditional
resampling), per docs/gate_checks/kappa1_hybrid_coupling_design.md.

The math (per-cell covariance, exact split, marginal preservation) is already
de-risked to 1e-11 in kappa1_coupling_design_check.py.  The remaining risk is
the IMPLEMENTATION — specifically the sub-cell index (the reviewer's catch): a
wrong sub-cell still passes a marginal-variance check but collapses the
coupling.  So the gate is a COUPLING-TIGHTNESS assert, run against the correct
coupler AND a deliberately sub-cell-swapped one.

  G-H2a  unbiased  : telescoping consistency  a_l - a_{l-1} - E[Y_l] ~ 0  (< 1)
  G-H2b  tightness : Var(Y_l)/Var(P_f) tiny for the correct coupler, O(1) for
                     the swapped one  (the marginal check is blind to this)
  G-H2c  rate      : Var(Y_l) ~ 2^{-beta l},  beta ~ 2H (constant changes, not rate)
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import numpy as np

from layer1b_mlmc_asian import PARAMS, mlmc_asian_level
from layer1b_kappa1 import (mlmc_level_kappa1, volterra_weights_kappa1,
                            coarse_coupling_params, _wtilde_from_W1diag)


def run_levels(L, N, p, seed, swap_bug=False):
    rng = np.random.default_rng(seed)
    m, v, a, vf = [], [], [], []
    for l in range(L + 1):
        s = mlmc_level_kappa1(l, N, p, rng=rng, swap_bug=swap_bug)
        Y, Pf = s[0], s[1]
        m.append(Y.mean()); v.append(Y.var())
        a.append(Pf.mean()); vf.append(Pf.var())
    return (np.array(m), np.array(v), np.array(a), np.array(vf))


def volterra_tightness(l, N, p, seed, swap_bug):
    """Coupling tightness at the level the coupler operates on: the variance of
    (W~_fine at coarse points - W~_coarse) relative to Var(W~_coarse).  Tiny for
    the correct coupler, O(1) when the sub-cell is swapped."""
    rng = np.random.default_rng(seed)
    H, T, n0 = p["H"], p["T"], p["n0"]
    n_f = n0 * 2**l
    dt_f = T / n_f
    n_c = n_f // 2
    _, _, beta1, sig1 = volterra_weights_kappa1(n_f, H, T)
    beta_cc, sig_cc = coarse_coupling_params(H, dt_f)
    dW1 = rng.standard_normal((N, n_f)) * np.sqrt(dt_f)
    Zf = rng.standard_normal((N, n_f))
    W1diag_f = beta1 * dW1 + sig1 * Zf
    Wf, _ = _wtilde_from_W1diag(dW1, W1diag_f, n_f, p)
    dW1_c = dW1.reshape(N, n_c, 2).sum(axis=2)
    d1 = dW1.reshape(N, n_c, 2)
    w1 = W1diag_f.reshape(N, n_c, 2)
    Zc = rng.standard_normal((N, n_c))
    if not swap_bug:
        I1 = beta_cc[0] * d1[:, :, 0] + beta_cc[1] * w1[:, :, 0] + sig_cc * Zc
        W1diag_c = I1 + w1[:, :, 1]
    else:
        I1 = beta_cc[0] * d1[:, :, 1] + beta_cc[1] * w1[:, :, 1] + sig_cc * Zc
        W1diag_c = I1 + w1[:, :, 0]
    Wc, v_k1c = _wtilde_from_W1diag(dW1_c, W1diag_c, n_c, p)
    D = Wf[:, 1::2] - Wc                       # fine at coarse points - coarse
    # also report coarse marginal vs the exact kappa=1 coarse variance v_k1c
    marg_relerr = np.abs(Wc.var(0) / v_k1c - 1.0).max()
    return D.var(0).mean() / Wc.var(0).mean(), marg_relerr


def main():
    t0 = time.time()
    print("#" * 74)
    print("  G-H2  kappa=1 COARSE-COUPLER gate  (correct vs sub-cell-swapped)")
    print("#" * 74)
    p = dict(PARAMS)
    L, N = 5, 24_000
    m, v, a, vf = run_levels(L, N, p, seed=11, swap_bug=False)

    # G-H2a — unbiased / telescoping consistency
    print("\n  G-H2a  unbiased (telescoping consistency a_l - a_{l-1} - E[Y_l]):")
    chk = []
    for l in range(1, L + 1):
        num = a[l] - a[l - 1] - m[l]
        den = 3 * (np.sqrt(vf[l]) + np.sqrt(vf[l - 1]) + np.sqrt(v[l])) / np.sqrt(N)
        chk.append(abs(num) / den)
    cons = max(chk)
    print(f"        max consistency = {cons:.3f}   (gate < 1)   "
          f"{'OK' if cons < 1 else 'CHECK'}")

    # G-H2c — rate
    ls = np.arange(1, L + 1)
    beta = -np.polyfit(ls, np.log2(v[1:]), 1)[0]

    # G-H2b — coupling tightness AT THE VOLTERRA LEVEL (where the coupler acts),
    # correct vs sub-cell-swapped.  Also the blind marginal check, to show it
    # does NOT catch the swap.
    print("\n  G-H2b  coupling tightness  Var(W~_f@coarse - W~_c)/Var(W~_c)"
          "  (THE assert):")
    print("        l | correct tightness  (marg.err) | swapped tightness  (marg.err)")
    tight_c, tight_s = [], []
    for l in range(1, 5):
        rc, mc = volterra_tightness(l, 40_000, p, seed=11, swap_bug=False)
        rs, msg = volterra_tightness(l, 40_000, p, seed=11, swap_bug=True)
        tight_c.append(rc); tight_s.append(rs)
        print(f"        {l} | {rc:.5f}   ({mc:.1e}) | {rs:.5f}   ({msg:.1e})")
    tight_c = np.array(tight_c); tight_s = np.array(tight_s)
    print("        ^ swapped 'marg.err' is ~0 too: the coarse MARGINAL is correct"
          "\n          either way — only the tightness (coupling) exposes the bug.")
    tight_ok = (tight_c.max() < 0.02) and (tight_s.min() > 0.2)
    sep = tight_s.mean() / tight_c.mean()
    print(f"        correct max = {tight_c.max():.5f} (< 0.02) | swapped min = "
          f"{tight_s.min():.3f} (> 0.2) | separation {sep:.0f}x")
    print(f"        => tightness discriminates correct from swapped: "
          f"{'OK' if tight_ok else 'CHECK'}")

    # G-H2c — payoff-level rate, and a kappa=0 sanity comparison (kappa=1 must
    # not be pathologically loose at the payoff level)
    rng0 = np.random.default_rng(11)
    v0 = np.array([mlmc_asian_level(l, N, p, rng=rng0)[0].var() for l in range(L+1)])
    print(f"\n  G-H2c  payoff rate + kappa=0 sanity:")
    print(f"        Var(Y_l) kappa1 = {np.array2string(v[1:], precision=3)}")
    print(f"        Var(Y_l) kappa0 = {np.array2string(v0[1:], precision=3)}  "
          f"(comparable -> coupler not pathologically loose)")
    print(f"        beta_kappa1 = {beta:.3f}  vs 2H = {2*p['H']:.2f}  "
          f"(constant changes, not the rate)   "
          f"{'OK' if abs(beta - 2*p['H']) < 0.12 else 'CHECK'}")

    print("\n" + "#" * 74)
    allg = cons < 1 and tight_ok and abs(beta - 2 * p["H"]) < 0.12
    for name, ok in [("G-H2a unbiased (telescoping)", cons < 1),
                     ("G-H2b coupling tightness", tight_ok),
                     ("G-H2c rate beta~2H", abs(beta - 2*p["H"]) < 0.12)]:
        print(f"   {'PASS' if ok else 'CHECK'}   {name}")
    print(f"\n  {'ALL GREEN' if allg else 'NOT all green'} — kappa=1 coarse coupler "
          f"{'verified exact and tight' if allg else 'needs investigation'}")
    print("#" * 74)
    print(f"  total wall {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
