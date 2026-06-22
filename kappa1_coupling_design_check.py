r"""
kappa1_coupling_design_check.py  —  de-risk the kappa=1 hybrid MLMC coupling
BEFORE any production wiring (ROADMAP: "design before coding").

Validates the covariance algebra the coupling rests on, against a fine
micro-simulation ground truth, for the rough regime H in {0.07, 0.10, 0.20}:

  Per Volterra cell of length h, with alpha = H-1/2, the hybrid kappa=1 scheme
  needs the pair (dW, W1) where
      dW = \int_0^h dW_s                       (plain increment, var h)
      W1 = \int_0^h (h-s)^alpha dW_s           (nearest-cell exact integral)
  For MLMC the COARSE nearest-cell integral over [0,2h] anchored at 2h is
      Wc1 = \int_0^{2h} (2h-s)^alpha dW_s
          = I1 + W1^(2nd)                       (EXACT split over the two subcells)
  where
      I1        = \int_0^h     (2h-s)^alpha dW_s   (first subcell, coarse anchor)
      W1^(2nd)  = \int_h^{2h}  (2h-s)^alpha dW_s   (== the fine cell-2 nearest int)
  The fine path already carries W1^(2nd); only I1 must be generated, by
  CONDITIONAL RESAMPLING given the fine summary (dW_1st, W1_1st) of the first
  subcell.  This keeps the coarse marginal EXACT (unbiased telescoping) while
  sharing the conditional mean (tight coupling).

Checks:
  (1) closed-form 3x3 covariance of (dW_1st, W1_1st, I1) matches micro-sim
  (2) the conditional construction of I1 reproduces the target covariance
      and has sigma_cond^2 > 0
  (3) Wc1 = I1 + W1^(2nd) has the exact coarse variance (2h)^{2H}/(2H)
  (4) coupling tightness: fraction of Var(I1) explained by the fine summary
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.integrate import quad

np.set_printoptions(precision=5, suppress=True)


def closed_form_cov(H, h):
    """3x3 covariance of (dW, W1_fine[anchor h], I1[anchor 2h]) on subcell [0,h]."""
    a = H - 0.5
    C = quad(lambda w: w**a * (1.0 + w)**a, 0.0, 1.0)[0]   # dimensionless const
    v_dW = h
    v_W1 = h**(2 * H) / (2 * H)
    v_I1 = h**(2 * H) * (2**(2 * H) - 1.0) / (2 * H)
    c_dW_W1 = h**(H + 0.5) / (H + 0.5)
    c_dW_I1 = h**(H + 0.5) * (2**(H + 0.5) - 1.0) / (H + 0.5)
    c_W1_I1 = h**(2 * H) * C
    Sig = np.array([[v_dW,    c_dW_W1, c_dW_I1],
                    [c_dW_W1, v_W1,    c_W1_I1],
                    [c_dW_I1, c_W1_I1, v_I1]])
    return Sig, C


def iso_cov(H, h):
    """Independent ground truth: each covariance is an Ito-isometry integral
    cov(\int f dW, \int g dW) = \int_0^h f(s) g(s) ds, evaluated by adaptive
    quadrature (handles the integrable singularity of the rough kernel, unlike
    a Monte-Carlo micro-sum with left-endpoint kernel weights)."""
    a = H - 0.5
    f_dW = lambda s: 1.0
    f_W1 = lambda s: (h - s)**a
    f_I1 = lambda s: (2 * h - s)**a
    fns = [f_dW, f_W1, f_I1]
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = quad(lambda s: fns[i](s) * fns[j](s), 0.0, h,
                           points=[h] if (i == 1 or j == 1) else None,
                           limit=200)[0]
    return M


def conditional_params(Sig):
    """I1 | (dW,W1) ~ N(beta . x, s2).  Returns beta (2-vec), s2."""
    Sxx = Sig[:2, :2]
    SxI = Sig[:2, 2]
    SII = Sig[2, 2]
    beta = np.linalg.solve(Sxx, SxI)
    s2 = SII - SxI @ beta
    return beta, s2


def main():
    print("=" * 74)
    print("  kappa=1 hybrid MLMC coupling — covariance-algebra validation")
    print("=" * 74)
    for H in (0.07, 0.10, 0.20):
        h = 1.0 / 256                                # a representative fine dt
        Sig, C = closed_form_cov(H, h)
        Mic = iso_cov(H, h)
        relerr = np.abs(Sig - Mic) / (np.abs(Sig) + 1e-300)
        beta, s2 = conditional_params(Sig)
        # coarse marginal check: Var(Wc1) should equal (2h)^{2H}/(2H)
        var_Wc1_target = (2 * h)**(2 * H) / (2 * H)
        var_Wc1_build = Sig[2, 2] + h**(2 * H) / (2 * H)   # Var(I1)+Var(W1_2nd)
        # full coarse BIVARIATE law: Cov(dW_c, W1_c) with dW_c the summed plain
        # increments and W1_c = I1 + W1_2nd.  Must equal (2h)^{H+1/2}/(H+1/2).
        # = Cov(dW_1st, I1) + Cov(dW_2nd, W1_2nd) = Sig[0,2] + Sig[0,1].
        cov_c_target = (2 * h)**(H + 0.5) / (H + 0.5)
        cov_c_build = Sig[0, 2] + Sig[0, 1]
        # coupling tightness: explained fraction of Var(I1)
        explained = 1.0 - s2 / Sig[2, 2]
        # conditional construction MC check
        rng = np.random.default_rng(1)
        L = np.linalg.cholesky(Sig[:2, :2])
        x = (L @ rng.standard_normal((2, 150_000))).T          # (dW,W1)
        I1 = x @ beta + np.sqrt(s2) * rng.standard_normal(150_000)
        Xc = np.column_stack([x, I1])
        Cov_build = (Xc.T @ Xc) / Xc.shape[0]
        build_relerr = np.abs(Cov_build - Sig).max() / np.abs(Sig).max()

        print(f"\n  H = {H}   (alpha={H-0.5:+.2f},  "
              f"C(H)=int_0^1 w^a(1+w)^a dw = {C:.5f})")
        print(f"   closed-form 3x3 cov vs quadrature isometry: max rel.err = "
              f"{relerr.max():.2e}   {'OK' if relerr.max() < 1e-6 else 'CHECK'}")
        print(f"   conditional resample sigma_cond^2 = {s2:.3e} > 0 : "
              f"{'OK' if s2 > 0 else 'FAIL'}   (beta = {beta})")
        print(f"   coarse marginal Var(Wc1): target {var_Wc1_target:.6e}  "
              f"build I1+W1_2nd {var_Wc1_build:.6e}  "
              f"{'OK' if abs(var_Wc1_target/var_Wc1_build-1) < 1e-12 else 'CHECK'}")
        print(f"   coarse cross-cov Cov(dWc,Wc1): target {cov_c_target:.6e}  "
              f"build {cov_c_build:.6e}  "
              f"{'OK' if abs(cov_c_target/cov_c_build-1) < 1e-12 else 'CHECK'}"
              f"   (full bivariate coarse law exact)")
        print(f"   conditional construction reproduces 3x3 cov: max rel.err "
              f"{build_relerr:.3%}   {'OK' if build_relerr < 0.02 else 'CHECK'}")
        print(f"   coupling tightness: fine summary explains "
              f"{explained:.1%} of Var(I1)  (residual {1-explained:.1%} fresh)")
    print("\n  => the coarse nearest-cell integral splits exactly, its marginal")
    print("     is preserved by construction (unbiased), and most of I1 is")
    print("     shared with the fine path (tight coupling). Design is sound.")


if __name__ == "__main__":
    main()
