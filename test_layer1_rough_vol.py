# -*- coding: utf-8 -*-
"""Regression tests for ROADMAP L1-1 (RVL-001 + RVL-002).

The teaching engine ``layer1_rough_vol.py`` was rewritten to the validated
discrete-variance construction:

  * RVL-001 — ``fbm_hybrid`` must produce a Volterra path whose Var(W̃_T) equals
    the discrete v_n = 2H·dt·cumsum(g²) (the old split-kernel construction
    over-subtracted near-diagonal terms and undershot the variance).
  * RVL-002 — ``rough_bergomi_paths`` must use that discrete v as its lognormal
    compensator, so E[V_t]/ξ₀ = 1 at every t (the old continuum t^{2H}
    compensator made it dip to ~0.35).

Both are pinned by seeded Monte-Carlo with a tolerance drawn from the estimator
standard error. The forward-variance check uses a 4·s.e. *family-wise* tolerance
on the worst of the ~n interior time points (testing every point at 3·s.e. has a
~32% family-wise false-positive rate; 4·s.e. is the correctly-calibrated max).
The target v_n comes from the independent, separately-validated
``roughvol_core.volterra_weights`` — not Layer 1's own helper — so these tests
genuinely fail on the pre-fix code.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from layer1_rough_vol import fbm_hybrid, rough_bergomi_paths
from roughvol_core import volterra_weights          # independent validated target

_N, _H = 128, 0.1


def test_fbm_hybrid_matches_discrete_variance():
    """RVL-001: Var(W̃_T) equals the discrete v_n within 4·s.e."""
    N = 40_000
    np.random.seed(2024)
    paths = fbm_hybrid(_N, _H, n_paths=N)            # (N, n); col -1 is W̃_T
    emp_var = float(np.var(paths[:, -1], ddof=1))
    _, v = volterra_weights(_N, _H)
    v_n = float(v[-1])
    ratio = emp_var / v_n
    se = ratio * np.sqrt(2.0 / (N - 1))              # W̃_T Gaussian ⇒ Var(S²)=2σ⁴/(N-1)
    assert abs(ratio - 1.0) < 4.0 * se, (
        f"Var(W̃_T)={emp_var:.5f} vs discrete v_n={v_n:.5f}: ratio={ratio:.5f}, "
        f"|ratio-1|={abs(ratio-1):.5f} exceeds 4·s.e.={4*se:.5f}"
    )


def test_rough_bergomi_forward_variance_is_flat():
    """RVL-002: E[V_t]/ξ₀ ≈ 1 at all t (worst interior point within 4·s.e.)."""
    N = 40_000
    xi0, eta, rho, S0, r = 0.04, 1.9, -0.7, 100.0, 0.0
    np.random.seed(2024)
    _, V = rough_bergomi_paths(_N, _H, xi0, eta, rho, S0, r, n_paths=N)
    Vn = V / xi0
    mean_t = Vn.mean(axis=0)                         # E[V_t]/ξ₀, t = 0..n
    se_t = Vn.std(axis=0, ddof=1) / np.sqrt(N)
    z = np.abs(mean_t[1:] - 1.0) / se_t[1:]          # interior t (t=0 is deterministic)
    worst = float(z.max())
    assert worst < 4.0, (
        f"E[V_t]/ξ₀ worst z-score {worst:.2f} ≥ 4 (family-wise) at t-index "
        f"{int(z.argmax()) + 1}; max|E[V_t]/ξ₀-1|={float(np.abs(mean_t[1:]-1).max()):.4f}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
