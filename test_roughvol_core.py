"""
test_roughvol_core.py — correctness tests for the shared rough-path engine
==========================================================================
These tests pin the properties that make roughvol_core trustworthy as the
foundation for Layer 1b (pricing) and Layer 1c (estimator audit). They are
the automated form of Layer 1b's Section 1 validation. If a future edit —
or an external tool overwriting files — breaks the variance normalisation
or the forward-variance compensator, these fail loudly instead of letting
corrupted results through.

Run:  pytest test_roughvol_core.py -v
"""

import numpy as np
import pytest

from roughvol_core import (
    volterra_weights,
    volterra_process,
    rough_log_variance_paths,
    rough_bergomi_paths,
)


# ──────────────────────────────────────────────────────────────────────────
# volterra_weights
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H", [0.05, 0.10, 0.30, 0.50, 0.70])
def test_weights_shapes_and_finiteness(H):
    g, v = volterra_weights(256, H, 1.0)
    assert g.shape == (256,) and v.shape == (256,)
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(v))
    assert np.all(v > 0)
    # discrete variance is monotone increasing in t
    assert np.all(np.diff(v) > 0)


def test_weights_reject_bad_H():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            volterra_weights(128, bad, 1.0)


def test_brownian_limit_H_half():
    # At H = 1/2 the kernel is flat (a = 0 => g_m = 1) and v_n -> T
    g, v = volterra_weights(500, 0.5, 1.0)
    assert np.allclose(g, 1.0, atol=1e-12)
    assert v[-1] == pytest.approx(1.0, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# volterra_process — the L1-1 regression guard
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H", [0.10, 0.30])
def test_volterra_variance_matches_discrete_formula(H):
    """
    THE key test. Empirical Var(W̃_T) must match the discrete formula v_n
    within Monte Carlo noise. This is precisely the property Layer 1's
    fbm_hybrid violates (Var ≈ 0.89 vs 1.0). A 3-sigma band on 60k paths.
    """
    n, N = 256, 60_000
    rng = np.random.default_rng(0)
    dW = rng.standard_normal((N, n)) * np.sqrt(1.0 / n)
    W = volterra_process(dW, H, 1.0)
    _, v = volterra_weights(n, H, 1.0)

    emp = W[:, -1].var()
    # variance of the sample-variance estimator ~ 2·sigma^4 / N  (Gaussian)
    se = np.sqrt(2.0) * v[-1] / np.sqrt(N)
    assert abs(emp - v[-1]) < 4 * se, (
        f"H={H}: empirical Var(W̃_T)={emp:.4f} vs discrete v_n={v[-1]:.4f}, "
        f"gap {abs(emp - v[-1]):.4f} exceeds 4·se={4*se:.4f} "
        f"— variance normalisation is broken (L1-1 regression)."
    )


def test_volterra_mean_zero():
    n, N = 128, 40_000
    rng = np.random.default_rng(1)
    dW = rng.standard_normal((N, n)) * np.sqrt(1.0 / n)
    W = volterra_process(dW, 0.1, 1.0)
    se = W[:, -1].std() / np.sqrt(N)
    assert abs(W[:, -1].mean()) < 4 * se


def test_volterra_linearity_in_increments():
    # W̃ is linear in dW: scaling increments scales the process identically
    n = 64
    rng = np.random.default_rng(2)
    dW = rng.standard_normal((10, n)) * np.sqrt(1.0 / n)
    W1 = volterra_process(dW, 0.2, 1.0)
    W2 = volterra_process(3.0 * dW, 0.2, 1.0)
    assert np.allclose(W2, 3.0 * W1, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────
# rough_log_variance_paths — Layer 1c's ground-truth generator
# ──────────────────────────────────────────────────────────────────────────

def test_log_variance_initial_condition():
    xi0 = 0.04
    t, logV = rough_log_variance_paths(128, 0.1, 100, xi0=xi0,
                                       rng=np.random.default_rng(3))
    assert t[0] == 0.0 and t[-1] == pytest.approx(1.0)
    assert logV.shape == (100, 129)
    assert np.allclose(logV[:, 0], np.log(xi0))


@pytest.mark.parametrize("H", [0.10, 0.30])
def test_forward_variance_is_exact(H):
    """
    E[V_t] = ξ₀ at every grid point (V = exp(log_V)). This is the
    compensator-correctness property; a missing −½η²v term (the bug the
    clobbered file reintroduced) makes E[V_t] drift away from ξ₀.
    """
    xi0, N = 0.04, 80_000
    t, logV = rough_log_variance_paths(200, H, N, eta=1.5, xi0=xi0,
                                       rng=np.random.default_rng(4))
    V = np.exp(logV)
    ratio = V.mean(axis=0) / xi0                    # should be ≈ 1 everywhere
    assert abs(ratio - 1.0).max() < 0.05, (
        f"H={H}: max |E[V_t]/xi0 - 1| = {abs(ratio-1).max():.4f} "
        f"— forward-variance compensator is wrong."
    )


def test_log_variance_hurst_is_recoverable():
    """
    Sanity that log V genuinely carries roughness H: the structure function
    E|log V_{t+Δ} − log V_t| should scale like Δ^H. We check the crude slope
    on clean paths is in the right ballpark (loose band — this is a smoke
    test, the real estimator audit is Layer 1c's job).
    """
    H_true, n, N = 0.1, 4096, 4000
    _, logV = rough_log_variance_paths(n, H_true, N, eta=1.5,
                                       rng=np.random.default_rng(5))
    lags = np.array([1, 2, 4, 8, 16, 32])
    m = [np.mean(np.abs(logV[:, lag:] - logV[:, :-lag])) for lag in lags]
    slope = np.polyfit(np.log(lags), np.log(m), 1)[0]
    assert 0.0 < slope < 0.3, f"structure-function slope {slope:.3f} off"


def test_reproducible_with_seed():
    a = rough_log_variance_paths(64, 0.15, 50, rng=np.random.default_rng(7))[1]
    b = rough_log_variance_paths(64, 0.15, 50, rng=np.random.default_rng(7))[1]
    assert np.array_equal(a, b)


# ──────────────────────────────────────────────────────────────────────────
# rough_bergomi_paths
# ──────────────────────────────────────────────────────────────────────────

def test_bergomi_shapes_and_initial_conditions():
    t, S, V = rough_bergomi_paths(128, 0.1, 200, S0=100.0, xi0=0.04,
                                  rng=np.random.default_rng(8))
    assert S.shape == (200, 129) and V.shape == (200, 129)
    assert np.allclose(S[:, 0], 100.0)
    assert np.allclose(V[:, 0], 0.04)
    assert np.all(S > 0)        # lognormal asset stays positive


def test_bergomi_martingale_under_zero_rate():
    # r = 0 => E[S_T] = S0 within MC noise (discounted asset is a martingale)
    N = 60_000
    _, S, _ = rough_bergomi_paths(100, 0.1, N, S0=100.0, r=0.0, eta=1.0,
                                  rho=-0.7, rng=np.random.default_rng(9))
    se = S[:, -1].std() / np.sqrt(N)
    assert abs(S[:, -1].mean() - 100.0) < 4 * se


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
