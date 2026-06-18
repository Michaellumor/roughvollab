"""
test_layer1c.py — tests for the roughness-estimator audit
=========================================================
Pins the GJR estimator's Rung-0 behaviour so a regression (or a clobber)
that breaks recovery fails CI. Tolerances mirror layer1c's ORACLE_TOLERANCE,
which is calibrated to the estimator's true finite-lag bias — not to slack.

Run:  pytest test_layer1c.py -v
"""

import numpy as np
import pytest

from roughvol_core import rough_log_variance_paths
from layer1c_roughness_audit import gjr_hurst, ORACLE_TOLERANCE


@pytest.mark.parametrize("H_true", [0.10, 0.30, 0.50])
def test_gjr_recovers_known_H_within_tolerance(H_true):
    """On clean spot-vol paths, GJR must recover H within its calibrated
    per-regime tolerance. This is the Rung-0 gate in unit-test form."""
    _, logV = rough_log_variance_paths(4096, H_true, 4000, eta=1.5,
                                       rng=np.random.default_rng(101))
    H_est = gjr_hurst(logV)
    assert abs(H_est - H_true) <= ORACLE_TOLERANCE[H_true], (
        f"H_true={H_true}: estimate {H_est:.4f} outside "
        f"tolerance {ORACLE_TOLERANCE[H_true]}"
    )


def test_gjr_bias_is_positive_and_grows_as_H_shrinks():
    """Document the estimator's signature: a systematic positive finite-lag
    bias that increases toward H → 0. If this ordering ever reverses, the
    estimator's behaviour has changed and the audit's premise needs review."""
    biases = {}
    for H_true in (0.05, 0.10, 0.30):
        _, logV = rough_log_variance_paths(4096, H_true, 4000, eta=1.5,
                                           rng=np.random.default_rng(101))
        biases[H_true] = gjr_hurst(logV) - H_true
    assert biases[0.05] > biases[0.10] > biases[0.30]
    assert biases[0.30] > -0.02            # essentially unbiased by H=0.3


def test_gjr_monofractal_r2_high_on_rough_bergomi():
    """Rough Bergomi log-variance is monofractal (ζ_q linear in q), so the
    diagnostic R² should be ~1. Real multifractal data would bend away."""
    _, logV = rough_log_variance_paths(4096, 0.1, 4000, eta=1.5,
                                       rng=np.random.default_rng(101))
    _, det = gjr_hurst(logV, return_detail=True)
    assert det["monofractal_r2"] > 0.98


def test_gjr_accepts_1d_input():
    """Single-path (1-D) input must work, not just (n_paths, n) matrices."""
    _, logV = rough_log_variance_paths(4096, 0.2, 1, eta=1.5,
                                       rng=np.random.default_rng(5))
    H_est = gjr_hurst(logV[0])         # pass a bare 1-D array
    assert 0.0 < H_est < 0.6


# ──────────────────────────────────────────────────────────────────────────
# Section 2 — Cont-Das p-variation estimator
# ──────────────────────────────────────────────────────────────────────────

from layer1c_roughness_audit import pvariation_hurst, PVAR_TOLERANCE


@pytest.mark.parametrize("H_true", [0.10, 0.30, 0.45])
def test_pvariation_recovers_known_H_within_tolerance(H_true):
    """Rung-0 gate as a unit test: the p-variation estimator must recover H
    within its calibrated per-regime tolerance on clean spot-vol paths."""
    _, logV = rough_log_variance_paths(8192, H_true, 120, eta=1.5,
                                       rng=np.random.default_rng(202))
    H_est = pvariation_hurst(logV)
    assert abs(H_est - H_true) <= PVAR_TOLERANCE[H_true], (
        f"H_true={H_true}: estimate {H_est:.4f} outside "
        f"tolerance {PVAR_TOLERANCE[H_true]}"
    )


def test_pvariation_bias_positive_and_grows_as_H_shrinks():
    """Document the estimator's signature — the SAME as GJR: a positive bias
    increasing toward H → 0. If this ordering reverses, behaviour changed."""
    biases = {}
    for H_true in (0.05, 0.10, 0.30):
        _, logV = rough_log_variance_paths(8192, H_true, 120, eta=1.5,
                                           rng=np.random.default_rng(202))
        biases[H_true] = pvariation_hurst(logV) - H_true
    assert biases[0.05] > biases[0.10] > biases[0.30]
    assert biases[0.30] > -0.02           # near-unbiased by H = 0.3


def test_pvariation_accepts_1d_input():
    """Single-path (1-D) input must work, not just (n_paths, n) matrices."""
    _, logV = rough_log_variance_paths(8192, 0.3, 1, eta=1.5,
                                       rng=np.random.default_rng(11))
    H_est = pvariation_hurst(logV[0])     # bare 1-D array
    assert 0.0 < H_est < 0.6


# ──────────────────────────────────────────────────────────────────────────
# Section 3 — MF-DFA estimator
# ──────────────────────────────────────────────────────────────────────────

from layer1c_roughness_audit import mfdfa_hurst, MFDFA_TOLERANCE


@pytest.mark.parametrize("H_true", [0.10, 0.30, 0.45])
def test_mfdfa_recovers_known_H_within_tolerance(H_true):
    """Rung-0 gate as a unit test: MF-DFA must recover H within its
    calibrated per-regime tolerance on clean spot-vol paths."""
    _, logV = rough_log_variance_paths(8192, H_true, 120, eta=1.5,
                                       rng=np.random.default_rng(303))
    H_est = mfdfa_hurst(logV)
    assert abs(H_est - H_true) <= MFDFA_TOLERANCE[H_true], (
        f"H_true={H_true}: estimate {H_est:.4f} outside "
        f"tolerance {MFDFA_TOLERANCE[H_true]}"
    )


def test_mfdfa_bias_is_negative_opposite_to_others():
    """MF-DFA's distinctive signature: it UNDER-estimates (negative bias) at
    small H — OPPOSITE to GJR and Cont–Das. This sign difference is the key
    audit finding; if it flips, the estimators' relationship has changed."""
    _, logV = rough_log_variance_paths(8192, 0.05, 120, eta=1.5,
                                       rng=np.random.default_rng(303))
    bias = mfdfa_hurst(logV) - 0.05
    assert bias < 0, f"MF-DFA bias at H=0.05 should be negative, got {bias:+.4f}"


def test_mfdfa_multifractal_h_flat_on_monofractal():
    """Rough Bergomi is monofractal: h(q) should be ≈ constant across q.
    Check h(2) and h(3) are close (the multifractality diagnostic)."""
    _, logV = rough_log_variance_paths(8192, 0.1, 120, eta=1.5,
                                       rng=np.random.default_rng(303))
    h2 = mfdfa_hurst(logV, q=2.0) + 1.0       # back to raw h(q)
    h3 = mfdfa_hurst(logV, q=3.0) + 1.0
    assert abs(h2 - h3) < 0.10, f"h(2)={h2:.3f}, h(3)={h3:.3f} — not monofractal"


def test_mfdfa_accepts_1d_input():
    """Single-path (1-D) input must work."""
    _, logV = rough_log_variance_paths(8192, 0.3, 1, eta=1.5,
                                       rng=np.random.default_rng(11))
    H_est = mfdfa_hurst(logV[0])
    assert 0.0 < H_est < 0.6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
