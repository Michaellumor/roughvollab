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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
