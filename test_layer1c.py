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


# ──────────────────────────────────────────────────────────────────────────
# Rung 1 — RV proxy (the corruption ladder's decisive mirage test)
# ──────────────────────────────────────────────────────────────────────────

from roughvol_core import rough_bergomi_paths
from layer1c_roughness_audit import realized_log_variance


def test_rung1_control_estimators_innocent_on_true_smooth_signal():
    """CONTROL: on the TRUE smooth (H=0.5) volatility, all three estimators
    must correctly report ≈ 0.5. This proves any spurious roughness seen via
    the proxy is the PROXY's doing, not a fault in the estimators."""
    _, S, V = rough_bergomi_paths(16384, 0.5, 50, eta=1.0,
                                  rng=np.random.default_rng(404))
    logV_true = np.log(V[:, 1:])
    idx = np.linspace(0, logV_true.shape[1] - 1, 512).astype(int)
    lvt = logV_true[:, idx]
    for est in (gjr_hurst, pvariation_hurst, mfdfa_hurst):
        H = est(lvt)
        assert H > 0.4, (
            f"{est.__name__} on TRUE smooth signal gave {H:.3f}; should be "
            f"≈0.5 — estimators must be innocent for the Rung-1 logic to hold"
        )


def test_rung1_proxy_manufactures_spurious_roughness_on_smooth_null():
    """THE SMOKING GUN: a genuinely SMOOTH (H=0.5) process, viewed through
    the RV proxy at a small window, must read as ROUGH — demonstrating the
    Cont–Das mirage. If a smooth truth produces a rough estimate, the
    roughness is purely a proxy artefact."""
    _, S, V = rough_bergomi_paths(16384, 0.5, 50, eta=1.0,
                                  rng=np.random.default_rng(404))
    log_rv = realized_log_variance(S, window=32)
    H_proxy = gjr_hurst(log_rv)
    assert H_proxy < 0.3, (
        f"smooth null through RV proxy gave H={H_proxy:.3f}; the artefact "
        f"should drive it well below the true 0.5 (toward the rough regime)"
    )


def test_rung1_artefact_severity_decreases_with_window():
    """The mirage's severity is set by the RV window: a SMALL window (noisy
    proxy) manufactures MORE spurious roughness than a LARGE window (cleaner
    proxy). So the estimated H on the smooth null should INCREASE toward the
    true 0.5 as the window grows."""
    _, S, V = rough_bergomi_paths(16384, 0.5, 50, eta=1.0,
                                  rng=np.random.default_rng(404))
    H_small = gjr_hurst(realized_log_variance(S, window=32))
    H_large = gjr_hurst(realized_log_variance(S, window=128))
    assert H_large > H_small, (
        f"larger window should give cleaner proxy (H {H_large:.3f}) than "
        f"small window (H {H_small:.3f}) — artefact must fade with window"
    )


def test_rung1_envelope_collapses_at_noisy_window():
    """Bias envelope: at a noisy window, the estimate COLLAPSES toward ≈0.1
    almost regardless of true H — so the spread of estimated H across the
    full true-H spectrum is small, and the smooth/persistent end reads far
    BELOW its true value (the observational-equivalence problem)."""
    ests = []
    for H_true in (0.05, 0.30, 0.70):
        _, S, V = rough_bergomi_paths(16384, H_true, 40, eta=1.0,
                                      rng=np.random.default_rng(505))
        ests.append(gjr_hurst(realized_log_variance(S, window=32)))
    span = max(ests) - min(ests)
    # despite true H ranging over 0.65, the proxy estimate barely moves
    assert span < 0.20, (
        f"noisy-window estimate spanned {span:.3f} across true H∈[0.05,0.70]; "
        f"the collapse means it should stay in a narrow band near ~0.1"
    )
    # and the genuinely smooth/persistent end must read spuriously low
    assert ests[-1] < 0.30, (
        f"true H=0.70 read as {ests[-1]:.3f} through the noisy proxy — should "
        f"be dragged far below truth into the spurious-rough band"
    )


def test_rung1_envelope_recovers_at_cleaner_window():
    """The collapse is not inevitable: at a CLEANER (larger) window the
    estimate tracks the true H far better, so the spread across the spectrum
    is materially larger than at the noisy window."""
    def span_at(window):
        ests = []
        for H_true in (0.05, 0.30, 0.70):
            _, S, V = rough_bergomi_paths(16384, H_true, 40, eta=1.0,
                                          rng=np.random.default_rng(505))
            ests.append(gjr_hurst(realized_log_variance(S, window)))
        return max(ests) - min(ests)
    assert span_at(128) > span_at(32), (
        "cleaner window should recover more of the true-H range than the "
        "noisy window — the collapse must depend on the sampling choice"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
