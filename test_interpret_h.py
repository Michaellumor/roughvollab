"""
test_interpret_h.py — tests for the de-biasing / envelope-inversion tool
========================================================================
The load-bearing test is `test_debias_recovers_known_true_H`: simulate a price
path at a KNOWN true H, push it through the same RV proxy the real pipeline
uses to get an observed Ĥ, then de-bias it against a matched envelope — the
implied true H must come back near the value we simulated. If de-biasing can
recover a truth we control, it is trustworthy on a truth we don't (real data).

Everything runs at a deliberately SMALL scale (short series, few Monte-Carlo
paths) so the suite stays fast; the machinery is identical at production scale.

Run:  pytest test_interpret_h.py -v
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys

import numpy as np
import pytest

from roughvol_core import rough_bergomi_paths
from layer1c_roughness_audit import realized_log_variance, gjr_hurst
from interpret_h import (
    observed_H_at,
    build_bias_curve,
    invert,
    interpret,
    BiasCurve,
    Interpretation,
)

# Small, fast conditions. Short enough to run quickly, long enough that the
# estimators are not completely degenerate.
N_OBS = 400
WINDOW = 48
N_MC = 6
GRID = np.array([0.05, 0.10, 0.20, 0.30, 0.45, 0.60])


# ──────────────────────────────────────────────────────────────────────────
# inversion mechanics
# ──────────────────────────────────────────────────────────────────────────

def test_invert_monotone_curve():
    grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mean = np.array([0.15, 0.24, 0.32, 0.41, 0.50])      # increasing
    h = invert(0.32, grid, mean)
    assert abs(h - 0.3) < 1e-9


def test_invert_interpolates_between_nodes():
    grid = np.array([0.1, 0.2, 0.3])
    mean = np.array([0.10, 0.20, 0.30])                  # identity curve
    assert abs(invert(0.25, grid, mean) - 0.25) < 1e-9


def test_invert_outside_range_is_nan():
    grid = np.array([0.1, 0.2, 0.3])
    mean = np.array([0.15, 0.24, 0.32])
    assert np.isnan(invert(0.05, grid, mean))            # below curve
    assert np.isnan(invert(0.99, grid, mean))            # above curve


def test_classify_inversion_below_and_above_floor():
    from interpret_h import _classify_inversion
    grid = np.array([0.05, 0.10, 0.20, 0.30])
    mean = np.array([0.116, 0.148, 0.21, 0.29])          # model floor ~0.116
    # real crypto-like observed 0.083 is BELOW the floor (the key real-data case)
    status, h = _classify_inversion(0.083, grid, mean)
    assert status == "below_floor" and h == []
    status, h = _classify_inversion(0.50, grid, mean)
    assert status == "above_ceiling"
    status, sols = _classify_inversion(0.20, grid, mean)
    assert status == "ok" and 0.10 < sols[0] < 0.20    # observed 0.20 -> true ~0.18


def test_nonmonotone_curve_is_multivalued():
    """A hump-shaped bias curve (the noisy-proxy regime) must flag rough≡smooth."""
    from interpret_h import _classify_inversion, _all_crossings, _is_monotone
    # mirrors the measured window=48 BTC curve: rises then falls
    grid = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.60])
    mean = np.array([0.115, 0.129, 0.160, 0.135, 0.091, 0.077, 0.038])
    assert not _is_monotone(mean), "hump curve must register as non-monotone"
    status, cand = _classify_inversion(0.084, grid, mean)
    assert status == "multivalued"
    # the in-grid solution sits on the smooth/falling branch (~0.43)
    assert any(c > 0.35 for c in cand)
    # a clean monotone increasing curve must NOT be flagged
    mono = np.array([0.12, 0.16, 0.21, 0.29, 0.34, 0.40, 0.48])
    assert _is_monotone(mono)
    assert _classify_inversion(0.25, grid, mono)[0] == "ok"


# ──────────────────────────────────────────────────────────────────────────
# the bias curve
# ──────────────────────────────────────────────────────────────────────────

def test_bias_curve_shape_and_monotonicity():
    c = build_bias_curve(GRID, n_obs=N_OBS, window=WINDOW, n_mc=N_MC, seed=1)
    assert isinstance(c, BiasCurve)
    for name in ("GJR", "Cont-Das", "MF-DFA"):
        assert c.mean[name].shape == GRID.shape
        assert c.std[name].shape == GRID.shape
    # GJR's observed Ĥ should rise with true H (allowing MC wobble): endpoints.
    g = c.mean["GJR"]
    assert g[-1] > g[0], "observed Ĥ should grow from rough to smooth true H"


# ──────────────────────────────────────────────────────────────────────────
# the load-bearing round-trip: recover a KNOWN true H
# ──────────────────────────────────────────────────────────────────────────

def test_debias_recovers_known_true_H():
    """At a CLEAN proxy (wide window), de-biasing must recover a known true H.

    window=48 (used by the fast tests above) is the noisy/collapse regime where
    the curve is non-invertible by design; recovery is only meaningful at a
    cleaner proxy, so this test deliberately uses a wide window — the same
    regime the real daily/5m series (window=288) lives in.
    """
    true_H = 0.45
    rt_n_obs, rt_window, rt_mc = 600, 576, 10
    _, S, _ = rough_bergomi_paths(rt_n_obs * rt_window, true_H, n_paths=1,
                                  rng=np.random.default_rng(123))
    x = realized_log_variance(S, rt_window)[0]
    observed_gjr = gjr_hurst(x)

    curve = build_bias_curve(GRID, n_obs=rt_n_obs, window=rt_window,
                             n_mc=rt_mc, seed=7)
    # the clean-proxy GJR curve must be monotone enough to invert
    assert np.all(np.diff(curve.mean["GJR"]) > -0.02)
    implied = invert(observed_gjr, curve.true_grid, curve.mean["GJR"])
    assert np.isfinite(implied), (
        f"observed Ĥ {observed_gjr:.3f} fell outside the curve "
        f"{np.round(curve.mean['GJR'], 3)}")
    assert abs(implied - true_H) < 0.10, (
        f"de-bias gave {implied:.3f} for true {true_H} "
        f"(observed Ĥ was {observed_gjr:.3f})")


# ──────────────────────────────────────────────────────────────────────────
# end-to-end interpret() on a simulated "real" series
# ──────────────────────────────────────────────────────────────────────────

def test_interpret_end_to_end(tmp_path):
    # fabricate a processed RV CSV from a known-H path, then interpret it
    import rv_series
    from kline_verifier import KlineData, INTERVAL_MS
    _, S, _ = rough_bergomi_paths(N_OBS * WINDOW, 0.30, n_paths=1,
                                  rng=np.random.default_rng(5))
    close = S[0]
    n = close.size
    ot = np.arange(n, dtype=np.int64) * INTERVAL_MS["1m"]
    kl = KlineData(open_time=ot, open=close, high=close, low=close, close=close,
                   volume=np.ones(n), close_time=ot + INTERVAL_MS["1m"] - 1,
                   n_trades=np.ones(n, np.int64), n_files=1)
    series = rv_series.build_rv_series(kl, sampling="1m", rv_bar="1h")
    csv = tmp_path / "rv.csv"
    rv_series._save_csv(series, csv)

    interp = interpret(str(csv), window=60, n_mc=N_MC, true_grid=GRID, seed=3)
    assert isinstance(interp, Interpretation)
    assert set(interp.observed) == {"GJR", "Cont-Das", "MF-DFA"}
    assert set(interp.implied) == {"GJR", "Cont-Das", "MF-DFA"}
    # report renders without error
    import io
    buf = io.StringIO()
    interp.report(stream=buf)
    assert "De-biasing" in buf.getvalue()
    assert "MODEL-CONDITIONAL" in buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────
# collapse / observational-equivalence flag
# ──────────────────────────────────────────────────────────────────────────

def test_collapse_flag_on_flat_curve():
    """A nearly-flat bias curve must mark the inversion as ill-posed."""
    grid = np.array([0.05, 0.10, 0.20, 0.30, 0.45, 0.60])
    flat_mean = np.array([0.095, 0.10, 0.105, 0.108, 0.11, 0.112])  # ~flat
    flat_std = np.full_like(flat_mean, 0.03)
    curve = BiasCurve(grid, {"GJR": flat_mean}, {"GJR": flat_std},
                      n_obs=N_OBS, window=WINDOW, n_mc=N_MC)
    from interpret_h import _local_slope, _FLAT_SLOPE
    h = invert(0.103, grid, flat_mean)
    slope = _local_slope(grid, flat_mean, h)
    assert abs(slope) < _FLAT_SLOPE, "flat curve should register as collapse"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
