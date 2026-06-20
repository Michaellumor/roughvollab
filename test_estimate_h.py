"""
test_estimate_h.py — tests for the estimator runner
===================================================
The runner must not change what the estimators say — `test_analyze_matches_
direct_estimators` pins that analyze() returns exactly gjr/pvariation/mfdfa's
own H. The rest check the value the runner ADDS: trust signals, the rough-vs-
smooth disagreement logic, the processed-CSV round-trip, and the sweep /
stability shapes. Recovery on a simulated smooth null confirms the whole
simulate->estimate path is wired correctly.

Run:  pytest test_estimate_h.py -v
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import io
import sys

import numpy as np
import pytest

from roughvol_core import rough_log_variance_paths
from kline_verifier import KlineData, INTERVAL_MS
from layer1c_roughness_audit import gjr_hurst, pvariation_hurst, mfdfa_hurst
import rv_series
from estimate_h import (
    EstimateResult,
    analyze,
    sampling_sweep,
    subwindow_stability,
    load_log_rv_csv,
    _disagreement_lines,
    _looks_like_rv_csv,
    _print_report,
)

MIN_MS = INTERVAL_MS["1m"]


def sim_log_vol(H, n=16384, seed=0):
    _, logV = rough_log_variance_paths(n, H, n_paths=1,
                                       rng=np.random.default_rng(seed))
    return logV[0]


def make_klines(start_ms, n, close=None):
    if close is None:
        rng = np.random.default_rng(1)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=n)))
    close = np.asarray(close, float)
    ot = start_ms + np.arange(n, dtype=np.int64) * MIN_MS
    return KlineData(open_time=ot, open=close, high=close, low=close,
                     close=close, volume=np.ones(n),
                     close_time=ot + MIN_MS - 1,
                     n_trades=np.ones(n, np.int64), n_files=1)


# ──────────────────────────────────────────────────────────────────────────
# faithfulness: the runner must echo the estimators exactly
# ──────────────────────────────────────────────────────────────────────────

def test_analyze_matches_direct_estimators():
    x = sim_log_vol(0.3, seed=2)
    results = {r.name: r.H for r in analyze(x)}
    assert results["GJR"] == gjr_hurst(x)
    assert results["Cont-Das"] == pvariation_hurst(x)
    assert results["MF-DFA"] == mfdfa_hurst(x)


def test_recovery_on_smooth_null():
    """All three should land near H=0.5 on a simulated Markov (smooth) path."""
    x = sim_log_vol(0.5, seed=0)
    for r in analyze(x):
        assert r.ok
        assert abs(r.H - 0.5) < 0.1, f"{r.name} gave {r.H:.3f} for true 0.5"


# ──────────────────────────────────────────────────────────────────────────
# trust signals
# ──────────────────────────────────────────────────────────────────────────

def test_trust_signals_render():
    x = sim_log_vol(0.3, seed=4)
    notes = {r.name: r.note for r in analyze(x, multifractal=True)}
    assert "R²" in notes["GJR"]
    assert "p*" in notes["Cont-Das"]
    assert "h(2)" in notes["MF-DFA"]
    assert "Δh(q)" in notes["MF-DFA"]          # multifractal probe ran


# ──────────────────────────────────────────────────────────────────────────
# disagreement logic
# ──────────────────────────────────────────────────────────────────────────

def _mk(name, H, ok=True):
    return EstimateResult(name, H, ok)


def test_disagreement_all_rough():
    out = "\n".join(_disagreement_lines(
        [_mk("GJR", 0.12), _mk("Cont-Das", 0.15), _mk("MF-DFA", 0.08)]))
    assert "below 0.5" in out
    assert "ROUGH" in out


def test_disagreement_straddles():
    out = "\n".join(_disagreement_lines(
        [_mk("GJR", 0.45), _mk("Cont-Das", 0.55), _mk("MF-DFA", 0.40)]))
    assert "STRADDLE" in out


def test_disagreement_all_smooth():
    out = "\n".join(_disagreement_lines(
        [_mk("GJR", 0.52), _mk("Cont-Das", 0.55), _mk("MF-DFA", 0.51)]))
    assert "no roughness" in out


def test_disagreement_needs_two():
    out = "\n".join(_disagreement_lines(
        [_mk("GJR", np.nan, ok=False), _mk("Cont-Das", 0.5)]))
    assert "fewer than two" in out


# ──────────────────────────────────────────────────────────────────────────
# nan handling on a too-short series
# ──────────────────────────────────────────────────────────────────────────

def test_short_series_does_not_crash():
    x = sim_log_vol(0.3, n=40, seed=0)[:40]
    results = analyze(x)
    assert len(results) == 3                  # all three reported, none raised
    buf = io.StringIO()
    _print_report(results, x.size, stream=buf)   # report renders regardless
    assert "Hurst estimates" in buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────
# processed-CSV round-trip
# ──────────────────────────────────────────────────────────────────────────

def test_csv_roundtrip(tmp_path):
    kl = make_klines(0, 5 * 1440)
    series = rv_series.build_rv_series(kl, sampling="5m", rv_bar="1d")
    path = tmp_path / "rv.csv"
    rv_series._save_csv(series, path)
    assert _looks_like_rv_csv(path)
    log_rv, log_bv, ms = load_log_rv_csv(path)
    np.testing.assert_allclose(log_rv, series.log_rv)
    assert ms.size == series.n_obs


def test_looks_like_rv_csv_rejects_klines(tmp_path):
    p = tmp_path / "klines.csv"
    p.write_text("1700000000000,100,101,99,100,1,1700000059999,100,5,0.5,50,0\n")
    assert not _looks_like_rv_csv(p)


# ──────────────────────────────────────────────────────────────────────────
# sampling sweep
# ──────────────────────────────────────────────────────────────────────────

def test_sampling_sweep_shape():
    kl = make_klines(0, 4 * 1440)
    rows = sampling_sweep(kl, samplings=("1m", "5m"), rv_bar="1d")
    assert len(rows) == 2
    for s, n_obs, res in rows:
        assert set(res.keys()) == {"GJR", "Cont-Das", "MF-DFA"}
        assert n_obs == 4                      # n_obs = days, same across sampling


# ──────────────────────────────────────────────────────────────────────────
# sub-window stability
# ──────────────────────────────────────────────────────────────────────────

def test_subwindow_stability_shape():
    x = sim_log_vol(0.3, n=16384, seed=0)
    stab = subwindow_stability(x, k_chunks=4)
    assert set(stab.keys()) == {"GJR", "Cont-Das", "MF-DFA"}
    for name, (mean, sd, vals) in stab.items():
        assert len(vals) == 4                  # one estimate per chunk


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
