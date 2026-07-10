"""
test_rv_series.py — tests for the realized-variance series builder
==================================================================
The load-bearing test is `test_matches_phase_a_rung1`: it imports the ACTUAL
Layer 1c proxy functions and asserts this module reproduces them byte-for-byte,
so a real-data log-RV series is the same object whose proxy-bias Phase A mapped.
The rest pin the real-data concerns the simulation never faces: calendar
bucketing, calendar-anchored subsampling, the ms->µs normalisation (a µs file
must yield the SAME RV as its ms image), gap accounting, and the honest
series-length report bands.

Run:  pytest test_rv_series.py -v
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless: before any mpl import

import io
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from kline_verifier import INTERVAL_MS, KlineData
from rv_series import (
    log_realized_variance,
    log_bipower_variance,
    build_rv_series,
    RVSeries,
)
# RVL-008: layer1c_roughness_audit is a first-party module — import it plainly
# (NOT pytest.importorskip), so a broken import makes the load-bearing round-trip
# test FAIL rather than silently SKIP behind a green suite.
import layer1c_roughness_audit as audit

DAY_MS = INTERVAL_MS["1d"]
MIN_MS = INTERVAL_MS["1m"]
MIDNIGHT = int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def gbm_close(n, base=100.0, seed=0):
    """A strictly-positive price path (geometric random walk)."""
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(0.0, 0.001, size=n)))


def make_klines(start_ms, n, unit="ms", close=None, interval_ms=MIN_MS):
    """A minimal contiguous KlineData (only close + timestamps matter for RV)."""
    if close is None:
        close = gbm_close(n)
    close = np.asarray(close, dtype=np.float64)
    ot_ms = start_ms + np.arange(n, dtype=np.int64) * interval_ms
    if unit == "ms":
        ot = ot_ms
        ct = ot + interval_ms - 1
    elif unit == "us":
        ot = ot_ms * 1000
        ct = ot + interval_ms * 1000 - 1
    else:
        raise ValueError(unit)
    f = close.copy()
    return KlineData(open_time=ot.astype(np.int64), open=f, high=f, low=f,
                     close=f, volume=np.ones(n), close_time=ct.astype(np.int64),
                     n_trades=np.ones(n, np.int64), n_files=1)


# ──────────────────────────────────────────────────────────────────────────
# the load-bearing test: identical convention to Layer 1c Rung 1
# ──────────────────────────────────────────────────────────────────────────

def test_matches_phase_a_rung1():
    """Our proxies must equal layer1c's realized/bipower log-variance exactly."""
    rng = np.random.default_rng(7)
    S = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=(4, 3001)), axis=1))
    for window in (10, 30, 100):
        np.testing.assert_array_equal(
            log_realized_variance(S, window),
            audit.realized_log_variance(S, window),
            err_msg=f"RV proxy diverged from Phase A at window={window}")
        np.testing.assert_array_equal(
            log_bipower_variance(S, window),
            audit.bipower_log_variance(S, window),
            err_msg=f"bipower diverged from Phase A at window={window}")


def test_primitive_1d_matches_2d_row():
    S = gbm_close(2001, seed=3)
    one_d = log_realized_variance(S, 50)
    two_d = log_realized_variance(S[None, :], 50)
    assert one_d.ndim == 1
    np.testing.assert_array_equal(one_d, two_d[0])


# ──────────────────────────────────────────────────────────────────────────
# calendar bucketing
# ──────────────────────────────────────────────────────────────────────────

def test_daily_bars_count_and_alignment():
    n_days = 5
    kl = make_klines(MIDNIGHT, n_days * 1440)         # contiguous 1-min, 5 days
    s = build_rv_series(kl, sampling="1m", rv_bar="1d")
    assert s.n_obs == n_days
    # each bar starts on a UTC midnight, one day apart
    assert s.period_start_ms[0] == MIDNIGHT
    assert np.all(np.diff(s.period_start_ms) == DAY_MS)
    # interior days see one return per minute (1440); day 0 lacks the carry-in
    assert s.n_returns[1] == 1440
    assert s.n_returns[0] == 1439
    assert s.n_gap_spanning.sum() == 0
    assert s.expected_returns_per_bar == 1440


def test_five_minute_sampling_anchors_to_grid():
    kl = make_klines(MIDNIGHT, 3 * 1440)
    s = build_rv_series(kl, sampling="5m", rv_bar="1d")
    assert s.expected_returns_per_bar == 288        # 1440 / 5
    assert s.n_returns[1] == 288                     # interior full day
    # coarser sampling => strictly fewer returns than the 1m proxy
    s1 = build_rv_series(kl, sampling="1m", rv_bar="1d")
    assert s.n_returns[1] < s1.n_returns[1]


def test_hourly_bars():
    kl = make_klines(MIDNIGHT, 2 * 1440)             # 2 days = 48 hours
    s = build_rv_series(kl, sampling="1m", rv_bar="1h")
    assert s.n_obs == 48
    assert s.expected_returns_per_bar == 60
    assert s.n_returns[1] == 60


# ──────────────────────────────────────────────────────────────────────────
# the loop-closure: a µs file must give the SAME RV as its ms image
# ──────────────────────────────────────────────────────────────────────────

def test_microsecond_timestamps_give_identical_rv():
    close = gbm_close(3 * 1440, seed=11)
    ms = build_rv_series(make_klines(MIDNIGHT, 3 * 1440, unit="ms", close=close))
    us = build_rv_series(make_klines(MIDNIGHT, 3 * 1440, unit="us", close=close))
    assert ms.n_obs == us.n_obs == 3
    np.testing.assert_array_equal(ms.period_start_ms, us.period_start_ms)
    np.testing.assert_allclose(ms.log_rv, us.log_rv, rtol=0, atol=0)
    np.testing.assert_array_equal(ms.n_returns, us.n_returns)


# ──────────────────────────────────────────────────────────────────────────
# gaps are accounted, not swallowed
# ──────────────────────────────────────────────────────────────────────────

def test_gap_is_flagged_and_localised():
    close = gbm_close(3 * 1440, seed=5)
    kl_full = make_klines(MIDNIGHT, 3 * 1440, close=close)
    # drop 30 interior minutes inside day 1 (indices 1440+100 .. +130)
    drop = slice(1440 + 100, 1440 + 130)
    keep = np.ones(close.size, bool); keep[drop] = False
    kl = KlineData(open_time=kl_full.open_time[keep], open=kl_full.open[keep],
                   high=kl_full.high[keep], low=kl_full.low[keep],
                   close=kl_full.close[keep], volume=kl_full.volume[keep],
                   close_time=kl_full.close_time[keep],
                   n_trades=kl_full.n_trades[keep], n_files=1)
    s = build_rv_series(kl, sampling="1m", rv_bar="1d")
    assert s.n_obs == 3
    # the gap sits in day 1: that bar loses ~30 returns and flags the carry-over
    assert s.n_returns[1] < 1440
    assert s.n_gap_spanning[1] >= 1
    # days 0 and 2 are untouched
    assert s.n_gap_spanning[0] == 0
    assert s.n_gap_spanning[2] == 0


# ──────────────────────────────────────────────────────────────────────────
# bipower
# ──────────────────────────────────────────────────────────────────────────

def test_bipower_present_and_toggleable():
    kl = make_klines(MIDNIGHT, 3 * 1440)
    with_bv = build_rv_series(kl, rv_bar="1d", jump_robust=True)
    assert np.isfinite(with_bv.log_bv).all()
    without = build_rv_series(kl, rv_bar="1d", jump_robust=False)
    assert not np.isfinite(without.log_bv).any()


# ──────────────────────────────────────────────────────────────────────────
# clean_log_rv drops sparse bars
# ──────────────────────────────────────────────────────────────────────────

def test_clean_log_rv_drops_sparse():
    # 2 full days + a stub partial day (200 minutes) -> the stub is sparse
    kl = make_klines(MIDNIGHT, 2 * 1440 + 200)
    s = build_rv_series(kl, sampling="1m", rv_bar="1d")
    assert s.n_obs == 3
    kept = s.clean_log_rv(min_fraction=0.5)            # 0.5*1440 = 720
    assert kept.size == 2                              # the 200-min day is dropped


# ──────────────────────────────────────────────────────────────────────────
# report bands (length honesty) — unit-tested on the container directly
# ──────────────────────────────────────────────────────────────────────────

def _series_with_n(n):
    return RVSeries(
        period_start_ms=np.arange(n, dtype=np.int64) * DAY_MS + MIDNIGHT,
        log_rv=np.full(n, -9.0), log_bv=np.full(n, -9.0),
        n_returns=np.full(n, 1440, np.int64),
        n_gap_spanning=np.zeros(n, np.int64),
        sampling="1m", rv_bar="1d", expected_returns_per_bar=1440,
        n_input_rows=n * 1440, unit="ms")


@pytest.mark.parametrize("n,flag", [(91, "TOO FEW"), (500, "usable"), (1500, "enough")])
def test_report_length_bands(n, flag):
    buf = io.StringIO()
    _series_with_n(n).report(stream=buf)
    out = buf.getvalue()
    assert flag in out
    assert "Rung 1" in out, "report must remind the reader this is the RV proxy"


def test_report_runs_on_empty():
    s = build_rv_series(make_klines(MIDNIGHT, 1))      # <2 rows -> empty
    assert s.n_obs == 0
    buf = io.StringIO()
    s.report(stream=buf)
    assert "no RV observations" in buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────
# input validation
# ──────────────────────────────────────────────────────────────────────────

def test_sampling_must_divide_base():
    kl = make_klines(MIDNIGHT, 100)
    with pytest.raises(ValueError):
        build_rv_series(kl, sampling="1s", base_interval="1m")   # 1s < 1m


def test_rv_bar_must_be_multiple_of_sampling():
    kl = make_klines(MIDNIGHT, 100)
    with pytest.raises(ValueError):
        build_rv_series(kl, sampling="1h", rv_bar="30m")          # 30m < 1h


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
