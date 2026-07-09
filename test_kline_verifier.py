"""
test_kline_verifier.py — tests for the diagnostic kline verifier
================================================================
The verifier never sees real Binance data here. Instead a small generator
(`clean_array`) builds valid 12-column Binance klines on a perfect UTC grid,
and a family of `inject_*` mutators each break exactly ONE invariant. Every
test asserts that the corresponding diagnostic fires with the right count and
that the unrelated diagnostics stay quiet — i.e. the verifier is specific, not
just sensitive.

Two facts get special attention because they are the ones that would silently
bias H downstream:
  * a GAP is reported but does NOT make the data "dirty" (is_clean stays True) —
    a missing minute is a fact about the trading calendar, not a corruption;
  * the 2025-01-01 ms->µs switch: a contiguous ms file + µs file must normalise
    to a single grid with ZERO spurious gap, while still being flagged as mixed.

Run:  pytest test_kline_verifier.py -v
"""

from __future__ import annotations

import io
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from kline_verifier import (
    INTERVAL_MS,
    KlineData,
    KlineDiagnostics,
    load_klines,
    detect_unit,
    verify_klines,
    diagnose,
)

# ──────────────────────────────────────────────────────────────────────────
# synthetic kline generator + on-disk writer
# ──────────────────────────────────────────────────────────────────────────

BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
# Columns that Binance stores as integers (and that MUST be written without
# scientific notation, or a 13–16 digit timestamp would lose its low digits).
_INT_COLS = {0, 6, 8, 11}

# A few known UTC anchors, all minute-aligned, computed (not hand-typed) so the
# arithmetic can't drift. START_MS lands on a clean minute/hour/day boundary.
START_MS = int(datetime(2024, 12, 1, tzinfo=timezone.utc).timestamp() * 1000)
# The last ms minute of 2024 and the first µs minute of 2025 — exactly one
# 1m bar apart, so concatenating them must NOT manufacture a gap.
BOUNDARY_LAST_MS = int(
    datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
JAN_FIRST_MS = int(
    datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def clean_array(start_ms: int, n: int, interval_ms: int = 60_000,
                unit: str = "ms", base: float = 100.0, seed: int = 0
                ) -> np.ndarray:
    """An (n, 12) array of valid Binance klines on a perfect grid.

    Prices are a gentle positive random walk with high = max(open,close)·(1+ε)
    and low = min(open,close)·(1−δ), so the OHLC bounds hold by construction and
    everything is strictly positive. Timestamps start at `start_ms` (or its µs
    image when unit='us') and step by exactly one interval. close_time is the
    exact Binance value open_time + interval − 1 in native units.
    """
    rng = np.random.default_rng(seed)
    close = base * np.cumprod(1.0 + rng.normal(0.0, 0.001, size=n))
    open_ = np.empty(n)
    open_[0] = base
    open_[1:] = close[:-1]                      # this bar opens at last close
    hi = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.005, size=n))
    lo = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.005, size=n))
    vol = rng.uniform(1.0, 100.0, size=n)       # strictly > 0 (no zero-vol bars)
    ntr = rng.integers(1, 1000, size=n)
    qv = vol * close

    ot_ms = start_ms + np.arange(n, dtype=np.int64) * interval_ms
    if unit == "ms":
        ot = ot_ms.astype(np.int64)
        ct = ot + interval_ms - 1
    elif unit == "us":
        ot = ot_ms.astype(np.int64) * 1000
        ct = ot + interval_ms * 1000 - 1
    else:
        raise ValueError(f"unit must be 'ms' or 'us', got {unit!r}")

    arr = np.empty((n, 12), dtype=np.float64)
    arr[:, 0] = ot          # exact: max µs ~1.7e15 < 2^53
    arr[:, 1] = open_
    arr[:, 2] = hi
    arr[:, 3] = lo
    arr[:, 4] = close
    arr[:, 5] = vol
    arr[:, 6] = ct
    arr[:, 7] = qv
    arr[:, 8] = ntr
    arr[:, 9] = vol * 0.5
    arr[:, 10] = qv * 0.5
    arr[:, 11] = 0
    return arr


def write_csv(path, arr: np.ndarray, header: bool = False) -> str:
    """Write `arr` as a Binance-style CSV; integer columns as plain ints."""
    lines = []
    if header:
        lines.append(",".join(BINANCE_COLS))
    for row in arr:
        fields = [str(int(round(v))) if j in _INT_COLS else repr(float(v))
                  for j, v in enumerate(row)]
        lines.append(",".join(fields))
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return str(path)


# ── defect injectors: each breaks exactly one invariant ───────────────────

def inject_gap(arr, start, length):
    """Delete `length` interior rows → a single gap of `length` missing bars."""
    return np.delete(arr, slice(start, start + length), axis=0)


def inject_duplicate(arr, idx):
    """Insert a copy of row `idx` right after it (1 duplicate timestamp)."""
    return np.insert(arr, idx + 1, arr[idx], axis=0)


def inject_nonmonotonic(arr, i):
    """Swap rows i and i+1 → one backward step, grid otherwise intact."""
    out = arr.copy()
    out[[i, i + 1]] = out[[i + 1, i]]
    return out


def inject_zero_price(arr, idx):
    """Zero the OHLC of one row (nonpositive, but bounds still hold)."""
    out = arr.copy()
    out[idx, 1:5] = 0.0
    return out


def inject_ohlc_violation(arr, idx):
    """Push high below low while keeping both positive (bounds break only)."""
    out = arr.copy()
    out[idx, 2] = out[idx, 3] * 0.9      # high = 0.9·low  > 0  but high < low
    return out


def inject_misalign(arr, offset_ms, interval_ms=60_000):
    """Shift open_time AND close_time of every row off the grid by offset_ms."""
    out = arr.copy()
    out[:, 0] += offset_ms
    out[:, 6] += offset_ms
    return out


def inject_negative_volume(arr, idx):
    out = arr.copy()
    out[idx, 5] = -1.0
    return out


# ──────────────────────────────────────────────────────────────────────────
# clean baseline
# ──────────────────────────────────────────────────────────────────────────

def test_clean_array_is_clean(tmp_path):
    path = write_csv(tmp_path / "clean.csv", clean_array(START_MS, 200))
    d = diagnose(path, interval="1m")
    assert d.n_rows == 200
    assert d.is_clean, "a perfectly-formed grid must be reported clean"
    assert d.n_gaps == 0
    assert d.n_duplicate_timestamps == 0
    assert d.n_misaligned == 0
    assert d.monotonic_increasing
    assert d.coverage_fraction == pytest.approx(1.0)
    assert d.unit == "ms"
    assert d.interval_matches


def test_report_runs_on_clean_data(tmp_path):
    path = write_csv(tmp_path / "clean.csv", clean_array(START_MS, 50))
    d = diagnose(path)
    buf = io.StringIO()
    d.report(stream=buf)
    out = buf.getvalue()
    assert "Kline diagnostics" in out
    assert "is_clean = True" in out


# ──────────────────────────────────────────────────────────────────────────
# header detection
# ──────────────────────────────────────────────────────────────────────────

def test_header_autodetected_and_skipped(tmp_path):
    path = write_csv(tmp_path / "h.csv", clean_array(START_MS, 30), header=True)
    d = diagnose(path, interval="1m")          # has_header="auto"
    assert d.n_rows == 30, "header row must not be counted as data"
    assert d.had_header
    assert d.n_header_files == 1
    assert d.is_clean


def test_no_header_autodetected(tmp_path):
    path = write_csv(tmp_path / "nh.csv", clean_array(START_MS, 30), header=False)
    d = diagnose(path, interval="1m")
    assert d.n_rows == 30
    assert not d.had_header


def test_header_force_flags(tmp_path):
    arr = clean_array(START_MS, 30)
    hp = write_csv(tmp_path / "h.csv", arr, header=True)
    np_ = write_csv(tmp_path / "n.csv", arr, header=False)
    assert load_klines(hp, has_header=True).n == 30
    assert load_klines(np_, has_header=False).n == 30


# ──────────────────────────────────────────────────────────────────────────
# timestamp units
# ──────────────────────────────────────────────────────────────────────────

def test_detect_unit_direct():
    ms = np.array([START_MS, START_MS + 60_000], dtype=np.int64)
    us = ms * 1000
    assert detect_unit(ms) == ("ms", 2, 0)
    assert detect_unit(us) == ("us", 0, 2)
    assert detect_unit(np.concatenate([ms, us])) == ("mixed", 2, 2)
    assert detect_unit(np.empty(0, dtype=np.int64)) == ("empty", 0, 0)


def test_pure_microsecond_file(tmp_path):
    path = write_csv(tmp_path / "us.csv", clean_array(JAN_FIRST_MS, 100, unit="us"))
    d = diagnose(path, interval="1m")
    assert d.unit == "us"
    assert not d.normalised_to_ms
    assert d.is_clean
    assert d.n_gaps == 0
    assert d.n_closetime_mismatch == 0


def test_mixed_units_ms_then_us_no_spurious_gap(tmp_path):
    """The headline ms->µs case: a Dec-2024 ms file followed by a Jan-2025 µs
    file is contiguous after normalisation — flagged mixed, but zero gap."""
    ms_arr = clean_array(BOUNDARY_LAST_MS - 4 * 60_000, 5, unit="ms")  # ...23:55–23:59
    us_arr = clean_array(JAN_FIRST_MS, 5, unit="us")                   # 00:00 onward
    ms_path = write_csv(tmp_path / "2024-12.csv", ms_arr)
    us_path = write_csv(tmp_path / "2025-01.csv", us_arr)

    d = diagnose([ms_path, us_path], interval="1m")
    assert d.unit == "mixed"
    assert d.n_ms == 5 and d.n_us == 5
    assert d.normalised_to_ms, "mixed units must record that a normalise happened"
    assert d.monotonic_increasing, "after //1000 the µs block continues the ms grid"
    assert d.n_gaps == 0, "the switch must not manufacture a gap"
    assert d.total_missing_bars == 0
    assert d.is_clean


def test_mixed_units_via_directory(tmp_path):
    """Same as above but discovered via a sorted directory glob."""
    write_csv(tmp_path / "2024-12.csv", clean_array(BOUNDARY_LAST_MS - 4 * 60_000, 5))
    write_csv(tmp_path / "2025-01.csv", clean_array(JAN_FIRST_MS, 5, unit="us"))
    d = diagnose(tmp_path, interval="1m")
    assert d.n_files == 2
    assert d.unit == "mixed"
    assert d.n_gaps == 0


# ──────────────────────────────────────────────────────────────────────────
# gaps & coverage  (a gap is reported but is NOT "dirty")
# ──────────────────────────────────────────────────────────────────────────

def test_gap_counted_but_not_dirty(tmp_path):
    arr = inject_gap(clean_array(START_MS, 60), start=20, length=7)
    path = write_csv(tmp_path / "gap.csv", arr)
    d = diagnose(path, interval="1m")
    assert d.n_gaps == 1
    assert d.total_missing_bars == 7
    assert d.largest_gap_bars == 7
    assert d.coverage_fraction < 1.0
    # design contract: a missing minute is a calendar fact, not corruption.
    assert d.is_clean, "a pure gap must leave is_clean True"


@pytest.mark.parametrize("length", [1, 5, 20])
def test_gap_missing_bar_count(tmp_path, length):
    arr = inject_gap(clean_array(START_MS, 100), start=40, length=length)
    d = diagnose(write_csv(tmp_path / "g.csv", arr), interval="1m")
    assert d.n_gaps == 1
    assert d.total_missing_bars == length


def test_two_gaps(tmp_path):
    arr = clean_array(START_MS, 100)
    arr = inject_gap(arr, start=60, length=3)     # delete later block first
    arr = inject_gap(arr, start=20, length=4)     # so earlier indices stay valid
    d = diagnose(write_csv(tmp_path / "g2.csv", arr), interval="1m")
    assert d.n_gaps == 2
    assert d.total_missing_bars == 7


# ──────────────────────────────────────────────────────────────────────────
# duplicates
# ──────────────────────────────────────────────────────────────────────────

def test_duplicate_timestamp(tmp_path):
    arr = inject_duplicate(clean_array(START_MS, 40), idx=10)
    d = diagnose(write_csv(tmp_path / "dup.csv", arr), interval="1m")
    assert d.n_duplicate_timestamps == 1
    assert d.n_duplicate_rows == 1
    assert not d.is_clean
    # an adjacent duplicate is also an equal-adjacent (non-increasing) step
    assert d.n_equal_adjacent >= 1
    assert d.n_gaps == 0, "the sorted-unique grid is unaffected by a duplicate"


# ──────────────────────────────────────────────────────────────────────────
# ordering
# ──────────────────────────────────────────────────────────────────────────

def test_nonmonotonic_backward_step(tmp_path):
    arr = inject_nonmonotonic(clean_array(START_MS, 40), i=15)
    d = diagnose(write_csv(tmp_path / "swap.csv", arr), interval="1m")
    assert not d.monotonic_increasing
    assert d.n_backward_steps == 1
    assert not d.is_clean
    assert d.n_gaps == 0, "swapping two rows keeps the timestamp set contiguous"
    assert d.n_duplicate_timestamps == 0


# ──────────────────────────────────────────────────────────────────────────
# prices
# ──────────────────────────────────────────────────────────────────────────

def test_zero_price_flagged_not_ohlc(tmp_path):
    arr = inject_zero_price(clean_array(START_MS, 30), idx=5)
    d = diagnose(write_csv(tmp_path / "zero.csv", arr), interval="1m")
    assert d.n_nonpositive_price_rows == 1
    assert d.n_ohlc_violations == 0, "all-zero OHLC still satisfies low<=high"
    assert not d.is_clean


def test_ohlc_violation_flagged_not_nonpositive(tmp_path):
    arr = inject_ohlc_violation(clean_array(START_MS, 30), idx=5)
    d = diagnose(write_csv(tmp_path / "ohlc.csv", arr), interval="1m")
    assert d.n_ohlc_violations == 1
    assert d.n_nonpositive_price_rows == 0, "high=0.9·low is still positive"
    assert not d.is_clean


def test_nan_price(tmp_path):
    arr = clean_array(START_MS, 30)
    arr[7, 4] = np.nan            # NaN close
    d = diagnose(write_csv(tmp_path / "nan.csv", arr), interval="1m")
    assert d.n_nan_price_rows == 1
    assert not d.is_clean


# ──────────────────────────────────────────────────────────────────────────
# volume
# ──────────────────────────────────────────────────────────────────────────

def test_negative_volume_is_fail(tmp_path):
    arr = inject_negative_volume(clean_array(START_MS, 30), idx=9)
    d = diagnose(write_csv(tmp_path / "negv.csv", arr), interval="1m")
    assert d.n_negative_volume == 1
    assert not d.is_clean


def test_zero_volume_is_only_a_note(tmp_path):
    arr = clean_array(START_MS, 30)
    arr[9, 5] = 0.0               # zero volume is legal (no trades that minute)
    d = diagnose(write_csv(tmp_path / "zerov.csv", arr), interval="1m")
    assert d.n_zero_volume == 1
    assert d.n_negative_volume == 0
    assert d.is_clean, "a zero-volume bar is a NOTE, not a defect"


# ──────────────────────────────────────────────────────────────────────────
# alignment & interval inference
# ──────────────────────────────────────────────────────────────────────────

def test_misaligned_timestamps(tmp_path):
    arr = inject_misalign(clean_array(START_MS, 40), offset_ms=30_000)
    d = diagnose(write_csv(tmp_path / "mis.csv", arr), interval="1m")
    assert d.n_misaligned == 40, "every row sits 30s off the minute grid"
    assert not d.is_clean
    assert d.n_gaps == 0, "a uniform shift preserves spacing"
    assert d.n_closetime_mismatch == 0, "open_time and close_time shifted together"


def test_interval_inference_mismatch(tmp_path):
    """5-minute data declared as 1m: inference should disagree with the label."""
    arr = clean_array(START_MS, 40, interval_ms=INTERVAL_MS["5m"])
    path = write_csv(tmp_path / "5m.csv", arr)
    d = diagnose(path, interval="1m")
    assert not d.interval_matches
    assert d.inferred_interval_ms == INTERVAL_MS["5m"]
    # and the same data verified at its true interval is clean
    d5 = diagnose(path, interval="5m")
    assert d5.interval_matches
    assert d5.is_clean


def test_closetime_mismatch(tmp_path):
    arr = clean_array(START_MS, 30)
    arr[12, 6] += 1000           # close_time one second too long
    d = diagnose(write_csv(tmp_path / "ct.csv", arr), interval="1m")
    assert d.n_closetime_mismatch == 1


# ──────────────────────────────────────────────────────────────────────────
# empty / degenerate inputs
# ──────────────────────────────────────────────────────────────────────────

def test_header_only_file(tmp_path):
    p = tmp_path / "empty.csv"
    with open(p, "w") as f:
        f.write(",".join(BINANCE_COLS) + "\n")
    d = diagnose(p, interval="1m")
    assert d.n_rows == 0
    buf = io.StringIO()
    d.report(stream=buf)
    assert "no data rows" in buf.getvalue()


def test_single_row_file(tmp_path):
    arr = clean_array(START_MS, 1)
    d = diagnose(write_csv(tmp_path / "one.csv", arr), interval="1m")
    assert d.n_rows == 1
    assert d.n_gaps == 0
    assert d.is_clean


def test_wrong_column_count_raises(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w") as f:
        f.write("1733011200000,100,101,99\n")   # only 4 columns
    with pytest.raises(ValueError):
        load_klines(p)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_klines("does_not_exist_12345.csv")


# ──────────────────────────────────────────────────────────────────────────
# RVL-036 + RVL-037: load_klines must not silently drop data on a BOM'd or
# blank-first-line CSV (and must not regress the good paths).
# ──────────────────────────────────────────────────────────────────────────

def _kline_csv_text(arr: np.ndarray, header: bool = False) -> str:
    """The exact text write_csv produces, so callers can prepend a BOM/blank."""
    lines = []
    if header:
        lines.append(",".join(BINANCE_COLS))
    for row in arr:
        fields = [str(int(round(v))) if j in _INT_COLS else repr(float(v))
                  for j, v in enumerate(row)]
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


def test_load_bom_headerless_csv(tmp_path):
    """RVL-036: a UTF-8 BOM'd headerless CSV loads all N rows, had_header=False."""
    arr = clean_array(START_MS, 30)
    p = tmp_path / "bom.csv"
    with open(p, "w", encoding="utf-8-sig") as f:      # utf-8-sig prepends the BOM
        f.write(_kline_csv_text(arr, header=False))
    d = load_klines(p)
    assert d.n == 30, "the BOM'd first bar must not be misread as a header and dropped"
    assert not d.had_header, "a BOM'd data row is not a header"


def test_load_leading_blank_line_csv(tmp_path):
    """RVL-037: a leading blank line must not classify the whole file as empty."""
    arr = clean_array(START_MS, 30)
    p = tmp_path / "blank.csv"
    with open(p, "w") as f:
        f.write("\n" + _kline_csv_text(arr, header=False))
    d = load_klines(p)
    assert d.n == 30, "a leading blank line must not drop the whole file"


def test_load_real_header_still_loads(tmp_path):
    """Guard: a genuine header row still loads correctly with had_header=True."""
    arr = clean_array(START_MS, 30)
    p = tmp_path / "hdr.csv"
    with open(p, "w") as f:
        f.write(_kline_csv_text(arr, header=True))
    d = load_klines(p)
    assert d.n == 30 and d.had_header


def test_load_truly_empty_file(tmp_path):
    """Guard: a genuinely empty (0-byte) file yields a well-formed empty KlineData."""
    p = tmp_path / "empty.csv"
    p.write_text("")
    d = load_klines(p)
    assert isinstance(d, KlineData)
    assert d.n == 0 and not d.had_header


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
