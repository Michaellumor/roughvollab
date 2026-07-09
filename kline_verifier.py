"""
kline_verifier.py — diagnostic verifier for Binance klines (Phase B, brick 2)
=============================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

Loads a Binance 1-minute kline CSV (or a set of them) and honestly reports
every data-quality issue it finds. It is DIAGNOSTIC, not a gate: it never
silently repairs the data and never returns a bare pass/fail. It returns a
KlineDiagnostics record listing what is wrong, how much, and where — so the
decision about whether a series is fit for a given estimator stays with the
researcher (ROADMAP Phase B: "the deliverable is what can be concluded at
stated confidence, not a verdict").

Why this module exists
----------------------
Real exchange dumps violate the tidy assumptions Phase A's estimators were
calibrated under. The defects that would silently bias Ĥ if fed in raw:

  * Gaps — exchange downtime / maintenance leaves missing minutes; treating a
    gap as a contiguous step corrupts the increment distribution.
  * Duplicates & non-monotone timestamps — concatenation or feed glitches.
  * The Binance ms→µs switch (2025-01-01) — open_time precision changed from
    milliseconds to microseconds. Naively concatenating a Dec-2024 file (ms)
    with a Jan-2025 file (µs) and reading both as ms places the 2025 bars in
    the year ~56000 and manufactures an astronomical "gap". This verifier
    detects mixed units, normalises to ms for analysis, and SAYS SO.
  * Zero / negative prices and broken OHLC invariants — malformed rows.

Catching these here, with counts and locations, is the honest-sourcing step
that lets Phase B apply each estimator only where its inputs are trustworthy.

The severity tags ([OK]/[NOTE]/[WARN]/[FAIL]) in the printed report are
descriptive aids, NOT a verdict — `.is_clean` is provided only as a
convenience flag. A gap, for instance, is a fact about the asset's trading
calendar, not necessarily an error.

Public API
----------
  load_klines(path_or_paths, has_header="auto")          -> KlineData
  detect_unit(open_time)                                 -> ("ms"|"us"|"mixed"|"empty", n_ms, n_us)
  verify_klines(data, interval="1m")                     -> KlineDiagnostics
  diagnose(path_or_paths, interval="1m")                 -> KlineDiagnostics
  KlineData, KlineDiagnostics  (dataclasses; .report() prints the summary)

CLI
---
  python kline_verifier.py data/spot/klines/BTCUSDT/1m/*.csv --interval 1m

Binance kline CSV schema (12 columns, no header in older files):
  open_time, open, high, low, close, volume, close_time, quote_volume,
  n_trades, taker_buy_base, taker_buy_quote, ignore
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "INTERVAL_MS",
    "KlineData",
    "KlineDiagnostics",
    "load_klines",
    "detect_unit",
    "canonical_ms",
    "verify_klines",
    "diagnose",
]

log = logging.getLogger("kline_verifier")

# Interval string -> milliseconds. Used as the *declared* grid; the verifier
# also infers the interval from the data and flags any disagreement.
INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000,
}

# Below this, an open_time is milliseconds; at/above it, microseconds. The two
# regimes for realistic dates (2017–2100) are ~1e12–4e12 (ms, 13 digits) and
# ~1.5e15–2.6e15 (µs, 16 digits); 1e14 sits cleanly in the empty gap between.
_US_THRESHOLD = 10**14

# float64 exactly represents integers below 2^53 ≈ 9.0e15; µs timestamps stay
# well under this (year ~2255 before µs reaches it), so loadtxt's float parse
# round-trips losslessly to int64. We assert this rather than assume it.
_MAX_EXACT_INT = 2**53

_N_COLS = 12
_MAX_EXAMPLES = 5  # cap stored example rows per issue (report stays readable)


@dataclass
class KlineData:
    """Typed columns of a (possibly concatenated) kline CSV.

    Timestamps are kept in their NATIVE units (ms or µs, as on disk) in
    open_time / close_time. `t_ms` is the canonical millisecond view used for
    all timing analysis (µs rows divided by 1000); `is_us` marks which rows
    were microseconds. Building t_ms is the single place unit normalisation
    happens, and verify_klines reports when it was necessary.
    """
    open_time: np.ndarray      # int64, native units
    open: np.ndarray           # float64
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    close_time: np.ndarray     # int64, native units
    n_trades: np.ndarray       # int64
    had_header: bool = False
    n_header_files: int = 0
    n_files: int = 1

    @property
    def n(self) -> int:
        return int(self.open_time.shape[0])


def _looks_like_header(line: str) -> bool:
    """True if the first comma-field of `line` is not a parseable number."""
    first = line.split(",", 1)[0].strip().strip('"')
    if not first:
        return True
    try:
        float(first)
        return False
    except ValueError:
        return True


def _resolve_paths(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]]
                   ) -> list[Path]:
    """Normalise the input into a sorted list of CSV paths.

    Accepts a single file, a directory (its *.csv, sorted), a glob string, or
    an explicit sequence of paths (order preserved — used to probe ordering).
    """
    if isinstance(path_or_paths, (list, tuple)):
        return [Path(p) for p in path_or_paths]
    p = Path(path_or_paths)
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    if any(ch in str(path_or_paths) for ch in "*?["):
        return sorted(Path(x) for x in _glob.glob(str(path_or_paths)))
    return [p]


def load_klines(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]],
                has_header: Union[bool, str] = "auto") -> KlineData:
    """Load one or many Binance kline CSVs into a KlineData.

    Multiple files are concatenated in the given order (a sorted directory/glob
    yields chronological order for Binance's YYYY-MM[-DD] filenames). Within-
    and across-file ordering is preserved so verify_klines can detect ordering
    defects — concatenation order is exactly where the ms→µs boundary bites.

    has_header: True/False to force, or "auto" to sniff each file's first line.
    """
    paths = _resolve_paths(path_or_paths)
    if not paths:
        raise FileNotFoundError(f"no CSV files matched: {path_or_paths!r}")

    blocks: list[np.ndarray] = []
    n_header_files = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        # RVL-036/037: utf-8-sig strips a UTF-8 BOM (PowerShell/Excel re-saves add
        # one) so it can't glue to the first field and be misread as a header;
        # skip leading blank lines so a blank first line doesn't classify the
        # whole file as empty. n_blank feeds skiprows for the load below.
        n_blank = 0
        with open(path, "r", encoding="utf-8-sig") as f:
            first = f.readline()
            while first != "" and not first.strip():
                n_blank += 1
                first = f.readline()
        if not first.strip():
            log.warning("empty file, skipping: %s", path)
            continue
        header = _looks_like_header(first) if has_header == "auto" else bool(has_header)
        if header:
            n_header_files += 1
        ncol = len([c for c in first.rstrip("\n").split(",")])
        if ncol < _N_COLS:
            raise ValueError(
                f"{path}: expected {_N_COLS} kline columns, found {ncol} "
                f"— is this a kline file? (aggTrades/trades have a different schema)"
            )
        # A header-only or empty file yields no data rows; that is a legitimate
        # (if degenerate) input we report on, not an error — so silence the
        # specific numpy "input contained no data" warning rather than leak it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
            arr = np.loadtxt(path, delimiter=",", encoding="utf-8-sig",
                             skiprows=n_blank + (1 if header else 0),
                             usecols=range(_N_COLS), ndmin=2)
        if arr.size:
            blocks.append(arr)

    if not blocks:
        # All files empty / header-only: return a well-formed empty KlineData.
        z_i = np.empty(0, dtype=np.int64)
        z_f = np.empty(0, dtype=np.float64)
        return KlineData(z_i, z_f, z_f, z_f, z_f, z_f, z_i, z_i,
                         had_header=n_header_files > 0,
                         n_header_files=n_header_files, n_files=len(paths))

    arr = np.vstack(blocks)

    ot = arr[:, 0]
    ct = arr[:, 6]
    if np.nanmax(np.abs(ot)) >= _MAX_EXACT_INT or np.nanmax(np.abs(ct)) >= _MAX_EXACT_INT:
        # Defensive: would only trip for µs timestamps past ~year 2255.
        raise ValueError("timestamp exceeds float64 exact-integer range "
                         "(2^53); load via integer parsing instead")

    return KlineData(
        open_time=ot.astype(np.int64),
        open=arr[:, 1].astype(np.float64),
        high=arr[:, 2].astype(np.float64),
        low=arr[:, 3].astype(np.float64),
        close=arr[:, 4].astype(np.float64),
        volume=arr[:, 5].astype(np.float64),
        close_time=ct.astype(np.int64),
        n_trades=arr[:, 8].astype(np.int64),
        had_header=n_header_files > 0,
        n_header_files=n_header_files,
        n_files=len(paths),
    )


def canonical_ms(open_time: np.ndarray) -> np.ndarray:
    """Map raw open_time values to canonical milliseconds.

    µs rows (>= _US_THRESHOLD, the 2025-01-01 switch) are floored by 1000; ms
    rows pass through. This is the SINGLE source of truth for the ms->µs
    normalisation: the verifier uses it for all timing analysis, and ANY
    downstream consumer that builds a returns/RV series from klines must apply
    exactly this mapping — otherwise a Dec-2024 / Jan-2025 boundary places the
    2025 bars in year ~56000 and manufactures an astronomical fake gap. µs
    minute bars are exact ms*1000, so // 1000 recovers ms losslessly.
    """
    ot = np.asarray(open_time)
    is_us = ot >= _US_THRESHOLD
    return np.where(is_us, ot // 1000, ot).astype(np.int64)


def detect_unit(open_time: np.ndarray) -> tuple[str, int, int]:
    """Classify timestamp units by magnitude.

    Returns (unit, n_ms, n_us) where unit is 'ms', 'us', 'mixed' or 'empty'.
    Per-value classification (not a single global test) is what lets us catch
    a file set that straddles the 2025-01-01 ms→µs boundary.
    """
    if open_time.size == 0:
        return "empty", 0, 0
    is_us = open_time >= _US_THRESHOLD
    n_us = int(is_us.sum())
    n_ms = int(open_time.size - n_us)
    if n_us == 0:
        return "ms", n_ms, 0
    if n_ms == 0:
        return "us", 0, n_us
    return "mixed", n_ms, n_us


@dataclass
class KlineDiagnostics:
    """Every finding from verify_klines. Counts + a few example rows each.

    This is a description of the data, not a verdict. `.report()` prints a
    severity-tagged human summary; `.is_clean` is a convenience flag only.
    """
    n_rows: int
    interval: str
    interval_ms: int

    # Provenance / format
    had_header: bool = False
    n_header_files: int = 0
    n_files: int = 1

    # Timestamp units
    unit: str = "unknown"
    n_ms: int = 0
    n_us: int = 0
    normalised_to_ms: bool = False    # True iff mixed units forced a normalise

    # Coverage / span
    span_start: Optional[str] = None
    span_end: Optional[str] = None
    n_unique_timestamps: int = 0
    expected_bars: int = 0
    coverage_fraction: float = 1.0

    # Interval inference
    inferred_interval_ms: Optional[int] = None
    interval_matches: bool = True

    # Ordering
    monotonic_increasing: bool = True
    n_backward_steps: int = 0
    n_equal_adjacent: int = 0
    backward_examples: list = field(default_factory=list)

    # Duplicates
    n_duplicate_timestamps: int = 0
    n_duplicate_rows: int = 0
    duplicate_examples: list = field(default_factory=list)

    # Gaps
    n_gaps: int = 0
    total_missing_bars: int = 0
    largest_gap_bars: int = 0
    largest_gap_at: Optional[str] = None
    gap_examples: list = field(default_factory=list)
    n_sub_interval: int = 0
    sub_interval_examples: list = field(default_factory=list)

    # Alignment
    n_misaligned: int = 0
    misaligned_examples: list = field(default_factory=list)

    # Prices
    n_nonpositive_price_rows: int = 0
    nonpositive_examples: list = field(default_factory=list)
    n_ohlc_violations: int = 0
    ohlc_examples: list = field(default_factory=list)
    n_nan_price_rows: int = 0

    # Volume
    n_negative_volume: int = 0
    n_zero_volume: int = 0
    n_nan_volume: int = 0
    negative_volume_examples: list = field(default_factory=list)

    # close_time
    n_closetime_mismatch: int = 0
    closetime_examples: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """No WARN/FAIL-level findings. Gaps and notes do NOT count as dirty."""
        return (self.monotonic_increasing
                and self.n_duplicate_timestamps == 0
                and self.n_misaligned == 0
                and self.n_nonpositive_price_rows == 0
                and self.n_ohlc_violations == 0
                and self.n_nan_price_rows == 0
                and self.n_negative_volume == 0
                and self.n_nan_volume == 0)

    def report(self, stream=sys.stdout) -> "KlineDiagnostics":
        """Print a severity-tagged human-readable summary; return self."""
        def line(s: str = "") -> None:
            print(s, file=stream)

        line("\n" + "=" * 70)
        line(f"  Kline diagnostics — {self.n_rows:,} rows, "
             f"interval '{self.interval}', {self.n_files} file(s)")
        line("=" * 70)
        if self.n_rows == 0:
            line("  [FAIL] no data rows loaded.")
            line("=" * 70 + "\n")
            return self

        line(f"  span        : {self.span_start}  ->  {self.span_end}")
        line(f"  timestamps  : {self.n_unique_timestamps:,} unique "
             f"of {self.n_rows:,} rows")

        # Format / units
        if self.had_header:
            line(f"  [NOTE] header row detected in {self.n_header_files} file(s) "
                 f"(skipped on load).")
        if self.unit == "mixed":
            line(f"  [WARN] mixed timestamp units: {self.n_ms:,} ms + "
                 f"{self.n_us:,} µs rows (the 2025-01-01 ms->µs switch).")
            line("         normalised to ms for analysis below — your own RV "
                 "loader must do the same.")
        elif self.unit in ("ms", "us"):
            line(f"  [OK]   timestamp unit: {self.unit}.")

        # Interval
        if self.inferred_interval_ms is not None and not self.interval_matches:
            line(f"  [WARN] inferred interval {self.inferred_interval_ms} ms "
                 f"!= declared {self.interval_ms} ms — wrong --interval?")

        # Coverage / gaps
        cov = self.coverage_fraction * 100.0
        tag = "OK" if self.n_gaps == 0 else "WARN"
        line(f"  [{tag}] coverage: {cov:.3f}%  "
             f"({self.expected_bars - self.total_missing_bars:,} present / "
             f"{self.expected_bars:,} expected over span)")
        if self.n_gaps:
            line(f"         {self.n_gaps:,} gap(s), {self.total_missing_bars:,} "
                 f"missing bar(s); largest {self.largest_gap_bars:,} bars "
                 f"at {self.largest_gap_at}")
            for s, e, miss in self.gap_examples:
                line(f"           gap: {s} -> {e}  ({miss:,} missing)")
        if self.n_sub_interval:
            line(f"  [WARN] {self.n_sub_interval:,} sub-interval step(s) "
                 f"(spacing < one bar) — irregular timestamps / wrong interval.")
            for a, b, d in self.sub_interval_examples:
                line(f"           {a} -> {b}  (+{d} ms)")

        # Ordering
        if self.monotonic_increasing:
            line("  [OK]   timestamps strictly increasing.")
        else:
            line(f"  [FAIL] not monotone: {self.n_backward_steps:,} backward "
                 f"step(s), {self.n_equal_adjacent:,} equal-adjacent.")
            for i, prev, cur in self.backward_examples:
                line(f"           row {i}: {prev} -> {cur}")

        # Duplicates
        if self.n_duplicate_timestamps:
            line(f"  [FAIL] {self.n_duplicate_timestamps:,} duplicate "
                 f"timestamp(s), {self.n_duplicate_rows:,} extra row(s).")
            for ts, c in self.duplicate_examples:
                line(f"           {ts}  x{c}")
        else:
            line("  [OK]   no duplicate timestamps.")

        # Alignment
        if self.n_misaligned:
            line(f"  [WARN] {self.n_misaligned:,} timestamp(s) not aligned to "
                 f"the {self.interval} grid (UTC).")
            for ts, off in self.misaligned_examples:
                line(f"           {ts}  (+{off} ms past boundary)")
        else:
            line("  [OK]   all timestamps aligned to the grid.")

        # Prices
        if self.n_nonpositive_price_rows:
            line(f"  [FAIL] {self.n_nonpositive_price_rows:,} row(s) with a "
                 f"zero/negative OHLC price.")
            for i, info in self.nonpositive_examples:
                line(f"           row {i}: {info}")
        if self.n_ohlc_violations:
            line(f"  [FAIL] {self.n_ohlc_violations:,} row(s) violate OHLC "
                 f"bounds (need low <= open,close <= high).")
            for i, info in self.ohlc_examples:
                line(f"           row {i}: {info}")
        if self.n_nan_price_rows:
            line(f"  [FAIL] {self.n_nan_price_rows:,} row(s) with NaN price(s).")
        if not (self.n_nonpositive_price_rows or self.n_ohlc_violations
                or self.n_nan_price_rows):
            line("  [OK]   prices positive and OHLC bounds hold.")

        # Volume
        if self.n_negative_volume:
            line(f"  [FAIL] {self.n_negative_volume:,} row(s) with negative volume.")
        if self.n_nan_volume:
            line(f"  [FAIL] {self.n_nan_volume:,} row(s) with NaN volume.")
        if self.n_zero_volume:
            line(f"  [NOTE] {self.n_zero_volume:,} zero-volume bar(s) "
                 f"(legal: no trades that minute; O=H=L=C).")

        # close_time
        if self.n_closetime_mismatch:
            line(f"  [WARN] {self.n_closetime_mismatch:,} row(s) where close_time "
                 f"!= open_time + interval - 1 (wrong interval / malformed).")

        line("=" * 70)
        line(f"  is_clean = {self.is_clean}  "
             f"(descriptive; gaps & notes are not 'dirty')")
        line("=" * 70 + "\n")
        return self


def _utc(ms: int) -> str:
    """Format a ms-since-epoch timestamp as a UTC string."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


def verify_klines(data: KlineData, interval: str = "1m") -> KlineDiagnostics:
    """Run every diagnostic check on a loaded KlineData. Pure: no I/O, no fix.

    All timing analysis runs on the canonical millisecond view; mixed-unit data
    is normalised here and the fact is recorded in the returned diagnostics.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unknown interval {interval!r}; "
                         f"known: {sorted(INTERVAL_MS)}")
    interval_ms = INTERVAL_MS[interval]
    interval_us = interval_ms * 1000

    d = KlineDiagnostics(n_rows=data.n, interval=interval, interval_ms=interval_ms,
                         had_header=data.had_header,
                         n_header_files=data.n_header_files, n_files=data.n_files)
    if data.n == 0:
        return d

    # ── timestamp units → canonical ms view ──────────────────────────────
    unit, n_ms, n_us = detect_unit(data.open_time)
    d.unit, d.n_ms, d.n_us = unit, n_ms, n_us
    is_us = data.open_time >= _US_THRESHOLD
    t_ms = canonical_ms(data.open_time)   # single source of truth (see helper)
    d.normalised_to_ms = (unit == "mixed")

    # ── ordering (file order) ────────────────────────────────────────────
    if data.n >= 2:
        step = np.diff(t_ms)
        back = np.nonzero(step < 0)[0]
        eq = np.nonzero(step == 0)[0]
        d.n_backward_steps = int(back.size)
        d.n_equal_adjacent = int(eq.size)
        d.monotonic_increasing = (back.size == 0 and eq.size == 0)
        d.backward_examples = [(int(i + 1), int(t_ms[i]), int(t_ms[i + 1]))
                               for i in back[:_MAX_EXAMPLES]]

    # ── duplicates ───────────────────────────────────────────────────────
    uniq, counts = np.unique(t_ms, return_counts=True)
    dup_mask = counts > 1
    d.n_duplicate_timestamps = int(dup_mask.sum())
    d.n_duplicate_rows = int((counts[dup_mask] - 1).sum())
    d.duplicate_examples = [(_utc(int(ts)), int(c))
                            for ts, c in zip(uniq[dup_mask][:_MAX_EXAMPLES],
                                             counts[dup_mask][:_MAX_EXAMPLES])]

    # ── coverage / gaps (sorted unique grid) ─────────────────────────────
    su = uniq  # np.unique already returns sorted unique
    d.n_unique_timestamps = int(su.size)
    d.span_start, d.span_end = _utc(int(su[0])), _utc(int(su[-1]))
    if su.size >= 2:
        diffs = np.diff(su)
        # inferred interval = most common positive spacing
        dv, dc = np.unique(diffs, return_counts=True)
        d.inferred_interval_ms = int(dv[np.argmax(dc)])
        d.interval_matches = (d.inferred_interval_ms == interval_ms)

        gap_idx = np.nonzero(diffs > interval_ms)[0]
        missing = (diffs[gap_idx] // interval_ms - 1).astype(np.int64)
        d.n_gaps = int(gap_idx.size)
        d.total_missing_bars = int(missing.sum())
        if gap_idx.size:
            biggest = gap_idx[np.argmax(diffs[gap_idx])]
            d.largest_gap_bars = int(diffs[biggest] // interval_ms - 1)
            d.largest_gap_at = _utc(int(su[biggest]))
            d.gap_examples = [(_utc(int(su[i])), _utc(int(su[i + 1])),
                               int(diffs[i] // interval_ms - 1))
                              for i in gap_idx[:_MAX_EXAMPLES]]

        sub_idx = np.nonzero((diffs > 0) & (diffs < interval_ms))[0]
        d.n_sub_interval = int(sub_idx.size)
        d.sub_interval_examples = [(_utc(int(su[i])), _utc(int(su[i + 1])),
                                    int(diffs[i])) for i in sub_idx[:_MAX_EXAMPLES]]

        span = int(su[-1] - su[0])
        d.expected_bars = span // interval_ms + 1
        d.coverage_fraction = d.n_unique_timestamps / d.expected_bars
    else:
        d.expected_bars = 1
        d.coverage_fraction = 1.0

    # ── alignment to the UTC grid ────────────────────────────────────────
    off = t_ms % interval_ms
    mis = np.nonzero(off != 0)[0]
    d.n_misaligned = int(mis.size)
    d.misaligned_examples = [(_utc(int(t_ms[i])), int(off[i]))
                             for i in mis[:_MAX_EXAMPLES]]

    # ── prices ───────────────────────────────────────────────────────────
    o, h, l, c = data.open, data.high, data.low, data.close
    nan_price = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
    d.n_nan_price_rows = int(nan_price.sum())
    finite = ~nan_price

    nonpos = finite & ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0))
    d.n_nonpositive_price_rows = int(nonpos.sum())
    for i in np.nonzero(nonpos)[0][:_MAX_EXAMPLES]:
        d.nonpositive_examples.append(
            (int(i), f"O={o[i]:g} H={h[i]:g} L={l[i]:g} C={c[i]:g}"))

    # low <= open,close <= high  and high >= low
    viol = finite & ((h < l) | (h < o) | (h < c) | (l > o) | (l > c))
    d.n_ohlc_violations = int(viol.sum())
    for i in np.nonzero(viol)[0][:_MAX_EXAMPLES]:
        d.ohlc_examples.append(
            (int(i), f"O={o[i]:g} H={h[i]:g} L={l[i]:g} C={c[i]:g}"))

    # ── volume ───────────────────────────────────────────────────────────
    v = data.volume
    nan_v = np.isnan(v)
    d.n_nan_volume = int(nan_v.sum())
    d.n_negative_volume = int((np.nan_to_num(v, nan=0.0) < 0).sum())
    d.n_zero_volume = int(((~nan_v) & (v == 0)).sum())

    # ── close_time consistency (native units, exact) ─────────────────────
    iv_native = np.where(is_us, interval_us, interval_ms).astype(np.int64)
    expected_close = data.open_time + iv_native - 1
    ct_bad = data.close_time != expected_close
    d.n_closetime_mismatch = int(ct_bad.sum())
    d.closetime_examples = [(int(i), int(data.close_time[i]),
                             int(expected_close[i]))
                            for i in np.nonzero(ct_bad)[0][:_MAX_EXAMPLES]]
    return d


def diagnose(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]],
             interval: str = "1m",
             has_header: Union[bool, str] = "auto") -> KlineDiagnostics:
    """Load + verify in one call. Returns the diagnostics (call .report())."""
    return verify_klines(load_klines(path_or_paths, has_header=has_header),
                         interval=interval)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Diagnose Binance kline CSV data quality (honest report).")
    p.add_argument("paths", nargs="+",
                   help="CSV file(s), a directory, or a glob")
    p.add_argument("--interval", default="1m", choices=sorted(INTERVAL_MS))
    p.add_argument("--header", default="auto",
                   choices=["auto", "yes", "no"],
                   help="treat the first row as a header (default: auto-detect)")
    args = p.parse_args(argv)
    has_header = {"auto": "auto", "yes": True, "no": False}[args.header]

    paths = args.paths[0] if len(args.paths) == 1 else args.paths
    diag = diagnose(paths, interval=args.interval, has_header=has_header)
    diag.report()
    return 0 if diag.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
