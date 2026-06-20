"""
rv_series.py — build the log-realized-variance series from verified klines
==========================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions   (Phase B, brick 3)

Turns verified Binance klines (via kline_verifier.load_klines) into the
observable proxy the Layer 1c estimators consume: a series of
log-realized-variance, log RV_t. This is the EMPIRICAL twin of Layer 1c's
Rung 1 — `layer1c_roughness_audit.realized_log_variance(S, window)` builds
exactly this object from *simulated* prices, and the corruption ladder proved
the proxy can manufacture roughness. So this module's whole reason for being
careful is that its output feeds the same estimators whose proxy-bias Phase A
already mapped: the convention here must match Rung 1 byte-for-byte (pinned by
test), or a real-data Ĥ cannot be read against those simulated bias maps.

Why each design choice is what it is
------------------------------------
  * RV proxy = Σ squared log-returns over a bar, reported as log(RV). Identical
    to realized_log_variance — verified by `test_matches_phase_a_rung1`.
  * Sampling frequency is a FIRST-CLASS knob, not a constant: Rung 1 showed the
    proxy's spurious-roughness bias grows as the RV window shrinks, and Rung 2
    showed 1-minute sampling is where microstructure noise bites hardest (the
    standard mitigation is to sample at ~5 min). You are meant to sweep it.
  * Bipower variation (jump-robust, Rung 3) is produced alongside RV so a
    real-data Ĥ can be compared with and without jump contamination.
  * Timestamps pass through kline_verifier.canonical_ms — the SAME ms->µs
    normalisation the verifier uses — so a Dec-2024/Jan-2025 boundary does not
    manufacture a fake gap. This closes the loop the verifier warned about.
  * RV bars are aligned to the UTC calendar (day / hour / 4h), and gap-spanning
    returns are counted and reported, not silently swallowed.

A sober note on series length
------------------------------
The Hurst estimators need a LONG series: GJR's default lags reach 89, MF-DFA
and Cont–Das sweep many scales. Three months of *daily* RV is ~91 points —
far too short to conclude anything about H. `.report()` says so loudly. For an
inferential run, pull years of history (daily RV -> hundreds/thousands of
points), or trade length against proxy-noise with a finer rv_bar.

Public API
----------
  log_realized_variance(S, window)        -> log RV   (Rung-1 identical proxy)
  log_bipower_variance(S, window)         -> log BV   (jump-robust twin)
  build_rv_series(source, sampling=, rv_bar=, ...) -> RVSeries
  RVSeries  (dataclass; .log_rv is the estimator input; .report() prints a
             severity-tagged honest summary; .clean_log_rv() drops sparse bars)

CLI
---
  python rv_series.py data/spot/klines/BTCUSDT/1m/*.csv --sampling 5m --rv-bar 1d
  (add --out data/processed/btc_rv.csv to save the committable series)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from kline_verifier import (
    INTERVAL_MS,
    KlineData,
    canonical_ms,
    detect_unit,
    load_klines,
)

__all__ = [
    "log_realized_variance",
    "log_bipower_variance",
    "build_rv_series",
    "RVSeries",
]

log = logging.getLogger("rv_series")

# Below this many RV observations the estimators are starved (GJR lags reach
# 89; MF-DFA / Cont–Das sweep many scales). Purely advisory — used by report().
_MIN_OBS_HARD = 250      # below: results are not interpretable
_MIN_OBS_SOFT = 1000     # below: usable but wide; above: comfortable

# Guard so log() never sees a zero RV (a bar with no price movement). Matches
# the +1e-300 floor used by Layer 1c so the proxies stay numerically identical.
_LOG_FLOOR = 1e-300


# ──────────────────────────────────────────────────────────────────────────
# the proxy primitives — convention-identical to Layer 1c Rung 1
# ──────────────────────────────────────────────────────────────────────────

def log_realized_variance(S: np.ndarray, window: int) -> np.ndarray:
    """log of realized variance (Σ squared log-returns) over fixed windows.

    Byte-identical convention to layer1c_roughness_audit.realized_log_variance
    (pinned by test_matches_phase_a_rung1). Accepts a 1-D price path -> 1-D
    output, or (n_paths, n_fine+1) -> (n_paths, n_windows), like Phase A.
    """
    S2 = np.atleast_2d(S)
    ret = np.diff(np.log(S2), axis=1)
    n_windows = ret.shape[1] // window
    rv = np.empty((S2.shape[0], n_windows))
    for w in range(n_windows):
        seg = ret[:, w * window:(w + 1) * window]
        rv[:, w] = np.sum(seg ** 2, axis=1)
    out = np.log(rv + _LOG_FLOOR)
    return out[0] if np.ndim(S) == 1 else out


def log_bipower_variance(S: np.ndarray, window: int) -> np.ndarray:
    """log of bipower variation (π/2 · Σ |r_t|·|r_{t-1}|) over fixed windows.

    Jump-robust RV alternative (Barndorff-Nielsen–Shephard). Byte-identical
    convention to layer1c_roughness_audit.bipower_log_variance (pinned by test).
    """
    S2 = np.atleast_2d(S)
    aret = np.abs(np.diff(np.log(S2), axis=1))
    prod = aret[:, 1:] * aret[:, :-1]
    n_windows = prod.shape[1] // window
    bv = (np.pi / 2) * np.array(
        [np.sum(prod[:, w * window:(w + 1) * window], axis=1)
         for w in range(n_windows)]).T
    out = np.log(bv + _LOG_FLOOR)
    return out[0] if np.ndim(S) == 1 else out


# ──────────────────────────────────────────────────────────────────────────
# the result container
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RVSeries:
    """A calendar-bucketed realized-variance series + provenance.

    `log_rv` is the array you hand to gjr_hurst / pvariation_hurst /
    mfdfa_hurst. Everything else is honest bookkeeping: which UTC period each
    observation covers, how many returns went into it, and how many of those
    returns spanned a data gap (so a sparse bar can be down-weighted or
    dropped via `clean_log_rv`).
    """
    period_start_ms: np.ndarray     # int64, UTC start of each RV bar
    log_rv: np.ndarray              # float64, the estimator input
    log_bv: np.ndarray              # float64, jump-robust twin (NaN if disabled)
    n_returns: np.ndarray           # int64, returns summed into each bar
    n_gap_spanning: np.ndarray      # int64, of those, how many crossed a gap

    sampling: str = "1m"
    rv_bar: str = "1d"
    base_interval: str = "1m"
    expected_returns_per_bar: int = 0
    symbol: Optional[str] = None
    n_input_rows: int = 0
    unit: str = "unknown"

    @property
    def n_obs(self) -> int:
        return int(self.log_rv.shape[0])

    def clean_log_rv(self, min_returns: Optional[int] = None,
                     min_fraction: float = 0.5) -> np.ndarray:
        """The log-RV series with under-populated bars dropped.

        A bar built from too few returns is a noisy RV estimate (and a partial
        first/last calendar day). Default keeps bars with at least
        `min_fraction` of the expected return count; pass `min_returns` to set
        an explicit floor. Dropping is for the analyst's convenience — the full
        series stays in `log_rv`.
        """
        if min_returns is None:
            min_returns = max(2, int(min_fraction * self.expected_returns_per_bar))
        return self.log_rv[self.n_returns >= min_returns]

    def report(self, stream=sys.stdout) -> "RVSeries":
        """Print a severity-tagged honest summary; return self."""
        def line(s: str = "") -> None:
            print(s, file=stream)

        def utc(ms: int) -> str:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC")

        line("\n" + "=" * 70)
        title = f"RV series — {self.n_obs:,} × {self.rv_bar} bars"
        if self.symbol:
            title += f"  [{self.symbol}]"
        line(f"  {title}")
        line("=" * 70)
        if self.n_obs == 0:
            line("  [FAIL] no RV observations produced (not enough data).")
            line("=" * 70 + "\n")
            return self

        line(f"  source      : {self.n_input_rows:,} klines, unit {self.unit}")
        line(f"  sampling    : {self.sampling} returns  "
             f"(~{self.expected_returns_per_bar:,} per {self.rv_bar} bar)")
        line(f"  span        : {utc(int(self.period_start_ms[0]))}  ->  "
             f"{utc(int(self.period_start_ms[-1]))}")

        med = int(np.median(self.n_returns))
        lo = int(self.n_returns.min())
        line(f"  returns/bar : median {med:,}, min {lo:,} "
             f"(expected {self.expected_returns_per_bar:,})")
        n_sparse = int((self.n_returns < 0.5 * self.expected_returns_per_bar).sum())
        if n_sparse:
            line(f"  [NOTE] {n_sparse:,} bar(s) under half-expected returns "
                 f"(partial day / gaps) — see clean_log_rv().")
        n_gap = int((self.n_gap_spanning > 0).sum())
        if n_gap:
            line(f"  [NOTE] {n_gap:,} bar(s) contain gap-spanning returns "
                 f"({int(self.n_gap_spanning.sum()):,} such returns total).")

        # The headline honesty: is this series long enough to mean anything?
        if self.n_obs < _MIN_OBS_HARD:
            line(f"  [WARN] only {self.n_obs:,} observations — TOO FEW to "
                 f"estimate H reliably.")
            line(f"         GJR lags reach 89; MF-DFA/Cont–Das need long "
                 f"series. Pull more history, or use a finer --rv-bar")
            line(f"         (trading proxy-noise for length).")
        elif self.n_obs < _MIN_OBS_SOFT:
            line(f"  [NOTE] {self.n_obs:,} observations — usable, but H "
                 f"estimates will be wide; ~{_MIN_OBS_SOFT:,}+ is comfortable.")
        else:
            line(f"  [OK]   {self.n_obs:,} observations — enough to estimate H.")

        line(f"  log RV      : mean {np.nanmean(self.log_rv):+.3f}, "
             f"sd {np.nanstd(self.log_rv):.3f}")
        if np.isfinite(self.log_bv).any():
            line(f"  log BV      : mean {np.nanmean(self.log_bv):+.3f} "
                 f"(jump-robust; compare Ĥ on RV vs BV).")
        line("  reminder: this is the RV PROXY (Layer 1c Rung 1). Read any Ĥ "
             "against the\n           simulated proxy-bias maps before "
             "claiming the roughness is real.")
        line("=" * 70 + "\n")
        return self


# ──────────────────────────────────────────────────────────────────────────
# the builder
# ──────────────────────────────────────────────────────────────────────────

def _interval_ms(name: str, what: str) -> int:
    if name not in INTERVAL_MS:
        raise ValueError(f"unknown {what} {name!r}; known: {sorted(INTERVAL_MS)}")
    return INTERVAL_MS[name]


def build_rv_series(source: Union[str, Path, Sequence, KlineData],
                    sampling: str = "1m",
                    rv_bar: str = "1d",
                    base_interval: str = "1m",
                    jump_robust: bool = True,
                    symbol: Optional[str] = None) -> RVSeries:
    """Build a calendar-bucketed log-RV series from verified klines.

    Parameters
    ----------
    source : a path / glob / list of kline CSVs (loaded via load_klines) OR an
             already-loaded KlineData.
    sampling : return granularity for the RV sum (e.g. '1m', '5m'). Must be a
               multiple of base_interval. Coarser sampling mitigates
               microstructure noise (Rung 2) at the cost of fewer returns.
    rv_bar   : the period one RV observation covers (e.g. '1d', '1h', '4h'),
               aligned to the UTC calendar.
    base_interval : the native bar size of the input klines (default '1m').
    jump_robust : also compute log bipower variation (Rung-3 robust twin).

    Returns
    -------
    RVSeries  (call .report(); feed .log_rv to the estimators).
    """
    data = source if isinstance(source, KlineData) else load_klines(source)

    base_ms = _interval_ms(base_interval, "base_interval")
    samp_ms = _interval_ms(sampling, "sampling")
    bar_ms = _interval_ms(rv_bar, "rv_bar")
    if samp_ms % base_ms != 0:
        raise ValueError(f"sampling {sampling} is not a multiple of base "
                         f"interval {base_interval}")
    if bar_ms % samp_ms != 0:
        raise ValueError(f"rv_bar {rv_bar} is not a multiple of sampling {sampling}")
    expected_per_bar = bar_ms // samp_ms

    # canonical ms (the shared ms->µs normalisation) + close prices, in order.
    t_ms = canonical_ms(data.open_time)
    close = np.asarray(data.close, dtype=np.float64)
    unit, _, _ = detect_unit(np.asarray(data.open_time))

    empty = RVSeries(
        period_start_ms=np.empty(0, np.int64),
        log_rv=np.empty(0), log_bv=np.empty(0),
        n_returns=np.empty(0, np.int64), n_gap_spanning=np.empty(0, np.int64),
        sampling=sampling, rv_bar=rv_bar, base_interval=base_interval,
        expected_returns_per_bar=expected_per_bar, symbol=symbol,
        n_input_rows=int(close.size), unit=unit)
    if close.size < 2:
        return empty

    # ── calendar-anchored subsample to the sampling grid ──────────────────
    # Keep bars whose timestamp lands on a sampling-grid mark (e.g. :00/:05 for
    # 5m), so sampling is reproducible and independent of where the file starts.
    keep = (t_ms % samp_ms) == 0
    ts = t_ms[keep]
    cs = close[keep]
    if ts.size < 2:
        return empty

    # ── returns on the sampled grid ───────────────────────────────────────
    r = np.diff(np.log(cs))
    t_end = ts[1:]                      # a return is stamped to its later bar
    dt = np.diff(ts)
    gap_spanning = dt > samp_ms         # a sampled step longer than nominal

    # ── bucket returns into UTC calendar bars ─────────────────────────────
    bucket = t_end // bar_ms           # integer UTC period index
    period_ids = np.unique(bucket)
    period_start = period_ids * bar_ms

    # adjacent |r|·|r_{-1}| products, each stamped to the later return's bar.
    if jump_robust and r.size >= 2:
        prod = np.abs(r[1:]) * np.abs(r[:-1])
        prod_bucket = bucket[1:]       # product k uses returns k, k-1 -> bar of k
    else:
        prod = None

    n_obs = period_ids.size
    log_rv = np.empty(n_obs)
    log_bv = np.full(n_obs, np.nan)
    n_ret = np.empty(n_obs, np.int64)
    n_gap = np.empty(n_obs, np.int64)

    for i, pid in enumerate(period_ids):
        m = bucket == pid
        seg = r[m]
        rv = np.sum(seg ** 2)
        log_rv[i] = np.log(rv + _LOG_FLOOR)
        n_ret[i] = int(seg.size)
        n_gap[i] = int(gap_spanning[m].sum())
        if prod is not None:
            pm = prod_bucket == pid
            if pm.any():
                bv = (np.pi / 2) * np.sum(prod[pm])
                log_bv[i] = np.log(bv + _LOG_FLOOR)

    return RVSeries(
        period_start_ms=period_start.astype(np.int64),
        log_rv=log_rv, log_bv=log_bv,
        n_returns=n_ret, n_gap_spanning=n_gap,
        sampling=sampling, rv_bar=rv_bar, base_interval=base_interval,
        expected_returns_per_bar=expected_per_bar, symbol=symbol,
        n_input_rows=int(close.size), unit=unit)


def _save_csv(series: RVSeries, path: Union[str, Path]) -> None:
    """Write the small, committable processed series (one row per RV bar)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = "period_start_utc,period_start_ms,log_rv,log_bv,n_returns,n_gap_spanning"
    lines = [header]
    for i in range(series.n_obs):
        ms = int(series.period_start_ms[i])
        iso = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        bv = series.log_bv[i]
        bv_s = "" if not np.isfinite(bv) else repr(float(bv))
        lines.append(f"{iso},{ms},{repr(float(series.log_rv[i]))},{bv_s},"
                     f"{int(series.n_returns[i])},{int(series.n_gap_spanning[i])}")
    p.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the log-realized-variance series from verified "
                    "klines (the Layer 1c Rung-1 proxy, on real data).")
    p.add_argument("paths", nargs="+", help="kline CSV file(s), a dir, or a glob")
    p.add_argument("--sampling", default="5m", choices=sorted(INTERVAL_MS),
                   help="return granularity for the RV sum (default 5m)")
    p.add_argument("--rv-bar", default="1d", choices=sorted(INTERVAL_MS),
                   help="period each RV observation covers (default 1d)")
    p.add_argument("--base-interval", default="1m", choices=sorted(INTERVAL_MS),
                   help="native bar size of the input klines (default 1m)")
    p.add_argument("--no-bipower", action="store_true",
                   help="skip the jump-robust bipower series")
    p.add_argument("--symbol", default=None, help="label for the report")
    p.add_argument("--out", default=None,
                   help="save the processed series to this CSV (committable)")
    args = p.parse_args(argv)

    paths = args.paths[0] if len(args.paths) == 1 else args.paths
    series = build_rv_series(paths, sampling=args.sampling, rv_bar=args.rv_bar,
                             base_interval=args.base_interval,
                             jump_robust=not args.no_bipower, symbol=args.symbol)
    series.report()
    if args.out:
        _save_csv(series, args.out)
        print(f"  wrote {series.n_obs:,} rows -> {args.out}\n")
    return 0 if series.n_obs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
