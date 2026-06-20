"""
RoughVolLab — equity_data.py
Phase B equity arm: free daily OHLC → range-based daily log-variance, in the
SAME CSV format estimate_h.py / interpret_h.py already read.

Why this module exists
----------------------
The crypto arm (rv_series.py) builds realized variance from intraday returns.
Free *intraday* equity history at a 2019–2025 span does not exist: Yahoo caps
1-minute data at ~60 days, and the Oxford-Man realized library was
discontinued. The ROADMAP's sanctioned fallback is a RANGE-BASED daily variance
(Parkinson / Garman–Klass) on free *daily* OHLC, clearly labelled lower
fidelity. This module fetches daily OHLC with the standard library (from stooq,
no API key, no pandas/requests — matching repo conventions) and writes
log(range-variance) one row per day, so the equity-vs-crypto Rung-5 gap
comparison falls straight out of the estimator/de-bias pipeline you already
have.

Fidelity caveats — state these in any write-up:
  * Range estimators capture the TRADING-SESSION (open→close) variance and MISS
    the overnight close→open gap, which for equities is real. That missing
    overnight move is itself part of the gap structure Rung 5 studies.
  * A range proxy is not the same statistical object as an intraday-return RV.
    An equity-vs-crypto Ĥ difference therefore mixes a CALENDAR difference with
    a PROXY difference — read it as suggestive, not a clean isolation. The clean
    isolation is the simulated Rung 5 (layer1c rung5_calendar).

Public API
----------
  download_stooq_daily(symbol, out_path, start=None, end=None) -> Path
  parse_ohlc_csv(path)                 -> dict(date_ms, o, h, l, c)
  garman_klass_log_var(o, h, l, c)     -> np.ndarray   # daily log-variance
  parkinson_log_var(h, l)              -> np.ndarray
  build_equity_rv_csv(in_csv, out_csv, estimator="gk") -> Path
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = [
    "download_stooq_daily",
    "parse_ohlc_csv",
    "garman_klass_log_var",
    "parkinson_log_var",
    "build_equity_rv_csv",
]

# stooq daily-history CSV endpoint: s=<ticker>&i=d. US tickers use a `.us`
# suffix (spy.us, qqq.us); indices use a caret (^spx). Free, no key.
_STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"
_LN2 = float(np.log(2.0))


# ──────────────────────────────────────────────────────────────────────────
# Download (stdlib only)
# ──────────────────────────────────────────────────────────────────────────

def download_stooq_daily(symbol: str, out_path: Union[str, Path],
                         start: Optional[str] = None,
                         end: Optional[str] = None) -> Path:
    """Fetch daily OHLC for `symbol` from stooq and save the raw CSV.

    `start`/`end` are optional 'YYYY-MM-DD' bounds. Raises with a clear message
    if stooq returns something other than an OHLC header (rate limit / bad
    ticker) — in that case download the CSV by hand from stooq.com or Yahoo and
    use build_equity_rv_csv on the file directly (parse_ohlc_csv reads both).
    """
    url = _STOOQ_URL.format(sym=symbol.lower())
    if start:
        url += "&d1=" + start.replace("-", "")
    if end:
        url += "&d2=" + end.replace("-", "")
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "Mozilla/5.0 (roughvollab)"})
    with urllib.request.urlopen(req, timeout=60) as resp:        # nosec - fixed host
        text = resp.read().decode("utf-8", "replace")
    first = text.splitlines()[0] if text.strip() else ""
    if not first.lower().startswith("date,"):
        raise RuntimeError(
            f"stooq did not return an OHLC CSV for {symbol!r}. First line: "
            f"{first!r}. Likely a rate limit or an unknown ticker — try again "
            f"later, try another ticker, or download the CSV manually and run "
            f"the build step on the file.")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# ──────────────────────────────────────────────────────────────────────────
# Parse + range-variance estimators
# ──────────────────────────────────────────────────────────────────────────

def parse_ohlc_csv(path: Union[str, Path]) -> dict:
    """Read a Date,Open,High,Low,Close[,...] CSV → arrays (date_ms, o, h, l, c).

    Tolerant of extra columns (Volume, Adj Close), so stooq AND Yahoo CSVs both
    work. Rows with missing or non-positive prices are dropped.
    """
    text = Path(path).read_text().strip().splitlines()
    reader = csv.DictReader(text)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: empty or unreadable CSV")
    cols = {c.lower(): c for c in reader.fieldnames}
    need = ("date", "open", "high", "low", "close")
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing} "
                         f"(header: {reader.fieldnames})")
    ms, o, h, l, c = [], [], [], [], []
    for row in reader:
        try:
            d = datetime.strptime(row[cols["date"]].strip(), "%Y-%m-%d")
            oo = float(row[cols["open"]]); hh = float(row[cols["high"]])
            ll = float(row[cols["low"]]);  cc = float(row[cols["close"]])
        except (ValueError, KeyError):
            continue
        if min(oo, hh, ll, cc) <= 0 or hh < ll:
            continue
        ms.append(int(d.replace(tzinfo=timezone.utc).timestamp() * 1000))
        o.append(oo); h.append(hh); l.append(ll); c.append(cc)
    if not ms:
        raise ValueError(f"{path}: no usable OHLC rows")
    order = np.argsort(ms)                       # stooq is ascending; be safe
    return dict(date_ms=np.asarray(ms)[order], o=np.asarray(o)[order],
                h=np.asarray(h)[order], l=np.asarray(l)[order],
                c=np.asarray(c)[order])


def garman_klass_log_var(o, h, l, c) -> np.ndarray:
    """Garman–Klass daily log-variance: log[0.5 (ln H/L)^2 − (2ln2−1)(ln C/O)^2].

    An efficient (~7× over close-to-close) unbiased estimator of the open→close
    variance under driftless GBM. Captures the trading session, not overnight.
    """
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    hl = np.log(h / l)
    co = np.log(c / o)
    var = 0.5 * hl ** 2 - (2.0 * _LN2 - 1.0) * co ** 2
    return np.log(np.clip(var, 1e-12, None))


def parkinson_log_var(h, l) -> np.ndarray:
    """Parkinson daily log-variance: log[(ln H/L)^2 / (4 ln2)]. High–low only."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    var = np.log(h / l) ** 2 / (4.0 * _LN2)
    return np.log(np.clip(var, 1e-12, None))


# ──────────────────────────────────────────────────────────────────────────
# Build the pipeline-compatible series
# ──────────────────────────────────────────────────────────────────────────

def build_equity_rv_csv(in_csv: Union[str, Path], out_csv: Union[str, Path],
                        estimator: str = "gk") -> Path:
    """OHLC CSV → a daily log-variance CSV readable by estimate_h/interpret_h.

    Header matches rv_series.py exactly. `log_rv` carries the range-based daily
    log-variance; `log_bv` is left empty (no jump-robust analogue here);
    n_returns is a nominal 4 (O,H,L,C used) and n_gap_spanning 0.
    """
    if estimator not in ("gk", "parkinson"):
        raise ValueError("estimator must be 'gk' or 'parkinson'")
    d = parse_ohlc_csv(in_csv)
    lrv = (garman_klass_log_var(d["o"], d["h"], d["l"], d["c"])
           if estimator == "gk" else parkinson_log_var(d["h"], d["l"]))
    header = "period_start_utc,period_start_ms,log_rv,log_bv,n_returns,n_gap_spanning"
    lines = [header]
    kept = 0
    for i in range(lrv.size):
        if not np.isfinite(lrv[i]):
            continue
        ms = int(d["date_ms"][i])
        iso = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        lines.append(f"{iso},{ms},{repr(float(lrv[i]))},,4,0")
        kept += 1
    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    span = (datetime.fromtimestamp(int(d['date_ms'][0]) / 1000, tz=timezone.utc),
            datetime.fromtimestamp(int(d['date_ms'][-1]) / 1000, tz=timezone.utc))
    print(f"  {estimator.upper()} range-variance: {kept} daily obs "
          f"({span[0]:%Y-%m-%d} → {span[1]:%Y-%m-%d}) → {p}")
    print(f"  reminder: range proxy = TRADING SESSION only (no overnight gap); "
          f"read Ĥ against the same Rung-1 caveats as crypto, plus the proxy "
          f"caveat in this module's docstring.")
    return p


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Equity arm: free daily OHLC → range-based daily "
                    "log-variance (pipeline-compatible with estimate_h.py).")
    ap.add_argument("in_csv", nargs="?", default=None,
                    help="existing OHLC CSV to build from (Date,Open,High,Low,"
                         "Close[,...]); omit if using --symbol to download")
    ap.add_argument("--symbol", default=None,
                    help="stooq ticker to download, e.g. spy.us, qqq.us, ^spx")
    ap.add_argument("--download", default=None,
                    help="path to save the raw OHLC CSV (default "
                         "data/equity/<symbol>.csv)")
    ap.add_argument("--out", default=None,
                    help="path for the processed log-variance CSV")
    ap.add_argument("--estimator", default="gk", choices=["gk", "parkinson"],
                    help="range-variance estimator (default gk = Garman–Klass)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD lower bound")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD upper bound")
    args = ap.parse_args(argv)

    in_csv = args.in_csv
    if args.symbol:
        raw = args.download or f"data/equity/{args.symbol.replace('^', '_')}.csv"
        print(f"  downloading {args.symbol} daily OHLC from stooq …")
        in_csv = download_stooq_daily(args.symbol, raw, args.start, args.end)
        print(f"  saved raw OHLC → {in_csv}")
    if in_csv is None:
        ap.error("provide an OHLC CSV path or --symbol to download")
    if args.out is None:
        ap.error("provide --out for the processed log-variance CSV")
    build_equity_rv_csv(in_csv, args.out, estimator=args.estimator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
