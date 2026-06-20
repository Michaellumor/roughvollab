"""
binance_data.py — Binance public-data access layer (Phase B, brick 1)
=====================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

This is the downloader for the Binance public data repository at
https://data.binance.vision — the first real-data brick of Phase B. Today it
fetches BTCUSDT 1-minute klines; the URL/path machinery is written generically
(market × period × data_type × symbol × interval) so the same code pulls
aggTrades or raw trades later by changing one argument, not rewriting the layer.

Why this module exists
----------------------
Phase A measured estimators on simulated paths where ground truth is known.
Phase B confronts them with real markets, and real exchange dumps are messy:
silent corruption, the ms→µs timestamp switch on 2025-01-01, partial months,
404s for dates that don't exist. Garbage in the corpus would make Layer 1c
measure the data-feed's defects, not the science. So this layer does two
defensive things at download time:

  1. Verifies the SHA-256 .CHECKSUM that Binance ships alongside every .zip,
     refusing to extract a corrupt archive (silent bit-rot is exactly the
     failure the verifier brick is built to catch downstream — better to catch
     it here too).
  2. Reports honestly. Every file ends in a typed status
     (downloaded / exists / not_found / checksum_failed / error) and the run
     prints a summary. Nothing is swallowed.

The actual HTTP GET is isolated behind one `fetcher` callable so the
orchestration (verify → extract → status) is unit-testable offline with an
in-memory fake feed — the sandbox can't reach data.binance.vision, the corpus
download happens on the local machine.

Repository layout (mirrors Binance, so re-downloads are detected):

  data/{market}/{data_type}/{symbol}/{interval}/{SYMBOL}-{interval}-{date}.csv

`data/` is gitignored (ROADMAP Phase B: raw downloads cached, not committed).

Public API
----------
  build_url(data_type, symbol, date_str, interval=, market=, period=) -> str
  month_range(start, end) / day_range(start, end)                    -> [str]
  expected_files(symbol, start, end, ...)                  -> [(date, url)]
  sha256_bytes(b) / sha256_file(path) / parse_checksum(text)
  download_range(data_type, symbol, start, end, ...)       -> [FileResult]
  download_klines(symbol="BTCUSDT", interval="1m", start, end, ...) -> [...]
  summarize(results)                                       -> dict + report

CLI
---
  python binance_data.py --symbol BTCUSDT --interval 1m \
      --start 2024-01 --end 2024-03 --out data
  (use --daily for daily files, --data-type aggTrades for agg trades, etc.)

Reference
---------
- Binance public data: https://github.com/binance/binance-public-data
  Note: respect Binance's terms when redistributing derived data (ROADMAP).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

__all__ = [
    "BASE_URL",
    "NotFound",
    "FileResult",
    "build_path",
    "build_url",
    "month_range",
    "day_range",
    "expected_files",
    "sha256_bytes",
    "sha256_file",
    "parse_checksum",
    "download_range",
    "download_klines",
    "summarize",
]

log = logging.getLogger("binance_data")

BASE_URL = "https://data.binance.vision"

# Markets, as the path segment Binance uses. Spot is one segment; the two
# futures venues are two ("futures/um" = USD-M, "futures/cm" = COIN-M).
MARKETS = ("spot", "futures/um", "futures/cm")

# Only kline-family data types carry an interval segment in the path AND the
# filename (e.g. .../klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip). aggTrades and
# trades have no interval level (.../aggTrades/BTCUSDT/BTCUSDT-aggTrades-...).
_INTERVAL_TYPES = frozenset(
    {"klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines"}
)

# Recognised kline intervals (used only for validation; the path is generic).
_KLINE_INTERVALS = frozenset(
    {"1s", "1m", "3m", "5m", "15m", "30m",
     "1h", "2h", "4h", "6h", "8h", "12h",
     "1d", "3d", "1w", "1mo"}
)


class NotFound(Exception):
    """Raised by the fetcher when the remote object returns HTTP 404.

    Distinguished from transient errors so the orchestrator can record a clean
    'not_found' status (a missing daily file for a future date is normal) and
    move on, rather than retrying or aborting the whole run.
    """


@dataclass
class FileResult:
    """Honest per-file outcome of a download attempt.

    status is one of:
      downloaded      — fetched (and verified, if a checksum was available)
      exists          — already present locally, skipped
      not_found       — remote returned 404 (date doesn't exist on Binance)
      checksum_failed — SHA-256 mismatch; archive NOT extracted
      error           — transient/unexpected failure after retries
    checksum_verified is True/False when a checksum was checked, None when the
    .CHECKSUM file itself was absent (verified=None is reported, not assumed OK).
    """
    date: str
    data_type: str
    symbol: str
    interval: Optional[str]
    url: str
    status: str
    checksum_verified: Optional[bool] = None
    path: Optional[str] = None
    n_bytes: int = 0
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────
# URL / path construction  (pure, fully unit-testable)
# ──────────────────────────────────────────────────────────────────────────

def build_path(data_type: str, symbol: str, date_str: str,
               interval: Optional[str] = None,
               market: str = "spot", period: str = "monthly") -> str:
    """Build the repository path (everything after BASE_URL) for one archive.

    Parameters
    ----------
    data_type : 'klines', 'aggTrades', 'trades', ...
    symbol    : e.g. 'BTCUSDT' (upper-cased internally)
    date_str  : 'YYYY-MM' for monthly, 'YYYY-MM-DD' for daily
    interval  : required iff data_type is a kline family type, else must be None
    market    : 'spot' | 'futures/um' | 'futures/cm'
    period    : 'monthly' | 'daily'

    Returns
    -------
    str : e.g. '/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip'
    """
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}, got {market!r}")
    if period not in ("monthly", "daily"):
        raise ValueError(f"period must be 'monthly' or 'daily', got {period!r}")
    symbol = symbol.upper()

    if data_type in _INTERVAL_TYPES:
        if interval is None:
            raise ValueError(f"{data_type} requires an interval (e.g. '1m')")
        if interval not in _KLINE_INTERVALS:
            log.warning("interval %r not in known set %s — proceeding anyway",
                        interval, sorted(_KLINE_INTERVALS))
        leaf_dir = f"{data_type}/{symbol}/{interval}"
        fname = f"{symbol}-{interval}-{date_str}.zip"
    else:
        if interval is not None:
            raise ValueError(f"{data_type} does not take an interval")
        leaf_dir = f"{data_type}/{symbol}"
        fname = f"{symbol}-{data_type}-{date_str}.zip"

    return f"/data/{market}/{period}/{leaf_dir}/{fname}"


def build_url(data_type: str, symbol: str, date_str: str,
              interval: Optional[str] = None,
              market: str = "spot", period: str = "monthly") -> str:
    """Full https URL for one archive (BASE_URL + build_path)."""
    return BASE_URL + build_path(data_type, symbol, date_str,
                                 interval=interval, market=market, period=period)


# ──────────────────────────────────────────────────────────────────────────
# Date enumeration
# ──────────────────────────────────────────────────────────────────────────

def _parse_month(s: str) -> date:
    """Parse 'YYYY-MM' (or the month part of 'YYYY-MM-DD') to a date on day 1."""
    s = s.strip()
    parts = s.split("-")
    if len(parts) < 2:
        raise ValueError(f"month must be 'YYYY-MM', got {s!r}")
    return date(int(parts[0]), int(parts[1]), 1)


def _parse_day(s: str) -> date:
    """Parse 'YYYY-MM-DD' to a date (rejects bare 'YYYY-MM' for daily clarity)."""
    s = s.strip()
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"day must be 'YYYY-MM-DD', got {s!r}")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    a, b = _parse_month(start), _parse_month(end)
    if b < a:
        raise ValueError(f"end {end!r} precedes start {start!r}")
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def day_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM-DD' strings from start to end."""
    a, b = _parse_day(start), _parse_day(end)
    if b < a:
        raise ValueError(f"end {end!r} precedes start {start!r}")
    out, d = [], a
    while d <= b:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def expected_files(symbol: str, start: str, end: str,
                   data_type: str = "klines", interval: Optional[str] = "1m",
                   market: str = "spot", period: str = "monthly"
                   ) -> list[tuple[str, str]]:
    """Enumerate (date_str, url) pairs to fetch for a date range.

    monthly → one entry per calendar month; daily → one per day.
    """
    dates = month_range(start, end) if period == "monthly" else day_range(start, end)
    return [(d, build_url(data_type, symbol, d, interval=interval,
                          market=market, period=period)) for d in dates]


# ──────────────────────────────────────────────────────────────────────────
# Checksums  (pure, unit-testable)
# ──────────────────────────────────────────────────────────────────────────

def sha256_bytes(b: bytes) -> str:
    """Hex SHA-256 of an in-memory byte string."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed in chunks (for re-verifying on disk)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_checksum(text: str | bytes) -> str:
    """Extract the hex digest from a Binance .CHECKSUM file.

    Format is the standard `sha256sum` output: '<hexdigest>  <filename>'.
    We take the first whitespace-delimited token and lower-case it.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    token = text.strip().split()[0] if text.strip() else ""
    return token.lower()


# ──────────────────────────────────────────────────────────────────────────
# HTTP  (the one impure function — isolated so the rest is testable offline)
# ──────────────────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: float = 30.0) -> bytes:
    """GET url, returning the body bytes. Raises NotFound on 404.

    Other HTTP/URL errors propagate (the caller's retry loop handles them).
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "roughvollab-binance-data/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotFound(url) from e
        raise


def _fetch_with_retry(fetcher: Callable[[str, float], bytes], url: str,
                      retries: int, backoff: float, timeout: float) -> bytes:
    """Call fetcher with exponential backoff. NotFound is not retried."""
    attempt = 0
    while True:
        try:
            return fetcher(url, timeout)
        except NotFound:
            raise
        except Exception as e:  # noqa: BLE001 — transient network errors
            attempt += 1
            if attempt > retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            log.warning("fetch failed (%s), retry %d/%d in %.1fs: %s",
                        type(e).__name__, attempt, retries, wait, url)
            time.sleep(wait)


# ──────────────────────────────────────────────────────────────────────────
# Archive handling
# ──────────────────────────────────────────────────────────────────────────

def _extract_csv(zip_bytes: bytes, dest_dir: Path) -> Path:
    """Extract the single .csv member of a kline/trades zip to dest_dir.

    Returns the path to the written CSV. Raises if no CSV member is present.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"no .csv member in archive (members: {zf.namelist()})")
        member = csv_members[0]
        # Flatten any internal path to just the basename under dest_dir.
        out_path = dest_dir / Path(member).name
        with zf.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

def _leaf_dir(out_dir: Path, market: str, data_type: str, symbol: str,
              interval: Optional[str]) -> Path:
    """Local cache directory mirroring the Binance layout."""
    parts = [out_dir, *market.split("/"), data_type, symbol.upper()]
    if interval is not None:
        parts.append(interval)
    return Path(*[str(p) for p in parts])


def download_range(data_type: str, symbol: str, start: str, end: str,
                   interval: Optional[str] = None,
                   market: str = "spot", period: str = "monthly",
                   out_dir: str | Path = "data",
                   *, verify: bool = True, extract: bool = True,
                   keep_zip: bool = False, force: bool = False,
                   retries: int = 3, backoff: float = 1.0, timeout: float = 30.0,
                   fetcher: Optional[Callable[[str, float], bytes]] = None
                   ) -> list[FileResult]:
    """Download a date range of Binance archives, verifying and extracting each.

    The network call is `fetcher` (defaults to `_http_get`); pass a fake one in
    tests to exercise this orchestration without touching the network.

    Returns one FileResult per expected file (nothing raises on a bad/missing
    file — the status records it). See FileResult for the status vocabulary.
    """
    fetcher = fetcher or _http_get
    out_dir = Path(out_dir)
    leaf = _leaf_dir(out_dir, market, data_type, symbol, interval)
    files = expected_files(symbol, start, end, data_type=data_type,
                           interval=interval, market=market, period=period)
    results: list[FileResult] = []

    for date_str, url in files:
        zip_name = url.rsplit("/", 1)[-1]
        csv_name = zip_name[:-4] + ".csv"
        csv_path = leaf / csv_name
        zip_path = leaf / zip_name
        res = FileResult(date=date_str, data_type=data_type,
                         symbol=symbol.upper(), interval=interval, url=url,
                         status="error")

        # Skip if we already have the artefact we'd produce.
        already = (csv_path.exists() if extract else zip_path.exists())
        if already and not force:
            res.status = "exists"
            res.path = str(csv_path if extract else zip_path)
            results.append(res)
            log.info("exists, skipping: %s", res.path)
            continue

        # Fetch the archive.
        try:
            zip_bytes = _fetch_with_retry(fetcher, url, retries, backoff, timeout)
        except NotFound:
            res.status = "not_found"
            results.append(res)
            log.info("not found (404): %s", url)
            continue
        except Exception as e:  # noqa: BLE001
            res.status = "error"
            res.error = f"{type(e).__name__}: {e}"
            results.append(res)
            log.error("error fetching %s: %s", url, res.error)
            continue
        res.n_bytes = len(zip_bytes)

        # Verify checksum if requested and available.
        if verify:
            try:
                chk_bytes = _fetch_with_retry(fetcher, url + ".CHECKSUM",
                                              retries, backoff, timeout)
                expected = parse_checksum(chk_bytes)
                actual = sha256_bytes(zip_bytes)
                if expected and actual != expected:
                    res.status = "checksum_failed"
                    res.checksum_verified = False
                    res.error = f"sha256 {actual[:12]}… != expected {expected[:12]}…"
                    results.append(res)
                    log.error("CHECKSUM MISMATCH (not extracting): %s", url)
                    continue
                res.checksum_verified = bool(expected)
            except NotFound:
                res.checksum_verified = None  # reported, not assumed OK
                log.warning("no .CHECKSUM for %s — proceeding unverified", url)
            except Exception as e:  # noqa: BLE001
                res.checksum_verified = None
                log.warning("checksum fetch failed for %s: %s", url, e)

        # Persist / extract.
        try:
            leaf.mkdir(parents=True, exist_ok=True)
            if keep_zip or not extract:
                zip_path.write_bytes(zip_bytes)
                res.path = str(zip_path)
            if extract:
                out_csv = _extract_csv(zip_bytes, leaf)
                res.path = str(out_csv)
            res.status = "downloaded"
            results.append(res)
            log.info("downloaded%s: %s",
                     "" if res.checksum_verified else " (unverified)", res.path)
        except Exception as e:  # noqa: BLE001
            res.status = "error"
            res.error = f"{type(e).__name__}: {e}"
            results.append(res)
            log.error("error writing/extracting %s: %s", url, res.error)

    return results


def download_klines(symbol: str = "BTCUSDT", interval: str = "1m",
                    start: str = "2024-01", end: str = "2024-01",
                    market: str = "spot", period: str = "monthly",
                    out_dir: str | Path = "data", **kwargs) -> list[FileResult]:
    """Convenience wrapper for the common case: kline archives over a range."""
    return download_range("klines", symbol, start, end, interval=interval,
                          market=market, period=period, out_dir=out_dir, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def summarize(results: Iterable[FileResult], *, printout: bool = True) -> dict:
    """Tally results by status and (optionally) print an honest run report."""
    results = list(results)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(results)
    n_unverified = sum(1 for r in results
                       if r.status == "downloaded" and r.checksum_verified is None)
    total_bytes = sum(r.n_bytes for r in results)

    if printout:
        print("\n" + "=" * 64)
        print(f"  Binance download summary — {total} file(s), "
              f"{total_bytes / 1e6:.1f} MB fetched")
        print("=" * 64)
        for status in ("downloaded", "exists", "not_found",
                       "checksum_failed", "error"):
            if counts.get(status):
                print(f"    {status:<16} {counts[status]}")
        if n_unverified:
            print(f"    (of downloaded, {n_unverified} had no .CHECKSUM "
                  f"— unverified)")
        problems = [r for r in results
                    if r.status in ("checksum_failed", "error")]
        if problems:
            print("\n  Problems:")
            for r in problems:
                print(f"    [{r.status}] {r.date}: {r.error}")
        print("=" * 64 + "\n")

    return {"total": total, "counts": counts,
            "unverified": n_unverified, "bytes": total_bytes}


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Binance public-data archives (data.binance.vision)."
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m",
                   help="kline interval (ignored for non-kline data types)")
    p.add_argument("--start", required=True,
                   help="YYYY-MM (monthly) or YYYY-MM-DD (daily)")
    p.add_argument("--end", required=True,
                   help="YYYY-MM (monthly) or YYYY-MM-DD (daily)")
    p.add_argument("--data-type", default="klines",
                   help="klines | aggTrades | trades | ...")
    p.add_argument("--market", default="spot", choices=list(MARKETS))
    p.add_argument("--daily", action="store_true",
                   help="use daily files instead of monthly")
    p.add_argument("--out", default="data", help="output cache directory")
    p.add_argument("--no-verify", action="store_true",
                   help="skip SHA-256 checksum verification")
    p.add_argument("--no-extract", action="store_true",
                   help="keep the .zip, do not extract the CSV")
    p.add_argument("--keep-zip", action="store_true",
                   help="keep the .zip alongside the extracted CSV")
    p.add_argument("--force", action="store_true",
                   help="re-download even if the file already exists")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    interval = args.interval if args.data_type in _INTERVAL_TYPES else None
    results = download_range(
        args.data_type, args.symbol, args.start, args.end,
        interval=interval, market=args.market,
        period="daily" if args.daily else "monthly",
        out_dir=args.out, verify=not args.no_verify,
        extract=not args.no_extract, keep_zip=args.keep_zip, force=args.force,
    )
    summary = summarize(results)
    # Non-zero exit if anything genuinely failed (not for benign 404s).
    bad = summary["counts"].get("checksum_failed", 0) + summary["counts"].get("error", 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
