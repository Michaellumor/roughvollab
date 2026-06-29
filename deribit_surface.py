"""
deribit_surface.py — Deribit option-surface fetcher + cleaner (Layer 4, real-market D39)
========================================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic Optimal Control
         under Market Frictions

Real-market arm of the calibration story (D32–D38 built/validated the engine in the
sandbox; this fetches the live target). Pulls the Deribit BTC (or ETH) option chain,
cleans it into a multi-maturity implied-vol SURFACE, and hands it to the validated
rough-Heston engine (`layer4_calibrate_surface.py`, D38). Crypto, because the data is
free + accessible and it is the high-roughness / high-vol-of-vol regime the lift was
built for; the same engine transfers to SPX unchanged when SPX option data is available.

Conventions (pinned from the live API, not assumed — see ROADMAP D39 / the plan):
  · IV is quoted in PERCENT (mark_iv 43.75 -> 0.4375 decimal).
  · MONEYNESS USES THE FORWARD: each expiry's `underlying_price` (NOT `index_price`,
    the spot). Options are inverse/BTC-settled, but that is IRRELEVANT in IV space —
    we calibrate to mark_iv directly.
  · The engine is S0=100, r=0, so each expiry is normalised to its own forward:
    K_norm = 100 * strike / forward(T);  T = days_to_expiry / 365.

The HTTP GET is isolated behind one injectable `fetcher` callable (the binance_data.py
pattern), so parsing + cleaning are unit-testable offline with a fake feed (the sandbox
reaches Deribit, but tests must not depend on a live, ever-changing chain). Raw snapshots
are saved under data/deribit/ (gitignored) so a calibration is reproducible.

Public API
----------
  fetch_instruments / fetch_book_summary / fetch_ticker        (raw, injectable)
  parse_instrument(inst, ticker, now_ms) -> OptionQuote
  clean(quotes, **thresholds) -> (kept, drops)
  to_grids_target_weights(kept) -> (grids_by_T, target_by_T, weights_by_T, meta)
  save_snapshot / load_snapshot
  fetch_and_clean(currency="BTC", ...) / clean_from_snapshot(path, ...)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

__all__ = [
    "DERIBIT_BASE", "DeribitError", "OptionQuote",
    "fetch_instruments", "fetch_book_summary", "fetch_order_book",
    "parse_instrument", "clean", "to_grids_target_weights",
    "save_snapshot", "load_snapshot", "fetch_and_clean", "clean_from_snapshot",
]

DERIBIT_BASE = "https://www.deribit.com/api/v2"

# Default calibration grid (the plan's pinned decisions; overridable).
WANT_EXPIRIES = ("10JUL26", "31JUL26", "28AUG26", "25SEP26", "25DEC26", "25JUN27")
# Put-wing + ATM, selected by Black-delta (forward/vol/T-normalised -> consistent depth).
PUT_DELTA_LO, PUT_DELTA_HI = -0.50, -0.10           # OTM-put wing (10Δ..50Δ)
ATM_ABS_LO, ATM_ABS_HI = 0.40, 0.60                 # near-ATM, either type
# Cleaning thresholds (recon-grounded).
OI_MIN = 5.0
SPREAD_PTS_MAX = 5.0                                 # ask_iv - bid_iv, vol points
SPREAD_FRAC_MAX = 0.15                               # ... or fraction of mark_iv
VEGA_MIN = 5.0
MONEYNESS_PREFILTER = (0.50, 1.20)                   # generous K/forward band before ticker calls


class DeribitError(Exception):
    """Deribit JSON-RPC returned an error payload, or the HTTP layer failed."""


# --------------------------------------------------------------------------- #
# HTTP layer (injectable fetcher; binance_data.py pattern)
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "roughvollab-deribit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _api_get(method: str, params: dict, *, fetcher: Optional[Callable[[str, float], bytes]] = None,
             timeout: float = 30.0, retries: int = 4, backoff: float = 0.5) -> object:
    """GET {BASE}/public/{method}?{params}; return payload['result']. Retries transient
    errors with exponential backoff; raises DeribitError on a JSON-RPC error."""
    fetcher = fetcher or _http_get
    url = f"{DERIBIT_BASE}/public/{method}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(retries + 1):
        try:
            payload = json.loads(fetcher(url, timeout))
            if "error" in payload and payload["error"]:
                raise DeribitError(f"{method}: {payload['error']}")
            return payload["result"]
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):                      # client error -> permanent, don't retry
                raise DeribitError(f"{method}: HTTP {e.code} (no retry): {e}")
            last = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise DeribitError(f"{method}: HTTP failed after {retries} retries: {e}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise DeribitError(f"{method}: HTTP failed after {retries} retries: {e}")
    raise DeribitError(f"{method}: {last}")


def _paced_map(fn, items, *, min_interval: float = 0.06, progress: bool = False) -> list:
    """Call fn(item) sequentially with rate pacing (Deribit public soft cap ~20 req/s;
    the limit is global, so sequential not threaded). Returns [fn(item), ...]."""
    out = []
    for i, it in enumerate(items):
        t0 = time.time()
        out.append(fn(it))
        if progress and (i + 1) % 25 == 0:
            print(f"    fetched {i+1}/{len(items)}", flush=True)
        dt = time.time() - t0
        if dt < min_interval:
            time.sleep(min_interval - dt)
    return out


# --------------------------------------------------------------------------- #
# raw fetchers (each takes an injectable fetcher for offline tests)
# --------------------------------------------------------------------------- #
def fetch_instruments(currency: str = "BTC", *, fetcher=None) -> list:
    return _api_get("get_instruments",
                    {"currency": currency, "kind": "option", "expired": "false"}, fetcher=fetcher)


def fetch_book_summary(currency: str = "BTC", *, fetcher=None) -> list:
    return _api_get("get_book_summary_by_currency",
                    {"currency": currency, "kind": "option"}, fetcher=fetcher)


def fetch_order_book(instrument_name: str, *, fetcher=None) -> dict:
    """Per-instrument quote (mark_iv/bid_iv/ask_iv/greeks/forward/OI). Deribit's method is
    `get_order_book` — `get_ticker` does NOT exist (verified: -32601 Method not found)."""
    return _api_get("get_order_book", {"instrument_name": instrument_name}, fetcher=fetcher)


# --------------------------------------------------------------------------- #
# parsing — from FIELDS, never the instrument name
# --------------------------------------------------------------------------- #
@dataclass
class OptionQuote:
    instrument: str
    expiry_ms: int
    T_years: float
    strike: float
    opt_type: str                       # 'call' | 'put'
    forward: float                      # underlying_price (per-expiry forward)
    spot: float                         # index_price
    mark_iv: float                      # PERCENT, as returned
    bid_iv: Optional[float]
    ask_iv: Optional[float]
    vega: Optional[float]
    delta: Optional[float]
    oi: float
    volume: float
    K_norm: float = float("nan")        # 100 * strike / forward (filled at clean)


def _expiry_label(instrument_name: str) -> str:
    return instrument_name.split("-")[1]


def parse_instrument(inst: dict, ticker: dict, now_ms: int) -> OptionQuote:
    exp_ms = int(inst["expiration_timestamp"])
    T = (exp_ms / 1000.0 - now_ms / 1000.0) / (365.0 * 86400.0)
    g = ticker.get("greeks") or {}
    stats = ticker.get("stats") or {}
    return OptionQuote(
        instrument=inst["instrument_name"], expiry_ms=exp_ms, T_years=T,
        strike=float(inst["strike"]), opt_type=inst["option_type"],
        forward=float(ticker.get("underlying_price") or 0.0),
        spot=float(ticker.get("index_price") or 0.0),
        mark_iv=float(ticker.get("mark_iv") or "nan"),
        bid_iv=ticker.get("bid_iv"), ask_iv=ticker.get("ask_iv"),
        vega=g.get("vega"), delta=g.get("delta"),
        oi=float(ticker.get("open_interest") or 0.0),
        volume=float(stats.get("volume") or 0.0),
    )


# --------------------------------------------------------------------------- #
# cleaning pipeline (pure; each filter tagged for the drop report)
# --------------------------------------------------------------------------- #
def _is_finite(x) -> bool:
    return x is not None and x == x and abs(float(x)) != float("inf")


def clean(quotes: list, *, oi_min=OI_MIN, spread_pts_max=SPREAD_PTS_MAX,
          spread_frac_max=SPREAD_FRAC_MAX, vega_min=VEGA_MIN,
          put_delta=(PUT_DELTA_LO, PUT_DELTA_HI), atm_abs=(ATM_ABS_LO, ATM_ABS_HI),
          arb_tol=1e-3):
    """Ordered filters. Returns (kept, drops) where drops = {reason: count}."""
    drops: dict = {}
    def drop(q, reason):
        drops[reason] = drops.get(reason, 0) + 1

    stage1 = []
    for q in quotes:
        if not (_is_finite(q.mark_iv) and 5.0 <= q.mark_iv <= 200.0):
            drop(q, "mark_iv_insane"); continue
        if not (q.oi >= oi_min):
            drop(q, "illiquid_oi"); continue
        if not (_is_finite(q.bid_iv) and _is_finite(q.ask_iv) and q.ask_iv > q.bid_iv):
            drop(q, "not_two_sided"); continue
        sp = q.ask_iv - q.bid_iv
        if sp > spread_pts_max or sp > spread_frac_max * q.mark_iv:
            drop(q, "wide_spread"); continue
        if not (_is_finite(q.vega) and abs(q.vega) >= vega_min):
            drop(q, "low_vega"); continue
        if not _is_finite(q.delta):
            drop(q, "no_delta"); continue
        d = q.delta
        in_put = (q.opt_type == "put" and put_delta[0] <= d <= put_delta[1])
        in_atm = (atm_abs[0] <= abs(d) <= atm_abs[1])
        if not (in_put or in_atm):
            drop(q, "outside_region"); continue
        if q.forward <= 0:
            drop(q, "no_forward"); continue
        q.K_norm = 100.0 * q.strike / q.forward
        stage1.append(q)

    # de-dup at (T, K_norm): keep the OTM side (put if K_norm < 100 else call)
    by_key: dict = {}
    for q in stage1:
        key = (round(q.expiry_ms), round(q.K_norm, 4))
        prefer_put = q.K_norm < 100.0
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = q
        else:
            q_is_pref = (q.opt_type == "put") == prefer_put
            cur_is_pref = (cur.opt_type == "put") == prefer_put
            if q_is_pref and not cur_is_pref:
                by_key[key] = q
            else:
                drop(q, "dup_strike")
    deduped = list(by_key.values())

    # light static no-arb: per expiry, total variance w(k)=IV^2 T must be ~convex in
    # k=ln(K_norm/100); drop interior points that bulge above the neighbour chord.
    kept = []
    from collections import defaultdict
    import math
    groups = defaultdict(list)
    for q in deduped:
        groups[q.expiry_ms].append(q)
    for exp, gq in groups.items():
        gq.sort(key=lambda q: q.K_norm)
        n = len(gq)
        ok = [True] * n
        if n >= 3:
            k = [math.log(q.K_norm / 100.0) for q in gq]
            w = [(q.mark_iv / 100.0) ** 2 * q.T_years for q in gq]
            for i in range(1, n - 1):
                if k[i + 1] == k[i - 1]:
                    continue
                t = (k[i] - k[i - 1]) / (k[i + 1] - k[i - 1])
                chord = w[i - 1] + t * (w[i + 1] - w[i - 1])
                if w[i] > chord + arb_tol:
                    ok[i] = False
        for keep_it, q in zip(ok, gq):
            if keep_it:
                kept.append(q)
            else:
                drop(q, "arb_convexity")
    return kept, drops


def to_grids_target_weights(kept: list, *, w_floor=0.2, w_cap=5.0):
    """Assemble engine-ready surface. grids_by_T[T] and target_by_T[T] / weights_by_T[T]
    are in LOCKSTEP, each block sorted by K_norm ascending (sorted(Ts) handled by the
    engine's stacking). Returns (grids_by_T, target_by_T, weights_by_T, meta)."""
    import math
    from collections import defaultdict
    groups = defaultdict(list)
    for q in kept:
        groups[q.T_years].append(q)
    grids_by_T, target_by_T, weights_by_T = {}, {}, {}
    raw_w = []
    for T, gq in groups.items():
        gq.sort(key=lambda q: q.K_norm)
        for q in gq:
            raw_w.append(1.0 / max(q.ask_iv - q.bid_iv, 1e-3))
    mean_w = (sum(raw_w) / len(raw_w)) if raw_w else 1.0
    import numpy as np
    for T, gq in groups.items():
        gq.sort(key=lambda q: q.K_norm)
        grids_by_T[T] = np.array([q.K_norm for q in gq])
        target_by_T[T] = np.array([q.mark_iv / 100.0 for q in gq])
        w = np.array([min(max((1.0 / max(q.ask_iv - q.bid_iv, 1e-3)) / mean_w, w_floor), w_cap)
                      for q in gq])
        weights_by_T[T] = w
    meta = {
        "n_kept": len(kept),
        "maturities": {f"{T:.4f}": len(grids_by_T[T]) for T in sorted(grids_by_T)},
        "forwards": {f"{T:.4f}": float(np.mean([q.forward for q in groups[T]])) for T in sorted(groups)},
        "atm_iv_by_T": {f"{T:.4f}": float(np.interp(100.0, grids_by_T[T], target_by_T[T]))
                        for T in sorted(grids_by_T)},
    }
    return grids_by_T, target_by_T, weights_by_T, meta


# --------------------------------------------------------------------------- #
# snapshot I/O (raw JSON -> reproducible offline calibration)
# --------------------------------------------------------------------------- #
def save_snapshot(currency: str, instruments: list, tickers: dict, *,
                  out_dir: str = "data/deribit", fetched_utc: Optional[str] = None) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = fetched_utc or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out / f"{currency}_{stamp}.json"
    path.write_text(json.dumps({"fetched_utc": stamp, "currency": currency,
                                "instruments": instruments, "tickers": tickers}), encoding="utf-8")
    return path


def load_snapshot(path) -> tuple:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["instruments"], d["tickers"], {"fetched_utc": d.get("fetched_utc"), "currency": d.get("currency")}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _select_candidates(instruments, summary_by_name, want_expiries, moneyness):
    """Pre-filter the chain (cheap, no ticker calls): selected expiries × OI floor ×
    generous moneyness band. Returns the instrument dicts that earn a ticker call."""
    want = set(want_expiries)
    out = []
    for inst in instruments:
        name = inst["instrument_name"]
        if _expiry_label(name) not in want:
            continue
        s = summary_by_name.get(name)
        if s is None:
            continue
        fwd = s.get("underlying_price") or 0.0
        oi = s.get("open_interest") or 0.0
        if fwd <= 0 or oi < OI_MIN:
            continue
        m = float(inst["strike"]) / fwd
        if not (moneyness[0] <= m <= moneyness[1]):
            continue
        out.append(inst)
    return out


def fetch_and_clean(currency: str = "BTC", *, want_expiries=WANT_EXPIRIES,
                    moneyness=MONEYNESS_PREFILTER, fetcher=None, save=True, verbose=True, **clean_kw):
    """Full online path: list chain -> book-summary pre-filter -> per-instrument tickers
    -> parse -> clean -> assemble. Saves a raw snapshot. Returns the clean-from-snapshot tuple."""
    if verbose:
        print(f"[deribit] fetching {currency} option chain ...", flush=True)
    instruments = fetch_instruments(currency, fetcher=fetcher)
    summary = fetch_book_summary(currency, fetcher=fetcher)
    summary_by_name = {s["instrument_name"]: s for s in summary}
    cands = _select_candidates(instruments, summary_by_name, want_expiries, moneyness)
    if verbose:
        print(f"[deribit] chain={len(instruments)}  pre-filtered candidates={len(cands)} "
              f"(expiries {want_expiries})", flush=True)
    names = [c["instrument_name"] for c in cands]
    tickers_list = _paced_map(lambda nm: fetch_order_book(nm, fetcher=fetcher), names, progress=verbose)
    tickers = {nm: tk for nm, tk in zip(names, tickers_list)}
    if save:
        path = save_snapshot(currency, cands, tickers)
        if verbose:
            print(f"[deribit] snapshot saved -> {path}", flush=True)
    return _assemble(cands, tickers, want_expiries=want_expiries, verbose=verbose, **clean_kw)


def clean_from_snapshot(path, *, want_expiries=WANT_EXPIRIES, verbose=True, **clean_kw):
    instruments, tickers, meta = load_snapshot(path)
    return _assemble(instruments, tickers, want_expiries=want_expiries, verbose=verbose,
                     snapshot=str(path), **clean_kw)


def _assemble(instruments, tickers, *, want_expiries, verbose=True, snapshot=None, **clean_kw):
    now_ms = int(time.time() * 1000)
    quotes = []
    for inst in instruments:
        tk = tickers.get(inst["instrument_name"])
        if tk is None:
            continue
        if _expiry_label(inst["instrument_name"]) not in set(want_expiries):
            continue
        quotes.append(parse_instrument(inst, tk, now_ms))
    kept, drops = clean(quotes, **clean_kw)
    grids, target, weights, meta = to_grids_target_weights(kept)
    meta["drops"] = drops
    meta["n_raw"] = len(quotes)
    meta["snapshot"] = snapshot
    if verbose:
        print(f"[deribit] cleaned: {len(quotes)} quotes -> {meta['n_kept']} kept | drops={drops}")
        print(f"[deribit] surface: {len(grids)} maturities {meta['maturities']}")
        print(f"[deribit] ATM IV by T: { {k: round(v,4) for k,v in meta['atm_iv_by_T'].items()} }")
    return grids, target, weights, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--snapshot", default=None, help="clean from a saved snapshot instead of fetching")
    a = ap.parse_args()
    if a.snapshot:
        clean_from_snapshot(a.snapshot)
    else:
        fetch_and_clean(a.currency)
