"""
estimate_h.py — run the roughness estimators on a real RV series
================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions   (Phase B, brick 4)

Takes a log-realized-variance series — either a processed RV CSV (from
rv_series.py) or kline CSVs it builds the series from — and runs all three
Layer 1c estimators (GJR, Cont–Das p-variation, MF-DFA), reporting each Ĥ
ALONGSIDE the trust signal that says whether to believe it, plus the
cross-estimator disagreement, which is the scientifically interesting part.

Why report disagreement, not a single number
---------------------------------------------
The audit (layer1c_roughness_audit) established that the three estimators
carry DIFFERENT small-H biases — GJR and Cont–Das bias UP, MF-DFA biases DOWN,
and the disagreement is intrinsic, not finite-sample. So a lone Ĥ is a trap:
the honest read is the spread across estimators and whether they even agree on
rough-versus-smooth (Ĥ below or above the Brownian 0.5). When they straddle
0.5, that ambiguity is the result — exactly the fact-or-artefact question the
project exists to referee — not something to average away.

Two diagnostics this adds on top of the bare estimators
-------------------------------------------------------
  * Sampling sweep — rebuild the RV series at 1m / 5m / 15m sampling and watch
    Ĥ move. This is the EMPIRICAL Rung-1 axis: the audit proved the proxy's
    spurious-roughness bias grows as the RV window shrinks, so strong drift of
    Ĥ with sampling is the proxy artefact showing itself on real data; a stable
    Ĥ is more credible. (Needs klines, since a pre-built CSV is one sampling.)
  * Sub-window stability — estimate Ĥ on K contiguous chunks. Preserves the
    within-chunk dependence (unlike a block bootstrap, which would shred the
    long-range dependence being measured and bias Ĥ toward 0.5), so it honestly
    shows whether roughness is stable across the sample or drifts in time.

Everything here still runs on the RV PROXY (Rung 1). Read any Ĥ against the
simulated proxy-bias maps before claiming the roughness is real.

Public API
----------
  analyze(log_vol, multifractal=True)        -> list[EstimateResult]
  sampling_sweep(klines, samplings, ...)      -> list[(sampling, n_obs, {name:H})]
  subwindow_stability(log_vol, k_chunks)      -> dict[name -> (mean, sd, [H...])]
  load_log_rv_csv(path)                       -> (log_rv, log_bv, period_start_ms)
  EstimateResult  (dataclass)

CLI
---
  python estimate_h.py data/spot/klines/BTCUSDT/1m/*.csv --sampling 5m --sweep 1m,5m,15m
  python estimate_h.py data/processed/btc_rv.csv --chunks 4
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")     # estimators live in a mpl module

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

from layer1c_roughness_audit import gjr_hurst, pvariation_hurst, mfdfa_hurst
from rv_series import build_rv_series

__all__ = [
    "EstimateResult",
    "analyze",
    "sampling_sweep",
    "subwindow_stability",
    "load_log_rv_csv",
]

# name -> estimator. Each takes (log_vol, return_detail=True) -> (H, detail).
ESTIMATORS: dict[str, Callable] = {
    "GJR": gjr_hurst,
    "Cont-Das": pvariation_hurst,
    "MF-DFA": mfdfa_hurst,
}

# Documented small-H bias directions, straight from the audit's calibration
# (printed as factual context, never used to silently "correct" an estimate).
BIAS_DIRECTION = {
    "GJR": "biases UP at small H (finite-lag)",
    "Cont-Das": "biases UP at small H",
    "MF-DFA": "biases DOWN at small H (intrinsic)",
}

# q values for the MF-DFA multifractal-width probe (flat h(q) => monofractal).
_MF_QS = np.array([-4.0, -2.0, 2.0, 4.0])


# ──────────────────────────────────────────────────────────────────────────
# result container
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class EstimateResult:
    name: str
    H: float
    ok: bool
    note: str = ""                 # human-readable trust signal
    detail: dict = field(default_factory=dict)


def _trust_note(name: str, H: float, detail: dict, x: np.ndarray,
                multifractal: bool) -> str:
    """One-line trust signal per estimator, from its own detail dict."""
    if name == "GJR":
        r2 = float(detail.get("monofractal_r2", np.nan))
        if not np.isfinite(r2):
            return "monofractal R² unavailable"
        flag = ("linear" if r2 >= 0.98 else
                "mild bend" if r2 >= 0.90 else "MULTIFRACTAL?")
        return f"monofractal R²={r2:.3f} ({flag})"
    if name == "Cont-Das":
        ps = float(detail.get("p_star", np.nan))
        if not np.isfinite(ps):
            return "no p-variation crossing — estimator could not resolve p*"
        return f"critical power p*={ps:.2f}  (H = 1/p*)"
    if name == "MF-DFA":
        hq = float(detail.get("h_q", np.nan))
        base = f"h(2)={hq:.3f}" if np.isfinite(hq) else "h(2) unavailable"
        if multifractal:
            w = _mfdfa_width(x)
            if np.isfinite(w):
                tag = "monofractal" if w < 0.05 else "multifractal spread"
                base += f"; Δh(q)={w:.3f} ({tag})"
        return base
    return ""


def _mfdfa_width(x: np.ndarray) -> float:
    """Spread of MF-DFA generalised Hurst across q (flat => monofractal)."""
    hs = []
    for q in _MF_QS:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                h = mfdfa_hurst(x, q=float(q))
            if np.isfinite(h):
                hs.append(h)
        except Exception:
            pass
    return float(max(hs) - min(hs)) if len(hs) >= 2 else np.nan


def _run_one(name: str, fn: Callable, x: np.ndarray,
             multifractal: bool) -> EstimateResult:
    try:
        # The runner deliberately probes estimators on series that may be too
        # short; degenerate scales raise RuntimeWarning ("empty slice") inside
        # the estimator and we convert the result to NaN below, so silence the
        # expected noise rather than leak it to the user's console.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            H, detail = fn(x, return_detail=True)
        H = float(H)
        ok = np.isfinite(H)
        if ok:
            note = _trust_note(name, H, detail, x, multifractal)
            if not (0.0 < H < 1.0):
                note = "OUTSIDE (0,1) — estimator broke on this series. " + note
        else:
            note = "estimate is NaN (series too short / degenerate for this estimator)"
        return EstimateResult(name, H if ok else float("nan"), ok, note, detail)
    except Exception as e:                       # never let one estimator abort
        return EstimateResult(name, float("nan"), False, f"failed: {e}")


def analyze(log_vol: np.ndarray, multifractal: bool = True
            ) -> list[EstimateResult]:
    """Run all three estimators on a log-volatility series."""
    x = np.asarray(log_vol, dtype=float)
    return [_run_one(n, fn, x, multifractal) for n, fn in ESTIMATORS.items()]


# ──────────────────────────────────────────────────────────────────────────
# cross-estimator disagreement (the interesting part)
# ──────────────────────────────────────────────────────────────────────────

def _disagreement_lines(results: list[EstimateResult]) -> list[str]:
    finite = [r for r in results if r.ok]
    if len(finite) < 2:
        return ["  [WARN] fewer than two estimators returned a value — "
                "the series is too short to compare them."]
    Hs = np.array([r.H for r in finite])
    spread = float(Hs.max() - Hs.min())
    out = [f"  spread Δ = {spread:.3f}  "
           f"(min {Hs.min():.3f} {finite[int(Hs.argmin())].name}, "
           f"max {Hs.max():.3f} {finite[int(Hs.argmax())].name})"]
    below = Hs < 0.5
    if below.all():
        out.append("  all estimators below 0.5 — consistent with ROUGH "
                   "volatility (but read against the proxy-bias maps).")
    elif (~below).all():
        out.append("  all estimators at/above 0.5 — no roughness detected "
                   "(consistent with a smooth/Markov vol).")
    else:
        out.append("  estimators STRADDLE 0.5 — they disagree on rough-vs-"
                   "smooth. That disagreement is the finding (the audit shows")
        out.append("  their small-H biases differ in SIGN), not something to "
                   "average away.")
    return out


# ──────────────────────────────────────────────────────────────────────────
# sampling sweep — the empirical Rung-1 axis
# ──────────────────────────────────────────────────────────────────────────

def sampling_sweep(klines_source, samplings: Sequence[str] = ("1m", "5m", "15m"),
                   rv_bar: str = "1d", base_interval: str = "1m"
                   ) -> list[tuple]:
    """Rebuild the RV series at each sampling and re-estimate Ĥ.

    Returns a list of (sampling, n_obs, {estimator_name: H}). Drift of Ĥ across
    samplings is the proxy artefact (Rung 1) showing itself on real data.
    """
    rows = []
    for s in samplings:
        series = build_rv_series(klines_source, sampling=s, rv_bar=rv_bar,
                                 base_interval=base_interval, jump_robust=False)
        if series.n_obs < 2:
            rows.append((s, series.n_obs, {n: np.nan for n in ESTIMATORS}))
            continue
        res = {r.name: r.H for r in analyze(series.log_rv, multifractal=False)}
        rows.append((s, series.n_obs, res))
    return rows


# ──────────────────────────────────────────────────────────────────────────
# sub-window stability — honest uncertainty for dependent data
# ──────────────────────────────────────────────────────────────────────────

def subwindow_stability(log_vol: np.ndarray, k_chunks: int = 4
                        ) -> dict[str, tuple]:
    """Estimate Ĥ on K contiguous, non-overlapping chunks.

    Preserves within-chunk dependence (a block bootstrap would destroy it and
    pull Ĥ toward 0.5), so the chunk-to-chunk spread reflects both estimation
    noise AND genuine time-variation in roughness — reported transparently.
    Returns {name: (mean, sd, [per-chunk H])}.
    """
    x = np.asarray(log_vol, dtype=float)
    chunks = np.array_split(x, k_chunks)
    per: dict[str, list] = {n: [] for n in ESTIMATORS}
    for ch in chunks:
        for r in analyze(ch, multifractal=False):
            per[r.name].append(r.H)
    out = {}
    for n, vals in per.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            out[n] = (float(finite.mean()), float(finite.std()), vals)
        else:
            out[n] = (np.nan, np.nan, vals)
    return out


# ──────────────────────────────────────────────────────────────────────────
# input: processed RV CSV
# ──────────────────────────────────────────────────────────────────────────

def load_log_rv_csv(path: Union[str, Path]):
    """Read a processed RV CSV (from rv_series.py) -> (log_rv, log_bv, ms)."""
    p = Path(path)
    lines = p.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"empty file: {path}")
    header = [h.strip() for h in lines[0].split(",")]
    try:
        i_rv = header.index("log_rv")
    except ValueError:
        raise ValueError(f"{path}: no 'log_rv' column (header: {header})")
    i_bv = header.index("log_bv") if "log_bv" in header else None
    i_ms = header.index("period_start_ms") if "period_start_ms" in header else None
    rv, bv, ms = [], [], []
    for ln in lines[1:]:
        f = ln.split(",")
        rv.append(float(f[i_rv]))
        if i_bv is not None:
            bv.append(float(f[i_bv]) if f[i_bv].strip() else np.nan)
        if i_ms is not None:
            ms.append(int(f[i_ms]))
    return (np.array(rv),
            np.array(bv) if bv else np.full(len(rv), np.nan),
            np.array(ms, dtype=np.int64) if ms else np.empty(0, np.int64))


def _looks_like_rv_csv(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not (p.is_file() and p.suffix == ".csv"):
        return False
    try:
        first = p.read_text().splitlines()[0]
    except Exception:
        return False
    return "log_rv" in first


# ──────────────────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────────────────

def _print_report(results: list[EstimateResult], n_obs: int,
                  span: Optional[tuple] = None, label: str = "",
                  on: str = "log RV", stream=sys.stdout) -> None:
    def line(s: str = "") -> None:
        print(s, file=stream)

    line("\n" + "=" * 70)
    line(f"  Hurst estimates on {on}" + (f"  [{label}]" if label else ""))
    line("=" * 70)
    line(f"  series      : {n_obs:,} observations")
    if span:
        def utc(ms): return datetime.fromtimestamp(
            ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        line(f"  span        : {utc(span[0])}  ->  {utc(span[1])}")
    if n_obs < 250:
        line(f"  [WARN] {n_obs:,} points is short — treat every Ĥ below as "
             f"indicative only, not inferential.")
    line("-" * 70)
    for r in results:
        h = f"{r.H:+.4f}" if r.ok else "  nan  "
        line(f"  {r.name:<10} Ĥ = {h}   {r.note}")
        line(f"  {'':<10} ({BIAS_DIRECTION.get(r.name, '')})")
    line("-" * 70)
    for s in _disagreement_lines(results):
        line(s)
    line("=" * 70)
    line("  this is the RV PROXY (Layer 1c Rung 1): read Ĥ against the "
         "simulated\n  proxy-bias maps before concluding the roughness is real.")
    line("=" * 70 + "\n")


def _print_sweep(rows: list[tuple], stream=sys.stdout) -> None:
    def line(s: str = "") -> None:
        print(s, file=stream)
    names = list(ESTIMATORS)
    line("  Sampling sweep — Ĥ vs RV window (drift = Rung-1 proxy artefact)")
    line("  " + "-" * 60)
    line(f"  {'sampling':<10}{'n_obs':>8}   " +
         "".join(f"{n:>12}" for n in names))
    for s, n_obs, res in rows:
        cells = "".join(
            f"{(f'{res[n]:+.3f}' if np.isfinite(res[n]) else 'nan'):>12}"
            for n in names)
        line(f"  {s:<10}{n_obs:>8}   {cells}")
    line("  " + "-" * 60)
    line("  (if Ĥ falls steadily as sampling gets finer, that's the proxy "
         "manufacturing\n   roughness; a flat row is the credible case.)\n")


def _print_stability(stab: dict, k: int, stream=sys.stdout) -> None:
    def line(s: str = "") -> None:
        print(s, file=stream)
    line(f"  Sub-window stability — Ĥ across {k} contiguous chunks")
    line("  " + "-" * 60)
    for name, (mean, sd, vals) in stab.items():
        cells = ", ".join(f"{v:+.3f}" if np.isfinite(v) else "nan" for v in vals)
        m = f"{mean:+.3f} ± {sd:.3f}" if np.isfinite(mean) else "nan"
        line(f"  {name:<10} {m}   [{cells}]")
    line("  " + "-" * 60)
    line("  (large chunk-to-chunk spread = roughness drifts in time, or the "
         "chunks are\n   too short to estimate; either way, not a single "
         "stable number.)\n")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Estimate the Hurst exponent of volatility from an RV "
                    "series (three estimators + trust signals + disagreement).")
    p.add_argument("source", nargs="+",
                   help="a processed RV CSV, or kline CSV file(s)/dir/glob")
    p.add_argument("--sampling", default="5m", help="RV sampling (klines input)")
    p.add_argument("--rv-bar", default="1d", help="RV bar (klines input)")
    p.add_argument("--base-interval", default="1m")
    p.add_argument("--sweep", default=None,
                   help="comma list of samplings to sweep, e.g. 1m,5m,15m "
                        "(klines input only)")
    p.add_argument("--chunks", type=int, default=0,
                   help="sub-window stability across this many chunks (e.g. 4)")
    p.add_argument("--symbol", default=None)
    args = p.parse_args(argv)

    single = args.source[0] if len(args.source) == 1 else args.source

    # ── route input: processed RV CSV vs klines ───────────────────────────
    if isinstance(single, str) and _looks_like_rv_csv(single):
        log_rv, _, ms = load_log_rv_csv(single)
        span = (int(ms[0]), int(ms[-1])) if ms.size else None
        results = analyze(log_rv)
        _print_report(results, log_rv.size, span, label=args.symbol or "",
                      on="log RV (from CSV)")
        if args.sweep:
            print("  [note] --sweep needs kline input (a processed CSV is "
                  "already one sampling); skipping.\n")
        if args.chunks > 1:
            _print_stability(subwindow_stability(log_rv, args.chunks), args.chunks)
        return 0

    # klines -> build RV at the chosen sampling, then analyze
    series = build_rv_series(single, sampling=args.sampling, rv_bar=args.rv_bar,
                             base_interval=args.base_interval, symbol=args.symbol)
    if series.n_obs < 2:
        print("  [FAIL] could not build an RV series (not enough data).")
        return 1
    span = (int(series.period_start_ms[0]), int(series.period_start_ms[-1]))
    results = analyze(series.log_rv)
    _print_report(results, series.n_obs, span, label=args.symbol or "",
                  on=f"log RV ({args.sampling} sampling, {args.rv_bar} bars)")

    if args.sweep:
        samplings = [s.strip() for s in args.sweep.split(",") if s.strip()]
        _print_sweep(sampling_sweep(single, samplings, rv_bar=args.rv_bar,
                                    base_interval=args.base_interval))
    if args.chunks > 1:
        _print_stability(subwindow_stability(series.log_rv, args.chunks),
                         args.chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
