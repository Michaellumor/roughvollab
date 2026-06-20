"""
interpret_h.py — de-bias a real-data Hurst estimate against the Rung-1 envelope
===============================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions   (Phase B, brick 5)

estimate_h.py reports the OBSERVED Ĥ on the RV proxy and states each
estimator's bias DIRECTION. This module does the quantitative step it stops
short of: it rebuilds the Rung-1 bias envelope at the SAME measurement
conditions as your real series (same number of RV observations, same returns-
per-bar), then INVERTS it — given the observed Ĥ, what true Hurst H could have
produced it through the proxy?

Why this is the actual Phase B question
---------------------------------------
The audit's envelope established an observational-equivalence problem: through
a noisy RV proxy the estimate COLLAPSES toward H ≈ 0.1 almost regardless of the
true H, so a raw market reading of Ĥ ≈ 0.1 is nearly uninformative. De-biasing
confronts that head-on: it maps observed Ĥ back to implied true H AND reports
where the map is ill-posed (the bias curve is locally flat — the collapse
zone), so a "rough" reading that is really just the proxy collapsing is flagged
as such rather than trusted.

The honest test is not a single de-biased number. It is: (a) do the three
estimators, which disagree on the raw observed Ĥ, AGREE on the implied true H
once de-biased? and (b) is that implied true H robustly below 0.5 even after
correction, or does it sit in / straddle the collapse zone where the proxy
cannot decide rough-from-smooth?

Heavy honesty caveats (printed in the report)
---------------------------------------------
  1. MODEL-CONDITIONAL. The bias curve is simulated under the rough-Bergomi /
     RFSV model (the same generator Phase A validated), with a fixed vol-of-vol
     and leverage. If real volatility is not generated that way, the implied
     true H is only as good as the model. Tune --eta/--rho/--xi0 toward an
     asset-calibrated setting before trusting the level.
  2. Rung-1 ONLY. This corrects the proxy / finite-sample bias. It does NOT
     remove residual microstructure noise (Rung 2); that shows up as Ĥ drifting
     with sampling in estimate_h.py --sweep, and should be checked separately.
  3. COLLAPSE. Where the bias curve is locally flat, the inversion is
     ill-posed; the implied H there is reported with a wide band and flagged.

Building the envelope is COMPUTE-HEAVY at a real series length (n_obs ~ 2500,
window ~ 288 => ~720k fine steps per Monte-Carlo path). Run it on your machine
with --n-mc 40+; the sandbox validates the machinery at a small scale.

Public API
----------
  observed_H_at(true_H, n_obs, window, n_mc, ...)   -> {name: array of Ĥ}
  build_bias_curve(true_grid, n_obs, window, ...)   -> BiasCurve
  invert(observed_H, true_grid, mean_curve)         -> implied true H
  interpret(source, window=, ...)                   -> Interpretation
  BiasCurve, Interpretation  (dataclasses; .report())

CLI
---
  python interpret_h.py data/processed/btc_rv.csv --window 288 --n-mc 40
  python interpret_h.py data/spot/klines/BTCUSDT/1m/*.csv --sampling 5m --rv-bar 1d --n-mc 40
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from roughvol_core import rough_bergomi_paths
from layer1c_roughness_audit import (
    gjr_hurst, pvariation_hurst, mfdfa_hurst, realized_log_variance,
)
from estimate_h import analyze, load_log_rv_csv, _looks_like_rv_csv
from rv_series import build_rv_series, INTERVAL_MS

__all__ = [
    "observed_H_at",
    "build_bias_curve",
    "invert",
    "interpret",
    "BiasCurve",
    "Interpretation",
]

ESTIMATORS = {"GJR": gjr_hurst, "Cont-Das": pvariation_hurst, "MF-DFA": mfdfa_hurst}

# True-H grid: dense in the empirically-relevant rough zone (small H), where the
# proxy's collapse lives and the fact-or-artefact debate actually sits.
DEFAULT_TRUE_GRID = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60])

# Below this local slope d(observed Ĥ)/d(true H), the inversion is ill-posed:
# a flat bias curve means many true H map to the same observed Ĥ (the collapse).
_FLAT_SLOPE = 0.25


def _est(fn, x) -> float:
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = fn(x)
        return float(h) if np.isfinite(h) else np.nan
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────────────────────────────
# the bias envelope, at matched measurement conditions
# ──────────────────────────────────────────────────────────────────────────

def observed_H_at(true_H: float, n_obs: int, window: int, n_mc: int,
                  eta: float = 1.0, rho: float = -0.7, xi0: float = 0.04,
                  seed: int = 0) -> dict:
    """Monte-Carlo distribution of SINGLE-PATH Ĥ when the true Hurst is true_H.

    Real data is one path of n_obs RV observations, so we simulate single paths
    (not pooled) of exactly that length and proxy window, and report the spread
    of Ĥ across n_mc realisations — the irreducible single-sample uncertainty.
    """
    n_fine = n_obs * window
    rng = np.random.default_rng(seed)
    out = {n: [] for n in ESTIMATORS}
    for _ in range(n_mc):
        _, S, _ = rough_bergomi_paths(n_fine, true_H, n_paths=1, eta=eta,
                                      rho=rho, xi0=xi0, rng=rng)
        x = realized_log_variance(S, window)[0]      # single-path daily log-RV
        for name, fn in ESTIMATORS.items():
            out[name].append(_est(fn, x))
    return {n: np.array(v) for n, v in out.items()}


@dataclass
class BiasCurve:
    true_grid: np.ndarray
    mean: dict                  # name -> mean observed Ĥ per true H
    std: dict                   # name -> std observed Ĥ per true H
    n_obs: int
    window: int
    n_mc: int
    eta: float = 1.0
    rho: float = -0.7
    xi0: float = 0.04


def build_bias_curve(true_grid=DEFAULT_TRUE_GRID, n_obs: int = 2500,
                     window: int = 288, n_mc: int = 40, eta: float = 1.0,
                     rho: float = -0.7, xi0: float = 0.04,
                     seed: int = 505) -> BiasCurve:
    """Simulate the observed-Ĥ vs true-H map at the given measurement conditions."""
    true_grid = np.asarray(true_grid, float)
    mean = {n: [] for n in ESTIMATORS}
    std = {n: [] for n in ESTIMATORS}
    for i, H in enumerate(true_grid):
        d = observed_H_at(H, n_obs, window, n_mc, eta, rho, xi0, seed + i)
        for name in ESTIMATORS:
            v = d[name][np.isfinite(d[name])]
            mean[name].append(float(v.mean()) if v.size else np.nan)
            std[name].append(float(v.std()) if v.size else np.nan)
    return BiasCurve(true_grid,
                     {n: np.array(mean[n]) for n in ESTIMATORS},
                     {n: np.array(std[n]) for n in ESTIMATORS},
                     n_obs, window, n_mc, eta, rho, xi0)


# ──────────────────────────────────────────────────────────────────────────
# inversion: observed Ĥ -> implied true H
# ──────────────────────────────────────────────────────────────────────────

def _is_monotone(mean_curve, tol: float = 0.012) -> bool:
    """Is the bias curve monotone (injective) over the grid?

    A non-monotone (hump-shaped) curve — the noisy-proxy regime — is NOT
    injective, so inverting it is fundamentally ambiguous even when only one
    solution happens to fall inside the grid (the twin sits off-grid). `tol`
    absorbs small Monte-Carlo wobble so genuine monotone curves are not
    misflagged.
    """
    m = np.asarray(mean_curve, float)
    m = m[np.isfinite(m)]
    if m.size < 2:
        return True
    d = np.diff(m)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


def _all_crossings(observed_H, true_grid, mean_curve):
    """All true-H solutions where the piecewise-linear bias curve == observed_H.

    The map true_H -> E[observed Ĥ] is NOT guaranteed monotone: at a noisy proxy
    window it is hump-shaped, so a rough and a much smoother true H can yield the
    SAME observed Ĥ (observational equivalence). We return ALL crossings, not one,
    so a multi-valued / non-identified inversion is EXPOSED rather than silently
    collapsed to a single branch.
    """
    g = np.asarray(true_grid, float)
    m = np.asarray(mean_curve, float)
    fin = np.isfinite(g) & np.isfinite(m)
    g, m = g[fin], m[fin]
    if g.size < 2:
        return []
    order = np.argsort(g)
    g, m = g[order], m[order]
    sols = []
    for i in range(g.size - 1):
        a, b = m[i], m[i + 1]
        if a == observed_H:
            sols.append(float(g[i]))
        if (a - observed_H) * (b - observed_H) < 0 and b != a:
            frac = (observed_H - a) / (b - a)
            sols.append(float(g[i] + frac * (g[i + 1] - g[i])))
    if m[-1] == observed_H:
        sols.append(float(g[-1]))
    out = []
    for s in sorted(sols):
        if not out or abs(s - out[-1]) > 1e-6:
            out.append(s)
    return out


def invert(observed_H: float, true_grid, mean_curve) -> float:
    """Implied true H if the inversion is UNIQUE; NaN if absent or ambiguous."""
    sols = _all_crossings(observed_H, true_grid, mean_curve)
    return sols[0] if len(sols) == 1 else np.nan


def _classify_inversion(observed_H: float, true_grid, mean_curve):
    """Why an inversion did or didn't work: (status, candidates).

    status in {'ok' (one solution), 'multivalued' (>=2 — non-identified),
    'below_floor', 'above_ceiling', 'uncalibrated'}. The two scientifically
    important failure modes: 'below_floor' (observed Ĥ rougher than the model
    produces at any true H — suspect Rung-2/3), and 'multivalued' (rough and
    smooth give the same observed Ĥ — true H not identified through this proxy).
    """
    g = np.asarray(true_grid, float)
    m = np.asarray(mean_curve, float)
    fin = np.isfinite(g) & np.isfinite(m)
    m = m[fin]
    if not np.isfinite(observed_H) or m.size < 2:
        return "uncalibrated", []
    sols = _all_crossings(observed_H, true_grid, mean_curve)
    # A non-monotone (hump) curve is non-injective: inversion is ambiguous even
    # if only one solution lands in-grid (the twin is off-grid). This is the
    # honest signal that rough and smooth are observationally equivalent.
    if not _is_monotone(mean_curve):
        return "multivalued", sols
    if not sols:
        if observed_H < m.min():
            return "below_floor", []
        if observed_H > m.max():
            return "above_ceiling", []
        return "uncalibrated", []
    return "ok", sols


def _local_slope(true_grid, mean_curve, true_H) -> float:
    """d(observed Ĥ)/d(true H) near true_H — small => ill-posed (collapse)."""
    g = np.asarray(true_grid, float)
    m = np.asarray(mean_curve, float)
    ok = np.isfinite(m) & np.isfinite(g)
    g, m = g[ok], m[ok]
    if g.size < 2 or not np.isfinite(true_H):
        return np.nan
    grad = np.gradient(m, g)
    return float(np.interp(true_H, g, grad))


# ──────────────────────────────────────────────────────────────────────────
# the interpretation
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Interpretation:
    observed: dict                 # name -> observed Ĥ on the real series
    implied: dict                  # name -> implied true H (NaN if uninvertible)
    implied_sd: dict               # name -> implied-H uncertainty (slope-propagated)
    collapsed: dict                # name -> bool (curve locally flat => ill-posed)
    reason: dict                   # name -> inversion status (ok/below_floor/...)
    candidates: dict               # name -> all implied-true-H solutions (>=2 => non-identified)
    curve: BiasCurve
    n_obs: int = 0
    label: str = ""

    def _finite_implied(self):
        return {n: h for n, h in self.implied.items()
                if np.isfinite(h) and not self.collapsed.get(n, False)}

    def report(self, stream=sys.stdout) -> "Interpretation":
        def line(s: str = "") -> None:
            print(s, file=stream)
        c = self.curve
        line("\n" + "=" * 72)
        line(f"  De-biasing Ĥ against the Rung-1 envelope"
             + (f"  [{self.label}]" if self.label else ""))
        line("=" * 72)
        line(f"  bias curve : rough-Bergomi MC — n_obs={c.n_obs:,}, "
             f"window={c.window}, n_mc={c.n_mc}")
        line(f"               eta={c.eta}, rho={c.rho}, xi0={c.xi0}  "
             f"(MODEL-CONDITIONAL)")
        line("-" * 72)
        line(f"  {'estimator':<10}{'observed Ĥ':>12}{'implied true H':>18}"
             f"   note")
        for name in ESTIMATORS:
            o = self.observed.get(name, np.nan)
            h = self.implied.get(name, np.nan)
            sd = self.implied_sd.get(name, np.nan)
            rsn = self.reason.get(name, "uncalibrated")
            o_s = f"{o:+.3f}" if np.isfinite(o) else "nan"
            if rsn == "below_floor":
                imp = "  —  "
                note = "BELOW model floor — too rough for Rung-1 to explain"
                if np.isfinite(o) and o < 0:
                    note = "observed Ĥ<0 (unphysical) — " + note
            elif rsn == "multivalued":
                cset = ", ".join(f"{c:.2f}" for c in self.candidates.get(name, []))
                cset = cset or "off-grid"
                imp = "  —  "
                note = f"NON-IDENTIFIED (non-monotone curve, rough≡smooth); soln(s): {cset}"
            elif rsn == "above_ceiling":
                imp, note = "  —  ", "observed Ĥ above calibrated range"
            elif rsn == "uncalibrated" or not np.isfinite(h):
                imp, note = "  —  ", "estimator returned no value (nan)"
            elif self.collapsed.get(name, False):
                imp = f"{h:.3f} ±{sd:.2f}" if np.isfinite(sd) else f"{h:.3f}"
                note = "COLLAPSE — curve flat here, implied H unreliable"
            else:
                imp = f"{h:.3f} ± {sd:.2f}" if np.isfinite(sd) else f"{h:.3f}"
                note = "well-posed"
            line(f"  {name:<10}{o_s:>12}{imp:>18}   {note}")
        line("-" * 72)

        obs = np.array([v for v in self.observed.values() if np.isfinite(v)])
        fin = self._finite_implied()
        if obs.size >= 2:
            line(f"  raw spread (observed Ĥ)   : {obs.max() - obs.min():.3f}")
        if len(fin) >= 2:
            hv = np.array(list(fin.values()))
            dspread = hv.max() - hv.min()
            line(f"  de-biased spread (true H) : {dspread:.3f}  "
                 f"over {len(fin)} well-posed estimator(s)")
            if obs.size >= 2 and dspread < (obs.max() - obs.min()):
                line("  -> de-biasing RECONCILES the estimators (they agree on "
                     "true H better than on raw Ĥ).")
            below = hv < 0.5
            if below.all():
                line(f"  -> implied true H ≈ {np.median(hv):.2f}, all below 0.5 — "
                     "consistent with GENUINE roughness")
                line("     (survives proxy de-biasing).")
            elif (~below).all():
                line("  -> implied true H at/above 0.5 — no roughness once "
                     "de-biased (the raw roughness was proxy artefact).")
            else:
                line("  -> implied true H STRADDLES 0.5 — even de-biased, the "
                     "estimators disagree on rough-vs-smooth.")
        else:
            n_below = sum(1 for r in self.reason.values() if r == "below_floor")
            n_multi = sum(1 for r in self.reason.values() if r == "multivalued")
            if n_multi >= 1:
                line(f"  -> {n_multi} estimator inversion(s) MULTI-VALUED: the same observed Ĥ "
                     "is produced by")
                line("     BOTH a rough and a much smoother true H. Through this proxy rough and "
                     "smooth are")
                line("     OBSERVATIONALLY EQUIVALENT — the true H is NOT identified. A single "
                     "de-biased")
                line("     number here would be an artefact of picking one branch of a "
                     "non-monotone curve.")
            if n_below >= 1:
                line(f"  -> {n_below} estimator(s) BELOW the model floor — rougher than "
                     "rough-Bergomi can")
                line("     produce via Rung-1; suspect microstructure (Rung 2) / jumps "
                     "(Rung 3).")
            if n_multi == 0 and n_below == 0:
                line("  [WARN] too few well-posed inversions to conclude — the series sits in "
                     "the proxy's")
                line("         collapse zone. This is itself the observational-equivalence "
                     "result.")
            line("     Decisive cross-check: estimate_h.py --sweep 1m,5m,15m — does observed Ĥ "
                 "move with")
            line("     sampling? (Stable across windows weakens the microstructure story; "
                 "drift supports it.)")
        line("=" * 72)
        line("  caveats: (1) MODEL-CONDITIONAL (rough Bergomi, params above);")
        line("           (2) corrects Rung-1 proxy bias only — check residual")
        line("               microstructure via estimate_h.py --sweep;")
        line("           (3) COLLAPSE rows: bias curve flat, implied H "
             "unreliable.")
        line("=" * 72 + "\n")
        return self


def interpret(source, window: Optional[int] = None, sampling: str = "5m",
              rv_bar: str = "1d", base_interval: str = "1m", n_mc: int = 40,
              true_grid=DEFAULT_TRUE_GRID, eta: float = 1.0, rho: float = -0.7,
              xi0: float = 0.04, label: str = "", seed: int = 505
              ) -> Interpretation:
    """Estimate Ĥ on a real RV series, then de-bias it against a matched envelope.

    `source` is a processed RV CSV or kline CSVs. For klines the proxy window is
    taken from the built series; for a CSV pass `window` (returns per RV bar).
    """
    # ── observed Ĥ + the measurement window ───────────────────────────────
    if isinstance(source, str) and _looks_like_rv_csv(source):
        log_rv, _, _ = load_log_rv_csv(source)
        if window is None:
            window = INTERVAL_MS[rv_bar] // INTERVAL_MS[sampling]
    else:
        series = build_rv_series(source, sampling=sampling, rv_bar=rv_bar,
                                 base_interval=base_interval)
        log_rv = series.log_rv
        window = series.expected_returns_per_bar
    n_obs = int(log_rv.size)
    observed = {r.name: r.H for r in analyze(log_rv, multifractal=False)}

    # ── matched bias envelope + inversion ─────────────────────────────────
    curve = build_bias_curve(true_grid, n_obs=n_obs, window=window, n_mc=n_mc,
                             eta=eta, rho=rho, xi0=xi0, seed=seed)
    implied, implied_sd, collapsed, reason, candidates = {}, {}, {}, {}, {}
    for name in ESTIMATORS:
        o = observed.get(name, np.nan)
        status, sols = _classify_inversion(o, curve.true_grid, curve.mean[name])
        reason[name] = status
        candidates[name] = sols
        h = sols[0] if status == "ok" else np.nan
        implied[name] = h
        slope = _local_slope(curve.true_grid, curve.mean[name], h)
        sd_obs = float(np.interp(h, curve.true_grid, curve.std[name])) \
            if np.isfinite(h) else np.nan
        implied_sd[name] = (sd_obs / abs(slope)) if (np.isfinite(slope)
                            and abs(slope) > 1e-9 and np.isfinite(sd_obs)) else np.nan
        collapsed[name] = bool(np.isfinite(slope) and abs(slope) < _FLAT_SLOPE)

    return Interpretation(observed, implied, implied_sd, collapsed, reason,
                          candidates, curve, n_obs=n_obs, label=label)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="De-bias an observed Hurst estimate against a matched "
                    "Rung-1 bias envelope (recover implied true H).")
    p.add_argument("source", nargs="+",
                   help="a processed RV CSV, or kline CSV file(s)/dir/glob")
    p.add_argument("--window", type=int, default=None,
                   help="returns per RV bar (CSV input; default 1d/5m = 288)")
    p.add_argument("--sampling", default="5m")
    p.add_argument("--rv-bar", default="1d")
    p.add_argument("--base-interval", default="1m")
    p.add_argument("--n-mc", type=int, default=40,
                   help="Monte-Carlo paths per true-H grid point (40+ on real "
                        "data; heavy)")
    p.add_argument("--eta", type=float, default=1.0, help="vol-of-vol (model)")
    p.add_argument("--rho", type=float, default=-0.7, help="leverage (model)")
    p.add_argument("--xi0", type=float, default=0.04, help="base variance (model)")
    p.add_argument("--symbol", default="")
    args = p.parse_args(argv)

    single = args.source[0] if len(args.source) == 1 else args.source
    print(f"\n  building bias envelope (n_mc={args.n_mc}) — this is the heavy "
          f"step, please wait...")
    interp = interpret(single, window=args.window, sampling=args.sampling,
                       rv_bar=args.rv_bar, base_interval=args.base_interval,
                       n_mc=args.n_mc, eta=args.eta, rho=args.rho, xi0=args.xi0,
                       label=args.symbol)
    interp.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
