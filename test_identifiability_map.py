"""
Tests for identifiability_map.py
================================
The classifier is exercised on hand-built bias curves (instant, deterministic)
so the *definition* of identifiability is pinned independently of the simulator;
one tiny end-to-end run checks the sweep/map plumbing.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from identifiability_map import (
    IDENTIFIED, DEBIASABLE, NON_IDENTIFIED, UNCALIBRATED,
    cell_status, build_identifiability_map, plot_identifiability_map,
    locate_observed, calibrate_eta, model_daily_logrv_sd, place_asset,
    AssetPlacement,
)
from interpret_h import build_bias_curve
from roughvol_core import rough_bergomi_paths
from layer1c_roughness_audit import realized_log_variance

# A clean, steep, monotone curve: observed Ĥ ≈ true H (slope ≈ 1), tight spread.
_G = np.array([0.05, 0.10, 0.20, 0.30, 0.45, 0.60])
_CLEAN_MEAN = _G.copy()
_TIGHT_STD = np.full_like(_G, 0.01)


# ── step 1: identifiability classifier ──────────────────────────────────────

@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_clean_steep_curve_identifies_rough_H(i):
    # Rough true H on a steep, tight curve → CI excludes 0.5 → IDENTIFIED.
    assert cell_status(_G, _CLEAN_MEAN, _TIGHT_STD, i) == IDENTIFIED


def test_smooth_H_is_not_identified_as_rough():
    # True H = 0.60 (smooth) cannot be called "rough" → not IDENTIFIED.
    assert cell_status(_G, _CLEAN_MEAN, _TIGHT_STD, 5) != IDENTIFIED


def test_nonmonotone_curve_is_non_identified_everywhere():
    # Hump-shaped curve (noisy-proxy regime): rough and smooth give the same Ĥ.
    hump = np.array([0.20, 0.30, 0.38, 0.30, 0.20, 0.12])
    for i in range(_G.size):
        assert cell_status(_G, hump, _TIGHT_STD, i) == NON_IDENTIFIED


def test_flat_curve_is_non_identified():
    # Near-flat curve: slope below the floor ⇒ ill-posed collapse.
    flat = np.full_like(_G, 0.22) + np.linspace(0, 0.01, _G.size)
    assert cell_status(_G, flat, _TIGHT_STD, 2) == NON_IDENTIFIED


def test_wide_band_is_debiasable_not_identified():
    # Steep monotone curve but huge single-sample spread ⇒ band reaches 0.5:
    # point-estimable, but the smooth null cannot be excluded.
    wide = np.full_like(_G, 0.30)
    assert cell_status(_G, _CLEAN_MEAN, wide, 1) == DEBIASABLE


def test_nan_mean_is_uncalibrated():
    m = _CLEAN_MEAN.copy(); m[2] = np.nan
    assert cell_status(_G, m, _TIGHT_STD, 2) == UNCALIBRATED


# ── step 4 hook: locating an observed value ─────────────────────────────────

def test_locate_below_floor_flags_rougher_than_model():
    # Build a real (small) curve, then probe an observed Ĥ rougher than its floor.
    curve = build_bias_curve(_G, n_obs=120, window=24, n_mc=3, eta=1.0, seed=1)
    floor = np.nanmin(curve.mean["GJR"])
    status, _ = locate_observed(floor - 0.05, curve, "GJR")
    assert status in {"below-floor", NON_IDENTIFIED}


# ── steps 2–3: sweep + plot plumbing (tiny, real simulation) ────────────────

def test_build_map_shape_and_valid_statuses_quick():
    eta_grid = np.array([1.0])
    window_grid = np.array([24])
    true_grid = np.array([0.05, 0.10, 0.45])
    imap = build_identifiability_map(eta_grid, window_grid, true_grid,
                                     n_obs=100, n_mc=3, progress=False)
    valid = {IDENTIFIED, DEBIASABLE, NON_IDENTIFIED, "below-floor",
             "above-ceiling", UNCALIBRATED}
    for name in ("GJR", "Cont-Das", "MF-DFA"):
        assert len(imap.status[name]) == 1            # one η
        assert len(imap.status[name][0]) == 1         # one Δ
        assert len(imap.status[name][0][0]) == 3      # three true-H cells
        assert set(imap.status[name][0][0]) <= valid
    # plotting must not raise (headless)
    plot_identifiability_map(imap, out=None, show=False)


# ── step 4: η calibration + asset placement ─────────────────────────────────

def _make_asset_csv(path, *, H, eta, n_obs, window, seed=7):
    """Write a synthetic asset CSV (a model path standing in for real data)."""
    rng = np.random.default_rng(seed)
    _, S, _ = rough_bergomi_paths(n_obs * window, H, n_paths=1, eta=eta, rng=rng)
    log_rv = realized_log_variance(S, window)[0]
    with open(path, "w") as f:
        f.write("period_start_ms,log_rv\n")
        for k, v in enumerate(log_rv):
            f.write(f"{k},{v}\n")


def test_calibrate_eta_recovers_known_eta():
    # Fixed-seed model std is a smooth monotone function of η, so calibrating to
    # a series generated at η*=1.5 (same draws) recovers it tightly.
    rng = np.random.default_rng(3)
    _, S, _ = rough_bergomi_paths(150 * 24, 0.10, n_paths=8, eta=1.5, rng=rng)
    obs_sd = float(np.nanmean(np.nanstd(realized_log_variance(S, 24), axis=1)))
    eta_hat = calibrate_eta(obs_sd, H=0.10, window=24, n_obs=150, n_paths=8, seed=3)
    assert abs(eta_hat - 1.5) < 0.3


def test_model_sd_increases_with_eta():
    lo = model_daily_logrv_sd(0.5, H=0.10, window=24, n_obs=120, n_paths=6, seed=1)
    hi = model_daily_logrv_sd(2.5, H=0.10, window=24, n_obs=120, n_paths=6, seed=1)
    assert hi > lo


def test_place_asset_smoke(tmp_path):
    csv = tmp_path / "fake_asset.csv"
    _make_asset_csv(str(csv), H=0.10, eta=1.5, n_obs=120, window=24, seed=9)
    pl = place_asset("FAKE", str(csv), window=24, n_mc=3, cal_paths=6)
    assert isinstance(pl, AssetPlacement)
    assert pl.eta_hat > 0 and np.isfinite(pl.observed_sd)
    valid = {IDENTIFIED, DEBIASABLE, NON_IDENTIFIED, "below-floor",
             "above-ceiling", UNCALIBRATED}
    assert set(pl.status.values()) <= valid
    assert set(pl.observed_H) == {"GJR", "Cont-Das", "MF-DFA"}


def test_plot_with_placements_no_raise(tmp_path):
    csv = tmp_path / "fake_asset.csv"
    _make_asset_csv(str(csv), H=0.10, eta=1.5, n_obs=120, window=24, seed=11)
    pl = place_asset("FAKE", str(csv), window=24, n_mc=3, cal_paths=6)
    imap = build_identifiability_map(np.array([1.5]), np.array([24]),
                                     np.array([0.05, 0.10, 0.45, 0.60]),
                                     n_obs=100, n_mc=3, progress=False)
    plot_identifiability_map(imap, out=None, show=False, placements=[pl])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
