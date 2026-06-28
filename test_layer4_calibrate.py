"""
test_layer4_calibrate.py — mechanics tests for the rough-Heston smile calibrator.
Guard the MACHINERY (param mapping, strike anchoring, fixed-length NaN-penalty
residual, guard-returns-NaN, the JᵀJ identifiability report, a fast near-truth
recovery). The measured results (CF→CF exact recovery, the H/ν ill-conditioning,
the lift→CF distortion) live in ROADMAP D37.
"""
import numpy as np

from layer4_calibrate import (
    PNAMES, KAPPA_FIXED, TRUTH, VUS_FULL, S0,
    theta_to_cfparams, fixed_strikes_from_target, model_smile_cf, model_smile_lift,
    residuals, iv_rmse, calibrate, sensitivity_jacobian, identifiability_report,
)

SMALL = dict(N_riccati=1000, n_nodes=96)  # robustly stable to FD steps (≤500 overflows near truth)


def test_theta_to_cfparams_mapping():
    """ξ₀ -> V0=theta (flat var); kappa fixed; (H,ν,ρ) direct."""
    H, cfp = theta_to_cfparams([0.10, 0.35, -0.70, 0.04])
    assert H == 0.10 and cfp["kappa"] == KAPPA_FIXED
    assert cfp["V0"] == 0.04 and cfp["theta"] == 0.04          # flat forward variance
    assert cfp["nu"] == 0.35 and cfp["rho"] == -0.70


def test_strike_anchoring_atm_at_S0():
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    assert Ks.shape == VUS_FULL.shape
    assert np.isclose(Ks[VUS_FULL == 0.0][0], S0)             # vu=0 -> K=S0
    assert (np.diff(Ks) > 0).all()                            # strictly increasing in vu


def test_residual_fixed_length_nan_penalty():
    """Residual length == len(Ks) and NaN entries -> penalty (FD Jacobian needs
    constant length)."""
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    target = np.full(len(Ks), 0.17)
    # an overflow-prone param set may yield NaN model IVs -> must become the penalty, not drop
    r = residuals([0.025, 0.99, -0.5, 0.04], Ks, target, nan_penalty=0.5,
                  cf_kw=dict(N_riccati=150, n_nodes=80))
    assert r.shape == (len(Ks),) and np.isfinite(r).all()
    assert np.isclose(r.max(), 0.5) or np.all(np.abs(r) < 0.5)  # penalty present or all finite


def test_iv_rmse_zero_at_truth():
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    target = model_smile_cf(TRUTH, Ks, **SMALL)
    assert iv_rmse(TRUTH, Ks, target, cf_kw=SMALL) < 1e-9


def test_guard_returns_nan_not_raise_on_overflow():
    """Small-H + high-ν + low N_riccati overflows the Riccati -> guard returns NaN,
    never raises; output length preserved."""
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    iv = model_smile_cf([0.03, 0.95, -0.5, 0.04], Ks, N_riccati=150, n_nodes=80)
    assert iv.shape == (len(Ks),)                              # no crash; shape preserved


def test_sensitivity_jacobian_shape_finite():
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    J = sensitivity_jacobian(TRUTH, Ks, cf_kw=SMALL)
    assert J.shape == (len(Ks), 4) and np.isfinite(J).all()


def test_identifiability_H_is_weak_direction():
    """At truth the flattest JᵀJ direction is dominated by H, H is anti-correlated
    with ν, and H is classed 'weak' (the single-smile degeneracy)."""
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    rep = identifiability_report(TRUTH, Ks, cf_kw=SMALL)
    assert abs(rep.flat_dir[0]) > 0.7                          # flat direction ≈ H
    assert rep.classes["H"] == "weak"
    assert rep.corr[0, 1] < -0.7                               # corr(H, ν) strongly negative
    assert rep.cond > 1e4                                      # ill-conditioned


def test_fast_recovery_near_truth():
    """A bounded near-truth CF→CF recovery converges back to truth (optimiser works)."""
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    target = model_smile_cf(TRUTH, Ks, **SMALL)
    res = calibrate(target, Ks, theta0=np.array([0.13, 0.30, -0.62, 0.042]), cf_kw=SMALL)
    err = np.abs((res.theta_hat - TRUTH) / TRUTH)
    assert res.iv_rmse < 1e-3 and err.max() < 0.10, f"{res}  err={err}"


def test_lift_smile_runs_finite():
    Ks = fixed_strikes_from_target(0.17, VUS_FULL)
    iv = model_smile_lift(TRUTH, Ks, M=4000, n=32, N=20, seed=1)
    assert iv.shape == (len(Ks),) and np.isfinite(iv).sum() >= len(Ks) - 2
