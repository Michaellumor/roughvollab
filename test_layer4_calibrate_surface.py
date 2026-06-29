"""
test_layer4_calibrate_surface.py — mechanics tests for the multi-maturity surface
calibrator. Guard the MACHINERY (n_params κ switch, √T strike scaling, stacked
fixed-length surface, the calibrate adapter, the shared identifiability path, and
that the surface out-conditions a single smile). The measured result (cond ÷~100×,
the (B) partial H-identification, the κ verdict) lives in ROADMAP D38.
"""
import numpy as np

import layer4_calibrate_surface as S
from layer4_calibrate import KAPPA_FIXED

SMALL = dict(N_riccati=900, n_nodes=96)          # fast; [0.25,1.0] is stable here
TS2 = [0.25, 1.0]
VUS5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def test_theta_to_cfparams_s_4_and_5():
    H, cfp4 = S.theta_to_cfparams_s([0.10, 0.35, -0.70, 0.04], 4)
    assert cfp4["kappa"] == KAPPA_FIXED and cfp4["V0"] == 0.04 and cfp4["theta"] == 0.04
    H, cfp5 = S.theta_to_cfparams_s([0.10, 0.35, -0.70, 0.04, 1.7], 5)
    assert cfp5["kappa"] == 1.7                  # 5-param frees kappa from theta[4]


def test_fixed_surface_strikes_sqrtT():
    a = {0.25: 0.17, 1.0: 0.17}
    grids = S.fixed_surface_strikes(a, VUS5, TS2)
    for T in TS2:
        assert np.isclose(grids[T][VUS5 == 0.0][0], S.S0)            # ATM -> S0
    # same vol-units => wider absolute log-strike range at longer T (∝ √T)
    span = {T: np.log(grids[T][-1] / grids[T][0]) for T in TS2}
    assert span[1.0] > span[0.25]


def test_surface_model_length_and_order():
    a = S.atm_by_T(S.TRUTH[4], TS2, **SMALL)
    grids = S.fixed_surface_strikes(a, VUS5, TS2)
    v = S.surface_model(S.TRUTH[4], grids, cf_kw=SMALL)
    assert v.shape == (len(TS2) * len(VUS5),)                        # stacked length
    # order = sorted(Ts) then K: first block is T=0.25
    first = S.model_smile_cf_T(S.TRUTH[4], grids[0.25], 0.25, **SMALL)
    assert np.allclose(v[:len(VUS5)], first, equal_nan=True)


def test_surface_iv_rmse_zero_at_truth():
    a = S.atm_by_T(S.TRUTH[4], TS2, **SMALL)
    grids = S.fixed_surface_strikes(a, VUS5, TS2)
    tgt = S.surface_model(S.TRUTH[4], grids, cf_kw=SMALL)
    model = S.make_surface_model(grids, 4)
    assert S.iv_rmse(S.TRUTH[4], None, tgt, model=model, cf_kw=SMALL) < 1e-9


def test_surface_jacobian_shape_finite():
    a = S.atm_by_T(S.TRUTH[4], TS2, **SMALL)
    grids = S.fixed_surface_strikes(a, VUS5, TS2)
    J = S.surface_jacobian(S.TRUTH[4], grids, cf_kw=SMALL)
    assert J.shape == (len(TS2) * len(VUS5), 4) and np.isfinite(J).all()


def test_surface_out_conditions_single_smile():
    """The core machinery claim: a 2-maturity surface is better-conditioned than one smile."""
    rs = S.ident_for([1.0], cf_kw=SMALL)
    rsurf = S.ident_for(TS2, cf_kw=SMALL)
    assert rsurf.cond < rs.cond                                      # surface improves conditioning
    assert abs(rsurf.flat[0]) > 0.5                                  # H still the weak direction (B, not A)


def test_calibrate_surface_adapter_recovers_near_truth():
    """The Ks=None adapter drives D37's calibrate over the stacked surface."""
    a = S.atm_by_T(S.TRUTH[4], TS2, **SMALL)
    grids = S.fixed_surface_strikes(a, VUS5, TS2)
    tgt = S.surface_model(S.TRUTH[4], grids, cf_kw=SMALL)
    res = S.calibrate_surface(tgt, grids, theta0=np.array([0.13, 0.30, -0.62, 0.042]), cf_kw=SMALL)
    assert res.theta_hat.shape == (4,) and res.iv_rmse < 1e-3
    # ξ₀ and ρ are the well-identified params -> tight even on a small surface
    assert abs(res.theta_hat[3] - 0.04) / 0.04 < 0.05 and abs(res.theta_hat[2] + 0.70) / 0.70 < 0.05
