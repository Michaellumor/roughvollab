"""
test_rough_heston_lifted.py — mechanics tests for brick 4b (lifted OU-factor
rough-Heston simulator). Guard the MACHINERY (reconstruction, exact-OU stiff
stability, the H=1/2 -> classical-Heston reduction, positivity). The measured
GATE C (H=1/2 prices) and GATE B (beta=2H) results live in ROADMAP D33.
"""
import numpy as np

from rough_heston_lifted import (lifted_setup, _lifted_from_increments,
                                 rough_heston_lifted_paths)
from rough_heston import PARAMS


def test_nu0_relaxes_to_theta():
    """nu=0: no noise -> V is deterministic; V0=theta stays at theta exactly."""
    t, S, V = rough_heston_lifted_paths(128, 0.10, 500, N=100,
                                        rng=np.random.default_rng(0), nu=0.0, V0=0.04)
    assert np.allclose(V, 0.04, atol=1e-12) and not np.isnan(V).any()
    # V0 != theta relaxes toward theta and stays deterministic
    t, S, V2 = rough_heston_lifted_paths(128, 0.10, 500, N=100,
                                         rng=np.random.default_rng(0), nu=0.0, V0=0.08)
    assert V2[:, -1].std() < 1e-10 and 0.04 < V2[:, -1].mean() < 0.08


def test_half_reduces_to_classical_heston():
    """GATE C (under the qe default): H=1/2 (single gamma=0 mode) -> classical
    Heston; lifted price matches the brick-2 CF reference to MC error."""
    from layer4_convergence import cf_reference
    p = dict(PARAMS)
    P_cf = cf_reference(0.5, p, 100.0, 2000)
    M = 60000
    t, S, V = rough_heston_lifted_paths(128, 0.5, M, N=4, rng=np.random.default_rng(7))
    pay = np.maximum(S[:, -1] - 100.0, 0.0)
    se = pay.std(ddof=1) / np.sqrt(M)
    assert abs(pay.mean() - P_cf) < 4 * se, f"lifted {pay.mean():.4f} vs CF {P_cf:.4f}"


def test_beta_2H_under_qe():
    """GATE B (smoke, under qe default): the lifted MLMC level-variance decay beta
    tracks 2H (H=0.20, 2H=0.40), reproducing brick-1's rough-Heston QE beta."""
    from rough_heston_lifted import measure_beta_lifted
    p = {**PARAMS, "nu": 0.20}
    r = measure_beta_lifted(0.20, p, 8, (1, 2, 3), 6000, 60, np.random.default_rng(7))
    assert 0.30 < r["beta"] < 0.50 and r["monotone"], f"beta={r['beta']:.3f}"


def test_stiff_factor_stable():
    """A huge gamma factor must NOT blow up under exact-OU (e^(-x dt) decay)."""
    p = dict(PARAMS)
    g = np.array([0.0, 1e14]); w = np.array([0.5, 0.5])
    dWV = np.random.default_rng(1).standard_normal((400, 128)) * np.sqrt(1 / 128)
    dWp = np.random.default_rng(2).standard_normal((400, 128)) * np.sqrt(1 / 128)
    S, V = _lifted_from_increments(dWV, dWp, 128, 0.10, p, g, w)
    assert np.isfinite(V).all() and np.isfinite(S).all() and np.abs(V).max() < 10.0


def test_bb_setup_has_zero_node():
    g, w = lifted_setup(0.10, 64, method="bb")
    assert np.isclose(g.min(), 0.0) and g.size > 1


def test_paths_finite_and_shapes():
    t, S, V = rough_heston_lifted_paths(64, 0.10, 200, N=80, rng=np.random.default_rng(3))
    assert S.shape == (200, 65) and V.shape == (200, 65)
    assert np.isfinite(S).all() and np.isfinite(V).all() and (S[:, 0] == 100.0).all()


def test_reconstruction_mean_near_theta():
    """E[V_T] ~ theta at the validated regime (the lift reconstructs the level)."""
    t, S, V = rough_heston_lifted_paths(128, 0.10, 20000, N=100,
                                        rng=np.random.default_rng(0))
    assert abs(V[:, -1].mean() - 0.04) < 0.005


# ---- Gate D (high-nu, brick 4c) smoke tests ----
def test_cf_price_guarded_highnu():
    """The brick-2 CF is usable as a high-nu reference: finite at the operating point
    (N_riccati=2000), and the guard returns NaN (not an exception) when the fractional
    Riccati overflows at too-low N_riccati (small H + high nu)."""
    from rough_heston_lifted import _cf_price_guarded
    p = dict(PARAMS, nu=0.40)
    ok = _cf_price_guarded(0.10, p, 100.0, 2000)
    assert np.isfinite(ok) and ok > 0
    guarded = _cf_price_guarded(0.05, p, 100.0, 300)     # overflow regime -> must not raise
    assert np.isnan(guarded) or np.isfinite(guarded)


def test_gate_d_highnu_qe_beats_trunc_on_price():
    """High-nu smoke (nu=0.40, H=0.10): lifted qe price is CLOSER to the CF known-answer
    than trunc (the lift's positivity win) and within a few % of CF."""
    from rough_heston_lifted import _cf_price_guarded
    p = dict(PARAMS, nu=0.40); K = 100.0; M = 20000
    P_cf = _cf_price_guarded(0.10, p, K, 2000)
    _, Sq, _ = rough_heston_lifted_paths(64, 0.10, M, 150, rng=np.random.default_rng(7),
                                         positivity="qe", nu=0.40)
    _, St, _ = rough_heston_lifted_paths(64, 0.10, M, 150, rng=np.random.default_rng(7),
                                         positivity="trunc", nu=0.40)
    eq = abs(np.maximum(Sq[:, -1] - K, 0).mean() - P_cf) / P_cf
    et = abs(np.maximum(St[:, -1] - K, 0).mean() - P_cf) / P_cf
    assert eq < et and eq < 0.06, f"qe err {eq:.3f} not < trunc {et:.3f} (& <6%)"


def test_lifted_non_anticipating():
    """Lifted V_i must be F_{t_i}-measurable: perturbing dWV[:, k] must leave
    V[:, :k+1] BIT-IDENTICAL and change only V[:, k+1:]. Left-point Ito /
    non-anticipation; beta=2H cannot catch a look-ahead here, so it is tested
    directly (mirrors the explicit test_non_anticipating in test_rough_heston.py)."""
    p = dict(PARAMS)
    gammas, weights = lifted_setup(0.10, 8)          # SOE nodes/weights; p is PARAMS
    n, k = 64, 30
    dt = p["T"] / n
    r = np.random.default_rng(99)
    dWV = r.standard_normal((200, n)) * np.sqrt(dt)
    dWp = r.standard_normal((200, n)) * np.sqrt(dt)
    _, V0 = _lifted_from_increments(dWV, dWp, n, 0.10, p, gammas, weights)
    dWV2 = dWV.copy(); dWV2[:, k] += 0.5             # perturb increment over [t_k, t_{k+1}]
    _, V1 = _lifted_from_increments(dWV2, dWp, n, 0.10, p, gammas, weights)
    assert np.array_equal(V0[:, :k + 1], V1[:, :k + 1]), \
        "LOOK-AHEAD: V_{<=k} changed when a FUTURE increment dWV[:,k] was perturbed"
    assert not np.array_equal(V0[:, k + 1:], V1[:, k + 1:]), \
        "V_{>k} must respond to dWV[:,k] (sanity that the perturbation took effect)"
