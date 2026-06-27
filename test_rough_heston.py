"""
test_rough_heston.py — pre-β sanity gates for the rough-Heston simulator.

These MUST all pass before the β = 2H validation harness is meaningful
(gate-check spec §5, build order step 2). They check the simulator is
well-formed, the variance stays valid, the forward variance and the H=½
classical-Heston limit hold (validating the Γ(H+½) renormalisation), and the
asset is a martingale. Tolerances are loose enough to tolerate the documented
full-truncation positivity bias but tight enough to catch a renormalisation /
structural bug.
"""
import numpy as np
import pytest

from rough_heston import rough_heston_paths, PARAMS


def _rng(seed=12345):
    return np.random.default_rng(seed)


def cir_variance(t, kappa, theta, nu, V0):
    """Exact variance of the classical CIR process V_t (the H=½ limit)."""
    e1 = np.exp(-kappa * t)
    e2 = np.exp(-2.0 * kappa * t)
    return V0 * (nu**2 / kappa) * (e1 - e2) \
        + theta * (nu**2 / (2.0 * kappa)) * (1.0 - e1)**2


def test_shapes():
    t, S, V = rough_heston_paths(n=64, H=0.10, n_paths=32, rng=_rng())
    assert t.shape == (65,)
    assert S.shape == (32, 65) and V.shape == (32, 65)
    assert np.allclose(S[:, 0], PARAMS["S0"])
    assert np.allclose(V[:, 0], PARAMS["V0"])


def test_variance_nonneg():
    _, _, V = rough_heston_paths(n=128, H=0.10, n_paths=256, rng=_rng())
    assert np.isfinite(V).all(), "variance has NaN/inf"
    assert (V >= 0.0).all(), "full-truncation V⁺ must keep V >= 0 everywhere"


def test_reproducible():
    a = rough_heston_paths(n=96, H=0.10, n_paths=40, rng=_rng(7))
    b = rough_heston_paths(n=96, H=0.10, n_paths=40, rng=_rng(7))
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_forward_variance_renorm():
    """Validate the Γ(H+½) renormalisation via E[V_t] ≈ θ in a MILD regime
    (small ν ⇒ truncation is rare, so E[V] reflects the kernel norm, not the
    positivity bias). A wrong kernel normalisation would shift E[V] by ~a Γ
    factor — caught easily here. (E[V]≈θ at the ROUGH default params is NOT
    asserted: full truncation deliberately biases it — see the next test.)"""
    theta = PARAMS["theta"]
    _, _, V = rough_heston_paths(n=128, H=0.10, n_paths=8000, rng=_rng(), nu=0.15)
    EV = V.mean(axis=0)
    assert abs(EV[1:].mean() / theta - 1.0) < 0.025, \
        f"renorm (time-avg) off: E[V]/θ={EV[1:].mean()/theta:.4f}"
    assert abs(EV[-1] / theta - 1.0) < 0.03, \
        f"renorm (terminal) off: E[V_T]/θ={EV[-1]/theta:.4f}"


def test_truncation_bias_documents_the_caveat():
    """Documents (does NOT defeat) the full-truncation positivity bias in the
    HIGH-ν regime (H=0.10, ν=0.40 — beyond the validated ν≤0.20 ceiling): E[V_t]
    is biased UP ~10–15% because the rough short-time variance blow-up sends V<0
    often and truncation clips it. This is why truncation is NOT the default and
    why ν>0.20 is outside the explicit scheme's validated range. Asserts only the
    sign and a sane bound, not smallness."""
    theta = PARAMS["theta"]
    _, _, V = rough_heston_paths(n=128, H=0.10, n_paths=8000, rng=_rng(),
                                 positivity="truncation", nu=0.40)
    bias = V.mean(axis=0)[1:].mean() / theta - 1.0
    assert 0.04 < bias < 0.30, \
        f"expected the documented +10–15% full-truncation E[V] bias; got {bias:.2%}"


def test_asset_martingale():
    """E[S_T] ≈ S0 at r=0 (the log-Euler asset is a martingale)."""
    S0 = PARAMS["S0"]
    _, S, _ = rough_heston_paths(n=64, H=0.10, n_paths=10000, rng=_rng())
    ST = S[:, -1]
    se = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - S0) < 4.0 * se, (
        f"E[S_T]={ST.mean():.3f} vs S0={S0} (4·se={4*se:.3f})")


def test_non_anticipating():
    """V_i must be F_{t_i}-measurable — depend ONLY on increments over [0, t_i]
    (dWV[:, :i]), never on a future increment. Perturbing dWV[:, k] must leave
    V[:, :k+1] BIT-IDENTICAL and change only V[:, k+1:]. This is the left-point
    Itô convention; β=2H cannot catch a look-ahead here, so it's tested directly."""
    from rough_heston import _rough_heston_from_increments
    n, k = 64, 30
    dt = PARAMS["T"] / n
    r = _rng(99)
    dWV  = r.standard_normal((200, n)) * np.sqrt(dt)
    dWp  = r.standard_normal((200, n)) * np.sqrt(dt)
    _, V0 = _rough_heston_from_increments(dWV, dWp, n, 0.10, PARAMS)
    dWV2 = dWV.copy(); dWV2[:, k] += 0.5            # perturb increment over [t_k, t_{k+1}]
    _, V1 = _rough_heston_from_increments(dWV2, dWp, n, 0.10, PARAMS)
    assert np.array_equal(V0[:, :k + 1], V1[:, :k + 1]), \
        "LOOK-AHEAD: V_{<=k} changed when a FUTURE increment dWV[:,k] was perturbed"
    assert not np.array_equal(V0[:, k + 1:], V1[:, k + 1:]), \
        "V_{>k} must respond to dWV[:,k] (sanity that the perturbation took effect)"


def test_heston_limit():
    """H=½ ⇒ flat kernel ⇒ classical Heston. Check the variance moments
    against the exact CIR formula (mild ν so truncation is rare and the
    classical limit is clean — isolates the renorm/structure check)."""
    kappa, theta, V0, T = 0.3, 0.04, 0.04, 1.0
    nu = 0.15                                 # near-Feller (2κθ=0.024 ≈ ν²=0.0225)
    _, _, V = rough_heston_paths(n=150, H=0.50, n_paths=6000,
                                 rng=_rng(2024), nu=nu)
    VT = V[:, -1]
    mean_err = abs(VT.mean() - theta) / theta
    var_meas = VT.var(ddof=1)
    var_true = cir_variance(T, kappa, theta, nu, V0)
    var_err = abs(var_meas - var_true) / var_true
    assert mean_err < 0.05, f"E[V_T]={VT.mean():.5f} vs θ={theta} ({mean_err:.2%})"
    assert var_err < 0.20, (
        f"Var(V_T)={var_meas:.6f} vs CIR {var_true:.6f} ({var_err:.2%})")
