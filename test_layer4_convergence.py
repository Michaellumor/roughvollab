"""
test_layer4_convergence.py — harness-MECHANICS tests for the weak-order study.

These guard the MACHINERY (CRN aggregation, the vectorised BS, Romano–Touzi
conditional-MC unbiasedness, the slope fit, the classifier) — NOT the heavy
statistical MC result. The measured weak order (α≫H banked; PARTIAL@H=0.05;
borderline@H=0.10; PASS@H=0.20) lives in ROADMAP D31, with its caveats; these
fast tests ensure the tools that produced it are correct.
"""
import numpy as np

from layer4_convergence import _agg, _bs_call_vec, _fit_slope, classify, mc_call_levels
from rough_heston import PARAMS as RH
from rough_heston_cf import bs_call


def test_agg_sums_fine_increments():
    """CRN coupling: a coarse increment must be the exact sum of its fine group."""
    dW = np.arange(8, dtype=float).reshape(1, 8)
    assert np.allclose(_agg(dW, 4), [[1, 5, 9, 13]])      # groups of 2
    assert np.allclose(_agg(dW, 2), [[6, 22]])            # groups of 4
    # variance scaling: summing g iid N(0,dt) → N(0, g·dt)
    dt = 1.0 / 64
    fine = np.random.default_rng(0).standard_normal((40000, 64)) * np.sqrt(dt)
    coarse = _agg(fine, 16)                               # group size 4 → var 4·dt
    assert abs(coarse.var() - 4 * dt) < 0.05 * 4 * dt


def test_bs_call_vec_matches_scalar():
    S = np.array([90.0, 100.0, 110.0]); sig = np.array([0.20, 0.25, 0.30])
    vec = _bs_call_vec(S, 100.0, 1.0, 0.0, sig)
    sc = np.array([bs_call(s, 100.0, 1.0, 0.0, v) for s, v in zip(S, sig)])
    assert np.allclose(vec, sc, atol=1e-10)


def test_conditional_mc_matches_bs_at_zero_volofvol():
    """ν→0 ⇒ variance pinned at θ ⇒ rough Heston is GBM(√θ); the Romano–Touzi
    conditional price must reproduce BS(√θ) in expectation (formula correctness)."""
    p = dict(RH); p["nu"] = 1e-6
    r = mc_call_levels(0.5, p, n0=32, L=0, M=60000, rng=np.random.default_rng(5),
                       K=100.0, conditional=True)
    bs = bs_call(p["S0"], 100.0, p["T"], p["r"], np.sqrt(p["theta"]))
    assert abs(r["cv_mean"][0] - bs) < 5 * max(r["cv_se"][0], 1e-4), \
        f"conditional MC {r['cv_mean'][0]:.5f} vs BS(√θ) {bs:.5f}"


def test_conditional_mc_unbiased_vs_plain():
    """Conditional MC only removes variance — its mean must match plain MC (same
    paths, same n) within a few s.e. (bias-preserving)."""
    p = dict(RH)
    rc = mc_call_levels(0.2, p, n0=16, L=1, M=40000, rng=np.random.default_rng(3),
                        K=100.0, conditional=True)
    rp = mc_call_levels(0.2, p, n0=16, L=1, M=40000, rng=np.random.default_rng(3),
                        K=100.0, conditional=False)
    diff = abs(rc["cv_mean"][-1] - rp["cv_mean"][-1])
    assert diff < 5 * np.hypot(rc["cv_se"][-1], rp["cv_se"][-1]), \
        f"conditional vs plain n=32 mean differ {diff:.4f}"


def test_fit_recovers_known_slope():
    """The fit must recover a known power-law exponent (guards the α machinery)."""
    dt = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    y = 0.7 * dt ** 0.8                                   # known α = 0.8
    a, a_se, n, mask = _fit_slope(dt, y, np.full_like(y, 1e-12))
    assert n == 5 and abs(a - 0.8) < 1e-6


def test_classify_cases():
    assert classify(1.00, 0.05, 0.10).startswith("PASS")
    assert classify(0.10, 0.02, 0.10).startswith("FAIL")
    assert classify(0.55, 0.05, 0.10).startswith("PARTIAL")


# ---- sim-callback generalization (any simulator can plug into the harness) ----
def test_sim_callback_default_matches_explicit():
    """sim=None must reproduce the explicit brick-1 core exactly (back-compat)."""
    from rough_heston import _rough_heston_from_increments
    p = dict(RH)
    r0 = mc_call_levels(0.10, p, 4, 2, 5000, np.random.default_rng(0), 110.0)
    sim = lambda dWV, dWp, n: _rough_heston_from_increments(dWV, dWp, n, 0.10, p, positivity="qe")
    r1 = mc_call_levels(0.10, p, 4, 2, 5000, np.random.default_rng(0), 110.0, sim=sim)
    assert np.allclose(r0["cv_mean"], r1["cv_mean"]) and np.allclose(r0["dmean"], r1["dmean"])


def test_lifted_sim_plugs_into_harness():
    """A lifted closure runs through the weak-order harness and yields finite biases
    (the reusable infrastructure — the value beyond D35's negative result)."""
    from rough_heston_lifted import _lifted_from_increments, lifted_setup
    p = dict(RH); H = 0.10
    g, w = lifted_setup(H, 40)
    sim = lambda dWV, dWp, n: _lifted_from_increments(dWV, dWp, n, H, p, g, w, "qe")
    r = mc_call_levels(H, p, 4, 2, 5000, np.random.default_rng(0), 110.0, conditional=True, sim=sim)
    assert np.all(np.isfinite(r["cv_mean"])) and np.all(np.isfinite(r["dmean"]))
