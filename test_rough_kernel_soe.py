"""
test_rough_kernel_soe.py — mechanics tests for brick 4a (sum-of-exponentials
kernel approximation). These guard the MACHINERY (spectral-measure closed forms,
the two constructions, the error metric, the H=1/2 guard); the measured error-vs-N
result + the AJ-EE-vs-BB verdict live in ROADMAP D32 / Gate A.
"""
import numpy as np
from scipy.integrate import quad

from rough_kernel_soe import (kernel, soe, soe_ajee, soe_bb, kernel_l2_metrics,
                              _mu_norm)


def test_kernel_value():
    from scipy.special import gamma
    t = np.array([0.01, 0.1, 0.5, 1.0])
    for H in (0.05, 0.1, 0.2):
        assert np.allclose(kernel(t, H), t ** (H - 0.5) / gamma(H + 0.5))


def test_ajee_moment_match_vs_numeric():
    """AJ-EE closed-form c_i, gamma_i must equal the numerical interval integrals
    of the spectral measure mu (eq 3.6)."""
    H, N = 0.1, 12
    a = H + 0.5
    pi_n = (N ** (-0.2)) * (np.sqrt(10.0) * (1 - 2 * H) / (5 - 2 * H)) ** 0.4
    eta = np.arange(N + 1) * pi_n
    mn = _mu_norm(H)
    mu = lambda x: mn * x ** (-(H + 0.5))
    g, c = soe_ajee(H, N)
    for i in range(1, N + 1):
        c_num = quad(mu, eta[i - 1], eta[i])[0]
        g_num = quad(lambda x: x * mu(x), eta[i - 1], eta[i])[0] / c_num
        assert abs(c[i - 1] - c_num) < 1e-9 * c_num
        assert abs(g[i - 1] - g_num) < 1e-9 * g_num


def test_ajee_realizes_algebraic_rate():
    """AJ-EE must realise its pinned algebraic rate -4H/5 (validates the build)."""
    H = 0.1
    Ns = [16, 32, 64, 128, 256]
    errs = [kernel_l2_metrics(H, *soe_ajee(H, N))["rel_l2"] for N in Ns]
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert abs(slope - (-0.8 * H)) < 0.02, f"AJ-EE slope {slope:.3f} vs -4H/5={-0.8*H}"


def test_bb_superpolynomial_and_beats_ajee():
    """BB converges fast (reaches the 1e-3 target by N~520 at H=0.1) and crushes
    AJ-EE at matched factor count."""
    H = 0.1
    g, w = soe_bb(H, 520)
    assert kernel_l2_metrics(H, g, w)["rel_l2"] < 1e-3
    # matched-N domination
    ga, wa = soe_ajee(H, 256); gb, wb = soe_bb(H, 256)
    ea = kernel_l2_metrics(H, ga, wa)["rel_l2"]
    eb = kernel_l2_metrics(H, gb, wb)["rel_l2"]
    assert eb < 0.05 * ea, f"BB {eb:.2e} not << AJ-EE {ea:.2e}"


def test_bb_has_zero_tail_node():
    g, w = soe_bb(0.1, 64)
    assert np.isclose(g.min(), 0.0) and w[np.argmin(g)] > 0


def test_half_guard_no_nan_constant_kernel():
    """H=1/2: K(t)=1 (flat); single gamma=0 mode reproduces it exactly, no NaN."""
    g, w = soe(0.5, 32)
    assert g.size == 1 and np.isclose(g[0], 0.0) and np.isclose(w[0], 1.0)
    m = kernel_l2_metrics(0.5, g, w, dt=1 / 128)
    assert m["rel_l2"] < 1e-12 and not np.isnan(m["rel_l2"])


def test_metric_is_zero_for_exact_reconstruction():
    """Sanity: a single mode that IS the (degenerate) kernel gives zero error."""
    g, w = soe(0.5, 8)
    assert kernel_l2_metrics(0.5, g, w)["rel_l2"] < 1e-12
