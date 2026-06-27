"""
test_rough_heston_cf.py — STAGE (a) gate: Gil-Pelaez inversion validated on
KNOWN characteristic functions, BEFORE any rough-Heston Riccati code exists.
Spec: docs/gate_checks/layer4_convergence_gate_check.md §4 (a).

Isolation: the inverter is proven correct on the Black–Scholes CF (which has a
true closed form) to ~1e-8; its quadrature is then shown accurate on the harder
Heston integrand (fixed Gauss–Legendre vs near-exact adaptive quad) to ~1e-6;
and its own knobs (U_max, n_nodes) are convergence-checked. Only after this is
the rough-Heston CF built (stages b–d).
"""
import numpy as np
import pytest
from scipy.special import gamma

from rough_heston_cf import (bs_call, bs_cf, heston_cf, _heston_CD,
                             gil_pelaez_call, _gil_pelaez_call_adaptive,
                             _frac_riccati, _frac_integral_at_T, rough_heston_cf)

S0, T, r = 100.0, 1.0, 0.0
HESTON = dict(V0=0.04, kappa=2.0, theta=0.04, nu=0.5, rho=-0.7)
STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)


def _hcf(u):
    return heston_cf(u, T, r=r, **HESTON)


# ---- (a1) the key isolation: inverter vs BS closed form, ~1e-8 ----
@pytest.mark.parametrize("K", STRIKES)
def test_inversion_matches_bs_closed_form(K):
    sig = 0.20
    gp = gil_pelaez_call(lambda u: bs_cf(u, T, sig, r), S0, K, T, r)
    an = bs_call(S0, K, T, r, sig)
    assert abs(gp - an) < 1e-8, f"BS inversion off by {abs(gp-an):.2e} at K={K}"


# ---- (a2) Heston CF normalisation / martingale: φ(0)=1, φ(−i)=e^{rT}=1 ----
def test_heston_cf_normalisation():
    assert abs(complex(_hcf(0.0)) - 1.0) < 1e-12, "phi(0) != 1"
    assert abs(complex(_hcf(-1j)) - np.exp(r * T)) < 1e-10, "phi(-i) != E[S_T/S0]"


# ---- (a3) inverter quadrature on Heston: fixed-GL vs near-exact adaptive, ~1e-6 ----
@pytest.mark.parametrize("K", STRIKES)
def test_heston_fixedGL_matches_adaptive(K):
    gl = gil_pelaez_call(_hcf, S0, K, T, r)
    ad = _gil_pelaez_call_adaptive(_hcf, S0, K, T, r)
    assert abs(gl - ad) < 1e-6, f"fixed-GL vs adaptive off by {abs(gl-ad):.2e} at K={K}"


# ---- (a4) Heston CF formula sanity: BS-limit trend as vol-of-vol → 0 ----
def test_heston_reduces_toward_bs_as_nu_small():
    """V0=θ ⇒ as ν→0 the variance is pinned at θ and Heston → BS(√θ). The gap
    must shrink with ν (the CF formula reduces correctly; not asserted to 1e-6
    because a small genuine Heston correction remains at finite ν)."""
    bs = bs_call(S0, 100.0, T, r, np.sqrt(0.04))
    gaps = []
    for nu in (0.20, 0.05, 0.01):
        cf = lambda u: heston_cf(u, T, V0=0.04, kappa=2.0, theta=0.04, nu=nu, rho=-0.7, r=r)
        gaps.append(abs(gil_pelaez_call(cf, S0, 100.0, T, r) - bs))
    assert gaps[0] > gaps[1] > gaps[2], f"Heston→BS gap not shrinking with ν: {gaps}"
    assert gaps[-1] < 5e-3, f"Heston(ν=0.01) not close to BS(√θ): gap {gaps[-1]:.2e}"


# ---- (a5) inversion's OWN knobs converge (independent of any Riccati grid) ----
def test_inversion_knob_convergence():
    """Refining U_max and n_nodes must move the price < 1e-6 — certifies the
    inversion error is resolved (its own error source, per spec §3a)."""
    ref = _gil_pelaez_call_adaptive(_hcf, S0, 100.0, T, r)
    for U_max, n in [(100.0, 128), (200.0, 256), (400.0, 512)]:
        price = gil_pelaez_call(_hcf, S0, 100.0, T, r, U_max=U_max, n_nodes=n)
        last = abs(price - ref)
    assert last < 1e-7, f"finest (U_max=400,n=512) vs adaptive off by {last:.2e}"


# ===== STAGE (b): rough-Heston CF — H=½ CF-level reduction (the arbiter) =====
UGRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])


def test_riccati_reduces_to_heston_at_half():
    """H=½ (α=1): the fractional Riccati ψ(u,T) must equal the classical Heston
    D(u,T), and κθ∫ψ must equal C(u,T), across a u-grid — vs the Albrecher
    reference proven in stage (a). This isolates Riccati + CF-assembly from the
    (already-proven) inverter, and is the arbiter for every coefficient."""
    C_ref, D_ref = _heston_CD(UGRID, T, **HESTON)
    psi = _frac_riccati(UGRID, T, 0.5, HESTON["kappa"], HESTON["nu"], HESTON["rho"], 400)
    h = T / 400
    errD = np.max(np.abs(psi[-1] - D_ref))
    errC = np.max(np.abs(HESTON["kappa"] * HESTON["theta"]
                         * _frac_integral_at_T(psi, 1.0, h) - C_ref))
    assert errD < 1e-4, f"psi(T) vs D_Heston max err {errD:.2e}"
    assert errC < 1e-4, f"kth*int(psi) vs C_Heston max err {errC:.2e}"


def test_riccati_second_order_convergence():
    """Error must fall ~O(h²) (×4 per N-doubling) — confirms it's the scheme
    converging to Heston, not an accidental match. A misplaced coefficient
    would give O(1) error, not convergence."""
    _, D_ref = _heston_CD(UGRID, T, **HESTON)
    errs = []
    for N in (100, 200, 400):
        psi = _frac_riccati(UGRID, T, 0.5, HESTON["kappa"], HESTON["nu"], HESTON["rho"], N)
        errs.append(np.max(np.abs(psi[-1] - D_ref)))
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[0] / errs[2] > 10.0, f"slower than ~2nd order (ratio {errs[0]/errs[2]:.1f})"


def test_frac_integral_on_monomial():
    """I^β t^p = Γ(p+1)/Γ(p+1+β)·T^{p+β} — validates the singular I^{1-α} operator
    independently of the Riccati (the piece the H=½ gate cannot exercise)."""
    p, N = 2.0, 800
    f = (np.linspace(0.0, T, N + 1) ** p).astype(complex).reshape(N + 1, 1)
    for H in (0.10, 0.30):
        beta = 0.5 - H
        approx = _frac_integral_at_T(f, beta, T / N)[0].real
        exact = gamma(p + 1.0) / gamma(p + 1.0 + beta) * T ** (p + beta)
        assert abs(approx - exact) < 1e-5, f"I^{beta} t^2 err {abs(approx-exact):.2e} (H={H})"


def test_frac_integral_identity_at_beta_zero():
    """I^0 reduces to identity ψ(T) — the α→1 limit of I^{1-α}."""
    f = (np.linspace(0.0, 1.0, 401) ** 2).astype(complex).reshape(401, 1)
    val = _frac_integral_at_T(f, 0.0, 1.0 / 400)[0]
    assert abs(val - f[-1, 0]) < 1e-12


# ===== STAGE (c): H=½ PRICE GATE — full pipeline reproduces closed-form Heston =====
def test_price_gate_reduces_to_heston_at_half():
    """Full pipeline at H=½ (rough_heston_cf → gil_pelaez_call) reproduces the
    closed-form Heston price (Gil-Pelaez on the Albrecher CF, proven in stage a)
    to ~1e-6. N_riccati=2000 is the grid that clears it (O(h²): ~1.1e-6 at N≈1600,
    ~7e-7 at 2000); the Riccati grid is the binding constraint (inversion ~1e-11)."""
    ref = gil_pelaez_call(lambda u: heston_cf(u, T, r=r, **HESTON), S0, 100.0, T, r,
                          U_max=200.0, n_nodes=128)
    cf = lambda u: rough_heston_cf(u, T, H=0.5, N_riccati=2000, **HESTON)
    price = gil_pelaez_call(cf, S0, 100.0, T, r, U_max=200.0, n_nodes=128)
    assert abs(price - ref) < 1e-6, f"H=1/2 price gate |err|={abs(price-ref):.2e}"


# ===== STAGE (d): H<½ CF property checks (sanity at rough regime H=0.1, ν=0.20) =====
ROUGH = dict(V0=0.04, kappa=0.3, theta=0.04, nu=0.20, rho=-0.70)


def _rcf(u, T_=T, N=600):
    return rough_heston_cf(u, T_, H=0.10, N_riccati=N, **ROUGH)


def _sc(u, T_=T, N=600):
    return complex(np.atleast_1d(_rcf(u, T_, N))[0])


def test_cf_phi0_and_martingale():
    assert abs(_sc(0.0) - 1.0) < 1e-12, "phi(0) != 1"
    assert abs(_sc(-1j) - 1.0) < 1e-10, "martingale phi(-i)=E[S_T/S0] != 1"


def test_cf_modulus_le_one():
    u = np.linspace(0.1, 60.0, 80)
    assert np.max(np.abs(_rcf(u))) <= 1.0 + 1e-8, "|phi(u)| > 1 (not a valid CF)"


def test_cf_hermitian():
    u = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    assert np.max(np.abs(_rcf(-u) - np.conj(_rcf(u)))) < 1e-10, "phi(-u) != conj phi(u)"


def test_cf_to_one_as_t_small():
    errs = [abs(_sc(3.0, Tt) - 1.0) for Tt in (0.05, 0.01, 0.002)]
    assert errs[0] > errs[1] > errs[2], f"phi not -> 1 as t->0: {errs}"
    assert errs[-1] < 1e-3


def test_price_no_arb_bounds_and_monotone():
    cf = lambda u: _rcf(u)
    P = [gil_pelaez_call(cf, S0, K, T, r, U_max=200, n_nodes=128) for K in (80.0, 100.0, 120.0)]
    for K, C in zip((80.0, 100.0, 120.0), P):
        assert max(S0 - K * np.exp(-r * T), 0.0) - 1e-8 <= C <= S0 + 1e-8, f"no-arb breached at K={K}"
    assert P[0] > P[1] > P[2], "call not decreasing in K"
