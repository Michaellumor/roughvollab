"""
rough_heston_cf.py — rough-Heston CF reference (Layer 4, brick 2)
================================================================
Simulation-free European option prices via the El Euch–Rosenbaum characteristic
function (fractional Riccati) + Gil-Pelaez Fourier inversion — the known-truth
the Layer-4 weak-order study (`layer4_convergence.py`, later) measures against.
Spec: docs/gate_checks/layer4_convergence_gate_check.md §2 / §6.

ACCEPTANCE GATE: at H=½ the rough-Heston CF must reduce to the CLASSICAL HESTON
CF, and the inverted prices must match the closed-form Heston price to ~1e-6.

Build order (two-stage validation, each gated before the next):
  (a) INVERSION on KNOWN CFs  ← THIS STAGE
        - invert the Black–Scholes CF → match the BS formula to ~1e-8
        - invert the Albrecher little-trap Heston CF → match the P1/P2 Heston
          price (fixed Gauss–Legendre vs near-exact adaptive quad) to ~1e-6
        - inversion knobs (U_max, n_nodes) convergence-checked on their own
      Inversion proven BEFORE any Riccati code is trusted — that is the
      isolation that makes a reference bug findable.
  (b) RICCATI reduction at H=½  (next: fractional Adams; ψ(u,T)==D_Heston)
  (c) FULL PIPELINE at H=½ == closed-form Heston (~1e-6)   ← the gate
  (d) H<½ CF property checks

This file (stage a) contains ONLY the inversion + the known-CF references. The
fractional Riccati / rough-Heston CF is added in stage (b) once (a) passes.

Convention (must match rough_heston.py exactly): rough Heston with kernel
(t-s)^{H-½}/Γ(H+½), drift κ(θ-V), diffusion ν√V, corr ρ; α = H+½. r = 0.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, gamma
from scipy.integrate import quad

__all__ = ["bs_call", "bs_cf", "heston_cf", "gil_pelaez_call"]


# --------------------------------------------------------------------------- #
# Black–Scholes: closed form + characteristic function (the inverter's anchor)
# --------------------------------------------------------------------------- #
def bs_call(S0, K, T, r, sigma):
    """Black–Scholes European call (analytic)."""
    if sigma * np.sqrt(T) == 0.0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def bs_cf(u, T, sigma, r=0.0):
    """Forward-log CF of log(S_T/S0) under Black–Scholes: E[exp(iu·log(S_T/S0))]."""
    u = np.asarray(u, dtype=complex)
    return np.exp(1j * u * (r - 0.5 * sigma ** 2) * T - 0.5 * sigma ** 2 * T * u ** 2)


# --------------------------------------------------------------------------- #
# Classical Heston CF — Albrecher et al. (2007) "little trap" (branch-stable)
# --------------------------------------------------------------------------- #
def _heston_CD(u, T, V0, kappa, theta, nu, rho):
    """Albrecher little-trap Heston C(u,T), D(u,T): log φ = C + D·V0 (r=0).
    At H=½ the rough-Heston fractional Riccati must give ψ(u,T)=D and κθ∫ψ=C
    — these are the stage-(b) gate references."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = kappa - rho * nu * iu
    d = np.sqrt(beta ** 2 + nu ** 2 * (u ** 2 + iu))
    g = (beta - d) / (beta + d)                       # little-trap: root with −d
    edt = np.exp(-d * T)
    D = ((beta - d) / nu ** 2) * (1.0 - edt) / (1.0 - g * edt)
    C = (kappa * theta / nu ** 2) * ((beta - d) * T
                                     - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    return C, D


def heston_cf(u, T, V0, kappa, theta, nu, rho, r=0.0):
    """Forward-log CF of log(S_T/S0) under classical Heston, Albrecher little-trap
    form (branch-cut-stable). nu is the Heston vol-of-vol (σ).

        dV = κ(θ−V)dt + ν√V dW,   d⟨W_S,W⟩ = ρ dt
    """
    C, D = _heston_CD(u, T, V0, kappa, theta, nu, rho)
    iu = 1j * np.asarray(u, dtype=complex)
    return np.exp(C + D * V0 + iu * r * T)


# --------------------------------------------------------------------------- #
# Gil-Pelaez European-call inversion (the reusable inverter for rough Heston)
# --------------------------------------------------------------------------- #
def gil_pelaez_call(cf, S0, K, T, r=0.0, U_max=200.0, n_nodes=256):
    """European call from a forward-log CF φ(u)=E[exp(iu·log(S_T/S0))], via the
    Gil-Pelaez P1/P2 representation with a FIXED Gauss–Legendre rule on [0,U_max].

        C = S0·P1 − K·e^{−rT}·P2
        P2 = ½ + (1/π)∫₀^∞ Re[e^{−iuk}·φ(u)/(iu)] du            (k = ln(K/S0))
        P1 = ½ + (1/π)∫₀^∞ Re[e^{−iuk}·φ(u−i)/(iu·φ(−i))] du

    Fixed nodes (not adaptive) by design: for rough Heston each φ(u) costs a
    fractional-Riccati solve, so the Riccati is solved once per fixed u-node.
    (U_max, n_nodes) are the inversion's OWN error source — convergence-checked
    separately from the Riccati grid.
    """
    k = np.log(K / S0)
    x, w = leggauss(n_nodes)                           # nodes on [-1, 1]
    u = 0.5 * U_max * (x + 1.0)                         # map to (0, U_max)
    wu = 0.5 * U_max * w
    phi = cf(u)
    phi_shift = cf(u - 1j)
    phi_mi = cf(-1j)                                    # φ(−i) = E[S_T/S0] = e^{rT}
    e = np.exp(-1j * u * k)
    integ2 = np.real(e * phi / (1j * u))
    integ1 = np.real(e * phi_shift / (1j * u * phi_mi))
    P2 = 0.5 + (1.0 / np.pi) * np.sum(wu * integ2)
    P1 = 0.5 + (1.0 / np.pi) * np.sum(wu * integ1)
    return S0 * P1 - K * np.exp(-r * T) * P2


def _gil_pelaez_call_adaptive(cf, S0, K, T, r=0.0):
    """Near-exact reference: same Gil-Pelaez price via adaptive scipy.quad on
    [0,∞). Used ONLY to validate the fixed Gauss–Legendre inverter's accuracy."""
    k = np.log(K / S0)
    phi_mi = complex(cf(-1j))
    f2 = lambda u: float(np.real(np.exp(-1j * u * k) * cf(u) / (1j * u)))
    f1 = lambda u: float(np.real(np.exp(-1j * u * k) * cf(u - 1j) / (1j * u * phi_mi)))
    P2 = 0.5 + (1.0 / np.pi) * quad(f2, 0.0, np.inf, limit=400)[0]
    P1 = 0.5 + (1.0 / np.pi) * quad(f1, 0.0, np.inf, limit=400)[0]
    return S0 * P1 - K * np.exp(-r * T) * P2


# --------------------------------------------------------------------------- #
# Rough-Heston CF — El Euch–Rosenbaum 2019 (arXiv:1609.02108, eq.(3)+main thm),
# mapped to our absolute-ν model (see module header). Fractional Riccati via
# Diethelm–Ford–Freed FABM; the I^{1-α} fractional integral is a SEPARATE op.
# --------------------------------------------------------------------------- #
def _frac_riccati(u, T, H, kappa, nu, rho, N):
    """Fractional Adams–Bashforth–Moulton solve of
        D^α ψ = −½(u²+iu) + (iuρν − κ)ψ + ½ν²ψ²,  ψ(·,0)=0,  α = H+½.
    Returns ψ on the time grid, shape (N+1, n_u). Vectorised over u.
    Reduces to the trapezoidal predictor–corrector (classical Heston ODE) at α=1."""
    u = np.asarray(u, dtype=complex).ravel()
    alpha = H + 0.5
    h = T / N
    c0 = -0.5 * (u ** 2 + 1j * u)
    c1 = 1j * u * rho * nu - kappa
    hn = 0.5 * nu ** 2
    F = lambda psi: c0 + c1 * psi + hn * psi ** 2
    psi = np.zeros((N + 1, u.size), dtype=complex)
    Fh = np.zeros((N + 1, u.size), dtype=complex)
    Fh[0] = F(psi[0])                                   # = c0  (ψ_0 = 0)
    ha, Ga, Ga2 = h ** alpha, gamma(alpha), gamma(alpha + 2.0)
    for k in range(N):
        j = np.arange(k + 1)
        b = (ha / alpha) * ((k + 1 - j) ** alpha - (k - j) ** alpha)     # AB weights
        pred = (1.0 / Ga) * (b @ Fh[:k + 1])
        a = np.empty(k + 1)                                              # AM weights
        a[0] = k ** (alpha + 1.0) - (k - alpha) * (k + 1.0) ** alpha
        if k >= 1:
            jj = np.arange(1, k + 1)
            a[1:] = ((k - jj + 2.0) ** (alpha + 1.0) + (k - jj) ** (alpha + 1.0)
                     - 2.0 * (k - jj + 1.0) ** (alpha + 1.0))
        psi[k + 1] = (ha / Ga2) * (F(pred) + (a @ Fh[:k + 1]))
        Fh[k + 1] = F(psi[k + 1])
    return psi


def _frac_integral_at_T(psi, beta, h):
    """(I^β ψ)(T) by product-trapezoidal weights (β=1 → ordinary trapezoid;
    β=0 → identity ψ(T)). Handles the singular (T−s)^{β−1} kernel for β<1.
    psi shape (N+1, n_u) → returns (n_u,)."""
    N = psi.shape[0] - 1
    c = np.empty(N + 1)
    c[N] = 1.0
    c[0] = (N - 1.0) ** (beta + 1.0) - (N - 1.0 - beta) * N ** beta
    if N >= 2:
        jj = np.arange(1, N)
        c[1:N] = ((N - jj + 1.0) ** (beta + 1.0) + (N - jj - 1.0) ** (beta + 1.0)
                  - 2.0 * (N - jj) ** (beta + 1.0))
    return (h ** beta / gamma(beta + 2.0)) * (c @ psi)


def rough_heston_cf(u, T, V0, kappa, theta, nu, rho, H, N_riccati=300):
    """Rough-Heston forward-log CF:  log φ = κθ·I¹ψ(T) + V₀·I^{1−α}ψ(T)."""
    alpha = H + 0.5
    h = T / N_riccati
    psi = _frac_riccati(u, T, H, kappa, nu, rho, N_riccati)
    int1 = _frac_integral_at_T(psi, 1.0, h)             # κθ term: ∫₀ᵀ ψ ds
    iqa = _frac_integral_at_T(psi, 1.0 - alpha, h)      # V0 term: I^{1−α}ψ(T)
    return np.exp(kappa * theta * int1 + V0 * iqa)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    S0, T, r = 100.0, 1.0, 0.0
    print("STAGE (a) — inversion validated on KNOWN CFs (no Riccati yet)\n")

    print("BS: gil_pelaez(bs_cf) vs analytic BS  (target ~1e-8)")
    sig = 0.20
    for K in (80.0, 100.0, 120.0):
        gp = gil_pelaez_call(lambda u: bs_cf(u, T, sig, r), S0, K, T, r)
        an = bs_call(S0, K, T, r, sig)
        print(f"   K={K:5.0f}  gp={gp:.10f}  analytic={an:.10f}  |err|={abs(gp-an):.2e}")

    print("\nHeston: fixed-GL vs adaptive-quad on Albrecher CF  (target ~1e-6)")
    hp = dict(V0=0.04, kappa=2.0, theta=0.04, nu=0.5, rho=-0.7)
    cf = lambda u: heston_cf(u, T, r=r, **hp)
    for K in (80.0, 100.0, 120.0):
        gl = gil_pelaez_call(cf, S0, K, T, r)
        ad = _gil_pelaez_call_adaptive(cf, S0, K, T, r)
        print(f"   K={K:5.0f}  fixedGL={gl:.10f}  adaptive={ad:.10f}  |err|={abs(gl-ad):.2e}")

    print("\nHeston CF sanity:  phi(0)=1, phi(-i)=1 (martingale, r=0)")
    print(f"   phi(0)  = {complex(cf(0.0)):.10f}")
    print(f"   phi(-i) = {complex(cf(-1j)):.10f}")

    # ---- STAGE (b): H=1/2 CF-level reduction (the arbiter) ----
    print("\nSTAGE (b) — H=1/2 fractional Riccati reduces to classical Heston (CF level)")
    print("   (arbiter for every coefficient; isolates Riccati+CF-assembly from the inverter)")
    ug = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    C_ref, D_ref = _heston_CD(ug, T, **hp)
    print(f"   {'N_riccati':>10} {'max|psi(T)-D_Heston|':>22} {'max|kth*int psi - C_Heston|':>30}")
    for N in (100, 200, 400, 800):
        psi = _frac_riccati(ug, T, 0.5, hp["kappa"], hp["nu"], hp["rho"], N)
        hh = T / N
        errD = np.max(np.abs(psi[-1] - D_ref))
        errC = np.max(np.abs(hp["kappa"] * hp["theta"]
                             * _frac_integral_at_T(psi, 1.0, hh) - C_ref))
        print(f"   {N:>10} {errD:>22.3e} {errC:>30.3e}")

    # ---- STAGE (b): the fractional integral on its own (H=1/2 can't test it) ----
    print("\nSTAGE (b) — fractional integral I^{1-alpha} validated independently (H<1/2)")
    print("   I^beta t^p = Gamma(p+1)/Gamma(p+1+beta) * T^{p+beta}   (p=2)")
    p = 2.0
    for H in (0.10, 0.30):
        beta = 0.5 - H                                  # 1 - alpha
        exact = gamma(p + 1.0) / gamma(p + 1.0 + beta) * T ** (p + beta)
        for N in (800,):
            tj = np.linspace(0.0, T, N + 1)
            f = (tj ** p).astype(complex).reshape(N + 1, 1)
            approx = _frac_integral_at_T(f, beta, T / N)[0].real
        print(f"   H={H} (beta={beta:.2f}): approx={approx:.8f} exact={exact:.8f} "
              f"|err|={abs(approx-exact):.2e}")
    # identity at beta -> 0
    f = np.linspace(0.0, 1.0, 401).astype(complex).reshape(401, 1) ** 2
    idn = _frac_integral_at_T(f, 0.0, 1.0 / 400)[0]
    print(f"   identity check I^0 f = f(T): {idn.real:.10f} vs {f[-1,0].real:.10f} "
          f"|err|={abs(idn - f[-1,0]):.1e}")
