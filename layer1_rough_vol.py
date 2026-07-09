"""
Layer 1 — Stochastic Simulation Core
=====================================
Rough Volatility: Theory and Numerical Implementation

Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

Structure
---------
  Section 1  Fractional Brownian Motion (exact Cholesky method)
  Section 2  Hybrid O(N log N) Scheme  (Bennedsen-Lunde-Pakkanen 2017)
  Section 3  Rough Bergomi and Rough Heston path simulation
  Section 4  Hurst exponent estimation from realised variance

Each section prints its results and saves a figure to ./output/.
Run the whole file with:  python layer1_rough_vol.py
Or run one section:       python layer1_rough_vol.py --section 2

Correction note (ROADMAP L1-1)
------------------------------
Section 2/3's rough-path construction was corrected under ROADMAP issue L1-1.
``fbm_hybrid`` now produces a Volterra path normalised to the discrete variance
v = 2H·dt·cumsum(g²), so Var(W̃_{t_i}) = v_i by construction, and
``rough_bergomi_paths`` uses that same v as its lognormal compensator (not the
continuum t^{2H}), so E[V_t] = ξ₀ holds exactly on the grid. This is the
readable teaching implementation; for production or correctness-critical work,
``roughvol_core.py`` is the separately-validated engine.
"""

import argparse
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import gamma

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("output", exist_ok=True)

# ── colour palette (matches project blueprint) ──────────────────────────────
TEAL   = "#1D9E75"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
GRAY   = "#888780"
AMBER  = "#BA7517"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.size":         11,
})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Fractional Brownian Motion (exact Cholesky method)
# ══════════════════════════════════════════════════════════════════════════════

def fbm_covariance(n: int, H: float) -> np.ndarray:
    """
    Build the N×N covariance matrix of fBm at times t = 1/N, 2/N, ..., 1.

    The covariance of fBm with Hurst exponent H is:
        Cov(B^H_s, B^H_t) = ½ (|s|^{2H} + |t|^{2H} - |t-s|^{2H})

    This is the exact representation — no approximation. It becomes the
    ground truth against which the fast hybrid scheme is validated.

    Parameters
    ----------
    n : int   Number of time steps.
    H : float Hurst exponent ∈ (0, 1).  H = 0.5 → standard Brownian motion.
              Rough volatility requires H ≈ 0.1.

    Returns
    -------
    C : np.ndarray  shape (n, n)  Covariance matrix.
    """
    t = np.arange(1, n + 1) / n          # times: 1/N, 2/N, ..., 1
    s, tt = np.meshgrid(t, t)             # pairwise grid
    C = 0.5 * (np.abs(s)**(2*H) + np.abs(tt)**(2*H) - np.abs(tt - s)**(2*H))
    return C


def fbm_exact(n: int, H: float, n_paths: int = 1) -> np.ndarray:
    """
    Simulate fBm paths using the exact Cholesky factorisation method.

    Algorithm:
      1. Build covariance matrix C (n×n).
      2. Compute lower-triangular Cholesky factor L  s.t.  C = L Lᵀ.
      3. Sample Z ~ N(0, I_n).
      4. Return X = L Z.

    Complexity: O(N³) — exact but slow. Used only for small N as ground truth.

    Parameters
    ----------
    n        : int   Number of time steps.
    H        : float Hurst exponent.
    n_paths  : int   Number of independent paths.

    Returns
    -------
    paths : np.ndarray  shape (n_paths, n)
    """
    C = fbm_covariance(n, H)
    L = cholesky(C, lower=True)
    Z = np.random.standard_normal((n, n_paths))
    return (L @ Z).T


def section1_fbm(show: bool = True):
    """Run Section 1: simulate fBm for several Hurst values, plot, report."""
    print("\n" + "="*70)
    print("SECTION 1 — Fractional Brownian Motion (exact Cholesky)")
    print("="*70)

    N = 200
    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = [CORAL, AMBER, GRAY, TEAL, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    t = np.linspace(0, 1, N)

    # ── left panel: sample paths ──
    ax = axes[0]
    for H, col in zip(hurst_values, colors):
        path = fbm_exact(N, H, n_paths=1)[0]
        ax.plot(t, path, color=col, linewidth=1.2, alpha=0.9,
                label=f"H = {H}")
    ax.set_title("Fractional Brownian motion — sample paths")
    ax.set_xlabel("t")
    ax.set_ylabel("$B^H_t$")
    ax.legend(fontsize=9)

    # ── right panel: variance scaling ──
    ax = axes[1]
    t_range = np.linspace(0.01, 1, 100)
    for H, col in zip(hurst_values, colors):
        theoretical_var = t_range**(2*H)
        ax.plot(t_range, theoretical_var, color=col, linewidth=1.5,
                label=f"H = {H}")
    ax.set_title("Theoretical variance  Var($B^H_t$) = $t^{2H}$")
    ax.set_xlabel("t")
    ax.set_ylabel("Variance")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("output/section1_fbm.png", dpi=150, bbox_inches="tight")
    print("  Figure saved: output/section1_fbm.png")

    # ── print key properties ──
    print("\n  Hurst exponent properties:")
    print(f"  {'H':>6}  {'Roughness':>12}  {'Self-similarity':>16}  "
          f"{'Var(B^H_1)':>12}")
    for H in hurst_values:
        roughness = "rough" if H < 0.5 else ("BM" if H == 0.5 else "smooth")
        print(f"  {H:>6.1f}  {roughness:>12}  {'t^'+str(2*H):>16}  "
              f"{'1.0':>12}")

    print("\n  Rough volatility regime: H ≈ 0.1  (Gatheral-Jaisson-Rosenbaum 2018)")
    print("  Standard Brownian motion: H = 0.5")
    print("  O(N³) exact method — ground truth for validation only")

    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Hybrid O(N log N) Scheme  (Bennedsen-Lunde-Pakkanen 2017)
# ══════════════════════════════════════════════════════════════════════════════

def _volterra_weights(n: int, H: float, T: float = 1.0):
    """
    κ = 0 hybrid-scheme convolution weights (self-contained; the same maths as
    the validated ``roughvol_core.volterra_weights``, kept local so this teaching
    module does the correct construction itself rather than importing the engine).

        g_m = (b_m · dt)^{H - 1/2}      b_m = Bennedsen–Lunde–Pakkanen (2017)
                                        optimal evaluation point in cell m
        v_i = 2H · dt · Σ_{m ≤ i} g_m²  =  Var(W̃_{t_i}) for the process
        W̃   = sqrt(2H) · (g ∗ dW),      dW ~ N(0, dt).

    Use v — NOT the continuum t^{2H} — in any lognormal compensator so that
    E[V_t] = ξ₀ holds exactly on the discrete grid.

    Returns
    -------
    g : np.ndarray (n,)   convolution kernel
    v : np.ndarray (n,)   discrete variance, v[i-1] = Var(W̃_{t_i})
    """
    a  = H - 0.5
    dt = T / n
    m  = np.arange(1, n + 1)
    if abs(a) < 1e-12:                  # H = 1/2: removable singularity — the
        g = np.ones(n)                  # kernel is flat, i.e. standard Brownian motion
    else:
        b = ((m**(a + 1) - (m - 1)**(a + 1)) / (a + 1)) ** (1.0 / a)
        g = (b * dt) ** a
    v = 2.0 * H * dt * np.cumsum(g**2)
    return g, v


def fbm_hybrid(n: int, H: float, n_paths: int = 1,
               n_wiener: int = None) -> np.ndarray:
    """
    Simulate the rough Volterra process W̃ via the κ = 0 optimal hybrid scheme of
    Bennedsen, Lunde & Pakkanen (2017).

        W̃_t = sqrt(2H) ∫₀ᵗ (t - s)^{H - 1/2} dW_s

    The power-law kernel is singular at s = t. The hybrid scheme places, in each
    cell m, an *optimal* evaluation point b_m (so the near-diagonal singularity is
    integrated accurately) and convolves the resulting kernel weights
    g_m = (b_m·dt)^{H-1/2} with the Brownian increments in a single FFT:

        W̃ = sqrt(2H) · (g ∗ dW),      dW ~ N(0, dt).

    This is O(N log N) and is normalised to the DISCRETE variance
    v_i = 2H·dt·Σ_{m≤i} g_m², so **Var(W̃_{t_i}) = v_i by construction** (v_i → 1
    as N → ∞). Getting this normalisation right is ROADMAP issue L1-1 (RVL-001):
    the previous split-kernel construction over-subtracted the near-diagonal
    terms and undershot the variance (~0.91).

    Parameters
    ----------
    n        : int   Number of time steps on [0, 1].
    H        : float Hurst exponent ∈ (0, 1)  (H = ½ ⇒ standard Brownian motion).
    n_paths  : int   Number of independent paths.
    n_wiener : int   Retained for API compatibility; unused by the κ = 0 optimal
                     scheme, which needs no explicit exact/FFT split.

    Returns
    -------
    paths : np.ndarray  shape (n_paths, n)   W̃_{t_1}, …, W̃_{t_n} for each path.
    """
    dt = 1.0 / n
    g, _ = _volterra_weights(n, H)
    # W̃ = sqrt(2H) · (g ∗ dW). The causal (linear) convolution gives
    # W̃_{t_i} = sqrt(2H) Σ_{m=1}^{i} g_m · dW_{i-m+1}, hence Var(W̃_{t_i}) = v_i.
    dW = np.random.standard_normal((n_paths, n)) * np.sqrt(dt)
    paths = np.sqrt(2.0 * H) * fftconvolve(dW, g[None, :], axes=1)[:, :n]
    return paths


def section2_convergence(show: bool = True):
    """
    Section 2: benchmark hybrid scheme vs exact Cholesky.
    Measure strong convergence rate: E[|error|²]^{1/2} vs N.
    Theoretical prediction: O(N^{-H}).
    """
    print("\n" + "="*70)
    print("SECTION 2 — Hybrid O(N log N) Scheme  (BLP 2017)")
    print("="*70)

    H = 0.1          # rough volatility regime
    N_ref = 512      # reference (fine) grid
    N_coarse_vals = [16, 32, 64, 128, 256]
    n_mc = 200       # Monte Carlo paths for error estimation

    print(f"\n  Hurst exponent H = {H}  (rough volatility regime)")
    print(f"  Reference grid N_ref = {N_ref}")
    print(f"  Monte Carlo paths for error: {n_mc}")
    print("\n  Computing strong convergence rates...")

    # Reference paths on fine grid
    t0 = time.time()
    ref_paths = fbm_exact(N_ref, H, n_paths=n_mc)
    print(f"  Exact reference computed in {time.time()-t0:.1f}s")

    errors_exact  = []
    errors_hybrid = []
    times_exact   = []
    times_hybrid  = []

    for N in N_coarse_vals:
        step = N_ref // N           # subsample reference to N points

        # ── exact Cholesky on coarse grid ──
        t0 = time.time()
        exact_coarse = fbm_exact(N, H, n_paths=n_mc)
        times_exact.append(time.time() - t0)

        # ── hybrid scheme on coarse grid ──
        t0 = time.time()
        hybrid_coarse = fbm_hybrid(N, H, n_paths=n_mc)
        times_hybrid.append(time.time() - t0)

        # strong L2 error vs reference (compare at final time point)
        ref_end    = ref_paths[:, -1]
        exact_end  = exact_coarse[:, -1]
        hybrid_end = hybrid_coarse[:, -1]

        err_e = np.sqrt(np.mean((exact_end - ref_end)**2))
        err_h = np.sqrt(np.mean((hybrid_end - ref_end)**2))

        errors_exact.append(err_e)
        errors_hybrid.append(err_h)

        print(f"    N={N:>4}  exact_err={err_e:.4f}  hybrid_err={err_h:.4f}  "
              f"t_exact={times_exact[-1]:.3f}s  t_hybrid={times_hybrid[-1]:.3f}s")

    # ── fit convergence rates (log-log slope) ──
    log_N  = np.log(N_coarse_vals)
    rate_e = np.polyfit(log_N, np.log(errors_exact),  1)[0]
    rate_h = np.polyfit(log_N, np.log(errors_hybrid), 1)[0]

    print(f"\n  Strong convergence rate (log-log slope):")
    print(f"    Exact Cholesky:  {rate_e:.3f}   (theoretical: -{H:.1f})")
    print(f"    Hybrid BLP:      {rate_h:.3f}   (theoretical: -{H:.1f})")

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # panel 1: convergence rates
    ax = axes[0]
    ax.loglog(N_coarse_vals, errors_exact,  "o-", color=TEAL,   label="Exact Cholesky")
    ax.loglog(N_coarse_vals, errors_hybrid, "s-", color=PURPLE, label="Hybrid BLP")
    N_arr = np.array(N_coarse_vals, dtype=float)
    ax.loglog(N_arr, 0.5 * N_arr**(-H), "--", color=GRAY, alpha=0.6,
              label=f"O(N^{{-{H}}})")
    ax.set_title("Strong convergence rate")
    ax.set_xlabel("N (steps)")
    ax.set_ylabel("L² error")
    ax.legend(fontsize=9)

    # panel 2: timing comparison
    ax = axes[1]
    ax.semilogy(N_coarse_vals, times_exact,  "o-", color=TEAL,   label="Exact O(N³)")
    ax.semilogy(N_coarse_vals, times_hybrid, "s-", color=PURPLE, label="Hybrid O(N log N)")
    ax.set_title("Wall-clock time")
    ax.set_xlabel("N (steps)")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=9)

    # panel 3: sample paths comparison
    ax = axes[2]
    N_demo = 128
    path_exact  = fbm_exact(N_demo,  H, n_paths=1)[0]
    path_hybrid = fbm_hybrid(N_demo, H, n_paths=1)[0]
    t_demo = np.linspace(0, 1, N_demo)
    ax.plot(t_demo, path_exact,  color=TEAL,   linewidth=1.2, label="Exact",  alpha=0.85)
    ax.plot(t_demo, path_hybrid, color=PURPLE, linewidth=1.0, label="Hybrid", alpha=0.85,
            linestyle="--")
    ax.set_title(f"Sample paths  H = {H}")
    ax.set_xlabel("t")
    ax.set_ylabel("$B^H_t$")
    ax.legend(fontsize=9)

    plt.suptitle("Section 2: Hybrid O(N log N) scheme — BLP 2017", y=1.01)
    plt.tight_layout()
    plt.savefig("output/section2_hybrid.png", dpi=150, bbox_inches="tight")
    print("  Figure saved: output/section2_hybrid.png")

    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Rough Bergomi and Rough Heston path simulation
# ══════════════════════════════════════════════════════════════════════════════

def rough_bergomi_paths(n: int, H: float, xi0: float, eta: float,
                        rho: float, S0: float, r: float,
                        n_paths: int = 1000) -> tuple:
    """
    Simulate stock price and variance paths under the rough Bergomi model.

    The rough Bergomi model (Bayer-Friz-Gatheral 2016):

        dS_t = S_t √(V_t) dW_t^S                   (stock under risk-neutral measure)
        V_t  = ξ₀ exp(η B^H_t - ½ η² t^{2H})       (rough stochastic variance)

    where B^H is fractional Brownian motion with Hurst exponent H,
    and W^S, B^H are correlated:  Corr(dW^S, dB^H) = ρ.

    Numerically the continuum compensator ½η²t^{2H} is replaced by the discrete
    variance ½η²v_t, v_t = 2H·dt·cumsum(g²), so E[V_t] = ξ₀ holds exactly on the
    grid (ROADMAP L1-1 / RVL-002).

    Key feature: V_t is log-normal, B^H gives rough sample paths of V.
    The negative correlation ρ < 0 produces the observed volatility skew.

    Parameters
    ----------
    n       : int    Number of time steps (T=1 year).
    H       : float  Hurst exponent (rough vol: ≈ 0.1).
    xi0     : float  Initial forward variance (typical: 0.04 = 20% vol).
    eta     : float  Vol-of-vol parameter.
    rho     : float  Correlation ∈ (-1, 0).
    S0      : float  Initial stock price.
    r       : float  Risk-free rate.
    n_paths : int    Number of Monte Carlo paths.

    Returns
    -------
    S : np.ndarray  shape (n_paths, n+1)  stock price paths
    V : np.ndarray  shape (n_paths, n+1)  instantaneous variance paths
    """
    dt   = 1.0 / n
    t    = np.linspace(0, 1, n + 1)

    # simulate correlated fBm and BM
    # dW^S = ρ dB^H + √(1-ρ²) dW^⊥
    B_H    = fbm_hybrid(n, H, n_paths=n_paths)         # shape (n_paths, n)
    dW_perp = np.random.standard_normal((n_paths, n)) * np.sqrt(dt)

    # reconstruct dB^H from B^H
    B_H_full = np.hstack([np.zeros((n_paths, 1)), B_H])   # prepend B^H_0 = 0

    S = np.zeros((n_paths, n + 1))
    V = np.zeros((n_paths, n + 1))
    S[:, 0] = S0
    V[:, 0] = xi0

    # RVL-002: discrete compensator v[k] = Var(W̃_{t_{k+1}}) from the corrected
    # kernel — keeps E[V_t] = ξ₀ exact on the grid. The continuum t^{2H} does not
    # match the discrete scheme's variance (it made E[V_t]/ξ₀ dip to ~0.35).
    _, v = _volterra_weights(n, H)
    for k in range(n):
        # variance at time t_{k+1}
        V[:, k+1] = xi0 * np.exp(
            eta * B_H_full[:, k+1] - 0.5 * eta**2 * v[k]
        )
        sqrt_V = np.sqrt(np.maximum(V[:, k], 1e-8))

        # correlated stock increment
        dW_S = rho * (B_H_full[:, k+1] - B_H_full[:, k]) \
               + np.sqrt(1 - rho**2) * dW_perp[:, k]

        S[:, k+1] = S[:, k] * np.exp(
            (r - 0.5 * V[:, k]) * dt + sqrt_V * dW_S
        )

    return S, V


def rough_heston_paths(n: int, H: float, V0: float, kappa: float,
                       theta: float, nu: float, rho: float, S0: float,
                       r: float, n_paths: int = 500) -> tuple:
    """
    Simulate stock price and variance paths under the rough Heston model.

    The rough Heston model (El Euch & Rosenbaum 2019):

        V_t = V₀ + (1/Γ(H+½)) ∫₀ᵗ (t-s)^{H-½} κ(θ - V_s) ds
              + (1/Γ(H+½)) ∫₀ᵗ (t-s)^{H-½} ν √(V_s) dW^V_s

    This is a Volterra integral equation — the variance is not Markovian.
    We discretise using an Euler scheme applied to the Volterra form.

    The standard Heston model is recovered in the limit H → ½.

    Parameters
    ----------
    n     : int    Number of time steps.
    H     : float  Hurst exponent.
    V0    : float  Initial variance.
    kappa : float  Mean-reversion speed.
    theta : float  Long-run variance.
    nu    : float  Vol-of-vol.
    rho   : float  Stock-vol correlation.
    S0    : float  Initial stock price.
    r     : float  Risk-free rate.

    Returns
    -------
    S : np.ndarray  shape (n_paths, n+1)
    V : np.ndarray  shape (n_paths, n+1)
    """
    dt    = 1.0 / n
    alpha = H - 0.5                                    # α ∈ (-½, 0)
    G_inv = 1.0 / gamma(H + 0.5)

    # Volterra kernel weights: w_k = (k*dt)^α * dt (rectangular quadrature)
    k_arr = np.arange(1, n + 1)
    weights = G_inv * (k_arr * dt)**alpha * dt         # shape (n,)

    # correlated Brownian increments
    Z1 = np.random.standard_normal((n_paths, n))
    Z2 = np.random.standard_normal((n_paths, n))
    dW_V = np.sqrt(dt) * Z1
    dW_S = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

    V = np.zeros((n_paths, n + 1))
    S = np.zeros((n_paths, n + 1))
    V[:, 0] = V0
    S[:, 0] = S0

    for k in range(n):
        # Volterra sum: ∑_{j=0}^{k-1} w_{k-j} [κ(θ - V_j) + ν √V_j dW^V_j / dt]
        # shape trick: weights[k-j-1] for j = 0..k-1
        if k == 0:
            drift_sum = 0.0
            diff_sum  = 0.0
        else:
            drift_integrand = kappa * (theta - V[:, :k])         # (n_paths, k)
            diff_integrand  = nu * np.sqrt(np.maximum(V[:, :k], 0)) \
                              * dW_V[:, :k] / dt                  # (n_paths, k)
            w_slice = weights[k-1::-1]                            # (k,) reversed
            drift_sum = (drift_integrand * w_slice).sum(axis=1)
            diff_sum  = (diff_integrand  * w_slice).sum(axis=1)

        V[:, k+1] = np.maximum(V0 + drift_sum + diff_sum, 0.0)
        sqrt_V = np.sqrt(np.maximum(V[:, k], 1e-8))
        S[:, k+1] = S[:, k] * np.exp(
            (r - 0.5 * V[:, k]) * dt + sqrt_V * dW_S[:, k]
        )

    return S, V


def section3_rough_models(show: bool = True):
    """Section 3: simulate and plot rough Bergomi and rough Heston paths."""
    print("\n" + "="*70)
    print("SECTION 3 — Rough Bergomi and Rough Heston path simulation")
    print("="*70)

    N       = 252        # daily steps (1 trading year)
    H       = 0.1        # rough volatility regime
    N_PATHS = 500
    S0, r   = 100.0, 0.05

    # ── Rough Bergomi parameters ──
    xi0 = 0.04    # initial forward variance (20% vol)
    eta = 1.9     # vol-of-vol
    rho = -0.9    # leverage effect

    print(f"\n  Rough Bergomi: H={H}, ξ₀={xi0}, η={eta}, ρ={rho}, N_paths={N_PATHS}")
    t0 = time.time()
    S_rB, V_rB = rough_bergomi_paths(N, H, xi0, eta, rho, S0, r, N_PATHS)
    print(f"  Simulated {N_PATHS} paths in {time.time()-t0:.2f}s")

    # ── Rough Heston parameters ──
    V0, kappa, theta, nu, rho_H = 0.04, 0.3, 0.04, 0.5, -0.7

    print(f"\n  Rough Heston: H={H}, V₀={V0}, κ={kappa}, θ={theta}, ν={nu}")
    t0 = time.time()
    S_rH, V_rH = rough_heston_paths(N, H, V0, kappa, theta, nu,
                                     rho_H, S0, r, n_paths=N_PATHS)
    print(f"  Simulated {N_PATHS} paths in {time.time()-t0:.2f}s")

    # ── summary statistics ──
    t_grid = np.linspace(0, 1, N + 1)

    for label, S, V in [("Rough Bergomi", S_rB, V_rB),
                         ("Rough Heston",  S_rH, V_rH)]:
        returns = np.log(S[:, 1:] / S[:, :-1])
        real_var = (returns**2).mean(axis=0) * N
        print(f"\n  {label}:")
        print(f"    Mean terminal price : {S[:, -1].mean():.2f}")
        print(f"    Std terminal price  : {S[:, -1].std():.2f}")
        print(f"    Mean realised vol   : {np.sqrt(real_var.mean()):.4f}")
        print(f"    Mean variance V_T   : {V[:, -1].mean():.4f}")

    # ── plot ──
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    def plot_panel(ax, paths, label, color, ylabel, n_show=30):
        for i in range(min(n_show, paths.shape[0])):
            ax.plot(t_grid, paths[i], color=color, linewidth=0.5, alpha=0.3)
        ax.plot(t_grid, paths.mean(axis=0), color=color, linewidth=2,
                label="Mean")
        ax.set_title(label)
        ax.set_xlabel("t  (years)")
        ax.set_ylabel(ylabel)

    plot_panel(fig.add_subplot(gs[0, 0]), S_rB, "Rough Bergomi — stock paths",
               TEAL,   "S_t")
    plot_panel(fig.add_subplot(gs[0, 1]), V_rB, "Rough Bergomi — variance paths",
               TEAL,   "V_t = σ²_t")
    plot_panel(fig.add_subplot(gs[1, 0]), S_rH, "Rough Heston — stock paths",
               PURPLE, "S_t")
    plot_panel(fig.add_subplot(gs[1, 1]), V_rH, "Rough Heston — variance paths",
               PURPLE, "V_t = σ²_t")

    # terminal distribution comparison
    ax_dist = fig.add_subplot(gs[:, 2])
    ax_dist.hist(np.log(S_rB[:, -1] / S0), bins=40, color=TEAL,
                 alpha=0.6, density=True, label="Rough Bergomi")
    ax_dist.hist(np.log(S_rH[:, -1] / S0), bins=40, color=PURPLE,
                 alpha=0.6, density=True, label="Rough Heston")
    ax_dist.set_title("Log-return distribution  T = 1yr")
    ax_dist.set_xlabel("log(S_T / S_0)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend()

    plt.suptitle("Section 3: Rough volatility model paths", y=1.01)
    plt.savefig("output/section3_models.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved: output/section3_models.png")

    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Hurst exponent estimation from realised variance
# ══════════════════════════════════════════════════════════════════════════════

def estimate_hurst_variogram(log_vol: np.ndarray,
                             lags: np.ndarray = None) -> tuple:
    """
    Estimate the Hurst exponent H using the variogram (structure function).

    For fBm B^H with Hurst exponent H, the variogram satisfies:

        m(q) ≡ E[|B^H_{t+q} - B^H_t|²]  =  q^{2H}

    Taking logs:  log m(q) = 2H · log q + const.

    So a log-log regression of the empirical variogram against the lag q
    gives a slope of 2H, from which H is recovered as slope / 2.

    This is exactly the estimator used by Gatheral-Jaisson-Rosenbaum (2018,
    Section 2.2), applied to the log-realised-variance series
        Y_k = log RV_k
    where RV_k is the daily realised variance computed from 5-min returns.
    The key empirical finding is H ≈ 0.1, far below H = 0.5 (Brownian).

    Parameters
    ----------
    log_vol : np.ndarray  Log-volatility level series (NOT first-differenced).
                          In GJR 2018 this is log(RV_k) for each day k.
    lags    : np.ndarray  Lag values q.  Default: geometric sequence up to N/4.

    Returns
    -------
    H_hat : float   Estimated Hurst exponent.
    ci    : tuple   Bootstrap 95% CI (H_lo, H_hi) over lag subsets.
    lags  : np.ndarray
    vario : np.ndarray  Empirical variogram values m(q).
    """
    T = len(log_vol)
    if lags is None:
        max_lag = max(4, T // 4)
        lags = np.unique(np.geomspace(1, max_lag, num=14).astype(int))
        lags = lags[(lags >= 1) & (lags < T // 3)]

    # empirical variogram: m(q) = (1/(T-q)) sum_{k=0}^{T-q-1} (Y_{k+q} - Y_k)^2
    vario = np.array([
        np.mean((log_vol[lag:] - log_vol[:-lag])**2)
        for lag in lags
    ])

    log_lag   = np.log(lags.astype(float))
    log_vario = np.log(vario)

    A      = np.column_stack([log_lag, np.ones_like(log_lag)])
    coeffs = np.linalg.lstsq(A, log_vario, rcond=None)[0]
    H_hat  = coeffs[0] / 2.0

    # bootstrap CI by resampling which lags to include
    n_boot = 500
    H_boot = np.zeros(n_boot)
    rng    = np.random.default_rng(0)
    for b in range(n_boot):
        idx       = rng.choice(len(lags), len(lags), replace=True)
        c_b       = np.linalg.lstsq(A[idx], log_vario[idx], rcond=None)[0]
        H_boot[b] = c_b[0] / 2.0

    ci = (np.percentile(H_boot, 2.5), np.percentile(H_boot, 97.5))
    return H_hat, ci, lags, vario


def section4_hurst(show: bool = True):
    """
    Section 4: estimate H from simulated rough vol paths.
    Tests whether our estimator recovers the true H.
    Also demonstrates the power analysis: can we distinguish H=0.1 from H=0.5?
    """
    print("\n" + "="*70)
    print("SECTION 4 — Hurst exponent estimation from realised variance")
    print("="*70)

    N = 1000     # high-frequency steps (e.g. 1 year of 4-min bars)
    H_true_vals = [0.1, 0.3, 0.5]
    n_paths_est = 50   # paths for Monte Carlo study of estimator

    print(f"\n  N = {N} steps per path,  {n_paths_est} paths per H value")
    print(f"\n  {'H_true':>8}  {'H_mean':>8}  {'H_std':>8}  "
          f"{'CI_lo':>8}  {'CI_hi':>8}  {'Coverage':>10}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col_idx, H_true in enumerate(H_true_vals):
        H_estimates = []
        ci_covers   = []

        for _ in range(n_paths_est):
            # simulate fBm level path — variogram is applied to levels (GJR 2018)
            path = fbm_hybrid(N, H_true, n_paths=1)[0]
            H_hat, ci, lags, vario = estimate_hurst_variogram(path)
            H_estimates.append(H_hat)
            ci_covers.append(ci[0] <= H_true <= ci[1])

        H_arr     = np.array(H_estimates)
        coverage  = np.mean(ci_covers)
        H_mean, H_std = H_arr.mean(), H_arr.std()
        # one representative CI
        rep_path = fbm_hybrid(N, H_true, n_paths=1)[0]
        _, ci_rep, lags_rep, vario_rep = estimate_hurst_variogram(rep_path)

        print(f"  {H_true:>8.2f}  {H_mean:>8.3f}  {H_std:>8.3f}  "
              f"  {ci_rep[0]:>6.3f}  {ci_rep[1]:>6.3f}  {coverage:>10.2%}")

        # panel: distribution of H_hat
        ax = axes[col_idx]
        ax.hist(H_arr, bins=15, color=[CORAL, TEAL, PURPLE][col_idx],
                alpha=0.75, density=True, edgecolor="white", linewidth=0.5)
        ax.axvline(H_true, color="black", linewidth=1.5, linestyle="--",
                   label=f"True H = {H_true}")
        ax.axvline(H_mean, color=AMBER, linewidth=1.2, linestyle="-",
                   label=f"Mean Ĥ = {H_mean:.3f}")
        ax.set_title(f"Estimator distribution  H = {H_true}")
        ax.set_xlabel("Estimated Ĥ")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    print("\n  Coverage should be ≈ 95% if CI is well-calibrated.")
    print("  Bias = |H_mean - H_true| should be small.")
    print("\n  Rough vol regime (H ≈ 0.1) is statistically distinguishable")
    print("  from standard BM (H = 0.5) — the key empirical claim of GJR 2018.")

    plt.suptitle("Section 4: Hurst exponent estimation — variogram method", y=1.01)
    plt.tight_layout()
    plt.savefig("output/section4_hurst.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved: output/section4_hurst.png")

    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 — Rough Volatility Simulation Core"
    )
    parser.add_argument(
        "--section", type=int, choices=[1, 2, 3, 4],
        help="Run only one section (default: run all four)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display figures interactively (just save to output/)"
    )
    args = parser.parse_args()

    show = not args.no_show

    print("\n" + "█"*70)
    print("  Layer 1 — Stochastic Simulation Core")
    print("  Rough Volatility: Theory and Numerical Implementation")
    print("  Project: RL as Numerical Approach to Stochastic Optimal Control")
    print("█"*70)

    sections = {
        1: section1_fbm,
        2: section2_convergence,
        3: section3_rough_models,
        4: section4_hurst,
    }

    if args.section:
        sections[args.section](show=show)
    else:
        for fn in sections.values():
            fn(show=show)

    print("\n" + "="*70)
    print("  All sections complete.  Figures in ./output/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
