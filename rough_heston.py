"""
rough_heston.py — native rough-Heston path simulator (Layer 4, brick 1)
=======================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

First brick of Layer 4. A rough-Heston (El Euch–Rosenbaum 2019) path
simulator that REUSES roughvol_core.py's κ=0 BLP Volterra kernel weights,
so the convergence study (layer4_convergence.py, NOT in this brick) measures
the *same* κ=0 discretisation already characterised at Layer 1b.

Model (variance is a rough / fractional CIR; the price is correlated):

    V_t = V0 + (1/Γ(H+½)) ∫₀ᵗ (t-s)^{H-½} κ(θ - V_s) ds
             + (1/Γ(H+½)) ∫₀ᵗ (t-s)^{H-½} ν √(V_s) dW_s^V
    dS_t = S_t √(V_t) dW_t^S ,   d⟨W^V, W^S⟩ = ρ dt

Discretised by an explicit Volterra–Euler recursion (V is NOT a closed-form
function of the increments — √V sits inside the integral — so it is stepped
forward, convolving the kernel against the running history each step; cost is
O(n²) per path-block, vs rough Bergomi's one-shot O(n log n) FFT).

Kernel reuse (the brick's premise — see §2 of the gate-check spec):
  roughvol_core.volterra_weights(n,H,T) -> g  with  g_m = (b_m·dt)^{H-½}
  (the BLP κ=0 optimal-point discretisation of (t-s)^{H-½}).  We reuse that
  g VECTOR and apply the rough-Heston normalisation ourselves:
      K_m = g_m / Γ(H+½)  =  K(b_m·dt),   K(τ) = τ^{H-½}/Γ(H+½)
  i.e. the kernel evaluated at the κ=0 optimal point of lag-cell m.  We do NOT
  call volterra_process (that bakes in rough Bergomi's √(2H) normalisation).
  The H=½ limit and the E[V_t]=θ test validate this renormalisation.

# ====================================================================== #
# ⚠️  POSITIVITY SCHEME & VALIDATED RANGE — ν ≤ 0.20 (chosen on evidence) #
# ---------------------------------------------------------------------- #
# positivity= selects how the variance is kept non-negative:             #
#   "qe"          Andersen Quadratic-Exponential on the conditionally-    #
#                 Gaussian step — THE DEFAULT and validated scheme.       #
#   "truncation"  full truncation V⁺ = max(V,0) — cheapest; OK only where #
#                 truncation is rare (low ν / large H).                   #
#   "reflection"  |V| — REJECTED (inflates E[V] +158% at ν=0.4; β worst). #
#                                                                        #
# VALIDATED RANGE — ν ≤ 0.20 (measured, rh_beta_gate.py):                 #
#   • QE reproduces β = 2H consistent with layer1b across                 #
#     H ∈ {.05,.10,.20,.35} for ν ≤ 0.20 (max|β−2H| ≤ 0.04). The β gate   #
#     PASSES there; this is the brick's acceptance regime.                #
#                                                                        #
# BEYOND ν ≈ 0.25 THE EXPLICIT SCHEME DEGRADES — both axes together, as   #
# V→0 events exceed ~10% of samples:                                      #
#   • β COLLAPSES: the V=0 clip/branch fires at DIFFERENT times on the    #
#     fine vs coarse MLMC grids → the coupling breaks. (Coupling is a     #
#     β-gate-only construct; the weak-order study does not use it.)       #
#   • the PRICED bias becomes SCHEME- and n-DEPENDENT (qe vs truncation   #
#     differ 2.4–5.3% at ν=0.4, halving as n doubles) → the weak-order    #
#     study is NOT trustworthy there either.                              #
#                                                                        #
# => This explicit hybrid Volterra–Euler simulator is VALIDATED FOR       #
#    ν ≤ 0.20 ONLY. The high-ν / SPX-calibrated regime (ν ≈ 0.3–0.4)      #
#    requires a MULTIFACTOR MARKOVIAN-LIFT simulator (Abi Jaber–El Euch), #
#    scoped as a separate Layer 4 EXTENSION — see the gate-check spec §5. #
# ====================================================================== #

References
----------
- El Euch & Rosenbaum (2019). The characteristic function of rough Heston
  models. Mathematical Finance.
- Bennedsen, Lunde & Pakkanen (2017). Hybrid scheme for BSS processes.
- Lord, Koekkoek & van Dijk (2010). A comparison of biased simulation
  schemes for stochastic volatility models. (full truncation = least biased
  of the simple fixes; revisit per the caveat above.)
"""

import numpy as np
from scipy.special import gamma as _gamma, erf as _erf

from roughvol_core import volterra_weights

__all__ = ["rough_heston_paths", "PARAMS"]

# Default parameters. ν is the Heston vol-of-vol (≠ rough Bergomi's η); κ, θ are
# new vs Bergomi. V0 = θ so E[V_t] = θ exactly in the continuum (forward-variance
# anchor). ν = 0.20 is the VALIDATED CEILING (see the positivity note below): QE
# reproduces β = 2H for ν ≤ 0.20; beyond ν ≈ 0.25 the explicit scheme degrades.
PARAMS = dict(T=1.0, V0=0.04, kappa=0.3, theta=0.04, nu=0.20, rho=-0.70,
              S0=100.0, r=0.0)


def _rh_kernel(n: int, H: float, T: float):
    """Rough-Heston κ=0 kernel weights K_m = g_m / Γ(H+½), reusing the BLP
    optimal-point weights g from roughvol_core (see module docstring §2)."""
    g, _ = volterra_weights(n, H, T)          # g_m = (b_m·dt)^{H-½}
    return g / _gamma(H + 0.5)                 # K_m = K(b_m·dt)


def _qe_map(m, s2, Z, psi_c=1.5):
    """Andersen (2008) Quadratic-Exponential positivity map for one step whose
    conditional law given history is Gaussian N(m, s2). Returns a non-negative
    V that matches (m, s2) and is a deterministic, coupling-preserving function
    of the driving standard normal Z (so MLMC fine/coarse stay coupled). Where
    the conditional mean m ≤ 0 the step is set to 0."""
    out = np.zeros_like(m, dtype=float)
    pos = m > 1e-14
    if not pos.any():
        return out
    # deterministic step where conditional variance ≈ 0 (V_{i-1}=0): V = m
    det = pos & (s2 <= 1e-300)
    out[det] = m[det]
    stoch = pos & ~det
    if not stoch.any():
        return out
    mp, s2p, Zp = m[stoch], s2[stoch], Z[stoch]
    psi = s2p / (mp * mp)
    res = np.empty_like(mp)
    quad = psi <= psi_c                                   # quadratic branch
    if quad.any():
        invpsi = 2.0 / psi[quad]
        b2 = invpsi - 1.0 + np.sqrt(invpsi) * np.sqrt(np.maximum(invpsi - 1.0, 0.0))
        a = mp[quad] / (1.0 + b2)
        res[quad] = a * (np.sqrt(b2) + Zp[quad]) ** 2
    ex = ~quad                                            # exponential branch
    if ex.any():
        psie = psi[ex]
        pe = (psie - 1.0) / (psie + 1.0)
        beta = (1.0 - pe) / mp[ex]
        U = np.clip(0.5 * (1.0 + _erf(Zp[ex] / np.sqrt(2.0))), 0.0, 1.0 - 1e-12)
        res[ex] = np.where(U <= pe, 0.0, np.log((1.0 - pe) / (1.0 - U)) / beta)
    out[stoch] = np.maximum(res, 0.0)
    return out


def _rough_heston_from_increments(dWV, dW_perp, n, H, p, positivity="qe"):
    """Core Volterra–Euler recursion from pre-drawn Brownian increments.

    Parameters
    ----------
    dWV, dW_perp : np.ndarray (n_paths, n)
        Independent Brownian increments ~ N(0, dt). dWV drives the variance;
        the correlated asset BM is dW^S = ρ·dWV + √(1-ρ²)·dW_perp.
        (Taking increments as input lets the β harness drive coupled
        fine/coarse paths from the SAME Brownian motion — coarse increments
        are the pairwise sums of fine.)
    n, H : grid steps, Hurst exponent. dt = p["T"]/n.
    p : params dict (PARAMS shape).

    Returns
    -------
    S, V : np.ndarray (n_paths, n+1)   asset and variance, col 0 = S0, V0.
    """
    T = p["T"]; dt = T / n
    V0 = p["V0"]; kappa = p["kappa"]; theta = p["theta"]
    nu = p["nu"]; rho = p["rho"]; S0 = p["S0"]; r = p["r"]

    K = _rh_kernel(n, H, T)                    # (n,)  K[m-1] = lag-m weight
    n_paths = dWV.shape[0]

    # ---- variance path: explicit Volterra–Euler, full-truncation V⁺ ----
    V = np.empty((n_paths, n + 1))
    V[:, 0] = V0
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n + 1):
        # cells j = 0..i-1 are at lag m = i-j (nearest cell j=i-1 -> m=1 -> K[0]);
        # weight slice in cell order is K[i-1], K[i-2], ..., K[0].
        w     = K[i - 1::-1]                    # (i,)
        Vp    = V[:, :i]                        # stored values (≥0 for all schemes)
        Vsqrt = np.sqrt(np.maximum(Vp, 0.0))    # √V⁺ for the coefficients
        drift = (kappa * (theta - Vp) * w).sum(axis=1)             # Lebesgue: × dt
        diff  = (nu * Vsqrt * dWV[:, :i] * w).sum(axis=1)          # Itô: dW carries √dt
        raw   = V0 + dt * drift + diff          # conditionally-Gaussian raw step
        # ---- POSITIVITY (selectable; chosen on EVIDENCE — see rh_beta_gate) ----
        if positivity == "truncation":          # full truncation V⁺ = max(V, 0)
            V[:, i] = np.maximum(raw, 0.0)
        elif positivity == "reflection":        # |V|
            V[:, i] = np.abs(raw)
        elif positivity == "qe":                # Andersen QE on the cond.-Gaussian step
            s_lag1 = K[0] * nu * Vsqrt[:, -1]                      # = K₀·ν·√V_{i-1}
            m  = raw - s_lag1 * dWV[:, i - 1]                      # E[V_i | F_{t_{i-1}}]
            s2 = (s_lag1 ** 2) * dt                                # Var[V_i | F_{t_{i-1}}]
            V[:, i] = _qe_map(m, s2, dWV[:, i - 1] / sqrt_dt)
        else:
            raise ValueError(f"unknown positivity scheme: {positivity!r}")

    # ---- asset path: log-Euler, left-point V⁺, correlated BM ----
    Vp_left = np.maximum(V[:, :-1], 0.0)        # (n_paths, n)
    dW_S = rho * dWV + np.sqrt(1.0 - rho ** 2) * dW_perp
    dlogS = (r - 0.5 * Vp_left) * dt + np.sqrt(Vp_left) * dW_S
    logS = np.concatenate([np.zeros((n_paths, 1)),
                           np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    return S, V


def rough_heston_paths(n: int, H: float, n_paths: int,
                       rng: np.random.Generator = None,
                       positivity: str = "qe", **overrides):
    """Simulate rough-Heston (S, V) paths on n steps of [0, T].

    Parameters
    ----------
    n, H, n_paths : grid steps, Hurst exponent, number of paths.
    rng : optional np.random.Generator (for reproducibility).
    **overrides : any of PARAMS (T, V0, kappa, theta, nu, rho, S0, r).

    Returns
    -------
    t : np.ndarray (n+1,)            time grid incl. 0
    S : np.ndarray (n_paths, n+1)    asset, S[:,0] = S0
    V : np.ndarray (n_paths, n+1)    variance, V[:,0] = V0  (V⁺, ≥ 0)
    """
    p = {**PARAMS, **overrides}
    rng = rng or np.random.default_rng()
    dt = p["T"] / n
    dWV    = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    dW_perp = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    S, V = _rough_heston_from_increments(dWV, dW_perp, n, H, p, positivity=positivity)
    t = np.linspace(0.0, p["T"], n + 1)
    return t, S, V


if __name__ == "__main__":
    # smoke run only (not a gate): confirm it executes and is well-formed.
    t, S, V = rough_heston_paths(n=128, H=0.10, n_paths=8,
                                 rng=np.random.default_rng(0))
    print(f"shapes  t{t.shape} S{S.shape} V{V.shape}")
    print(f"V>=0    {(V >= 0).all()}   finite {np.isfinite(V).all()}")
    print(f"E[V_T]  {V[:, -1].mean():.5f}  (theta={PARAMS['theta']})")
    print(f"E[S_T]  {S[:, -1].mean():.3f}  (S0={PARAMS['S0']})")
