"""
roughvol_core.py — Shared rough-path engine (validated foundation)
==================================================================
Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

This module holds the ONE trusted rough-volatility path generator that the
rest of the library imports. It is the κ=0 optimal-discretisation Volterra
scheme extracted verbatim from the validated Layer 1b pricing engine
(layer1b_mlmc_asian.py), whose Section 1 confirms:

  - Var(W̃_t) matches the discrete formula v_i (not the biased continuum
    t^{2H}), and
  - the lognormal variance process has E[V_t] = ξ₀ exactly at every grid
    point — no forward-variance drift bias.

Why this module exists
----------------------
Layer 1 (layer1_rough_vol.py) has a known normalisation bug (ROADMAP issue
L1-1): its fbm_hybrid produces Var(B^H_1) ≈ 0.89 while its rBergomi
compensator assumes 1.0. Any analysis built on that engine would measure the
bug, not the science. Layer 1c (the roughness-estimator audit) lives or dies
on its ground-truth paths being correct, so it imports from HERE, never from
Layer 1.

The accompanying test suite (test_roughvol_core.py) pins the correctness
properties so a future edit (or a clobber) that breaks the normalisation
fails CI instead of silently corrupting results.

Public API
----------
  volterra_weights(n, H, T)                  -> (g, v)
  volterra_process(dW, H, T)                 -> W̃ paths   (n_paths, n)
  rough_log_variance_paths(...)              -> (t, log_V) for H-estimation
  rough_bergomi_paths(...)                   -> (t, S, V)  asset + variance

References
----------
- Bennedsen, Lunde & Pakkanen (2017). Hybrid scheme for BSS processes.
- Bayer, Friz & Gatheral (2016). Pricing under rough volatility.
- Gatheral, Jaisson & Rosenbaum (2018). Volatility is rough.
"""

import numpy as np
from scipy.signal import fftconvolve

__all__ = [
    "volterra_weights",
    "volterra_process",
    "rough_log_variance_paths",
    "rough_bergomi_paths",
]


def volterra_weights(n: int, H: float, T: float = 1.0):
    """
    Convolution weights for the optimal-discretisation (κ = 0) hybrid scheme.

    Parameters
    ----------
    n : int      number of time steps on [0, T]
    H : float    Hurst exponent in (0, 1)
    T : float    horizon

    Returns
    -------
    g : np.ndarray (n,)
        Kernel g_m = (b_m·dt)^{H-1/2} with b_m the BLP optimal evaluation
        points, so that  W̃_i = sqrt(2H) · (g ∗ dW)_i.
    v : np.ndarray (n,)
        Discrete variance v_i = Var(W̃_{t_i}) = 2H·dt·cumsum(g²). Use this
        (not t^{2H}) in any lognormal compensator to keep E[V_t] = ξ₀ exact.
    """
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must be in (0,1), got {H}")
    a  = H - 0.5
    dt = T / n
    m  = np.arange(1, n + 1)
    if abs(a) < 1e-12:
        # H = 1/2: kernel exponent a -> 0 is a removable singularity. The
        # factor (b_m·dt)^a -> 1, so the kernel is flat — standard Brownian
        # motion. This is Layer 1c's smooth (Markovian) null hypothesis.
        g = np.ones(n)
    else:
        b = ((m**(a + 1) - (m - 1)**(a + 1)) / (a + 1)) ** (1.0 / a)
        g = (b * dt) ** a
    v  = 2.0 * H * dt * np.cumsum(g**2)
    return g, v


def volterra_process(dW: np.ndarray, H: float, T: float = 1.0) -> np.ndarray:
    """
    Build the Volterra (rough) process W̃ from pre-drawn Brownian increments.

    Parameters
    ----------
    dW : np.ndarray (n_paths, n)
        Brownian increments already scaled by sqrt(dt) (i.e. ~ N(0, dt)).
    H, T : as above. dt = T / n is inferred from dW's shape.

    Returns
    -------
    W̃ : np.ndarray (n_paths, n)
        W̃_{t_i} for i = 1..n, exact to the κ=0 scheme.
    """
    n = dW.shape[1]
    g, _ = volterra_weights(n, H, T)
    return np.sqrt(2.0 * H) * fftconvolve(dW, g[None, :], axes=1)[:, :n]


def rough_log_variance_paths(n: int, H: float, n_paths: int,
                             T: float = 1.0, eta: float = 1.0,
                             xi0: float = 0.04,
                             rng: np.random.Generator = None):
    """
    Simulate rough log-variance paths  log V_t  under the RFSV / rough
    Bergomi variance model — the ground-truth generator for Layer 1c.

        V_t = ξ₀ · exp( η·W̃_t − ½·η²·Var(W̃_t) )
    =>  log V_t = log ξ₀ + η·W̃_t − ½·η²·v_t

    By construction this has E[V_t] = ξ₀ (validated), and log V_t is an
    affine function of the rough process W̃_t, so its Hurst exponent equals
    H. This is exactly the object the GJR / Cont–Das estimators consume.

    Parameters
    ----------
    n        grid steps on [0, T]   (so n+1 points incl. t=0)
    H        true Hurst exponent — the quantity estimators must recover
    n_paths  number of independent paths
    eta      vol-of-vol (scales the log-variance fluctuations)
    xi0      base forward variance
    rng      optional np.random.Generator for reproducibility

    Returns
    -------
    t      : np.ndarray (n+1,)            time grid including 0
    log_V  : np.ndarray (n_paths, n+1)    log-variance, column 0 = log ξ₀
    """
    rng = rng or np.random.default_rng()
    dt  = T / n
    dW  = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    W   = volterra_process(dW, H, T)                 # (n_paths, n)
    _, v = volterra_weights(n, H, T)

    log_V = np.empty((n_paths, n + 1))
    log_V[:, 0]  = np.log(xi0)
    log_V[:, 1:] = np.log(xi0) + eta * W - 0.5 * eta**2 * v[None, :]
    t = np.linspace(0.0, T, n + 1)
    return t, log_V


def rough_bergomi_paths(n: int, H: float, n_paths: int, T: float = 1.0,
                        eta: float = 1.0, rho: float = -0.7,
                        xi0: float = 0.04, S0: float = 100.0, r: float = 0.0,
                        rng: np.random.Generator = None):
    """
    Full rough Bergomi asset + variance simulation (log-Euler asset).

    Returns
    -------
    t : (n+1,)                 time grid
    S : (n_paths, n+1)         asset price paths, S[:,0] = S0
    V : (n_paths, n+1)         instantaneous variance, V[:,0] = ξ₀

    Forward variance is exact (E[V_t] = ξ₀) by the discrete compensator.
    """
    rng = rng or np.random.default_rng()
    dt  = T / n
    dW1 = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    dW2 = rng.standard_normal((n_paths, n)) * np.sqrt(dt)

    W = volterra_process(dW1, H, T)
    _, v = volterra_weights(n, H, T)

    V = np.empty((n_paths, n + 1))
    V[:, 0]  = xi0
    V[:, 1:] = xi0 * np.exp(eta * W - 0.5 * eta**2 * v[None, :])

    V_left = V[:, :-1]                                # left endpoints
    dW_S   = rho * dW1 + np.sqrt(1.0 - rho**2) * dW2
    dlogS  = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * dW_S
    logS   = np.concatenate([np.zeros((n_paths, 1)),
                             np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    t = np.linspace(0.0, T, n + 1)
    return t, S, V
