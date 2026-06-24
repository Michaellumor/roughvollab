"""
layer1b_kappa1.py — the kappa=1 BLP hybrid Volterra scheme: FINE path + the
exact MLMC COARSE COUPLER (split + conditional resampling).

Separate module: the production engine (layer1b_mlmc_asian.py) keeps its kappa=0
default untouched.  Adoption as the project default is a LATER decision.

Fine path (design doc §2):
    W~_{t_i} = sqrt(2H) [ W_{i,1} + sum_{m=2}^i g_m dW_{i-m+1} ],
    W_{i,1}  = beta1 dW_i + sig1 Z_i,   (the exact nearest-cell integral)
    v_i^{k1} = dt^{2H} + 2H dt sum_{m=2}^i g_m^2   (exact compensator variance).

Coarse coupler (design doc §4, with the §9 index fix the reviewer caught):
    for 0-indexed arrays a coarse cell j spans fine cells [2j, 2j+1];
    W1diag_c[j] = I1[j] + W1diag_f[2j+1]            (reuse the RIGHT sub-cell)
    I1[j]       = beta . (dW_f[2j], W1diag_f[2j]) + sig_c Z_c[j]   (cond. on LEFT)
with (beta, sig_c) from the per-cell 3x3 covariance of (dW, W1, I1) — validated
to 1e-11 in kappa1_coupling_design_check.py.  Reusing the WRONG sub-cell passes
a marginal check but destroys the coupling; the gate asserts coupling tightness.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import quad

from layer1b_mlmc_asian import (PARAMS, volterra_weights,
                                volterra_weights_kappa1)


# ── weights and per-cell constants ───────────────────────────────────────────
# volterra_weights_kappa1 now lives in layer1b_mlmc_asian.py (single source of
# truth, returning (g_hyb, v_k1, c_near, sig_perp)); imported above.


def coarse_coupling_params(H, h):
    """I1 | (dW_left, W1_left) ~ N(beta . x, sig_c^2) on a sub-cell of length h.
    Returns beta (2-vector), sig_c.  (h = the FINE cell length dt_f.)"""
    a = H - 0.5
    C = quad(lambda w: w**a * (1.0 + w)**a, 0.0, 1.0)[0]
    v_dW = h
    v_W1 = h**(2 * H) / (2 * H)
    v_I1 = h**(2 * H) * (2**(2 * H) - 1.0) / (2 * H)
    c_dW_W1 = h**(H + 0.5) / (H + 0.5)
    c_dW_I1 = h**(H + 0.5) * (2**(H + 0.5) - 1.0) / (H + 0.5)
    c_W1_I1 = h**(2 * H) * C
    Sxx = np.array([[v_dW, c_dW_W1], [c_dW_W1, v_W1]])
    SxI = np.array([c_dW_I1, c_W1_I1])
    beta = np.linalg.solve(Sxx, SxI)
    sig_c = np.sqrt(max(v_I1 - SxI @ beta, 0.0))
    return beta, sig_c


# ── path builders driven by an explicit W1diag (fine OR coarse) ───────────────
def _wtilde_from_W1diag(dW1, W1diag, n, p):
    """W~ = sqrt(2H)(W1diag + Riemann tail conv); returns (W~, v_k1)."""
    H = p["H"]
    g_hyb, v_k1, _, _ = volterra_weights_kappa1(n, p["H"], p["T"])
    conv = fftconvolve(dW1, g_hyb[None, :], axes=1)[:, :n]
    return np.sqrt(2.0 * H) * (W1diag + conv), v_k1


def _asian_from_W1diag(dW1, W1diag, dW2, n, p):
    """Discounted arithmetic-Asian payoff for a kappa=1 path with given W1diag."""
    eta, rho = p["eta"], p["rho"]
    xi0, S0, K, T, r = p["xi0"], p["S0"], p["K"], p["T"], p["r"]
    dt = T / n
    W_tilde, v_k1 = _wtilde_from_W1diag(dW1, W1diag, n, p)
    V_left = np.empty_like(dW1)
    V_left[:, 0] = xi0
    V_left[:, 1:] = xi0 * np.exp(eta * W_tilde[:, :-1]
                                 - 0.5 * eta**2 * v_k1[None, :-1])
    dW_S = rho * dW1 + np.sqrt(1.0 - rho**2) * dW2
    dlogS = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * dW_S
    logS = np.concatenate([np.zeros((dW1.shape[0], 1)),
                           np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    A = (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1]) / n
    return np.exp(-r * T) * np.maximum(A - K, 0.0)


# ── kept for the G-H1 fine-path gate (backward-compatible names) ──────────────
def volterra_kappa1(dW1, Z, n, p):
    _, _, beta1, sig1 = volterra_weights_kappa1(n, p["H"], p["T"])
    return _wtilde_from_W1diag(dW1, beta1 * dW1 + sig1 * Z, n, p)


def kappa1_payoff(dW1, Z, dW2, n, p, payoff="asian"):
    _, _, beta1, sig1 = volterra_weights_kappa1(n, p["H"], p["T"])
    W1diag = beta1 * dW1 + sig1 * Z
    if payoff == "european":
        eta, rho = p["eta"], p["rho"]
        xi0, S0, K, T, r = p["xi0"], p["S0"], p["K"], p["T"], p["r"]
        dt = T / n
        W_tilde, v_k1 = _wtilde_from_W1diag(dW1, W1diag, n, p)
        V_left = np.empty_like(dW1)
        V_left[:, 0] = xi0
        V_left[:, 1:] = xi0 * np.exp(eta * W_tilde[:, :-1]
                                     - 0.5 * eta**2 * v_k1[None, :-1])
        dW_S = rho * dW1 + np.sqrt(1.0 - rho**2) * dW2
        dlogS = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * dW_S
        ST = S0 * np.exp(dlogS.sum(axis=1))
        return np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return _asian_from_W1diag(dW1, W1diag, dW2, n, p)


def draw_increments(N, n, p, rng):
    dt = p["T"] / n
    dW1 = rng.standard_normal((N, n)) * np.sqrt(dt)
    dW2 = rng.standard_normal((N, n)) * np.sqrt(dt)
    Z = rng.standard_normal((N, n))
    return dW1, Z, dW2


# ── the coupled MLMC level estimator ─────────────────────────────────────────
def mlmc_level_kappa1(l, N, p=PARAMS, batch=5000, rng=None, swap_bug=False):
    """N coupled samples of Y_l = P_f - P_c under the kappa=1 hybrid scheme.

    swap_bug=True deliberately reuses the WRONG sub-cell (conditions I1 on the
    right, reuses the left) — for the gate to show coupling-tightness catches it.
    Returns out[0]=Y_l, out[1]=P_f.
    """
    rng = rng or np.random.default_rng()
    H, T, n0 = p["H"], p["T"], p["n0"]
    n_f = n0 * 2**l
    dt_f = T / n_f
    batch = max(200, min(batch, 2_560_000 // n_f))
    _, _, beta1_f, sig1_f = volterra_weights_kappa1(n_f, H, T)
    if l > 0:
        n_c = n_f // 2
        beta_cc, sig_cc = coarse_coupling_params(H, dt_f)   # h = fine cell length
    out = np.empty((2, N))
    done = 0
    while done < N:
        nb = min(batch, N - done)
        dW1 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
        dW2 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
        Zf = rng.standard_normal((nb, n_f))
        W1diag_f = beta1_f * dW1 + sig1_f * Zf
        P_f = _asian_from_W1diag(dW1, W1diag_f, dW2, n_f, p)
        if l == 0:
            Y = P_f
        else:
            dW1_c = dW1.reshape(nb, n_c, 2).sum(axis=2)
            dW2_c = dW2.reshape(nb, n_c, 2).sum(axis=2)
            d1 = dW1.reshape(nb, n_c, 2)
            w1 = W1diag_f.reshape(nb, n_c, 2)
            Zc = rng.standard_normal((nb, n_c))
            if not swap_bug:
                # CORRECT: condition I1 on the LEFT sub-cell (2j), reuse RIGHT (2j+1)
                I1 = beta_cc[0] * d1[:, :, 0] + beta_cc[1] * w1[:, :, 0] + sig_cc * Zc
                W1diag_c = I1 + w1[:, :, 1]
            else:
                # WRONG: condition on the RIGHT, reuse the LEFT (anchor mismatch)
                I1 = beta_cc[0] * d1[:, :, 1] + beta_cc[1] * w1[:, :, 1] + sig_cc * Zc
                W1diag_c = I1 + w1[:, :, 0]
            P_c = _asian_from_W1diag(dW1_c, W1diag_c, dW2_c, n_c, p)
            Y = P_f - P_c
        out[0, done:done + nb] = Y
        out[1, done:done + nb] = P_f
        done += nb
    return out
