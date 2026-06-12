"""
Layer 1b — Multilevel Monte Carlo Pricing under Rough Volatility
=================================================================
Rough Volatility: Theory and Numerical Implementation

Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions

Structure
---------
  Section 1  Coupled rough Bergomi engine + correctness validation
  Section 2  Level statistics: weak rate α, variance rate β, cost rate γ
  Section 3  Adaptive MLMC driver (Giles 2008) vs standard Monte Carlo
  Section 4  Complexity under roughness: β as a function of H

Each section prints its results and saves a figure to ./output/.
Run the whole file with:  python layer1b_mlmc_asian.py
Or run one section:       python layer1b_mlmc_asian.py --section 2
Reduced-cost run:         python layer1b_mlmc_asian.py --quick

Mathematical background
-----------------------
We price an arithmetic Asian call under the rough Bergomi model
(Bayer-Friz-Gatheral 2016):

    dS_t = S_t sqrt(V_t) dW_t^S
    V_t  = xi0 * exp( eta * W~_t  -  (eta^2 / 2) * Var(W~_t) )
    W~_t = sqrt(2H) * INT_0^t (t-s)^{H-1/2} dW_s        (Volterra process)

with Corr(dW^S, dW) = rho.  The payoff is

    P = exp(-rT) * max( A - K, 0 ),   A = trapezoidal average of S on [0, T].

Multilevel Monte Carlo (Giles 2008) writes

    E[P^L] = E[P^0] + SUM_{l=1}^{L} E[P^l - P^{l-1}]

where P^l is the payoff computed on a grid with n_l = n_0 * 2^l steps, and
estimates each correction term with *coupled* fine/coarse paths driven by the
same Brownian increments.  If

    |E[P^l - P]|  ~  c1 * 2^{-alpha l}        (weak rate)
    Var[P^l - P^{l-1}] ~ c2 * 2^{-beta  l}    (level variance rate)
    Cost_l        ~  c3 * 2^{ gamma l}        (cost rate, here gamma = 1)

then the total cost to reach RMS accuracy eps is (Giles' complexity theorem):

    beta >  gamma :  O( eps^{-2} )
    beta == gamma :  O( eps^{-2} log(eps)^2 )
    beta <  gamma :  O( eps^{-2 - (gamma - beta)/alpha} )

Why rough volatility is the interesting (hard) case: the hybrid /
optimal-discretisation schemes for the Volterra process have strong rate
O(n^{-H}) (Bennedsen-Lunde-Pakkanen 2017), which suggests the pessimistic
bound beta >= 2H — for H ~ 0.1 that is beta ~ 0.2 << gamma = 1, the worst
Giles regime.  Whether the integral-average payoff buys a better rate than this
pathwise bound is exactly what Sections 2 and 4 measure.  Our runs say no:
with the kappa = 0 coupling, the measured beta tracks 2H closely — the bound
is tight, because the Volterra strong error acts as a slowly-decaying common
factor across the path that the time-average cannot cancel.  That negative
result is the scientific point: it quantifies why naive MLMC struggles under
rough volatility and motivates the antithetic / conditional-MC couplings
listed as Layer 1b extensions in ROADMAP.md.

Numerical scheme and coupling
-----------------------------
The Volterra process is simulated with the optimal-discretisation Riemann
scheme (the kappa = 0 member of the BLP hybrid family):

    W~_{t_i} = sqrt(2H) * SUM_{j=1}^{i} g_{i-j+1} * dW_j,
    g_m      = (b_m * dt)^{H - 1/2},
    b_m      = [ (m^{a+1} - (m-1)^{a+1}) / (a+1) ]^{1/a},   a = H - 1/2,

i.e. a discrete convolution, evaluated with FFT in O(n log n) per path-block.
kappa = 0 is chosen deliberately: the process is a *pure function of the
Brownian increments dW*, so the MLMC coupling is exact — the coarse path is
generated from the pairwise-summed fine increments, and the telescoping
identity holds by construction.  (The kappa = 1 scheme has a better error
constant but requires a second, correlated Gaussian per cell whose coarse-
level counterpart is not a simple function of the fine variables; it is
listed as a refinement in ROADMAP.md.)

The lognormal compensator uses the *discrete* variance

    v_i = 2H * dt * SUM_{m<=i} g_m^2

rather than the continuum limit t_i^{2H}.  This makes E[V_{t_i}] = xi0 hold
exactly at every level and every grid point (validated in Section 1), so the
forward-variance curve carries no discretisation bias — all level bias lives
in the asset discretisation and the payoff average, which is what MLMC is
built to handle.

References
----------
- Giles (2008). Multilevel Monte Carlo path simulation. Operations Research.
- Bayer, Friz & Gatheral (2016). Pricing under rough volatility. Quant. Finance.
- Bennedsen, Lunde & Pakkanen (2017). Hybrid scheme for Brownian
  semistationary processes. Finance and Stochastics.
- McCrickerd & Pakkanen (2018). Turbocharging Monte Carlo pricing for the
  rough Bergomi model. Quantitative Finance.
"""

import argparse
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import norm

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

# ── model / contract defaults ───────────────────────────────────────────────
# Bayer-Friz-Gatheral SPX calibration is (H, eta, rho) ~ (0.07, 1.9, -0.9);
# slightly tamer defaults are used here so estimator variances stay modest
# on a laptop.  Override via PARAMS or the CLI of your own experiments.
PARAMS = dict(
    H    = 0.10,     # Hurst exponent (rough regime)
    eta  = 1.50,     # vol-of-vol
    rho  = -0.70,    # leverage correlation
    xi0  = 0.04,     # flat forward variance (20% vol)
    S0   = 100.0,
    K    = 100.0,
    T    = 1.0,
    r    = 0.0,
    n0   = 32,       # coarsest grid: n_0 steps on [0, T]
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Coupled rough Bergomi engine + correctness validation
# ══════════════════════════════════════════════════════════════════════════════

def volterra_weights(n: int, H: float, T: float) -> tuple:
    """
    Convolution weights for the optimal-discretisation (kappa = 0) scheme.

    Returns
    -------
    g : np.ndarray, shape (n,)
        Kernel values g_m = (b_m dt)^{H-1/2} with b_m the BLP optimal
        evaluation points, so that  W~_i = sqrt(2H) * (g * dW)_i.
    v : np.ndarray, shape (n,)
        Discrete variance  v_i = Var(W~_{t_i}) = 2H dt cumsum(g^2),
        used in the lognormal compensator (exact forward variance).
    """
    a  = H - 0.5
    dt = T / n
    m  = np.arange(1, n + 1)
    b  = ((m**(a + 1) - (m - 1)**(a + 1)) / (a + 1)) ** (1.0 / a)
    g  = (b * dt) ** a
    v  = 2.0 * H * dt * np.cumsum(g**2)
    return g, v


def _paths_from_increments(dW1, dW2, n, p, payoff="asian"):
    """
    Rough Bergomi payoff from given Brownian increments (one grid level).

    dW1 : (N, n) increments driving the Volterra process  (scaled sqrt(dt))
    dW2 : (N, n) orthogonal increments for the asset
    Returns the discounted payoff per path, shape (N,).
    """
    H, eta, rho = p["H"], p["eta"], p["rho"]
    xi0, S0, K, T, r = p["xi0"], p["S0"], p["K"], p["T"], p["r"]
    dt = T / n

    g, v = volterra_weights(n, H, T)
    # Volterra process at t_1..t_n via FFT convolution along the time axis
    W_tilde = np.sqrt(2.0 * H) * fftconvolve(dW1, g[None, :], axes=1)[:, :n]

    # variance at left endpoints t_0..t_{n-1}; exact forward variance E[V]=xi0
    V_left = np.empty_like(dW1)
    V_left[:, 0]  = xi0
    V_left[:, 1:] = xi0 * np.exp(eta * W_tilde[:, :-1]
                                 - 0.5 * eta**2 * v[None, :-1])

    # log-Euler asset increments with correlated driver
    dW_S  = rho * dW1 + np.sqrt(1.0 - rho**2) * dW2
    dlogS = (r - 0.5 * V_left) * dt + np.sqrt(V_left) * dW_S
    logS  = np.concatenate([np.zeros((dW1.shape[0], 1)),
                            np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)                                # (N, n+1) incl. S0

    if payoff == "european":
        return np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0)
    # trapezoidal arithmetic average on the level's own grid
    A = (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1]) / n
    return np.exp(-r * T) * np.maximum(A - K, 0.0)


def mlmc_asian_level(l: int, N: int, p: dict = PARAMS,
                     payoff: str = "asian", batch: int = 5000,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Draw N coupled samples of Y_l = P_f - P_c at level l (Y_0 = P_0).

    Fine grid n_f = n0 * 2^l; coarse grid n_c = n_f / 2.  Both levels are
    deterministic functions of the SAME fine Brownian increments, with the
    coarse increments obtained by pairwise summation — exact MLMC coupling.

    Returns
    -------
    out : np.ndarray, shape (2, N)
        out[0] = Y_l samples (the MLMC correction),
        out[1] = P_f samples (fine-level payoff, for consistency checks).
    """
    rng = rng or np.random.default_rng()
    n_f  = p["n0"] * 2**l
    dt_f = p["T"] / n_f
    # cap batch x n_f so peak memory stays flat as levels deepen
    batch = max(200, min(batch, 2_560_000 // n_f))
    out  = np.empty((2, N))

    done = 0
    while done < N:
        nb  = min(batch, N - done)
        dW1 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)
        dW2 = rng.standard_normal((nb, n_f)) * np.sqrt(dt_f)

        P_f = _paths_from_increments(dW1, dW2, n_f, p, payoff)
        if l == 0:
            Y = P_f
        else:
            n_c    = n_f // 2
            dW1_c  = dW1.reshape(nb, n_c, 2).sum(axis=2)
            dW2_c  = dW2.reshape(nb, n_c, 2).sum(axis=2)
            P_c    = _paths_from_increments(dW1_c, dW2_c, n_c, p, payoff)
            Y = P_f - P_c

        out[0, done:done + nb] = Y
        out[1, done:done + nb] = P_f
        done += nb
    return out


def _bs_call(S0, K, T, r, sigma):
    """Black-Scholes European call (validation anchor for eta -> 0)."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def section1_validation(show: bool = True, quick: bool = False):
    """
    Three correctness anchors before any pricing is trusted:

      (a) Var(W~_t) matches the discrete formula v_i and approaches t^{2H};
      (b) E[V_t] = xi0 exactly, at every grid point (forward-variance check) —
          this is the property Layer 1's rBergomi violates (~18% bias at
          eta = 1.9 from a mis-normalised Volterra process);
      (c) with eta = 0 the model degenerates to Black-Scholes, and the
          European-payoff price must match the closed form.
    """
    print("\n" + "─" * 70)
    print("  SECTION 1 — Engine validation")
    print("─" * 70)
    rng = np.random.default_rng(1)
    p = dict(PARAMS)
    N = 40_000 if quick else 100_000
    n = 256

    # (a)+(b): simulate Volterra + variance directly
    dW1 = rng.standard_normal((N, n)) * np.sqrt(p["T"] / n)
    g, v = volterra_weights(n, p["H"], p["T"])
    W = np.sqrt(2 * p["H"]) * fftconvolve(dW1, g[None, :], axes=1)[:, :n]
    V = p["xi0"] * np.exp(p["eta"] * W - 0.5 * p["eta"]**2 * v[None, :])

    t = np.linspace(p["T"] / n, p["T"], n)
    var_emp = W.var(axis=0)
    fwd_err = np.abs(V.mean(axis=0) / p["xi0"] - 1.0)
    print(f"  (a) Var(W~_T): empirical {var_emp[-1]:.4f} | discrete v_n "
          f"{v[-1]:.4f} | continuum T^2H {p['T']**(2*p['H']):.4f}")
    print(f"  (b) forward-variance check  max_t |E[V_t]/xi0 - 1| "
          f"= {fwd_err.max():.4f}   (MC noise ~ "
          f"{3*np.exp(p['eta']**2*v[-1]/2)/np.sqrt(N):.4f})")

    # (c): eta = 0, European payoff vs Black-Scholes
    p0 = dict(p, eta=0.0)
    samp = mlmc_asian_level(3, N // 2, p0, payoff="european",
                            rng=np.random.default_rng(2))
    mc, se = samp[1].mean(), samp[1].std() / np.sqrt(N // 2)
    bs = _bs_call(p["S0"], p["K"], p["T"], p["r"], np.sqrt(p["xi0"]))
    z  = abs(mc - bs) / se
    print(f"  (c) eta=0 European: MC {mc:.4f} ± {se:.4f} | BS {bs:.4f} "
          f"| z = {z:.2f}  {'OK' if z < 3 else '** FAIL **'}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, var_emp, color=TEAL, lw=2, label="empirical Var(W~_t)")
    ax[0].plot(t, v, "--", color=CORAL, lw=1.5, label="discrete v_i (used)")
    ax[0].plot(t, t**(2 * p["H"]), ":", color=GRAY, lw=1.5,
               label=r"continuum $t^{2H}$")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("variance")
    ax[0].set_title("Volterra normalisation"); ax[0].legend(frameon=False)

    ax[1].plot(t, V.mean(axis=0) / p["xi0"], color=PURPLE, lw=2)
    ax[1].axhline(1.0, color=GRAY, ls="--", lw=1)
    ax[1].set_ylim(0.95, 1.05)
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$E[V_t]\,/\,\xi_0$")
    ax[1].set_title("Forward variance is exact (no drift bias)")

    fig.suptitle("Layer 1b §1 — engine validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/layer1b_validation.png", dpi=150)
    if show: plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Level statistics: estimating alpha, beta, gamma
# ══════════════════════════════════════════════════════════════════════════════

def estimate_rates(L: int = 6, N: int = 20_000, p: dict = PARAMS,
                   payoff: str = "asian", seed: int = 7,
                   verbose: bool = True) -> dict:
    """
    Giles-style mlmc_test: fixed N samples on each level l = 0..L, then

        alpha  from regression of log2 |E[Y_l]|   on l   (weak rate)
        beta   from regression of log2 Var[Y_l]   on l   (variance rate)
        gamma  from regression of log2 Cost_l     on l   (cost rate)

    Also runs the telescoping consistency check
        a_l - a_{l-1} - m_l  ≈ 0   within Monte Carlo noise,
    where a_l = E[P_f^l]; a coupling bug shows up here immediately.
    """
    rng = np.random.default_rng(seed)
    m_l, v_l, a_l, vf_l, c_l, chk = [], [], [], [], [], []

    for l in range(L + 1):
        t0 = time.time()
        s = mlmc_asian_level(l, N, p, payoff, rng=rng)
        wall = time.time() - t0
        Y, Pf = s[0], s[1]
        m_l.append(Y.mean());  v_l.append(Y.var())
        a_l.append(Pf.mean()); vf_l.append(Pf.var())
        c_l.append(p["n0"] * 2**l * (1.5 if l else 1.0))   # fine + coarse work
        if l:
            num = a_l[l] - a_l[l - 1] - m_l[l]
            den = 3 * (np.sqrt(vf_l[l]) + np.sqrt(vf_l[l - 1])
                       + np.sqrt(v_l[l])) / np.sqrt(N)
            chk.append(abs(num) / den)
        if verbose:
            se = np.sqrt(v_l[-1] / N)
            print(f"  l={l}  n={p['n0']*2**l:5d}  "
                  f"E[Y]={m_l[-1]:+.5f} (se {se:.5f})  "
                  f"V[Y]={v_l[-1]:.5f}  E[P]={a_l[-1]:.4f}  ({wall:.1f}s)")

    m, v = np.abs(m_l), np.array(v_l)
    ls = np.arange(1, L + 1)
    alpha = -np.polyfit(ls, np.log2(m[1:]), 1)[0]
    beta  = -np.polyfit(ls, np.log2(v[1:]), 1)[0]
    gamma =  np.polyfit(np.arange(L + 1), np.log2(c_l), 1)[0]

    if verbose:
        print(f"\n  consistency check (should be < 1): max "
              f"{max(chk):.3f}")
        ses = np.sqrt(v[1:] / N)
        if (m[1:] < 2 * ses).all():
            print("  ** |E[Y_l]| below 2 s.e. on every level: the alpha "
                  "regression is noise-dominated — increase N **")
        print(f"  alpha = {alpha:.3f}   beta = {beta:.3f}   "
              f"gamma = {gamma:.3f}")
        reg = ("beta > gamma  =>  O(eps^-2)" if beta > gamma else
               "beta = gamma  =>  O(eps^-2 log^2 eps)" if abs(beta-gamma) < .05
               else f"beta < gamma  =>  O(eps^-(2+{(gamma-beta):.2f}/alpha))")
        print(f"  Giles regime: {reg}")

    return dict(m=np.array(m_l), v=v, a=np.array(a_l), c=np.array(c_l),
                vP=vf_l[-1], alpha=alpha, beta=beta, gamma=gamma,
                L=L, N=N, consistency=max(chk))


def section2_rates(show: bool = True, quick: bool = False):
    print("\n" + "─" * 70)
    print("  SECTION 2 — Level statistics and convergence rates")
    print("─" * 70)
    L = 5 if quick else 6
    N = 10_000 if quick else 20_000
    res = estimate_rates(L=L, N=N)

    ls = np.arange(res["L"] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(ls, np.log2(res["v"]), "o-", color=TEAL, lw=2,
               label=r"$\log_2 V[Y_\ell]$")
    ax[0].plot(ls[1:], np.log2(res["v"][1]) - res["beta"] * (ls[1:] - 1),
               "--", color=GRAY, lw=1.2,
               label=fr"slope $-\beta = {-res['beta']:.2f}$")
    ax[0].set_xlabel(r"level $\ell$"); ax[0].set_ylabel(r"$\log_2$ variance")
    ax[0].set_title("Level variance decay"); ax[0].legend(frameon=False)

    ax[1].plot(ls, np.log2(np.abs(res["m"])), "s-", color=CORAL, lw=2,
               label=r"$\log_2 |E[Y_\ell]|$")
    ax[1].plot(ls[1:], np.log2(abs(res["m"][1])) - res["alpha"] * (ls[1:] - 1),
               "--", color=GRAY, lw=1.2,
               label=fr"slope $-\alpha = {-res['alpha']:.2f}$")
    ax[1].set_xlabel(r"level $\ell$"); ax[1].set_ylabel(r"$\log_2$ |mean|")
    ax[1].set_title("Weak error decay"); ax[1].legend(frameon=False)

    fig.suptitle("Layer 1b §2 — MLMC rates under rough Bergomi "
                 f"(H = {PARAMS['H']})", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/layer1b_rates.png", dpi=150)
    if show: plt.show()
    plt.close(fig)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Adaptive MLMC driver (Giles 2008)
# ══════════════════════════════════════════════════════════════════════════════

def mlmc_run(eps: float, p: dict = PARAMS, alpha: float = None,
             beta: float = None, N0: int = 2_000, Lmin: int = 2,
             Lmax: int = 9, seed: int = 11, verbose: bool = True) -> dict:
    """
    Giles' adaptive MLMC.  Chooses per-level sample sizes

        N_l = ceil( (2/eps^2) sqrt(V_l / C_l) * SUM_k sqrt(V_k C_k) )

    (variance target eps^2/2) and grows L until the bias test

        max over last 3 levels of  |m_l| / (2^alpha - 1) * 2^{-alpha(L-l)}
            <=  eps / sqrt(2)

    passes.  Rates alpha, beta may be supplied (from Section 2) or are
    re-estimated by regression on the fly.
    """
    rng = np.random.default_rng(seed)
    L = Lmin
    Nl    = np.zeros(L + 1)
    sums  = np.zeros((2, L + 1))            # sum Y, sum Y^2
    costl = np.zeros(L + 1)
    dNl   = np.full(L + 1, N0, dtype=float)
    a_fix, b_fix = alpha, beta

    while dNl.sum() > 0:
        for l in range(L + 1):
            if dNl[l] < 1: continue
            n_new = int(dNl[l])
            s = mlmc_asian_level(l, n_new, p, rng=rng)
            sums[0, l] += s[0].sum()
            sums[1, l] += (s[0]**2).sum()
            Nl[l]    += n_new
            costl[l] += n_new * p["n0"] * 2**l * (1.5 if l else 1.0)

        ml = np.abs(sums[0] / Nl)
        Vl = np.maximum(0.0, sums[1] / Nl - (sums[0] / Nl)**2)
        Cl = costl / Nl

        # regression-estimate rates if not fixed (levels >= 1); floors follow
        # Giles' practice and guard against noise-dominated regressions
        # (a negative alpha makes 2^alpha - 1 < 0 and corrupts the bias test)
        if a_fix is not None:
            alpha = max(0.5, a_fix)
        elif L >= 2:
            alpha = max(0.5, -np.polyfit(np.arange(1, L + 1), np.log2(
                np.maximum(ml[1:], 1e-12)), 1)[0])
        else:
            alpha = 0.5
        if b_fix is not None:
            beta = max(0.1, b_fix)
        elif L >= 2:
            beta = max(0.1, -np.polyfit(np.arange(1, L + 1), np.log2(
                np.maximum(Vl[1:], 1e-12)), 1)[0])
        else:
            beta = 0.5

        Ns  = np.ceil(2.0 / eps**2 * np.sqrt(Vl / Cl)
                      * np.sum(np.sqrt(Vl * Cl)))
        dNl = np.maximum(0.0, Ns - Nl)

        if (dNl <= 0.01 * Nl).all():                     # samples converged
            # Giles' bias test: extrapolate the last (up to) 3 level means
            # forward to level L assuming O(2^{-alpha l}) decay
            offs = np.arange(min(3, L))                  # offsets from L
            tail = ml[L - offs] * 2.0 ** (-alpha * offs)
            rem  = tail.max() / (2**alpha - 1)
            if rem > eps / np.sqrt(2.0):
                if L == Lmax:
                    print("  ** Lmax reached before bias target — "
                          "result may be biased **")
                    break
                L += 1
                Nl    = np.append(Nl, 0.0)
                sums  = np.append(sums, np.zeros((2, 1)), axis=1)
                costl = np.append(costl, 0.0)
                Vl    = np.append(Vl, Vl[-1] / 2**beta)
                Cl    = np.append(Cl, Cl[-1] * 2)
                Ns  = np.ceil(2.0 / eps**2 * np.sqrt(Vl / Cl)
                              * np.sum(np.sqrt(Vl * Cl)))
                dNl = np.maximum(0.0, Ns - Nl)

    price = (sums[0] / Nl).sum()
    out = dict(eps=eps, price=price, L=L, Nl=Nl.astype(int),
               Vl=Vl, cost=costl.sum(), alpha=alpha, beta=beta)
    if verbose:
        print(f"  eps={eps:<7g} price={price:.4f}  L={L}  "
              f"N_l={out['Nl']}  cost={out['cost']:.3g}")
    return out


def section3_mlmc(show: bool = True, quick: bool = False, rates: dict = None):
    print("\n" + "─" * 70)
    print("  SECTION 3 — Adaptive MLMC vs standard Monte Carlo")
    print("─" * 70)
    eps_list = ([0.20, 0.10, 0.05] if quick else
                [0.20, 0.10, 0.05, 0.025])
    a0 = rates["alpha"] if rates else None
    b0 = rates["beta"]  if rates else None

    runs = [mlmc_run(e, alpha=a0, beta=b0) for e in eps_list]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r0, mk in zip(runs, ["o", "s", "^", "d"]):
        ax[0].semilogy(np.arange(r0["L"] + 1), r0["Nl"], mk + "-",
                       lw=1.8, label=fr"$\varepsilon$ = {r0['eps']:g}")
    ax[0].set_xlabel(r"level $\ell$"); ax[0].set_ylabel(r"$N_\ell$")
    ax[0].set_title("Optimal samples per level"); ax[0].legend(frameon=False)

    # standard MC at the same accuracy: same bias (so same finest grid as
    # the MLMC run used) and variance target eps^2/2  =>
    #     cost_MC = 2 Var(P) eps^{-2} * n0 * 2^{L(eps)}
    eps  = np.array([r0["eps"] for r0 in runs])
    cost = np.array([r0["cost"] for r0 in runs])
    varP = rates["vP"] if rates else float(runs[0]["Vl"][0])
    cost_mc = np.array([2.0 * varP / r0["eps"]**2
                        * PARAMS["n0"] * 2**r0["L"] for r0 in runs])

    ax[1].loglog(eps, eps**2 * cost, "o-", color=TEAL, lw=2, label="MLMC")
    ax[1].loglog(eps, eps**2 * cost_mc, "s--", color=CORAL, lw=2,
                 label="standard MC")
    ax[1].set_xlabel(r"accuracy $\varepsilon$")
    ax[1].set_ylabel(r"$\varepsilon^2 \times$ cost")
    ax[1].set_title("Complexity: flat line = O($\\varepsilon^{-2}$)")
    ax[1].legend(frameon=False)

    fig.suptitle("Layer 1b §3 — adaptive MLMC (Giles 2008)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/layer1b_mlmc.png", dpi=150)
    if show: plt.show()
    plt.close(fig)

    ratio = cost_mc[-1] / cost[-1]
    print(f"\n  cost ratio (std MC / MLMC) at eps = {eps[-1]:g}:  {ratio:.2f}x")
    if ratio < 1.5:
        print("  NOTE: naive coupled MLMC does not convincingly beat standard"
              "\n  MC here. With beta ~ 2H << gamma = 1, level corrections"
              "\n  shrink too slowly to repay refinement — the expected"
              "\n  pathology of the beta < gamma regime, and the motivation"
              "\n  for the antithetic / conditional-MC couplings in ROADMAP.md.")
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Complexity under roughness: beta as a function of H
# ══════════════════════════════════════════════════════════════════════════════

def section4_roughness(show: bool = True, quick: bool = False):
    """
    The headline experiment: how does the MLMC variance rate beta degrade as
    volatility gets rougher?  Theory's pessimistic bound from the pathwise
    strong rate is beta >= 2H; the smoothing effect of the Asian average
    typically buys more.  The gap IS the research question.
    """
    print("\n" + "─" * 70)
    print("  SECTION 4 — Variance decay rate vs Hurst exponent")
    print("─" * 70)
    H_list = [0.05, 0.10, 0.20, 0.35] if not quick else [0.10, 0.30]
    N = 6_000 if quick else 12_000
    L = 5

    betas, alphas = [], []
    for Hx in H_list:
        p = dict(PARAMS, H=Hx)
        print(f"\n  H = {Hx}")
        res = estimate_rates(L=L, N=N, p=p, seed=23, verbose=True)
        betas.append(res["beta"]); alphas.append(res["alpha"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(H_list, betas, "o-", color=TEAL, lw=2, ms=7,
            label=r"measured $\beta$ (Asian payoff)")
    hh = np.linspace(min(H_list), max(H_list), 50)
    ax.plot(hh, 2 * hh, "--", color=GRAY, lw=1.5,
            label=r"pathwise bound $2H$")
    ax.axhline(1.0, color=CORAL, ls=":", lw=1.5,
               label=r"$\gamma = 1$ (regime boundary)")
    ax.set_xlabel("Hurst exponent H")
    ax.set_ylabel(r"variance decay rate $\beta$")
    ax.set_title("MLMC efficiency vs roughness")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("output/layer1b_beta_vs_H.png", dpi=150)
    if show: plt.show()
    plt.close(fig)

    print("\n  H      beta    alpha")
    for Hx, b, a in zip(H_list, betas, alphas):
        print(f"  {Hx:<6g} {b:<7.3f} {a:.3f}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1b — MLMC Asian pricing under rough Bergomi"
    )
    parser.add_argument("--section", type=int, choices=[1, 2, 3, 4],
                        help="Run only one section (default: all four)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display figures (just save to output/)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sample sizes for a fast pass")
    args = parser.parse_args()
    show, quick = not args.no_show, args.quick

    print("\n" + "█" * 70)
    print("  Layer 1b — Multilevel Monte Carlo under Rough Volatility")
    print("  Arithmetic Asian options in the rough Bergomi model")
    print("  Project: RL as Numerical Approach to Stochastic Optimal Control")
    print("█" * 70)

    if args.section == 1:
        section1_validation(show, quick)
    elif args.section == 2:
        section2_rates(show, quick)
    elif args.section == 3:
        rates = estimate_rates(L=5 if quick else 6,
                               N=10_000 if quick else 20_000, verbose=False)
        section3_mlmc(show, quick, rates)
    elif args.section == 4:
        section4_roughness(show, quick)
    else:
        section1_validation(show, quick)
        rates = section2_rates(show, quick)
        section3_mlmc(show, quick, rates)
        section4_roughness(show, quick)

    print("\n" + "=" * 70)
    print("  Layer 1b complete.  Figures in ./output/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
