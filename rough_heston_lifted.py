"""
rough_heston_lifted.py — Layer 4 brick 4b: lifted (multifactor OU) rough-Heston
simulator. N Markovian OU factors reconstruct the rough variance instead of the
O(n^2) Volterra history-convolution -> O(N*n) per path.

Source-pinned lifted SDE (Abi Jaber, "Lifting the Heston model", arXiv:1810.04868,
Eq 2.2-2.3), in the brick-1-consistent (folded-drift) form so beta and H=1/2 prices
are directly comparable to rough_heston.py:

    dU_i = (-x_i U_i + kappa(theta - V)) dt + nu*sqrt(V) dW,   U_i(0)=0
    V    = V0 + sum_i c_i U_i

Each factor carries its own mean-reversion x_i (= gamma_i from brick 4a); the CIR
coupling kappa(theta-V) and diffusion nu*sqrt(V) dW are on the AGGREGATE V, fed to
every factor, all sharing ONE Brownian W (which also drives the asset via rho).
(c_i, x_i) = brick-4a Bayer-Breneis (weights, gammas) from rough_kernel_soe.soe.

CRUX-2 integration: EXACT-OU / exponential-Euler factor decay e^(-x_i dt) (exact at
any x_i; explicit Euler is unstable across the ~1e14 gamma-span), shared-DeltaW
envelope for the noise (exact joint noise is O(N^2); the envelope keeps O(N*n)).

CRUX-3 positivity: DEFAULT "qe" -- brick-1's Andersen QE ported to the lift (applied
to the aggregate conditional-Gaussian step, with an effective increment so the factors
still reconstruct the QE'd V): positive + smooth (coupling-preserving) + unbiased.
GATE B selected it on EVIDENCE -- "trunc" (the control) COLLAPSES beta at H<=0.10,
nu=0.20 (frequent near-0 breaks the MLMC coupling), and implicit-drift / naive-Alfonsi
were tried and ruled out (weak / GATE-C-biased). See ROADMAP D33 for the full journey.
"""
import numpy as np

from rough_heston import PARAMS
from rough_kernel_soe import soe


def lifted_setup(H, N, T=1.0, method="bb"):
    """Factor nodes/weights (x_i, c_i) for the lift, from brick-4a's SOE kernel."""
    gammas, weights = soe(H, N, method=method, T=T)
    return gammas, weights


def _lifted_from_increments(dWV, dW_perp, n, H, p, gammas, weights, positivity="qe"):
    """Core lifted recursion from pre-drawn Brownian increments (shape (M, n),
    ~N(0,dt)). The SAME (gammas, weights) are used for any n, so the beta harness
    can drive coupled fine/coarse paths from one Brownian (coarse = pairwise sums).
    Returns (S, V): (M, n+1), col 0 = S0, V0."""
    T = p["T"]; dt = T / n
    V0 = p["V0"]; kappa = p["kappa"]; theta = p["theta"]
    nu = p["nu"]; rho = p["rho"]; S0 = p["S0"]; r = p["r"]
    x = np.asarray(gammas, float); c = np.asarray(weights, float)
    M = dWV.shape[0]

    # ---- per-factor exact-OU step coefficients (depend on x_i, dt only) ----
    xdt = x * dt
    decay = np.exp(-xdt)                                  # e^(-x_i dt); x=0 -> 1
    e1 = -np.expm1(-xdt)                                  # 1 - e^(-x_i dt)
    pos = x > 0.0
    env = np.where(pos, e1 / np.where(pos, xdt, 1.0), 1.0)   # (1-e^-xdt)/(xdt); x=0 -> 1
    drift_coef = env * dt                                 # (1-e^-xdt)/x ; x=0 -> dt

    # ---- variance path: exact-OU factor recursion ----
    # CRUX-3 positivity (selected on EVIDENCE via GATE B, not theory):
    #   "qe"    : DEFAULT, validated. Port brick-1 Andersen QE to the lift -- apply
    #             _qe_map to the AGGREGATE conditional-Gaussian step (mean m, var s2),
    #             then back out an effective increment dWtil so the factors still
    #             reconstruct the QE'd V. Positive + smooth (coupling-preserving) +
    #             unbiased (moment-matched). Reproduces brick-1's beta=2H.
    #   "trunc" : control. Explicit drift kappa(theta - V+), diffusion sqrt(V+).
    #             Correct at low near-0, but COLLAPSES beta at H<=0.10, nu=0.20
    #             (frequent near-0 -> the V+ clip breaks the MLMC coupling) -- the
    #             brick-1 truncation failure recurring; it is why QE is needed.
    # (Implicit-drift and naive-Alfonsi were tried and ruled out -- see ROADMAP D33.)
    if positivity not in ("qe", "trunc"):
        raise ValueError(f"unknown positivity {positivity!r} (use 'qe' or 'trunc')")
    Dk = kappa * float(c @ drift_coef)                   # = kappa * sum_i c_i*drift_coef_i
    E = float(c @ env)                                   # = sum_i c_i*env_i
    sqrt_dt = np.sqrt(dt)
    if positivity == "qe":
        from rough_heston import _qe_map                 # reuse brick-1 Andersen QE
    U = np.zeros((M, x.size))
    V = np.empty((M, n + 1)); V[:, 0] = V0
    for i in range(n):
        if positivity == "qe":
            Vstar = V[:, i]; sq = np.sqrt(np.maximum(Vstar, 0.0))
            decayU = (U * decay) @ c
            m = V0 + decayU + Dk * (theta - Vstar)       # cond. mean of factor-aggregate
            s2 = (nu * E * sq) ** 2 * dt                  # cond. variance
            Vnew = _qe_map(m, s2, dWV[:, i] / sqrt_dt)    # QE: >=0, smooth in dW
            denom = nu * E * sq
            dWtil = np.where(denom > 0, (Vnew - m) / np.where(denom > 0, denom, 1.0),
                             dWV[:, i])                   # effective incr: V0+U@c == Vnew
            cdrift = kappa * (theta - Vstar)
            U = U * decay + cdrift[:, None] * drift_coef + (nu * sq)[:, None] * env * dWtil[:, None]
        else:                                            # "trunc" (control)
            Vpos = np.maximum(V[:, i], 0.0); sq = np.sqrt(Vpos)
            cdrift = kappa * (theta - Vpos)
            U = U * decay + cdrift[:, None] * drift_coef + (nu * sq * dWV[:, i])[:, None] * env
        V[:, i + 1] = V0 + U @ c                          # reconstruct aggregate (== Vnew for qe)

    # ---- asset path: log-Euler, left-point V+, correlated BM (matches brick 1) ----
    Vp_left = np.maximum(V[:, :-1], 0.0)
    dW_S = rho * dWV + np.sqrt(1.0 - rho ** 2) * dW_perp
    dlogS = (r - 0.5 * Vp_left) * dt + np.sqrt(Vp_left) * dW_S
    logS = np.concatenate([np.zeros((M, 1)), np.cumsum(dlogS, axis=1)], axis=1)
    S = S0 * np.exp(logS)
    return S, V


def rough_heston_lifted_paths(n, H, n_paths, N, rng=None, method="bb",
                              positivity="qe", **overrides):
    """Simulate lifted rough-Heston (S, V) paths. N = nominal factor count (the
    actual count is len(gammas)). **overrides: any PARAMS field."""
    p = {**PARAMS, **overrides}
    rng = rng or np.random.default_rng()
    dt = p["T"] / n
    dWV = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    dW_perp = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
    gammas, weights = lifted_setup(H, N, T=p["T"], method=method)
    S, V = _lifted_from_increments(dWV, dW_perp, n, H, p, gammas, weights, positivity)
    t = np.linspace(0.0, p["T"], n + 1)
    return t, S, V


# --------------------------------------------------------------------------- #
#  Reconstruction sanity                                                        #
# --------------------------------------------------------------------------- #
def reconstruction_sanity():
    print("RECONSTRUCTION SANITY")
    p = dict(PARAMS)
    for V0 in (0.04, 0.08):                               # nu=0 -> V relaxes to theta
        t, S, V = rough_heston_lifted_paths(128, 0.10, 4000, N=100,
                                            rng=np.random.default_rng(0), nu=0.0, V0=V0)
        print(f"  nu=0, V0={V0}: V[T] mean={V[:, -1].mean():.5f} (theta=0.04), "
              f"std={V[:, -1].std():.2e}, nan={np.isnan(V).any()}")
    t, S, V = rough_heston_lifted_paths(128, 0.10, 20000, N=100,
                                        rng=np.random.default_rng(0))
    print(f"  nu=0.2, H=0.10: E[V_T]={V[:, -1].mean():.5f} (theta=0.04), "
          f"near0 freq={(V < 1e-6).mean():.1%}, nan={np.isnan(V).any()}")
    # stiff stability: inject a huge gamma factor, confirm no blow-up
    g = np.array([0.0, 1e14]); w = np.array([0.5, 0.5])
    dWV = np.random.default_rng(1).standard_normal((1000, 128)) * np.sqrt(1 / 128)
    dWp = np.random.default_rng(2).standard_normal((1000, 128)) * np.sqrt(1 / 128)
    S, V = _lifted_from_increments(dWV, dWp, 128, 0.10, p, g, w)
    print(f"  stiff (gamma=1e14): finite={np.isfinite(V).all()}, "
          f"max|V|={np.abs(V).max():.3f}  -> exact-OU stable")


# --------------------------------------------------------------------------- #
#  GATE C: H=1/2 price vs CF (integrator anchor)                                #
# --------------------------------------------------------------------------- #
def gate_c(K=100.0, n_list=(32, 64, 128, 256), M=200000, N=8, seed=7, N_riccati=2000):
    from layer4_convergence import cf_reference
    p = dict(PARAMS)
    P_cf = cf_reference(0.5, p, K, N_riccati)
    print(f"GATE C -- H=1/2 lifted European call vs CF (classical Heston) | "
          f"K={K} M={M} | P_CF={P_cf:.5f}")
    print(f"  {'n':>5} {'price':>10} {'95% CI':>10} {'bias':>10} {'bias/se':>8}")
    for n in n_list:
        t, S, V = rough_heston_lifted_paths(n, 0.5, M, N, rng=np.random.default_rng(seed))
        pay = np.maximum(S[:, -1] - K, 0.0)
        price = pay.mean(); se = pay.std(ddof=1) / np.sqrt(M)
        print(f"  {n:>5} {price:>10.5f} {1.96*se:>10.5f} {price-P_cf:>+10.5f} "
              f"{(price-P_cf)/se:>+8.1f}")
    return P_cf


# --------------------------------------------------------------------------- #
#  GATE B: beta = 2H (coupling / rate)                                          #
# --------------------------------------------------------------------------- #
def _lifted_mlmc_level(l, M, H, p, n0, gammas, weights, rng, K=100.0, positivity="qe"):
    from rh_beta_gate import _asian_call
    n_f = n0 * 2 ** l; n_c = n_f // 2
    dt_f = p["T"] / n_f
    dWV_f = rng.standard_normal((M, n_f)) * np.sqrt(dt_f)
    dWp_f = rng.standard_normal((M, n_f)) * np.sqrt(dt_f)
    Sf, _ = _lifted_from_increments(dWV_f, dWp_f, n_f, H, p, gammas, weights, positivity)
    dWV_c = dWV_f.reshape(M, n_c, 2).sum(axis=2)          # pairwise-summed fine (CRN)
    dWp_c = dWp_f.reshape(M, n_c, 2).sum(axis=2)
    Sc, _ = _lifted_from_increments(dWV_c, dWp_c, n_c, H, p, gammas, weights, positivity)
    Y = _asian_call(Sf, K, n_f) - _asian_call(Sc, K, n_c)
    return Y.mean(), Y.var(ddof=1), _asian_call(Sf, K, n_f).mean()


def measure_beta_lifted(H, p, n0, levels, M, N, rng, method="bb", K=100.0, positivity="qe"):
    gammas, weights = lifted_setup(H, N, T=p["T"], method=method)
    rows = [_lifted_mlmc_level(l, M, H, p, n0, gammas, weights, rng, K, positivity) for l in levels]
    vY = np.array([r[1] for r in rows])
    ls = np.asarray(levels, float)
    slope, intercept = np.polyfit(ls, np.log2(vY), 1)
    resid = np.log2(vY) - (slope * ls + intercept)
    dof = len(ls) - 2
    se = (np.sqrt((resid ** 2).sum() / dof / ((ls - ls.mean()) ** 2).sum())
          if dof > 0 else np.nan)
    return dict(beta=-slope, beta_se=se, vY=vY, monotone=bool(np.all(np.diff(vY) < 0)),
                n_factors=int(gammas.size))


def gate_b(H_grid=(0.05, 0.10, 0.20, 0.35), N=150, n0=16, levels=(1, 2, 3, 4, 5),
           M=20000, seed=7, method="bb"):
    from rh_beta_gate import L1B_BETA
    p = dict(PARAMS)
    print(f"GATE B -- lifted beta=2H | N(nominal)={N} n0={n0} levels={list(levels)} "
          f"M={M} nu={p['nu']}")
    print(f"  {'H':>6} {'2H':>6} {'L1b':>6} {'beta':>8} {'±se':>7} {'|b-2H|':>8} "
          f"{'mono':>5} {'Nfac':>5} {'verdict':>8}")
    ok = True
    for H in H_grid:
        r = measure_beta_lifted(H, p, n0, levels, M, N, np.random.default_rng(seed), method)
        dev = abs(r["beta"] - 2 * H)
        verdict = "OK" if (dev <= 0.05 and r["monotone"]) else "OFF"
        ok = ok and verdict == "OK"
        print(f"  {H:6.2f} {2*H:6.2f} {L1B_BETA.get(round(H,2), float('nan')):6.2f} "
              f"{r['beta']:8.3f} {r['beta_se']:7.3f} {dev:8.3f} {str(r['monotone']):>5} "
              f"{r['n_factors']:>5} {verdict:>8}")
    print(f"  GATE B: {'consistent with layer1b' if ok else 'NOT consistent'}")
    return ok


def gate_b_nfactors(H=0.10, N_list=(25, 50, 100, 200), n0=16, levels=(1, 2, 3, 4, 5),
                    M=20000, seed=7):
    """The N_factors finding: how many factors does the RATE (beta) need?"""
    p = dict(PARAMS)
    print(f"\nGATE B -- N_factors finding at H={H} (2H={2*H})")
    print(f"  {'N_nom':>6} {'Nfac':>5} {'beta':>8} {'±se':>7} {'|b-2H|':>8} {'mono':>5}")
    for N in N_list:
        r = measure_beta_lifted(H, p, n0, levels, M, N, np.random.default_rng(seed))
        print(f"  {N:>6} {r['n_factors']:>5} {r['beta']:>8.3f} {r['beta_se']:>7.3f} "
              f"{abs(r['beta']-2*H):>8.3f} {str(r['monotone']):>5}")


if __name__ == "__main__":
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--gate-c", action="store_true")
    ap.add_argument("--gate-b", action="store_true")
    ap.add_argument("--nfactors", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.sanity:
        reconstruction_sanity()
    if a.gate_c:
        gate_c(M=50000 if a.quick else 200000)
    if a.gate_b:
        if a.quick:
            gate_b(N=50, levels=(1, 2, 3), M=4000)
        else:
            gate_b()
    if a.nfactors:
        gate_b_nfactors(M=4000 if a.quick else 20000)
    if not any((a.sanity, a.gate_c, a.gate_b, a.nfactors)):
        reconstruction_sanity()
