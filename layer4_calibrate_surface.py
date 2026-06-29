"""
layer4_calibrate_surface.py — MULTI-MATURITY surface calibration (sandbox, synthetic
known-answer). Extends the single-maturity engine (D37, layer4_calibrate.py) to a JOINT
IV surface across maturities, and tests the load-bearing hypothesis:

  D37 showed a single smile CANNOT identify H (flat eigen-direction; H~ν degenerate −0.82;
  cond(JᵀJ)≈6e5). The reason is structural — H controls the TERM STRUCTURE of the skew
  (rough-vol power law, ATM-skew ∝ T^{H−1/2}). A single maturity sees the skew at one T,
  not its decay where H lives. HYPOTHESIS: adding maturities → term-structure info →
  breaks the H~ν degeneracy → the surface IDENTIFIES H.

Reuses D37's core unchanged (calibrate, residuals, iv_rmse, CalibResult). Calibrates
against the CF (exact/fast); the lift is OUT (its validation role is done, D36/D37).
Real-market surface fitting is OUT (the later step on the user's machine — this unblocks it).
Spec §5/§8; ROADMAP D38.
"""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread/process -> clean pool parallelism (the EXP3 fix)
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rough_heston import PARAMS
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_iv
from layer4_calibrate import KAPPA_FIXED, calibrate, iv_rmse, CalibResult

S0, R = PARAMS["S0"], PARAMS["r"]
TS = [0.10, 0.25, 0.50, 1.00, 2.00]                       # maturity grid (~4.5 octaves; short end carries H)
VUS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])    # log-moneyness in std-devs (|vu|<=2)
NOISE_VUS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])         # leaner grid for the (recovery-heavy) noise ensemble
TRUTH = {4: np.array([0.10, 0.35, -0.70, 0.04]),
         5: np.array([0.10, 0.35, -0.70, 0.04, 0.30])}    # [+kappa]
PN = {4: ["H", "nu", "rho", "xi0"], 5: ["H", "nu", "rho", "xi0", "kappa"]}
LB = {4: np.array([0.02, 0.05, -0.99, 1e-3]), 5: np.array([0.02, 0.05, -0.99, 1e-3, 0.02])}
UB = {4: np.array([0.49, 1.00, 0.00, 0.25]), 5: np.array([0.49, 1.00, 0.00, 0.25, 5.0])}
INIT_FAR = {4: np.array([0.20, 0.20, -0.40, 0.05]), 5: np.array([0.20, 0.20, -0.40, 0.05, 1.0])}
NR_SAFE, NN = 1000, 128                                    # cheapest all-finite surface knobs (35/35 at truth)


# --------------------------------------------------------------------------- #
# T-parameterised CF smile + surface model
# --------------------------------------------------------------------------- #
def theta_to_cfparams_s(theta, n_params=4):
    H, nu, rho, xi0 = theta[:4]
    kappa = KAPPA_FIXED if n_params == 4 else theta[4]
    return H, dict(V0=xi0, kappa=kappa, theta=xi0, nu=nu, rho=rho)


def model_smile_cf_T(theta, Ks, T, *, n_params=4, N_riccati=NR_SAFE, n_nodes=NN):
    """Guarded CF IV per strike at maturity T (NaN where Riccati overflows / inversion fails)."""
    H, cfp = theta_to_cfparams_s(theta, n_params)
    cf = lambda u: rough_heston_cf(u, T, H=H, N_riccati=N_riccati, **cfp)
    out = np.full(len(Ks), np.nan)
    for i, K in enumerate(Ks):
        try:
            px = gil_pelaez_call(cf, S0, K, T, R, n_nodes=n_nodes)
            if np.isfinite(px) and px > 1e-12:
                out[i] = bs_iv(px, S0, K, T, R)
        except Exception:
            pass
    return out


def atm_by_T(theta, Ts, *, n_params=4, **cf_kw):
    return {T: model_smile_cf_T(theta, np.array([S0]), T, n_params=n_params, **cf_kw)[0] for T in Ts}


def fixed_surface_strikes(atm_by_T_, vus, Ts):
    """K(T) = S0·exp(vu·σ_atm(T)·√T) — same #std-devs at every T; anchored ONCE from the target."""
    return {T: S0 * np.exp(vus * atm_by_T_[T] * np.sqrt(T)) for T in Ts}


def surface_model(theta, grids_by_T, *, n_params=4, cf_kw=None):
    """Stack per-T smiles into ONE fixed-length vector (frozen order: sorted(Ts), then K)."""
    cf_kw = cf_kw or {}
    return np.concatenate([model_smile_cf_T(theta, grids_by_T[T], T, n_params=n_params, **cf_kw)
                           for T in sorted(grids_by_T)])


def make_surface_model(grids_by_T, n_params=4):
    """Adapter so D37's calibrate (model(theta, Ks, **cf_kw)) drives the surface; Ks is ignored."""
    def _m(theta, _ks_ignored, **cf_kw):
        return surface_model(theta, grids_by_T, n_params=n_params, cf_kw=cf_kw or None)
    return _m


def build_target(truth, Ts, *, n_params=4, vus=VUS, cf_kw=None):
    """Synthetic CF surface at the known params: returns (grids_by_T, stacked_target)."""
    a = atm_by_T(truth, Ts, n_params=n_params, **(cf_kw or {}))
    grids = fixed_surface_strikes(a, vus, Ts)
    return grids, surface_model(truth, grids, n_params=n_params, cf_kw=cf_kw)


def calibrate_surface(stacked_target, grids_by_T, *, n_params=4, theta0=None, bounds=None,
                      cf_kw=None, max_nfev=200):
    theta0 = INIT_FAR[n_params] if theta0 is None else np.asarray(theta0, float)
    bounds = (LB[n_params], UB[n_params]) if bounds is None else bounds
    return calibrate(stacked_target, Ks=None, theta0=theta0, bounds=bounds,
                     model=make_surface_model(grids_by_T, n_params), cf_kw=cf_kw, max_nfev=max_nfev)


# --------------------------------------------------------------------------- #
# identifiability (dimension-agnostic; shared by single-T and surface)
# --------------------------------------------------------------------------- #
@dataclass
class SurfIdent:
    cond: float
    eig: np.ndarray
    flat: np.ndarray
    sens: np.ndarray
    corr: np.ndarray
    pnames: list

    def line(self, tag):
        pn = self.pnames
        i_h = pn.index("H"); i_nu = pn.index("nu")
        return (f"  [{tag:28s}] cond={self.cond:.2e}  |flat[H]|={abs(self.flat[i_h]):.2f}  "
                f"corr(H,ν)={self.corr[i_h, i_nu]:+.2f}  "
                f"flat=(" + ",".join(f"{pn[k]}:{self.flat[k]:+.2f}" for k in range(len(pn))) + ")")


def surface_jacobian(theta, grids_by_T, *, n_params=4, rel=0.01, abs_=1e-4, cf_kw=None):
    base = np.array(theta[:n_params], float)
    J = []
    for j in range(n_params):
        h = rel * abs(base[j]) + abs_
        tp = base.copy(); tp[j] += h
        tm = base.copy(); tm[j] -= h
        J.append((surface_model(tp, grids_by_T, n_params=n_params, cf_kw=cf_kw)
                  - surface_model(tm, grids_by_T, n_params=n_params, cf_kw=cf_kw)) / (2 * h))
    return np.array(J).T                                  # (n_strikes, n_params)


def ident_from_J(J, pnames):
    good = np.isfinite(J).all(axis=1)
    J = J[good]
    npar = J.shape[1]
    if J.shape[0] < npar:
        raise ValueError(f"surface ident: {J.shape[0]} finite rows < {npar} params (raise N_riccati)")
    G = J.T @ J
    d = np.sqrt(np.diag(G))
    corr = G / np.outer(d, d)
    eig, V = np.linalg.eigh(G)
    return SurfIdent(eig[-1] / eig[0], eig, V[:, 0], d, corr, list(pnames))


def ident_for(Ts, *, n_params=4, vus=VUS, cf_kw=None):
    grids, _ = build_target(TRUTH[n_params], Ts, n_params=n_params, vus=vus, cf_kw=cf_kw)
    J = surface_jacobian(TRUTH[n_params], grids, n_params=n_params, cf_kw=cf_kw)
    return ident_from_J(J, PN[n_params])


def _fmt(theta_hat, truth, pnames):
    return "  ".join(f"{n}={v:+.4f}({100*(v-truth[i])/abs(truth[i]):+.0f}%)"
                     for i, (n, v) in enumerate(zip(pnames, theta_hat)))


# --------------------------------------------------------------------------- #
# noise ensemble (parallel, capped pool — the D37 OOM lesson)
# --------------------------------------------------------------------------- #
def _surf_ens_worker(args):
    seed, sigma, target, grids, n_params, theta0, cf_kw = args
    rng = np.random.default_rng(seed)
    noisy = target + rng.standard_normal(len(target)) * sigma
    return calibrate_surface(noisy, grids, n_params=n_params, theta0=theta0, cf_kw=cf_kw).theta_hat


def surface_noise_ensemble(target, grids, *, sigma, n_draws=10, seed=100, n_params=4,
                           theta0=None, cf_kw=None, parallel=True):
    args = [(seed + d, sigma, target, grids, n_params, theta0, cf_kw) for d in range(n_draws)]
    if parallel:
        try:
            import os
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(4, (os.cpu_count() or 4))) as ex:
                recs = []
                for i, r in enumerate(ex.map(_surf_ens_worker, args)):
                    recs.append(r)
                    print(f"        member {i+1:2d}/{len(args)}  H={r[0]:.4f}", flush=True)
                recs = np.array(recs)
        except Exception:
            recs = np.array([_surf_ens_worker(a) for a in args])
    else:
        recs = np.array([_surf_ens_worker(a) for a in args])
    return recs.mean(0), recs.std(0)


# --------------------------------------------------------------------------- #
# the four experiments
# --------------------------------------------------------------------------- #
def exp_cf(n_params=4, N_riccati=1200, n_nodes=140):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    print(f"\n=== EXP 1: SURFACE CF→CF gate ({n_params}-param, T={TS}) ===")
    grids, target = build_target(TRUTH[n_params], TS, n_params=n_params, cf_kw=cf_kw)
    res = calibrate_surface(target, grids, n_params=n_params, cf_kw=cf_kw)
    print(f"  truth : " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(PN[n_params], TRUTH[n_params])))
    print(f"  recov : {_fmt(res.theta_hat, TRUTH[n_params], PN[n_params])}")
    print(f"  IV-RMSE={res.iv_rmse*100:.5f}pp  nfev={res.nfev}  -> H recovered: "
          f"{abs(res.theta_hat[0]-TRUTH[n_params][0])/TRUTH[n_params][0] < 0.05}")


def exp_ident(N_riccati=NR_SAFE, n_nodes=NN):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    print(f"\n=== EXP 2: ★ IDENTIFIABILITY — single-T vs surface (the H hypothesis) ===")
    print("  (D37 single-T=1 baseline: cond≈6e5, H~ν=−0.82, H is the flat direction)")
    cases = [("SINGLE T=1.0", [1.00]), ("SHORT-END {0.1,0.25}", [0.10, 0.25]),
             ("SURFACE {0.25..2}", [0.25, 0.5, 1.0, 2.0]), ("SURFACE {0.1..2}", TS)]
    base = None
    for tag, Ts in cases:
        r = ident_for(Ts, n_params=4, cf_kw=cf_kw)
        if base is None:
            base = r.cond
        print(r.line(tag) + (f"  [cond ÷{base/r.cond:.0e}× vs single]" if r.cond < base else ""))


def exp_noise(sigmas=(0.001, 0.003, 0.005), n_draws=10, N_riccati=900, n_nodes=100):
    """Cheaper ensemble (NOISE_VUS=5 strikes, capped pool); RUN WITH BLAS threads=1 in the env
    (OMP/OPENBLAS/MKL_NUM_THREADS=1) so the 4 workers don't oversubscribe — the diagnosed cause
    of the 2h+ stall (each member converges at nfev~6; the cost was per-eval × pool contention)."""
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    print(f"\n=== EXP 3: NOISE-ROBUSTNESS + span sweep (H spread; D37 single-smile: 62% @0.1pp) ===")
    near = TRUTH[4] * np.array([1.25, 0.9, 0.95, 1.03])           # near-truth init (local sensitivity)
    for tag, Ts in (("SINGLE T=1.0", [1.00]), ("SURFACE {0.1..2}", TS)):
        grids, target = build_target(TRUTH[4], Ts, vus=NOISE_VUS, cf_kw=cf_kw)
        print(f"  [{tag}]", flush=True)
        for sg in sigmas:
            _, std = surface_noise_ensemble(target, grids, sigma=sg, n_draws=n_draws,
                                            theta0=near, cf_kw=cf_kw)
            rel = std / np.abs(TRUTH[4])
            print(f"     σ={sg*100:.1f}pp  H_spread={rel[0]*100:5.0f}%  "
                  + " ".join(f"{PN[4][i]}={rel[i]*100:4.0f}%" for i in (1, 2, 3)), flush=True)
    print("  span sweep (cond vs #maturities):", flush=True)
    for Ts in ([1.0], [0.5, 1.0, 2.0], [0.25, 0.5, 1.0, 2.0], TS):
        r = ident_for(Ts, vus=NOISE_VUS, cf_kw=cf_kw)
        print(f"     {len(Ts)} mat {str(Ts):28s} cond={r.cond:.2e}  |flat[H]|={abs(r.flat[0]):.2f}", flush=True)


def exp_kappa(N_riccati=NR_SAFE, n_nodes=NN):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    print(f"\n=== EXP 4: κ DECISION (5-param surface) ===")
    r4 = ident_for(TS, n_params=4, cf_kw=cf_kw)
    r5 = ident_for(TS, n_params=5, cf_kw=cf_kw)
    print(r4.line("4-param surface"))
    print(r5.line("5-param surface (+κ)"))
    ik = PN[5].index("kappa")
    print(f"  κ: sens={r5.sens[ik]:.3g}  |flat[κ]|={abs(r5.flat[ik]):.2f}  "
          f"(degenerate if |flat[κ]|→1 or cond inflates vs 4-param)")
    print(f"  cond 5-param vs 4-param: {r5.cond:.2e} vs {r4.cond:.2e} "
          f"({'INFLATES — κ harms' if r5.cond > 5*r4.cond else 'comparable'})")
    print("  5-param CF→CF recovery (is κ recoverable?):")
    grids, target = build_target(TRUTH[5], TS, n_params=5, cf_kw=cf_kw)
    res = calibrate_surface(target, grids, n_params=5, cf_kw=cf_kw)
    print(f"     {_fmt(res.theta_hat, TRUTH[5], PN[5])}  IV-RMSE={res.iv_rmse*100:.4f}pp")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["cf", "ident", "noise", "kappa", "all"], default="ident")
    a = ap.parse_args()
    for e in (["cf", "ident", "noise", "kappa"] if a.exp == "all" else [a.exp]):
        {"cf": exp_cf, "ident": exp_ident, "noise": exp_noise, "kappa": exp_kappa}[e]()
