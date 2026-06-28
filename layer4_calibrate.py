"""
layer4_calibrate.py — rough-Heston single-maturity smile CALIBRATION engine,
validated against synthetic CF known-answer smiles (SPX prerequisite, sandbox).

Fits θ = [H, ν, ρ, ξ₀] to a target implied-vol smile by minimising IV-RMSE in IV
space (the calibration-relevant metric, per D36). Calibrates against the CF (fast,
analytic) — the optimiser evaluates the smile many times. Real-market SPX data is
OUT of scope (a documented later step on the user's machine); here we apply the
project's KNOWN-ANSWER discipline: validate the calibrator against CF-generated
truth before trusting it.

Param mapping (single-maturity reduction): ξ₀ → V0 = theta (flat forward variance),
kappa FIXED = 0.30 (a single maturity cannot identify mean-reversion), (H, ν, ρ)
direct. T=1, S0=100, r=0.

Four known-answer experiments (run via --exp):
  cf     CF→CF recovery gate    — does the optimiser recover known params? (exact)
  ident  identifiability report — JᵀJ cond / correlation / flat direction; which
                                  params a single smile pins (ξ₀,ρ) vs degenerate (H·ν)
  noise  noise-robustness       — perturb target, recalibrate ensemble → realistic spread
  lift   lift→CF calib-grade    — calibrate with the LIFT (D36's ~1pp call-wing bias)
                                  in the loop; does the bias distort recovered params?
Spec: docs/gate_checks/layer4_convergence_gate_check.md §5/§8; ROADMAP D37.
"""
import sys
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rough_heston import PARAMS
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_iv, bs_vega

PNAMES = ("H", "nu", "rho", "xi0")
KAPPA_FIXED = 0.30
S0, T, R = PARAMS["S0"], PARAMS["T"], PARAMS["r"]
TRUTH = np.array([0.10, 0.35, -0.70, 0.04])           # SPX-like known answer
VUS_FULL = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
VUS_PUT = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
LB = np.array([0.02, 0.05, -0.99, 1e-3])
UB = np.array([0.49, 1.00, 0.00, 0.25])
INIT_FAR = np.array([0.20, 0.20, -0.40, 0.05])


# --------------------------------------------------------------------------- #
# model smiles: CF (analytic, the calibration model) and LIFT (MC, the diagnostic)
# --------------------------------------------------------------------------- #
def theta_to_cfparams(theta):
    H, nu, rho, xi0 = theta
    return H, dict(V0=xi0, kappa=KAPPA_FIXED, theta=xi0, nu=nu, rho=rho)


def model_smile_cf(theta, Ks, *, N_riccati=1000, n_nodes=160):
    """Guarded CF IV per strike (NaN where the Riccati overflows / inversion fails)."""
    H, cfp = theta_to_cfparams(theta)
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


def atm_iv_cf(theta, **cf_kw):
    return model_smile_cf(theta, np.array([S0]), **cf_kw)[0]


def model_smile_lift(theta, Ks, *, seed=7, M=50000, n=128, N=80):
    """LIFT IV per strike via the conditional-MC smile (one run → analytic BS across
    strikes; CRN via FIXED seed → smooth objective, the bias not MC noise drives it)."""
    from rough_heston_lifted import _lifted_from_increments, lifted_setup
    from layer4_convergence import _bs_call_vec
    H, nu, rho, xi0 = theta
    p = dict(PARAMS, nu=nu, rho=rho, V0=xi0, theta=xi0)
    g, w = lifted_setup(H, N, T=T)
    rng = np.random.default_rng(seed); dt = T / n
    dWV = rng.standard_normal((M, n)) * np.sqrt(dt)
    dWp = rng.standard_normal((M, n)) * np.sqrt(dt)
    _, V = _lifted_from_increments(dWV, dWp, n, H, p, g, w, "qe")
    Vl = np.maximum(V[:, :-1], 0.0)
    Mt = (np.sqrt(Vl) * dWV).sum(axis=1)
    It = Vl.sum(axis=1) * dt
    S_eff = S0 * np.exp(rho * Mt - 0.5 * rho ** 2 * It)
    sig_eff = np.sqrt(np.maximum((1.0 - rho ** 2) * It / T, 1e-300))
    out = np.full(len(Ks), np.nan)
    for i, K in enumerate(Ks):
        px = _bs_call_vec(S_eff, K, T, R, sig_eff).mean()
        if np.isfinite(px) and px > 1e-12:
            out[i] = bs_iv(px, S0, K, T, R)
    return out


def fixed_strikes_from_target(atm_iv_target, vus):
    """Market strikes set ONCE from the target ATM IV — never recomputed per candidate."""
    return S0 * np.exp(vus * atm_iv_target)


# --------------------------------------------------------------------------- #
# residual / optimiser
# --------------------------------------------------------------------------- #
def residuals(theta, Ks, target_iv, model=model_smile_cf, weights=None,
              nan_penalty=0.5, cf_kw=None):
    """Fixed-LENGTH residual model_IV − target_IV; NaN → +penalty (NOT dropped —
    least_squares' FD Jacobian needs a constant length)."""
    iv = model(theta, Ks, **(cf_kw or {}))
    r = iv - target_iv
    r = np.where(np.isfinite(r), r, nan_penalty)
    return r if weights is None else r * weights


def iv_rmse(theta, Ks, target_iv, model=model_smile_cf, cf_kw=None):
    iv = model(theta, Ks, **(cf_kw or {}))
    m = np.isfinite(iv) & np.isfinite(target_iv)
    return float(np.sqrt(np.mean((iv[m] - target_iv[m]) ** 2))) if m.any() else np.nan


@dataclass
class CalibResult:
    theta_hat: np.ndarray
    iv_rmse: float
    finite_mask: np.ndarray
    nfev: int
    success: bool
    message: str = ""

    def __str__(self):
        return ("  ".join(f"{n}={v:+.4f}" for n, v in zip(PNAMES, self.theta_hat))
                + f"  | IV-RMSE={self.iv_rmse*100:.4f}pp  nfev={self.nfev}")


def calibrate(target_iv, Ks, *, theta0=None, bounds=None, model=model_smile_cf,
              weights=None, cf_kw=None, max_nfev=120):
    theta0 = INIT_FAR.copy() if theta0 is None else np.asarray(theta0, float)
    lb, ub = (LB, UB) if bounds is None else bounds
    sol = least_squares(residuals, theta0, bounds=(lb, ub), method="trf",
                        jac="2-point", diff_step=2e-3, xtol=1e-12, ftol=1e-12,
                        max_nfev=max_nfev,
                        kwargs=dict(Ks=Ks, target_iv=target_iv, model=model,
                                    weights=weights, cf_kw=cf_kw))
    iv = model(sol.x, Ks, **(cf_kw or {}))
    mask = np.isfinite(iv) & np.isfinite(target_iv)
    rmse = float(np.sqrt(np.mean((iv[mask] - target_iv[mask]) ** 2))) if mask.any() else np.nan
    return CalibResult(sol.x, rmse, mask, int(sol.nfev), bool(sol.success), sol.message)


# --------------------------------------------------------------------------- #
# identifiability: JᵀJ at θ → cond / correlation / flat direction / classes
# --------------------------------------------------------------------------- #
@dataclass
class IdentReport:
    cond: float
    eigvals: np.ndarray
    flat_dir: np.ndarray
    sens: np.ndarray                       # ||dIV/dp|| per param
    corr: np.ndarray
    classes: dict
    degenerate_pairs: list = field(default_factory=list)

    def __str__(self):
        lines = [f"  cond(JᵀJ)={self.cond:.2e}   "
                 f"eigs=" + np.array2string(self.eigvals, formatter={'float': lambda x: f'{x:.2e}'}),
                 "  ||dIV/dp||: " + ", ".join(f"{n}={self.sens[i]:.3g}" for i, n in enumerate(PNAMES)),
                 "  flattest dir: " + ", ".join(f"{n}={self.flat_dir[i]:+.2f}" for i, n in enumerate(PNAMES)),
                 "  classes: " + ", ".join(f"{n}={self.classes[n]}" for n in PNAMES)]
        if self.degenerate_pairs:
            lines.append("  degenerate (|corr|>0.8): "
                         + ", ".join(f"{a}~{b} ({c:+.2f})" for a, b, c in self.degenerate_pairs))
        lines.append("  correlation:")
        for i, n in enumerate(PNAMES):
            lines.append("     " + n.rjust(4) + " "
                         + np.array2string(self.corr[i], formatter={'float': lambda x: f'{x:+.2f}'}))
        return "\n".join(lines)


def sensitivity_jacobian(theta, Ks, *, rel=0.01, abs_=1e-4, cf_kw=None):
    cf_kw = cf_kw or {}
    J = np.zeros((len(Ks), 4))
    for j in range(4):
        h = rel * abs(theta[j]) + abs_
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        J[:, j] = (model_smile_cf(tp, Ks, **cf_kw) - model_smile_cf(tm, Ks, **cf_kw)) / (2 * h)
    return J


def identifiability_report(theta, Ks, *, cf_kw=None):
    J = sensitivity_jacobian(theta, Ks, cf_kw=cf_kw)
    good = np.isfinite(J).all(axis=1)                   # drop any strike whose FD step overflowed
    J = J[good]
    if J.shape[0] < 4:
        raise ValueError(f"identifiability: only {J.shape[0]} finite strikes (<4); raise N_riccati")
    G = J.T @ J
    d = np.sqrt(np.diag(G))
    corr = G / np.outer(d, d)
    eig, Vec = np.linalg.eigh(G)
    flat = Vec[:, 0]                                    # smallest-eigenvalue direction
    classes = {n: ("weak" if abs(flat[i]) > 0.5 else "identified")
               for i, n in enumerate(PNAMES)}
    degen = [(PNAMES[i], PNAMES[j], corr[i, j])
             for i in range(4) for j in range(i + 1, 4) if abs(corr[i, j]) > 0.8]
    return IdentReport(eig[-1] / eig[0], eig, flat, d, corr, classes, degen)


# --------------------------------------------------------------------------- #
# noise-robustness ensemble (parallel over draws)
# --------------------------------------------------------------------------- #
def _ensemble_worker(args):
    seed, sigma, Ks, target, cf_kw = args
    rng = np.random.default_rng(seed)
    noisy = target + rng.standard_normal(len(target)) * sigma
    return calibrate(noisy, Ks, cf_kw=cf_kw).theta_hat


def noise_ensemble(Ks, target_iv, *, sigma_noise, n_draws=24, seed=100,
                   cf_kw=None, parallel=True):
    args = [(seed + d, sigma_noise, Ks, target_iv, cf_kw) for d in range(n_draws)]
    if parallel:
        try:
            import os
            from concurrent.futures import ProcessPoolExecutor
            workers = min(6, (os.cpu_count() or 4))    # cap: each worker = full python+numpy baseline (~200MB)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                recs = np.array(list(ex.map(_ensemble_worker, args)))
        except Exception:
            recs = np.array([_ensemble_worker(a) for a in args])
    else:
        recs = np.array([_ensemble_worker(a) for a in args])
    return dict(mean=recs.mean(0), std=recs.std(0), recs=recs)


# --------------------------------------------------------------------------- #
# the four experiments
# --------------------------------------------------------------------------- #
def _grids(atm_iv):
    return {"FULL": fixed_strikes_from_target(atm_iv, VUS_FULL),
            "PUT+ATM": fixed_strikes_from_target(atm_iv, VUS_PUT)}


def exp_cf_recovery(N_riccati=1500, n_nodes=200):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    atm = atm_iv_cf(TRUTH, **cf_kw)
    print(f"\n=== EXP 1: CF→CF recovery gate (N_riccati={N_riccati}, n_nodes={n_nodes}) ===")
    print(f"  truth: " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(PNAMES, TRUTH)) + f"  σ_ATM={atm:.4f}")
    for tag, Ks in _grids(atm).items():
        target = model_smile_cf(TRUTH, Ks, **cf_kw)
        res = calibrate(target, Ks, theta0=INIT_FAR, cf_kw=cf_kw)
        err = 100 * (res.theta_hat - TRUTH) / np.abs(TRUTH)
        print(f"  [{tag:8s}] init far {INIT_FAR} ->")
        print(f"            recovered {res}")
        print(f"            rel err %: " + ", ".join(f"{n}={err[i]:+.2f}" for i, n in enumerate(PNAMES)))


def exp_identifiability(N_riccati=1000, n_nodes=160):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    atm = atm_iv_cf(TRUTH, **cf_kw)
    print(f"\n=== EXP 3: identifiability (JᵀJ at truth) ===")
    conds = {}
    for tag, Ks in _grids(atm).items():
        rep = identifiability_report(TRUTH, Ks, cf_kw=cf_kw)
        conds[tag] = rep.cond
        print(f"\n  [{tag}]\n{rep}")
    print(f"\n  PUT+ATM worse-conditioned than FULL: "
          f"{conds['PUT+ATM'] > conds['FULL']} ({conds['PUT+ATM']:.2e} vs {conds['FULL']:.2e})")


def exp_noise(sigmas=(0.001, 0.003, 0.005), n_draws=16, N_riccati=1000, n_nodes=96):
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    atm = atm_iv_cf(TRUTH, **cf_kw)
    Ks = fixed_strikes_from_target(atm, VUS_FULL)
    target = model_smile_cf(TRUTH, Ks, **cf_kw)
    print(f"\n=== EXP 2: noise-robustness ensemble (FULL grid, {n_draws} draws) ===")
    print(f"  truth: " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(PNAMES, TRUTH)))
    print(f"  {'σ(pp)':>6} " + " ".join(f"{n+'_mean':>10} {n+'_std':>9}" for n in PNAMES))
    for sg in sigmas:
        e = noise_ensemble(Ks, target, sigma_noise=sg, n_draws=n_draws, cf_kw=cf_kw)
        row = f"  {sg*100:>6.1f} "
        for i in range(4):
            row += f"{e['mean'][i]:>10.4f} {e['std'][i]:>9.4f}"
        print(row)
        rel = e['std'] / np.abs(TRUTH)
        print(f"         rel spread std/|truth|: " + ", ".join(f"{n}={rel[i]:.2f}" for i, n in enumerate(PNAMES)))


def exp_lift(M=40000, n=128, N=80, N_riccati=1500, n_nodes=200):
    """LIFT→CF: target = CF-truth; calibrate with the LIFT in the loop (CRN)."""
    cf_kw = dict(N_riccati=N_riccati, n_nodes=n_nodes)
    lift_kw = dict(M=M, n=n, N=N, seed=7)
    atm = atm_iv_cf(TRUTH, **cf_kw)
    print(f"\n=== EXP 4: LIFT→CF calibration-grade (M={M}, n={n}, N={N}) ===")
    print(f"  truth: " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(PNAMES, TRUTH)))
    for tag, vus in (("FULL", VUS_FULL), ("PUT+ATM", VUS_PUT)):
        Ks = fixed_strikes_from_target(atm, vus)
        target = model_smile_cf(TRUTH, Ks, **cf_kw)        # CF-truth target
        res = calibrate(target, Ks, theta0=INIT_FAR, model=model_smile_lift, cf_kw=lift_kw)
        err = 100 * (res.theta_hat - TRUTH) / np.abs(TRUTH)
        print(f"  [{tag:8s}] lift-calibrated {res}")
        print(f"            distortion %: " + ", ".join(f"{n}={err[i]:+.2f}" for i, n in enumerate(PNAMES)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["cf", "ident", "noise", "lift", "all"], default="cf")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        cf_kw = dict(N_riccati=600, n_nodes=120)
        atm = atm_iv_cf(TRUTH, **cf_kw)
        Ks = fixed_strikes_from_target(atm, VUS_FULL)
        target = model_smile_cf(TRUTH, Ks, **cf_kw)
        res = calibrate(target, Ks, theta0=np.array([0.13, 0.30, -0.60, 0.042]), cf_kw=cf_kw)
        print("quick CF→CF (near-truth init):", res)
        print(identifiability_report(TRUTH, Ks, cf_kw=cf_kw))
    else:
        for e in (["cf", "ident", "noise", "lift"] if a.exp == "all" else [a.exp]):
            {"cf": exp_cf_recovery, "ident": exp_identifiability,
             "noise": exp_noise, "lift": exp_lift}[e]()
