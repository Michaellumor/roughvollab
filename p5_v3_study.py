"""
p5_v3_study.py — P5 v3's development study (docs/protocols/P5_v3_plan.md, "The candidate
sampler (A)" and "The development study"; the author's answers in D91 bind its verdict rule;
D93 commits this script and its settings before the run).

The study runs candidate (A), v3's start and first proposal for the exact reference's
sampler, on the 36 pilot pairs v2's gate G2 saw (D89): the 18 of the first attempt and the
18 of the retry, each y read from v2's committed records, each pair keyed
default_rng([83, i, j]) with i counted within its v2 set, so that the 36 pairs share 18
generator streams. Everything but the start and the first proposal is p5_g2_reference.py as
committed after the pool fix (D92): reference_pair, _run_with_repeat, accept, the
reweighting rule and the direct runs, grid_check, whole_box_check and g2_report. Only
run_chains is replaced, by candidate_run_chains below, installed inside each worker process
(a patch made in the parent does not reach spawned workers, D92's review).

Candidate (A), once per sampler run, inside the run (Decision 2). Its draws from the run's
generator come in this order: one block of 16 × 4 uniforms for the optimiser's starts; one
block of 4 × 4 standard normals for the chains' start perturbations; then the proposals and
acceptance draws, as v2 draws them.
  1. The modes: scipy.optimize.least_squares (method trf, bounds [0, 1]⁴, jac 2-point;
     ftol = xtol = 1e-10, gtol = 1e-8, max_nfev = 2,000) on r(x) = (y − G̃(θ(x)))/s from the 16
     starts; the exit status and the evaluations of each start are recorded. For a direct run
     under prior A or B each kept solution is then polished by L-BFGS-B on
     ½Σr² − log P(x_H), its gradient Jᵀr (J by the bounded 2-point difference, step √ε·max(1,
     |x|), inward at a face) plus the prior's analytic gradient, H's bounds ε = 1e-6 inside the
     faces (ftol = 1e-10, gtol = 1e-5, maxiter = 200).
  2. Kept: in order of log-target from the best, solutions distinct (unit distance above 0.01
     from every solution kept) whose Laplace mass log f + ½ log det Σ is within log 10⁴ of the
     largest mass, at most four; m and the starts reaching each are reported.
  3. Σ = (JᵀJ)⁻¹ from J's singular values, V S⁻² Vᵀ; for a direct run Σ_P = (JᵀJ +
     diag(κ, 0, 0, 0))⁻¹, κ = −∂² log P/∂x_H², from the singular values of J stacked with
     √κ e₁ᵀ; the eigenvalues clipped to [1e-10, 1/12], the count clipped recorded. Σ_A and Σ_B
     at each kept solution are recorded for the ratio estimator's route.
  4. Chain k starts at kept solution k mod m plus a draw from N(0, 4 Σ_s), Σ_s being Σ with its
     eigenvalues capped at 0.0025, clipped to the box ([ε, 1 − ε] in H for a direct run).
  5. The first proposal is 2.38²/4 · Σ of the chain's solution; the adaptation every 500
     burn-in steps, the lengths, the acceptance, the repeat, the reweighting and the direct
     runs, and the storage are v2's.
The ratio estimator's route (Decision 3, the gradient route) is the same candidate given a
callable target with a gradient: the modes by L-BFGS-B from the 16 starts with that gradient,
Σ the pair's Σ_P from the reference record (a preconditioner only). The study runs it on two
stand-in targets per pair and prior: the reference's own log-target under the prior (which
must give accepted runs within G2's tolerances of the least-squares route's figures), and the
same target displaced in H by 0.02 towards the centre of the range; the stand-in's gradient
is the bounded 2-point difference of its log-target in unit coordinates (G3 supplies the
network's autograd gradient); the generators are default_rng([107, p, 0, i, j]) and
default_rng([107, p, 0, i, j, 1]), G3's family with the training seed q = 0.

The runs (--run, the author's, from the merged main): the 36 references at v2's base lengths
(10,000 and 40,000) with the one repeat at doubled lengths, at pool 4; g2_report's grid checks
on every accepted pair (grid_check everywhere, whole_box_check at s = 0.005, v2's tolerances
and the D81 rule) and its projection of the test set's 1,200 references against 1,344
core-hours, the optimiser's seconds inside each run's seconds; at every noise level, on the
whole-box grid, the share of the posterior mass under A outside the first grid's box, compared
with nothing; the stand-ins; then the verdict (pass: all 36 accepted, every comparison within
tolerance, the projection within budget; a failure by its kind, non-adaptation or slow mixing,
a grid miss recomputing the pair's first grid at 64 points per dimension) and the report,
with the 14 pairs v2 failed beside v2's own attempts. Nothing here runs a gate; the records
are v3's development, declared under line 198 of the frozen draft.

Usage (from the repository root):
    py -3 p5_v3_study.py                 the design, the settings, a cost estimate; holds
    py -3 p5_v3_study.py --run           the study (the author's); options --pool K (default 4),
                                         --emulator PATH (G̃ by default; another file is refused),
                                         --allow-other-versions (recorded)

Outputs (output/ is git-ignored; the files are force-added by name after the run, D94):
    output/p5v3_study.jsonl              header, launches, the 36 pair records, the mass and stand-in lines
    output/p5v3_study_samples.npz        the quantised H samples of every accepted run
    output/p5v3_study_checks.json        g2_report's grid checks and projection (its "gate" key is the function's label)
    output/p5v3_study_report.json        the verdict and the report
Never v2's paths (output/p5_g2_*, output/p5_g2r_*) or the names reserved for v3's G2 (output/p5v3_g2_*).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (p5_g2_reference.py)

import sys
import json
import math
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import eigh as _geigh

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g2_reference as R
import p5_g1_emulator as E
from p5_g0_pilot import git_state, utc_now, append_line, _rel, read_jsonl
from p5_g1_emulator import BOX, DIMS, NOISE_LEVELS, box_arrays

# --------------------------------------------------------------------------- #
# The candidate's constants (the plan, "The candidate sampler (A)", steps 1 to 5): the study's
# settings, committed before the run and written into the record's header
# --------------------------------------------------------------------------- #
N_STARTS = 16                                       # optimiser starts, uniform on [0, 1]⁴ (step 1)
LS_METHOD, LS_JAC = "trf", "2-point"                # least_squares (step 1)
LS_FTOL, LS_XTOL, LS_GTOL, LS_MAX_NFEV = 1e-10, 1e-10, 1e-8, 2000   # its tolerances and evaluation cap, declared (D93)
FD_REL_STEP = math.sqrt(np.finfo(float).eps)        # the 2-point step √ε·max(1, |x|), least_squares' own convention
LBFGS_FTOL, LBFGS_GTOL, LBFGS_MAXITER = 1e-10, 1e-5, 200   # the polish and the gradient route (L-BFGS-B), declared (D93)
EPS_H = 1e-6                                        # H's bounds for a direct run or a prior-weighted target (step 1)
KAPPA_U_FLOOR = 0.01                                # the prior's density and curvature for the keep rule, and for the recorded Σ_A and Σ_B,
#                                                     are taken at x_H clipped to [0.01, 0.99]: a solution on a face records no degenerate Σ_P
KEEP_MAX, DISTINCT, MASS_LOG = 4, 0.01, math.log(1e4)   # step 2: at most four, distinct beyond 0.01, within log 10⁴ of the mass
EIG_LO, EIG_HI = 1e-10, 1.0 / 12.0                  # step 3: Σ's eigenvalues clipped to this range
PERTURB_CAP, PERTURB_MULT = 0.0025, 4.0             # step 4: Σ_s caps the eigenvalues; the draw is N(0, 4 Σ_s)
SCALE = R.SCALE                                     # step 5: 2.38²/4 (v2's)
BURN, SAMPLE = R.BURN, R.SAMPLE                     # v2's base lengths, 10,000 and 40,000; the repeat doubles them
N_CHAINS, ADAPT_EVERY = R.N_CHAINS, R.ADAPT_EVERY   # 4 chains, adaptation every 500 burn-in steps (v2's)

# The stand-ins (Decision 3): the gradient route on the reference's own log-target
SEED_STANDIN = 107                                  # G3's SEED_NRE: default_rng([107, p, q, i, j]) (line 75, D81)
PRIOR_INDEX = {"A": 1, "B": 2}                      # p5_g3_calibrators.PRIOR_INDEX
STANDIN_Q = 0                                       # no training seed: the stand-in's q
DISPLACE_H = 0.02                                   # the displaced target: H shifted by this, towards the centre of the range
STANDIN_VARIANTS = ("own", "displaced")

# The verdict (the plan, "The verdict on (A)"), binding by D91
NONADAPT_ACC, NONADAPT_RHAT = 0.01, 1.1             # non-adaptation: acceptance below 1%, or R-hat above 1.1
GRID_MISS_N = 64                                    # a grid miss: the first grid recomputed at 64 points per dimension
SEALED_SEEDS = (R.SEED_TEST, R.SEED_TEST_NOISE)     # 20261002, 20261022: never drawn, refused wherever a seed is read

TASK_KWARGS = {"burn": BURN, "sample": SAMPLE}      # the lengths in force (v2's base); the acceptance and the Kish floor are the reference's
GRID_N, WHOLE_N = R.GRID_N, R.WHOLE_N               # G2's grids, 32⁴ on the support and 48³ × 32 over the whole box (line 143)
POOL_DEFAULT = 4
OUT_DIR = R.OUT_DIR
SET_NAME = "p5v3_study"
PATHS = {"jsonl": os.path.join(OUT_DIR, "p5v3_study.jsonl"),
         "samples": os.path.join(OUT_DIR, "p5v3_study_samples.npz"),
         "checks": os.path.join(OUT_DIR, "p5v3_study_checks.json"),
         "report": os.path.join(OUT_DIR, "p5v3_study_report.json")}
SOURCES = {"g2": R.SETS["g2"]["jsonl"], "g2r": R.SETS["g2r"]["jsonl"]}      # v2's committed records (D89), read only
SOURCE_REPORTS = {"g2": R.SETS["g2"]["report"], "g2r": R.SETS["g2r"]["report"]}
N_PAIRS_EXPECTED = 36


def candidate_constants():
    """The constants of (A) as the record's header carries them (also inside `lengths`, so
    that a resume with other constants is refused by launch)."""
    return {"n_starts": N_STARTS, "ls_method": LS_METHOD, "ls_jac": LS_JAC, "ls_ftol": LS_FTOL, "ls_xtol": LS_XTOL,
            "ls_gtol": LS_GTOL, "ls_max_nfev": LS_MAX_NFEV, "fd_rel_step": FD_REL_STEP, "lbfgs_ftol": LBFGS_FTOL,
            "lbfgs_gtol": LBFGS_GTOL, "lbfgs_maxiter": LBFGS_MAXITER, "eps_H": EPS_H, "keep_max": KEEP_MAX,
            "distinct": DISTINCT, "mass_log": MASS_LOG, "eig_lo": EIG_LO, "eig_hi": EIG_HI, "perturb_cap": PERTURB_CAP,
            "perturb_mult": PERTURB_MULT, "kappa_u_floor": KAPPA_U_FLOOR, "scale": SCALE, "n_chains": N_CHAINS, "adapt_every": ADAPT_EVERY,
            "draw_order": "16x4 uniform starts, 4x4 standard normal perturbations, then proposals and acceptance draws"}


def standin_constants():
    return {"seed": SEED_STANDIN, "prior_index": dict(PRIOR_INDEX), "q": STANDIN_Q, "displace_H": DISPLACE_H,
            "variants": list(STANDIN_VARIANTS), "gradient": "bounded 2-point finite difference of the log-target in unit "
                                                            "coordinates, step fd_rel_step (G3 supplies autograd)",
            "sigma": "the pair's Σ_P from its record: the direct run's at its best solution when the pair's figures under "
                     "the prior are direct, else the uniform run's with the prior's curvature added"}


def refuse_sealed(seeds, where):
    """Refuses when any integer among `seeds` (nested lists and dicts allowed) is a sealed seed."""
    def ints(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from ints(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from ints(x)
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            yield int(v)
    bad = sorted({v for v in ints(seeds) if v in SEALED_SEEDS})
    if bad:
        raise SystemExit(f"{where} refused: the sealed seed{'s' if len(bad) > 1 else ''} {bad} must never be drawn "
                         "(line 93; the plan, 'What v3 does not change')")


# --------------------------------------------------------------------------- #
# The pieces of (A): the prior's derivatives, the residuals, the bounded difference, Σ
# --------------------------------------------------------------------------- #
def prior_grad_curv(u, which):
    """(d/du log P(u), −d²/du² log P(u)) for prior `which` at u = x_H, the scaled Beta's
    argument (p5_g2_reference.prior_logpdf drops the constant Jacobian): (0, 0) for 'uniform'."""
    if which == "uniform":
        return 0.0, 0.0
    a, b = R.PRIOR_AB[which]
    u = float(u)
    return (a - 1.0) / u - (b - 1.0) / (1.0 - u), (a - 1.0) / u ** 2 + (b - 1.0) / (1.0 - u) ** 2


def residuals(x, y, s, em, box=BOX):
    """r(x) = (y − G̃(θ(x)))/s at one unit-cube point, so that ½Σr² = −log L."""
    th = R.unit_to_theta(np.clip(np.asarray(x, dtype=float), 0.0, 1.0), box)
    return (np.asarray(y, dtype=float) - em.evaluate_many(th.reshape(1, 4))[0]) / float(s)


def fd_jacobian(fun, x, lo, hi, rel_step=FD_REL_STEP):
    """The 2-point finite-difference Jacobian of `fun` at x (a vector of residuals, or a
    scalar): the step rel_step·max(1, |x_i|) forward, or backward where the forward point
    would leave [lo, hi], and the actual difference (x_i + h) − x_i as the divisor, as
    scipy's approx_derivative does. Returns (J, fun(x))."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(fun(x), dtype=float))
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = rel_step * max(1.0, abs(x[i]))
        if x[i] + h > hi[i]:
            h = -h
        xi = x.copy()
        xi[i] += h
        J[:, i] = (np.atleast_1d(np.asarray(fun(xi), dtype=float)) - f0) / (xi[i] - x[i])
    return J, f0


def cov_from_jacobian(J, kappa=0.0, eig_lo=EIG_LO, eig_hi=EIG_HI):
    """Σ = (JᵀJ + diag(κ, 0, 0, 0))⁻¹ from the singular values of J stacked with the row √κ e₁ᵀ
    (κ = 0: J alone), as V S⁻² Vᵀ, never by inverting JᵀJ; the eigenvalues clipped to
    [eig_lo, eig_hi]. Returns (Σ, the eigenvalues before clipping, the number clipped)."""
    Ja = np.asarray(J, dtype=float)
    if kappa > 0.0:
        Ja = np.vstack([Ja, np.sqrt(kappa) * np.eye(1, Ja.shape[1])])
    if Ja.shape[0] < Ja.shape[1]:
        raise ValueError("cov_from_jacobian: J needs at least as many residuals as parameters")
    _, S, Vt = np.linalg.svd(Ja, full_matrices=False)                # S has one value per parameter
    with np.errstate(divide="ignore"):
        eig = np.where(S > 0.0, 1.0 / S ** 2, np.inf)                # a zero singular value: a flat direction, clipped to eig_hi
    clipped = np.clip(eig, eig_lo, eig_hi)
    cov = (Vt.T * clipped) @ Vt
    return 0.5 * (cov + cov.T), eig, int(np.sum((eig < eig_lo) | (eig > eig_hi)))


def capped_cov(cov, cap=PERTURB_CAP):
    """Σ_s: Σ with its eigenvalues capped at `cap` (step 4)."""
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    return (V * np.minimum(w, cap)) @ V.T


def keep_solutions(xs, logf, covs, keep_max=KEEP_MAX, distinct=DISTINCT, mass_log=MASS_LOG):
    """Step 2. xs [n, 4] the solutions, logf [n] their log-targets, covs [n, 4, 4] their
    (clipped) Σ. In order of log-target from the best: a solution is kept when its unit
    distance to every solution already kept exceeds `distinct` and its Laplace mass,
    log f + ½ log det Σ, is within `mass_log` of the largest mass among the solutions, up to
    `keep_max` of them. Returns (kept indices, masses [n], reached [n]: the position in the
    kept list of the kept solution within `distinct` of solution i, or −1)."""
    xs, logf = np.asarray(xs, dtype=float), np.asarray(logf, dtype=float)
    mass = np.array([lf + 0.5 * np.linalg.slogdet(c)[1] if np.isfinite(lf) else -np.inf for lf, c in zip(logf, covs)])
    best = np.max(mass) if np.any(np.isfinite(mass)) else -np.inf
    kept = []
    for i in np.argsort(-logf, kind="stable"):
        if len(kept) >= keep_max or not np.isfinite(mass[i]):
            break
        if any(np.linalg.norm(xs[i] - xs[k]) <= distinct for k in kept):
            continue
        if mass[i] >= best - mass_log:
            kept.append(int(i))
    reached = np.full(len(xs), -1, dtype=int)
    for i in range(len(xs)):
        for pos, k in enumerate(kept):
            if np.linalg.norm(xs[i] - xs[k]) <= distinct:
                reached[i] = pos
                break
    return kept, mass, reached


def chain_starts(solutions, covs, z, lo_b, hi_b, mult=PERTURB_MULT, cap=PERTURB_CAP):
    """Step 4: chain k at solution k mod m plus _chol_pd(mult·Σ_s) z_k, clipped to [lo_b, hi_b].
    Returns (starts [n_chains, 4], the unclipped starts, the number of coordinates clipped,
    the solution index of each chain)."""
    m = len(solutions)
    raw = np.empty_like(np.asarray(z, dtype=float))
    which = [k % m for k in range(raw.shape[0])]
    for k in range(raw.shape[0]):
        Ls = R._chol_pd(mult * capped_cov(covs[which[k]], cap))
        raw[k] = np.asarray(solutions[which[k]], dtype=float) + Ls @ np.asarray(z[k], dtype=float)
    x = np.clip(raw, lo_b, hi_b)
    return x, raw, int(np.sum(x != raw)), which


def _lbfgs(f_and_g, x0, lo_b, hi_b):
    x0 = np.clip(np.asarray(x0, dtype=float), lo_b, hi_b)
    res = minimize(f_and_g, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo_b, hi_b)),
                   options={"ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL, "maxiter": LBFGS_MAXITER})
    return res


def _modes_least_squares(starts, y, s, em, prior, box, lo_b, hi_b):
    """Steps 1 to 3 by the least-squares route: least_squares from each start on the box
    [0, 1]⁴; the keep rule on log f = −cost and Σ = (JᵀJ)⁻¹; for a direct run the polish of
    each kept solution by L-BFGS-B on ½Σr² − log P(x_H) within [lo_b, hi_b], and Σ_P at the
    polished solution."""
    fun = lambda x: residuals(x, y, s, em, box)                              # noqa: E731
    sols, costs, Js, status, nfev = [], [], [], [], []
    for x0 in starts:
        res = least_squares(fun, np.clip(x0, 0.0, 1.0), jac=LS_JAC, bounds=(np.zeros(4), np.ones(4)), method=LS_METHOD,
                            ftol=LS_FTOL, xtol=LS_XTOL, gtol=LS_GTOL, max_nfev=LS_MAX_NFEV)
        sols.append(res.x)
        costs.append(float(res.cost))
        Js.append(np.asarray(res.jac, dtype=float))
        status.append(int(res.status))
        nfev.append(int(res.nfev))
    sols, cost = np.asarray(sols), np.asarray(costs)
    u_keep = np.clip(sols[:, 0], KAPPA_U_FLOOR, 1.0 - KAPPA_U_FLOOR)       # the prior's density and curvature for the keep rule
    if prior == "uniform":
        logf = -cost
        covs = [cov_from_jacobian(J)[0] for J in Js]
    else:                                                                  # the run's log-target, −½Σr² + log P(x_H) (step 2)
        logf = -cost + np.array([float(R.prior_logpdf(R.H_LO + u * R.H_W, prior)) for u in u_keep])
        covs = [cov_from_jacobian(J, prior_grad_curv(u, prior)[1])[0] for J, u in zip(Js, u_keep)]
    if not np.any(np.isfinite(logf)):
        raise RuntimeError(f"candidate (A): no finite least-squares solution from {N_STARTS} starts")
    kept_idx, mass, reached = keep_solutions(sols, logf, covs)
    kept, pos_map, n_merged = [], {}, 0
    for pos, i in enumerate(kept_idx):
        x, J, polish, lf = sols[i], Js[i], None, float(logf[i])
        if prior != "uniform":                                                 # the polish (step 1, direct runs)
            def f_and_g(xx):
                Jx, r = fd_jacobian(fun, xx, lo_b, hi_b)
                g1, _ = prior_grad_curv(xx[0], prior)
                grad = Jx.T @ r
                grad[0] -= g1
                return 0.5 * float(r @ r) - float(R.prior_logpdf(R.unit_to_theta(xx, box)[0], prior)), grad
            res = _lbfgs(f_and_g, x, lo_b, hi_b)
            x = np.asarray(res.x, dtype=float)
            J, r = fd_jacobian(fun, x, lo_b, hi_b)
            lf = -0.5 * float(r @ r) + float(R.prior_logpdf(R.unit_to_theta(x, box)[0], prior))   # the log-target at the polished point
            polish = {"status": int(res.status), "nit": int(res.nit), "fun": float(res.fun), "message": str(res.message)}
            same = [q for q, k_ in enumerate(kept) if np.linalg.norm(x - k_["x"]) <= DISTINCT]
            if same:                                                           # two solutions polished to one point: one kept
                pos_map[pos] = same[0]
                n_merged += 1
                continue
        pos_map[pos] = len(kept)
        u_rec = min(max(float(x[0]), KAPPA_U_FLOOR), 1.0 - KAPPA_U_FLOOR)      # Σ_A and Σ_B recorded with κ at a floored x_H
        kap = {p: prior_grad_curv(u_rec, p)[1] for p in ("A", "B")}
        cov, eig, n_clip = cov_from_jacobian(J, prior_grad_curv(float(x[0]), prior)[1] if prior != "uniform" else 0.0)
        kept.append({"x": x, "logf": lf, "logf_ls": float(-cost[i]), "mass_ls": float(mass[i]), "eig": eig, "n_clipped": n_clip,
                     "sigma": cov, "sigma_A": cov_from_jacobian(J, kap["A"])[0], "sigma_B": cov_from_jacobian(J, kap["B"])[0],
                     "n_starts_reached": 0, "polish": polish})
    reached = np.array([pos_map[int(r_)] if r_ >= 0 else -1 for r_ in reached], dtype=int)
    for q, k_ in enumerate(kept):
        k_["n_starts_reached"] = int(np.sum(reached == q))
    return {"route": "least_squares", "kept": kept, "starts_elsewhere": int(np.sum(reached < 0)), "n_merged_after_polish": n_merged,
            "optimiser": {"status": status, "nfev": nfev, "cost": costs, "reached": [int(v) for v in reached]}}


def _modes_gradient(starts, target, box, lo_b, hi_b):
    """Steps 1 to 3 by the gradient route (Decision 3): L-BFGS-B from each start on
    −target with the target's gradient in unit coordinates; the keep rule with Σ =
    target.sigma for every solution (so the mass rule reduces to log f within log 10⁴); Σ
    the record's Σ_P throughout, a preconditioner only."""
    for name in ("grad_unit", "sigma"):
        if not hasattr(target, name):
            raise TypeError(f"the gradient route needs a target with `{name}`; a plain callable is not run with the candidate")
    b_lo, b_hi = getattr(target, "bounds_unit", (np.zeros(4), np.ones(4)))
    lo_b, hi_b = np.maximum(lo_b, b_lo), np.minimum(hi_b, b_hi)
    sigma, eig, n_clip = clip_cov(np.asarray(target.sigma, dtype=float))

    def f_and_g(xx):
        f = float(target(R.unit_to_theta(xx, box).reshape(1, 4))[0])
        return -f, -np.asarray(target.grad_unit(xx), dtype=float)
    sols, fs, status, nit = [], [], [], []
    for x0 in starts:
        res = _lbfgs(f_and_g, x0, lo_b, hi_b)
        sols.append(np.asarray(res.x, dtype=float))
        fs.append(float(res.fun))
        status.append(int(res.status))
        nit.append(int(res.nit))
    sols, logf = np.asarray(sols), -np.asarray(fs)
    if not np.any(np.isfinite(logf)):
        raise RuntimeError(f"candidate (A), gradient route: no finite solution from {N_STARTS} starts")
    kept_idx, mass, reached = keep_solutions(sols, logf, [sigma] * len(sols))
    kept = [{"x": sols[i], "logf": float(logf[i]), "logf_ls": None, "mass_ls": float(mass[i]), "eig": eig, "n_clipped": n_clip,
             "sigma": sigma, "sigma_A": None, "sigma_B": None, "n_starts_reached": int(np.sum(reached == pos)), "polish": None}
            for pos, i in enumerate(kept_idx)]
    return {"route": "gradient", "kept": kept, "starts_elsewhere": int(np.sum(reached < 0)), "n_merged_after_polish": 0,
            "optimiser": {"status": status, "nit": nit, "fun": fs, "reached": [int(v) for v in reached]},
            "bounds_unit": (lo_b, hi_b)}


def clip_cov(sigma, eig_lo=EIG_LO, eig_hi=EIG_HI):
    """A given Σ (the record's Σ_P) with its eigenvalues clipped to [eig_lo, eig_hi], as the
    candidate clips its own: (Σ, eigenvalues before clipping, the number clipped)."""
    w, V = np.linalg.eigh(np.asarray(sigma, dtype=float))
    clipped = np.clip(w, eig_lo, eig_hi)
    cov = (V * clipped) @ V.T
    return 0.5 * (cov + cov.T), w, int(np.sum((w < eig_lo) | (w > eig_hi)))


def _proposal_vs_sample(cov, samples):
    """Per chain, the sampling proposal's covariance against the chain's sample covariance:
    the generalised eigenvalues of (C_prop, C_sample), about 2.38²/4 if the adaptation is
    clean, and the ratio of traces; None where the sample covariance is not positive definite."""
    out = []
    for k in range(samples.shape[0]):
        cs = np.atleast_2d(np.cov(samples[k].T, ddof=1))
        try:
            np.linalg.cholesky(cs)
            ge = _geigh(cov[k], cs, eigvals_only=True)
            out.append({"gen_eig": [float(v) for v in ge], "trace_ratio": float(np.trace(cov[k]) / np.trace(cs))})
        except (np.linalg.LinAlgError, ValueError):
            out.append(None)
    return out


# --------------------------------------------------------------------------- #
# Candidate (A): run_chains with the anchored starts and the preconditioned first proposal
# --------------------------------------------------------------------------- #
_DIAG = []                                          # this process's candidate diagnostics, drained by _study_worker


def candidate_run_chains(y, s, em, rng, prior="uniform", burn=BURN, sample=SAMPLE, n_chains=N_CHAINS,
                         adapt_every=ADAPT_EVERY, box=BOX, target=None):
    """Candidate (A) in place of p5_g2_reference.run_chains: the same signature and return
    (samples [n_chains, sample, 4] in unit coordinates, the acceptance over the sampling
    steps, the final proposal covariances, the lengths), plus `candidate`, the run's
    diagnostics (also appended to this process's list). The draws from `rng`: the 16 × 4
    uniform starts, the n_chains × 4 standard normals, then mh_step's, as v2. With `target`
    (a callable of θ in box units with `grad_unit(x)` and `sigma`, see StandInTarget) the
    modes come from the gradient route."""
    d = 4
    t0 = time.perf_counter()
    starts = rng.uniform(size=(N_STARTS, d))                         # block 1
    z = rng.standard_normal((n_chains, d))                           # block 2
    lo_b, hi_b = np.zeros(d), np.ones(d)
    if prior != "uniform" or target is not None:
        lo_b[0], hi_b[0] = EPS_H, 1.0 - EPS_H                        # the prior's density vanishes at the faces
    if target is None:
        opt = _modes_least_squares(starts, y, s, em, prior, box, lo_b, hi_b)
    else:
        opt = _modes_gradient(starts, target, box, lo_b, hi_b)
        lo_b, hi_b = opt["bounds_unit"]
    kept = opt["kept"]
    m = len(kept)
    if m == 0:
        raise RuntimeError("candidate (A): no solution kept; the run is recorded as an error, not run with v2's start")
    x, raw, n_clip, which = chain_starts([k_["x"] for k_ in kept], [k_["sigma"] for k_ in kept], z, lo_b, hi_b)
    opt_seconds = time.perf_counter() - t0
    lt = R.log_target(x, y, s, em, prior, box, target)
    cov = np.stack([SCALE * kept[which[k]]["sigma"] for k in range(n_chains)])      # step 5: the first proposal
    L = np.stack([R._chol_pd(c) for c in cov])
    hist = np.empty((n_chains, burn + 1, d))
    hist[:, 0] = x
    acc_first = np.zeros(n_chains)
    n_first = min(burn, adapt_every)
    for t in range(1, burn + 1):                                     # v2's burn-in and adaptation, unchanged
        x, lt, acc = R.mh_step(x, lt, L, y, s, em, prior, rng, box, target)
        hist[:, t] = x
        if t <= n_first:
            acc_first += acc
        if t % adapt_every == 0:
            cov = np.stack([R.adapted_cov(hist[k, :t + 1]) for k in range(n_chains)])
            L = np.stack([R._chol_pd(c) for c in cov])
    samples = np.empty((n_chains, sample, d))
    n_acc = 0
    for t in range(sample):                                          # v2's sampling phase, unchanged
        x, lt, acc = R.mh_step(x, lt, L, y, s, em, prior, rng, box, target)
        samples[:, t] = x
        n_acc += int(acc.sum())
    diag = {"sampler": "A", "route": opt["route"], "prior": prior, "burn": int(burn), "sample": int(sample),
            "n_chains": int(n_chains), "optimiser_seconds": opt_seconds, "n_starts": N_STARTS, "starts": starts,
            "optimiser": opt["optimiser"], "m": m, "starts_elsewhere": opt["starts_elsewhere"], "kept": kept,
            "n_merged_after_polish": opt["n_merged_after_polish"],
            "chain_solution": which, "chain_start_raw": raw, "chain_start": hist[:, 0].copy(),
            "n_start_clipped": n_clip, "bounds_unit": [lo_b, hi_b],
            "acceptance_first_%d" % adapt_every: (acc_first / float(n_first)) if n_first else None,
            "proposal_vs_sample": _proposal_vs_sample(cov, samples) if sample > 1 else None}
    _DIAG.append(diag)
    return {"samples": samples, "acceptance": n_acc / float(n_chains * sample), "cov": cov,
            "burn": int(burn), "sample": int(sample), "prior": prior, "candidate": diag}


_V2_RUN_CHAINS = R.run_chains                       # v2's sampler, for the tests and the uninstall
_R_WORKER = R._worker                               # p5_g2_reference's worker, which the study's worker wraps
SAMPLER_TAG = "A (p5_v3_study.candidate_run_chains)"   # every pair record of the study carries it; the verdict requires it


def install_candidate():
    """Sets p5_g2_reference.run_chains to the candidate in this process (idempotent).
    _run_with_repeat looks run_chains up as the module's global, so every run, repeat and
    direct run then uses it."""
    if R.run_chains is not candidate_run_chains:
        R.run_chains = candidate_run_chains
    return R.run_chains


def uninstall_candidate():
    R.run_chains = _V2_RUN_CHAINS


def _jsonable(v):
    """numpy arrays and scalars to lists and Python numbers; inf and nan to None (the record
    is written with allow_nan=False)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _study_worker(args):
    """The study's pool worker, a module-level function: in its own process it installs the
    candidate, then calls p5_g2_reference's worker, and attaches the candidate's diagnostics
    of every run of the pair to the record. Under spawn the executor pickles it by name, so
    the worker process installs the candidate itself (D92's review)."""
    install_candidate()
    del _DIAG[:]
    rec, stores = _R_WORKER(args)
    if len(_DIAG) != len(R.pair_attempts(rec)):                      # the candidate ran every sampler run of the pair, or nothing is recorded
        raise RuntimeError(f"the candidate ran {len(_DIAG)} of the pair's {len(R.pair_attempts(rec))} sampler runs")
    rec["candidate"] = _jsonable(_DIAG)
    rec["sampler"] = SAMPLER_TAG
    job = args[0]
    if "v2_status" in job:
        rec["v2_status"] = job["v2_status"]
    del _DIAG[:]
    return rec, stores


# --------------------------------------------------------------------------- #
# The stand-in targets (Decision 3)
# --------------------------------------------------------------------------- #
class StandInTarget:
    """The reference's own log-target under prior `prior`, log P(H) + log L(θ; y, s), as a
    callable of θ rows in box units for the sampler's `target`, with the gradient route's
    attributes: grad_unit(x), the bounded 2-point finite difference of the log-target in unit
    coordinates; sigma, the pair's Σ_P from its record; bounds_unit, the support in unit
    coordinates. With `shift` ≠ 0 the whole target is displaced in H: g(θ) = f(θ − shift·e_H),
    −inf where θ − shift·e_H leaves the box."""

    def __init__(self, y, s, em, prior, sigma, shift=0.0, box=BOX):
        self.y, self.s, self.em, self.prior, self.shift, self.box = np.asarray(y, dtype=float), float(s), em, prior, float(shift), box
        self.sigma = np.asarray(sigma, dtype=float)
        lo, hi = box_arrays(box)
        du = self.shift / (hi[0] - lo[0])
        self.bounds_unit = (np.array([max(0.0, du) + EPS_H, 0.0, 0.0, 0.0]), np.array([min(1.0, 1.0 + du) - EPS_H, 1.0, 1.0, 1.0]))

    def __call__(self, th):
        th = np.asarray(th, dtype=float).reshape(-1, 4)
        lo, hi = box_arrays(self.box)
        th_l = th.copy()
        th_l[:, 0] -= self.shift
        tol = 1e-12 * (hi - lo)
        inside = np.all((th_l >= lo - tol) & (th_l <= hi + tol), axis=1)
        out = np.full(th.shape[0], -np.inf)
        if inside.any():
            th_in = np.clip(th_l[inside], lo, hi)
            out[inside] = R.log_lik(th_in, self.y, self.s, self.em) + R.prior_logpdf(th_in[:, 0], self.prior)
        return out

    def grad_unit(self, x):
        f = lambda xx: self(R.unit_to_theta(xx, self.box).reshape(1, 4))[0]        # noqa: E731
        J, _ = fd_jacobian(f, x, self.bounds_unit[0], self.bounds_unit[1])
        return J[0]


def sigma_P_from_record(rec, prior):
    """The pair's Σ_P under `prior` from its study record: the accepted direct run's Σ at its
    best kept solution when the pair's figures under the prior are direct, else the accepted
    uniform run's Σ with the prior's curvature added (sigma_A or sigma_B)."""
    diags = rec.get("candidate") or []
    if rec["under"][prior]["source"] == "direct":
        runs = [d for d in diags if d["prior"] == prior]
        return np.asarray(runs[-1]["kept"][0]["sigma"], dtype=float)
    runs = [d for d in diags if d["prior"] == "uniform"]
    return np.asarray(runs[-1]["kept"][0]["sigma_" + prior], dtype=float)


def standin_key(prior, i, j, variant):
    key = [SEED_STANDIN, PRIOR_INDEX[prior], STANDIN_Q, int(i), int(j)] + ([1] if variant == "displaced" else [])
    refuse_sealed(key, "stand-in")
    return key


def standin_shift(rec, prior, variant):
    """The displaced target's shift: DISPLACE_H towards the centre of H's range from the
    pair's posterior mean under the prior; 0 for the own target."""
    if variant != "displaced":
        return 0.0
    lo, hi = BOX["H"]
    return DISPLACE_H if rec["under"][prior]["mean_H"] < 0.5 * (lo + hi) else -DISPLACE_H


def standin_run(job, rec, prior, variant, em, burn=BURN, sample=SAMPLE, rhat_max=R.RHAT_MAX, ess_min=R.ESS_MIN):
    """One stand-in run: the gradient route on the pair's own log-target under `prior`
    ('own'), or on it displaced in H ('displaced'), by _run_with_repeat with the candidate
    installed. Returns the record line: the attempts, the accepted run's figures, and for
    'own' the comparison with the record's figures under the prior by G2's tolerance rule
    (R._compare; the keys compared are the prior's own: mean_H, c and the quantiles under A,
    mean_H under B)."""
    install_candidate()
    del _DIAG[:]
    shift = standin_shift(rec, prior, variant)
    sigma = sigma_P_from_record(rec, prior)
    tgt = StandInTarget(job["y"], job["s"], em, prior, sigma, shift)
    key = standin_key(prior, job["i"], job["j"], variant)
    rng = np.random.default_rng(key)
    t0 = time.perf_counter()
    chains, attempts = R._run_with_repeat(None, None, None, rng, "uniform", burn, sample, BOX, rhat_max, ess_min, target=tgt)
    if len(_DIAG) != len(attempts) or any(d["route"] != "gradient" for d in _DIAG):   # the candidate's gradient route ran, or nothing is recorded
        raise RuntimeError("the stand-in was not run by the candidate's gradient route")
    line = {"kind": "standin", "task": job["label"], "set": job["set"], "i": int(job["i"]), "j": int(job["j"]), "s": float(job["s"]),
            "prior": prior, "variant": variant, "shift_H": shift, "seed": key, "attempts": attempts,
            "accepted": chains is not None, "sigma_P": _jsonable(sigma), "candidate": _jsonable(_DIAG),
            "figures": None, "compare": None, "seconds": None, "utc": None}
    del _DIAG[:]
    if chains is not None:
        pooled = R.unit_to_theta(chains["samples"].reshape(-1, 4))
        w = np.ones(pooled.shape[0])
        fig = R.summaries(pooled[:, 0], pooled[:, 3], w)
        fig["c"] = R.contraction(fig["var_H"])
        line["figures"] = fig
        if variant == "own":
            under = rec["under"]
            g = {"A": {"mean_H": under["A"]["mean_H"], "q05_H": under["A"]["q05_H"], "q95_H": under["A"]["q95_H"]},
                 "B": {"mean_H": under["B"]["mean_H"]}, "c_A": rec["c_A"]}
            if prior == "A":
                g["A"] = {"mean_H": fig["mean_H"], "q05_H": fig["q05_H"], "q95_H": fig["q95_H"]}
                g["c_A"] = fig["c"]
                keys = ["d_mean", "d_c", "d_q05", "d_q95"]
            else:
                g["B"] = {"mean_H": fig["mean_H"]}
                keys = ["d_mean_B"]
            cmp_ = R._compare(rec, g, R.TOL_MEAN, R.TOL_C, R.TOL_Q)
            # both figures are Monte Carlo estimates of one posterior, so the tolerance's Monte Carlo term is on their
            # difference: the larger of the absolute value and tol_mc_k × √(se_ref² + se_standin²), the stand-in's standard
            # errors by mc_errors on its own accepted run (its ESS of H, its variance of H), as for a direct run
            ess_s = float(attempts[-1]["ess_H"])
            own_rec = {"under": {"A": {"source": "direct", "var_H": fig["var_H"]}, "B": {"source": "direct", "var_H": fig["var_H"]}},
                       "runs": {"A": [{"ess_H": ess_s}], "B": [{"ess_H": ess_s}]}, "c_A": fig["c"], "neff": {}}
            se_s = R.mc_errors(own_rec)
            absolute = {"d_mean": R.TOL_MEAN, "d_c": R.TOL_C, "d_mean_B": R.TOL_MEAN, "d_q05": R.TOL_Q, "d_q95": R.TOL_Q}
            tol = {k: max(absolute[k], R.TOL_MC_K * math.sqrt(cmp_["mc_se"][k] ** 2 + se_s[k] ** 2)) for k in keys}
            line["compare"] = {"keys": keys, "d": {k: cmp_[k] for k in keys}, "tol": tol,
                               "mc_se_ref": {k: cmp_["mc_se"][k] for k in keys}, "mc_se_standin": {k: float(se_s[k]) for k in keys},
                               "tol_ref_only": {k: cmp_["tol"][k] for k in keys},
                               "passed": bool(all(cmp_[k] <= tol[k] for k in keys))}
        else:
            line["compare"] = {"mean_H_shift": fig["mean_H"] - rec["under"][prior]["mean_H"], "shift_H": shift}
    line["seconds"] = time.perf_counter() - t0
    line["utc"] = utc_now()
    return line


def _standin_job(args):
    job, rec, prior, variant, kw = args
    em = R.get_emulator(kw.pop("emulator_path", None))
    return standin_run(job, rec, prior, variant, em, **kw)


# --------------------------------------------------------------------------- #
# The mass diagnostic: the posterior mass under A outside the first grid's box
# --------------------------------------------------------------------------- #
def mass_outside(grid_lo, grid_hi, xi0_axis, y, s, em, n=R.WHOLE_N, box=BOX, prior="A", chunk=8192):
    """On the whole-box grid of line 143 (n cell-centred points in each of H, ν and ρ over the
    whole box, ξ₀ on the first grid's points), the share of the posterior mass under `prior`
    in the cells whose (H, ν, ρ) centre lies outside the first grid's box [grid_lo, grid_hi].
    Compared with nothing."""
    lo, hi = box_arrays(box)
    axes = [R.cell_centres(lo[k], hi[k], n) for k in range(3)] + [np.asarray(xi0_axis, dtype=float)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    logl = np.empty(mesh.shape[0])
    for a in range(0, mesh.shape[0], chunk):
        logl[a:a + chunk] = R.log_lik(mesh[a:a + chunk], y, s, em)
    logw = logl + R.prior_logpdf(mesh[:, 0], prior)
    w = np.exp(logw - np.max(logw))
    glo, ghi = np.asarray(grid_lo, dtype=float), np.asarray(grid_hi, dtype=float)
    inside = np.all((mesh[:, :3] >= glo[:3]) & (mesh[:, :3] <= ghi[:3]), axis=1)
    total = float(np.sum(w))
    return {"share_outside": float(1.0 - np.sum(w[inside]) / total), "n_cells": int(mesh.shape[0]),
            "n_cells_inside": int(np.sum(inside)), "n_per_dim": [int(n)] * 3 + [int(len(xi0_axis))],
            "grid_lo": [float(v) for v in glo], "grid_hi": [float(v) for v in ghi], "prior": prior}


def _mass_job(args):
    job, grid, kw = args
    em = R.get_emulator(kw.pop("emulator_path", None))
    t0 = time.perf_counter()
    xi0_axis = R.cell_centres(grid["grid_lo"][3], grid["grid_hi"][3], int(grid["n_per_dim"]))   # the first grid's ξ₀ points
    out = mass_outside(grid["grid_lo"], grid["grid_hi"], xi0_axis, np.asarray(job["y"], dtype=float), job["s"], em,
                       n=kw.get("n", WHOLE_N))
    return dict({"kind": "mass", "task": job["label"], "set": job["set"], "i": int(job["i"]), "j": int(job["j"]),
                 "s": float(job["s"])}, **out, seconds=time.perf_counter() - t0, utc=utc_now())


# --------------------------------------------------------------------------- #
# v2's records, the study's jobs, header and record
# --------------------------------------------------------------------------- #
def sha256_lf(path):
    """The SHA-256 of a text file's bytes with LF line endings: the committed blob's hash, as
    D89 lists it, whatever the working copy's line endings (git stores LF, Windows checks out
    CRLF). None when there is no such file."""
    if not path or not os.path.exists(path):
        return None
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def read_sources(sources=None):
    """v2's two G2 records (D89), read only: {name: {path, sha256_lf, hdr, launches, tasks}}.
    Refuses a missing record, a set other than 18 pairs, or a sealed seed among the header's."""
    out = {}
    for name, p in (sources or SOURCES).items():
        hdr, launches, tasks = read_jsonl(p)
        if hdr is None:
            raise SystemExit(f"no record at {_rel(p)}: the study runs on v2's 36 pilot pairs, read from its committed "
                             "records (D89)")
        refuse_sealed([hdr.get("seeds"), hdr.get("seed_ref")], f"reading {_rel(p)}")
        if len(tasks) != N_PAIRS_EXPECTED // 2:
            raise SystemExit(f"{_rel(p)} holds {len(tasks)} pair records, not {N_PAIRS_EXPECTED // 2}")
        out[name] = {"path": p, "sha256_lf": sha256_lf(p), "hdr": hdr, "launches": launches, "tasks": tasks}
    return out


def study_jobs(sources):
    """The 36 pairs as launch's jobs: v2's label, its set (g2 or g2r), i, j, s, θ and y from
    the record, and v2's status; the generator key stays [83, i, j] (reference_pair)."""
    jobs = []
    for name in sorted(sources):
        for label in sorted(sources[name]["tasks"]):
            rec = sources[name]["tasks"][label]
            jobs.append({"label": label, "set": name, "i": int(rec["i"]), "j": int(rec["j"]), "s": float(rec["s"]),
                         "theta": [float(v) for v in rec["theta"]], "y": np.asarray(rec["y"], dtype=float),
                         "v2_status": rec["status"]})
            refuse_sealed(rec.get("seed"), f"pair {label}")
    return jobs


def check_surfaces(jobs, em, sources, atol=1e-12):
    """The declared check that binds the study's surfaces to G̃ and v2's seeds: every pair's θ
    and y in the record equal g2_surfaces(retry) priced through the emulator given."""
    for name in sources:
        pts, eps, _ = R.g2_surfaces(retry=(name == "g2r"))
        Y = R.surfaces(pts, eps, em)
        for job in (j for j in jobs if j["set"] == name):
            if not (np.allclose(job["theta"], pts[job["i"]], rtol=0, atol=atol) and np.allclose(job["y"], Y[job["i"], job["j"]], rtol=0, atol=atol)):
                raise SystemExit(f"refused: pair {job['label']}'s θ or y in {_rel(sources[name]['path'])} is not the surface "
                                 "g2_surfaces gives through this emulator (line 143; G̃, D88)")
    return True


def study_header(pool, emulator_path, sources=None, versions=None, task_kwargs=None):
    """The record's header: p5_g2_reference.header's (commit, clean tree, the versions, G̃'s
    SHA-256, v2's constants), the study's script and sources, the candidate's constants, the
    stand-ins' settings, and `lengths` with the lengths in force and the candidate's constants
    (launch refuses a resume whose `lengths` differ)."""
    kw = dict(task_kwargs or {})
    seeds = {"pair_key": "default_rng([83, i, j]), i counted within the pair's v2 set (line 68; the plan)", "seed_ref": R.SEED_REF}
    src = {}
    if sources:
        for name, s_ in sources.items():
            seeds[name] = s_["hdr"].get("seeds")
            src[name] = {"path": _rel(s_["path"]), "sha256_lf": s_["sha256_lf"], "commit": s_["hdr"].get("commit"),
                         "lengths": s_["hdr"].get("lengths"), "emulator_sha256": s_["hdr"].get("emulator_sha256")}
    extra = {"script": os.path.basename(__file__), "reference_script": os.path.basename(R.__file__),
             "study": "P5 v3 development study of candidate (A) on v2's 36 pilot pairs (the plan, D91; D93)",
             "sources": src, "candidate": candidate_constants(), "standin": standin_constants(),
             "verdict_rule": {"pass": "all pairs accepted, every comparison within tolerance, projected cost within budget",
                              "non_adaptation": {"acceptance_below": NONADAPT_ACC, "rhat_above": NONADAPT_RHAT},
                              "slow_mixing": {"rhat_at_most": R.RHAT_MAX, "ess_H_below": R.ESS_MIN}, "grid_miss_n": GRID_MISS_N}}
    n_points = sum(len({t["i"] for t in s_["tasks"].values()}) for s_ in sources.values()) if sources else 0
    hdr = R.header(SET_NAME, pool, emulator_path, n_points, seeds, extra, versions)
    hdr["lengths"] = {"burn": int(kw.get("burn", BURN)), "sample": int(kw.get("sample", SAMPLE)),
                      "kish_min": kw.get("kish_min", R.KISH_MIN), "rhat_max": kw.get("rhat_max", R.RHAT_MAX),
                      "ess_min": kw.get("ess_min", R.ESS_MIN), "levels": [float(s_) for s_ in NOISE_LEVELS],
                      "candidate": candidate_constants()}
    return hdr


def run_references(jobs, paths, hdr, pool, emulator_path, task_kwargs=None):
    """Phase 1: p5_g2_reference.launch over the jobs with the study's worker in place of its
    own (so each worker installs the candidate in its process); resumable, samples before
    records, nothing written over. Returns the count of pairs done now."""
    kw = dict(task_kwargs or {})
    kw.setdefault("burn", BURN)
    kw.setdefault("sample", SAMPLE)
    if pool != 1:
        kw["emulator_path"] = emulator_path
    worker = R._worker
    R._worker = _study_worker
    try:
        return R.launch(pool, jobs, paths["jsonl"], paths["samples"], hdr, kw)
    finally:
        R._worker = worker


def read_study_record(jsonl_path):
    """The header, the launches, the pair records by label (read_jsonl's), and the study's own
    lines: the mass lines and the stand-in lines by their keys."""
    hdr, launches, tasks = read_jsonl(jsonl_path)
    mass, standins, errors = {}, {}, []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("kind") == "mass":
                    mass[obj["task"]] = obj
                elif obj.get("kind") == "standin":
                    standins[(obj["task"], obj["prior"], obj["variant"])] = obj
                elif obj.get("kind") in ("mass_error", "standin_error", "task_error"):
                    errors.append(obj)
    return hdr, launches, tasks, mass, standins, errors


def _pool_run(pool, fn, items, jsonl_path, what):
    """A pool over `items` (each a tuple for `fn`), every result appended to the record as it
    arrives; an exception recorded as a `{what}_error` line, the run continuing."""
    done, n_err = [], 0

    def take(item, res, err):
        nonlocal n_err
        if err is not None:
            n_err += 1
            append_line(jsonl_path, {"kind": f"{what}_error", "task": item[0]["label"], "utc": utc_now(), "error": repr(err)})
            print(f"  {what} {item[0]['label']} raised {err!r}; recorded, continuing", flush=True)
            return
        append_line(jsonl_path, res)
        done.append(res)
    if pool == 1:
        for it in items:
            try:
                take(it, fn(it), None)
            except Exception as e:                                    # noqa: BLE001
                take(it, None, e)
    else:
        ex = ProcessPoolExecutor(max_workers=pool)
        try:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                try:
                    take(futs[fut], fut.result(), None)
                except Exception as e:                                # noqa: BLE001
                    take(futs[fut], None, e)
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    return done, n_err


def run_checks(jobs, paths, em, pool, emulator_path, versions=None, task_kwargs=None):
    """Phase 2: g2_report over the study's record (the grid checks on every accepted pair,
    whole_box_check at the largest level, the projection and the go), written to
    paths['checks']; read back, not rewritten, when the file exists."""
    hdr, _, tasks = read_jsonl(paths["jsonl"])
    kw = dict(task_kwargs or {})
    lengths = (int(kw.get("burn", BURN)), int(kw.get("sample", SAMPLE)))
    if os.path.exists(paths["checks"]):
        with open(paths["checks"], encoding="utf-8") as f:
            rep = json.load(f)
        if (rep.get("lengths") or {}) != {"burn": lengths[0], "sample": lengths[1]} or \
                (rep.get("emulator") or {}).get("sha256") != (hdr or {}).get("emulator_sha256"):
            raise SystemExit(f"checks refused: {_rel(paths['checks'])} was made at other lengths or with another emulator than "
                             "the record's; a re-run needs the old file moved aside by hand, and the move declared")
        print(f"  checks read from {_rel(paths['checks'])} (not rewritten)", flush=True)
        return rep
    present = [j for j in jobs if j["label"] in tasks]                   # a pair that erred on every attempt has no record:
    if len(present) < len(jobs):                                         # the checks run on the rest, and the verdict reports it missing
        print(f"  checks: {len(jobs) - len(present)} pair(s) without a record are left to the verdict", flush=True)
    chk = {"jsonl": paths["jsonl"], "samples": paths["samples"], "report": paths["checks"]}
    return R.g2_report(present, chk, em, pool, NOISE_LEVELS, grid_n=GRID_N, whole_n=WHOLE_N, emulator_path=emulator_path,
                       lengths=lengths, attempt=1, versions=versions)


def run_mass(jobs, paths, checks, pool, emulator_path):
    """Phase 3: the mass diagnostic for every accepted pair with a first grid, one `mass` line
    each; pairs with a line already are skipped."""
    _, _, tasks, mass, _, _ = read_study_record(paths["jsonl"])
    rows = {r["task"]: r for r in checks["pairs"]}
    items = []
    for job in jobs:
        row = rows.get(job["label"])
        if job["label"] in mass or row is None or "grid" not in row:
            continue
        kw = {"n": WHOLE_N, **({"emulator_path": emulator_path} if pool != 1 else {})}
        items.append((job, row["grid"], kw))
    print(f"\n=== MASS: {len(items)} pairs to do ({len(mass)} already) ===", flush=True)
    done, n_err = _pool_run(pool, _mass_job, items, paths["jsonl"], "mass")
    return read_study_record(paths["jsonl"])[3], n_err


def run_standins(jobs, paths, pool, emulator_path, task_kwargs=None):
    """Phase 4: the stand-ins, two variants under each prior for every accepted pair, one
    `standin` line each; those with a line already are skipped."""
    _, _, tasks, _, standins, _ = read_study_record(paths["jsonl"])
    kw = {k: v for k, v in dict(task_kwargs or {}).items() if k in ("burn", "sample", "rhat_max", "ess_min")}
    items = []
    for job in jobs:
        rec = tasks.get(job["label"])
        if rec is None or rec.get("status") != "ok":
            continue
        for prior in ("A", "B"):
            for variant in STANDIN_VARIANTS:
                if (job["label"], prior, variant) in standins:
                    continue
                items.append((job, rec, prior, variant, dict(kw, **({"emulator_path": emulator_path} if pool != 1 else {}))))
    print(f"\n=== STAND-INS: {len(items)} runs to do ({len(standins)} already) ===", flush=True)
    done, n_err = _pool_run(pool, _standin_job, items, paths["jsonl"], "standin")
    return read_study_record(paths["jsonl"])[4], n_err


# --------------------------------------------------------------------------- #
# The verdict and the report
# --------------------------------------------------------------------------- #
def _max_rhat(att):
    vals = [v for v in (att.get("rhat") or {}).values()]
    if not vals or any(v is None for v in vals):
        return float("inf")
    return max(float(v) for v in vals)


def failure_kind(att, acc_min=NONADAPT_ACC, rhat_bad=NONADAPT_RHAT, rhat_max=R.RHAT_MAX, ess_min=R.ESS_MIN):
    """The kind of a run that was not accepted (the plan): 'non-adaptation' when its acceptance
    is below 1% or its R-hat above 1.1; 'slow mixing' when R-hat is within 1.01 but the
    effective size of H is below 2,000; a run fitting neither is labelled as such."""
    rh, ess = _max_rhat(att), att.get("ess_H")
    if att.get("acceptance", 0.0) < acc_min or rh > rhat_bad:
        return "non-adaptation"
    if rh <= rhat_max and (ess is None or ess < ess_min):
        return "slow mixing"
    return "neither kind (R-hat between %g and %g with acceptance at least %g)" % (rhat_max, rhat_bad, acc_min)


def v2_comparison(jobs, tasks, sources):
    """The pairs v2 failed, beside what v2's sampler did on them at the lengths each ran: for
    the first attempt's, one and two times the study's lengths; for the retry's, two and four."""
    out = []
    for job in jobs:
        v2 = sources[job["set"]]["tasks"][job["label"]]
        if v2["status"] == "ok":
            continue
        st = tasks.get(job["label"])
        row = {"task": job["label"], "set": job["set"], "s": job["s"], "v2_status": v2["status"],
               "v2_lengths_in_units_of_the_studys": [1, 2] if job["set"] == "g2" else [2, 4],
               "v2_runs": {k: [{"burn": a["burn"], "sample": a["sample"], "acceptance": a["acceptance"], "max_rhat": _max_rhat(a),
                                "ess_H": a["ess_H"], "accepted": a["accepted"]} for a in v] for k, v in v2["runs"].items()},
               "study_status": None if st is None else st["status"], "study_runs": None}
        if st is not None:
            row["study_runs"] = {k: [{"burn": a["burn"], "sample": a["sample"], "acceptance": a["acceptance"], "max_rhat": _max_rhat(a),
                                      "ess_H": a["ess_H"], "accepted": a["accepted"]} for a in v] for k, v in st["runs"].items()}
        out.append(row)
    return out


def per_pair_rows(jobs, tasks, checks, mass):
    """The per-pair report the plan lists, from the record, the candidate's diagnostics, the
    checks and the mass lines."""
    rows = {r["task"]: r for r in (checks or {}).get("pairs", [])}
    out = []
    for job in jobs:
        rec = tasks.get(job["label"])
        row = {"task": job["label"], "set": job["set"], "i": job["i"], "j": job["j"], "s": job["s"], "v2_status": job["v2_status"],
               "status": None if rec is None else rec["status"]}
        if rec is None:
            out.append(row)
            continue
        acc_key = "acceptance_first_%d" % ADAPT_EVERY
        runs = []
        for d in rec.get("candidate", []):
            runs.append({"prior": d["prior"], "burn": d["burn"], "sample": d["sample"], "m": d["m"],
                         "starts_per_solution": [k["n_starts_reached"] for k in d["kept"]], "starts_elsewhere": d["starts_elsewhere"],
                         "eigenvalues": [k["eig"] for k in d["kept"]], "n_clipped": [k["n_clipped"] for k in d["kept"]],
                         "optimiser_status": d["optimiser"].get("status"), acc_key: d.get(acc_key),
                         "n_start_clipped": d["n_start_clipped"], "optimiser_seconds": d["optimiser_seconds"],
                         "proposal_vs_sample": d.get("proposal_vs_sample")})
        row.update({"runs": rec["runs"], "candidate_runs": runs, "direct": rec.get("direct"), "seconds": rec.get("seconds"),
                    "under": rec.get("under"), "c_A": rec.get("c_A"),
                    "kinds": {k: [failure_kind(a) for a in v if not a["accepted"]] for k, v in rec["runs"].items()},
                    "grid": (rows.get(job["label"]) or {}).get("grid"), "whole_box": (rows.get(job["label"]) or {}).get("whole_box"),
                    "checks_passed": (rows.get(job["label"]) or {}).get("passed"),
                    "mass_outside_first_grid": (mass.get(job["label"]) or {}).get("share_outside")})
        out.append(row)
    return out


def verdict(jobs, tasks, checks, standins, n_expected=None):
    """The verdict on (A), binding by D91: pass when every pair is accepted, every comparison
    is within tolerance and the projected cost is within the budget; else fail, with the kind
    of each failure. The gradient route's outcome (every own stand-in accepted and within
    tolerance) is reported beside it, not inside it. `n_expected` is N_PAIRS_EXPECTED, 36."""
    n_expected = N_PAIRS_EXPECTED if n_expected is None else int(n_expected)
    missing = [j["label"] for j in jobs if j["label"] not in tasks]
    not_candidate = [j["label"] for j in jobs if j["label"] in tasks and tasks[j["label"]].get("sampler") != SAMPLER_TAG]
    not_accepted = [j["label"] for j in jobs if j["label"] in tasks and tasks[j["label"]]["status"] != "ok"]
    rows = {r["task"]: r for r in (checks or {}).get("pairs", [])}
    misses = [j["label"] for j in jobs if j["label"] in rows and rows[j["label"]]["status"] == "ok" and not rows[j["label"]]["passed"]]
    go = bool(checks and checks.get("go"))
    complete = not missing and not not_candidate and len(jobs) == n_expected and checks is not None
    passed = bool(complete and not not_accepted and not misses and go)
    kinds = {lab: {k: [failure_kind(a) for a in v if not a["accepted"]] for k, v in tasks[lab]["runs"].items()} for lab in not_accepted}
    own = [v for v in standins.values() if v["variant"] == "own"]
    disp = [v for v in standins.values() if v["variant"] == "displaced"]
    n_own_expected = 2 * sum(1 for j in jobs if j["label"] in tasks and tasks[j["label"]]["status"] == "ok")
    grad_ok = bool(own) and len(own) == n_own_expected and all(v["accepted"] and (v["compare"] or {}).get("passed") for v in own)
    return {"passed": passed, "complete": complete, "n_pairs": len(jobs), "n_expected": n_expected, "missing": missing,
            "not_candidate": not_candidate, "not_accepted": not_accepted, "kinds": kinds, "grid_misses": misses, "go": go,
            "projected_core_hours": ((checks or {}).get("projection") or {}).get("projected_core_hours"),
            "budget_core_hours": (checks or {}).get("budget_core_hours"),
            "gradient_route": {"passed": grad_ok, "n_own": len(own), "n_own_expected": n_own_expected,
                               "own_not_accepted": [v["task"] + "/" + v["prior"] for v in own if not v["accepted"]],
                               "own_outside_tolerance": [v["task"] + "/" + v["prior"] for v in own if v["accepted"] and not (v["compare"] or {}).get("passed")],
                               "displaced_not_accepted": [v["task"] + "/" + v["prior"] for v in disp if not v["accepted"]],
                               "n_displaced": len(disp)},
            "reading": ("(A) passes the development study" if passed else
                        "(A) fails the development study" if complete else "the study is incomplete")}


def grid_miss_recompute(jobs, tasks, misses, pool, emulator_path, n=None):
    """A grid miss: the pair's first grid recomputed at n points per dimension (GRID_MISS_N,
    the plan), by p5_g2_reference._grid_job, to separate quadrature error from sampler error.
    At most two workers: a 64⁴ grid holds about 1 GB per worker while it is built."""
    n = GRID_MISS_N if n is None else int(n)
    pool = min(int(pool), 2)
    out = {}
    items = []
    for job in jobs:
        if job["label"] in misses:
            kw = {"grid_n": n, "whole_n": WHOLE_N, "extend": R.EXTEND, "box": BOX, "tol_mean": R.TOL_MEAN, "tol_c": R.TOL_C,
                  "tol_q": R.TOL_Q, "whole": False}
            if pool != 1:
                kw["emulator_path"] = emulator_path
            items.append((job, tasks[job["label"]], kw))
    if pool == 1:
        for it in items:
            out[it[0]["label"]] = R._grid_job(it)
    else:
        ex = ProcessPoolExecutor(max_workers=pool)
        try:
            futs = {ex.submit(R._grid_job, it): it[0]["label"] for it in items}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        finally:
            ex.shutdown(wait=True)
    return out


def write_report(jobs, paths, sources, checks, hdr, versions=None, pool=None, emulator_path=None, recompute=True):
    """Phase 5: the verdict and the report, written once (never over)."""
    E.refuse_overwrite([paths["report"]], "study report")
    rec_hdr, launches, tasks, mass, standins, errors = read_study_record(paths["jsonl"])
    hdr = rec_hdr or hdr                                                 # the record's header, not a resume's freshly built one
    v = verdict(jobs, tasks, checks, standins)
    grid64 = {}
    if recompute and v["grid_misses"] and emulator_path is not None:
        grid64 = grid_miss_recompute(jobs, tasks, v["grid_misses"], pool or 1, emulator_path)
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    done_keys = set(tasks) | set(mass) | {k[0] for k in standins}
    unresolved = [e for e in errors if not (e.get("kind") == "task_error" and e.get("task") in tasks
                                            or e.get("kind") == "mass_error" and e.get("task") in mass
                                            or e.get("kind") == "standin_error" and e.get("task") in done_keys)]
    report = {"study": hdr.get("study"), "script": os.path.basename(__file__), "verdict": v,
              "note": "the checks file is g2_report's output; its 'gate' key is the function's label, and the study runs no gate",
              "lengths": hdr.get("lengths"), "candidate": candidate_constants(), "standin": standin_constants(),
              "projection": (checks or {}).get("projection"), "go": (checks or {}).get("go"),
              "budget_core_hours": (checks or {}).get("budget_core_hours"), "cost": (checks or {}).get("cost"),
              "pairs": per_pair_rows(jobs, tasks, checks, mass),
              "v2_failed_pairs": v2_comparison(jobs, tasks, sources) if sources else None,
              "mass_outside_first_grid": {k: m_["share_outside"] for k, m_ in mass.items()},
              "standins": [standins[k] for k in sorted(standins)], "grid_miss_recomputed": _jsonable(grid64),
              "errors": errors, "errors_unresolved": unresolved,
              "sources": hdr.get("sources"), "emulator": {"path": hdr.get("emulator"), "sha256": hdr.get("emulator_sha256")},
              "header_commit": hdr.get("commit"), "launches": launches, "report_commit": commit, "report_clean_tree": clean,
              "versions": versions or E.version_check(E.CORE_VERSIONS, check=False), "utc_report": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=1, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
def estimate(sources_reports=None, pool=POOL_DEFAULT, n_pairs=N_PAIRS_EXPECTED):
    """A cost estimate from v2's G2 reports (read only), in core-hours and wall hours at
    `pool`: the median seconds per 50,000-step unit over both attempts' runs, about four units
    of sampling per pair, the first grids, the whole-box checks at the largest level, the
    whole-box mass grids at every level, and the stand-ins (four runs per pair, with repeats
    counted at 1.2 units). Nothing is priced or sampled."""
    unit_s, grid_s, whole_s, units = [], [], [], []
    for name, p in (sources_reports or SOURCE_REPORTS).items():
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        pr = rep.get("projection") or {}
        if pr.get("median_unit_s") and pr.get("base_steps"):
            unit_s.append(pr["median_unit_s"] * (BURN + SAMPLE) / pr["base_steps"])
            units.append(pr["mean_units"] * pr["base_steps"] / (BURN + SAMPLE))
        c = rep.get("cost") or {}
        if c.get("grid_mean_s"):
            grid_s.append(c["grid_mean_s"])
        if c.get("whole_box_mean_s"):
            whole_s.append(c["whole_box_mean_s"])
    if not unit_s:
        return None
    u, g, w, k = float(np.mean(unit_s)), float(np.mean(grid_s or [0.0])), float(np.mean(whole_s or [0.0])), float(np.mean(units))
    n_level_max = n_pairs // len(NOISE_LEVELS)
    parts = {"sampling": n_pairs * k * u, "first_grids": n_pairs * g, "whole_box_checks": n_level_max * w,
             "mass_grids": n_pairs * w, "standins": n_pairs * 4 * 1.2 * u}
    core_h = sum(parts.values()) / 3600.0
    return {"unit_s_per_%d_steps" % (BURN + SAMPLE): u, "units_per_pair": k, "grid_s": g, "whole_box_s": w,
            "parts_core_hours": {a: b / 3600.0 for a, b in parts.items()}, "core_hours": core_h, "wall_hours_at_pool": core_h / pool,
            "pool": pool, "from": [_rel(p) for p in (sources_reports or SOURCE_REPORTS).values() if os.path.exists(p)]}


def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 v3 development study (see module docstring)")
    ap.add_argument("--run", action="store_true", help="run the study (the author's; from the merged main)")
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--emulator", default=None, help="G̃ is the build G1 passes (p5_g1_emulator.require_g1); another file is refused (D82)")
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy or scipy other than p5_g1_emulator.CORE_VERSIONS; recorded (D81)")
    a = ap.parse_args(argv)
    if a.pool < 1:
        raise SystemExit(f"refused: --pool {a.pool}; the pool is at least 1")
    vers = E.version_check(E.CORE_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions)
    cc = candidate_constants()
    print("=== P5 v3 development study: candidate (A) on v2's 36 pilot pairs ===", flush=True)
    print(f"  reference: {os.path.basename(R.__file__)} as committed, run_chains replaced by candidate_run_chains inside each worker", flush=True)
    print(f"  lengths {BURN}/{SAMPLE} (the repeat doubles them); chains {N_CHAINS}; adapt every {ADAPT_EVERY}; "
          f"R-hat<={R.RHAT_MAX} ESS_H>={R.ESS_MIN:.0f}; (Kish/steps) x ESS_H>={R.KISH_MIN:.0f}; pairs keyed [83, i, j]", flush=True)
    print(f"  (A) step 1: {N_STARTS} starts; least_squares {LS_METHOD}, jac {LS_JAC}, bounds [0,1]^4, ftol {LS_FTOL:g} xtol {LS_XTOL:g} "
          f"gtol {LS_GTOL:g} max_nfev {LS_MAX_NFEV}; polish (direct runs) L-BFGS-B ftol {LBFGS_FTOL:g} gtol {LBFGS_GTOL:g} "
          f"maxiter {LBFGS_MAXITER}, H in [{EPS_H:g}, 1-{EPS_H:g}]; fd step {FD_REL_STEP:.3e}", flush=True)
    print(f"  (A) step 2: keep <= {KEEP_MAX}, distinct > {DISTINCT}, Laplace mass within log 1e4 = {MASS_LOG:.4f}", flush=True)
    print(f"  (A) step 3: Sigma from J's singular values, eigenvalues clipped to [{EIG_LO:g}, {EIG_HI:.6f}]; the prior's density and "
          f"curvature for the keep rule and the recorded Sigma_A, Sigma_B at x_H clipped to [{KAPPA_U_FLOOR}, {1 - KAPPA_U_FLOOR}]", flush=True)
    print(f"  (A) step 4: start = solution k mod m + N(0, {PERTURB_MULT:g} Sigma_s), Sigma_s eigenvalues capped at {PERTURB_CAP}", flush=True)
    print(f"  (A) step 5: first proposal {SCALE:.4f} x Sigma; then v2's adaptation, lengths, acceptance, repeat, reweighting, direct runs", flush=True)
    print(f"  stand-ins (Decision 3): own target and displaced (H by {DISPLACE_H} towards the centre), generators "
          f"[{SEED_STANDIN}, p, {STANDIN_Q}, i, j] and [..., 1], Sigma_P from the record, gradient by finite difference", flush=True)
    print(f"  checks: g2_report (grids {GRID_N}^4 on the support +{R.EXTEND:.0%}, {WHOLE_N}^3 x {GRID_N} at s = {max(NOISE_LEVELS)}; "
          f"tolerances {R.TOL_MEAN}, {R.TOL_C}, {R.TOL_Q}, each >= {R.TOL_MC_K:g} MC standard errors); mass outside the first grid at every level", flush=True)
    print(f"  verdict: pass = all pairs accepted, every comparison within tolerance, projection <= {R.BUDGET_CORE_H:.0f} core-hours; "
          f"non-adaptation acc < {NONADAPT_ACC} or R-hat > {NONADAPT_RHAT}; slow mixing R-hat <= {R.RHAT_MAX} and ESS_H < {R.ESS_MIN:.0f}; "
          f"grid miss -> first grid at {GRID_MISS_N}^4", flush=True)
    print(f"  records: {_rel(PATHS['jsonl'])}, {_rel(PATHS['samples'])}, {_rel(PATHS['checks'])}, {_rel(PATHS['report'])}", flush=True)
    print(f"  numpy {vers['numpy']} scipy {vers['scipy']} (expected {E.CORE_VERSIONS}); sealed seeds {SEALED_SEEDS} refused", flush=True)
    sources = None
    try:
        sources = read_sources()
        for name, s_ in sources.items():
            n_ok = sum(1 for t in s_["tasks"].values() if t["status"] == "ok")
            print(f"  source {name}: {_rel(s_['path'])} sha256 (LF) {s_['sha256_lf'][:12]}..., commit {str(s_['hdr'].get('commit'))[:7]}, "
                  f"lengths {s_['hdr'].get('lengths', {}).get('burn')}/{s_['hdr'].get('lengths', {}).get('sample')}, "
                  f"{len(s_['tasks'])} pairs, {n_ok} ok, G~ {str(s_['hdr'].get('emulator_sha256'))[:12]}...", flush=True)
            for label in sorted(s_["tasks"]):
                t = s_["tasks"][label]
                print(f"    {label}: s={t['s']} key [{R.SEED_REF}, {t['i']}, {t['j']}] theta={[round(v, 4) for v in t['theta']]} "
                      f"v2 {t['status']}" + (f" (direct {t['direct']})" if t.get("direct") else ""), flush=True)
    except SystemExit as e:
        print(f"  sources: {e}", flush=True)
    est = estimate(pool=a.pool)
    if est:
        print(f"  estimate (from v2's reports, {', '.join(est['from'])}): {est['unit_s_per_%d_steps' % (BURN + SAMPLE)]:.1f} s per "
              f"{BURN + SAMPLE} steps, {est['units_per_pair']:.2f} units per pair; "
              + ", ".join(f"{k} {v_:.1f}" for k, v_ in est["parts_core_hours"].items())
              + f"; total {est['core_hours']:.1f} core-hours, {est['wall_hours_at_pool']:.1f} h on {a.pool} workers", flush=True)
    else:
        print("  estimate: v2's G2 reports not found; no figure", flush=True)
    if not a.run:
        print("\nHOLD: nothing run, no emulator loaded, nothing written. --run is the author's, from the merged main (D93).", flush=True)
        return
    if sources is None:
        raise SystemExit("refused: the study needs v2's two G2 records")
    E.refuse_overwrite([PATHS["report"]], "study")
    em_path = E.emulator_for_run(a.emulator)                                            # G̃, the build G1 passes (D82)
    sha = E.sha256_file(em_path)
    for name, s_ in sources.items():
        if s_["hdr"].get("emulator_sha256") != sha:
            raise SystemExit(f"refused: {_rel(s_['path'])} was made with emulator {s_['hdr'].get('emulator_sha256')}, not G̃ ({sha})")
    em = R.get_emulator(em_path)
    jobs = study_jobs(sources)
    check_surfaces(jobs, em, sources)
    if len(jobs) != N_PAIRS_EXPECTED:
        raise SystemExit(f"refused: {len(jobs)} pairs, not {N_PAIRS_EXPECTED}")
    kw = dict(TASK_KWARGS)
    hdr = study_header(a.pool, em_path, sources, vers, kw)
    run_references(jobs, PATHS, hdr, a.pool, em_path, kw)
    checks = run_checks(jobs, PATHS, em, a.pool, em_path, vers, kw)
    print(f"=== checks: {'passed' if checks['passed'] else 'NOT passed'}; reference failures {checks['n_reference_failures']}; "
          f"projected {checks['projection']['projected_core_hours']} core-hours ({'go' if checks['go'] else 'no go'}) ===", flush=True)
    run_mass(jobs, PATHS, checks, a.pool, em_path)
    run_standins(jobs, PATHS, a.pool, em_path, kw)
    rep = write_report(jobs, PATHS, sources, checks, hdr, vers, a.pool, em_path)
    v = rep["verdict"]
    print(f"\n=== VERDICT: {v['reading']}; not accepted {len(v['not_accepted'])}, grid misses {len(v['grid_misses'])}, "
          f"go {v['go']}; gradient route {'passed' if v['gradient_route']['passed'] else 'NOT passed'} ===", flush=True)
    for row in rep["pairs"]:
        print(f"  {row['task']}: {row['status']} (v2 {row['v2_status']}) m={[r['m'] for r in row.get('candidate_runs', [])]} "
              f"checks={row.get('checks_passed')} mass_out={row.get('mass_outside_first_grid')}", flush=True)
    print(f"  written: {_rel(PATHS['report'])}", flush=True)


if __name__ == "__main__":
    main(check_versions=True)
