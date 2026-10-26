"""
p5_g2_reference.py — P5 gate G2, the exact reference (docs/protocols/P5_protocol_v3_draft.md,
Objects "The exact reference", D66, with v3's start and first proposal, D95; Gates, G2; the ξ₀
means of D67; the priors of D69).

Per scored (surface, noise level) pair: the posterior of θ = (H, ν, ρ, ξ₀) on the box
under the likelihood Gaussian with the known s on the 35 quotes, log L = −Σ(y − G̃(θ))² /
(2 s²), and the uniform box prior, by adaptive random-walk Metropolis on the emulator G̃
in the box mapped affinely to [0, 1]⁴. One generator per pair, default_rng([83, i, j]) for
the i-th θ of the set in question and the j-th noise level, serves, in this order, the
optimiser's starts, the chains' start perturbations, and the proposals and acceptance
draws (z then u within a step). G2's retry is a set of its own, its i counted within it,
so that a retry pair shares only the generator's stream with the first attempt's pair of
the same i and j (D82, D95).

The start and the first proposal (line 68; candidate (A) of the plan, D91, developed in
D93, passed in D94 and written in here, D95), once per sampler run, before the chains:
  1. The modes: scipy.optimize.least_squares (method trf, bounds [0, 1]⁴, jac 2-point;
     ftol = xtol = 1e-10, gtol = 1e-8, max_nfev = 2,000) on r(x) = (y − G̃(θ(x)))/s from 16
     starts uniform on [0, 1]⁴; each start's exit status and evaluations are recorded. For a
     direct run under prior A or B each kept solution is then polished by L-BFGS-B on
     ½Σr² − log P(x_H), its gradient Jᵀr (J by the bounded 2-point difference, step √ε·max(1,
     |x|), inward at a face) less the prior's analytic gradient d log P/du, H's bounds ε = 1e-6
     inside the faces (ftol = 1e-10, gtol = 1e-5, maxiter = 200).
  2. Kept: in order of log-target from the best, solutions distinct (unit distance above 0.01
     from every solution kept) whose Laplace mass log f + ½ log det Σ is within log 10⁴ of the
     largest mass, at most four; m and the starts reaching each are recorded.
  3. Σ = (JᵀJ)⁻¹ from J's singular values, V S⁻² Vᵀ; for a direct run Σ_P = (JᵀJ +
     diag(κ, 0, 0, 0))⁻¹, κ = −∂² log P/∂x_H², from the singular values of J stacked with
     √κ e₁ᵀ; the eigenvalues clipped to [1e-10, 1/12], the count clipped recorded. Σ_A and Σ_B
     at each kept solution of every run are recorded (the uniform run's read for Σ_P); the keep
     rule's prior density and curvature for a direct run, and κ for the recorded Σ_A and Σ_B,
     are taken at x_H clipped to [0.01, 0.99] (KAPPA_U_FLOOR); and the pair's record carries Σ_P
     under each prior for the ratio estimator's sampler (p5_g3_calibrators.py).
  4. Chain k starts at kept solution k mod m plus a draw from N(0, 4 Σ_s), Σ_s being Σ with its
     eigenvalues capped at 0.0025, clipped to the box ([ε, 1 − ε] in H for a direct run or a
     run with a target, within the target's own bounds).
  5. The first proposal is 2.38²/4 · Σ of the chain's solution.
Then, as in v2: at burn-in steps 500, 1,000, …, 10,000 the proposal is 2.38²/4 × the
empirical covariance of that chain's steps so far (the start included), its diagonal floored
at 1e-10; the adaptation at step 10,000 is the sampling proposal, fixed over the 40,000
sampling steps; a proposal with any coordinate outside the box is rejected without
evaluation. Accepted when, on the sampling steps with each chain split into halves, the
classical between-within split R-hat (Gelman and others; not rank-normalised) is at most
1.01 on each parameter and the effective sample size of H over the eight half-chains, by
the variogram between-within estimator, is at least 2,000; otherwise repeated once with
both lengths doubled, the start run again from the same generator continuing; a second
failure is a reference failure for the pair, recorded. Priors A and B by self-normalised
importance weights ∝ the prior's density of H at each pooled sampling step, accepted when
the weighted chain's effective size, (Kish's (Σw)²/Σw² / the number of pooled sampling
steps) × the effective sample size of H above, is at least 1,000 under that prior (D81),
else a direct run under that prior with the same sampler and acceptance, from the same
generator continuing; a direct run that fails after its repeat is also a reference failure
for the pair. Weighted means, variances, quantiles (the smallest sample value at which the
cumulative weight reaches q) and interval probabilities of the pooled sampling steps.
Reported per pair: the posterior means of H under A and B, its variance under A and the
contraction c against the analytic Var_prior(H) of prior A, the equal-tailed 90% intervals
under A and under B, the coverage covariate (the mass each prior puts on the 90% interval
of H under the exact posterior under that prior; a pair well covered when both are at least
0.001, D81), the means of ξ₀ under A and B, Σ_P under A and B (the accepted direct run's Σ
at its best kept solution where the figures under that prior are direct, else the uniform
run's Σ with that prior's curvature added), the acceptance rate, R-hat and the effective
sample sizes, and each sampler run's start (the kept solutions with Σ, Σ_A and Σ_B, the
chains' starts, the acceptance over the first 500 burn-in steps, the final proposal against
the sample covariance); the H samples of every accepted run are stored (quantised to 16
bits on the box's range of H) so that the posterior probability under A of any interval can
be computed later. With a `target`, a callable object with grad_unit(x) and sigma
(p5_g3_calibrators.NRETarget), the modes come from the gradient route: L-BFGS-B from the 16
starts with the target's gradient (and its value, when the target gives both), Σ the given
sigma, its eigenvalues clipped to [1e-10, 1/12] as the start clips its own, for every
solution, a preconditioner only; a plain callable is refused.

G2 (line 143): on 18 pilot surfaces (4 points uniform on the box from default_rng(151) and
the two fixed points H = 0.04 and 0.46 at the box's centre in ν, ρ and ξ₀, at the three
noise levels, one noise draw each, from default_rng(157) for the four and default_rng(163)
for the two; v3's seeds, D95), the MCMC posterior of H against a cell-centred tensor grid of
32 points per dimension over the smallest axis-aligned box holding the four chains'
sampling-phase samples, each side extended by 20% and clipped to the box, the posterior
being the prior-weighted likelihood summed over cells; at the largest noise level a second
grid of 48 cell-centred points in each of H, ν and ρ over the whole box with ξ₀ on the first
grid's 32 points; tolerance on every pilot surface and both grids: under prior A the
posterior mean of H within 0.001, c within 0.01 and the 5% and 95% quantiles of H within
0.002; under prior B the posterior mean of H within 0.001, by the route the pair used
(D81); each tolerance the larger of that value and 4 Monte Carlo standard errors of the
MCMC figure, from the effective sample size of the run it came from ((Kish / pooled steps)
× ESS of H reweighted, ESS of H direct) by line 89's Gaussian reading (D81); a reference
failure on any pilot surface fails the gate (D73). A failure is followed by one retry
(--pilot-retry) with the burn-in and sampling lengths doubled, on fresh surfaces (4 points
from default_rng(167), noise from default_rng(173); the two fixed points with noise from
default_rng(179)); if the retry passes, the doubled lengths are the reference's for every
later run; if it fails, P5 stops. Go or no-go: the passing attempt's median core-seconds of
one sampler run (its optimiser included), times its mean number of runs per pair (direct
runs and repeats counted, each in units of that attempt's lengths), times the 1,200 test
pairs, against 1,344 core-hours (G0's two weeks of four cores); above it, P5 stops (D81).
G3's 48 pilot surfaces (default_rng(97) points, default_rng(101) noise) and the sealed test
set get the same reference, only once G2 has passed with a go (line 144, line 93): the test
set is N = 400 θ, the first 400 rows of default_rng(20261002)'s uniform draw on the box,
with noise default_rng(20261022).standard_normal((400, 3, 35)) (line 93, D81).

The emulator is G̃, the build G1 passed under v2 (D88), adopted by v3 (line 142):
p5_g1_emulator.require_g1 returns it, and an --emulator naming another file is refused;
every record and report carries G̃'s SHA-256 (D82). --test refuses an N other than 400 or a
noise seed other than 20261022 before any draw, and no report is written over (D82). The
constants of the start go into each record's header, inside `lengths`, so that a resume
with other constants is refused as one with other lengths is (D95). The tests use toy
emulators in memory, and a toy emulator file for the pool paths (D92); the transcription
check of D95 (p5_v3_transcription.py) re-runs this sampler on the development study's 36
pairs and compares its records with the study's. Run from the command line, the script
refuses numpy and scipy other than the core interpreter's (p5_g1_emulator.CORE_VERSIONS)
unless --allow-other-versions, recorded (D81).

Usage (from the repository root, after v3's freeze):
    py -3 p5_g2_reference.py                   the design; holds
    py -3 p5_g2_reference.py --pilot           G2: the 18 references, the grid checks, the report
    py -3 p5_g2_reference.py --pilot-retry     the one retry, only after a failed first attempt
    py -3 p5_g2_reference.py --g3-pilot        the 48 references of G3's pilot surfaces, after G2
    py -3 p5_g2_reference.py --test            the sealed set, N = 400, after G2
    options: --emulator PATH (G̃ by default; another file is refused), --pool K (default 4)

Outputs (output/ is git-ignored; the files are force-added by name after a run). v3's records
go under new names; v2's (output/p5_g2_*, p5_g2r_*) stay in place as history (D89, D95):
    output/p5v3_g2_pilot.jsonl, output/p5v3_g2_pilot_samples.npz, output/p5v3_g2_report.json
    output/p5v3_g2r_pilot.jsonl, output/p5v3_g2r_pilot_samples.npz, output/p5v3_g2r_report.json (the retry)
    output/p5v3_g3_pilot_refs.jsonl, output/p5v3_g3_pilot_refs_samples.npz
    output/p5v3_ref_test.jsonl, output/p5v3_ref_test_samples.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (calibrate_btc.py)

import sys
import json
import math
import time
import argparse
import platform
from statistics import NormalDist
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import beta as _beta
from scipy.optimize import least_squares, minimize
from scipy.linalg import eigh as _geigh

_NORMAL = NormalDist()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g0_pilot as P
import p5_g1_emulator as E
from p5_g0_pilot import git_state, utc_now, append_line, _rel, read_jsonl
from p5_g1_emulator import BOX, DIMS, NOISE_LEVELS, N_QUOTES, box_arrays

# --------------------------------------------------------------------------- #
# The design, as written in (lines 62 to 68, 81, 89, 93, 143, 144)
# --------------------------------------------------------------------------- #
SEED_REF = 83                                       # default_rng([83, i, j]) per pair, i and j from 0 (line 68)
SEED_G2_POINTS, N_G2_POINTS, SEED_G2_NOISE = 151, 4, 157     # G2's pilot surfaces (line 143): v3's seeds (D95)
FIXED_POINTS = ((0.04, 0.20, -0.80, 0.145), (0.46, 0.20, -0.80, 0.145))   # G2's two fixed points, the box's centre (D81)
SEED_G2_FIXED_NOISE = 163                           # their noise (D81; v3's seed, D95)
SEED_G2R_POINTS, SEED_G2R_NOISE, SEED_G2R_FIXED_NOISE = 167, 173, 179     # the one retry's fresh surfaces (D81; v3's seeds, D95)
SEEDS_V2_G2 = (79, 89, 139, 131, 137, 149)          # v2's G2 seeds (D81, D89), not reused (the plan, item 3; D95)
SEED_G3_POINTS, N_G3_POINTS, SEED_G3_NOISE = 97, 16, 101     # G3's pilot surfaces (line 144)
SEED_TEST = 20261002                                # the sealed test θ (line 93)
N_TEST, SEED_TEST_NOISE = 400, 20261022             # the number of test θ and the test noise (line 93, D81)
N_TEST_PAIRS = 3 * N_TEST                           # 1,200
N_CHAINS = 4
BURN, SAMPLE, ADAPT_EVERY = 10000, 40000, 500
SCALE = 2.38 ** 2 / 4                               # 1.4161
DIAG_FLOOR = 1e-10
# The start and the first proposal (line 68; candidate (A), D91 to D95): the constants as the development study ran them (D93)
N_STARTS = 16                                       # optimiser starts, uniform on [0, 1]⁴ (step 1)
LS_METHOD, LS_JAC = "trf", "2-point"                # least_squares (step 1)
LS_FTOL, LS_XTOL, LS_GTOL, LS_MAX_NFEV = 1e-10, 1e-10, 1e-8, 2000   # its tolerances and evaluation cap
FD_REL_STEP = math.sqrt(np.finfo(float).eps)        # the 2-point step √ε·max(1, |x|), least_squares' own convention
LBFGS_FTOL, LBFGS_GTOL, LBFGS_MAXITER = 1e-10, 1e-5, 200   # the polish and the gradient route (L-BFGS-B)
EPS_H = 1e-6                                        # H's bounds for a direct run or a prior-weighted target (step 1)
KAPPA_U_FLOOR = 0.01                                # the prior's density and curvature for the keep rule, and for the recorded Σ_A and Σ_B,
#                                                     are taken at x_H clipped to [0.01, 0.99]: a solution on a face records no degenerate Σ_P
KEEP_MAX, DISTINCT, MASS_LOG = 4, 0.01, math.log(1e4)   # step 2: at most four, distinct beyond 0.01, within log 10⁴ of the mass
EIG_LO, EIG_HI = 1e-10, 1.0 / 12.0                  # step 3: Σ's eigenvalues clipped to this range (1/12, the variance of a uniform coordinate)
PERTURB_CAP, PERTURB_MULT = 0.0025, 4.0             # step 4: Σ_s caps the eigenvalues; the draw is N(0, 4 Σ_s)
RHAT_MAX, ESS_MIN, KISH_MIN = 1.01, 2000.0, 1000.0  # KISH_MIN: the floor on (Kish / pooled steps) × ESS of H (D81)
GRID_N, EXTEND, WHOLE_N = 32, 0.20, 48              # G2's grids (line 143)
TOL_MEAN, TOL_C, TOL_Q = 0.001, 0.01, 0.002         # G2's tolerance: means of H, c; the 5% and 95% quantiles (D81)
TOL_MC_K = 4.0                                      # each at least this many Monte Carlo standard errors (D81)
COVER_MIN = 0.001                                   # a pair is well covered when both priors' masses reach this (D81)
BUDGET_CORE_H = 1344.0                              # G0's two weeks of four cores (line 141): the go or no-go (D81)
H_LO, H_W = BOX["H"][0], BOX["H"][1] - BOX["H"][0]  # 0.02, 0.46
PRIOR_AB = {"A": (2.5, 9.0), "B": (235.0 / 23.0, 376.0 / 69.0)}   # line 62 (D69)
VAR_PRIOR_A = H_W ** 2 * 36.0 / 2645.0              # 0.00288, analytic (line 66)
POOL_DEFAULT = 4
QUANT = 65535                                       # H stored as uint16 on [H_LO, H_LO + H_W]

OUT_DIR = P.OUT_DIR
EMULATOR_PATH = E.paths("base")["npz"]
# v3's records go under new names (the plan, item 5; D95); v2's files, output/p5_g2_* and p5_g2r_*, stay in place (D89)
SETS = {"g2": {"jsonl": os.path.join(OUT_DIR, "p5v3_g2_pilot.jsonl"),
               "samples": os.path.join(OUT_DIR, "p5v3_g2_pilot_samples.npz"),
               "report": os.path.join(OUT_DIR, "p5v3_g2_report.json")},
        "g2r": {"jsonl": os.path.join(OUT_DIR, "p5v3_g2r_pilot.jsonl"),
                "samples": os.path.join(OUT_DIR, "p5v3_g2r_pilot_samples.npz"),
                "report": os.path.join(OUT_DIR, "p5v3_g2r_report.json")},
        "g3": {"jsonl": os.path.join(OUT_DIR, "p5v3_g3_pilot_refs.jsonl"),
               "samples": os.path.join(OUT_DIR, "p5v3_g3_pilot_refs_samples.npz")},
        "test": {"jsonl": os.path.join(OUT_DIR, "p5v3_ref_test.jsonl"),
                 "samples": os.path.join(OUT_DIR, "p5v3_ref_test_samples.npz")}}


# --------------------------------------------------------------------------- #
# Priors, the model, coordinates
# --------------------------------------------------------------------------- #
def prior_logpdf(H, which):
    """log density of H under 'uniform' (a constant, 0), 'A' or 'B' (the scaled Betas of
    line 62; the Jacobian 1/0.46 is a constant and is dropped). −inf outside the range."""
    H = np.asarray(H, dtype=float)
    if which == "uniform":
        return np.zeros_like(H)
    a, b = PRIOR_AB[which]
    with np.errstate(divide="ignore", invalid="ignore"):
        return _beta.logpdf((H - H_LO) / H_W, a, b)


def unit_to_theta(x, box=BOX):
    lo, hi = box_arrays(box)
    return lo + np.asarray(x, dtype=float) * (hi - lo)


def theta_to_unit(theta, box=BOX):
    lo, hi = box_arrays(box)
    return (np.asarray(theta, dtype=float) - lo) / (hi - lo)


def log_lik(thetas, y, s, em):
    """log L = −Σ(y − G̃(θ))² / (2 s²) per row of thetas (box units)."""
    G = em.evaluate_many(np.asarray(thetas, dtype=float).reshape(-1, 4))
    return -np.sum((G - np.asarray(y, dtype=float)[None, :]) ** 2, axis=1) / (2.0 * s * s)


def log_target(x, y, s, em, prior, box=BOX, target=None):
    """log posterior (up to a constant) at unit-cube points x [M, 4]; −inf outside. With
    `target`, a callable of points in box units returning the log density, the sampler
    serves another target with the same mechanics (the ratio estimator's posterior under
    The calibrators uses it with log P(H) + logit)."""
    x = np.asarray(x, dtype=float).reshape(-1, 4)
    out = np.full(x.shape[0], -np.inf)
    inside = np.all((x >= 0.0) & (x <= 1.0), axis=1)
    if inside.any():
        th = unit_to_theta(x[inside], box)
        out[inside] = target(th) if target is not None else log_lik(th, y, s, em) + prior_logpdf(th[:, 0], prior)
    return out


# --------------------------------------------------------------------------- #
# The sampler (line 68): the Cholesky factor, the adaptation, one Metropolis step (v2's, unchanged)
# --------------------------------------------------------------------------- #
def _chol_pd(c):
    """Cholesky factor; a matrix that is not positive definite after the diagonal floor
    (a chain that moved along a line) has 1e-10 × I added, tenfold each time, until it is."""
    jitter = DIAG_FLOOR
    for _ in range(12):
        try:
            return np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            c = c + jitter * np.eye(c.shape[0])
            jitter *= 10.0
    raise np.linalg.LinAlgError("proposal covariance could not be made positive definite")


def adapted_cov(steps):
    """2.38²/4 × the empirical covariance of one chain's steps so far, diagonal floored."""
    c = SCALE * np.atleast_2d(np.cov(np.asarray(steps, dtype=float).T, ddof=1))
    d = c.shape[0]
    c[np.diag_indices(d)] = np.maximum(np.diag(c), DIAG_FLOOR)
    return c


def mh_step(x, lt, L, y, s, em, prior, rng, box=BOX, target=None):
    """One Metropolis step for every chain: z then u are drawn from rng in that order,
    whether or not a proposal lies inside the box. Returns (x, lt, accepted)."""
    n, d = x.shape
    z = rng.standard_normal((n, d))
    prop = x + np.einsum("kij,kj->ki", L, z)
    inside = np.all((prop >= 0.0) & (prop <= 1.0), axis=1)
    lt_prop = np.full(n, -np.inf)
    if inside.any():
        lt_prop[inside] = log_target(prop[inside], y, s, em, prior, box, target)
    u = rng.uniform(size=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        accept = inside & (np.log(u) < lt_prop - lt)
    x = x.copy()
    lt = lt.copy()
    x[accept] = prop[accept]
    lt[accept] = lt_prop[accept]
    return x, lt, accept


# --------------------------------------------------------------------------- #
# The start and the first proposal (line 68; candidate (A), written in from
# p5_v3_study.candidate_run_chains as the development study ran it, D93 to D95): the prior's
# derivatives, the residuals, the bounded difference, Σ, the keep rule, the chain starts
# --------------------------------------------------------------------------- #
def prior_grad_curv(u, which):
    """(d/du log P(u), −d²/du² log P(u)) for prior `which` at u = x_H, the scaled Beta's
    argument (prior_logpdf drops the constant Jacobian): (0, 0) for 'uniform'."""
    if which == "uniform":
        return 0.0, 0.0
    a, b = PRIOR_AB[which]
    u = float(u)
    return (a - 1.0) / u - (b - 1.0) / (1.0 - u), (a - 1.0) / u ** 2 + (b - 1.0) / (1.0 - u) ** 2


def residuals(x, y, s, em, box=BOX):
    """r(x) = (y − G̃(θ(x)))/s at one unit-cube point, so that ½Σr² = −log L."""
    th = unit_to_theta(np.clip(np.asarray(x, dtype=float), 0.0, 1.0), box)
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


def clip_cov(sigma, eig_lo=EIG_LO, eig_hi=EIG_HI):
    """A given Σ (the record's Σ_P, for the gradient route) with its eigenvalues clipped to
    [eig_lo, eig_hi], as the start clips its own: (Σ, eigenvalues before clipping, the number
    clipped)."""
    w, V = np.linalg.eigh(np.asarray(sigma, dtype=float))
    clipped = np.clip(w, eig_lo, eig_hi)
    cov = (V * clipped) @ V.T
    return 0.5 * (cov + cov.T), w, int(np.sum((w < eig_lo) | (w > eig_hi)))


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
        Ls = _chol_pd(mult * capped_cov(covs[which[k]], cap))
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
        logf = -cost + np.array([float(prior_logpdf(H_LO + u * H_W, prior)) for u in u_keep])
        covs = [cov_from_jacobian(J, prior_grad_curv(u, prior)[1])[0] for J, u in zip(Js, u_keep)]
    if not np.any(np.isfinite(logf)):
        raise RuntimeError(f"the start found no finite least-squares solution from {N_STARTS} starts")
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
                return 0.5 * float(r @ r) - float(prior_logpdf(unit_to_theta(xx, box)[0], prior)), grad
            res = _lbfgs(f_and_g, x, lo_b, hi_b)
            x = np.asarray(res.x, dtype=float)
            J, r = fd_jacobian(fun, x, lo_b, hi_b)
            lf = -0.5 * float(r @ r) + float(prior_logpdf(unit_to_theta(x, box)[0], prior))   # the log-target at the polished point
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
    """Steps 1 to 3 by the gradient route (Decision 3; line 75): L-BFGS-B from each start on
    −target with the target's gradient in unit coordinates (the value and the gradient from
    `value_grad_unit(x)` where the target gives both, as the ratio estimator's float64 copy
    does; else the value from the target itself and the gradient from `grad_unit(x)`); the
    keep rule with Σ = target.sigma for every solution (so the mass rule reduces to log f
    within log 10⁴); Σ the record's Σ_P throughout, a preconditioner only."""
    for name in ("grad_unit", "sigma"):
        if not hasattr(target, name):
            raise TypeError(f"the gradient route needs a target with `{name}`; a plain callable is not run with this sampler")
    b_lo, b_hi = getattr(target, "bounds_unit", (np.zeros(4), np.ones(4)))
    lo_b, hi_b = np.maximum(lo_b, b_lo), np.minimum(hi_b, b_hi)
    sigma, eig, n_clip = clip_cov(np.asarray(target.sigma, dtype=float))
    value_grad = getattr(target, "value_grad_unit", None)

    def f_and_g(xx):
        if value_grad is not None:
            f, g = value_grad(xx)
            return -float(f), -np.asarray(g, dtype=float)
        f = float(target(unit_to_theta(xx, box).reshape(1, 4))[0])
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
        raise RuntimeError(f"the start, gradient route: no finite solution from {N_STARTS} starts")
    kept_idx, mass, reached = keep_solutions(sols, logf, [sigma] * len(sols))
    kept = [{"x": sols[i], "logf": float(logf[i]), "logf_ls": None, "mass_ls": float(mass[i]), "eig": eig, "n_clipped": n_clip,
             "sigma": sigma, "sigma_A": None, "sigma_B": None, "n_starts_reached": int(np.sum(reached == pos)), "polish": None}
            for pos, i in enumerate(kept_idx)]
    return {"route": "gradient", "kept": kept, "starts_elsewhere": int(np.sum(reached < 0)), "n_merged_after_polish": 0,
            "optimiser": {"status": status, "nit": nit, "fun": fs, "reached": [int(v) for v in reached]},
            "bounds_unit": (lo_b, hi_b)}


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


def sampler_constants():
    """The constants of the start and first proposal, as each record's header carries them
    inside `lengths` (so that launch refuses a resume with other constants) and as the
    development study's header carried them under `candidate` (D93, D94; the same keys)."""
    return {"n_starts": N_STARTS, "ls_method": LS_METHOD, "ls_jac": LS_JAC, "ls_ftol": LS_FTOL, "ls_xtol": LS_XTOL,
            "ls_gtol": LS_GTOL, "ls_max_nfev": LS_MAX_NFEV, "fd_rel_step": FD_REL_STEP, "lbfgs_ftol": LBFGS_FTOL,
            "lbfgs_gtol": LBFGS_GTOL, "lbfgs_maxiter": LBFGS_MAXITER, "eps_H": EPS_H, "keep_max": KEEP_MAX,
            "distinct": DISTINCT, "mass_log": MASS_LOG, "eig_lo": EIG_LO, "eig_hi": EIG_HI, "perturb_cap": PERTURB_CAP,
            "perturb_mult": PERTURB_MULT, "kappa_u_floor": KAPPA_U_FLOOR, "scale": SCALE, "n_chains": N_CHAINS, "adapt_every": ADAPT_EVERY,
            "draw_order": "16x4 uniform starts, 4x4 standard normal perturbations, then proposals and acceptance draws"}


def run_chains(y, s, em, rng, prior="uniform", burn=BURN, sample=SAMPLE, n_chains=N_CHAINS,
               adapt_every=ADAPT_EVERY, box=BOX, target=None):
    """The adaptive random-walk Metropolis of line 68 with v3's start and first proposal
    (candidate (A), written in from p5_v3_study.candidate_run_chains as the development
    study ran it, D93 to D95). Returns a dict with the sampling-phase samples [n_chains,
    sample, 4] in unit coordinates, the acceptance rate over the sampling steps pooled over
    chains, the final proposal covariances, the lengths, and `start`, the run's start
    diagnostics. The draws from `rng`: the 16 × 4 uniform starts, the n_chains × 4 standard
    normals, then mh_step's, as v2 drew them. `target` (see log_target) substitutes another
    log density with the same mechanics; it must carry `grad_unit(x)` and `sigma`
    (p5_g3_calibrators.NRETarget), and the modes then come from the gradient route."""
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
        raise RuntimeError("the start kept no solution; the run is recorded as an error, not run from a uniform start")
    x, raw, n_clip, which = chain_starts([k_["x"] for k_ in kept], [k_["sigma"] for k_ in kept], z, lo_b, hi_b)
    opt_seconds = time.perf_counter() - t0
    lt = log_target(x, y, s, em, prior, box, target)
    cov = np.stack([SCALE * kept[which[k]]["sigma"] for k in range(n_chains)])      # step 5: the first proposal
    L = np.stack([_chol_pd(c) for c in cov])
    hist = np.empty((n_chains, burn + 1, d))
    hist[:, 0] = x
    acc_first = np.zeros(n_chains)
    n_first = min(burn, adapt_every)
    for t in range(1, burn + 1):                                     # v2's burn-in and adaptation, unchanged
        x, lt, acc = mh_step(x, lt, L, y, s, em, prior, rng, box, target)
        hist[:, t] = x
        if t <= n_first:
            acc_first += acc
        if t % adapt_every == 0:
            cov = np.stack([adapted_cov(hist[k, :t + 1]) for k in range(n_chains)])
            L = np.stack([_chol_pd(c) for c in cov])
    samples = np.empty((n_chains, sample, d))
    n_acc = 0
    for t in range(sample):                                          # v2's sampling phase, unchanged
        x, lt, acc = mh_step(x, lt, L, y, s, em, prior, rng, box, target)
        samples[:, t] = x
        n_acc += int(acc.sum())
    start = {"route": opt["route"], "prior": prior, "burn": int(burn), "sample": int(sample),
             "n_chains": int(n_chains), "optimiser_seconds": opt_seconds, "n_starts": N_STARTS, "starts": starts,
             "optimiser": opt["optimiser"], "m": m, "starts_elsewhere": opt["starts_elsewhere"], "kept": kept,
             "n_merged_after_polish": opt["n_merged_after_polish"],
             "chain_solution": which, "chain_start_raw": raw, "chain_start": hist[:, 0].copy(),
             "n_start_clipped": n_clip, "bounds_unit": [lo_b, hi_b],
             "acceptance_first_%d" % adapt_every: (acc_first / float(n_first)) if n_first else None,
             "proposal_vs_sample": _proposal_vs_sample(cov, samples) if sample > 1 else None}
    return {"samples": samples, "acceptance": n_acc / float(n_chains * sample), "cov": cov,
            "burn": int(burn), "sample": int(sample), "prior": prior, "start": start}


# --------------------------------------------------------------------------- #
# Diagnostics (line 68): split R-hat and the between-within effective sample size
# --------------------------------------------------------------------------- #
def split_chains(x):
    """[m, n] -> [2m, n // 2]: each chain split into halves (an odd last step dropped)."""
    x = np.asarray(x, dtype=float)
    h = x.shape[1] // 2
    return np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)


def _bw(c):
    """Between and within variances of chains c [m, n], and var⁺ = (n−1)/n W + B/n."""
    m, n = c.shape
    means = c.mean(axis=1)
    B = n * means.var(ddof=1)
    W = c.var(axis=1, ddof=1).mean()
    varp = (n - 1) / n * W + B / n
    return B, W, varp


def split_rhat(x):
    """The classical split R-hat of Gelman and others, √(var⁺ / W), on x [m, n]."""
    B, W, varp = _bw(split_chains(x))
    return float(np.sqrt(varp / W)) if W > 0 else float("inf")


def _variogram(c):
    """V_t = (1 / (m (n − t))) Σ_j Σ_i (ψ_{i,j} − ψ_{i−t,j})², t = 1..n−1, exactly, by FFT
    (each chain centred first: the variogram is shift-invariant and the centring keeps
    the cancellation between the cross term and the sums of squares small)."""
    c = np.asarray(c, dtype=float)
    c = c - c.mean(axis=1, keepdims=True)
    m, n = c.shape
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(c, nfft, axis=1)
    r = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :n]          # r[j, t] = Σ_i ψ_i ψ_{i−t}
    sq = np.cumsum(c ** 2, axis=1)
    tot = sq[:, -1:]
    t = np.arange(1, n)
    head = sq[:, n - 1 - t]                                           # Σ_{i < n−t} ψ_i²
    tail = tot - np.concatenate([np.zeros((m, 1)), sq[:, :n - 1]], axis=1)[:, t]   # Σ_{i ≥ t} ψ_i²
    V = (head + tail - 2.0 * r[:, 1:]) / (n - t)
    return V.mean(axis=0)


def ess_bw(x):
    """The between-within effective sample size (the variogram estimator of Gelman and
    others): ρ_t = 1 − V_t / (2 var⁺), summed up to the first odd T for which
    ρ_{T+1} + ρ_{T+2} < 0; n_eff = m n / (1 + 2 Σ_{t=1}^{T} ρ_t), on the split chains."""
    c = split_chains(x)
    m, n = c.shape
    B, W, varp = _bw(c)
    if varp <= 0:
        return float("nan")
    rho = 1.0 - _variogram(c) / (2.0 * varp)                          # rho[t−1] is ρ_t
    T = n - 1                                                         # no negative pair: every lag counts
    k = 1
    while k + 1 < len(rho):
        if rho[k] + rho[k + 1] < 0:                                   # ρ_{k+1} + ρ_{k+2} < 0 with T = k, k odd
            T = k
            break
        k += 2
    return float(m * n / (1.0 + 2.0 * np.sum(rho[:T])))


def accept(samples, rhat_max=RHAT_MAX, ess_min=ESS_MIN):
    """The acceptance of line 68 on samples [m, n, 4]: R-hat per parameter at most
    rhat_max and the effective sample size of H at least ess_min; the effective sample
    sizes of the other parameters are reported beside it. Returns (accepted, dict)."""
    rh = [split_rhat(samples[:, :, p]) for p in range(samples.shape[2])]
    ess = [ess_bw(samples[:, :, p]) for p in range(samples.shape[2])]
    ok = bool(all(r <= rhat_max for r in rh) and ess[0] >= ess_min)
    return ok, {"rhat": {d: rh[i] for i, d in enumerate(DIMS)}, "ess": {d: ess[i] for i, d in enumerate(DIMS)},
                "ess_H": ess[0], "accepted": ok}


# --------------------------------------------------------------------------- #
# Weighted summaries (line 68), the contraction (line 81), the stored samples
# --------------------------------------------------------------------------- #
def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def wvar(x, w):
    mu = wmean(x, w)
    return float(np.sum(w * (x - mu) ** 2) / np.sum(w))


def wquantile(x, w, q):
    """The smallest sample value at which the cumulative normalised weight reaches q."""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) / np.sum(ws)
    return float(xs[min(int(np.searchsorted(cum, q, side="left")), len(xs) - 1)])   # rounding can leave cum < 1


def kish(w):
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def interval_prob(x, w, lo, hi):
    """The weighted probability of lo ≤ x ≤ hi."""
    m = (x >= lo) & (x <= hi)
    return float(np.sum(w[m]) / np.sum(w))


def contraction(var_H):
    return float(1.0 - var_H / VAR_PRIOR_A)


def coverage_mass(lo, hi, which):
    """The mass prior `which` ('A' or 'B') puts on [lo, hi] in H: the coverage covariate
    (line 99, D81) when [lo, hi] is the equal-tailed 90% interval of H under the exact
    posterior under that prior. From the lower tail or the upper, whichever holds lo below
    one half, so that a small mass far in a tail keeps its digits."""
    a, b = PRIOR_AB[which]
    ul = min(max((float(lo) - H_LO) / H_W, 0.0), 1.0)
    uh = min(max((float(hi) - H_LO) / H_W, 0.0), 1.0)
    if uh <= ul:
        return 0.0
    if _beta.cdf(ul, a, b) < 0.5:
        return float(_beta.cdf(uh, a, b) - _beta.cdf(ul, a, b))
    return float(_beta.sf(ul, a, b) - _beta.sf(uh, a, b))


def coverage(under, cover_min=COVER_MIN):
    """Both priors' coverage masses from a pair's summaries, and whether the pair is well covered."""
    cov = {p: coverage_mass(under[p]["q05_H"], under[p]["q95_H"], p) for p in ("A", "B")}
    return cov, bool(all(v >= cover_min for v in cov.values()))


def quantise_H(H):
    return np.round((np.asarray(H, dtype=float) - H_LO) / H_W * QUANT).astype(np.uint16)


def dequantise_H(q):
    return H_LO + np.asarray(q, dtype=float) / QUANT * H_W


def summaries(H, xi0, w):
    """Weighted mean and variance of H, the equal-tailed 90% interval, the mean of ξ₀."""
    return {"mean_H": wmean(H, w), "var_H": wvar(H, w), "q05_H": wquantile(H, w, 0.05),
            "q95_H": wquantile(H, w, 0.95), "mean_xi0": wmean(xi0, w)}


# --------------------------------------------------------------------------- #
# One pair: the uniform run, the repeat, the reweighting, the direct runs
# --------------------------------------------------------------------------- #
def _finite(v):
    return float(v) if np.isfinite(v) else None


def _start_of(run):
    """A run's start diagnostics: `start` from run_chains, or `candidate` where the development
    study's candidate_run_chains (D93) is installed in its place, which returns the same
    diagnostics under that key."""
    return run.get("start", run.get("candidate"))


def _run_with_repeat(y, s, em, rng, prior, burn, sample, box, rhat_max=RHAT_MAX, ess_min=ESS_MIN, target=None):
    """One run; on failure of the acceptance, one repeat with doubled lengths, the start and
    first proposal run again from the same generator continuing. Returns (run or None, list
    of attempt records, each carrying the run's `start`)."""
    attempts = []
    for k, (b, n) in enumerate(((burn, sample), (2 * burn, 2 * sample))):
        t0 = time.perf_counter()
        run = run_chains(y, s, em, rng, prior, b, n, box=box, target=target)
        ok, diag = accept(run["samples"], rhat_max, ess_min)
        attempts.append({"attempt": k + 1, "burn": b, "sample": n, "acceptance": run["acceptance"],
                         "rhat": {d: _finite(v) for d, v in diag["rhat"].items()},
                         "ess": {d: _finite(v) for d, v in diag["ess"].items()}, "ess_H": _finite(diag["ess_H"]),
                         "accepted": ok, "seconds": time.perf_counter() - t0, "start": _jsonable(_start_of(run))})
        if ok:
            return run, attempts
    return None, attempts


def reference_pair(y, s, em, i, j, seed=SEED_REF, burn=BURN, sample=SAMPLE, kish_min=KISH_MIN, box=BOX,
                   rhat_max=RHAT_MAX, ess_min=ESS_MIN, cover_min=COVER_MIN):
    """The exact reference for one (surface, noise level) pair. Returns (record, stores):
    the record holds the summaries under A and B, each with Σ_P for the ratio estimator's
    sampler (the accepted direct run's Σ at its best kept solution where the figures under
    that prior are direct, else the uniform run's Σ with that prior's curvature added; line
    75, D95), c, the coverage covariate, the diagnostics with each run's start, the status
    and the seconds; stores holds the quantised H samples of each accepted run ('u', 'A',
    'B'). `kish_min` is the floor on the weighted chain's effective
    size, (Kish / pooled sampling steps) × the accepted uniform run's ESS of H (line 68,
    D81; the name is kept from D73, when the floor was on Kish's figure alone)."""
    rng = np.random.default_rng([seed, int(i), int(j)])
    t0 = time.perf_counter()
    rec = {"i": int(i), "j": int(j), "s": float(s), "seed": [seed, int(i), int(j)], "status": "ok",
           "runs": {}, "under": {}, "kish": {}, "neff": {}, "direct": [], "support_lo": None, "support_hi": None,
           "c_A": None, "coverage": None, "well_covered": None, "seconds": None}
    stores = {}
    run, attempts = _run_with_repeat(y, s, em, rng, "uniform", burn, sample, box, rhat_max, ess_min)
    rec["runs"]["uniform"] = attempts
    if run is None:
        rec["status"] = "reference_failure"
        rec["seconds"] = time.perf_counter() - t0
        return rec, stores
    pooled = unit_to_theta(run["samples"].reshape(-1, 4), box)
    H, xi0 = pooled[:, 0], pooled[:, 3]
    n_pooled, ess_H = H.shape[0], float(attempts[-1]["ess_H"])
    rec["support_lo"] = [float(v) for v in pooled.min(axis=0)]
    rec["support_hi"] = [float(v) for v in pooled.max(axis=0)]
    stores["u"] = quantise_H(H)
    for which in ("A", "B"):
        w = np.exp(prior_logpdf(H, which))
        w = w / np.sum(w)
        rec["kish"][which] = kish(w)
        rec["neff"][which] = rec["kish"][which] / n_pooled * ess_H        # the weighted chain's effective size (D81)
        if rec["neff"][which] >= kish_min:
            rec["under"][which] = dict(summaries(H, xi0, w), source="reweighted",
                                       sigma_P=_jsonable(_start_of(run)["kept"][0]["sigma_" + which]))
            continue
        rec["direct"].append(which)
        run_p, attempts_p = _run_with_repeat(y, s, em, rng, which, burn, sample, box, rhat_max, ess_min)
        rec["runs"][which] = attempts_p
        if run_p is None:
            rec["status"] = "reference_failure"
            rec["seconds"] = time.perf_counter() - t0
            return rec, stores
        pooled_p = unit_to_theta(run_p["samples"].reshape(-1, 4), box)
        w1 = np.ones(pooled_p.shape[0])
        rec["under"][which] = dict(summaries(pooled_p[:, 0], pooled_p[:, 3], w1), source="direct",
                                   sigma_P=_jsonable(_start_of(run_p)["kept"][0]["sigma"]))
        stores[which] = quantise_H(pooled_p[:, 0])
    rec["c_A"] = contraction(rec["under"]["A"]["var_H"])
    rec["coverage"], rec["well_covered"] = coverage(rec["under"], cover_min)
    rec["seconds"] = time.perf_counter() - t0
    return rec, stores


def weights_for(rec, stores, which="A"):
    """The H samples and weights that the pair's summaries under `which` rest on
    (the reweighted uniform run, or the direct run), from the stored quantised samples."""
    if rec["under"][which]["source"] == "direct":
        H = dequantise_H(stores[which])
        return H, np.ones_like(H)
    H = dequantise_H(stores["u"])
    w = np.exp(prior_logpdf(H, which))
    return H, w / np.sum(w)


# --------------------------------------------------------------------------- #
# Surfaces: the pilot sets (lines 143, 144) and the sealed test set (line 93)
# --------------------------------------------------------------------------- #
def pilot_points(seed, n, box=BOX):
    lo, hi = box_arrays(box)
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 4))


def noise_draws(seed, n, n_levels=len(NOISE_LEVELS), n_quotes=N_QUOTES):
    """ε[i, j] for the i-th point and the j-th level: standard_normal((n, n_levels, n_quotes))."""
    return np.random.default_rng(seed).standard_normal((n, n_levels, n_quotes))


def test_points(n=N_TEST, box=BOX):
    """The first n rows of the sealed draw, default_rng(20261002).uniform over the box,
    columns (H, ν, ρ, ξ₀); n = 400 (line 93, D81); the stream is sequential, so the set for
    n is the head of the set for any larger n."""
    return pilot_points(SEED_TEST, n, box)


def test_noise(n=N_TEST):
    """The test set's noise, default_rng(20261022).standard_normal((n, 3, 35)), indexed
    [θ, noise level 0.1, 0.3, 0.5 in that order, quote in the observable's order] (line 93, D81)."""
    return noise_draws(SEED_TEST_NOISE, n)


def g2_surfaces(retry=False, box=BOX):
    """G2's pilot set (line 143, D81; v3's seeds, D95): the 4 points uniform on the box, then
    the two fixed points; their noise, [6, 3, 35]. The first attempt: points default_rng(151),
    noise default_rng(157) for the four and default_rng(163) for the two; the one retry: fresh
    points default_rng(167), noise default_rng(173) and default_rng(179)."""
    seeds = ((SEED_G2R_POINTS, SEED_G2R_NOISE, SEED_G2R_FIXED_NOISE) if retry
             else (SEED_G2_POINTS, SEED_G2_NOISE, SEED_G2_FIXED_NOISE))
    pts = np.vstack([pilot_points(seeds[0], N_G2_POINTS, box), np.asarray(FIXED_POINTS, dtype=float)])
    eps = np.concatenate([noise_draws(seeds[1], N_G2_POINTS), noise_draws(seeds[2], len(FIXED_POINTS))], axis=0)
    return pts, eps, {"points": seeds[0], "noise": seeds[1], "fixed_points": [list(p) for p in FIXED_POINTS],
                      "fixed_noise": seeds[2]}


def surfaces(points, eps, em, levels=NOISE_LEVELS):
    """y[i, j] = G̃(θ_i) + s_j ε[i, j]."""
    G = em.evaluate_many(points)
    return G[:, None, :] + np.asarray(levels, dtype=float)[None, :, None] * np.asarray(eps, dtype=float)


# --------------------------------------------------------------------------- #
# G2's grids (line 143)
# --------------------------------------------------------------------------- #
def cell_centres(lo, hi, n):
    return lo + (np.arange(n) + 0.5) / n * (hi - lo)


def extended_support(support_lo, support_hi, extend=EXTEND, box=BOX):
    lo_b, hi_b = box_arrays(box)
    lo, hi = np.asarray(support_lo, dtype=float), np.asarray(support_hi, dtype=float)
    w = hi - lo
    return np.maximum(lo - extend * w, lo_b), np.minimum(hi + extend * w, hi_b)


def grid_quantiles(h_axis, H_cells, w, qs=(0.05, 0.95)):
    """Quantiles of H on a cell-centred grid: the marginal mass of each H cell, spread
    uniformly over the cell's width, so that the distribution function is linear within a
    cell and the quantile is read off it without the step of one cell (D81)."""
    h_axis = np.asarray(h_axis, dtype=float)
    mass = np.bincount(np.searchsorted(h_axis, H_cells), weights=w, minlength=h_axis.size)[:h_axis.size]
    half = (h_axis[1] - h_axis[0]) / 2.0 if h_axis.size > 1 else 0.0
    edges = np.concatenate([[h_axis[0] - half], h_axis + half])
    cdf = np.concatenate([[0.0], np.cumsum(mass)]) / np.sum(mass)
    return [float(np.interp(q, cdf, edges)) for q in qs]


def grid_posterior(axes, y, s, em, prior="A", chunk=8192, priors=("A", "B")):
    """The prior-weighted likelihood summed over the cells of the tensor grid with the given
    axes (box units), under each prior of `priors` from one evaluation of the likelihood per
    cell, never holding the quotes of more than `chunk` cells at once: per prior the mean
    and variance of H and the mean of ξ₀, and under A the 5% and 95% quantiles of H (D81).
    The keys mean_H, var_H and mean_xi0 at the top are those under `prior`."""
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    n = mesh.shape[0]
    logl = np.empty(n)
    for a in range(0, n, chunk):
        logl[a:a + chunk] = log_lik(mesh[a:a + chunk], y, s, em)
    out = {"n_cells": int(n)}
    for p in sorted(set(priors) | {prior}):
        logw = logl + prior_logpdf(mesh[:, 0], p)
        w = np.exp(logw - np.max(logw))
        res = {"mean_H": wmean(mesh[:, 0], w), "var_H": wvar(mesh[:, 0], w), "mean_xi0": wmean(mesh[:, 3], w)}
        if p == "A":
            res["q05_H"], res["q95_H"] = grid_quantiles(axes[0], mesh[:, 0], w)
        out[p] = res
    out.update({k: out[prior][k] for k in ("mean_H", "var_H", "mean_xi0")})
    return out


def grid_check(rec, y, s, em, n=GRID_N, extend=EXTEND, box=BOX, tol_mean=TOL_MEAN, tol_c=TOL_C, tol_q=TOL_Q):
    """The grid of line 143 over the sampled support, against the pair's figures under A and B."""
    lo, hi = extended_support(rec["support_lo"], rec["support_hi"], extend, box)
    axes = [cell_centres(lo[k], hi[k], n) for k in range(4)]
    t0 = time.perf_counter()
    g = grid_posterior(axes, y, s, em, "A")
    g["c_A"] = contraction(g["var_H"])
    g["grid_lo"], g["grid_hi"], g["n_per_dim"] = [float(v) for v in lo], [float(v) for v in hi], int(n)
    g["seconds"] = time.perf_counter() - t0
    return _compare(rec, g, tol_mean, tol_c, tol_q), axes


def whole_box_check(rec, y, s, em, xi0_axis, n=WHOLE_N, box=BOX, tol_mean=TOL_MEAN, tol_c=TOL_C, tol_q=TOL_Q):
    """The second grid of line 143: n cell-centred points in H, ν and ρ over the whole box,
    ξ₀ on the first grid's points; the same comparison and tolerance."""
    lo, hi = box_arrays(box)
    axes = [cell_centres(lo[k], hi[k], n) for k in range(3)] + [np.asarray(xi0_axis, dtype=float)]
    t0 = time.perf_counter()
    g = grid_posterior(axes, y, s, em, "A")
    g["c_A"] = contraction(g["var_H"])
    g["n_per_dim"], g["seconds"] = [int(n)] * 3 + [int(len(xi0_axis))], time.perf_counter() - t0
    return _compare(rec, g, tol_mean, tol_c, tol_q)


def run_neff(rec, which):
    """The effective sample size of the run a pair's figures under `which` came from (line 143,
    D81): (Kish / pooled sampling steps) × the uniform run's ESS of H for a reweighted figure,
    the accepted direct run's ESS of H for a direct one."""
    if rec["under"][which]["source"] == "direct":
        return float(rec["runs"][which][-1]["ess_H"])
    return float(rec["neff"][which])


def mc_errors(rec):
    """The Monte Carlo standard errors of the MCMC figures G2 compares, by the Gaussian reading
    of line 89, from n_eff of the run each came from (run_neff): a posterior mean sd / √n_eff;
    c (1 − c)·√(2 / (n_eff − 1)); the q quantile √(q(1 − q) / n_eff) / (φ(z_q) / sd); sd the
    MCMC posterior standard deviation of H under that prior (D81). An n_eff below 2, which
    the acceptance criteria exclude, is taken as 2."""
    nA, nB = max(run_neff(rec, "A"), 2.0), max(run_neff(rec, "B"), 2.0)
    sdA, sdB = math.sqrt(rec["under"]["A"]["var_H"]), math.sqrt(rec["under"]["B"]["var_H"])

    def se_q(q):
        return math.sqrt(q * (1.0 - q) / nA) / (_NORMAL.pdf(_NORMAL.inv_cdf(q)) / sdA)
    return {"n_eff_A": nA, "n_eff_B": nB, "d_mean": sdA / math.sqrt(nA),
            "d_c": (1.0 - rec["c_A"]) * math.sqrt(2.0 / (nA - 1.0)), "d_mean_B": sdB / math.sqrt(nB),
            "d_q05": se_q(0.05), "d_q95": se_q(0.95)}


def _compare(rec, g, tol_mean, tol_c, tol_q=TOL_Q, k_mc=TOL_MC_K):
    """G2's tolerance on one grid (line 143, D81): under A the mean of H, c and the 5% and
    95% quantiles of H; under B the mean of H, by the route the pair used (reweighted or
    direct); each tolerance the larger of its absolute value and k_mc Monte Carlo standard
    errors of the MCMC figure (mc_errors), on either grid."""
    uA, uB = rec["under"]["A"], rec["under"]["B"]
    d = {"d_mean": abs(uA["mean_H"] - g["A"]["mean_H"]), "d_c": abs(rec["c_A"] - g["c_A"]),
         "d_mean_B": abs(uB["mean_H"] - g["B"]["mean_H"]),
         "d_q05": abs(uA["q05_H"] - g["A"]["q05_H"]), "d_q95": abs(uA["q95_H"] - g["A"]["q95_H"])}
    d = {k: float(v) for k, v in d.items()}
    se = mc_errors(rec)
    absolute = {"d_mean": tol_mean, "d_c": tol_c, "d_mean_B": tol_mean, "d_q05": tol_q, "d_q95": tol_q}
    tol = {k: max(float(v), k_mc * se[k]) for k, v in absolute.items()}
    passed = all(d[k] <= tol[k] for k in absolute)
    return dict(g, mcmc_mean_H=uA["mean_H"], mcmc_c_A=rec["c_A"], mcmc_mean_H_B=uB["mean_H"], route_B=uB["source"],
                mcmc_q05_H=uA["q05_H"], mcmc_q95_H=uA["q95_H"], **d, tol_mean=tol_mean, tol_c=tol_c, tol_q=tol_q,
                k_mc=float(k_mc), mc_se={k: float(se[k]) for k in absolute},
                n_eff={"A": se["n_eff_A"], "B": se["n_eff_B"]}, tol=tol, passed=bool(passed))


# --------------------------------------------------------------------------- #
# Records, the pool, the sets
# --------------------------------------------------------------------------- #
_EM = {"path": None, "em": None}


def get_emulator(path=None):
    """The emulator for this process, loaded once; a test may install an object directly."""
    if _EM["em"] is not None and (path is None or _EM["path"] == path):
        return _EM["em"]
    path = path or EMULATOR_PATH
    if not os.path.exists(path):
        raise SystemExit(f"no emulator at {_rel(path)}: G̃ is the build G1 passed under v2 (line 142, D88)")
    _EM["path"], _EM["em"] = path, E.load_emulator(path)
    return _EM["em"]


def set_emulator(em):
    _EM["path"], _EM["em"] = "<object>", em


def _worker(args):
    job, kw = args
    em = get_emulator(kw.pop("emulator_path", None))          # every call: a worker's later pairs carry the path too (D92)
    rec, stores = reference_pair(job["y"], job["s"], em, job["i"], job["j"], **kw)
    rec.update(kind="task", task=job["label"], set=job["set"], theta=job["theta"], y=list(map(float, job["y"])),
               utc=utc_now())
    return rec, {f"{job['label']}_{k}": v for k, v in stores.items()}


def header(set_name, pool, emulator_path, n_points, seeds, extra=None, versions=None):
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    v = versions or E.version_check(E.CORE_VERSIONS, check=False)
    em_meta = {}
    if os.path.exists(emulator_path):
        with np.load(emulator_path, allow_pickle=False) as z:
            em_meta = {"stage": str(z["stage"]), "commit": str(z["commit"]), "nodes": [int(v_) for v_ in z["values"].shape[:4]]}
    return dict({"kind": "header", "script": os.path.basename(__file__), "commit": commit, "clean_tree": clean,
                 "set": set_name, "n_points": int(n_points), "seeds": seeds, "seed_ref": SEED_REF,
                 "emulator": _rel(emulator_path), "emulator_meta": em_meta,
                 "emulator_sha256": E.sha256_file(emulator_path),          # G̃'s hash, checked where records meet (D82)
                 "box": {d: list(BOX[d]) for d in DIMS}, "noise_levels": list(NOISE_LEVELS),
                 "n_chains": N_CHAINS, "burn": BURN, "sample": SAMPLE, "adapt_every": ADAPT_EVERY,
                 "scale": SCALE, "diag_floor": DIAG_FLOOR,
                 "start": sampler_constants(),                          # the start and first proposal (line 68; D95)
                 "rhat_max": RHAT_MAX, "ess_min": ESS_MIN, "kish_min": KISH_MIN,
                 "reweighting_rule": "(Kish / pooled sampling steps) x ESS of H >= kish_min (D81)",
                 "cover_min": COVER_MIN, "priors": {k: list(v_) for k, v_ in PRIOR_AB.items()}, "var_prior_A": VAR_PRIOR_A,
                 "grid_n": GRID_N, "extend": EXTEND, "whole_n": WHOLE_N, "tol_mean": TOL_MEAN, "tol_c": TOL_C,
                 "tol_q": TOL_Q, "tol_mc_k": TOL_MC_K,
                 "tolerance_rule": "each tolerance the larger of its value and tol_mc_k Monte Carlo standard errors (D81)",
                 "quant": QUANT, "python": platform.python_version(), "numpy": v["numpy"],
                 "scipy": v["scipy"], "versions": v, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                 "pool": int(pool), "utc_created": utc_now()}, **(extra or {}))


def samples_dir(samples_path):
    """The per-pair store beside the consolidated archive: one .npy per stored array."""
    base = samples_path[:-4] if samples_path.endswith(".npz") else samples_path
    return base + "_parts"


def save_pair_samples(samples_path, st):
    """Each array written to its own file atomically (a temporary name, then replaced)."""
    d = samples_dir(samples_path)
    os.makedirs(d, exist_ok=True)
    for k, v in st.items():
        tmp = os.path.join(d, k + ".tmp.npy")
        np.save(tmp, v)
        os.replace(tmp, os.path.join(d, k + ".npy"))


def load_samples(samples_path):
    """The consolidated archive, if any, overlaid by the per-pair files."""
    out = {}
    if os.path.exists(samples_path):
        with np.load(samples_path, allow_pickle=False) as z:
            out.update({k: np.array(z[k]) for k in z.files})
    d = samples_dir(samples_path)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy") and not f.endswith(".tmp.npy"):
                out[f[:-4]] = np.load(os.path.join(d, f), allow_pickle=False)
    return out


def consolidate_samples(samples_path, stores):
    """The consolidated compressed archive, written once, atomically."""
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    tmp = samples_path + ".tmp.npz"
    np.savez_compressed(tmp, **stores)
    os.replace(tmp, samples_path)


def samples_complete(rec, stores):
    """Whether every array a record's status implies is present in the store."""
    if rec.get("status") != "ok":
        return True
    need = [rec["task"] + "_u"] + [rec["task"] + "_" + p for p in rec.get("direct", [])]
    return all(k in stores for k in need)


def launch(pool, jobs, jsonl_path, samples_path, header_obj, task_kwargs=None):
    """The pool over `jobs`, skipping pairs already in the jsonl whose samples are stored;
    per pair the samples are written first, then the record line; the consolidated
    archive is written once at the end. Returns the count of pairs done now."""
    task_kwargs = dict(task_kwargs or {})
    hdr, launches, tasks = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, header_obj)
    else:
        for key in ("set", "n_points", "seeds", "burn", "sample", "emulator", "emulator_meta", "emulator_sha256", "lengths"):
            if hdr.get(key) != header_obj.get(key):
                raise SystemExit(f"launch refused: {_rel(jsonl_path)} was started with {key}={hdr.get(key)!r}, "
                                 f"this call has {header_obj.get(key)!r}")
    stores = load_samples(samples_path)
    done = {k for k, r in tasks.items() if samples_complete(r, stores)}
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    append_line(jsonl_path, {"kind": "launch", "utc": utc_now(), "pool": int(pool), "n_done_before": len(done),
                             "commit": commit, "clean_tree": clean, "versions": header_obj.get("versions")})
    todo = [j for j in jobs if j["label"] not in done]
    print(f"\n=== RUN: {len(todo)} pairs to do ({len(done)} already in {_rel(jsonl_path)}), pool={pool} ===", flush=True)
    t0 = time.time()
    n = n_err = 0

    def take(job, rec, st, err):
        nonlocal n, n_err
        if err is not None:
            n_err += 1
            append_line(jsonl_path, {"kind": "task_error", "task": job["label"], "utc": utc_now(), "error": repr(err)})
            print(f"  pair {job['label']} raised {err!r}; recorded, continuing", flush=True)
            return
        save_pair_samples(samples_path, st)                   # samples first, then the record that implies them
        stores.update(st)
        append_line(jsonl_path, rec)
        n += 1
        a = rec["under"].get("A", {})
        print(f"  [{n:3d}/{len(todo)}] {rec['task']}: {rec['status']} mean_H_A={a.get('mean_H')} c_A={rec['c_A']} "
              f"direct={rec['direct']} {rec['seconds']:.0f}s elapsed={(time.time() - t0) / 60:.1f}min", flush=True)

    if pool == 1:
        for job in todo:
            try:
                rec, st = _worker((job, dict(task_kwargs)))
                take(job, rec, st, None)
            except Exception as e:
                take(job, None, None, e)
    else:
        ex = ProcessPoolExecutor(max_workers=pool)
        try:
            futs = {ex.submit(_worker, (job, dict(task_kwargs))): job for job in todo}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    rec, st = fut.result()
                    take(job, rec, st, None)
                except Exception as e:
                    take(job, None, None, e)
        except BaseException:
            print("\n  stopping: pending pairs cancelled; re-run to resume", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    if stores:
        consolidate_samples(samples_path, stores)
    print(f"=== done: {n} pairs ({n_err} errors) in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    return n


def make_jobs(set_name, points, eps, em, levels=NOISE_LEVELS):
    Y = surfaces(points, eps, em, levels)
    jobs = []
    for i in range(points.shape[0]):
        for j, s in enumerate(levels):
            jobs.append({"label": f"{set_name}_{i:04d}_{j}", "set": set_name, "i": i, "j": j, "s": float(s),
                         "theta": [float(v) for v in points[i]], "y": Y[i, j]})
    return jobs


def load_references(jsonl_path, samples_path=None):
    """The header, the launches, the pair records by label, and the stored samples."""
    hdr, launches, tasks = read_jsonl(jsonl_path)
    stores = load_samples(samples_path) if samples_path else {}
    return hdr, launches, tasks, stores


def run_set(set_name, points, eps, em, pool, paths, emulator_path, seeds, levels=NOISE_LEVELS, task_kwargs=None,
            extra_header=None, versions=None):
    jobs = make_jobs(set_name, points, eps, em, levels)
    hdr = header(set_name, pool, emulator_path, points.shape[0], seeds, extra_header, versions)
    kw = dict(task_kwargs or {})
    hdr["lengths"] = {"burn": kw.get("burn", BURN), "sample": kw.get("sample", SAMPLE), "kish_min": kw.get("kish_min", KISH_MIN),
                      "rhat_max": kw.get("rhat_max", RHAT_MAX), "ess_min": kw.get("ess_min", ESS_MIN),
                      "levels": [float(s) for s in levels],
                      "start": sampler_constants()}                     # the lengths in force; burn and sample: as written;
    #                                                                     the start's constants, so a resume with others is refused (D95)
    if pool != 1:
        kw["emulator_path"] = emulator_path
    launch(pool, jobs, paths["jsonl"], paths["samples"], hdr, kw)
    return jobs


def _grid_job(args):
    """One pair's grid checks, in a worker (the emulator loaded once per process)."""
    job, rec, kw = args
    em = get_emulator(kw.pop("emulator_path", None))          # every call, as in _worker (D92)
    y = np.asarray(job["y"], dtype=float)
    gc, axes = grid_check(rec, y, job["s"], em, kw["grid_n"], kw["extend"], kw["box"], kw["tol_mean"], kw["tol_c"],
                          kw["tol_q"])
    row = {"grid": gc, "passed": gc["passed"]}
    if kw["whole"]:
        wb = whole_box_check(rec, y, job["s"], em, axes[3], kw["whole_n"], kw["box"], kw["tol_mean"], kw["tol_c"],
                             kw["tol_q"])
        row["whole_box"] = wb
        row["passed"] = bool(row["passed"] and wb["passed"])
    return row


def project_cost(attempt_lists, n_items, base_steps, cores=1.0):
    """The projected cost of n_items items of work (pairs, or posteriors) from the measured
    ones (line 143, D81): the median, over every sampler run measured, of its seconds per
    unit of `base_steps` (burn-in plus sampling steps of the lengths in force), times the
    mean number of such units per measured item (a repeat at doubled lengths, or a direct
    run, counted at its own length), times n_items, times `cores` (the cores a run holds:
    1 for a worker of the pool, the torch threads for a scoring process). In core-hours."""
    unit_s, per_item = [], []
    for attempts in attempt_lists:
        units = 0.0
        for a in attempts:
            u = (float(a["burn"]) + float(a["sample"])) / float(base_steps)
            units += u
            if a.get("seconds") is not None and u > 0:
                unit_s.append(float(a["seconds"]) / u)
        per_item.append(units)
    if not unit_s or not per_item:
        return {"median_unit_s": None, "mean_units": None, "n_items": int(n_items), "projected_core_hours": None,
                "n_runs_measured": len(unit_s), "n_items_measured": len(per_item)}
    med, mean_units = float(np.median(unit_s)), float(np.mean(per_item))
    return {"median_unit_s": med, "mean_units": mean_units, "n_items": int(n_items), "cores": float(cores),
            "projected_core_hours": n_items * med * mean_units * float(cores) / 3600.0,
            "n_runs_measured": len(unit_s), "n_items_measured": len(per_item), "base_steps": int(base_steps)}


def pair_attempts(rec):
    """Every sampler run of a pair's reference: the uniform run and any direct runs, repeats included."""
    return [a for k in ("uniform", "A", "B") for a in rec.get("runs", {}).get(k, [])]


def g2_report(jobs, paths, em, pool, levels=NOISE_LEVELS, grid_n=GRID_N, whole_n=WHOLE_N, extend=EXTEND, box=BOX,
              tol_mean=TOL_MEAN, tol_c=TOL_C, emulator_path=None, tol_q=TOL_Q, lengths=None, attempt=1,
              n_test_pairs=N_TEST_PAIRS, budget_core_h=BUDGET_CORE_H, versions=None):
    """The grid checks on every accepted pair (a pool over pairs), the gate's verdict and the
    go or no-go on cost (D81); written as JSON. `passed` is the checks' verdict (every pair
    accepted and within the tolerance on every grid); `go` is the projected cost of the test
    set's references within the budget, at this attempt's lengths, which are the record's
    (its header's `lengths`; a `lengths` given that differs is refused). Refuses if a pair is
    missing from the record or its stored surface is not the one the current emulator gives
    (a rebuilt emulator)."""
    if pool != 1 and not emulator_path:                       # the grid checks' workers load it from the path (D92)
        raise SystemExit("report refused: at a pool above 1 the workers load the emulator from emulator_path, and "
                         "none was given")
    hdr, launches, tasks, stores = load_references(paths["jsonl"], paths["samples"])
    hl = (hdr or {}).get("lengths")
    rec_lengths = (int(hl["burn"]), int(hl["sample"])) if hl else None
    if lengths is not None and rec_lengths is not None and tuple(int(v) for v in lengths) != rec_lengths:
        raise SystemExit(f"report refused: the record was made at lengths {rec_lengths}, this call says {tuple(lengths)}")
    lengths = rec_lengths or (tuple(lengths) if lengths is not None else (BURN, SAMPLE))
    out, all_pass, n_fail, grid_jobs = [], True, 0, []
    j_max = int(np.argmax(levels))
    kw = {"grid_n": grid_n, "whole_n": whole_n, "extend": extend, "box": box, "tol_mean": tol_mean, "tol_c": tol_c,
          "tol_q": tol_q}
    for job in jobs:
        rec = tasks.get(job["label"])
        if rec is None:
            raise SystemExit(f"report refused: pair {job['label']} is not in the record")
        if not np.allclose(rec["y"], job["y"], rtol=0, atol=1e-12):
            raise SystemExit(f"report refused: pair {job['label']}'s surface in the record differs from the current "
                             "emulator's; the record was made with another emulator")
        row = {"task": job["label"], "i": job["i"], "j": job["j"], "s": job["s"], "status": rec["status"],
               "seconds": rec["seconds"], "runs": rec["runs"], "kish": rec["kish"], "direct": rec["direct"]}
        out.append(row)
        if rec["status"] != "ok":
            n_fail += 1
            all_pass = False
            continue
        grid_jobs.append((len(out) - 1, (job, rec, dict(kw, whole=(job["j"] == j_max),
                                                        **({"emulator_path": emulator_path} if pool != 1 else {})))))
    E.refuse_overwrite([paths["report"]], "report")                    # the retry and the go key on it (D82)
    if pool == 1:
        results = [(idx, _grid_job(args)) for idx, args in grid_jobs]
    else:
        ex = ProcessPoolExecutor(max_workers=pool)
        try:
            futs = {ex.submit(_grid_job, args): idx for idx, args in grid_jobs}
            results = [(futs[f], f.result()) for f in as_completed(futs)]
        finally:
            ex.shutdown(wait=True)
    for idx, row in results:
        out[idx].update(row)
        all_pass = all_pass and row["passed"]
    secs = [r["seconds"] for r in tasks.values() if r.get("status") == "ok"]
    gsecs = [r["grid"]["seconds"] for r in out if "grid" in r]
    wsecs = [r["whole_box"]["seconds"] for r in out if "whole_box" in r]
    proj = project_cost([pair_attempts(tasks[j["label"]]) for j in jobs], n_test_pairs, int(lengths[0]) + int(lengths[1]))
    go = bool(proj["projected_core_hours"] is not None and proj["projected_core_hours"] <= budget_core_h)
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"gate": "G2", "attempt": int(attempt), "lengths": {"burn": int(lengths[0]), "sample": int(lengths[1])},
              "passed": bool(all_pass), "n_pairs": len(jobs), "n_reference_failures": n_fail,
              "emulator": {"path": (hdr or {}).get("emulator"), "sha256": (hdr or {}).get("emulator_sha256")},
              "tolerance": {"mean_H": tol_mean, "c": tol_c, "quantiles_H_A": tol_q, "mean_H_B": tol_mean,
                            "mc_k": TOL_MC_K, "rule": "the larger of the value and mc_k Monte Carlo standard errors of "
                                                      "the MCMC figure, per pair and quantity (D81)"},
              "grid_n": grid_n, "whole_n": whole_n, "extend": extend,
              "go": go, "budget_core_hours": budget_core_h, "projection": proj,
              "cost": {"posterior_seconds": secs, "posterior_mean_s": float(np.mean(secs)) if secs else None,
                       "posterior_median_s": float(np.median(secs)) if secs else None,
                       "grid_mean_s": float(np.mean(gsecs)) if gsecs else None,
                       "whole_box_mean_s": float(np.mean(wsecs)) if wsecs else None, "pool": int(pool),
                       "note": "wall seconds inside a worker with `pool` workers running, one BLAS thread each: "
                               "core-seconds, as G0 read them (line 141)"},
              "pairs": out, "header": hdr, "launches": launches, "report_commit": commit, "report_clean_tree": clean,
              "versions": versions or E.version_check(E.CORE_VERSIONS, check=False), "utc_report": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    return report


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def g2_status(first_report=None, retry_report=None):
    """G2's outcome from its reports (line 143, D81). Returns {passed, p5_stops, retry_due,
    attempt, burn, sample, go, projected_core_hours, reason}: `passed` means the later runs
    (G3's pilot references, the test set) may be made, at lengths `burn` and `sample`. The
    first attempt failing its checks makes the one retry due; the retry failing its checks,
    or a passing attempt's projected cost above the budget, stops P5."""
    first_report = first_report or SETS["g2"]["report"]
    retry_report = retry_report or SETS["g2r"]["report"]
    out = {"passed": False, "p5_stops": False, "retry_due": False, "attempt": None, "burn": None, "sample": None,
           "go": None, "projected_core_hours": None, "emulator_sha256": None, "reason": None}
    r1 = _load_json(first_report)
    if r1 is None:
        return dict(out, reason=f"G2 has not been run: no report at {_rel(first_report)}")
    r = r1
    if not r1.get("passed"):
        r2 = _load_json(retry_report)
        if r2 is None:
            return dict(out, retry_due=True, reason="G2 failed its checks on the first attempt; the one retry "
                                                    "(--pilot-retry) is due")
        if not r2.get("passed"):
            return dict(out, p5_stops=True, attempt=2, reason="G2 failed its checks on the first attempt and on the "
                                                              "retry: P5 stops (line 143)")
        r = r2
    out.update(attempt=int(r.get("attempt", 1)), burn=int(r["lengths"]["burn"]), sample=int(r["lengths"]["sample"]),
               go=bool(r.get("go")), projected_core_hours=(r.get("projection") or {}).get("projected_core_hours"),
               emulator_sha256=(r.get("emulator") or {}).get("sha256"))       # the passing attempt's G̃ (D82)
    if not r.get("go"):
        return dict(out, p5_stops=True, reason=f"G2's projected cost of the test set's references, "
                                               f"{out['projected_core_hours']} core-hours, is above "
                                               f"{r.get('budget_core_hours')}: P5 stops (line 143)")
    return dict(out, passed=True, reason=f"G2 passed on attempt {out['attempt']} with a go; the reference's lengths are "
                                         f"burn {out['burn']}, sample {out['sample']}")


def require_g2(first_report=None, retry_report=None):
    """The status, or a refusal: the later runs are made only after G2 has passed with a go."""
    st = g2_status(first_report, retry_report)
    if not st["passed"]:
        raise SystemExit(f"refused: G2 has not passed with a go ({st['reason']}); the later references are made only "
                         "after it (lines 93, 143)")
    return st


# --------------------------------------------------------------------------- #
def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 gate G2 reference (see module docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="G2: the 18 pilot references, the grid checks and the report")
    mode.add_argument("--pilot-retry", action="store_true",
                      help="G2's one retry, with doubled lengths on fresh surfaces, only after a failed first attempt")
    mode.add_argument("--g3-pilot", action="store_true", help="the 48 references of G3's pilot surfaces, after G2")
    mode.add_argument("--test", action="store_true", help="the sealed test set (N = 400), after G2")
    ap.add_argument("--n", type=int, default=N_TEST, help="number of test θ (line 93: 400; any other value is refused, D82)")
    ap.add_argument("--noise-seed", type=int, default=SEED_TEST_NOISE,
                    help="the test set's noise seed (line 93: 20261022; any other value is refused, D82)")
    ap.add_argument("--emulator", default=None,
                    help="G̃ is the build G1 passes (p5_g1_emulator.require_g1); another file is refused (D82)")
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy or scipy other than p5_g1_emulator.CORE_VERSIONS; recorded (D81)")
    a = ap.parse_args(argv)
    vers = E.version_check(E.CORE_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions)

    print("=== P5 G2 reference ===", flush=True)
    print(f"  chains={N_CHAINS} burn={BURN} sample={SAMPLE} adapt_every={ADAPT_EVERY} scale={SCALE:.4f} "
          f"floor={DIAG_FLOOR}; R-hat<={RHAT_MAX} ESS_H>={ESS_MIN:.0f}; "
          f"(Kish/steps) x ESS_H>={KISH_MIN:.0f}", flush=True)
    print(f"  start (A, line 68; D95): {N_STARTS} starts; least_squares {LS_METHOD}, jac {LS_JAC}, bounds [0,1]^4, "
          f"ftol {LS_FTOL:g} xtol {LS_XTOL:g} gtol {LS_GTOL:g} max_nfev {LS_MAX_NFEV}; polish (direct runs) L-BFGS-B "
          f"ftol {LBFGS_FTOL:g} gtol {LBFGS_GTOL:g} maxiter {LBFGS_MAXITER}, H in [{EPS_H:g}, 1-{EPS_H:g}]; fd step "
          f"{FD_REL_STEP:.3e}; keep <= {KEEP_MAX}, distinct > {DISTINCT}, mass within log 1e4 = {MASS_LOG:.4f}; Sigma "
          f"eigenvalues in [{EIG_LO:g}, {EIG_HI:.6f}], kappa at x_H in [{KAPPA_U_FLOOR}, {1 - KAPPA_U_FLOOR}]; "
          f"start = solution + N(0, {PERTURB_MULT:g} Sigma_s), cap {PERTURB_CAP}; first proposal {SCALE:.4f} x Sigma",
          flush=True)
    print(f"  priors A={PRIOR_AB['A']} B=({PRIOR_AB['B'][0]:.4f}, {PRIOR_AB['B'][1]:.4f}); Var_prior(A)={VAR_PRIOR_A:.6f}; "
          f"well covered: both masses >= {COVER_MIN}", flush=True)
    print(f"  G2 (v3's seeds, D95): {N_G2_POINTS} points rng({SEED_G2_POINTS}) + {len(FIXED_POINTS)} fixed {FIXED_POINTS} x {len(NOISE_LEVELS)} "
          f"levels, noise rng({SEED_G2_NOISE}) and rng({SEED_G2_FIXED_NOISE}); grids {GRID_N}^4 on the support "
          f"+{EXTEND:.0%}, {WHOLE_N}^3 x {GRID_N}; tol mean {TOL_MEAN} (A and B), c {TOL_C}, quantiles {TOL_Q}, "
          f"each at least {TOL_MC_K:g} Monte Carlo standard errors", flush=True)
    print(f"  retry: lengths doubled, points rng({SEED_G2R_POINTS}), noise rng({SEED_G2R_NOISE}) and rng({SEED_G2R_FIXED_NOISE}); "
          f"go if the test set's {N_TEST_PAIRS} pairs project to <= {BUDGET_CORE_H:.0f} core-hours (a run's seconds include its optimiser)", flush=True)
    print(f"  records: {_rel(SETS['g2']['jsonl'])}, {_rel(SETS['g2r']['jsonl'])}, {_rel(SETS['g3']['jsonl'])}, "
          f"{_rel(SETS['test']['jsonl'])} (v3's names; v2's files stay)", flush=True)
    print(f"  test set: N = {N_TEST}, theta rng({SEED_TEST}), noise rng({SEED_TEST_NOISE}); "
          f"numpy {vers['numpy']} scipy {vers['scipy']} (expected {E.CORE_VERSIONS})", flush=True)
    if not (a.pilot or a.pilot_retry or a.g3_pilot or a.test):
        print("\nHOLD: nothing run. --pilot is G2, run after v3's freeze on G̃, the build G1 passed under v2 (line 142).", flush=True)
        return
    if a.test and (a.n != N_TEST or a.noise_seed != SEED_TEST_NOISE):                   # before any draw (D82)
        raise SystemExit(f"refused: the test set is N = {N_TEST} with noise seed {SEED_TEST_NOISE} (line 93); this call "
                         f"has --n {a.n} --noise-seed {a.noise_seed}")
    st = None
    if a.pilot_retry:
        r1 = _load_json(SETS["g2"]["report"])
        if r1 is None:
            raise SystemExit("refused: --pilot-retry follows a failed first attempt, and there is no first report")
        if r1.get("passed"):
            raise SystemExit("refused: G2 passed its checks on the first attempt; there is no retry (line 143)")
    if a.g3_pilot or a.test:
        st = require_g2()
    em_path = E.emulator_for_run(a.emulator)                                            # G̃, the build G1 passes (D82)
    em = get_emulator(em_path)
    if a.pilot or a.pilot_retry:
        retry = bool(a.pilot_retry)
        set_name = "g2r" if retry else "g2"
        lengths = (2 * BURN, 2 * SAMPLE) if retry else (BURN, SAMPLE)
        pts, eps, seeds = g2_surfaces(retry)
        jobs = run_set(set_name, pts, eps, em, a.pool, SETS[set_name], em_path, seeds,
                       task_kwargs={"burn": lengths[0], "sample": lengths[1]}, versions=vers)
        rep = g2_report(jobs, SETS[set_name], em, a.pool, emulator_path=em_path, lengths=lengths,
                        attempt=2 if retry else 1, versions=vers)
        print(f"=== G2 attempt {rep['attempt']}: checks {'passed' if rep['passed'] else 'NOT passed'}; reference failures "
              f"{rep['n_reference_failures']}; projected {rep['projection']['projected_core_hours']} core-hours for the "
              f"test set ({'go' if rep['go'] else 'no go'}) ===", flush=True)
        for r in rep["pairs"]:
            g = r.get("grid", {})
            print(f"  {r['task']}: {r['status']} d_mean={g.get('d_mean')} d_c={g.get('d_c')} d_mean_B={g.get('d_mean_B')} "
                  f"d_q05={g.get('d_q05')} d_q95={g.get('d_q95')} passed={r.get('passed')}"
                  + (f" whole_box passed={r['whole_box']['passed']}" if "whole_box" in r else ""), flush=True)
        print(f"  written: {_rel(SETS[set_name]['report'])}", flush=True)
        print(f"  status: {g2_status()['reason']}", flush=True)
        return
    if a.g3_pilot:
        pts, eps = pilot_points(SEED_G3_POINTS, N_G3_POINTS), noise_draws(SEED_G3_NOISE, N_G3_POINTS)
        run_set("g3", pts, eps, em, a.pool, SETS["g3"], em_path, {"points": SEED_G3_POINTS, "noise": SEED_G3_NOISE},
                task_kwargs={"burn": st["burn"], "sample": st["sample"]}, extra_header={"g2": st}, versions=vers)
        return
    if a.test:
        pts, eps = test_points(a.n), noise_draws(a.noise_seed, a.n)
        run_set("test", pts, eps, em, a.pool, SETS["test"], em_path, {"points": SEED_TEST, "noise": int(a.noise_seed)},
                task_kwargs={"burn": st["burn"], "sample": st["sample"]},
                extra_header={"n_test": int(a.n), "n_test_written": N_TEST, "noise_seed_written": SEED_TEST_NOISE, "g2": st},
                versions=vers)


if __name__ == "__main__":
    main(check_versions=True)
