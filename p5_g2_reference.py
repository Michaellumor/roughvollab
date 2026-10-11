"""
p5_g2_reference.py — P5 gate G2, the exact reference (docs/protocols/P5_protocol_v2_draft.md,
Objects "The exact reference", D66; Gates, G2; the ξ₀ means of D67; the priors of D69).

Per scored (surface, noise level) pair: the posterior of θ = (H, ν, ρ, ξ₀) on the box
under the likelihood Gaussian with the known s on the 35 quotes, log L = −Σ(y − G̃(θ))² /
(2 s²), and the uniform box prior, by adaptive random-walk Metropolis on the emulator G̃
in the box mapped affinely to [0, 1]⁴. One generator per pair, default_rng([83, i, j]) for
the i-th θ of the set in question and the j-th noise level, serves the starts, the
proposals and the acceptance draws (in that order within a step). Four chains from
uniform starts; the proposal covariance 2.38²/4 × 1/12 × I at the start and, at burn-in
steps 500, 1,000, …, 10,000, 2.38²/4 × the empirical covariance of that chain's steps so
far (the start included), its diagonal floored at 1e-10; the adaptation at step 10,000
is the sampling proposal, fixed over the 40,000 sampling steps; a proposal with any
coordinate outside the box is rejected without evaluation. Accepted when, on the
sampling steps with each chain split into halves, the classical between-within split
R-hat (Gelman and others; not rank-normalised) is at most 1.01 on each parameter and
the effective sample size of H over the eight half-chains, by the variogram
between-within estimator, is at least 2,000; otherwise repeated once with both lengths
doubled from fresh starts of the same generator; a second failure is a reference failure
for the pair, recorded. Priors A and B by self-normalised importance weights ∝ the
prior's density of H at each pooled sampling step, with Kish's effective sample size
(Σw)²/Σw² at least 1,000 under each, else a direct run under that prior with the same
sampler and acceptance, from the same generator continuing. Weighted means, variances,
quantiles (the smallest sample value at which the cumulative weight reaches q) and
interval probabilities of the pooled sampling steps. Reported per pair: the posterior
means of H under A and B, its variance under A and the contraction c against the
analytic Var_prior(H) of prior A, the equal-tailed 90% interval under A, the means of ξ₀
under A and B, the acceptance rate, R-hat and the effective sample sizes; the H samples
of every accepted run are stored (quantised to 16 bits on the box's range of H) so that
the posterior probability under A of any interval can be computed later.

G2 (line 143): on the 12 pilot surfaces (4 points uniform on the box from default_rng(79)
at the three noise levels, one noise draw each from default_rng(89)), the MCMC posterior
of H against a cell-centred tensor grid of 32 points per dimension over the smallest
axis-aligned box holding the four chains' sampling-phase samples, each side extended by
20% and clipped to the box, the posterior being the prior-A-weighted likelihood summed
over cells; at the largest noise level a second grid of 48 cell-centred points in each
of H, ν and ρ over the whole box with ξ₀ on the first grid's 32 points; tolerance under
prior A on every pilot surface: the posterior mean of H within 0.001 and c within 0.01.
The cost per posterior is measured. G3's 48 pilot surfaces (default_rng(97) points,
default_rng(101) noise) get the same reference (line 144). The sealed test set is the
first N rows of default_rng(20261002)'s uniform draw on the box (line 93), N and the
noise seed being arguments until line 94's amendment declares them.

Nothing here runs before freezing: the emulator (output/p5_g1_emulator.npz) is built at
G1 after freezing, and this script needs it. The tests use toy emulators in memory.

Usage (from the repository root, after G1):
    py -3 p5_g2_reference.py                   the design; holds
    py -3 p5_g2_reference.py --pilot           G2: the 12 references, the grid checks, the report
    py -3 p5_g2_reference.py --g3-pilot        the 48 references of G3's pilot surfaces
    py -3 p5_g2_reference.py --test --n N --noise-seed S     the sealed set (after line 94's amendment)
    options: --emulator PATH (default output/p5_g1_emulator.npz), --pool K (default 4)

Outputs (output/ is git-ignored; the files are force-added by name after a run):
    output/p5_g2_pilot.jsonl, output/p5_g2_pilot_samples.npz, output/p5_g2_report.json
    output/p5_g3_pilot_refs.jsonl, output/p5_g3_pilot_refs_samples.npz
    output/p5_ref_test.jsonl, output/p5_ref_test_samples.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (calibrate_btc.py)

import sys
import json
import time
import argparse
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import beta as _beta

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
SEED_REF = 83                                       # default_rng([83, i, j]) per pair (line 68)
SEED_G2_POINTS, N_G2_POINTS, SEED_G2_NOISE = 79, 4, 89       # G2's pilot surfaces (line 143)
SEED_G3_POINTS, N_G3_POINTS, SEED_G3_NOISE = 97, 16, 101     # G3's pilot surfaces (line 144)
SEED_TEST = 20261002                                # the sealed test θ (line 93)
N_CHAINS = 4
BURN, SAMPLE, ADAPT_EVERY = 10000, 40000, 500
SCALE = 2.38 ** 2 / 4                               # 1.4161
INIT_VAR = 1.0 / 12.0                               # the variance of the uniform start
DIAG_FLOOR = 1e-10
RHAT_MAX, ESS_MIN, KISH_MIN = 1.01, 2000.0, 1000.0
GRID_N, EXTEND, WHOLE_N = 32, 0.20, 48              # G2's grids (line 143)
TOL_MEAN, TOL_C = 0.001, 0.01                       # G2's tolerance under prior A
H_LO, H_W = BOX["H"][0], BOX["H"][1] - BOX["H"][0]  # 0.02, 0.46
PRIOR_AB = {"A": (2.5, 9.0), "B": (235.0 / 23.0, 376.0 / 69.0)}   # line 62 (D69)
VAR_PRIOR_A = H_W ** 2 * 36.0 / 2645.0              # 0.00288, analytic (line 66)
POOL_DEFAULT = 4
QUANT = 65535                                       # H stored as uint16 on [H_LO, H_LO + H_W]

OUT_DIR = P.OUT_DIR
EMULATOR_PATH = E.paths("base")["npz"]
SETS = {"g2": {"jsonl": os.path.join(OUT_DIR, "p5_g2_pilot.jsonl"),
               "samples": os.path.join(OUT_DIR, "p5_g2_pilot_samples.npz"),
               "report": os.path.join(OUT_DIR, "p5_g2_report.json")},
        "g3": {"jsonl": os.path.join(OUT_DIR, "p5_g3_pilot_refs.jsonl"),
               "samples": os.path.join(OUT_DIR, "p5_g3_pilot_refs_samples.npz")},
        "test": {"jsonl": os.path.join(OUT_DIR, "p5_ref_test.jsonl"),
                 "samples": os.path.join(OUT_DIR, "p5_ref_test_samples.npz")}}


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


def log_target(x, y, s, em, prior, box=BOX):
    """log posterior (up to a constant) at unit-cube points x [M, 4]; −inf outside."""
    x = np.asarray(x, dtype=float).reshape(-1, 4)
    out = np.full(x.shape[0], -np.inf)
    inside = np.all((x >= 0.0) & (x <= 1.0), axis=1)
    if inside.any():
        th = unit_to_theta(x[inside], box)
        out[inside] = log_lik(th, y, s, em) + prior_logpdf(th[:, 0], prior)
    return out


# --------------------------------------------------------------------------- #
# The sampler (line 68)
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


def mh_step(x, lt, L, y, s, em, prior, rng, box=BOX):
    """One Metropolis step for every chain: z then u are drawn from rng in that order,
    whether or not a proposal lies inside the box. Returns (x, lt, accepted)."""
    n, d = x.shape
    z = rng.standard_normal((n, d))
    prop = x + np.einsum("kij,kj->ki", L, z)
    inside = np.all((prop >= 0.0) & (prop <= 1.0), axis=1)
    lt_prop = np.full(n, -np.inf)
    if inside.any():
        lt_prop[inside] = log_target(prop[inside], y, s, em, prior, box)
    u = rng.uniform(size=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        accept = inside & (np.log(u) < lt_prop - lt)
    x = x.copy()
    lt = lt.copy()
    x[accept] = prop[accept]
    lt[accept] = lt_prop[accept]
    return x, lt, accept


def run_chains(y, s, em, rng, prior="uniform", burn=BURN, sample=SAMPLE, n_chains=N_CHAINS,
               adapt_every=ADAPT_EVERY, box=BOX):
    """The adaptive random-walk Metropolis of line 68. Returns a dict with the
    sampling-phase samples [n_chains, sample, 4] in unit coordinates, the acceptance rate
    over the sampling steps pooled over chains, the final proposal covariances, lengths."""
    d = 4
    x = rng.uniform(size=(n_chains, d))
    lt = log_target(x, y, s, em, prior, box)
    cov = np.tile(SCALE * INIT_VAR * np.eye(d), (n_chains, 1, 1))
    L = np.stack([_chol_pd(c) for c in cov])
    hist = np.empty((n_chains, burn + 1, d))
    hist[:, 0] = x
    for t in range(1, burn + 1):
        x, lt, _ = mh_step(x, lt, L, y, s, em, prior, rng, box)
        hist[:, t] = x
        if t % adapt_every == 0:
            cov = np.stack([adapted_cov(hist[k, :t + 1]) for k in range(n_chains)])
            L = np.stack([_chol_pd(c) for c in cov])
    samples = np.empty((n_chains, sample, d))
    n_acc = 0
    for t in range(sample):
        x, lt, acc = mh_step(x, lt, L, y, s, em, prior, rng, box)
        samples[:, t] = x
        n_acc += int(acc.sum())
    return {"samples": samples, "acceptance": n_acc / float(n_chains * sample), "cov": cov,
            "burn": int(burn), "sample": int(sample), "prior": prior}


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


def _run_with_repeat(y, s, em, rng, prior, burn, sample, box, rhat_max=RHAT_MAX, ess_min=ESS_MIN):
    """One run; on failure of the acceptance, one repeat with doubled lengths from fresh
    starts of the same generator. Returns (run or None, list of attempt records)."""
    attempts = []
    for k, (b, n) in enumerate(((burn, sample), (2 * burn, 2 * sample))):
        t0 = time.perf_counter()
        run = run_chains(y, s, em, rng, prior, b, n, box=box)
        ok, diag = accept(run["samples"], rhat_max, ess_min)
        attempts.append({"attempt": k + 1, "burn": b, "sample": n, "acceptance": run["acceptance"],
                         "rhat": {d: _finite(v) for d, v in diag["rhat"].items()},
                         "ess": {d: _finite(v) for d, v in diag["ess"].items()}, "ess_H": _finite(diag["ess_H"]),
                         "accepted": ok, "seconds": time.perf_counter() - t0})
        if ok:
            return run, attempts
    return None, attempts


def reference_pair(y, s, em, i, j, seed=SEED_REF, burn=BURN, sample=SAMPLE, kish_min=KISH_MIN, box=BOX,
                   rhat_max=RHAT_MAX, ess_min=ESS_MIN):
    """The exact reference for one (surface, noise level) pair. Returns (record, stores):
    the record holds the summaries under A and B, c, the diagnostics, the status and the
    seconds; stores holds the quantised H samples of each accepted run ('u', 'A', 'B')."""
    rng = np.random.default_rng([seed, int(i), int(j)])
    t0 = time.perf_counter()
    rec = {"i": int(i), "j": int(j), "s": float(s), "seed": [seed, int(i), int(j)], "status": "ok",
           "runs": {}, "under": {}, "kish": {}, "direct": [], "support_lo": None, "support_hi": None,
           "c_A": None, "seconds": None}
    stores = {}
    run, attempts = _run_with_repeat(y, s, em, rng, "uniform", burn, sample, box, rhat_max, ess_min)
    rec["runs"]["uniform"] = attempts
    if run is None:
        rec["status"] = "reference_failure"
        rec["seconds"] = time.perf_counter() - t0
        return rec, stores
    pooled = unit_to_theta(run["samples"].reshape(-1, 4), box)
    H, xi0 = pooled[:, 0], pooled[:, 3]
    rec["support_lo"] = [float(v) for v in pooled.min(axis=0)]
    rec["support_hi"] = [float(v) for v in pooled.max(axis=0)]
    stores["u"] = quantise_H(H)
    for which in ("A", "B"):
        w = np.exp(prior_logpdf(H, which))
        w = w / np.sum(w)
        rec["kish"][which] = kish(w)
        if rec["kish"][which] >= kish_min:
            rec["under"][which] = dict(summaries(H, xi0, w), source="reweighted")
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
        rec["under"][which] = dict(summaries(pooled_p[:, 0], pooled_p[:, 3], w1), source="direct")
        stores[which] = quantise_H(pooled_p[:, 0])
    rec["c_A"] = contraction(rec["under"]["A"]["var_H"])
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


def test_points(n, box=BOX):
    """The first n rows of the sealed draw, default_rng(20261002).uniform over the box;
    the stream is sequential, so the set for n is the head of the set for any larger n."""
    return pilot_points(SEED_TEST, n, box)


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


def grid_posterior(axes, y, s, em, prior="A", chunk=8192):
    """Mean and variance of H and the mean of ξ₀ of the prior-weighted likelihood summed
    over the cells of the tensor grid with the given axes (box units), never holding the
    quotes of more than `chunk` cells at once. Returns the dict and the number of cells."""
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    n = mesh.shape[0]
    logw = np.empty(n)
    for a in range(0, n, chunk):
        th = mesh[a:a + chunk]
        logw[a:a + chunk] = log_lik(th, y, s, em) + prior_logpdf(th[:, 0], prior)
    w = np.exp(logw - np.max(logw))
    return {"mean_H": wmean(mesh[:, 0], w), "var_H": wvar(mesh[:, 0], w), "mean_xi0": wmean(mesh[:, 3], w),
            "n_cells": int(n)}


def grid_check(rec, y, s, em, n=GRID_N, extend=EXTEND, box=BOX, tol_mean=TOL_MEAN, tol_c=TOL_C):
    """The grid of line 143 over the sampled support, against the pair's figures under A."""
    lo, hi = extended_support(rec["support_lo"], rec["support_hi"], extend, box)
    axes = [cell_centres(lo[k], hi[k], n) for k in range(4)]
    t0 = time.perf_counter()
    g = grid_posterior(axes, y, s, em, "A")
    g["c_A"] = contraction(g["var_H"])
    g["grid_lo"], g["grid_hi"], g["n_per_dim"] = [float(v) for v in lo], [float(v) for v in hi], int(n)
    g["seconds"] = time.perf_counter() - t0
    return _compare(rec, g, tol_mean, tol_c), axes


def whole_box_check(rec, y, s, em, xi0_axis, n=WHOLE_N, box=BOX, tol_mean=TOL_MEAN, tol_c=TOL_C):
    """The second grid of line 143: n cell-centred points in H, ν and ρ over the whole box,
    ξ₀ on the first grid's points; the same comparison and tolerance."""
    lo, hi = box_arrays(box)
    axes = [cell_centres(lo[k], hi[k], n) for k in range(3)] + [np.asarray(xi0_axis, dtype=float)]
    t0 = time.perf_counter()
    g = grid_posterior(axes, y, s, em, "A")
    g["c_A"] = contraction(g["var_H"])
    g["n_per_dim"], g["seconds"] = [int(n)] * 3 + [int(len(xi0_axis))], time.perf_counter() - t0
    return _compare(rec, g, tol_mean, tol_c)


def _compare(rec, g, tol_mean, tol_c):
    d_mean = abs(rec["under"]["A"]["mean_H"] - g["mean_H"])
    d_c = abs(rec["c_A"] - g["c_A"])
    return dict(g, mcmc_mean_H=rec["under"]["A"]["mean_H"], mcmc_c_A=rec["c_A"], d_mean=float(d_mean),
                d_c=float(d_c), tol_mean=tol_mean, tol_c=tol_c, passed=bool(d_mean <= tol_mean and d_c <= tol_c))


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
        raise SystemExit(f"no emulator at {_rel(path)}: G1 is built after freezing (line 54)")
    _EM["path"], _EM["em"] = path, E.load_emulator(path)
    return _EM["em"]


def set_emulator(em):
    _EM["path"], _EM["em"] = "<object>", em


def _worker(args):
    job, kw = args
    em = get_emulator(kw.pop("emulator_path", None)) if _EM["em"] is None else _EM["em"]
    rec, stores = reference_pair(job["y"], job["s"], em, job["i"], job["j"], **kw)
    rec.update(kind="task", task=job["label"], set=job["set"], theta=job["theta"], y=list(map(float, job["y"])),
               utc=utc_now())
    return rec, {f"{job['label']}_{k}": v for k, v in stores.items()}


def header(set_name, pool, emulator_path, n_points, seeds, extra=None):
    import scipy
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    em_meta = {}
    if os.path.exists(emulator_path):
        with np.load(emulator_path, allow_pickle=False) as z:
            em_meta = {"stage": str(z["stage"]), "commit": str(z["commit"]), "nodes": [int(v) for v in z["values"].shape[:4]]}
    return dict({"kind": "header", "script": os.path.basename(__file__), "commit": commit, "clean_tree": clean,
                 "set": set_name, "n_points": int(n_points), "seeds": seeds, "seed_ref": SEED_REF,
                 "emulator": _rel(emulator_path), "emulator_meta": em_meta,
                 "box": {d: list(BOX[d]) for d in DIMS}, "noise_levels": list(NOISE_LEVELS),
                 "n_chains": N_CHAINS, "burn": BURN, "sample": SAMPLE, "adapt_every": ADAPT_EVERY,
                 "scale": SCALE, "init_var": INIT_VAR, "diag_floor": DIAG_FLOOR,
                 "rhat_max": RHAT_MAX, "ess_min": ESS_MIN, "kish_min": KISH_MIN,
                 "priors": {k: list(v) for k, v in PRIOR_AB.items()}, "var_prior_A": VAR_PRIOR_A,
                 "grid_n": GRID_N, "extend": EXTEND, "whole_n": WHOLE_N, "tol_mean": TOL_MEAN, "tol_c": TOL_C,
                 "quant": QUANT, "python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(),
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
        for key in ("set", "n_points", "seeds", "burn", "sample", "emulator", "emulator_meta", "lengths"):
            if hdr.get(key) != header_obj.get(key):
                raise SystemExit(f"launch refused: {_rel(jsonl_path)} was started with {key}={hdr.get(key)!r}, "
                                 f"this call has {header_obj.get(key)!r}")
    stores = load_samples(samples_path)
    done = {k for k, r in tasks.items() if samples_complete(r, stores)}
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    append_line(jsonl_path, {"kind": "launch", "utc": utc_now(), "pool": int(pool), "n_done_before": len(done),
                             "commit": commit, "clean_tree": clean})
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
            extra_header=None):
    jobs = make_jobs(set_name, points, eps, em, levels)
    hdr = header(set_name, pool, emulator_path, points.shape[0], seeds, extra_header)
    kw = dict(task_kwargs or {})
    hdr["lengths"] = {"burn": kw.get("burn", BURN), "sample": kw.get("sample", SAMPLE), "kish_min": kw.get("kish_min", KISH_MIN),
                      "rhat_max": kw.get("rhat_max", RHAT_MAX), "ess_min": kw.get("ess_min", ESS_MIN),
                      "levels": [float(s) for s in levels]}
    if pool != 1:
        kw["emulator_path"] = emulator_path
    launch(pool, jobs, paths["jsonl"], paths["samples"], hdr, kw)
    return jobs


def _grid_job(args):
    """One pair's grid checks, in a worker (the emulator loaded once per process)."""
    job, rec, kw = args
    em = get_emulator(kw.pop("emulator_path", None)) if _EM["em"] is None else _EM["em"]
    y = np.asarray(job["y"], dtype=float)
    gc, axes = grid_check(rec, y, job["s"], em, kw["grid_n"], kw["extend"], kw["box"], kw["tol_mean"], kw["tol_c"])
    row = {"grid": gc, "passed": gc["passed"]}
    if kw["whole"]:
        wb = whole_box_check(rec, y, job["s"], em, axes[3], kw["whole_n"], kw["box"], kw["tol_mean"], kw["tol_c"])
        row["whole_box"] = wb
        row["passed"] = bool(row["passed"] and wb["passed"])
    return row


def g2_report(jobs, paths, em, pool, levels=NOISE_LEVELS, grid_n=GRID_N, whole_n=WHOLE_N, extend=EXTEND, box=BOX,
              tol_mean=TOL_MEAN, tol_c=TOL_C, emulator_path=None):
    """The grid checks on every accepted pair (a pool over pairs) and the gate's verdict;
    written as JSON. Refuses if a pair is missing from the record or its stored surface
    is not the one the current emulator gives (a rebuilt emulator)."""
    hdr, launches, tasks, stores = load_references(paths["jsonl"], paths["samples"])
    out, all_pass, n_fail, grid_jobs = [], True, 0, []
    j_max = int(np.argmax(levels))
    kw = {"grid_n": grid_n, "whole_n": whole_n, "extend": extend, "box": box, "tol_mean": tol_mean, "tol_c": tol_c}
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
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"gate": "G2", "passed": bool(all_pass), "n_pairs": len(jobs), "n_reference_failures": n_fail,
              "tolerance": {"mean_H": tol_mean, "c": tol_c}, "grid_n": grid_n, "whole_n": whole_n, "extend": extend,
              "cost": {"posterior_seconds": secs, "posterior_mean_s": float(np.mean(secs)) if secs else None,
                       "posterior_median_s": float(np.median(secs)) if secs else None,
                       "grid_mean_s": float(np.mean(gsecs)) if gsecs else None,
                       "whole_box_mean_s": float(np.mean(wsecs)) if wsecs else None, "pool": int(pool),
                       "note": "wall seconds inside a worker with `pool` workers running; read with pool"},
              "pairs": out, "header": hdr, "launches": launches, "report_commit": commit, "report_clean_tree": clean,
              "utc_report": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P5 gate G2 reference (see module docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="G2: the 12 pilot references, the grid checks and the report")
    mode.add_argument("--g3-pilot", action="store_true", help="the 48 references of G3's pilot surfaces")
    mode.add_argument("--test", action="store_true", help="the sealed test set; needs --n and --noise-seed")
    ap.add_argument("--n", type=int, default=None, help="number of test θ (line 94's amendment)")
    ap.add_argument("--noise-seed", type=int, default=None, help="the test set's noise seed (line 93, to be declared)")
    ap.add_argument("--emulator", default=EMULATOR_PATH)
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    a = ap.parse_args(argv)

    print("=== P5 G2 reference ===", flush=True)
    print(f"  chains={N_CHAINS} burn={BURN} sample={SAMPLE} adapt_every={ADAPT_EVERY} scale={SCALE:.4f} "
          f"init_var={INIT_VAR:.4f} floor={DIAG_FLOOR}; R-hat<={RHAT_MAX} ESS_H>={ESS_MIN:.0f} Kish>={KISH_MIN:.0f}", flush=True)
    print(f"  priors A={PRIOR_AB['A']} B=({PRIOR_AB['B'][0]:.4f}, {PRIOR_AB['B'][1]:.4f}); Var_prior(A)={VAR_PRIOR_A:.6f}", flush=True)
    print(f"  G2: {N_G2_POINTS} points rng({SEED_G2_POINTS}) x {len(NOISE_LEVELS)} levels, noise rng({SEED_G2_NOISE}); "
          f"grids {GRID_N}^4 on the support +{EXTEND:.0%}, {WHOLE_N}^3 x {GRID_N}; tol mean {TOL_MEAN}, c {TOL_C}", flush=True)
    if not (a.pilot or a.g3_pilot or a.test):
        print("\nHOLD: nothing run. --pilot needs the emulator built at G1 after freezing (line 54).", flush=True)
        return
    if a.test and (a.n is None or a.noise_seed is None):
        raise SystemExit("--test needs --n and --noise-seed, the values of line 94's amendment")
    em = get_emulator(a.emulator)
    if a.pilot:
        pts, eps = pilot_points(SEED_G2_POINTS, N_G2_POINTS), noise_draws(SEED_G2_NOISE, N_G2_POINTS)
        jobs = run_set("g2", pts, eps, em, a.pool, SETS["g2"], a.emulator,
                       {"points": SEED_G2_POINTS, "noise": SEED_G2_NOISE})
        rep = g2_report(jobs, SETS["g2"], em, a.pool, emulator_path=a.emulator)
        print(f"=== G2: {'passed' if rep['passed'] else 'NOT passed'}; reference failures {rep['n_reference_failures']}; "
              f"posterior median {rep['cost']['posterior_median_s']}s, grid mean {rep['cost']['grid_mean_s']}s ===", flush=True)
        for r in rep["pairs"]:
            g = r.get("grid", {})
            print(f"  {r['task']}: {r['status']} d_mean={g.get('d_mean')} d_c={g.get('d_c')} passed={r.get('passed')}"
                  + (f" whole_box d_mean={r['whole_box']['d_mean']} d_c={r['whole_box']['d_c']}" if "whole_box" in r else ""),
                  flush=True)
        print(f"  written: {_rel(SETS['g2']['report'])}", flush=True)
        return
    if a.g3_pilot:
        pts, eps = pilot_points(SEED_G3_POINTS, N_G3_POINTS), noise_draws(SEED_G3_NOISE, N_G3_POINTS)
        run_set("g3", pts, eps, em, a.pool, SETS["g3"], a.emulator, {"points": SEED_G3_POINTS, "noise": SEED_G3_NOISE})
        return
    if a.test:
        pts, eps = test_points(a.n), noise_draws(a.noise_seed, a.n)
        run_set("test", pts, eps, em, a.pool, SETS["test"], a.emulator, {"points": SEED_TEST, "noise": int(a.noise_seed)},
                extra_header={"n_test": int(a.n)})


if __name__ == "__main__":
    main()
