"""
test_p5_v3_study.py — mechanics tests for P5 v3's development study script (D93). They
guard the MACHINERY of candidate (A) and of the study on toy emulators only: the linear toy
of test_p5_g2_reference.py in memory, and the toy emulator file of D92's test for the pool
paths. The settings as literals; the candidate recovering a known Gaussian posterior; the
order of its draws from the generator; mode retention and the mass rule; Σ and its clipping;
the direct-run polish and ε; the candidate installed inside spawned and forked workers; the
diagnostics recorded; the stand-in targets; the mass diagnostic; the verdict; the hold mode
running nothing and the refusals; and a small end-to-end run of --run on toy records.
Nothing here touches G̃, and the sealed test set's drawing functions are replaced by ones
that fail if called wherever the entry point runs (D82).
"""
import json
import functools
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

import p5_g2_reference as R
import p5_g1_emulator as E
import p5_v3_study as S
from test_p5_g2_reference import LinearEm, toy_surface, _toy_emulator_file, _no_sealed_draws, LO, HI, CENTRE

KW = {"burn": 300, "sample": 900, "kish_min": 50.0, "rhat_max": 1.5, "ess_min": 20.0}     # short toy lengths, relaxed acceptance
KW_E2E = {"burn": 200, "sample": 600, "kish_min": 20.0, "rhat_max": 2.0, "ess_min": 10.0}  # shorter still, for the end-to-end run


@pytest.fixture(autouse=True)
def _v2_sampler_and_cache_restored():
    """Every test leaves p5_g2_reference as it found it: v2's run_chains, its worker, no emulator cached."""
    yield
    S.uninstall_candidate()
    R._worker = S._R_WORKER
    R._EM["path"], R._EM["em"] = None, None


def _untimed2(d):
    """A record without its times (the optimiser's seconds included), which differ from run to run."""
    if isinstance(d, dict):
        return {k: _untimed2(v) for k, v in d.items() if k not in ("utc", "seconds", "optimiser_seconds")}
    if isinstance(d, list):
        return [_untimed2(v) for v in d]
    return d


def test_the_settings_are_the_declared_ones():
    assert S.N_STARTS == 16 and (S.LS_METHOD, S.LS_JAC) == ("trf", "2-point")
    assert (S.LS_FTOL, S.LS_XTOL, S.LS_GTOL, S.LS_MAX_NFEV) == (1e-10, 1e-10, 1e-8, 2000)
    assert (S.LBFGS_FTOL, S.LBFGS_GTOL, S.LBFGS_MAXITER) == (1e-10, 1e-5, 200) and S.EPS_H == 1e-6
    assert (S.KEEP_MAX, S.DISTINCT) == (4, 0.01) and abs(S.MASS_LOG - np.log(1e4)) < 1e-15
    assert (S.EIG_LO, S.EIG_HI) == (1e-10, 1 / 12) and (S.PERTURB_CAP, S.PERTURB_MULT) == (0.0025, 4.0)
    assert S.SCALE == R.SCALE and abs(S.SCALE - 2.38 ** 2 / 4) < 1e-15
    assert (S.BURN, S.SAMPLE, S.N_CHAINS, S.ADAPT_EVERY) == (10000, 40000, 4, 500) == (R.BURN, R.SAMPLE, R.N_CHAINS, R.ADAPT_EVERY)
    assert S.TASK_KWARGS == {"burn": 10000, "sample": 40000} and (S.GRID_N, S.WHOLE_N) == (32, 48) == (R.GRID_N, R.WHOLE_N)
    assert abs(S.FD_REL_STEP - np.sqrt(np.finfo(float).eps)) < 1e-30 and abs(S.FD_REL_STEP - 1.4901161193847656e-08) < 1e-23
    assert (S.SEED_STANDIN, S.STANDIN_Q, S.DISPLACE_H) == (107, 0, 0.02) and S.PRIOR_INDEX == {"A": 1, "B": 2}
    assert S.STANDIN_VARIANTS == ("own", "displaced")
    import p5_g3_calibrators as C                                                       # the ratio estimator's own constants
    assert S.SEED_STANDIN == C.SEED_NRE and S.PRIOR_INDEX == C.PRIOR_INDEX
    assert S.SEALED_SEEDS == (20261002, 20261022) and (S.NONADAPT_ACC, S.NONADAPT_RHAT, S.GRID_MISS_N) == (0.01, 1.1, 64)
    assert S.N_PAIRS_EXPECTED == 36 and S.POOL_DEFAULT == 4 and S.SET_NAME == "p5v3_study"
    for p in S.PATHS.values():                                                          # new names, never v2's or v3's G2
        assert Path(p).name.startswith("p5v3_study") and "p5v3_g2" not in p
    assert not set(S.PATHS.values()) & {v for d in R.SETS.values() for v in d.values()}
    assert S.SOURCES == {"g2": R.SETS["g2"]["jsonl"], "g2r": R.SETS["g2r"]["jsonl"]}
    cc = S.candidate_constants()
    assert cc["n_starts"] == 16 and cc["ls_ftol"] == 1e-10 and cc["eps_H"] == 1e-6 and cc["perturb_cap"] == 0.0025
    assert cc["mass_log"] == S.MASS_LOG and cc["scale"] == S.SCALE and "uniform starts" in cc["draw_order"]
    sc = S.standin_constants()
    assert sc["seed"] == 107 and sc["q"] == 0 and "finite difference" in sc["gradient"] and "autograd" in sc["gradient"]
    S.refuse_sealed([83, 79, {"a": [131, (137,)]}, 89], "x")                             # nothing sealed: passes
    with pytest.raises(SystemExit, match="sealed seed.*20261002"):
        S.refuse_sealed([83, 20261002], "x")
    with pytest.raises(SystemExit, match="20261022"):
        S.refuse_sealed({"noise": 20261022}, "x")
    assert S.standin_key("A", 3, 1, "own") == [107, 1, 0, 3, 1] and S.standin_key("B", 0, 2, "displaced") == [107, 2, 0, 0, 2, 1]


def test_the_candidate_recovers_the_gaussian_posterior_and_records_its_diagnostics():
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    mean, cov = em.posterior(y, s)
    sd = np.sqrt(np.diag(cov))
    run = S.candidate_run_chains(y, s, em, np.random.default_rng([R.SEED_REF, 0, 0]), "uniform", burn=2000, sample=4000)
    d = run["candidate"]
    assert set(run) >= {"samples", "acceptance", "cov", "burn", "sample", "prior", "candidate"}        # run_chains' return
    assert run["samples"].shape == (4, 4000, 4) and np.all(run["samples"] >= 0) and np.all(run["samples"] <= 1)
    assert d["route"] == "least_squares" and d["prior"] == "uniform" and d["m"] == 1 and d["starts_elsewhere"] == 0
    assert d["kept"][0]["n_starts_reached"] == 16 and len(d["optimiser"]["status"]) == 16                # one mode, all starts
    assert all(st in (1, 2, 3, 4) for st in d["optimiser"]["status"])                                  # least_squares converged
    assert all(n <= S.LS_MAX_NFEV for n in d["optimiser"]["nfev"]) and d["optimiser"]["reached"] == [0] * 16
    cov_unit = cov / np.outer(HI - LO, HI - LO)                                                         # the Gaussian in unit coordinates
    assert np.allclose(d["kept"][0]["sigma"], cov_unit, rtol=1e-4, atol=0) and d["kept"][0]["n_clipped"] == 0
    assert np.allclose(np.sort(d["kept"][0]["eig"]), np.sort(np.linalg.eigvalsh(cov_unit)), rtol=1e-4)
    assert np.allclose(R.unit_to_theta(d["kept"][0]["x"]), mean, atol=1e-6)
    assert d["kept"][0]["polish"] is None and d["kept"][0]["sigma_A"][0, 0] < d["kept"][0]["sigma"][0, 0]   # the prior's curvature
    assert d["kept"][0]["logf"] == d["kept"][0]["logf_ls"] <= 0.0 and np.isfinite(d["kept"][0]["mass_ls"])   # a uniform run: no polish
    assert run["cov"].shape == (4, 4, 4) and d["chain_solution"] == [0, 0, 0, 0] and d["n_merged_after_polish"] == 0
    assert d["n_start_clipped"] == 0 and d["optimiser_seconds"] > 0 and d["n_starts"] == 16
    acc = d["acceptance_first_500"]
    assert len(acc) == 4 and all(0 < a_ <= 1 for a_ in acc)                                              # a preconditioned proposal moves
    assert all(r is not None and len(r["gen_eig"]) == 4 and r["trace_ratio"] > 0 for r in d["proposal_vs_sample"])
    assert 0.5 < np.mean([r["trace_ratio"] for r in d["proposal_vs_sample"]]) / S.SCALE < 2.0            # about 2.38²/4, loosely
    ok, diag = R.accept(run["samples"], rhat_max=1.1, ess_min=100)
    assert ok
    pooled = R.unit_to_theta(run["samples"].reshape(-1, 4))
    for p, dim in enumerate(E.DIMS):
        assert abs(pooled[:, p].mean() - mean[p]) < 4 * sd[p] / np.sqrt(diag["ess"][dim]) + 1e-9
    assert 0.1 < run["acceptance"] < 0.6
    run2 = S.candidate_run_chains(y, s, em, np.random.default_rng([R.SEED_REF, 0, 0]), "uniform", burn=2000, sample=4000)
    assert np.array_equal(run["samples"], run2["samples"])                                               # the same generator, the same chains
    assert json.dumps(S._jsonable(d), allow_nan=False)                                                  # the record takes it
    assert S._jsonable({"a": np.array([1.0, np.inf]), "b": np.float64(2.0), "c": np.int64(3), "d": np.bool_(True)}) == \
        {"a": [1.0, None], "b": 2.0, "c": 3, "d": True}


def test_the_generator_is_drawn_in_the_declared_order():
    """Block 1: 16 × 4 uniforms for the starts; block 2: 4 × 4 standard normals for the
    perturbations; then mh_step's draws, as v2's loop makes them from the candidate's starts
    and first proposals."""
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    key = [R.SEED_REF, 1, 2]
    run = S.candidate_run_chains(y, s, em, np.random.default_rng(key), "uniform", burn=600, sample=800)
    d = run["candidate"]
    rng = np.random.default_rng(key)
    U = rng.uniform(size=(16, 4))
    Z = rng.standard_normal((4, 4))
    assert np.array_equal(np.asarray(d["starts"]), U)
    for k in range(4):
        sol = d["kept"][d["chain_solution"][k]]
        L = R._chol_pd(S.PERTURB_MULT * S.capped_cov(sol["sigma"]))
        z = np.linalg.solve(L, np.asarray(d["chain_start_raw"][k]) - sol["x"])
        assert np.allclose(z, Z[k], atol=1e-12)
    assert np.array_equal(np.asarray(d["chain_start"]), np.clip(np.asarray(d["chain_start_raw"]), 0.0, 1.0))
    x = np.asarray(d["chain_start"]).copy()
    lt = R.log_target(x, y, s, em, "uniform")
    cov = np.stack([S.SCALE * d["kept"][d["chain_solution"][k]]["sigma"] for k in range(4)])
    L = np.stack([R._chol_pd(c) for c in cov])
    hist = np.empty((4, 601, 4))
    hist[:, 0] = x
    for t in range(1, 601):
        x, lt, _ = R.mh_step(x, lt, L, y, s, em, "uniform", rng)
        hist[:, t] = x
        if t % 500 == 0:
            cov = np.stack([R.adapted_cov(hist[k, :t + 1]) for k in range(4)])
            L = np.stack([R._chol_pd(c) for c in cov])
    samples = np.empty((4, 800, 4))
    for t in range(800):
        x, lt, _ = R.mh_step(x, lt, L, y, s, em, "uniform", rng)
        samples[:, t] = x
    assert np.array_equal(samples, run["samples"]) and np.array_equal(cov, run["cov"])
    # the candidate's chains differ from v2's on the same generator: the starts and the first proposal changed
    v2 = S._V2_RUN_CHAINS(y, s, em, np.random.default_rng(key), "uniform", burn=600, sample=800)
    assert not np.array_equal(v2["samples"], run["samples"]) and "candidate" not in v2


class _TwoGaussians:
    """A synthetic target for the gradient route: log(N(m1, sd² I) + w2 N(m2, sd² I)) in unit
    coordinates, with its exact gradient and Σ = sd² I."""

    def __init__(self, m1, m2, sd, w2=1.0):
        self.m1, self.m2, self.sd, self.w2 = np.asarray(m1, float), np.asarray(m2, float), float(sd), float(w2)
        self.sigma = sd ** 2 * np.eye(4)

    def _parts(self, x):
        x = np.asarray(x, float).reshape(-1, 4)
        a = -np.sum((x - self.m1) ** 2, axis=1) / (2 * self.sd ** 2)
        b = np.log(self.w2) - np.sum((x - self.m2) ** 2, axis=1) / (2 * self.sd ** 2)
        return x, a, b

    def __call__(self, th):
        _, a, b = self._parts(R.theta_to_unit(th))
        return np.logaddexp(a, b)

    def grad_unit(self, x):
        x, a, b = self._parts(x)
        pa = np.exp(a - np.logaddexp(a, b))[:, None]
        return (pa * (-(x - self.m1) / self.sd ** 2) + (1 - pa) * (-(x - self.m2) / self.sd ** 2))[0]


def test_mode_retention_and_the_mass_rule():
    I = 1e-3 * np.eye(4)
    xs = np.array([[0.2] * 4, [0.205] * 4, [0.7] * 4, [0.5] * 4, [0.9] * 4, [0.1] * 4, [0.4] * 4])
    logf = np.array([0.0, -0.5, -1.0, -np.log(1e4) - 1e-6, -2.0, -3.0, -4.0])
    kept, mass, reached = S.keep_solutions(xs, logf, [I] * 7)
    # in order of log f: 0 kept; 1 within 0.01 of 0 (distance 0.01, not above it); 2 kept; 3 below the mass rule
    # (the masses share one log det, so the rule is on log f); 4 and 5 kept, four in all; 6 beyond the four
    assert kept == [0, 2, 4, 5] and reached.tolist() == [0, 0, 1, -1, 2, 3, -1]
    assert np.allclose(mass, logf + 0.5 * np.linalg.slogdet(I)[1])
    kept3, _, _ = S.keep_solutions(xs, logf, [I] * 7, keep_max=3)
    assert kept3 == [0, 2, 4]
    xs2 = np.array([[0.2] * 4, [0.2 + 0.0051] * 4])                                   # distance 0.0102: distinct
    assert S.keep_solutions(xs2, np.array([0.0, -0.1]), [I] * 2)[0] == [0, 1]
    # the mass rule is on the largest Laplace mass: a sharp spike with the highest log f but under 1e-4 of the mass is dropped
    spike, wide = 1e-10 * np.eye(4), (1 / 12) * np.eye(4)
    kept_s, mass_s, _ = S.keep_solutions(np.array([[0.3] * 4, [0.7] * 4]), np.array([0.0, -30.0]), [spike, wide])
    assert mass_s[1] > mass_s[0] and kept_s == [1]
    kept_t, _, _ = S.keep_solutions(np.array([[0.3] * 4, [0.7] * 4]), np.array([0.0, -30.0]), [spike, spike])
    assert kept_t == [0]                                                                 # equal Σ: the rule is on log f again
    assert S.keep_solutions(np.array([[0.3] * 4]), np.array([-np.inf]), [I])[0] == []
    # a bimodal target through the gradient route: both modes found and kept, the chains at both, R-hat fails (the plan:
    # such a pair fails, as G2 requires); with the second mode under 1e-4 of the mass, m = 1
    tg = _TwoGaussians([0.3] * 4, [0.7] * 4, 0.02)
    run = S.candidate_run_chains(None, None, None, np.random.default_rng(5), "uniform", burn=500, sample=1000, target=tg)
    d = run["candidate"]
    assert d["route"] == "gradient" and d["m"] == 2 and sum(k["n_starts_reached"] for k in d["kept"]) == 16 and d["starts_elsewhere"] == 0
    assert np.allclose(sorted(k["x"][0] for k in d["kept"]), [0.3, 0.7], atol=1e-4) and d["chain_solution"] == [0, 1, 0, 1]
    assert all(np.allclose(k["sigma"], tg.sigma) and k["n_clipped"] == 0 and k["sigma_A"] is None for k in d["kept"])
    assert all(st in (0, 1, 2) for st in d["optimiser"]["status"]) and len(d["optimiser"]["nit"]) == 16
    ok, diag = R.accept(run["samples"], 1.01, 100)
    assert not ok and diag["rhat"]["H"] > 1.5                                           # the chains sit on different modes
    assert d["bounds_unit"][0][0] == S.EPS_H and d["bounds_unit"][1][0] == 1 - S.EPS_H and d["bounds_unit"][0][1] == 0.0
    run1 = S.candidate_run_chains(None, None, None, np.random.default_rng(5), "uniform", burn=500, sample=1000,
                                  target=_TwoGaussians([0.3] * 4, [0.7] * 4, 0.02, w2=1e-6))
    d1 = run1["candidate"]
    assert d1["m"] == 1 and abs(d1["kept"][0]["x"][0] - 0.3) < 1e-4 and d1["starts_elsewhere"] > 0
    assert R.accept(run1["samples"], 1.1, 50)[0]
    with pytest.raises(TypeError, match="grad_unit"):                                    # a plain callable is not run with the candidate
        S.candidate_run_chains(None, None, None, np.random.default_rng(5), "uniform", burn=10, sample=10,
                               target=lambda th: np.zeros(np.asarray(th).reshape(-1, 4).shape[0]))


def test_sigma_from_the_jacobian_and_its_clipping():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((35, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    sv = np.array([1e6, 100.0, 4.0, 0.1])
    J = (Q * sv) @ V.T
    cov, eig, n = S.cov_from_jacobian(J)
    assert np.allclose(np.sort(eig), np.sort(1 / sv ** 2), rtol=1e-8) and n == 2        # 1e-12 below the floor, 100 above the cap; 1/16 inside
    assert np.allclose(np.sort(np.linalg.eigvalsh(cov)), np.sort(np.clip(1 / sv ** 2, 1e-10, 1 / 12)), rtol=1e-6)
    assert np.allclose(cov, cov.T)
    sv2 = np.array([50.0, 20.0, 8.0, 4.0])
    J2 = (Q * sv2) @ V.T
    cov2, eig2, n2 = S.cov_from_jacobian(J2)
    assert n2 == 0 and np.allclose(cov2, np.linalg.inv(J2.T @ J2), rtol=1e-8)
    kappa = 30.0
    covP, eigP, nP = S.cov_from_jacobian(J2, kappa)
    assert nP == 0 and np.allclose(covP, np.linalg.inv(J2.T @ J2 + np.diag([kappa, 0, 0, 0])), rtol=1e-8)
    assert covP[0, 0] < cov2[0, 0]
    J3 = J2.copy()
    J3[:, 3] = 0.0                                                                       # a flat direction: its eigenvalue is infinite, clipped to 1/12
    cov3, eig3, n3 = S.cov_from_jacobian(J3)
    assert n3 == 1 and np.isinf(np.max(eig3)) and abs(np.max(np.linalg.eigvalsh(cov3)) - 1 / 12) < 1e-12
    with pytest.raises(ValueError, match="at least as many residuals"):
        S.cov_from_jacobian(np.ones((3, 4)))
    # Σ_s: the eigenvalues capped at 0.0025, the eigenvectors kept
    cs = S.capped_cov(np.diag([0.01, 0.001, 0.002, 0.1]))
    assert np.allclose(np.diag(cs), [0.0025, 0.001, 0.002, 0.0025]) and np.allclose(cs - np.diag(np.diag(cs)), 0)
    w, Vc = np.linalg.eigh(covP)
    assert np.allclose(S.capped_cov(covP), (Vc * np.minimum(w, 0.0025)) @ Vc.T)
    # a given Σ (the record's Σ_P) clipped as the candidate clips its own
    cg, wg, ng = S.clip_cov(np.diag([1e-12, 0.5, 0.001, 0.01]))
    assert ng == 2 and np.allclose(np.diag(cg), [1e-10, 1 / 12, 0.001, 0.01])
    # the bounded difference: the step is √ε·max(1, |x|), backward at the upper face, the actual difference the divisor
    f = lambda x: np.array([x[0] ** 2, 3.0 * x[1], x[2] * x[3]])                      # noqa: E731
    x0 = np.array([0.3, 1.0, 0.5, 0.25])
    Jf, f0 = S.fd_jacobian(f, x0, np.zeros(4), np.ones(4))
    assert np.allclose(f0, f(x0)) and np.allclose(Jf, [[0.6, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0.25, 0.5]], atol=1e-6)
    seen = []
    S.fd_jacobian(lambda x: seen.append(x.copy()) or np.zeros(1), x0, np.zeros(4), np.ones(4))
    steps = np.array([v - x0 for v in seen[1:]])
    assert np.allclose(np.diag(steps), [S.FD_REL_STEP, -S.FD_REL_STEP, S.FD_REL_STEP, S.FD_REL_STEP], rtol=1e-6)
    assert np.allclose(steps - np.diag(np.diag(steps)), 0)


def test_the_prior_derivatives_the_direct_run_polish_and_epsilon():
    for which in ("A", "B"):
        for u in (0.05, 0.3, 0.8):
            g, k = S.prior_grad_curv(u, which)
            h = 1e-5
            lp = lambda uu: float(R.prior_logpdf(R.H_LO + uu * R.H_W, which))          # noqa: E731
            assert abs(g - (lp(u + h) - lp(u - h)) / (2 * h)) < 1e-5 * max(1.0, abs(g))
            assert abs(k + (lp(u + h) - 2 * lp(u) + lp(u - h)) / h ** 2) < 1e-3 * abs(k) and k > 0
    assert S.prior_grad_curv(0.3, "uniform") == (0.0, 0.0)
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    mean, cov = em.posterior(y, s)
    run = S.candidate_run_chains(y, s, em, np.random.default_rng([R.SEED_REF, 0, 1]), "A", burn=300, sample=300)
    d = run["candidate"]
    k0 = d["kept"][0]
    assert d["route"] == "least_squares" and d["prior"] == "A" and d["m"] == 1 and k0["polish"] is not None
    assert k0["polish"]["nit"] >= 1 and k0["polish"]["status"] in (0, 2)                 # converged, or the line search at the noise floor
    assert d["bounds_unit"][0][0] == S.EPS_H and d["bounds_unit"][1][0] == 1 - S.EPS_H and list(d["bounds_unit"][0][1:]) == [0, 0, 0]
    fun = lambda xx: S.residuals(xx, y, s, em)                                          # noqa: E731
    J, r = S.fd_jacobian(fun, k0["x"], d["bounds_unit"][0], d["bounds_unit"][1])
    g = J.T @ r
    g[0] -= S.prior_grad_curv(k0["x"][0], "A")[0]
    assert np.max(np.abs(g)) < 1e-3 and S.EPS_H <= k0["x"][0] <= 1 - S.EPS_H           # a stationary point of −log L − log P
    assert k0["logf"] > k0["logf_ls"] - 30 and np.isfinite(k0["mass_ls"]) and d["n_merged_after_polish"] == 0
    assert abs(k0["logf"] - (-0.5 * float(r @ r) + float(R.prior_logpdf(R.unit_to_theta(k0["x"])[0], "A")))) < 1e-9   # the log-target at the polished point
    # it is the mode of the Gaussian × Beta: the exact quadratic's gradient balances the prior's there
    th = R.unit_to_theta(k0["x"])
    g_exact = -(HI - LO) * (np.linalg.solve(cov, th - mean))                          # ∇_x log L of the exact Gaussian
    g_exact[0] += S.prior_grad_curv(k0["x"][0], "A")[0]
    assert np.max(np.abs(g_exact)) < 1e-3
    # Σ_P adds the prior's curvature in x_H: narrower in H than the likelihood's Σ, equal elsewhere up to that coupling
    sigma_L = S.cov_from_jacobian(J)[0]
    assert k0["sigma"][0, 0] < sigma_L[0, 0] and k0["n_clipped"] == 0
    assert np.allclose(k0["sigma"], np.linalg.inv(np.linalg.inv(sigma_L) + np.diag([S.prior_grad_curv(k0["x"][0], "A")[1], 0, 0, 0])), rtol=1e-6)
    assert np.allclose(k0["sigma"], k0["sigma_A"]) and not np.allclose(k0["sigma"], k0["sigma_B"])
    # a surface whose likelihood mode lies beyond the lower face of H: under the uniform prior the solution sits on the
    # face and starts clip to 0; under A the polish moves it inside and the starts clip to ε
    th_out = CENTRE.copy()
    th_out[0] = LO[0] - 0.05
    y2 = em.evaluate_many(th_out)[0] + s * np.random.default_rng(3).standard_normal(35)
    run_u = S.candidate_run_chains(y2, s, em, np.random.default_rng([R.SEED_REF, 0, 2]), "uniform", burn=300, sample=300)
    du = run_u["candidate"]
    assert all(k["x"][0] < 1e-6 for k in du["kept"]) and np.min(np.asarray(du["chain_start"])[:, 0]) == 0.0   # trf ends a hair inside
    assert du["n_start_clipped"] >= 1 and np.all(np.asarray(du["chain_start"]) >= 0.0)
    run_a = S.candidate_run_chains(y2, s, em, np.random.default_rng([R.SEED_REF, 0, 2]), "A", burn=300, sample=300)
    da = run_a["candidate"]
    assert all(S.EPS_H <= k["x"][0] < 0.05 for k in da["kept"]) and np.all(np.asarray(da["chain_start"])[:, 0] >= S.EPS_H)
    assert da["n_start_clipped"] >= 1 and np.min(np.asarray(da["chain_start"])[:, 0]) == S.EPS_H
    assert np.all(np.isfinite(R.log_target(np.asarray(da["chain_start"]), y2, s, em, "A")))   # every chain starts at finite density
    # chain_starts on its own: k mod m, the cap, the clip count
    sols = [np.array([0.01, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.99])]
    covs = [np.diag([0.01, 0.01, 0.01, 0.01]), np.diag([1e-4, 1e-4, 1e-4, 1e-4])]
    z = np.array([[-3.0, 0, 0, 0], [0, 0, 0, 0.3], [0, 0, 0, 0], [0, 0, 0, 0]])
    lo_b, hi_b = np.array([S.EPS_H, 0, 0, 0]), np.array([1 - S.EPS_H, 1, 1, 1])
    x, raw, n_clip, which = S.chain_starts(sols, covs, z, lo_b, hi_b)
    assert which == [0, 1, 0, 1] and n_clip == 1
    assert abs(raw[0, 0] - (0.01 - 2 * 0.05 * 3)) < 1e-12 and x[0, 0] == S.EPS_H          # capped at 0.0025, × 4, z = −3
    assert abs(raw[1, 3] - (0.99 + 2 * 0.01 * 0.3)) < 1e-12 and x[1, 3] == raw[1, 3] < 1.0   # 1e-4 under the cap: sd 0.01, × 2
    assert np.array_equal(x[2], sols[0]) and np.array_equal(x[3], sols[1])


def _toy_jobs(em, levels=(0.003, 0.005)):
    pts, eps = np.array([CENTRE, CENTRE + 0.1 * (HI - LO)]), R.noise_draws(89, 2, 2)
    jobs = R.make_jobs("g2", pts, eps, em, levels)
    for j in jobs:
        j["v2_status"] = "ok"
    return jobs, pts, eps


def _paths(tmp_path, name):
    return {"jsonl": str(tmp_path / f"{name}.jsonl"), "samples": str(tmp_path / f"{name}.npz"),
            "checks": str(tmp_path / f"{name}_checks.json"), "report": str(tmp_path / f"{name}_report.json")}


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_the_candidate_is_installed_inside_the_workers(tmp_path, monkeypatch, method):
    """At pool 2 each worker installs the candidate in its own process (the parent never does):
    every record carries the candidate's diagnostics, no pair errs, and the records equal a
    pool-1 run's, the generator being per pair; a resume with other constants is refused."""
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"the {method} start method is not available on this platform")
    monkeypatch.setattr(R, "ProcessPoolExecutor",
                        functools.partial(R.ProcessPoolExecutor, mp_context=multiprocessing.get_context(method)))
    monkeypatch.setattr(R, "EMULATOR_PATH", str(tmp_path / "no_g_tilde.npz"))          # no G̃ here or in fork's workers
    _no_sealed_draws(monkeypatch)
    em_file = _toy_emulator_file(tmp_path / "toy_em.npz", LinearEm())
    R._EM["path"], R._EM["em"] = None, None
    em = R.get_emulator(em_file)                                                        # from the file, as main loads G̃
    jobs, pts, eps = _toy_jobs(em)
    hdr2 = S.study_header(2, em_file, None, None, KW)
    assert hdr2["script"] == "p5_v3_study.py" and hdr2["reference_script"] == "p5_g2_reference.py" and hdr2["set"] == "p5v3_study"
    assert hdr2["lengths"]["candidate"] == S.candidate_constants() and hdr2["candidate"] == S.candidate_constants()
    assert (hdr2["lengths"]["burn"], hdr2["lengths"]["sample"], hdr2["lengths"]["kish_min"]) == (300, 900, 50.0)
    assert hdr2["emulator_sha256"] == E.sha256_file(em_file) and hdr2["burn"] == 10000 and hdr2["seed_ref"] == 83
    assert hdr2["standin"]["seed"] == 107 and "autograd" in hdr2["standin"]["gradient"] and hdr2["verdict_rule"]["grid_miss_n"] == 64
    S.uninstall_candidate()
    assert R.run_chains is S._V2_RUN_CHAINS
    n = S.run_references(jobs, _paths(tmp_path, "two"), hdr2, 2, em_file, KW)
    assert n == 4 and R.run_chains is S._V2_RUN_CHAINS and R._worker is S._R_WORKER    # the workers installed it, the parent did not
    text = Path(_paths(tmp_path, "two")["jsonl"]).read_text(encoding="utf-8")
    assert '"task_error"' not in text
    hdr, launches, tasks, mass, standins, errors = S.read_study_record(_paths(tmp_path, "two")["jsonl"])
    assert sorted(tasks) == [j["label"] for j in jobs] and [l["pool"] for l in launches] == [2] and errors == [] and mass == {}
    assert hdr["lengths"]["candidate"]["n_starts"] == 16 and hdr["emulator_sha256"] == E.sha256_file(em_file)
    # the later phases at pool 2 as well: the checks, the mass lines and the stand-ins, every worker loading the toy file
    monkeypatch.setattr(S, "GRID_N", 4)
    monkeypatch.setattr(S, "WHOLE_N", 4)
    checks2 = S.run_checks(jobs, _paths(tmp_path, "two"), em, 2, em_file, None, KW)
    assert checks2["n_pairs"] == 4 and checks2["cost"]["pool"] == 2 and all(r["status"] == "ok" for r in checks2["pairs"])
    mass2, n_err = S.run_mass(jobs, _paths(tmp_path, "two"), checks2, 2, em_file)
    assert n_err == 0 and sorted(mass2) == [j["label"] for j in jobs] and all(0 <= m_["share_outside"] <= 1 for m_ in mass2.values())
    st2, n_err2 = S.run_standins(jobs, _paths(tmp_path, "two"), 2, em_file, KW)
    assert n_err2 == 0 and len(st2) == 16 and all(v["accepted"] for v in st2.values())
    assert all(v["compare"]["passed"] for v in st2.values() if v["variant"] == "own")
    assert R.run_chains is S._V2_RUN_CHAINS and "_error" not in Path(_paths(tmp_path, "two")["jsonl"]).read_text(encoding="utf-8")
    for t in tasks.values():
        assert t["status"] == "ok" and t["sampler"] == S.SAMPLER_TAG and t["v2_status"] == "ok" and t["set"] == "g2"
        attempts = [(k, a_) for k in ("uniform", "A", "B") for a_ in t["runs"].get(k, [])]
        assert len(t["candidate"]) == len(attempts) >= 1                               # one diagnostics entry per sampler run, in order
        for c, (prior, a) in zip(t["candidate"], attempts):
            assert (c["prior"], c["burn"], c["sample"]) == (prior, a["burn"], a["sample"])
            assert c["m"] >= 1 and len(c["acceptance_first_500"]) == 4 and c["optimiser_seconds"] > 0 and c["route"] == "least_squares"
            assert len(c["kept"]) == c["m"] and all(len(k["eig"]) == 4 and len(k["sigma"]) == 4 for k in c["kept"])
    hdr1 = S.study_header(1, em_file, None, None, KW)
    assert S.run_references(jobs, _paths(tmp_path, "one"), hdr1, 1, em_file, KW) == 4
    assert R.run_chains is S.candidate_run_chains                                       # at pool 1 the worker ran in the parent
    _, _, ref, *_ = S.read_study_record(_paths(tmp_path, "one")["jsonl"])
    assert {k: _untimed2(v) for k, v in tasks.items()} == {k: _untimed2(v) for k, v in ref.items()}
    st2, st1 = R.load_samples(_paths(tmp_path, "two")["samples"]), R.load_samples(_paths(tmp_path, "one")["samples"])
    assert sorted(st2) == sorted(st1) and all(np.array_equal(st2[k], st1[k]) for k in st1)
    # the same jobs under v2's sampler give other chains, and records without the diagnostics
    S.uninstall_candidate()
    R.run_set("g2", pts, eps, em, 1, {**_paths(tmp_path, "v2"), "report": str(tmp_path / "v2.json")}, em_file,
              {"points": 79, "noise": 89}, (0.003, 0.005), KW)
    _, _, v2t, v2s = R.load_references(_paths(tmp_path, "v2")["jsonl"], _paths(tmp_path, "v2")["samples"])
    assert "candidate" not in v2t["g2_0000_0"] and not np.array_equal(v2s["g2_0000_0_u"], st1["g2_0000_0_u"])
    # a resume with other constants is refused, as one with other lengths is
    monkeypatch.setattr(S, "DISTINCT", 0.02)
    hdr3 = S.study_header(1, em_file, None, None, KW)
    assert hdr3["lengths"] != hdr1["lengths"]
    with pytest.raises(SystemExit, match="launch refused: .* lengths"):
        S.run_references(jobs, _paths(tmp_path, "one"), hdr3, 1, em_file, KW)
    monkeypatch.setattr(S, "DISTINCT", 0.01)
    hdr4 = S.study_header(1, em_file, None, None, dict(KW, burn=400))
    with pytest.raises(SystemExit, match="launch refused: .* lengths"):
        S.run_references(jobs, _paths(tmp_path, "one"), hdr4, 1, em_file, dict(KW, burn=400))
    assert S.run_references(jobs, _paths(tmp_path, "one"), hdr1, 1, em_file, KW) == 0          # the same header: a resume with nothing to do


def test_the_stand_in_targets_and_the_gradient_route(monkeypatch):
    """Decision 3 on the toy: the pair's own log-target under each prior, passed as a target
    with a finite-difference gradient and Σ_P from the record, gives accepted runs within G2's
    tolerances of the least-squares route's figures; displaced in H, it is accepted and its
    mean moves by the shift."""
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    R.set_emulator(em)
    job = {"label": "g2_0000_1", "set": "g2", "i": 0, "j": 1, "s": s, "theta": [float(v) for v in CENTRE], "y": y, "v2_status": "ok"}
    kw = {"burn": 1000, "sample": 3000, "kish_min": 50.0, "rhat_max": 1.2, "ess_min": 50.0}
    rec, stores = S._study_worker((job, dict(kw)))
    assert rec["status"] == "ok" and rec["direct"] == [] and len(rec["candidate"]) == 1 and rec["v2_status"] == "ok"
    rec = json.loads(json.dumps(rec, allow_nan=False))                                 # as the record holds it
    sigA = S.sigma_P_from_record(rec, "A")
    assert np.allclose(sigA, rec["candidate"][0]["kept"][0]["sigma_A"]) and sigA.shape == (4, 4)
    # the own target equals the reference's log-target under the prior; displaced, it is the translate
    tg = S.StandInTarget(y, s, em, "A", sigA)
    th = R.unit_to_theta(np.random.default_rng(1).uniform(0.1, 0.9, size=(5, 4)))
    assert np.allclose(tg(th), R.log_target(R.theta_to_unit(th), y, s, em, "A"))
    assert np.allclose(tg.bounds_unit[0], [S.EPS_H, 0, 0, 0]) and np.allclose(tg.bounds_unit[1], [1 - S.EPS_H, 1, 1, 1])
    g_fd = tg.grad_unit(R.theta_to_unit(th[0]))
    mean, cov = em.posterior(y, s)
    g_ex = -(HI - LO) * np.linalg.solve(cov, th[0] - mean)
    g_ex[0] += S.prior_grad_curv(R.theta_to_unit(th[0])[0], "A")[0]
    assert np.allclose(g_fd, g_ex, rtol=1e-5, atol=1e-6)
    td = S.StandInTarget(y, s, em, "A", sigA, shift=0.02)
    assert np.allclose(td(th), tg(th - np.array([0.02, 0, 0, 0]))) and np.isinf(td(np.array([[LO[0] + 0.01, 0.2, -0.8, 0.1]])))[0]
    du = 0.02 / (HI[0] - LO[0])
    assert abs(td.bounds_unit[0][0] - (du + S.EPS_H)) < 1e-15 and abs(td.bounds_unit[1][0] - (1 - S.EPS_H)) < 1e-15
    tn = S.StandInTarget(y, s, em, "B", sigA, shift=-0.02)
    assert abs(tn.bounds_unit[0][0] - S.EPS_H) < 1e-15 and abs(tn.bounds_unit[1][0] - (1 - du - S.EPS_H)) < 1e-15
    # the runs
    for prior in ("A", "B"):
        own = S.standin_run(job, rec, prior, "own", em, **{k: kw[k] for k in ("burn", "sample", "rhat_max", "ess_min")})
        assert own["kind"] == "standin" and own["accepted"] and own["compare"]["passed"] and own["shift_H"] == 0.0
        assert own["seed"] == [107, S.PRIOR_INDEX[prior], 0, 0, 1] and own["prior"] == prior and own["variant"] == "own"
        assert own["compare"]["keys"] == (["d_mean", "d_c", "d_q05", "d_q95"] if prior == "A" else ["d_mean_B"])
        assert all(own["compare"]["d"][k] <= own["compare"]["tol"][k] for k in own["compare"]["keys"])
        for k in own["compare"]["keys"]:                                                    # the Monte Carlo term is on the difference
            c_ = own["compare"]
            assert c_["tol"][k] >= c_["tol_ref_only"][k] and c_["mc_se_standin"][k] > 0
            assert abs(c_["tol"][k] - max({"d_mean": 0.001, "d_c": 0.01, "d_q05": 0.002, "d_q95": 0.002, "d_mean_B": 0.001}[k],
                                          4.0 * np.hypot(c_["mc_se_ref"][k], c_["mc_se_standin"][k]))) < 1e-15
        assert own["candidate"][0]["route"] == "gradient" and own["candidate"][0]["m"] == 1 and np.allclose(own["sigma_P"], S.sigma_P_from_record(rec, prior))
        assert abs(own["figures"]["mean_H"] - rec["under"][prior]["mean_H"]) < 0.005
        json.dumps(own, allow_nan=False)
        disp = S.standin_run(job, rec, prior, "displaced", em, **{k: kw[k] for k in ("burn", "sample", "rhat_max", "ess_min")})
        assert disp["accepted"] and disp["seed"] == [107, S.PRIOR_INDEX[prior], 0, 0, 1, 1] and abs(disp["shift_H"]) == 0.02
        assert disp["shift_H"] == S.standin_shift(rec, prior, "displaced")
        assert abs(disp["compare"]["mean_H_shift"] - disp["shift_H"]) < 0.005                # the posterior moved with the target
        assert disp["candidate"][0]["bounds_unit"][0][0] >= S.EPS_H
    assert S.standin_shift(rec, "A", "own") == 0.0
    assert S.standin_shift({"under": {"A": {"mean_H": 0.1}}}, "A", "displaced") == 0.02        # towards the centre of H's range
    assert S.standin_shift({"under": {"A": {"mean_H": 0.4}}}, "A", "displaced") == -0.02
    # a pair whose figures under A are direct takes the direct run's Σ_P
    rec2, _ = S._study_worker((job, dict(kw, kish_min=1e12)))
    assert rec2["direct"] == ["A", "B"] and [c["prior"] for c in rec2["candidate"]][-2:] == ["A", "B"]
    assert np.allclose(S.sigma_P_from_record(rec2, "A"), rec2["candidate"][-2]["kept"][0]["sigma"])
    assert np.allclose(S.sigma_P_from_record(rec2, "B"), rec2["candidate"][-1]["kept"][0]["sigma"])


def test_the_mass_outside_the_first_grid():
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    xi0_axis = R.cell_centres(LO[3], HI[3], 8)
    whole = S.mass_outside(LO, HI, xi0_axis, y, s, em, n=10)                              # the first grid the whole box: nothing outside
    assert whole["share_outside"] == 0.0 and whole["n_cells"] == 10 ** 3 * 8 == whole["n_cells_inside"]
    assert whole["n_per_dim"] == [10, 10, 10, 8] and whole["prior"] == "A" and whole["grid_lo"] == [float(v) for v in LO]
    mean, cov = em.posterior(y, s)
    sd = np.sqrt(np.diag(cov))
    lo4, hi4 = np.maximum(mean - 4 * sd, LO), np.minimum(mean + 4 * sd, HI)
    near = S.mass_outside(lo4, hi4, xi0_axis, y, s, em, n=16)
    assert 0.0 <= near["share_outside"] < 0.05 and near["n_cells_inside"] < near["n_cells"]
    lo1, hi1 = np.maximum(mean - sd, LO), np.minimum(mean + sd, HI)
    tight = S.mass_outside(lo1, hi1, xi0_axis, y, s, em, n=16)
    assert tight["share_outside"] > near["share_outside"]                                   # a smaller box leaves more outside
    far_lo, far_hi = LO.copy(), LO + 0.1 * (HI - LO)                                        # a box away from the posterior
    assert S.mass_outside(far_lo, far_hi, xi0_axis, y, s, em, n=16)["share_outside"] > 0.99
    # against an independent sum on a small grid: ξ₀ never counts, only (H, ν, ρ) place a cell outside
    n = 4
    axes = [R.cell_centres(LO[k], HI[k], n) for k in range(3)] + [xi0_axis]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    w = np.exp(R.log_lik(mesh, y, s, em) + R.prior_logpdf(mesh[:, 0], "A"))
    inside = np.all((mesh[:, :3] >= lo4[:3]) & (mesh[:, :3] <= hi4[:3]), axis=1)
    out = S.mass_outside(lo4, hi4, xi0_axis, y, s, em, n=n)
    assert abs(out["share_outside"] - (1 - w[inside].sum() / w.sum())) < 1e-12 and out["n_cells_inside"] == int(inside.sum())
    out_xi = S.mass_outside(np.array([lo4[0], lo4[1], lo4[2], HI[3] + 1]), hi4, xi0_axis, y, s, em, n=n)
    assert out_xi["share_outside"] == out["share_outside"]


def _att(acc, rh, ess, accepted=False, burn=10000, sample=40000):
    return {"attempt": 1, "burn": burn, "sample": sample, "acceptance": acc, "rhat": {"H": rh, "nu": 1.0, "rho": 1.0, "xi0": 1.0},
            "ess": {}, "ess_H": ess, "accepted": accepted, "seconds": 1.0}


def test_the_verdict_the_kinds_and_the_comparison_with_v2():
    assert S.failure_kind(_att(0.005, 1.0, 5000)) == "non-adaptation"                     # acceptance below 1%
    assert S.failure_kind(_att(0.05, 1.2, 100)) == "non-adaptation"                       # R-hat above 1.1
    assert S.failure_kind(_att(0.05, None, 100)) == "non-adaptation"                      # an infinite R-hat (None in the record)
    assert S.failure_kind(_att(0.1, 1.005, 500)) == "slow mixing"                         # R-hat within 1.01, ESS_H below 2,000
    assert S.failure_kind(_att(0.1, 1.05, 500)).startswith("neither kind")
    assert S.failure_kind(_att(0.01, 1.1, 1999)).startswith("neither kind")              # at the bounds: not below 1%, not above 1.1
    assert S.failure_kind(_att(0.01, 1.01, 1999)) == "slow mixing"
    jobs = [{"label": f"p{k}", "set": "g2" if k < 2 else "g2r", "i": k, "j": 0, "s": 0.001, "v2_status": "ok" if k == 0 else "reference_failure"}
            for k in range(3)]
    ok_rec = {"status": "ok", "runs": {"uniform": [_att(0.2, 1.0, 3000, True)]}, "under": {"A": {"source": "reweighted"}, "B": {"source": "reweighted"}},
              "sampler": S.SAMPLER_TAG}
    tasks = {"p0": ok_rec, "p1": {"status": "reference_failure", "sampler": S.SAMPLER_TAG,
                                  "runs": {"uniform": [_att(0.005, 1.3, 4), _att(0.02, 1.005, 900, burn=20000, sample=80000)]}}}
    checks = {"pairs": [{"task": "p0", "status": "ok", "passed": True}, {"task": "p1", "status": "reference_failure", "passed": False}],
              "go": True, "projection": {"projected_core_hours": 50.0}, "budget_core_hours": 1344.0}
    v = S.verdict(jobs, tasks, checks, {}, n_expected=3)
    assert not v["passed"] and not v["complete"] and v["missing"] == ["p2"] and v["not_accepted"] == ["p1"]
    assert v["kinds"]["p1"]["uniform"] == ["non-adaptation", "slow mixing"] and v["reading"] == "the study is incomplete"
    assert v["go"] and v["projected_core_hours"] == 50.0 and v["gradient_route"]["passed"] is False
    tasks["p2"] = ok_rec
    checks["pairs"].append({"task": "p2", "status": "ok", "passed": True})
    v = S.verdict(jobs, tasks, checks, {}, n_expected=3)
    assert v["complete"] and not v["passed"] and v["reading"] == "(A) fails the development study" and v["grid_misses"] == []
    tasks["p1"] = ok_rec
    checks["pairs"][1] = {"task": "p1", "status": "ok", "passed": True}
    own = {(f"p{k}", p, "own"): {"task": f"p{k}", "prior": p, "variant": "own", "accepted": True, "compare": {"passed": True}}
           for k in range(3) for p in ("A", "B")}
    v = S.verdict(jobs, tasks, checks, own, n_expected=3)
    assert v["passed"] and v["reading"] == "(A) passes the development study" and v["kinds"] == {}
    assert v["gradient_route"]["passed"] and v["gradient_route"]["n_own"] == 6 == v["gradient_route"]["n_own_expected"]
    checks["pairs"][2] = {"task": "p2", "status": "ok", "passed": False}                      # a grid miss fails the study
    v = S.verdict(jobs, tasks, checks, own, n_expected=3)
    assert not v["passed"] and v["grid_misses"] == ["p2"] and v["not_accepted"] == []
    checks["pairs"][2]["passed"] = True
    checks["go"] = False                                                                       # so does a projection above the budget
    assert not S.verdict(jobs, tasks, checks, own, n_expected=3)["passed"]
    checks["go"] = True
    own[("p1", "A", "own")] = dict(own[("p1", "A", "own")], compare={"passed": False})           # the gradient route beside the verdict
    v = S.verdict(jobs, tasks, checks, own, n_expected=3)
    assert v["passed"] and not v["gradient_route"]["passed"] and v["gradient_route"]["own_outside_tolerance"] == ["p1/A"]
    own[("p1", "A", "own")]["accepted"] = False
    assert S.verdict(jobs, tasks, checks, own, n_expected=3)["gradient_route"]["own_not_accepted"] == ["p1/A"]
    assert not S.verdict(jobs, tasks, checks, own, n_expected=36)["complete"]                  # 36 pairs are expected
    v2run = dict(tasks, p2={k: v for k, v in ok_rec.items() if k != "sampler"})                 # a record v2's sampler made: not the study's
    vv = S.verdict(jobs, v2run, checks, own, n_expected=3)
    assert vv["not_candidate"] == ["p2"] and not vv["complete"] and not vv["passed"]
    # the pairs v2 failed, beside v2's attempts at the lengths each ran
    sources = {"g2": {"tasks": {"p0": {"status": "ok"}, "p1": {"status": "reference_failure", "runs": {"uniform": [_att(0.003, 12.4, 4), _att(0.003, 7.2, 5, burn=20000, sample=80000)]}}}},
               "g2r": {"tasks": {"p2": {"status": "reference_failure", "runs": {"uniform": [_att(0.004, 6.9, 4, burn=20000, sample=80000), _att(0.019, 25.9, 4, burn=40000, sample=160000)]}}}}}
    rows = S.v2_comparison(jobs, tasks, sources)
    assert [r["task"] for r in rows] == ["p1", "p2"] and rows[0]["v2_lengths_in_units_of_the_studys"] == [1, 2] and rows[1]["v2_lengths_in_units_of_the_studys"] == [2, 4]
    assert rows[0]["v2_runs"]["uniform"][0]["max_rhat"] == 12.4 and rows[0]["study_status"] == "ok" and rows[0]["study_runs"]["uniform"][0]["accepted"] is True
    assert rows[1]["v2_runs"]["uniform"][1]["ess_H"] == 4 and rows[1]["v2_status"] == "reference_failure"
    # the per-pair rows carry the candidate's diagnostics, the checks and the mass share
    diag = {"prior": "uniform", "burn": 10000, "sample": 40000, "m": 1, "kept": [{"n_starts_reached": 16, "eig": [1, 2, 3, 4], "n_clipped": 0}],
            "starts_elsewhere": 0, "optimiser": {"status": [2] * 16}, "acceptance_first_500": [0.3] * 4, "n_start_clipped": 0,
            "optimiser_seconds": 0.1, "proposal_vs_sample": None}
    tasks["p0"] = dict(ok_rec, candidate=[diag], direct=[], seconds=3.0, c_A=0.5)
    rows = S.per_pair_rows(jobs, tasks, checks, {"p0": {"share_outside": 0.01}})
    assert rows[0]["candidate_runs"][0]["m"] == 1 and rows[0]["candidate_runs"][0]["starts_per_solution"] == [16]
    assert rows[0]["mass_outside_first_grid"] == 0.01 and rows[0]["checks_passed"] is True and rows[0]["kinds"] == {"uniform": []}
    assert rows[2]["mass_outside_first_grid"] is None


def _toy_sources(tmp_path, em, n_points=2):
    """Two toy records in v2's form, one per set, each with n_points × 3 pairs whose θ and y are
    g2_surfaces(retry) priced through `em` (so check_surfaces passes), headers with v2's
    seeds and the emulator's SHA-256."""
    paths = {}
    for name in ("g2", "g2r"):
        pts, eps, seeds = R.g2_surfaces(retry=(name == "g2r"))
        Y = R.surfaces(pts[:n_points], eps[:n_points], em)
        p = tmp_path / f"src_{name}.jsonl"
        hdr = {"kind": "header", "set": name, "seeds": seeds, "seed_ref": R.SEED_REF, "commit": "toy", "lengths": {"burn": 10000, "sample": 40000},
               "emulator_sha256": E.sha256_file(R._EM["path"]) if R._EM["path"] and Path(str(R._EM["path"])).exists() else None}
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(hdr) + "\n")
            for i in range(n_points):
                for j, s in enumerate(R.NOISE_LEVELS):
                    f.write(json.dumps({"kind": "task", "task": f"{name}_{i:04d}_{j}", "set": name, "i": i, "j": j, "s": s, "seed": [83, i, j],
                                        "theta": [float(v) for v in pts[i]], "y": [float(v) for v in Y[i, j]],
                                        "status": "ok" if (i + j) % 2 else "reference_failure", "direct": [],
                                        "runs": {"uniform": [_att(0.1, 1.0, 3000, True)]}}) + "\n")
        paths[name] = str(p)
    return paths


def test_hold_mode_runs_nothing_and_the_refusals(tmp_path, monkeypatch, capsys):
    _no_sealed_draws(monkeypatch)                                                            # a draw would fail the test (D82)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(S, "PATHS", {k: str(out_dir / Path(v).name) for k, v in S.PATHS.items()})
    monkeypatch.setattr(S, "SOURCES", {"g2": str(tmp_path / "no_g2.jsonl"), "g2r": str(tmp_path / "no_g2r.jsonl")})
    monkeypatch.setattr(S, "SOURCE_REPORTS", {"g2": str(tmp_path / "no.json"), "g2r": str(tmp_path / "no2.json")})

    def boom(*a, **k):
        raise AssertionError("the emulator was touched in hold mode")
    with monkeypatch.context() as m:
        for mod, name in ((E, "require_g1"), (E, "emulator_for_run"), (E, "load_emulator"), (R, "get_emulator"), (R, "launch")):
            m.setattr(mod, name, boom)
        S.main([])                                                                           # no flag: the design, then hold
        out = capsys.readouterr().out
        assert "HOLD: nothing run" in out and "no record at" in out and "least_squares trf" in out and "max_nfev 2000" in out
        assert "estimate: v2's G2 reports not found" in out and "sealed seeds (20261002, 20261022) refused" in out
        assert list(out_dir.iterdir()) == [] and R.run_chains is S._V2_RUN_CHAINS
        # with records present (toy ones), hold mode lists the pairs and still writes nothing, loads nothing
        R.set_emulator(LinearEm())
        src = _toy_sources(tmp_path, R._EM["em"], n_points=6)
        m.setattr(S, "SOURCES", src)
        S.main([])
        out = capsys.readouterr().out
        assert "source g2:" in out and "18 pairs" in out and "g2r_0005_2: s=0.005 key [83, 5, 2]" in out and "HOLD" in out
        assert list(out_dir.iterdir()) == []
        est = S.estimate({"g2": R.SETS["g2"]["report"], "g2r": R.SETS["g2r"]["report"]}, pool=4)   # v2's committed reports, read only
        if est is not None:
            assert est["core_hours"] > 0 and est["wall_hours_at_pool"] == est["core_hours"] / 4 and set(est["parts_core_hours"]) == \
                {"sampling", "first_grids", "whole_box_checks", "mass_grids", "standins"}
    # a sealed seed in a source's header is refused before anything else
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"kind": "header", "seeds": {"points": 79, "noise": 20261022}, "seed_ref": 83}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed seed.*20261022"):
        S.read_sources({"g2": str(bad)})
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"kind": "header", "seeds": {"points": 79}, "seed_ref": 83}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="holds 0 pair records, not 18"):
        S.read_sources({"g2": str(empty)})
    with pytest.raises(SystemExit, match="no record at"):
        S.read_sources({"g2": str(tmp_path / "absent.jsonl")})
    # --run: the report existing is refused before the emulator is touched; another --emulator than G̃ is refused before any load;
    # sources made with another emulator are refused
    monkeypatch.setattr(S, "SOURCES", _toy_sources(tmp_path, LinearEm(), n_points=6))
    Path(S.PATHS["report"]).write_text("{}", encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr(E, "require_g1", boom)
        with pytest.raises(SystemExit, match="study refused: .*p5v3_study_report.json exists"):
            S.main(["--run", "--pool", "1"])
    Path(S.PATHS["report"]).unlink()
    em_file = tmp_path / "em.npz"
    np.savez(em_file, stage=np.array("base"), commit=np.array("x"), values=np.zeros((2, 1, 1, 2, 3)))
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: str(em_file))
    with pytest.raises(SystemExit, match="is not G̃"):
        S.main(["--run", "--pool", "1", "--emulator", str(tmp_path / "other.npz")])
    with pytest.raises(SystemExit, match="was made with emulator None, not G̃"):
        S.main(["--run", "--pool", "1"])
    assert list(out_dir.iterdir()) == []
    with pytest.raises(SystemExit, match="--pool 0"):
        S.main(["--pool", "0"])
    # the version guard, as the reference's
    monkeypatch.setattr(E, "CORE_VERSIONS", {"numpy": "0.0.0", "scipy": "0.0.0"})
    S.main([])
    with pytest.raises(SystemExit, match="refused: numpy"):
        S.main([], check_versions=True)
    S.main(["--allow-other-versions"], check_versions=True)
    assert list(out_dir.iterdir()) == []


def test_a_small_run_end_to_end_on_toy_records(tmp_path, monkeypatch, capsys):
    """--run on toy records at pool 1: the pre-flight, the references with the candidate, the
    checks, the mass lines, the stand-ins, the verdict and the report; then a second --run is
    refused, the report existing."""
    _no_sealed_draws(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(S, "PATHS", {k: str(out_dir / Path(v).name) for k, v in S.PATHS.items()})
    monkeypatch.setattr(S, "SOURCE_REPORTS", {"g2": str(tmp_path / "no.json"), "g2r": str(tmp_path / "no2.json")})
    monkeypatch.setattr(R, "EMULATOR_PATH", str(tmp_path / "no_g_tilde.npz"))
    em_file = _toy_emulator_file(tmp_path / "toy_em.npz", LinearEm())
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: em_file)             # the toy file stands in for G̃
    R._EM["path"], R._EM["em"] = None, None
    em = R.get_emulator(em_file)
    monkeypatch.setattr(S, "SOURCES", _toy_sources(tmp_path, em, n_points=1))                 # 3 pairs per set, 6 in all
    monkeypatch.setattr(S, "N_PAIRS_EXPECTED", 6)
    monkeypatch.setattr(S, "TASK_KWARGS", dict(KW_E2E))
    monkeypatch.setattr(S, "GRID_N", 4)
    monkeypatch.setattr(S, "WHOLE_N", 4)
    monkeypatch.setattr(S, "GRID_MISS_N", 6)                                                 # a miss on the coarse grid recomputes at 6, not 64
    sources = S.read_sources()
    jobs = S.study_jobs(sources)
    assert [j["label"] for j in jobs] == ["g2_0000_0", "g2_0000_1", "g2_0000_2", "g2r_0000_0", "g2r_0000_1", "g2r_0000_2"]
    assert S.check_surfaces(jobs, em, sources)
    other = [dict(j, y=np.asarray(j["y"]) + 1e-6) for j in jobs]
    with pytest.raises(SystemExit, match="is not the surface g2_surfaces gives"):
        S.check_surfaces(other, em, sources)
    S.main(["--run", "--pool", "1"])
    out = capsys.readouterr().out
    assert "=== RUN: 6 pairs to do" in out and "=== checks:" in out and "=== MASS: 6 pairs to do" in out
    assert "=== STAND-INS: 24 runs to do" in out and "=== VERDICT:" in out
    assert sorted(p.name for p in out_dir.iterdir()) == ["p5v3_study.jsonl", "p5v3_study_checks.json", "p5v3_study_report.json",
                                                         "p5v3_study_samples.npz", "p5v3_study_samples_parts"]
    hdr, launches, tasks, mass, standins, errors = S.read_study_record(S.PATHS["jsonl"])
    assert hdr["set"] == "p5v3_study" and hdr["n_points"] == 2 and hdr["sources"]["g2r"]["sha256_lf"] == S.sha256_lf(S.SOURCES["g2r"])
    crlf, lf = tmp_path / "crlf.jsonl", tmp_path / "lf.jsonl"
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    assert S.sha256_lf(str(crlf)) == S.sha256_lf(str(lf)) == E.sha256_file(str(lf)) != E.sha256_file(str(crlf))   # the committed blob's hash
    assert S.sha256_lf(str(tmp_path / "absent")) is None
    assert hdr["seeds"]["g2"]["points"] == 79 and hdr["seeds"]["g2r"]["points"] == 131 and hdr["lengths"]["candidate"]["n_starts"] == 16
    assert hdr["emulator_sha256"] == E.sha256_file(em_file) and hdr["script"] == "p5_v3_study.py"
    assert (hdr["lengths"]["burn"], hdr["lengths"]["sample"], hdr["lengths"]["kish_min"], hdr["lengths"]["ess_min"]) == (200, 600, 20.0, 10.0)
    assert len(tasks) == 6 and all(t["status"] == "ok" and t["candidate"] for t in tasks.values()) and errors == []
    assert {t["set"] for t in tasks.values()} == {"g2", "g2r"} and {t["v2_status"] for t in tasks.values()} == {"ok", "reference_failure"}
    assert sorted(mass) == sorted(tasks) and all(0 <= m_["share_outside"] <= 1 and m_["n_per_dim"] == [4, 4, 4, 4] for m_ in mass.values())
    assert len(standins) == 24 and all(v["accepted"] for v in standins.values())
    assert all(v["compare"]["passed"] for v in standins.values() if v["variant"] == "own")
    checks = json.load(open(S.PATHS["checks"], encoding="utf-8"))
    assert checks["gate"] == "G2" and checks["n_pairs"] == 6 and checks["lengths"] == {"burn": 200, "sample": 600}
    assert checks["grid_n"] == 4 and checks["whole_n"] == 4 and sum("whole_box" in r for r in checks["pairs"]) == 2
    rep = json.load(open(S.PATHS["report"], encoding="utf-8"))
    json.dumps(rep, allow_nan=False)
    v = rep["verdict"]
    assert v["complete"] and v["missing"] == [] and v["not_accepted"] == [] and (v["n_pairs"], v["n_expected"]) == (6, 6)
    assert v["not_candidate"] == [] and rep["errors_unresolved"] == [] and rep["header_commit"] == hdr["commit"]
    assert v["passed"] == (v["grid_misses"] == [] and v["go"]) and v["gradient_route"]["passed"] and v["gradient_route"]["n_own"] == 12
    assert rep["projection"]["n_items"] == 1200 and rep["projection"]["base_steps"] == 800 and "runs no gate" in rep["note"]
    assert len(rep["pairs"]) == 6 and all(r["mass_outside_first_grid"] is not None and r["candidate_runs"] for r in rep["pairs"])
    assert [r["task"] for r in rep["v2_failed_pairs"]] == [j["label"] for j in jobs if j["v2_status"] == "reference_failure"]
    assert rep["emulator"]["sha256"] == E.sha256_file(em_file) and rep["candidate"] == S.candidate_constants() and rep["errors"] == []
    assert set(rep["grid_miss_recomputed"]) == set(v["grid_misses"])                           # each miss: the first grid again, finer
    assert all(g["grid"]["n_per_dim"] == 6 and g["grid"]["n_cells"] == 6 ** 4 and "whole_box" not in g for g in rep["grid_miss_recomputed"].values())
    with pytest.raises(SystemExit, match="study refused: .*exists"):                           # no record is written over
        S.main(["--run", "--pool", "1"])
    other = dict(checks, lengths={"burn": 999, "sample": 600})                                  # a checks file from other lengths is refused
    (tmp_path / "other_checks.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit, match="checks refused"):
        S.run_checks(jobs, dict(S.PATHS, checks=str(tmp_path / "other_checks.json")), em, 1, em_file, None, KW_E2E)
    # a resume: nothing to do, the checks read back, nothing rewritten
    Path(S.PATHS["report"]).rename(tmp_path / "report_moved_aside.json")
    before = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    S.main(["--run", "--pool", "1"])
    out = capsys.readouterr().out
    assert "=== RUN: 0 pairs to do (6 already" in out and "checks read from" in out and "MASS: 0 pairs to do (6 already)" in out
    assert "STAND-INS: 0 runs to do (24 already)" in out
    after = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    assert after["p5v3_study_checks.json"] == before["p5v3_study_checks.json"] and after["p5v3_study_samples.npz"] == before["p5v3_study_samples.npz"]
    rep2 = json.load(open(S.PATHS["report"], encoding="utf-8"))
    assert rep2["verdict"] == v and rep2["mass_outside_first_grid"] == rep["mass_outside_first_grid"]
