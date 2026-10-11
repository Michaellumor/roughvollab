"""
test_p5_g2_reference.py — mechanics tests for the P5 gate G2 reference script. They
guard the MACHINERY on toy emulators in memory: the literals written into the draft
(lines 62 to 68, 81, 93, 143, 144); the split R-hat and the between-within effective
sample size on synthetic chains; the weighted summaries; the sampler on a linear
emulator, whose posterior is a known Gaussian; the repeat and failure rules; the
reweighting and the direct-run path; the grid checks; the draws; the record, resume
and report on a toy pilot. Nothing here touches the real emulator, which is built at G1
after freezing.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta as _beta

import p5_g2_reference as R
import p5_g1_emulator as E

LO, HI = E.box_arrays(E.BOX)
CENTRE = (LO + HI) / 2


class LinearEm:
    """G̃(θ) = a + M θ: the posterior under the uniform prior is a Gaussian on the box."""

    def __init__(self, seed=5, scale=0.03):
        rng = np.random.default_rng(seed)
        self.M = rng.normal(0.0, scale, size=(35, 4))
        self.a = 0.2 + rng.uniform(0, 0.05, size=35)
        self.box = E.BOX

    def evaluate_many(self, thetas):
        return self.a[None, :] + np.asarray(thetas, dtype=float).reshape(-1, 4) @ self.M.T

    def posterior(self, y, s):
        """Mean and covariance of the untruncated Gaussian posterior."""
        cov = s * s * np.linalg.inv(self.M.T @ self.M)
        mean = cov @ self.M.T @ (np.asarray(y) - self.a) / (s * s)
        return mean, cov


def toy_surface(em, theta=CENTRE, s=0.003, seed=11):
    return em.evaluate_many(theta)[0] + s * np.random.default_rng(seed).standard_normal(35)


def test_literals_are_the_written_ones():
    assert (R.SEED_REF, R.SEED_G2_POINTS, R.N_G2_POINTS, R.SEED_G2_NOISE) == (83, 79, 4, 89)
    assert (R.SEED_G3_POINTS, R.N_G3_POINTS, R.SEED_G3_NOISE, R.SEED_TEST) == (97, 16, 101, 20261002)
    assert (R.N_CHAINS, R.BURN, R.SAMPLE, R.ADAPT_EVERY) == (4, 10000, 40000, 500)
    assert abs(R.SCALE - 2.38 ** 2 / 4) < 1e-15 and R.INIT_VAR == 1 / 12 and R.DIAG_FLOOR == 1e-10
    assert (R.RHAT_MAX, R.ESS_MIN, R.KISH_MIN) == (1.01, 2000.0, 1000.0)
    assert (R.GRID_N, R.EXTEND, R.WHOLE_N, R.TOL_MEAN, R.TOL_C) == (32, 0.20, 48, 0.001, 0.01)
    assert R.PRIOR_AB["A"] == (2.5, 9.0) and abs(R.PRIOR_AB["B"][0] - 235 / 23) < 1e-15 and abs(R.PRIOR_AB["B"][1] - 376 / 69) < 1e-15
    a, b = R.PRIOR_AB["B"]
    assert abs(a + b - 47 / 3) < 1e-12 and abs(a / (a + b) - 15 / 23) < 1e-15                   # line 66
    for which in ("A", "B"):
        aa, bb = R.PRIOR_AB[which]
        assert abs(_beta(aa, bb).var() - 36 / 2645) < 1e-15                                  # equal variances
    assert abs(R.VAR_PRIOR_A - 0.46 ** 2 * 36 / 2645) < 1e-18 and abs(R.VAR_PRIOR_A - 0.00288) < 1e-17
    assert R.H_LO == 0.02 and abs(R.H_W - 0.46) < 1e-15 and R.NOISE_LEVELS == (0.001, 0.003, 0.005)
    assert {R.SEED_REF, R.SEED_G2_POINTS, R.SEED_G2_NOISE}.isdisjoint({20261002, 7, 61, 62, 73, 97, 101})
    assert R.SETS["g2"]["jsonl"].endswith("p5_g2_pilot.jsonl") and R.SETS["test"]["samples"].endswith("p5_ref_test_samples.npz")


def test_split_rhat_and_ess_on_synthetic_chains():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((4, 4000))
    assert abs(R.split_rhat(iid) - 1.0) < 0.01
    ess = R.ess_bw(iid)
    assert 0.8 * 16000 < ess < 1.3 * 16000
    phi, n = 0.5, 20000                                                   # AR(1): ESS ≈ mn (1 − φ)/(1 + φ)
    ar = np.empty((4, n))
    for k in range(4):
        e = rng.standard_normal(n)
        ar[k, 0] = e[0] / np.sqrt(1 - phi ** 2)
        for t in range(1, n):
            ar[k, t] = phi * ar[k, t - 1] + e[t]
    ess_ar = R.ess_bw(ar)
    assert abs(ess_ar / (4 * n / 3) - 1.0) < 0.25
    shifted = iid + np.array([[0.0], [0.0], [3.0], [3.0]])
    assert R.split_rhat(shifted) > 1.01
    v = R._variogram(np.array([[1.0, 2.0, 4.0, 7.0]]))                  # (1 + 4 + 9)/3, (9 + 25)/2, 36
    assert np.allclose(v, [14 / 3, 17.0, 36.0])
    x = np.arange(10.0)[None, :]
    assert R.split_chains(np.stack([x[0], x[0] + 1])).shape == (4, 5)


def test_weighted_summaries_kish_interval_and_quantisation():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    w = np.array([1.0, 1.0, 2.0, 0.0])
    assert abs(R.wmean(x, w) - 0.225) < 1e-15 and abs(R.wvar(x, w) - (0.125 ** 2 + 0.025 ** 2 + 2 * 0.075 ** 2) / 4) < 1e-15
    assert R.wquantile(x, w, 0.25) == 0.1 and R.wquantile(x, w, 0.5) == 0.2 and R.wquantile(x, w, 0.51) == 0.3
    assert R.wquantile(x, w, 1.0) == 0.3                                   # the zero-weight point is never reached
    w3 = np.array([0.1, 0.2, 0.7]) * (1 - 1e-16)                           # cumulative weight short of 1 by rounding
    assert R.wquantile(np.array([1.0, 2.0, 3.0]), w3 / w3.sum() * (1 - 2e-16), 1.0) == 3.0
    assert abs(R.kish(w) - 16 / 6) < 1e-15 and R.kish(np.ones(7)) == 7.0
    assert R.interval_prob(x, w, 0.15, 0.35) == 0.75 and R.interval_prob(x, w, 0.0, 1.0) == 1.0
    H = np.array([0.02, 0.48, 0.1234567])
    q = R.quantise_H(H)
    assert q.dtype == np.uint16 and q[0] == 0 and q[1] == 65535
    assert np.max(np.abs(R.dequantise_H(q) - H)) <= 0.46 / 65535 / 2 + 1e-12
    assert R.contraction(R.VAR_PRIOR_A) == 0.0 and abs(R.contraction(0.1 * R.VAR_PRIOR_A) - 0.9) < 1e-12
    assert np.all(np.isfinite(R.prior_logpdf(np.array([0.05, 0.3]), "A"))) and R.prior_logpdf(0.3, "uniform") == 0.0
    assert np.isinf(R.prior_logpdf(0.01, "A"))


def test_sampler_on_a_linear_emulator_matches_its_gaussian_posterior():
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    mean, cov = em.posterior(y, s)
    sd = np.sqrt(np.diag(cov))
    assert np.all(mean - 4 * sd > LO) and np.all(mean + 4 * sd < HI)       # the truncation is negligible
    rng = np.random.default_rng([R.SEED_REF, 0, 0])
    run = R.run_chains(y, s, em, rng, "uniform", burn=2000, sample=4000)
    pooled = R.unit_to_theta(run["samples"].reshape(-1, 4))
    ok, diag = R.accept(run["samples"], rhat_max=1.1, ess_min=100)
    assert ok and diag["ess_H"] > 100 and diag["ess_H"] == diag["ess"]["H"] and set(diag["ess"]) == set(E.DIMS)
    for p, d in enumerate(E.DIMS):
        se = sd[p] / np.sqrt(diag["ess"][d])                               # each parameter's own effective size
        assert abs(pooled[:, p].mean() - mean[p]) < 4 * se + 1e-9
    assert 0.1 < run["acceptance"] < 0.6
    assert run["samples"].shape == (4, 4000, 4) and np.all(run["samples"] >= 0) and np.all(run["samples"] <= 1)
    run2 = R.run_chains(y, s, em, np.random.default_rng([R.SEED_REF, 0, 0]), "uniform", burn=2000, sample=4000)
    assert np.array_equal(run["samples"], run2["samples"])                 # the same generator, the same chains
    # a proposal outside the box is rejected without evaluation: log_target is -inf there
    assert np.isinf(R.log_target(np.array([[1.2, 0.5, 0.5, 0.5]]), y, s, em, "uniform"))[0]
    # the adapted covariance is 2.38^2/4 times the empirical one, diagonal floored
    steps = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    c = R.adapted_cov(steps)
    assert np.allclose(np.diag(c), R.DIAG_FLOOR) and np.allclose(c - np.diag(np.diag(c)), 0)
    steps2 = np.random.default_rng(3).uniform(size=(600, 4))
    assert np.allclose(R.adapted_cov(steps2), R.SCALE * np.cov(steps2.T, ddof=1))


def test_repeat_then_reference_failure(monkeypatch):
    em = LinearEm()
    y = toy_surface(em)
    monkeypatch.setattr(R, "split_rhat", lambda x: 1.5)
    rec, stores = R.reference_pair(y, 0.003, em, 2, 1, burn=200, sample=400)
    assert rec["status"] == "reference_failure" and stores == {}
    att = rec["runs"]["uniform"]
    assert [(a["burn"], a["sample"]) for a in att] == [(200, 400), (400, 800)]           # doubled lengths
    assert all(a["accepted"] is False and a["rhat"]["H"] == 1.5 for a in att)
    assert rec["seed"] == [83, 2, 1] and rec["under"] == {} and rec["c_A"] is None


def test_reweighting_direct_runs_and_grid_against_the_gaussian():
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    mean, cov = em.posterior(y, s)
    # the A-weighted posterior by exact Gaussian draws, as the independent yardstick
    draws = np.random.default_rng(21).multivariate_normal(mean, cov, size=400000)
    wA = np.exp(R.prior_logpdf(draws[:, 0], "A"))
    ref_mean_A, ref_c_A = R.wmean(draws[:, 0], wA), R.contraction(R.wvar(draws[:, 0], wA))
    rec, stores = R.reference_pair(y, s, em, 0, 2, burn=2000, sample=6000, kish_min=100.0, rhat_max=1.1, ess_min=100.0)
    assert rec["status"] == "ok" and rec["direct"] == [] and set(stores) == {"u"}
    assert rec["under"]["A"]["source"] == "reweighted" and rec["kish"]["A"] >= 100
    ess = rec["runs"]["uniform"][0]["ess_H"]
    se = np.sqrt(cov[0, 0] / ess)
    assert abs(rec["under"]["A"]["mean_H"] - ref_mean_A) < 5 * se
    assert abs(rec["under"]["A"]["q05_H"] - R.wquantile(draws[:, 0], wA, 0.05)) < 8 * se
    assert abs(rec["c_A"] - ref_c_A) < 0.05
    H, w = R.weights_for(rec, stores, "A")
    assert abs(R.wmean(H, w) - rec["under"]["A"]["mean_H"]) < 1e-4                        # from the stored samples
    # the direct-run path, forced by a Kish floor no reweighting can meet
    rec2, stores2 = R.reference_pair(y, s, em, 0, 2, burn=2000, sample=4000, kish_min=1e12, rhat_max=1.1, ess_min=100.0)
    assert rec2["direct"] == ["A", "B"] and set(stores2) == {"u", "A", "B"}
    assert rec2["under"]["A"]["source"] == "direct" and "A" in rec2["runs"] and "B" in rec2["runs"]
    assert abs(rec2["under"]["A"]["mean_H"] - ref_mean_A) < 5 * se + 5e-4
    H2, w2 = R.weights_for(rec2, stores2, "A")
    assert np.all(w2 == 1.0) and len(H2) == 4 * 4000
    # the grid of line 143 on the sampled support against the same yardstick
    gc, axes = R.grid_check(rec, y, s, em, n=12)
    assert gc["n_cells"] == 12 ** 4 and len(axes) == 4 and all(len(ax) == 12 for ax in axes)
    assert abs(gc["mean_H"] - ref_mean_A) < 1e-3 and abs(gc["c_A"] - ref_c_A) < 0.02
    assert gc["passed"] == (gc["d_mean"] <= 0.001 and gc["d_c"] <= 0.01)
    lo_e, hi_e = R.extended_support(rec["support_lo"], rec["support_hi"])
    assert np.all(lo_e >= LO) and np.all(hi_e <= HI) and np.all(lo_e <= rec["support_lo"]) and np.all(hi_e >= rec["support_hi"])
    assert gc["grid_lo"] == [float(v) for v in lo_e] and axes[0][0] == lo_e[0] + 0.5 / 12 * (hi_e[0] - lo_e[0])
    wb = R.whole_box_check(rec, y, s, em, axes[3], n=6)
    assert wb["n_cells"] == 6 ** 3 * 12 and wb["n_per_dim"] == [6, 6, 6, 12]
    assert wb["d_mean"] >= 0 and isinstance(wb["passed"], bool) and wb["mcmc_c_A"] == rec["c_A"]   # mechanics: coarse grid
    # the support extension clips at the box
    lo_c, hi_c = R.extended_support(LO + 0.001, HI - 0.001)
    assert np.array_equal(lo_c, LO) and np.array_equal(hi_c, HI)


def test_draws_surfaces_and_the_sealed_set():
    pts = R.pilot_points(79, 4)
    assert pts.shape == (4, 4) and np.all(pts >= LO) and np.all(pts <= HI)
    assert np.array_equal(pts, np.random.default_rng(79).uniform(LO, HI, size=(4, 4)))
    eps = R.noise_draws(89, 4)
    assert eps.shape == (4, 3, 35) and np.array_equal(eps, np.random.default_rng(89).standard_normal((4, 3, 35)))
    em = LinearEm()
    Y = R.surfaces(pts, eps, em)
    assert Y.shape == (4, 3, 35)
    assert np.allclose(Y[2, 1], em.evaluate_many(pts[2])[0] + 0.003 * eps[2, 1])
    p5, p3 = R.pilot_points(79, 5), R.pilot_points(79, 3)                 # the head property of a uniform draw
    assert np.array_equal(p5[:3], p3)                                      # (checked on a pilot seed: line 95 keeps
    import inspect                                                         #  the sealed stream for the test set)
    assert "SEED_TEST" in inspect.getsource(R.test_points) and R.SEED_TEST == 20261002
    g3 = R.pilot_points(97, 16)
    assert g3.shape == (16, 4) and not np.array_equal(g3[:4], pts)
    assert R.cell_centres(0.0, 1.0, 4).tolist() == [0.125, 0.375, 0.625, 0.875]
    with pytest.raises(SystemExit, match="needs --n and --noise-seed"):
        R.main(["--test", "--n", "3", "--emulator", "no_such_file.npz"])              # refused before any load
    with pytest.raises(SystemExit, match="no emulator"):
        R.main(["--pilot", "--emulator", "no_such_file.npz"])                         # G1 is built after freezing
    R.main([])                                                                        # no flag: the design, then hold


@pytest.fixture
def toy_emulator():
    """A linear toy installed as the process's emulator for pool-1 runs (the pool path
    loads from a file per worker and cannot be exercised with an in-memory toy)."""
    em = LinearEm()
    R.set_emulator(em)
    try:
        yield em
    finally:
        R._EM["path"], R._EM["em"] = None, None


def test_toy_pilot_record_resume_and_report(tmp_path, capsys, toy_emulator):
    em = toy_emulator
    paths = {"jsonl": str(tmp_path / "r.jsonl"), "samples": str(tmp_path / "r.npz"), "report": str(tmp_path / "r.json")}
    pts = np.array([CENTRE, CENTRE + 0.1 * (HI - LO)])
    eps = R.noise_draws(89, 2)
    kw = {"burn": 500, "sample": 1500, "kish_min": 50.0, "rhat_max": 1.2, "ess_min": 50.0}
    levels = (0.003,)
    jobs = R.run_set("g2", pts, eps, em, 1, paths, "no_such_file.npz", {"points": 79, "noise": 89}, levels, kw)
    jobs2 = R.run_set("g2", pts, eps, em, 1, paths, "no_such_file.npz", {"points": 79, "noise": 89}, levels, kw)
    hdr, launches, tasks, stores = R.load_references(paths["jsonl"], paths["samples"])
    assert [j["label"] for j in jobs] == ["g2_0000_0", "g2_0001_0"] == [j["label"] for j in jobs2]
    assert all(np.array_equal(a["y"], b["y"]) for a, b in zip(jobs, jobs2))
    assert hdr["set"] == "g2" and hdr["burn"] == 10000 and hdr["seed_ref"] == 83 and hdr["priors"]["A"] == [2.5, 9.0]
    assert [l["n_done_before"] for l in launches] == [0, 2] and set(tasks) == {"g2_0000_0", "g2_0001_0"}
    assert all(t["status"] == "ok" and t["runs"]["uniform"][0]["burn"] == 500 for t in tasks.values())
    assert set(stores) >= {"g2_0000_0_u", "g2_0001_0_u"} and stores["g2_0000_0_u"].dtype == np.uint16
    assert Path(paths["samples"]).exists() and sorted(Path(R.samples_dir(paths["samples"])).glob("*.npy"))
    rec = tasks["g2_0000_0"]
    assert rec["theta"] == [float(v) for v in CENTRE] and len(rec["y"]) == 35 and rec["seed"] == [83, 0, 0]
    assert set(rec["under"]) == {"A", "B"} and rec["c_A"] == R.contraction(rec["under"]["A"]["var_H"])
    assert set(rec["runs"]["uniform"][0]["ess"]) == set(E.DIMS)
    rep = R.g2_report(jobs, paths, em, 1, levels, grid_n=8, whole_n=4)
    assert rep["gate"] == "G2" and rep["n_pairs"] == 2 and len(rep["pairs"]) == 2
    assert all("grid" in r and "whole_box" in r for r in rep["pairs"])               # one level: it is the largest
    assert rep["pairs"][0]["grid"]["n_cells"] == 8 ** 4 and rep["pairs"][0]["whole_box"]["n_cells"] == 4 ** 3 * 8
    assert rep["passed"] == all(r["passed"] for r in rep["pairs"]) and rep["cost"]["posterior_median_s"] > 0
    json.dumps(json.loads(Path(paths["report"]).read_text(encoding="utf-8")), allow_nan=False)
    with pytest.raises(SystemExit, match="launch refused"):
        R.run_set("g2", pts, eps, em, 1, paths, "no_such_file.npz", {"points": 79, "noise": 89}, levels,
                  dict(kw, burn=600))
    # a pair recorded without its samples is done again on resume; a surface from another emulator is refused
    for f in Path(R.samples_dir(paths["samples"])).glob("g2_0001_0_*.npy"):
        f.unlink()
    Path(paths["samples"]).unlink()
    R.run_set("g2", pts, eps, em, 1, paths, "no_such_file.npz", {"points": 79, "noise": 89}, levels, kw)
    other = [dict(j, y=np.asarray(j["y"]) + 1e-6) for j in jobs]
    with pytest.raises(SystemExit, match="another emulator"):
        R.g2_report(other, paths, em, 1, levels, grid_n=4, whole_n=4)
    out = capsys.readouterr().out
    assert "2 pairs to do" in out and "0 pairs to do (2 already" in out and "1 pairs to do (1 already" in out
