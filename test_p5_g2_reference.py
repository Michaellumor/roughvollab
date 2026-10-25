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
    assert gc["passed"] == all(gc[k] <= gc["tol"][k] for k in ("d_mean", "d_c", "d_mean_B", "d_q05", "d_q95"))   # D81's rule
    lo_e, hi_e = R.extended_support(rec["support_lo"], rec["support_hi"])
    assert np.all(lo_e >= LO) and np.all(hi_e <= HI) and np.all(lo_e <= rec["support_lo"]) and np.all(hi_e >= rec["support_hi"])
    assert gc["grid_lo"] == [float(v) for v in lo_e] and axes[0][0] == lo_e[0] + 0.5 / 12 * (hi_e[0] - lo_e[0])
    wb = R.whole_box_check(rec, y, s, em, axes[3], n=6)
    assert wb["n_cells"] == 6 ** 3 * 12 and wb["n_per_dim"] == [6, 6, 6, 12]
    assert wb["d_mean"] >= 0 and isinstance(wb["passed"], bool) and wb["mcmc_c_A"] == rec["c_A"]   # mechanics: coarse grid
    # the support extension clips at the box
    lo_c, hi_c = R.extended_support(LO + 0.001, HI - 0.001)
    assert np.array_equal(lo_c, LO) and np.array_equal(hi_c, HI)


def test_draws_surfaces_and_the_sealed_set(tmp_path, monkeypatch):
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
    monkeypatch.setitem(R.SETS, "g2", dict(R.SETS["g2"], report=str(tmp_path / "no_g2.json")))
    _no_sealed_draws(monkeypatch)                                                     # a draw would fail the test (D82)
    with pytest.raises(SystemExit, match="G2 has not passed"):
        R.main(["--test", "--emulator", "no_such_file.npz"])                          # refused before any load (D81)
    with pytest.raises(SystemExit, match=r"--emulator no_such_file\.npz is not G"):
        R.main(["--pilot", "--emulator", "no_such_file.npz"])                         # G1's build is committed (D88): not G̃, refused before any load
    monkeypatch.setattr(E, "paths", lambda stage="base": {k: str(tmp_path / f"no_g1_{stage}.{k}") for k in ("jsonl", "npz", "csv", "report")})
    with pytest.raises(SystemExit, match="no emulator"):
        R.main(["--pilot", "--emulator", "no_such_file.npz"])                         # with no G1 build (an empty folder), no emulator
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


# ---- D81: the coverage covariate, the weighted effective size, G2's added tolerance, fixed points, retry, go ----
def test_coverage_covariate_and_the_weighted_effective_size():
    """Line 99 and line 68 (D81): the mass each prior puts on the 90% interval of H under the
    exact posterior under that prior; a pair well covered when both reach 0.001; reweighting
    accepted when (Kish / pooled steps) × ESS of H reaches the floor, else a direct run."""
    # the floor acts on (Kish / steps) × ESS_H, not on Kish's figure alone: a floor between the two
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    kw = dict(burn=2000, sample=6000, rhat_max=1.1, ess_min=100.0)
    rec0, _ = R.reference_pair(y, s, em, 0, 2, kish_min=1.0, **kw)
    n_pooled, ess_H = 4 * 6000, rec0["runs"]["uniform"][-1]["ess_H"]
    neff_A = rec0["kish"]["A"] / n_pooled * ess_H
    assert neff_A < rec0["kish"]["A"]                                                # autocorrelated chains: fewer
    floor = 0.5 * (neff_A + rec0["kish"]["A"])                                       # Kish above it, the effective size below
    rec1, st1 = R.reference_pair(y, s, em, 0, 2, kish_min=floor, **kw)
    assert rec1["kish"]["A"] >= floor and "A" in rec1["direct"] and rec1["under"]["A"]["source"] == "direct"
    for which in ("A", "B"):
        assert abs(rec0["neff"][which] - rec0["kish"][which] / n_pooled * ess_H) < 1e-9 * rec0["kish"][which]
    assert "A" in st1 and rec1["coverage"] is not None and isinstance(rec1["well_covered"], bool)
    assert rec1["coverage"] == R.coverage(rec1["under"])[0]                          # the record's covariate (D81)
    assert {"q05_H", "q95_H"} <= set(rec1["under"]["B"])                             # the 90% interval under B, reported
    for which in ("A", "B"):
        a, b = R.PRIOR_AB[which]
        for lo, hi in ((0.05, 0.15), (0.30, 0.45), (0.40, 0.47), (0.02, 0.06)):
            exact = _beta.cdf((hi - 0.02) / 0.46, a, b) - _beta.cdf((lo - 0.02) / 0.46, a, b)
            assert abs(R.coverage_mass(lo, hi, which) - exact) <= 1e-12 + 1e-9 * exact
    assert R.coverage_mass(0.0, 1.0, "A") == 1.0 and R.coverage_mass(0.3, 0.2, "A") == 0.0
    tiny = R.coverage_mass(0.44, 0.46, "A")                              # far in A's upper tail: its digits kept
    assert 0 < tiny < 1e-6 and abs(tiny - (_beta.sf(0.42 / 0.46, 2.5, 9) - _beta.sf(0.44 / 0.46, 2.5, 9))) < 1e-18
    under = {"A": {"q05_H": 0.10, "q95_H": 0.14}, "B": {"q05_H": 0.10, "q95_H": 0.14}}
    cov, wc = R.coverage(under)
    assert cov["A"] > 0.2 and cov["B"] < 0.001 and wc is False                     # low H: B puts almost nothing there
    cov2, wc2 = R.coverage({"A": {"q05_H": 0.17, "q95_H": 0.23}, "B": {"q05_H": 0.20, "q95_H": 0.26}})
    assert wc2 is True and min(cov2.values()) >= 0.001


def test_g2_grid_adds_the_mean_under_B_and_the_quantiles_under_A():
    """Line 143 (D81): the grid gives the posterior under B and A's 5% and 95% quantiles from
    the same likelihood evaluations; the tolerance adds the mean under B (by the pair's route)
    within 0.001 and A's quantiles within 0.002."""
    assert (R.TOL_MEAN, R.TOL_C, R.TOL_Q) == (0.001, 0.01, 0.002)
    # quantiles on a cell-centred grid, the mass spread uniformly within each cell: exact for a uniform law
    axis = R.cell_centres(0.1, 0.3, 10)
    q05, q95 = R.grid_quantiles(axis, axis, np.ones(10))
    assert abs(q05 - 0.11) < 1e-12 and abs(q95 - 0.29) < 1e-12
    em = LinearEm()
    s = 0.003
    y = toy_surface(em, s=s)
    mean, cov = em.posterior(y, s)
    rec, stores = R.reference_pair(y, s, em, 0, 2, burn=2000, sample=6000, kish_min=100.0, rhat_max=1.1, ess_min=100.0)
    gc, axes = R.grid_check(rec, y, s, em, n=32)
    draws = np.random.default_rng(21).multivariate_normal(mean, cov, size=400000)
    wA, wB = np.exp(R.prior_logpdf(draws[:, 0], "A")), np.exp(R.prior_logpdf(draws[:, 0], "B"))
    assert abs(gc["B"]["mean_H"] - R.wmean(draws[:, 0], wB)) < 2e-4                  # the grid under B, against exact draws
    assert abs(gc["A"]["q05_H"] - R.wquantile(draws[:, 0], wA, 0.05)) < 1e-3
    assert abs(gc["A"]["q95_H"] - R.wquantile(draws[:, 0], wA, 0.95)) < 1e-3
    assert gc["route_B"] == rec["under"]["B"]["source"] and gc["mcmc_mean_H_B"] == rec["under"]["B"]["mean_H"]
    assert gc["passed"] == all(gc[k] <= gc["tol"][k] for k in ("d_mean", "d_c", "d_mean_B", "d_q05", "d_q95"))
    # a quantile off by more than its tolerance fails the pair although its mean and c match
    off = json.loads(json.dumps(rec))
    over = gc["tol"]["d_q05"] + 0.0005
    off["under"]["A"]["q05_H"] = gc["A"]["q05_H"] + over
    off["under"]["A"]["mean_H"], off["c_A"] = gc["A"]["mean_H"], gc["c_A"]
    off["under"]["B"]["mean_H"], off["under"]["A"]["q95_H"] = gc["B"]["mean_H"], gc["A"]["q95_H"]
    bad = R._compare(off, gc, R.TOL_MEAN, R.TOL_C, R.TOL_Q)
    assert bad["d_mean"] < 1e-12 and bad["d_c"] < 1e-12 and abs(bad["d_q05"] - over) < 1e-12 and bad["passed"] is False
    off["under"]["A"]["q05_H"] = gc["A"]["q05_H"]
    off["under"]["B"]["mean_H"] = gc["B"]["mean_H"] + gc["tol"]["d_mean_B"] + 0.0005    # the mean under B off
    assert R._compare(off, gc, R.TOL_MEAN, R.TOL_C, R.TOL_Q)["passed"] is False
    off["under"]["B"]["mean_H"] = gc["B"]["mean_H"]
    assert R._compare(off, gc, R.TOL_MEAN, R.TOL_C, R.TOL_Q)["passed"] is True
    wb = R.whole_box_check(rec, y, s, em, axes[3], n=6)                              # the second grid carries the same checks
    assert {"d_mean_B", "d_q05", "d_q95"} <= set(wb) and "B" in wb
    assert wb["k_mc"] == 4.0 and wb["mc_se"] == gc["mc_se"] and wb["n_eff"] == gc["n_eff"]   # and the same rule


def test_g2_pilot_set_fixed_points_retry_and_go_no_go(tmp_path, monkeypatch):
    """Line 143 (D81): 18 pilot surfaces (4 uniform points and the two fixed ones); one retry
    with doubled lengths on fresh surfaces; the go or no-go on the projected cost; the later
    runs only after G2 has passed with a go."""
    assert R.FIXED_POINTS == ((0.04, 0.20, -0.80, 0.145), (0.46, 0.20, -0.80, 0.145))
    assert (R.SEED_G2_FIXED_NOISE, R.SEED_G2R_POINTS, R.SEED_G2R_NOISE, R.SEED_G2R_FIXED_NOISE) == (139, 131, 137, 149)
    used = {20261002, 20261022, 7, 61, 62, 73, 79, 83, 89, 97, 101, 107, 109, 113, 127}
    assert {139, 131, 137, 149}.isdisjoint(used) and R.BUDGET_CORE_H == 1344.0 == 2 * 7 * 24 * 4
    centre = (LO + HI) / 2
    assert np.allclose(np.array(R.FIXED_POINTS)[:, 1:], centre[1:])                  # ν, ρ, ξ₀ at the box's centre
    pts, eps, seeds = R.g2_surfaces()
    assert pts.shape == (6, 4) and eps.shape == (6, 3, 35)
    assert np.array_equal(pts[:4], R.pilot_points(79, 4)) and np.array_equal(pts[4:], np.array(R.FIXED_POINTS))
    assert np.array_equal(eps[:4], R.noise_draws(89, 4)) and np.array_equal(eps[4:], R.noise_draws(139, 2))
    assert seeds == {"points": 79, "noise": 89, "fixed_points": [list(p) for p in R.FIXED_POINTS], "fixed_noise": 139}
    ptr, epr, sr = R.g2_surfaces(retry=True)
    assert np.array_equal(ptr[:4], R.pilot_points(131, 4)) and np.array_equal(ptr[4:], pts[4:])
    assert np.array_equal(epr[:4], R.noise_draws(137, 4)) and np.array_equal(epr[4:], R.noise_draws(149, 2))
    assert len(R.make_jobs("g2", pts, eps, LinearEm())) == 18
    # the projection: the median seconds per run of the lengths in force, times the mean runs per pair, times the pairs
    att = [[{"burn": 10, "sample": 40, "seconds": 2.0}],
           [{"burn": 10, "sample": 40, "seconds": 3.0}, {"burn": 20, "sample": 80, "seconds": 8.0}],   # a repeat: 2 units
           [{"burn": 10, "sample": 40, "seconds": 2.0}, {"burn": 10, "sample": 40, "seconds": 2.5}]]   # a direct run
    pr = R.project_cost(att, 1200, 50)
    assert pr["median_unit_s"] == 2.5 and abs(pr["mean_units"] - (1 + 3 + 2) / 3) < 1e-12
    assert abs(pr["projected_core_hours"] - 1200 * 2.5 * 2.0 / 3600) < 1e-12
    # G2's status from its reports
    first, retry = str(tmp_path / "g2.json"), str(tmp_path / "g2r.json")

    def rep(path, passed, go, attempt, burn=10000, sample=40000, hours=100.0):
        Path(path).write_text(json.dumps({"gate": "G2", "passed": passed, "go": go, "attempt": attempt,
                                          "lengths": {"burn": burn, "sample": sample}, "budget_core_hours": 1344.0,
                                          "projection": {"projected_core_hours": hours}}), encoding="utf-8")
    st = R.g2_status(first, retry)
    assert not st["passed"] and not st["p5_stops"] and "not been run" in st["reason"]
    rep(first, True, True, 1)
    st = R.g2_status(first, retry)
    assert st["passed"] and (st["burn"], st["sample"]) == (10000, 40000) and st["attempt"] == 1
    rep(first, True, False, 1, hours=2000.0)
    assert R.g2_status(first, retry)["p5_stops"] and not R.g2_status(first, retry)["passed"]   # no go: P5 stops
    rep(first, False, True, 1)
    st = R.g2_status(first, retry)
    assert st["retry_due"] and not st["passed"] and not st["p5_stops"]
    rep(retry, True, True, 2, 20000, 80000)
    st = R.g2_status(first, retry)
    assert st["passed"] and (st["burn"], st["sample"]) == (20000, 80000) and st["attempt"] == 2   # doubled for every later run
    rep(retry, False, True, 2, 20000, 80000)
    st = R.g2_status(first, retry)
    assert st["p5_stops"] and not st["passed"] and "retry" in st["reason"]
    with pytest.raises(SystemExit, match="G2 has not passed"):
        R.require_g2(first, retry)
    # the entry point: the retry only after a failed first attempt; the later runs only after G2
    monkeypatch.setitem(R.SETS, "g2", dict(R.SETS["g2"], report=first))
    monkeypatch.setitem(R.SETS, "g2r", dict(R.SETS["g2r"], report=retry))
    rep(first, True, True, 1)
    with pytest.raises(SystemExit, match="there is no retry"):
        R.main(["--pilot-retry", "--emulator", "no_such_file.npz"])
    Path(first).unlink()
    with pytest.raises(SystemExit, match="no first report"):
        R.main(["--pilot-retry", "--emulator", "no_such_file.npz"])
    with pytest.raises(SystemExit, match="G2 has not passed"):
        R.main(["--g3-pilot", "--emulator", "no_such_file.npz"])
    rep(first, True, True, 1)
    _no_sealed_draws(monkeypatch)                                                    # a draw would fail the test (D82)
    monkeypatch.setattr(E, "paths", lambda stage="base": {k: str(tmp_path / f"no_g1_{stage}.{k}") for k in ("jsonl", "npz", "csv", "report")})
    with pytest.raises(SystemExit, match="no emulator"):                             # past G2's gate, the emulator is needed
        R.main(["--test", "--emulator", "no_such_file.npz"])


def _no_sealed_draws(monkeypatch):
    """The sealed test set's drawing functions replaced by ones that fail if called, so that
    a refusal is seen to come before any draw (D82)."""
    def fail(*a, **k):
        raise AssertionError("a draw from the sealed test set was attempted")
    orig = R.noise_draws
    monkeypatch.setattr(R, "test_points", fail)
    monkeypatch.setattr(R, "test_noise", fail)
    monkeypatch.setattr(R, "noise_draws",
                        lambda seed, n, *a, **k: fail() if int(seed) == R.SEED_TEST_NOISE else orig(seed, n, *a, **k))


class _NoDraw:
    """Stands in for numpy's generator where a test must not draw from a sealed stream: it
    records the seed and the call and returns zeros of the shape asked for."""

    def __init__(self, log, seed):
        self.log, self.seed = log, seed

    def uniform(self, lo, hi, size):
        self.log.append((self.seed, "uniform", tuple(np.atleast_1d(lo)), tuple(np.atleast_1d(hi)), tuple(size)))
        return np.zeros(size)

    def standard_normal(self, shape):
        self.log.append((self.seed, "standard_normal", tuple(shape)))
        return np.zeros(shape)


def test_the_test_set_is_fixed_and_the_entry_point_guards_the_versions(monkeypatch):
    """Line 93 (D81): N = 400 θ, the first 400 rows of default_rng(20261002).uniform over the
    box; noise default_rng(20261022).standard_normal((400, 3, 35)); the defaults of --test.
    The definitions are checked without drawing from the sealed streams (line 95): numpy's
    generator is replaced by a recorder; the head property of a uniform draw is checked on a
    pilot seed in test_draws_surfaces_and_the_sealed_set. Run from the command line, the
    script refuses other numpy and scipy versions unless --allow-other-versions, recorded
    in the header."""
    assert (R.N_TEST, R.SEED_TEST, R.SEED_TEST_NOISE, R.N_TEST_PAIRS) == (400, 20261002, 20261022, 1200)
    log = []
    with monkeypatch.context() as m:
        m.setattr(np.random, "default_rng", lambda seed=None: _NoDraw(log, seed))
        th, ep = R.test_points(), R.test_noise()
        R.test_points(600)
    assert log == [(20261002, "uniform", tuple(LO), tuple(HI), (400, 4)), (20261022, "standard_normal", (400, 3, 35)),
                   (20261002, "uniform", tuple(LO), tuple(HI), (600, 4))]
    assert th.shape == (400, 4) and ep.shape == (400, 3, 35) and not th.any() and not ep.any()   # the recorder's zeros
    import inspect
    src = inspect.getsource(R.main)
    assert "default=N_TEST" in src and "default=SEED_TEST_NOISE" in src
    assert "main(check_versions=True)" in inspect.getsource(R).split("__main__")[-1]
    monkeypatch.setattr(E, "CORE_VERSIONS", {"numpy": "0.0.0", "scipy": "0.0.0"})
    R.main([])                                                                      # a call: nothing refused
    with pytest.raises(SystemExit, match="refused: numpy"):
        R.main([], check_versions=True)
    R.main(["--allow-other-versions"], check_versions=True)
    h = R.header("test", 1, "no_such_file.npz", 400, {"points": 20261002, "noise": 20261022},
                 versions=E.version_check(E.CORE_VERSIONS, allow=True, check=True))
    assert h["versions"]["override"] is True and h["scipy"] == h["versions"]["scipy"] and h["tol_q"] == 0.002


def test_the_retry_and_the_hand_off_of_g2s_lengths(tmp_path, monkeypatch, toy_emulator):
    """Line 143 (D81). On a toy pair: every launch line records the versions, an override
    included, and the report takes its lengths from the record and refuses others. Through
    the entry point, with the runs replaced by recorders (nothing sampled, and nothing drawn
    from the sealed streams): --pilot-retry runs the set g2r on the retry's fresh surfaces
    at doubled lengths and reports attempt 2; once G2 has passed, its lengths reach
    --g3-pilot and --test, whose defaults are N = 400 and the noise seed 20261022."""
    paths = {"jsonl": str(tmp_path / "r.jsonl"), "samples": str(tmp_path / "r.npz"), "report": str(tmp_path / "r.json")}
    kw = {"burn": 300, "sample": 900, "kish_min": 50.0, "rhat_max": 1.5, "ess_min": 20.0}
    over = E.version_check({"numpy": "0.0.0", "scipy": "0.0.0"}, allow=True, check=True)
    jobs = R.run_set("g2", np.array([CENTRE]), R.noise_draws(89, 1), toy_emulator, 1, paths, "no_such_file.npz",
                     {"points": 79, "noise": 89}, (0.003,), kw, versions=over)
    R.run_set("g2", np.array([CENTRE]), R.noise_draws(89, 1), toy_emulator, 1, paths, "no_such_file.npz",
              {"points": 79, "noise": 89}, (0.003,), kw)                                   # a resume, without the override
    hdr, launches, tasks, _ = R.load_references(paths["jsonl"], paths["samples"])
    assert [l["versions"]["override"] for l in launches] == [True, False] and hdr["versions"]["override"] is True
    with pytest.raises(SystemExit, match="made at lengths"):
        R.g2_report(jobs, paths, toy_emulator, 1, (0.003,), grid_n=4, whole_n=4, lengths=(10000, 40000))
    rep = R.g2_report(jobs, paths, toy_emulator, 1, (0.003,), grid_n=4, whole_n=4)
    assert rep["lengths"] == {"burn": 300, "sample": 900} and rep["projection"]["base_steps"] == 1200
    assert set(rep["versions"]) >= {"numpy", "scipy", "override"}
    # the entry point, on recorders
    first, retry = str(tmp_path / "g2.json"), str(tmp_path / "g2r.json")
    monkeypatch.setitem(R.SETS, "g2", dict(R.SETS["g2"], report=first))
    monkeypatch.setitem(R.SETS, "g2r", dict(R.SETS["g2r"], report=retry))
    calls = {"run_set": [], "report": [], "draws": []}

    def fake_run_set(set_name, points, eps, em, pool, paths_, emulator_path, seeds, levels=R.NOISE_LEVELS,
                     task_kwargs=None, extra_header=None, versions=None):
        calls["run_set"].append({"set": set_name, "n": len(points), "seeds": seeds, "kw": dict(task_kwargs or {}),
                                 "extra": extra_header or {}})
        return ["job"]

    def fake_report(jobs_, paths_, em, pool, **k):
        calls["report"].append(k)
        return {"attempt": k["attempt"], "passed": False, "go": True, "n_reference_failures": 0,
                "projection": {"projected_core_hours": 1.0}, "pairs": []}
    monkeypatch.setattr(R, "get_emulator", lambda path=None: toy_emulator)
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: "no_such_file.npz")     # G̃ stands in (D82)
    monkeypatch.setattr(R, "run_set", fake_run_set)
    monkeypatch.setattr(R, "g2_report", fake_report)
    monkeypatch.setattr(R, "test_points", lambda n=R.N_TEST, box=R.BOX: calls["draws"].append(("theta", n)) or np.zeros((n, 4)))
    orig_noise = R.noise_draws

    def noise(seed, n, *a, **k):
        if seed == R.SEED_TEST_NOISE:                                                   # the sealed stream: recorded, not drawn
            calls["draws"].append(("noise", seed, n))
            return np.zeros((n, 3, 35))
        return orig_noise(seed, n, *a, **k)
    monkeypatch.setattr(R, "noise_draws", noise)
    Path(first).write_text(json.dumps({"gate": "G2", "passed": False, "go": True, "attempt": 1,
                                       "lengths": {"burn": 10000, "sample": 40000}, "budget_core_hours": 1344.0,
                                       "projection": {"projected_core_hours": 100.0}}), encoding="utf-8")
    R.main(["--pilot-retry", "--emulator", "no_such_file.npz"])
    rs, rp = calls["run_set"][-1], calls["report"][-1]
    assert rs["set"] == "g2r" and rs["n"] == 6 and (rs["kw"]["burn"], rs["kw"]["sample"]) == (20000, 80000)
    assert rs["seeds"]["points"] == 131 and rs["seeds"]["noise"] == 137 and rs["seeds"]["fixed_noise"] == 149
    assert rp["attempt"] == 2 and tuple(rp["lengths"]) == (20000, 80000)
    Path(retry).write_text(json.dumps({"gate": "G2", "passed": True, "go": True, "attempt": 2,
                                       "lengths": {"burn": 20000, "sample": 80000}, "budget_core_hours": 1344.0,
                                       "projection": {"projected_core_hours": 200.0}}), encoding="utf-8")
    R.main(["--g3-pilot", "--emulator", "no_such_file.npz"])
    g3 = calls["run_set"][-1]
    assert g3["set"] == "g3" and (g3["kw"]["burn"], g3["kw"]["sample"]) == (20000, 80000) and g3["extra"]["g2"]["attempt"] == 2
    R.main(["--test", "--emulator", "no_such_file.npz"])
    t = calls["run_set"][-1]
    assert t["set"] == "test" and t["n"] == 400 and t["seeds"] == {"points": 20261002, "noise": 20261022}
    assert (t["kw"]["burn"], t["kw"]["sample"]) == (20000, 80000) and t["extra"]["n_test"] == 400
    assert calls["draws"] == [("theta", 400), ("noise", 20261022, 400)]                 # recorded, never drawn


def test_g2_tolerances_are_at_least_four_monte_carlo_standard_errors():
    """Line 143 (D81, the author's decision of 22 October 2026): each G2 tolerance is the
    larger of its absolute value and 4 Monte Carlo standard errors of the MCMC figure, from
    the effective sample size of the run it came from ((Kish / pooled steps) × ESS of H for a
    reweighted figure, the accepted direct run's ESS of H for a direct one), by line 89's
    Gaussian reading: sd / √n for a mean, (1 − c)·√(2 / (n − 1)) for c, √(q(1 − q) / n) /
    (φ(z_q) / sd) for a quantile."""
    from statistics import NormalDist
    assert R.TOL_MC_K == 4.0
    phi = NormalDist().pdf(NormalDist().inv_cdf(0.05))                                    # 0.1031
    sdA, sdB = 0.017, 0.020
    rec = {"c_A": 0.90, "neff": {"A": 1000.0, "B": 1.0e5},
           "runs": {"uniform": [{"ess_H": 9000.0}], "B": [{"ess_H": 1500.0}, {"ess_H": 2500.0}]},
           "under": {"A": {"source": "reweighted", "mean_H": 0.200, "var_H": sdA ** 2, "q05_H": 0.172, "q95_H": 0.228},
                     "B": {"source": "direct", "mean_H": 0.250, "var_H": sdB ** 2}}}
    se = {"d_mean": sdA / np.sqrt(1000.0), "d_c": 0.10 * np.sqrt(2.0 / 999.0), "d_mean_B": sdB / np.sqrt(2500.0),
          "d_q05": np.sqrt(0.05 * 0.95 / 1000.0) / (phi / sdA)}
    se["d_q95"] = se["d_q05"]
    absolute = {"d_mean": 0.001, "d_c": 0.01, "d_mean_B": 0.001, "d_q05": 0.002, "d_q95": 0.002}

    def grid(dm=0.0, dc=0.0, dmB=0.0, dq=0.0):
        return {"A": {"mean_H": 0.200 + dm, "q05_H": 0.172 + dq, "q95_H": 0.228}, "B": {"mean_H": 0.250 + dmB},
                "c_A": 0.90 + dc}

    def cmp(r, **g):
        return R._compare(r, grid(**g), 0.001, 0.01, 0.002)
    out = cmp(rec)
    for k in absolute:
        assert abs(out["mc_se"][k] - se[k]) < 1e-15 and abs(out["tol"][k] - max(absolute[k], 4 * se[k])) < 1e-15
    assert out["n_eff"] == {"A": 1000.0, "B": 2500.0} and out["passed"] is True and out["k_mc"] == 4.0
    assert out["tol"]["d_mean"] > 0.001 and out["tol"]["d_c"] > 0.01 and out["tol"]["d_q05"] > 0.002   # 4 SE govern here
    assert cmp(rec, dm=0.0018)["passed"] is True and cmp(rec, dm=0.0025)["passed"] is False    # above 0.001, within 4 SE
    assert cmp(rec, dc=0.015)["passed"] is True and cmp(rec, dc=0.019)["passed"] is False
    assert cmp(rec, dq=0.004)["passed"] is True and cmp(rec, dq=0.005)["passed"] is False
    assert cmp(rec, dmB=0.0014)["passed"] is True                                         # B direct: ESS 2,500, 4 SE 0.0016
    reweighted_B = json.loads(json.dumps(rec))
    reweighted_B["under"]["B"]["source"] = "reweighted"                                  # B reweighted: n_eff 100,000
    assert cmp(reweighted_B, dmB=0.0014)["passed"] is False and cmp(reweighted_B)["tol"]["d_mean_B"] == 0.001
    big = json.loads(json.dumps(rec))
    big["neff"]["A"] = 1.0e6                                                              # a large n_eff: the absolute values govern
    assert all(cmp(big)["tol"][k] == absolute[k] for k in ("d_mean", "d_c", "d_q05", "d_q95"))
    assert cmp(big, dm=0.0018)["passed"] is False


def test_the_test_set_and_the_emulator_are_fixed_before_any_draw_or_run(tmp_path, monkeypatch, toy_emulator):
    """Lines 93 and 142 (D82): --test refuses an N other than 400 or a noise seed other than
    20261022 before any draw; from the command line the emulator is G̃ (require_g1) and an
    --emulator naming another file is refused before any run; every reference record and
    G2's report carry the emulator's SHA-256, a resume with a changed emulator file is
    refused, G2's status hands the hash on, and no G2 report is written over."""
    _no_sealed_draws(monkeypatch)
    with pytest.raises(SystemExit, match="the test set is N = 400 with noise seed 20261022"):
        R.main(["--test", "--n", "300"])
    with pytest.raises(SystemExit, match="the test set is N = 400 with noise seed 20261022"):
        R.main(["--test", "--noise-seed", "7"])
    em_file = tmp_path / "em.npz"
    np.savez(em_file, stage=np.array("base"), commit=np.array("x"), values=np.zeros((2, 1, 1, 2, 3)))
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: str(em_file))
    with pytest.raises(SystemExit, match="is not G̃"):
        R.main(["--pilot", "--emulator", str(tmp_path / "other.npz")])
    paths = {"jsonl": str(tmp_path / "r.jsonl"), "samples": str(tmp_path / "r.npz"), "report": str(tmp_path / "r.json")}
    kw = {"burn": 300, "sample": 900, "kish_min": 50.0, "rhat_max": 1.5, "ess_min": 20.0}
    args = (np.array([CENTRE]), R.noise_draws(89, 1), toy_emulator, 1, paths, str(em_file), {"points": 79, "noise": 89},
            (0.003,), kw)
    jobs = R.run_set("g2", *args)
    hdr = R.load_references(paths["jsonl"])[0]
    assert hdr["emulator_sha256"] == E.sha256_file(str(em_file)) and hdr["emulator_sha256"] is not None
    rep = R.g2_report(jobs, paths, toy_emulator, 1, (0.003,), grid_n=4, whole_n=4)
    assert rep["emulator"]["sha256"] == hdr["emulator_sha256"]
    with pytest.raises(SystemExit, match="report refused: .* exists; a re-run needs the old file moved aside"):
        R.g2_report(jobs, paths, toy_emulator, 1, (0.003,), grid_n=4, whole_n=4)
    first = tmp_path / "g2.json"
    first.write_text(json.dumps({"gate": "G2", "passed": True, "go": True, "attempt": 1, "lengths": {"burn": 10, "sample": 40},
                                 "budget_core_hours": 1344.0, "projection": {"projected_core_hours": 1.0},
                                 "emulator": rep["emulator"]}), encoding="utf-8")
    assert R.g2_status(str(first), str(tmp_path / "no_retry.json"))["emulator_sha256"] == hdr["emulator_sha256"]
    np.savez(em_file, stage=np.array("base"), commit=np.array("x"), values=np.ones((2, 1, 1, 2, 3)))   # another file
    with pytest.raises(SystemExit, match="launch refused: .* emulator_sha256"):
        R.run_set("g2", *args)
