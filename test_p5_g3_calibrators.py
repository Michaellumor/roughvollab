"""
test_p5_g3_calibrators.py — mechanics tests for the P5 gate G3 calibrators script. The
numpy-only tests run in the core (torch-free) environment; the training tests need torch
and SKIP without it (run them in .venv-p5, see requirements-p5.txt), as Layer 3's do. They
guard the MACHINERY on toy emulators in memory: the literals written into the draft
(lines 72, 73, 75, 144); the draws and their nesting; the models, losses and the batch
construction; the training loop and its stop statuses; the outputs in box units; the ratio
estimator's posterior through the reference's sampler; G3's checks and ladder on a toy.
Nothing here trains on the real emulator, which is built at G1 after freezing.
"""
import copy
import json
import inspect
from pathlib import Path

import numpy as np
import pytest

import p5_g3_calibrators as C
import p5_g2_reference as R
import p5_g1_emulator as E

_needs_torch = pytest.mark.skipif(not C._HAS_TORCH, reason="torch not in this env (run in .venv-p5)")
LO, HI = E.box_arrays(E.BOX)
CENTRE = (LO + HI) / 2


class LinearEm:
    def __init__(self, seed=5, scale=0.03):
        rng = np.random.default_rng(seed)
        self.M = rng.normal(0.0, scale, size=(35, 4))
        self.a = 0.2 + rng.uniform(0, 0.05, size=35)
        self.box = E.BOX

    def evaluate_many(self, thetas):
        return self.a[None, :] + np.asarray(thetas, dtype=float).reshape(-1, 4) @ self.M.T


# ---- numpy only ----
def test_literals_are_the_written_ones():
    assert C.HIDDEN == (128, 128, 128) and C.N_MEMBERS == 5 and C.LR == 1e-3 and C.BETAS == (0.9, 0.999) and C.BATCH == 256
    assert (C.PATIENCE, C.MAX_EPOCHS, C.VAR_FLOOR, C.N_VALID) == (50, 1000, 1e-6, 4000)
    assert (C.SEED_TRAIN, C.SEED_VALID, C.SEED_NRE, C.TORCH_BASE) == (109, 113, 107, 109_000_000)
    assert C.LADDER == (10000, 20000, 40000, 80000, 160000) and C.DRAW_N == 320000 == C.BUDGET_SURFACES
    assert C.BUDGET_WALL_S == 4 * 3600 and (C.SLOPE_MIN, C.STABILITY_TOL, C.N_SEEDS, C.RETRAIN_SEED) == (0.95, 0.002, 5, 6)
    assert C.PRIOR_INDEX == {"A": 1, "B": 2} and C.LEVEL_INDEX == {0.001: 1, 0.003: 2, 0.005: 3} and C.Z90 == 1.645
    assert C.PRIOR_AB is R.PRIOR_AB and C.CALIBRATORS == ("ensemble", "nre")
    seeds = {C.torch_seed(p, j, q, m) for p in (1, 2) for j in (1, 2, 3) for q in range(1, 7) for m in range(0, 6)}
    assert len(seeds) == 2 * 3 * 6 * 6 and C.torch_seed(1, 1, 1, 1) == 109_011_011 and C.torch_seed(2, 3, 5, 0) == 109_023_050
    assert all(2 * n == C.LADDER[k + 1] for k, n in enumerate(C.LADDER[:-1])) and 2 * C.LADDER[-1] == C.DRAW_N
    assert {C.SEED_TRAIN, C.SEED_VALID, C.SEED_NRE}.isdisjoint({20261002, 7, 61, 62, 73, 79, 83, 89, 97, 101})


def test_draws_layout_nesting_and_standardiser():
    th, ep = C.draw_block([109, 1, 2, 1], 500, "A")
    assert th.shape == (500, 4) and ep.shape == (500, 35)
    assert np.all(th >= LO) and np.all(th <= HI)
    rng = np.random.default_rng([109, 1, 2, 1])
    a, b = R.PRIOR_AB["A"]
    H = C.H_LO + C.H_W * rng.beta(a, b, size=500)                         # the written order: H, then the rest, then ε
    assert C.H_LO == 0.02 and abs(C.H_W - 0.46) < 1e-15
    other = rng.uniform(LO[1:], HI[1:], size=(500, 3))
    eps = rng.standard_normal((500, 35))
    assert np.array_equal(th[:, 0], H) and np.array_equal(th[:, 1:], other) and np.array_equal(ep, eps)
    th2, ep2 = C.training_block("A", 2, 1, n_block=500)                   # the block; a run at n takes its head
    assert np.array_equal(th2, th) and np.array_equal(ep2, ep)
    thB, _ = C.training_block("B", 2, 1, n_block=500)
    assert thB[:, 0].mean() > th[:, 0].mean()                             # B's mass sits higher in H
    vt, ve = C.validation_block("A", 2, n=50)
    assert vt.shape == (50, 4) and np.array_equal(vt, C.draw_block([113, 1, 2], 50, "A")[0])
    em = LinearEm()
    Y = C.make_surfaces(th, ep, 0.003, em, chunk=128)
    assert np.allclose(Y, em.evaluate_many(th) + 0.003 * ep)
    std = C.Standardiser(Y[:200])
    Z = std(Y[:200])
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12) and np.allclose(Z.std(axis=0), 1, atol=1e-12)
    assert np.allclose(C.Standardiser.from_json(std.to_json())(Y[:5]), std(Y[:5]))
    assert list(C._batches(600, 256)) == [slice(0, 256), slice(256, 512), slice(512, 600)]     # the partial batch kept
    assert [s.stop - s.start for s in C._batches(4000, 256)][-1] == 160                     # the validation's final 160


def test_g3_checks_and_budget_on_hand_built_numbers():
    rows = [{"H_exact": 0.1 + 0.01 * k, "xi0_exact": 0.05 + 0.02 * k, "i": k} for k in range(8)]
    pred_n = [(r["H_exact"] + 0.004, 0.01 + 0.98 * r["xi0_exact"]) for r in rows]               # slope 0.98, |d| 0.004
    pred_2n = [(r["H_exact"] - 0.0055, 0.5 * r["xi0_exact"]) for r in rows]                     # |d| 0.0055
    c = C.g3_checks(pred_n, pred_2n, rows)
    assert abs(c["slope"] - 0.98) < 1e-12 and c["slope_ok"] and abs(c["median_abs_d_n"] - 0.004) < 1e-12
    assert abs(c["median_abs_d_2n"] - 0.0055) < 1e-12 and c["stable"] and c["passed"]        # |0.0055 − 0.004| ≤ 0.002
    pred_far = [(r["H_exact"] + 0.01, 0.5 * r["xi0_exact"]) for r in rows]
    c2 = C.g3_checks(pred_far, pred_2n, rows)
    assert not c2["slope_ok"] and c2["stable"] is False and not c2["passed"]                 # slope 0.5; |d| moves 0.0045
    pred_n[3] = None                                                                          # a failed pair is excluded
    c3 = C.g3_checks(pred_n, pred_2n, rows)
    assert c3["n_used_n"] == 7 and c3["n_used_2n"] == 8 and c3["slope_ok"]
    run = {"status": "ok", "seconds": 100.0}
    assert C.budget_ok(run, 320000) and not C.budget_ok(run, 320001)
    assert not C.budget_ok({"status": "over_budget", "seconds": 100.0}, 10000)
    assert not C.budget_ok({"status": "ok", "seconds": 4 * 3600 + 1}, 10000)
    assert C.run_label("nre", "B", 3, 2, 20000) == "nre_p2_j3_q2_n20000"
    assert "SEED_NRE" in inspect.getsource(C.nre_posterior) and "PRIOR_INDEX" in inspect.getsource(C.nre_posterior)


def test_pilot_surfaces_from_reference_records():
    refs = {}
    for i in range(3):
        for j in range(3):
            refs[f"g3_{i:04d}_{j}"] = {"set": "g3", "i": i, "j": j, "status": "ok" if (i, j) != (1, 2) else "reference_failure",
                                       "theta": [0.1, 0.2, -0.8, 0.1], "y": [0.2] * 35,
                                       "under": {"A": {"mean_H": 0.1 + i, "mean_xi0": 0.05}, "B": {"mean_H": 0.3, "mean_xi0": 0.06}}}
    rows, removed = C.pilot_surfaces(refs, "A", 3)                         # j = 3 is the record's j = 2
    assert [r["i"] for r in rows] == [0, 2] and removed == ["g3_0001_2"] and rows[1]["H_exact"] == 2.1
    rows_b, removed_b = C.pilot_surfaces(refs, "B", 1)
    assert len(rows_b) == 3 and removed_b == [] and rows_b[0]["xi0_exact"] == 0.06


# ---- torch ----
@_needs_torch
def test_models_losses_and_batch_construction():
    torch = C.torch
    member, nre = C.make_mlp(35, 4), C.make_mlp(39, 1)
    assert C.n_params(member) == 38148 and C.n_params(nre) == 38273                           # D68's counts
    assert [m.out_features for m in member if isinstance(m, torch.nn.Linear)] == [128, 128, 128, 4]
    assert sum(isinstance(m, torch.nn.ReLU) for m in member) == 3
    out = torch.tensor([[0.3, 0.6, 0.0, 0.0]])                                                 # softplus(0) = ln 2
    t = torch.tensor([[0.3, 0.5]])
    var = np.log(2.0) + 1e-6
    expect = 0.5 * (np.log(2 * np.pi * var) + 0.0 / var) + 0.5 * (np.log(2 * np.pi * var) + 0.01 / var)
    assert abs(C.member_loss(out, t).item() - expect) < 1e-5
    torch.manual_seed(0)
    xb, tb = torch.randn(4, 35), torch.rand(4, 4)
    loss = C.nre_loss(nre, xb, tb)
    joint = nre(torch.cat([xb, tb], 1)).squeeze(1)
    marg = nre(torch.cat([xb, torch.roll(tb, 1, 0)], 1)).squeeze(1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    expect = (bce(joint, torch.ones(4), reduction="sum") + bce(marg, torch.zeros(4), reduction="sum")) / 8
    assert abs(loss.item() - expect.item()) < 1e-6
    assert torch.equal(torch.roll(tb, 1, 0)[0], tb[3]) and torch.equal(torch.roll(tb, 1, 0)[1], tb[0])    # the cyclic shift


@_needs_torch
def test_training_loop_stops_and_statuses(monkeypatch):
    em = LinearEm()
    th, ep = C.training_block("A", 1, 1, n_block=600)
    Y = C.make_surfaces(th, ep, 0.003, em)
    std = C.Standardiser(Y)
    X, Tu = std(Y), R.theta_to_unit(th)
    vt, ve = C.validation_block("A", 1, n=100)
    Xv, Tv = std(C.make_surfaces(vt, ve, 0.003, em)), R.theta_to_unit(vt)
    r = C.train_net("member", X, Tu[:, [0, 3]], Xv, Tv[:, [0, 3]], C.torch_seed(1, 1, 1, 1), patience=3, max_epochs=40,
                    threads=1)
    assert r["status"] == "ok" and r["best_val"] == min(r["history"]) and r["epochs"] == len(r["history"]) <= 40
    assert r["history"][-1] >= r["best_val"] and r["history"][0] > r["best_val"]                 # it learnt something
    r2 = C.train_net("member", X, Tu[:, [0, 3]], Xv, Tv[:, [0, 3]], C.torch_seed(1, 1, 1, 1), patience=3, max_epochs=40,
                     threads=1)
    assert r2["history"] == r["history"]                                                          # the same seed, the same run
    r3 = C.train_net("nre", X, Tu, Xv, Tv, C.torch_seed(1, 1, 1, 0), patience=2, max_epochs=5, threads=1)
    assert r3["status"] == "ok" and r3["kind"] == "nre" and len(r3["history"]) <= 5
    r4 = C.train_net("member", X, Tu[:, [0, 3]], Xv, Tv[:, [0, 3]], 1, deadline=C.time.perf_counter() - 1, max_epochs=10,
                     threads=1)
    assert r4["status"] == "over_budget" and r4["epochs"] == 1
    bad = X.copy()
    bad[0, 0] = np.inf
    r5 = C.train_net("member", bad, Tu[:, [0, 3]], Xv, Tv[:, [0, 3]], 1, max_epochs=3, threads=1)
    assert r5["status"] == "nonfinite" and r5["epochs"] == 0 and r5["history"] == []
    badv = Xv.copy()
    badv[0, 0] = np.inf                                                                           # the validation path
    r6 = C.train_net("member", X, Tu[:, [0, 3]], badv, Tv[:, [0, 3]], 1, max_epochs=3, threads=1)
    assert r6["status"] == "nonfinite" and r6["history"] == [] and r6["best_val"] is None
    json.dumps({k: v for k, v in r6.items() if k != "net"}, allow_nan=False)                      # recordable


@_needs_torch
def test_calibrator_outputs_in_box_units_and_the_nre_posterior(tmp_path):
    em = LinearEm()
    th, ep = C.training_block("A", 1, 1, n_block=800)
    vt, ve = C.validation_block("A", 1, n=100)
    run = C.train_calibrator("ensemble", "A", 1, 1, 800, th, ep, 0.003, em, vt, ve, threads=1, patience=3, max_epochs=15)
    assert run["status"] == "ok" and len(run["nets"]) == 5 and run["label"] == "ensemble_p1_j1_q1_n800"
    assert [m["seed"] for m in run["members"]] == [C.torch_seed(1, 1, 1, m) for m in range(1, 6)]
    Y = C.make_surfaces(th[:7], ep[:7], 0.003, em)
    pr = C.ensemble_predict(run, Y)
    assert pr["H_hat"].shape == (7,) and np.all(pr["var_alea"] > 0) and np.all(pr["var_epi"] >= 0)
    assert np.allclose(pr["var_total"], pr["var_alea"] + pr["var_epi"])
    big = copy.deepcopy(run)                                                                      # a raw variance of 800
    with C.torch.no_grad():
        big["nets"][0][-1].bias[2] += 800.0
    assert np.all(np.isfinite(C.ensemble_predict(big, Y)["var_total"]))                           # softplus overflow-safe
    assert np.allclose(pr["hi"] - pr["lo"], 2 * 1.645 * np.sqrt(pr["var_total"]))
    # the mapping back to box units: recompute one member by hand
    torch = C.torch
    with torch.no_grad():
        outs = np.stack([net(torch.as_tensor(run["std"](Y), dtype=torch.float32)).numpy() for net in run["nets"]])
    H_hat = 0.02 + 0.46 * outs[:, :, 0].mean(axis=0)
    assert np.allclose(pr["H_hat"], H_hat) and np.allclose(pr["var_epi"], outs[:, :, 0].var(axis=0) * 0.46 ** 2)
    base = C.save_run(str(tmp_path), run)
    loaded = C.load_run(str(tmp_path), run["label"])
    pr2 = C.ensemble_predict(loaded, Y)
    assert np.allclose(pr2["H_hat"], pr["H_hat"]) and Path(base + ".json").exists() and loaded["status"] == "ok"
    # the ratio estimator's posterior: a hand-built logit whose posterior under A is a known Gaussian in H
    nre = C.train_calibrator("nre", "A", 1, 1, 400, th, ep, 0.003, em, vt, ve, threads=1, patience=2, max_epochs=3)
    assert nre["status"] == "ok" and len(nre["nets"]) == 1 and nre["members"][0]["seed"] == C.torch_seed(1, 1, 1, 0)
    mu, sd = 0.20, 0.01

    class Quadratic(torch.nn.Module):
        def forward(self, z):                                                  # logit = −(H_unit − mu_unit)² / (2 sd_unit²)
            Hu = z[:, 35]
            return (-((Hu - (mu - 0.02) / 0.46) ** 2) / (2 * (sd / 0.46) ** 2)).unsqueeze(1)
    nre["nets"] = [Quadratic()]
    po = C.nre_posterior(nre, Y[0], i=0, burn=1000, sample=3000, rhat_max=1.1, ess_min=50)
    assert po["status"] == "ok" and po["n_pooled"] == 4 * 3000 and po["lo"] < po["H_hat"] < po["hi"]
    wA = lambda h: np.exp(R.prior_logpdf(h, "A")) * np.exp(-(h - mu) ** 2 / (2 * sd ** 2))
    grid = np.linspace(0.02, 0.48, 200001)
    w = wA(grid)
    mean_A = np.sum(grid * w) / np.sum(w)
    ess = po["attempts"][-1]["ess"]["H"]                                                     # the accepted attempt's
    direct, _ = R._run_with_repeat(None, None, None, np.random.default_rng([107, 1, 1, 0, 1]), "uniform", 1000, 3000,
                                   E.BOX, 1.1, 50, target=C.nre_target(nre, Y[0]))
    assert R.unit_to_theta(direct["samples"].reshape(-1, 4))[:, 0].mean() == po["H_hat"]          # the generator's arguments
    assert abs(po["H_hat"] - mean_A) < 4 * sd / np.sqrt(ess) + 1e-9                     # the prior A tilts the mean down
    assert abs(po["xi0_hat"] - CENTRE[3]) < 0.02                                           # ξ₀ flat: uniform on its range
    po2 = C.nre_posterior(nre, Y[0], i=0, burn=1000, sample=3000, rhat_max=1.1, ess_min=50)
    assert po2["H_hat"] == po["H_hat"]                                                     # default_rng([107, 1, 1, 0, 1])


@_needs_torch
def test_toy_g3_ladder_end_to_end(tmp_path):
    em = LinearEm()
    refs = {}
    rng = np.random.default_rng(3)
    for i in range(3):
        theta = rng.uniform(LO, HI)
        for j in range(3):
            y = em.evaluate_many(theta)[0] + E.NOISE_LEVELS[j] * rng.standard_normal(35)
            refs[f"g3_{i:04d}_{j}"] = {"set": "g3", "task": f"g3_{i:04d}_{j}", "i": i, "j": j, "status": "ok",
                                       "theta": theta.tolist(), "y": y.tolist(),
                                       "under": {p: {"mean_H": float(theta[0]), "mean_xi0": float(theta[3])} for p in "AB"}}
    paths = {"jsonl": str(tmp_path / "ladder.jsonl"), "nets": str(tmp_path / "nets"), "report": str(tmp_path / "g3.json")}
    kw = dict(em=em, refs=refs, jsonl_path=paths["jsonl"], nets_dir=paths["nets"], report_path=paths["report"], threads=1,
              ladder=(64, 128), n_block=256, n_valid=40, levels=(0.003,), priors=("A",), calibrators=("ensemble", "nre"),
              train_kw={"patience": 2, "max_epochs": 3}, post_kw={"burn": 100, "sample": 200, "rhat_max": 5.0, "ess_min": 1.0})
    rep = C.run_g3(**kw)
    assert rep["gate"] == "G3" and set(rep["cells"]) == {"ensemble_A_j2", "nre_A_j2"} and rep["ladder"] == [64, 128]
    cell = rep["cells"]["ensemble_A_j2"]
    assert cell["n_pilot"] == 3 and cell["removed_references"] == [] and 1 <= len(cell["rungs"]) <= 2
    assert all(set(r) >= {"slope", "median_abs_d_n", "median_abs_d_2n", "passed", "budget_n", "budget_2n"} for r in cell["rungs"])
    assert (cell["n_star"] in (64, 128)) == cell["passed"] and rep["passed"] == (not rep["cells_not_passing"])
    hdr, launches, tasks = R.read_jsonl(paths["jsonl"])
    runs = sorted(k for k in tasks if k.startswith("ensemble"))
    assert runs[0].startswith("ensemble_p1_j2_q1_n") and all(Path(paths["nets"], k + ".pt").exists() for k in runs)
    assert hdr["ladder"] == [64, 128] and hdr["n_block"] == 256
    json.dumps(json.loads(Path(paths["report"]).read_text(encoding="utf-8")), allow_nan=False)
    Path(paths["report"]).rename(paths["report"] + ".aside")                               # a re-run moves it aside (D82)
    rep2 = C.run_g3(**kw)                                                                   # resume: every run reused
    assert len(R.read_jsonl(paths["jsonl"])[2]) == len(tasks) and rep2["cells"]["nre_A_j2"]["rungs"][0]["n"] == 64
    # the refusal without a passed G3
    with pytest.raises(SystemExit, match="G3 has not passed"):
        C.train_seeds(em, {"passed": False, "cells": {}}, jsonl_path=str(tmp_path / "t0.jsonl"), nets_dir=str(tmp_path / "n0"))


@_needs_torch
def test_seeds_scoring_and_the_retraining_rule(tmp_path, monkeypatch):
    """After a (forced) pass of G3: seeds 2 to 5 at n*, the one retraining from seed 6 on a
    non-finite loss, the second failure leaving its slot empty, and the scoring of a set."""
    em = LinearEm()
    refs = {}
    rng = np.random.default_rng(4)
    for i in range(3):
        theta = rng.uniform(LO, HI)
        y = em.evaluate_many(theta)[0] + 0.003 * rng.standard_normal(35)
        refs[f"g3_{i:04d}_1"] = {"set": "g3", "task": f"g3_{i:04d}_1", "i": i, "j": 1, "status": "ok",
                                 "theta": theta.tolist(), "y": y.tolist(),
                                 "under": {p: {"mean_H": float(theta[0]), "mean_xi0": float(theta[3])} for p in "AB"}}
    orig_checks = C.g3_checks
    monkeypatch.setattr(C, "g3_checks", lambda pn, p2, rows, **kw: orig_checks(pn, p2, rows, slope_min=-np.inf, tol=np.inf))
    paths = {"jsonl": str(tmp_path / "ladder.jsonl"), "nets": str(tmp_path / "nets"), "report": str(tmp_path / "g3.json")}
    tk = {"patience": 2, "max_epochs": 2}
    pk = {"burn": 50, "sample": 100, "rhat_max": 5.0, "ess_min": 1.0}
    rep = C.run_g3(em, refs, paths["jsonl"], paths["nets"], paths["report"], threads=1, ladder=(32,), n_block=64, n_valid=16,
                   levels=(0.003,), priors=("A",), train_kw=tk, post_kw=pk)
    assert rep["passed"] and {c["n_star"] for c in rep["cells"].values()} == {32}
    # seeds 2 to 4 with the member of seed 3 forced non-finite, so 3 is replaced by 6; then 4 forced too: empty
    orig_train = C.train_calibrator

    def flaky(kind, prior, j, q, n, *args, **kw):
        run = orig_train(kind, prior, j, q, n, *args, **kw)
        if kind == "ensemble" and q in (3, 4):
            run["status"] = "nonfinite"
        return run
    monkeypatch.setattr(C, "train_calibrator", flaky)
    trained = C.train_seeds(em, rep, jsonl_path=str(tmp_path / "train.jsonl"), nets_dir=str(tmp_path / "nets2"),
                            nets_g3=paths["nets"], threads=1, n_block=64, n_valid=16, seeds=(2, 3, 4), train_kw=tk)
    ens, nre = trained["ensemble_A_j2"], trained["nre_A_j2"]
    assert ens["n_star"] == 32 and set(ens["seeds"]) == {1, 2, 3} and ens["seeds"][3].endswith("_q6_n32") and ens["empty_slots"] == [4]
    assert set(nre["seeds"]) == {1, 2, 3, 4} and nre["empty_slots"] == []
    lines = [json.loads(l) for l in Path(tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    retrains = [l for l in lines if l["kind"] == "retrain"]
    assert [(r["failed_q"], r["replacement_q"]) for r in retrains] == [(3, 6), (4, None)]
    assert sum(1 for l in lines if l["kind"] == "task" and "_q6_" in l["task"]) == 1              # seed 6 trained once
    n = C.score_set("g3", refs, em, trained, nets_dir=str(tmp_path / "nets2"), nets_g3=paths["nets"],
                    out_path=str(tmp_path / "scores.jsonl"), threads=1, post_kw=pk)
    assert n == 3 * 3 + 4 * 3                                                                 # ensemble 3 seeds, nre 4, 3 pairs
    rows = [json.loads(l) for l in Path(tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "header" and rows[0]["threads"] == 1
    rows = rows[1:]
    assert {r["calibrator"] for r in rows} == {"ensemble", "nre"} and all("H_hat" in r and "lo" in r for r in rows)
    assert {r["q"] for r in rows if r["calibrator"] == "ensemble"} == {1, 2, 3} and all("_q6_" in r["run"] for r in rows if r["q"] == 3 and r["calibrator"] == "ensemble")
    assert C.score_set("g3", refs, em, trained, nets_dir=str(tmp_path / "nets2"), nets_g3=paths["nets"],
                       out_path=str(tmp_path / "scores.jsonl"), threads=1, post_kw=pk) == 0                 # resume


def test_entry_points_hold_and_refuse():
    C.main([])                                                                             # the design, then hold
    if C._HAS_TORCH:
        with pytest.raises(SystemExit, match="no emulator"):
            C.main(["--g3", "--emulator", "no_such_file.npz"])
    else:
        with pytest.raises(SystemExit, match="torch is not in this environment"):
            C.main(["--g3", "--emulator", "no_such_file.npz"])


# ---- D81: per-cell pass or fail, the largest n within budget, the scoring cost (numpy only, with stand-ins) ----
def _fake_ladder(monkeypatch, d_by_n, slope_by_kind, seconds_by_n=None):
    """Stand-ins for training and prediction, so that G3's logic runs without torch: a run
    is a record; a network trained at n predicts Ĥ = Ĥ_exact + d_by_n[n] and ξ₀ with the
    kind's slope; the ratio estimator's prediction carries one sampler run per surface."""
    calls = {"train": [], "predict": []}

    def fake_train(kind, prior, j, q, n, theta, eps, s, em, val, jsonl_path, nets_dir, threads, train_kw):
        calls["train"].append((kind, prior, j, q, n))
        secs = (seconds_by_n or {}).get(n, 1.0)
        return {"label": C.run_label(kind, prior, j, q, n), "kind": kind, "prior": prior, "j": j, "q": q, "n": n,
                "status": "ok", "seconds": secs}, True

    def fake_predict(run, rows, burn=R.BURN, sample=R.SAMPLE, rhat_max=R.RHAT_MAX, ess_min=R.ESS_MIN):
        calls["predict"].append((run["kind"], run["n"]))
        preds = [(r["H_exact"] + d_by_n[run["n"]], slope_by_kind[run["kind"]] * r["xi0_exact"]) for r in rows]
        if run["kind"] == "ensemble":
            return preds, [], {"seconds": 0.02, "n": len(rows), "attempts": None}
        return preds, [], {"seconds": 1.0, "n": len(rows), "burn": burn, "sample": sample,
                           "attempts": [[{"burn": burn, "sample": sample, "seconds": 2.0}] for _ in rows]}
    monkeypatch.setattr(C, "get_or_train", fake_train)
    monkeypatch.setattr(C, "predict_rows", fake_predict)
    return calls


def _g3_refs(n=4, j_ref=1):
    return {f"g3_{i:04d}_{j_ref}": {"set": "g3", "task": f"g3_{i:04d}_{j_ref}", "i": i, "j": j_ref, "status": "ok",
                                    "theta": [0.1, 0.2, -0.8, 0.1], "y": [0.2] * 35,
                                    "under": {p: {"mean_H": 0.1 + 0.05 * i, "mean_xi0": 0.05 + 0.04 * i} for p in "AB"}}
            for i in range(n)}


def test_g3_cells_pass_on_their_own_at_the_largest_n_within_budget(tmp_path, monkeypatch):
    """Line 144 (D81): every rung is run; a cell's training-set size is the largest n at which
    both checks hold and the runs at n and 2n are within the budget; each (calibrator, prior,
    level) cell passes or fails on its own; P5 stops only if every cell fails."""
    d_by_n = {100: 0.010, 200: 0.0085, 400: 0.0080}               # |d| moves 0.0015, then 0.0005: both rungs stable
    calls = _fake_ladder(monkeypatch, d_by_n, {"ensemble": 1.0, "nre": 0.5})   # the ratio estimator misses ξ₀'s slope
    paths = {"jsonl": str(tmp_path / "l.jsonl"), "nets": str(tmp_path / "n"), "report": str(tmp_path / "g3.json")}
    rep = C.run_g3(LinearEm(), _g3_refs(), paths["jsonl"], paths["nets"], paths["report"], threads=1, ladder=(100, 200),
                   n_block=400, n_valid=16, levels=(0.003,), priors=("A",), post_kw={"burn": 10, "sample": 40})
    ens, nre = rep["cells"]["ensemble_A_j2"], rep["cells"]["nre_A_j2"]
    assert len(ens["rungs"]) == 2 and all(r["passed"] for r in ens["rungs"])           # both rungs run, both pass
    assert ens["passed"] and ens["n_star"] == 200                                        # the largest, not the smallest
    assert not nre["passed"] and nre["n_star"] is None and len(nre["rungs"]) == 2
    assert rep["passed"] is False and rep["p5_stops"] is False                          # one cell failing does not stop P5
    assert rep["cells_passing"] == ["ensemble_A_j2"] and rep["cells_not_passing"] == ["nre_A_j2"]
    assert sorted(set(calls["predict"])) == sorted(calls["predict"])                    # each network predicted once
    assert ens["cost_at_n_star"] == {"seconds": 0.02, "n": 4, "attempts": None}
    # a run over the wall budget at 2n fails that rung, so the largest n within budget is the smaller one
    calls2 = _fake_ladder(monkeypatch, d_by_n, {"ensemble": 1.0, "nre": 1.0}, seconds_by_n={400: 5 * 3600.0})
    rep2 = C.run_g3(LinearEm(), _g3_refs(), str(tmp_path / "l2.jsonl"), paths["nets"], str(tmp_path / "g3b.json"), threads=1,
                    ladder=(100, 200), n_block=400, n_valid=16, levels=(0.003,), priors=("A",),
                    post_kw={"burn": 10, "sample": 40})
    assert all(c["n_star"] == 100 for c in rep2["cells"].values()) and rep2["passed"] and not rep2["p5_stops"]
    assert rep2["cells"]["ensemble_A_j2"]["rungs"][1]["passed"] is False and "budget" in rep2["cells"]["ensemble_A_j2"]["rungs"][1]["reason"]
    # every cell failing stops P5
    _fake_ladder(monkeypatch, d_by_n, {"ensemble": 0.5, "nre": 0.5})
    rep3 = C.run_g3(LinearEm(), _g3_refs(), str(tmp_path / "l3.jsonl"), paths["nets"], str(tmp_path / "g3c.json"), threads=1,
                    ladder=(100, 200), n_block=400, n_valid=16, levels=(0.003,), priors=("A",),
                    post_kw={"burn": 10, "sample": 40})
    assert rep3["p5_stops"] is True and rep3["cells_passing"] == [] and "every cell" in rep3["p5_stops_reason"]


def test_g3_scoring_cost_projection_and_per_cell_seeds(tmp_path, monkeypatch):
    """Line 143's budget at G3 (D81): the projected cost of scoring both calibrators on the test
    set's pairs, five seeds each, against 1,344 core-hours; the seeds of a passing cell only."""
    cells = {"nre_A_j1": {"calibrator": "nre", "passed": True,
                          "cost_at_n_star": {"burn": 10000, "sample": 40000, "n": 2, "seconds": 0.0,
                                             "attempts": [[{"burn": 10000, "sample": 40000, "seconds": 20.0}],
                                                          [{"burn": 10000, "sample": 40000, "seconds": 30.0},
                                                           {"burn": 20000, "sample": 80000, "seconds": 100.0}]]}},
             "ensemble_A_j1": {"calibrator": "ensemble", "passed": True, "cost_at_n_star": {"seconds": 0.04, "n": 16}},
             "nre_B_j1": {"calibrator": "nre", "passed": False, "cost_at_n_star": None}}
    pr = C.scoring_projection(cells, threads=4)
    nre = pr["per_cell"]["nre_A_j1"]
    assert nre["median_unit_s"] == 30.0 and nre["mean_units"] == 2.0                    # units 1 and 1 + 2; seconds 20, 30, 50
    assert abs(nre["projected_core_hours"] - 400 * 5 * 30.0 * 2.0 * 4 / 3600) < 1e-9
    assert abs(pr["per_cell"]["ensemble_A_j1"]["projected_core_hours"] - 400 * 5 * 0.0025 * 4 / 3600) < 1e-12
    assert "nre_B_j1" not in pr["per_cell"] and pr["go"] and pr["budget_core_hours"] == 1344.0
    cells["nre_A_j1"]["cost_at_n_star"]["attempts"][0][0]["seconds"] = 2000.0                 # far too slow: no go
    cells["nre_A_j1"]["cost_at_n_star"]["attempts"][1][0]["seconds"] = 2000.0
    assert C.scoring_projection(cells, threads=4)["go"] is False
    # the seeds: the passing cell's only; none when P5 stops
    calls = _fake_ladder(monkeypatch, {100: 0.010, 200: 0.0085, 400: 0.0080}, {"ensemble": 1.0, "nre": 0.5})
    rep = C.run_g3(LinearEm(), _g3_refs(), str(tmp_path / "l.jsonl"), str(tmp_path / "n"), str(tmp_path / "g3.json"),
                   threads=1, ladder=(100, 200), n_block=400, n_valid=16, levels=(0.003,), priors=("A",),
                   post_kw={"burn": 10, "sample": 40})
    calls["train"].clear()
    trained = C.train_seeds(LinearEm(), rep, jsonl_path=str(tmp_path / "t.jsonl"), nets_dir=str(tmp_path / "n2"),
                            nets_g3=str(tmp_path / "n"), threads=1, n_block=400, n_valid=16, seeds=(2, 3))
    assert set(trained) == {"ensemble_A_j2"} and trained["ensemble_A_j2"]["n_star"] == 200
    assert {c[0] for c in calls["train"]} == {"ensemble"} and {c[3] for c in calls["train"]} == {2, 3}   # the failed cell untrained
    with pytest.raises(SystemExit, match="P5 stops"):
        C.train_seeds(LinearEm(), dict(rep, p5_stops=True, p5_stops_reason="cost"), jsonl_path=str(tmp_path / "t2.jsonl"))
    assert set(C.header("g3", 1)["versions"]) >= {"numpy", "scipy", "expected", "override"}
    assert C.header("g3", 1)["versions"]["expected"] == E.LEAF_VERSIONS


def test_g3_fails_closed_and_records_an_override(tmp_path, monkeypatch):
    """A level with no accepted pilot reference fails its cell rather than stopping the run; a
    passing cell whose scoring cost cannot be projected makes the projection incomplete and
    no go; a projected cost above the budget stops P5 through run_g3; a run under
    --allow-other-versions is recorded in the record it writes to and in the report; the
    command line refuses other numpy and scipy versions; the scores' header lists the
    networks scored (D81)."""
    for kind in ("ensemble", "nre"):
        preds, fails, cost = C.predict_rows({"kind": kind}, [], burn=10, sample=40)
        assert preds == [] and fails == [] and cost["n"] == 0
        assert C.g3_checks(preds, preds, [])["passed"] is False
    cells = {"ensemble_A_j1": {"calibrator": "ensemble", "passed": True, "cost_at_n_star": None},
             "ensemble_A_j2": {"calibrator": "ensemble", "passed": True, "cost_at_n_star": {"seconds": 0.04, "n": 16}}}
    pr = C.scoring_projection(cells, threads=1)
    assert pr["complete"] is False and pr["cells_not_projected"] == ["ensemble_A_j1"] and pr["go"] is False
    _fake_ladder(monkeypatch, {100: 0.010, 200: 0.0085, 400: 0.0080}, {"ensemble": 1.0, "nre": 1.0})
    over = E.version_check({"numpy": "0.0.0", "scipy": "0.0.0"}, allow=True, check=True)
    jl = str(tmp_path / "l.jsonl")
    rep = C.run_g3(LinearEm(), _g3_refs(), jl, str(tmp_path / "n"), str(tmp_path / "g3.json"), threads=1,
                   ladder=(100, 200), n_block=400, n_valid=16, levels=(0.003,), priors=("A",),
                   post_kw={"burn": 10, "sample": 40}, versions=over, budget_core_h=1e-9)
    assert rep["cells_passing"] and rep["p5_stops"] is True and "above the budget" in rep["p5_stops_reason"]
    assert rep["versions"]["override"] is True and rep["scoring_projection"]["complete"] is True
    lines = [json.loads(l) for l in Path(jl).read_text(encoding="utf-8").splitlines()]
    assert [l["kind"] for l in lines[:2]] == ["header", "launch"] and lines[1]["versions"]["override"] is True
    Path(tmp_path, "g3.json").rename(tmp_path / "g3.first.json")                       # a re-run moves the report aside (D82)
    C.run_g3(LinearEm(), _g3_refs(), jl, str(tmp_path / "n"), str(tmp_path / "g3.json"), threads=1, ladder=(100, 200),
             n_block=400, n_valid=16, levels=(0.003,), priors=("A",), post_kw={"burn": 10, "sample": 40})
    assert sum(1 for l in Path(jl).read_text(encoding="utf-8").splitlines() if '"launch"' in l) == 1   # no override, no line
    assert C.scored_networks({"ensemble_A_j1": {"seeds": {"1": "a", "3": "b"}}, "nre_B_j3": {"seeds": {1: "c", 2: "d"}}}) == \
        [["ensemble", "A", 1, 1], ["ensemble", "A", 1, 3], ["nre", "B", 3, 1], ["nre", "B", 3, 2]]
    assert "main(check_versions=True)" in inspect.getsource(C).split("__main__")[-1]
    monkeypatch.setattr(E, "LEAF_VERSIONS", {"numpy": "0.0.0", "scipy": "0.0.0"})
    C.main([])                                                                             # a call: nothing refused
    with pytest.raises(SystemExit, match="refused: numpy"):
        C.main([], check_versions=True)
    C.main(["--allow-other-versions"], check_versions=True)


def test_the_test_set_is_scored_only_after_stage1(tmp_path):
    """Line 101 (D81): scoring the test set refuses without Stage 1's output for its
    references, with one made from other references, or one that stopped P5; a scores file
    begun after another Stage 1 output is refused on resume; the header records the Stage
    1 output's digest (with torch present; without it, the scoring stops for want of torch
    once Stage 1 is in order). The emulator hashes it reads, Stage 1's among them, must all
    agree (D82)."""
    import p5_stage2 as S2
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps({"kind": "header", "set": "test", "n_points": 1}) + "\n"
                    + json.dumps({"kind": "task", "task": "test_0000_0", "status": "ok"}) + "\n", encoding="utf-8")
    out = tmp_path / "scores.jsonl"
    g = "e" * 64                                                                          # a stand-in for G̃'s hash
    kw = dict(out_path=str(out), threads=1, refs_path=str(refs), stage1_dir=str(tmp_path / "s1"),
              hashes={"G̃": g, "test references": g, "G3 report": g})
    with pytest.raises(SystemExit, match="no Stage 1 output"):
        C.score_set("test", {}, None, {}, **kw)
    (tmp_path / "s1").mkdir()
    s1 = tmp_path / "s1" / "p5_stage1_report.json"
    s1.write_text(json.dumps({"refs_digest": "0" * 64, "p5_stops": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="other references"):
        C.score_set("test", {}, None, {}, **kw)
    s1.write_text(json.dumps({"refs_digest": S2.refs_digest(str(refs)), "p5_stops": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="P5 stopped at Stage 1"):
        C.score_set("test", {}, None, {}, **kw)
    s1.write_text(json.dumps({"refs_digest": S2.refs_digest(str(refs)), "p5_stops": False, "emulator": {"sha256": g}}),
                  encoding="utf-8")
    out.write_text(json.dumps({"kind": "header", "mode": "score", "set": "test", "stage1_refs_digest": "f" * 64}) + "\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit, match="begun after another Stage 1 output"):
        C.score_set("test", {}, None, {}, **kw)
    out.unlink()
    with pytest.raises(SystemExit, match="the emulator hashes differ"):                  # one hash not G̃'s (D82)
        C.score_set("test", {}, None, {}, **dict(kw, hashes=dict(kw["hashes"], **{"G3 report": "f" * 64})))
    with pytest.raises(SystemExit, match="no emulator hash in test references"):
        C.score_set("test", {}, None, {}, **dict(kw, hashes=dict(kw["hashes"], **{"test references": None})))
    assert not out.exists()                                                               # refused before any header
    if C._HAS_TORCH:
        assert C.score_set("test", {}, None, {}, **kw) == 0
        h = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert h["stage1_refs_digest"] == S2.refs_digest(str(refs)) and h["expected"] == []
    else:
        with pytest.raises(SystemExit, match="torch"):
            C.score_set("test", {}, None, {}, **kw)


# ---- D82: a complete pilot record made with G̃; resumed runs; nothing written over ----
def test_g3_needs_a_complete_pilot_record_made_with_g_tilde():
    """Line 144 (D82): G3 and the training of seeds run only when all 48 G3-pilot pairs (16
    surfaces at 3 levels) are recorded, accepted or failed, and the record was made with G̃."""
    g = "e" * 64
    hdr = {"kind": "header", "set": "g3", "n_points": 16, "emulator_sha256": g}
    tasks = {f"g3_{i:04d}_{j}": {"set": "g3", "status": "ok" if (i + j) % 5 else "reference_failure"}
             for i in range(16) for j in range(3)}
    assert C.require_pilot(hdr, tasks, g) is True                                         # failures count as recorded
    gone = tasks.pop("g3_0007_2")
    with pytest.raises(SystemExit, match="holds 47 of its 48 pairs \\(missing g3_0007_2\\)"):
        C.require_pilot(hdr, tasks, g)
    tasks["g3_0007_2"] = gone
    with pytest.raises(SystemExit, match="made with another emulator than G̃"):
        C.require_pilot(hdr, tasks, "f" * 64)
    with pytest.raises(SystemExit, match="holds 0 of its 48 pairs"):
        C.require_pilot(None, {}, g)
    assert C.require_pilot(hdr, dict(tasks, g3_0016_0={"set": "g3", "status": "ok"}), g) is True   # extra pairs ignored


def test_g3_resumes_only_on_a_matching_record_and_writes_nothing_over(tmp_path, monkeypatch):
    """Line 144 (D82): G3 refuses to write over its report, the refusal coming before any
    training; a resumed G3 or seed training reuses saved networks only when the record's
    header matches the call, the emulator included; the emulator is recorded."""
    _fake_ladder(monkeypatch, {100: 0.010, 200: 0.0085, 400: 0.0080}, {"ensemble": 1.0, "nre": 1.0})
    A, B = {"path": "output/a.npz", "sha256": "a" * 64}, {"path": "output/a.npz", "sha256": "b" * 64}
    jl, rp, nets = str(tmp_path / "l.jsonl"), str(tmp_path / "g3.json"), str(tmp_path / "n")
    kw = dict(threads=1, ladder=(100, 200), n_block=400, n_valid=16, levels=(0.003,), priors=("A",),
              post_kw={"burn": 10, "sample": 40})
    rep = C.run_g3(LinearEm(), _g3_refs(), jl, nets, rp, emulator=A, **kw)
    assert rep["emulator"] == A and json.loads(Path(jl).read_text(encoding="utf-8").splitlines()[0])["emulator"] == A
    with pytest.raises(SystemExit, match="G3 refused: .* exists; a re-run needs the old file moved aside"):
        C.run_g3(LinearEm(), _g3_refs(), jl, nets, rp, emulator=A, **kw)
    Path(rp).rename(rp + ".aside")
    with pytest.raises(SystemExit, match="resume refused: .* emulator sha256"):
        C.run_g3(LinearEm(), _g3_refs(), jl, nets, rp, emulator=B, **kw)
    with pytest.raises(SystemExit, match="resume refused: .* post_kw"):
        C.run_g3(LinearEm(), _g3_refs(), jl, nets, rp, emulator=A, **dict(kw, post_kw={"burn": 20, "sample": 80}))
    rep2 = C.run_g3(LinearEm(), _g3_refs(), jl, nets, rp, emulator=A, **kw)                # the same call resumes
    assert rep2["cells_passing"] == rep["cells_passing"] and not rep2["p5_stops"]
    tj = str(tmp_path / "t.jsonl")
    tkw = dict(jsonl_path=tj, nets_dir=str(tmp_path / "n2"), nets_g3=nets, threads=1, n_block=400, n_valid=16, seeds=(2,))
    C.train_seeds(LinearEm(), rep2, emulator=A, **tkw)
    assert json.loads(Path(tj).read_text(encoding="utf-8").splitlines()[0])["emulator"] == A
    with pytest.raises(SystemExit, match="resume refused: .* emulator sha256"):
        C.train_seeds(LinearEm(), rep2, emulator=B, **tkw)
