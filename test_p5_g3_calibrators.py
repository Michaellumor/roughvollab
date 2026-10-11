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
