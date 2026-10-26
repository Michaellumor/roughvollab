"""
test_p5_stage2.py — mechanics tests for the P5 Stage 1 and Stage 2 analysis script, on
synthetic records in the scripts' formats: the literals written into the draft (lines 85
to 89, 101, 108, 111, 121 to 124, 128, 132 to 134); the binning, the floors and the counts;
the measures and the identity for E; m on a toy sample store; line 75's exclusions; the
seed floor; the weighted median and the crossed bootstrap; the verdict rule on every row;
the null; the re-scoring; the dose-response; an end-to-end run with a reference failure
and a network failure planted. Nothing here touches real outputs.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import p5_stage2 as S
import p5_g2_reference as R
from p5_g0_pilot import append_line


def test_literals_are_the_written_ones():
    assert S.CUTS == (0.9, 0.5) and S.SHIFTS == (0.05, -0.05) and S.WINDOW == 0.02
    assert (S.FLOOR_SHARE, S.FLOOR_COUNT, S.GAP, S.N_SEEDS) == (0.10, 100, 0.20, 5)
    assert (S.N_BOOT, S.SEED_BOOT, S.CI, S.OVERWIDE, S.NULL_BAND) == (2000, 127, (2.5, 97.5), 0.95, (0.85, 0.95))
    h = S.HYPOTHESES
    assert h["H-a"] == {"name": "Faithful when informed", "measure": "abs_dA", "bin": "Identified", "direction": "below",
                        "expected": 0.01, "refuted": 0.03, "seed_floor": False}
    assert (h["H-b"]["measure"], h["H-b"]["bin"], h["H-b"]["expected"], h["H-b"]["refuted"]) == ("E", "Partial", 0.15, 0.05)
    assert (h["H-c"]["measure"], h["H-c"]["bin"], h["H-c"]["expected"], h["H-c"]["refuted"], h["H-c"]["seed_floor"]) == ("E", "Identified", 0.05, 0.02, True)
    assert (h["H-d"]["measure"], h["H-d"]["bin"], h["H-d"]["direction"], h["H-d"]["expected"], h["H-d"]["refuted"]) == ("m", "Flat", "below", 0.80, 0.85)
    assert S.SEED_BOOT not in (20261002, 7, 61, 62, 73, 79, 83, 89, 97, 101, 107, 109, 113)
    assert S.DEFAULT_INPUTS["refs"].endswith("p5v3_ref_test.jsonl") and S.DEFAULT_INPUTS["scores"].endswith("p5_scores_test.jsonl")


def _pairs(cs, levels=None, i0=0):
    out = {}
    for k, c in enumerate(cs):
        s = 0.001 if levels is None else levels[k]
        j = S.LEVEL_INDEX[float(s)]
        lab = f"test_{i0 + k:04d}_{j - 1}"
        out[lab] = {"label": lab, "i": i0 + k, "j": j, "s": float(s), "theta": [0.1 + 0.01 * k, 0.2, -0.8, 0.1], "c": float(c),
                    "H_exact": {"A": 0.10, "B": 0.30}, "xi0_exact": {"A": 0.1, "B": 0.1},
                    "sources": {"A": "reweighted", "B": "reweighted"}, "var_H_A": 0.001}
    return out


def test_binning_window_negative_floor_and_distinct_theta():
    assert [S.bin_of(c) for c in (0.95, 0.9, 0.89, 0.5, 0.49, -0.2)] == ["Identified", "Identified", "Partial", "Partial", "Flat", "Flat"]
    assert S.bin_of(0.9, (0.95, 0.55)) == "Partial" and S.bin_of(0.86, (0.85, 0.45)) == "Identified"
    cs = [0.95] * 120 + [0.7] * 150 + [0.2] * 25 + [-0.1] * 5 + [0.91, 0.885, 0.515, 0.485]
    pairs = _pairs(cs)
    s1, bins = S.stage1(pairs, n_failures=3)
    assert s1["n_accepted"] == 304 and s1["n_reference_failures"] == 3 and s1["n_negative_c"] == 5
    assert s1["bins"]["Identified"]["count"] == 121 and s1["bins"]["Partial"]["count"] == 152 and s1["bins"]["Flat"]["count"] == 31
    assert s1["near_cutoff"] == {"0.9": {"at_or_above": 1, "below": 1}, "0.5": {"at_or_above": 1, "below": 1}}   # per bin
    assert s1["c_by_level"]["0.001"]["n"] == 304 and s1["c_by_level"]["0.003"]["quartiles"] is None
    assert not s1["bins"]["Identified"]["untestable"] and s1["bins"]["Flat"]["untestable"]   # 31 < 100
    assert s1["bins"]["Partial"]["n_distinct_theta"] == 152 and s1["bins"]["Identified"]["per_level"]["0.001"] == 121
    # the count governs below N = 334: 12% of 304 is 36 pairs, still under 100
    small = _pairs([0.95] * 36 + [0.7] * 268)
    assert S.stage1(small)[0]["bins"]["Identified"]["untestable"]
    # distinct θ: pairs sharing a θ across levels count once
    multi = _pairs([0.95, 0.95, 0.95], levels=[0.001, 0.003, 0.005])
    for lab in multi:
        multi[lab]["i"] = 7
    assert S.stage1(multi)[0]["bins"]["Identified"]["n_distinct_theta"] == 1


def _store_for(pairs, mean=0.10, sd=0.02, seed=0):
    """A sample store in the reference's format: 'u' samples from a normal, quantised."""
    rng = np.random.default_rng(seed)
    stores = {}
    for lab in pairs:
        H = np.clip(rng.normal(mean, sd, 20000), 0.02, 0.48)
        stores[f"{lab}_u"] = R.quantise_H(H)
    return stores


def _scores(pairs, slots=(1, 2, 3, 4, 5), biasA=0.0, biasB=0.0, width=0.03, fail=()):
    rows = {}
    for cal in S.CALIBRATORS:
        for q in slots:
            for lab, p in pairs.items():
                for prior, bias in (("A", biasA), ("B", biasB)):
                    key = (cal, prior, q, lab)
                    if (cal, prior, q, lab) in fail:
                        rows[key] = {"calibrator": cal, "prior": prior, "q": q, "pair": lab, "status": "failure"}
                        continue
                    H_hat = p["H_exact"][prior] + bias + 0.001 * q
                    rows[key] = {"calibrator": cal, "prior": prior, "q": q, "pair": lab, "status": "ok", "H_hat": H_hat,
                                 "lo": H_hat - width, "hi": H_hat + width}
    return rows


def test_measures_identity_m_and_exclusions():
    pairs = _pairs([0.95, 0.7])
    stores = _store_for(pairs)
    lab0, lab1 = sorted(pairs)
    scores = _scores(pairs, biasA=0.004, biasB=0.010, fail={("nre", "B", 2, lab1)})
    slots = {"ensemble": (1, 2, 3, 4, 5), "nre": (1, 2, 3, 4, 5)}
    cells = S.measures(pairs, scores, stores, slots)
    c = cells[("ensemble", 1, lab0)]
    assert abs(c["dA"] - 0.005) < 1e-12 and abs(c["dB"] - 0.011) < 1e-12 and c["abs_dA"] == abs(c["dA"])
    assert abs(c["S_exact"] - 1.0) < 1e-12 and abs(c["S_net"] - (0.311 - 0.105) / 0.20) < 1e-12
    assert abs(c["E"] - (c["dB"] - c["dA"]) / 0.20) < 1e-12                               # line 109's identity
    H, w = R.weights_for({"under": {"A": {"source": "reweighted"}}}, S.pair_store(stores, lab0), "A")
    assert c["m"] == R.interval_prob(H, w, c["lo_A"], c["hi_A"]) and 0.5 < c["m"] < 1.0
    f = cells[("nre", 2, lab1)]                                                             # the B network failed
    assert f["failure"] == "B" and f["okA"] and not f["okB"]
    assert f["dA"] is not None and f["m"] is not None and f["dB"] is None and f["S_net"] is None and f["E"] is None
    assert cells[("nre", 2, lab0)]["failure"] is None and cells[("ensemble", 2, lab1)]["E"] is not None
    bins = {lab0: "Identified", lab1: "Partial"}
    ff = S.failure_floor(cells, bins, "nre", "Partial", (1, 2, 3, 4, 5), floor_count=1)
    assert ff["n_pairs"] == 1 and ff["n_failed"] == 1 and ff["n_failed_cells"] == 1 and ff["untestable"] and ff["failed"] == [lab1]
    assert not S.failure_floor(cells, bins, "ensemble", "Partial", (1, 2, 3, 4, 5), floor_count=1)["untestable"]
    # a vacant slot (no rows for (nre, B, level 1, q = 5)) is excluded, not a failure; a missing row elsewhere is refused
    present = {(cal, pr, 1, q) for cal in S.CALIBRATORS for pr in S.PRIORS for q in (1, 2, 3, 4, 5)} - {("nre", "B", 1, 5)}
    sparse = {k: v for k, v in scores.items() if k[:3] != ("nre", "B", 5)}
    cells2 = S.measures(pairs, sparse, stores, slots, present=present)
    v = cells2[("nre", 5, lab0)]
    assert v["vacant"] == "B" and v["failure"] is None and v["dA"] is not None and v["E"] is None
    assert S.failure_floor(cells2, bins, "nre", "Identified", (1, 2, 3, 4, 5), floor_count=1)["n_failed"] == 0
    with pytest.raises(SystemExit, match="incomplete"):
        S.measures(pairs, {k: v for k, v in scores.items() if k != ("ensemble", "A", 3, lab0)}, stores, slots,
                   present={(cal, pr, 1, q) for cal in S.CALIBRATORS for pr in S.PRIORS for q in (1, 2, 3, 4, 5)})


def test_seed_floor_on_hand_built_numbers():
    pairs = _pairs([0.95, 0.95])
    labs = sorted(pairs)
    cells = {}
    for q in (1, 2, 3):
        for k, lab in enumerate(labs):
            cells[("ensemble", q, lab)] = {"H_hat_A": 0.10 + 0.002 * q, "H_hat_B": 0.30 + 0.010 * q if k == 0 else None}
    bins = {lab: "Identified" for lab in labs}
    sf = S.seed_floor(cells, bins, "Identified", "ensemble", (1, 2, 3))
    # A: |diff| in {0.002, 0.004, 0.002} per pair -> /0.20 -> {0.01, 0.02, 0.01}, over two pairs: median 0.01
    # B: only pair 0 has values: {0.05, 0.10, 0.05} -> median 0.05; the floor is the larger
    assert abs(sf["per_prior"]["A"] - 0.01) < 1e-12 and abs(sf["per_prior"]["B"] - 0.05) < 1e-12 and abs(sf["floor"] - 0.05) < 1e-12
    assert sf["n_pairs"] == 2


def test_weighted_median_and_crossed_bootstrap():
    assert S.weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == 2.0 and S.weighted_median([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]) == 2.0
    assert S.weighted_median([1.0, 2.0, 3.0], [1, 0, 5]) == 3.0 and S.weighted_median([1.0, 2.0], [0, 0]) is None
    M = np.full((30, 5), 0.7)
    rng = np.random.default_rng(S.SEED_BOOT)
    interval, reps, info = S.bootstrap_median(M, rng, n_boot=200)
    assert interval == [0.7, 0.7] and len(reps) == 200 and info == {"n_live_pairs": 30, "n_live_seeds": 5, "n_dropped_replicates": 0}
    M2 = np.random.default_rng(1).normal(0.0, 1.0, (40, 5))
    M2[:, 4] = np.nan                                                                     # a cell left with four seeds
    M2[3, :] = np.nan                                                                     # a pair with no live cell
    i1, r1, f1 = S.bootstrap_median(M2, np.random.default_rng(S.SEED_BOOT), n_boot=300)
    i2, r2, f2 = S.bootstrap_median(M2, np.random.default_rng(S.SEED_BOOT), n_boot=300)
    assert i1 == i2 and np.array_equal(r1, r2) and f1 == {"n_live_pairs": 39, "n_live_seeds": 4, "n_dropped_replicates": 0}
    point, per_seed, n_cells = S.point_and_seed_medians(M2)
    assert n_cells == 156 and per_seed[4] is None and all(v is not None for v in per_seed[:4])
    assert i1[0] < point < i1[1] and point == S.weighted_median(M2[np.isfinite(M2)], np.ones(156))   # the same lower median
    # the multiplicities: a pair drawn k times and a seed drawn l times enters k·l times
    M3 = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert S.weighted_median(M3.ravel(), np.outer([2, 0], [1, 3]).ravel()) == 10.0            # weights 2, 6, 0, 0
    assert S.bootstrap_median(np.full((3, 2), np.nan), rng)[:2] == (None, [])
    # the lower median on a discrete sample: the point and every replicate come from the same convention
    M4 = np.array([[0.0], [1.0]])
    assert S.point_and_seed_medians(M4)[0] == 0.0 and set(S.bootstrap_median(M4, rng, n_boot=50)[1]) <= {0.0, 1.0}
    # the vectorised row-wise median equals weighted_median row by row, NaNs and zero weights included
    M5 = np.random.default_rng(2).normal(size=(50, 5))
    M5[np.random.default_rng(3).uniform(size=M5.shape) < 0.3] = np.nan
    M5[7, :] = np.nan
    for l_seed in ([1, 1, 1, 1, 1], [2, 0, 1, 0, 2], [0, 0, 3, 0, 0]):
        got = S.rowwise_weighted_median(M5, np.array(l_seed))
        for a in range(50):
            fin = np.isfinite(M5[a])
            w = np.array(l_seed, dtype=float) * fin
            exp = S.weighted_median(M5[a][fin], w[fin]) if w.sum() > 0 else None
            assert (exp is None and np.isnan(got[a])) or got[a] == exp


def test_verdict_rule_on_every_row_and_the_null():
    ha, hb, hc, hd = (S.HYPOTHESES[k] for k in ("H-a", "H-b", "H-c", "H-d"))
    assert S.verdict([0.002, 0.009], ha) == "Confirmed" and S.verdict([0.031, 0.05], ha) == "Refuted"
    assert S.verdict([0.005, 0.012], ha) == "Inconclusive" and S.verdict([0.02, 0.025], ha) == "Inconclusive"
    assert S.verdict([0.16, 0.30], hb) == "Confirmed" and S.verdict([0.0, 0.04], hb) == "Refuted" and S.verdict([0.04, 0.20], hb) == "Inconclusive"
    assert S.verdict([0.06, 0.09], hc, floor=0.03) == "Confirmed" and S.verdict([0.06, 0.09], hc, floor=0.07) == "Inconclusive"
    assert S.verdict([0.06, 0.09], hc, floor=None) == "Inconclusive" and S.verdict([0.0, 0.015], hc, floor=0.03) == "Refuted"
    assert S.verdict([0.6, 0.79], hd) == "Confirmed" and S.verdict([0.86, 0.99], hd) == "Refuted"     # over-wide is Refuted
    assert S.verdict([0.96, 0.99], hd) == "Refuted" and S.verdict([0.84, 0.90], hd) == "Inconclusive" and S.verdict(None, hd) == "Untestable"
    assert S.verdict([0.005, 0.01], ha) == "Inconclusive" and S.verdict([0.03, 0.04], ha) == "Inconclusive"   # the ends are not "beyond"
    assert S.verdict([0.15, 0.3], hb) == "Inconclusive" and S.verdict([0.0, 0.05], hb) == "Inconclusive"


N_SYNTH = 150                                                                            # per level; 449 accepted pairs
EM_SHA = "e" * 64                                                                        # a stand-in for G̃'s hash (D82)
EM_INFO = {"path": "output/p5_g1_emulator.npz", "sha256": EM_SHA}


def _write_inputs(tmp_path, n_per_level=N_SYNTH, slots=(1, 2, 3, 4, 5), seed=5, plant=True, faithful=False, lost=(),
                  fail_refs=(), n_points=None):
    """Synthetic reference and score files in the scripts' formats: c spread over the three
    bins by level, the networks' Ĥ near the exact means with a prior-dependent bias in the
    weaker bins, one reference failure and one network failure (one θ's three pairs,
    for the ratio estimator under B at slot 2) planted. With `faithful`, the networks are
    faithful (d near 0, E near 0, the exact 90% interval, so m near 0.90). `lost` names the
    (calibrator, prior, level j from 1) cells that failed G3: they get no scores and the G3
    report written beside the files marks them failed (D81). `fail_refs` adds reference
    failures at (i, j from 0). Every header carries the same stand-in emulator hash (D82)."""
    rng = np.random.default_rng(seed)
    refs_path, samples_path, scores_path = (str(tmp_path / n) for n in ("refs.jsonl", "refs.npz", "scores.jsonl"))
    g3_path = str(tmp_path / "g3_report.json")
    append_line(refs_path, {"kind": "header", "set": "test", "n_points": n_points or n_per_level,
                            "seeds": {"points": 20261002, "noise": 1}, "emulator": EM_INFO["path"],
                            "emulator_sha256": EM_SHA, "g2": {"passed": True, "emulator_sha256": EM_SHA}})
    stores = {}
    pairs = []
    for i in range(n_per_level):
        theta = [0.02 + 0.46 * rng.beta(2, 2), 0.2, -0.8, 0.1]
        for j, s in enumerate(S.NOISE_LEVELS):
            c = float(np.clip(rng.normal((0.95, 0.70, 0.25)[j], 0.05), -0.1, 0.999))
            lab = f"test_{i:04d}_{j}"
            mA, mB = theta[0] + rng.normal(0, 0.002), theta[0] + (1 - c) * 0.2 + rng.normal(0, 0.002)
            sd = 0.054 * np.sqrt(max(1 - c, 1e-3))
            status = "reference_failure" if ((plant and i == 3 and j == 1) or (i, j) in set(fail_refs)) else "ok"
            rec = {"kind": "task", "task": lab, "set": "test", "i": i, "j": j, "s": float(s), "status": status, "theta": theta,
                   "y": [0.2] * 35, "c_A": c, "seed": [83, i, j], "runs": {}, "kish": {}, "direct": [],
                   "under": {"A": {"mean_H": mA, "var_H": sd ** 2, "q05_H": mA - 1.645 * sd, "q95_H": mA + 1.645 * sd,
                                   "mean_xi0": 0.1, "source": "reweighted"},
                             "B": {"mean_H": mB, "var_H": sd ** 2, "q05_H": mB - 1.645 * sd, "q95_H": mB + 1.645 * sd,
                                   "mean_xi0": 0.1, "source": "reweighted"}},
                   "seconds": 1.0}
            if faithful:                                        # the exact A posterior is the stored sample itself
                rec["direct"], rec["under"]["A"]["source"] = ["A"], "direct"
            append_line(refs_path, rec)
            if status == "ok":
                stores[f"{lab}_u"] = R.quantise_H(np.clip(rng.normal(mA, sd, 20000), 0.02, 0.48))
                if faithful:
                    stores[f"{lab}_A"] = stores[f"{lab}_u"]
            pairs.append((lab, i, j, c, mA, mB, sd))
    np.savez_compressed(samples_path, **stores)
    lost = set(lost)
    expected = sorted([cal, prior, j, q] for cal in S.CALIBRATORS for prior in S.PRIORS for j in (1, 2, 3)
                      if (cal, prior, j) not in lost for q in slots)              # the producer's list of networks (D81)
    append_line(scores_path, {"kind": "header", "mode": "score", "set": "test", "threads": 1, "expected": expected,
                              "stage1_refs_digest": S.refs_digest(refs_path),    # as made after Stage 1 on these references
                              "emulator": EM_INFO})
    for cal in S.CALIBRATORS:
        for q in slots:
            for lab, i, j, c, mA, mB, sd in pairs:
                for prior, mean in (("A", mA), ("B", mB)):
                    if (cal, prior, j + 1) in lost:
                        continue                                                       # a cell that failed G3: no networks
                    run_label = f"{cal}_p{1 if prior == 'A' else 2}_j{j + 1}_q{q}_n10000"     # the producer's run label
                    if plant and cal == "nre" and prior == "B" and q == 2 and i == 5:
                        append_line(scores_path, {"kind": "task", "task": f"{run_label}|{lab}", "run": run_label, "pair": lab,
                                                  "calibrator": cal, "prior": prior, "j": j + 1, "q": q, "i": i, "status": "failure"})
                        continue
                    leak = 0.0 if (faithful or c >= 0.9) else 0.08 * (1 - c)        # a prior-dependent bias off the identified bin
                    H_hat = mean + (leak if prior == "B" else -leak) + rng.normal(0, 0.001)
                    width = 1.645 * sd * (1.0 if (faithful or c >= 0.5) else 0.7)     # too narrow in the flat bin
                    append_line(scores_path, {"kind": "task", "task": f"{run_label}|{lab}", "run": run_label, "pair": lab,
                                              "calibrator": cal, "prior": prior, "j": j + 1, "q": q, "i": i, "status": "ok",
                                              "H_hat": H_hat, "xi0_hat": 0.1, "lo": H_hat - width, "hi": H_hat + width})
    Path(g3_path).write_text(json.dumps({"gate": "G3", "cells": {f"{cal}_{prior}_j{j}": {"passed": (cal, prior, j) not in lost}
                                                                 for cal in S.CALIBRATORS for prior in S.PRIORS
                                                                 for j in (1, 2, 3)}, "emulator": EM_INFO}), encoding="utf-8")
    return {"refs": refs_path, "samples": samples_path, "scores": scores_path, "g3_report": g3_path}


def test_end_to_end_on_synthetic_inputs(tmp_path, capsys):
    inputs = _write_inputs(tmp_path)
    out_dir = str(tmp_path / "out")
    S.run_stage1(inputs, out_dir)                                                        # Stage 1 first, from the references (D81)
    rep = S.run(inputs, out_dir, n_boot=150)
    s1 = rep["stage1"]
    n_acc = 3 * N_SYNTH - 1
    assert s1["n_accepted"] == n_acc and rep["n_reference_failures"] == 1 and rep["reference_failures"] == ["test_0003_1"]
    assert sum(v["count"] for v in s1["bins"].values()) == n_acc and s1["bins"]["Identified"]["count"] >= 100
    assert rep["seed_slots"] == {"ensemble": [1, 2, 3, 4, 5], "nre": [1, 2, 3, 4, 5]}
    ens, nre = rep["stage2"]["ensemble"], rep["stage2"]["nre"]
    for cal in (ens, nre):
        for h, row in cal["hypotheses"].items():
            assert row["verdict"] in ("Confirmed", "Refuted", "Inconclusive", "Untestable")
            if row["verdict"] != "Untestable":
                assert row["interval"][0] <= row["point"] <= row["interval"][1] and len(row["per_seed"]) == 5
                assert set(row["per_level"]) <= {"0.001", "0.003", "0.005"}
    v = {h: r["verdict"] for h, r in ens["hypotheses"].items()}
    assert v == {"H-a": "Confirmed", "H-b": "Confirmed", "H-c": "Refuted", "H-d": "Confirmed"}   # the planted effects, by direction
    assert ens["hypotheses"]["H-d"]["share_over_wide"] is not None and ens["hypotheses"]["H-c"]["seed_floor"] is not None
    assert ens["hypotheses"]["H-a"]["bootstrap"]["n_live_seeds"] == 5 and ens["hypotheses"]["H-a"]["n_failed_pairs"] == 0
    assert all(S.bin_of(c, tuple(rep["re_scoring"]["+0.05"]["cuts"])) == b
               for c, b in ((0.95, "Identified"), (0.949, "Partial"), (0.55, "Partial"), (0.549, "Flat")))      # no float noise
    assert "calibrators" in rep["re_scoring"]["-0.05"] and rep["re_scoring"]["-0.05"]["calibrators"]["ensemble"]["seed_floor"]["Identified"]["floor"] is not None
    assert sum(nre["failure_floor"][b]["n_failed"] for b in S.BINS) == 3                 # θ_5 at each of the three levels
    assert sum(ens["failure_floor"][b]["n_failed"] for b in S.BINS) == 0
    assert not rep["null"]["ensemble"]["stands"] and set(rep["re_scoring"]) == {"+0.05", "-0.05"}
    assert rep["re_scoring"]["+0.05"]["cuts"] == [0.95, 0.55] and rep["re_scoring"]["-0.05"]["cuts"] == [0.85, 0.45]
    dr = rep["dose_response"]["ensemble"]
    assert dr["spearman"] < 0 and dr["interval"][0] <= dr["spearman"] <= dr["interval"][1] and dr["n_pairs"] == n_acc
    assert {k: rep["bootstrap"][k] for k in ("n_boot", "seed", "interval", "ci")} == {"n_boot": 150, "seed": 127, "interval": "percentile", "ci": [2.5, 97.5]}
    assert "lower weighted median" in rep["bootstrap"]["median"] and "narrower than nominal" in rep["bootstrap"]["note"]
    assert "without adjustment" in rep["multiplicity"]
    rows = list(csv.DictReader(Path(out_dir, "p5_stage1.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == n_acc and set(r["bin"] for r in rows) == set(S.BINS)
    cells = list(csv.DictReader(Path(out_dir, "p5_stage2_cells.csv").read_text(encoding="utf-8").splitlines()))
    assert len(cells) == 2 * 5 * n_acc
    failed = [r for r in cells if r["failure"]]
    assert len(failed) == 3 and all(r["calibrator"] == "nre" and r["q"] == "2" and r["E"] == "" and r["dA"] != "" for r in failed)
    json.dumps(json.loads(Path(out_dir, "p5_stage2_report.json").read_text(encoding="utf-8")), allow_nan=False)
    assert Path(out_dir, "p5_stage1.png").exists() == S._HAS_MPL
    _aside(out_dir)                                                                     # a re-run moves the report aside (D82)
    rep2 = S.run(inputs, out_dir, n_boot=150)
    assert rep2["verdicts"] == rep["verdicts"] and rep2["stage2"]["ensemble"]["hypotheses"]["H-a"]["interval"] == ens["hypotheses"]["H-a"]["interval"]
    S.main([])
    out = capsys.readouterr().out
    assert "HOLD" in out
    with pytest.raises(SystemExit, match="no refs"):
        S.run({"refs": str(tmp_path / "missing.jsonl"), "samples": inputs["samples"], "scores": inputs["scores"]}, out_dir)
    assert rep["n_pairs_expected"] == 3 * N_SYNTH and rep["n_pairs_missing_from_record"] == 0
    # a duplicated well-formed row is refused; a truncated line is tolerated as the producer's reader tolerates it
    dup = str(tmp_path / "dup.jsonl")
    lines = Path(inputs["scores"]).read_text(encoding="utf-8").splitlines()
    Path(dup).write_text("\n".join(lines + [lines[5]]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="more than one row"):
        S.load_scores(dup)
    trunc = str(tmp_path / "trunc.jsonl")
    Path(trunc).write_text("\n".join(lines + [lines[5][:40]]) + "\n", encoding="utf-8")
    assert len(S.load_scores(trunc)[1]) == len(S.load_scores(inputs["scores"])[1])
    # references with a missing sample store entry are refused; mismatched sets are refused
    bad_store = str(tmp_path / "bad.npz")
    st = dict(np.load(inputs["samples"]))
    st.pop("test_0001_0_u")
    np.savez_compressed(bad_store, **st)
    with pytest.raises(SystemExit, match="no stored samples"):
        S.load_refs(inputs["refs"], bad_store)
    g3 = str(tmp_path / "g3scores.jsonl")
    Path(g3).write_text("\n".join([json.dumps({"kind": "header", "mode": "score", "set": "g3"})] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="inputs refused"):
        S.run({"refs": inputs["refs"], "samples": inputs["samples"], "scores": g3}, str(tmp_path / "out2"), n_boot=5)


def test_floors_and_untestable_rows_on_a_thin_bin(tmp_path):
    inputs = _write_inputs(tmp_path, n_per_level=20, plant=False)                        # 60 pairs: every bin under 100
    S.run_stage1(inputs, str(tmp_path / "out"))                                          # Stage 1 first (D81)
    rep = S.run(inputs, str(tmp_path / "out"), n_boot=50)
    for cal in S.CALIBRATORS:
        assert all(r["verdict"] == "Untestable" and r["untestable_reason"] == "stage 1 floor" for r in rep["stage2"][cal]["hypotheses"].values())
        assert not rep["null"][cal]["stands"] and rep["null"][cal]["all_four_refuted"] is False
    assert all(v["untestable"] for v in rep["stage1"]["bins"].values())


# ---- D81: the null, Stage 1 alone, the secondary analysis, G3's per-cell losses, the failure stop ----
def _aside(out_dir):
    """What a re-run needs (D82): Stage 2's report and cells CSV moved aside, the earlier
    copies of a previous move replaced."""
    for name in ("p5_stage2_report.json", "p5_stage2_cells.csv"):
        p = Path(out_dir, name)
        if p.exists():
            p.replace(p.with_name(p.name + ".aside"))


def _scored(inputs, n_boot=120, lost=None):
    """Stage 2's machinery on the files without the run's input checks: the hypotheses at the
    written cut-offs (the null's test runs on code that predates run_stage1)."""
    hdr, pairs, stores, failures = S.load_refs(inputs["refs"], inputs["samples"])
    _, scores, slots, present = S.load_scores(inputs["scores"])
    cells = S.measures(pairs, scores, stores, slots, present=present)
    kw = {} if lost is None else {"lost": lost}
    return S.score_hypotheses(pairs, cells, slots, n_boot=n_boot, n_failures=len(failures), **kw)[0]


def test_the_null_stands_for_a_faithful_calibrator_and_not_for_an_unfaithful_one(tmp_path):
    """Line 128 (D81): H-a Confirmed, H-b, H-c and H-d Refuted, H-d's interval within
    [0.85, 0.95]. A planted faithful calibrator (d near 0, E near 0, m near 0.90) meets it;
    the planted unfaithful one does not."""
    good = _scored(_write_inputs(tmp_path / "good", faithful=True, plant=False))
    for cal in S.CALIBRATORS:
        v = {h: r["verdict"] for h, r in good["calibrators"][cal]["hypotheses"].items()}
        assert v == {"H-a": "Confirmed", "H-b": "Refuted", "H-c": "Refuted", "H-d": "Refuted"}, (cal, v)
        hd = good["calibrators"][cal]["hypotheses"]["H-d"]
        assert 0.85 <= hd["interval"][0] and hd["interval"][1] <= 0.95 and abs(hd["point"] - 0.90) < 0.02
        assert good["calibrators"][cal]["null"]["stands"] is True
    bad = _scored(_write_inputs(tmp_path / "bad"))
    for cal in S.CALIBRATORS:
        assert bad["calibrators"][cal]["null"]["stands"] is False


def test_null_reading_on_hand_built_verdicts():
    """v1's rule (all four Refuted) is not the null any more (line 128, D81)."""
    refuted4 = {"H-a": "Refuted", "H-b": "Refuted", "H-c": "Refuted", "H-d": "Refuted"}
    faithful = {"H-a": "Confirmed", "H-b": "Refuted", "H-c": "Refuted", "H-d": "Refuted"}
    nr = S.null_reading(refuted4, [0.88, 0.92])
    assert nr["stands"] is False and nr["all_four_refuted"] is True
    assert S.null_reading(faithful, [0.85, 0.95])["stands"] is True                      # closed ends
    assert S.null_reading(faithful, [0.86, 0.96])["stands"] is False                     # over-wide: not within the band
    assert S.null_reading(dict(faithful, **{"H-c": "Inconclusive"}), [0.88, 0.92])["stands"] is False
    assert S.null_reading(dict(faithful, **{"H-b": "Untestable"}), [0.88, 0.92])["stands"] is False
    assert S.null_reading(faithful, None)["stands"] is False


def test_stage1_alone_from_the_references_and_stage2_refuses_without_it(tmp_path):
    inputs = _write_inputs(tmp_path)
    for k in ("samples", "scores", "g3_report"):                       # Stage 1 needs no samples, scores or G3 report
        Path(inputs[k]).rename(inputs[k] + ".away")
    out = str(tmp_path / "s1")
    rep1 = S.run_stage1({"refs": inputs["refs"]}, out)
    for k in ("samples", "scores", "g3_report"):
        Path(inputs[k] + ".away").rename(inputs[k])
    s1 = rep1["stage1"]
    assert s1["n_accepted"] == 3 * N_SYNTH - 1 and rep1["reference_failures"]["n_failures"] == 1
    assert rep1["refs_sha256"] == S.sha256_of(inputs["refs"]) and not rep1["p5_stops"]
    assert rep1["refs_digest"] == S.refs_digest(inputs["refs"])
    assert set(rep1["untestable_both_calibrators"]) == {b for b in S.BINS if s1["bins"][b]["untestable"]}
    hdr, pairs, _, _ = S.load_refs(inputs["refs"], inputs["samples"])
    for b in S.BINS:                                                    # the share well covered per bin, from the references
        labs = [lab for lab, p in pairs.items() if S.bin_of(p["c"]) == b]
        n_wc = sum(1 for lab in labs if pairs[lab]["well_covered"])
        assert s1["bins"][b]["n_well_covered"] == n_wc and rep1["coverage"]["per_bin"][b]["n"] == n_wc
    assert 0 < rep1["coverage"]["n_well_covered"] < len(pairs) and rep1["coverage"]["cover_min"] == 0.001
    assert rep1["coverage"]["secondary_stage1"]["n_accepted"] == rep1["coverage"]["n_well_covered"]
    for lab, p in pairs.items():                                        # the covariate: each prior's mass on its 90% interval
        assert all(0.0 <= p["coverage"][q] <= 1.0 for q in S.PRIORS)
        assert p["well_covered"] == (p["coverage"]["A"] >= 0.001 and p["coverage"]["B"] >= 0.001)
    rows = list(csv.DictReader(Path(out, "p5_stage1.csv").read_text(encoding="utf-8").splitlines()))
    assert {"cover_A", "cover_B", "well_covered"} <= set(rows[0]) and len(rows) == len(pairs)
    assert set(rep1["versions"]) >= {"numpy", "scipy", "expected", "override"}
    # Stage 2 refuses without Stage 1's output, or with one made from other references
    with pytest.raises(SystemExit, match="no Stage 1 output"):
        S.run(inputs, str(tmp_path / "elsewhere"), n_boot=5)
    S.run(inputs, out, n_boot=5)                                        # with it, Stage 2 runs
    with open(inputs["refs"], "a", encoding="utf-8") as f:              # a re-run with nothing to do appends a launch line:
        f.write(json.dumps({"kind": "launch", "utc": "later"}) + "\n")  # the records Stage 1 read are unchanged
    with pytest.raises(SystemExit, match="exists; a re-run needs the old file moved aside"):   # no report written over (D82)
        S.run(inputs, out, n_boot=5)
    _aside(out)
    S.run(inputs, out, n_boot=5)
    refs_text, scores_text = Path(inputs["refs"]).read_text(encoding="utf-8"), Path(inputs["scores"]).read_text(encoding="utf-8")
    Path(inputs["refs"]).write_text(refs_text.replace('"task": "test_0000_0"', '"task": "test_0000_0", "note": "edited"'),
                                    encoding="utf-8")
    with pytest.raises(SystemExit, match="other references"):           # a pair's record changed
        S.run(inputs, out, n_boot=5)
    Path(inputs["refs"]).write_text(refs_text, encoding="utf-8")
    hdr_line, rest = scores_text.split("\n", 1)
    Path(inputs["scores"]).write_text(hdr_line.replace(S.refs_digest(inputs["refs"]), "0" * 64) + "\n" + rest, encoding="utf-8")
    with pytest.raises(SystemExit, match="not made after this Stage 1 output"):
        S.run(inputs, out, n_boot=5)
    gone = [l for l in rest.splitlines() if '"calibrator": "ensemble", "prior": "A", "j": 1, "q": 5' in l]
    assert len(gone) == sum(1 for lab in pairs if lab.endswith("_0"))
    Path(inputs["scores"]).write_text(hdr_line + "\n" + "".join(l + "\n" for l in rest.splitlines() if l not in gone),
                                      encoding="utf-8")
    with pytest.raises(SystemExit, match="the scoring is incomplete"):  # a slot not scored is not read as an empty one
        S.run(inputs, out, n_boot=5)
    Path(inputs["scores"]).write_text(scores_text, encoding="utf-8")
    s1_path = Path(out, "p5_stage1_report.json")
    s1_text = s1_path.read_text(encoding="utf-8")
    s1_edit = json.loads(s1_text)
    s1_edit["stage1"]["bins"]["Flat"]["untestable"] = not s1_edit["stage1"]["bins"]["Flat"]["untestable"]
    s1_path.write_text(json.dumps(s1_edit), encoding="utf-8")
    _aside(out)
    with pytest.raises(SystemExit, match="its Stage 1 differs"):        # the declaration stands as made
        S.run(inputs, out, n_boot=5)
    s1_path.write_text(s1_text, encoding="utf-8")
    # reference failures above 10% at a level stop P5 at Stage 1, and Stage 2 then refuses
    many = _write_inputs(tmp_path / "many", n_per_level=40, plant=False, fail_refs={(i, 2) for i in range(5)})
    rep_m = S.run_stage1(many, str(tmp_path / "m"))
    lv = rep_m["reference_failures"]["by_level"]["0.005"]
    assert lv["n_failures"] == 5 and lv["n_test_pairs"] == 40 and rep_m["p5_stops"]
    assert rep_m["reference_failures"]["levels_over"] == ["0.005"]
    assert sum(map(sum, rep_m["reference_failures"]["by_true_H_nu"]["counts"])) == 5
    with pytest.raises(SystemExit, match="P5 stopped at Stage 1"):
        S.run(many, str(tmp_path / "m"), n_boot=5)
    four = _write_inputs(tmp_path / "four", n_per_level=40, plant=False, fail_refs={(i, 2) for i in range(4)})
    assert not S.run_stage1(four, str(tmp_path / "f"))["p5_stops"]                    # 10% exactly does not exceed it
    short = _write_inputs(tmp_path / "short", n_per_level=20, plant=False, n_points=25)
    with pytest.raises(SystemExit, match="not complete"):
        S.run_stage1(short, str(tmp_path / "sh"))


def test_secondary_analysis_and_g3_cells_that_failed(tmp_path):
    inputs = _write_inputs(tmp_path, lost={("nre", "B", 2)})         # the level that holds the Partial pairs here
    out = str(tmp_path / "out")
    S.run_stage1(inputs, out)
    rep = S.run(inputs, out, n_boot=60)
    hdr, pairs, _, _ = S.load_refs(inputs["refs"], inputs["samples"])
    sec = rep["secondary"]
    assert sec["n_pairs"] == sum(1 for p in pairs.values() if p["well_covered"]) and "secondary" in sec["label"]
    assert set(sec["verdicts"]) == set(S.CALIBRATORS) and all(v["null"]["stands"] is None for v in sec["calibrators"].values())
    assert sec["stage1"]["n_accepted"] == sec["n_pairs"] and all(isinstance(r["stands"], bool) for r in rep["null"].values())
    nre, ens = rep["stage2"]["nre"]["hypotheses"], rep["stage2"]["ensemble"]["hypotheses"]
    assert nre["H-b"]["levels_pooled"] == [1, 3] and nre["H-c"]["levels_pooled"] == [1, 3]       # E needs the B network
    assert nre["H-a"]["levels_pooled"] == [1, 2, 3] and nre["H-d"]["levels_pooled"] == [1, 2, 3]  # A's networks all passed
    assert ens["H-b"]["levels_pooled"] == [1, 2, 3] and rep["g3_levels_lost"] == {"nre_B": [2]}
    assert "lost" in nre["H-c"]["per_level"]["0.003"] and "lost" not in nre["H-d"]["per_level"].get("0.003", {})
    n_partial = sum(1 for p in pairs.values() if S.bin_of(p["c"]) == "Partial")
    n_partial_13 = sum(1 for p in pairs.values() if S.bin_of(p["c"]) == "Partial" and p["j"] in (1, 3))
    assert nre["H-b"]["n_pairs"] == n_partial_13 < n_partial == ens["H-b"]["n_pairs"]
    # the floors are taken on the kept levels' pairs: too few Partial pairs at levels 1 and 3 for the ratio estimator
    assert n_partial_13 < 100 and nre["H-b"]["failure_floor"]["n_pairs"] == n_partial_13
    assert nre["H-b"]["verdict"] == "Untestable" and nre["H-b"]["untestable_reason"] == "calibrator failure floor"
    assert rep["stage2"]["nre"]["failure_floor"]["Partial"]["n_pairs"] == n_partial               # every level, reported
    assert nre["H-c"]["failure_floor"]["n_pairs"] == nre["H-c"]["n_pairs"] >= 100 and nre["H-c"]["verdict"] != "Untestable"
    assert ens["H-b"]["verdict"] != "Untestable" and ens["H-b"]["failure_floor"]["n_pairs"] == n_partial
    # every level of the ratio estimator under A lost: its hypotheses are untestable for it, the ensemble's not
    allA = _write_inputs(tmp_path / "allA", lost={("nre", "A", 1), ("nre", "A", 2), ("nre", "A", 3)})
    S.run_stage1(allA, str(tmp_path / "a"))
    repA = S.run(allA, str(tmp_path / "a"), n_boot=40)
    assert all(r["verdict"] == "Untestable" and "G3" in r["untestable_reason"] for r in repA["stage2"]["nre"]["hypotheses"].values())
    assert repA["stage2"]["ensemble"]["hypotheses"]["H-a"]["verdict"] != "Untestable" and not repA["null"]["nre"]["stands"]
    assert repA["dose_response"]["nre"]["n_pairs"] == 0 and repA["dose_response"]["nre"]["spearman"] is None   # no cell left
    # scores for a cell that failed G3, or none for a cell that passed, are refused
    g3 = json.loads(Path(inputs["g3_report"]).read_text(encoding="utf-8"))
    g3["cells"]["ensemble_A_j1"]["passed"] = False
    Path(inputs["g3_report"]).write_text(json.dumps(g3), encoding="utf-8")
    with pytest.raises(SystemExit, match="failed G3"):
        S.run(inputs, out, n_boot=5)
    g3["cells"]["ensemble_A_j1"]["passed"], g3["cells"]["nre_B_j2"]["passed"] = True, True
    Path(inputs["g3_report"]).write_text(json.dumps(g3), encoding="utf-8")
    with pytest.raises(SystemExit, match="passed G3"):
        S.run(inputs, out, n_boot=5)
    del g3["cells"]["nre_B_j2"]
    Path(inputs["g3_report"]).write_text(json.dumps(g3), encoding="utf-8")
    with pytest.raises(SystemExit, match="no cell nre_B_j2"):
        S.run(inputs, out, n_boot=5)


def test_entry_point_records_and_guards_the_versions(tmp_path, monkeypatch, capsys):
    """Run from the command line (check_versions), the script refuses numpy and scipy other
    than the core interpreter's unless --allow-other-versions, which is recorded (D81);
    called from a test, nothing is refused."""
    import inspect
    import p5_g1_emulator as EM
    assert "main(check_versions=True)" in inspect.getsource(S).split("__main__")[-1]
    monkeypatch.setattr(EM, "CORE_VERSIONS", {"numpy": "0.0.0", "scipy": "0.0.0"})
    S.main([])                                                                          # a call, not a command-line run
    with pytest.raises(SystemExit, match="refused: numpy .* is installed, 0.0.0 is expected"):
        S.main([], check_versions=True)
    S.main(["--allow-other-versions"], check_versions=True)                            # the override runs
    inputs = _write_inputs(tmp_path, n_per_level=20, plant=False)
    S.main(["--stage1", "--refs", inputs["refs"], "--out-dir", str(tmp_path / "o"), "--allow-other-versions"],
           check_versions=True)
    rep = json.loads(Path(tmp_path, "o", "p5_stage1_report.json").read_text(encoding="utf-8"))
    assert rep["versions"]["override"] is True and rep["versions"]["expected"] == {"numpy": "0.0.0", "scipy": "0.0.0"}
    assert rep["versions"]["mismatch"]["numpy"][1] == "0.0.0" and rep["numpy"] == np.__version__
    assert "HOLD" in capsys.readouterr().out


def test_stage1_from_the_command_line_with_an_empty_bin(tmp_path, capsys):
    """A bin with no pair (here every c at 0.95, so Partial and Flat are empty) is reported as
    0 pairs well covered of 0, its share left undefined, and the command line prints it
    without failing; the bin is untestable under the floor (D81)."""
    inputs = _write_inputs(tmp_path, n_per_level=40, plant=False)
    text = Path(inputs["refs"]).read_text(encoding="utf-8").splitlines()
    recs = [json.loads(l) for l in text]
    for r in recs:
        if r.get("kind") == "task":
            r["c_A"] = 0.95
    Path(inputs["refs"]).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    S.main(["--stage1", "--refs", inputs["refs"], "--out-dir", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "Identified 120" in out and "Partial 0" in out and "well covered n/a" in out
    rep = json.loads(Path(tmp_path, "o", "p5_stage1_report.json").read_text(encoding="utf-8"))
    for b in ("Partial", "Flat"):
        v = rep["stage1"]["bins"][b]
        assert v["count"] == 0 and v["n_well_covered"] == 0 and v["share_well_covered"] is None and v["untestable"]


def test_the_emulator_hashes_must_agree_where_records_meet(tmp_path):
    """Line 142 (D82): Stage 1 refuses references whose header carries no emulator hash, or
    one other than G2's, records the hash and writes nothing over; Stage 2 refuses when the
    references, the scores, G3's report and Stage 1's output do not carry one hash, and
    records it. The dose-response is also reported on the well-covered pairs (line 134)."""
    inputs = _write_inputs(tmp_path, n_per_level=40, plant=False)
    lines = Path(inputs["refs"]).read_text(encoding="utf-8").splitlines()
    hdr = json.loads(lines[0])

    def with_header(h):
        Path(inputs["refs"]).write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n", encoding="utf-8")
    with_header(dict(hdr, emulator_sha256=None))
    with pytest.raises(SystemExit, match="Stage 1 refused: no emulator hash in test references"):
        S.run_stage1(inputs, str(tmp_path / "o1"))
    with_header(dict(hdr, g2={"passed": True, "emulator_sha256": "f" * 64}))
    with pytest.raises(SystemExit, match="Stage 1 refused: the emulator hashes differ"):
        S.run_stage1(inputs, str(tmp_path / "o1"))
    with_header(hdr)
    out = str(tmp_path / "o")
    assert S.run_stage1(inputs, out)["emulator"]["sha256"] == EM_SHA
    with pytest.raises(SystemExit, match="Stage 1 refused: .* exists; a re-run needs the old file moved aside"):
        S.run_stage1(inputs, out)
    s_lines = Path(inputs["scores"]).read_text(encoding="utf-8").splitlines()
    s_other = dict(json.loads(s_lines[0]), emulator={"path": "x", "sha256": "f" * 64})
    Path(inputs["scores"]).write_text("\n".join([json.dumps(s_other)] + s_lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Stage 2 refused: the emulator hashes differ"):
        S.run(inputs, out, n_boot=5)
    Path(inputs["scores"]).write_text("\n".join(s_lines) + "\n", encoding="utf-8")
    g3 = json.loads(Path(inputs["g3_report"]).read_text(encoding="utf-8"))
    Path(inputs["g3_report"]).write_text(json.dumps(dict(g3, emulator=None)), encoding="utf-8")
    with pytest.raises(SystemExit, match="Stage 2 refused: no emulator hash in G3 report"):
        S.run(inputs, out, n_boot=5)
    Path(inputs["g3_report"]).write_text(json.dumps(g3), encoding="utf-8")
    rep = S.run(inputs, out, n_boot=20)
    assert rep["emulator"]["sha256"] == EM_SHA
    _, pairs, _, _ = S.load_refs(inputs["refs"], inputs["samples"])
    n_wc = sum(1 for p in pairs.values() if p["well_covered"])
    dr = rep["secondary"]["dose_response"]
    assert set(dr) == set(S.CALIBRATORS) and all(v["n_pairs"] <= n_wc for v in dr.values())
    assert "about 160 at n = 160,000" in rep["secondary"]["covered_means"]
