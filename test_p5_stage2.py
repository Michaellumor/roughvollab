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
    assert S.DEFAULT_INPUTS["refs"].endswith("p5_ref_test.jsonl") and S.DEFAULT_INPUTS["scores"].endswith("p5_scores_test.jsonl")


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


def _write_inputs(tmp_path, n_per_level=N_SYNTH, slots=(1, 2, 3, 4, 5), seed=5, plant=True):
    """Synthetic reference and score files in the scripts' formats: c spread over the three
    bins by level, the networks' Ĥ near the exact means with a prior-dependent bias in the
    weaker bins, one reference failure and one network failure (one θ's three pairs,
    for the ratio estimator under B at slot 2) planted."""
    rng = np.random.default_rng(seed)
    refs_path, samples_path, scores_path = (str(tmp_path / n) for n in ("refs.jsonl", "refs.npz", "scores.jsonl"))
    append_line(refs_path, {"kind": "header", "set": "test", "n_points": n_per_level, "seeds": {"points": 20261002, "noise": 1}})
    append_line(scores_path, {"kind": "header", "mode": "score", "set": "test", "threads": 1})
    stores = {}
    pairs = []
    for i in range(n_per_level):
        theta = [0.02 + 0.46 * rng.beta(2, 2), 0.2, -0.8, 0.1]
        for j, s in enumerate(S.NOISE_LEVELS):
            c = float(np.clip(rng.normal((0.95, 0.70, 0.25)[j], 0.05), -0.1, 0.999))
            lab = f"test_{i:04d}_{j}"
            mA, mB = theta[0] + rng.normal(0, 0.002), theta[0] + (1 - c) * 0.2 + rng.normal(0, 0.002)
            sd = 0.054 * np.sqrt(max(1 - c, 1e-3))
            status = "reference_failure" if (plant and i == 3 and j == 1) else "ok"
            rec = {"kind": "task", "task": lab, "set": "test", "i": i, "j": j, "s": float(s), "status": status, "theta": theta,
                   "y": [0.2] * 35, "c_A": c, "seed": [83, i, j], "runs": {}, "kish": {}, "direct": [],
                   "under": {"A": {"mean_H": mA, "var_H": sd ** 2, "q05_H": mA - 1.645 * sd, "q95_H": mA + 1.645 * sd,
                                   "mean_xi0": 0.1, "source": "reweighted"},
                             "B": {"mean_H": mB, "var_H": sd ** 2, "q05_H": 0, "q95_H": 1, "mean_xi0": 0.1, "source": "reweighted"}},
                   "seconds": 1.0}
            append_line(refs_path, rec)
            if status == "ok":
                stores[f"{lab}_u"] = R.quantise_H(np.clip(rng.normal(mA, sd, 20000), 0.02, 0.48))
            pairs.append((lab, i, j, c, mA, mB, sd))
    np.savez_compressed(samples_path, **stores)
    for cal in S.CALIBRATORS:
        for q in slots:
            for lab, i, j, c, mA, mB, sd in pairs:
                for prior, mean in (("A", mA), ("B", mB)):
                    run_label = f"{cal}_p{1 if prior == 'A' else 2}_j{j + 1}_q{q}_n10000"     # the producer's run label
                    if plant and cal == "nre" and prior == "B" and q == 2 and i == 5:
                        append_line(scores_path, {"kind": "task", "task": f"{run_label}|{lab}", "run": run_label, "pair": lab,
                                                  "calibrator": cal, "prior": prior, "j": j + 1, "q": q, "i": i, "status": "failure"})
                        continue
                    leak = 0.0 if c >= 0.9 else 0.08 * (1 - c)                        # a prior-dependent bias off the identified bin
                    H_hat = mean + (leak if prior == "B" else -leak) + rng.normal(0, 0.001)
                    width = 1.645 * sd * (1.0 if c >= 0.5 else 0.7)                   # too narrow in the flat bin
                    append_line(scores_path, {"kind": "task", "task": f"{run_label}|{lab}", "run": run_label, "pair": lab,
                                              "calibrator": cal, "prior": prior, "j": j + 1, "q": q, "i": i, "status": "ok",
                                              "H_hat": H_hat, "xi0_hat": 0.1, "lo": H_hat - width, "hi": H_hat + width})
    return {"refs": refs_path, "samples": samples_path, "scores": scores_path}


def test_end_to_end_on_synthetic_inputs(tmp_path, capsys):
    inputs = _write_inputs(tmp_path)
    out_dir = str(tmp_path / "out")
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
    rep = S.run(inputs, str(tmp_path / "out"), n_boot=50)
    for cal in S.CALIBRATORS:
        assert all(r["verdict"] == "Untestable" and r["untestable_reason"] == "stage 1 floor" for r in rep["stage2"][cal]["hypotheses"].values())
        assert not rep["null"][cal]["stands"] and rep["null"][cal]["all_four_refuted"] is False
    assert all(v["untestable"] for v in rep["stage1"]["bins"].values())
