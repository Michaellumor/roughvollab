"""
p5_stage2.py — P5 Stage 1 and Stage 2: from the reference's records and the calibrators'
scores to the eight verdicts (docs/protocols/P5_protocol_v2_draft.md, Stratification lines
79 to 89 (D70), Stage 1 lines 97 to 101, Stage 2 lines 103 to 113, Hypotheses lines 115 to
128 (D71), Analysis plan lines 130 to 135).

Inputs: the sealed test set's references (p5_g2_reference.py --test: per pair the status,
c under A, the posterior means of H and ξ₀ under A and B, the stored H samples) and the
calibrators' scores (p5_g3_calibrators.py --score test: per calibrator, training prior,
seed slot q and pair, Ĥ, the ξ₀ estimate, the 90% interval, or a failure).

Stage 1 (lines 79 to 101): each accepted pair binned by c = 1 − Var_post(H | y) / Var_prior(H)
under prior A, Identified at c ≥ 0.9, Partial at 0.5 ≤ c < 0.9, Flat below (a negative c
under Flat and counted); the pairs within 0.02 of a cut-off counted per bin; the shares
pooled and per level; the distinct θ per bin; the floor: a bin with fewer than 10% of the
accepted pairs pooled over the levels, or fewer than 100, makes its hypotheses untestable
for both calibrators; a reference failure removes its pair and is counted.

Stage 2 (lines 105 to 113): per calibrator, seed q and pair, d_P = Ĥ_net,P − Ĥ_exact,P;
S = (Ĥ_B − Ĥ_A) / 0.20 for the network (the A- and B-trained networks of the same seed
slot) and for the reference; E = S_net − S_exact; m = the exact posterior probability
under A of the A-trained network's 90% interval, from the stored samples. A failed
network's pair is excluded at that seed from every measure that uses its output, S and E
included, and kept in the measures that use only the other prior's network; a calibrator
whose failed pairs in a bin exceed 10% of the bin's pairs, or leave fewer than 100, is
untestable in that bin (line 75). The seed floor per prior and bin: the median over the
bin's pairs and the unordered pairs of seeds of abs(S) between two networks trained on
that prior; the larger prior's value; a seed pair with a failed network left out for that
pair; fixed, recomputed on each shifted bin.

Hypotheses (lines 119 to 128): H-a, the median of abs(d) in Identified, expected below
0.01, refuted above 0.03; H-b, the median E in Partial, above 0.15, refuted below 0.05;
H-c, the median E in Identified, above 0.05 and above the seed floor, refuted below 0.02;
H-d, the median m in Flat, below 0.80, refuted above 0.85, an interval wholly above 0.95
Refuted on that row with the share of cells above 0.95 reported. Scored per calibrator,
pooled over levels, a seed slot one unit across the levels; the per-level breakdown
reported without verdicts; the eight verdicts read individually at 95% without
adjustment; the null per calibrator when all four are Refuted and H-d's interval lies
within [0.85, 0.95], an Inconclusive or untestable outcome leaving it unproven.

Analysis plan (lines 132 to 134): the point estimate is the median over the bin's
(pair, seed) cells; the crossed bootstrap resamples the seed indices (five, or the four a
cell has) and the bin's pair indices independently with replacement, a cell entering with
the product of its multiplicities, and takes the weighted median; 2,000 replicates from
default_rng(127) (the seed is this script's, declared in D75); the 2.5 and 97.5
percentiles; the per-seed medians beside each interval. Confirmed: the whole interval on
the expected side of the expected value; Refuted: the whole interval beyond the
refutation threshold; Inconclusive otherwise. The re-scoring moves both cut-offs by
+0.05 and by −0.05 with the floors applied to each shifted bin, reported and deciding
nothing. The dose-response: per calibrator, pooled, Spearman's correlation over the
accepted pairs of c with the median over seeds of abs(d), its interval by the same
bootstrap, descriptive. Line 135's exploratory items are not implemented here.

Cost and memory, by arithmetic: the store holds every accepted pair's 160,000 16-bit samples
(about 1 GB at 3,000 pairs, read into memory); the prior-A weights are computed once per pair;
the bootstraps are 2,000 replicates per row and the dose-response's per-pair medians are
vectorised, so a run at N = 1,000 test θ is minutes, not hours.

Usage (from the repository root, after the runs):
    py -3 p5_stage2.py              the design; holds
    py -3 p5_stage2.py --run        the analysis; writes output/p5_stage1.csv, output/p5_stage1.png,
                                    output/p5_stage2_cells.csv, output/p5_stage2_report.json
    options: --refs PATH --samples PATH --scores PATH --out-dir DIR (the test set's files by default)
"""
import os
import sys
import csv
import json
import argparse
import platform
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g0_pilot as P
import p5_g2_reference as R
from p5_g0_pilot import git_state, utc_now, read_jsonl, _rel
from p5_g1_emulator import NOISE_LEVELS

# --------------------------------------------------------------------------- #
# The literals (lines 85 to 89, 101, 108, 111, 121 to 124, 128, 132 to 134)
# --------------------------------------------------------------------------- #
CUTS = (0.9, 0.5)                                   # Identified at or above the first; Flat below the second
SHIFTS = (0.05, -0.05)                              # the re-scoring: both cut-offs moved together
WINDOW = 0.02                                       # pairs within it of a cut-off are counted
FLOOR_SHARE, FLOOR_COUNT = 0.10, 100                # Stage 1's floor (pooled) and line 75's per-calibrator floor
GAP = 0.20                                          # the shift ratio's denominator
N_SEEDS = 5
N_BOOT, SEED_BOOT, CI = 2000, 127, (2.5, 97.5)
OVERWIDE, NULL_BAND = 0.95, (0.85, 0.95)
CALIBRATORS, PRIORS, BINS = ("ensemble", "nre"), ("A", "B"), ("Identified", "Partial", "Flat")
LEVEL_INDEX = {float(s): j + 1 for j, s in enumerate(NOISE_LEVELS)}
HYPOTHESES = {
    "H-a": {"name": "Faithful when informed", "measure": "abs_dA", "bin": "Identified", "direction": "below",
            "expected": 0.01, "refuted": 0.03, "seed_floor": False},
    "H-b": {"name": "Excess prior dependence", "measure": "E", "bin": "Partial", "direction": "above",
            "expected": 0.15, "refuted": 0.05, "seed_floor": False},
    "H-c": {"name": "Leak into the identified regime", "measure": "E", "bin": "Identified", "direction": "above",
            "expected": 0.05, "refuted": 0.02, "seed_floor": True},
    "H-d": {"name": "Interval infidelity", "measure": "m", "bin": "Flat", "direction": "below",
            "expected": 0.80, "refuted": 0.85, "seed_floor": False},
}
MEASURES = ("dA", "dB", "abs_dA", "S_net", "S_exact", "E", "m")

OUT_DIR = P.OUT_DIR
DEFAULT_INPUTS = {"refs": R.SETS["test"]["jsonl"], "samples": R.SETS["test"]["samples"],
                  "scores": os.path.join(OUT_DIR, "p5_scores_test.jsonl")}


def out_paths(out_dir=None):
    d = out_dir or OUT_DIR
    return {"stage1_csv": os.path.join(d, "p5_stage1.csv"), "stage1_png": os.path.join(d, "p5_stage1.png"),
            "cells_csv": os.path.join(d, "p5_stage2_cells.csv"), "report": os.path.join(d, "p5_stage2_report.json")}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_refs(jsonl_path, samples_path):
    """The reference records by pair label, their stored samples, and the failures."""
    hdr, launches, tasks, stores = R.load_references(jsonl_path, samples_path)
    pairs, failures = {}, []
    for lab, rec in sorted(tasks.items()):
        if rec.get("status") != "ok":
            failures.append(lab)
            continue
        if not R.samples_complete(rec, stores):
            raise SystemExit(f"refs refused: no stored samples for {lab} (status ok); the store is incomplete")
        pairs[lab] = {"label": lab, "i": int(rec["i"]), "j": int(rec["j"]) + 1, "s": float(rec["s"]),
                      "theta": [float(v) for v in rec["theta"]], "c": float(rec["c_A"]),
                      "H_exact": {p: float(rec["under"][p]["mean_H"]) for p in PRIORS},
                      "xi0_exact": {p: float(rec["under"][p]["mean_xi0"]) for p in PRIORS},
                      "sources": {p: rec["under"][p]["source"] for p in PRIORS},
                      "var_H_A": float(rec["under"]["A"]["var_H"])}
    return hdr, pairs, stores, failures


def pair_store(stores, label):
    """The pair's samples under the keys weights_for expects ('u', 'A', 'B')."""
    return {k: stores[f"{label}_{k}"] for k in ("u", "A", "B") if f"{label}_{k}" in stores}


def load_scores(jsonl_path):
    """Score rows by (calibrator, prior, q, pair); the header; the seed slots per calibrator;
    the (calibrator, prior, level, slot) combinations that hold any row. The producer's task
    labels are unique per (run, pair) and a malformed line (a crash mid-write) is skipped
    and re-scored on resume, as read_jsonl allows; two well-formed rows under one label,
    which would silently replace each other, are refused."""
    hdr, tasks, seen = None, {}, []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("kind") == "header":
                hdr = obj
            elif obj.get("kind") == "task":
                seen.append(obj["task"])
                tasks[obj["task"]] = obj
    dup = sorted({t for t in seen if seen.count(t) > 1})
    if dup:
        raise SystemExit(f"scores refused: {_rel(jsonl_path)} holds more than one row under {dup[:5]}"
                         + (f" and {len(dup) - 5} more" if len(dup) > 5 else ""))
    rows, slots, present = {}, {}, set()
    for rec in tasks.values():
        key = (rec["calibrator"], rec["prior"], int(rec["q"]), rec["pair"])
        if key in rows:
            raise SystemExit(f"scores refused: two rows for {key}")
        rows[key] = rec
        slots.setdefault(rec["calibrator"], set()).add(int(rec["q"]))
        present.add((rec["calibrator"], rec["prior"], int(rec["j"]), int(rec["q"])))
    return hdr, rows, {k: tuple(sorted(v)) for k, v in slots.items()}, present


# --------------------------------------------------------------------------- #
# Stage 1 (lines 79 to 101)
# --------------------------------------------------------------------------- #
def bin_of(c, cuts=CUTS):
    hi, lo = cuts
    return "Identified" if c >= hi else ("Partial" if c >= lo else "Flat")


def stage1(pairs, cuts=CUTS, window=WINDOW, floor_share=FLOOR_SHARE, floor_count=FLOOR_COUNT, n_failures=0):
    """Bins, counts, shares, the window counts, negative c, distinct θ, the floor."""
    bins = {lab: bin_of(p["c"], cuts) for lab, p in pairs.items()}
    n = len(pairs)
    out = {"cuts": list(cuts), "n_accepted": n, "n_reference_failures": int(n_failures),
           "n_negative_c": int(sum(1 for p in pairs.values() if p["c"] < 0)), "bins": {}}
    for b in BINS:
        labs = [lab for lab, bb in bins.items() if bb == b]
        per_level = {str(s): sum(1 for lab in labs if pairs[lab]["j"] == LEVEL_INDEX[float(s)]) for s in NOISE_LEVELS}
        thetas = {pairs[lab]["i"] for lab in labs}
        count = len(labs)
        share = count / n if n else 0.0
        untestable = bool(count < floor_count or share < floor_share)
        out["bins"][b] = {"count": count, "share": share, "per_level": per_level, "n_distinct_theta": len(thetas),
                          "untestable": untestable,
                          "floor": {"share_min": floor_share, "count_min": floor_count}}
    out["near_cutoff"] = {f"{cut:g}": {"at_or_above": int(sum(1 for p in pairs.values() if 0 <= p["c"] - cut <= window)),
                                       "below": int(sum(1 for p in pairs.values() if 0 < cut - p["c"] <= window))}
                          for cut in cuts}                                  # per bin: the side of the cut-off is the bin
    out["window"] = window
    out["c_by_level"] = {}
    for s in NOISE_LEVELS:
        j = LEVEL_INDEX[float(s)]
        cs = [p["c"] for p in pairs.values() if p["j"] == j]
        out["c_by_level"][str(s)] = {"n": len(cs), "quartiles": [float(v) for v in np.percentile(cs, [25, 50, 75])] if cs else None}
    return out, bins


def write_stage1(pairs, bins, path_csv, path_png=None):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "i", "j", "s", "H", "nu", "rho", "xi0", "c_A", "bin"])
        for lab, p in sorted(pairs.items()):
            w.writerow([lab, p["i"], p["j"], p["s"], *[repr(float(v)) for v in p["theta"]], repr(float(p["c"])), bins[lab]])
    if path_png and _HAS_MPL and pairs:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for s in NOISE_LEVELS:
            cs = [p["c"] for p in pairs.values() if p["j"] == LEVEL_INDEX[float(s)]]
            if cs:
                axes[0].hist(cs, bins=40, range=(min(-0.2, min(cs)), 1.0), alpha=0.5, label=f"s = {s:g}")
        axes[0].set_xlabel("c under A")
        axes[0].set_ylabel("pairs")
        axes[0].legend()
        H = [p["theta"][0] for p in pairs.values()]
        nu = [p["theta"][1] for p in pairs.values()]
        c = [p["c"] for p in pairs.values()]
        for ax, x, name in ((axes[1], H, "true H"), (axes[2], nu, "true ν")):
            ax.scatter(x, c, s=4, alpha=0.4)
            ax.set_xlabel(name)
            ax.set_ylabel("c under A")
            for cut in CUTS:
                ax.axhline(cut, color="grey", lw=0.6)
        fig.tight_layout()
        fig.savefig(path_png, dpi=120)
        plt.close(fig)
        return True
    return False


# --------------------------------------------------------------------------- #
# Stage 2 measures (lines 105 to 111) and line 75's exclusions
# --------------------------------------------------------------------------- #
def measures(pairs, scores, stores, slots, calibrators=CALIBRATORS, gap=GAP, present=None):
    """One cell per (calibrator, seed slot q, pair) with the measures that its networks'
    statuses allow: dA and m from the A-trained network, dB from the B-trained, S_net and
    E from both at the same slot; S_exact from the reference. A failed network's measures
    are None with the failure recorded. A slot with no rows at all for that (calibrator,
    prior, level) is vacant (line 75: a cell left with four seeds): its cells are excluded
    but not counted as failures. A row missing while its slot has rows at that level is
    an incomplete scoring and is refused."""
    cells = {}
    weights_cache = {}                                      # (H, w) under A per pair, computed once and reused

    def weights_A(lab):
        if lab not in weights_cache:
            p = pairs[lab]
            weights_cache[lab] = R.weights_for({"under": {"A": {"source": p["sources"]["A"]}}}, pair_store(stores, lab), "A")
        return weights_cache[lab]

    for cal in calibrators:
        for q in slots.get(cal, ()):
            for lab, p in pairs.items():
                rA, rB = scores.get((cal, "A", q, lab)), scores.get((cal, "B", q, lab))
                vacant = ""
                for prior, r in (("A", rA), ("B", rB)):
                    if r is None:
                        if present is not None and (cal, prior, p["j"], int(q)) not in present:
                            vacant += prior
                        else:
                            raise SystemExit(f"scores refused: no row for {(cal, prior, q, lab)} although the slot is "
                                             "scored at that level; the scoring is incomplete")
                okA = rA is not None and rA.get("status") == "ok"
                okB = rB is not None and rB.get("status") == "ok"
                S_exact = (p["H_exact"]["B"] - p["H_exact"]["A"]) / gap
                failed = "".join(pr for pr, r in (("A", rA), ("B", rB)) if r is not None and r.get("status") != "ok")
                cell = {"calibrator": cal, "q": int(q), "pair": lab, "bin": None, "s": p["s"], "j": p["j"], "i": p["i"],
                        "okA": okA, "okB": okB, "S_exact": S_exact, "failure": failed or None, "vacant": vacant or None,
                        "dA": None, "dB": None, "abs_dA": None, "S_net": None, "E": None, "m": None,
                        "H_hat_A": None, "H_hat_B": None, "lo_A": None, "hi_A": None}
                if okA:
                    cell["H_hat_A"], cell["lo_A"], cell["hi_A"] = float(rA["H_hat"]), float(rA["lo"]), float(rA["hi"])
                    cell["dA"] = cell["H_hat_A"] - p["H_exact"]["A"]
                    cell["abs_dA"] = abs(cell["dA"])
                    H, w = weights_A(lab)
                    cell["m"] = R.interval_prob(H, w, cell["lo_A"], cell["hi_A"])
                if okB:
                    cell["H_hat_B"] = float(rB["H_hat"])
                    cell["dB"] = cell["H_hat_B"] - p["H_exact"]["B"]
                if okA and okB:
                    cell["S_net"] = (cell["H_hat_B"] - cell["H_hat_A"]) / gap
                    cell["E"] = cell["S_net"] - S_exact
                cells[(cal, int(q), lab)] = cell
    return cells


def failure_floor(cells, bins, calibrator, bin_name, slots, floor_share=FLOOR_SHARE, floor_count=FLOOR_COUNT):
    """Line 75: a calibrator's failed pairs in a bin, read as the pairs on which any of its
    networks failed at any seed (declared in D75; the per-cell count is reported beside
    it), above 10% of the bin's pairs, or leaving fewer than 100, make the bin untestable
    for that calibrator. A vacant slot is not a failure."""
    labs = [lab for lab, b in bins.items() if b == bin_name]
    failed = [lab for lab in labs if any(cells[(calibrator, q, lab)]["failure"] for q in slots)]
    n, nf = len(labs), len(failed)
    n_cells_failed = sum(1 for lab in labs for q in slots if cells[(calibrator, q, lab)]["failure"])
    return {"n_pairs": n, "n_failed": nf, "share_failed": nf / n if n else 0.0, "n_failed_cells": n_cells_failed,
            "n_cells": n * len(slots), "reading": "a pair failed at any seed",
            "untestable": bool(n and (nf > floor_share * n or n - nf < floor_count)) or not n, "failed": failed}


def seed_floor(cells, bins, bin_name, calibrator, slots, gap=GAP):
    """Line 111: per prior the median over the bin's pairs and the unordered seed pairs of
    abs(S) between two networks trained on that prior; the floor the larger prior's value."""
    labs = [lab for lab, b in bins.items() if b == bin_name]
    out = {}
    for prior in PRIORS:
        key = "H_hat_A" if prior == "A" else "H_hat_B"
        vals = []
        for lab in labs:
            for q1, q2 in combinations(slots, 2):
                a, b = cells[(calibrator, q1, lab)][key], cells[(calibrator, q2, lab)][key]
                if a is not None and b is not None:
                    vals.append(abs(a - b) / gap)
        out[prior] = float(np.median(vals)) if vals else None
    present = [v for v in out.values() if v is not None]
    return {"per_prior": out, "floor": max(present) if present else None, "n_pairs": len(labs)}


# --------------------------------------------------------------------------- #
# The statistic, the crossed bootstrap, the verdicts (lines 132, 133)
# --------------------------------------------------------------------------- #
def weighted_median(values, weights):
    """The smallest value whose cumulative weight reaches half the total (the lower
    weighted median; with unit weights and an even count, the lower of the two middles)."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    keep = w > 0
    v, w = v[keep], w[keep]
    if v.size == 0:
        return None
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cum, 0.5 * cum[-1], side="left")])


def rowwise_weighted_median(M, l_seed):
    """The lower weighted median of each row of M [n, m] under the seed weights l_seed [m],
    NaN cells carrying no weight; NaN for a row with no weight. Equals weighted_median
    row by row (a sort along the row, NaN last; the cumulative weight; the first column
    at which it reaches half the total), vectorised for the dose-response's replicates."""
    M = np.asarray(M, dtype=float)
    finite = np.isfinite(M)
    order = np.argsort(np.where(finite, M, np.inf), axis=1, kind="stable")
    Ms = np.take_along_axis(M, order, axis=1)
    W = np.take_along_axis(finite * np.asarray(l_seed, dtype=float)[None, :], order, axis=1)
    cum = np.cumsum(W, axis=1)
    total = cum[:, -1]
    idx = np.argmax(cum >= 0.5 * total[:, None], axis=1)
    out = Ms[np.arange(M.shape[0]), idx]
    out[total <= 0] = np.nan
    return out


def cell_matrix(cells, calibrator, measure, labs, slots):
    """[n_pairs, n_seeds] of the measure, NaN where the cell is excluded."""
    M = np.full((len(labs), len(slots)), np.nan)
    for a, lab in enumerate(labs):
        for b, q in enumerate(slots):
            v = cells[(calibrator, q, lab)][measure]
            if v is not None:
                M[a, b] = v
    return M


def point_and_seed_medians(M):
    """The point estimate and the per-seed medians, by the same lower median as the
    replicates (unit weights), on the same cells."""
    vals = M[np.isfinite(M)]
    point = weighted_median(vals, np.ones(vals.size)) if vals.size else None
    per_seed = []
    for b in range(M.shape[1]):
        col = M[:, b][np.isfinite(M[:, b])]
        per_seed.append(weighted_median(col, np.ones(col.size)) if col.size else None)
    return point, per_seed, int(vals.size)


def bootstrap_median(M, rng, n_boot=N_BOOT, ci=CI):
    """The crossed bootstrap of line 132 on the live cells: the pairs with at least one
    finite cell and the seed columns with at least one finite cell (a dead slot is not
    resampled: "the four a cell has been left with"); pair indices and seed indices
    resampled independently with replacement; a cell enters with the product of its
    multiplicities; the lower weighted median per replicate; the percentile interval.
    Returns (interval, replicates, info) with the counts of live pairs and seeds and of
    replicates without a live cell (dropped)."""
    finite = np.isfinite(M)
    live_p, live_s = finite.any(axis=1), finite.any(axis=0)
    M, finite = M[np.ix_(live_p, live_s)], finite[np.ix_(live_p, live_s)]
    info = {"n_live_pairs": int(live_p.sum()), "n_live_seeds": int(live_s.sum()), "n_dropped_replicates": 0}
    n_pairs, n_seeds = M.shape
    if not finite.any():
        return None, [], info
    reps = np.empty(n_boot)
    for r in range(n_boot):
        k_pair = np.bincount(rng.integers(0, n_pairs, n_pairs), minlength=n_pairs)
        l_seed = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
        w = np.outer(k_pair, l_seed).astype(float) * finite
        med = weighted_median(M[finite], w[finite]) if w[finite].sum() > 0 else None
        reps[r] = np.nan if med is None else med
    ok = np.isfinite(reps)
    info["n_dropped_replicates"] = int((~ok).sum())
    if not ok.any():
        return None, [], info
    lo, hi = np.percentile(reps[ok], ci)
    return [float(lo), float(hi)], reps, info


def verdict(interval, hyp, floor=None):
    """Line 133 on a one-sided row: Confirmed when the whole interval lies on the expected
    side of the expected value (and, for H-c, wholly above the seed floor, which a row
    with no seed floor cannot meet); Refuted when it lies wholly beyond the refutation
    threshold; Inconclusive otherwise."""
    if interval is None:
        return "Untestable"
    lo, hi = interval
    if hyp["direction"] == "below":
        if hi < hyp["expected"]:
            return "Confirmed"
        if lo > hyp["refuted"]:
            return "Refuted"
        return "Inconclusive"
    if lo > hyp["expected"] and (not hyp["seed_floor"] or (floor is not None and lo > floor)):
        return "Confirmed"
    if hi < hyp["refuted"]:
        return "Refuted"
    return "Inconclusive"


def score_hypotheses(pairs, cells, slots, cuts=CUTS, rng_seed=SEED_BOOT, n_boot=N_BOOT, n_failures=0,
                     calibrators=CALIBRATORS, hypotheses=HYPOTHESES):
    """Stage 1 at these cut-offs, then every hypothesis for every calibrator."""
    s1, bins = stage1(pairs, cuts, n_failures=n_failures)
    out = {"cuts": list(cuts), "stage1": s1, "calibrators": {}}
    for ci_, cal in enumerate(calibrators):
        sl = slots.get(cal, ())
        res = {"seed_slots": list(sl), "hypotheses": {}, "failure_floor": {}, "seed_floor": {}}
        for b in BINS:
            res["failure_floor"][b] = failure_floor(cells, bins, cal, b, sl)
            res["seed_floor"][b] = seed_floor(cells, bins, b, cal, sl)
        for hi_, (hid, hyp) in enumerate(hypotheses.items()):
            rng = np.random.default_rng([rng_seed, ci_, hi_])          # one stream per (calibrator, row)
            b = hyp["bin"]
            labs = sorted(lab for lab, bb in bins.items() if bb == b)
            ff = res["failure_floor"][b]
            row = {"name": hyp["name"], "measure": hyp["measure"], "bin": b, "expected": hyp["expected"],
                   "refuted": hyp["refuted"], "direction": hyp["direction"], "n_pairs": len(labs),
                   "n_failed_pairs": ff["n_failed"], "n_failed_cells": ff["n_failed_cells"]}
            untestable = s1["bins"][b]["untestable"] or ff["untestable"]
            row["untestable_reason"] = ("stage 1 floor" if s1["bins"][b]["untestable"] else
                                        "calibrator failure floor" if ff["untestable"] else None)
            if untestable or not sl:
                row.update(point=None, interval=None, per_seed=None, n_cells=0, verdict="Untestable")
            else:
                M = cell_matrix(cells, cal, hyp["measure"], labs, sl)
                point, per_seed, n_cells = point_and_seed_medians(M)
                interval, reps, info = bootstrap_median(M, rng, n_boot)
                fl = res["seed_floor"][b]["floor"] if hyp["seed_floor"] else None
                row.update(point=point, interval=interval, per_seed=per_seed, n_cells=n_cells, bootstrap=info,
                           seed_floor=fl, verdict=verdict(interval, hyp, fl) if point is not None else "Untestable")
                if hyp["measure"] == "m":
                    vals = M[np.isfinite(M)]
                    per_pair = [weighted_median(M[a][np.isfinite(M[a])], np.ones(int(np.isfinite(M[a]).sum())))
                                for a in range(M.shape[0]) if np.isfinite(M[a]).any()]
                    row["share_over_wide"] = float(np.mean([v > OVERWIDE for v in per_pair])) if per_pair else None
                    row["share_over_wide_cells"] = float(np.mean(vals > OVERWIDE)) if vals.size else None
                row["per_level"] = {}
                for s in NOISE_LEVELS:
                    labs_s = [lab for lab in labs if pairs[lab]["j"] == LEVEL_INDEX[float(s)]]
                    if labs_s:
                        Ms = cell_matrix(cells, cal, hyp["measure"], labs_s, sl)
                        pt, _, nc = point_and_seed_medians(Ms)
                        row["per_level"][str(s)] = {"median": pt, "n_cells": nc, "n_pairs": len(labs_s)}
            res["hypotheses"][hid] = row
        hd = res["hypotheses"]["H-d"]
        all_refuted = all(r["verdict"] == "Refuted" for r in res["hypotheses"].values())
        hd_in_band = bool(hd.get("interval") and NULL_BAND[0] <= hd["interval"][0] and hd["interval"][1] <= NULL_BAND[1])
        res["null"] = {"stands": bool(all_refuted and hd_in_band), "all_four_refuted": all_refuted,
                       "H_d_interval_within_band": hd_in_band, "band": list(NULL_BAND),
                       "note": "an Inconclusive or untestable outcome anywhere leaves the null unproven (line 128)"}
        out["calibrators"][cal] = res
    return out, bins


def dose_response(pairs, cells, slots, rng_seed=SEED_BOOT, n_boot=N_BOOT, calibrators=CALIBRATORS, ci=CI):
    """Line 134: per calibrator, pooled, Spearman's correlation over the accepted pairs of c
    with the lower median over seeds of abs(d_A) (a pair with no accepted seed left out
    and counted), its interval by the crossed bootstrap over the live pairs and seeds
    (the per-pair weighted median over the resampled seeds; the correlation on the pairs
    expanded by their multiplicities, ties from the repeats taking average ranks; a
    replicate without variation or with fewer than three pairs dropped and counted).
    Descriptive; no verdict."""
    out = {}
    labs = sorted(pairs)
    for ci_, cal in enumerate(calibrators):
        sl = slots.get(cal, ())
        if not sl:
            out[cal] = None
            continue
        rng = np.random.default_rng([rng_seed, ci_, 9])
        M = cell_matrix(cells, cal, "abs_dA", labs, sl)
        c = np.array([pairs[lab]["c"] for lab in labs])
        keep = np.isfinite(M).any(axis=1)
        n_dropped_pairs = int((~keep).sum())
        M, c = M[keep], c[keep]
        live_s = np.isfinite(M).any(axis=0)
        M = M[:, live_s]
        n = M.shape[0]
        med = np.array([weighted_median(M[a][np.isfinite(M[a])], np.ones(int(np.isfinite(M[a]).sum()))) for a in range(n)])
        rho = float(spearmanr(c, med).statistic) if n >= 3 and np.ptp(c) > 0 and np.ptp(med) > 0 else None
        reps, n_dropped = [], 0
        for r in range(n_boot):
            k_pair = np.bincount(rng.integers(0, n, n), minlength=n)
            l_seed = np.bincount(rng.integers(0, M.shape[1], M.shape[1]), minlength=M.shape[1])
            mm = rowwise_weighted_median(M, l_seed)
            idx = np.repeat(np.arange(n), k_pair)
            idx = idx[np.isfinite(mm[idx])]
            if len(idx) >= 3 and np.ptp(c[idx]) > 0 and np.ptp(mm[idx]) > 0:
                reps.append(float(spearmanr(c[idx], mm[idx]).statistic))
            else:
                n_dropped += 1
        interval = [float(v) for v in np.percentile(reps, ci)] if reps else None
        out[cal] = {"spearman": rho, "interval": interval, "n_pairs": int(n), "n_dropped_pairs": n_dropped_pairs,
                    "n_live_seeds": int(live_s.sum()), "expected_sign": "negative", "n_replicates": len(reps),
                    "n_dropped_replicates": n_dropped, "note": "descriptive; no verdict (line 134)"}
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def write_cells(cells, bins, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    cols = ["calibrator", "q", "pair", "i", "j", "s", "bin", "okA", "okB", "failure", "vacant", "H_hat_A", "H_hat_B",
            "lo_A", "hi_A", "dA", "dB", "abs_dA", "S_net", "S_exact", "E", "m"]
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key in sorted(cells):
            cell = dict(cells[key], bin=bins.get(cells[key]["pair"]))
            w.writerow(["" if cell[c] is None else (repr(float(cell[c])) if isinstance(cell[c], float) else cell[c]) for c in cols])


def run(inputs=None, out_dir=None, rng_seed=SEED_BOOT, n_boot=N_BOOT, calibrators=CALIBRATORS):
    inputs, paths = dict(DEFAULT_INPUTS, **(inputs or {})), out_paths(out_dir)
    for k, pth in inputs.items():
        if not (os.path.exists(pth) or (k == "samples" and os.path.isdir(R.samples_dir(pth)))):
            raise SystemExit(f"no {k} at {_rel(pth)}: the test set's references and scores follow the runs after freezing")
    ref_hdr, pairs, stores, failures = load_refs(inputs["refs"], inputs["samples"])
    score_hdr, scores, slots, present = load_scores(inputs["scores"])
    if not pairs:
        raise SystemExit("no accepted reference pairs")
    if ref_hdr and score_hdr and ref_hdr.get("set") != score_hdr.get("set"):
        raise SystemExit(f"inputs refused: the references are the set {ref_hdr.get('set')!r} and the scores the set "
                         f"{score_hdr.get('set')!r}")
    n_expected = 3 * int(ref_hdr["n_points"]) if ref_hdr and ref_hdr.get("n_points") else None
    n_missing = (n_expected - len(pairs) - len(failures)) if n_expected is not None else None
    cells = measures(pairs, scores, stores, slots, calibrators, present=present)
    main_res, bins = score_hypotheses(pairs, cells, slots, CUTS, rng_seed, n_boot, len(failures), calibrators)
    shifted = {}
    for sh in SHIFTS:
        cuts = (round(CUTS[0] + sh, 10), round(CUTS[1] + sh, 10))      # 0.95 and 0.55; 0.85 and 0.45, free of float noise
        res, _ = score_hypotheses(pairs, cells, slots, cuts, rng_seed, n_boot, len(failures), calibrators)
        shifted[f"{sh:+.2f}"] = {"cuts": list(cuts), "verdicts": {cal: {h: r["verdict"] for h, r in v["hypotheses"].items()}
                                                                  for cal, v in res["calibrators"].items()},
                                 "stage1": res["stage1"], "calibrators": res["calibrators"],
                                 "note": "the intervals, floors and seed floors recomputed on the shifted bins; reported "
                                         "beside the verdicts; decides nothing (lines 89, 111)"}
    dr = dose_response(pairs, cells, slots, rng_seed, n_boot, calibrators)
    write_stage1(pairs, bins, paths["stage1_csv"], paths["stage1_png"])
    write_cells(cells, bins, paths["cells_csv"])
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"stage1": main_res["stage1"], "stage2": main_res["calibrators"], "re_scoring": shifted,
              "dose_response": dr, "verdicts": {cal: {h: r["verdict"] for h, r in v["hypotheses"].items()}
                                                for cal, v in main_res["calibrators"].items()},
              "null": {cal: v["null"] for cal, v in main_res["calibrators"].items()},
              "multiplicity": "the eight verdicts are read individually at 95% without adjustment (line 126)",
              "bootstrap": {"n_boot": n_boot, "seed": rng_seed, "streams": "default_rng([seed, calibrator index, row index]); "
                                                                            "the dose-response uses row index 9",
                            "interval": "percentile", "ci": list(CI), "median": "the lower weighted median, for the point, "
                                                                                 "the per-seed medians and every replicate",
                            "note": "a percentile interval over five seeds is narrower than nominal; declared, not corrected "
                                    "(line 132); the per-seed medians are reported beside each interval"},
              "m_note": "m is the exact posterior probability under A of the A-trained network's interval (line 110)",
              "n_reference_failures": len(failures), "reference_failures": failures,
              "n_pairs_expected": n_expected, "n_pairs_missing_from_record": n_missing,
              "seed_slots": {cal: list(v) for cal, v in slots.items()},
              "seed_slot_note": "a slot is one unit across levels and priors; a seed-6 replacement sits in the slot it "
                                "replaced, per (prior, level) cell (line 75)",
              "inputs": {k: _rel(v) for k, v in inputs.items()}, "input_headers": {"refs": ref_hdr, "scores": score_hdr},
              "files": {k: _rel(v) for k, v in paths.items()}, "commit": commit, "clean_tree": clean,
              "python": platform.python_version(), "numpy": np.__version__, "utc": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    text = json.dumps(report, indent=1, allow_nan=False)                 # serialised first: no truncated file on an error
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(text)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="P5 Stage 1 and Stage 2 analysis (see module docstring)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--refs", default=DEFAULT_INPUTS["refs"])
    ap.add_argument("--samples", default=DEFAULT_INPUTS["samples"])
    ap.add_argument("--scores", default=DEFAULT_INPUTS["scores"])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    print("=== P5 Stage 1 and Stage 2 ===", flush=True)
    print(f"  cut-offs {CUTS} (re-scored at {SHIFTS}); window {WINDOW}; floor {FLOOR_SHARE:.0%} or {FLOOR_COUNT} pairs; "
          f"gap {GAP}; bootstrap {N_BOOT} replicates from default_rng({SEED_BOOT}), percentile {CI}", flush=True)
    for h, hyp in HYPOTHESES.items():
        print(f"  {h} {hyp['name']}: median {hyp['measure']} in {hyp['bin']}, expected {hyp['direction']} {hyp['expected']}"
              f"{' and above the seed floor' if hyp['seed_floor'] else ''}, refuted "
              f"{'above' if hyp['direction'] == 'below' else 'below'} {hyp['refuted']}", flush=True)
    if not a.run:
        print("\nHOLD: nothing analysed. --run needs the test set's references and scores, after the runs.", flush=True)
        return
    rep = run({"refs": a.refs, "samples": a.samples, "scores": a.scores}, a.out_dir)
    print(f"=== Stage 1: {rep['stage1']['n_accepted']} accepted pairs, {rep['n_reference_failures']} reference failures; bins "
          + ", ".join(f"{b} {v['count']} ({v['share']:.1%}{', untestable' if v['untestable'] else ''})"
                      for b, v in rep["stage1"]["bins"].items()) + " ===", flush=True)
    for cal, vs in rep["verdicts"].items():
        print(f"  {cal}: " + ", ".join(f"{h} {v}" for h, v in vs.items()) + f"; null {'stands' if rep['null'][cal]['stands'] else 'unproven'}",
              flush=True)
    print(f"  written: {rep['files']['report']}", flush=True)


if __name__ == "__main__":
    main()
