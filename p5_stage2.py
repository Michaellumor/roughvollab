"""
p5_stage2.py — P5 Stage 1 and Stage 2: from the reference's records and the calibrators'
scores to the eight verdicts (docs/protocols/P5_protocol_v2_draft.md, Stratification lines
79 to 89 (D70), Stage 1 lines 97 to 101, Stage 2 lines 103 to 113, Hypotheses lines 115 to
128 (D71), Analysis plan lines 130 to 135).

Inputs: the sealed test set's references (p5_g2_reference.py --test: per pair the status,
c under A, the posterior means of H and ξ₀ under A and B, the stored H samples) and the
calibrators' scores (p5_g3_calibrators.py --score test: per calibrator, training prior,
seed slot q and pair, Ĥ, the ξ₀ estimate, the 90% interval, or a failure).

Stage 1 (lines 79 to 101), run alone and first (--stage1), from the references only and
before any score exists (D81): each accepted pair binned by c = 1 − Var_post(H | y) /
Var_prior(H) under prior A, Identified at c ≥ 0.9, Partial at 0.5 ≤ c < 0.9, Flat below (a
negative c under Flat and counted); the pairs within 0.02 of a cut-off counted per bin; the
shares pooled and per level; the distinct θ per bin; the floor: a bin with fewer than 10%
of the accepted pairs pooled over the levels, or fewer than 100, makes its hypotheses
untestable for both calibrators; a reference failure removes its pair and is counted. The
coverage covariate per pair (the mass each prior puts on the 90% interval of H under the
exact posterior under that prior; well covered when both are at least 0.001) and the share
of pairs well covered per bin, with the floors of the secondary set declared beside them.
Reference failures reported by noise level and by true (H, ν); above 10% of the test pairs
at any level, P5 stops at Stage 1 (D81). The output records a digest of the reference
records (refs_digest); the scoring of the test set (p5_g3_calibrators.py --score test)
refuses to start without this output, and records its digest in the scores' header; Stage
2 refuses to run without it, made from the same references, and refuses scores not made
after it or not covering every network their header lists (D81). The references, Stage 1's
output, G3's report and the scores carry G̃'s SHA-256, and Stage 1 and Stage 2 refuse when
the hashes they read are missing or differ; neither writes over an existing report or CSV
(D82).

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
adjustment; the null per calibrator when H-a is Confirmed, H-b, H-c and H-d are Refuted
and H-d's interval lies wholly within [0.85, 0.95], an Inconclusive or untestable outcome
leaving it unproven (line 128, D81). A (calibrator, prior, level) cell that failed G3 has
no networks: every measure needing one loses that level for that calibrator, the other
levels pooled as before, the calibrator's floors taken on the pairs of those levels, and a
hypothesis left with no level is untestable for that calibrator (line 144, D81). The
secondary analysis: H-a to H-d re-scored on the
well-covered pairs only, with the same measures, thresholds, floors (recomputed on that
set), bootstrap streams and verdict rule, reported beside the primary verdicts and
labelled secondary; the null is read on the primary verdicts only (D81).

Analysis plan (lines 132 to 134): the point estimate is the median over the bin's
(pair, seed) cells; the crossed bootstrap resamples the seed indices (five, or the four a
cell has) and the bin's pair indices independently with replacement, a cell entering with
the product of its multiplicities, and takes the weighted median; 2,000 replicates from
default_rng([127, calibrator, row]) (the seed is this script's, declared in D75; the
dose-response's row is 9); the 2.5 and 97.5
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

Run from the command line, the script refuses numpy and scipy other than the core
interpreter's (p5_g1_emulator.CORE_VERSIONS) unless --allow-other-versions, which is
recorded (D81).

Usage (from the repository root, after the runs):
    py -3 p5_stage2.py              the design; holds
    py -3 p5_stage2.py --stage1     Stage 1 from the references alone, before any score: writes
                                    output/p5_stage1_report.json, output/p5_stage1.csv, output/p5_stage1.png
    py -3 p5_stage2.py --run        Stage 2, after Stage 1 and the scoring; writes
                                    output/p5_stage2_cells.csv, output/p5_stage2_report.json
    options: --refs PATH --samples PATH --scores PATH --g3-report PATH --out-dir DIR (the test set's files by default)
"""
import os
import sys
import csv
import json
import hashlib
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
import p5_g1_emulator as EM
import p5_g2_reference as R
from p5_g0_pilot import git_state, utc_now, read_jsonl, _rel
from p5_g1_emulator import NOISE_LEVELS, BOX

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
NEEDS = {"abs_dA": ("A",), "dA": ("A",), "m": ("A",), "dB": ("B",), "E": ("A", "B"), "S_net": ("A", "B")}   # networks per measure
REF_FAIL_MAX = 0.10                                 # reference failures above this share at any level stop P5 (D81)
H_EDGES = np.linspace(BOX["H"][0], BOX["H"][1], 6)  # the failure report's bins in true H and ν (D81)
NU_EDGES = np.linspace(BOX["nu"][0], BOX["nu"][1], 4)
SECONDARY_LABEL = ("secondary: H-a to H-d re-scored on the well-covered pairs only, with the same measures, "
                   "thresholds, floors (recomputed on that set), bootstrap streams and verdict rule; reported beside the "
                   "primary verdicts; the null is read on the primary verdicts only (D81)")

OUT_DIR = P.OUT_DIR
DEFAULT_INPUTS = {"refs": R.SETS["test"]["jsonl"], "samples": R.SETS["test"]["samples"],
                  "scores": os.path.join(OUT_DIR, "p5_scores_test.jsonl"),
                  "g3_report": os.path.join(OUT_DIR, "p5_g3_report.json")}


def out_paths(out_dir=None):
    d = out_dir or OUT_DIR
    return {"stage1_report": os.path.join(d, "p5_stage1_report.json"),
            "stage1_csv": os.path.join(d, "p5_stage1.csv"), "stage1_png": os.path.join(d, "p5_stage1.png"),
            "cells_csv": os.path.join(d, "p5_stage2_cells.csv"), "report": os.path.join(d, "p5_stage2_report.json")}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refs_digest(jsonl_path):
    """The SHA-256 of what Stage 1 reads from the references: the header and every pair's
    record, by label, as canonical JSON. Launch lines are left out, so a re-run of the
    references with nothing left to do, which appends one, does not change it (D81)."""
    hdr, _, tasks = read_jsonl(jsonl_path)
    h = hashlib.sha256(json.dumps(hdr, sort_keys=True, allow_nan=False).encode("utf-8"))
    for lab in sorted(tasks):
        h.update(b"\n" + json.dumps(tasks[lab], sort_keys=True, allow_nan=False).encode("utf-8"))
    return h.hexdigest()


def require_stage1(refs_path=None, out_dir=None):
    """Stage 1's output for these references, or a refusal (line 101, D81): it must exist,
    have been made from these references (by refs_digest) and not have stopped P5. The
    scoring of the test set (p5_g3_calibrators.py --score test) and Stage 2 require it."""
    refs_path, path = refs_path or DEFAULT_INPUTS["refs"], out_paths(out_dir)["stage1_report"]
    if not os.path.exists(path):
        raise SystemExit(f"no Stage 1 output at {_rel(path)}: run p5_stage2.py --stage1 first; Stage 1 is declared "
                         "from the references alone, before any network is scored (line 101)")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("refs_digest") != refs_digest(refs_path):
        raise SystemExit("refused: the Stage 1 output was made from other references than these")
    if rep.get("p5_stops"):
        raise SystemExit("refused: P5 stopped at Stage 1 (reference failures above 10% at a noise level)")
    return rep


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_refs(jsonl_path, samples_path, check_samples=True, details=False):
    """The reference records by pair label, their stored samples, and the failures (their
    labels; with `details`, the records' i, j and θ beside them). The coverage covariate per
    pair is recomputed from the record's 90% intervals and, where the record carries it,
    checked against the record's (D81). With check_samples False (Stage 1, which needs no
    samples) the store is not read."""
    hdr, launches, tasks, stores = R.load_references(jsonl_path, samples_path if check_samples else None)
    pairs, failures, fail_recs = {}, [], []
    for lab, rec in sorted(tasks.items()):
        if rec.get("status") != "ok":
            failures.append(lab)
            fail_recs.append({"label": lab, "i": int(rec["i"]), "j": int(rec["j"]) + 1,
                              "theta": [float(v) for v in rec["theta"]] if rec.get("theta") else None})
            continue
        if check_samples and not R.samples_complete(rec, stores):
            raise SystemExit(f"refs refused: no stored samples for {lab} (status ok); the store is incomplete")
        cov, wc = R.coverage(rec["under"])
        if rec.get("coverage") is not None:
            for p in PRIORS:
                if not np.isclose(float(rec["coverage"][p]), cov[p], rtol=1e-9, atol=1e-12):
                    raise SystemExit(f"refs refused: {lab}'s recorded coverage under {p} differs from its 90% interval's")
        pairs[lab] = {"label": lab, "i": int(rec["i"]), "j": int(rec["j"]) + 1, "s": float(rec["s"]),
                      "theta": [float(v) for v in rec["theta"]], "c": float(rec["c_A"]),
                      "H_exact": {p: float(rec["under"][p]["mean_H"]) for p in PRIORS},
                      "xi0_exact": {p: float(rec["under"][p]["mean_xi0"]) for p in PRIORS},
                      "sources": {p: rec["under"][p]["source"] for p in PRIORS},
                      "var_H_A": float(rec["under"]["A"]["var_H"]), "coverage": cov, "well_covered": wc}
    return (hdr, pairs, stores, failures, fail_recs) if details else (hdr, pairs, stores, failures)


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
        known = [pairs[lab]["well_covered"] for lab in labs if "well_covered" in pairs[lab]]
        n_wc = int(sum(bool(v) for v in known)) if (known or not count) else None    # an empty bin: 0 of 0
        out["bins"][b] = {"count": count, "share": share, "per_level": per_level, "n_distinct_theta": len(thetas),
                          "untestable": untestable,
                          "floor": {"share_min": floor_share, "count_min": floor_count},
                          "n_well_covered": n_wc,                                      # the coverage covariate (D81)
                          "share_well_covered": n_wc / count if (count and n_wc is not None) else None}
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
        w.writerow(["pair", "i", "j", "s", "H", "nu", "rho", "xi0", "c_A", "bin", "cover_A", "cover_B", "well_covered"])
        for lab, p in sorted(pairs.items()):
            cov = p.get("coverage") or {}
            w.writerow([lab, p["i"], p["j"], p["s"], *[repr(float(v)) for v in p["theta"]], repr(float(p["c"])), bins[lab],
                        *["" if cov.get(q) is None else repr(float(cov[q])) for q in PRIORS],
                        "" if "well_covered" not in p else int(bool(p["well_covered"]))])
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


def reference_failure_report(fail_recs, n_accepted_by_level, n_points=None, share_max=REF_FAIL_MAX):
    """Reference failures by noise level and by true (H, ν) (D81): per level the count and
    its share of that level's test pairs (n_points when the record's header gives it, else
    the pairs recorded at that level); the counts on a grid of five bins of true H and
    three of true ν over the box; the failures themselves. Above `share_max` at any level,
    P5 stops at Stage 1."""
    by_level = {}
    for s in NOISE_LEVELS:
        j = LEVEL_INDEX[float(s)]
        nf = sum(1 for f in fail_recs if f["j"] == j)
        denom = int(n_points) if n_points else nf + int(n_accepted_by_level.get(j, 0))
        by_level[str(s)] = {"n_failures": nf, "n_test_pairs": denom, "share": nf / denom if denom else 0.0}
    grid = np.zeros((len(H_EDGES) - 1, len(NU_EDGES) - 1), dtype=int)
    for f in fail_recs:
        if f.get("theta"):
            a = min(max(int(np.searchsorted(H_EDGES, f["theta"][0], side="right")) - 1, 0), grid.shape[0] - 1)
            b = min(max(int(np.searchsorted(NU_EDGES, f["theta"][1], side="right")) - 1, 0), grid.shape[1] - 1)
            grid[a, b] += 1
    over = [s for s, v in by_level.items() if v["share"] > share_max]
    return {"n_failures": len(fail_recs), "by_level": by_level,
            "by_true_H_nu": {"H_edges": [float(v) for v in H_EDGES], "nu_edges": [float(v) for v in NU_EDGES],
                             "counts": grid.tolist()},
            "failures": fail_recs, "share_max": share_max, "levels_over": over, "p5_stops": bool(over),
            "rule": "reference failures above 10% of the test pairs at any noise level stop P5 at Stage 1 (D81)"}


def run_stage1(inputs=None, out_dir=None, versions=None):
    """Stage 1 alone, from the references only, before any score exists (line 101, D81):
    the bins, counts, shares, window counts, negative c, distinct θ, the floor and the
    untestable declarations; the coverage covariate and the share of pairs well covered per
    bin, with Stage 1 on the well-covered set declared beside it for the secondary
    analysis; the reference-failure report and its stop rule; the CSV and the figure.
    Records the references' digest (refs_digest), which the scoring of the test set and
    Stage 2 check, and the file's SHA-256 beside it. Refuses an incomplete record, an
    existing report or CSV (no record written over), and references whose header carries no
    emulator hash or another one than G2's (line 142, D82); records the emulator."""
    inputs, paths = dict(DEFAULT_INPUTS, **(inputs or {})), out_paths(out_dir)
    if not os.path.exists(inputs["refs"]):
        raise SystemExit(f"no refs at {_rel(inputs['refs'])}: the test set's references follow G2 after freezing")
    EM.refuse_overwrite([paths["stage1_report"], paths["stage1_csv"]], "Stage 1")
    ref_hdr, pairs, _, failures, fail_recs = load_refs(inputs["refs"], None, check_samples=False, details=True)
    named = {"test references": (ref_hdr or {}).get("emulator_sha256")}
    if (ref_hdr or {}).get("g2") is not None:
        named["G2's report, in the references' header"] = ref_hdr["g2"].get("emulator_sha256")
    EM.require_same_emulator(named, "Stage 1")
    if not pairs:
        raise SystemExit("no accepted reference pairs")
    n_points = int(ref_hdr["n_points"]) if ref_hdr and ref_hdr.get("n_points") else None
    if n_points is not None and len(pairs) + len(failures) != 3 * n_points:
        raise SystemExit(f"Stage 1 refused: the record holds {len(pairs) + len(failures)} pairs of the {3 * n_points} "
                         "its header declares; the references are not complete")
    s1, bins = stage1(pairs, n_failures=len(failures))
    wc = {lab: p for lab, p in pairs.items() if p["well_covered"]}
    s1_wc, _ = stage1(wc, n_failures=len(failures))
    acc_by_level = {j: sum(1 for p in pairs.values() if p["j"] == j) for j in LEVEL_INDEX.values()}
    rf = reference_failure_report(fail_recs, acc_by_level, n_points)
    v = versions or EM.version_check(EM.CORE_VERSIONS, check=False)
    write_stage1(pairs, bins, paths["stage1_csv"], paths["stage1_png"])
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    report = {"stage": 1, "stage1": s1, "coverage": {"cover_min": R.COVER_MIN, "n_well_covered": len(wc),
                                                      "share_well_covered": len(wc) / len(pairs),
                                                      "per_bin": {b: {"n": s1["bins"][b]["n_well_covered"],
                                                                      "share": s1["bins"][b]["share_well_covered"]}
                                                                  for b in BINS},
                                                      "secondary_stage1": s1_wc},
              "reference_failures": rf, "p5_stops": rf["p5_stops"],
              "untestable_both_calibrators": [b for b in BINS if s1["bins"][b]["untestable"]],
              "untestable_secondary": [b for b in BINS if s1_wc["bins"][b]["untestable"]],
              "refs": _rel(inputs["refs"]), "refs_digest": refs_digest(inputs["refs"]),
              "refs_sha256": sha256_of(inputs["refs"]), "refs_header": ref_hdr,
              "emulator": {"path": ref_hdr.get("emulator"), "sha256": ref_hdr.get("emulator_sha256")},
              "files": {k: _rel(paths[k]) for k in ("stage1_report", "stage1_csv", "stage1_png")},
              "commit": commit, "clean_tree": clean, "python": platform.python_version(),
              "numpy": v["numpy"], "scipy": v["scipy"], "versions": v, "utc": utc_now(),
              "note": "declared from the references alone, before any network scores the test set (line 101)"}
    os.makedirs(os.path.dirname(paths["stage1_report"]), exist_ok=True)
    text = json.dumps(report, indent=1, allow_nan=False)
    with open(paths["stage1_report"], "w", encoding="utf-8") as f:
        f.write(text)
    return report


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


def levels_kept(lost, calibrator, measure):
    """The levels (1-based j) at which every network a measure needs passed G3 for this
    calibrator (line 144, D81): `lost` maps (calibrator, prior) to the levels whose cell failed."""
    lost = lost or {}
    gone = set()
    for prior in NEEDS.get(measure, PRIORS):
        gone |= set(lost.get((calibrator, prior), ()))
    return [j for j in sorted(LEVEL_INDEX.values()) if j not in gone]


def null_reading(verdicts, hd_interval, band=NULL_BAND):
    """Line 128 (D81): the null stands for a calibrator when H-a is Confirmed, H-b, H-c and
    H-d are Refuted, and H-d's interval lies wholly within the band (closed ends); an
    Inconclusive or untestable outcome anywhere leaves it unproven."""
    hd_in_band = bool(hd_interval and band[0] <= hd_interval[0] and hd_interval[1] <= band[1])
    pattern = (verdicts.get("H-a") == "Confirmed" and all(verdicts.get(h) == "Refuted" for h in ("H-b", "H-c", "H-d")))
    return {"stands": bool(pattern and hd_in_band), "H_a_confirmed_and_others_refuted": bool(pattern),
            "all_four_refuted": all(verdicts.get(h) == "Refuted" for h in ("H-a", "H-b", "H-c", "H-d")),   # v1's rule, reported only
            "H_d_interval_within_band": hd_in_band, "band": list(band),
            "rule": "H-a Confirmed; H-b, H-c and H-d Refuted; H-d's interval wholly within the band (line 128, D81)",
            "note": "an Inconclusive or untestable outcome anywhere leaves the null unproven (line 128)"}


def score_hypotheses(pairs, cells, slots, cuts=CUTS, rng_seed=SEED_BOOT, n_boot=N_BOOT, n_failures=0,
                     calibrators=CALIBRATORS, hypotheses=HYPOTHESES, lost=None, read_null=True):
    """Stage 1 at these cut-offs, then every hypothesis for every calibrator, each pooled over
    the levels at which the networks its measure needs passed G3 (`lost`, D81). The
    calibrator's floors (line 75's failure floor and, for H-c, the seed floor) are taken on
    the pairs of the levels the hypothesis keeps; the per-bin floors over every level are
    reported beside them (D81)."""
    s1, bins = stage1(pairs, cuts, n_failures=n_failures)
    out = {"cuts": list(cuts), "stage1": s1, "calibrators": {}}
    for ci_, cal in enumerate(calibrators):
        sl = slots.get(cal, ())
        res = {"seed_slots": list(sl), "hypotheses": {}, "failure_floor": {}, "seed_floor": {},
               "levels_lost_g3": {p: sorted((lost or {}).get((cal, p), ())) for p in PRIORS}}
        for b in BINS:
            res["failure_floor"][b] = failure_floor(cells, bins, cal, b, sl)
            res["seed_floor"][b] = seed_floor(cells, bins, b, cal, sl)
        for hi_, (hid, hyp) in enumerate(hypotheses.items()):
            rng = np.random.default_rng([rng_seed, ci_, hi_])          # one stream per (calibrator, row)
            b = hyp["bin"]
            kept = levels_kept(lost, cal, hyp["measure"])
            bins_kept = {lab: bb for lab, bb in bins.items() if pairs[lab]["j"] in kept}
            labs = sorted(lab for lab, bb in bins_kept.items() if bb == b)
            ff = failure_floor(cells, bins_kept, cal, b, sl)           # on the kept levels' pairs (D81)
            row = {"name": hyp["name"], "measure": hyp["measure"], "bin": b, "expected": hyp["expected"],
                   "refuted": hyp["refuted"], "direction": hyp["direction"], "n_pairs": len(labs),
                   "n_failed_pairs": ff["n_failed"], "n_failed_cells": ff["n_failed_cells"], "levels_pooled": kept,
                   "failure_floor": {k: ff[k] for k in ("n_pairs", "n_failed", "share_failed", "untestable")}}
            untestable = s1["bins"][b]["untestable"] or not kept or ff["untestable"]
            row["untestable_reason"] = ("stage 1 floor" if s1["bins"][b]["untestable"] else
                                        "G3: no level left for this calibrator (line 144)" if not kept else
                                        "calibrator failure floor" if ff["untestable"] else None)
            if untestable or not sl:
                row.update(point=None, interval=None, per_seed=None, n_cells=0, verdict="Untestable")
            else:
                M = cell_matrix(cells, cal, hyp["measure"], labs, sl)
                point, per_seed, n_cells = point_and_seed_medians(M)
                interval, reps, info = bootstrap_median(M, rng, n_boot)
                fl = seed_floor(cells, bins_kept, b, cal, sl)["floor"] if hyp["seed_floor"] else None
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
                    j = LEVEL_INDEX[float(s)]
                    if j not in kept:
                        row["per_level"][str(s)] = {"lost": "a network this measure needs failed G3 at this level"}
                        continue
                    labs_s = [lab for lab in labs if pairs[lab]["j"] == j]
                    if labs_s:
                        Ms = cell_matrix(cells, cal, hyp["measure"], labs_s, sl)
                        pt, _, nc = point_and_seed_medians(Ms)
                        row["per_level"][str(s)] = {"median": pt, "n_cells": nc, "n_pairs": len(labs_s)}
            res["hypotheses"][hid] = row
        verdicts = {h: r["verdict"] for h, r in res["hypotheses"].items()}
        res["null"] = (null_reading(verdicts, res["hypotheses"]["H-d"].get("interval")) if read_null else
                       {"stands": None, "note": "not read on this analysis: the null is read on the primary verdicts only"})
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
        if n == 0 or M.shape[1] == 0:                                   # no accepted cell at all (D81: G3 may leave none)
            out[cal] = {"spearman": None, "interval": None, "n_pairs": 0, "n_dropped_pairs": n_dropped_pairs,
                        "n_live_seeds": int(live_s.sum()), "expected_sign": "negative", "n_replicates": 0,
                        "n_dropped_replicates": 0, "note": "no pair with an accepted cell; descriptive; no verdict (line 134)"}
            continue
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


def g3_lost_levels(g3_report, calibrators=CALIBRATORS):
    """From G3's report (line 144, D81): {(calibrator, prior): [levels j whose cell failed]};
    refuses a report that lacks a cell."""
    lost, cells = {}, g3_report.get("cells", {})
    for cal in calibrators:
        for prior in PRIORS:
            for j in sorted(LEVEL_INDEX.values()):
                key = f"{cal}_{prior}_j{j}"
                if key not in cells:
                    raise SystemExit(f"inputs refused: the G3 report has no cell {key}")
                if not cells[key].get("passed"):
                    lost.setdefault((cal, prior), []).append(j)
    return lost


def check_scores_against_g3(present, lost, calibrators=CALIBRATORS):
    """A cell that failed G3 has no scores; a cell that passed has some (D81)."""
    for cal in calibrators:
        for prior in PRIORS:
            for j in sorted(LEVEL_INDEX.values()):
                has = any((cal, prior, j, q) in present for q in range(1, 10))
                failed = j in lost.get((cal, prior), ())
                if failed and has:
                    raise SystemExit(f"inputs refused: scores exist for ({cal}, {prior}, level {j}), a cell that failed G3")
                if not failed and not has:
                    raise SystemExit(f"inputs refused: no scores for ({cal}, {prior}, level {j}), a cell that passed G3")


def run(inputs=None, out_dir=None, rng_seed=SEED_BOOT, n_boot=N_BOOT, calibrators=CALIBRATORS, versions=None):
    """Stage 2 (lines 103 to 135). Refuses, before the stored samples are read: without the
    references; where P5 stopped at Stage 1; without the samples or the scores; on sets that
    differ; without Stage 1's output for these references (require_stage1); on scores not
    made after that output, or not covering every network their header lists; without G3's
    report, or on scores that disagree with its cells (D81); when the references, the
    scores, G3's report and Stage 1's output do not all carry one emulator hash; and before
    any measure is computed, when its report or cells CSV exists (no record is written over)
    (D82). The dose-response is also reported on the well-covered pairs, deciding nothing."""
    inputs, paths = dict(DEFAULT_INPUTS, **(inputs or {})), out_paths(out_dir)
    if not os.path.exists(inputs["refs"]):
        raise SystemExit(f"no refs at {_rel(inputs['refs'])}: the test set's references and scores follow the runs after freezing")
    if os.path.exists(paths["stage1_report"]):                         # P5 stopped at Stage 1: no scores follow it
        with open(paths["stage1_report"], encoding="utf-8") as f:
            s1_early = json.load(f)
        if s1_early.get("p5_stops") and s1_early.get("refs_digest") == refs_digest(inputs["refs"]):
            raise SystemExit("Stage 2 refused: P5 stopped at Stage 1 (reference failures above 10% at a noise level)")
    for k in ("samples", "scores"):
        pth = inputs[k]
        if not (os.path.exists(pth) or (k == "samples" and os.path.isdir(R.samples_dir(pth)))):
            raise SystemExit(f"no {k} at {_rel(pth)}: the test set's references and scores follow the runs after freezing")
    ref_hdr = read_jsonl(inputs["refs"])[0]
    score_hdr, scores, slots, present = load_scores(inputs["scores"])
    if ref_hdr and score_hdr and ref_hdr.get("set") != score_hdr.get("set"):
        raise SystemExit(f"inputs refused: the references are the set {ref_hdr.get('set')!r} and the scores the set "
                         f"{score_hdr.get('set')!r}")
    # Stage 1 comes first, from these references, before any score (line 101, D81)
    s1_rep = require_stage1(inputs["refs"], out_dir)
    if (score_hdr or {}).get("stage1_refs_digest") != s1_rep["refs_digest"]:
        raise SystemExit("Stage 2 refused: the scores were not made after this Stage 1 output (their header names "
                         "another Stage 1 output, or none; line 101, D81)")
    listed = (score_hdr or {}).get("expected")
    if listed is None:
        raise SystemExit("Stage 2 refused: the scores' header does not list the networks to be scored (D81)")
    expected = {(str(c), str(p), int(j), int(q)) for c, p, j, q in listed}
    missing, extra = sorted(expected - present), sorted(present - expected)
    if missing:
        raise SystemExit(f"Stage 2 refused: the scoring is incomplete: no rows for {missing[:5]}"
                         + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""))
    if extra:
        raise SystemExit(f"Stage 2 refused: rows for networks the scores' header does not list: {extra[:5]}")
    if not os.path.exists(inputs["g3_report"]):
        raise SystemExit(f"no g3_report at {_rel(inputs['g3_report'])}: G3's per-cell verdicts are an input (line 144)")
    with open(inputs["g3_report"], encoding="utf-8") as f:
        g3 = json.load(f)
    lost = g3_lost_levels(g3, calibrators)
    check_scores_against_g3(present, lost, calibrators)
    emulator_sha = EM.require_same_emulator({"test references": (ref_hdr or {}).get("emulator_sha256"),
                                             "scores": ((score_hdr or {}).get("emulator") or {}).get("sha256"),
                                             "G3 report": (g3.get("emulator") or {}).get("sha256"),
                                             "Stage 1 output": (s1_rep.get("emulator") or {}).get("sha256")}, "Stage 2")
    EM.refuse_overwrite([paths["report"], paths["cells_csv"]], "Stage 2")
    ref_hdr, pairs, stores, failures = load_refs(inputs["refs"], inputs["samples"])     # the stored samples, read last
    if not pairs:
        raise SystemExit("no accepted reference pairs")
    n_expected = 3 * int(ref_hdr["n_points"]) if ref_hdr and ref_hdr.get("n_points") else None
    n_missing = (n_expected - len(pairs) - len(failures)) if n_expected is not None else None
    cells = measures(pairs, scores, stores, slots, calibrators, present=present)
    main_res, bins = score_hypotheses(pairs, cells, slots, CUTS, rng_seed, n_boot, len(failures), calibrators, lost=lost)
    for b in BINS:                                                     # the declaration of Stage 1 stands as made
        if main_res["stage1"]["bins"][b]["untestable"] != s1_rep["stage1"]["bins"][b]["untestable"]:
            raise SystemExit("Stage 2 refused: its Stage 1 differs from the Stage 1 output declared before scoring")
    shifted = {}
    for sh in SHIFTS:
        cuts = (round(CUTS[0] + sh, 10), round(CUTS[1] + sh, 10))      # 0.95 and 0.55; 0.85 and 0.45, free of float noise
        res, _ = score_hypotheses(pairs, cells, slots, cuts, rng_seed, n_boot, len(failures), calibrators, lost=lost,
                                  read_null=False)
        shifted[f"{sh:+.2f}"] = {"cuts": list(cuts), "verdicts": {cal: {h: r["verdict"] for h, r in v["hypotheses"].items()}
                                                                  for cal, v in res["calibrators"].items()},
                                 "stage1": res["stage1"], "calibrators": res["calibrators"],
                                 "note": "the intervals, floors and seed floors recomputed on the shifted bins; reported "
                                         "beside the verdicts; decides nothing (lines 89, 111)"}
    wc = {lab: p for lab, p in pairs.items() if p["well_covered"]}
    sec_res, _ = score_hypotheses(wc, cells, slots, CUTS, rng_seed, n_boot, len(failures), calibrators, lost=lost,
                                  read_null=False)
    secondary = {"label": SECONDARY_LABEL, "n_pairs": len(wc), "share_of_accepted": len(wc) / len(pairs),
                 "stage1": sec_res["stage1"], "calibrators": sec_res["calibrators"],
                 "verdicts": {cal: {h: r["verdict"] for h, r in v["hypotheses"].items()}
                              for cal, v in sec_res["calibrators"].items()},
                 "dose_response": dose_response(wc, cells, slots, rng_seed, n_boot, calibrators),   # decides nothing (D82)
                 "covered_means": "a mass of 0.001 is about 10 of a network's training surfaces at n = 10,000 and "
                                  "about 160 at n = 160,000 (line 136, D82)"}
    dr = dose_response(pairs, cells, slots, rng_seed, n_boot, calibrators)
    write_cells(cells, bins, paths["cells_csv"])                       # Stage 1's CSV and figure stay as declared
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    v = versions or EM.version_check(EM.CORE_VERSIONS, check=False)
    report = {"stage1": main_res["stage1"], "stage2": main_res["calibrators"], "re_scoring": shifted,
              "secondary": secondary,
              "dose_response": dr, "verdicts": {cal: {h: r["verdict"] for h, r in v_["hypotheses"].items()}
                                                for cal, v_ in main_res["calibrators"].items()},
              "null": {cal: v_["null"] for cal, v_ in main_res["calibrators"].items()},
              "g3_levels_lost": {f"{c}_{p}": js for (c, p), js in lost.items()},
              "stage1_output": {"file": _rel(paths["stage1_report"]), "refs_digest": s1_rep.get("refs_digest"),
                                "refs_sha256": s1_rep.get("refs_sha256"), "made_utc": s1_rep.get("utc")},
              "emulator": {"path": (ref_hdr or {}).get("emulator"), "sha256": emulator_sha},
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
              "inputs": {k: _rel(p) for k, p in inputs.items()}, "input_headers": {"refs": ref_hdr, "scores": score_hdr},
              "files": {k: _rel(p) for k, p in paths.items()}, "commit": commit, "clean_tree": clean,
              "python": platform.python_version(), "numpy": v["numpy"], "scipy": v["scipy"], "versions": v,
              "utc": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    text = json.dumps(report, indent=1, allow_nan=False)                 # serialised first: no truncated file on an error
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(text)
    return report


def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 Stage 1 and Stage 2 analysis (see module docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--stage1", action="store_true", help="Stage 1 from the references alone, before any score")
    mode.add_argument("--run", action="store_true", help="Stage 2, after Stage 1 and the scoring")
    ap.add_argument("--refs", default=DEFAULT_INPUTS["refs"])
    ap.add_argument("--samples", default=DEFAULT_INPUTS["samples"])
    ap.add_argument("--scores", default=DEFAULT_INPUTS["scores"])
    ap.add_argument("--g3-report", default=DEFAULT_INPUTS["g3_report"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy or scipy other than p5_g1_emulator.CORE_VERSIONS; recorded (D81)")
    a = ap.parse_args(argv)
    vers = EM.version_check(EM.CORE_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions)
    print("=== P5 Stage 1 and Stage 2 ===", flush=True)
    print(f"  cut-offs {CUTS} (re-scored at {SHIFTS}); window {WINDOW}; floor {FLOOR_SHARE:.0%} or {FLOOR_COUNT} pairs; "
          f"gap {GAP}; bootstrap {N_BOOT} replicates from default_rng([{SEED_BOOT}, calibrator, row]), percentile {CI}",
          flush=True)
    for h, hyp in HYPOTHESES.items():
        print(f"  {h} {hyp['name']}: median {hyp['measure']} in {hyp['bin']}, expected {hyp['direction']} {hyp['expected']}"
              f"{' and above the seed floor' if hyp['seed_floor'] else ''}, refuted "
              f"{'above' if hyp['direction'] == 'below' else 'below'} {hyp['refuted']}", flush=True)
    print(f"  null: H-a Confirmed, H-b, H-c and H-d Refuted, H-d's interval within {list(NULL_BAND)}; secondary analysis on "
          f"pairs with both priors' coverage >= {R.COVER_MIN}; reference failures above {REF_FAIL_MAX:.0%} at a level stop "
          "P5 at Stage 1", flush=True)
    if a.stage1:
        rep1 = run_stage1({"refs": a.refs}, a.out_dir, versions=vers)
        print(f"=== Stage 1: {rep1['stage1']['n_accepted']} accepted pairs, {rep1['reference_failures']['n_failures']} "
              "reference failures; bins " + ", ".join(f"{b} {v['count']} ({v['share']:.1%}{', untestable' if v['untestable'] else ''}"
                                                      "; well covered " + ("n/a" if v["share_well_covered"] is None else
                                                                           f"{v['share_well_covered']:.0%}") + ")"
                                                      for b, v in rep1["stage1"]["bins"].items())
              + (" ; P5 STOPS at Stage 1" if rep1["p5_stops"] else "") + " ===", flush=True)
        print(f"  written: {rep1['files']['stage1_report']}", flush=True)
        return
    if not a.run:
        print("\nHOLD: nothing analysed. --stage1 needs the test set's references; --run needs Stage 1 and the scores, "
              "after the runs.", flush=True)
        return
    rep = run({"refs": a.refs, "samples": a.samples, "scores": a.scores, "g3_report": a.g3_report}, a.out_dir,
              versions=vers)
    print(f"=== Stage 1: {rep['stage1']['n_accepted']} accepted pairs, {rep['n_reference_failures']} reference failures; bins "
          + ", ".join(f"{b} {v['count']} ({v['share']:.1%}{', untestable' if v['untestable'] else ''})"
                      for b, v in rep["stage1"]["bins"].items()) + " ===", flush=True)
    for cal, vs in rep["verdicts"].items():
        print(f"  {cal}: " + ", ".join(f"{h} {v}" for h, v in vs.items()) + f"; null {'stands' if rep['null'][cal]['stands'] else 'unproven'}",
              flush=True)
    print(f"  written: {rep['files']['report']}", flush=True)


if __name__ == "__main__":
    main(check_versions=True)
