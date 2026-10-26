"""
p5_v3_transcription.py — P5 v3's transcription check (D95). The sampler's start and first
proposal, written into p5_g2_reference.py from p5_v3_study.candidate_run_chains as the
development study ran it (D93, D94), is re-run on the study's 36 pairs at the study's
lengths and must reproduce the study's records exactly. The same generator per pair,
default_rng([83, i, j]), and one BLAS thread per process make the chains deterministic, so
the stored samples, the summaries and the start's diagnostics must agree to the last bit;
only the seconds and the times differ.

What it reads, read only: the study's record and samples, output/p5v3_study.jsonl and
output/p5v3_study_samples.npz (committed, D94), and G̃ through p5_g1_emulator.require_g1.
What it writes, and nothing else under output/: its own record output/p5v3_transcription.jsonl,
its samples output/p5v3_transcription_samples.npz (with the _parts folder beside it) and its
report output/p5v3_transcription_report.json; never v2's records (output/p5_g2_*, p5_g2r_*),
never the names reserved for v3's gates (output/p5v3_g2_*, p5v3_g2r_*, p5v3_g3_pilot_refs*,
p5v3_ref_test*), never the study's.

The run (--run, the author's, from the branch's code commit before the merge): the versions
against the pinned ones (refused unless --allow-other-versions, recorded); G̃, an --emulator
naming another file refused before any load; the report must not exist; the study's header
must carry G̃'s SHA-256 and, under lengths.candidate, the start's constants as
p5_g2_reference.sampler_constants() now gives them; p5_g2_reference.run_chains must be the
module's own (nothing replaced); the 36 pairs come from the study's task lines (label, set,
i, j, s, θ, y), each y checked against v2's G2 surfaces (v2's seeds, from the study's header)
priced through G̃, as the study checked them, the sealed seeds refused wherever a seed is read;
then p5_g2_reference.launch over them with its own worker at the study's lengths, 10,000 and
40,000, resumable, the samples written before the record; then the comparison, pair by pair,
and the report, written once.

The comparison, after the JSON round trip the records themselves make: every stored sample
array of the pair equal (numpy's array_equal; the key sets equal); the record's status, i, j,
s, seed, theta, y, direct, kish, neff, support_lo, support_hi, c_A, coverage and well_covered
equal, and under each prior the figures and the source; each sampler run's attempt, burn,
sample, acceptance, rhat, ess, ess_H and accepted equal; each run's start equal to the
study's `candidate` entry for that run on every field but optimiser_seconds (the study's
entries also carry a `sampler` tag, which the start does not); and Σ_P under each prior equal
to what the study's rule reads from its own record (p5_v3_study.sigma_P_from_record). The
report gives `reproduced`, true or false, every differing field per pair, the counts, and the
provenance (commit, clean tree, versions, G̃'s SHA-256, the study record's SHA-256 with LF line
endings, D94's blob hash).

Usage (from the repository root):
    py -3 p5_v3_transcription.py              the design, the study's pairs, the rule; holds
    py -3 p5_v3_transcription.py --run        the check (the author's); --pool K (default 4),
                                              --emulator PATH (G̃ by default; another file is refused),
                                              --allow-other-versions (recorded)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (p5_g2_reference.py)

import sys
import json
import time
import argparse

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g2_reference as R
import p5_g1_emulator as E
import p5_v3_study as S                      # the study's record names, its Σ_P rule and its sealed-seed check; nothing of it runs
from p5_g0_pilot import git_state, utc_now, _rel, read_jsonl
from p5_g1_emulator import BOX, NOISE_LEVELS

SET_NAME = "p5v3_transcription"
OUT_DIR = R.OUT_DIR
PATHS = {"jsonl": os.path.join(OUT_DIR, "p5v3_transcription.jsonl"),
         "samples": os.path.join(OUT_DIR, "p5v3_transcription_samples.npz"),
         "report": os.path.join(OUT_DIR, "p5v3_transcription_report.json")}
STUDY = {"jsonl": S.PATHS["jsonl"], "samples": S.PATHS["samples"]}          # read only (D94)
N_PAIRS_EXPECTED = S.N_PAIRS_EXPECTED                                        # 36
POOL_DEFAULT = 4
SEALED_SEEDS = S.SEALED_SEEDS                                                # 20261002, 20261022: never drawn

RECORD_KEYS = ("status", "set", "i", "j", "s", "seed", "theta", "y", "direct", "kish", "neff", "support_lo", "support_hi",
               "c_A", "coverage", "well_covered")
UNDER_KEYS = ("mean_H", "var_H", "q05_H", "q95_H", "mean_xi0", "source")
ATTEMPT_KEYS = ("attempt", "burn", "sample", "acceptance", "rhat", "ess", "ess_H", "accepted")
START_SKIP = ("optimiser_seconds", "sampler")                                 # the times differ; the study's tag is not a field of the start


def read_study(paths=None, n_expected=None):
    """The study's record and samples (D94), read only: {hdr, launches, tasks, stores, sha256_lf}.
    Refuses a missing record, a pair count other than 36, a record without the study's
    diagnostics or tag, a sealed seed among the header's, or samples missing for a record."""
    paths = paths or STUDY
    n_expected = N_PAIRS_EXPECTED if n_expected is None else int(n_expected)
    hdr, launches, tasks = read_jsonl(paths["jsonl"])
    if hdr is None:
        raise SystemExit(f"no study record at {_rel(paths['jsonl'])}: the check reproduces the development study's records (D94)")
    S.refuse_sealed([hdr.get("seeds"), hdr.get("seed_ref")], f"reading {_rel(paths['jsonl'])}")
    if len(tasks) != n_expected:
        raise SystemExit(f"{_rel(paths['jsonl'])} holds {len(tasks)} pair records, not {n_expected}")
    for lab, rec in sorted(tasks.items()):
        S.refuse_sealed(rec.get("seed"), f"pair {lab}")
        if rec.get("sampler") != S.SAMPLER_TAG or not rec.get("candidate"):
            raise SystemExit(f"pair {lab} in {_rel(paths['jsonl'])} was not made by the study's candidate (no tag or diagnostics)")
        if len(rec["candidate"]) != len(R.pair_attempts(rec)):
            raise SystemExit(f"pair {lab}: {len(rec['candidate'])} diagnostics entries for {len(R.pair_attempts(rec))} sampler runs")
    stores = R.load_samples(paths["samples"])
    missing = [lab for lab, rec in sorted(tasks.items()) if not R.samples_complete(rec, stores)]
    if missing:
        raise SystemExit(f"the study's samples lack the arrays of {', '.join(missing[:3])}{' and more' if len(missing) > 3 else ''}")
    return {"hdr": hdr, "launches": launches, "tasks": tasks, "stores": stores, "sha256_lf": S.sha256_lf(paths["jsonl"]),
            "paths": dict(paths)}


def check_constants(hdr):
    """The study's start constants (its header's lengths.candidate, D93) against the written-in
    sampler's (p5_g2_reference.sampler_constants), key by key; refuses a difference or a missing
    key. Returns the constants compared."""
    study = ((hdr or {}).get("lengths") or {}).get("candidate") or {}
    ours = R.sampler_constants()
    if not study:
        raise SystemExit("the study's header carries no start constants (lengths.candidate); the check needs them")
    bad = {k: (study.get(k, "<missing>"), v) for k, v in ours.items() if study.get(k, "<missing>") != v}
    if bad:
        raise SystemExit("the written-in sampler's constants differ from the study's: "
                         + "; ".join(f"{k} study {a!r} written {b!r}" for k, (a, b) in sorted(bad.items())))
    return ours


def study_jobs(tasks):
    """The study's pairs as launch's jobs: the label, the set (g2 or g2r), i, j, s, θ and y from
    the study's record, and the study's status; the generator key stays [83, i, j]."""
    jobs = []
    for label in sorted(tasks):
        rec = tasks[label]
        jobs.append({"label": label, "set": rec["set"], "i": int(rec["i"]), "j": int(rec["j"]), "s": float(rec["s"]),
                     "theta": [float(v) for v in rec["theta"]], "y": np.asarray(rec["y"], dtype=float),
                     "study_status": rec["status"]})
    return jobs


def v2_surfaces(seeds, retry):
    """v2's G2 surfaces of one attempt from the seeds the study's header recorded (v2's, D81,
    D89; p5_g2_reference now holds v3's), as g2_surfaces built them: the 4 points, the two
    fixed points, their noise."""
    sd = seeds["g2r" if retry else "g2"]
    S.refuse_sealed(sd, "the study's seeds")
    pts = np.vstack([R.pilot_points(int(sd["points"]), R.N_G2_POINTS), np.asarray(sd["fixed_points"], dtype=float)])
    eps = np.concatenate([R.noise_draws(int(sd["noise"]), R.N_G2_POINTS), R.noise_draws(int(sd["fixed_noise"]), len(sd["fixed_points"]))],
                         axis=0)
    return pts, eps


def check_surfaces(jobs, em, seeds, atol=1e-12):
    """Every pair's θ and y in the study's record equal v2's surfaces priced through the
    emulator given, as the study checked them against G̃ (D93)."""
    for name in ("g2", "g2r"):
        pts, eps = v2_surfaces(seeds, retry=(name == "g2r"))
        Y = R.surfaces(pts, eps, em)
        for job in (j for j in jobs if j["set"] == name):
            if not (np.allclose(job["theta"], pts[job["i"]], rtol=0, atol=atol) and np.allclose(job["y"], Y[job["i"], job["j"]], rtol=0, atol=atol)):
                raise SystemExit(f"refused: pair {job['label']}'s θ or y in the study's record is not v2's surface priced through "
                                 "this emulator (line 143; G̃, D88)")
    return True


def transcription_header(pool, emulator_path, study, versions=None, task_kwargs=None):
    """The check's header: p5_g2_reference.header's (commit, clean tree, the versions, G̃'s
    SHA-256, the constants, the start's constants), the study's provenance, and `lengths` with
    the lengths in force and the start's constants (launch refuses a resume whose `lengths`
    differ)."""
    kw = dict(task_kwargs or {})
    hdr_s = study["hdr"]
    seeds = {"pair_key": "default_rng([83, i, j]), i counted within the pair's v2 set (line 68; the study, D93)",
             "seed_ref": R.SEED_REF, "g2": (hdr_s.get("seeds") or {}).get("g2"), "g2r": (hdr_s.get("seeds") or {}).get("g2r")}
    S.refuse_sealed(seeds, "the check's header")
    extra = {"script": os.path.basename(__file__), "reference_script": os.path.basename(R.__file__),
             "check": "P5 v3 transcription check: p5_g2_reference.run_chains, the start written in from the development "
                      "study's candidate, re-run on the study's 36 pairs at the study's lengths (D95)",
             "study": {"path": _rel(study["paths"]["jsonl"]), "sha256_lf": study["sha256_lf"], "commit": hdr_s.get("commit"),
                       "lengths": hdr_s.get("lengths"), "emulator_sha256": hdr_s.get("emulator_sha256"), "n_pairs": len(study["tasks"]),
                       "samples": _rel(study["paths"]["samples"])},
             "comparison": {"record_keys": list(RECORD_KEYS), "under_keys": list(UNDER_KEYS) + ["sigma_P"],
                            "attempt_keys": list(ATTEMPT_KEYS), "start": "every field but " + ", ".join(START_SKIP),
                            "samples": "every stored array equal, the key sets equal"}}
    n_points = len({(t["set"], t["i"]) for t in study["tasks"].values()})
    hdr = R.header(SET_NAME, pool, emulator_path, n_points, seeds, extra, versions)
    sl = hdr_s.get("lengths") or {}
    hdr["lengths"] = {"burn": int(kw.get("burn", sl.get("burn", R.BURN))), "sample": int(kw.get("sample", sl.get("sample", R.SAMPLE))),
                      "kish_min": kw.get("kish_min", sl.get("kish_min", R.KISH_MIN)), "rhat_max": kw.get("rhat_max", sl.get("rhat_max", R.RHAT_MAX)),
                      "ess_min": kw.get("ess_min", sl.get("ess_min", R.ESS_MIN)), "levels": [float(s_) for s_ in NOISE_LEVELS],
                      "start": R.sampler_constants()}
    return hdr


def run_references(jobs, paths, hdr, pool, emulator_path, task_kwargs=None):
    """p5_g2_reference.launch over the study's pairs with the reference's own worker (nothing
    replaced; refused if run_chains is not the module's own), resumable, samples before
    records. Returns the count of pairs done now."""
    if getattr(R.run_chains, "__module__", None) != R.__name__:
        raise SystemExit("refused: p5_g2_reference.run_chains is not the module's own function; the check runs the written-in sampler")
    kw = {k: v for k, v in dict(task_kwargs or {}).items() if k in ("burn", "sample", "kish_min", "rhat_max", "ess_min")}
    kw.setdefault("burn", int(hdr["lengths"]["burn"]))
    kw.setdefault("sample", int(hdr["lengths"]["sample"]))
    if pool != 1:
        kw["emulator_path"] = emulator_path
    return R.launch(pool, jobs, paths["jsonl"], paths["samples"], hdr, kw)


def _eq(a, b):
    """Exact equality of JSON-like values (the records' numbers as JSON holds them)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool) and a == b
    return a == b


def compare_pair(label, study_rec, new_rec, study_stores, new_stores):
    """The differences between the study's record of a pair and the check's: a list of field
    names, empty when the pair is reproduced exactly."""
    diffs = []
    if new_rec is None:
        return ["record missing"]
    # the stored samples
    pre = label + "_"
    ks, kn = {k for k in study_stores if k.startswith(pre)}, {k for k in new_stores if k.startswith(pre)}
    if ks != kn:
        diffs.append(f"samples keys {sorted(ks)} vs {sorted(kn)}")
    for k in sorted(ks & kn):
        if not np.array_equal(np.asarray(study_stores[k]), np.asarray(new_stores[k])):
            diffs.append(f"samples {k}")
    # the record
    for k in RECORD_KEYS:
        if not _eq(study_rec.get(k), new_rec.get(k)):
            diffs.append(k)
    us, un = study_rec.get("under") or {}, new_rec.get("under") or {}
    if set(us) != set(un):
        diffs.append(f"under priors {sorted(us)} vs {sorted(un)}")
    for p in sorted(set(us) & set(un)):
        for k in UNDER_KEYS:
            if not _eq(us[p].get(k), un[p].get(k)):
                diffs.append(f"under.{p}.{k}")
        if study_rec.get("status") == "ok":
            try:
                sig = S.sigma_P_from_record(study_rec, p)
                if "sigma_P" not in un[p] or not np.array_equal(np.asarray(un[p]["sigma_P"], dtype=float), sig):
                    diffs.append(f"under.{p}.sigma_P")
            except (KeyError, IndexError, TypeError) as e:                  # noqa: BLE001
                diffs.append(f"under.{p}.sigma_P (the study's rule failed: {e!r})")
    # the sampler runs
    rs, rn = study_rec.get("runs") or {}, new_rec.get("runs") or {}
    if list(rs) != list(rn):
        diffs.append(f"runs priors {list(rs)} vs {list(rn)}")
    for p in rs:
        if p not in rn:
            continue
        if len(rs[p]) != len(rn[p]):
            diffs.append(f"runs.{p}: {len(rs[p])} vs {len(rn[p])} attempts")
            continue
        for n_, (a, b) in enumerate(zip(rs[p], rn[p])):
            for k in ATTEMPT_KEYS:
                if not _eq(a.get(k), b.get(k)):
                    diffs.append(f"runs.{p}[{n_}].{k}")
    # the start of each run against the study's candidate entry, in pair_attempts order
    cands, atts = study_rec.get("candidate") or [], R.pair_attempts(new_rec)
    if len(cands) != len(atts):
        diffs.append(f"start: {len(cands)} study entries vs {len(atts)} runs")
    for n_, (c, a) in enumerate(zip(cands, atts)):
        st = a.get("start")
        if st is None:
            diffs.append(f"start[{n_}] missing")
            continue
        keys = (set(c) | set(st)) - set(START_SKIP)
        for k in sorted(keys):
            if not _eq(c.get(k), st.get(k)):
                diffs.append(f"start[{n_}].{k}")
    return diffs


def compare(study, paths):
    """Every pair of the study against the check's record and samples: {label: [differences]}."""
    hdr, launches, tasks, stores = R.load_references(paths["jsonl"], paths["samples"])
    return {lab: compare_pair(lab, rec, tasks.get(lab), study["stores"], stores) for lab, rec in sorted(study["tasks"].items())}, hdr, launches


def write_report(study, paths, diffs, hdr, launches, versions=None, emulator_path=None):
    """The report, written once (never over): `reproduced`, the differences per pair, the counts
    and the provenance."""
    E.refuse_overwrite([paths["report"]], "transcription report")
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    n_pairs = len(study["tasks"])
    bad = {k: v for k, v in diffs.items() if v}
    reproduced = bool(n_pairs == N_PAIRS_EXPECTED and len(diffs) == n_pairs and not bad)
    n_arrays = sum(1 for lab in study["tasks"] for k in study["stores"] if k.startswith(lab + "_"))
    n_runs = sum(len(R.pair_attempts(r)) for r in study["tasks"].values())
    report = {"check": (hdr or {}).get("check"), "script": os.path.basename(__file__), "reproduced": reproduced,
              "reading": ("the written-in sampler reproduces the development study's records exactly" if reproduced else
                          "the written-in sampler does NOT reproduce the development study's records"),
              "n_pairs": n_pairs, "n_pairs_expected": N_PAIRS_EXPECTED, "n_pairs_compared": len(diffs),
              "n_pairs_differing": len(bad), "n_sample_arrays_compared": n_arrays, "n_sampler_runs_compared": n_runs,
              "differences": bad, "comparison": (hdr or {}).get("comparison"),
              "lengths": (hdr or {}).get("lengths"), "start": R.sampler_constants(),
              "study": (hdr or {}).get("study"), "record": {"path": _rel(paths["jsonl"]), "samples": _rel(paths["samples"])},
              "emulator": {"path": (hdr or {}).get("emulator"), "sha256": (hdr or {}).get("emulator_sha256")},
              "header_commit": (hdr or {}).get("commit"), "launches": launches, "report_commit": commit, "report_clean_tree": clean,
              "versions": versions or E.version_check(E.CORE_VERSIONS, check=False), "utc_report": utc_now()}
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(R._jsonable(report), f, indent=1, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 v3 transcription check (see module docstring)")
    ap.add_argument("--run", action="store_true", help="run the check (the author's; from the branch's code commit)")
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--emulator", default=None, help="G̃ is the build G1 passes (p5_g1_emulator.require_g1); another file is refused (D82)")
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy or scipy other than p5_g1_emulator.CORE_VERSIONS; recorded (D81)")
    a = ap.parse_args(argv)
    if a.pool < 1:
        raise SystemExit(f"refused: --pool {a.pool}; the pool is at least 1")
    vers = E.version_check(E.CORE_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions)
    cc = R.sampler_constants()
    print("=== P5 v3 transcription check: the written-in start on the development study's 36 pairs ===", flush=True)
    print(f"  sampler: {os.path.basename(R.__file__)}.run_chains as written (start: {cc['n_starts']} starts, least_squares "
          f"{cc['ls_method']} {cc['ls_jac']}, keep <= {cc['keep_max']}, distinct > {cc['distinct']}, eigenvalues in "
          f"[{cc['eig_lo']:g}, {cc['eig_hi']:.6f}], cap {cc['perturb_cap']}, first proposal {cc['scale']:.4f} x Sigma)", flush=True)
    print(f"  lengths: the study's, {R.BURN}/{R.SAMPLE} with the one repeat; pairs keyed [{R.SEED_REF}, i, j]; "
          f"R-hat<={R.RHAT_MAX} ESS_H>={R.ESS_MIN:.0f}; (Kish/steps) x ESS_H>={R.KISH_MIN:.0f}", flush=True)
    print(f"  reads: {_rel(STUDY['jsonl'])}, {_rel(STUDY['samples'])} (read only); writes: {_rel(PATHS['jsonl'])}, "
          f"{_rel(PATHS['samples'])}, {_rel(PATHS['report'])}; nothing else under output/", flush=True)
    print(f"  rule: every stored sample array equal; the record's {', '.join(RECORD_KEYS)}; under each prior "
          f"{', '.join(UNDER_KEYS)} and sigma_P (the study's rule); each run's {', '.join(ATTEMPT_KEYS)}; each run's start "
          f"against the study's candidate entry on every field but {', '.join(START_SKIP)}", flush=True)
    print(f"  numpy {vers['numpy']} scipy {vers['scipy']} (expected {E.CORE_VERSIONS}); sealed seeds {SEALED_SEEDS} refused", flush=True)
    study = None
    try:
        study = read_study()
        h = study["hdr"]
        n_ok = sum(1 for t in study["tasks"].values() if t["status"] == "ok")
        print(f"  study: {_rel(STUDY['jsonl'])} sha256 (LF) {str(study['sha256_lf'])[:12]}..., commit {str(h.get('commit'))[:7]}, "
              f"lengths {h.get('lengths', {}).get('burn')}/{h.get('lengths', {}).get('sample')}, {len(study['tasks'])} pairs, {n_ok} ok, "
              f"{len(study['stores'])} sample arrays, G~ {str(h.get('emulator_sha256'))[:12]}...", flush=True)
        try:
            check_constants(h)
            print("  constants: the study's start constants equal the written-in sampler's", flush=True)
        except SystemExit as e:
            print(f"  constants: {e}", flush=True)
        for label in sorted(study["tasks"]):
            t = study["tasks"][label]
            print(f"    {label}: s={t['s']} key {t['seed']} {t['status']} runs={len(R.pair_attempts(t))}"
                  + (f" (direct {t['direct']})" if t.get("direct") else ""), flush=True)
    except SystemExit as e:
        print(f"  study: {e}", flush=True)
    if not a.run:
        print("\nHOLD: nothing run, no emulator loaded, nothing written. --run is the author's, from the branch's code commit (D95).",
              flush=True)
        return
    if study is None:
        raise SystemExit("refused: the check needs the development study's record and samples (D94)")
    E.refuse_overwrite([PATHS["report"]], "transcription check")
    check_constants(study["hdr"])
    em_path = E.emulator_for_run(a.emulator)                                            # G̃, the build G1 passed (D82, D88)
    sha = E.sha256_file(em_path)
    if study["hdr"].get("emulator_sha256") != sha:
        raise SystemExit(f"refused: the study's record was made with emulator {study['hdr'].get('emulator_sha256')}, not G̃ ({sha})")
    em = R.get_emulator(em_path)
    jobs = study_jobs(study["tasks"])
    check_surfaces(jobs, em, study["hdr"].get("seeds") or {})
    sl = study["hdr"]["lengths"]                                                        # the study's lengths and acceptance, in force
    kw = {"burn": int(sl["burn"]), "sample": int(sl["sample"]), **{k: sl[k] for k in ("kish_min", "rhat_max", "ess_min") if k in sl}}
    hdr = transcription_header(a.pool, em_path, study, vers, kw)
    run_references(jobs, PATHS, hdr, a.pool, em_path, kw)
    diffs, rec_hdr, launches = compare(study, PATHS)
    rep = write_report(study, PATHS, diffs, rec_hdr or hdr, launches, vers, em_path)
    print(f"\n=== TRANSCRIPTION: {'reproduced' if rep['reproduced'] else 'NOT reproduced'}: {rep['n_pairs_compared']} pairs compared, "
          f"{rep['n_pairs_differing']} differing, {rep['n_sample_arrays_compared']} sample arrays, {rep['n_sampler_runs_compared']} "
          f"sampler runs ===", flush=True)
    for lab, d in sorted(diffs.items()):
        print(f"  {lab}: {'equal' if not d else 'DIFFERS in ' + ', '.join(d)}", flush=True)
    print(f"  written: {_rel(PATHS['report'])}", flush=True)


if __name__ == "__main__":
    main(check_versions=True)
