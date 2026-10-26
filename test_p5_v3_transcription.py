"""
test_p5_v3_transcription.py — mechanics tests for P5 v3's transcription check (D95), on toy
emulators and toy records only: the names (never v2's, the study's or v3's gate records); a
toy study record made by the study's own worker (p5_v3_study._study_worker, the candidate
installed) reproduced by the check run through p5_g2_reference's own worker, the written-in
sampler, on the reweighted and the direct-run paths; a changed sample array or record field
caught and named; the refusals (a report that exists, no study record, other constants,
another emulator, a sealed seed, a patched run_chains); hold mode writing nothing; v2's
surfaces from the study's seeds. Nothing here touches G̃, and the sealed test set's drawing
functions are replaced by ones that fail if called wherever the entry point runs (D82).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import p5_g2_reference as R
import p5_g1_emulator as E
import p5_v3_study as S
import p5_v3_transcription as T
from test_p5_g2_reference import LinearEm, _toy_emulator_file, _no_sealed_draws, LO, HI, CENTRE

KW = {"burn": 300, "sample": 900, "kish_min": 50.0, "rhat_max": 1.5, "ess_min": 20.0}     # short toy lengths, relaxed acceptance
V2_SEEDS = {"g2": {"points": 79, "noise": 89, "fixed_points": [list(p) for p in R.FIXED_POINTS], "fixed_noise": 139},
            "g2r": {"points": 131, "noise": 137, "fixed_points": [list(p) for p in R.FIXED_POINTS], "fixed_noise": 149}}


@pytest.fixture(autouse=True)
def _written_in_sampler_restored():
    """Every test leaves p5_g2_reference as it found it: its own run_chains and worker, no emulator cached."""
    yield
    S.uninstall_candidate()
    R._worker = S._R_WORKER
    R._EM["path"], R._EM["em"] = None, None


def test_names_and_literals():
    assert T.SET_NAME == "p5v3_transcription" and T.N_PAIRS_EXPECTED == 36 == S.N_PAIRS_EXPECTED and T.POOL_DEFAULT == 4
    names = {Path(p).name for p in T.PATHS.values()}
    assert names == {"p5v3_transcription.jsonl", "p5v3_transcription_samples.npz", "p5v3_transcription_report.json"}
    others = {Path(v).name for d in R.SETS.values() for v in d.values()} | {Path(v).name for v in S.PATHS.values()} | \
        {"p5_g2_pilot.jsonl", "p5_g2_pilot_samples.npz", "p5_g2_report.json", "p5_g2r_pilot.jsonl", "p5_g2r_pilot_samples.npz",
         "p5_g2r_report.json"}
    assert not names & others
    assert T.STUDY == {"jsonl": S.PATHS["jsonl"], "samples": S.PATHS["samples"]}
    assert T.SEALED_SEEDS == (20261002, 20261022) == S.SEALED_SEEDS
    assert "status" in T.RECORD_KEYS and "c_A" in T.RECORD_KEYS and "source" in T.UNDER_KEYS and "accepted" in T.ATTEMPT_KEYS
    assert set(T.START_SKIP) == {"optimiser_seconds", "sampler"}
    assert T._eq(1.0, 1) and not T._eq(1.0, True) and T._eq(None, None) and T._eq("a", "a") and not T._eq(0.1, 0.1 + 1e-17)
    assert T._eq([1.0, {"a": None}], [1.0, {"a": None}]) and not T._eq([1.0], [1.0000001])


def _toy_jobs(em, levels=(0.003, 0.005)):
    pts, eps = np.array([CENTRE, CENTRE + 0.1 * (HI - LO)]), R.noise_draws(157, 2, 2)
    jobs = R.make_jobs("g2", pts, eps, em, levels)
    for j in jobs:
        j["v2_status"] = "ok"
    return jobs


def _paths(tmp_path, name):
    return {"jsonl": str(tmp_path / f"{name}.jsonl"), "samples": str(tmp_path / f"{name}.npz"),
            "report": str(tmp_path / f"{name}_report.json")}


def _toy_study(tmp_path, em_file, jobs, kw=KW):
    """A toy study record in the study's form: the study's worker (the candidate installed in
    this process at pool 1) over the jobs, one launch, a header with the study's constants and
    v2's seeds."""
    paths = _paths(tmp_path, "study")
    hdr = S.study_header(1, em_file, None, None, kw)
    hdr["seeds"] = dict(hdr["seeds"], **V2_SEEDS)
    S.run_references(jobs, paths, hdr, 1, em_file, kw)
    S.uninstall_candidate()
    assert R.run_chains is S._V2_RUN_CHAINS and R.run_chains.__module__ == "p5_g2_reference"
    return paths


def test_a_toy_study_record_is_reproduced_and_a_change_is_caught(tmp_path, monkeypatch):
    _no_sealed_draws(monkeypatch)
    monkeypatch.setattr(R, "EMULATOR_PATH", str(tmp_path / "no_g_tilde.npz"))
    em_file = _toy_emulator_file(tmp_path / "toy_em.npz", LinearEm())
    R._EM["path"], R._EM["em"] = None, None
    em = R.get_emulator(em_file)
    jobs = _toy_jobs(em)
    sp = _toy_study(tmp_path, em_file, jobs)
    study = T.read_study(sp, n_expected=4)
    assert len(study["tasks"]) == 4 and study["sha256_lf"] == S.sha256_lf(sp["jsonl"]) and all(t["candidate"] for t in study["tasks"].values())
    assert T.check_constants(study["hdr"]) == R.sampler_constants() == S.candidate_constants()
    tj = T.study_jobs(study["tasks"])
    assert [j["label"] for j in tj] == sorted(t["label"] for t in jobs) and all(j["study_status"] == "ok" for j in tj)
    # the check through p5_g2_reference's own worker, the written-in sampler
    hdr = T.transcription_header(1, em_file, study, None, KW)
    assert hdr["set"] == "p5v3_transcription" and hdr["lengths"]["start"] == R.sampler_constants() and hdr["study"]["n_pairs"] == 4
    assert (hdr["lengths"]["burn"], hdr["lengths"]["sample"], hdr["lengths"]["kish_min"]) == (300, 900, 50.0)
    assert hdr["study"]["sha256_lf"] == study["sha256_lf"] and hdr["emulator_sha256"] == E.sha256_file(em_file) and hdr["seeds"]["g2"]["points"] == 79
    cp = _paths(tmp_path, "check")
    assert T.run_references(tj, cp, hdr, 1, em_file, KW) == 4
    assert R.run_chains is S._V2_RUN_CHAINS                                            # nothing replaced
    diffs, rec_hdr, launches = T.compare(study, cp)
    assert sorted(diffs) == sorted(study["tasks"]) and all(d == [] for d in diffs.values()), diffs
    _, _, ctasks, cstores = R.load_references(cp["jsonl"], cp["samples"])
    assert "candidate" not in ctasks["g2_0000_0"] and all("start" in a for t in ctasks.values() for a in R.pair_attempts(t))
    for lab in ctasks:
        for p in ("A", "B"):
            assert np.array_equal(np.asarray(ctasks[lab]["under"][p]["sigma_P"]), S.sigma_P_from_record(study["tasks"][lab], p))
    rep = T.write_report(study, cp, diffs, rec_hdr, launches)
    assert rep["reproduced"] is False and rep["n_pairs_compared"] == 4 and rep["n_pairs_differing"] == 0   # 4 pairs, not the 36 expected
    monkeypatch.setattr(T, "N_PAIRS_EXPECTED", 4)
    Path(cp["report"]).unlink()
    rep = T.write_report(study, cp, diffs, rec_hdr, launches)
    assert rep["reproduced"] is True and "exactly" in rep["reading"] and rep["n_sample_arrays_compared"] == len(cstores) == 4
    assert rep["n_sampler_runs_compared"] == 4 and rep["study"]["sha256_lf"] == study["sha256_lf"] and rep["differences"] == {}
    json.dumps(json.loads(Path(cp["report"]).read_text(encoding="utf-8")), allow_nan=False)
    with pytest.raises(SystemExit, match="transcription report refused: .*exists"):
        T.write_report(study, cp, diffs, rec_hdr, launches)
    # the direct-run path: the study's worker and the reference's own worker on one pair with a Kish floor no reweighting meets
    job = next(j for j in jobs if j["label"] == "g2_0001_1")
    rec_s, st_s = S._study_worker((dict(job), dict(KW, kish_min=1e12)))
    S.uninstall_candidate()
    rec_r, st_r = R._worker((dict(job), dict(KW, kish_min=1e12)))
    rec_s, rec_r = json.loads(json.dumps(rec_s, allow_nan=False)), json.loads(json.dumps(rec_r, allow_nan=False))
    assert rec_s["direct"] == ["A", "B"] == rec_r["direct"] and len(rec_s["candidate"]) == 3 == len(R.pair_attempts(rec_r))
    assert T.compare_pair("g2_0001_1", rec_s, rec_r, st_s, st_r) == []
    for p in ("A", "B"):
        assert np.array_equal(np.asarray(rec_r["under"][p]["sigma_P"]), S.sigma_P_from_record(rec_s, p))
        assert rec_r["under"][p]["sigma_P"] == rec_r["runs"][p][-1]["start"]["kept"][0]["sigma"]        # the direct run's Σ_P
    # a changed sample array, a changed figure, a changed attempt field, a changed start field and a changed Σ_P are each caught and named
    bad = json.loads(json.dumps(study["tasks"]))
    bad["g2_0000_0"]["under"]["A"]["mean_H"] += 1e-12
    bad["g2_0000_1"]["candidate"][0]["kept"][0]["x"][2] += 1e-15
    bad["g2_0001_0"]["runs"]["uniform"][0]["acceptance"] = 0.5
    bad["g2_0001_1"]["candidate"][0]["kept"][0]["sigma_B"][0][0] *= 1.0000001
    bst = dict(study["stores"])
    bst["g2_0000_1_u"] = bst["g2_0000_1_u"].copy()
    bst["g2_0000_1_u"][7] += 1
    d2, _, _ = T.compare(dict(study, tasks=bad, stores=bst), cp)
    assert d2["g2_0000_0"] == ["under.A.mean_H"] and d2["g2_0001_0"] == ["runs.uniform[0].acceptance"]
    assert sorted(d2["g2_0000_1"]) == ["samples g2_0000_1_u", "start[0].kept"]
    assert d2["g2_0001_1"] == ["under.B.sigma_P", "start[0].kept"]
    # the times differ from run to run and are not compared; a missing pair is a difference
    for t in bad.values():
        for c in t["candidate"]:
            c["optimiser_seconds"] = 0.0
    bad["g2_0000_0"]["under"]["A"]["mean_H"] -= 1e-12
    bad["g2_0001_0"]["runs"]["uniform"][0]["acceptance"] = study["tasks"]["g2_0001_0"]["runs"]["uniform"][0]["acceptance"]
    d3, _, _ = T.compare(dict(study, tasks=bad, stores=study["stores"]), cp)
    assert d3["g2_0000_0"] == [] and d3["g2_0001_0"] == []
    assert T.compare_pair("g2_0000_0", study["tasks"]["g2_0000_0"], None, study["stores"], cstores) == ["record missing"]


def test_the_entry_point_on_toy_records_the_refusals_and_hold_mode(tmp_path, monkeypatch, capsys):
    _no_sealed_draws(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(T, "PATHS", {k: str(out_dir / Path(v).name) for k, v in T.PATHS.items()})
    monkeypatch.setattr(R, "EMULATOR_PATH", str(tmp_path / "no_g_tilde.npz"))
    em_file = _toy_emulator_file(tmp_path / "toy_em.npz", LinearEm())
    R._EM["path"], R._EM["em"] = None, None
    em = R.get_emulator(em_file)
    jobs = _toy_jobs(em)
    sp = _toy_study(tmp_path, em_file, jobs)
    monkeypatch.setattr(T, "STUDY", {"jsonl": sp["jsonl"], "samples": sp["samples"]})
    monkeypatch.setattr(T, "N_PAIRS_EXPECTED", 4)

    def boom(*a, **k):
        raise AssertionError("the emulator was touched in hold mode")
    with monkeypatch.context() as m:                                                    # hold mode: the design, the study, nothing run
        for mod, name in ((E, "require_g1"), (E, "emulator_for_run"), (E, "load_emulator"), (R, "get_emulator"), (R, "launch")):
            m.setattr(mod, name, boom)
        T.main([])
        out = capsys.readouterr().out
        assert "HOLD: nothing run" in out and "4 pairs, 4 ok" in out and "constants: the study's start constants equal" in out
        assert "g2_0001_1: s=0.005 key [83, 1, 1] ok runs=1" in out and "sealed seeds (20261002, 20261022) refused" in out
        assert list(out_dir.iterdir()) == []
    # --run end to end on the toy: the toy file stands in for G̃; another --emulator is refused before any load
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: em_file)
    with pytest.raises(SystemExit, match="is not G̃"):
        T.main(["--run", "--pool", "1", "--emulator", str(tmp_path / "other.npz")])
    assert list(out_dir.iterdir()) == []
    with pytest.raises(SystemExit, match="is not v2's surface"):                          # the toy θ are not v2's surfaces
        T.main(["--run", "--pool", "1"])
    assert list(out_dir.iterdir()) == []
    monkeypatch.setattr(T, "check_surfaces", lambda jobs_, em_, seeds, atol=1e-12: True)   # stood down for the toy (tested on its own below)
    T.main(["--run", "--pool", "1"])
    out = capsys.readouterr().out
    assert "=== RUN: 4 pairs to do" in out and "=== TRANSCRIPTION: reproduced: 4 pairs compared, 0 differing, 4 sample arrays, 4 sampler runs" in out
    assert sorted(p.name for p in out_dir.iterdir()) == ["p5v3_transcription.jsonl", "p5v3_transcription_report.json",
                                                         "p5v3_transcription_samples.npz", "p5v3_transcription_samples_parts"]
    rep = json.load(open(T.PATHS["report"], encoding="utf-8"))
    assert rep["reproduced"] is True and rep["n_pairs_compared"] == 4 and rep["emulator"]["sha256"] == E.sha256_file(em_file)
    assert rep["study"]["sha256_lf"] == S.sha256_lf(sp["jsonl"]) and rep["lengths"]["start"] == R.sampler_constants()
    assert (rep["lengths"]["burn"], rep["lengths"]["sample"], rep["lengths"]["kish_min"], rep["lengths"]["ess_min"]) == (300, 900, 50.0, 20.0)
    hdr = R.read_jsonl(T.PATHS["jsonl"])[0]
    assert hdr["set"] == "p5v3_transcription" and hdr["script"] == "p5_v3_transcription.py" and hdr["seeds"]["g2"]["points"] == 79
    with pytest.raises(SystemExit, match="transcription check refused: .*exists"):          # no record is written over
        T.main(["--run", "--pool", "1"])
    # the refusals: other constants in the study's header; a sealed seed; another emulator; a missing record; a patched run_chains
    st = T.read_study(sp, 4)
    other = json.loads(json.dumps(st["hdr"]))
    other["lengths"]["candidate"]["distinct"] = 0.02
    with pytest.raises(SystemExit, match="constants differ from the study's: distinct study 0.02 written 0.01"):
        T.check_constants(other)
    with pytest.raises(SystemExit, match="no start constants"):
        T.check_constants({"lengths": {"burn": 1}})
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(json.dumps({"kind": "header", "seeds": {"g2": {"points": 20261002}}, "seed_ref": 83}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed seed.*20261002"):
        T.read_study({"jsonl": str(sealed), "samples": str(tmp_path / "none.npz")}, 4)
    with pytest.raises(SystemExit, match="no study record at"):
        T.read_study({"jsonl": str(tmp_path / "absent.jsonl"), "samples": str(tmp_path / "none.npz")}, 4)
    with pytest.raises(SystemExit, match="holds 4 pair records, not 36"):
        T.read_study(sp, 36)
    Path(T.PATHS["report"]).unlink()
    em2 = tmp_path / "em2.npz"
    np.savez(em2, stage=np.array("base"), commit=np.array("x"), values=np.zeros((2, 1, 1, 2, 3)))
    monkeypatch.setattr(E, "require_g1", lambda base=None, refined=None: str(em2))
    with pytest.raises(SystemExit, match="was made with emulator .*, not G̃"):
        T.main(["--run", "--pool", "1"])
    S.install_candidate()                                                               # a patched run_chains is refused before any launch
    with pytest.raises(SystemExit, match="not the module's own function"):
        T.run_references(T.study_jobs(st["tasks"]), _paths(tmp_path, "patched"), T.transcription_header(1, em_file, st, None, KW), 1, em_file, KW)
    S.uninstall_candidate()
    with pytest.raises(SystemExit, match="--pool 0"):
        T.main(["--pool", "0"])
    monkeypatch.setattr(E, "CORE_VERSIONS", {"numpy": "0.0.0", "scipy": "0.0.0"})       # the version guard, as the reference's
    T.main([])
    with pytest.raises(SystemExit, match="refused: numpy"):
        T.main([], check_versions=True)
    T.main(["--allow-other-versions"], check_versions=True)


def test_v2_surfaces_from_the_studys_seeds(monkeypatch):
    """The study's pairs are v2's G2 surfaces, drawn with v2's seeds from the study's header,
    while p5_g2_reference now holds v3's; check_surfaces refuses a surface that is not v2's."""
    _no_sealed_draws(monkeypatch)
    pts, eps = T.v2_surfaces(V2_SEEDS, retry=False)
    assert pts.shape == (6, 4) and eps.shape == (6, 3, 35)
    assert np.array_equal(pts[:4], R.pilot_points(79, 4)) and np.array_equal(pts[4:], np.array(R.FIXED_POINTS))
    assert np.array_equal(eps[:4], R.noise_draws(89, 4)) and np.array_equal(eps[4:], R.noise_draws(139, 2))
    ptr, epr = T.v2_surfaces(V2_SEEDS, retry=True)
    assert np.array_equal(ptr[:4], R.pilot_points(131, 4)) and np.array_equal(epr[:4], R.noise_draws(137, 4)) and np.array_equal(epr[4:], R.noise_draws(149, 2))
    assert not np.array_equal(pts[:4], R.g2_surfaces()[0][:4])                           # v3's seeds give other points
    em = LinearEm()
    Y = R.surfaces(pts, eps, em)
    jobs = [{"label": f"g2_{i:04d}_{j}", "set": "g2", "i": i, "j": j, "s": float(s), "theta": [float(v) for v in pts[i]], "y": Y[i, j]}
            for i in range(6) for j, s in enumerate(R.NOISE_LEVELS)]
    assert T.check_surfaces(jobs, em, V2_SEEDS) is True
    other = [dict(j, y=np.asarray(j["y"]) + 1e-6) for j in jobs]
    with pytest.raises(SystemExit, match="is not v2's surface"):
        T.check_surfaces(other, em, V2_SEEDS)
    with pytest.raises(SystemExit, match="sealed seed"):
        T.v2_surfaces({"g2": dict(V2_SEEDS["g2"], noise=20261022)}, retry=False)
