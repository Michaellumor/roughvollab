"""
test_p5_g0_pilot.py — mechanics tests for the P5 gate G0 scoping script. They guard the
MACHINERY on cheap settings: the lattice is the literal one and sits inside the draft's
box; no random draws; the script's cached pricing path equals the repository's wrapper
bit for bit; the ladder's bookkeeping and early stop; a non-finite solve is a status,
not an exception; the output schema; the Black–Scholes known answer. Nothing here is a
measurement of the pricer: the scoping run's output is the measurement, declared after
the run in the draft under "What was known before freezing" (L139) and in the ROADMAP.
"""
import csv
import json
import inspect
from pathlib import Path

import numpy as np

import p5_g0_pilot as P
import layer4_calibrate_surface as S

D38 = P.D38_POINT
TRIPLE = D38[:3]
VUS3 = np.array([-1.0, 0.0, 1.0])
CHEAP = {"A": (100.0, 32), "C": (100.0, 64), "D": (200.0, 64)}      # (U_max, n_nodes); finite at N 200, T 0.10


def test_lattice_is_the_literal_one_inside_the_box():
    assert len(P.TRIPLES) == 60 and len(set(P.TRIPLES)) == 60
    assert len(P.TASKS) == 300 and len(P.TS) == 5
    corners = {(H, nu, rho) for H in (0.02, 0.48) for nu in (0.05, 1.00) for rho in (0.00, -0.99)}
    assert set(P.TRIPLES[:8]) == corners                 # the 8 corner triples come first
    assert P.TRIPLES[8] == TRIPLE                        # then D38's
    assert 0.001 in P.XI0_VALUES and 0.25 in P.XI0_VALUES and 0.04 in P.XI0_VALUES
    for H, nu, rho in P.TRIPLES:                         # the draft's box, H upper bound 0.48
        assert 0.02 <= H <= 0.48 and 0.05 <= nu <= 1.00 and -0.99 <= rho <= 0.00
    assert all(0.001 <= x <= 0.25 for x in P.XI0_VALUES)
    assert P.KAPPA == 0.30 and P.TOL == 1e-4
    assert P.LADDER == (1000, 2000, 4000, 8000, 16000)
    assert P.SETTINGS == {"A": (200.0, 128), "C": (200.0, 256), "D": (400.0, 256)}
    assert P.SETTINGS["A"] == (200.0, S.NN)              # the base is the repository's own setting
    assert [T for _, T in P.TASKS[:5]] == list(P.TS)     # tasks run triple by triple, maturity by maturity


def test_no_random_draws():
    src = inspect.getsource(P)
    assert P.SEED is None
    assert "np.random" not in src and "default_rng" not in src and "RandomState" not in src
    assert "20261002" not in src                         # the sealed seed never appears


def test_pricing_path_equals_the_wrapper_bit_for_bit():
    T, N, n = 0.25, 200, 32
    rec = P.run_task(TRIPLE, T, ladder=(N,), settings={"A": (200.0, n), "C": (200.0, 2 * n), "D": (400.0, 2 * n)},
                     xi0s=(D38[3],), vus=VUS3)
    atm = S.atm_by_T(D38, [T], N_riccati=N, n_nodes=n)
    grids = S.fixed_surface_strikes(atm, VUS3, [T])
    ref = S.model_smile_cf_T(D38, grids[T], T, N_riccati=N, n_nodes=n)
    key = P._xi_key(D38[3])
    mine = np.array([np.nan if q is None else q for q in rec["rungs"][0]["quotes"]["A"][key]["quotes"]])
    assert np.array_equal(mine, ref, equal_nan=True)     # exact, not approximate
    assert rec["anchors"][key]["sigma_atm"] == atm[T]
    assert np.array_equal(np.array(rec["anchors"][key]["K"]), grids[T])
    d, n_q = P.identity_gate(N=N, n_nodes=n, Ts=(T,), vus=VUS3)
    assert d == 0.0 and n_q == 3


def test_ladder_bookkeeping_and_early_stop():
    T = 0.10
    # tolerance 1.0: rung 200 meets every comparison, so rung 400 is its partner only
    r = P.run_task(TRIPLE, T, ladder=(200, 400), settings=CHEAP, tol=1.0, xi0s=(0.04,), vus=VUS3)
    assert r["stopped_at"] == 200 and r["exhausted"] is False
    assert [(g["N"], g["role"]) for g in r["rungs"]] == [(200, "full"), (400, "partner_only")]
    first = r["rungs"][0]
    assert set(first["solves"]) == {"A", "C", "D"} and set(r["rungs"][1]["solves"]) == {"A"}
    assert all(c["met"] is True for c in (first["comparisons"][k] for k in ("N2", "nodes", "umax")))
    assert first["comparisons"]["N2"]["n_total"] == 3
    for sol in first["solves"].values():
        assert sol["status"] == "ok" and sol["seconds"] >= 0.0
    for q in first["quotes"].values():
        assert q["0.04"]["top_modulus"] is not None and q["0.04"]["status"] == ["ok"] * 3
    assert r["anchors"]["0.04"]["anchor_N"] == 200
    # tolerance 0: nothing can be met, the ladder is exhausted, the top rung has no partner
    r0 = P.run_task(TRIPLE, T, ladder=(200, 400), settings=CHEAP, tol=0.0, xi0s=(0.04,), vus=VUS3)
    assert r0["stopped_at"] is None and r0["exhausted"] is True
    assert [(g["N"], g["role"]) for g in r0["rungs"]] == [(200, "full"), (400, "full")]
    assert r0["rungs"][0]["comparisons"]["N2"]["met"] is False
    assert r0["rungs"][1]["comparisons"]["N2"]["status"] == "not_computed"
    assert r0["rungs"][1]["comparisons"]["N2"]["met"] is None
    assert P.first_finite_N(r0) == {"0.04": 200}


def test_nonfinite_solve_is_a_status_not_an_exception():
    r = P.run_task((0.02, 1.00, -0.99), 2.0, ladder=(20, 40), settings={"A": (400.0, 16), "C": (400.0, 32),
                                                                        "D": (800.0, 32)}, xi0s=(0.04, 0.25), vus=VUS3)
    for g in r["rungs"]:
        assert g["solves"]["A"]["status"] in ("riccati_nonfinite", "integral_nonfinite")
        assert g["solves"]["A"]["failed_at"] in ("u", "u_shift", "mi")
        assert g["solves"]["C"]["status"] == "skipped_base_nonfinite"
        assert g["solves"]["D"]["status"] == "skipped_base_nonfinite"
        for key in ("0.04", "0.25"):
            assert g["quotes"]["A"][key]["status"] == [g["solves"]["A"]["status"]] * 3
            assert g["quotes"]["A"][key]["quotes"] == [None] * 3
        assert g["comparisons"]["nodes"]["met"] is False and g["comparisons"]["nodes"]["status"] == "skipped_base_nonfinite"
    assert r["anchors"] == {} and r["stopped_at"] is None and r["exhausted"] is True
    assert P.first_finite_N(r) == {"0.04": None, "0.25": None}
    json.dumps(r, allow_nan=False)                       # the record carries no NaN


def test_output_schema(tmp_path):
    jl, js, cs = (str(tmp_path / n) for n in ("t.jsonl", "t.json", "t.csv"))
    r = P.run_task(TRIPLE, 0.10, ladder=(200, 400), settings=CHEAP, tol=0.0, xi0s=(0.04, 0.25), vus=VUS3)
    P.append_line(jl, P.header(pool=2))
    P.append_line(jl, {"kind": "launch", "utc": P.utc_now(), "pool": 2, "n_done_before": 0,
                       "timing_alone_s": None, "known_answers": None})
    P.append_line(jl, r)
    s = P.report(jl, js, cs)
    hdr, launches, tasks = P.read_jsonl(jl)
    assert hdr["kind"] == "header" and hdr["seed"] is None and hdr["n_tasks"] == 300
    assert hdr["settings"] == {"A": [200.0, 128], "C": [200.0, 256], "D": [400.0, 256]}
    assert len(launches) == 1 and list(tasks) == [r["task"]]
    assert s["n_tasks_done"] == 1 and s["n_tasks_total"] == 300
    assert json.dumps(json.loads(Path(js).read_text(encoding="utf-8")), allow_nan=False)
    rows = list(csv.DictReader(Path(cs).read_text(encoding="utf-8").splitlines()))
    assert list(rows[0].keys()) == P.CSV_COLUMNS
    assert len(rows) == 2 * 3 * 2                        # xi0 x z x rungs
    assert {row["status_A"] for row in rows} == {"ok"} and {row["atm_status_A"] for row in rows} == {"ok", "anchored"}
    assert all(row["quote_A"] != "" and row["K"] != "" and row["log10_modulus_A"] != "" for row in rows)
    assert s["cost"]["per_triple"][P.task_id(TRIPLE, 0.10)[:-6]]["n_ok_by_N"] == {str(N): 0 for N in P.LADDER}
    t = s["tasks"][0]
    assert t["task"] == P.task_id(TRIPLE, 0.10) and t["rungs"][1]["comparisons"]["N2"]["status"] == "not_computed"


def test_pool_launch_writes_and_resumes(tmp_path, capsys):
    jl, js, cs = (str(tmp_path / n) for n in ("t.jsonl", "t.json", "t.csv"))
    tasks = ((TRIPLE, 0.10), (TRIPLE, 0.25))
    kw = {"ladder": (200, 400), "settings": CHEAP, "tol": 1.0, "xi0s": (0.04,), "vus": VUS3}
    n1 = P.launch(2, jl, js, cs, tasks=tasks, task_kwargs=kw)
    n2 = P.launch(2, jl, js, cs, tasks=tasks, task_kwargs=kw)   # a second launch finds nothing to do
    assert (n1, n2) == (2, 0)
    hdr, launches, done = P.read_jsonl(jl)
    assert hdr["pool"] == 2 and [l["n_done_before"] for l in launches] == [0, 2]
    assert set(done) == {P.task_id(TRIPLE, 0.10), P.task_id(TRIPLE, 0.25)}
    assert all(rec["stopped_at"] == 200 for rec in done.values())
    s = json.loads(Path(js).read_text(encoding="utf-8"))
    assert s["n_tasks_done"] == 2 and [t["T"] for t in s["tasks"]] == [0.10, 0.25]   # ordered as TASKS
    assert all(l["commit"] == hdr["commit"] for l in launches)                       # provenance on every launch
    assert len(list(csv.DictReader(Path(cs).read_text(encoding="utf-8").splitlines()))) == 2 * 3 * 2
    out = capsys.readouterr().out
    assert "2 tasks to do" in out and "0 tasks to do (2 already" in out


def test_known_answer_black_scholes():
    rows = P.known_answers(settings={"A": (200.0, 128)}, Ts=(1.0,), xi0s=(0.04,), vus=np.array([0.0]), heston_pairs=())
    assert [(r["model"], r["setting"]) for r in rows] == [("bs", "adaptive"), ("bs", "A")]
    for r in rows:                                       # the fixed rule and the adaptive reference, both against bs_call
        assert abs(r["diff_price"]) < 1e-6 and abs(r["diff_iv"]) < 1e-6
    assert set(P.known_answers_summary(rows)) == {"bs/adaptive", "bs/A"}
    est = P.estimate({1000: 0.32, 4000: 4.4}, ladder=(1000, 2000), n_tasks=10)
    assert est["anchor_N"] == 4000 and abs(est["solve_s"][2000] - 1.1) < 1e-9
    assert est["bound_core_s"] == 10 * 5.0 * (est["solve_s"][1000] + est["solve_s"][2000])
