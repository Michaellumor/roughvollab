"""
test_p5_g0_main.py — mechanics tests for the P5 gate G0 main-run script. They guard the
MACHINERY on cheap settings: the settings are the literal values written into the draft
(line 140, D61); the stop rule and the partner on a cheap ladder; the criterion of line
140 on hand-built records (a known box, D38's tier before five-versus-four, the share and
its tie-break, "there and at every finer resolution checked", the cost limit as the median
over triples, confirmation attempts complete or incomplete, used up); the confirming draw;
the output schema. Nothing here is a measurement of the pricer.
"""
import csv
import json
import inspect
from pathlib import Path

import numpy as np

import p5_g0_main as M
import p5_g0_pilot as P

TRIPLE = M.D38_POINT[:3]
VUS3 = np.array([-1.0, 0.0, 1.0])
CHEAP = {"A": (100.0, 32), "C": (100.0, 64), "D": (200.0, 64)}      # finite at N 200, T 0.10


def test_settings_are_the_literal_values():
    assert M.U1 == 250.0 and M.SPACING == 200.0 / 128.0
    assert M.BASE_T == {0.10: (790.0, 504), 0.25: (500.0, 320), 0.50: (350.0, 224), 1.00: (250.0, 160), 2.00: (180.0, 112)}
    for T, (U, n) in M.BASE_T.items():
        assert abs(U - M.U1 / np.sqrt(T)) < 5 and n == 8 * round(U / M.SPACING / 8)      # nearest multiple of 8
        assert M.SETTINGS_T[T] == {"A": (U, n), "C": (U, 2 * n), "D": (2 * U, 2 * n)}
    assert M.LADDER_MAIN == (2000, 4000, 8000) and M.PARTNER == 16000
    assert M.H_VALUES == (0.02, 0.05, 0.10, 0.25, 0.48) and M.H_RANGE == (0.02, 0.48)
    assert M.NU_VALUES == (0.05, 0.35, 0.50, 0.65, 1.00) and M.RHO_VALUES == (0.00, -0.35, -0.70, -0.90, -0.99)
    assert M.XI0_MAIN == (0.02, 0.04, 0.07, 0.10, 0.16, 0.25)
    assert len(M.TRIPLES_MAIN) == 125 and len(set(M.TRIPLES_MAIN)) == 125 and len(M.TASKS_MAIN) == 625
    corners = {(H, nu, rho) for H in (0.02, 0.48) for nu in (0.05, 1.00) for rho in (0.00, -0.99)}
    assert set(M.TRIPLES_MAIN[:8]) == corners and M.TRIPLES_MAIN[8] == TRIPLE
    assert M.SEEDS == (61, 62) and M.N_CONFIRM == 8 and M.N_ATTEMPTS == 2 and M.POOL_DEFAULT == 4
    assert M.COST_LIMIT_S == 1200.0 and M.TOL == 1e-4
    assert M.RANGES == {"nu": (0.05, 1.00), "rho": (-0.99, 0.00), "xi0": (0.001, 0.25)}
    src = inspect.getsource(M)
    assert "20261002" not in src and "default_rng(7)" not in src


def test_stop_rule_and_partner_on_a_cheap_ladder():
    # at tol 1e-4 the U_max comparison fails by truncation (about 0.002) while the raised quote is
    # stable, so the second rung is final by the truncation-only clause; the partner gives its N2
    r = M.run_task_main(TRIPLE, 0.10, ladder=(200, 400), partner=800, settings=CHEAP, xi0s=(0.04, 0.02), vus=VUS3)
    assert [(g["N"], g["role"]) for g in r["rungs"]] == [(200, "full"), (400, "full"), (800, "partner_only")]
    assert r["final_N"] == 400 and r["stop_reason"] == "truncation_only" and r["exhausted"] is True
    assert r["rungs"][1]["comparisons"]["N2"]["met"] is True and r["rungs"][1]["comparisons"]["umax"]["met"] is False
    assert r["kind"] == "task" and r["attempt"] is None and r["partner"] == 800
    # with a third rung the same rung is final before it, and the third rung is partner only
    r3 = M.run_task_main(TRIPLE, 0.10, ladder=(200, 400, 800), partner=1600, settings=CHEAP, xi0s=(0.04, 0.02), vus=VUS3)
    assert [(g["N"], g["role"]) for g in r3["rungs"]] == [(200, "full"), (400, "full"), (800, "partner_only")]
    assert r3["final_N"] == 400 and r3["exhausted"] is False
    # a loose tolerance: everything met at the first rung
    r1 = M.run_task_main(TRIPLE, 0.10, ladder=(200, 400), partner=800, settings=CHEAP, tol=1.0, xi0s=(0.04,), vus=VUS3)
    assert r1["final_N"] == 200 and r1["stop_reason"] == "all_met" and r1["exhausted"] is False
    assert [(g["N"], g["role"]) for g in r1["rungs"]] == [(200, "full"), (400, "partner_only")]
    # tolerance 0: nothing is ever final; the ladder is exhausted and the partner still made
    r0 = M.run_task_main(TRIPLE, 0.10, ladder=(200, 400), partner=800, settings=CHEAP, tol=0.0, xi0s=(0.04,), vus=VUS3)
    assert r0["final_N"] is None and r0["stop_reason"] is None and r0["exhausted"] is True
    assert r0["rungs"][-1]["N"] == 800 and r0["rungs"][1]["comparisons"]["N2"]["met"] is False
    json.dumps(r0, allow_nan=False)


def test_nonfinite_skips_the_partner_and_confirm_records():
    r = M.run_task_main((0.02, 1.00, -0.99), 2.0, ladder=(20, 40), partner=80,
                        settings={"A": (400.0, 16), "C": (400.0, 32), "D": (800.0, 32)}, xi0s=(0.04,), vus=VUS3)
    assert all(g["solves"]["A"]["status"] in ("riccati_nonfinite", "integral_nonfinite") for g in r["rungs"])
    assert [g["role"] for g in r["rungs"]] == ["full", "full"]                  # no partner on a non-finite top
    assert r["rungs"][1]["comparisons"]["N2"]["status"] == "skipped_base_nonfinite"
    assert r["rungs"][0]["solves"]["C"]["status"] == "skipped_base_nonfinite" and r["final_N"] is None
    c = M.run_task_main(TRIPLE, 0.25, ladder=(200,), partner=None, settings=CHEAP, xi0s=(0.04,), vus=VUS3,
                        kind="confirm", attempt=1, label="C1_00_T0.25", candidate={"bounds": {"nu": [0.05, 0.35]}})
    assert c["task"] == "C1_00_T0.25" and c["kind"] == "confirm" and c["attempt"] == 1 and c["partner"] is None
    assert c["rungs"][0]["comparisons"]["N2"]["status"] == "not_computed"


# ---- the criterion on hand-built records ------------------------------------------------
LAD = M.LADDER_MAIN
Z3 = [-1.0, 0.0, 1.0]


def fake_rung(N, xi0s, pass_fn, secs):
    keyed = {M._xi_key(x): x for x in xi0s}
    q = lambda: {k: {"cf_status": "ok", "top_modulus": 0.0, "top_log10_modulus": -5.0, "atm_status": "ok",
                     "quotes": [0.2] * len(Z3), "status": ["ok"] * len(Z3)} for k in keyed}
    diffs = lambda bad: {k: [(0.0 if pass_fn(x, N) else bad) for _ in Z3] for k, x in keyed.items()}
    comp = lambda d: {"met": all(v == 0.0 for vs in d.values() for v in vs), "n_met": 0,
                      "n_total": len(Z3) * len(keyed), "max_abs_diff": 0.0, "diffs": d}
    return {"N": N, "role": "full",
            "solves": {s: {"N": N, "U_max": 1.0, "n_nodes": 1, "seconds": secs * (N / 2000) ** 2,
                           "status": "ok", "failed_at": None} for s in "ACD"},
            "quotes": {"A": q(), "C": q(), "D": q()},
            "comparisons": {"N2": comp(diffs(1.0)), "nodes": comp(diffs(0.0)), "umax": comp(diffs(1.0))}}


def fake_record(H, nu, rho, T, xi0s, pass_fn, secs=1.0, kind="task", attempt=None, candidate=None, label=None):
    return {"kind": kind, "task": label or M.task_id((H, nu, rho), T), "attempt": attempt, "candidate": candidate,
            "H": H, "nu": nu, "rho": rho, "T": T, "xi0s": list(xi0s), "z": Z3, "tol": 1e-4, "ladder": list(LAD),
            "anchors": {}, "rungs": [fake_rung(N, xi0s, pass_fn, secs) for N in LAD], "stopped_at": None,
            "final_N": None, "stop_reason": None, "exhausted": True, "seconds_total": 1.0}


def rule(nu, T):
    def pf(x, N):                                    # nu 1.0 fails at T 2.0; xi0 0.04 passes from 4000; else from 2000
        if nu == 1.0 and T == 2.0:
            return False
        return N >= (4000 if x == 0.04 else 2000)
    return pf


def lattice_records(rule_fn=rule, secs=1.0):
    return [fake_record(H, nu, rho, T, (0.04, 0.25), rule_fn(nu, T), secs=secs)
            for H in (0.02, 0.48) for nu in (0.05, 0.35, 1.0) for rho in (-0.7, 0.0) for T in M.TS]


def confirm_records(points, candidate, attempt, rule_fn=rule, Ts=M.TS, prefix=None):
    prefix = prefix or f"C{attempt}_"
    return [fake_record(p[0], p[1], p[2], T, (p[3],), rule_fn(p[1], T), kind="confirm", attempt=attempt,
                        candidate=candidate, label=f"{prefix}{i:02d}_T{T:.2f}") for i, p in enumerate(points) for T in Ts]


def test_criterion_order_on_hand_built_records():
    recs = lattice_records()
    st = M.apply_criterion(recs)
    assert st["n_candidates"] == 3 * 1 * 1 and st["n_points"] == 12         # C(3,2) x C(2,2) x C(2,2)
    assert st["grid"] == "five" and st["tier"] == "containing D38's point"
    t = st["taken"]
    assert t["bounds"] == {"nu": [0.05, 0.35], "rho": [-0.7, 0.0], "xi0": [0.04, 0.25]}
    assert t["resolution"] == {str(T): 4000 for T in M.TS}                   # xi0 0.04 passes only from 4000
    assert abs(t["share"] - (0.30 / 0.95) * (0.70 / 0.99) * (0.21 / 0.249)) < 1e-12
    assert t["cost_s"] == 5 * 4.0 and t["cost_n_triples"] == 12 and st["tie"] == []  # median over triples of the summed base solves
    assert len(st["passing"]["five"]) == 1 and len(st["passing"]["four"]) == 3   # nu 1.0 passes only without T 2.00
    best4 = max(st["passing"]["four"], key=lambda c: c["share"])
    assert best4["bounds"]["nu"] == [0.05, 1.0]                             # four would be wider; five is taken
    assert M.apply_criterion(recs, cost_limit=0.001)["taken"] is None        # nothing admissible
    # the cost is the median over triples: one slow triple does not move it, three do
    slow = lattice_records(secs=1.0)
    for r in slow:
        if (r["H"], r["nu"], r["rho"]) == (0.02, 0.05, -0.7):
            for g in r["rungs"]:
                g["solves"]["A"]["seconds"] *= 1000
    assert M.apply_criterion(slow)["taken"]["cost_s"] == 20.0


def test_d38_tier_comes_before_five_versus_four():
    # every point passes at xi0 0.25 on five maturities, but xi0 0.04 fails at T 2.00 everywhere:
    # the only five-maturity candidates exclude D38's xi0, so the draft's order takes the four-maturity
    # grid with a candidate containing D38's point
    def r2(nu, T):
        return lambda x, N: not (x == 0.04 and T == 2.0)
    recs = [fake_record(H, nu, rho, T, (0.04, 0.10, 0.25), r2(nu, T))
            for H in (0.02, 0.48) for nu in (0.05, 0.35) for rho in (-0.7, 0.0) for T in M.TS]
    st = M.apply_criterion(recs)
    assert [c["bounds"]["xi0"] for c in st["passing"]["five"]] == [[0.1, 0.25]]
    assert st["tier"] == "containing D38's point" and st["grid"] == "four"
    assert st["taken"]["bounds"] == {"nu": [0.05, 0.35], "rho": [-0.7, 0.0], "xi0": [0.04, 0.25]}
    assert set(st["taken"]["resolution"]) == {"0.1", "0.25", "0.5", "1.0"}


def test_tie_break_is_the_written_one_and_not_float_noise():
    # nu widths 0.35-0.05 and 0.65-0.35 are equal in value but not in floating point; both candidates
    # contain D38's point only if nu 0.35 is inside, so make rho the tied dimension instead: widths
    # 0.35 (rho -0.70..-0.35 and -0.35..0.00) with everything passing
    recs = [fake_record(H, nu, rho, T, (0.04, 0.25), lambda x, N: True)
            for H in (0.02, 0.48) for nu in (0.05, 0.35) for rho in (-0.7, -0.35, 0.0) for T in M.TS]
    st = M.apply_criterion(recs)
    assert st["taken"]["bounds"]["rho"] == [-0.7, 0.0] and st["tie"] == []     # the widest wins outright
    # remove the widest by making rho -0.70 fail with rho 0.00 inside the same box: build two equal boxes
    def r3(rho_lo_fails):
        return lambda x, N: True
    narrow = [r for r in recs if r["rho"] in (-0.7, -0.35)]
    narrow += [fake_record(H, nu, 0.0, T, (0.04, 0.25), lambda x, N: True)
               for H in (0.02, 0.48) for nu in (0.05, 0.35) for T in M.TS]
    # candidates rho [-0.7, -0.35] and [-0.35, 0.0] have equal width; the smaller lower bound wins the tie
    for r in narrow:
        for g in r["rungs"]:
            if r["rho"] == -0.7 and r["T"] == 0.1:
                pass
    st2 = M.apply_criterion([r for r in narrow] + [])
    keys = [M.rank_key({k: tuple(v) for k, v in c["bounds"].items()})[:4] for c in st2["passing"]["five"]]
    a, b = ({"nu": (0.05, 0.35), "rho": (-0.7, -0.35), "xi0": (0.04, 0.25)}, {"nu": (0.05, 0.35), "rho": (-0.35, 0.0), "xi0": (0.04, 0.25)})
    assert M.rank_key(a)[:4] == M.rank_key(b)[:4]                                   # equal to nine places
    assert (0.65 - 0.35) != (0.35 - 0.05) and round(0.65 - 0.35, 9) == round(0.35 - 0.05, 9)
    assert M.rank_key(a) > M.rank_key(b)                                            # the smaller lower bound on rho wins
    assert st2["taken"]["bounds"]["rho"] == [-0.7, 0.0]                             # (the widest still passes here)


def test_passes_from_requires_every_finer_rung():
    rec = fake_record(0.1, 0.35, -0.7, 0.1, (0.04,), lambda x, N: N in (2000, 8000))     # passes at 2000, fails at 4000
    assert M.passes_from(rec, "0.04", 2000) is False and M.passes_from(rec, "0.04", 4000) is False
    assert M.passes_from(rec, "0.04", 8000) is True
    assert M.point_resolution({0.1: rec}, "0.04", (0.1,), LAD) == {0.1: 8000}
    # a task stopped early has no finer full rung: it passes from its final rung onward
    short = fake_record(0.1, 0.35, -0.7, 0.1, (0.04,), lambda x, N: True)
    short["rungs"] = short["rungs"][:1]
    assert M.passes_from(short, "0.04", 2000) is True and M.passes_from(short, "0.04", 4000) is False


def test_confirmation_state_machine():
    recs = lattice_records()
    cs = M.criterion_state(recs)
    assert cs["state"].startswith("attempt 1 pending") and cs["next_draw"]["attempt"] == 1 and cs["next_draw"]["seed"] == 61
    cand = cs["next_draw"]["candidate"]
    assert set(cand) == {"bounds", "grid", "resolution"} and cand["grid"] == "five"
    b = cand["bounds"]
    pts = cs["next_draw"]["points"]
    assert len(pts) == 8 and pts == M.draw_confirm({k: tuple(v) for k, v in b.items()}, 61)   # reproducible
    for H, nu, rho, xi0 in pts:
        assert 0.02 <= H <= 0.48 and b["nu"][0] <= nu <= b["nu"][1] and b["rho"][0] <= rho <= b["rho"][1]
        assert b["xi0"][0] <= xi0 <= b["xi0"][1]
    good = confirm_records(pts, cand, 1)
    # an incomplete attempt is neither confirmed nor failed; its draw is restored so --confirm can resume,
    # and the order is applied without its partial records
    cs1 = M.criterion_state(recs + good[:7])
    assert cs1["state"].startswith("attempt 1 incomplete: 7 of 40") and cs1["attempts_done"] == []
    assert cs1["next_draw"] == cs["next_draw"] and cs1["taken"]["bounds"] == b
    cs2 = M.criterion_state(recs + good)
    assert cs2["state"].startswith("confirmed") and cs2["taken"]["bounds"] == b and cs2["next_draw"] is None
    # confirming points that pass only from 8000 move the resolution: not confirmed, attempt 2 drawn
    late = confirm_records(pts, cand, 1, rule_fn=lambda nu, T: (lambda x, N: N >= 8000))
    cs3 = M.criterion_state(recs + late)
    assert cs3["state"].startswith("attempt 2 pending") and cs3["next_draw"]["seed"] == 62
    assert cs3["next_draw"]["candidate"]["resolution"] == {str(T): 8000 for T in M.TS}
    # attempt 2 confirms the moved candidate
    cand2 = cs3["next_draw"]["candidate"]
    good2 = confirm_records(cs3["next_draw"]["points"], cand2, 2, rule_fn=lambda nu, T: (lambda x, N: N >= 8000))
    assert M.criterion_state(recs + late + good2)["state"].startswith("confirmed: the candidate taken is the one confirmed by attempt 2")
    # attempt 1 fails everywhere: every candidate containing those points is struck, attempt 2 is drawn
    # for the next candidate; if attempt 2 then moves that candidate's resolution, the two attempts are
    # used up and no candidate passes; if attempt 2's points fail everywhere too, nothing passes at all
    bad1 = confirm_records(pts, cand, 1, rule_fn=lambda nu, T: (lambda x, N: False))
    cs4 = M.criterion_state(recs + bad1)
    assert cs4["state"].startswith("attempt 2 pending") and cs4["next_draw"]["candidate"]["bounds"] != b
    cand2 = cs4["next_draw"]["candidate"]
    late2 = confirm_records(cs4["next_draw"]["points"], cand2, 2, rule_fn=lambda nu, T: (lambda x, N: N >= 8000))
    cs5 = M.criterion_state(recs + bad1 + late2)
    assert cs5["state"].startswith("confirmations used up") and cs5["taken"] is None and cs5["next_draw"] is None
    bad2 = confirm_records(cs4["next_draw"]["points"], cand2, 2, rule_fn=lambda nu, T: (lambda x, N: False))
    cs6 = M.criterion_state(recs + bad1 + bad2)
    assert cs6["state"].startswith("no candidate passes") and cs6["taken"] is None and cs6["next_draw"] is None


def test_output_schema_and_estimate(tmp_path):
    jl, js, cs = (str(tmp_path / n) for n in ("m.jsonl", "m.json", "m.csv"))
    recs = lattice_records()[:3]
    conf = [fake_record(0.1, 0.2, -0.5, 0.1, (0.05,), rule(0.2, 0.1), kind="confirm", attempt=1,
                        candidate={"bounds": {"nu": [0.05, 0.35]}, "grid": "five", "resolution": {}}, label="C1_00_T0.10")]
    M.append_line(jl, M.header_main(2))
    for r in recs + conf:
        M.append_line(jl, r)
    s = M.report_main(jl, js, cs)
    hdr, launches, tasks, confirms = M.read_main(jl)
    assert hdr["n_tasks"] == 625 and hdr["ladder"] == [2000, 4000, 8000] and hdr["partner"] == 16000
    assert hdr["base_T"]["0.1"] == [790.0, 504] and hdr["base_T"]["2.0"] == [180.0, 112] and hdr["seeds"] == [61, 62]
    assert s["n_tasks_done"] == 3 and s["n_confirm_done"] == 1 and len(tasks) == 3 and list(confirms) == ["C1_00_T0.10"]
    json.dumps(json.loads(Path(js).read_text(encoding="utf-8")), allow_nan=False)
    rows = list(csv.DictReader(Path(cs).read_text(encoding="utf-8").splitlines()))
    assert list(rows[0].keys()) == ["kind", "attempt"] + P.CSV_COLUMNS
    assert len(rows) == 3 * 2 * 3 * 3 + 1 * 1 * 3 * 3                   # tasks: xi0 x z x rungs; confirm: 1 xi0
    assert {r["kind"] for r in rows} == {"task", "confirm"}
    est = M.estimate_main({T: {"A": 1.0, "C": 1.5, "D": 1.5} for T in M.TS}, tasks=M.TASKS_MAIN[:5])
    p = est["per_T"][0.1]
    assert p["rung_s"][2000] == 4.0 and p["full_s"] == 4.0 * (1 + 4 + 16) + 1.0 * 64 and p["stop_at_s"][2000] == 4.0 + 4.0
    assert est["bound_core_s"] == 5 * p["full_s"]
