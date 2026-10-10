"""
p5_g0_main.py — P5 gate G0, the main run (docs/protocols/P5_protocol_v2_draft.md, Gates,
G0, line 140; its settings written in with the scoping run's output in view, D61).

The main run prices the pilot points below at the settings below and records, for each
point and maturity, the same quantities as the scoping run (p5_g0_pilot.py): the seconds
of each group of three Riccati solves; whether each quote is finite and at which stage it
failed; the three comparisons (`N_riccati` doubled, node count doubled, `U_max` raised) at
the tolerance 0.0001; and the modulus of the characteristic function at the highest node.
The criterion of line 140 is then applied to that output by `--criterion`, which is a
reading of the record, not a pricing: it takes the box, the grid and the resolution by the
written order and draws the confirming points; `--confirm` prices them. Nothing here
compares quotes across parameter values.

Settings (line 140, as values):
  base inversion setting per maturity: `U_max` = 250/sqrt(T) rounded, nodes at the scoping
  spacing (`U_max`/1.5625 to the nearest multiple of 8); node count doubled at the same
  `U_max`; `U_max` doubled with nodes doubled, compared with the doubled-node setting;
  ladder `N_riccati` 2000, 4000, 8000 per maturity, with a partner base solve at 16000 for
  the top rung's `N_riccati` comparison (16000 itself is not a resolution: it is tried
  only as the partner, on cost); the stop rule in `run_task_main`; the lattice below at
  six xi0 values; a candidate box has distinct lattice bounds in each of nu, rho and xi0;
  the cost of a resolution is the median over the triples priced of the summed seconds of
  their base solves over the grid (line 140); eight confirming points per attempt, two
  attempts, seeds 61 and 62; the tie in share broken by the share of nu, then rho, then
  xi0, then the smaller lower bound on xi0, nu and rho in turn, shares compared to nine
  decimal places. No flag changes a point or a numerical setting; `--pool` defaults to
  the written four and the launch line records the value used.

Usage (from the repository root):
    py -3 p5_g0_main.py              identity gate, design, timing, estimate; holds
    py -3 p5_g0_main.py --run        the worker pool over the 625 tasks; resumable
    py -3 p5_g0_main.py --report     regenerate output/p5_g0_main.json and _quotes.csv
    py -3 p5_g0_main.py --criterion  apply line 140's order to the output; draw the confirming points
    py -3 p5_g0_main.py --confirm    price the confirming points of the pending attempt
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import csv
import json
import time
import argparse
import platform
import statistics
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g0_pilot as P
from p5_g0_pilot import (S0, R, TS, VUS, KAPPA, TOL, STATUS_OK, D38_POINT, solve_cf, price_setting,
                         skipped_setting, compare, not_computed, first_finite_N, task_id, append_line,
                         read_jsonl, csv_rows, task_summary, utc_now, git_state, _rel, _f, _xi_key)

# --------------------------------------------------------------------------- #
# The settings (line 140)
# --------------------------------------------------------------------------- #
U1 = 250.0                                          # U_max at T = 1; U_max(T) = U1 / sqrt(T), rounded
SPACING = 200.0 / 128.0                             # the scoping run's node spacing, U_max / n_nodes
BASE_T = {0.10: (790.0, 504), 0.25: (500.0, 320), 0.50: (350.0, 224), 1.00: (250.0, 160), 2.00: (180.0, 112)}


def settings_for(T):
    """A: the base (U_max, nodes); C: nodes doubled; D: U_max doubled with nodes doubled."""
    U, n = BASE_T[float(T)]
    return {"A": (U, n), "C": (U, 2 * n), "D": (2.0 * U, 2 * n)}


SETTINGS_T = {T: settings_for(T) for T in TS}
LADDER_MAIN = (2000, 4000, 8000)
PARTNER = 16000                                     # base solve only, for the top rung's N_riccati comparison
H_VALUES = P.H_VALUES                               # (0.02, 0.05, 0.10, 0.25, 0.48): the range of H is not narrowed
NU_VALUES = (0.05, 0.35, 0.50, 0.65, 1.00)
RHO_VALUES = (0.00, -0.35, -0.70, -0.90, -0.99)
XI0_MAIN = (0.02, 0.04, 0.07, 0.10, 0.16, 0.25)
H_RANGE = (0.02, 0.48)
RANGES = {"nu": (0.05, 1.00), "rho": (-0.99, 0.00), "xi0": (0.001, 0.25)}   # the proposed ranges (line 37), for the share
SEEDS = (61, 62)                                    # confirming draws; neither the sealed seed nor the probe's (7)
N_CONFIRM = 8
N_ATTEMPTS = 2
COST_LIMIT_S = 1200.0                               # line 140's cost limit, core-seconds per (H, nu, rho) over the grid
POOL_DEFAULT = 4
GRIDS = {"five": tuple(TS), "four": tuple(T for T in TS if T != 2.0)}

OUT_DIR = P.OUT_DIR
JSONL_PATH = os.path.join(OUT_DIR, "p5_g0_main_tasks.jsonl")
JSON_PATH = os.path.join(OUT_DIR, "p5_g0_main.json")
CSV_PATH = os.path.join(OUT_DIR, "p5_g0_main_quotes.csv")
CRITERION_PATH = os.path.join(OUT_DIR, "p5_g0_main_criterion.json")


def build_triples():
    """The 125 (H, nu, rho) triples: the 8 corner triples, D38's, then the rest with H ascending."""
    corners = [(H, nu, rho) for H in (H_VALUES[0], H_VALUES[-1])
               for nu in (NU_VALUES[0], NU_VALUES[-1])
               for rho in (RHO_VALUES[0], RHO_VALUES[-1])]
    out = corners + [D38_POINT[:3]]
    for t in [(H, nu, rho) for H in H_VALUES for nu in NU_VALUES for rho in RHO_VALUES]:
        if t not in out:
            out.append(t)
    return tuple(out)


TRIPLES_MAIN = build_triples()
TASKS_MAIN = tuple((t, T) for t in TRIPLES_MAIN for T in TS)


# --------------------------------------------------------------------------- #
# One task, with the main run's stop rule and partner
# --------------------------------------------------------------------------- #
def _met(cmp_, key, i, tol):
    d = cmp_["diffs"][key][i] if cmp_ and "diffs" in cmp_ else None
    return d is not None and abs(d) < tol


def _quote(q, key, i):
    return q[key]["quotes"][i] if q else None


def rung_is_final(rung, prev_rung, tol):
    """Line 140's stop rule, written in D61: a rung is final if every quote at every xi0
    either meets all three comparisons, or fails only the U_max comparison with the
    raised-U_max quote unchanged (< tol) from the rung before, so that more steps cannot
    change the outcome. Needs the rung's N2 comparison already filled in."""
    comps = rung["comparisons"]
    if not all(c in comps for c in ("N2", "nodes", "umax")):
        return False
    qa = rung["quotes"]["A"]
    for key in qa:
        for i in range(len(qa[key]["quotes"])):
            n2, nd, um = (_met(comps[c], key, i, tol) for c in ("N2", "nodes", "umax"))
            if n2 and nd and um:
                continue
            if n2 and nd and not um and prev_rung is not None:
                d_now = _quote(rung["quotes"].get("D"), key, i)
                d_prev = _quote(prev_rung["quotes"].get("D"), key, i)
                if d_now is not None and d_prev is not None and abs(d_now - d_prev) < tol:
                    continue
            return False
    return True


def run_task_main(triple, T, *, ladder=LADDER_MAIN, partner=PARTNER, settings=None, tol=TOL,
                  xi0s=XI0_MAIN, vus=VUS, kind="task", attempt=None, label=None, candidate=None):
    """Climb the ladder at the maturity's settings. At each rung: the base solve A; the
    previous rung's N_riccati comparison; if the previous rung is final (rung_is_final)
    stop, this rung recorded as partner only; else, if the base solve is finite, the C and
    D solves and the nodes and umax comparisons. At the top rung a partner base solve at
    `partner` gives its N_riccati comparison, and the top rung is then judged final or not.
    The partner is skipped where the top rung's base solve is not finite (as the
    comparison solves are). The record has the scoping run's shape (p5_g0_pilot.run_task)
    plus kind, attempt, candidate, final_N and stop_reason; `exhausted` is True whenever
    the top rung was reached, final or not."""
    H, nu, rho = (float(x) for x in triple)
    T = float(T)
    settings = settings or settings_for(T)
    vus = np.asarray(vus, dtype=float)
    UA, nA = settings["A"]
    rec = {"kind": kind, "task": label or task_id(triple, T), "attempt": attempt, "candidate": candidate,
           "H": H, "nu": nu, "rho": rho, "kappa": float(KAPPA), "T": T,
           "xi0s": [float(x) for x in xi0s], "z": [float(z) for z in vus], "tol": float(tol),
           "ladder": [int(n) for n in ladder], "partner": int(partner) if partner else None,
           "settings": {k: [float(v[0]), int(v[1])] for k, v in settings.items()},
           "anchors": {}, "rungs": [], "stopped_at": None, "final_N": None, "stop_reason": None,
           "exhausted": False, "seconds_total": None}
    anchors = rec["anchors"]
    t_start = time.perf_counter()
    prev, prev2 = None, None
    for N in ladder:
        rung = {"N": int(N), "role": "full", "solves": {}, "quotes": {}, "comparisons": {}}
        solA, intsA, uA = solve_cf(H, nu, rho, T, N, UA, nA)
        rung["solves"]["A"] = solA
        rung["quotes"]["A"] = price_setting(intsA, uA, T, UA, nA, xi0s, vus, anchors, N, allow_anchor=True,
                                            solve_status=solA["status"])
        if prev is not None:
            prev["comparisons"]["N2"] = compare(rung["quotes"]["A"], prev["quotes"]["A"], tol)
            if rung_is_final(prev, prev2, tol):
                rec["stopped_at"] = rec["final_N"] = prev["N"]
                rec["stop_reason"] = "all_met" if all(prev["comparisons"][c]["met"] for c in ("N2", "nodes", "umax")) \
                    else "truncation_only"
                rung["role"] = "partner_only"
                rec["rungs"].append(rung)
                break
        if solA["status"] == STATUS_OK:
            for name in ("C", "D"):
                U, n = settings[name]
                sol, ints, u = solve_cf(H, nu, rho, T, N, U, n)
                rung["solves"][name] = sol
                rung["quotes"][name] = price_setting(ints, u, T, U, n, xi0s, vus, anchors, N, allow_anchor=False,
                                                     solve_status=sol["status"])
            rung["comparisons"]["nodes"] = compare(rung["quotes"]["C"], rung["quotes"]["A"], tol)
            rung["comparisons"]["umax"] = compare(rung["quotes"]["D"], rung["quotes"]["C"], tol)
        else:
            for name in ("C", "D"):
                rung["solves"][name] = {"N": int(N), "U_max": float(settings[name][0]),
                                        "n_nodes": int(settings[name][1]), "seconds": None,
                                        "status": "skipped_base_nonfinite", "failed_at": None}
                rung["quotes"][name] = skipped_setting(xi0s, vus, "skipped_base_nonfinite")
            for c in ("nodes", "umax"):
                rung["comparisons"][c] = dict(not_computed(rung["quotes"]["A"]), met=False,
                                              status="skipped_base_nonfinite")
        rec["rungs"].append(rung)
        prev2, prev = prev, rung
    else:
        rec["exhausted"] = True
        if prev is not None and partner and prev["solves"]["A"]["status"] != STATUS_OK:
            prev["comparisons"]["N2"] = dict(not_computed(prev["quotes"]["A"]), met=False, status="skipped_base_nonfinite")
        elif prev is not None and partner:
            pr = {"N": int(partner), "role": "partner_only", "solves": {}, "quotes": {}, "comparisons": {}}
            solP, intsP, uP = solve_cf(H, nu, rho, T, partner, UA, nA)
            pr["solves"]["A"] = solP
            pr["quotes"]["A"] = price_setting(intsP, uP, T, UA, nA, xi0s, vus, anchors, partner, allow_anchor=True,
                                              solve_status=solP["status"])
            prev["comparisons"]["N2"] = compare(pr["quotes"]["A"], prev["quotes"]["A"], tol)
            rec["rungs"].append(pr)
            if rung_is_final(prev, prev2, tol):
                rec["stopped_at"] = rec["final_N"] = prev["N"]
                rec["stop_reason"] = "all_met" if all(prev["comparisons"][c]["met"] for c in ("N2", "nodes", "umax")) \
                    else "truncation_only"
        elif prev is not None:
            prev["comparisons"]["N2"] = not_computed(prev["quotes"]["A"])
    rec["seconds_total"] = time.perf_counter() - t_start
    return rec


# --------------------------------------------------------------------------- #
# The criterion (line 140), applied to the record
# --------------------------------------------------------------------------- #
def quote_passes(rung, key, i, tol):
    """A quote passes at a rung if it is finite under A, C and D and meets all three comparisons."""
    q = rung["quotes"]
    if any(name not in q for name in ("A", "C", "D")):
        return False
    if any(q[name][key]["status"][i] != STATUS_OK for name in ("A", "C", "D")):
        return False
    return all(_met(rung["comparisons"].get(c), key, i, tol) for c in ("N2", "nodes", "umax"))


def passes_from(rec, key, N, tol=TOL):
    """The point passes at resolution N for this xi0: at the rung N and at every finer
    full rung the record holds (line 140: "there and at every finer resolution checked")."""
    full = [r for r in rec["rungs"] if r["role"] == "full"]
    if not any(r["N"] == N for r in full):
        return False
    n = len(rec["z"])
    for r in full:
        if r["N"] >= N and not all(quote_passes(r, key, i, tol) for i in range(n)):
            return False
    return True


def point_resolution(recs_by_T, key, Ts, ladder, tol=TOL):
    """Per maturity, the coarsest N on the ladder at which the point passes for this xi0,
    or None where it passes at none."""
    out = {}
    for T in Ts:
        rec = recs_by_T.get(T)
        out[T] = next((N for N in ladder if rec is not None and passes_from(rec, key, N, tol)), None)
    return out


def collect_points(records):
    """(H, nu, rho) -> {T: record} for every pilot and confirmed point in the record;
    confirmed points carry their own single xi0."""
    pts = {}
    for rec in records:
        pts.setdefault((rec["H"], rec["nu"], rec["rho"]), {})[rec["T"]] = rec
    return pts


def cost_table(records, Ts, ladder):
    """For information only: the median seconds of the finite base solve at each (T, N)
    over the record, with the pool running."""
    c = {}
    for T in Ts:
        for N in ladder:
            secs = [r["solves"]["A"]["seconds"] for rec in records if rec["T"] == T
                    for r in rec["rungs"] if r["N"] == N and r["solves"]["A"]["status"] == STATUS_OK
                    and r["solves"]["A"]["seconds"] is not None]
            c[(T, N)] = statistics.median(secs) if secs else None
    return c


def cost_of_resolution(points, grid, res):
    """Line 140's cost quantity (D59): the Riccati solves of one (H, nu, rho) over the
    grid's maturities at the base setting, as the median over the points priced, with
    the pool running. Here: for each triple priced, the sum over the grid of the seconds
    of its finite base solve at rung res[T]; the median over the triples that have that
    solve at every maturity of the grid. Returns (median, n_triples)."""
    sums = []
    for triple, recs in points.items():
        total = 0.0
        for T in grid:
            rec = recs.get(T)
            rung = next((r for r in rec["rungs"] if r["N"] == res[T]), None) if rec else None
            sol = rung["solves"]["A"] if rung else None
            if sol is None or sol["status"] != STATUS_OK or sol["seconds"] is None:
                total = None
                break
            total += sol["seconds"]
        if total is not None:
            sums.append(total)
    return (statistics.median(sums) if sums else None), len(sums)


def share_of(bounds):
    s = {}
    for k in ("nu", "rho", "xi0"):
        lo, hi = RANGES[k]
        s[k] = (bounds[k][1] - bounds[k][0]) / (hi - lo)
    return s["nu"] * s["rho"] * s["xi0"], s


def contains_d38(bounds):
    H, nu, rho, xi0 = D38_POINT
    return (bounds["nu"][0] <= nu <= bounds["nu"][1] and bounds["rho"][0] <= rho <= bounds["rho"][1]
            and bounds["xi0"][0] <= xi0 <= bounds["xi0"][1])


def evaluate_candidate(bounds, points, grid, ladder, costs, tol=TOL, cost_limit=COST_LIMIT_S):
    """Does the candidate pass on the grid? Every point inside it (every H; its corners are
    lattice points) must pass at some ladder N at every maturity of the grid, for every
    xi0 inside the candidate; the resolution is the per-maturity maximum over points of
    the coarsest passing N (the coarsest at which all pass). A point whose task stopped at
    a coarser rung by the stop rule counts as passing at every finer N (the stop rule's
    claim; line 140 as written in by D61). Admissible if line 140's cost quantity at that
    resolution (cost_of_resolution) is within the limit; a resolution at which no triple was
    priced at every maturity of the grid has no cost and is not admissible (cost_n_triples
    records 0). `costs` (the per-(T, N) medians) is kept for information."""
    res = {T: 0 for T in grid}
    n_points = 0
    for (H, nu, rho), recs in points.items():
        if not (bounds["nu"][0] <= nu <= bounds["nu"][1] and bounds["rho"][0] <= rho <= bounds["rho"][1]):
            continue
        any_rec = next(iter(recs.values()))
        keys = [(_xi_key(x), x) for x in any_rec["xi0s"] if bounds["xi0"][0] <= x <= bounds["xi0"][1]]
        for key, _ in keys:
            n_points += 1
            pr = point_resolution(recs, key, grid, ladder, tol)
            for T in grid:
                if pr[T] is None:
                    return {"passes": False, "n_points": n_points, "fails_at": {"point": [H, nu, rho], "xi0": key, "T": T}}
                res[T] = max(res[T], pr[T])
    if n_points == 0:
        return {"passes": False, "n_points": 0, "fails_at": "no point inside"}
    cost, n_tr = cost_of_resolution(points, grid, res)
    admissible = cost is not None and cost <= cost_limit
    return {"passes": admissible, "admissible": admissible, "n_points": n_points,
            "resolution": {str(T): res[T] for T in grid}, "cost_s": _f(cost), "cost_n_triples": n_tr}


def candidates_from(points):
    """Every box whose bounds on nu, rho and xi0 are values priced (lattice values) and
    distinct in each (a box, not a face or an edge), H full."""
    nus = sorted({nu for (_, nu, _) in points})
    rhos = sorted({rho for (_, _, rho) in points})
    xis = sorted({x for recs in points.values() for rec in recs.values() for x in rec["xi0s"]})
    out = []
    for a, b in itertools.combinations(nus, 2):
        for c, d in itertools.combinations(rhos, 2):
            for e, f in itertools.combinations(xis, 2):
                out.append({"nu": (a, b), "rho": (c, d), "xi0": (e, f)})
    return out


def rank_key(bounds):
    """Line 140's ranking: the share, then the share of nu, rho and xi0, then the smaller
    lower bound on xi0, nu and rho in turn; shares compared to nine decimal places so that
    equal lattice widths are equal (0.65 - 0.35 is not 0.35 - 0.05 in floating point)."""
    total, s = share_of(bounds)
    return (round(total, 9), round(s["nu"], 9), round(s["rho"], 9), round(s["xi0"], 9),
            -bounds["xi0"][0], -bounds["nu"][0], -bounds["rho"][0])


def _bkey(b):
    return tuple(tuple(float(x) for x in b[k]) for k in ("nu", "rho", "xi0"))


def apply_criterion(records, ladder=LADDER_MAIN, tol=TOL, cost_limit=COST_LIMIT_S, lattice_only=True):
    """Line 140's order. Candidates are boxes with lattice bounds; a candidate passes on a
    grid at its resolution if admissible and every pilot point inside passes. The
    candidates considered are the passing ones (on either grid) that contain D38's point,
    or all passing ones if none contains it; if any of them passes on five maturities the
    grid is the five and only those are kept, otherwise the four; then the largest share
    of the proposed ranges, ties by rank_key. Returns the full account."""
    pilot = [r for r in records if r["kind"] in ("task", "confirm")]
    points = collect_points(pilot)
    lattice_points = {k: v for k, v in points.items()
                      if any(r["kind"] == "task" for r in v.values())} if lattice_only else points
    cands = candidates_from(lattice_points)
    costs = cost_table(pilot, TS, ladder)
    out = {"n_candidates": len(cands), "n_points": len(points), "costs": {f"{T}/{N}": _f(v) for (T, N), v in costs.items()},
           "passing": {"five": [], "four": []}}
    for name in ("five", "four"):
        grid = GRIDS[name]
        for b in cands:
            ev = evaluate_candidate(b, points, grid, ladder, costs, tol, cost_limit)
            if ev["passes"]:
                total, s = share_of(b)
                out["passing"][name].append({"bounds": {k: list(v) for k, v in b.items()}, "share": total, "shares": s,
                                             "contains_d38": contains_d38(b), **ev})
    passing_any = {_bkey(c["bounds"]): c for name in ("five", "four") for c in out["passing"][name]}
    if not passing_any:
        out["grid"], out["tier"], out["taken"] = None, None, None
        out["verdict"] = "no candidate passes on either grid"
        return out
    with_d38 = {k: c for k, c in passing_any.items() if c["contains_d38"]}
    tier_keys = set(with_d38) if with_d38 else set(passing_any)
    out["tier"] = "containing D38's point" if with_d38 else "all passing (none contains D38's point)"
    five_keys = {_bkey(c["bounds"]) for c in out["passing"]["five"]}
    chosen_grid = "five" if (tier_keys & five_keys) else "four"
    out["grid"] = chosen_grid
    tier = [c for c in out["passing"][chosen_grid] if _bkey(c["bounds"]) in tier_keys]
    tier_sorted = sorted(tier, key=lambda c: rank_key({k: tuple(v) for k, v in c["bounds"].items()}), reverse=True)
    out["taken"] = tier_sorted[0]
    top_key = rank_key({k: tuple(v) for k, v in tier_sorted[0]["bounds"].items()})[:4]
    out["tie"] = [c["bounds"] for c in tier_sorted[1:]
                  if rank_key({k: tuple(v) for k, v in c["bounds"].items()})[:4] == top_key]
    out["verdict"] = "a candidate is taken"
    return out


def draw_confirm(bounds, seed, n=N_CONFIRM):
    """n points uniform in the candidate, H over its full range."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        H = rng.uniform(*H_RANGE)
        nu = rng.uniform(*bounds["nu"])
        rho = rng.uniform(*bounds["rho"])
        xi0 = rng.uniform(*bounds["xi0"])
        pts.append([round(float(H), 6), round(float(nu), 6), round(float(rho), 6), round(float(xi0), 6)])
    return pts


def criterion_state(records, ladder=LADDER_MAIN, tol=TOL, n_per_attempt=None):
    """The order applied, then the confirmation state. An attempt counts only when all its
    confirming tasks are priced (N_CONFIRM points at every maturity). The candidate is
    confirmed when the one taken after the attempt's points are included equals, in
    bounds, grid and resolution, the one the attempt was drawn for (line 140: "if the
    candidate then no longer passes on that grid at that resolution the order is applied
    again"). Otherwise the next attempt is drawn, or, after N_ATTEMPTS, no candidate
    passes."""
    n_per_attempt = n_per_attempt or N_CONFIRM * len(TS)
    confirms = [r for r in records if r["kind"] == "confirm"]
    attempts = sorted({r["attempt"] for r in confirms})
    if attempts:
        k = attempts[-1]
        n_k = sum(1 for r in confirms if r["attempt"] == k)
        if n_k < n_per_attempt:
            # the attempt is judged only when complete: the order is applied without its records,
            # and its draw (deterministic) is restored so that --confirm can resume it
            out = apply_criterion([r for r in records if not (r["kind"] == "confirm" and r["attempt"] == k)], ladder, tol)
            out["attempts_done"] = attempts[:-1]
            cand = next(r["candidate"] for r in confirms if r["attempt"] == k)
            out["state"] = f"attempt {k} incomplete: {n_k} of {n_per_attempt} confirming tasks priced; run --confirm"
            out["next_draw"] = {"attempt": k, "seed": SEEDS[k - 1], "candidate": cand,
                                "points": draw_confirm({kk: tuple(v) for kk, v in cand["bounds"].items()}, SEEDS[k - 1])}
            return out
    out = apply_criterion(records, ladder, tol)
    out["attempts_done"] = attempts
    if out["taken"] is None:
        out["state"] = "no candidate passes: the draft is not frozen as it stands (line 140)"
        out["next_draw"] = None
        return out
    taken = {"bounds": out["taken"]["bounds"], "grid": out["grid"], "resolution": out["taken"]["resolution"]}
    last = None
    if attempts:
        last = next(r["candidate"] for r in confirms if r["attempt"] == attempts[-1])
    if last is not None and last == taken:
        out["state"] = f"confirmed: the candidate taken is the one confirmed by attempt {attempts[-1]}"
        out["next_draw"] = None
    elif len(attempts) >= N_ATTEMPTS:
        out["state"] = f"confirmations used up ({N_ATTEMPTS}): no candidate passes (line 140)"
        out["taken"] = None
        out["next_draw"] = None
    else:
        k = len(attempts) + 1
        out["state"] = f"attempt {k} pending: price the confirming points, then re-apply"
        out["next_draw"] = {"attempt": k, "seed": SEEDS[k - 1], "candidate": taken,
                            "points": draw_confirm({kk: tuple(v) for kk, v in taken["bounds"].items()}, SEEDS[k - 1])}
    return out


# --------------------------------------------------------------------------- #
# Output, pool, estimate
# --------------------------------------------------------------------------- #
def header_main(pool):
    import scipy
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    return {"kind": "header", "script": os.path.basename(__file__), "commit": commit, "clean_tree": clean,
            "seeds": list(SEEDS), "n_confirm": N_CONFIRM, "n_attempts": N_ATTEMPTS, "tol": TOL,
            "U1": U1, "spacing": SPACING, "base_T": {str(T): [float(v[0]), int(v[1])] for T, v in BASE_T.items()},
            "settings_T": {str(T): {k: [float(v[0]), int(v[1])] for k, v in s.items()} for T, s in SETTINGS_T.items()},
            "ladder": list(LADDER_MAIN), "partner": PARTNER, "cost_limit_s": COST_LIMIT_S,
            "H_values": list(H_VALUES), "nu_values": list(NU_VALUES), "rho_values": list(RHO_VALUES),
            "xi0_values": list(XI0_MAIN), "triples": [list(t) for t in TRIPLES_MAIN], "n_tasks": len(TASKS_MAIN),
            "TS": list(TS), "z": [float(z) for z in VUS], "S0": float(S0), "r": float(R), "kappa": float(KAPPA),
            "d38_point": list(D38_POINT), "ranges": {k: list(v) for k, v in RANGES.items()},
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "pool": int(pool), "utc_created": utc_now()}


def read_main(jsonl_path=None):
    """Header, launches, and every task and confirm record (by id, last wins). Paths
    default to the module's at call time, so a test can point them elsewhere."""
    jsonl_path = jsonl_path or JSONL_PATH
    hdr, launches, tasks = read_jsonl(jsonl_path)
    confirms = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("kind") == "confirm":
                    confirms[obj["task"]] = obj
    return hdr, launches, tasks, confirms


def report_main(jsonl_path=None, json_path=None, csv_path=None):
    jsonl_path, json_path, csv_path = jsonl_path or JSONL_PATH, json_path or JSON_PATH, csv_path or CSV_PATH
    hdr, launches, tasks, confirms = read_main(jsonl_path)
    ordered = [tasks[task_id(t, T)] for t, T in TASKS_MAIN if task_id(t, T) in tasks]
    ordered += [rec for k, rec in tasks.items() if k not in {task_id(t, T) for t, T in TASKS_MAIN}]
    ordered += [confirms[k] for k in sorted(confirms)]
    summary = {"header": hdr, "launches": launches, "n_tasks_done": len(tasks), "n_tasks_total": len(TASKS_MAIN),
               "n_confirm_done": len(confirms), "utc_report": utc_now(),
               "cost": P.cost_summary(tasks, ladder=LADDER_MAIN),
               "stop_reasons": {k: sum(1 for r in tasks.values() if r["stop_reason"] == k)
                                for k in ("all_met", "truncation_only", None)},
               "tasks": [dict(task_summary(r), kind=r["kind"], attempt=r.get("attempt"), final_N=r.get("final_N"),
                              stop_reason=r.get("stop_reason")) for r in ordered]}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, allow_nan=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "attempt"] + P.CSV_COLUMNS)
        w.writeheader()
        for rec in ordered:
            for row in csv_rows(rec):
                w.writerow({"kind": rec["kind"], "attempt": rec.get("attempt"),
                            **{k: ("" if v is None else v) for k, v in row.items()}})
    return summary


def _worker_main(job):
    triple, T, kw = job
    return run_task_main(triple, T, **kw)


def launch_main(pool, jobs, jsonl_path=None, json_path=None, csv_path=None, timing=None):
    """The pool over `jobs` = [(triple, T, kwargs)], skipping ids already in the jsonl.
    One line per finished task; a task that raises is recorded and the rest go on."""
    jsonl_path, json_path, csv_path = jsonl_path or JSONL_PATH, json_path or JSON_PATH, csv_path or CSV_PATH
    hdr, launches, tasks, confirms = read_main(jsonl_path)
    done = set(tasks) | set(confirms)
    if hdr is None:
        append_line(jsonl_path, header_main(pool))
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    append_line(jsonl_path, {"kind": "launch", "utc": utc_now(), "pool": int(pool), "n_done_before": len(done),
                             "commit": commit, "clean_tree": clean, "timing_alone_s": timing})
    todo = [j for j in jobs if (j[2].get("label") or task_id(j[0], j[1])) not in done]
    print(f"\n=== RUN: {len(todo)} tasks to do ({len(done)} already in {_rel(jsonl_path)}), pool={pool} ===", flush=True)
    t0 = time.time()
    n = n_err = 0
    ex = ProcessPoolExecutor(max_workers=pool)
    try:
        futs = {ex.submit(_worker_main, job): job for job in todo}
        for fut in as_completed(futs):
            triple, T, kw = futs[fut]
            lab = kw.get("label") or task_id(triple, T)
            try:
                rec = fut.result()
            except Exception as e:
                n_err += 1
                append_line(jsonl_path, {"kind": "task_error", "task": lab, "utc": utc_now(), "error": repr(e)})
                print(f"  task {lab} raised {e!r}; recorded, continuing", flush=True)
                continue
            append_line(jsonl_path, rec)
            report_main(jsonl_path, json_path, csv_path)
            n += 1
            print(f"  [{n:3d}/{len(todo)}] {rec['task']}: final_N={rec['final_N']} ({rec['stop_reason']}) "
                  f"exhausted={rec['exhausted']} first_finite_N(A)={first_finite_N(rec)} "
                  f"task={rec['seconds_total']:.0f}s elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    except BaseException:
        print("\n  stopping: pending tasks cancelled; re-run to resume", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    print(f"=== done: {n} tasks ({n_err} errors) in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    return n


def estimate_main(timing, ladder=LADDER_MAIN, partner=PARTNER, tasks=TASKS_MAIN):
    """timing: {T: {"A": s, "C": s, "D": s}} for the three settings at N_ref = 2000, alone.
    Cost model per setting c(N) = t * (N / 2000)^2 (the recursion is O(N^2)). A task that
    exhausts the ladder costs the sum over rungs of (A + C + D) plus the partner's A."""
    per_T = {}
    for T, tt in timing.items():
        T = float(T)
        rung = {N: sum(tt[s] * (N / 2000.0) ** 2 for s in ("A", "C", "D")) for N in ladder}
        full = sum(rung.values()) + tt["A"] * (partner / 2000.0) ** 2
        cum = {}
        run = 0.0
        for i, N in enumerate(ladder):
            run += rung[N]
            nxt = tt["A"] * (ladder[i + 1] / 2000.0) ** 2 if i + 1 < len(ladder) else tt["A"] * (partner / 2000.0) ** 2
            cum[N] = run + nxt
        per_T[T] = {"rung_s": rung, "full_s": full, "stop_at_s": cum, "base_s": {N: tt["A"] * (N / 2000.0) ** 2 for N in ladder}}
    bound = sum(per_T[float(T)]["full_s"] for _, T in tasks)
    guess = 0.0
    for (H, nu, rho), T in tasks:
        p = per_T[float(T)]
        if H >= 0.25:
            guess += p["stop_at_s"][4000]
        elif H >= 0.10:
            guess += p["stop_at_s"][8000]
        else:
            guess += 0.5 * p["full_s"] + 0.5 * p["stop_at_s"][8000]
    return {"per_T": per_T, "bound_core_s": bound, "guess_core_s": guess}


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P5 gate G0 main run (see module docstring)")
    ap.add_argument("--run", action="store_true", help="launch the worker pool over the pilot points (resumable)")
    ap.add_argument("--report", action="store_true", help="regenerate the JSON and CSV from the jsonl and exit")
    ap.add_argument("--criterion", action="store_true", help="apply line 140's order to the record and exit")
    ap.add_argument("--confirm", action="store_true", help="price the pending attempt's confirming points")
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT, help="worker processes (default: 4, as written in)")
    a = ap.parse_args(argv)

    if a.report:
        s = report_main()
        print(f"report: {s['n_tasks_done']}/{s['n_tasks_total']} tasks, {s['n_confirm_done']} confirm -> "
              f"{_rel(JSON_PATH)}, {_rel(CSV_PATH)}", flush=True)
        return
    if a.criterion:
        hdr, launches, tasks, confirms = read_main()
        st = criterion_state(list(tasks.values()) + list(confirms.values()))
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CRITERION_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(st, utc=utc_now()), f, indent=1, allow_nan=False)
        print(f"=== criterion (line 140) over {st['n_points']} points, {st['n_candidates']} candidates ===", flush=True)
        print(f"  passing on five maturities: {len(st['passing']['five'])}; on four: {len(st['passing']['four'])}; "
              f"grid: {st['grid']}; tier: {st.get('tier')}", flush=True)
        if st["taken"]:
            t = st["taken"]
            print(f"  taken: nu {t['bounds']['nu']} rho {t['bounds']['rho']} xi0 {t['bounds']['xi0']}  share={t['share']:.4f} "
                  f"resolution={t['resolution']} cost={t['cost_s']:.0f}s over {t['cost_n_triples']} triples  "
                  f"ties={len(st['tie'])}", flush=True)
        print(f"  state: {st['state']}", flush=True)
        if st["next_draw"]:
            print(f"  next draw (attempt {st['next_draw']['attempt']}, seed {st['next_draw']['seed']}): "
                  f"{len(st['next_draw']['points'])} points -> run --confirm", flush=True)
        print(f"  written: {_rel(CRITERION_PATH)}", flush=True)
        return
    if a.confirm:
        with open(CRITERION_PATH, encoding="utf-8") as f:
            st = json.load(f)
        nd = st.get("next_draw")
        if not nd:
            raise SystemExit("no pending draw in the criterion file; run --criterion first")
        jobs = []
        for i, (H, nu, rho, xi0) in enumerate(nd["points"]):
            for T in TS:
                jobs.append(((H, nu, rho), T, {"xi0s": (xi0,), "kind": "confirm", "attempt": nd["attempt"],
                                               "label": f"C{nd['attempt']}_{i:02d}_T{T:.2f}", "candidate": nd["candidate"]}))
        print(f"=== confirm attempt {nd['attempt']} (seed {nd['seed']}): {len(nd['points'])} points x {len(TS)} maturities ===",
              flush=True)
        launch_main(a.pool, jobs)
        print("  now run --criterion again", flush=True)
        return

    print("=== P5 G0 main run ===", flush=True)
    print(f"  tol={TOL}  ladder={LADDER_MAIN} partner={PARTNER}  pool={a.pool}  seeds={SEEDS}", flush=True)
    for T in TS:
        print(f"  T={T}: A={SETTINGS_T[T]['A']} C={SETTINGS_T[T]['C']} D={SETTINGS_T[T]['D']}", flush=True)
    print(f"  lattice: H{H_VALUES} x nu{NU_VALUES} x rho{RHO_VALUES} = {len(TRIPLES_MAIN)} triples; "
          f"xi0{XI0_MAIN}; {len(TASKS_MAIN)} tasks; first nine: {TRIPLES_MAIN[:9]}", flush=True)

    t0 = time.perf_counter()
    d, n = P.identity_gate()
    print(f"\n=== identity gate at the scoping base (200, 128): max |path - model_smile_cf_T| over {n} quotes = {d!r} "
          f"({time.perf_counter() - t0:.0f}s) ===", flush=True)
    if d != 0.0:
        raise SystemExit("identity gate failed")

    print("\n=== timing alone (D38's triple, N=2000, the three settings per maturity) ===", flush=True)
    H, nu, rho, _ = D38_POINT
    timing = {}
    for T in TS:
        timing[str(T)] = {}
        status = {}
        for name, (U, nn) in SETTINGS_T[T].items():
            rec, _, _ = solve_cf(H, nu, rho, T, 2000, U, nn)
            timing[str(T)][name] = rec["seconds"]
            status[name] = rec["status"]
        print(f"  T={T}: " + "  ".join(f"{k}={v:.2f}s ({status[k]})" for k, v in timing[str(T)].items())
              + "   (a solve that is not finite stops at its failing stage and reads as cheap)", flush=True)
    est = estimate_main({float(k): v for k, v in timing.items()})
    print("\n=== estimate (O(N^2) from N=2000; one core, alone) ===", flush=True)
    top = LADDER_MAIN[-1]
    for T in TS:
        p = est["per_T"][T]
        print(f"  T={T}: full ladder {p['full_s']:7.0f}s/task; stopping at " +
              " ".join(f"{N}:{p['stop_at_s'][N]:.0f}s" for N in LADDER_MAIN) +
              f"; base solves at {top} {p['base_s'][top]:.0f}s", flush=True)
    print(f"  one triple's base solves over the five maturities at {top}: "
          f"{sum(est['per_T'][T]['base_s'][top] for T in TS):.0f}s alone (the cost limit is {COST_LIMIT_S:.0f}s with the pool running)",
          flush=True)
    print(f"  bound (every task exhausts): {est['bound_core_s'] / 3600:.0f} core-hours = "
          f"{est['bound_core_s'] / 3600 / a.pool:.0f} h on {a.pool} workers, before the pool's slowdown (2.2 to 2.6x in the scoping run)",
          flush=True)
    print(f"  guess (not a measurement; H>=0.25 stop by 4000, H 0.10 by 8000, H<=0.05 half exhaust): "
          f"{est['guess_core_s'] / 3600:.0f} core-hours = {est['guess_core_s'] / 3600 / a.pool:.0f} h on {a.pool} workers, before the slowdown",
          flush=True)
    if not a.run:
        print("\nHOLD: nothing written. Add --run to launch (mains power, sleep off; Ctrl-C stops, --run resumes).", flush=True)
        return
    jobs = [(t, T, {}) for t, T in TASKS_MAIN]
    launch_main(a.pool, jobs, timing=timing)
    print("  now run --criterion", flush=True)


if __name__ == "__main__":
    main()
