"""
p5_g1_emulator.py — P5 gate G1, the emulator G̃ (docs/protocols/P5_protocol_v2_draft.md,
Objects "Emulator form and design", D64; Gates, G1).

G̃ is a tensor-product Chebyshev interpolant of the forward map G on the box, one
interpolant per quote of the grid, in (H, ν, ρ, ξ₀) mapped affinely to [−1, 1]⁴, on
Chebyshev nodes of the first kind, evaluated by the barycentric formula. It is
deterministic: no training, no seed, exact at its nodes. The node values are priced by
the path of p5_g0_main.py at the resolution and inversion settings written in under
Gates (N_riccati 8000; U_max 790, 500, 350, 250, 180 with 504, 320, 224, 160, 112 nodes),
each surface's strikes anchored on its own ATM quote at that setting: the pilot's
solve_cf / make_cf / quote / price_setting, unchanged, three Riccati solves per
(H, ν, ρ, T) shared across strikes and ξ₀. A training or held-out point with a quote that
is not finite is priced once more at N_riccati 16000 with the same inversion settings,
its quotes then taken from that pricing (the triple is solved again; its other surfaces
keep their quotes); a quote still not finite stops the build and P5 (line 54, D81). The
held-out set is 200 points uniform on the box from numpy's
default_rng(73), priced the same way; the emulator error e of G1 is the 99th percentile
over their 7,000 quotes of |G̃ − G|, with the maximum and the per-maturity 99th
percentiles beside it. G1: Q1 is claimed at a noise level s only where e ≤ s / 5. If e
exceeds s / 5 at the smallest level (0.1 vol point, so 0.0002), the node count is doubled
once in the dimension whose Chebyshev coefficient tail is largest and the build repeated
at the new nodes; there is no second refinement (line 142). G̃ for everything after G1 is
the build G1 passes: require_g1() returns the refined build when the base report calls for
the refinement and the base build otherwise, and refuses a stage that stopped or is
incomplete; G2 and G3 take their emulator from it on the command line, and every later
record carries its SHA-256 (line 142, D82). No report, CSV or emulator is written over (D82).

The build is G1 and is made after freezing (lines 7 and 54). This file is committed
before it; its tests run the pricer at cheap settings on D38's point, as G0's identity gate
did, and one of them builds a 2 × 2 × 2 × 2 toy interpolant from the pricer at N_riccati 200
on a small box near D38's point, which is not the emulator of the design (declared, D81).

Run from the command line, the script refuses numpy and scipy versions other than the core
interpreter's on the author's machine on 22 October 2026 (CORE_VERSIONS) unless
--allow-other-versions is given; the versions and any override are recorded in the header
(D81). Imports, tests and CI are unaffected.

Usage (from the repository root):
    py -3 p5_g1_emulator.py                 design, identity gate at the scoping base (N_riccati 1000, 128 nodes), estimate; holds
    py -3 p5_g1_emulator.py --run           the 768 node triples and the 200 held-out points; resumable
    py -3 p5_g1_emulator.py --assemble      the .npz, the held-out CSV and the report with G1's verdicts
    py -3 p5_g1_emulator.py --refine        the one permitted refinement, only when the report calls for it
    py -3 p5_g1_emulator.py --assemble --stage refined
    py -3 p5_g1_emulator.py --estimate      the cost from D62's 288 core-seconds per triple; no pricing

Outputs (output/ is git-ignored; the files are force-added by name after the build):
    output/p5_g1_tasks.jsonl        header, launches, one line per priced triple or held-out point
    output/p5_g1_emulator.npz       node values [16, 8, 6, 12, 35], nodes, box, settings
    output/p5_g1_heldout.csv        one row per held-out quote: G, G̃ and their difference
    output/p5_g1_report.json        e, the maximum, per-maturity 99th percentiles, G1's verdicts,
                                    the coefficient tails and the refinement decision
    the same four names with _refined for the refined stage
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (calibrate_btc.py)

import sys
import csv
import json
import time
import hashlib
import argparse
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import p5_g0_pilot as P
import p5_g0_main as M
from p5_g0_pilot import (S0, R, TS, VUS, KAPPA, STATUS_OK, D38_POINT, solve_cf, price_setting, _xi_key, _f,
                         git_state, utc_now, append_line, _rel, read_jsonl)

# --------------------------------------------------------------------------- #
# The design, as written in (lines 39, 41, 54, 141, 142)
# --------------------------------------------------------------------------- #
DIMS = ("H", "nu", "rho", "xi0")
BOX = {"H": (0.02, 0.48), "nu": (0.05, 0.35), "rho": (-0.90, -0.70), "xi0": (0.04, 0.25)}   # D62's box (line 39)
NODES = (16, 8, 6, 12)                              # first-kind Chebyshev nodes per dimension (line 54)
N_RICCATI = 8000                                    # the resolution at every maturity (line 141, D62)
N_REPRICE = 16000                                   # the one re-pricing of a point with a quote not finite (line 54, D81)
BASE_T = M.BASE_T                                   # T -> (U_max, n_nodes): 790/504 ... 180/112 (line 141)
SEED_HELDOUT = 73                                   # the held-out draw (line 54)
N_HELDOUT = 200
RATIO = 5                                           # G1: e <= s / RATIO (line 142)
NOISE_LEVELS = (0.001, 0.003, 0.005)                # s in {0.1, 0.3, 0.5} vol point, as decimals (line 56)
COST_PER_TRIPLE_S = 288.0                           # D62: median core-seconds per (H, nu, rho) over the grid
POOL_DEFAULT = 4
N_QUOTES = len(TS) * len(VUS)                       # 35, ordered by maturity then z, both ascending (line 45)

OUT_DIR = P.OUT_DIR

# The environments the runs use (D81): the core interpreter for G1, G2 and Stage 1 and 2, the
# P5 leaf (.venv-p5, requirements-p5.txt) for G3; as installed on the author's machine on
# 22 October 2026. numpy's generator streams are fixed by seed only together with its version.
CORE_VERSIONS = {"numpy": "2.4.4", "scipy": "1.17.1"}
LEAF_VERSIONS = {"numpy": "2.5.4", "scipy": "1.18.1"}
LEAF_TORCH = "2.14.1"                               # the leaf's torch, its local build tag (+cpu) aside (D82)


def versions_now(torch_too=False):
    import scipy
    out = {"numpy": np.__version__, "scipy": scipy.__version__}
    if torch_too:
        try:
            import torch
            out["torch"] = torch.__version__.split("+")[0]
        except Exception:
            out["torch"] = None
    return out


def version_check(expected, allow=False, script=None, check=True, torch_expected=None):
    """The installed numpy and scipy (and torch, when `torch_expected` is given) against
    `expected`, for a script's output header: {numpy, scipy, [torch,] expected, checked,
    override, mismatch}. With `check` (a command-line run), a mismatch is refused unless
    `allow` (--allow-other-versions), which is recorded as the override. Without `check` (a
    call from a test or another script) nothing is refused and the record says so."""
    now = versions_now(torch_expected is not None)
    exp = dict(expected, **({"torch": torch_expected} if torch_expected is not None else {}))
    bad = {k: [now[k], v] for k, v in exp.items() if now[k] != v}
    if check and bad and not allow:
        raise SystemExit(f"{script or 'this script'} refused: "
                         + "; ".join(f"{k} {a} is installed, {b} is expected" for k, (a, b) in bad.items())
                         + ". The draws are fixed by seed only together with these versions (D81); "
                           "--allow-other-versions runs anyway and is recorded.")
    return dict(now, expected=dict(exp), checked=bool(check), override=bool(check and bad and allow), mismatch=bad)


def paths(stage="base"):
    suf = "" if stage == "base" else "_" + stage
    return {"jsonl": os.path.join(OUT_DIR, f"p5_g1{suf}_tasks.jsonl"),
            "npz": os.path.join(OUT_DIR, f"p5_g1{suf}_emulator.npz"),
            "csv": os.path.join(OUT_DIR, f"p5_g1{suf}_heldout.csv"),
            "report": os.path.join(OUT_DIR, f"p5_g1{suf}_report.json")}


# --------------------------------------------------------------------------- #
# G̃ for everything after G1 (D82): which build, its hash, the checks where records meet
# --------------------------------------------------------------------------- #
def sha256_file(path):
    """The SHA-256 of a file's bytes, or None when there is no such file."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def emulator_info(path):
    """The emulator a run uses, for its output header: the path and the SHA-256 of the .npz."""
    return {"path": _rel(path) if path else None, "sha256": sha256_file(path)}


def require_same_emulator(hashes, where):
    """Refuses unless every emulator hash `where` reads ({source: SHA-256}) is present and
    all are equal (line 142, D82); returns the hash."""
    missing = sorted(k for k, v in hashes.items() if not v)
    if missing:
        raise SystemExit(f"{where} refused: no emulator hash in {', '.join(missing)}; every record after G1 carries "
                         "G̃'s (line 142, D82)")
    if len(set(hashes.values())) > 1:
        raise SystemExit(f"{where} refused: the emulator hashes differ ("
                         + "; ".join(f"{k} {v[:12]}" for k, v in sorted(hashes.items())) + "); everything after G1 "
                         "uses G̃ (line 142, D82)")
    return next(iter(hashes.values()))


def refuse_overwrite(files, what):
    """No record is written over (D82): refuses when any of `files` exists. A re-run needs the
    old file moved aside by hand, and the move declared (line 141)."""
    there = [p for p in files if p and os.path.exists(p)]
    if there:
        raise SystemExit(f"{what} refused: {', '.join(_rel(p) for p in there)} exists; a re-run needs the old file moved "
                         "aside by hand, and the move declared (line 141, D82)")


def stage_stopped(jsonl_path):
    """The task at which a stage's launch stopped on a quote still not finite after the
    re-pricing (its "stop" line), or None."""
    if not os.path.exists(jsonl_path):
        return None
    stop = None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("kind") == "stop":
                stop = obj.get("task")
    return stop


def _stage_report(p, stage):
    """The stage's report if it and its .npz are there and it is a complete assembly of that
    stage (assemble writes it only from a complete, finite record), else None."""
    if not (os.path.exists(p["report"]) and os.path.exists(p["npz"])):
        return None
    with open(p["report"], encoding="utf-8") as f:
        rep = json.load(f)
    need = ("stage", "nodes", "n_train", "n_heldout", "e_p99", "verdicts")
    if any(k not in rep for k in need) or rep["stage"] != stage or rep["n_train"] != int(np.prod(rep["nodes"][:3])):
        return None
    return rep


def require_g1(base=None, refined=None):
    """G̃, the emulator G1 passes to everything after it (lines 54 and 142, D82): the refined
    build when the base report calls for the one refinement, the base build otherwise.
    Refuses when the base build stopped on a quote still not finite after the re-pricing,
    when its report is missing or incomplete, and when the refinement called for stopped, is
    missing or is incomplete. `base` and `refined` are paths(stage) dicts (the defaults are
    output/'s). Returns the path of G̃'s .npz."""
    base, refined = base or paths("base"), refined or paths("refined")
    stop = stage_stopped(base["jsonl"])
    if stop:
        raise SystemExit(f"no emulator: G1's base build stopped at {stop} on a quote still not finite after the "
                         f"re-pricing at N_riccati {N_REPRICE}, which stops P5 (line 54)")
    rep = _stage_report(base, "base")
    if rep is None:
        raise SystemExit(f"no emulator: G1's base build is missing or incomplete (no complete report and emulator at "
                         f"{_rel(base['report'])}); G̃ is the build G1 passes (lines 54, 142)")
    if not rep.get("refine"):
        return base["npz"]
    stop = stage_stopped(refined["jsonl"])
    if stop:
        raise SystemExit(f"no emulator: G1's refined build stopped at {stop} on a quote still not finite after the "
                         f"re-pricing at N_riccati {N_REPRICE}, which stops P5 as at the base (lines 54, 142)")
    if _stage_report(refined, "refined") is None:
        raise SystemExit(f"no emulator: the base report calls for the one refinement and the refined build is missing "
                         f"or incomplete (no complete report and emulator at {_rel(refined['report'])}); after it the "
                         "refined build is G̃ (line 142)")
    return refined["npz"]


def emulator_for_run(given=None, base=None, refined=None):
    """The command line's emulator (D82): G̃ from require_g1(), and a refusal of an --emulator
    that names another file."""
    g = require_g1(base, refined)
    if given and os.path.normcase(os.path.abspath(given)) != os.path.normcase(os.path.abspath(g)):
        raise SystemExit(f"refused: --emulator {given} is not G̃, the build G1 passes ({_rel(g)}); everything after G1 "
                         "uses G̃ (line 142, D82)")
    return g


# --------------------------------------------------------------------------- #
# Chebyshev interpolation on [-1, 1]^4: first-kind nodes, barycentric evaluation
# --------------------------------------------------------------------------- #
def cheb_nodes(n):
    """Chebyshev nodes of the first kind, cos((2k+1)π/(2n)), in ascending order."""
    k = np.arange(n)
    return -np.cos((2 * k + 1) * np.pi / (2 * n))


def bary_weights(n):
    """Barycentric weights for the first-kind nodes in ascending order: (-1)^k sin((2k+1)π/(2n)),
    up to a common sign, which the formula cancels (Berrut and Trefethen's weights)."""
    k = np.arange(n)
    return (-1.0) ** k * np.sin((2 * k + 1) * np.pi / (2 * n))


def bary_matrix(x, n):
    """Rows of barycentric weights: p(x_m) = sum_k A[m, k] f(node_k). A point within
    64·eps of a node, absolute in unit coordinates (where a node mapped to the box and
    back lands, the map losing a few digits on a narrow box), takes the node's value."""
    nodes, w = cheb_nodes(n), bary_weights(n)
    x = np.asarray(x, dtype=float).reshape(-1)
    diff = x[:, None] - nodes[None, :]
    exact = np.abs(diff) <= 64.0 * np.finfo(float).eps
    with np.errstate(divide="ignore", invalid="ignore"):
        A = w[None, :] / diff
        A = A / A.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    A[hit] = exact[hit].astype(float)
    return A


def box_arrays(box=BOX):
    lo = np.array([box[d][0] for d in DIMS], dtype=float)
    hi = np.array([box[d][1] for d in DIMS], dtype=float)
    return lo, hi


def to_unit(thetas, box=BOX):
    """(H, nu, rho, xi0) in the box -> [-1, 1]^4, affinely."""
    lo, hi = box_arrays(box)
    t = np.asarray(thetas, dtype=float).reshape(-1, 4)
    return 2.0 * (t - lo) / (hi - lo) - 1.0


def from_unit(x, box=BOX):
    lo, hi = box_arrays(box)
    x = np.asarray(x, dtype=float).reshape(-1, 4)
    return lo + (x + 1.0) * (hi - lo) / 2.0


def node_grid(nodes_per_dim=NODES, box=BOX):
    """The node coordinates per dimension, in box units, ascending."""
    lo, hi = box_arrays(box)
    return [lo[i] + (cheb_nodes(n) + 1.0) * (hi[i] - lo[i]) / 2.0 for i, n in enumerate(nodes_per_dim)]


def evaluate_many(values, thetas, box=BOX, chunk=512, tol=1e-12):
    """G̃ at M points: values [n0, n1, n2, n3, Q] -> [M, Q]. Points outside the box (beyond
    tol in unit coordinates) raise: the interpolant is not for extrapolation."""
    values = np.asarray(values, dtype=float)
    x = to_unit(thetas, box)
    if np.any(np.abs(x) > 1.0 + tol):
        raise ValueError("evaluate_many: a point lies outside the box")
    x = np.clip(x, -1.0, 1.0)
    out = np.empty((x.shape[0], values.shape[-1]), dtype=float)
    n0, n1, n2, n3 = values.shape[:4]
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        W0, W1, W2, W3 = (bary_matrix(xs[:, i], n) for i, n in enumerate((n0, n1, n2, n3)))
        t = np.einsum("ma,abcdq->mbcdq", W0, values, optimize=False)
        t = np.einsum("mb,mbcdq->mcdq", W1, t, optimize=False)
        t = np.einsum("mc,mcdq->mdq", W2, t, optimize=False)
        out[s:s + chunk] = np.einsum("md,mdq->mq", W3, t, optimize=False)
    return out


def evaluate(values, theta, box=BOX):
    return evaluate_many(values, np.asarray(theta, dtype=float).reshape(1, 4), box)[0]


def cheb_coeffs(values, axis):
    """Chebyshev coefficients along one axis of node values on first-kind nodes:
    c_j = (2/n) sum_k f(x_k) cos(j θ_k), θ_k = arccos(x_k), c_0 halved, so that
    f = sum_j c_j T_j along that axis. The cosine transform; no further pricing."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    theta = np.arccos(cheb_nodes(n))
    C = (2.0 / n) * np.cos(np.outer(np.arange(n), theta))     # [j, k]
    C[0] /= 2.0
    return np.moveaxis(np.tensordot(C, np.moveaxis(values, axis, 0), axes=(1, 0)), 0, axis)


def cheb_coeffs_4d(values):
    """The coefficients of the tensor-product Chebyshev expansion: the transform applied
    along each of the four parameter axes in turn (the quote axis untouched)."""
    c = np.asarray(values, dtype=float)
    for axis in range(4):
        c = cheb_coeffs(c, axis)
    return c


def tails(values):
    """Per dimension, the largest absolute coefficient of the tensor-product expansion at
    that dimension's highest index, over all quotes and all coefficient indices in the
    other three dimensions (line 142, read as the full expansion; D72)."""
    c = cheb_coeffs_4d(values)
    return [float(np.max(np.abs(np.take(c, -1, axis=axis)))) for axis in range(4)]


def refine_dimension(tail_values):
    """The dimension to double: the largest tail; the lowest index among exact ties."""
    return int(np.argmax(np.asarray(tail_values, dtype=float)))


# --------------------------------------------------------------------------- #
# Pricing one (H, nu, rho) at the written settings: the pilot's path, unchanged
# --------------------------------------------------------------------------- #
def price_triple(H, nu, rho, xi0s, N=N_RICCATI, settings=None, Ts=TS, vus=VUS):
    """For each maturity one set of Riccati solves at (N, U_max, n_nodes), then for each
    xi0 the ATM quote, the strikes anchored on it and the seven quotes, through
    price_setting with allow_anchor=True (the base setting of G0's path). Returns a
    record with quotes[n_xi0][n_T][n_z] (None where not finite), statuses, timings."""
    settings = settings or BASE_T
    vus = np.asarray(vus, dtype=float)
    xi0s = [float(x) for x in xi0s]
    rec = {"H": float(H), "nu": float(nu), "rho": float(rho), "kappa": float(KAPPA), "xi0s": xi0s, "N": int(N),
           "Ts": [float(T) for T in Ts], "z": [float(z) for z in vus], "per_T": {}, "quotes": None, "status": None,
           "all_finite": None, "seconds_total": None}
    quotes = [[[None] * len(vus) for _ in Ts] for _ in xi0s]
    status = [[[None] * len(vus) for _ in Ts] for _ in xi0s]
    t0 = time.perf_counter()
    for j, T in enumerate(Ts):
        T = float(T)
        U, n = settings[T]
        sol, ints, u = solve_cf(H, nu, rho, T, N, U, n)
        anchors = {}
        q = price_setting(ints, u, T, U, n, xi0s, vus, anchors, N, allow_anchor=True, solve_status=sol["status"])
        per = {"solve": sol, "U_max": float(U), "n_nodes": int(n), "xi0": {}}
        for i, xi0 in enumerate(xi0s):
            key = _xi_key(xi0)
            e = q[key]
            a = anchors.get(key)
            per["xi0"][key] = {"sigma_atm": a["sigma_atm"] if a else None, "K": a["K"] if a else None,
                               "atm_status": e["atm_status"], "cf_status": e["cf_status"],
                               "top_log10_modulus": e["top_log10_modulus"]}
            quotes[i][j] = [_f(v) for v in e["quotes"]]
            status[i][j] = list(e["status"])
        rec["per_T"][str(T)] = per
    rec["quotes"], rec["status"] = quotes, status
    rec["all_finite"] = all(v is not None for xi in quotes for row in xi for v in row)
    rec["seconds_total"] = time.perf_counter() - t0
    return rec


def flat_quotes(rec, i_xi0):
    """The 35 quotes of one surface in the observable's order: by maturity, then by z."""
    return [v for row in rec["quotes"][i_xi0] for v in row]


# --------------------------------------------------------------------------- #
# Tasks: the node triples (train) and the held-out points
# --------------------------------------------------------------------------- #
def train_jobs(nodes_per_dim=NODES, box=BOX, stage="base"):
    g = node_grid(nodes_per_dim, box)
    jobs = []
    for a, H in enumerate(g[0]):
        for b, nu in enumerate(g[1]):
            for c, rho in enumerate(g[2]):
                label = f"{'N' if stage == 'base' else 'R'}{a:02d}_{b:02d}_{c:02d}"
                jobs.append({"label": label, "role": "train", "stage": stage, "index": [a, b, c],
                             "H": float(H), "nu": float(nu), "rho": float(rho), "xi0s": [float(x) for x in g[3]]})
    return jobs


def heldout_points(seed=SEED_HELDOUT, n=N_HELDOUT, box=BOX):
    """n points uniform on the box from default_rng(seed), columns (H, nu, rho, xi0)."""
    lo, hi = box_arrays(box)
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 4))


def heldout_jobs(seed=SEED_HELDOUT, n=N_HELDOUT, box=BOX):
    pts = heldout_points(seed, n, box)
    return [{"label": f"HO{i:03d}", "role": "heldout", "stage": "base", "index": i,
             "H": float(p[0]), "nu": float(p[1]), "rho": float(p[2]), "xi0s": [float(p[3])]} for i, p in enumerate(pts)]


def run_job(job, N=N_RICCATI, settings=None, Ts=TS, vus=VUS):
    rec = price_triple(job["H"], job["nu"], job["rho"], job["xi0s"], N=N, settings=settings, Ts=Ts, vus=vus)
    return dict(rec, kind="task", task=job["label"], role=job["role"], stage=job["stage"], index=job["index"],
                utc=utc_now())


def run_job_with_reprice(job, kw, reprice_n=N_REPRICE):
    """The job at its settings; if any quote is not finite, the job is priced once more at
    N_riccati reprice_n with the same inversion settings (one set of Riccati solves serves
    every ξ₀ of a triple), and each point, a surface (H, ν, ρ, ξ₀), with a quote not finite
    takes all its quotes from that pricing; the triple's other surfaces keep theirs (line 54,
    D81). Returns (record, first), `first` the record of the first pricing, kept for the
    log, or None when it was finite. The record keeps the first pricing's N and per-maturity
    detail, with `repriced_xi0` (the surfaces re-priced), `N_by_xi0` (per surface, the N its
    quotes come from), `N_reprice` and `per_T_reprice`, which holds the strikes and ATM vols of
    the re-priced surfaces; its seconds are both pricings' (D82)."""
    rec = run_job(job, **kw)
    if rec["all_finite"] or reprice_n is None:
        return rec, None
    rec2 = run_job(job, **dict(kw, N=int(reprice_n)))
    bad = [i for i, xi in enumerate(rec["quotes"]) if any(v is None for row in xi for v in row)]
    out = dict(rec, quotes=[rec2["quotes"][i] if i in bad else rec["quotes"][i] for i in range(len(rec["quotes"]))],
               status=[rec2["status"][i] if i in bad else rec["status"][i] for i in range(len(rec["status"]))],
               seconds_total=rec["seconds_total"] + rec2["seconds_total"], repriced=True, repriced_xi0=bad,
               N_by_xi0=[int(reprice_n) if i in bad else int(rec["N"]) for i in range(len(rec["quotes"]))],
               N_reprice=int(reprice_n), per_T_reprice=rec2["per_T"],
               reprice_note="per_T is the first pricing's; the surfaces in repriced_xi0 take their quotes, strikes and "
                            "ATM vols from per_T_reprice")
    out["all_finite"] = all(v is not None for xi in out["quotes"] for row in xi for v in row)
    return out, dict(rec, kind="first_attempt")


def _worker(args):
    job, kw, reprice_n = args
    return run_job_with_reprice(job, kw, reprice_n)


def header(pool, nodes_per_dim=NODES, box=BOX, stage="base", N=N_RICCATI, settings=None, versions=None):
    settings = settings or BASE_T
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    v = versions or version_check(CORE_VERSIONS, check=False)
    return {"kind": "header", "script": os.path.basename(__file__), "commit": commit, "clean_tree": clean,
            "stage": stage, "box": {d: list(box[d]) for d in DIMS}, "nodes": list(nodes_per_dim), "N": int(N),
            "n_reprice": N_REPRICE,
            "base_T": {str(float(T)): [float(v_[0]), int(v_[1])] for T, v_ in settings.items()},
            "seed_heldout": SEED_HELDOUT, "n_heldout": N_HELDOUT, "ratio": RATIO, "noise_levels": list(NOISE_LEVELS),
            "n_train": int(np.prod(nodes_per_dim[:3])), "n_surfaces": int(np.prod(nodes_per_dim)),
            "TS": list(TS), "z": [float(z) for z in VUS], "S0": float(S0), "r": float(R), "kappa": float(KAPPA),
            "d38_point": list(D38_POINT), "cost_per_triple_s": COST_PER_TRIPLE_S,
            "python": platform.python_version(), "numpy": v["numpy"], "scipy": v["scipy"], "versions": v,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "pool": int(pool), "utc_created": utc_now()}


def launch(pool, jobs, jsonl_path, header_obj, task_kwargs=None, stop_on_nonfinite=True, reprice_n=N_REPRICE):
    """The pool over `jobs`, skipping labels already in the jsonl. One line per finished
    task. A task whose quotes are not all finite is priced once more at N_riccati
    `reprice_n` with the same inversion settings (line 54, D81), the first pricing kept
    as a "first_attempt" line and the second recorded as the task; if a quote is still not
    finite, with stop_on_nonfinite, the launch stops (the build and P5 stop): the pending
    tasks are cancelled, those already running are recorded as they finish, and the stop
    is recorded; a later launch on that record refuses. A task that raises is recorded as
    task_error and is tried again on resume. Returns (n_done, stopped_at)."""
    task_kwargs = task_kwargs or {}
    header_obj = dict(header_obj, n_reprice=reprice_n)          # the re-pricing this launch applies, recorded
    hdr, launches, done = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, header_obj)
    else:
        for key in ("nodes", "box", "N", "base_T", "stage", "n_reprice"):
            if hdr.get(key) != header_obj.get(key):
                raise SystemExit(f"launch refused: {_rel(jsonl_path)} was started with {key}={hdr.get(key)!r}, "
                                 f"this call has {header_obj.get(key)!r}")
    if stop_on_nonfinite:
        bad = sorted(k for k, r in done.items() if r.get("all_finite") is False)
        if bad:
            raise SystemExit(f"launch refused: the record holds a quote that is not finite ({', '.join(bad)}); "
                             "the build stopped there (line 54)")
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    append_line(jsonl_path, {"kind": "launch", "utc": utc_now(), "pool": int(pool), "n_done_before": len(done),
                             "commit": commit, "clean_tree": clean, "versions": header_obj.get("versions")})
    todo = [j for j in jobs if j["label"] not in done]
    print(f"\n=== RUN: {len(todo)} tasks to do ({len(done)} already in {_rel(jsonl_path)}), pool={pool} ===", flush=True)
    t0 = time.time()
    state = {"n": 0, "n_err": 0, "stopped_at": None}

    def take(job, out, err):
        """Record one result (the first pricing too, when the point was re-priced); True when
        the launch must stop."""
        if err is not None:
            state["n_err"] += 1
            append_line(jsonl_path, {"kind": "task_error", "task": job["label"], "utc": utc_now(), "error": repr(err)})
            print(f"  task {job['label']} raised {err!r}; recorded, continuing", flush=True)
            return False
        rec, first = out
        if first is not None:
            append_line(jsonl_path, first)
            print(f"  {rec['task']}: a quote not finite at N_riccati {first['N']}; re-priced once at {rec['N_reprice']}, "
                  f"surfaces {rec['repriced_xi0']} (all_finite={rec['all_finite']})", flush=True)
        append_line(jsonl_path, rec)
        state["n"] += 1
        print(f"  [{state['n']:3d}/{len(todo)}] {rec['task']} ({rec['role']}): all_finite={rec['all_finite']} "
              f"task={rec['seconds_total']:.0f}s elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
        if stop_on_nonfinite and not rec["all_finite"]:
            state["stopped_at"] = rec["task"]
            reason = ("a quote that is not finite after the one re-pricing at N_riccati "
                      f"{reprice_n} stops the build, and P5 (line 54, D81)" if reprice_n is not None else
                      "a quote that is not finite stops the build (line 54)")
            append_line(jsonl_path, {"kind": "stop", "task": rec["task"], "utc": utc_now(), "reason": reason})
            print(f"  STOP: {rec['task']}: {reason}", flush=True)
            return True
        return False

    if pool == 1:                                       # in this process: no pool, same record
        for job in todo:
            try:
                out, err = _worker((job, task_kwargs, reprice_n)), None
            except Exception as e:
                out, err = None, e
            if take(job, out, err):
                break
    else:
        ex = ProcessPoolExecutor(max_workers=pool)
        try:
            futs = {ex.submit(_worker, (job, task_kwargs, reprice_n)): job for job in todo}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    out, err = fut.result(), None
                except Exception as e:
                    out, err = None, e
                if take(job, out, err):
                    ex.shutdown(wait=False, cancel_futures=True)
                    for f2, j2 in futs.items():          # tasks already running finish and are recorded
                        if f2 is fut or f2.cancelled():
                            continue
                        try:
                            o2, e2 = f2.result(), None
                        except Exception as e:
                            o2, e2 = None, e
                        take(j2, o2, e2)
                    break
        except BaseException:
            print("\n  stopping: pending tasks cancelled; re-run to resume", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    n, n_err, stopped_at = state["n"], state["n_err"], state["stopped_at"]
    print(f"=== done: {n} tasks ({n_err} errors) in {(time.time() - t0) / 60:.1f} min"
          + (f"; stopped at {stopped_at}" if stopped_at else "") + " ===", flush=True)
    return n, stopped_at


# --------------------------------------------------------------------------- #
# Assembly: the node values, the held-out error, G1's verdicts, the refinement rule
# --------------------------------------------------------------------------- #
def read_stage(jsonl_path):
    hdr, launches, tasks = read_jsonl(jsonl_path)
    train = {k: r for k, r in tasks.items() if r.get("role") == "train"}
    held = {k: r for k, r in tasks.items() if r.get("role") == "heldout"}
    return hdr, launches, train, held


def check_complete(expected_labels, records, what):
    """Every expected label present and all its quotes finite; else a list of reasons."""
    problems = []
    for lab in expected_labels:
        r = records.get(lab)
        if r is None:
            problems.append(f"{what} {lab}: missing")
        elif not r["all_finite"]:
            bad = sorted({s for xi in r["status"] for row in xi for s in row if s != STATUS_OK})
            problems.append(f"{what} {lab}: a quote is not finite ({', '.join(bad)})")
    return problems


def values_from(train, nodes_per_dim, n_quotes=N_QUOTES, stage="base"):
    V = np.full(tuple(nodes_per_dim) + (n_quotes,), np.nan)
    for a in range(nodes_per_dim[0]):
        for b in range(nodes_per_dim[1]):
            for c in range(nodes_per_dim[2]):
                r = train[f"{'N' if stage == 'base' else 'R'}{a:02d}_{b:02d}_{c:02d}"]
                for d in range(nodes_per_dim[3]):
                    V[a, b, c, d] = flat_quotes(r, d)
    return V


def heldout_table(held, n):
    thetas, G = [], []
    for i in range(n):
        r = held[f"HO{i:03d}"]
        thetas.append([r["H"], r["nu"], r["rho"], r["xi0s"][0]])
        G.append(flat_quotes(r, 0))
    return np.array(thetas, dtype=float), np.array(G, dtype=float)


def g1_verdicts(e, ratio=RATIO, levels=NOISE_LEVELS):
    return {f"{s:g}": {"bound": s / ratio, "claimed": bool(e <= s / ratio)} for s in levels}


def assess(values, thetas, G, box=BOX, Ts=TS, vus=VUS, ratio=RATIO, levels=NOISE_LEVELS, stage="base"):
    """e, the maximum, the per-maturity 99th percentiles, G1's verdicts, the tails and,
    at the base stage, the refinement decision (e above the smallest level's bound) with
    its dimension; at the refined stage there is no second refinement (line 142)."""
    Gt = evaluate_many(values, thetas, box)
    diff = np.abs(Gt - G)
    nz = len(vus)
    per_T = {str(float(T)): float(np.percentile(diff[:, j * nz:(j + 1) * nz], 99)) for j, T in enumerate(Ts)}
    e = float(np.percentile(diff, 99))
    tl = tails(values)
    refine = bool(e > min(levels) / ratio) if stage == "base" else None
    return {"e_p99": e, "max_abs": float(np.max(diff)), "per_T_p99": per_T, "n_quotes": int(diff.size),
            "percentile_method": "numpy.percentile default (linear interpolation)",
            "verdicts": g1_verdicts(e, ratio, levels), "tails": {d: tl[i] for i, d in enumerate(DIMS)},
            "refine": refine, "refine_dimension": DIMS[refine_dimension(tl)] if refine else None}, Gt, diff


def assemble(stage="base", nodes_per_dim=None, base_paths=None, stage_paths=None, n_heldout=N_HELDOUT,
             Ts=TS, vus=VUS, versions=None):
    """Read the stage's record, refuse unless complete and finite, write the .npz, the
    held-out CSV and the report (with the numpy and scipy of this call, D81). The held-out
    points are read from the base record for either stage (drawn once). Returns the report."""
    bp, sp = base_paths or paths("base"), stage_paths or paths(stage)
    hdr, launches, train, _ = read_stage(sp["jsonl"])
    _, _, _, held = read_stage(bp["jsonl"])
    if hdr is None:
        raise SystemExit(f"nothing to assemble: {_rel(sp['jsonl'])} has no record")
    nodes_per_dim = tuple(nodes_per_dim or hdr["nodes"])
    labels = [j["label"] for j in train_jobs(nodes_per_dim, {d: tuple(hdr["box"][d]) for d in DIMS}, stage)]
    problems = check_complete(labels, train, "train") + check_complete([f"HO{i:03d}" for i in range(n_heldout)], held,
                                                                       "heldout")
    if problems:
        raise SystemExit("assemble refused:\n  " + "\n  ".join(problems[:20])
                         + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""))
    refuse_overwrite([sp.get("npz"), sp.get("csv"), sp.get("report")], "assemble")           # no record written over (D82)
    box = {d: tuple(hdr["box"][d]) for d in DIMS}
    nq = len(Ts) * len(vus)
    V = values_from(train, nodes_per_dim, nq, stage)
    assert np.all(np.isfinite(V))
    thetas, G = heldout_table(held, n_heldout)
    rep, Gt, diff = assess(V, thetas, G, box, Ts, vus, stage=stage)
    grid = node_grid(nodes_per_dim, box)
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(sp["npz"]), exist_ok=True)
    np.savez(sp["npz"], values=V, nodes_H=grid[0], nodes_nu=grid[1], nodes_rho=grid[2], nodes_xi0=grid[3],
             box_lo=box_arrays(box)[0], box_hi=box_arrays(box)[1], TS=np.array([float(T) for T in Ts]),
             z=np.asarray(vus, dtype=float), N=np.array(hdr["N"]),
             base_T=np.array([[float(T)] + list(hdr["base_T"][str(float(T))]) for T in Ts]),
             stage=np.array(stage), commit=np.array(str(hdr.get("commit"))), quote_order=np.array("by T then z, ascending"))
    with open(sp["csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "H", "nu", "rho", "xi0", "T", "z", "G", "G_tilde", "diff"])
        for i in range(thetas.shape[0]):
            for j, T in enumerate(Ts):
                for k, z in enumerate(vus):
                    q = j * len(vus) + k
                    w.writerow([f"HO{i:03d}", *[repr(float(x)) for x in thetas[i]], float(T), float(z),
                                repr(float(G[i, q])), repr(float(Gt[i, q])), repr(float(Gt[i, q] - G[i, q]))])
    report = {"stage": stage, "nodes": list(nodes_per_dim), "n_train": len(labels), "n_surfaces": int(np.prod(V.shape[:4])),
              "n_heldout": int(thetas.shape[0]), "box": {d: list(box[d]) for d in DIMS}, "N": hdr["N"],
              "base_T": hdr["base_T"], "seed_heldout": SEED_HELDOUT, "ratio": RATIO, "noise_levels": list(NOISE_LEVELS),
              **rep, "seconds_train_total": float(sum(r["seconds_total"] for r in train.values())),
              "seconds_heldout_total": float(sum(r["seconds_total"] for r in held.values())),
              "n_reprice": hdr.get("n_reprice"),
              "repriced_points": sorted(k for k, r in list(train.items()) + list(held.items()) if r.get("repriced")),
              "versions": versions or version_check(CORE_VERSIONS, check=False),
              "record_commit": hdr.get("commit"), "assembled_commit": commit, "assembled_clean_tree": clean,
              "launches": launches, "utc_assembled": utc_now(),
              "files": {k: _rel(v) for k, v in sp.items()}}
    if stage == "base" and rep["refine"]:
        new = list(nodes_per_dim)
        new[DIMS.index(rep["refine_dimension"])] *= 2
        report["refined_nodes"] = new
    with open(sp["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    return report


class Emulator:
    """G̃ loaded from the .npz: evaluate(theta) -> 35 quotes; evaluate_many(thetas) -> [M, 35]."""

    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as z:   # copied out; the file is closed
            self.values = np.array(z["values"])
            self.box = {d: (float(z["box_lo"][i]), float(z["box_hi"][i])) for i, d in enumerate(DIMS)}
            self.TS, self.z = np.array(z["TS"]), np.array(z["z"])
            self.N, self.stage = int(z["N"]), str(z["stage"])
        self.nodes = tuple(int(n) for n in self.values.shape[:4])

    def evaluate_many(self, thetas):
        return evaluate_many(self.values, thetas, self.box)

    def evaluate(self, theta):
        return self.evaluate_many(np.asarray(theta, dtype=float).reshape(1, 4))[0]


def load_emulator(npz_path=None):
    return Emulator(npz_path or paths("base")["npz"])


# --------------------------------------------------------------------------- #
def estimate(nodes_per_dim=NODES, n_heldout=N_HELDOUT, pool=POOL_DEFAULT, per_triple_s=COST_PER_TRIPLE_S):
    n_sets = int(np.prod(nodes_per_dim[:3])) + n_heldout
    core_h = n_sets * per_triple_s / 3600.0
    return {"n_sets": n_sets, "core_hours": core_h, "wall_hours": core_h / pool, "per_triple_s": per_triple_s}


def main(argv=None, check_versions=False):
    ap = argparse.ArgumentParser(description="P5 gate G1 emulator (see module docstring)")
    ap.add_argument("--run", action="store_true", help="price the node triples and the held-out points (resumable)")
    ap.add_argument("--assemble", action="store_true", help="write the .npz, the held-out CSV and the report")
    ap.add_argument("--refine", action="store_true", help="price the refined node triples, if the base report calls for it")
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate from D62 and exit; no pricing")
    ap.add_argument("--stage", default="base", choices=("base", "refined"))
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--allow-other-versions", action="store_true",
                    help="run with numpy or scipy other than CORE_VERSIONS; recorded (D81)")
    a = ap.parse_args(argv)
    vers = version_check(CORE_VERSIONS, a.allow_other_versions, os.path.basename(__file__), check=check_versions)

    if a.estimate:
        e = estimate(pool=a.pool)
        print(f"estimate from D62's {e['per_triple_s']:.0f} core-s per triple: {e['n_sets']} sets of solves, "
              f"{e['core_hours']:.1f} core-hours, {e['wall_hours']:.1f} h on {a.pool} workers (unmeasured here)", flush=True)
        return
    if a.assemble:
        rep = assemble(a.stage, versions=vers)
        print(f"=== G1 ({a.stage}): e = {rep['e_p99']:.6f} (max {rep['max_abs']:.6f}) over {rep['n_quotes']} held-out quotes ===",
              flush=True)
        for s, v in rep["verdicts"].items():
            print(f"  s = {s}: bound {v['bound']:.6f} -> {'claimed' if v['claimed'] else 'not claimed'}", flush=True)
        print(f"  tails: " + ", ".join(f"{d}={t:.2e}" for d, t in rep["tails"].items()), flush=True)
        if a.stage == "base":
            print(f"  refine: {rep['refine']}" + (f" in {rep['refine_dimension']} -> nodes {rep['refined_nodes']}; run --refine"
                                                  if rep["refine"] else " (no refinement)"), flush=True)
        print(f"  written: {rep['files']['npz']}, {rep['files']['csv']}, {rep['files']['report']}", flush=True)
        return
    if a.refine:
        with open(paths("base")["report"], encoding="utf-8") as f:
            rep = json.load(f)
        if not rep.get("refine"):
            raise SystemExit("the base report does not call for a refinement; nothing to do")
        nodes = tuple(rep["refined_nodes"])
        jobs = train_jobs(nodes, stage="refined")
        print(f"=== refinement: {rep['refine_dimension']} doubled -> nodes {nodes}; {len(jobs)} triples ===", flush=True)
        e = estimate(nodes, 0, a.pool)
        print(f"  estimate: {e['core_hours']:.1f} core-hours, {e['wall_hours']:.1f} h on {a.pool} workers", flush=True)
        n, stopped = launch(a.pool, jobs, paths("refined")["jsonl"], header(a.pool, nodes, stage="refined", versions=vers))
        if not stopped:
            print("  now run --assemble --stage refined", flush=True)
        return

    print("=== P5 G1 emulator ===", flush=True)
    print(f"  box: " + ", ".join(f"{d} {BOX[d]}" for d in DIMS), flush=True)
    print(f"  nodes: {NODES} = {int(np.prod(NODES[:3]))} triples, {int(np.prod(NODES))} surfaces; N_riccati={N_RICCATI}", flush=True)
    for T in TS:
        print(f"  T={T}: U_max={BASE_T[T][0]} n_nodes={BASE_T[T][1]}", flush=True)
    print(f"  held-out: {N_HELDOUT} points, default_rng({SEED_HELDOUT}); G1: e <= s/{RATIO} at s in {NOISE_LEVELS}", flush=True)
    t0 = time.perf_counter()
    d, n = P.identity_gate()
    print(f"\n=== identity gate at the scoping base (N_riccati 1000, U_max 200, 128 nodes): "
          f"max |path - model_smile_cf_T| over {n} quotes = {d!r} ({time.perf_counter() - t0:.0f}s) ===", flush=True)
    if d != 0.0:
        raise SystemExit("identity gate failed")
    e = estimate(pool=a.pool)
    print(f"\n=== estimate from D62's {e['per_triple_s']:.0f} core-s per triple: {e['n_sets']} sets, "
          f"{e['core_hours']:.1f} core-hours, {e['wall_hours']:.1f} h on {a.pool} workers ===", flush=True)
    if not a.run:
        print("\nHOLD: nothing priced. Run with --run to build (G1 is run after freezing, line 7).", flush=True)
        return
    jobs = train_jobs() + heldout_jobs()
    n, stopped = launch(a.pool, jobs, paths("base")["jsonl"], header(a.pool, versions=vers))
    if not stopped:
        print("  now run --assemble", flush=True)


if __name__ == "__main__":
    main(check_versions=True)
