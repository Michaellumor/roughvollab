"""
p5_g0_pilot.py — P5 gate G0, the scoping run (docs/protocols/P5_protocol_v2_draft.md,
Gates, G0; staged in D59).

The scoping run has no pass or fail. It measures, on the author's machine, the cost of
the rough-Heston pricer over the box and the grid proposed under Objects, and at each
point and maturity it prices, whether and from which `N_riccati` among those it tries
the quotes are finite, and which of G0's three comparisons are met. Its points and the
values of `N_riccati`, node count and `U_max` it tries are those of this file as
committed before it is run (L139). No flag changes a point or a setting.

What it computes is G0's list and nothing else: the seconds of each group of three
Riccati solves with their fractional integrals; whether each quote is finite and, if
not, at which stage it failed; the three comparisons (`N_riccati` doubled, node count
doubled, `U_max` raised) at the tolerance 0.0001; the inversion applied to the
Black–Scholes and classical Heston characteristic functions, for which a reference
price is available (the exact formula; adaptive quadrature, whose own accuracy is
checked on the Black–Scholes case); and the modulus of the characteristic function at
the highest node, recorded and deciding nothing (with its log10, which stays finite
where the exponential overflows). Nothing that compares quotes across parameter
values: no Jacobian, Fisher information, posterior, calibration or emulator. There
are no random draws, so there is no seed.

Pricing path. The repository's wrapper (`layer4_calibrate_surface.model_smile_cf_T`)
solves the fractional Riccati three times per strike. This script solves it three times
per (H, ν, ρ, T, `N_riccati`, u-grid), as `rough_heston_cf` does for one call of
`gil_pelaez_call`, and reuses the solve across strikes and across ξ₀: ξ₀ enters only
the assembly `exp(κ·θ·I¹ψ + V0·I^{1−α}ψ)` with V0 = θ = ξ₀ (`rough_heston_cf.py:219-226`),
not `_frac_riccati`. The assembly here is that expression, term for term, and
`identity_gate` checks before anything else that this path reproduces the wrapper on
D38's 35 quotes with a maximum difference of exactly 0.0 (as `calibrate_btc.py`'s cache
was verified). Overflow is recorded as a status, never raised.

Usage (from the repository root):
    py -3 p5_g0_pilot.py            identity gate, known answers, timing, design, estimate; holds
    py -3 p5_g0_pilot.py --run      the same, then the worker pool over the 300 tasks; resumable
    py -3 p5_g0_pilot.py --report   regenerate output/p5_g0_pilot.json and _quotes.csv only

Outputs (output/ is git-ignored; the files are force-added by name after the run):
    output/p5_g0_pilot_tasks.jsonl  the working record: a header line, a line per launch,
                                    a line per completed task (appended as each finishes)
    output/p5_g0_pilot.json         header, launches, known answers, per-task summaries, cost
    output/p5_g0_pilot_quotes.csv   one row per triple × ξ₀ × T × z × rung
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 BLAS thread per process, set before numpy (calibrate_btc.py)

import sys
import csv
import json
import time
import argparse
import platform
import subprocess
import warnings
import datetime
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.polynomial.legendre import leggauss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import layer4_calibrate_surface as S
from layer4_calibrate import KAPPA_FIXED
from rough_heston_cf import (_frac_riccati, _frac_integral_at_T, gil_pelaez_call, bs_iv,
                             bs_cf, bs_call, heston_cf, _gil_pelaez_call_adaptive)

# --------------------------------------------------------------------------- #
# The grid and the fixed parameter: from the repository, not retyped
# --------------------------------------------------------------------------- #
S0, R = S.S0, S.R                                   # 100, 0
TS = tuple(float(T) for T in S.TS)                  # (0.10, 0.25, 0.50, 1.00, 2.00)
VUS = np.asarray(S.VUS, dtype=float)                # z in {-2, -1, -0.5, 0, 0.5, 1, 2}
KAPPA = KAPPA_FIXED                                 # 0.30

# --------------------------------------------------------------------------- #
# The settings the scoping run tries (L139: "those of the script as committed")
# --------------------------------------------------------------------------- #
TOL = 1e-4                                          # 0.01 vol point, absolute, per quote, per comparison
LADDER = (1000, 2000, 4000, 8000, 16000)            # N_riccati, climbed in order per task
SETTINGS = {                                        # name -> (U_max, n_nodes)
    "A": (200.0, 128),                              # base: the repository's own (NR_SAFE/NN, gil_pelaez_call default)
    "C": (200.0, 256),                              # node count doubled
    "D": (400.0, 256),                              # U_max doubled at the doubled node count
}
COMPARISONS = {                                     # name -> (quotes compared, against)
    "N2": ("A at 2N", "A at N"),                    # N_riccati doubled: the next rung's base
    "nodes": ("C", "A"),                            # node count doubled, same U_max
    "umax": ("D", "C"),                             # U_max raised, same node count
}
H_VALUES = (0.02, 0.05, 0.10, 0.25, 0.48)
NU_VALUES = (0.05, 0.35, 0.65, 1.00)
RHO_VALUES = (0.00, -0.70, -0.99)
XI0_VALUES = (0.001, 0.01, 0.04, 0.10, 0.25)        # share each solve; the box's bounds are the ends
D38_POINT = (0.10, 0.35, -0.70, 0.04)               # (H, nu, rho, xi0); D38's truth
KNOWN_HESTON = ((0.35, -0.70), (1.00, -0.99))       # (nu, rho) for the classical-Heston known answer
SEED = None                                         # no draws are made; the points are the lattice below

STATUS_OK = "ok"
STATUS_ORDER = ("ok", "riccati_nonfinite", "integral_nonfinite", "cf_nonfinite", "price_nonfinite",
                "price_too_small", "iv_failed", "atm_unavailable", "skipped_base_nonfinite", "not_computed")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
JSONL_PATH = os.path.join(OUT_DIR, "p5_g0_pilot_tasks.jsonl")
JSON_PATH = os.path.join(OUT_DIR, "p5_g0_pilot.json")
CSV_PATH = os.path.join(OUT_DIR, "p5_g0_pilot_quotes.csv")


def build_triples():
    """The 60 (H, nu, rho) triples: the 8 corner triples first, then D38's, then the
    rest of the lattice with H ascending. With XI0_VALUES this contains the 16 corners
    of the proposed box and D38's point."""
    corners = [(H, nu, rho) for H in (H_VALUES[0], H_VALUES[-1])
               for nu in (NU_VALUES[0], NU_VALUES[-1])
               for rho in (RHO_VALUES[0], RHO_VALUES[-1])]
    out = corners + [D38_POINT[:3]]
    for t in [(H, nu, rho) for H in H_VALUES for nu in NU_VALUES for rho in RHO_VALUES]:
        if t not in out:
            out.append(t)
    return tuple(out)


TRIPLES = build_triples()
TASKS = tuple((t, T) for t in TRIPLES for T in TS)


def task_id(triple, T):
    H, nu, rho = triple
    return f"H{H:.2f}_nu{nu:.2f}_rho{rho:+.2f}_T{T:.2f}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _f(x):
    """float, or None where not finite (JSON never gets a bare NaN)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _xi_key(xi0):
    return f"{xi0:g}"


def u_nodes(U_max, n_nodes):
    """The Gauss–Legendre nodes exactly as gil_pelaez_call builds them."""
    x, _ = leggauss(n_nodes)
    return 0.5 * U_max * (x + 1.0)


# --------------------------------------------------------------------------- #
# One solve: the three fractional-Riccati solves behind one gil_pelaez_call
# --------------------------------------------------------------------------- #
def solve_cf(H, nu, rho, T, N, U_max, n_nodes):
    """Solve the fractional Riccati for cf(u), cf(u - i) and cf(-i) on the inverter's
    u-grid, as rough_heston_cf does per call, and return the two fractional integrals
    each needs. Returns (record, integrals, u). record: N, U_max, n_nodes, seconds,
    status (ok | riccati_nonfinite | integral_nonfinite) and the stage that failed."""
    u = u_nodes(U_max, n_nodes)
    alpha = H + 0.5
    h = T / N
    rec = {"N": int(N), "U_max": float(U_max), "n_nodes": int(n_nodes),
           "seconds": None, "status": STATUS_OK, "failed_at": None}
    ints = {}
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        for key, uu in (("u", u), ("u_shift", u - 1j), ("mi", np.array([-1j]))):
            psi = _frac_riccati(uu, T, H, KAPPA, nu, rho, N)
            if not np.all(np.isfinite(psi)):
                rec["status"], rec["failed_at"] = "riccati_nonfinite", key
                break
            int1 = _frac_integral_at_T(psi, 1.0, h)             # kappa*theta term (rough_heston_cf.py:223)
            iqa = _frac_integral_at_T(psi, 1.0 - alpha, h)      # V0 term            (rough_heston_cf.py:224)
            if not (np.all(np.isfinite(int1)) and np.all(np.isfinite(iqa))):
                rec["status"], rec["failed_at"] = "integral_nonfinite", key
                break
            ints[key] = (int1, iqa)
    rec["seconds"] = time.perf_counter() - t0
    return rec, ints, u


def make_cf(ints, u, xi0):
    """Assemble the CF for one xi0 from the solved integrals (rough_heston_cf.py:225) and
    return a cf(u) that answers only on the solved u-grid, raising on anything else.
    Returns (cf, status, modulus): status ok | cf_nonfinite; modulus holds the CF's
    modulus at the highest node and its log10, which stays finite where exp overflows."""
    mod = {"top_modulus": None, "top_log10_modulus": None}
    if len(ints) < 3:
        return None, "cf_nonfinite", mod
    table = {}
    finite = True
    with np.errstate(all="ignore"):
        for key, (int1, iqa) in ints.items():
            expo = KAPPA * xi0 * int1 + xi0 * iqa               # kappa*theta*int1 + V0*iqa, theta = V0 = xi0
            phi = np.exp(expo)
            finite = finite and bool(np.all(np.isfinite(phi)))
            table[key] = phi
            if key == "u":
                mod["top_modulus"] = _f(np.abs(phi[-1]))
                mod["top_log10_modulus"] = _f(np.real(expo[-1]) / np.log(10.0))
    if not finite:
        return None, "cf_nonfinite", mod
    lookup = {u.tobytes(): table["u"], (u - 1j).tobytes(): table["u_shift"]}

    def cf(uu):
        if np.ndim(uu) == 0:
            if complex(uu) != -1j:
                raise KeyError("cf asked off the solved grid: scalar %r" % (uu,))
            return table["mi"]
        v = lookup.get(np.asarray(uu).tobytes())
        if v is None:
            raise KeyError("cf asked off the solved u-grid")
        return v
    return cf, STATUS_OK, mod


def quote(cf, K, T, U_max, n_nodes):
    """One implied-vol quote through gil_pelaez_call and bs_iv, with the wrapper's guards
    (model_smile_cf_T: finite price above 1e-12, else NaN). Returns (iv, status)."""
    with np.errstate(all="ignore"):
        px = gil_pelaez_call(cf, S0, K, T, R, U_max=U_max, n_nodes=n_nodes)
        if not np.isfinite(px):
            return None, "price_nonfinite"
        if px <= 1e-12:
            return None, "price_too_small"
        iv = bs_iv(px, S0, K, T, R)
    if not np.isfinite(iv):
        return None, "iv_failed"
    return float(iv), STATUS_OK


def strikes_for(sigma_atm, T, vus):
    """K = S0 * exp(z * sigma_atm * sqrt(T)), through the repository's own function."""
    return S.fixed_surface_strikes({T: sigma_atm}, np.asarray(vus, dtype=float), [T])[T]


def price_setting(ints, u, T, U_max, n_nodes, xi0s, vus, anchors, N, allow_anchor, solve_status=STATUS_OK):
    """Quotes for every xi0 at one setting. anchors: xi0 -> dict(sigma_atm, anchor_N, K);
    filled here from the ATM quote when allow_anchor (the base setting) and missing.
    A solve that failed gives every quote the stage at which it failed."""
    out = {}
    for xi0 in xi0s:
        key = _xi_key(xi0)
        if solve_status != STATUS_OK:
            cf, cf_status, mod = None, solve_status, {"top_modulus": None, "top_log10_modulus": None}
        else:
            cf, cf_status, mod = make_cf(ints, u, xi0)
        entry = {"cf_status": cf_status, "top_modulus": mod["top_modulus"],
                 "top_log10_modulus": mod["top_log10_modulus"], "atm_status": cf_status,
                 "quotes": [None] * len(vus), "status": [cf_status] * len(vus)}
        if cf is not None:
            if key in anchors:
                entry["atm_status"] = "anchored"                 # strikes fixed earlier in this task
            elif allow_anchor:
                sigma, st = quote(cf, S0, T, U_max, n_nodes)      # the ATM quote; its stage of failure is kept
                entry["atm_status"] = st
                if st == STATUS_OK:
                    anchors[key] = {"xi0": float(xi0), "sigma_atm": sigma, "anchor_N": int(N),
                                    "K": [float(k) for k in strikes_for(sigma, T, vus)]}
            else:
                entry["atm_status"] = "not_anchoring"             # comparison settings never anchor
            if key in anchors:
                for i, K in enumerate(anchors[key]["K"]):
                    entry["quotes"][i], entry["status"][i] = quote(cf, K, T, U_max, n_nodes)
            else:
                entry["status"] = ["atm_unavailable"] * len(vus)
        out[key] = entry
    return out


def skipped_setting(xi0s, vus, status):
    return {_xi_key(xi0): {"cf_status": status, "top_modulus": None, "top_log10_modulus": None,
                           "atm_status": status, "quotes": [None] * len(vus), "status": [status] * len(vus)}
            for xi0 in xi0s}


def compare(q_new, q_ref, tol):
    """Per-quote differences q_new - q_ref in vol units, and whether every quote is
    finite in both and within tol. Returns dict(met, n_met, n_total, max_abs_diff, diffs)."""
    diffs, n_met, n_total, mx = {}, 0, 0, None
    for key in q_ref:
        d = []
        for a, b in zip(q_new[key]["quotes"], q_ref[key]["quotes"]):
            n_total += 1
            if a is None or b is None:
                d.append(None)
                continue
            dd = a - b
            d.append(dd)
            mx = abs(dd) if mx is None else max(mx, abs(dd))
            if abs(dd) < tol:
                n_met += 1
        diffs[key] = d
    return {"met": bool(n_total > 0 and n_met == n_total), "n_met": n_met, "n_total": n_total,
            "max_abs_diff": mx, "diffs": diffs}


def not_computed(q_ref):
    return {"met": None, "n_met": 0, "n_total": sum(len(v["quotes"]) for v in q_ref.values()),
            "max_abs_diff": None, "diffs": {k: [None] * len(v["quotes"]) for k, v in q_ref.items()},
            "status": "not_computed"}


# --------------------------------------------------------------------------- #
# One task: one (H, nu, rho) at one maturity, up the ladder
# --------------------------------------------------------------------------- #
def run_task(triple, T, *, ladder=LADDER, settings=SETTINGS, tol=TOL, xi0s=XI0_VALUES, vus=VUS):
    """Climb the ladder. At each rung: the base solve A; then, if the previous rung now
    meets all three comparisons (its N2 partner being this base), stop with this rung
    recorded as partner only; else, if the base solve is finite, the C and D solves and
    the nodes and umax comparisons. The top rung has no partner: N2 is not computed.
    Returns one JSON-serialisable record with no NaN."""
    H, nu, rho = (float(x) for x in triple)
    T = float(T)
    vus = np.asarray(vus, dtype=float)
    UA, nA = settings["A"]
    rec = {"kind": "task", "task": task_id(triple, T), "H": H, "nu": nu, "rho": rho, "kappa": float(KAPPA),
           "T": T, "xi0s": [float(x) for x in xi0s], "z": [float(z) for z in vus], "tol": float(tol),
           "ladder": [int(n) for n in ladder], "settings": {k: [float(v[0]), int(v[1])] for k, v in settings.items()},
           "anchors": {}, "rungs": [], "stopped_at": None, "exhausted": False, "seconds_total": None}
    anchors = rec["anchors"]
    t_start = time.perf_counter()
    prev = None
    for N in ladder:
        rung = {"N": int(N), "role": "full", "solves": {}, "quotes": {}, "comparisons": {}}
        solA, intsA, uA = solve_cf(H, nu, rho, T, N, UA, nA)
        rung["solves"]["A"] = solA
        rung["quotes"]["A"] = price_setting(intsA, uA, T, UA, nA, xi0s, vus, anchors, N, allow_anchor=True,
                                            solve_status=solA["status"])
        if prev is not None:
            prev["comparisons"]["N2"] = compare(rung["quotes"]["A"], prev["quotes"]["A"], tol)
            if all(prev["comparisons"][c]["met"] for c in ("N2", "nodes", "umax")):
                rec["stopped_at"] = prev["N"]
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
        prev = rung
    else:
        rec["exhausted"] = True
        if prev is not None:
            prev["comparisons"]["N2"] = not_computed(prev["quotes"]["A"])
    rec["seconds_total"] = time.perf_counter() - t_start
    return rec


def first_finite_N(rec, setting="A"):
    """Per xi0: the first rung at which all quotes at `setting` are ok, else None."""
    out = {}
    for key in (_xi_key(x) for x in rec["xi0s"]):
        out[key] = None
        for rung in rec["rungs"]:
            q = rung["quotes"].get(setting)
            if q and all(s == STATUS_OK for s in q[key]["status"]):
                out[key] = rung["N"]
                break
    return out


# --------------------------------------------------------------------------- #
# Identity gate: this pricing path == the repository's wrapper, bit for bit
# --------------------------------------------------------------------------- #
def identity_gate(theta=D38_POINT, N=S.NR_SAFE, U_max=200.0, n_nodes=S.NN, Ts=TS, vus=VUS):
    """Max absolute difference between this script's quotes and model_smile_cf_T on the
    wrapper's own standardised grid (strikes anchored by atm_by_T), NaN-aware: NaN on
    both sides counts 0, on one side inf. Must be exactly 0.0."""
    H, nu, rho, xi0 = theta
    vus = np.asarray(vus, dtype=float)
    atm = S.atm_by_T(theta, list(Ts), N_riccati=N, n_nodes=n_nodes)
    grids = S.fixed_surface_strikes(atm, vus, list(Ts))
    max_diff, n = 0.0, 0
    for T in Ts:
        ref = S.model_smile_cf_T(theta, grids[T], T, N_riccati=N, n_nodes=n_nodes)
        _, ints, u = solve_cf(H, nu, rho, T, N, U_max, n_nodes)
        cf, _, _ = make_cf(ints, u, xi0)
        mine = np.full(len(vus), np.nan)
        if cf is not None:
            sigma, st_atm = quote(cf, S0, T, U_max, n_nodes)
            if st_atm == STATUS_OK:
                Ks = strikes_for(sigma, T, vus)
                if not np.array_equal(Ks, grids[T]):
                    max_diff = np.inf
                for i, K in enumerate(Ks):
                    v, _ = quote(cf, K, T, U_max, n_nodes)
                    mine[i] = np.nan if v is None else v
        for a, b in zip(mine, ref):
            n += 1
            if np.isnan(a) and np.isnan(b):
                continue
            d = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.inf
            max_diff = max(max_diff, d)
    return float(max_diff), n


# --------------------------------------------------------------------------- #
# Known answers: the inversion alone, on CFs with a reference price (no Riccati)
# --------------------------------------------------------------------------- #
def known_answers(settings=SETTINGS, Ts=TS, xi0s=XI0_VALUES, vus=VUS, heston_pairs=KNOWN_HESTON):
    """Black–Scholes (reference: bs_call, the exact formula) and classical Heston
    (reference: _gil_pelaez_call_adaptive, adaptive quadrature, whose accuracy at the
    box's ends is not established) through gil_pelaez_call at each setting. The adaptive
    quadrature is also applied to the Black–Scholes CF at each (T, xi0), against bs_call,
    as the check of the reference method itself (model "bs", setting "adaptive").
    Strikes standardised on the reference's own ATM vol. One row per model, parameter
    set, T, z and setting: prices, implied vols and their differences."""
    rows = []
    vus = np.asarray(vus, dtype=float)
    for T in Ts:
        for xi0 in xi0s:
            sigma = float(np.sqrt(xi0))
            Ks = strikes_for(sigma, T, vus)
            for z, K in zip(vus, Ks):
                ref = bs_call(S0, K, T, R, sigma)
                with warnings.catch_warnings(record=True) as w_bs, np.errstate(all="ignore"):
                    warnings.simplefilter("always")
                    px_ad = _gil_pelaez_call_adaptive(lambda uu: bs_cf(uu, T, sigma, R), S0, K, T, R)
                    iv_ad = bs_iv(px_ad, S0, K, T, R) if (np.isfinite(px_ad) and px_ad > 1e-12) else np.nan
                rows.append({"model": "bs", "T": T, "xi0": float(xi0), "nu": None, "rho": None, "z": float(z),
                             "K": float(K), "setting": "adaptive", "U_max": None, "n_nodes": None,
                             "ref_price": _f(ref), "price": _f(px_ad), "diff_price": _f(px_ad - ref),
                             "ref_iv": sigma, "iv": _f(iv_ad), "diff_iv": _f(iv_ad - sigma), "ref_warning": bool(w_bs)})
                for name, (U, n) in settings.items():
                    with np.errstate(all="ignore"):
                        px = gil_pelaez_call(lambda uu: bs_cf(uu, T, sigma, R), S0, K, T, R, U_max=U, n_nodes=n)
                        iv = bs_iv(px, S0, K, T, R) if (np.isfinite(px) and px > 1e-12) else np.nan
                    rows.append({"model": "bs", "T": T, "xi0": float(xi0), "nu": None, "rho": None, "z": float(z),
                                 "K": float(K), "setting": name, "U_max": float(U), "n_nodes": int(n),
                                 "ref_price": _f(ref), "price": _f(px), "diff_price": _f(px - ref),
                                 "ref_iv": sigma, "iv": _f(iv), "diff_iv": _f(iv - sigma), "ref_warning": False})
            for nu, rho in heston_pairs:
                cf = lambda uu, T=T, xi0=xi0, nu=nu, rho=rho: heston_cf(uu, T, V0=xi0, kappa=KAPPA, theta=xi0,
                                                                        nu=nu, rho=rho, r=R)
                with warnings.catch_warnings(record=True) as w_atm, np.errstate(all="ignore"):
                    warnings.simplefilter("always")
                    atm_ref = _gil_pelaez_call_adaptive(cf, S0, S0, T, R)
                    sigma_h = bs_iv(atm_ref, S0, S0, T, R)
                if not np.isfinite(sigma_h):
                    rows.append({"model": "heston", "T": T, "xi0": float(xi0), "nu": nu, "rho": rho, "z": 0.0,
                                 "K": float(S0), "setting": None, "U_max": None, "n_nodes": None,
                                 "ref_price": _f(atm_ref), "price": None, "diff_price": None, "ref_iv": None,
                                 "iv": None, "diff_iv": None, "ref_warning": bool(w_atm)})
                    continue
                Ks = strikes_for(float(sigma_h), T, vus)
                for z, K in zip(vus, Ks):
                    with warnings.catch_warnings(record=True) as w, np.errstate(all="ignore"):
                        warnings.simplefilter("always")
                        ref = _gil_pelaez_call_adaptive(cf, S0, K, T, R)
                        ref_iv = bs_iv(ref, S0, K, T, R) if (np.isfinite(ref) and ref > 1e-12) else np.nan
                    for name, (U, n) in settings.items():
                        with np.errstate(all="ignore"):
                            px = gil_pelaez_call(cf, S0, K, T, R, U_max=U, n_nodes=n)
                            iv = bs_iv(px, S0, K, T, R) if (np.isfinite(px) and px > 1e-12) else np.nan
                        rows.append({"model": "heston", "T": T, "xi0": float(xi0), "nu": nu, "rho": rho,
                                     "z": float(z), "K": float(K), "setting": name, "U_max": float(U),
                                     "n_nodes": int(n), "ref_price": _f(ref), "price": _f(px),
                                     "diff_price": _f(px - ref), "ref_iv": _f(ref_iv), "iv": _f(iv),
                                     "diff_iv": _f(iv - ref_iv), "ref_warning": bool(w or w_atm)})
    return rows


def known_answers_summary(rows):
    """Per model and setting: row count, rows with an implied vol on both sides, the
    max |diff_iv| over them, and the count of quadrature warnings on the reference."""
    out = {}
    for model, name in sorted({(r["model"], r["setting"]) for r in rows if r["setting"] is not None}):
        sel = [r for r in rows if r["model"] == model and r["setting"] == name]
        ivs = [abs(r["diff_iv"]) for r in sel if r["diff_iv"] is not None]
        out[f"{model}/{name}"] = {"n": len(sel), "n_iv": len(ivs),
                                  "max_abs_diff_iv": max(ivs) if ivs else None,
                                  "n_ref_warning": sum(1 for r in sel if r["ref_warning"])}
    return out


# --------------------------------------------------------------------------- #
# Cost estimate from two timed solves made alone
# --------------------------------------------------------------------------- #
def estimate(timing, ladder=LADDER, n_tasks=len(TASKS), n_T=len(TS)):
    """timing: {N: seconds} for the three base solves at a few N, measured alone. The
    fractional Adams recursion is O(N^2), so the cost model is c(N) = t_top * (N/N_top)^2
    anchored at the largest timed N; the exponent fitted between the smallest and the
    largest timed N is reported beside it. A full ladder costs the sum over rungs of
    (1 + 2 + 2) c(N): base, nodes doubled, U_max raised at doubled nodes. Returns the
    per-rung costs, the bound (every task on the full ladder) and a guess (labelled so)
    of where the ladders stop."""
    Ns = sorted(int(k) for k in timing)
    N_lo, N_top = Ns[0], Ns[-1]
    t_lo, t_top = float(timing[N_lo] if N_lo in timing else timing[str(N_lo)]), \
        float(timing[N_top] if N_top in timing else timing[str(N_top)])
    p = float(np.log(t_top / t_lo) / np.log(N_top / N_lo)) if (N_top > N_lo and t_lo > 0 and t_top > 0) else None
    c = {N: t_top * (N / float(N_top)) ** 2 for N in ladder}
    rung_cost = {N: 5.0 * c[N] for N in ladder}
    cum = {}
    run = 0.0
    for i, N in enumerate(ladder):
        run += rung_cost[N]
        partner = c[ladder[i + 1]] if i + 1 < len(ladder) else 0.0
        cum[N] = run + partner                       # ladder stops at N: all rungs to N plus the partner base
    full = run
    bound_core_s = n_tasks * full
    # Guess, not a measurement: H >= 0.25 stop by 2000; H 0.10 by 4000; at H <= 0.05 half the
    # triples run the full ladder and half stop at 8000.
    def stop_cost(target):
        return cum[max([N for N in ladder if N <= target] or [ladder[0]])]
    guess = 0.0
    for (H, nu, rho), T in TASKS:
        if H >= 0.25:
            guess += stop_cost(2000)
        elif H >= 0.10:
            guess += stop_cost(4000)
        else:
            guess += 0.5 * full + 0.5 * stop_cost(8000)
    return {"p_fitted": p, "anchor_N": N_top, "anchor_s": t_top, "solve_s": c, "rung_s": rung_cost,
            "stop_at_s": cum, "full_ladder_s": full,
            "triple_base_s": {N: n_T * c[N] for N in ladder},
            "bound_core_s": bound_core_s, "guess_core_s": guess}


# --------------------------------------------------------------------------- #
# Output: jsonl working record -> summary JSON + quotes CSV
# --------------------------------------------------------------------------- #
def git_state(repo_dir):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=repo_dir, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, cwd=repo_dir, check=True).stdout.strip()
        return commit, (dirty == "")
    except Exception:
        return None, None


def header(pool):
    import scipy
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    return {"kind": "header", "script": os.path.basename(__file__), "commit": commit, "clean_tree": clean,
            "seed": SEED, "tol": TOL, "ladder": list(LADDER),
            "settings": {k: [float(v[0]), int(v[1])] for k, v in SETTINGS.items()},
            "comparisons": {k: list(v) for k, v in COMPARISONS.items()},
            "H_values": list(H_VALUES), "nu_values": list(NU_VALUES), "rho_values": list(RHO_VALUES),
            "xi0_values": list(XI0_VALUES), "triples": [list(t) for t in TRIPLES], "n_tasks": len(TASKS),
            "TS": list(TS), "z": [float(z) for z in VUS], "S0": float(S0), "r": float(R), "kappa": float(KAPPA),
            "d38_point": list(D38_POINT), "known_heston": [list(p) for p in KNOWN_HESTON],
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "pool": int(pool), "utc_created": utc_now()}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_line(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")


def _rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:                                  # another drive on Windows
        return os.path.abspath(path)


def read_jsonl(path):
    """Header, launch lines and completed tasks (by id, last wins). A malformed line,
    as a crash mid-write can leave, is reported and skipped, never raised."""
    hdr, launches, tasks = None, [], {}
    if not os.path.exists(path):
        return hdr, launches, tasks
    with open(path, encoding="utf-8") as f:
        for n_line, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  warning: {_rel(path)} line {n_line} is not valid JSON and is skipped", flush=True)
                continue
            if obj.get("kind") == "header":
                hdr = obj
            elif obj.get("kind") == "launch":
                launches.append(obj)
            elif obj.get("kind") == "task":
                tasks[obj["task"]] = obj
    return hdr, launches, tasks


def task_summary(rec):
    rungs = []
    for rung in rec["rungs"]:
        comp = {c: {"met": v.get("met"), "n_met": v.get("n_met"), "n_total": v.get("n_total"),
                    "max_abs_diff": v.get("max_abs_diff"), "status": v.get("status", "computed")}
                for c, v in rung["comparisons"].items()}
        rungs.append({"N": rung["N"], "role": rung["role"],
                      "solves": {k: {"seconds": v["seconds"], "status": v["status"], "failed_at": v["failed_at"]}
                                 for k, v in rung["solves"].items()},
                      "n_ok_A": sum(s == STATUS_OK for q in rung["quotes"]["A"].values() for s in q["status"]),
                      "atm_status_A": {k: q.get("atm_status") for k, q in rung["quotes"]["A"].items()},
                      "top_modulus_A": {k: q["top_modulus"] for k, q in rung["quotes"]["A"].items()},
                      "top_log10_modulus_A": {k: q.get("top_log10_modulus") for k, q in rung["quotes"]["A"].items()},
                      "comparisons": comp})
    return {"task": rec["task"], "H": rec["H"], "nu": rec["nu"], "rho": rec["rho"], "T": rec["T"],
            "stopped_at": rec["stopped_at"], "exhausted": rec["exhausted"], "seconds_total": rec["seconds_total"],
            "first_finite_N_A": first_finite_N(rec, "A"),
            "anchors": {k: {"sigma_atm": a["sigma_atm"], "anchor_N": a["anchor_N"]} for k, a in rec["anchors"].items()},
            "rungs": rungs}


def cost_summary(tasks):
    """Per triple and rung N: the sum over the five maturities of the base solve's
    seconds, None unless all five have that rung with a finite solve (a solve that
    failed carries only the time to the failing stage), and the median over triples.
    n_ok_by_N counts the maturities whose base solve at N was finite."""
    by_triple = {}
    for rec in tasks.values():
        key = f"H{rec['H']:.2f}_nu{rec['nu']:.2f}_rho{rec['rho']:+.2f}"
        by_triple.setdefault(key, {})[rec["T"]] = rec
    per_triple = {}
    for key, recs in by_triple.items():
        per_N, n_ok = {}, {}
        for N in LADDER:
            secs, ok = [], 0
            for T in TS:
                rec = recs.get(T)
                rung = next((r for r in rec["rungs"] if r["N"] == N), None) if rec else None
                sol = rung["solves"]["A"] if rung else None
                if sol is not None and sol["status"] == STATUS_OK and sol["seconds"] is not None:
                    secs.append(sol["seconds"])
                    ok += 1
            per_N[str(N)] = sum(secs) if ok == len(TS) else None
            n_ok[str(N)] = ok
        per_triple[key] = {"base_solves_s_by_N": per_N, "n_ok_by_N": n_ok, "n_maturities_done": len(recs),
                           "seconds_total": sum(r["seconds_total"] or 0.0 for r in recs.values())}
    median = {}
    for N in LADDER:
        vals = [v["base_solves_s_by_N"][str(N)] for v in per_triple.values() if v["base_solves_s_by_N"][str(N)] is not None]
        median[str(N)] = {"n_triples": len(vals), "median_s": statistics.median(vals) if vals else None}
    return {"per_triple": per_triple, "median_base_solves_s_by_N": median}


CSV_COLUMNS = ["task", "H", "nu", "rho", "T", "xi0", "z", "N", "role", "sigma_atm", "anchor_N", "atm_status_A", "K",
               "quote_A", "status_A", "quote_C", "status_C", "quote_D", "status_D",
               "diff_N2", "diff_nodes", "diff_umax",
               "modulus_A", "modulus_C", "modulus_D", "log10_modulus_A", "log10_modulus_C", "log10_modulus_D",
               "seconds_A", "seconds_C", "seconds_D"]


def csv_rows(rec):
    rows = []
    for rung in rec["rungs"]:
        for xi0 in rec["xi0s"]:
            key = _xi_key(xi0)
            anchor = rec["anchors"].get(key)
            qa = rung["quotes"]["A"][key]
            qc = rung["quotes"].get("C", {}).get(key)
            qd = rung["quotes"].get("D", {}).get(key)
            comp = rung["comparisons"]
            for i, z in enumerate(rec["z"]):
                def _d(name):
                    c = comp.get(name)
                    return None if c is None else c["diffs"][key][i]
                rows.append({
                    "task": rec["task"], "H": rec["H"], "nu": rec["nu"], "rho": rec["rho"], "T": rec["T"],
                    "xi0": xi0, "z": z, "N": rung["N"], "role": rung["role"],
                    "sigma_atm": anchor["sigma_atm"] if anchor else None,
                    "anchor_N": anchor["anchor_N"] if anchor else None,
                    "atm_status_A": qa.get("atm_status"),
                    "K": anchor["K"][i] if anchor else None,
                    "quote_A": qa["quotes"][i], "status_A": qa["status"][i],
                    "quote_C": qc["quotes"][i] if qc else None, "status_C": qc["status"][i] if qc else "not_computed",
                    "quote_D": qd["quotes"][i] if qd else None, "status_D": qd["status"][i] if qd else "not_computed",
                    "diff_N2": _d("N2"), "diff_nodes": _d("nodes"), "diff_umax": _d("umax"),
                    "modulus_A": qa["top_modulus"], "modulus_C": qc["top_modulus"] if qc else None,
                    "modulus_D": qd["top_modulus"] if qd else None,
                    "log10_modulus_A": qa.get("top_log10_modulus"),
                    "log10_modulus_C": qc.get("top_log10_modulus") if qc else None,
                    "log10_modulus_D": qd.get("top_log10_modulus") if qd else None,
                    "seconds_A": rung["solves"]["A"]["seconds"],
                    "seconds_C": rung["solves"].get("C", {}).get("seconds"),
                    "seconds_D": rung["solves"].get("D", {}).get("seconds")})
    return rows


def report(jsonl_path=JSONL_PATH, json_path=JSON_PATH, csv_path=CSV_PATH):
    """Regenerate the summary JSON and the quotes CSV from the jsonl working record."""
    hdr, launches, tasks = read_jsonl(jsonl_path)
    ordered = [tasks[task_id(t, T)] for t, T in TASKS if task_id(t, T) in tasks]
    ordered += [rec for k, rec in tasks.items() if k not in {task_id(t, T) for t, T in TASKS}]
    known = next((l["known_answers"] for l in reversed(launches) if l.get("known_answers")), None)
    summary = {"header": hdr, "launches": [{k: v for k, v in l.items() if k != "known_answers"} for l in launches],
               "n_tasks_done": len(tasks), "n_tasks_total": len(TASKS), "utc_report": utc_now(),
               "known_answers_summary": known_answers_summary(known) if known else None,
               "known_answers": known, "cost": cost_summary(tasks), "tasks": [task_summary(r) for r in ordered]}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, allow_nan=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for rec in ordered:
            for row in csv_rows(rec):
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return summary


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #
def _worker(args):
    triple, T, kw = args
    return run_task(triple, T, **kw)


def launch(pool, jsonl_path=JSONL_PATH, json_path=JSON_PATH, csv_path=CSV_PATH, known=None, timing=None,
           tasks=TASKS, task_kwargs=None):
    """The pool over `tasks`, skipping those already in the jsonl. `task_kwargs` exist for
    the mechanics tests (a cheap ladder); the run passes none, so every task uses the
    module's settings."""
    hdr, launches, done = read_jsonl(jsonl_path)
    if hdr is None:
        append_line(jsonl_path, header(pool))
    commit, clean = git_state(os.path.dirname(os.path.abspath(__file__)))
    append_line(jsonl_path, {"kind": "launch", "utc": utc_now(), "pool": int(pool), "n_done_before": len(done),
                             "commit": commit, "clean_tree": clean, "timing_alone_s": timing, "known_answers": known})
    todo = [(t, T, task_kwargs or {}) for t, T in tasks if task_id(t, T) not in done]
    print(f"\n=== RUN: {len(todo)} tasks to do ({len(done)} already in {_rel(jsonl_path)}), pool={pool} ===",
          flush=True)
    t0 = time.time()
    n = 0
    n_err = 0
    ex = ProcessPoolExecutor(max_workers=pool)
    try:
        futs = {ex.submit(_worker, job): job for job in todo}
        for fut in as_completed(futs):
            triple, T, _ = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:                      # one task's failure is recorded; the rest go on
                n_err += 1
                append_line(jsonl_path, {"kind": "task_error", "task": task_id(triple, T), "utc": utc_now(),
                                         "error": repr(e)})
                print(f"  task {task_id(triple, T)} raised {e!r}; recorded, continuing", flush=True)
                continue
            append_line(jsonl_path, rec)
            report(jsonl_path, json_path, csv_path)
            n += 1
            ff = first_finite_N(rec)
            print(f"  [{n:3d}/{len(todo)}] {rec['task']}: stopped_at={rec['stopped_at']} "
                  f"exhausted={rec['exhausted']} first_finite_N(A)={ff} "
                  f"task={rec['seconds_total']:.0f}s elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    except BaseException:                               # Ctrl-C or a fatal error: do not let pending tasks run on
        print("\n  stopping: pending tasks cancelled; re-run with --run to resume", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    print(f"=== done: {n} tasks ({n_err} errors) in {(time.time() - t0) / 60:.1f} min; outputs beside "
          f"{_rel(jsonl_path)} ===", flush=True)
    return n


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P5 gate G0 scoping run (see module docstring)")
    ap.add_argument("--run", action="store_true", help="launch the worker pool over the 300 tasks (resumable)")
    ap.add_argument("--report", action="store_true", help="regenerate the JSON and CSV from the jsonl and exit")
    ap.add_argument("--pool", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="worker processes (default: half the logical processors)")
    a = ap.parse_args(argv)

    if a.report:
        s = report()
        print(f"report: {s['n_tasks_done']}/{s['n_tasks_total']} tasks -> {_rel(JSON_PATH)}, {_rel(CSV_PATH)}",
              flush=True)
        return

    print("=== P5 G0 scoping run ===", flush=True)
    print(f"  tol={TOL}  ladder={LADDER}  settings={SETTINGS}  seed={SEED}", flush=True)
    print(f"  lattice: H{H_VALUES} x nu{NU_VALUES} x rho{RHO_VALUES} = {len(TRIPLES)} triples; "
          f"xi0{XI0_VALUES}; T{TS}; z{tuple(float(z) for z in VUS)}; {len(TASKS)} tasks", flush=True)
    print(f"  first nine triples: {TRIPLES[:9]}", flush=True)

    t0 = time.perf_counter()
    d, n = identity_gate()
    print(f"\n=== identity gate: max |this path - model_smile_cf_T| over {n} quotes = {d!r} "
          f"({time.perf_counter() - t0:.0f}s) ===", flush=True)
    if d != 0.0:
        raise SystemExit("identity gate failed: this script's pricing path does not reproduce the wrapper")

    t0 = time.perf_counter()
    known = known_answers()
    ks = known_answers_summary(known)
    print(f"\n=== known answers ({len(known)} rows, {time.perf_counter() - t0:.0f}s): max |diff_iv| by model/setting ===",
          flush=True)
    for k, v in ks.items():
        mx = "n/a" if v["max_abs_diff_iv"] is None else f"{v['max_abs_diff_iv']:.2e}"
        print(f"  {k:10s} n={v['n']:4d} max|diff_iv|={mx}  ref_warnings={v['n_ref_warning']}", flush=True)
    for r in known:
        if r["model"] == "bs" and r["T"] == min(TS) and r["xi0"] == min(XI0_VALUES) and r["z"] == 0.0:
            print(f"  bs ATM T={r['T']} xi0={r['xi0']} {r['setting']}: iv={r['iv']} ref={r['ref_iv']} "
                  f"diff_iv={r['diff_iv']}", flush=True)

    print("\n=== timing alone (D38's triple, T=0.10, setting A) ===", flush=True)
    H, nu, rho, _ = D38_POINT
    timing = {}
    for N in (1000, 2000, 4000):
        rec, _, _ = solve_cf(H, nu, rho, 0.10, N, *SETTINGS["A"])
        timing[str(N)] = rec["seconds"]
        print(f"  N={N}: three base solves {rec['seconds']:.2f}s ({rec['status']})", flush=True)
    est = estimate(timing)
    print(f"\n=== estimate (O(N^2) anchored at N={est['anchor_N']}: {est['anchor_s']:.2f}s; fitted exponent "
          f"{est['p_fitted']:.2f}; one core, alone) ===", flush=True)
    for N in LADDER:
        print(f"  N={N:6d}: base solves {est['solve_s'][N]:7.1f}s  rung (A+C+D) {est['rung_s'][N]:7.1f}s  "
              f"ladder stopping here {est['stop_at_s'][N]:7.1f}s/task  "
              f"one triple's base solves over {len(TS)} T: {est['triple_base_s'][N]:7.1f}s", flush=True)
    print(f"  bound (every task on the full ladder): {est['bound_core_s'] / 3600:.1f} core-hours "
          f"= {est['bound_core_s'] / 3600 / a.pool:.1f} h on {a.pool} workers, if no slowdown", flush=True)
    print(f"  guess (not a measurement; H>=0.25 stop by 2000, H 0.10 by 4000, H<=0.05 half full): "
          f"{est['guess_core_s'] / 3600:.1f} core-hours = {est['guess_core_s'] / 3600 / a.pool:.1f} h on {a.pool} workers",
          flush=True)

    if not a.run:
        print("\nHOLD: nothing written. Add --run to launch (mains power, sleep off; Ctrl-C stops, --run resumes).",
              flush=True)
        return
    launch(a.pool, known=known, timing=timing)


if __name__ == "__main__":
    main()
