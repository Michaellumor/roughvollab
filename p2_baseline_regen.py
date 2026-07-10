"""
p2_baseline_regen.py — regenerate the kappa=0 NAIVE baseline numbers for
p2_paper.tex on THIS machine, through the production functions in
layer1b_mlmc_asian.py.  READ-ONLY: the engine is imported, never modified.

Seeds (documented): estimate_rates -> 7, adaptive/mlmc_run -> 11, H-sweep -> 23.
H = 0.10, default PARAMS.  Prints a paste-ready \newcommand block, the four
tab:cost rows, and a changed-values list (vs any local p2_paper.tex it can find,
plus the documented beta-sweep reference).

Run:  python p2_baseline_regen.py
"""

import os
import re
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import layer1b_mlmc_asian as L
from layer1b_mlmc_asian import (PARAMS, _volterra, mlmc_asian_level,
                                estimate_rates, mlmc_run, _bs_call)

# ── documented reference values for the diff (from the prompt / antithetic gate)
DOC_REF = {"bA": 0.120, "bB": 0.219, "bC": 0.418, "bD": 0.726}


# ── VALIDATION (Section 5) ───────────────────────────────────────────────────
def validation():
    p = dict(PARAMS)                                   # H = 0.10
    H, eta, xi0, T = p["H"], p["eta"], p["xi0"], p["T"]
    n, N = 256, 200_000

    # (1) Volterra variance at n=256 — empirical vs discrete analytic (cont = 1)
    rng = np.random.default_rng(1)
    dW1 = rng.standard_normal((N, n)) * np.sqrt(T / n)
    W, v = _volterra(dW1, n, p, kappa=0)               # production Volterra
    VarWe = float(W[:, -1].var())
    VarWd = float(v[-1])
    VarWe_se = VarWe * np.sqrt(2.0 / N)

    # (2) forward variance: max_t |E[V_t]/xi0 - 1| and the MC-noise bound
    V = xi0 * np.exp(eta * W - 0.5 * eta**2 * v[None, :])
    fwdv = float(np.abs(V.mean(0) / xi0 - 1.0).max())
    fwdn = float(3 * np.exp(eta**2 * v[-1] / 2) / np.sqrt(N))

    # (3) Black-Scholes anchor at eta=0 (European, level 3, seed 2)
    p0 = dict(p, eta=0.0)
    Nb = 100_000
    samp = mlmc_asian_level(3, Nb, p0, payoff="european",
                            rng=np.random.default_rng(2))
    eurMC = float(samp[1].mean())
    eurSE = float(samp[1].std() / np.sqrt(Nb))
    eurBS = float(_bs_call(p["S0"], p["K"], T, p["r"], np.sqrt(xi0)))
    zbs = abs(eurMC - eurBS) / eurSE

    return dict(VarWe=(VarWe, VarWe_se), VarWd=(VarWd, 0.0),
                fwdv=(fwdv, fwdn), fwdn=(fwdn, 0.0),
                eurMC=(eurMC, eurSE), eurSE=(eurSE, 0.0),
                eurBS=(eurBS, 0.0), zbs=(zbs, 0.0))


# ── RATES (Section 6) ────────────────────────────────────────────────────────
def rates():
    # main run, seed 7: H=0.10, L=6, N=20000 -> beta, gamma, consistency, vP, alpha
    main = estimate_rates(L=6, N=20_000, p=PARAMS, seed=7, verbose=False)
    cons = float(main["consistency"])
    # H-sweep, seed 23, L=5, N=12000 (matches the antithetic gate config)
    sweep = {}
    for H in (0.05, 0.10, 0.20, 0.35):
        r = estimate_rates(L=5, N=12_000, p=dict(PARAMS, H=H), seed=23,
                           verbose=False)
        sweep[H] = float(r["beta"])
    # RVL-033: the canonical H=0.10 baseline for the paper is bBmain (the seed-7
    # main run, L=6/N=20000); bB=sweep[0.10] is the lower-fidelity β-vs-H trend point.
    out = dict(bA=(sweep[0.05], None), bB=(sweep[0.10], None),
               bC=(sweep[0.20], None), bD=(sweep[0.35], None),
               bBmain=(float(main["beta"]), None),
               gammaval=(float(main["gamma"]), None),
               telcons=(cons, None))
    return out, main


# ── COST (Section 6, tab:cost) ───────────────────────────────────────────────
def cost(main):
    # naive adaptive MLMC with the seed-7 rates fixed (as section3_mlmc does),
    # seed 11; eps in {0.2,0.1,0.05,0.025}
    a0, b0, vP = main["alpha"], main["beta"], main["vP"]
    n0 = PARAMS["n0"]
    eps_list = [0.2, 0.1, 0.05, 0.025]
    runs = []
    for e in eps_list:
        r = mlmc_run(e, alpha=a0, beta=b0, seed=11, verbose=False)
        cost_mc = 2.0 * vP / e**2 * n0 * 2**r["L"]
        runs.append(dict(eps=e, L=int(r["L"]), cost=float(r["cost"]),
                         price=float(r["price"]), cost_mc=cost_mc,
                         ratio=cost_mc / r["cost"]))
    costratio = runs[-1]["ratio"]                       # std MC / MLMC at eps=0.025
    asianpx = runs[-1]["price"]                          # finest-eps price
    Ls = [r["L"] for r in runs]
    mono = all(Ls[i] <= Ls[i + 1] for i in range(len(Ls) - 1))
    return runs, dict(costratio=(costratio, None), asianpx=(asianpx, None)), mono


# ── LaTeX formatting ─────────────────────────────────────────────────────────
FMT = {"VarWe": "%.4f", "VarWd": "%.4f", "fwdv": "%.4f", "fwdn": "%.4f",
       "eurMC": "%.4f", "eurSE": "%.4f", "eurBS": "%.4f", "zbs": "%.2f",
       "telcons": "%.3f", "bA": "%.3f", "bB": "%.3f", "bC": "%.3f", "bD": "%.3f",
       "bBmain": "%.3f", "gammaval": "%.3f", "costratio": "%.2f", "asianpx": "%.4f"}
ORDER = ["VarWe", "VarWd", "fwdv", "fwdn", "eurMC", "eurSE", "eurBS", "zbs",
         "telcons", "bA", "bB", "bC", "bD", "bBmain", "gammaval",
         "costratio", "asianpx"]


def sci(x):
    e = int(np.floor(np.log10(abs(x)))) if x else 0
    m = x / 10**e
    return f"{m:.2f}\\times10^{{{e}}}"


def load_placeholders():
    for cand in ("p2_paper.tex", os.path.join("OVERLEAF", "p2_paper.tex"),
                 os.path.join("OVERLEAF", "P2", "p2_paper.tex")):
        if os.path.exists(cand):
            txt = open(cand, encoding="utf-8", errors="ignore").read()
            ph = {}
            for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt):
                try:
                    ph[m.group(1)] = float(re.sub(r"[^0-9eE.+-]", "", m.group(2)))
                except ValueError:
                    pass
            return ph, cand
    return {}, None


def main():
    vals = {}
    v = validation(); vals.update(v)
    r, main = rates(); vals.update(r)
    runs, c, mono = cost(main); vals.update(c)

    print("=" * 74)
    print("  PASTE-READY MACRO BLOCK for p2_paper.tex  (kappa=0 baseline)")
    print("=" * 74)
    for k in ORDER:
        val = vals[k][0]
        print(f"\\newcommand{{\\{k}}}{{{FMT[k] % val}}}")

    print("\n" + "=" * 74)
    print("  tab:cost rows  (eps & L & total cost & std-MC/MLMC & price)")
    print("=" * 74)
    for rr in runs:
        print(f"  ${rr['eps']:g}$ & {rr['L']} & ${sci(rr['cost'])}$ & "
              f"${rr['ratio']:.2f}$ & ${rr['price']:.4f}$ \\\\")
    Ls = [rr["L"] for rr in runs]
    mono_str = ("monotone" if mono else
                "** NON-MONOTONE in eps (adaptive-selector sensitive, "
                "as on the antithetic gate) **")
    print(f"\n  (finest level L per row: {Ls}  ->  {mono_str})")

    # ── changed-values list ──────────────────────────────────────────────────
    ph, src = load_placeholders()
    ref = dict(DOC_REF); ref.update(ph)               # parsed file overrides DOC
    print("\n" + "=" * 74)
    print("  CHANGED VALUES (differ from placeholder by > 1 s.e.)")
    if src:
        print(f"  placeholders parsed from: {src}")
    else:
        print("  p2_paper.tex not found locally -> only the documented beta-sweep")
        print("  reference is diffed; compare the rest against Overleaf by eye.")
    print("=" * 74)
    any_changed = False
    for k in ORDER:
        if k not in ref:
            continue
        new, se = vals[k][0], (vals[k][1] or 0.0)
        old = ref[k]
        tol = max(se, 1e-3)                            # 1 s.e., floor 0.001
        if abs(new - old) > tol:
            any_changed = True
            print(f"  \\{k}: placeholder {old:g}  ->  regenerated "
                  f"{FMT[k] % new}   (|Δ|={abs(new-old):.4f} > {tol:.4f})")
    if not any_changed:
        print("  none of the diffable macros changed beyond 1 s.e. "
              "(beta sweep reproduces 0.120/0.219/0.418/0.726).")

    # full value+se listing so every macro can be eyeballed vs Overleaf
    print("\n  ALL regenerated values (value +/- s.e.):")
    for k in ORDER:
        val, se = vals[k]
        s = f" +/- {se:.4f}" if se else ""
        print(f"    {k:<10s} = {FMT[k] % val}{s}")


if __name__ == "__main__":
    main()
