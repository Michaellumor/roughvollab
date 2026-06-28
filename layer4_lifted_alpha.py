"""
layer4_lifted_alpha.py — H=0.10 resolution: re-run brick-3's weak-order (alpha)
study on the LIFTED simulator to resolve the borderline H=0.10 via fine grids the
explicit O(n^2) Volterra sim couldn't reach (the lift is O(N*n)).

Reuses brick-3's harness (layer4_convergence.measure_alpha / mc_call_levels) through
its `sim` callback — swaps the explicit core for the lifted QE-port core with
grid-INDEPENDENT SOE factors, so (a) the CRN coupling holds and (b) the coupled
estimator cancels the SOE kernel bias EXACTLY (P_l and P_{l-1} share the same kernel
-> P_SOE cancels in the difference; only the discretization survives).

Validation ladder (gate each):
  1. H=1/2 anchor on the lift (recover the D31 anchor ~1.1 OTM).
  2. GATE 2 (known-answer): match brick-3's RESOLVED alpha at H=0.20 (~1.01) and
     H=0.05 (~0.74) at coarse grids n<=64 -> licenses the fine-grid extension.
  3. Fine grids at H=0.10: push n past the explicit reach; does alpha converge?

Probed SOE floor at N=150 (~189 factors) is ~2e-4 — >=1 order below the fine-grid b_n
(~4.5e-3 at n=512), so both estimators are valid; the coupled is the SOE-immune anchor.
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rough_heston import PARAMS
from rough_heston_lifted import _lifted_from_increments, lifted_setup
from layer4_convergence import (measure_alpha, mc_call_levels, cf_reference,
                                _fit_slope, _combine)

# brick-3 (D31) resolved alpha, OTM K=110, prec-weighted -- the known answers.
BRICK3 = {0.5: 1.10, 0.20: 1.01, 0.05: 0.74}


def lifted_sim(H, p, N=150, positivity="qe"):
    """Bind a lifted path generator sim(dWV,dWp,n)->(S,V) with grid-independent SOE
    factors (same gammas/weights for every n)."""
    g, w = lifted_setup(H, N, T=p["T"])
    sim = lambda dWV, dWp, n: _lifted_from_increments(dWV, dWp, n, H, p, g, w, positivity)
    return sim, int(g.size)


def _alpha_over_windows(H, p, N, M, seed, K=110.0, n0=4, L=6, N_riccati=4000):
    """One lifted conditional-MC run over n=n0..n0*2^L; fit absolute & coupled alpha
    over nested windows ns<=n_max to test window-convergence (the brick-3 issue)."""
    sim, nfac = lifted_sim(H, p, N)
    res = mc_call_levels(H, p, n0, L, M, np.random.default_rng(seed), K,
                         conditional=True, sim=sim)
    P_CF = cf_reference(H, p, K, N_riccati)
    ns = res["ns"]; dt = p["T"] / ns
    b = np.abs(res["cv_mean"] - P_CF)
    dtl = p["T"] / ns[1:]
    rows = []
    for nmax in (64, 128, 256, 512, 1024):
        if nmax > ns[-1]:
            continue
        ia = ns <= nmax; ic = ns[1:] <= nmax
        a_dir, a_dir_se, nd, _ = _fit_slope(dt[ia], b[ia], res["cv_se"][ia])
        a_cpl, a_cpl_se, nc, _ = _fit_slope(dtl[ic], res["dmean"][ic], res["dse"][ic])
        a_comb, _ = _combine(a_dir, a_dir_se, a_cpl, a_cpl_se)
        rows.append((nmax, a_dir, a_cpl, a_comb, nd, nc))
    return dict(P_CF=P_CF, nfac=nfac, ns=ns, b=b, cv_se=res["cv_se"],
                dmean=res["dmean"], dse=res["dse"], rows=rows)


def run(N=150, M=200000, seed=7, L_fine=6, do_fine=True):
    p = dict(PARAMS)                                     # nu=0.20 (brick-3 regime)
    print(f"H=0.10 RESOLUTION — lifted weak-order re-run | N={N} M={M} OTM K=110 nu={p['nu']}")

    print("\n[LADDER 1] H=1/2 anchor on the lift (expect ~1.1 OTM, D31):")
    sim, nfac = lifted_sim(0.5, p, N)
    r = measure_alpha(0.5, p, 4, 4, M, 110.0, 4000, seed, conditional=True, sim=sim)
    print(f"  a_dir={r['a_dir']:.3f}  a_cpl={r['a_cpl']:.3f}  a_comb={r['a_comb']:.3f}  "
          f"(factors={nfac}; brick-3 ~{BRICK3[0.5]})")

    print("\n[LADDER 2] GATE 2 — match brick-3 alpha at coarse grids n<=64 (license to extend):")
    print(f"  {'H':>5} {'a_dir':>7} {'a_cpl':>7} {'a_comb':>7} {'brick3':>7} {'|diff|':>7} {'verdict':>8}")
    gate2_ok = True
    for H in (0.20, 0.05):
        sim, nfac = lifted_sim(H, p, N)
        r = measure_alpha(H, p, 4, 4, M, 110.0, 4000, seed, conditional=True, sim=sim)
        diff = abs(r["a_comb"] - BRICK3[H]); ok = diff <= 0.10
        gate2_ok = gate2_ok and ok
        print(f"  {H:>5} {r['a_dir']:>7.3f} {r['a_cpl']:>7.3f} {r['a_comb']:>7.3f} "
              f"{BRICK3[H]:>7.2f} {diff:>7.3f} {'OK' if ok else 'MISS':>8}")
    print(f"  GATE 2: {'PASS — SOE preserves the weak rate, extend to fine grids' if gate2_ok else 'FAIL — STOP (SOE perturbs the weak rate)'}")
    if not (gate2_ok and do_fine):
        return

    print(f"\n[LADDER 3] FINE-GRID H=0.10 — alpha vs window n_max (does it converge?):")
    res = _alpha_over_windows(0.10, p, N, M, seed, n0=4, L=L_fine)
    print(f"  CF={res['P_CF']:.5f}  factors={res['nfac']}")
    print(f"  {'n_max':>6} {'a_dir':>7} {'a_cpl':>7} {'a_comb':>7} {'nd':>3} {'nc':>3}")
    for nmax, ad, ac, acomb, nd, nc in res["rows"]:
        print(f"  {nmax:>6} {ad:>7.3f} {ac:>7.3f} {acomb:>7.3f} {nd:>3} {nc:>3}")
    print("  brick-3 (explicit): 0.84 (n<=64) vs 0.95 (n<=128) — the borderline window-sensitivity")
    last = res["rows"][-1]
    print(f"  -> finest window n<=({last[0]}): a_comb={last[3]:.3f} -> "
          f"{classify_alpha(last[3])}")


def classify_alpha(a):
    if a >= 0.95:
        return "(ii) PASS-like (alpha~1)"
    if a < 0.85:
        return "(i) PARTIAL-like (alpha<1 penalty)"
    return "(iii) still borderline (0.85-0.95)"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--anchor-gate", action="store_true", help="ladder 1+2 only")
    a = ap.parse_args()
    if a.quick:
        run(N=80, M=40000, L_fine=5, do_fine=True)
    elif a.anchor_gate:
        run(N=150, M=200000, do_fine=False)
    else:
        run(N=150, M=200000, L_fine=6)
