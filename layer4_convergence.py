"""
layer4_convergence.py — weak-order (α) study of the κ=0 hybrid scheme (Layer 4, brick 3)
========================================================================================
Measures the WEAK convergence order α of the rough-Heston MC simulator (brick 1,
`rough_heston.py`) by pricing a smooth European call across refining Δt and
comparing to the brick-2 CF known-truth (`rough_heston_cf.py`). Regime: ν ≤ 0.20.

Committed prediction (spec §3, set BEFORE measuring): α > H, expect ≈ 1.
Outcomes (pre-specified): PASS (α>H,≈1) / PARTIAL (H<α<1) / FAIL-but-publishable (α≈H).

H=½ ANCHOR — pinned expectation (NOT assumed): Euler-type log-Heston weak order is
1 if Feller ν_F=2κθ/σ²>1, else ≈ν_F (arXiv:2106.10926, Thm 1.1). At our params
ν_F=2·0.3·0.04/0.20²=0.60<1, so the Euler floor is ≈0.60; QE variance may beat it,
the left-point asset Euler caps at 1, and the ATM-call kink (non-C⁶) may degrade it.
Expected H=½ order ∈ [~0.6, 1] — the anchor MEASURES which, validating the harness
on a regime with literature guidance before any H<½ α is trusted.

NOT BEING FOOLED — two estimators + two guards:
  • primary   : absolute bias  b_n = |E[P_n^MC] − P_CF|  (vs the CF known-truth), with a
                BS-price CONTROL VARIATE to shrink σ and widen the bias-dominated window.
  • cross-check: coupled level-means  Y_l = E[P_{n_l} − P_{n_{l-1}}]  (CRN, low-variance;
                the Giles weak rate). CV cancels in the difference.
  Guard 1 (noise<bias): CRN across the refined grid; report MC s.e. band; fit ONLY where
    b_n > 5·s.e.; require the two estimators' slopes to agree.
  Guard 2 (ref<bias): N_riccati from brick-2's curve so reference error ≥1 order below the
    smallest fitted b_n; reported + certified.

Scope: SPX calibration / multifactor lift / exotic payoffs are OUT.
"""
import numpy as np
from scipy.special import ndtr

from rough_heston import PARAMS as RH_PARAMS, _rough_heston_from_increments
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_call


def _bs_call_vec(S, K, T, r, sigma):
    """Vectorised Black–Scholes call (S, sigma may be arrays); for the Romano–Touzi
    conditional price."""
    sig = np.maximum(sigma, 1e-12)
    srt = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / srt
    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d1 - srt)

_CF_KEYS = ("V0", "kappa", "theta", "nu", "rho")


def _agg(dW, n_target):
    """Aggregate fine BM increments (M, n_f) → coarse (M, n_target) by summing
    consecutive groups (CRN coupling: coarse increment = sum of fine)."""
    M, n_f = dW.shape
    g = n_f // n_target
    return dW.reshape(M, n_target, g).sum(axis=2)


def mc_call_levels(H, p, n0, L, M, rng, K, positivity="qe", sigma_cv=None, conditional=False,
                   sim=None):
    """CRN-coupled MC European-call across levels l=0..L (n_l = n0·2^l), all driven
    by ONE finest Brownian path (coarse = aggregated fine), with a BS-price control
    variate built from the (grid-independent) total asset BM W_S(T).

    `sim(dWV, dWp, n) -> (S, V)` is the path generator; default is the explicit
    brick-1 core. Pass a lifted closure (rough_heston_lifted._lifted_from_increments
    with bound grid-independent gammas/weights) to run the study on the lift — the
    coupled differences then also cancel the SOE kernel bias exactly.

    Returns dict: ns, cv_mean[L+1], cv_se[L+1] (CV-corrected E[P_n]),
                  dmean[L], dse[L] (coupled E[P_l−P_{l-1}] and its s.e.),
                  cv_var_ratio (Var reduction from the control variate)."""
    if sim is None:
        sim = lambda dWV_, dWp_, n_: _rough_heston_from_increments(dWV_, dWp_, n_, H, p,
                                                                   positivity=positivity)
    T, S0, r, rho = p["T"], p["S0"], p["r"], p["rho"]
    n_max = n0 * 2 ** L
    dt_f = T / n_max
    dWV = rng.standard_normal((M, n_max)) * np.sqrt(dt_f)
    dWp = rng.standard_normal((M, n_max)) * np.sqrt(dt_f)

    # BS control variate: total asset BM W_S(T) ~ N(0,T), grid-independent
    WS_T = rho * dWV.sum(axis=1) + np.sqrt(1.0 - rho ** 2) * dWp.sum(axis=1)
    sig = sigma_cv if sigma_cv is not None else np.sqrt(p["theta"])
    pay_bs = np.maximum(S0 * np.exp((r - 0.5 * sig ** 2) * T + sig * WS_T) - K, 0.0)
    bs_exact = bs_call(S0, K, T, r, sig)
    var_bs = np.var(pay_bs)

    ns, payoffs = [], []
    for l in range(L + 1):
        n_l = n0 * 2 ** l
        dWV_l, dWp_l = _agg(dWV, n_l), _agg(dWp, n_l)
        S, V = sim(dWV_l, dWp_l, n_l)
        if conditional:
            # Romano–Touzi: integrate out the orthogonal asset BM analytically.
            # Conditional on (V-path, W^V): call = BS(S_eff, K, T, 0, sig_eff),
            #   M = ∫√V dW^V,  I = ∫V dt (left-point);  S_eff = S0·exp(ρM − ½ρ²I),
            #   sig_eff² = (1−ρ²)·I/T.  Same V/M/I discretisation ⇒ same weak bias.
            Vl = np.maximum(V[:, :-1], 0.0)
            Mt = (np.sqrt(Vl) * dWV_l).sum(axis=1)
            It = Vl.sum(axis=1) * (T / n_l)
            S_eff = S0 * np.exp(rho * Mt - 0.5 * rho ** 2 * It)
            sig_eff = np.sqrt(np.maximum((1.0 - rho ** 2) * It / T, 1e-300))
            payoffs.append(_bs_call_vec(S_eff, K, T, r, sig_eff))
        else:
            payoffs.append(np.maximum(S[:, -1] - K, 0.0))
        ns.append(n_l)
    payoffs = np.asarray(payoffs)                       # (L+1, M)

    cv_mean, cv_se, vr = [], [], []
    for l in range(L + 1):
        pr = payoffs[l]
        beta = np.cov(pr, pay_bs)[0, 1] / var_bs        # optimal CV coefficient
        prc = pr - beta * (pay_bs - bs_exact)
        cv_mean.append(prc.mean()); cv_se.append(prc.std(ddof=1) / np.sqrt(M))
        vr.append(prc.var() / pr.var())
    dmean, dse = [], []
    for l in range(1, L + 1):                            # coupled differences (CV cancels)
        d = payoffs[l] - payoffs[l - 1]
        dmean.append(d.mean()); dse.append(d.std(ddof=1) / np.sqrt(M))
    return dict(ns=np.asarray(ns), cv_mean=np.asarray(cv_mean), cv_se=np.asarray(cv_se),
                dmean=np.asarray(dmean), dse=np.asarray(dse),
                cv_var_ratio=float(np.mean(vr)))


def cf_reference(H, p, K, N_riccati):
    cf = lambda u: rough_heston_cf(u, p["T"], H=H, N_riccati=N_riccati,
                                   **{k: p[k] for k in _CF_KEYS})
    return gil_pelaez_call(cf, p["S0"], K, p["T"], p["r"], U_max=200.0, n_nodes=128)


def _fit_slope(dt, y, yse, k_sigma=5.0):
    """Fit slope α of log|y| vs log(dt) over the bias-dominated window (|y| > k·s.e.).
    Returns (alpha, alpha_se, n_used, mask). alpha_se = nan if <3 usable points
    (no regression dof); alpha = nan if <2."""
    y = np.abs(np.asarray(y, float))
    mask = y > k_sigma * np.asarray(yse, float)
    if mask.sum() < 2:
        return np.nan, np.nan, int(mask.sum()), mask
    x, ly = np.log(dt[mask]), np.log(y[mask])
    coef = np.polyfit(x, ly, 1)
    a = float(coef[0])
    if mask.sum() >= 3:
        resid = ly - np.polyval(coef, x)
        a_se = float(np.sqrt((resid ** 2).sum() / (mask.sum() - 2)
                             / ((x - x.mean()) ** 2).sum()))
    else:
        a_se = np.nan
    return a, a_se, int(mask.sum()), mask


def classify(alpha, alpha_se, H):
    """PASS / PARTIAL / FAIL-but-publishable per the committed prediction (α>H, ≈1)."""
    if np.isnan(alpha):
        return "INCONCLUSIVE (no bias-dominated fit)"
    ci = 1.96 * alpha_se if not np.isnan(alpha_se) else 0.10
    if abs(alpha - H) <= ci:
        return "FAIL-but-publishable (α≈H: roughness bottlenecks the weak rate)"
    if alpha >= 0.85:
        return "PASS (α>H and ≈1: weak converges classically)"
    if alpha > H:
        return "PARTIAL (H<α<1: quantified roughness penalty)"
    return "BELOW-H (unexpected)"


def _combine(a1, se1, a2, se2):
    """Inverse-variance-weighted combination of the two slope estimators —
    down-weights the imprecise one (at OTM the absolute b_n is the tighter one,
    so it dominates; the noisy coupled tail does not swing the verdict). Falls
    back to whichever estimator is finite."""
    ests = [(a, se) for a, se in ((a1, se1), (a2, se2))
            if not np.isnan(a) and not np.isnan(se) and se > 0]
    if not ests:
        alphas = [a for a in (a1, a2) if not np.isnan(a)]
        return (float(np.mean(alphas)) if alphas else np.nan), np.nan
    w = np.array([1.0 / se ** 2 for _, se in ests])
    a = float((w * np.array([e for e, _ in ests])).sum() / w.sum())
    return a, float(1.0 / np.sqrt(w.sum()))


def measure_alpha(H, p, n0, L, M, K, N_riccati, seed, positivity="qe", conditional=False, sim=None):
    rng = np.random.default_rng(seed)
    res = mc_call_levels(H, p, n0, L, M, rng, K, positivity=positivity, conditional=conditional,
                         sim=sim)
    P_CF = cf_reference(H, p, K, N_riccati)
    ns, dt = res["ns"], p["T"] / res["ns"]
    b = np.abs(res["cv_mean"] - P_CF)                    # absolute bias vs known-truth
    a_dir, a_dir_se, nd, mdir = _fit_slope(dt, b, res["cv_se"])
    dtl = p["T"] / ns[1:]
    a_cpl, a_cpl_se, nc, mcpl = _fit_slope(dtl, res["dmean"], res["dse"])
    a_comb, a_comb_se = _combine(a_dir, a_dir_se, a_cpl, a_cpl_se)
    return dict(H=H, K=K, P_CF=P_CF, ns=ns, dt=dt, b=b, cv_se=res["cv_se"],
                dmean=res["dmean"], dse=res["dse"],
                a_dir=a_dir, a_dir_se=a_dir_se, a_cpl=a_cpl, a_cpl_se=a_cpl_se,
                a_comb=a_comb, a_comb_se=a_comb_se,
                n_dir=nd, n_cpl=nc, mask_dir=mdir, cv_var_ratio=res["cv_var_ratio"])


def _report(r, label):
    print(f"\n=== {label}  (H={r['H']}, P_CF={r['P_CF']:.6f}, "
          f"CV var-ratio={r['cv_var_ratio']:.3f}) ===")
    print(f"  {'n':>5} {'dt':>9} {'b_n=|E[P]-CF|':>14} {'MC s.e.':>10} {'b/se':>7} {'used?':>6}")
    for i, n in enumerate(r["ns"]):
        used = "fit" if r["mask_dir"][i] else "noise"
        print(f"  {n:>5} {r['dt'][i]:>9.5f} {r['b'][i]:>14.3e} {r['cv_se'][i]:>10.2e} "
              f"{r['b'][i]/r['cv_se'][i]:>7.1f} {used:>6}")
    print(f"  coupled  Y_l = E[P_l - P_(l-1)]  (CRN, low-variance):")
    print(f"  {'n':>5} {'dt':>9} {'|Y_l|':>12} {'se':>10} {'Y/se':>7}")
    for i in range(len(r["dmean"])):
        ni, dti = r["ns"][i + 1], r["dt"][i + 1]
        print(f"  {ni:>5} {dti:>9.5f} {abs(r['dmean'][i]):>12.3e} {r['dse'][i]:>10.2e} "
              f"{abs(r['dmean'][i])/r['dse'][i]:>7.1f}")
    def _fmt(a, se):
        return f"{a:.3f} ± {1.96*se:.3f}" if not np.isnan(se) else f"{a:.3f} (CI n/a, <3 pts)"
    print(f"  alpha (absolute b_n) = {_fmt(r['a_dir'], r['a_dir_se'])}  [{r['n_dir']} bias-dom pts]")
    print(f"  alpha (coupled Y_l)  = {_fmt(r['a_cpl'], r['a_cpl_se'])}  [{r['n_cpl']} bias-dom pts]")
    if not (np.isnan(r["a_dir"]) or np.isnan(r["a_cpl"])):
        gap = abs(r["a_dir"] - r["a_cpl"])
        comb = 1.96 * np.sqrt(np.nan_to_num(r["a_dir_se"]) ** 2 + np.nan_to_num(r["a_cpl_se"]) ** 2)
        verdict = "AGREE" if (comb == 0 or gap <= comb) else "DISAGREE"
        print(f"  estimator agreement: |Δα|={gap:.3f} vs combined 95% CI {comb:.3f} -> {verdict}")
    print(f"  alpha (prec-weighted)= {_fmt(r['a_comb'], r['a_comb_se'])}"
          f"  -> {classify(r['a_comb'], r['a_comb_se'], r['H'])}")


def plot_weak_order(results, path="output/layer4_weak_order.png"):
    """Two-panel diagnostic: (L) log-log priced bias vs Δt per H with the fitted
    slope; (R) α(H) with the α=H strong-order line and the α=1 classical line.
    Regenerated by `python layer4_convergence.py --sweep`."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Hs = sorted(results)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(Hs)))
    for H, c in zip(Hs, colors):
        r = results[H]; dt, b, m = r["dt"], r["b"], r["mask_dir"]
        axL.errorbar(dt, b, yerr=r["cv_se"], fmt="o", ms=3, color=c, alpha=0.35)
        axL.plot(dt[m], b[m], "o", ms=6, color=c,
                 label=f"H={H}  α={r['a_comb']:.2f}")
        if m.sum() >= 2:                                 # fitted slope over bias window
            xs = dt[m]
            axL.plot(xs, b[m][0] * (xs / xs[0]) ** r["a_comb"], "-", color=c, lw=1.2)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("Δt = T/n"); axL.set_ylabel("|E[P] − P_CF|  (priced bias)")
    axL.set_title("Weak-order bias vs Δt  (OTM call, ν=0.20)")
    axL.legend(fontsize=8); axL.grid(True, which="both", alpha=0.2)
    Harr = np.array(Hs)
    acomb = np.array([results[H]["a_comb"] for H in Hs])
    aerr = np.array([1.96 * np.nan_to_num(results[H]["a_comb_se"]) for H in Hs])
    axR.errorbar(Harr, acomb, yerr=aerr, fmt="s-", color="C3", capsize=4,
                 label="measured α (prec-weighted)")
    xs = np.linspace(0, max(Hs) * 1.1, 50)
    axR.plot(xs, xs, "k--", lw=1, label="α = H  (strong order)")
    axR.axhline(1.0, color="gray", ls=":", lw=1, label="α = 1  (classical weak)")
    axR.set_xlabel("H (Hurst)"); axR.set_ylabel("weak order α")
    axR.set_title("α(H): weak ≫ strong, penalty as H→0")
    axR.set_ylim(0, 1.3); axR.legend(fontsize=8); axR.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


def plot_alpha_only(results, path="output/layer4_weak_order_alpha.png"):
    """Single-panel α(H) — the RIGHT panel of plot_weak_order (measured α with the α=H
    strong-order and α=1 classical reference lines), matching the weak-order paper's
    Figure 1 caption (a clean α-vs-H panel). Same style/colours as plot_weak_order."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Hs = sorted(results)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    Harr = np.array(Hs)
    acomb = np.array([results[H]["a_comb"] for H in Hs])
    aerr = np.array([1.96 * np.nan_to_num(results[H]["a_comb_se"]) for H in Hs])
    ax.errorbar(Harr, acomb, yerr=aerr, fmt="s-", color="C3", capsize=4,
                label="measured α (prec-weighted)")
    xs = np.linspace(0, max(Hs) * 1.1, 50)
    ax.plot(xs, xs, "k--", lw=1, label="α = H  (strong order)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="α = 1  (classical weak)")
    ax.set_xlabel("H (Hurst)"); ax.set_ylabel("weak order α")
    ax.set_title("α(H): measured weak order vs Hurst  (OTM call, ν=0.20)")
    ax.set_ylim(0, 1.3); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  single-panel figure -> {path}")


def _save_sweep_json(results, path="output/layer4_weak_order_results.json"):
    """Persist the scalar α(H) results so the figure is reproducible from data."""
    import json, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ("P_CF", "a_dir", "a_dir_se", "a_cpl", "a_cpl_se", "a_comb", "a_comb_se")
    out = {str(H): {k: float(results[H][k]) for k in keys} for H in sorted(results)}
    json.dump(out, open(path, "w"), indent=2)
    print(f"  results -> {path}")


if __name__ == "__main__":
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="H=1/2 anchor, coarse-n pilot")
    ap.add_argument("--sweep", action="store_true",
                    help="OTM H-sweep (D31 measurement) + regenerate the figure")
    a = ap.parse_args()
    p = dict(RH_PARAMS)                                  # brick-1 params: nu=0.20
    nu_F = 2 * p["kappa"] * p["theta"] / p["nu"] ** 2
    print(f"Feller index nu_F = 2*kappa*theta/nu^2 = {nu_F:.3f}  "
          f"(<1 -> Euler-type weak order ~ nu_F per arXiv:2106.10926)")
    if a.sweep:
        # The D31 measurement: OTM call, Romano–Touzi conditional MC, n=4..64.
        results = {}
        for H in (0.05, 0.10, 0.20):
            r = measure_alpha(H, p, n0=4, L=4, M=500000, K=110.0, N_riccati=2000,
                              seed=11, conditional=True)
            _report(r, f"SWEEP — H={H} OTM K=110 (conditional MC)")
            results[H] = r
        plot_weak_order(results)
        plot_alpha_only(results)
        _save_sweep_json(results)
    elif a.pilot:
        r = measure_alpha(0.5, p, n0=4, L=5, M=120000, K=100.0, N_riccati=2000, seed=7)
        _report(r, "PILOT — H=1/2 anchor (coarse-n, bias-resolving)")
    else:
        r = measure_alpha(0.5, p, n0=16, L=5, M=300000, K=100.0, N_riccati=2000, seed=7)
        _report(r, "STAGE 1 — H=1/2 anchor")
