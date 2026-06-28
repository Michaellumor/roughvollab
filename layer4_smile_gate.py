"""
layer4_smile_gate.py — OTM-smile gate (SPX prerequisite): validate the lifted
simulator's high-ν OTM IMPLIED-VOL smile vs the CF known-answer at H≈0.10, BEFORE
any market calibration. Closes the gap thread-2 surfaced (4c validated ATM only;
the H=0.05 corner showed ~20% OTM tail-amplification — does it reach the SPX-
relevant H=0.10?).

Efficiency: the Romano–Touzi conditional MC makes the strike sweep ~free — one
lifted run (n, M) gives per-path (S_eff, sig_eff); the lifted price at ANY strike
is the analytic mean of bs_call(S_eff, K, sig_eff). CF smile = gil_pelaez across
strikes. Compare in IV space (the calibration-relevant metric); report only where
the BS inversion is reliable (IV s.e. = price_se/vega below tolerance).
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rough_heston import PARAMS
from rough_heston_lifted import _lifted_from_increments, lifted_setup
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_iv, bs_vega
from layer4_convergence import _bs_call_vec

_CF = ("V0", "kappa", "theta", "nu", "rho")
VUS = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)   # log-moneyness in vol-units


def _cf(p, H, N_riccati):
    return lambda u: rough_heston_cf(u, p["T"], H=H, N_riccati=N_riccati,
                                     **{k: p[k] for k in _CF})


def smile_gate(H=0.10, nu_list=(0.30, 0.40), N=150, n=256, M=100000, N_riccati=4000,
               seed=7, vus=VUS, iv_se_tol=0.001, match_tol=0.005, plot=True):
    """Lifted vs CF implied-vol smile at H, high ν. iv_se_tol, match_tol in vol
    (decimal: 0.001 = 0.1 vol-pt). Returns {nu: (vus, iv_cf, iv_lift, diff, reliable)}."""
    S0, r, T = PARAMS["S0"], PARAMS["r"], PARAMS["T"]
    print(f"OTM-SMILE GATE — lifted vs CF implied-vol smile | H={H} N={N} n={n} M={M} "
          f"N_riccati={N_riccati}")
    out = {}
    for nu in nu_list:
        p = dict(PARAMS, nu=nu)
        cf = _cf(p, H, N_riccati)
        sig_atm = bs_iv(gil_pelaez_call(cf, S0, S0, T, r), S0, S0, T, r)   # vol-unit scale
        Ks = np.array([S0 * np.exp(vu * sig_atm) for vu in vus])

        # ---- ONE lifted conditional-MC run -> per-path (S_eff, sig_eff) ----
        g, w = lifted_setup(H, N, T=T)
        rng = np.random.default_rng(seed); dt = T / n
        dWV = rng.standard_normal((M, n)) * np.sqrt(dt)
        dWp = rng.standard_normal((M, n)) * np.sqrt(dt)
        _, V = _lifted_from_increments(dWV, dWp, n, H, p, g, w, "qe")
        Vl = np.maximum(V[:, :-1], 0.0)
        Mt = (np.sqrt(Vl) * dWV).sum(axis=1)
        It = Vl.sum(axis=1) * dt
        S_eff = S0 * np.exp(p["rho"] * Mt - 0.5 * p["rho"] ** 2 * It)
        sig_eff = np.sqrt(np.maximum((1.0 - p["rho"] ** 2) * It / T, 1e-300))

        print(f"\n  ν={nu}  σ_ATM(CF)={sig_atm:.4f}")
        print(f"  {'vu':>5} {'K':>7} {'IV_cf%':>7} {'IV_lift%':>8} {'diff(pp)':>9} "
              f"{'se(pp)':>7} {'reliable':>8}")
        iv_cf_a, iv_lift_a, diff_a, rel_a = [], [], [], []
        for vu, K in zip(vus, Ks):
            iv_cf = bs_iv(gil_pelaez_call(cf, S0, K, T, r), S0, K, T, r)
            px = _bs_call_vec(S_eff, K, T, r, sig_eff)
            m, se = px.mean(), px.std(ddof=1) / np.sqrt(M)
            iv_lift = bs_iv(m, S0, K, T, r)
            vg = bs_vega(S0, K, T, r, iv_lift) if np.isfinite(iv_lift) else np.nan
            iv_se = se / vg if (np.isfinite(vg) and vg > 1e-9) else np.nan
            diff = (iv_lift - iv_cf) * 100 if (np.isfinite(iv_lift) and np.isfinite(iv_cf)) else np.nan
            ok = bool(np.isfinite(iv_se) and iv_se < iv_se_tol)
            iv_cf_a.append(iv_cf); iv_lift_a.append(iv_lift); diff_a.append(diff); rel_a.append(ok)
            print(f"  {vu:>5.1f} {K:>7.2f} {iv_cf*100:>7.2f} {iv_lift*100:>8.2f} {diff:>9.3f} "
                  f"{iv_se*100 if np.isfinite(iv_se) else float('nan'):>7.3f} {'Y' if ok else '-':>8}")
        diff_a = np.array(diff_a); rel_a = np.array(rel_a); vus_a = np.array(vus)
        out[nu] = (vus_a, np.array(iv_cf_a), np.array(iv_lift_a), diff_a, rel_a)
        _classify(nu, vus_a, diff_a, rel_a, match_tol)
    if plot:
        _plot(out, H)
    return out


def _classify(nu, vus, diff, reliable, match_tol):
    thr = match_tol * 100                                    # vol points
    atm = abs(diff[np.argmin(np.abs(vus))])
    rel = reliable & np.isfinite(diff)
    if not rel.any():
        print(f"     -> ν={nu}: INCONCLUSIVE (no reliable strikes)"); return
    maxd = np.abs(diff[rel]).max()
    if maxd < thr:
        v = f"(A) MATCH — max|IV diff|={maxd:.2f}pp < {thr:.1f}pp over reliable range; OTM pricing validated"
    else:
        worst_vu = vus[rel][np.argmax(np.abs(diff[rel]))]
        deepest = np.abs(vus[rel]).max()
        if abs(worst_vu) >= deepest - 1e-9:
            v = (f"(C) DEEP-OTM LIMIT — max|diff|={maxd:.2f}pp at the deepest reliable vu={worst_vu:+.1f} "
                 f"(grows toward OTM); calibrate to |vu|<{abs(worst_vu):.1f}")
        else:
            v = f"(B) WING DIVERGENCE — max|diff|={maxd:.2f}pp at vu={worst_vu:+.1f} (ATM clean)"
    print(f"     -> ν={nu}: ATM diff={atm:.3f}pp (anchor ~0) | {v}")


def _plot(out, H, path="output/layer4_smile_gate.png"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nus = sorted(out)
    fig, axes = plt.subplots(1, len(nus), figsize=(5.5 * len(nus), 4.4), squeeze=False)
    for ax, nu in zip(axes[0], nus):
        vus, iv_cf, iv_lift, diff, rel = out[nu]
        ax.plot(vus, iv_cf * 100, "k-o", ms=4, label="CF (truth)")
        ax.plot(vus, iv_lift * 100, "C3--s", ms=4, label="lifted")
        ax.plot(vus[rel], iv_lift[rel] * 100, "C3s", ms=7, label="lifted (reliable)")
        ax.set_xlabel("log-moneyness (vol-units)"); ax.set_ylabel("implied vol (%)")
        ax.set_title(f"H={H}, ν={nu}"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle("OTM-smile gate — lifted vs CF implied-vol smile")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        smile_gate(nu_list=(0.40,), N=80, n=128, M=30000, plot=False)
    else:
        smile_gate()
