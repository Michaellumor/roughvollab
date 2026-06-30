"""Layer-4 — cross-market identifiability + fit diagnostics (D41 BTC, D42 ETH).

At a calibrated optimum θ̂ (calibrate_btc.py --calibrate), report — for a live Deribit
surface — the things the calibration paper needs:
  · H-IDENTIFIABILITY: jacobian cond(JᵀJ), |flat[H]|, corr(H,ν) — is H pinned or degenerate?
  · the MECHANISM: per-maturity H-sensitivity ||∂IV/∂H|| (does the H-signal live at the short
    end and decay to ~flat at the long tenor, per D41?);
  · FIT + PUT-TAIL: per-maturity IV-RMSE split put/atm/call, and the SIGNED deep-put residual
    mean(model−market) — does rough-Heston UNDER-produce the crash-fear put wing?
  · --span-compare (BTC/D41): full vs 1-year-tenor-dropped ident at the same θ̂.

FD per parameter picks the side (±h) with the most finite rows so the 1-year tenor stays finite
at the railed H=0.02 (central H−h would overflow it). Read-only; writes two figures to output/.

Reproduce θ̂:  python calibrate_btc.py --currency {BTC,ETH} --calibrate   (≈72-94 min at N=8000)
Run:          python analysis/layer4_span_identifiability.py --currency ETH
"""
import argparse, sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
warnings.filterwarnings("ignore")
import numpy as np
import calibrate_btc as C
import layer4_calibrate_surface as S
import deribit_surface as D

# documented calibrated optima at the N=8000 per-maturity schedule (calibrate_btc --calibrate):
THETA = {
    "BTC": [0.0200, 0.7139, -0.3063, 0.2363],   # D41 (full-span; H railed, 0/69 NaN)
    "ETH": [0.0200, 0.7129, -0.3013, 0.3973],   # D42 (full-span; H railed, 0/46 NaN)
}
CF_KW = dict(n_nodes=160)
DEEP_PUT_K = 90.0          # K_norm < 90  →  >10% OTM put (the crash-fear wing)


def fd_jacobian(model, theta, n_par=4):
    base = model(theta, None, **CF_KW)
    print(f"  base NaN at θ̂: {int(np.isnan(base).sum())}/{len(base)}")
    J = np.full((len(base), n_par), np.nan)
    for j in range(n_par):
        h = 0.01 * abs(theta[j]) + 1e-4
        best = None
        for s in (+1, -1):
            tp = theta.copy(); tp[j] += s * h
            ivp = model(tp, None, **CF_KW)
            nf = int(np.isfinite(ivp).sum())
            if best is None or nf > best[0]:
                best = (nf, s, ivp)
        nf, s, ivp = best
        J[:, j] = (ivp - base) / (s * h)
        print(f"    ∂/∂{S.PN[4][j]:<3}: {'fwd' if s > 0 else 'bwd'}  finite {nf}/{len(base)}")
    return J, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--theta", default=None, help="H,nu,rho,xi0 (overrides the documented θ̂)")
    ap.add_argument("--span-compare", action="store_true", help="D41 BTC: full vs 1-yr-dropped ident")
    a = ap.parse_args()
    cur = a.currency.upper()
    theta = np.array([float(x) for x in a.theta.split(",")] if a.theta else THETA[cur], float)
    snap = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "deribit", f"{cur}_*.json")))[-1]
    grids, target_by_T, weights_by_T, meta = D.clean_from_snapshot(snap, verbose=False)
    Ts = sorted(grids)
    PN = list(S.PN[4]); iH, inu = PN.index("H"), PN.index("nu")
    model = C.make_cached_surface_model(grids, 4)
    off = 0; offs = {}
    for T in Ts:
        offs[T] = (off, off + len(grids[T])); off += len(grids[T])
    print(f"=== {cur}  θ̂ = H={theta[0]:.4f} ν={theta[1]:.4f} ρ={theta[2]:+.4f} ξ₀={theta[3]:.4f} "
          f"({len(target_by_T and [x for T in Ts for x in grids[T]])} pts, {len(Ts)} maturities) ===")

    print("\n[1] H-IDENTIFIABILITY (jacobian at θ̂):")
    J, base = fd_jacobian(model, theta)
    rep = S.ident_from_J(J, PN)
    print(f"    cond(JᵀJ)={rep.cond:.2e}  λ_min={rep.eig[0]:.2e}  |flat[H]|={abs(rep.flat[iH]):.3f}  "
          f"corr(H,ν)={rep.corr[iH, inu]:+.3f}")

    print("\n[2] MECHANISM — per-maturity H-sensitivity ||∂IV/∂H|| (vol-pts per unit H):")
    sens = {}
    for T in Ts:
        a0, b0 = offs[T]; col = J[a0:b0, iH]
        sens[T] = float(np.sqrt(np.nansum(col ** 2)) * 100)
        print(f"    T={T:.3f} ({b0-a0:2d}): ||∂IV/∂H||={sens[T]:6.1f}")

    print("\n[3] FIT + PUT-TAIL (model vs market at θ̂; signed = model−market, − ⇒ UNDER-produces):")
    iv_m = {T: base[offs[T][0]:offs[T][1]] for T in Ts}
    all_dp = []
    for T in Ts:
        K = grids[T]; mkt = target_by_T[T]; mdl = iv_m[T]; m = np.isfinite(mdl) & np.isfinite(mkt)
        def rmse(msk):
            q = msk & m
            return float(np.sqrt(np.mean((mdl[q] - mkt[q]) ** 2)) * 100) if q.any() else float("nan")
        dp = (K < DEEP_PUT_K) & m
        sdp = float(np.mean((mdl[dp] - mkt[dp])) * 100) if dp.any() else float("nan")
        all_dp += list((mdl[dp] - mkt[dp]) * 100)
        print(f"    T={T:.3f}: RMSE all={rmse(np.ones(len(K), bool)):.2f} put={rmse(K<97):.2f} "
              f"atm={rmse((K>=97)&(K<=103)):.2f} call={rmse(K>103):.2f} | deep-put(K<90) signed={sdp:+.2f} (n={int(dp.sum())})")
    overall_sdp = float(np.mean(all_dp)) if all_dp else float("nan")
    print(f"    → OVERALL deep-put signed residual = {overall_sdp:+.2f} vol-pts "
          f"({'UNDER-produces the crash tail' if overall_sdp < -0.2 else 'no systematic undershoot'})")

    _fig_hsens(cur, Ts, sens, rep, iH, inu)
    _fig_fit(cur, Ts, grids, target_by_T, iv_m, theta)

    if a.span_compare:
        print("\n[4] SPAN (D41): full vs 1-yr-dropped (same θ̂):")
        a1, b1 = offs[Ts[-1]]; mask = np.ones(len(base), bool); mask[a1:b1] = False
        rf, rr = S.ident_from_J(J, PN), S.ident_from_J(J[mask], PN)
        print(f"    FULL: cond={rf.cond:.2e} |flat[H]|={abs(rf.flat[iH]):.3f} | DROP-1yr: cond={rr.cond:.2e} "
              f"|flat[H]|={abs(rr.flat[iH]):.3f} | Δ|flat[H]|={abs(rf.flat[iH])-abs(rr.flat[iH]):+.3f}")

    return dict(cur=cur, cond=rep.cond, flatH=abs(rep.flat[iH]), corrHnu=rep.corr[iH, inu],
                sens=sens, deep_put=overall_sdp)


def _fig_hsens(cur, Ts, sens, rep, iH, inu, path=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (h-sens figure skipped: {e})"); return
    path = path or os.path.join(os.path.dirname(__file__), "..", "output", f"{cur.lower()}_h_sensitivity.png")
    months = [T * 12 for T in Ts]; vals = [sens[T] for T in Ts]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(months, vals, "o-", color="#7F77DD" if cur == "ETH" else "#D85A30", lw=2.2, ms=8)
    for m, v in zip(months, vals):
        ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xlabel("maturity (months)"); ax.set_ylabel("H-sensitivity  ||∂IV/∂H||  (vol-pts/unit H)")
    ax.set_title(f"{cur}: H-signal vs maturity — short end peaks, long tenor ~flat\n"
                 f"(|flat[H]|={abs(rep.flat[iH]):.2f}, corr(H,ν)={rep.corr[iH,inu]:+.2f} — H↔ν degeneracy)")
    ax.grid(alpha=0.3); ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(path, dpi=140); print(f"  figure → {os.path.normpath(path)}")


def _fig_fit(cur, Ts, grids, target_by_T, iv_m, theta, path=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (fit figure skipped: {e})"); return
    path = path or os.path.join(os.path.dirname(__file__), "..", "output", f"{cur.lower()}_smile_fit.png")
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.0 * len(Ts), 3.2), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        k = np.log(grids[T] / 100.0)
        ax.plot(k, target_by_T[T] * 100, "ko", ms=4, label="market")
        ax.plot(k, iv_m[T] * 100, "C3-", label="model")
        ax.set_title(f"T={T:.3f}"); ax.set_xlabel("ln(K/F)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("IV %"); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{cur} fit: H={theta[0]:.3f} ν={theta[1]:.2f} ρ={theta[2]:.2f} ξ₀={theta[3]:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); print(f"  figure → {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
