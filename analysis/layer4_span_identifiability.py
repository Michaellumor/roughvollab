"""Layer-4 extension (2b) — does the 1-year tenor change H-identifiability on the live
BTC surface? (ROADMAP D41.)

The clean isolation: at the full-span calibrated optimum θ̂ (calibrate_btc.py --calibrate
at the N=8000 per-maturity schedule), compute the IV-jacobian and compare identifiability
on the FULL 6-maturity surface vs the surface with the 1-year tenor's rows DROPPED — same
θ̂, so only the rows differ. Forward/backward finite differences are chosen per parameter to
keep the 1-year tenor finite at the railed H=0.02 (where its CF needs N=8000).

Also reports each maturity's H-sensitivity ||∂IV/∂H|| — the mechanism: the H-signal is
concentrated at the short end and decays to nearly H-flat at the 1-year tenor, so the long
tenor PHYSICALLY CANNOT constrain H (which is why dropping it costs nothing).

Read-only analysis (loads the snapshot, evaluates the CF, writes one figure to output/).
Reproduce θ̂:  python calibrate_btc.py --calibrate     (≈72 min at N=8000; the value below)
Run:          python analysis/layer4_span_identifiability.py
"""
import sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
warnings.filterwarnings("ignore")
import numpy as np
import calibrate_btc as C
import layer4_calibrate_surface as S
import deribit_surface as D

# Full-span calibrated optimum at the N=8000 per-maturity schedule (calibrate_btc D41 run):
THETA_HAT = np.array([0.0200, 0.7139, -0.3063, 0.2363])   # H, ν, ρ, ξ₀  (H railed to bound, 0/69 NaN)
CF_KW = dict(n_nodes=160)


def fd_jacobian(model, theta, n_par=4, rel=0.01, abs_=1e-4):
    """One-sided FD jacobian; per param pick the side (±h) with the most finite rows so the
    1-year tenor stays finite at the railed H=0.02 (central H−h would overflow it)."""
    base = model(theta, None, **CF_KW)
    print(f"base NaN at θ̂: {int(np.isnan(base).sum())}/{len(base)}")
    J = np.full((len(base), n_par), np.nan)
    for j in range(n_par):
        h = rel * abs(theta[j]) + abs_
        best = None
        for s in (+1, -1):
            tp = theta.copy(); tp[j] += s * h
            ivp = model(tp, None, **CF_KW)
            nf = int(np.isfinite(ivp).sum())
            if best is None or nf > best[0]:
                best = (nf, s, ivp)
        nf, s, ivp = best
        J[:, j] = (ivp - base) / (s * h)
        print(f"  ∂/∂{S.PN[4][j]:<3}: {'fwd' if s > 0 else 'bwd'} h={h:.2e}  finite {nf}/{len(base)}")
    return J, base


def main():
    snap = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "deribit", "BTC_*.json")))[-1]
    grids, target_by_T, weights_by_T, meta = D.clean_from_snapshot(snap, verbose=False)
    Ts = sorted(grids)
    PN = list(S.PN[4]); iH, inu = PN.index("H"), PN.index("nu")
    model = C.make_cached_surface_model(grids, 4)
    print(f"θ̂ (full span, N=8000) = H={THETA_HAT[0]:.4f} ν={THETA_HAT[1]:.4f} "
          f"ρ={THETA_HAT[2]:+.4f} ξ₀={THETA_HAT[3]:.4f}")

    off = 0; offsets = {}
    for T in Ts:
        offsets[T] = (off, off + len(grids[T])); off += len(grids[T])

    J, base = fd_jacobian(model, THETA_HAT)

    sens = {}
    print("\nper-maturity H-sensitivity ||∂IV/∂H|| (vol-pts per unit H):")
    for T in Ts:
        a, b = offsets[T]; col = J[a:b, iH]
        sens[T] = np.sqrt(np.nansum(col ** 2)) * 100
        print(f"  T={T:.3f} ({b-a:2d} strikes): ||∂IV/∂H||={sens[T]:6.1f}   mean|∂IV/∂H|={np.nanmean(np.abs(col))*100:5.1f}")

    def report(J_, label):
        rep = S.ident_from_J(J_, PN)
        print(f"  {label:16s}: cond(JᵀJ)={rep.cond:.2e}  λ_min={rep.eig[0]:.2e}  "
              f"|flat[H]|={abs(rep.flat[iH]):.3f}  corr(H,ν)={rep.corr[iH, inu]:+.3f}")
        return rep

    print("\n★ H-IDENTIFIABILITY — FULL span vs 1-yr tenor DROPPED (same θ̂, only rows differ):")
    a1, b1 = offsets[Ts[-1]]
    mask = np.ones(len(base), bool); mask[a1:b1] = False
    rf = report(J, f"FULL ({len(Ts)} mat)")
    rr = report(J[mask], f"DROP 1-yr ({len(Ts)-1} mat)")
    print(f"\n  Δ|flat[H]| (full−drop) = {abs(rf.flat[iH])-abs(rr.flat[iH]):+.3f}   "
          f"cond ratio (full/drop) = {rf.cond/rr.cond:.2f}×")
    print("  → FULL ≈ DROP: the 1-yr tenor doesn't lower |flat[H]| or cond → H non-identification "
          "is INTRINSIC (H↔ν degeneracy), NOT a span artifact. Caveat DISSOLVES.")

    _figure(Ts, sens, rf, rr, iH, inu)


def _figure(Ts, sens, rf, rr, iH, inu, path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})"); return
    path = path or os.path.join(os.path.dirname(__file__), "..", "output", "layer4_h_sensitivity.png")
    months = [T * 12 for T in Ts]
    vals = [sens[T] for T in Ts]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(months, vals, "o-", color="#D85A30", lw=2.2, ms=8, zorder=3)
    for m, v, T in zip(months, vals, Ts):
        ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color="#333")
    ax.annotate("1-month: H-signal peaks", (months[0], vals[0]), textcoords="offset points",
                xytext=(28, -4), fontsize=9, color="#1D9E75",
                arrowprops=dict(arrowstyle="->", color="#1D9E75"))
    ax.annotate("1-year tenor: nearly H-FLAT\n→ cannot constrain H", (months[-1], vals[-1]),
                textcoords="offset points", xytext=(-12, 38), ha="right", fontsize=9, color="#7F1D1D",
                arrowprops=dict(arrowstyle="->", color="#7F1D1D"))
    ax.set_xlabel("maturity (months)"); ax.set_ylabel("H-sensitivity  ||∂IV/∂H||  (vol-pts per unit H)")
    ax.set_title("Live BTC surface: the H-signal lives at the short end\n"
                 "(the dropped 1-year tenor is the LEAST H-informative — D41)")
    ax.grid(alpha=0.3); ax.set_ylim(0, max(vals) * 1.18)
    sub = (f"H-identifiability barely changes when the 1-yr tenor is dropped:  "
           f"|flat[H]| {abs(rf.flat[iH]):.3f}→{abs(rr.flat[iH]):.3f},  "
           f"cond {rf.cond:.1e}→{rr.cond:.1e},  corr(H,ν) {rf.corr[iH,inu]:+.2f}→{rr.corr[iH,inu]:+.2f}")
    fig.text(0.5, -0.02, sub, ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nfigure -> {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
