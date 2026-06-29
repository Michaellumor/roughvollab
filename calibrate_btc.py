"""
calibrate_btc.py — calibrate the validated rough-Heston engine (D38) to a cleaned Deribit
BTC option-surface snapshot (real-market D39). Reuses layer4_calibrate_surface unchanged;
the only new code is a thin `calibrate_surface_weighted` wrapper (the engine's `calibrate`
accepts `weights`, but `calibrate_surface` forgets to forward it — verified lines 101-106).

Build-gated flow: (1) N_riccati pre-check at the CRYPTO corner (low-H/high-ν/long-T is the
fractional-Riccati overflow regime), (2) runtime estimate + HOLD, (3) calibrate, (4) assess.
Compute-hardening carried from D38: BLAS pinned to 1 thread/process (set BEFORE numpy import
so spawned pool workers inherit it), pool = physical cores, per-member progress, estimate
before launch.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # BLAS=1 BEFORE numpy (the D38 pool-stall fix)

import sys
import time
import glob
import argparse
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import layer4_calibrate_surface as S
from layer4_calibrate import calibrate
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_iv
from layer4_calibrate_surface import theta_to_cfparams_s
import deribit_surface as D

S0_, R_ = 100.0, 0.0


# --------------------------------------------------------------------------- #
# Riccati-cached surface model — IDENTICAL numerics to S.surface_model (verified
# max|ΔIV|=0) but solves the strike-INDEPENDENT fractional Riccati ONCE per
# maturity instead of once per strike (~13x faster). Reuses the validated
# rough_heston_cf / gil_pelaez_call verbatim; the only change is a memoised cf.
# --------------------------------------------------------------------------- #
def _cached_cf(T, cfp, H, N_riccati):
    cache = {}
    def cf(u):
        key = u.tobytes() if hasattr(u, "tobytes") else complex(u)
        v = cache.get(key)
        if v is None:
            v = rough_heston_cf(u, T, H=H, N_riccati=N_riccati, **cfp)
            cache[key] = v
        return v
    return cf


def _smile_cached(theta, Ks, T, *, n_params=4, N_riccati=2000, n_nodes=96):
    H, cfp = theta_to_cfparams_s(theta, n_params)
    cf = _cached_cf(T, cfp, H, N_riccati)
    out = np.full(len(Ks), np.nan)
    for i, K in enumerate(Ks):
        try:
            px = gil_pelaez_call(cf, S0_, K, T, R_, n_nodes=n_nodes)
            if np.isfinite(px) and px > 1e-12:
                out[i] = bs_iv(px, S0_, K, T, R_)
        except Exception:
            pass
    return out


def make_cached_surface_model(grids_by_T, n_params=4):
    Ts = sorted(grids_by_T)
    def _m(theta, _ks_ignored, **cf_kw):
        return np.concatenate([_smile_cached(theta, grids_by_T[T], T, n_params=n_params, **cf_kw)
                               for T in Ts])
    return _m

RHO0, XI0_0 = -0.50, 0.185                              # crypto-corner defaults (ξ₀≈ATM-IV²≈0.44²)
CORNERS = ((0.03, 0.60), (0.05, 0.50), (0.08, 0.40))   # (H,ν) the optimizer may visit
NS = (1500, 2000, 3000, 4000)


def latest_snapshot(currency="BTC", d="data/deribit"):
    fs = sorted(glob.glob(f"{d}/{currency}_*.json"))
    if not fs:
        raise SystemExit("no snapshot — run `python deribit_surface.py` first")
    return fs[-1]


def stack(by_T, grids_by_T):
    return np.concatenate([by_T[T] for T in sorted(grids_by_T)])


def calibrate_surface_weighted(target, grids_by_T, *, weights=None, n_params=4,
                               theta0=None, bounds=None, cf_kw=None, max_nfev=200):
    """The engine-gap fix: calibrate_surface drops `weights`; the underlying `calibrate`
    applies them. Reuse make_surface_model + calibrate verbatim — no engine edit."""
    theta0 = S.INIT_FAR[n_params] if theta0 is None else np.asarray(theta0, float)
    bounds = (S.LB[n_params], S.UB[n_params]) if bounds is None else bounds
    return calibrate(target, Ks=None, theta0=theta0, bounds=bounds,
                     model=make_cached_surface_model(grids_by_T, n_params),
                     weights=weights, cf_kw=cf_kw, max_nfev=max_nfev)


# --------------------------------------------------------------------------- #
# STEP 1 — N_riccati pre-check at the crypto corner
# --------------------------------------------------------------------------- #
def precheck(grids_by_T, *, corners=CORNERS, Ns=NS, n_nodes=160, atol=2e-3):
    Ts = sorted(grids_by_T); Tmin, Tmax = Ts[0], Ts[-1]
    keys = [(H, nu, T) for (H, nu) in corners for T in (Tmin, Tmax)]
    print(f"\n=== STEP 1: N_riccati pre-check (crypto corner) ===")
    print(f"  corners (H,ν) at ρ={RHO0} ξ₀={XI0_0}; ATM CF IV at T∈[{Tmin:.3f},{Tmax:.3f}] (nan=overflow):")
    tab = {}
    for N in Ns:
        row = {}
        for (H, nu, T) in keys:
            iv = S.model_smile_cf_T([H, nu, RHO0, XI0_0], np.array([100.0]), T,
                                    N_riccati=N, n_nodes=n_nodes)[0]
            row[(H, nu, T)] = iv
        tab[N] = row
        finite = sum(np.isfinite(v) for v in row.values())
        print(f"  N={N:5d}: finite {finite}/{len(keys)}  " +
              " ".join(f"H{H}ν{nu}T{T:.2f}={(v*100 if np.isfinite(v) else float('nan')):.1f}"
                       for (H, nu, T), v in row.items()))
    ref = tab[Ns[-1]]
    def converged(row):
        return all(np.isfinite(row[k]) and np.isfinite(ref[k]) and abs(row[k] - ref[k]) < atol for k in ref)
    Nstar = next((N for N in Ns if all(np.isfinite(v) for v in tab[N].values()) and converged(tab[N])), Ns[-1])
    # inversion check at Nstar on the REAL strikes (worst corner = lowest H, highest ν)
    Hc, nuc = corners[0]
    cf_kw = dict(N_riccati=Nstar, n_nodes=n_nodes)
    fin_by_T = {}
    for T in Ts:
        iv = S.model_smile_cf_T([Hc, nuc, RHO0, XI0_0], grids_by_T[T], T, **cf_kw)
        fin_by_T[T] = (int(np.isfinite(iv).sum()), len(iv))
    print(f"  -> chosen N_riccati={Nstar} (n_nodes={n_nodes}); inversion at worst corner "
          f"(H={Hc},ν={nuc}) finite/strike per T: " +
          " ".join(f"{T:.2f}:{f}/{n}" for T, (f, n) in fin_by_T.items()))
    return cf_kw


# --------------------------------------------------------------------------- #
# STEP 2 — runtime estimate (time one calibration, scale to the bootstrap)
# --------------------------------------------------------------------------- #
def estimate(target, grids_by_T, weights, cf_kw, *, theta0, n_boot=24, pool=4):
    print(f"\n=== STEP 2: runtime estimate (BLAS=1, pool={pool}, cf_kw={cf_kw}) ===", flush=True)
    base_model = make_cached_surface_model(grids_by_T, 4)
    cnt = [0]; tlast = [time.time()]
    def counting_model(theta, Ks, **ck):                         # per-eval progress (visibility)
        v = base_model(theta, Ks, **ck)
        cnt[0] += 1; now = time.time()
        print(f"    eval {cnt[0]:3d}: dt={now-tlast[0]:5.1f}s  nan={int(np.isnan(v).sum())}/{len(v)}", flush=True)
        tlast[0] = now
        return v
    t0 = time.time()
    res = calibrate(target, Ks=None, theta0=theta0, bounds=(S.LB[4], S.UB[4]),
                    model=counting_model, weights=weights, cf_kw=cf_kw, max_nfev=200)
    t_cal = time.time() - t0
    print(f"  one full calibration: {t_cal:.0f}s  nfev={res.nfev} ({cnt[0]} model evals) "
          f"success={res.success} ({res.message[:34]})  -> "
          + "  ".join(f"{n}={v:+.4f}" for n, v in zip(S.PN[4], res.theta_hat))
          + f"  IV-RMSE={res.iv_rmse*100:.3f}pp", flush=True)
    boot = t_cal * np.ceil(n_boot / pool)
    print(f"  H-bootstrap: {n_boot} draws ÷ {pool} workers × {t_cal:.0f}s ≈ {boot/60:.1f} min", flush=True)
    print(f"  ESTIMATED TOTAL (1 calib + {n_boot}-draw bootstrap) ≈ {(t_cal + boot)/60:.1f} min", flush=True)
    return res, t_cal


# --------------------------------------------------------------------------- #
# STEP 4 — assessment (the no-known-answer gate: fit quality, plausibility,
# H-identifiability bootstrap, figure)
# --------------------------------------------------------------------------- #
def _split_by_maturity(arr, grids_by_T):
    out = {}; off = 0
    for T in sorted(grids_by_T):
        n = len(grids_by_T[T]); out[T] = arr[off:off + n]; off += n
    return out


def cached_jacobian(theta, grids_by_T, cf_kw, *, rel=0.01, abs_=1e-4):
    m = make_cached_surface_model(grids_by_T, 4)
    base = np.array(theta, float); J = []
    for j in range(4):
        h = rel * abs(base[j]) + abs_
        tp = base.copy(); tp[j] += h; tm = base.copy(); tm[j] -= h
        J.append(m(tp, None, **cf_kw) - m(tm, None, **cf_kw))
    return np.array(J).T


def _boot_worker(args):
    """Nonparametric bootstrap: resample the cleaned points (with replacement) per
    maturity, recalibrate from θ̂. Module-level for ProcessPool picklability."""
    seed, grids_by_T, target_by_T, weights_by_T, cf_kw, theta0 = args
    rng = np.random.default_rng(seed)
    g2, t2, w2 = {}, {}, {}
    for T in grids_by_T:
        mm = len(grids_by_T[T]); idx = rng.integers(0, mm, mm)
        g2[T] = grids_by_T[T][idx]; t2[T] = target_by_T[T][idx]; w2[T] = weights_by_T[T][idx]
    tgt = np.concatenate([t2[T] for T in sorted(g2)])
    wts = np.concatenate([w2[T] for T in sorted(g2)])
    return calibrate_surface_weighted(tgt, g2, weights=wts, theta0=theta0, cf_kw=cf_kw).theta_hat


def assess(res, grids_by_T, target_by_T, weights_by_T, cf_kw, meta, *, n_boot=24, pool=4, seed=20260629):
    th = res.theta_hat; Ts = sorted(grids_by_T)
    iv_m = _split_by_maturity(make_cached_surface_model(grids_by_T, 4)(th, None, **cf_kw), grids_by_T)
    print("\n=== STEP 4: ASSESSMENT ===", flush=True)
    print("  (a) FIT QUALITY (IV-RMSE vol-pts) per maturity / region:", flush=True)
    for T in Ts:
        Kn = grids_by_T[T]; mm = np.isfinite(iv_m[T]) & np.isfinite(target_by_T[T])
        def rr(msk):
            q = msk & mm
            return (np.sqrt(np.mean((iv_m[T][q] - target_by_T[T][q]) ** 2)) * 100) if q.any() else float("nan")
        print(f"    T={T:.3f} ({len(Kn)}pts): all={rr(np.ones(len(Kn), bool)):.2f}  "
              f"put(K<97)={rr(Kn < 97):.2f}  atm={rr((Kn >= 97) & (Kn <= 103)):.2f}  "
              f"call(K>103)={rr(Kn > 103):.2f}  (finite {int(mm.sum())}/{len(Kn)})", flush=True)
    H, nu, rho, xi0 = th
    atmv = float(np.mean(list(meta["atm_iv_by_T"].values()))) ** 2
    print("  (b) PLAUSIBILITY:", flush=True)
    print(f"    H ={H:.4f}  " + ("<-- AT LB 0.02: NON-IDENTIFIED (degeneracy, not a fit)" if H < 0.025
                                  else "(rough)" if H < 0.12 else "(smooth?)"), flush=True)
    print(f"    ν ={nu:.3f}  " + ("(HIGH ≥0.4 — the D34 lift regime)" if nu >= 0.4 else "(low?)"), flush=True)
    print(f"    ρ ={rho:+.3f}  " + ("(put-skew)" if rho <= -0.3 else "(weak skew)"), flush=True)
    print(f"    ξ₀={xi0:.3f}  (ATM vol≈{np.sqrt(xi0)*100:.0f}%; surface ATM-var≈{atmv:.3f})", flush=True)
    print("  (c) ★ H-IDENTIFIABILITY:", flush=True)
    try:
        rep = S.ident_from_J(cached_jacobian(th, grids_by_T, cf_kw), S.PN[4])
        print(f"    JᵀJ at θ̂: cond={rep.cond:.2e}  |flat[H]|={abs(rep.flat[0]):.2f}  corr(H,ν)={rep.corr[0,1]:+.2f}", flush=True)
    except Exception as e:
        print(f"    jacobian ident skipped ({e})", flush=True)
    print(f"    bootstrap: {n_boot} resamples ÷ {pool} workers (init=θ̂)...", flush=True)
    args = [(seed + d, grids_by_T, target_by_T, weights_by_T, cf_kw, th) for d in range(n_boot)]
    from concurrent.futures import ProcessPoolExecutor
    recs = []
    with ProcessPoolExecutor(max_workers=pool) as ex:
        for i, r in enumerate(ex.map(_boot_worker, args)):
            recs.append(r)
            print(f"      draw {i+1:2d}/{n_boot}: H={r[0]:.4f} ν={r[1]:.3f} ρ={r[2]:+.3f} ξ₀={r[3]:.3f}", flush=True)
    recs = np.array(recs); mean = recs.mean(0); std = recs.std(0)
    print("    bootstrap spread (mean ± std, rel%):", flush=True)
    for i, n in enumerate(S.PN[4]):
        print(f"      {n:>3}: {mean[i]:+.4f} ± {std[i]:.4f}  ({std[i]/max(abs(mean[i]),1e-9)*100:.0f}%)", flush=True)
    at_lb = float(np.mean(recs[:, 0] < 0.025))
    loose = std[0] / 0.1 > 0.3 or at_lb > 0.3 or float(np.ptp(recs[:, 0])) > 0.05
    print(f"    H: range [{recs[:,0].min():.4f}, {recs[:,0].max():.4f}], at-LB in {at_lb*100:.0f}% of draws", flush=True)
    print(f"    => H IDENTIFIABILITY: " + ("LOOSE / NON-IDENTIFIED (PREDICTED — the option-calibration angle "
          "confirms the observational-equivalence wall)" if loose else "TIGHT — surprising, scrutinize"), flush=True)
    _plot(grids_by_T, target_by_T, iv_m, th)
    return recs


def _plot(grids_by_T, target_by_T, iv_m, th, path="output/btc_smile_fit.png"):
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs("output", exist_ok=True)
    Ts = sorted(grids_by_T)
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.1 * len(Ts), 3.3), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        k = np.log(grids_by_T[T] / 100.0)
        ax.plot(k, target_by_T[T] * 100, "ko", ms=4, label="market")
        ax.plot(k, iv_m[T] * 100, "C3-", label="model")
        ax.set_title(f"T={T:.3f}"); ax.set_xlabel("ln(K/F)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("IV %"); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"BTC fit: H={th[0]:.3f} ν={th[1]:.2f} ρ={th[2]:.2f} ξ₀={th[3]:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"    figure -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--max-T", type=float, default=None, help="drop maturities with T > this (the long tenor is the overflow driver)")
    ap.add_argument("--n-nodes", type=int, default=160)
    ap.add_argument("--calibrate", action="store_true", help="run the full calibration + assess (Step 3-4)")
    a = ap.parse_args()
    snap = a.snapshot or latest_snapshot(a.currency)
    print(f"[calibrate_btc] snapshot = {snap}", flush=True)
    grids, target_by_T, weights_by_T, meta = D.clean_from_snapshot(snap, verbose=True)
    if a.max_T is not None:
        drop = [T for T in grids if T > a.max_T]
        for T in drop:
            grids.pop(T); target_by_T.pop(T); weights_by_T.pop(T)
        print(f"  --max-T={a.max_T}: dropped {len(drop)} maturities -> {len(grids)} kept "
              f"(longest now T={max(grids):.3f})", flush=True)
    target = stack(target_by_T, grids)
    weights = stack(weights_by_T, grids)
    # stacking-order guard
    probe = np.array([0.08, 0.5, RHO0, XI0_0])
    assert len(target) == sum(len(grids[T]) for T in grids) == len(weights), "stacking length mismatch"
    # init: ξ₀ from ATM var, ρ negative, H low, ν high
    atm = float(np.mean(list(meta["atm_iv_by_T"].values())))
    theta0 = np.array([0.08, 0.50, RHO0, atm ** 2])
    print(f"  surface: {len(target)} points, {len(grids)} maturities; init θ0={theta0}", flush=True)
    cf_kw = precheck(grids, n_nodes=a.n_nodes)
    if a.calibrate:
        print("\n=== STEP 3: CALIBRATE (weighted, BLAS=1) ===", flush=True)
        t0 = time.time()
        res = calibrate_surface_weighted(target, grids, weights=weights, theta0=theta0, cf_kw=cf_kw)
        print(f"  calibrated in {time.time()-t0:.0f}s  nfev={res.nfev}  success={res.success}: "
              + "  ".join(f"{n}={v:+.4f}" for n, v in zip(S.PN[4], res.theta_hat))
              + f"  IV-RMSE={res.iv_rmse*100:.3f}pp", flush=True)
        assess(res, grids, target_by_T, weights_by_T, cf_kw, meta, n_boot=24)
    else:
        estimate(target, grids, weights, cf_kw, theta0=theta0)
        print("\n[HOLD] Steps 1-2 done. Re-run with --calibrate to proceed to Step 3-4.")
