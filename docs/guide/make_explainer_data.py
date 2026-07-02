"""
make_explainer_data.py -- regenerate the baked-in DATA for explainer.html
=========================================================================
The interactive study explainer (docs/guide/explainer.html) embeds all of
its demo data as a JS constant so it works offline from file:// with zero
external requests. This script regenerates that data from the repo's
TRUSTED modules only:

  - roughvol_core.py        (kappa=0 Volterra engine, 18 tests)
  - rough_heston_cf.py      (El Euch-Rosenbaum CF + Gil-Pelaez, 23 tests)
  - layer1c_roughness_audit.py  (GJR estimator + RV proxy, Rung-0 gated)

It deliberately does NOT import layer1_rough_vol.py (known issue L1-1).

Everything is seeded; re-running reproduces the same data. Kept in the repo
so the explainer's numbers are regenerable (the D44 lesson: keep the driver).

Usage (from repo root):
    python docs/guide/make_explainer_data.py
Writes docs/guide/explainer_data.js and, if docs/guide/explainer.html
exists with /*DATA_START*/ ... /*DATA_END*/ markers, splices the fresh
data in between them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from roughvol_core import (volterra_process, volterra_weights,          # noqa: E402
                           rough_bergomi_paths)
from rough_heston_cf import rough_heston_cf, gil_pelaez_call, bs_iv     # noqa: E402
from layer1c_roughness_audit import gjr_hurst, realized_log_variance    # noqa: E402

SEED = 42


def _round_sig(x, sig=4):
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(x, (list, tuple)):
        return [_round_sig(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round_sig(v, sig) for k, v in x.items()}
    return x


def hurst_slider_data():
    """Demo 2: same Brownian noise pushed through the Volterra kernel at
    several H -- common random numbers make the path morph, not resample."""
    n, T = 512, 1.0
    H_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    rng = np.random.default_rng(SEED)
    dW = rng.standard_normal((1, n)) * np.sqrt(T / n)
    eta, xi0 = 1.5, 0.04

    paths, vols, var_t = {}, {}, {}
    for H in H_grid:
        W = volterra_process(dW, H, T)[0]
        _, v = volterra_weights(n, H, T)
        sigma = np.sqrt(xi0 * np.exp(eta * W - 0.5 * eta**2 * v))
        key = f"{H:.2f}"
        paths[key] = W.tolist()
        vols[key] = (100.0 * sigma).tolist()      # vol in percent
        var_t[key] = v.tolist()                   # discrete Var(W~_t)
    t = np.linspace(T / n, T, n)
    return dict(H_grid=H_grid, t=t.tolist(), W=paths, vol_pct=vols,
                var_t=var_t, eta=eta, xi0=xi0)


def proxy_artefact_data():
    """Demo 3: the Cont-Das mirage (Layer 1c Rung 1). Prices generated with
    KNOWN H; the realized-variance proxy is fed to the repo's GJR estimator.
    Smooth truth (H=0.5) reads rough through a noisy proxy window."""
    n_fine, n_paths, eta = 32768, 200, 1.5
    windows = [16, 32, 64, 128, 256]
    lags = np.array([1, 2, 4, 8, 16])
    out = {}
    for label, H_true in [("smooth", 0.50), ("rough", 0.10)]:
        rng = np.random.default_rng(SEED)
        _, S, V = rough_bergomi_paths(n_fine, H_true, n_paths, eta=eta,
                                      rng=rng)
        H_proxy, H_control = [], []
        for w in windows:
            log_rv = realized_log_variance(S, w)
            H_proxy.append(float(gjr_hurst(log_rv, lags=lags)))
            # control: estimator on the TRUE log-variance at the same grid
            H_control.append(float(gjr_hurst(np.log(V[:, w::w]), lags=lags)))
        # one example pair for the visual (window=32, first path, <=512 pts)
        if label == "smooth":
            log_rv32 = realized_log_variance(S[:1], 32)[0]
            true_lv = np.log(V[0, 32::32])
            m = min(len(log_rv32), len(true_lv), 512)
            example = dict(log_rv=log_rv32[:m].tolist(),
                           log_v_true=true_lv[:m].tolist())
        out[label] = dict(H_true=H_true, H_proxy=H_proxy,
                          H_control=H_control)
    out["windows"] = windows
    out["example_smooth_w32"] = example
    out["n_fine"] = n_fine
    out["n_paths"] = n_paths
    out["eta"] = eta
    return out


def smile_degeneracy_data():
    """Demo 5: rough-Heston IV smiles on an (H, nu) grid -- the flat valley.
    CF evaluations are memoised per u-array so each grid cell solves the
    fractional Riccati only twice (P1's shifted u and P2's u)."""
    H_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    nu_grid = [0.20, 0.28, 0.36, 0.44, 0.52, 0.60]
    V0 = theta = 0.04
    kappa, rho, T, r, S0 = 0.30, -0.70, 0.25, 0.0, 1.0
    N_riccati = 600
    ks = np.linspace(-0.4, 0.4, 15)
    Ks = np.exp(ks)

    smiles = []
    for H in H_grid:
        row = []
        for nu in nu_grid:
            cache = {}

            def cf(u, H=H, nu=nu, cache=cache):
                key = u.tobytes() if hasattr(u, "tobytes") else repr(u)
                if key not in cache:
                    cache[key] = rough_heston_cf(np.asarray(u, complex), T,
                                                 V0, kappa, theta, nu, rho,
                                                 H, N_riccati=N_riccati)
                return cache[key]

            ivs = []
            try:
                for K in Ks:
                    price = gil_pelaez_call(cf, S0, K, T, r)
                    iv = bs_iv(price, S0, K, T, r)
                    ivs.append(float(iv) if np.isfinite(iv) else None)
                if any(v is None for v in ivs):
                    row.append(None)
                else:
                    row.append([100.0 * v for v in ivs])   # IV in percent
            except Exception:
                row.append(None)                            # overflow cell
        smiles.append(row)
        print(f"  smile row H={H} done", file=sys.stderr)
    return dict(H_grid=H_grid, nu_grid=nu_grid, k=ks.tolist(),
                smiles=smiles, T=T, V0=V0, kappa=kappa, theta=theta,
                rho=rho, N_riccati=N_riccati)


def mlmc_data():
    """Demo 4: measured level-variance decay beta vs the 2H bound.
    Numbers are the committed Layer-1b results (README / ROADMAP table);
    no recomputation -- they are pinned measurements."""
    return dict(measured=[[0.05, 0.125], [0.10, 0.226],
                          [0.20, 0.422], [0.35, 0.721]],
                source="layer1b_mlmc_asian.py runs, ROADMAP 'Measured results'")


def main():
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse",
                                 "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    print("hurst slider ...", file=sys.stderr)
    hurst = hurst_slider_data()
    print("rv-proxy artefact (slow: 2x200 paths x 32768 steps) ...",
          file=sys.stderr)
    proxy = proxy_artefact_data()
    print("smile grid (slow: 36 CF cells) ...", file=sys.stderr)
    smile = smile_degeneracy_data()

    data = dict(meta=dict(seed=SEED, commit=commit,
                          generator="docs/guide/make_explainer_data.py"),
                hurst=hurst, proxy=proxy, smile=smile, mlmc=mlmc_data())
    js = "const DATA = " + json.dumps(_round_sig(data),
                                      separators=(",", ":")) + ";\n"

    out_js = HERE / "explainer_data.js"
    out_js.write_text(js, encoding="utf-8")
    print(f"wrote {out_js} ({len(js)/1e3:.0f} kB)", file=sys.stderr)

    html = HERE / "explainer.html"
    if html.exists():
        text = html.read_text(encoding="utf-8")
        a, b = "/*DATA_START*/", "/*DATA_END*/"
        if a in text and b in text:
            pre, rest = text.split(a, 1)
            _, post = rest.split(b, 1)
            html.write_text(pre + a + "\n" + js + b + post, encoding="utf-8")
            print("spliced DATA into explainer.html", file=sys.stderr)


if __name__ == "__main__":
    main()
