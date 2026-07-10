"""
rough_kernel_soe.py — Layer 4 brick 4a: sum-of-exponentials (SOE) approximation
of the rough kernel, the foundation of the multifactor Markovian lift.

    K(t) = t^(H-1/2) / Gamma(H+1/2)   ~   K^N(t) = sum_i w_i exp(-gamma_i t)

Each exponential becomes a Markovian OU factor in the lifted simulator (brick 4b).
This brick is SIMULATION-FREE: it builds the SOE nodes/weights (two literature-
pinned constructions) and measures the kernel-approximation error vs the number of
factors N (Gate A). No Monte Carlo.

Spectral (Laplace) representation -- the foundation of every SOE rule, valid H in (0, 1/2)
[Abi Jaber-El Euch eq (1.3); Bayer-Breneis eq (1.3)]:

    K(t) = int_0^inf exp(-gamma t) mu(dgamma),
    mu(dgamma) = gamma^(-H-1/2) / (Gamma(H+1/2) Gamma(1/2-H)) dgamma.

Every SOE method is a quadrature of this integral.

Method A -- Abi Jaber & El Euch (arXiv:1801.10359; SIAM J. Fin. Math. 10(2), 2019).
    Moment-matched nodes/weights on a partition (closed form), uniform optimal mesh.
    Algebraic convergence ||K^n - K||_{2,T} <= C_H n^(-4H/5).

Method B -- Bayer & Breneis (arXiv:2108.05048; Quantitative Finance 23(1), 2023).
    Gauss-Legendre quadrature (level m) on geometric subintervals of the spectral
    integral + a gamma=0 tail node. Superpolynomial convergence ~ exp(-c sqrt(N)).

The kernel-approximation error is the relative L2[0,T] norm (the papers' norm; it
absorbs the integrable t->0 singularity), computed in CLOSED FORM via the lower
incomplete gamma function -- no singular numerical quadrature needed.
"""
import numpy as np
from scipy.special import gamma as _gamma, gammainc  # gammainc(a,x) = regularised lower P(a,x)


def kernel(t, H):
    """Analytic rough kernel K(t) = t^(H-1/2) / Gamma(H+1/2)."""
    return np.asarray(t, float) ** (H - 0.5) / _gamma(H + 0.5)


def _mu_norm(H):
    """Normalising constant of the spectral measure: 1/(Gamma(H+1/2)Gamma(1/2-H))."""
    return 1.0 / (_gamma(H + 0.5) * _gamma(0.5 - H))


# --------------------------------------------------------------------------- #
#  Method A: Abi Jaber-El Euch -- closed-form moment matching                   #
# --------------------------------------------------------------------------- #
def soe_ajee(H, N, T=1.0):
    """N-factor SOE via AJ-EE moment matching on the uniform optimal mesh.
    c_i = int_{eta_{i-1}}^{eta_i} mu,  gamma_i = (1/c_i) int gamma mu  (eq 3.6).
    Uniform mesh eta_i = i*pi_n with the optimal pi_n (eq 3.10).
    Returns (gammas[N], weights[N])."""
    a = H + 0.5                                  # alpha
    p, q = 1.0 - a, 2.0 - a                       # 1/2 - H, 3/2 - H  (both > 0)
    pi_n = (N ** (-0.2) / T) * (np.sqrt(10.0) * (1 - 2 * H) / (5 - 2 * H)) ** 0.4
    eta = np.arange(N + 1) * pi_n                 # eta_0 = 0 .. eta_N
    ep = eta ** p
    norm = _mu_norm(H) / (1.0 - a)                # = 1/((1-a) Gamma(a) Gamma(1-a))
    c = (ep[1:] - ep[:-1]) * norm                 # weights c_i
    g = ((1 - a) / (2 - a)) * (eta[1:] ** q - eta[:-1] ** q) / (ep[1:] - ep[:-1])
    return g, c


# --------------------------------------------------------------------------- #
#  Method B: Bayer-Breneis -- Gauss-Legendre on geometric subintervals          #
# --------------------------------------------------------------------------- #
def soe_bb(H, N, T=1.0, alpha=1.6, beta=0.4275, a=1.0, b=1.0):
    """~N-factor SOE via Gauss-Legendre quadrature of level m on n geometric
    subintervals of the spectral integral + a gamma=0 tail node (BB construction).
    Optimal split m ~ (beta/A) sqrt(N), n ~ (A/beta) sqrt(N) (so n*m ~ N); the
    exponential range spread (xi_0, xi_n ~ exp(+-c sqrt(N))) yields the
    superpolynomial rate exp(-2 alpha/A sqrt(N)). Both alpha=1.6 and beta=0.4275
    are empirically-tuned constants of the Bayer-Breneis SOE method
    (arXiv:2108.05048, sec. 4.2; full reference in the module header), not free
    parameters: alpha=1.6 gives the best realised rate across H in {0.05,0.1,0.2}
    for this folded-Gauss-Legendre variant, and beta=0.4275 sets the m/n
    (quadrature-level / interval) split. Returns (gammas, weights);
    total factors = n*m + 1."""
    cH = _mu_norm(H)
    aexp = H + 0.5
    A = np.sqrt(1.0 / H + 1.0 / (1.5 - H))        # A_H
    sN = np.sqrt(N)
    m = max(1, int(round((beta / A) * sN)))       # quadrature level per interval
    n = max(1, int(round((A / beta) * sN)))       # number of geometric subintervals
    xi0 = a * np.exp(-alpha * sN / ((1.5 - H) * A))
    xin = b * np.exp(alpha * sN / (H * A))
    edges = xi0 * (xin / xi0) ** (np.arange(n + 1) / n)
    xleg, wleg = np.polynomial.legendre.leggauss(m)            # nodes/weights on [-1,1]
    gammas = [0.0]                                             # zero / tail node ...
    weights = [cH / (0.5 - H) * xi0 ** (0.5 - H)]             # ... weight for [0, xi0]
    for j in range(n):
        lo, hi = edges[j], edges[j + 1]
        xk = 0.5 * (hi - lo) * xleg + 0.5 * (hi + lo)          # map to [lo, hi]
        wk = 0.5 * (hi - lo) * wleg                            # interval Jacobian
        gammas.extend(xk.tolist())
        weights.extend((wk * cH * xk ** (-aexp)).tolist())     # fold in c_H x^(-a)
    return np.array(gammas), np.array(weights)


def soe(H, N, method="ajee", T=1.0, **kw):
    """Dispatcher. H >= 1/2 is the degenerate (classical) case: K(t)=1 constant,
    reproduced exactly by a single gamma=0 mode -- the lift is unnecessary there."""
    if H >= 0.5:
        return np.array([0.0]), np.array([1.0])
    if method == "ajee":
        return soe_ajee(H, N, T)
    if method == "bb":
        return soe_bb(H, N, T, **kw)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
#  Error metrics (closed form -- Gate A)                                        #
# --------------------------------------------------------------------------- #
def _int_K_exp(H, g, T):
    """int_0^T K(t) exp(-g t) dt for each node g (g may be 0).
    = g^(-a) P(a, gT)  (a=H+1/2);  g=0 -> T^a/(a Gamma(a))."""
    a = H + 0.5
    g = np.asarray(g, float)
    out = np.empty_like(g)
    nz = g > 0
    out[nz] = g[nz] ** (-a) * gammainc(a, g[nz] * T)
    out[~nz] = T ** a / (a * _gamma(a))
    return out


def kernel_l2_metrics(H, g, w, T=1.0, dt=None):
    """Closed-form error metrics for K^N (nodes g, weights w) vs K on [0,T]:
      rel_l2   -- relative L2[0,T] error (primary; absorbs the t->0 singularity);
      mass_rel -- relative error of the integrated fast mass int_0^dt K (if dt given);
      linf_rel -- relative L-inf error on resolved lags [dt,T] (if dt given)."""
    a = H + 0.5
    g = np.asarray(g, float); w = np.asarray(w, float)
    Knorm2 = T ** (2 * H) / (2 * H * _gamma(a) ** 2)           # ||K||_{2,T}^2
    cross = float(w @ _int_K_exp(H, g, T))                     # int K K^N
    G = g[:, None] + g[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(G > 0, (1.0 - np.exp(-G * T)) / np.where(G > 0, G, 1.0), T)
    quad = float(w @ M @ w)                                    # int (K^N)^2
    rel_l2 = np.sqrt(max(Knorm2 - 2 * cross + quad, 0.0) / Knorm2)
    out = {"rel_l2": rel_l2, "N": int(g.size)}
    if dt is not None:
        massK = dt ** a / _gamma(a + 1.0)
        gg = np.where(g > 0, g, 1.0)
        massN = float(np.sum(np.where(g > 0, w * (1.0 - np.exp(-g * dt)) / gg, w * dt)))
        out["mass_rel"] = abs(massK - massN) / massK
        tg = np.logspace(np.log10(dt), np.log10(T), 400)
        Kt = kernel(tg, H)
        KN = (w[None, :] * np.exp(-np.outer(tg, g))).sum(axis=1)
        out["linf_rel"] = float(np.max(np.abs(KN - Kt) / Kt))
    return out


# --------------------------------------------------------------------------- #
#  Gate A driver                                                                #
# --------------------------------------------------------------------------- #
def _fit_rate(Ns, errs, kind):
    """Fit the empirical convergence rate. kind='alg': slope of log(err) vs log(N)
    (algebraic, AJ-EE expects -4H/5). kind='exp': slope of log(err) vs sqrt(N)
    (superpolynomial, BB expects -c)."""
    Ns, errs = np.asarray(Ns, float), np.asarray(errs, float)
    ok = errs > 0
    x = (np.log(Ns) if kind == "alg" else np.sqrt(Ns))[ok]
    y = np.log(errs[ok])
    if x.size < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def gate_a(H_list=(0.05, 0.10, 0.20), N_list=(16, 32, 64, 128, 256, 512),
           dt=1.0 / 128, T=1.0, target=1e-3, plot=True):
    """Gate A: kernel-error vs N for both methods, at each H. Prints tables, the
    fitted rate vs the pinned theoretical rate, the N_factors(target) curve, and
    the H=1/2 guard. Returns a results dict."""
    res = {}
    print(f"GATE A -- rough-kernel sum-of-exponentials | T={T} dt={dt:.5f} | "
          f"target rel-L2 <= {target:g}")
    print(f"spectral measure mu(dg) = g^(-H-1/2)/(Gamma(H+1/2)Gamma(1/2-H)) dg, valid H in (0,1/2)\n")
    for H in H_list:
        res[H] = {}
        print(f"==================== H = {H} ====================")
        for method, kind, thy in (("ajee", "alg", -0.8 * H), ("bb", "exp", None)):
            rows = []
            for N in N_list:
                g, w = soe(H, N, method=method, T=T)
                mtr = kernel_l2_metrics(H, g, w, T=T, dt=dt)
                rows.append((mtr["N"], mtr["rel_l2"], mtr["linf_rel"], mtr["mass_rel"]))
            res[H][method] = rows
            rate = _fit_rate([r[0] for r in rows], [r[1] for r in rows], kind)
            label = ("AJ-EE  (algebraic, theory slope -4H/5="
                     f"{thy:+.3f} vs logN)" if method == "ajee"
                     else "BB     (superpoly, slope vs sqrt(N))")
            print(f"  [{method.upper()}] {label}")
            print(f"    {'N':>5} {'rel_L2':>11} {'Linf[dt,T]':>11} {'mass[0,dt]':>11}")
            for n_, l2, li, ma in rows:
                print(f"    {n_:>5} {l2:>11.3e} {li:>11.3e} {ma:>11.3e}")
            print(f"    fitted rate = {rate:+.3f}  ({'per log N' if kind=='alg' else 'per sqrt(N)'})")
            # N_factors(target)
            hit = [n_ for n_, l2, *_ in rows if l2 <= target]
            print(f"    smallest N with rel-L2 <= {target:g}: "
                  f"{min(hit) if hit else 'not reached in N_list'}\n")
        # recommendation: fewest factors to reach target; tie-break on best error
        def first_hit(rows):
            h = [n_ for n_, l2, *_ in rows if l2 <= target]
            return min(h) if h else None
        def best(rows):
            return min(l2 for _, l2, *_ in rows)
        na, nb = first_hit(res[H]["ajee"]), first_hit(res[H]["bb"])
        ba, bb_ = best(res[H]["ajee"]), best(res[H]["bb"])
        winner = "BB" if bb_ < ba else "AJ-EE"
        sa = f"reaches at N>={na}" if na else f"not reached (best {ba:.1e})"
        sb = f"reaches at N>={nb}" if nb else f"not reached (best {bb_:.1e})"
        print(f"  --> at H={H}, target {target:g}: AJ-EE {sa}; BB {sb}  => recommend {winner} for 4b\n")
    # H = 1/2 guard
    g, w = soe(0.5, 16)
    m05 = kernel_l2_metrics(0.5, g, w, T=T, dt=dt)
    print(f"H=1/2 guard: K(t)=1 (flat); single gamma=0 mode -> rel_L2 = {m05['rel_l2']:.2e} "
          f"(nan? {np.isnan(m05['rel_l2'])})  [classical case, lift unnecessary]")
    if plot:
        _plot_gate_a(res, H_list, N_list, dt, T, target)
    return res


def _plot_gate_a(res, H_list, N_list, dt, T, target, path="output/layer4_kernel_soe.png"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, len(H_list), figsize=(5 * len(H_list), 4.4), sharey=True)
    if len(H_list) == 1:
        axes = [axes]
    for ax, H in zip(axes, H_list):
        for method, mk, col in (("ajee", "o-", "C0"), ("bb", "s-", "C3")):
            rows = res[H][method]
            ax.plot([r[0] for r in rows], [r[1] for r in rows], mk, color=col,
                    label=method.upper())
        ax.axhline(target, color="gray", ls=":", lw=1, label=f"target {target:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N factors"); ax.set_title(f"H = {H}")
        ax.grid(True, which="both", alpha=0.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("relative L2[0,T] kernel error")
    fig.suptitle("Gate A -- SOE kernel error vs N (AJ-EE vs Bayer-Breneis)")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    import sys, argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run Gate A (error vs N)")
    a = ap.parse_args()
    if a.gate:
        gate_a()
    else:
        # quick smoke: print a few errors
        for H in (0.05, 0.1, 0.2):
            for meth in ("ajee", "bb"):
                g, w = soe(H, 64, method=meth)
                print(H, meth, "N=", g.size, "rel_l2=",
                      f"{kernel_l2_metrics(H, g, w, dt=1/128)['rel_l2']:.3e}")
