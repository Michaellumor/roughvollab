"""
execution_alpha_phase1.py — PHASE 1 of the execution-RL gate-check
(execution_rl_gate_check.md §6): the CHEAP vol-aware heuristic probe = the
KILL-SWITCH.  Honest job: does ANY vol-aware policy beat Almgren-Chriss on the
MATCHED-RISK frontier in the rough market?  If not, deep RL is not pursued.
A negative result is expected, valid, and reported plainly.  NO deep RL here.
NOT tuned to win.  Builds on the Phase-0 functions in execution_alpha.py.

THE HEURISTIC (vol-reactive, signed strength so the direction is TESTED):
  - AC trades dh_AC[k] for urgency kappa (== risk-aversion lambda)
  - causal signal (no look-ahead in the timing): z[k] = (V[k]-xi0)/xi0, clip[-1,3]
  - dh[k] = dh_AC[k]*max(0, 1 + theta*z[k]), renormalised so sum(dh)=q0
  - theta>0 = "faster when V high", theta<0 = "slower when high", theta=0 = AC.
    Swept over [-2,2]; the sign is not assumed.

THE BAR (committed before any numbers — beats AC only if ALL hold):
  (i)   fixed-theta* Pareto front below AC by MORE than the seed s.e. at
        multiple risk levels;
  (ii)  the winning theta* is clearly nonzero;
  (iii) the edge is sign-stable across seeds {5,11,23};
  (iv)  SANITY: at H=0.49 (near-Markovian) the heuristic must NOT beat AC.
Verdict uses a FIXED theta* tested across seeds (not the noise-selected
envelope, which is <= AC by construction and overfits).
"""

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import execution_alpha as ea          # Phase-0 foundation (validated env + AC)

XI0, S0, T = ea.MKT["xi0"], ea.MKT["S0"], ea.MKT["T"]
Q0, N, ETA, TAU = ea.Q0, ea.N_STEPS, ea.ETA_IMP, ea.MKT["T"] / ea.N_STEPS
KAPPAS = np.array([0.3, 0.6, 1.0, 1.6, 2.4, 3.4, 4.8, 6.6, 9.0, 12.0])
THETAS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
SEEDS = [5, 11, 23]
NPATH = 40_000


# ── the path-dependent vol-reactive heuristic ───────────────────────────────
def _causal_schedule(kappa, theta, V):
    """CAUSAL vol-reactive schedule — NO look-ahead.  At each step trade a
    state-dependent fraction of the ACTUAL remaining inventory: AC's
    fraction-of-remaining phi_AC[k] modulated by the current vol signal z[k],
    forcing full liquidation at the last step.  theta=0 reproduces AC exactly.
    Because x_k is F_k-measurable and the mid is a martingale, E[x_k dS_k]=0
    => E[inv_pnl]=0 (the precondition); the old global renormalisation broke
    this by making the schedule depend on the whole vol path."""
    n_paths = V.shape[0]
    x_ac = ea.ac_holdings(Q0, N, T, kappa)
    dh_ac = x_ac[:-1] - x_ac[1:]
    phi = np.zeros(N)                                  # AC fraction-of-remaining
    pos = x_ac[:-1] > 1e-12
    phi[pos] = dh_ac[pos] / x_ac[:-1][pos]
    z = np.clip((V[:, :N] - XI0) / XI0, -1.0, 3.0)     # causal vol signal
    q = np.full(n_paths, Q0)
    dh = np.empty((n_paths, N)); x = np.empty((n_paths, N + 1)); x[:, 0] = Q0
    for k in range(N):
        g = (np.ones(n_paths) if k == N - 1            # clear all remaining at T
             else np.clip(phi[k] * (1.0 + theta * z[:, k]), 0.0, 1.0))
        trade = q * g
        dh[:, k] = trade
        q = q - trade
        x[:, k + 1] = q
    return dh, x


def heuristic_detail(kappa, theta, V, S):
    """(E[temp], E[inv_pnl], inv_se, E[cost], risk) — for the precondition proof."""
    dh, x = _causal_schedule(kappa, theta, V)
    temp = ETA * (dh**2).sum(axis=1) / TAU
    inv = -(x[:, :-1] * np.diff(S, axis=1)).sum(axis=1)
    cost = temp + inv
    return (float(temp.mean()), float(inv.mean()),
            float(inv.std() / np.sqrt(len(inv))), float(cost.mean()), float(cost.std()))


def heuristic_eval(kappa, theta, V, S):
    """(mean cost, inventory risk) for the CAUSAL vol-reactive schedule."""
    _, im, _, cm, cs = heuristic_detail(kappa, theta, V, S)
    return cm, cs


def eval_grid(H, eta_vol, seeds):
    """grid[seed][(kappa,theta)] = (cost, risk)."""
    grid = {}
    for sd in seeds:
        _, S, V = ea.simulate_market(H, eta_vol, NPATH, N, T, S0, XI0,
                                     ea.MKT["rho"], sd)
        grid[sd] = {(k, th): heuristic_eval(k, th, V, S)
                    for k in KAPPAS for th in THETAS}
    return grid


# ── frontier helpers ────────────────────────────────────────────────────────
def theta_curve(grid_seed, theta):
    """(cost,risk) points over kappa at fixed theta, sorted by risk."""
    pts = [grid_seed[(k, theta)] for k in KAPPAS]
    return sorted(pts, key=lambda p: p[1])


def pooled_curve(grid, seeds, theta):
    """seed-averaged (cost,risk) over kappa at fixed theta."""
    pts = []
    for k in KAPPAS:
        cs = np.mean([grid[sd][(k, theta)][0] for sd in seeds])
        rs = np.mean([grid[sd][(k, theta)][1] for sd in seeds])
        pts.append((cs, rs))
    return sorted(pts, key=lambda p: p[1])


def cost_at(curve, risks):
    r = [p[1] for p in curve]; c = [p[0] for p in curve]
    order = np.argsort(r)
    return np.interp(risks, np.array(r)[order], np.array(c)[order])


def matched_gap(ac_curve, h_curve, risks):
    """AC_cost(R) - heuristic_cost(R) at matched risk (positive = heuristic wins)."""
    return cost_at(ac_curve, risks) - cost_at(h_curve, risks)


def pareto_front(points):
    """lower-left Pareto envelope (minimise cost & risk)."""
    pts = sorted(points, key=lambda p: (p[1], p[0]))
    front, best = [], np.inf
    for c, r in pts:
        if c < best - 1e-12:
            front.append((c, r)); best = c
    return sorted(front, key=lambda p: p[1])


# ── the comparison + the committed bar ──────────────────────────────────────
def risk_levels_from(ac_curve, n=6):
    r = [p[1] for p in ac_curve]
    return np.linspace(min(r) * 1.08, max(r) * 0.92, n)


def run_comparison(H, eta_vol, label):
    grid = eval_grid(H, eta_vol, SEEDS)
    ac_pool = pooled_curve(grid, SEEDS, 0.0)
    risks = risk_levels_from(ac_pool)
    # pooled matched-risk gap for each fixed theta
    pooled_gap = {th: matched_gap(ac_pool, pooled_curve(grid, SEEDS, th), risks)
                  for th in THETAS}
    # per-seed gap for each theta (paired within seed: same paths for AC & heur)
    per_seed_gap = {th: np.array([
        matched_gap(theta_curve(grid[sd], 0.0), theta_curve(grid[sd], th), risks)
        for sd in SEEDS]) for th in THETAS}
    return dict(grid=grid, ac_pool=ac_pool, risks=risks,
                pooled_gap=pooled_gap, per_seed_gap=per_seed_gap, H=H, label=label)


def evaluate_bar(rough, sanity):
    nonzero = [th for th in THETAS if abs(th) > 1e-9]
    # theta* = nonzero theta with the largest mean pooled gap (the candidate edge)
    theta_star = max(nonzero, key=lambda th: rough["pooled_gap"][th].mean())
    pg = rough["pooled_gap"][theta_star]                 # gap per risk level
    ps = rough["per_seed_gap"][theta_star]               # (3 seeds, nlevels)
    se = ps.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))    # seed s.e. per level
    seed_means = ps.mean(axis=1)                         # mean gap per seed

    # (i) pooled gap > s.e. at multiple risk levels
    bar_i = int(np.sum(pg > se)) >= 2 and pg.mean() > se.mean()
    # (ii) theta* clearly nonzero
    bar_ii = abs(theta_star) >= 0.5
    # (iii) sign-stable & nonvanishing across seeds
    same_sign = np.all(seed_means > 0) or np.all(seed_means < 0)
    bar_iii = same_sign and seed_means.mean() > seed_means.std(ddof=1) / np.sqrt(len(SEEDS))
    # (iv) sanity: at H=0.49 theta* must NOT beat AC (gap within s.e.)
    sg = sanity["per_seed_gap"][theta_star]
    sse = sg.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    sgp = sanity["pooled_gap"][theta_star]
    bar_iv_ok = sgp.mean() <= max(sse.mean(), 1e-9) * 1.5   # near-zero edge at H=0.49
    return dict(theta_star=theta_star, pooled_gap=pg, seed_gap=ps, se=se,
                seed_means=seed_means, bar_i=bar_i, bar_ii=bar_ii,
                bar_iii=bar_iii, bar_iv_ok=bar_iv_ok,
                sanity_gap=sgp, sanity_se=sse)


# ── dashboard (regenerate; panel 1 now carries the heuristic) ────────────────
def panel_frontier(ax, rough, naive_pt, theta_star):
    ac = rough["ac_pool"]
    ar = [p[1] for p in ac]; ac_c = [p[0] for p in ac]
    ax.plot(ar, ac_c, "-o", color=ea.AC_C, lw=2, ms=4, label="Almgren–Chriss (θ=0)")
    hc = pooled_curve(rough["grid"], SEEDS, theta_star)
    hr = [p[1] for p in hc]; hcc = [p[0] for p in hc]
    ax.plot(hr, hcc, "-s", color=ea.RL_C, lw=2, ms=4,
            label=f"vol-heuristic (θ*={theta_star:+.1f})")
    # full Pareto envelope over all (kappa,theta) — shown faint (overfits in-sample)
    env = pareto_front([rough["grid"][SEEDS[0]][(k, th)]
                        for k in KAPPAS for th in THETAS])
    ax.plot([p[1] for p in env], [p[0] for p in env], ":", color=ea.GRAY, lw=1,
            alpha=0.7, label="envelope over all θ (in-sample, ↓ biased)")
    ax.plot(naive_pt["risk"], naive_pt["cost"], "s", color=ea.NAIVE_C, ms=10,
            label="naive (floor)", zorder=5)
    ax.set_xlabel("inventory risk (std of shortfall)"); ax.set_ylabel("expected cost")
    ax.set_title("① Cost–risk frontier — heuristic vs AC")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")


def panel_hsweep_probe(ax, hsweep, theta_star):
    Hs = sorted(hsweep)
    gap = [hsweep[h]["pooled_gap"][theta_star].mean() for h in Hs]
    Hp = np.linspace(0.05, 0.5, 50)
    pred = np.maximum(0.0, (0.5 - Hp) / 0.45)**1.3
    pred = pred / pred.max() * max(max(gap), 1e-6) * 1.1   # scaled shape only
    ax.plot(Hp, pred, "--", color=ea.RL_C, lw=1.5, alpha=0.6,
            label="predicted shape (grows as H↓)")
    ax.plot(Hs, gap, "o-", color=ea.CORAL, lw=2, ms=7,
            label=f"measured heuristic edge (θ*={theta_star:+.1f})")
    ax.axhline(0, color=ea.GRAY, lw=1)
    ax.set_xlabel("Hurst exponent H (rougher ←)")
    ax.set_ylabel("heuristic−AC gap (matched risk)")
    ax.set_title("② Edge vs roughness — Phase-1 probe")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")


def build_dashboard(gx1, rough, naive_pt, V_sample, verdict, hsweep):
    th = verdict["theta_star"]
    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27,
                           top=0.88, bottom=0.13, left=0.06, right=0.975)
    panel_frontier(fig.add_subplot(gs[0, 0]), rough, naive_pt, th)
    panel_hsweep_probe(fig.add_subplot(gs[0, 1]), hsweep, th)
    ea.panel_markov_sanity(fig.add_subplot(gs[0, 2]), gx1)
    ea.panel_oos_placeholder(fig.add_subplot(gs[1, 0]))
    ea.panel_trajectory(fig.add_subplot(gs[1, 1]), V_sample)
    # cost-split reuses the Phase-0 AC frontier (deterministic temp costs)
    rf = ea.ac_frontier(rough["S"], Q0, N, T, ETA, KAPPAS)
    ea.panel_cost_split(fig.add_subplot(gs[1, 2]), rf, naive_pt)

    met = verdict["bar_met"]
    fig.suptitle("RoughVolLab — Execution-Alpha Audit  (Phase 1: vol-heuristic kill-switch)",
                 fontsize=20, fontweight="bold", x=0.5, y=0.965, color=ea.INK)
    fig.text(0.5, 0.925,
             "Does ANY cheap vol-aware policy beat Almgren–Chriss on the "
             "matched-risk frontier?  — the gate before deep RL.",
             ha="center", fontsize=12, color=ea.GRAY)
    head = ("BAR MET — an exploitable vol-timing edge exists; Phase 2 (deep RL) justified."
            if met else
            "KILL-SWITCH FIRED — no vol-aware heuristic beats Almgren–Chriss; deep RL NOT pursued.")
    g = verdict
    verdict_txt = (
        f"{head}\n"
        f"θ* = {g['theta_star']:+.1f}  ·  pooled matched-risk gap = "
        f"{g['pooled_gap'].mean():+.4f} ± {g['se'].mean():.4f} (seed s.e.)  ·  "
        f"per-seed gaps {{5,11,23}} = [{', '.join(f'{x:+.4f}' for x in g['seed_means'])}]\n"
        f"(i) gap>s.e. at multiple risks: {g['bar_i']}   "
        f"(ii) θ* clearly nonzero: {g['bar_ii']}   "
        f"(iii) sign-stable across seeds: {g['bar_iii']}   "
        f"(iv) H=0.49 sanity (no edge): {g['bar_iv_ok']}   "
        f"[H=0.49 gap {g['sanity_gap'].mean():+.4f}]")
    fig.text(0.5, 0.05, verdict_txt, ha="center", va="center", fontsize=9.3,
             color=ea.INK, wrap=True,
             bbox=dict(boxstyle="round,pad=0.6",
                       fc=("#EAF5EF" if met else "#FBEDE8"), ec=ea.GRAY, lw=0.9))
    fig.text(0.5, 0.012,
             f"rough-Bergomi (H={ea.MKT['H']}, η={ea.MKT['eta_vol']}) · liquidate "
             f"q0={Q0:g}/{N} steps · linear impact η={ETA:g} · seeds {SEEDS} · "
             f"{NPATH} paths · execution_alpha_phase1.py (no deep RL)",
             ha="center", fontsize=7.5, color=ea.GRAY)
    out = "roughvollab_alpha_audit.png"
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    return out


def precondition_proof():
    """Prove the causal schedule has NO look-ahead: E[inv_pnl] ~ 0 (within MC
    s.e.) for every theta — vs the old global-renorm artifact of +/-2..4."""
    print("\n" + "=" * 74)
    print("  PRECONDITION — E[inv_pnl] ~ 0 (no look-ahead) for the causal schedule")
    print("=" * 74)
    _, S, V = ea.simulate_market(ea.MKT["H"], ea.MKT["eta_vol"], NPATH, N, T,
                                 S0, XI0, ea.MKT["rho"], 11)
    worst_im, worst_z = 0.0, 0.0
    print("    theta |  E[inv_pnl]   s.e.    z=|E|/s.e.   (must be ~0; |z|<4 ⇒ =0)")
    for th in (-2.0, -1.0, 0.0, 1.0, 2.0):
        # scan kappa, keep the worst |E[inv_pnl]| as the headline number
        ims = [(abs(heuristic_detail(k, th, V, S)[1]), k) for k in KAPPAS]
        _, im, se, _, _ = heuristic_detail(1.0, th, V, S)
        z = abs(im) / se if se > 0 else 0.0
        wk = max(ims)[0]
        worst_im = max(worst_im, wk); worst_z = max(worst_z, z)
        print(f"   {th:+5.1f} | {im:+9.4f}   {se:.4f}   {z:6.2f}")
    ok = worst_z < 4.0
    print(f"\n   PRECONDITION NUMBER: max |E[inv_pnl]| over the (θ,κ) grid = "
          f"{worst_im:.4f}   (old global-renorm: ±2–4)")
    print(f"   => {'PASS — look-ahead removed, E[inv_pnl]=0 within MC noise' if ok else 'FAIL — still biased, STOP'}")
    return worst_im, ok


def main():
    t0 = time.time()
    print("\n" + "█" * 74)
    print("  EXECUTION-ALPHA  PHASE 1 — vol-heuristic kill-switch (no deep RL)")
    print("█" * 74)

    pre_num, pre_ok = precondition_proof()
    if not pre_ok:
        print("\n  ** precondition failed — not reading the frontier. Fix first. **")
        return

    print("\n  evaluating heuristic vs AC on the matched-risk frontier (H=0.10) …")
    rough = run_comparison(ea.MKT["H"], ea.MKT["eta_vol"], "rough H=0.10")
    print("  sanity comparison at H=0.49 (near-Markovian) …")
    sanity = run_comparison(0.49, ea.MKT["eta_vol"], "near-Markov H=0.49")
    verdict = evaluate_bar(rough, sanity)
    th = verdict["theta_star"]

    # H spot-check (does the edge grow as H falls?) — fixed theta*
    print("  H spot-check {0.05, 0.10, 0.20, 0.49} at fixed theta* …")
    hsweep = {0.10: rough, 0.49: sanity}
    for H in (0.05, 0.20):
        hsweep[H] = run_comparison(H, ea.MKT["eta_vol"], f"H={H}")

    print("\n" + "=" * 74)
    print(f"  RESULTS  (theta* = {th:+.1f}, the best nonzero vol-timing strength)")
    print("=" * 74)
    print(f"   pooled matched-risk gap (H=0.10):  mean {verdict['pooled_gap'].mean():+.4f}"
          f"  vs seed s.e. {verdict['se'].mean():.4f}")
    print(f"   per-seed mean gap {SEEDS}: "
          f"[{', '.join(f'{x:+.4f}' for x in verdict['seed_means'])}]")
    print(f"   per-risk gap:  "
          f"[{', '.join(f'{x:+.4f}' for x in verdict['pooled_gap'])}]")
    print(f"   H-spot-check gap(θ*): " +
          "  ".join(f"H={h}:{hsweep[h]['pooled_gap'][th].mean():+.4f}"
                    for h in sorted(hsweep)))
    print(f"   H=0.49 SANITY gap: {verdict['sanity_gap'].mean():+.4f} "
          f"(s.e. {verdict['sanity_se'].mean():.4f})")
    print("\n   THE BAR (committed before results):")
    print(f"     (i)   gap > s.e. at multiple risk levels : {verdict['bar_i']}")
    print(f"     (ii)  theta* clearly nonzero (|θ*|>=0.5) : {verdict['bar_ii']}")
    print(f"     (iii) sign-stable across seeds           : {verdict['bar_iii']}")
    print(f"     (iv)  H=0.49 sanity (heuristic NOT > AC) : {verdict['bar_iv_ok']}")

    bar_met = (verdict["bar_i"] and verdict["bar_ii"]
               and verdict["bar_iii"] and verdict["bar_iv_ok"])
    verdict["bar_met"] = bar_met
    # broken-comparison guard: edge at H=0.49 as big as at H=0.10
    broken = (not verdict["bar_iv_ok"]) and verdict["bar_i"]
    print("\n" + "─" * 74)
    if broken:
        print("  ** BROKEN-COMPARISON FLAG: heuristic 'beats' AC even at H=0.49 — the")
        print("     edge is H-independent (likely unmatched risk / look-ahead). NOT a")
        print("     win. STOP and inspect before any further claim. **")
    elif bar_met:
        print("  VERDICT: BAR MET — a vol-timing edge survives the discipline.")
        print(f"  Winning strength θ*={th:+.1f} => "
              f"{'trade FASTER when V is high' if th > 0 else 'trade SLOWER when V is high'}.")
        print("  Phase 2 (deep RL) is justified.")
    else:
        print("  VERDICT: KILL-SWITCH FIRED — no vol-aware heuristic beats Almgren–Chriss")
        print("  on the matched-risk frontier. Under linear impact, rough-volatility")
        print("  structure offers no executable execution edge here. Deep RL NOT pursued.")
        print("  (Clean negative result, per spec §7 — theta/lambda/signal NOT tuned.)")
    print("─" * 74)

    # rebuild the dashboard with panel 1 carrying the heuristic
    gx1 = ea.gate_x1(KAPPAS)
    _, S, V = ea.simulate_market(ea.MKT["H"], ea.MKT["eta_vol"], NPATH, N, T,
                                 S0, XI0, ea.MKT["rho"], 11)
    rough["S"] = S
    xn = ea.naive_holdings(Q0, N, T)
    nec, _, ntc, nrisk = ea.evaluate(xn, S, ETA, TAU)
    naive_pt = dict(risk=nrisk, cost=nec, temp=ntc, kappa=0.0)
    out = build_dashboard(gx1, rough, naive_pt, V[0], verdict, hsweep)
    print(f"\n  dashboard updated: {out}   ({time.time()-t0:.0f}s)\n")


if __name__ == "__main__":
    main()
