"""
execution_alpha.py — Layer 2/3 execution-alpha gate-check, PHASE 0 ONLY.

Spec: execution_rl_gate_check.md.  The question: can an RL agent learn an
execution schedule that beats Almgren-Chriss (AC) — the Markovian optimum — in a
ROUGH (non-Markovian) volatility market?  This module builds the foundation that
must be green before any agent is trained:

  * the execution environment over the rough-Bergomi engine (roughvol_core),
  * Almgren-Chriss optimal schedule + naive proportional (the baselines),
  * G-X1: the Markovian sanity check — at constant vol the simulated AC frontier
    must match the analytic AC frontier (the "BS anchor" of this project),
  * the Phase-0 slice of roughvollab_alpha_audit.png (panels that are drawable
    with NO RL: frontier, Markovian sanity, sample trajectory, cost split;
    the RL panels are honest "Phase 2" placeholders).

NO RL is trained here (deliberately — see the spec's hard kill-switch §7).

COST MODEL (specified explicitly, per spec §1):
  Linear temporary PRICE impact => quadratic temporary COST per step
      temp_cost = eta_imp * sum_j (n_j)^2 / tau ,   n_j = shares traded in step j
  (the standard Almgren-Chriss form that gives the sinh schedule; the square-root
  |v|^1.5 variant is a documented Phase-3 robustness item, not built here.)
  Permanent impact gamma_imp = 0 for now.  r = 0, so the mid is a martingale and
  E[cost] = temp_cost; the inventory-risk term has mean ~0 and supplies the risk.

Implementation shortfall for a holdings schedule x (x_0=q0, x_N=0):
      IS = eta_imp * sum n_j^2 / tau            (temporary impact, deterministic)
         - sum_{j=0}^{N-1} x_j * (S_{j+1}-S_j)  (inventory P&L on the mid path)
  cost  = E[IS]   (~ temp_cost);   risk = std[IS]   (the inventory risk).

Run:  python execution_alpha.py
"""

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from roughvol_core import rough_bergomi_paths

# ── palette (spec §9): RL teal, AC purple, naive gray ───────────────────────
TEAL, PURPLE, CORAL, GRAY, AMBER = "#1D9E75", "#7F77DD", "#D85A30", "#888780", "#BA7517"
INK = "#222222"
RL_C, AC_C, NAIVE_C = TEAL, PURPLE, GRAY
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

# ── market / contract defaults (the rough-Bergomi engine) ───────────────────
MKT = dict(H=0.10, eta_vol=1.5, rho=-0.70, xi0=0.04, S0=100.0, T=1.0)
Q0, N_STEPS, ETA_IMP = 1.0, 50, 0.10       # liquidate 1 unit over 50 steps


def _log(m):
    print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT + BASELINES
# ════════════════════════════════════════════════════════════════════════════
def simulate_market(H, eta_vol, n_paths, N, T, S0, xi0, rho, seed):
    """Rough-Bergomi mid-price + variance paths on the N-step decision grid."""
    t, S, V = rough_bergomi_paths(N, H, n_paths, T=T, eta=eta_vol, rho=rho,
                                  xi0=xi0, S0=S0, r=0.0,
                                  rng=np.random.default_rng(seed))
    return t, S, V


def ac_holdings(q0, N, T, kappa):
    """Almgren-Chriss optimal holdings trajectory (sinh), urgency kappa.
    kappa -> 0 reduces to the linear TWAP (== naive proportional)."""
    t = np.linspace(0.0, T, N + 1)
    if kappa < 1e-6:
        x = q0 * (1.0 - t / T)                      # TWAP / naive limit
    else:
        x = q0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    x[-1] = 0.0
    return x


def naive_holdings(q0, N, T):
    return ac_holdings(q0, N, T, 0.0)               # linear = AC at kappa->0


def evaluate(x, S, eta_imp, tau):
    """Realised cost & risk of a (deterministic) holdings schedule x on paths S.
    Returns (E_cost, risk, temp_cost, risk_only) — risk == std of the IS."""
    n = x[:-1] - x[1:]                               # shares traded per step
    temp_cost = eta_imp * np.sum(n**2) / tau         # deterministic
    dS = np.diff(S, axis=1)                          # (n_paths, N)
    inv_pnl = -(x[None, :-1] * dS).sum(axis=1)       # inventory P&L per path
    cost = temp_cost + inv_pnl
    return float(cost.mean()), float(cost.std()), float(temp_cost), float(inv_pnl.std())


def ac_frontier(S, q0, N, T, eta_imp, kappas):
    """Sweep AC urgency -> realised (risk, cost) frontier on paths S."""
    tau = T / N
    out = []
    for k in kappas:
        x = ac_holdings(q0, N, T, k)
        ec, _, tc, risk = evaluate(x, S, eta_imp, tau)
        out.append(dict(kappa=k, risk=risk, cost=ec, temp=tc))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  G-X1 — Markovian sanity: simulated AC frontier == analytic AC frontier
# ════════════════════════════════════════════════════════════════════════════
def gate_x1(kappas):
    """Constant vol (eta_vol=0 => V=xi0) makes the mid a GBM, where AC is
    provably optimal and the inventory risk is sigma*S0*sqrt(tau*sum x_j^2).
    The environment PASSES if its simulated risk matches that closed form."""
    print("\n" + "=" * 72)
    print("  G-X1  Markovian sanity — simulated AC frontier vs analytic (const vol)")
    print("=" * 72)
    T, S0, xi0 = MKT["T"], MKT["S0"], MKT["xi0"]
    tau, sigma = T / N_STEPS, np.sqrt(xi0)
    # eta_vol=0 -> constant variance; H is then irrelevant (use 0.45, no singularity)
    _, S, _ = simulate_market(0.45, 0.0, 40000, N_STEPS, T, S0, xi0, MKT["rho"], 11)
    print("    kappa |  risk_sim   risk_analytic   ratio   cost_sim  cost_analytic")
    sims, anas = [], []
    worst = 0.0
    for k in kappas:
        x = ac_holdings(Q0, N_STEPS, T, k)
        ec, _, tc, risk_sim = evaluate(x, S, ETA_IMP, tau)
        risk_ana = sigma * S0 * np.sqrt(tau * np.sum(x[:-1]**2))
        ratio = risk_sim / risk_ana if risk_ana > 0 else np.nan
        worst = max(worst, abs(ratio - 1.0))
        sims.append((risk_sim, ec)); anas.append((risk_ana, tc))
        print(f"   {k:5.2f} | {risk_sim:8.4f}   {risk_ana:10.4f}    {ratio:5.3f}"
              f"   {ec:8.4f}   {tc:10.4f}")
    ok = worst < 0.04
    print(f"\n   max |risk_sim/risk_analytic - 1| = {worst:.3f}   "
          f"=> {'PASS — environment reproduces AC-optimality' if ok else 'FAIL — env broken, STOP'}")
    return dict(sim=sims, ana=anas, ok=ok, worst=worst)


# ════════════════════════════════════════════════════════════════════════════
#  DASHBOARD  (Phase-0 panels live; RL panels honest placeholders)
# ════════════════════════════════════════════════════════════════════════════
RLPH = ("Phase 2 — needs the trained RL agent\n"
        "(this panel is built once Phase 1's cheap\n"
        "heuristic probe clears the bar; see spec §6)")


def _rl_placeholder(ax, title):
    ax.text(0.5, 0.5, f"{title}\n\n{RLPH}", ha="center", va="center", fontsize=9,
            color=GRAY, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F7F6F2", ec=GRAY, lw=0.7))
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def panel_frontier(ax, rough_front, naive_pt):
    r = [p["risk"] for p in rough_front]; c = [p["cost"] for p in rough_front]
    ax.plot(r, c, "-", color=AC_C, lw=2, label="Almgren–Chriss (rough mkt)")
    ax.plot(r, c, "o", color=AC_C, ms=4)
    ax.plot(naive_pt["risk"], naive_pt["cost"], "s", color=NAIVE_C, ms=10,
            label="naive proportional (floor)", zorder=5)
    ax.annotate("RL must push\nthis frontier ↙", xy=(r[len(r)//2], c[len(c)//2]),
                xytext=(0.45, 0.7), textcoords="axes fraction", fontsize=8,
                color=RL_C, ha="center",
                arrowprops=dict(arrowstyle="->", color=RL_C, lw=1.2))
    ax.set_xlabel("inventory risk  (std of implementation shortfall)")
    ax.set_ylabel("expected cost")
    ax.set_title("① Cost–risk frontier (rough market)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def panel_hsweep_prediction(ax):
    ax.set_title("② RL-vs-AC edge vs roughness  (CENTERPIECE — Phase 2)")
    H = np.linspace(0.05, 0.5, 50)
    pred = np.maximum(0.0, (0.5 - H) / 0.45)**1.3     # grows as H↓, →0 at H→½ (shape only)
    ax.plot(H, pred, "--", color=RL_C, lw=2, label="PREDICTED edge (to be tested)")
    ax.axhline(0, color=GRAY, lw=1)
    ax.fill_between(H, 0, pred, color=RL_C, alpha=0.07)
    ax.text(0.30, 0.55, "flat line here ⇒\nedge is a simulator artifact,\nnot rough structure",
            transform=ax.transAxes, fontsize=8, color=CORAL, ha="center")
    ax.set_xlabel("Hurst exponent H  (rougher ←)")
    ax.set_ylabel("RL−AC frontier gap  (cost saved, a.u.)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(0.05, 0.5)


def panel_markov_sanity(ax, gx1):
    sr = [p[0] for p in gx1["sim"]]; sc = [p[1] for p in gx1["sim"]]
    ar = [p[0] for p in gx1["ana"]]; ac = [p[1] for p in gx1["ana"]]
    ax.plot(ar, ac, "-", color=GRAY, lw=2.5, label="analytic AC (known optimum)")
    ax.plot(sr, sc, "o", color=AC_C, ms=6, label="simulated AC (this env)")
    ax.set_xlabel("inventory risk"); ax.set_ylabel("expected cost")
    tag = "PASS" if gx1["ok"] else "FAIL"
    ax.set_title(f"③ G-X1 Markovian sanity — {tag} "
                 f"(max dev {gx1['worst']*100:.1f}%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def panel_oos_placeholder(ax):
    _rl_placeholder(ax, "④ In-sample vs out-of-sample edge (Phase 2)")


def panel_trajectory(ax, V_sample):
    t = np.linspace(0.0, MKT["T"], N_STEPS + 1)
    x_ac = ac_holdings(Q0, N_STEPS, MKT["T"], 3.0)
    x_nv = naive_holdings(Q0, N_STEPS, MKT["T"])
    ax.plot(t, x_nv, "-", color=NAIVE_C, lw=2, label="naive inventory")
    ax.plot(t, x_ac, "-", color=AC_C, lw=2, label="AC inventory (κ=3)")
    ax.set_xlabel("time t"); ax.set_ylabel("inventory  q(t)")
    ax.set_title("⑤ Schedules + a rough-vol path (RL overlay in Phase 2)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    axv = ax.twinx(); axv.grid(False)
    axv.plot(t, V_sample, color=AMBER, lw=1.1, alpha=0.7)
    axv.set_ylabel("instantaneous variance  V(t)", color=AMBER, fontsize=8)
    axv.tick_params(axis="y", labelcolor=AMBER, labelsize=7)


def panel_cost_split(ax, rough_front, naive_pt):
    # representative AC point at ~half the naive risk (a lower-risk schedule)
    target = 0.5 * naive_pt["risk"]
    ac_pt = min(rough_front, key=lambda p: abs(p["risk"] - target))
    methods = ["naive", f"AC (κ={ac_pt['kappa']:.1f})"]
    impact = [naive_pt["temp"], ac_pt["temp"]]
    risk = [naive_pt["risk"], ac_pt["risk"]]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, impact, w, color=CORAL, alpha=0.85, label="impact cost",
           edgecolor=INK, lw=0.5)
    ax.bar(x + w/2, risk, w, color=PURPLE, alpha=0.7, label="inventory risk (std)",
           edgecolor=INK, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_title("⑥ Cost split — buying less risk costs impact")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.text(0.5, 0.92, "compare only at MATCHED risk (panel ①) — never raw cost",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=GRAY)


def build_dashboard(gx1, rough_front, naive_pt, V_sample):
    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27,
                           top=0.88, bottom=0.13, left=0.06, right=0.975)
    panel_frontier(fig.add_subplot(gs[0, 0]), rough_front, naive_pt)
    panel_hsweep_prediction(fig.add_subplot(gs[0, 1]))
    panel_markov_sanity(fig.add_subplot(gs[0, 2]), gx1)
    panel_oos_placeholder(fig.add_subplot(gs[1, 0]))
    panel_trajectory(fig.add_subplot(gs[1, 1]), V_sample)
    panel_cost_split(fig.add_subplot(gs[1, 2]), rough_front, naive_pt)

    fig.suptitle("RoughVolLab — Execution-Alpha Audit  (Phase 0: environment + baselines)",
                 fontsize=20, fontweight="bold", x=0.5, y=0.965, color=INK)
    fig.text(0.5, 0.925,
             "Can RL beat Almgren–Chriss when liquidating in a rough market?   "
             "— Phase 0 validates the environment; no agent trained yet.",
             ha="center", fontsize=12, color=GRAY)
    g1 = "PASS" if gx1["ok"] else "FAIL"
    verdict = (
        f"PHASE-0 STATUS  —  G-X1 (Markovian sanity): {g1} "
        f"(env reproduces the analytic AC optimum to {gx1['worst']*100:.1f}%).   "
        "Baselines live: Almgren–Chriss frontier + naive floor in the rough "
        "market.   NO RL trained — the centerpiece H-sweep (②) and the OOS gate "
        "(④) are the bar the agent must clear in Phase 2.   "
        "The discipline: compare on the matched-risk frontier only; a flat ② "
        "would mean any RL win is a simulator artifact, not rough structure.")
    fig.text(0.5, 0.045, verdict, ha="center", va="center", fontsize=9.5,
             color=INK, wrap=True,
             bbox=dict(boxstyle="round,pad=0.6", fc="#F4F3EE", ec=GRAY, lw=0.8))
    fig.text(0.5, 0.012,
             f"rough-Bergomi market (H={MKT['H']}, η={MKT['eta_vol']}, "
             f"ρ={MKT['rho']}) · liquidate q0={Q0:g} over {N_STEPS} steps · "
             f"impact η={ETA_IMP:g} · seeds 11 · execution_alpha.py (Phase 0)",
             ha="center", fontsize=7.5, color=GRAY)
    out = "roughvollab_alpha_audit.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def main():
    t0 = time.time()
    print("\n" + "█" * 72)
    print("  RoughVolLab — EXECUTION-ALPHA gate-check  (PHASE 0, no RL)")
    print("█" * 72)
    kappas = np.array([0.3, 0.6, 1.0, 1.6, 2.4, 3.4, 4.8, 6.6, 9.0, 12.0])

    # 1) G-X1 — the env must reproduce the known AC optimum before we trust it
    gx1 = gate_x1(kappas)

    # 2) rough-market baselines: AC frontier + naive floor
    print("\n" + "=" * 72)
    print("  Rough-market baselines (H=0.10): AC frontier + naive floor")
    print("=" * 72)
    _, S, V = simulate_market(MKT["H"], MKT["eta_vol"], 40000, N_STEPS, MKT["T"],
                              MKT["S0"], MKT["xi0"], MKT["rho"], 11)
    tau = MKT["T"] / N_STEPS
    rough_front = ac_frontier(S, Q0, N_STEPS, MKT["T"], ETA_IMP, kappas)
    xn = naive_holdings(Q0, N_STEPS, MKT["T"])
    nec, _, ntc, nrisk = evaluate(xn, S, ETA_IMP, tau)
    naive_pt = dict(risk=nrisk, cost=nec, temp=ntc, kappa=0.0)
    print(f"   naive:  risk={nrisk:.4f}  cost={nec:.4f}  (impact {ntc:.4f})")
    lo = min(rough_front, key=lambda p: p["risk"])
    print(f"   AC fastest (κ={lo['kappa']:.1f}): risk={lo['risk']:.4f} "
          f"cost={lo['cost']:.4f}  — AC buys risk down to "
          f"{lo['risk']/nrisk*100:.0f}% of naive's at higher impact cost")
    # rough vs Markovian risk at the naive schedule (the structure RL might use)
    risk_markov = np.sqrt(MKT["xi0"]) * MKT["S0"] * np.sqrt(tau * np.sum(xn[:-1]**2))
    print(f"   inventory risk on the SAME naive schedule:  rough {nrisk:.4f}  vs "
          f"const-vol {risk_markov:.4f}  ({nrisk/risk_markov:.2f}× — rough vol is the structure)")

    # 3) dashboard
    print("\n" + "=" * 72)
    _log("building Phase-0 dashboard …")
    V_sample = V[0]
    out = build_dashboard(gx1, rough_front, naive_pt, V_sample)
    _log(f"dashboard written: {out}   ({time.time()-t0:.0f}s)")
    print("─" * 72)
    print(f"  PHASE 0 {'GREEN — Phase 1 (cheap heuristic probe) may proceed' if gx1['ok'] else 'RED — env broken, fix before any agent'}")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    main()
