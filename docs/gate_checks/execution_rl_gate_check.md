# Gate-check spec — execution-RL agent (Layer 3 / "execution alpha")

**The question (not a claim).** Can a reinforcement-learning agent learn an execution
schedule that reduces total trading cost — relative to (i) the Almgren–Chriss optimal
schedule and (ii) naive proportional trading — when liquidating/hedging in a *rough*
(non-Markovian) volatility market, and **by how much**? The honest answer may be a clear
improvement, a marginal one, none, or *negative* (RL fails to beat a well-tuned classical
baseline — a common and reportable outcome).

> No target number. An earlier framing carried "up to 15%"; that was a hypothesis written
> as a result. This spec commits to measuring the number, not hitting it. The result is
> whatever the run says, including "AC wins."

---

## 0. Scope discipline — what this is, and is NOT

**IN:** execution alpha only — schedule a known quantity of trading to minimise cost+risk
in the rough-Bergomi market. This is the one of the three proposed alphas that is
well-posed and falsifiable, and it connects to the parked Layer 2 (Almgren–Chriss) work
and to execution/hedging research at Oxford-Man.

**OUT (deliberately, for now):**
- *Signal alpha* ("buy rough / sell smooth" via the Layer 1c estimator) — **contradicts
  P3.** P3's central result is that roughness is *non-identified*: the RV proxy
  manufactures spurious roughness and estimates collapse toward Ĥ≈0.1. A strategy that
  sorts assets by Ĥ trades on a quantity P3 proves is largely an artifact. It needs its
  own prior investigation — *does any tradeable signal survive the observational-
  equivalence wall?* — before it is a strategy. Not built here.
- *Model alpha* ("arbitrage Asian mispricings Heston misses" via Layer 1b) — softer but
  same tension: calling the rough price "true" and Heston "wrong" assumes the market is
  rough, which P3 says you can't establish from data. It is a model-*disagreement* trade
  with real model risk, not a clean arbitrage. Not built here.

This spec is about #3 only. #1 and #2 stay as open research questions, not features.

---

## 1. The problem and the environment

**Task.** Over horizon $[0,T]$ with $N$ decision steps, trade an initial position $q_0$ to
zero (liquidation) — or track a hedge target — choosing the trade rate at each step. The
agent observes market state; it pays a temporary (slippage) cost for trading fast and bears
inventory risk for trading slow. Classic cost-vs-risk trade-off.

**Market = the existing rough Bergomi engine** (`layer1b_mlmc_asian.py`; H=0.10, η=1.5,
ρ=−0.70, ξ₀=0.04, S₀=100, T=1). The price path is driven by the rough variance process, so
volatility is **stochastic, persistent, and non-Markovian** — the property that makes this
different from classical Almgren–Chriss (which assumes constant or Markovian vol). *This is
the entire point*: if the market were constant-vol, AC is provably optimal and RL can only
match it. The rough vol is the structure RL might exploit.

**Cost model (this IS the experiment — specify it explicitly, do not hand-wave).**
- Temporary impact: cost of trading $v$ shares in a step $= \eta_{\text{imp}}\,|v|^{p}$
  (linear $p=1$ first; square-root $p=1.5$ as a documented variant). $\eta_{\text{imp}}$ is
  the impact coefficient.
- Permanent impact: $\gamma_{\text{imp}}\,v$ moving the mid (start with $\gamma_{\text{imp}}=0$,
  add later).
- Execution price each step = mid (from the rough-Bergomi path) ± temporary impact.
- **Critical caveat:** the agent must not be able to exploit a *quirk of the simulator*
  (e.g. predict the next mid from a discretisation artifact). The cost model and the
  observation set are what protect against this — see §5.

**State** (what the agent observes at step $k$): remaining inventory $q_k$, time-to-go
$T-t_k$, current instantaneous variance $V_{t_k}$ (or a noisy proxy of it), and recent
price moves. Whether $V_{t_k}$ is observed *cleanly* or only via a proxy is a deliberate
choice — clean $V$ is the optimistic case; a proxy is realistic and connects to P3's point
that you can't observe roughness cleanly.

**Action:** trade rate $v_k$ in the step (continuous; bounded). **Reward:** negative of
(execution cost + $\lambda\cdot$ inventory-risk penalty) for the step, where $\lambda$ is
the risk-aversion. $\lambda$ is fixed and *shared with the baselines* — see §3.

---

## 2. The committed prediction (to be falsified)

**Predicted, with reasoning, before building:**
- **RL will at best modestly beat Almgren–Chriss, not dramatically — and may not beat it
  at all.** AC is optimal for *arithmetic* impact with constant vol; the rough market
  violates the constant-vol assumption, so there is *room* for a state-dependent policy
  (trade slower when vol is high/persistent). But AC is a strong, well-tuned baseline, and
  RL agents frequently fail to beat such baselines once risk is matched. The honest
  prediction is a *small* edge from vol-timing the schedule, **if any**.
- **RL will beat naive proportional trading** (the floor) with high confidence — naive
  ignores both impact-optimality and vol, so beating it is necessary but *not sufficient*
  to call the project a success. **Beating naive is the floor; beating AC is the bar.**
- **The edge (if real) comes specifically from the non-Markovian structure** — i.e. the RL
  advantage over AC should *grow with* roughness (smaller H) and *vanish* as H→½ (where the
  market becomes Markovian and AC regains optimality). This is the sharp, falsifiable
  signature.

### Falsifiable predictions
| quantity | predicted | falsified if |
|---|---|---|
| RL cost vs naive (matched risk) | RL cheaper, clearly | RL ≥ naive |
| RL cost vs Almgren–Chriss (matched risk) | RL marginally cheaper *or* tied | RL clearly worse than AC |
| RL edge over AC vs roughness | grows as H↓, →0 as H→½ | edge independent of H (⇒ not from rough structure ⇒ likely an artifact) |
| edge survives out-of-sample seeds (§5) | yes, stable | edge vanishes/flips OOS ⇒ overfit |

**The H-dependence row is the most important test.** If RL beats AC but the edge does *not*
grow with roughness, the win is probably the agent exploiting the simulator, not the rough
structure — a false positive. The edge *must* track H to be a real result.

---

## 3. The baselines (the comparison is the whole project)

Both run in the *same* environment, *same* cost model, *same* risk-aversion $\lambda$, *same*
seeds.

1. **Almgren–Chriss** — the closed-form optimal schedule for the cost+risk objective
   (linear impact, quadratic risk). This is the parked Layer 2 work; it provides the
   *optimal Markovian* benchmark. The RL agent must beat *this* to justify itself.
2. **Naive proportional** — trade $q_0/N$ per step regardless of state. The floor.

**Matched-risk comparison (the AC analogue of pinned-L).** Cost alone is meaningless without
matching risk: an agent can always look "cheaper" by bearing more inventory risk. Compare on
the **cost–risk efficient frontier**: sweep $\lambda$, plot realised (cost, risk) for each
method, and compare curves. RL wins only if its frontier lies *below* AC's. A single
(cost, risk) point is not a result; the frontier is. *Do not compare raw costs at different
realised risk levels* — that is the execution-alpha version of the free-running-L artifact.

---

## 4. Validation gates

- **G-X1 — sanity / known limit.** Set H→½ (Markovian) and constant vol. AC is provably
  optimal; the RL agent must *converge to AC's frontier*, not beat it. If RL "beats" AC in
  the constant-vol limit, the environment or the matching is broken — **stop and debug.**
  (This is the BS-anchor of this project: a case with a known optimal answer.)
- **G-X2 — floor.** RL beats naive proportional on the frontier. Necessary, not sufficient.
- **G-X3 — the bar.** RL vs AC on the matched-risk frontier in the *rough* market. Report
  the frontier gap (cost reduction at matched risk), seed-averaged.
- **G-X4 — the mechanism test.** Sweep H ∈ {0.05, 0.10, 0.20, 0.35, 0.45}; measure the
  RL-vs-AC frontier gap at each. **Predicted:** gap grows as H↓, →0 at H→½. This is what
  proves the edge is the rough structure, not a simulator artifact.
- **G-X5 — out-of-sample (the overfitting gate, §5).** The G-X3 edge must hold on held-out
  seeds and at least one held-out regime. If it vanishes OOS, it was memorisation.

---

## 5. Overfitting protocol (RL memorises simulators — this is non-negotiable)

The single biggest risk: the agent learns the *training seeds*, not a *policy*. Guards:
- **Train/test seed split.** Train on one seed set, report *all* headline numbers on a
  disjoint test seed set. No number quoted from training seeds.
- **Held-out regime.** Train at one (H, η); evaluate at a nearby unseen (H, η). A real
  execution policy should transfer; a memorised one won't.
- **Frontier, not point** (§3) — prevents the "cheaper by taking more risk" illusion.
- **Seed-stability / sign-flip check.** If the RL-vs-AC edge flips sign across test seeds,
  it is inside the noise — report it as such (the discipline that caught the antithetic
  adaptive-L phantom and the κ=1 proxy offset, applied to RL).
- **AC must be tuned, not strawman.** Use AC's *optimal* parameters for the cost model, not
  a deliberately weak version. Beating a crippled baseline proves nothing.

---

## 6. Build plan (phased — do NOT build it all at once)

1. **Phase 0 — the environment + baselines, NO RL yet.** Build the execution env over the
   rough-Bergomi engine, the cost model, AC, and naive. Run **G-X1** (H→½ ⇒ AC optimal,
   the env reproduces the known answer) and confirm AC beats naive in the rough market.
   *This must be green before any agent is trained* — same as validating the fine path
   before the coupler. If the env is wrong, every RL number is wrong.
2. **Phase 1 — a tabular / simple-policy agent.** Before deep RL, a simple state-dependent
   policy (e.g. discretised state, Q-learning, or even a hand-designed "trade slower when V
   is high" heuristic) as a *cheap probe*: does *any* vol-aware policy beat AC in the rough
   market? If even a simple heuristic can't, deep RL likely won't either — and you've
   learned the answer for almost no cost. **This phase can kill the project early and
   cheaply.**
3. **Phase 2 — deep RL (only if Phase 1 shows promise).** PPO/DDPG or similar, with the full
   G-X3/G-X4/G-X5 protocol. The H-sweep (G-X4) is the centerpiece result.
4. **Phase 3 — robustness/variants.** Square-root impact, permanent impact, noisy V proxy,
   hedging (not just liquidation).

---

## 7. The hard kill-switch

**The project stops, honestly, if:**
- **G-X1 fails** (env can't reproduce AC-optimality in the Markovian limit) → the
  environment is broken; no result is trustworthy until fixed.
- **Phase 1 shows no vol-aware policy beats AC** → the rough structure offers no executable
  edge for this cost model; deep RL is unlikely to find one. Report the negative result
  (which is itself interesting: "rough vol does not create exploitable execution alpha under
  linear impact") and stop.
- **G-X4 shows the edge is H-independent** → the win is a simulator artifact, not the rough
  structure. Not a real result.
- **G-X5 shows the edge vanishes OOS** → overfit. Not a real result.

**You must be willing to report "Almgren–Chriss wins" or "no exploitable execution alpha."**
If there is no version of the outcome you'd accept as negative, this is not an experiment.
Half of the value here is a *credible* negative result — "we tested whether RL beats classical
execution in rough markets, rigorously, and it does/doesn't" — which is publishable and
honest either way.

---

## 8. Honest scope note

This is a **months-long project done properly**, not a weekend. It is also, of the three
alphas, the one *furthest from RoughVolLab's current identity* — it is really a **Layer 2
(Almgren–Chriss) + RL** project that *uses* the rough-Bergomi engine as its market, rather
than an extension of the pricing/identifiability work. That is fine — but worth seeing
clearly before committing: it is a new front, and its strength (falsifiability, the H-sweep
signature, the link to Cartea/execution at Oxford-Man) depends entirely on the discipline
above being held, especially the matched-risk frontier and the OOS protocol.

**Acceptance for the spec itself:** Phase 0 green (env reproduces AC-optimality at H→½; AC
beats naive in the rough market) is the gate before any agent is trained. Only then does
Phase 1 (the cheap heuristic probe) run, and only on its promise does deep RL begin.

---

### Recommended first action
Build **Phase 0 only** — the rough-Bergomi execution environment, Almgren–Chriss, naive
proportional, and **G-X1** (the H→½ Markovian sanity check). No RL. Confirm the environment
reproduces the known optimal answer before trusting it to evaluate an agent. That is the
foundation everything else stands on — exactly as the κ=1 fine path was validated before the
coupler was wired.

---

## 9. Output dashboard — `roughvollab_alpha_audit.png`

The run exports a six-panel figure. **It is an *audit*, not a summary** — every panel must
be able to show a *failure* as clearly as a success, or it is marketing, not evidence. The
panels are the gates of §4 made visual; they are not chosen for prettiness.

| panel | shows | the gate it visualises | failure it must expose |
|---|---|---|---|
| 1. Efficient frontier | (cost, risk) curves for RL / AC / naive, swept over λ | G-X2, G-X3 | curves overlapping ⇒ RL doesn't beat AC at matched risk |
| 2. H-sweep (**centerpiece**) | RL-vs-AC frontier gap vs H, prediction overlaid | G-X4 | a *flat* line ⇒ edge is a simulator artifact, not roughness |
| 3. Markovian sanity | at H→½, RL frontier converging onto AC | G-X1 | RL "beating" AC here ⇒ broken environment |
| 4. In-sample vs OOS | RL-vs-AC edge: training seeds vs held-out seeds | G-X5 | bars diverging ⇒ overfit |
| 5. Sample trajectory | inventory(t) for RL vs AC, variance path beneath | interpretability | RL *not* vol-timing ⇒ no real mechanism |
| 6. Cost decomposition | impact cost vs inventory-risk cost per method | matched-risk integrity | "cheaper" hiding higher risk-bearing |

**Phasing:** panels **3 and 6 are drawable from Phase 0** (baselines + G-X1, no RL) — build
them first as a plumbing check that the figure pipeline works before any agent exists.
Panels 1, 2, 4, 5 require the trained agent (Phase 2+). A Phase-0 dashboard with panels 3
and 6 (and AC-vs-naive on panel 1) is a valid early artifact.

**Honest-figure rules:** show the matched-risk frontier, never raw cost at unmatched risk
(panel 1); always overlay the *predicted* H-dependence on panel 2 so a flat result is
visibly a failure; put training and OOS *side by side* on panel 4 (never OOS alone). A
dashboard that cannot show these tests failing is not auditing anything.

Styling: follow the project palette (TEAL #1D9E75, PURPLE #7F77DD, CORAL #D85A30, GRAY
#888780, AMBER #BA7517) for consistency with the rest of RoughVolLab. RL = teal, AC =
purple, naive = gray throughout, so the three methods are identifiable at a glance across
every panel.
