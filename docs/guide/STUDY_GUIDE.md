# RoughVolLab — the study guide

*The reference text for understanding this project. Four levels: read Level 0
today, Level 1 this week, Level 2 as you work with each layer, Level 3 over
the summer. The [interactive explainer](explainer.html) is the intuition
playground for the same ideas — this guide adds the depth; the explainer makes
it visible. Written for a maths undergraduate who has done first year and
half-forgotten it; every prerequisite is restated when used.*

---

## Level 0 — the elevator pitch

In 2018 a famous paper ("Volatility is rough", Gatheral–Jaisson–Rosenbaum)
claimed that market volatility wiggles with an extremely jagged texture —
a fractional process with Hurst exponent H ≈ 0.1 instead of Brownian motion's
0.5. The claim caught fire: it explains steep short-dated option skews, and a
whole modelling industry ("rough volatility") grew around it.

**RoughVolLab interrogates that claim instead of assuming it.** Five
questions, five honestly-earned answers:

1. **Can you measure the roughness from price history?** *Mostly no* — the
   realized-variance proxy manufactures spurious roughness; real assets
   (BTC, ETH, SPX) sit in the regime where rough and smooth are
   observationally equivalent. *(Layer 1c + Phase B → paper P3)*
2. **Can rough models be priced cheaply?** *Yes, but the celebrated tool
   (multilevel Monte Carlo) fails here* — its variance-decay rate is β ≈ 2H,
   structurally too slow; conditional single-grid Monte Carlo wins instead.
   *(Layer 1b → paper P2)*
3. **Can the rough structure be traded?** *No* — a vol-reactive execution
   policy is ~5 standard errors worse than the classical Almgren–Chriss
   schedule at matched risk. *(Layer 2)*
4. **Can you recover the roughness from option prices?** *No* — H is the flat
   direction of the calibration problem (it trades off against vol-of-vol ν);
   on live Deribit BTC and ETH surfaces it rails to its bound. *(Layer 4 →
   paper P1)*
5. **Does roughness make hedging better?** *Not beyond generic effects* —
   deep hedging beats delta-hedging under transaction costs in any model; the
   roughness-specific increment is statistically zero. *(Layer 3)*

Four negatives out of five — and that is the point. The project's real
product is *methodology*: every claim sits behind a falsifiable prediction, a
validation against a known answer, and a written verdict. Four seductive
false positives (including one at z = 5.6 significance) were caught by those
gates. The one-line summary: **roughness may be there, but the data can't
prove it and the market doesn't pay for it — and proving that carefully is
worth more than assuming otherwise.**

---

## Level 1 — the concept map

Read these in order; each builds on the last. Explainer links give the
interactive version.

| # | Concept | One paragraph | Explainer |
|---|---------|----------------|-----------|
| 1 | **Return & volatility** | A return is a percentage price change; volatility σ is the statistical size of returns, quoted per year. It clusters (calm/storm regimes) and is never directly observable. | [primer](explainer.html#primer) |
| 2 | **Brownian motion** | The infinite-speed limit of a coin-flip walk: independent increments, spread √t. The default noise source of all continuous-time finance. | [primer](explainer.html#primer) |
| 3 | **Stochastic volatility model** | Two coupled noise streams: one drives the price, one drives σ itself, correlated ρ < 0 (crashes are volatile). | [primer](explainer.html#primer) |
| 4 | **Monte Carlo** | Price = average payoff over simulated futures. Universal but slow: error ∝ 1/√N (Central Limit Theorem). | [demo 1](explainer.html#mc) |
| 5 | **fBm & the Hurst exponent** | Fractional Brownian motion generalises BM with memory: Var(B_t) = t^{2H}. H < ½ ⇒ anti-persistent, jagged, long memory. "Rough volatility" = log σ behaves like fBm with H ≈ 0.1. | [demo 2](explainer.html#hurst) |
| 6 | **Volterra representation** | Build fBm-like processes by convolving Brownian increments with the kernel (t−s)^{H−½}. This is *the* computational trick — `roughvol_core.py` is exactly this convolution done right. | [demo 2 maths](explainer.html#hurst) |
| 7 | **Rough Bergomi** | V_t = ξ₀·exp(η·W̃_t − ½η²·Var W̃_t): lognormal variance around forward level ξ₀, shaken by the rough driver. The compensator term makes E[V_t] = ξ₀ *exactly* — if you use the discrete variance (the repo's HIGH bug RVL-001 is using the continuum one). | [demo 2 maths](explainer.html#hurst) |
| 8 | **Realized variance & the proxy problem** | You estimate σ² by summing squared returns in a window. The estimate = truth × noisy multiplier, and that noise *reads as roughness*. The project's central mechanism. | [demo 3](explainer.html#proxy) |
| 9 | **Estimators & the corruption ladder** | Three independent H-estimators (GJR structure functions, Cont–Das p-variation, MF-DFA), audited against five corruptions: proxy noise, microstructure, jumps, finite samples, seasonality. Their biases *disagree in sign* — the audit's headline. | [demo 3 maths](explainer.html#proxy) |
| 10 | **Identifiability** | A parameter is identifiable if the data could, in principle, pin it down. Two failure modes here: the non-monotone bias curve (one observed Ĥ, two true H's) and the flat Jacobian direction (H ↔ ν trade-off). | [demos 3 & 5](explainer.html#smile) |
| 11 | **MLMC** | Giles' telescoping trick: cheap coarse simulations + few fine corrections. Wins iff level-variance decays faster than cost grows (β > γ). Rough paths give β ≈ 2H ≪ 1 ⇒ structural failure. | [demo 4](explainer.html#mlmc) |
| 12 | **Conditional MC ("turbocharging")** | Given the volatility path, the price is exactly lognormal — so integrate it out analytically (Romano–Touzi) instead of simulating it. The estimator that actually won P2. | [demo 4 maths](explainer.html#mlmc) |
| 13 | **Characteristic functions** | φ(u) = E[e^{iu·logS}] determines the whole distribution; option prices follow by Fourier inversion (Gil-Pelaez). Rough Heston has a semi-closed CF via a *fractional Riccati* ODE — a simulation-free ground truth for validating simulators. | [demo 5](explainer.html#smile) |
| 14 | **Calibration & the flat valley** | Fit model parameters to market smiles by least squares. The Jacobian's smallest eigen-direction points along H (|flat[H]| ≈ 0.99 single-smile): the market smile cannot pin H down. | [demo 5](explainer.html#smile) |
| 15 | **Weak vs strong convergence** | Strong = pathwise error (O(n^{−H}), brutal); weak = pricing-bias error (measured α ≈ 1 for H ≥ 0.2 — much better). Averages forgive what paths cannot. *(Paper P4)* | — |
| 16 | **Deep hedging & CVaR** | Train a neural network hedging policy by minimising Conditional Value-at-Risk (Rockafellar–Uryasev). Beats delta under frictions generically ("the Buehler edge"); roughness adds nothing on top. | — |

**How to read ROADMAP.md** (the project's single source of truth): start at
the Status board, then read decisions **D20–D23** (the P2 estimator verdict),
**D31/D35/D44** (weak order + the reproducibility correction), **D37–D42**
(the calibration arc: single smile → surface → live BTC → live ETH),
**D40/D43** (the hedging null and the hardest-won false-positive catch). The
log is append-only: history is never rewritten, corrections get new entries —
that's the audit trail that makes the negatives credible.

---

## Level 2 — the layers, mechanically

Each entry: the question, what the code computes, where, the verdict, how to
reproduce, and what's known-broken (IDs from [`ERROR_AUDIT.md`](../../ERROR_AUDIT.md)).

### The core — `roughvol_core.py` *(trust this)*
The ONE trusted path engine: `volterra_weights` (κ=0 BLP kernel + the
discrete variance v_i = 2H·dt·Σg²), `volterra_process` (FFT convolution),
`rough_log_variance_paths` (ground truth for estimator audits),
`rough_bergomi_paths` (asset + variance). Pinned by `test_roughvol_core.py`
(18 tests): variance matches the discrete formula, E[V_t] = ξ₀ exactly.
Everything downstream imports from here — **never** from Layer 1.

### Layer 1 — `layer1_rough_vol.py` *(pedagogy; known-broken)*
Cholesky-exact fBm, the hybrid scheme, first Hurst estimates; four teaching
sections with figures. ⚠️ Carries the repo's two HIGH bugs: **RVL-001**
(`fbm_hybrid` over-subtracts near-diagonal kernel terms → Var(B^H_1) ≈ 0.91
measured vs 1.0 assumed) and **RVL-002** (continuum compensator → E[V_t] up
to 65% low). Both verified at runtime 2026-07-02; fix plan + acceptance test
in the audit. Fixing them is the recommended first hands-on exercise.

### Layer 1b — `layer1b_mlmc_asian.py`, `layer1b_kappa1.py` *(Q2: pricing)*
Exact MLMC coupling for rough-Bergomi Asian options (coarse = pairwise-summed
fine increments — exact because W̃ is a *pure function* of increments).
Measured β = {0.125, 0.226, 0.422, 0.721} at H = {0.05, 0.10, 0.20, 0.35}:
the pathwise bound β = 2H is *tight*, so naive MLMC loses to standard MC
(cost ratio ≈ 0.6× at ε = 0.025). The P2 estimator programme then tested
rescues: antithetic coupling **refuted** (variance factor 1.44× < cost factor
2.5/1.5), conditional MC **won** — as single-grid turbocharging, not
multilevel (std-MC/conditional-MLMC = 0.41–0.45). κ=1 exact near-cell
integration sharpens constants ~1.3–1.5×, can't change the rate (and its
MLMC coarse coupler is intentionally not wired — RVL-025).
Reproduce: `python layer1b_mlmc_asian.py --section 3 --quick --no-show`.
Issues: RVL-007, RVL-010, RVL-024, RVL-025, RVL-033.

### Layer 1c — `layer1c_roughness_audit.py`, `identifiability_map.py` *(Q1)*
Three H-estimators validated on clean paths (Rung 0), then attacked with the
corruption ladder (Rungs 1–5). Headline mechanisms: the RV proxy makes smooth
read rough (Rung 1 — the Cont–Das mirage, live in explainer demo 3); iid
microstructure noise fakes roughness while persistent noise fakes smoothness
(Rung 2 — bias *direction* depends on noise structure); jumps fake roughness,
bipower variation defends (Rung 3); MF-DFA fabricates roughness in short
samples (Rung 4); weekly seasonality splits the estimators up/down and
deseasonalising cures it (Rung 5). `identifiability_map.py` compiles it into
the P3 phase diagram over (vol-of-vol η, window Δ): identified / de-biasable
/ non-identified cells, with BTC/ETH/SPX placed — all outside the identified
region. Reproduce: `python layer1c_roughness_audit.py --section 1 --quick`.
Issues: RVL-003 (in the inversion), RVL-006, RVL-018, RVL-029.

### Phase B — `binance_data.py` → `kline_verifier.py` → `rv_series.py` → `estimate_h.py` → `interpret_h.py` *(Q1 on real data)*
The real-data arm: download + SHA-verify 7 years of Binance 1-minute klines
(BTC/ETH 2019–2025, in `data/spot/`), verify integrity (including the
2025-01-01 ms→µs timestamp switch — one canonical normaliser), build log-RV
series, run all three estimators with subwindow-stability uncertainty, then
de-bias against the matched simulation envelope. Verdict: GJR reads Ĥ ≈ 0.08
(stable, sampling-invariant) but the de-biasing inversion is **non-identified**
— crypto's vol-of-vol (η ≥ 1.5) forces the regime where rough and smooth
proxies coincide. `equity_data.py` adds the SPX leg via range-based variance.
Reproduce: `run_phaseb.md` is the runbook. Issues: RVL-008, RVL-015, RVL-020,
RVL-031, RVL-034.

### Layer 2 — `execution_alpha.py`, `execution_alpha_phase1.py` *(Q3)*
Execution environment (rough-Bergomi market, linear impact) + the classical
Almgren–Chriss closed-form baseline, validated to 0.7% against the analytic
frontier (gate G-X1). Phase 1 probe: a *causal* vol-reactive schedule
modulation — kill-switch fired: ~5 s.e. WORSE than AC at matched risk, no
edge growing with roughness, so deep RL (Phase 2) was never built. The first
run looked good and was a look-ahead artifact — caught by the built-in
causality gate. Issues: RVL-013, RVL-014.

### Layer 3 — `layer3_deep_hedging.py` *(Q5; isolated torch venv)*
Buehler-style deep hedging: neural policy, CVaR objective
(Rockafellar–Uryasev dual), self-computed path signatures, causality guard
(E[Σδ·dS] = 0 z-test). Gate 1: recovers Black–Scholes delta in the
frictionless limit. Gate 2 (the contrast): deep-vs-delta edge on a rough
market (H = 0.10) minus the same edge on a smooth control (H = 0.5, identical
generator) — generic edge +1.1 CVaR (8/8 seeds, both markets), roughness
increment +0.06 ± 0.04 (z = 1.4, null). D43's friction sweep hardened the
null: a z = 5.6 "emergence" at c = 0.02 dissolved under a 3000-epoch budget
(the DELTA_gap untrained control held exactly constant — the perfect
convergence control). ⚠️ Requires torch in `.venv-layer3` (not created on
this machine; Python 3.14 wheel availability unverified — RVL-035).

### Layer 4 — `rough_heston*.py`, `rough_kernel_soe.py`, `layer4_*.py`, `deribit_surface.py`, `calibrate_btc.py` *(Q4 + convergence)*
Four bricks, each gated before the next: (1) rough-Heston Volterra–Euler
simulator with Andersen-QE positivity (validated ν ≤ 0.20); (2) the
El Euch–Rosenbaum characteristic function via fractional Riccati +
Gil-Pelaez inversion — the simulation-free reference, anchored at H = ½
against closed-form Heston (which caught a factor-κ² coefficient error from a
secondary source — the "source-pinning" lesson, D30); (3) weak-order study:
α ≈ 1 for H ≥ 0.20, α ≈ 0.74 at H = 0.05, H = 0.10 honestly unresolved
(pre-asymptotic; paper P4); (4) the sum-of-exponentials Markovian lift
(O(N·n) vs O(n²)) — preserves prices and β, does NOT preserve weak order
(intrinsic noise-envelope perturbation, D35/D44), unlocks high vol-of-vol
(ν ≤ 0.40). Calibration engines (single smile → multi-maturity surface →
live Deribit): H non-identified via three independent diagnostics (flat
Jacobian eigen-direction, bootstrap railing, valley constancy), replicated
BTC → ETH (D42), plus a genuine model finding — rough Heston under-produces
crypto's crash-fear put tail. Issues: RVL-004, RVL-005, RVL-011, RVL-012,
RVL-021, RVL-026, RVL-027, RVL-028, RVL-030.

### The papers — `OVERLEAF/P1–P4`
- **P2** — *Turbocharged vs multilevel Monte Carlo for rough-volatility Asian
  options*: MLMC does not earn its place; conditional single-grid MC wins.
- **P3** — *When is volatility roughness identifiable?* The estimator audit +
  identifiability map + real BTC/ETH placement. **Target: SIURO** (needs a
  project advisor's letter — find a Salford academic sponsor).
- **P1** — *Calibration*: H non-identifiability from option surfaces, live
  BTC/ETH, + the put-tail model gap.
- **P4** — *Weak order*: α ≫ H (weak convergence survives roughness), the
  lift's honest failure, H = 0.10 honestly open.
Reading order: P3 → P2 → P1 → P4 (matches the layer story).

---

## Level 3 — the maths, gently

*Prerequisites you half-remember, restated: variance Var(X) = E[X²] − E[X]²;
for a Gaussian X ~ N(0, σ²), E[e^X] = e^{σ²/2}; a log-log plot turns power
laws y = c·x^a into straight lines of slope a; big-O(n^{−H}) means "shrinks
at least like n^{−H}".*

### 3.1 From Brownian motion to fBm
Brownian motion: Var(B_t) = t, independent increments. Fractional Brownian
motion B^H is the Gaussian process with
Cov(B^H_s, B^H_t) = ½(s^{2H} + t^{2H} − |t−s|^{2H}).
Check H = ½ recovers Cov = min(s,t) — Brownian. For H < ½, expand
E[(B^H_{t+Δ} − B^H_t)(B^H_{t+2Δ} − B^H_{t+Δ})] < 0: consecutive increments
anti-correlate — the jaggedness you see in demo 2. Paths are Hölder-H: the
smaller H, the rougher. *(Mandelbrot–Van Ness 1968 — the `mvn1968` Zotero
entry.)*

### 3.2 The Volterra construction and its discretisation
The Riemann–Liouville form W̃_t = √(2H)∫₀ᵗ(t−s)^{H−½}dW_s has
Var(W̃_t) = t^{2H} (integrate the kernel squared). Discretise on n steps:
W̃_i = √(2H)·Σ_j g_{i−j}·ΔW_j with g_m = (b_m·dt)^{H−½} at BLP's optimal
points b_m — a convolution, so FFT gives all n values in O(n log n)
(`volterra_weights` / `volterra_process`). Crucially the DISCRETE variance is
v_i = 2H·dt·Σ_{j≤i} g_j², which at usable n is visibly below t^{2H} (0.83 vs
1.0 at n = 128, H = 0.1 — rough kernels converge slowly). Every downstream
formula must use v_i, not t^{2H}. This one distinction separates the trusted
engine from the buggy Layer 1 (RVL-001/002) — measured live this session:
the wrong choice loses 20–65% of forward variance.

### 3.3 Rough Bergomi and the exact compensator
V_t = ξ₀·exp(η·W̃_t − ½η²·v_t). Since η·W̃_t ~ N(0, η²v_t), the Gaussian mgf
gives E[V_t] = ξ₀·e^{−½η²v_t}·e^{½η²v_t} = ξ₀ exactly — *an algebraic
identity, not an approximation*, provided the v in the exponent is the same v
the process actually has. The asset follows log-Euler:
dlogS = (r − ½V)dt + √V·(ρ dW₁ + √(1−ρ²) dW₂).

### 3.4 Monte Carlo and the Giles theorem
Plain MC at accuracy ε: need Var/N ≤ ε² samples AND a grid fine enough that
bias ≤ ε — total cost ~ ε^{−2−1/α}. MLMC telescopes
E[P_L] = E[P₀] + Σ E[P_ℓ − P_{ℓ−1}] with coupled pairs; with
V_ℓ ∝ 2^{−βℓ} and C_ℓ ∝ 2^{γℓ}, optimal cost ~ ε^{−2} if β > γ,
ε^{−2}log²ε if β = γ, ε^{−2−(γ−β)/α} if β < γ. Here strong error O(n^{−H})
⇒ β = 2H (variance = error²) — at H = 0.1, β = 0.2 ≪ γ = 1: the worst
regime, and the measured β table shows the bound is attained. The Asian
payoff's time-average can't help: the Volterra discretisation error is a
common factor along the path, and averaging doesn't cancel common factors.

### 3.5 Conditional Monte Carlo (the winner)
Condition on the variance path (equivalently the vol-driving Brownian W₁).
Given it, log S_T is Gaussian with known mean/variance (Romano–Touzi 1997) —
so the European part prices by Black–Scholes *inside* the simulation, and
only the variance path is simulated. Variance drops ~4.2× per grid. The
subtle P2 finding: it drops MLMC's level-difference variance *less* (~3.2×,
because coupling had already cancelled the shared part), so conditioning is
best used *instead of* MLMC, not inside it.

### 3.6 Rough Heston and the fractional Riccati
Rough Heston: V_t = V₀ + (1/Γ(H+½))∫₀ᵗ(t−s)^{H−½}[κ(θ−V_s)ds + ν√V_s dW_s].
Its characteristic function is exp(κθ·I¹ψ + V₀·I^{1−α}ψ) where ψ solves the
*fractional* Riccati D^α ψ = −½(u²+iu) + (iuρν−κ)ψ + ½ν²ψ², α = H+½
(El Euch–Rosenbaum 2019). Solved numerically by a fractional Adams
predictor–corrector (`_frac_riccati`); prices by Gil-Pelaez quadrature. At
H = ½ this collapses to classical Heston's closed form — the anchor gate that
catches coefficient errors (it caught θλ vs θ/λ, a 4× error, in D30). The
CF is the *non-circular* reference: validating a simulator against another
simulator only measures self-consistency; validating against the CF measures
truth.

### 3.7 Calibration as an inverse problem
Least squares: θ̂ = argmin Σ(IV_model(θ) − IV_mkt)². Near θ̂ the loss is
½(θ−θ̂)ᵀJᵀJ(θ−θ̂): the eigenvalues of JᵀJ price each direction's
identifiability. Single rough-Heston smile: cond(JᵀJ) ≈ 6×10⁵ with the soft
eigenvector ≈ pure H. Mechanism: ATM skew ≈ c·ν·T^{H−½} — one number
constraining a product. Multi-maturity term structure helps (~100× on cond,
clean data) but the H↔ν correlation ≈ −0.85 persists, and live crypto
surfaces rail H to the bound while fitting the smile to ~0.9 vol-points.
Diagnosis toolkit worth remembering: **flat eigen-direction + bootstrap
railing + equal-RMSE valley walk** = structural non-identifiability.

### 3.8 The estimation side: why Ĥ lies
GJR structure functions: m(q,Δ) = E|Δ_Δ log σ|^q ∝ Δ^{qH} — two nested
regressions give Ĥ. Feed it log RV = log(true IV × noisy multiplier): the
multiplier's i.i.d. noise dominates short-lag increments and drags the slope
down → spurious roughness (demo 3's red curve: smooth truth reads Ĥ = 0.12 at
window 64). The de-bias inversion (simulate the bias curve true-H → E[Ĥ] at
YOUR window/length, then invert) fails exactly when the curve is non-monotone
— which crypto's vol-of-vol guarantees. Model-conditional, and honest about
it: the answer "multivalued" is reported as such (`interpret_h.py`; RVL-003
notes the monotonicity check needs a trend bound).

---

## Explain it to someone

**30 seconds:** "There's a famous claim that market volatility is 'rough' —
statistically much more jagged than standard models assume. I work on a
research codebase that stress-tests that claim end to end: can you measure
the roughness, price with it, trade it, hedge with it? Four of the five
answers came out negative — carefully, reproducibly negative — which matters
because the measurement methods themselves turn out to manufacture roughness
out of nothing."

**5 minutes:** add: what H is (texture dial, 0.5 = coin-flip smooth, 0.1 =
markets' alleged value — show explainer demo 2); the mirage (smooth truth
reads Ĥ ≈ 0.12 through a realistic estimation window — demo 3); the flat
valley (a completely different H fits option prices equally well — demo 5);
and the discipline (predictions committed before running, negatives published
with mechanisms, four false positives caught by gates).

**30 minutes:** walk the five questions with the layer table (Level 2),
demo each claim in the explainer, sketch the Giles theorem and the β = 2H
bound on paper (Level 3.4), and finish with the papers P3 → P2 → P1 → P4 and
what's still open (the H = 0.10 weak order, the L1-1 fix, the SIURO
submission).

---

## Appendix — working on this repo (human or AI)

1. **Read `ROADMAP.md` first, every session.** It is the single source of
   truth; its decisions log is append-only — never rewrite history, append
   corrections (D44 is the model).
2. **Fix work comes from [`ERROR_AUDIT.md`](../../ERROR_AUDIT.md):** one
   issue = one commit; run the issue's acceptance test + the full suite;
   update the issue's Status; append a ROADMAP entry.
3. **Trust boundaries:** import paths from `roughvol_core.py`, never
   `layer1_rough_vol.py`; validate new CF code at the H = ½ anchor; keep
   torch inside `.venv-layer3` (the core stays numpy/scipy-only).
4. **Environment:** Windows, Python 3.14, `python -m pip` (pip not on PATH),
   set `PYTHONIOENCODING=utf-8` before running repo scripts (Greek glyphs).
   Baseline: 252 passed, 2 torch-skips, ~21 min.
5. **Reproducibility ethos:** every figure has a committed generating driver
   (D44 exists because one didn't); every Monte Carlo number ships a standard
   error; every comparison is matched-accuracy or matched-risk.
6. **References:** import `docs/references/roughvollab.bib` into Zotero;
   reading order in `docs/references/ZOTERO_SETUP.md`.

*Guide written 2026-07-02 (session: full deep-read + adversarial audit).
Companion: [explainer.html](explainer.html) · [ERROR_AUDIT.md](../../ERROR_AUDIT.md).*
