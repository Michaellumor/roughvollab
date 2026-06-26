# RoughVolLab — Roadmap & Project Memory

> **Read this file first, every session.** It is the single source of truth
> for project state, design decisions, and measured results. Chat history
> does not persist; this file does. Append to the decisions log every
> working session — never rewrite history.

---

## Programme

**Title:** Reinforcement Learning as a Numerical Approach to Stochastic
Optimal Control under Market Frictions.

**Mission:** a unified, pedagogically structured, publication-grade Python
platform for rough stochastic volatility — simulation, multilevel Monte
Carlo pricing, market frictions, and risk-aware RL hedging. Every module is
individually citable; every numerical claim in code or docs must be backed
by a run that actually happened.

**Author:** Michael Lumor, Department of Mathematics, University of Salford
(independent research programme; ORCID 0009-0000-0326-3891).

---

## Status board

| Module | Contents | Status | Last touched |
|---|---|---|---|
| `layer1_rough_vol.py` | fBm (Cholesky + hybrid), rBergomi/rHeston paths, Hurst estimation | ✅ complete — **1 known issue (L1-1)** | 2026-06-12 |
| `layer1b_mlmc_asian.py` | Coupled rBergomi engine, Giles rates, adaptive MLMC, β-vs-H study; opt-in antithetic / conditional / κ=1 estimator flags (P2) | ✅ v0.1 + P2 estimators, validated | 2026-06-23 |
| `layer1b_kappa1.py` | Exact near-cell (κ=1) Volterra module + coarse coupler (split + conditional resampling) | ✅ G-H1 / G-H2 pass | 2026-06-23 |
| `roughvol_core.py` | Shared tested rough-path engine (κ=0 Volterra) + `test_roughvol_core.py` | ✅ 18 tests pass | 2026-06-13 |
| `layer1c_roughness_audit.py` | Roughness-estimator audit. 3 estimators (§1–3) + corruption ladder Rungs 1–5 complete (RV proxy + envelope; microstructure noise + subsampling; jumps + bipower; finite-sample; calendar/day-of-week seasonality + deseasonalise) (+`test_layer1c.py`) | ✅ estimators + full ladder | 2026-06-20 |
| `identifiability_map.py` | Layer 1c capstone (P3): identifiability map over (η, Δ) — classifier + factorial sweep + phase diagram + per-asset η-calibration & placement (+`test_identifiability_map.py`) | ✅ 15 tests pass | 2026-06-21 |
| `paper_outputs.py` | Reproducibility — regenerates P3 figures (fig1 bias curves, fig2 map + asset overlay) + prints all paper numbers in one run | ✅ reuses tested modules | 2026-06-21 |
| `binance_data.py`, `kline_verifier.py`, `rv_series.py` | Phase B data layer: download + SHA-verify Binance klines → log-RV proxy (Rung-1 twin) | ✅ 66 tests pass | 2026-06-20 |
| `estimate_h.py`, `interpret_h.py` | Phase B analysis: 3 estimators + disagreement; de-bias vs matched Rung-1 envelope | ✅ 21 tests pass | 2026-06-20 |
| `equity_data.py` | Equity arm: free daily OHLC → Garman–Klass/Parkinson range log-variance (Rung-5 gap leg), pipeline-compatible | ✅ 6 tests; run on SPX | 2026-06-20 |
| `execution_alpha.py` | Execution env (rough-Bergomi) + Almgren–Chriss + naive baselines (G-X1) | ✅ Phase 0 validated | 2026-06-24 |
| `execution_alpha_phase1.py` | Execution kill-switch probe — causal vol-reactive heuristic | ✅ Phase 1 — kill-switch fired (negative) | 2026-06-24 |
| `layer2_frictions.py` | Almgren–Chriss + rough-market execution (spec: `layer2_piece1_gate_check.md`) | ✅ AC baseline built & validated in `execution_alpha.py` (G-X1, 0.7%) — dedicated `layer2_frictions.py` module not yet split out | 2026-06-24 |
| `layer4_convergence.py` | Convergence study, SPX calibration, diagnostics | 🔜 Still ahead — needs spec before code | — |
| `docs/gate_checks/` | Gate-check specs + recorded verdicts (index) | ✅ living | 2026-06-26 |
| `ROADMAP.md` | This file — project memory | living document | 2026-06-12 |

---

## Layer 1 — Stochastic simulation core (complete, 1 known issue)

Cholesky-exact fBm, BLP hybrid scheme, rBergomi and rough Heston path
simulation, Hurst estimation from realised variance. Four sections, figures
to `output/`.

### KNOWN ISSUE L1-1 — Volterra normalisation (logged 2026-06-12)

`fbm_hybrid` produces a process with **Var(B^H_1) ≈ 0.89** (measured,
n = 128, 3000 paths, H = 0.1), but `rough_bergomi_paths` uses the
compensator −½η²t^{2H}, which assumes Var = t^{2H} = 1. Consequence:
**E[V_t] ≈ 0.82·ξ₀ at η = 1.9** — an ~18% forward-variance bias. Note the
empirical 0.89 also disagrees with the un-normalised Riemann–Liouville
theory value 1/(2H·Γ(H+½)²) ≈ 2.25, so the hybrid implementation is
internally inconsistent too (suspect: double-subtraction in the
`smooth_corr` loop).

**Impact:** cosmetic for Layer 1's path plots; fatal for pricing — which is
why Layer 1b has its own verified Volterra engine and does **not** import
Layer 1's.

**Fix plan:** rewrite `fbm_hybrid` against the κ=0 weights in
`layer1b_mlmc_asian.volterra_weights` (vectorised, FFT), normalise so
Var(W̃_t) matches the discrete formula, and switch the rBergomi compensator
to the discrete variance. **Acceptance test:** empirical Var(W̃_T) within
MC noise of the discrete v_n; max_t |E[V_t]/ξ₀ − 1| within 3 s.e. at
N = 100k. (These are exactly Layer 1b §1's checks — reuse them.)

---

## Layer 1b — MLMC Asian pricing under rough Bergomi (v0.1 complete)

**File:** `layer1b_mlmc_asian.py` · four sections · figures to `output/` ·
flags `--section {1..4}`, `--no-show`, `--quick`.

**Engine:** κ=0 optimal-discretisation Volterra scheme (BLP family),
FFT convolution, pure function of Brownian increments → **exact MLMC
coupling** (coarse = pairwise-summed fine increments). Lognormal
compensator uses the **discrete** variance v_i = 2H·dt·Σg², so
E[V_t] = ξ₀ exactly at every level. Log-Euler asset, trapezoidal arithmetic
Asian call. Defaults: H=0.10, η=1.50, ρ=−0.70, ξ₀=0.04, S0=K=100, T=1,
r=0, n₀=32 (BFG SPX calibration ≈ (0.07, 1.9, −0.9) noted as the wilder
real-world regime).

**Validation (all passed, 2026-06-12):**
- Var(W̃_T): empirical 0.8496 vs discrete 0.8529 (continuum t^{2H} = 1.0 —
  the 15% gap at n=256 is rough-vol slow convergence, and is why the
  discrete compensator matters).
- Forward variance: max_t |E[V_t]/ξ₀ − 1| = 0.034, within MC noise (~0.039).
- η=0 European vs Black–Scholes: z = 0.53.
- Telescoping consistency check < 0.51 on every run.

**Measured results (N = 20k main / 12k sweep, L = 5–6):**

| H | measured β | pathwise bound 2H |
|---|---|---|
| 0.05 | 0.125 | 0.10 |
| 0.10 | 0.226 | 0.20 |
| 0.20 | 0.422 | 0.40 |
| 0.35 | 0.721 | 0.70 |

- **β tracks 2H across the roughness spectrum — the pathwise bound is
  tight.** The Asian time-average buys no extra variance decay: the Volterra
  strong error is a slowly-decaying common factor the average cannot cancel.
- α is **unmeasurable** at this N (|E[Y_l]| ≈ 0.01–0.04 vs s.e. ≈ 0.02);
  level bias is already small at n₀ = 32. The code flags noise-dominated
  regressions explicitly.
- Adaptive MLMC (Giles): ε = 0.2 → L=2, cost 7.1e5; ε = 0.1 → L=3, 3.1e6;
  ε = 0.05 → L=7, 1.6e8; ε = 0.025 → L=7, 6.4e8. Price ≈ 4.17–4.20
  (ATM Asian; BS European anchor 7.97).
- **Honest headline: naive coupled MLMC loses to standard MC here**
  (cost ratio ≈ 0.6× at ε = 0.025) — the expected β < γ pathology. This
  negative result is the point: it quantifies why rough volatility needs
  specialised estimators, and seeds publication P1 below.

**Extensions (ordered):**
1. **Antithetic coupling** (Giles–Szpruch style) adapted to the Volterra
   convolution — target: β > 2H empirically.
2. **Conditional MC / turbocharging** (McCrickerd–Pakkanen) as a control
   variate inside each level.
3. κ=1 hybrid coupling (needs the correlated nearest-cell Gaussian to be
   coupled across levels — non-trivial; design before coding).
4. Rough Heston MLMC (reuse the driver; new kernel).
5. Large-N weak-rate study to actually measure α (≥10⁶ samples/level).
6. MLMC for risk measures (nested simulation / market-risk direction —
   PhD-relevant).

---

## Layer 1c — Roughness-estimator audit (specced 2026-06-13, not started)

**File:** `layer1c_roughness_audit.py` (house style: sections, `--section`,
`--no-show`, `--quick`; figures to `output/`).

**Goal:** quantify how reliably the Hurst exponent can be estimated from
realized-volatility data, using Layer 1b's *verified* simulator as ground
truth — then apply only the audited, bias-corrected estimators to real
crypto and equity data, with honest uncertainty. This joins the live
methodological debate (Gatheral–Jaisson–Rosenbaum's H ≈ 0.1 vs the
Cont–Das "spurious roughness" critique) at the level where an
undergraduate-led project can genuinely contribute: careful operating
characteristics, open code, reproducible pipeline. Statistics on real
data — the QR-relevant gap in the portfolio.

**Positioning (prior art, read before coding):** GJR (2018) log-RV
regression and its simulation appendix; Takaishi (2019) MF-DFA roughness
of Bitcoin; Cont & Das (2024) normalised p-variation estimator + the
proxy-error mechanism for spurious roughness; Fukasawa–Takabatake–
Westphal quasi-likelihood approach ("Is volatility rough?"); arXiv
2507.00575 (Bitcoin RV multifractality — argues against a single H);
arXiv 2511.03314 (finite-sample dependence of measured H). The basic
"is BTC vol rough?" question is taken and contested — the audit angle is
the contribution.

### Phase A — simulation-grounded audit

**Estimators (core three, two stretch):**
1. GJR structure-function regression: m(q, Δ) = E|log σ_{t+Δ} − log σ_t|^q
   regressed on log Δ for q ∈ {0.5, 1, 1.5, 2, 3}; H from slope/q;
   monofractality check via linearity of ζ_q in q.
2. MF-DFA generalized Hurst h(q) (Takaishi's tool) — doubles as the
   multifractality diagnostic.
3. Cont–Das normalised p-variation roughness index.
4. *(stretch)* Fukasawa–Takabatake–Westphal quasi-likelihood.
5. *(stretch)* Wavelet (Abry–Veitch) estimator.

**Ground-truth generators** (the design's most important element — the
audit MUST include non-rough nulls, or it cannot detect spurious
roughness):
- Rough truth: RFSV log-vol via the Layer 1b κ=0 Volterra engine,
  H ∈ {0.05, 0.10, 0.20, 0.30, 0.45}.
- Smooth null: Markovian log-OU stochastic vol (effective H = 1/2).
- Long-memory null: fractional log-vol with H = 0.7.
- *(optional)* regime-switching vol as a "neither" null.

**Corruption ladder** (every estimator × every truth, rung by rung):
- Rung 0 — oracle: true spot σ on the grid. Sanity tier.
- Rung 1 — RV proxy: σ unobserved; daily RV from n intraday returns
  (n ∈ {26, 78, 288} ≈ 15-min/5-min/1-min on a 24/7 calendar). This is
  the Cont–Das spurious-roughness mechanism.
- Rung 2 — microstructure noise: iid and AR(1) noise on log-prices
  before RV; noise-to-signal γ ∈ {0, 0.5, 1, 2}; subsampled RV as
  mitigation.
- Rung 3 — price jumps: compound Poisson; bipower variation
  (Barndorff-Nielsen–Shephard) as the jump-robust RV alternative.
- Rung 4 — finite sample: T ∈ {250, 1000, 2500} daily RV observations;
  reproduce the direction of the known H-vs-sample-size effect.
- Rung 5 *(simulated version DONE 2026-06-20; real-data leg remains)* — calendar
  effects. Two distinct things hide under this name. **(a) The CONTROLLED
  artefact** — a deterministic day-of-week SEASONALITY in volatility — is now
  built (`rung5_calendar` + helpers `add_weekly_seasonality` / `deseasonalize`,
  +2 tests): inject a growing weekly cycle into known-H log-vol, measure the
  bias, test the deseasonalise mitigation. Result: GJR biases UP (smoother),
  MF-DFA DOWN (rougher), Cont–Das eventually breaks — the SAME estimator-
  dependent sign-split as Rungs 2–3; the bias grows with amplitude and
  deseasonalising (subtract the period-7 mean cycle) removes it cleanly — a
  REMOVABLE artefact, unlike finite-sample. Crypto's weak day-of-week amplitude
  ⇒ minimal bias ⇒ the Phase B BTC/ETH reading is clean of calendar effects.
  **(b) The DATA-STRUCTURE artefact** — real overnight/weekend GAPS (equity
  5-day, real NYSE hours) vs 24/7 crypto, the natural experiment — still belongs
  with REAL data (simulating invented gaps would be circular and weak) and needs
  the equity arm; it is the remaining leg. So the simulated ladder is now
  complete through Rung 5's controlled seasonality; the gapped equity-vs-crypto
  comparison is Phase-B data work, not a simulated rung.

**Protocol:** ≥500 Monte Carlo replications per cell; core grid =
truths × core estimators × rungs 0–4 at one realistic setting each,
plus targeted sensitivity sweeps. Report bias, RMSE, and 95% CI
coverage (asymptotic CIs where the method provides them; moving-block
bootstrap otherwise). Batched generation per Layer 1b conventions (D8).

**Phase A validation criteria (all must pass before Phase B):**
- Rung 0, T = 2500: every core estimator recovers known H with
  |bias| < 0.01 and CI coverage in [93%, 97%] — else implementation bug.
- Reproduces three known qualitative results: (i) RV-proxy error inflates
  apparent roughness on smooth/long-memory nulls (Cont–Das artefact);
  (ii) measured H falls as the sampling period coarsens (finite-sample
  effect); (iii) GJR regression on clean rough spot vol recovers H
  (their simulation appendix).
- Deliverable: an operating-characteristics table — for each estimator,
  the corruption regimes where |bias| < 0.05 ("trustworthy zone").

### Phase B — real data (only with audited estimators)

**STATUS (2026-06-20): CRYPTO ARM COMPLETE.** Pipeline built and tested
(`binance_data` → `kline_verifier` → `rv_series` → `estimate_h` → `interpret_h`,
87 tests; runbook `run_phaseb.md`). Analysis run on BTCUSDT + ETHUSDT, 2019–2025,
2,557 daily obs. **Finding (full write-up: `PHASE_B_FINDINGS.md`):** the apparent
ultra-roughness of crypto vol (GJR Ĥ ≈ 0.08) is real and sampling-invariant but
**NOT IDENTIFIED** as a property of the latent volatility — seen only by the
model-assuming estimator (Cont–Das cannot resolve, MF-DFA unphysical),
microstructure ruled out by the sweep, and de-biasing non-identified at the
data-calibrated vol-of-vol (η ≥ 1.5, where rough ≡ smooth through the proxy).
Empirical vindication of the Cont–Das/Rogers artefact position. Methodological
note: planned bootstrap CIs were replaced by **sub-window stability** — a moving-block bootstrap would shred the long-range dependence being measured. Still open
(not load-bearing for the finding): jump-robust (bipower) re-run; deeper equity
coverage (the first leg is done — SPX range-variance run 2026-06-20, Ĥ≈0.13,
also non-identified — see the equity decisions-log entry; more tickers would
test stability); Hayashi–Yoshida / Epps refinement for asynchronous ticks; Rung
5's real gap leg is now done (SPX-vs-crypto). The spec below is retained as the
original plan.

- Crypto: BTC, ETH (+ one liquid alt) from public exchange kline APIs,
  1-min bars, full history; 5-min RV with subsampling; bipower variant.
- Equity benchmark: honest sourcing required — the Oxford-Man realized
  library is no longer updated; use archived OMI data if obtainable,
  else range-based estimators (Parkinson / Garman–Klass) on free OHLC,
  clearly labelled as lower-fidelity.
- Analysis: apply only estimators inside their trustworthy zone;
  Ĥ with bootstrap CIs; ζ_q multifractality diagnostic; rolling-window
  stability; the Ĥ-vs-Δ signature plot (the debate's key figure).
- Honest framing: the deliverable is *what can be concluded at stated
  confidence*, not a verdict. "Indistinguishable from proxy artefact"
  is an acceptable conclusion (working agreement §4).
- Data handling: raw downloads cached under `data/` (gitignored);
  small processed RV series (CSV, few MB) committed for
  reproducibility, with a note on exchange API terms.
- **Irregular / random observation times (captured 2026-06-20, ML)** — real
  ticks do NOT arrive on a regular clock: they are generated by a point
  process (Poisson / Hawkes), clustered in busy periods and sparse in quiet
  ones, and across assets they are *asynchronous*. The "every k-th tick"
  subsampling of Rung 2 quietly assumes a synchronous regular grid that real
  data violates. Forcing irregular ticks onto a common grid itself introduces
  bias (the **Epps effect**). Confront this in Phase B with real ticks in
  hand; consider **Hayashi–Yoshida**-style estimation, designed for
  asynchronous, irregularly-spaced observations without grid imposition.
  Likely a FOUNDATIONAL refinement to the observation model touching all
  rungs, not a single new rung — so it belongs here at the real-data stage,
  not as a simulated ladder rung. (Connects to the same point-process maths
  as the Rung 3 jumps.)

**Module alignment:** Year 2 Statistics + Numerical Analysis; Year 3
Mathematical Statistics (estimator theory: bias, consistency, coverage)
+ Programming & Optimisation.

**Dependencies:** AFTER the L1-1 fix and the pytest/CI milestone — the
audit inherits its credibility from a tested simulator. Sequence per
D10.

---

## Layer 2 — Market frictions (spec)

**Goal:** non-linear execution frictions under rough vol; demonstrate the
breakdown of Markovian dynamic programming.

**Contents:** Almgren–Chriss temporary/permanent impact with closed-form
classical solutions as anchors; rough execution slippage (impact driven by
the rough variance path); empirical demonstration that conditioning on path
history changes the conditional law (non-Markovianity) — this motivates
Layer 3's signature features.

**Validation criteria:** reproduce Almgren–Chriss optimal trajectories
analytically in the classical limit; quantify Markov-projection error vs a
path-dependent benchmark.

**Key refs:** Almgren & Chriss (2001); Gatheral, Jaisson & Rosenbaum (2018).

## Layer 3 — RL hedging engine (spec)

**Goal:** risk-aware deep hedging on the non-Markovian state via path
signatures.

**Contents:** truncated signature features of (t, S, realised-var) path;
actor–critic with CVaR-sensitive objective; baselines: BS delta,
delta-vega, and the Layer 2 friction-aware strategies.

**Validation criteria:** recover BS delta (η→0, frictionless) within
tolerance; beat delta hedging on CVaR of terminal P&L under frictions with
statistical significance across seeds.

**Key refs:** Buehler et al. (2019); signature methods (Lyons; Kidger &
Lyons for ML practice).

## Layer 4 — Convergence & calibration (spec)

**Goal:** consolidated convergence study (weak/strong rates across layers)
and SPX smile calibration of rBergomi; reproduce BFG-style fits;
end-to-end reproducibility harness.

**Validation criteria:** calibrated (H, η, ρ) in the published
neighbourhood; documented seeds; one-command reproduction of every figure.

---

## Conventions (all layers)

- File pattern: `layerN_*.py`, self-contained, `--section k`, `--no-show`,
  `--quick`; banner prints; sections named `sectionK_*`.
- Palette: TEAL `#1D9E75`, PURPLE `#7F77DD`, CORAL `#D85A30`,
  GRAY `#888780`, AMBER `#BA7517`. White background, no top/right spines,
  grid alpha 0.25, font 11.
- `np.random.seed(42)` at import; per-experiment `default_rng(seed)`.
- Runtime figures regenerate into `output/`; the copies committed for the
  README live flat at the repo root. dpi 150.
- Heavy loops vectorised over paths; batch size 5000 to cap memory.
- Every printed rate comes with its noise context (s.e. or an explicit
  noise-domination warning). No naked regressions.

---

## Decisions log (append-only)

**2026-06-12**
- **D1** Layer 1b uses the κ=0 optimal-discretisation scheme, *not* κ=1:
  the process is then a pure function of Brownian increments, making the
  MLMC coupling exact by pairwise summation. κ=1 deferred (extension 3).
- **D2** Lognormal compensator uses the **discrete** variance
  v_i = 2H·dt·cumsum(g²), not t^{2H}: forward variance exact at every
  level; kills the bias class that broke Layer 1 (L1-1).
- **D3** MLMC driver floors rates at α ≥ 0.5, β ≥ 0.1 on *both* the
  regression and the externally-supplied path. Bug story: a
  noise-dominated regression returned α = −0.01, making 2^α − 1 < 0 and
  the bias test pass vacuously (L stuck at 2). Floors follow Giles'
  practice.
- **D4** ε schedule [0.20, 0.10, 0.05, 0.025] (quick: drop the last) —
  affordable in the β < γ worst regime.
- **D5** README + CITATION reframed from "PhD programme at
  Manchester/Oxford" (false, reputationally dangerous) to **independent
  research programme, University of Salford** (true, and a stronger story).
  Broken Zenodo DOI badge removed; DOI to be minted at first tagged
  release.
- **D6** `requirements.txt` added (numpy ≥1.24, scipy ≥1.10,
  matplotlib ≥3.7) — README referenced it but it didn't exist.
- **D7** (amended) Figures are committed flat at the repo root
  (`layer1b_beta_vs_H.png` etc.) — GitHub web upload flattens folders.
  Runtime figures still regenerate into `output/` locally.
- **D8** Per-level batch size capped at `2_560_000 // n_f` samples
  (floor 200): a fixed batch of 5000 at n = 4096 spiked peak RAM to ~2–3 GB
  and got the process OOM-killed. The cap keeps peak memory flat as levels
  deepen; estimator statistics are unaffected.
- **D9** GitHub web uploads (per-file downloads + manual renaming)
  cyclically scrambled names↔contents across all repo files and dropped
  `layer1_rough_vol.py`; repaired by re-uploading the full verified set in
  one commit. Adopted flat root layout. Future commits should go through a
  local git clone or Claude Code — file names and contents then travel
  together by construction.

**2026-06-13**
- **D10** Layer 1c specced: a roughness-estimator audit chosen as the
  next research focus over any new standalone project. Rationale: fills
  the portfolio's empirical-statistics gap (QR-aligned), joins the live
  GJR vs Cont–Das estimator debate where the verified Layer 1b simulator
  is a genuine advantage, aligns with Year 2/3 statistics modules, and
  adds no new repo. Agreed sequence: Year 2 marks → L1-1 fix +
  pytest/CI → Layer 1c → P3 arXiv note + JOSS submission. Marks remain
  first; one focus at a time.
- **D11** *(2026-06-13)* Started Layer 1c the disciplined way
  (extract-and-test first). Created `roughvol_core.py` — the single
  trusted rough-path engine, lifted verbatim from validated L1b and
  pinned by `test_roughvol_core.py` (18 tests). Two are L1-1 regression
  guards: empirical variance must match the discrete formula, and forward
  variance E[V_t]=ξ₀ must hold — the bug the clobber reintroduced cannot
  return silently again. Added `rough_log_variance_paths` (returns the
  log-vol path, the object estimators consume). Edge case found + fixed:
  at H=0.5 the κ=0 weight exponent a→0 divides by zero; handled as the
  flat Brownian kernel — which is also Layer 1c's smooth null. First
  estimator (GJR structure-function) built with a Rung-0 oracle gate;
  passes for H∈{0.05,…,0.70} (`test_layer1c.py`, +6 tests). **Calibration
  finding:** the spec's flat "|bias|<0.01 for every estimator" is NOT
  achievable for GJR even on clean spot-vol paths — it carries a
  systematic positive finite-lag bias growing as H→0 (≈+0.06 at H=0.05,
  ≈+0.006 at H=0.3, essentially unbiased for H≥0.3). This is a real
  estimator property, not a bug; the gate is now per-regime
  (`ORACLE_TOLERANCE`), and quantifying this bias is itself part of the
  audit's contribution. Figure: layer1c_oracle_gate.png.
- **D12** *(2026-06-18)* Built Layer 1c §2 — the Cont–Das normalised
  p-variation estimator (`pvariation_hurst`) — through the same Rung-0
  oracle gate, the §1 way (probe → build → validate → test → log).
  Mechanism: sweep power p; the p-variation V_p(s) ∝ s^{1−pH} has scaling
  exponent (1−pH) that crosses zero at the critical p* = 1/H, so H = 1/p*.
  Model-free by construction (does NOT assume fBm) — exactly why Cont & Das
  built it: to referee the roughness claim without the circularity of
  presupposing a rough model. Gate passes for H∈{0.05,…,0.45}
  (`test_layer1c.py`, +5 tests → 29 total). **Headline finding:** the
  p-variation estimator carries the SAME bias signature as GJR — positive,
  growing as H→0 (≈+0.07 at H=0.05, ≈+0.009 at H=0.3, essentially unbiased
  by H=0.45). Build probe established it is PART finite-sample (bias at
  H=0.05 shrank +0.089→+0.068 going n=4096→8192) and PART intrinsic to the
  rough regime. That two *independent* estimators — one assuming fBm (GJR),
  one model-free (Cont–Das) — agree small-H roughness is hard to measure
  precisely is itself evidence relevant to the fact-or-artefact debate, and
  strengthens publication seed P3. Figure overlays both biases:
  layer1c_pvariation_gate.png.
- **D13** *(2026-06-18)* Built Layer 1c §3 — the MF-DFA estimator
  (`mfdfa_hurst`) — through the same Rung-0 oracle gate, completing the
  three core estimators. Steps: profile (cumulative sum) → non-overlapping
  windows → detrend each (fit and SUBTRACT an order-1 polynomial; the trend
  is removed, not measured) → q-th order fluctuation F_q(s) → slope is the
  generalised Hurst h(q); since the profile integrates the ≈fBm log-vol,
  H = h(2) − 1. Gate passes for H∈{0.05,…,0.45} (`test_layer1c.py`, +6
  tests → 35 total). **Headline finding — sharper than §2's:** MF-DFA's
  bias runs OPPOSITE to GJR and Cont–Das. Those two OVER-estimate roughness
  at small H (positive bias); MF-DFA UNDER-estimates (negative bias, ≈−0.023
  at H=0.05 → −0.009 at H=0.45), and its bias is INTRINSIC (barely changes
  with n, unlike the partly-finite-sample bias of the other two). That three
  independent estimators disagree even in the SIGN of their small-H bias is
  strong evidence that small-H roughness measurements are estimator-
  dependent — a sharper point for the fact-or-artefact debate than mere
  agreement, and a stronger spine for publication seed P3. Figure overlays
  all three biases: layer1c_mfdfa_gate.png. Three core estimators now
  validated; corruption ladder (Rung 1 = RV proxy first) is the next arc.
- **D14** *(2026-06-19)* Built corruption ladder **Rung 1 — the RV proxy**
  (`rung1_rv_proxy`, `realized_log_variance`), the decisive test of the
  Cont–Das mirage. Spot volatility is unobservable; the proxy estimates it
  as log realized variance over windows of high-frequency returns. **The
  decisive design (settled at the pre-build gate): corrupt a SMOOTH (H=0.5)
  null** — if the estimators then report rough H, the roughness is purely a
  proxy artefact, since the truth has none. **Headline result:** they do.
  Control first — on the TRUE smooth volatility all three estimators
  correctly read ≈0.5 (GJR 0.51, Cont–Das 0.48, MF-DFA 0.49), proving the
  estimators are innocent. Then through the RV proxy at window=32, the SAME
  smooth process reads GJR 0.16 / Cont–Das 0.05 / MF-DFA 0.02 — i.e. the
  empirical H≈0.1 signature, **manufactured entirely by the proxy.** Added
  nuance (window sweep): the artefact's severity is set by the RV window —
  small windows (noisier proxy) produce severe spurious roughness, large
  windows recover toward 0.5. So the mirage is real *and* its magnitude is a
  function of the sampling choice — a sharper, actionable finding than a
  flat "the proxy fools everything." 3 tests (control, smoking gun, window-
  dependence) → 38 total. Bug caught by RUNNING (not assuming):
  `rough_bergomi_paths` was unimported — fixed. Robustness: `_safe_estimate`
  wrapper handles degenerate (very small window) proxies that break the
  p-variation zero-crossing, returning nan rather than crashing — itself
  information (the proxy can be too corrupt to resolve). Strongly advances
  publication seed P3 (this is arguably its centrepiece result). Figure:
  layer1c_rung1_rvproxy.png. Next rungs: microstructure noise (R2),
  jumps (R3 — use the captured fractional-jump-diffusion controlled null),
  finite-sample (R4).
- **D15** *(2026-06-19)* Extended Rung 1 with the **bias envelope**
  (`rung1_bias_envelope`) — sweeping the TRUE H across the full validation
  spectrum (0.05–0.70) through the RV proxy at two windows (32 noisy, 128
  cleaner), mapping estimated-vs-true H. Where the smooth null PROVES the
  artefact exists, the envelope CHARACTERISES it. **Headline (stronger than
  expected):** at the noisy window the GJR estimate spans only ≈0.09 across
  the entire true-H range — i.e. the proxy COLLAPSES the estimate toward
  Ĥ ≈ 0.1 almost regardless of the true roughness. A market reading of
  Ĥ ≈ 0.1 is therefore nearly uninformative about the true H. This exposes
  an **observational-equivalence problem** in the empirically-relevant
  ultra-rough zone: a genuinely smooth process dragged down to 0.1 and a
  genuinely rough process are indistinguishable through a noisy proxy
  (Michael's framing, articulated at the design gate). Crucially, the
  cleaner window (128) partly recovers the diagonal — so the collapse is a
  function of the SAMPLING CHOICE, not inevitable, tying back to the
  window-dependence finding. The shaded Ĥ≈0.1 band in the figure marks where
  true and spurious roughness cannot be separated. Honest caveat baked into
  the framing: the smooth null remains the CLEAN evidence (artefact provable
  there); the envelope's rough end is interpretively murky BY DESIGN, and
  that murkiness is itself the finding — it explains why the fact-or-artefact
  debate is so hard to settle where it actually lives. +2 tests (collapse at
  noisy window, recovery at cleaner window) → 40 total. Figure:
  layer1c_rung1_envelope.png.
- **D16** *(2026-06-19)* Built corruption-ladder **Rung 2 — microstructure
  noise** (`add_microstructure_noise`, `rung2_microstructure`,
  `realized_log_variance_subsampled`). Where Rung 1 corrupts via the
  finite-sample proxy math, Rung 2 poisons the OBSERVED PRICE itself:
  Y_t = X_t + η_t, so ΔY = ΔX + η_t − η_{t-1} carries an MA(1) structure
  with negative autocorrelation ≈ −σ²_η (the bid-ask-bounce signature).
  **Direction settled by probe (and it overturned the tempting intuition):**
  a first reasoning said iid noise → H≈0.5; the probe falsified this. Because
  the noise enters through DIFFERENCING-then-SQUARING, it becomes
  anti-persistent in the RV series, which reads as ROUGHNESS — so noise drags
  Ĥ DOWN toward 0, not up. Confirmed: GJR on a true H=0.1 path fell
  0.131 → 0.008 as γ (noise-to-signal) went 0 → 2; same downward pull on a
  smooth H=0.5 null (0.153 → 0.002). So Rung 2 manufactures spurious
  roughness like Rung 1, but via a DIFFERENT mechanism (noise-induced
  negative autocorrelation vs. estimation chatter) — the two COMPOUND.
  **Mitigation — subsampled RV:** the noise is tick-to-tick independent but
  the signal persists, so taking every k-th tick dilutes noise relative to
  signal and recovers the estimate (GJR climbs 0.009 → 0.042 subsampling 1→4
  at γ=2). Honest caveat: at this EXTREME γ=2 the recovery is only partial;
  at moderate γ subsampling restores the estimate much more fully. Bug caught
  by RUNNING (not assuming): first subsampling impl shrank the RV window with
  the subsample (`window//step`), which destroyed the estimate and made the
  mitigation look ineffective — fixed to keep the window fixed, after which
  recovery is clear. +3 tests (downward direction, smooth-path corruption,
  subsampling recovery) → 43 total. Figure:
  output/layer1c_rung2_microstructure.png. Next: jumps (R3 — fractional
  jump-diffusion controlled null), then AR(1) noise variant, finite-sample
  (R4).
- **D17** *(2026-06-20)* Built corruption-ladder **Rung 3 — price jumps**
  (`add_compound_poisson_jumps`, `bipower_log_variance`, `rung3_jumps`).
  Question: can the estimators tell true fractal roughness from jump noise?
  A jump is a LOCAL singularity (Hölder exponent 0 at one instant);
  roughness is GLOBAL. Through a finite window both inject extreme small-
  scale variation, so estimators suffer an IDENTIFICATION FAILURE. Controlled
  null: SMOOTH (H=0.5) base + compound-Poisson jumps, so any roughness is
  purely the jump mirage. **Result — baseline prediction CONFIRMED, competing
  prediction NOT:** at the pre-build gate two directions were predicted —
  (baseline) independent jumps → Ĥ DOWN via point-singularity flattening, and
  (competing) clustered jumps → Ĥ UP via persistence/variance-plateaus. The
  probe confirmed the baseline (jumps drag GJR to −0.02, MF-DFA to −0.33 on a
  smooth null — the mirage) but the competing case did NOT appear: clustered
  jumps ALSO collapse downward (GJR −0.02). So for PRICE jumps the downward
  mirage dominates; the upward effect, if real, needs conditions not reached
  (jumps in volatility, or true self-excitation) — logged honestly as a
  tested-but-unconfirmed hypothesis. **Mitigation — BIPOWER VARIATION**
  (Barndorff-Nielsen–Shephard): instead of squaring returns (one jump²
  dominates), it pairs ADJACENT |returns| |r_t|·|r_{t-1}|; because jumps are
  isolated, a jump-return is multiplied by its CLEAN neighbour and stays
  bounded. Recovers GJR −0.02 → 0.06 (partial — reduces, not erases, jump
  sensitivity). +3 tests (jump mirage, bipower recovery, clustered-also-down)
  → 46 total. Gate note: this rung had a CORRECTLY-CALIBRATED prediction —
  baseline right, competing held open and ruled out by probe (contrast Rung
  2, where the first prediction was wrong). Figure:
  output/layer1c_rung3_jumps.png. Next: finite-sample (R4), then AR(1) noise
  variant; then Phase B (real data).
- **D18** *(2026-06-20)* Built corruption-ladder **Rung 4 — finite sample**
  (`rung4_finite_sample`), completing the ladder's core. Unlike Rungs 1–3
  this rung does NOT poison the data — it is clean and matches true H; the
  corruption is EPISTEMOLOGICAL (forcing an asymptotic estimator into a
  truncated timeline). On clean True H=0.1 paths, sweep T ∈ {8000…250} and
  measure each estimator's bias. **Result — an honest ESTIMATOR-DEPENDENT
  split, with a bold prediction partly confirmed and partly falsified:** at
  the gate the prediction was (a) Ĥ shifts UP as T shrinks, and (b) a
  striking claim — finite-sample bias is "exclusively an upward gravity
  well, structurally impossible to fabricate false roughness." The probe
  (plus a disentangling probe separating finite-sample effect from baseline
  bias) found: GJR and Cont–Das carry a roughly CONSTANT upward bias (bias
  change ≈ −0.003/−0.005 from T=8000→250 — essentially NO finite-sample
  effect), and never read below true H — so the claim HOLDS for them. But
  MF-DFA shows a GENUINE finite-sample push DOWN (bias −0.02 → −0.08 as T
  shrinks) and reads BELOW true H at small T — so the claim FAILS for
  MF-DFA, which fabricates extra roughness from small samples. So finite-
  sample bias is estimator-dependent: regression-on-spot-vol estimators
  (GJR) resist it; large-scale-aggregating estimators (MF-DFA) are
  vulnerable. Crucially, unlike Rungs 1–3 there is NO mitigation — financial
  history is finite. This bears directly on anyone measuring H≈0.1 from a few
  years of daily data. Honest note: the "upward, can't-fake-roughness" claim
  was elegant and DIRECTIONALLY right (dominant effect is upward for most
  estimators) but NOT universal — logged as partially falsified. +4 tests
  (GJR T-stable, MF-DFA drifts down, claim holds for GJR, claim breaks for
  MF-DFA) → 50 total. This completes the Rung 2/3/4 arc of prediction-vs-
  evidence: Rung 2 wrong→corrected, Rung 3 right→confirmed, Rung 4 partly
  right→refined. Figure: output/layer1c_rung4_finitesample.png. Remaining
  before Phase B: optional AR(1) noise variant, optional Rung 5 (calendar).
- **D19** *(2026-06-20)* Built the **AR(1) persistent-noise variant**
  (`rung2_ar1_noise`; `add_microstructure_noise` upgraded with a φ
  parameter) — the parked Rung 2 Option 2. Where Rung 2's core iid noise is
  memoryless (→ MA(1) negative autocorrelation → spurious ROUGHNESS, Ĥ down),
  real frictions are often PERSISTENT (stale quotes, VWAP/TWAP child-order
  pressure, slow liquidity replenishment), modelled as AR(1):
  η_t = φ·η_{t-1} + shock. Controlled comparison (single variable = noise
  persistence): fix smooth H=0.5, RV window, γ=1.0; sweep φ. **Prediction
  CONFIRMED (a second clean confirmation):** the downward push weakens at low
  φ and REVERSES upward as φ grows — GJR 0.02 (φ=0, iid baseline) → 0.13
  (φ=0.95), a lift of +0.115; MF-DFA −0.36 → −0.03. So iid noise fakes
  ROUGHNESS while persistent noise fakes SMOOTHNESS — the DIRECTION of the
  microstructure artefact depends on the noise's temporal structure. Deepens
  the fact-or-artefact problem: not even the sign of the bias is fixed
  (frictions can manufacture either illusion). +1 test (persistence lifts Ĥ)
  → 51 total. Figure: layer1c_rung2_ar1.png. Remaining before Phase B:
  optional Rung 5 (calendar effects).

- **2026-06-20 — Phase B crypto arm built and analysed; the headline finding.**
  Built the real-data pipeline (downloader+verifier+RV-series+estimator-runner+
  de-biaser, 87 tests, runbook). Ran it on BTCUSDT+ETHUSDT 2019–2025 (2,557 daily
  obs each). Result chain: (1) GJR reads Ĥ ≈ 0.08 (BTC) / 0.07 (ETH), clean
  monofractal, but Cont–Das returns nan and MF-DFA returns negative/unphysical at
  every sampling — only the model-assuming estimator sees roughness. (2) Sampling
  sweep 1m/5m/15m is flat (GJR even ticks *up* at 1m), which RULES OUT
  microstructure noise (it has the opposite signature). (3) De-biasing against a
  matched Rung-1 envelope is NON-IDENTIFIED: observed below the model floor at 5m,
  bias curve non-monotone (rough≡smooth) at 30m. (4) Robustness/calibration:
  matching the model's log-RV variability to the data's (sd≈1.13) forces η≥1.5 for
  every H, and at η≥1.5 the verdict is non-identified — so it does NOT rest on an
  assumed η. Conclusion: crypto vol's latent roughness is not identifiable from
  RV-proxy data — empirical Cont–Das/Rogers. Bug caught + fixed mid-analysis: the
  de-bias tool assumed a monotone bias curve and reported a false "true H = 0.43"
  on the non-monotone noisy-proxy curve; now detects non-monotonicity and reports
  NON-IDENTIFIED. Write-up: PHASE_B_FINDINGS.md. Open (optional): bipower re-run,
  equity arm, Hayashi–Yoshida/Epps, Rung 5.

- **2026-06-20 — Rung 5 (calendar effects): controlled simulated version built.**
  Closed the long-deferred Rung 5 with its controlled-artefact half — a
  deterministic day-of-week seasonality. `rung5_calendar` + helpers
  `add_weekly_seasonality` / `deseasonalize` in layer1c, +2 fast tests. Finding:
  a deterministic weekly cycle biases the estimators in OPPOSITE directions —
  GJR up (toward smooth: the cycle is more predictable than rough noise), MF-DFA
  down (toward rough), Cont–Das breaks at large amplitude — the same
  sign-disagreement as Rungs 2–3; the bias grows with amplitude; deseasonalising
  (subtract the period-7 mean cycle) recovers every estimator exactly (a
  removable artefact, unlike Rung 4). Ties to Phase B: crypto's weak day-of-week
  amplitude means the BTC/ETH roughness reading is uncontaminated by calendar
  effects. NOT done (deliberately — needs equity data): the real overnight/
  weekend GAP natural experiment (equity-gapped vs crypto-continuous), the
  remaining data leg. Figure: layer1c_rung5_calendar.png. Closing message and
  `--rung 5` CLI updated; the simulated corruption ladder is now Rungs 1–5.

- **2026-06-20 — Equity arm built AND run; Rung-5 gap leg closed.** Free
  intraday equity history at a 2019–2025 span does not exist, so — per the
  ROADMAP's sanctioned fallback — the equity arm uses a RANGE-BASED daily
  variance on free daily OHLC. `equity_data.py` (+`test_equity_data.py`, 6
  tests): a stdlib stooq downloader + Garman–Klass / Parkinson daily
  log-variance builder that emits the SAME CSV header as rv_series.py, so
  estimate_h / interpret_h read it unchanged (verified end-to-end: a built
  series loads through load_log_rv_csv and runs the three estimators). RUN on
  real data: S&P 500 (^GSPC, 1,759 daily obs 2019–2025) via **yfinance** —
  stooq now sits behind a JavaScript bot-wall so the auto-downloader correctly
  errors out; yfinance, or a manually-downloaded CSV, is the route (the builder
  reads both). **Result: SPX GJR Ĥ ≈ 0.132 — LESS rough than crypto's ~0.08**,
  the right direction for a calendar/gap effect; and de-biasing SPX is
  NON-IDENTIFIED too (GJR above the calibrated range, MF-DFA multi-valued,
  Cont–Das nan). So the observational-equivalence wall holds across BOTH asset
  classes — continuous crypto AND gapped equity — a stronger, more general
  finding than crypto alone. Fidelity caveats (in the module): the range proxy
  is TRADING-SESSION only (no overnight gap), and an equity-vs-crypto Ĥ gap
  mixes a calendar difference with a proxy (range-vs-RV) difference — so read
  SPX-vs-crypto as suggestive corroboration, not the clean isolation (that is
  the simulated Rung 5). Both legs of Rung 5 now exist: clean simulated
  isolation + real cross-asset comparison. Possible next: more equity tickers
  (QQQ, ^FTSE) to check whether ~0.13 is stable across equities.
**2026-06-20 (cont.) — Layer 1c capstone: identifiability map (P3)**
- **L1c-MAP** Built `identifiability_map.py` (+ `test_identifiability_map.py`,
  11 green) — the operating-characteristics deliverable P3 calls for, made
  formal. (1) `cell_status` = an operational identifiability definition:
  non-monotone curve ⇒ multivalued (rough ≡ smooth); |local slope| < _FLAT_SLOPE
  ⇒ collapse; else identified iff the ±zσ single-sample band separates the
  cell's Ĥ from the smooth null E[Ĥ | H=½]. Reuses interpret_h's audited
  `_is_monotone`/`_local_slope`, so the map and the Phase-B interpreter share
  ONE definition. (2) `build_identifiability_map` sweeps the bias curve over
  (η, Δ). (3) `plot_identifiability_map` = the phase diagram (estimator × Δ
  panels, η × H cells). New file only; nothing tested touched. Commit fe3688c.
- **P3 framing upgrade** from "can we trust roughness estimates? (audit)" to
  "here is the identifiable region; the major assets fall outside it —
  reconciling GJR-rough (H≈0.1) and Cont–Das-artefact as two readings of one
  ill-posed inverse problem."
- **Open** step 4 (calibrate η per asset, drop BTC/ETH/SPX on the map via
  `locate_observed`); step 5 (microstructure-noise robustness — needs a γ param
  threaded into build_bias_curve, a deliberate edit to a tested module); FTW
  quasi-likelihood (stretch #4) to harden inference from labels to identified
  sets. Does NOT advance Layers 2–4 (the RL/frictions destination).
  **2026-06-20 (cont.) — Layer 1c step 4: assets located on the map**
- **L1c-MAP-4** Added η calibration + asset placement to identifiability_map.py
  (+4 tests, 15 green). calibrate_eta pins η by bisection so model daily-log-RV
  std matches the asset's (Phase B calibration, reusable; clamps at the grid edge
  when the asset is more variable than the model spans). place_asset loads a Phase
  B RV CSV, calibrates η, reuses interpret() to invert each estimator's Ĥ against
  the matched curve; status uses the map's identified/de-biasable/non-identified
  definition. plot_identifiability_map(..., placements=) overlays each asset as a
  star (identified), bracket (multivalued), or off-grid arrow (below-floor) at
  (implied H, η̂). Run --assets CSV:LABEL:WINDOW; set --eta to span calibrated η.
  Open: step 5 (microstructure-noise robustness axis), FTW quasi-likelihood.
  **2026-06-21 — P3: real-data placement + paper drafted**
- **L1c-MAP-runs** Ran the map on real processed RV (grid η∈{0.5,1.5,2.5,3.5}, Δ∈{48,96,288}, n_obs=2500, n_mc=40). Identifiable fractions: GJR id 12% / non-id 85% / de-bias 4%; MF-DFA id 30% / non-id 61% / de-bias 10%; Cont–Das non-id 92% / uncal 8% — region real but concentrated at fine Δ. Placement at calibrated η̂: BTC (n=2557, η̂=1.57) GJR 0.083→non-identified, Cont–Das NaN→uncalibrated, MF-DFA −0.057→below-floor; ETH (n=2557, η̂=1.45) GJR 0.070→non-identified, Cont–Das NaN→uncalibrated, MF-DFA −0.065→non-identified. Headline holds: no identified reading on any asset, on a map where the method works elsewhere.
- **Sampling sweep** (1m/5m/15m) — GJR flat/rising as sampling fines (BTC 0.092/0.083/0.083; ETH 0.077/0.070/0.070) ⇒ microstructure-noise confound ruled out. BTC monofractal (Δh≈0.04); ETH multifractal spread (Δh≈0.08).
- **SPX** (Garman–Klass range proxy): η̂ clamps at floor 0.20; GJR 0.132 & MF-DFA above-ceiling, Cont–Das undefined — non-identified but proxy-confounded; weak corroboration only, NOT a clean second asset class.
- **P3 drafted** content-complete (~3.8k words; abstract→§1–6→reproducibility→refs→appendix), all numbers from the runs above. Title upgraded: "When is volatility roughness identifiable? A simulation-grounded audit of Hurst estimation from realized variance, with application to cryptocurrency." Overleaf-ready LaTeX built (pdflatex-clean, 10pp) + "Use of generative AI" statement. Remaining: insert the 2 figures.
- **Citations** all 16 verified vs originals. Fixed: Rogers = book chapter (Options — 45 Years…, ch.9, 173–184), not a working paper; Fukasawa–Takabatake–Westphal = 2019 arXiv:1905.04852 "Is volatility rough?" + 2022 Math. Finance 32(4):1086–1132 "Consistent estimation…". Added: SIAM *Rough Volatility* (2024); Takaishi 2025 (FRL 74, 106683). Confirmed Cont–Das = 2024 (Sankhya B 86:191–223).
- **Committed:** paper_outputs.py (one-command P3 figure/number regen) added to the repo as a reproducibility asset; reuses already-tested modules, so it ships without its own test (output/ stays gitignored). No advance on Layers 2–4.

**2026-06-21 — Layer 2 Piece 1 spec**
**L2-P1-spec** Wrote the gate-check spec for classical Almgren–Chriss (cost E/V, sinh optimum, λ=0 linear limit, efficient frontier; 7 validation gates). All closed-form targets verified numerically before commit. Build not started — acceptance = 7 gates green before any rough-vol work. Spec file: layer2_piece1_gate_check.md.

**2026-06-23 — P2 estimator programme + κ=1 (D20–D23)**
- **D20** *(2026-06-23)* **Antithetic coupling — REFUTED.** Mechanism/prediction: antithetic sampling cancels odd-order terms — predicted it reduces variance but **not** the rate β, so it cannot rescue MLMC. Result: β unchanged at ≈2H (identical to naive across H); variance reduction ×1.438 mean, but matched-L efficiency **0.908×** — i.e. ~9–11% *costlier* once accuracy is matched. Artifact caught (G-A4): a free-running adaptive-L selector manufactured phantom ratios (anti÷naive swung 0.13×→1.10×, sign-unstable across seeds). **Rule adopted: pin L (matched accuracy) for every estimator comparison.** Seeds 7/23/11. Driver `p2_antithetic_gatecheck.py`.
- **D21** *(2026-06-23)* **Conditional MC — CONFIRMED (the P2 winner).** Mechanism/prediction: conditioning on the variance path (geometric-Asian control variate) removes a large noise share; pre-check predicted the removed share is *higher* single-level than level-difference → conditioning helps standard MC **more** than MLMC. Result: unbiased, β≈2H; single-level variance reduction **4.16×** vs level-difference **3.19×**; four-way matched-L cost cond-stdMC < cond-MLMC < naive-stdMC < naive-MLMC at every ε/seed; decisive κ-invariant ratio **std-MC / cond-MLMC = 0.41–0.45 < 1** → conditional MLMC does **not** beat conditional standard MC; cond-stdMC ≈3.3× cheaper than naive-stdMC. Seeds 11/99/1234. Driver `p2_conditional_verify.py`.
- **D22** *(2026-06-23)* **κ=1 hybrid (exact near-cell integration) — VALIDATED, adopted narrowly.** Mechanism/prediction: exact near-cell variance integration tightens the coupling; predicted it sharpens the *constant* (bias) without touching β. Result: covariance closed forms verified to 1e-11–1e-13; fine-path Var(W̃_T) closes to **0.9996** (analytic) vs κ=0's **0.853** (~500× closer); coupling tightness 0.001 vs 0.39–0.58 (552× separation). G-H4 adoption: conditional std MC is bias-limited → κ=1 bias ratio 0.59× (cuts ~40%), variance penalty 1.13×, grid halves, **net cost ratio 0.79×/0.68× (~1.3–1.5× cheaper)**, stable across seeds 5/11/23. **κ=1 NOT default** — it changes the constant, not the MLMC verdict; β unchanged, so the P2 conclusion is untouched. (Two real bugs caught and fixed en route: a gate-evading sub-cell-reuse off-by-one; a proxy-offset artifact — switched to coupled/paired biases.) Module `layer1b_kappa1.py`; drivers `gh1_/gh2_/gh4_kappa1_*.py`, `kappa1_coupling_design_check.py`.
- **D23** *(2026-06-23)* **P2 reframe — "turbocharged versus multilevel."** Decision: reframed the paper from "make MLMC pay" to the honest finding. **Verdict: for arithmetic-Asian options under rough Bergomi, MLMC does not earn its place. Conditioning is the route that pays — as single-grid turbocharging, not multilevel — and κ=1 sharpens that winner.** Recommendation: conditional standard MC on the κ=1 variance path. P1 (exact-coupling baseline) absorbed into P2's baseline rather than published standalone.

**2026-06-24 — Execution-RL arc, Phase 0–1 (D24–D26)**
- **D24** *(2026-06-24)* **Execution-RL Phase 0 — VALIDATED.** Mechanism/prediction: build the rough-Bergomi execution environment + Almgren–Chriss + naive baselines (advances the parked L2-P1 Almgren–Chriss build); gate G-X1 — in the Markovian limit (H→½, constant vol) AC is provably optimal, so the simulated AC frontier must reproduce the closed-form optimum. Result: **G-X1 PASS — simulated AC frontier matches the analytic optimum to 0.7%** across the frontier → the environment can be trusted (the project's "BS anchor"). Framing finding: on the naive schedule, rough-market inventory risk is **0.95×** the constant-vol risk — roughness barely moves *aggregate* risk, so any RL edge must come from **path-dependent vol-timing**, not a different risk level. Module `execution_alpha.py`; spec `docs/gate_checks/execution_rl_gate_check.md`.
- **D25** *(2026-06-24)* **Execution-RL Phase 1 — KILL-SWITCH FIRED (clean negative).** Mechanism/prediction: cheap probe — a causal vol-reactive heuristic (signed strength θ, signal z=(V−ξ₀)/ξ₀) modulating the AC schedule. Committed bar (set before reading numbers): beats AC only if its matched-risk Pareto front sits below AC's by > seed SE, with θ*≠0, sign-stable across seeds {5,11,23}, **and** does NOT beat AC at H=0.49 (Markovian sanity). Result: **KILL-SWITCH FIRED.** With look-ahead removed, the heuristic is ~5 s.e. **worse** than AC (gap −0.025), per-seed [−0.034, −0.017, −0.023], no edge that grows with roughness. **Verdict: under linear impact, rough-volatility structure offers no executable execution edge — a simple vol-aware policy cannot beat classical Almgren–Chriss, so deep RL is NOT pursued.** Scope: linear impact + single causal vol signal; Phase-3 variants (sqrt/permanent impact, noisy V proxy) deliberately untested — no fishing. Module `execution_alpha_phase1.py`.
- **D26** *(2026-06-24)* **Look-ahead artifact — CAUGHT and corrected (the discipline working).** What happened: Phase 1's first run showed a spurious **+1.52** "edge" at H=0.10 that was **larger (+2.21) at H=0.49** — H-independent, the signature of an artifact, with gaps exceeding the AC cost itself (impossible for a real matched-risk reduction). Bar (iv) fired: STOP-and-show, do not report a win. Mechanism: global schedule renormalization used the *entire* vol path, and with leverage ρ=−0.70 the vol path is correlated with the price path — covertly timing inventory to future moves. Fix: causal per-step renormalization (uses only current state). **Precondition proven before re-reading results: max|E[inv_pnl]| = 0.025, z<0.5** (≈0 on a martingale mid) → comparison valid. Same matched-accuracy/sanity discipline that caught the antithetic phantom-L (D20) and the κ=1 proxy offset (D22), now in execution.

**2026-06-26 — Audit trail (D27)**
- **D27** *(2026-06-26)* **Audit trail committed and reconciled.** PR #1 (docs/gate-checks index, `754a390`), PR #2 (antithetic + conditional estimators + κ=1 module, `d778074`), PR #3 (execution-RL Phase 0/1 — env, baselines, kill-switch negative result, `9284a0d`) all **merged**. Local `main` reconciled to `origin/main` (`06d4c09`); **remote branches deleted (local copies remain)**. P3 `identifiability_map.py` edit parked in `stash@{0}` for its own session.
---

## Publication seeds

- **P1 (ready to draft):** *"Is the pathwise bound tight? Naive multilevel
  Monte Carlo under rough Bergomi."* Empirical note: β ≈ 2H across
  H ∈ [0.05, 0.35] with exact κ=0 coupling; cost comparison showing naive
  MLMC ≤ standard MC in the practical regime; why the Asian average fails
  to smooth the Volterra error. Figures already exist (§2, §3, §4).
- **P2:** improved estimators — antithetic and conditional-MC couplings
  (extensions 1–2), benchmarked against P1's baseline.
- **P3 (drafted 2026-06-21):** *"When is volatility roughness identifiable?
  A simulation-grounded audit of Hurst estimation from realized variance,
  with application to cryptocurrency."* Content-complete draft (markdown +
  Overleaf-ready LaTeX, 10pp): identifiability map + real-asset placement
  (BTC/ETH/SPX all non-identified at calibrated η̂); 2 figures, 2 tables,
  16 verified refs, AI-use statement. Remaining: insert the 2 figures.
  Target: arXiv q-fin.ST, then SIURO.
- Long-range: MLMC for market-risk measures (nested estimation) — aligns
  with the PhD direction.

---

## Working agreement

1. Read this file before touching code. In Claude Code: **Plan Mode first**
   (Shift+Tab), subagents for diagnostics.
2. Append to the decisions log every session, with date and rationale.
3. Never declare anything "tested" or "ruled out" without having run that
   exact thing, by name, in that session.
4. Negative results are results. Claims in docstrings must match the
   evidence in `output/` — when they diverge, the docstring changes.
5. Commit messages reference the layer and decision IDs (e.g. "L1b: D3
   rate floors").
