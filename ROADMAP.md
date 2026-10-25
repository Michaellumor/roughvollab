# RoughVolLab — Roadmap & Project Memory

> **Read this file first, every session.** It is the single source of truth
> for project state, design decisions, and measured results. Chat history
> does not persist; this file does. Append to the decisions log every
> working session — never rewrite history.

---

## Programme

**Title:** RoughVolLab — interrogating rough volatility: is it identifiable,
priceable, tradeable, and hedgeable? A rough-stochastic-volatility research
programme.

**Mission:** a unified, pedagogically structured, publication-grade Python
platform for rough stochastic volatility — covering simulation, convergence
analysis, multilevel Monte Carlo pricing, identifiability and calibration,
market frictions, and (as an exploratory extension) risk-aware deep hedging.
Every module is individually citable; every numerical claim in code or docs
must be backed by a run that actually happened.

**Author:** Michael Lumor, Department of Mathematics, University of Salford
(independent research programme; ORCID 0009-0000-0326-3891).

**Research arcs (README framing) ↔ layers:** Arc 1 — identifiability (Layer 1c) ·
Arc 2 — pricing (Layer 1b) · Arc 3 — execution (Layer 2).

---

## Status board

| Module | Contents | Status | Last touched |
|---|---|---|---|
| `layer1_rough_vol.py` | fBm (Cholesky + hybrid), rBergomi/rHeston paths, Hurst estimation | ✅ complete — **L1-1 resolved** (PR #53) | 2026-07-09 |
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
| `layer3_deep_hedging.py` | Deep-hedging engine — Buehler-style direct policy optimisation, CVaR objective, self-computed signatures; isolated torch venv, deletion-safe leaf. Finds: deep beats delta under frictions, roughness adds no hedging edge beyond it | ✅ built (7 tests, isolated venv; core torch-free) | 2026-06-30 |
| `rough_heston_cf.py` · `rough_heston_lifted.py` · `layer4_calibrate*.py` · `deribit_surface.py` | Rough-Heston convergence + Markovian lift (O(N·n) vs O(n²)) + high-ν pricing + calibration engine (single-smile → multi-maturity surface → live Deribit BTC); see the Layer-4 narrative + spec §5/§8 | ✅ built (D31–D39) | 2026-06-29 |
| `docs/gate_checks/` | Gate-check specs + recorded verdicts (index) | ✅ living | 2026-06-26 |
| `ROADMAP.md` | This file — project memory | living document | 2026-06-27 |
| `OVERLEAF/P1` | Paper — *Identifying roughness from an option surface* (Q4; calibration route). Single-smile → multi-maturity → live Deribit BTC/ETH; H non-identifiable, rails to bound | ✅ submitted — Applied Mathematical Finance, 26 Sept 2026 (ID 266158357); IMAMAT declined on scope 23 Sept; QF held in reserve. Venue copies: OVERLEAF/P1/imamat/ (frozen), amf/ | 2026-09-26 |
| `OVERLEAF/P2` | Paper — *Turbocharged vs multilevel MC* (Q2; pricing). MLMC does not pay; conditional single-grid wins (0.41–0.45); confirms Giles's β<γ theory | ✅ submission-shaped for SIURO (PR #83); packet delivered to the project advisor 20 Aug; awaiting advisor letter | 2026-08-20 |
| `OVERLEAF/P3` | Paper — *When is volatility roughness identifiable?* (Q1; RV route). Identifiability map + BTC/ETH/SPX non-identified; 3 estimators, intrinsic biases | ✅ preprint posted — SSRN 7346318 (approved 26 Aug 2026); pins closed (PR #101: ftw merged, Cont–Das → Sankhyā B) | 2026-08-26 |
| `OVERLEAF/P4` | Paper — *Weak convergence is faster than strong* (foundational). Weak order ≫ strong (H), measured in rough Heston and tested against the Gaussian-volatility weak-rate theory; rank-one-lift negative result; underpins P2's single-grid conclusion | ✅ preprint posted — SSRN 7346418 (26 Aug 2026; PR #104 title/pins). Journal: SIFIN Short Communications, desk-declined on scope (31 Aug, no referee reports); resubmitted to IJTAF 31 Aug, at referees (~5–6 months). Per-venue copies: OVERLEAF/P4/sifin/ (frozen record), ijtaf/ (PRs #118/#120/#122) | 2026-09-02 |

---

## Layer 1 — Stochastic simulation core (complete)

Cholesky-exact fBm, BLP hybrid scheme, rBergomi and rough Heston path
simulation, Hurst estimation from realised variance. Four sections, figures
to `output/`.

### KNOWN ISSUE L1-1 — Volterra normalisation (logged 2026-06-12; RESOLVED 2026-07-09) ✅

**✅ RESOLVED (2026-07-09, PR #53 — RVL-001/002/016/017; see D47):** the fix
plan below was carried out exactly. The bug record is kept verbatim as honest
history — the present-tense description is how it was logged, not the current
state.

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

**Fix plan (carried out — PR #53):** rewrite `fbm_hybrid` against the κ=0 weights in
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

## Layer 1c — Roughness-estimator audit (Phase A + B complete; P3 drafted)

**File:** `layer1c_roughness_audit.py` (house style: sections, `--section`,
`--no-show`, `--quick`; figures to `output/`).

**Goal:** quantify how reliably the Hurst exponent can be estimated from
realised-volatility data, using Layer 1b's *verified* simulator as ground
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
a committed pytest suite; runbook `run_phaseb.md`). Analysis run on BTCUSDT + ETHUSDT, 2019–2025,
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

## Layer 2 — Market frictions (execution baseline built; rough-friction spec)

**Status — execution arc built (negative result).** The Almgren–Chriss
execution baseline is **built and validated** in `execution_alpha.py`: gate
**G-X1** reproduces the analytic AC frontier to **0.7%** in the Markovian limit
(the environment's "BS anchor"). A cheap causal vol-reactive execution probe
(`execution_alpha_phase1.py`) was then run as a kill-switch: on the matched-risk
efficient frontier it is **~5 s.e. worse** than AC with no edge that grows with
roughness → **no executable execution edge under linear impact; deep RL not
pursued.** See decisions **D24–D26** (including the look-ahead artefact caught
and corrected en route). This is the **execution** arc — distinct from the
Layer 3 deep-hedging engine below.

**Goal (remaining):** non-linear execution frictions under rough vol; demonstrate
the breakdown of Markovian dynamic programming.

**Contents (remaining):** rough execution slippage (impact driven by the rough
variance path); empirical demonstration that conditioning on path history changes
the conditional law (non-Markovianity) — this motivates Layer 3's signature
features. (Almgren–Chriss closed-form anchors: done, above.)

**Validation criteria:** ✅ reproduce Almgren–Chriss optimal trajectories
analytically in the classical limit (G-X1, 0.7%); ⬜ quantify Markov-projection
error vs a path-dependent benchmark.

**Key refs:** Almgren & Chriss (2001); Gatheral, Jaisson & Rosenbaum (2018).

## Layer 3 — Deep-hedging engine (built — D40)

> **Note — not the execution work.** This is the deep-**hedging** engine
> (risk-aware deep hedging of a derivative position via path signatures), **built (D40)** and **distinct from the Layer 2 execution arc** above (Almgren–Chriss
> liquidation + the execution-alpha probe, which is done → kill-switch fired,
> D24–D26). Layer 2 = *executing/liquidating* a position; Layer 3 = *hedging* a
> derivative. Do not re-conflate the two.

**File:** `layer3_deep_hedging.py` (built — D40). Separate suite `test_layer3_deep_hedging.py`; torch isolated in `.venv-layer3` (the core stays torch-free).

**Goal:** risk-aware deep hedging on the non-Markovian state via path
signatures.

**Contents:** truncated signature features of (t, S, realised-var) path;
direct policy optimisation (Buehler-style) with a CVaR objective; baselines: BS delta,
delta-vega, and the Layer 2 friction-aware strategies.

**Validation criteria:** recover BS delta (η→0, frictionless) within
tolerance; beat delta hedging on CVaR of terminal P&L under frictions with
statistical significance across seeds.

**Result (D40):** Gate 1 recovers BS-delta (frictionless); under frictions deep hedging beats delta (+1.1 CVaR, 8/8 seeds — the generic Buehler edge), but the roughness-specific increment is modest/absent (+0.06 ± 0.04, z=1.4) — roughness adds no hedging edge beyond frictions.

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
  de-biaser, a committed pytest suite, runbook). Ran it on BTCUSDT+ETHUSDT 2019–2025 (2,557 daily
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

**2026-06-27 — Documentation reconciliation (D28)**
- **D28** *(2026-06-27)* **Docs caught up to the three-arc state; one misstep made and corrected — recorded honestly.** A run of documentation-only PRs brought README + ROADMAP into line with the three-research-arc framing and fixed stale layer labels:
  - **PR #4** — README/ROADMAP rewritten around the three arcs (identifiability / pricing / execution).
  - **PR #5** — removed stale planned-layer rows from the status tables (dropped the redundant `layer3_rl_hedging.py` row; marked Layer 2 done).
  - **PR #6** — **corrected the execution work to Layer 2** (it had been mislabelled "Layer 3" in the README); kept the deep-hedging engine as the distinct, still-unbuilt Layer 3.
  - **PR #7 — MISSTEP:** replaced the flagship `roughvollab_module_map.png` (the maths-to-layer map the course tutors prefer) with a native Mermaid architecture diagram, and added a Layer 3 Structure-table row. The Mermaid swap was the wrong call — it discarded the tutor-facing flagship to gain editability.
  - **PR #8 — CORRECTION:** reverted the Mermaid swap, restoring the original PNG embed byte-identical from history (`7c889d1^`); **kept** the (good, independent) Layer 3 row.
  - **PR #9 — the durable fix:** added `module_map.py`, a regenerable matplotlib generator for the flagship, and regenerated `roughvollab_module_map.png` from it with CURRENT statuses (Layer 2 execution built/negative, L1c complete, L3 unbuilt, L4 spec next; "161 pytest tests + gate-checks"). The flagship is now both tutor-facing AND version-controlled. **Lesson from the #7→#8 round-trip: don't replace a flagship asset to make it editable — make the asset regenerable instead.**
  - **This entry** sits in a final ROADMAP↔README reconciliation: fixed the stale `## Layer 1c` "not started" header (→ Phase A + B complete), added the missing `layer3_deep_hedging.py` status-board row + named the file in the `## Layer 3` spec + renamed that header to "Deep-hedging engine", corrected the Publication-seeds P1/P2 entries to match D23 (P1 absorbed into P2; P2 = "turbocharged versus multilevel", concluded), and added an arc↔layer crosswalk to the Programme. All D1–D27, layer specs, gate-check records, and Phase-A/B detail preserved.

**2026-06-27 — Layer 4 brick 1: rough-Heston simulator (D29)**
- **D29** *(2026-06-27)* **Native rough-Heston simulator built and validated for ν ≤ 0.20 — via two instructive FAILs the gate caught.** `rough_heston.py` reuses `roughvol_core.py`'s κ=0 BLP Volterra weights (g/Γ(H+½)) in an explicit Volterra–Euler recursion (conditionally Gaussian per step) + a correlated log-Euler asset; acceptance = reproduce the layer1b β=2H level-variance decay (`rh_beta_gate.py`: MLMC coupling, arithmetic-Asian functional). The PASS came only after two failures — these, not the PASS, are the intellectual content:
  - **FAIL 1 — full truncation broke β (a wrong assumption, refuted by the gate).** The plan assumed full truncation (V⁺=max(V,0)) was fine for the β gate because "β is a strong-error property, insensitive to the positivity bias." FALSE: at the default rough params (H=0.10, ν=0.40) β collapsed (0.026/0.076/0.210/0.544 vs 2H=0.10/0.20/0.40/0.70). Diagnosis: the non-smooth clip fires at DIFFERENT times on the fine vs coarse MLMC grids → it BREAKS THE COUPLING; β-suppression tracks the V==0 frequency exactly (β=0.075 at 35% V==0 vs 0.223≈2H at 4%). Truncation corrupts β *itself*, not just the weak-order bias. (Same "discipline catches a wrong claim" pattern as the antithetic phantom-L (D20) and the look-ahead artifact (D26).)
  - **FAIL 2 — NO positivity map recovers β at high ν.** Built three swappable schemes and measured β-vs-H at ν=0.40 (N=20000): full truncation, reflection |V|, and Andersen QE (on the conditionally-Gaussian step). NONE recovered β=2H. Reflection worst (β<0 at small H; E[V] inflated +158%). QE best (E[V] bias +4%, closest β) but still failed at H≥0.20. So the V→0 coupling break is intrinsic to the rough/high-ν regime, not fixable by the positivity map.
  - **Diagnosis path that settled the fork.** A paired price diagnostic (qe vs truncation, identical increments) separated the two high-ν pathologies: (a) the coupling break is a β-GATE-ONLY artifact — the weak-order study is single-grid vs the CF reference, no coupling; but (b) the positivity scheme *separately* CONTAMINATES the priced bias at high ν (qe vs truncation 2.4–5.3% at ν=0.4, n-dependent; <0.5% at ν=0.15). Both onset together as V==0 events exceed ~10% (ν≳0.25) and both vanish at mild ν.
  - **RESOLUTION — validated regime ν ≤ 0.20, QE default.** QE PASSES the β=2H gate for ν ≤ 0.20: β=0.070/0.167/0.384/0.737 vs 2H, max|β−2H|=0.037, consistent with layer1b (0.13/0.23/0.42/0.72); PASS at ν=0.15 and 0.20, fails at 0.25. `PARAMS['nu']` set to 0.20 (validated ceiling), positivity='qe' default; full truncation kept selectable (cheapest where truncation is rare); reflection rejected. 8 sanity tests pass (incl. a non-anticipation guard β can't catch; H=½→classical-Heston and E[V]=θ renorm checks).
  - **BOUNDARY FINDING (a result in its own right).** The explicit hybrid Volterra–Euler scheme's strong-order *coupling* (β) and weak-order *priced bias* BOTH degrade as V→0 events exceed ~10% of samples (ν≳0.25); the scheme is validated for ν ≤ 0.20. A genuine characterisation of where the method stops being trustworthy, not a config limit.
  - **HIGH-ν EXTENSION (scoped, not silenced).** Real rough-Heston SPX/index calibration is high-vol-of-vol / Feller-violated (V hits 0 often) — almost certainly ν > 0.20 — so the ν≤0.20 weak-order study is METHOD-VALIDATION and the market-relevant SPX claim requires a MULTIFACTOR MARKOVIAN-LIFT simulator (Abi Jaber–El Euch: rough kernel ≈ sum of exponentials → Markovian factors + per-factor QE/Alfonsi), scoped as an explicit Layer 4 extension (its own brick + gate) in `docs/gate_checks/layer4_convergence_gate_check.md` §5/§8. Files: `rough_heston.py`, `test_rough_heston.py`, `rh_beta_gate.py`.

**2026-06-27 — Layer 4 brick 2: rough-Heston CF reference (D30)**
- **D30** *(2026-06-27)* **Simulation-free rough-Heston CF reference built and certified — and a coefficient error caught by source-pinning + the H=½ arbiter.** `rough_heston_cf.py`: the El Euch–Rosenbaum characteristic function via a fractional Riccati (Diethelm–Ford–Freed fractional Adams–Bashforth–Moulton) + Gil-Pelaez Fourier inversion, with an Albrecher little-trap classical-Heston CF as the independent reference. This is the known-truth the weak-order study (brick 3) prices against. 23 tests pass.
  - **CONVENTION CATCH (the intellectual content).** The CF was pinned from the PRIMARY source (arXiv:1609.02108, eq.(3)+main thm), NOT memory/secondary — and it mattered: a secondary web extraction gave the CF integral coefficient as **θ/λ**, but the paper gives **θλ (= κθ)** — a **factor-κ² error** (κ=2 in the test set → 4× wrong) that would have made EVERY price wrong. The H=½ CF-level gate confirmed κθ to **8e-6** (the `κθ∫ψ == C_Heston` check would have been O(1)-wrong under θ/λ). Mapped ER's relative-vol-of-vol (λν) convention to our absolute-ν model (κ=λ, ν=λ·ν_ER); the H=½ reduction is the arbiter for every coefficient.
  - **TWO-STAGE ISOLATION (why the bug was findable).** Gated build order: (a) the Gil-Pelaez INVERTER proven FIRST on KNOWN CFs — Black–Scholes to **1e-13** (true closed form), Heston integrand fixed-GL-vs-adaptive to **1e-11** — BEFORE any Riccati existed; (b) then the fractional Riccati, gated at H=½ on ψ(u,T)==D_Heston and κθ∫ψ==C_Heston (O(h²) → 8e-6 at N=800); (c) the full pipeline H=½ price gate == closed-form Heston to **~1e-6** (N_riccati≈1600); (d) H<½ property checks (φ(0)=1, **martingale φ(−i)=1 exact**, |φ|≤1, Hermitian, φ→1 as t→0, no-arb). Certifying the inverter independently made any pipeline failure isolable to the Riccati/CF-assembly — that isolation is what made the θλ-vs-θ/λ bug findable rather than buried.
  - **CERTIFICATION (relative, for brick 3).** At the rough regime (H=0.10, ν=0.20) the reference converges at **order p ≈ 1.60 = 1+H+½** (lower than H=½'s O(h²): the I^{1−α} fractional integral is active and ψ~t^α is non-smooth at 0). Fit `err ≈ 0.68·(T/N_riccati)^1.60`; to reach reference error ≤ X use **N_riccati ≥ {1e-5: 1040, 1e-6: 4370, 1e-7: 18400, 1e-8: 77400}**. Brick 3 picks N_riccati so the reference sits ≥1 order below its smallest weak-order bias (a brick-3 decision — hence the curve, not a single number).
  - **COST FLAG.** The FABM is **O(N²)** per CF-set (N=6400 → 12.5 s). Adequate for brick 3 as scoped (ν≤0.20, reference once per strike at 1e-5–1e-6 → N≈1040–4370 → ~0.1–5 s). A **sum-of-exponentials kernel approximation** (fast convolution) is NOT needed there, but WOULD be needed for a ≤1e-7 reference (N≥18k → ~100 s+) or for SPX-calibration loops (repeated CF evals — the high-ν multifactor-lift regime). Files: `rough_heston_cf.py`, `test_rough_heston_cf.py`.

**2026-06-27 — Layer 4 brick 3: weak-order convergence study (D31)**
- **D31** *(2026-06-27)* **Weak order α of the κ=0 hybrid measured against the CF reference — the strong-order H-pessimism does NOT carry to pricing (α ≫ H, banked), with a real roughness penalty at the rough end.** `layer4_convergence.py` prices an OTM call (K=110, ν=0.20) at n = 4…64, fits α in `bias ≈ C·Δtᵅ` against the brick-2 CF (the known truth), and classifies each H against the §3 prediction. `test_layer4_convergence.py`: 6 harness-mechanics tests (CRN aggregation, vectorised BS, conditional-MC unbiasedness, fit-on-synthetic-power-law, classifier) — the MC *result* lives here in D31; the tests guard the *machinery*.
  - **THE HARNESS.** Two independent bias estimators that must agree (anti-artifact gate): the **absolute** b_n = |E[Pₙ] − P_CF| (vs the CF) and the **coupled** Yₗ = E[Pₗ − P_{l−1}] (CRN across levels — one finest Brownian path aggregated down → low-variance). Variance reduction: a Black–Scholes **control variate** plus the lever that actually widened the bias window — **Romano–Touzi conditional MC** (integrate the orthogonal asset Brownian out analytically → the call becomes a BS price conditional on the variance path; bias-preserving, same V/M/I discretisation). The absolute-vs-CF estimator could not resolve α at all without it.
  - **H=½ ANCHOR (harness validated first).** Recovered weak order **≈0.89 (absolute) / 0.94 (coupled) ATM, ≈1.14 / 1.09 OTM** — inside the literature-pinned range **[0.6, 1]** for Euler-type Heston discretisation at our Feller index ν_F = 2κθ/ν² = 0.60 (arXiv:2106.10926: order = 1 if ν_F > 1 else ~ν_F). The pricing **kink** was identified — **ATM α < OTM α** (the at-the-money payoff is harder to price) — so the H-sweep uses the cleaner **OTM** strike.
  - **THE RESULT (OTM K=110, ν=0.20).** **α ≫ H at every H — robust, banked, across both estimators and all fit windows:** the strong order O(n^{−H}) does NOT bound the weak/pricing rate. Per H: **PASS @ H=0.20** (α ≈ 1.0 ≈ 1 — weak converges classically; reproducible `--sweep` prec-weighted 1.01); **PARTIAL @ H=0.05** (α ≈ 0.74, robustly **< 1** at **8σ** by the tight absolute estimator, window-stable 0.74–0.77 — a genuine roughness penalty at the rough end); **borderline-pending-finer-grids @ H=0.10** (α ≈ 0.84–0.95; the single-window `--sweep` prec-weighted reads 0.87→auto-PASS, but the cross-window systematic 0.84↔0.95 spans PARTIAL-to-PASS, so it is **not robustly distinguishable from 1**). Trend: α rises with H and dips below 1 as H → 0. The §3 directional prediction (α > H, ≈ 1) is now **measured** — α ≫ H confirmed; ≈ 1 for H ≥ 0.20; penalty emerging at H ≤ 0.05; mid-range borderline.
  - **THE CAVEATS (the intellectual honesty).** (i) The two estimators agree only *marginally* at tightened precision — the absolute (tight, ±0.06) sits ~0.2 below the coupled (loose, ±0.21); traced to **one low-SNR coupled tail point** (n=64) inflating the coupled slope — over its clean sub-window (n = 8–32) the coupled gives ≈ 0.75, agreeing with the absolute (precision-weighted ≈ 0.75). Not a real split. (ii) **H=0.10 carries ~±0.1 SYSTEMATIC window-sensitivity** (absolute α = 0.95 over n ≤ 128 vs 0.84 over n ≤ 64) — **pre-asymptotic, not statistical**, so more paths at the same coarse grids will NOT resolve it; finer grids would. (iii) The accessible grids are coarse (the O(n²) sim + a memory cap stop ~n = 64–128 at usable M); the bias is not a perfectly clean single power law there.
  - **THE CONTINGENCY.** A definitive α(H) curve — and in particular resolving H=0.10's borderline — needs **finer grids** than the O(n²) sim affords at capped cost, i.e. the **multifactor Markovian-lift extension** (sum-of-exponentials kernel → cheap fine grids + more paths; the same lift flagged in D30's cost note and §5 for ≤1e-7 references and SPX-calibration loops). Recorded as borderline-pending-finer-grids, NOT forced to PASS/PARTIAL. Files: `layer4_convergence.py`, `test_layer4_convergence.py`, `output/layer4_weak_order.png`.

**2026-06-28 — Layer 4 brick 4a: sum-of-exponentials kernel approximation (D32)**
- **D32** *(2026-06-28)* **Sum-of-exponentials approximation of the rough kernel built and gated — the foundation of the multifactor Markovian lift; BOTH constructions pinned from the literature, validated, and selected on evidence (BB wins).** `rough_kernel_soe.py` approximates K(t)=t^(H−1/2)/Γ(H+½) ≈ Σ wᵢ·e^(−γᵢt) (each exponential → a Markovian OU factor in 4b). Simulation-free: builds the nodes/weights + measures the kernel error vs the factor count N (Gate A, closed-form L²). `test_rough_kernel_soe.py`: 7 mechanics tests (μ closed-forms vs numerical ∫; AJ–EE realizes its pinned rate; BB superpolynomial + beats AJ–EE; γ=0 tail node; H=½ guard). 36/36 tests pass with the brick-2/3 suites.
  - **SOURCE-PINNED (the θλ discipline).** Spectral representation K(t)=∫₀^∞ e^(−γt) μ(dγ), μ(dγ)=γ^(−H−1/2)/(Γ(H+½)Γ(½−H)) dγ — pinned from **arXiv:1801.10359** (Abi Jaber–El Euch, eq 1.3) and **arXiv:2108.05048** (Bayer–Breneis, eq 1.3), NOT invented. Every SOE rule is a quadrature of this integral. Two constructions: **A** = AJ–EE moment-matched closed-form nodes/weights on the uniform optimal mesh; **B** = Bayer–Breneis Gauss–Legendre quadrature on geometric subintervals + a γ=0 tail node.
  - **BOTH VALIDATED BEFORE SELECTION (Gate A is the selector).** AJ–EE realizes its pinned algebraic rate ‖K^n−K‖₂ ~ n^(−4H/5) **EXACTLY** (fitted −0.040 / −0.080 / −0.160 at H=0.05/0.10/0.20 vs theory −4H/5 — proves the build faithful), but the uniform mesh is **impractical**: its γ-nodes top out too low to resolve fast lags, so it never reaches the target (best rel-L² 0.67/0.45/0.19 at N=512). **BB is superpolynomial** (exp(−c√N); H=0.20 fitted −0.628/√N) and **wins unambiguously** at every H. Gate A's closed-form L²[0,T] error (incomplete-gamma; absorbs the integrable t→0 singularity; no simulation) is the arbiter.
  - **THE BB FRONTIER (a CONSERVATIVE lower bound).** Global α=1.6 (range constant), **NOT per-H optimized** — per-H BB §4.2 tuning is a 4b refinement that only improves it. Reaches rel-L²≤1e-3 at **N≈130 (H=0.20), ~250–520 (H=0.10), ~512–1024 (H=0.05)** — the conservative/upper-bound factor counts, not the floor.
  - **SCOPING.** Geometric AJ–EE NOT built — BB *is* the principled geometric+quadrature method, so a third hybrid variant would muddy the comparison without adding a real option. Uniform AJ–EE (faithful to its theory) is the baseline; BB the winner.
  - **THE ECONOMIC JUSTIFICATION (justifies the whole lift).** The lift costs **O(N·n)** per path (N OU factors × n steps, Markovian recursion) vs the Volterra **O(n²)** — so it **wins when N < n**, i.e. at the **fine grids** (n≳256–1024) needed to resolve brick-3's borderline H=0.10, where N(130–520) < n. It is **costlier** at brick-3's coarse n(4–64) where N>n — but the lift isn't needed there. **The lift pays off precisely where brick 3 could not go.**
  - **TARGET (provisional, relative) + GUARD.** rel-L²≤1e-3 ≈ 1 order below brick-3's smallest priced bias (~1e-2), delivered as the **N_factors(target) curve** (like brick-2's N_riccati(X)); the **final N is a 4b decision** — 4b's measured β/price sensitivity to kernel error sets it. **H=½ guard:** K≡1 (flat) reproduced exactly by a single γ=0 mode → rel-L²=0, no NaN (the classical case, lift unnecessary). Files: `rough_kernel_soe.py`, `test_rough_kernel_soe.py`, `output/layer4_kernel_soe.png`.

**2026-06-28 — Layer 4 brick 4b: lifted OU-factor rough-Heston simulator (D33)**
- **D33** *(2026-06-28)* **The multifactor Markovian-lift SIMULATOR built and validated — both known-answer gates green, reproducing brick-1; the POSITIVITY journey (four schemes, gated on evidence) is the intellectual content.** `rough_heston_lifted.py`: N Markovian OU factors reconstruct the rough variance (V = V0 + Σcᵢ Uᵢ, dUᵢ = (−xᵢUᵢ + κ(θ−V))dt + ν√V dW) instead of the O(n²) Volterra convolution → **O(N·n) per path**. Source-pinned SDE (Abi Jaber, "Lifting the Heston model", arXiv:1810.04868, Eq 2.2–2.3), brick-1-consistent folded-drift form; (cᵢ,xᵢ) = brick-4a Bayer–Breneis (weights, gammas). `test_rough_heston_lifted.py`: 7 mechanics tests; 43/43 pass with the brick-2/3/4a suites.
  - **CRUX-1 (factor dynamics).** Each factor carries its OWN mean-reversion xᵢ (=γᵢ); the CIR coupling κ(θ−V) and diffusion ν√V dW act on the AGGREGATE V, fed to every factor, all sharing ONE Brownian W (which also drives the asset via ρ — matching brick 1). Folded-drift form so β and H=½ prices are directly comparable to brick 1.
  - **CRUX-2 (stiff integration).** The γ-span reaches ~1e14 at small H → explicit Euler is unstable. **Exact-OU / exponential-Euler** (e^(−xᵢdt) decay, exact at any xᵢ) + **shared-ΔW envelope** for the noise (exact joint noise is O(N²); the envelope keeps O(N·n)). The envelope was VALIDATED despite being stiff-factor-inexact (see disambiguation).
  - **CRUX-3 (positivity) — the journey, FOUR schemes gated on evidence (the brick-1 lesson: don't pick from theory):**
    1. **Truncation (√V⁺) COLLAPSES β** at H≤0.10, ν=0.20 (β 0.068/0.158 vs L1b 0.13/0.23) — the brick-1 truncation failure recurring. The near-0 frequency is **13.2%/8.9%** (intrinsic to rough Heston here, NOT an SOE artifact — flat in N as Σcᵢ explodes, because the fast factors' weights are killed by their envelope), and the non-smooth V⁺ clip fires at different times fine/coarse → the MLMC coupling breaks.
    2. **DISAMBIGUATION (truncation vs the new shared-ΔW noise envelope) — reported separately, not lumped.** (a) β RECOVERS at milder ν=0.10 (near-0 → ~1%; H=0.05 β 0.068→0.139≈L1b) → truncation. (b) Replacing the envelope with the EXACT joint OU noise does NOT fix β (stays collapsed, 0.063→0.037, even worse) → NOT the envelope. (c) The envelope is ν-INDEPENDENT (depends only on xᵢ,dt) so it cannot cause the ν-dependent collapse, and at H=0.05/ν=0.10 (stiff factors present, truncation off) β is correct. **Verdict: truncation broke β; the noise envelope is EXONERATED and VALIDATED** — a real, slightly surprising result (stiff-factor-inexact yet correct β).
    3. **Implicit-drift FAILS** (≈ truncation): the implicit −κV damping is only ~1% (the fast factors dominating Σcᵢ have drift_coef ∝ 1/xᵢ ≈ 0), so near-0 barely moves (13.0%→12.4%) and β stays collapsed.
    4. **Naive-Alfonsi (semi-implicit √V_new) keeps V≥0 but is BIASED** — the implicit diffusion injects an Itô drift, failing GATE C by **+0.89 (+33 s.e.)** at H=½. Ruled out (do not ship a known-biased scheme).
    5. **QE-port WORKS (the fix).** Brick-1's Andersen QE DOES port: apply `_qe_map` to the AGGREGATE conditional-Gaussian step (mean m, var s²), then **back out an effective increment** dWtil = (V_new − m)/(νE√V) so the factors still reconstruct the QE'd V. Positive + smooth (coupling-preserving) + unbiased (moment-matched) — the aggregate-QE + effective-increment insight that gets around "QE doesn't port to a sum of factors."
  - **BOTH GATES GREEN (QE default, ν=0.20), reproducing brick-1.** GATE C (H=½): unbiased, price 7.596 vs CF 7.589 (+0.3 s.e.). GATE B (β=2H): **0.066 / 0.175 / 0.397 / 0.699** at H = 0.05/0.10/0.20/0.35, **max|β−2H| = 0.034**, monotone — matching brick-1's OWN rough-Heston QE β (0.070/0.167/0.384/0.737, max 0.037, D29) to ~MC noise. H=0.05 β=0.066 confirmed STABLE across more levels/M/seeds (0.060–0.068) → faithful rough-Heston-QE small-H behavior, not a lift artifact.
  - **OVER-STEEPENING RESOLVED — a truncation artifact.** Truncation gave β=0.804 at H=0.35 (OFF); QE gives 0.699 ≈ 2H. The over-steepening flagged earlier was caused by the positivity scheme, NOT pre-asymptotic and NOT a second systematic — the correct QE scheme removes it.
  - **N_FACTORS ECONOMY.** The rate (β) needs only ~tens of factors (β stable from N≈25–40), far below brick-4a's ~250–520 price target — the lift's cost win confirmed. O(N·n) vs Volterra O(n²) → the lift wins at the fine grids (n≳256–1024) needed to resolve brick-3's borderline H=0.10.
  - **SCOPE.** Validated at ν≤0.20 (brick-1's regime). High-ν (ν≈0.3–0.4, GATE D), resolving H=0.10 (the weak-order re-run on the lift), and SPX calibration are brick 4c+. Files: `rough_heston_lifted.py`, `test_rough_heston_lifted.py`.

**2026-06-28 — Layer 4 brick 4c: high-ν validation (Gate D) of the lifted simulator (D34)**
- **D34** *(2026-06-28)* **High-ν Gate D — the lift BREAKS the explicit scheme's high-ν PRICE boundary (SPX-relevant pricing delivered to ν≈0.40) while the MLMC β-rate extends only to ν≤0.30: a precisely-characterized SPLIT boundary, classified (B) PARTIAL.** Gate D ran the lifted QE-port simulator at ν∈{0.25,0.30,0.40}, H∈{0.05,0.10,0.20} on TWO checks mirroring the explicit scheme's two documented high-ν failures (§5/§8), using the brick-2 CF as a genuine **high-ν known-answer** (verified usable: H=½ rough-CF == closed-form Heston to 1e-7 at *any* ν, self-converges in N_riccati, φ(−i)=1; pinned N_riccati=4000 + NaN-guard for the small-H/high-ν Riccati ψ² overflow). Driver `rough_heston_lifted.py --gate-d`; high-ν smoke tests added (9/9 pass).
  - **THE PRICING WIN (the headline).** Lifted-qe prices CONVERGE to the CF known-answer at high ν — **<0.5% at H≥0.10 across all ν to 0.40** — vs the explicit scheme's **2.4–5.3% scheme-dependent** bias. The SPX-relevant pricing capability, delivered. Truncation (the control) diverges **2–9%** vs the CF, confirming the explicit failure recurs and that QE is what fixes it. **One caveat (measured, not inferred):** at the extreme corner **H=0.05/ν=0.40** (roughest H, highest ν, ~34% near-0) qe−CF drops with n (3.44→2.23→2.03% at n=64/128/256) but **flattens at ~2%** — partly discretization plus a small real residual that does NOT clean to <0.5% like H≥0.10. So pricing is clean to ν=0.40 for H≥0.10; a ~2% residual remains at the single H=0.05/ν=0.40 cell.
  - **THE CONVERGENCE-VS-N EVIDENCE (the proof).** At ν=0.40, H=0.10: **qe−CF → 0 as n grows** (1.57→0.92→0.48%) = discretization, CORRECT; **trunc−CF PLATEAUS ~5%** (7.85→6.44→5.47%) = a real positivity bias. This is WHY qe is correct and trunc is not — the CF high-ν known-answer made Gate D a *truth-check* (correctness), not mere scheme-agreement.
  - **THE β-RATE BOUNDARY (the honest partial).** The MLMC level-variance rate β=2H is **clean to ν≤0.30** (qe max|β−2H| = 0.052 / 0.056 at ν=0.25/0.30, monotone in H); at **ν=0.40 it degrades** (H=0.20: β=0.305 vs 2H=0.40, dev 0.095) under 16–34% near-0. qe **vastly beats trunc everywhere** (trunc collapses to 0.016–0.200 at ν=0.40), but does not hold β=2H to tolerance at ν=0.40.
  - **THE SPLIT (the finding).** The two boundaries moved **DIFFERENTLY**: the **price boundary → ν≈0.40** (H≥0.10; the explicit price failure broken), the **MLMC-rate boundary → ν≤0.30**. Pre-specified outcome **(B) PARTIAL** — not forced to (A) HOLDS or (C) BREAKS. The lift delivers the high-ν *pricing* SPX needs while the stricter MLMC-variance-rate property is characterized at ν≤0.30. Remaining for the SPX application (4d+): the H=0.05/ν=0.40 price residual and the weak-order α re-run on the lift. Files: `rough_heston_lifted.py` (gate_d), `test_rough_heston_lifted.py`.

**2026-06-28 — Layer 4: H=0.10 resolution attempt via the lift — a STOP (D35)**
- **D35** *(2026-06-28)* **Attempted to resolve brick-3's borderline H=0.10 by re-running the weak-order (α) study on the LIFTED simulator (the fine grids the explicit O(n²) couldn't reach) — the validation ladder's known-answer gate caught that the lift does NOT preserve the weak order, so the attempt STOPS and H=0.10 REMAINS OPEN. A clean negative result; the reusable `sim`-callback harness generalization is banked.** Driver `layer4_lifted_alpha.py`; the weak-order harness (`mc_call_levels`/`measure_alpha`) now takes a `sim` callback so ANY path generator plugs in (default = the explicit brick-1 core). 8 mechanics tests pass.
  - **THE GOAL.** brick-3 left H=0.10 borderline (α 0.84↔0.95, ±0.1 *systematic* window-sensitivity — "finer grids would resolve it", D31). The lift (4b/4c-validated, O(N·n)) affords those fine grids — so re-measure α on the lift past n≈64–128.
  - **THE LADDER + THE STOP.** **Gate 1 (H=½ anchor): PASS** — lifted a_cpl=1.000, a_comb=0.979 (≈ D31's ~1.1); harness + conditional MC + coupled estimator sound on the lift (H=½ = 1 factor = exact classical Heston). **Gate 2 (known-answer — does the lift reproduce brick-3's EXPLICIT α at the resolved coarse cells?): FAIL** — lifted α systematically BELOW brick-3 at n≤64: **H=0.20 → 0.74 (vs 1.01), H=0.05 → 0.40 (vs 0.74)**; the **SOE-floor-immune coupled** estimator confirms it (0.80, 0.66 — not a reference-floor artifact). |diff| 0.27/0.34 ≫ the ±0.10 bar.
  - **THE DISAMBIGUATION (the core).** N-sweep at H=0.20: a_cpl is **FLAT at 0.854 across N=150→300→600** (139→553 factors). More factors do NOT move it → the perturbation is **NOT the SOE kernel truncation** (which more N would fix) but **INTRINSIC to the shared-ΔW NOISE ENVELOPE** — the O(N·n) cost-advantage approximation that under-weights fast-factor noise. It was fine for **β=2H (a rate)** and **prices (a level)** — validated 4b/4c — but it **PERTURBS the weak ORDER (a finer quantity), N-independently.**
  - **THE COROLLARY.** The lift's O(N·n) cost advantage *is* the noise-envelope approximation — which is EXACTLY what perturbs the weak rate. An exact-joint-noise lift (O(N²) Cholesky) might preserve the weak order but forfeits the cost win (no cheaper than the explicit O(n²) for this purpose). So the *cheap* lift cannot measure the weak order.
  - **THE VERDICT.** The lift is validated for β/prices (4b/4c) but **NOT for the weak-order measurement**; the H=0.10-resolution-via-the-lift application **DOES NOT DELIVER**; **brick-3's H=0.10 borderline REMAINS OPEN** (recorded as such, not forced). The fine-grid extension was NOT run — Gate 2 gates it. Resolving H=0.10's true α needs the **explicit sim at finer n** (the validated weak-order tool, O(n²)) or an **exact-noise lift** (O(N²)) — neither is the cheap path.
  - **THE GATE'S VALUE (the methodological point).** Gate 2 — *demand the lift reproduce a KNOWN answer (brick-3's resolved α) before trusting it on the unknown (H=0.10)* — caught this BEFORE a false resolution: the lift's α≈0.85 would have LOOKED like a clean H=0.10 answer (borderline-to-PASS) and been stated confidently. The known-answer-gate discipline prevented a confidently-wrong result. Files: `layer4_convergence.py` (`sim` callback), `layer4_lifted_alpha.py`, `test_layer4_convergence.py`.

**2026-06-28 — Layer 4: OTM-smile gate — high-ν OTM smile validated vs CF, SPX-viable on the skew wing (D36)**
- **D36** *(2026-06-28)* **OTM-smile gate (the SPX prerequisite) — the lift's high-ν OTM implied-vol smile validated vs the CF at H≈0.10: (B) WING DIVERGENCE, ASYMMETRIC — ATM + the OTM PUT wing clean (~0.2pp), OTM CALL wing over-priced ~1pp. SPX put-wing/ATM skew calibration is VIABLE.** `layer4_smile_gate.py`: one conditional-MC lifted run → per-path (S_eff, sig_eff) → analytic BS across all strikes (the lifted smile) vs the gil-pelaez CF smile, compared in **implied-vol space** with a vega-reliability mask. New BS-IV inverter (`bs_put`/`bs_vega`/`bs_iv` in `rough_heston_cf.py`). 5 mechanics tests; 52/52 full suite.
  - **PURPOSE.** Validate the lift's high-ν OTM *smile* (IV space, across strikes — where SPX calibration lives) vs the CF known-answer BEFORE any market calibration, closing the gap thread-2 surfaced (4c validated ATM only; does the H=0.05 ~20% OTM tail-amplification reach the SPX-relevant H≈0.10?). Calibrating before this = fitting parameters to a possibly-biased tail (can't separate calibration failure from pricing failure).
  - **RESULT — (B) WING DIVERGENCE, ASYMMETRIC** (H=0.10, ν∈{0.30,0.40}, T=1): **ATM + the entire OTM PUT wing are CLEAN** (lifted−CF ≤ ~0.2pp; ATM +0.10 / +0.18pp confirms 4c). The **OTM CALL wing is over-priced**: +0.46→**+0.81pp** (ν=0.30) and +0.82→**+1.18pp** (ν=0.40), **peaking at vu≈1–1.5 then easing** (NOT a deep-OTM blow-up). So thread-2's tail-amplification **reaches H=0.10 but mildly and on the call wing only** (vs ~20% at H=0.05).
  - **MECHANISM (connects to D35).** The call wing (high S_T) is the **low-integrated-variance tail**, where the QE/shared-ΔW envelope's variance-distribution perturbation shows up; the **high-variance put/crash wing is matched**. The price-level fingerprint of D35's envelope perturbation, now mapped across the smile (asymmetric in the variance tail). Characterized — not chased further (diminishing returns).
  - **SPX VIABILITY CALL (the gate's purpose).** Equity skew lives in the **PUT wing** (the CF's steep downside skew — IV ~28% at −2.5vu vs ~13% at +2vu). The lift is **clean there + ATM → SPX put-wing/ATM skew calibration is VIABLE.** The OTM **call wing** carries a ~1pp bias (growing with ν) → **exclude it from the fit OR carry a documented ~1pp caveat**. Not (A) clean, not (C) a deep-OTM cliff — a characterized, asymmetric, modest bias.
  - **ν-RANGE BOUND (honest scope).** Tested at **ν=0.30–0.40**; the put-wing-clean claim holds across this range. The real SPX-implied ν should fall in/near it — to be confirmed when calibration runs (the call-wing bias GROWS with ν, so the real-SPX ν sets the caveat's size).
  - **METHOD.** The **conditional MC makes the strike sweep ~free** (one lifted run → analytic BS across all K). A **vega-reliability mask** bounds the IV inversion (put wing reliable to −2.5vu; the thin-vega deep call wing is the inversion limit). Files: `layer4_smile_gate.py`, `test_layer4_smile_gate.py`, `rough_heston_cf.py` (bs_put/bs_vega/bs_iv), `output/layer4_smile_gate.png`.

**2026-06-28 — Layer 4: SPX calibration engine — built + validated vs synthetic CF known-answers; H unidentifiable from one smile (D37)**
- **D37** *(2026-06-28)* **SPX calibration engine (sandbox, synthetic known-answer) — fits θ=[H,ν,ρ,ξ₀] to an IV smile via least_squares (TRF, IV space) against the CF; validated CF→CF EXACT; the single-maturity inverse problem identifies ξ₀/ρ/ν but NOT H; the lift is calibration-grade for ν/ρ/ξ₀ (its bias quarantines in H).** `layer4_calibrate.py` + `test_layer4_calibrate.py` (9 tests; full suite green). Real-market SPX data is not reachable in-sandbox → the run on real SPX is a **documented later step on the user's machine**.
  - **GOAL / setup.** Fit θ=[H,ν,ρ,ξ₀] by minimising IV-RMSE (IV space — the calibration-relevant metric, per D36) against the CF (fast/analytic; the optimiser evaluates the smile many times). Single-maturity reduction: **ξ₀→V0=θ** (flat forward variance), **kappa FIXED=0.30** (a single maturity cannot identify mean-reversion). The project's KNOWN-ANSWER discipline applied to calibration: validate vs CF-generated truth before any real use. (Truth = H=0.10, ν=0.35, ρ=−0.70, ξ₀=0.04; FULL grid vu∈{−2.5…+2.0}, PUT+ATM vu∈{−2.5…0}.)
  - **(1) CF→CF GATE — (A) PASS, EXACT.** From a *distant* init, recovers all 4 params to **0.00% error, IV-RMSE 0.0000**, both grids (nfev 8/14). The optimiser + objective are validated against the exact model; no global search needed for noise-free synthetic.
  - **(2) IDENTIFIABILITY (JᵀJ at truth).** **ξ₀/ρ/ν IDENTIFIED; H WEAK** (the flat eigen-direction ≈ pure H). cond(JᵀJ) = **6.2e5 (FULL) / 2.4e7 (PUT+ATM — WORSE)**; sensitivity ξ₀(6.6)≫ρ(0.31)>ν(0.26)>H(0.042); degenerate **H~ν (−0.82 full / −0.92 put)**. A single-maturity smile pins an **H·ν combination, not H alone** — and dropping the call wing worsens H/ν separation.
  - **(3) NOISE-ROBUSTNESS (perturb target, recalibrate ensemble).** ξ₀/ρ **rock-solid** (rel spread ~1–2%), ν **stable** (2–10%), **H BLOWS UP** — 62% spread at 0.1pp noise → 153% at 0.5pp, with H_mean drifting 0.089→0.141. **A single smile cannot identify H under realistic market noise.**
  - **(4) ★ LIFT→CF CALIBRATION-GRADE** (target = CF-truth; model-in-loop = LIFT, conditional-MC + CRN). The lift's ~1pp call-wing bias (D36) is **ABSORBED INTO H** (+166% FULL / +82% PUT), leaving **ν/ρ/ξ₀ CLEAN on the FULL grid (~5%, IV-RMSE 0.018pp)**; the **PUT-WING-only is WORSE** (ν +18%, ρ +12%) — its ill-conditioning spreads the misfit. **CONDITIONING BEATS WING-CLEANLINESS for calibration.**
  - **THE UNIFYING FINDING.** **H is the weak parameter EVERYWHERE** — both random noise (3) AND the lift's systematic bias (4) corrupt **H specifically**, because it is the flat eigen-direction of the single-maturity inverse problem. The calibration-context echo of the project's H-identifiability theme (`interpret_h.py` / `identifiability_map.py`).
  - **ACTIONABLE SPX GUIDANCE.** (1) **Calibrate with the CF** (exact — the lift isn't needed for vanilla-smile calibration). (2) From one maturity **trust ξ₀/ρ/ν, DO NOT trust H** — fix it (roughness literature) or add maturities → the **multi-maturity surface is the well-motivated next step**. (3) If the lift MUST be used (path-dependent products, no CF), its bias lands on H (which you're fixing anyway), so calibrate **ν/ρ/ξ₀ on the FULL grid** (clean ~5%) and **DON'T restrict to the put wing** — this REFINES D36's pricing-guidance for the *calibration* context.
  - **HARDENING (infra note, honest).** The four experiments first hit process-oversubscription **OOMs** (a default 16-worker pool + concurrent runs starved a memory-tight machine) and one contention-induced `eigh` NaN — diagnosed as **contention, not logic**; fixed (NaN-robust `identifiability_report`, 6-worker pool cap, fixed M for reproducibility) and re-run serially/bounded to clean results.
  - **SCOPE.** OUT: real-market SPX fetch + fitting to *market* vols (later step, user's machine); multi-maturity surface. IN: the engine + the CF→CF gate + the noise/identifiability characterisation + the lift→CF calibration-grade test. Files: `layer4_calibrate.py`, `test_layer4_calibrate.py`.

**2026-06-29 — Layer 4: multi-maturity surface calibration — the term structure makes H usable at low-noise data (D38)**
- **D38** *(2026-06-29)* **Multi-maturity SURFACE calibration — extends D37 to a joint IV surface across maturities and tests whether the term structure identifies H (which a single smile structurally cannot, D37). Verdict: (B) IMPROVES STRONGLY, PARTIAL — the surface makes H USABLE at realistic low-noise data (H-spread ~106%→~10% at σ=0.1pp) without structurally breaking the H~ν degeneracy; κ stays fixed.** `layer4_calibrate_surface.py` + `test_layer4_calibrate_surface.py` (44 tests green). Sandbox-buildable (CF-based, no market data — the prerequisite for the your-machine real-market run). Imports D37's core (calibrate/residuals/IdentReport) unchanged; adds T-parameterised smiles, √T-scaled per-T strike grids, a stacked surface residual, and a dimension-agnostic identifiability report.
  - **(1) SURFACE CF→CF GATE.** Recovers all params incl. H **EXACTLY (+0%)** in the noiseless limit — the optimiser/objective work on the surface. (NOTE: noiseless-exact recovery does NOT mean H is well-identified under noise — EXP2/EXP3 are the real tests; this gate only validates the machinery.)
  - **(2) IDENTIFIABILITY.** Surface cond(JᵀJ) improves **~100×** vs single smile (8.97e5 → 8.5e3) — BUT H stays the flat direction (|flat[H]|≈0.89) and H~ν stays correlated (−0.85). The surface pins the **H·ν COMBINATION** far better **without STRUCTURALLY breaking the degeneracy**.
  - **(3) ★ NOISE-ROBUSTNESS (the practical payoff).** H-spread at matched knobs (this run's own single-T baseline — N=900/nn=100/5-strikes/same seeds, NOT D37's number; see caveat a): σ=0.1pp **~106%→~10% (~10×)**; σ=0.3pp ~190%→~29% (~6.5×); σ=0.5pp ~194%→~48% (~4×). ν/ρ/ξ₀ stay tight throughout (surface ≤10%). At clean data the surface pins H to **~10% (H≈0.10±0.01) — genuinely USABLE** — vs hopeless (~106%) from one smile. The gain SHRINKS as noise grows (10×→6.5×→4×) **because the soft H~ν direction persists** (EXP2): the surface adds enough info to pin H usefully at low noise, not to eliminate the soft direction. This degradation curve is the evidence for **(B)-not-(A)**.
  - **(4) κ DECISION — KEEP κ FIXED.** Freeing κ **inflates cond 550×** (8.5e3 → 4.66e6); κ becomes the degenerate direction (|flat[κ]|=0.96), recoverable only noise-free (under noise it would blow up like H). With flat ξ₀ the drift κ(θ−V) cancels in expectation → the surface can't usefully pin κ. The 4-param choice (D37) is now **evidence-confirmed on the surface**.
  - **VERDICT — (B) IMPROVES STRONGLY, PARTIAL.** The surface makes H usable at realistic (low-noise) data where a single smile cannot, WITHOUT structurally breaking the H~ν degeneracy. The calibration-context refinement of D37's "H is the weak parameter everywhere": even with the term structure H remains the soft direction — the surface **MITIGATES** its weak identifiability (usable at low noise, degrading under noise), it does **not CURE** it.
  - **ACTIONABLE REAL-MARKET GUIDANCE (span sweep).** cond falls monotonically with maturities (1→9.6e5, 3→5.2e4, 4→2.0e4, 5→9.8e3); the **SHORT maturity (T=0.1) gives the biggest single jump** (4→5 halves cond, |flat[H]| 0.95→0.89) — the T^(H−1/2) skew-term-structure signal. → for the real-market run: use **≥5 maturities INCLUDING a short (~1-month) tenor**; more/wider keeps tightening. Keep κ fixed.
  - **HONEST CAVEATS (load-bearing).** (a) **MATCHED-KNOB COMPARISON:** the headline ratios use THIS run's own single-T baseline (~106%, matched knobs), NOT D37's recorded single-smile spread (62%, finer knobs). The single-smile H-spread is itself knob- and ensemble-sensitive — 62% (D37, finer CF) → 96% → 106% (here, trimmed-for-cost knobs) across runs; the comparison is internally consistent (matched knobs) and that sensitivity is itself a minor finding. (b) **SPREAD-ESTIMATE NOISE:** H-spread is a scatter over 10 draws; 10-draw estimates carry real sampling error (the 62/96/106 drift demonstrates it) → the reported spreads are APPROXIMATE (~10/~29/~48%) and the factors "roughly" (~10× etc.), NOT precise; the qualitative finding (order-of-magnitude tightening at low noise, degrading under noise) is robust, the exact factors are not. (c) **H REMAINS STRUCTURALLY SOFT:** the surface improves conditioning + practical identifiability at low noise; it does not eliminate the H~ν degeneracy (EXP2 |flat[H]|=0.89, corr −0.85) — why it's (B) not (A).
  - **INFRASTRUCTURE LESSON (like D37's oversubscription note).** The surface noise-ensemble is expensive (~436s/recovery: 25 CF inversions × ~0.5s × ~30 evals). A first run **STALLED 2+ hours** with active CPU but zero output — diagnosed (via per-eval instrumentation that REFUTED both the max_nfev and tolerance hypotheses) as raw cost × **POOL OVERSUBSCRIPTION** (numpy/BLAS spawning threads inside each of 6 pool workers, stacking past the 4 physical cores), NOT convergence (calibrations converge at nfev~6, status=gtol). FIX (baked into the committed module): pin BLAS to **1 thread/worker** (OMP/OPENBLAS/MKL=1, set at module level before numpy import so spawned workers inherit it), cap the pool at **physical cores (4)**, keep the ensemble at **10 draws** (never shrink the spread to save time — fix COST not the measurement), trim strikes 7→5 only after a conditioning check confirmed it (cond 9.8e3 vs 8.5e3, identical H-flatness), print **per-member progress**, and **ESTIMATE runtime before launching**. Files: `layer4_calibrate_surface.py`, `test_layer4_calibrate_surface.py`.

**2026-06-29 — Layer 4: real-market calibration — rough-Heston fit to live Deribit BTC options; H non-identified, model under-fits the crash tail (D39)**
- **D39** *(2026-06-29)* **Real-market run (task 1) — the validated surface engine (D38) calibrated to LIVE Deribit BTC options. Rough-Heston FITS the gross smile (~0.9pp) and recovers ξ₀/ρ/ν plausibly, but (1) H is NON-IDENTIFIED (3 independent diagnostics — the option-calibration angle confirming the observational-equivalence wall) and (2) the model UNDER-PRODUCES crypto's extreme put-tail.** `deribit_surface.py` (fetcher+cleaner, 8 offline tests) + `calibrate_btc.py` (driver) + a saved snapshot (reproducible offline) + the fit figure. Crypto chosen: free/accessible data + the high-roughness/high-ν regime the lift was built for (SPX option data unavailable — the engine transfers unchanged if/when it is). All D38 guidance carried (≥5 maturities incl. short, calibrate vs CF, κ fixed, H soft). Runs on the user's machine (data not reachable in-sandbox).
  - **THE DATA (Phase 0/1).** Deribit public API (no auth): 874 BTC options / 12 expiries. Conventions pinned from the LIVE API (not assumed): IV in %, **moneyness via the FORWARD** (`underlying_price`, not spot), inverse/BTC settlement IRRELEVANT in IV space. Cleaned: OI≥5, two-sided quotes, spread-weighted, vega mask → **63 points / 5 maturities (11 days → 6 months)**, put-wing+ATM, forward-normalised (K_norm=100·K/forward). Snapshotted for offline reproducibility.
  - **THE BTC FIT** (full weighted calibration vs CF, converged): **H=0.0201 (railed to LB) · ν=0.636 · ρ=−0.371 · ξ₀=0.230 · IV-RMSE 0.888pp.**
  - **★ FINDING 1 — H IS NON-IDENTIFIED** (the predicted headline), via THREE independent diagnostics: (a) **Jacobian at θ̂: |flat[H]|=0.99, cond 5.2e4, corr(H,ν)=−0.96** — H is ~entirely the flat eigen-direction (the D37/D38 diagnostic, now even more extreme); THIS is the primary evidence. (b) H **rails to the LB (0.02) in 100% of 24 bootstrap resamples**. (c) **Cross-run instability** — two identical-config runs landed at DIFFERENT (ν,ρ)=(0.54,−0.46) vs (0.64,−0.37) at the SAME ~0.9pp RMSE: the flat H~ν~ρ valley made operationally visible (ξ₀ robust ~0.23 both runs; ν/ρ drift along the valley; H unpinnable). **THEME TIE:** the option-calibration angle independently CONFIRMS the observational-equivalence wall (Phase B crypto Ĥ≈0.08; D37/D38 soft-direction) — H-identifiability is hard in real crypto markets, now shown from TWO independent angles (realised variance AND option calibration).
  - **★ FINDING 2 — THE MODEL UNDER-PRODUCES CRYPTO'S EXTREME PUT-TAIL** (a distinct, novel model-limit finding — NOT buried under Finding 1): the figure shows rough-Heston captures the gross put-skew (~0.9pp) at every tenor BUT **systematically undershoots the steep deep-put wing** (market left-tail above the model) and the longer-tenor curvature (T=0.49 market convex, model skew-only). Even at MAXIMAL roughness (H at the floor) the model can't reach crypto's crash-fear tail steepness. This **COMPOUNDS Finding 1**: the optimiser rails H to the bound TRYING (and failing) to reach the steep tail — so H-at-bound is partly non-identification, partly the model straining for a tail it can't produce. **Real crypto is not rough-Heston; its left tail is steeper than the model** (future-modelling motivation: jumps / a steeper kernel).
  - **PLAUSIBILITY (the identified parameters).** ν=0.64 **HIGH** (the D34 lift regime) ✓; ρ=−0.37 put-skew ✓; ξ₀=0.23 (ATM vol ≈48% ≈ market) ✓; the model fits the gross smile (~0.9pp) ✓. Three of four predictions confirmed; the fourth (H soft) confirmed as outright non-identification.
  - **LOAD-BEARING CAVEATS (these make the result honest).** (1) **THE WARM-START BOOTSTRAP** inits at θ̂ (warm, for speed) → all draws stuck at the bound → **H-std≈0; that ≈0 is "RAILED TO THE BOUNDARY", NOT tight identification** — the opposite reading would be wrong. The verdict rests on the JACOBIAN + cross-run instability + 100%-at-LB, NOT the (degenerate) std. A cold/dispersed-init bootstrap would show the valley WIDTH (~150 min) but the jacobian already proves the flat direction. (2) **REDUCED ≤6mo SURFACE** — the 1-year tenor was dropped (the CF OVERFLOWS even at N_riccati=4000 at the rough/high-ν/long-T corner — itself a computational-limit finding). Per D38 the long tenor carries H term-structure span, so dropping it WEAKENS H-identification further → Finding 1 is COMPOUNDED (intrinsic difficulty AND missing span — we cannot fully separate them). **OPEN QUESTION (future work):** does adding the long tenor back (via per-maturity N_riccati) tighten H, or is it non-identified even with full span? (3) **H=0.02 is near the CF-overflow floor** (H<0.02 NaNs the long tenor) — so the LB-pin is partly "data wants very-rough H" + partly the numerical floor; either way non-identified. (4) **ν/ρ carry valley-coupled uncertainty** (ν~0.5–0.65, ρ~−0.37 to −0.46 across runs); **ξ₀ (~0.23) is the robustly-identified parameter** (the level is pinned, the valley-coupled params are not).
  - **INFRASTRUCTURE/METHOD (honest, like D37/D38).** (i) Phase 0 recon caught the Plan agent's PHANTOM endpoint (`get_ticker` doesn't exist → `get_order_book`) and a cf_kw-dict-vs-keyword signature mismatch — verify the real API/signatures from source, don't assume. (ii) **THE 13× RICCATI-CACHE:** the fractional Riccati was re-solved per-strike though it depends only on (T, params); memoising the CF per maturity gave **BIT-IDENTICAL IVs (max diff 0.00)** and **13× speedup** (no engine edit, no science tradeoff) — turned a ~4-hour run into ~40 min. (iii) Buffering discipline: tee (not grep) + per-eval/per-draw progress + smoke-before-the-long-run (caught a numpy-2 `.ptp()` bug) + runtime-estimate-before-launch. Live-fetch is NOT in tests (offline fake-fetcher only); the snapshot makes the calibration reproducible offline.
  - **VERDICT.** Rough-Heston CALIBRATES to a real options market (BTC) — recovers ξ₀/ρ/ν plausibly, fits the gross smile to ~0.9pp — and **H is NON-IDENTIFIABLE** (the option-calibration angle confirming the obs-equivalence wall), while the model **UNDER-PRODUCES crypto's extreme put-tail**. A non-identified H is the EXPECTED, theme-confirming result, measured now from LIVE market data. SPX is the unchanged-engine transfer for later; ETH is the natural comparison follow-up. Files: `deribit_surface.py`, `test_deribit_surface.py`, `calibrate_btc.py`, snapshot, figure.

**2026-06-30 — Layer 3: deep-hedging engine — deep beats delta under frictions, but roughness adds no hedging edge beyond it (D40)**
- **D40** *(2026-06-30)* **Layer 3 deep-hedging engine — the fourth "is roughness USEFUL?" question (after execution's no-edge, Q3): does roughness give a HEDGING edge beyond the generic frictions effect? Answer: NO. Deep hedging beats delta under frictions (the Buehler edge, +1.1 CVaR, 8/8 seeds), but the roughness increment is MODEST/ABSENT (+0.060±0.044, z=1.4) and path-signatures add no advantage over the instantaneous Markovian state.** Built as a clean, isolated, gate-disciplined core (Buehler-style direct policy optimisation, NOT actor-critic). `layer3_deep_hedging.py` + `test_layer3_deep_hedging.py` (separate suite). **STRICT ISOLATION (deletion-safe leaf):** torch lives only in `.venv-layer3` (the repo's first ML dep; NOT committed — `requirements-layer3.txt` documents the recreate); the core stays numpy/scipy-only and its CI never imports torch.
  - **THE SHARPENED QUESTION (on-theme).** Deep hedging beats delta under frictions in ANY model (generic Buehler effect) — so the experiment is a CONTRAST: the deep-vs-delta CVaR edge on a ROUGH market (H=0.10) vs the SAME edge on a SMOOTH control (H=0.5, identical generator/params, only H differs). The roughness-specific result is (rough-edge − smooth-edge).
  - **PHASE 0 (foundation, verified).** Self-computed truncated signature in numpy (signatory is dead on py3.14; iisignature risky) — VERIFIED against four known answers: straight-line = exp-tensor (err 0), Chen/reparam invariance (err 3.5e-18), closed-loop level-1 = 0, unit-triangle signed area = 0.5 exact. torch autograd sanity (∇‖x‖²=2x, err 0). Signature dims: 39 (depth 3), 120 (depth 4).
  - **GATE 1 (framework validation — the CF→CF analogue).** In the frictionless GBM complete-market limit the trained CVaR-policy RECOVERS BS-delta: mean|δ−Φ(d₁)| = 0.016 (sig) / 0.018 (simple), P&L-std ratio 1.03–1.04 (**MATCHES** delta, does not beat it — correct, since delta is variance-optimal frictionless; beating it would be the red flag). Both feature modes recover delta → framework AND signature pipeline correct. Causality guard CAUSAL (z=1.6); a unit test confirms a LOOK-AHEAD policy FIRES the guard (the Layer-2 trap, explicitly defended).
  - **GATE 2 + CONTRAST (the result)** — well-trained simple (t,S,√V) features, c=0.01, 8 seeds, disjoint test set, paired: **ROUGH edge (deep beats delta) = +1.143 ± 0.029 (8/8 seeds); SMOOTH edge = +1.083 ± 0.052 (8/8); ★ ROUGHNESS INCREMENT = +0.060 ± 0.044 (z=1.4, 6/8) → MODEST/ABSENT (PREDICTED).** Deep clearly beats delta under frictions (the generic Buehler edge); the rough structure adds little beyond it.
  - **SCRUTINY 1 — false positive CAUGHT.** The signature-feature run gave increment +0.167 (z=2.2) and was flagged SUSPECT. The simple-feature control revealed the sig net had trained unevenly (rough +0.146, smooth −0.021) — NOT a roughness edge; with the well-trained simple policy the increment is +0.060 (z=1.4, n.s.). Scrutinising the positive result (the ML-false-positive discipline) dissolved the red herring.
  - **SCRUTINY 2 — over-claimed negative AVOIDED (the mirror).** The matched-budget sig underperformance (+0.15 vs simple +1.14) could be read as "signatures don't help," but a FAIR-budget check (hidden=64, epochs=1000, n_train=50k vs the matched hidden=32/500ep/20k) showed the sig net CATCHES UP to **+1.031 ± 0.089 ≈ simple +1.14** → the matched-budget gap was a TRAINING ARTIFACT (the 39-dim path-history input needs more budget than the 3-dim state; an intermediate hidden=128/1200ep/20k run OVERFIT to −0.286 — more capacity without more data overfits). **SOFTENED sub-claim:** signatures MATCH but do not BEAT the simple Markovian state, at ~2.5× training cost → the path-history provides no hedging ADVANTAGE, only added cost (not "worse").
  - **VERDICT.** Deep hedging beats delta under frictions as expected (+1.1 CVaR, 8/8 seeds), but **the rough structure adds little hedging edge beyond frictions** (increment +0.06, z=1.4, n.s.), and path-signatures provide no advantage over the instantaneous Markovian state (match at higher cost). The fourth "is roughness useful?" question answered NO — consistent across all four layers: roughness is hard to identify (Q1 realised-variance / Q4 option-surface), not cheaply priceable via MLMC (Q2), not tradeable (Q3 execution), and not a hedging edge beyond frictions (Q-hedge here).
  - **ISOLATION / INFRA (honest).** torch 2.12 in `.venv-layer3` (gitignored, not committed); core numpy/scipy-only, CI torch-free. Test suite: **CORE env 5 passed / 2 torch-skipped** (proving the file is harmless to the torch-free core), **venv 7/7**. Phase-0 hit the Windows cp1252 print-encoding gotcha (fixed via stdout-utf8). Runtime-estimate-before-launch held (timed one training → ~16min Gate-2 contrast). Files: `layer3_deep_hedging.py`, `test_layer3_deep_hedging.py`, `requirements-layer3.txt`.

**2026-06-30 — Layer 4: extension (2b) — full-span H test (per-maturity N_riccati): the reduced-span caveat dissolves**
- **D41** *(2026-06-30)* **Layer-4 extension (2b) — the calibration paper's reduced-span caveat DISSOLVES. With the full 6-maturity span (incl. the 1-year tenor) made computable by a per-maturity `N_riccati` schedule, H is STILL non-identified on the live BTC surface — and the jacobian comparison shows adding the 1-year tenor changes H-identifiability by essentially nothing (Δ|flat[H]| = −0.001). The non-identification is an intrinsic H↔ν degeneracy, NOT the dropped span; the per-maturity H-sensitivity proves the long tenor is the LEAST H-informative maturity (5.2 vs 64.5 at the short end). D39's reduced span was not the cause.**
  - **GOAL.** Resolve D39's open caveat. D39 found H non-identified on the live BTC surface BUT dropped the 1-year tenor (the CF overflows at long-T/high-ν under uniform N=4000), and the D38 intuition held the long tenor carried the most H term-structure signal — so the non-identification was COMPOUNDED (intrinsic difficulty + missing span) and couldn't be separated. (2b) asks: does adding the 1-year tenor back TIGHTEN H, or is H non-identified even with the FULL span?
  - **PHASE-0 MINI-FINDING — D39's "1-year tenor intractable" was a UNIFORM-N artifact, not fundamental.** A per-maturity CONSTANT-STEP schedule `N(T)=clip(round(T/h),800,8000)`, h=1.23e-4 → N = 800,800,1314,1937,3964,8000 across the six maturities (short tenors cheap, the long tenor gets N=8000) makes the full surface finite + converged at every corner INCLUDING the railed H=0.02/ν=0.63 at the 1-year tenor. A probe showed the H=0.02/T=0.986 overflow MOVES with N (nan at N=6000 → finite + converged at N=8000, IV 0.4285 bit-identical through N=14000 — a RESOLUTION limit, not a wall). The 13× Riccati-cache stayed active (3 solves/maturity, strike-independent u-grid — verified: CF calls = 3 regardless of strike count). **★ VALIDITY CONFIRMED:** NaN-at-solution = 0/69 at the converged θ̂ (was 6/69 at N=6000) — the 1-year tenor is genuinely PRESENT where H lands, so "H rails with full span" is a CLEAN result, not the tenor dropping out at the rail (the same "a result that loses its key constraint at the solution is confounded" discipline as the warm-start / signature-artifact catches).
  - **THE RESULT (the jacobian comparison at θ̂, full span vs 1-year-tenor-dropped — same θ̂, only the rows differ, so it isolates the tenor's marginal H-information; forward-diff on H so the tenor stays finite).** FULL (6 mat): cond(JᵀJ)=9.83e3, λ_min=6.97e-3, |flat[H]|=0.118, corr(H,ν)=−0.876. DROP 1-yr (5 mat): cond=1.01e4, λ_min=6.20e-3, |flat[H]|=0.119, corr(H,ν)=−0.911. Δ: cond ×0.97, **|flat[H]| −0.001** — adding the 1-year tenor changes H-identifiability by ESSENTIALLY NOTHING → H non-identification is INTRINSIC, not a span artifact. **THE CAVEAT DISSOLVES** (the "Likely" outcome).
  - **★ THE MECHANISM (per-maturity H-sensitivity ||∂IV/∂H||, the valuable part).** T=0.027 (1mo) 64.5 → 0.084: 51.2 → 0.161: 49.4 → 0.238: 31.6 → 0.487: 22.9 → T=0.986 (1yr): 5.2. The H-signal is CONCENTRATED AT THE SHORT END and decays MONOTONICALLY to nearly H-flat at the 1-year tenor (5.2 vs 64.5) — so the long tenor PHYSICALLY CANNOT constrain H, which is WHY dropping it cost nothing. This OVERTURNS the plausible prior (and the paper's reduced-span premise) that the long tenor carried the most H-signal: the opposite is true — H lives at the short end. Figure: `output/layer4_h_sensitivity.png`.
  - **THE TRUE DEGENERACY — H↔ν (corr ≈ −0.88 to −0.91), NOT span.** The short maturities carry the H-signal, but H there trades off against ν, and the long tenor doesn't break the tie. H rails to 0.02 COMPENSATED by ν=0.71 — with the tenor PRESENT and contributing (it does move ν 0.63→0.71 and ρ −0.36→−0.31 vs the N=6000 run; just not H). So the non-identification is a STRUCTURAL PARAMETER DEGENERACY (H↔ν), not a missing-data (span) problem.
  - **VERDICT + WHAT IT MEANS FOR THE PAPER.** Roughness is non-identifiable from this BTC option surface, and now we know WHY — not because the long tenor was missing (it is H-flat and could not have helped) but because of intrinsic H↔ν degeneracy. This RESOLVES the calibration paper's open caveat: the reduced span was NOT the cause. The paper can state the non-identification is intrinsic (the dropped tenor is the LEAST H-informative, shown by the sensitivity profile), removing the caveat.
  - **SCOPE / VALIDITY.** This is a LOCAL diagnostic at the fitted optimum θ̂ (strong — the degeneracy, the H-railing, and the tenor-negligibility all agree). GLOBAL valley-flatness confirmation (a dispersed cold multi-start, ~3–4h — does H scatter at ~equal IV-RMSE?) is RESERVED for the calibration-paper update; the local jacobian already settles the SPAN question (2b) targets. One full-span calibration at N=8000 ≈ 72 min (cache active; the cost is the genuine O(N²) fractional-Adams Riccati); the full dispersed bootstrap (~6h) was NOT run — the jacobian comparison answered the question directly and cheaply (the runtime-estimate discipline: don't run 6h when a ~30-min jacobian comparison answers it mechanistically).
  - **Files.** `calibrate_btc.py` (per-maturity N schedule `n_riccati_of_T` + the per-maturity stability gate in `precheck` + the NaN-at-solution validity instrumentation), `analysis/layer4_span_identifiability.py` (the jacobian comparison + the H-sensitivity figure), `output/layer4_h_sensitivity.png`. Reuses the validated engine (`layer4_calibrate_surface`, `rough_heston_cf`) unchanged. Runs on the user's machine (Deribit data + compute).
- **D42** *(2026-06-30)* **Layer-4 extension (2a) — CROSS-MARKET replication: live ETH options show the SAME two findings as BTC (D39/D41). H is non-identified (rails to the bound; H↔ν degeneracy corr(H,ν)=−0.855, if anything MORE degenerate than BTC — λ_min 3.6e-3 vs 7.1e-3) and rough-Heston UNDER-produces the crash-fear put tail (signed deep-put −0.42 vs BTC −0.35 vol-pts, worse at the long tenor). Roughness non-identifiability from crypto option surfaces is a STRUCTURAL cross-market phenomenon (H↔ν degeneracy + short-end-concentrated H-signal), NOT a BTC idiosyncrasy — broadening D39/D41 from "BTC" to "crypto."**
  - **GOAL.** Replicate D39 (H non-identified) + D41 (the H↔ν mechanism + the put-tail undershoot) on a SECOND live market (Deribit ETH). A clean replication broadens the finding from BTC to crypto and strengthens the calibration paper's spine. The pipeline is currency-agnostic and inherits D41's per-maturity N=8000 work.
  - **★ DATA FINDING — cross-market calibration needs SCALE-INVARIANT cleaning + bounds (a methodological contribution).** BTC-tuned ABSOLUTE thresholds silently collapse a cheaper market: under `VEGA_MIN=5` the ETH surface cleaned to 1 maturity / 4 points (vs the full 6 maturities) — vega ∝ underlying price (ETH ~$1.6k vs BTC ~$61k, ~40× lower) AND ∝ √T, so only the longest, highest-vega tenor cleared the floor. FIX (committed): a FORWARD-SCALED vega floor `vega_min × median_forward/FORWARD_REF` (uniform per snapshot) — scale-invariant, transfers to SPX, BTC BIT-IDENTICAL by construction (BTC median == FORWARD_REF → ×1; verified old-vs-new at the same instant: identical grids/targets/weights). Recovered ETH to 6 maturities / 46 points incl. the 1-year tenor. A SECOND scale artifact surfaced in Phase 0: ETH's higher vol (ATM IV ~58%, ξ₀≈0.34) exceeded the BTC-tuned ξ₀ upper bound 0.25 (≈50% vol) → the init was out of bounds. FIX (committed): a DATA-DRIVEN ξ₀ cap `max(0.25, 1.2×max_ATM_var)` — BTC stays EXACTLY 0.25 (its max ATM-var 0.20 → floor binds → D39/D41 bit-identical), ETH gets 0.442. Same philosophy as the vega floor: replace absolute BTC-tuned constants with data-driven, scale-invariant ones. (Honest note: the absolute SPREAD cap was NOT made scale-invariant — loosening it would re-admit 13 BTC quotes and break bit-identical; the vega fix alone recovered ETH, so the spread cap stays absolute — a documented non-generalized filter.)
  - **PHASE 0 (stability + timing).** The per-maturity N schedule is currency-agnostic (keys on T) → every ETH tenor finite + converged + inverts at the crypto corner, incl. the 1-year at N=8000. Validity: NaN-at-solution = 0/46 at the converged θ̂ (the 1-year tenor stays IN where H lands → the full-span test is valid, the D41 discipline). One full-span ETH calibration ≈ 94 min (cache active); the ~8h bootstrap was NOT run — the jacobian + fit diagnostics at θ̂ settle it (the D41 lesson). θ̂: H=0.0200 (RAILED), ν=0.713, ρ=−0.301, ξ₀=0.397 (interior to the 0.442 bound — headroom sufficient), IV-RMSE 1.63pp.
  - **★ THE RESULT — ETH replicates BTC at THREE levels (each at θ̂, forward-diff on H).** (1) RESULT: H rails to 0.0200 on both; ν/ρ nearly coincide (ETH ν=0.713/ρ=−0.301 vs BTC 0.714/−0.306). (2) MECHANISM: same H↔ν degeneracy (corr(H,ν) ≈ −0.86 both) — ETH MORE degenerate (cond 7.95e3, λ_min 3.58e-3 < BTC 7.06e-3; |flat[H]| 0.070 < 0.118, ETH's H even less pinned). (3) REASON: per-maturity H-sensitivity ||∂IV/∂H|| decays identically (BTC 65.2→5.2, ETH 65.7→5.9) — the H-signal at the short end, the 1-year tenor H-flat, on BOTH (the D41 mechanism is crypto-general). (4) PUT-TAIL: ETH under-produces the crash tail too, slightly MORE (signed deep-put residual −0.42 vs BTC −0.35; worst at long tenors, 1-yr RMSE 3.06pp vs 2.43pp). Honest note: the signed residual isolates model-vs-market on deep puts (a real MODEL gap), but ETH's higher overall RMSE also reflects thinner/noisier data (46 pts vs 69) — the undershoot finding holds; read the magnitude with ETH's data quality in mind.
  - **VERDICT (the pre-committed "Likely" outcome — CONFIRMED).** Roughness non-identifiability from crypto option surfaces is a CROSS-MARKET STRUCTURAL phenomenon (the H↔ν degeneracy + the short-end-concentrated H-signal), not a BTC idiosyncrasy; same for the put-tail undershoot. Broadens D39/D41 from "BTC" to "crypto (BTC+ETH)" and strengthens the calibration paper's generality.
  - **Files.** `deribit_surface.py` (forward-scaled vega floor + `FORWARD_REF`), `calibrate_btc.py` (data-driven ξ₀ bound, threaded through estimate/calibrate/assess), `analysis/layer4_span_identifiability.py` (currency-generalized `--currency`/`--theta` + signed put-tail + fit figure), the ETH snapshot `data/deribit/ETH_*.json`, figures `output/eth_smile_fit.png` + `output/eth_h_sensitivity.png` (+ `output/btc_h_sensitivity.png`). BTC bit-identical preserved (both fixes are ×1 for BTC; verified same-instant signature); the 8 offline `test_deribit_surface` + Layer-4 suite pass.
- **D43** *(2026-07-01)* **Layer-3 extension (3) — the friction sweep: D40's roughness-hedging null is ROBUST across transaction-cost levels (c ∈ [0, 0.05]) — roughness adds NO exploitable hedging edge at any friction. An apparent "emergence" at c=0.02 (increment +0.362, z=5.6) survived significance, 8/8 seed-consistency, TWO-budget reproducibility, an edge-convergence fix, AND a baseline decomposition — and was STILL a residual-convergence artifact, unmasked only by a 3000ep finer-convergence check (the increment dissolved to +0.033, z=0.7, while the untrained DELTA_gap control held EXACTLY constant). The hardest-won false-positive catch in the project.**
  - **GOAL.** Robustness check on D40. Does the roughness-hedging null (roughness adds no edge beyond the generic Buehler frictions effect) hold across friction levels, or does the roughness-specific increment emerge at some c? The D40 contrast (rough H=0.10 vs smooth H=0.5 — SAME generator/params, only H differs; simple/Markovian (t,S,√V) features; 8 seeds, paired, disjoint test set) run at each c ∈ {0, 0.005, 0.01, 0.02, 0.05}.
  - **HEADLINE.** The null is ROBUST: at every c, roughness adds no robust exploitable hedging edge. c=0.01 reproduces D40 EXACTLY (+0.060, z=1.4). The generic Buehler edge (deep beats delta) grows with c as expected — the backdrop, not the result.
  - **★ THE SCRUTINY CHAIN (the methodological star of the project).** An apparent emergence at c=0.02 (+0.362, z=5.6, 7/8) was scrutinised in THREE escalating rounds and was STILL an artifact — every signal that normally means "real" fired, and it was still residual undertraining:
    · **ROUND 1 (edge convergence).** The generic Buehler edge MUST grow monotonically with c (physics — more cost → delta over-trades more → deep's edge grows); the 500ep edges instead made a V-shape (the smooth edge COLLAPSED to +0.04 at c=0.02) → undertraining. Cosine-annealed 1500ep monotonised the edges — BUT the increment PERSISTED (+0.326, z=7.2, 8/8). So the edge non-monotonicity was an artifact (fixed), yet the increment survived.
    · **ROUND 2 (baseline decomposition).** Decomposed the increment into DELTA_gap (baseline) vs DEEP_gap (policy): edge = CVaR(δ)−CVaR(deep) per market; increment = DELTA_gap − DEEP_gap. The DELTA_gap was MONOTONIC (−0.036→−0.086→−0.205) — ruling out the D40-style constant-σ delta-baseline-misspecification confound; the sign-flip lived in the DEEP terms, which LOOKED like a real deep-policy roughness effect.
    · **ROUND 3 (finer convergence — the decisive one).** The DEEP_gap is a difference-of-differences, far finer than the edges — so "edges converged" did NOT guarantee "DEEP_gap converged." Pushed c=0.02 to 3000ep: the DEEP_gap shrank 71% (−0.412 → −0.120 ± 0.026) and the increment DISSOLVED (+0.326/z7.2/8-of-8 → +0.033/z0.7/5-of-8, sign-random), WHILE the DELTA_gap held EXACTLY constant (−0.086 → −0.086 — the perfect control: the untrained baseline CANNOT convergence-shift). The artifact signature: the increment shrinks with budget (+0.362 → +0.326 → +0.033) — a real effect is budget-STABLE; one that dissolves as you train harder is a convergence artifact. → residual per-market deep-convergence undertraining, NOT a real roughness effect.
  - **★ METHODOLOGICAL FINDING (the transferable lesson).** (1) Deep-hedging training budgets must SCALE WITH FRICTION (the c=0.01-tuned 500ep budget under-trained at c≥0.02). (2) CRITICALLY: the EDGES converging does NOT guarantee the finer roughness CONTRAST (a difference-of-differences) converged — and statistical significance (z=7.2) + seed-consistency (8/8) + budget-reproducibility (500ep AND 1500ep) ALL FAILED to guarantee convergence. A z=7.2/8-of-8/two-budget result can still be residual undertraining, detectable only by escalating the budget until the fine quantity stabilises. The DELTA_gap (untrained baseline) is the perfect convergence control — it can't shift, so it isolates what's training-dependent.
  - **VERDICT.** Roughness adds no robust exploitable hedging edge at any friction level — D40's null holds across c ∈ [0, 0.05]. The apparent high-friction emergence was residual deep-convergence undertraining (confirmed by the budget-dependence + the DELTA_gap control), NOT a real roughness×frictions effect. Consistent with the spine (roughness not usefully exploitable — Q1 identify, Q2 price, Q3 execute, Q-hedge). Mapping the crossover / the c=0.05 secondary check were unnecessary once the largest anomaly (c=0.02) proved convergence-unstable.
  - **ISOLATION / INFRA.** `.venv-layer3` (torch), core torch-free (venv 7/7; core 5 passed / 2 torch-skipped — the D40 deletion-safe isolation holds). `friction_sweep` + `--sweep`/`--costs`/`--lr-schedule` (cosine, default-OFF so D40 stays reproducible) / `--checkpoint` in `layer3_deep_hedging.py`; per-c checkpoint JSON (resumable); causal guard active per-training (`assert_causal` on the deep policy — r=0 → S a martingale). Sweep ≈56 min; each 3000ep escalation point ≈68 min. Figure: `output/layer3_friction_convergence.png`.
- **D44** *(2026-07-01)* **Weak-order Figure 2 reproducibility correction — the lifted N-sweep value is ≈0.80, NOT 0.854. D35's line-811 ad-hoc N-sweep number does not reproduce under the documented gate-2 setup; D35 is LEFT INTACT as the historical record, and this entry records the correction.**
  - **THE RE-RUN.** Regenerating `fig_lift_nsweep.png` from a now-committed driver (`layer4_lifted_alpha.py --nsweep`) gives coupled a_cpl = 0.804 / 0.803 / 0.803 across N = 150 / 300 / 600 (flat, spread 0.001, ~0.20 below the 1.01 explicit-scheme truth). The negative-result SCIENCE is unchanged — flat across N (N-independent), below truth → the perturbation is intrinsic to the shared-increment NOISE ENVELOPE, not SOE kernel truncation (more factors do not close the gap).
  - **RECONCILIATION WITH D35.** The reproducible 0.80 MATCHES D35's own gate-2 record (line 810, a_cpl ≈ 0.80). D35's line-811 N-sweep value of **0.854** came from a window/config that was never saved and does NOT reproduce under the documented gate-2 setup (n≤64, seed=7, M=200000). So the discrepancy is D35-INTERNAL (its gate-2 0.80 vs its N-sweep 0.854), resolved in favour of the reproducible 0.80.
  - **RESOLUTION.** The figure plots the reproducible 0.80 (NOT fudged to 0.854); the paper text (`OVERLEAF/P4/weakorder_paper_clean.tex`) was corrected 0.854 → ≈0.80 so text and figure agree. Separately, `fig_alpha_H.png` (Figure 1) had been the WRONG figure — a Layer-2 execution-alpha audit — and was replaced with the real single-panel α(H) from `layer4_convergence.py --sweep` (α = 0.75 / 0.86 / 1.01 at H = 0.05 / 0.10 / 0.20, matching the caption).
  - **REPRODUCIBILITY GAP CLOSED.** The N-sweep now has a committed driver (`--nsweep` + `plot_nsweep`) + data (`output/layer4_lifted_nsweep.json`) + figure — it was run in D35 but never saved. Runtime measured before launch (D41/D42 discipline): `--sweep` ≈10 min, `--nsweep` ≈9.5 min (coarse gate-2 grids, no OOM — the fine-grid H=0.10 regime was the expensive one). Core stays torch-free.
- **D45 (2026-07-02): Publication structure finalised at four papers.** The seeds
  are updated from the original three-paper plan (P1-absorbed / P2-pricing /
  P3-identifiability) to the shipped four: P1 option-surface identification, P2
  pricing, P3 realised-variance identification, P4 weak-order convergence. The
  "P1" label is repurposed (absorbed-baseline → option-surface paper); the naive-MLMC
  baseline remains inside P2 per D23. Seeds section updated in place (current-plan,
  not append-only log); this entry records the change.
- **D46** *(2026-07-03)* **Public interactive tour shipped + wired into the README.** The six browser explorers (engine + Q1–Q5) and their hub are published via GitHub Pages from `docs/tour/` (hub = `index.html`), live at `https://michaellumor.github.io/roughvollab/tour/`. README now leads with a tour link and the five-questions poster is a clickable entry point. Each explorer runs client-side with no build step or dependencies beyond web fonts; every reported number is traced to its decision (D20–D23 pricing, D24–D26 execution, D37–D39/D41 calibration, D40/D43 hedging, Q1 estimators), and the illustrative visuals are labelled as such. Docs/infra only — no core code touched; `.nojekyll` added, Pages source = `/docs` on `main`.
- **D47** *(2026-07-09)* **L1-1 resolved — `fbm_hybrid` Volterra normalisation corrected; the quarantined Layer 1 teaching engine now matches the validated core.** The long-logged KNOWN ISSUE L1-1 (above) is fixed. `fbm_hybrid` was rewritten to the discrete-variance construction — `√(2H)·` FFT-convolution of the BLP κ=0 weights via a self-contained `_volterra_weights` — and the rBergomi compensator switched from the continuum `t^{2H}` to the discrete `v[k] = Var(W̃_{t_{k+1}})`, removing the ~18% forward-variance bias (E[V_t] → ξ₀). Both acceptance tests from the original fix plan pass under Monte Carlo, now pinned by `test_layer1_rough_vol.py`. Shipped as PR #53 (RVL-001/002/016/017). **Numbering / scope note (honest):** this is *not* a new result — the correct discrete-variance compensator and its regression guards have existed and been enforced in `roughvol_core.py` / Layer 1b since **D2** and **D11**; L1-1 was only that the quarantined Layer 1 teaching module (nothing imports it — `roughvol_core.py` is the pricing engine, so pricing was never affected) had not been brought in line. Recorded here as **D47**, the next sequential entry, dated to the fix's merge (2026-07-09); the bug it closes was first logged 2026-06-12.
- **D48** *(2026-07-25)* **Audit tail: source lost, reconstructed, made durable.** A read-only survey found the audit page's claim that the remaining ~15 tail items were "logged in the ROADMAP" to be inaccurate: the ROADMAP holds no item list, and the original itemised audit body (2026-07-02/05) survives only as its reconciliation header. All 15 item natures were reconstructed from dated session records (2026-07-11, 2026-07-19). Decision: commit `docs/AUDIT_TAIL.md` as the tail's system of record; correct the audit page to point at it with accurate counts; retire RVL-015/032/034/043/044 (cosmetic, detailed blocks unrecoverable — retire rather than re-audit); keep 10 items open at recovered-spec/nature level, RVL-009 flagged as the next real loop. Honest-tracking principle: a public "still tracked" claim must point at a file that actually contains the items.
- **D49** *(2026-08-02)* **Dissertation citation state sealed.** Figure-reproduction script committed (`plot_rho_sweep.py`, PR #79, issue #78), fulfilling the stated principle that every reported figure is reproducible from committed code; annotated tag `v1.0-dissertation` pushed at `de47ca1` as the repository state cited by the BSc dissertation. Open audit-tail items continue per `docs/AUDIT_TAIL.md`.
- **D50** *(2026-08-19)* **Publication pack, mechanical passes: P2, P3, P1 made
  submission-shaped; verified reference file committed.** One-day prep arc through
  the issue→branch→PR loop, chat-Claude as constructor/verifier, Claude Code as
  executor. P2 (PR #83): draft date removed; SIURO project-advisor placeholder;
  Bourgey–De Marco pinned to the JCF 26(2), 53–82 journal record. P3 (PR #85):
  advisor placeholder; abstract re-scoped to what was measured ("none of the three
  estimators … from daily RV"); explicit scope sentence excluding quasi-likelihood
  (ftw2022) and GMM (bolko2023) estimators from the audit; figure TODO closed; the
  ftw2019/ftw2022 merge deferred until the Wiley record is confirmed. P1 (PR #87):
  all four editorial flag blocks removed (texts preserved in the PR body); the
  flat-direction diagnostic harmonised to |flat[H]| = 0.118 — the §5.4 Jacobian
  diagnostic — throughout; abstract cut 401→250 words; skew-term-structure
  positioning paragraph added on five source-verified references (Fukasawa 2017
  QF; Bayer–Friz–Gulisashvili–Horvath–Stemper 2019 QF; El Euch–Fukasawa–Gatheral–
  Rosenbaum 2019 SIFIN; El Euch–Gatheral–Rosenbaum 2019 Risk; Bayer et al. deep
  calibration); the dangling fig:btcfit now referenced. The reference working file
  (`OVERLEAF/_refs/new_references_verified.tex`; every entry marked VERIFIED
  against the publisher/arXiv record or LEAD-to-pin) committed in PR #89.
  Discipline: every PR independently compiled by chat-Claude before merge (P2 9pp,
  P3 10pp, P1 11pp; zero undefined references). Outstanding: advisor names in the
  four placeholders; bfn2022/ak2024 metadata pins; the ftw merge.
- **D51** *(2026-08-20)* **P4 premise corrected and reframed; lift negative result
  mechanised and situated (PRs #89, #91, #93).** The draft's founding claim — that
  the hybrid scheme's weak order "has never been measured" — was found false
  against the established weak-rate literature (Gassiat, SIFIN 14(2) 2023:
  (3H+½)∧1 left-point, H+½ weighted hybrid; Friz–Salkeld–Wagenhofer, AAP 35(1)
  2025: matching lower bound, phase transition at H=1/6; Bayer–Hall–Tempone, IJTAF
  2023; Bayer–Fukasawa–Nakahara, SIFIN 2022; Bonesini–Jacquier–Pannier 2023).
  Phase 1 (PR #89) inserted the eight-entry literature and boxed the claims.
  Phase 2a (PR #91) reframed the paper as an empirical weak-rate measurement of the
  direct rough-Heston discretisation tested against that Gaussian-volatility
  theory — none of the theorems covers square-root Volterra dynamics — with the
  measurements reproducing the proved phase structure (α ≈ 1 at H = 0.20 > 1/6;
  α ≈ 0.74, below 1 at 8 SE, at H = 0.05 < 1/6) and the sub-transition exponent
  sitting above 3H+½ = 0.65, stated as an observation, not a refutation; a new
  method paragraph records what conditioning rates (bias-preserving by the code's
  own construction — M, I left-point — a smooth OTM functional of the discretised
  integrals). Phase 2b (PR #93) rewrote the lift section on the mechanism read off
  `rough_heston_lifted.py`: the shared Brownian is the Abi Jaber (2019)
  lifted-Heston *model* (aj2019 added, QF 19(12), 1995–2013); the approximation is
  rank-one noise — each per-factor OU integral replaced by its L²-projection
  env_i·ΔW onto the shared increment — preserving each factor's covariance with ΔW
  (why β = 2H and prices validated) while under-representing stiff-factor variance
  by x_i·Δt/2, independent of N; the flat N-sweep recast as confirming the
  Bayer–Breneis L¹ kernel bound (bb2023), localising the fault in the scheme; the
  BB second-order weak scheme (bb2024) engaged as the non-rank-one counterexample
  that acquits the lift but not the surrogate purpose; the once-dangling bbf2023
  now cited at the Gauss–Jacobi quadrature; "second contribution" priority framing
  removed. All Phase-2 swaps SHA-256-gated and independently compiled (12pp, 0
  undefined references, 0 boxes). Open on P4: title/subtitle still predates the
  reframe; advisor placeholder; venue call.
- **D52** *(2026-09-02)* **Dissemination and the P4 venue sequence (PRs
  #97–#130).** Project-advisor names set (P1, P2) and placeholders stripped
  from the arXiv-track papers (PR #97). P3 and P4 posted to SSRN 24 Aug,
  approved and distributed 26 Aug (7346318, 7346418; a ligature-mangled title
  corrected before posting); README carries both links (PR #110); P1's
  self-reference now cites P3's SSRN record (PR #108). P4 went to SIFIN Short
  Communications 30 Aug (siamonline copy; Declarations section brought to
  SIAM's AI-policy v2.0 wording, PRs #112/#114) and was desk-declined on scope
  31 Aug — a placement outcome, not a finding, since the paper is empirical
  and SIFIN's scope asks for theorems or new algorithms — and resubmitted the
  same day to IJTAF on the precedent of Bayer–Hall–Tempone (IJTAF 2023):
  ws-ijtaf copy, abstract cut to 195 words, natbib author-year citations,
  bibliography closed at 19 cited / 19 defined (PRs #118/#120/#122); at
  referees from ~2 Sept. Principle adopted: submission copies live per venue
  under OVERLEAF/P4/<venue>/ and are frozen as the record of what each venue
  held; the master stays canonical. Alongside: guide gained the
  Information/Reliability/Efficiency section (PR #106), a public seed for the
  P-vs-Q roughness question (PR #116), the custom domain roughvollab.com with
  root-relative links (PR #124), and a revised author note, strapline and
  headline (PRs #128/#130). A Mathematics Today feature derived from P3 was
  submitted 28 Aug (off-repo).
- **D53** *(2026-09-26)* **P1's venue sequence: IMAMAT declined, AMF
  submitted (PRs #147, #149, #151).** P1 went to the IMA Journal of Applied
  Mathematics on a faculty head's suggestion (IMAMAT-2026-211, submitted
  10 Sept) and was desk-declined on scope on 23 Sept — "more suitable for a
  financial mathematics journal", no referee reports. Second scope decline of
  the programme, after SIFIN's on P4. Venue rule adopted in response: choose
  the journal the paper's own bibliography cites most, since a stated scope
  is a superset of what a journal actually publishes. P1 cites Quantitative
  Finance four times, so QF was the pick — but its portal charges a $162
  submission fee, and Applied Mathematical Finance is free, also Taylor &
  Francis, accepts any standard layout for initial review, and publishes the
  implied-volatility-surface and skew work P1 sits beside. **P1 submitted to
  AMF on 26 Sept 2026, submission ID 266158357**; QF is held as the reserve
  venue. A venue-neutral copy — stock article class, nothing vendored, no
  \date so it rebuilds identically — is frozen at OVERLEAF/P1/amf/.
  Referees suggested: Gassiat, Rosenbaum, Jacquier; Reisinger excluded as an
  AMF Editor-in-Chief.
- **D54** *(2026-10-02)* **P5 protocol frozen as a pre-registration.** The
  protocol (`docs/protocols/P5_protocol.md`) is committed before any P5 code,
  network or surrogate exists; this commit's timestamp is the record. Its
  question narrowed under a literature search: Brigo, Huser and Leonte (2026)
  and Bayer, Horvath, Muguruza, Stemper and Tomas (2019/2025) are the state of
  the art in prior-conditional deep inference, but neither varies its prior,
  and none of the four papers reviewed computes a Jacobian, Fisher information
  or Cramér–Rao bound. P5 tests what in-prior checks cannot: whether a deep
  ensemble and a neural-ratio posterior return their training prior along
  flat directions. Bayer et al. report H well identified for rough Bergomi on
  SPX, in tension with P1; the protocol treats identifiability as
  setting-specific and measures it by flatness quintile. Four predictions with
  refutation thresholds, a sealed test-set seed (20261002) and three pilot
  gates are fixed; any later change is a dated amendment. Drafted with AI
  assistance; every decision confirmed by the author.
- **D55** *(2026-10-03)* **P5 protocol under amendment — v2 draft recorded; v1
  remains the registered protocol.** The v1 protocol (D54) is under amendment;
  the draft is `docs/protocols/P5_protocol_v2_draft.md`. v1 remains the
  registered protocol until v2 is frozen and the OSF registration is amended;
  the OSF amendment is pending and will be made by the author at freeze.
  - **No code yet.** At the time of the amendment no P5 code, pilot, network or
    surrogate exists in the repository (file and import search at `e4a1a80`).
  - **Reason.** v1's four predictions follow from Bayes' rule and the parameter
    box, so an exact posterior would confirm them as readily as a network. v2
    asks instead whether a learned calibrator is more prior-driven than the
    exact posterior under its own training prior.
  - **Three repo-grounded changes.** (i) "P1's grid" was ambiguous between the
    D37 smile, the D38 synthetic surface and the D39 Deribit grid, and v2 names
    D38's. (ii) v1's single noise level of 0.1 vol point is the level at which
    D38 found the surface pins H to about 10%, so v2 makes noise a design
    factor. (iii) The pricer cannot afford the design directly, so v2 defines
    the model as a frozen emulator used by both the training data and the
    exact reference.
  - **What was seen.** On 3 Oct 2026 a timing and finiteness probe of the
    existing pricer was run by Claude in a claude.ai chat sandbox, on a shallow
    clone at commit `e4a1a8027ce7576040e92c4476f1bd8e25d1561f`, not on the
    author's machine; the script is not in the repo. Setup: 35-quote surface
    (`TS` × `VUS`), strikes standardised on each point's own ATM vol,
    per-maturity CF memoisation, `N_riccati` 1000, `n_nodes` 128, κ 0.30, one
    core. Results: at θ = (0.10, 0.35, −0.70, 0.04), 1.67 s and 0 of 35
    non-finite; over 24 points uniform on the box (H capped at 0.48, numpy
    `default_rng(7)`), mean 1.58 s and 9 of 24 with at least one non-finite
    maturity. Follow-up on four failing points, ATM quote only: finite at
    T ≤ 0.25 with N 1000 in all four; T = 0.5 needed N 2000 in two; T = 1.0
    needed N 4000 in two; T = 2.0 was finite at N 2000 in two and still
    non-finite at N 4000 in the other two. No Jacobian, Fisher information,
    Cramér–Rao bound, posterior or calibration was computed.
  - **Nothing decided.** The draft has 16 items marked OPEN and an unticked
    checklist; none is decided by this entry.
  - Drafted with AI assistance; decisions remain the author's.
- **D56** *(2026-10-03)* **P1: power-law skew claim softened; two references
  added (text and source comments only; no results change).** The
  introduction said Fukasawa's model is consistent with "the power law the
  SPX skew obeys in time-to-maturity", which asserted a disputed market fact
  in the author's voice. It now reads "a power-law decay of the SPX skew in
  time-to-maturity". Two sentences follow the skew-asymptotics citations:
  the first records the dispute, citing Guyon and El Amrani (Risk.net, 2023)
  and Abi Jaber and Li (Mathematical Finance, 2025); the second states that
  the paper works within the rough model, asking what a surface can identify
  and reporting where the model fails to fit live surfaces, and does not
  compare it with non-rough alternatives. Changed in the master
  (`OVERLEAF/P1/calibration_paper_revised.tex`) and in the AMF copy
  (`OVERLEAF/P1/amf/p1_qf.tex`); no result, number, figure or other claim
  changes.
  - **Not in the version under review.** The manuscript under review at
    Applied Mathematical Finance (submitted 26 Sept 2026, submission ID
    266158357; D53) does not contain the change. It is queued for the
    revision. Nothing was sent to the journal and no submitted PDF was
    regenerated or replaced.
  - **Frozen AMF copy.** The copy D53 records as frozen for AMF:
    `OVERLEAF/P1/amf/p1_qf.tex` at commit
    `0130d36b1c175c5a5eea4e2acc86db9481a00d1a`, blob `6055761`; the file is
    unchanged from its addition in `5ee9795` through its move to `amf/` in
    `185afde`. This blob first entered git on 28 Sept 2026, after the address
    and \date{} edits. Whether it is identical to the source submitted on
    26 Sept is not established from the repository; the PDF held by the
    journal is the authority. From this entry on, `amf/` is the working copy
    for the revision; this supersedes the freeze recorded in D53. The header
    comment of `p1_qf.tex` now records that commit and blob and the
    working-copy status.
  - **SSRN is not updated.**
  - **Other copy.** `OVERLEAF/P1/imamat/p1_imamat.tex` keeps the original
    sentence; it is the frozen record of a declined submission.
  - **Also queued for the revision.** The abstract and the introduction's
    first sentence state that rough models reproduce the short-maturity
    skew; that statement is unchanged here and is queued for the revision.
  - **References.** Guyon and El Amrani verified at the Risk.net article
    page; Abi Jaber and Li verified against the Crossref record deposited by
    Wiley (the Wiley page could not be read). Both are recorded in
    `OVERLEAF/_refs/new_references_verified.tex`. The SSRN preprint of the
    Risk article is not cited: its details could not be confirmed at source.
  - Drafted with AI assistance; decisions remain the author's.
- **D57** *(2026-10-04)* **P1 compiles in CI: the master and the AMF copy are
  built on every push to main and every pull request (CI only; no paper or
  code changes).** A second workflow, `.github/workflows/papers.yml`, compiles
  `OVERLEAF/P1/calibration_paper_revised.tex` and `OVERLEAF/P1/amf/p1_qf.tex`
  with pdfLaTeX and attaches the PDFs and logs to the run. The job fails on a
  LaTeX error or an undefined citation or reference. Until now the papers
  were compiled by hand, outside the repository.
  - **Toolchain.** TinyTeX-1 v2026.10 (TeX Live 2026), checked against a
    SHA-256 held in the workflow, plus `mathtools`, `cmap`, `microtype` and
    `cm-super` from the TeX Live snapshot of 30 Sept 2026. The runner is
    pinned to `ubuntu-24.04` and the three GitHub actions to commit SHAs.
  - **Why not the runner's own packages.** Ubuntu 24.04 packages TeX Live
    2023; Overleaf's logs of 3 Oct 2026 report TeX Live 2026 for both files.
  - **Scope.** The two P1 files only. The other papers and the frozen venue
    copies are not compiled.
  - **Local build.** TinyTeX-1 v2026.10 with the same four packages was
    installed on the author's Windows machine on 3 Oct 2026, outside the
    repository; both files compile there to 11 pages with no undefined
    citations.
  - Drafted with AI assistance; decisions remain the author's.
- **D58** *(2026-10-05)* **P5 v2 draft: observable decided; box and grid to
  be fixed as values from G0's pilot, to be run before freezing (one of the
  16 OPEN items decided; nothing frozen).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md`. The decision sheet
  (`docs/protocols/P5_v2_decision_sheet.md`) was the working aid; it is not
  updated, describes the draft as at commit `28e441b` (4 Oct 2026) and still
  places G0 after freezing. For this entry no measurement was made, nothing
  was run and no code was written. v1 remains the registered protocol.
  - **Observable: self-standardised quotes.** For maturity T and standardised
    moneyness point z, the quote is the model implied vol at strike
    S0·exp(z·σ_ATM(θ, T)·√T), where σ_ATM(θ, T) is θ's own noiseless implied
    vol at strike S0. The reference and the networks take as input the noisy
    quotes in a fixed order and nothing else that depends on θ; the strikes
    and the noiseless σ_ATM are withheld. That closes the leak in D38's
    convention, which anchors strikes once from the target's noiseless ATM
    vol. The fixed absolute grid is not chosen. No claim is made about quotes
    at fixed strikes, and D38's figures do not carry over to this observable
    as measured.
  - **Box and grid: not decided; the route is.** Both stay OPEN. The record
    cannot settle them: no coordinates are recorded for the failing points of
    the 3 October probe (D55), no result above `N_riccati` 4000 is recorded
    for T = 2.00, and what T = 2.00 adds about H has not been measured in
    this repository. G0's pilot is therefore run before freezing, on the
    author's machine, from a script committed with its output, first on the
    box and the grid as proposed (all five maturities). What it showed is
    declared under "What was known before freezing", and the box, the grid
    and the resolution (the `N_riccati` at which the emulator is built) are
    then written in as values on which G0's check has passed. After freezing
    no gate changes them. The earlier wording, which implied that G0 followed
    freezing and under which G0 could shrink the box or the grid, is
    withdrawn.
  - **What the pilot computes.** Beyond the existing tests: timing,
    finiteness and the change in each quote when `N_riccati` doubles. No
    Jacobian, Fisher information, posterior, calibration or emulator. It does
    not measure what T = 2.00 adds about H, so it bears on whether T = 2.00
    stays only through finiteness, the change when `N_riccati` doubles and
    cost. It uses no point of the sealed test set (seed 20261002).
  - **Owed before the pilot is run.** G0's tolerance and its unit, the pilot
    points, the starting `N_riccati`, the Fourier inversion settings, any
    limit on cost and any criterion for choosing the box and the grid from
    the pilot's output are marked OPEN and are to be written into the draft
    before the run. Whether that choice is made by such a criterion or by a
    judgement declared as made on seen output is not decided here. Every run,
    passing or failing, is to be declared with its settings.
  - **What this changes about v2.** v1 was frozen before any P5 code existed.
    v2 will not be: the pilot's script and output will exist at freezing and
    will be declared; no emulator, reference posterior or trained network is
    built before freezing. G0's run over the box is not among the pilots v1
    declares (its gates G1 to G3); v1's G2 asks only that the existing
    verification tests pass, as G0 also does. v1 asks that a departure be a
    dated amendment stating what changed, why and whether any data had been
    seen: the change is that a pilot of the pricer precedes the freezing of
    v2; the reason is that the record cannot settle the box and the grid;
    what had been seen is declared in the draft's section "What was known
    before freezing", and the repository holds no script or output of the
    pilot. A further entry is owed when the pilot has been run.
  - **Also in the draft.** The prior-knowledge section now declares the
    fitted ν of 0.71 and the overflow recorded on live BTC, and that D38's
    figures were measured with strikes anchored once on the target. The unit
    of the quotes is written in (implied vols are decimals and one vol point
    is 0.01, as in `layer4_calibrate_surface.py`). Scope lists quotes at
    fixed strikes as out. The priors of H and the 0.20 between their means
    are noted as written for the proposed range of H.
  - **Checklist.** No box is ticked; three are reworded. 15 items remain
    OPEN.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D59** *(2026-10-05)* **P5 v2 draft: G0's tolerance, comparisons, cost
  limit and criterion written in; the pilot staged as a scoping run and a
  main run (no item closes; nothing frozen).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md` (lines 15, 139 and 176). For this
  entry nothing was run, no measurement was made and no code was written; the
  repository holds no script or output of the pilot. What is said below about
  the pricer is reasoning from the code or arithmetic on recorded numbers. v1
  remains the registered protocol. The decision sheet is not updated.
  - **Why the check gains comparisons.** G0 as drafted compared each quote at
    `N_riccati` and at twice that. Both pricings use the same Gauss–Legendre
    nodes and the same `U_max` (`gil_pelaez_call` in `rough_heston_cf.py`),
    so an error of the inversion is, to leading order, the same in both and
    does not show in the difference. Arithmetic for a Gaussian characteristic
    function with σ√T = 0.01 (ξ₀ = 0.001, T = 0.10): cutting the integral at
    `U_max` 200 scales the at-the-money time value by 1 − 2Φ(−2), which
    lowers the implied vol by 4.6%, about 0.14 vol point. That is arithmetic
    on a Gaussian case, not a measurement of the rough pricer. The one
    existing test that varies `U_max` is on the classical Heston
    characteristic function at T = 1 and asserts only its finest setting
    (`test_rough_heston_cf.py`).
  - **Why two runs.** A reading of the Riccati recursion: for the highest
    node, whether the solve stays finite depends on ν · `U_max` · h^(H+½),
    with h the step T / `N_riccati`, and on H, ρ and the number of steps. The
    recorded outcomes at D38's point (D55) and at the railed H on live BTC
    (D39, D41) are consistent with it; they lie at H of 0.10 and 0.02 and,
    where ρ is recorded, at ρ between −0.70 and −0.31. Raising `U_max` to
    reduce truncation raises that quantity. The prediction, untested, is that
    the box as proposed does not pass at any affordable setting. Where it
    stops depends on numbers not yet measured: that threshold across H and ρ,
    where truncation bites, and what the solves cost at those settings on the
    author's machine. A scoping run is to measure them first.
  - **Fixed now.** Existing verification tests: the repository's test suite,
    run on the author's machine. Tolerance: 0.01 vol point (0.0001),
    absolute, on every quote and in each comparison separately; a tenth of
    the smallest proposed noise level and half of G1's proposed allowance at
    that level. Comparisons: `N_riccati` doubled, the node count doubled,
    `U_max` raised; by how much, with which node count and at which
    `N_riccati`, is among the main run's settings. They are checks of
    convergence, not a bound on the pricer's error. Cost limit: 1,200
    core-seconds for the Riccati solves of one (H, ν, ρ) over the grid's
    maturities, at the node count from which the comparisons start, as the
    median over the points priced, on the author's machine with the pilot's
    worker pool running; two weeks of four cores for 4,000 such sets of
    solves. Criterion: the box, the grid and the resolution are taken from
    the main run's output by a written order, given in the draft. The range
    of H is not narrowed; candidates that contain D38's point come first,
    then five maturities before four, then the largest share of the proposed
    ranges. The candidate taken is confirmed on fresh points; those points
    count as pilot points from then on, and if it then no longer passes the
    order is applied again. If no candidate passes, the draft is not frozen
    as it stands and anything adopted afterwards is declared as a judgement
    made on seen output. D58 left open whether the choice would be by
    criterion or by judgement; it is by criterion. Seeds: pilot seeds are
    stated with each run; none is the sealed seed 20261002 or the probe's 7.
  - **What the pilot computes.** D58's list is widened: the comparisons in
    node count and `U_max`, the inversion applied to characteristic functions
    for which a reference price is available (Black–Scholes and classical
    Heston), the stage at which a non-finite quote failed, and the modulus of
    the characteristic function at the highest node, which decides nothing.
    Beyond the existing tests, as in D58: D58's exclusions stand (no
    Jacobian, Fisher information, posterior, calibration or emulator), and
    the draft now states the limit as nothing that compares quotes across
    parameter values.
  - **Scoping run.** To be made first, on the box and the grid as proposed,
    the 16 corners included. It has no pass or fail. It measures cost on the
    author's machine and, at each point and maturity it prices, whether and
    from which `N_riccati` among those it tries the quotes are finite, and
    which comparisons are met. Its points and the values of `N_riccati`, node
    count and `U_max` it tries are those of the script as committed before it
    is run.
  - **Still OPEN at G0.** For the main run: the resolutions tried, the
    Fourier inversion settings, the pilot points and the candidate bounds,
    the size of the worker pool, the number of confirming points and of
    confirmations attempted, and how a tie in share is broken. They are to be
    written into the draft before the main run, with the scoping run's output
    in view, and declared as such. After it: the resolution at which the
    emulator is built.
  - **Change from D58.** Of what D58 listed as owed before the pilot is run,
    the tolerance, the cost limit and the criterion are written in now, and
    the rest is to be written into the draft before the main run. For the
    scoping run the points and the numerical settings it tries are fixed by
    the script as committed before it is run, not by the draft.
  - **What the fixed values rest on.** The tolerance rests on the smallest
    noise level and on G1's ratio, and the cost limit on an emulator design
    of about 4,000 sets of solves. All three are still OPEN in the draft.
  - **Counts.** 15 items remain OPEN. No checklist box is ticked; one is
    reworded.
  - A further entry is owed after each run.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D60** *(2026-10-10)* **P5 G0 scoping run made and declared; its output
  committed (no item closes; nothing frozen).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md` (line 15 reworded; one bullet added
  under "What was known before freezing", line 21). The run was made by the
  author on his machine from `p5_g0_pilot.py` as committed in #184 (`1c6fc13`),
  with the working tree clean at `d394b08`: one launch, 2026-10-09 09:48:53 UTC
  to 23:42:47 UTC (ending at 00:42 on 10 Oct local time), 300 tasks, 0 errors,
  a pool of 4 workers, on the author's laptop (an Intel i7-1165G7 with 4 cores;
  the output records 8 logical processors and Windows 11), Python 3.14.3, numpy
  2.4.4, scipy 1.17.1. No random draws were made, so there is no seed. The run
  has no pass or fail and nothing below is a decision: the box, the grid, the
  resolution and every main-run setting remain OPEN. v1 remains the registered
  protocol. The decision sheet is not updated.
  - **Settings as committed.** Base inversion setting `U_max` 200 with 128
    nodes, the repository's own (`U_max` 200 is `gil_pelaez_call`'s default;
    128 nodes is `NN` in `layer4_calibrate_surface.py`, as `model_smile_cf_T`
    passes it; `gil_pelaez_call`'s own default is 256). Comparison settings:
    (200, 256) for the node count doubled, against the base; (400, 256) for
    `U_max` raised, against (200, 256); `N_riccati` doubled, the base at the
    next rung. Ladder `N_riccati` 1000, 2000, 4000, 8000, 16000, climbed per
    (H, ν, ρ, T) until a rung met all three comparisons on its 35 quotes; the
    comparison solves skipped at a rung whose base solve was not finite; the
    top rung without a partner. Tolerance 0.0001 per quote, per comparison.
    Points: H {0.02, 0.05, 0.10, 0.25, 0.48} × ν {0.05, 0.35, 0.65, 1.00} × ρ
    {0.00, −0.70, −0.99}, 60 triples, each at ξ₀ {0.001, 0.01, 0.04, 0.10,
    0.25}, which share each solve; the 16 corners of the proposed box and D38's
    point are among them; κ 0.30. Grid: D38's five maturities and seven z,
    strikes standardised on each point's own ATM quote at the first rung where
    it was finite under the base setting, then fixed for that task. Before the
    run the script reproduced `model_smile_cf_T` on D38's 35 quotes with a
    maximum difference of exactly 0.0 (the script stops otherwise; the value
    itself is not in the output).
  - **Cost.** With the pool running, the three base solves of one task took a
    median 0.8, 2.4, 9.0, 38.1 and 156 s at `N_riccati` 1000, 2000, 4000, 8000
    and 16000, over the finite solves; 2.2 to 2.6 times the same solves made
    alone (0.31, 1.05 and 4.05 s at 1000, 2000 and 4000, from the launch line).
    The cost quantity of L139, the base solves of one (H, ν, ρ) over the five
    maturities, had median 4.0, 12.3, 44.4, 189.5 and 774.9 s at those
    `N_riccati`, over the triples whose five maturities all had a finite base
    solve at that rung: 32, 37, 37, 45 and 49 of the 60 (at 16000 the eleven
    others are the six ν 0.05 triples with a task that stopped at 1000 and five
    with a non-finite solve at one maturity); the largest at 16000 was 856 s.
    The cost at a raised `U_max` was measured only at 400 and only where that
    solve was finite. The whole run was 833.9 min of wall and 55.5 core-hours.
  - **Finiteness, Riccati stage (base setting).** At T = 0.10 the solve was
    finite from `N_riccati` 1000 everywhere except (H 0.02, ν 1.00, ρ 0) and
    (0.05, 1.00, 0), which needed 2000. At T = 1.00 and 2.00 the steps needed
    rose with ν and fell with H and with |ρ|: at T = 2.00, ν 0.35 needed 1000
    to 8000, ν 0.65 up to 16000, and at ν 1.00 nothing below H 0.10 was finite
    through 16000 at ρ 0. Six tasks were non-finite at 16000: (0.02, 0.65, 0),
    (0.02, 1.00, each of the three ρ) and (0.05, 1.00, 0) at T = 2.00, and
    (0.02, 1.00, 0) at T = 1.00. 79 of the 300 tasks had a non-finite base
    solve at some rung. At (400, 256), 26 of the 283 tasks with a finite base
    solve at 16000 were non-finite there (11 at H 0.02, 10 at 0.05, 5 at 0.10;
    12 at T = 2.00, 8 at 1.00, 5 at 0.50, 1 at 0.25), and 177 were finite from
    1000; at the six tasks above no comparison solve was made. At (200, 256) no
    task was non-finite where the base solve was finite.
  - **Finiteness, inversion stage (base setting, each task's top rung).** At ξ₀
    0.001, 290 of 2,100 quotes were not finite: 242 with a price at or below
    1e-12 and 48 with no implied vol, at every maturity, the former all at z =
    +1 and +2. At ξ₀ 0.01, 0.04, 0.10 and 0.25, 73, 58, 55 and 37 quotes had a
    price at or below 1e-12, all but three at ρ −0.99 and mostly at z = +2; 42
    per ξ₀ more were non-finite at the Riccati stage. The modulus of the
    characteristic function at the top node, just below u = 200 (median over
    tasks), was 10^−0.1 to 10^−0.3 at ξ₀ 0.001, 10^−0.6 to 10^−2.5 at 0.01,
    10^−2.6 to 10^−10 at 0.04, below 10^−6 at 0.10 and below 10^−16 at 0.25,
    falling with T; at the top node of the (400, 256) setting and ξ₀ 0.001 it
    was still 10^−0.1 to 10^−0.7.
  - **Comparisons.** `N_riccati` doubled, 8000 against 16000: met on every
    finite pair, largest difference 4.3e-5; 1000 against 2000: met on all but
    14 of 7,367 finite pairs, largest 6.9e-4. Node count doubled, at each
    task's top rung: met on every finite pair but 5 of 9,777 (all at ρ −0.99, T
    = 2.00, z = +2, ξ₀ 0.25; up to 0.0136); over all rungs, 17 of 42,175.
    `U_max` raised, at the top rung: met on 9%, 48%, 78%, 92% and 98% of finite
    pairs at ξ₀ 0.001, 0.01, 0.04, 0.10 and 0.25; at ξ₀ 0.04 from 62% at T =
    0.10 to 93% at T = 2.00; at ξ₀ ≥ 0.04 on 100% at ν 0.05 and at ν 1.00 from
    58% at T = 0.10 to 87% at T = 2.00; largest difference 0.062. At D38's
    point: met at every maturity for ξ₀ ≥ 0.10; at ξ₀ 0.04 one miss, 0.00014 at
    T = 0.10; at ξ₀ 0.01 up to 0.0148; at ξ₀ 0.001 up to 0.0107. Tasks on which
    all three comparisons were met on all 35 quotes at some rung: 11 of 300,
    all at ν 0.05 and T ≥ 1.00 (ρ 0 at each H at T = 1.00 and 2.00, and (0.25,
    0.05, −0.70) at T = 2.00); the other 289 climbed to 16000. Counting only
    the quotes at ξ₀ ≥ 0.01, ≥ 0.04 or ≥ 0.10, the node and `U_max` comparisons
    at the top rung were met on 105, 175 and 199 tasks.
  - **Known answers (no Riccati solve).** Black–Scholes through the base
    setting: largest error 0.0058 in vol over 175 rows, 0.0014 at the money at
    T = 0.10 and ξ₀ 0.001 (the D59 arithmetic, now measured); 2.2e-6 at (400,
    256); the adaptive quadrature used as the Heston reference, 2.2e-11.
    Classical Heston through the base setting: up to 0.098 in vol, and 0.070 at
    (400, 256), with 29 to 30 of 350 rows lacking an implied vol on one side.
  - **What the output bears on.** A reading of the output, labelled as such; it
    decides nothing. At the top rung the `U_max` comparison was met on 9% of
    finite pairs at ξ₀ 0.001 and 48% at 0.01; at T = 0.10 and ξ₀ ≥ 0.04, on 75%
    at ν 0.65 and 58% at ν 1.00. The logarithm of the modulus at the top node
    falls in proportion to ξ₀ and more slowly with T. Only `U_max` 200 and 400
    were tried; what a quote needs was not measured. At (400, 256) 26 tasks
    were non-finite at 16000 where the base solve was finite, and at (200, 256)
    none: the raised `U_max`, not the doubled node count, cost finiteness, at
    low H, high ν and long maturity. The node-count comparison at `U_max` 200
    was met on all finite pairs but 5 at the top rung and 17 of 42,175 over all
    rungs. The median of L139's cost quantity at 16000, 774.9 s over the 49
    triples with a finite base solve there at all five maturities, is below
    1,200 core-seconds; at a raised `U_max` the cost was not measured beyond
    400. Whether any resolution is admissible on any grid is for the main run.
  - **Not decided.** The box, the grid and the resolution, and every main-run
    setting listed OPEN at L139. The 11 tasks on which every comparison was met
    are not a box: they are one value of ν at two maturities. The range of H is
    not narrowed by anything here.
  - **Files.** `output/p5_g0_pilot_tasks.jsonl` (12,803,200 bytes; the working
    record: a header, one launch line, one line per task),
    `output/p5_g0_pilot.json` (3,294,100 bytes; the summary) and
    `output/p5_g0_pilot_quotes.csv` (19,775,497 bytes; one row per triple, ξ₀,
    maturity, z and rung, 51,345 rows), committed unmodified; the folder is
    git-ignored, so they were added by name. `py -3 p5_g0_pilot.py --report`
    regenerates the second and the third from the first. The numbers above were
    read from the JSON and the CSV by a script in a scratch folder, not in the
    repository.
  - **Draft.** Line 15: "Nothing from G0's pilot (see Gates) is declared here
    yet; every run of it is to be declared in this section before freezing."
    becomes "G0's scoping run is declared below (D60); the main run is to be
    declared here before freezing." One bullet is added after the probe bullet
    (new line 21), so the draft goes from 178 to 179 lines and the decision
    sheet's references to lines from 21 on are off by one; the sheet is not
    updated (D59). Lines 37, 39, 139 and 176 of the committed draft (now 38,
    40, 140 and 177) are unchanged.
  - **Counts.** 15 items remain OPEN. No checklist box is ticked.
  - The main run's settings sitting is owed next, with this output in view;
    what it sets is declared as such.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D61** *(2026-10-10)* **P5 v2 draft: the main run's settings written in with
  the scoping output in view; `p5_g0_main.py` committed (no item closes;
  nothing frozen).** Recorded in `docs/protocols/P5_protocol_v2_draft.md` (line
  140, the OPEN span for the main run, now values). The settings were chosen
  with the scoping run's output (D60) in view, as line 140 allowed, and are
  declared as such. No run was made for this entry; the main run follows. The
  box, the grid and the resolution remain OPEN, as does the resolution at which
  the emulator is built. v1 remains the registered protocol. The decision sheet
  is not updated.
  - **Settings.** Resolutions: one `N_riccati` per maturity; ladder 2000, 4000,
    8000, climbed per point and maturity; a base solve at 16000 only for the
    `N_riccati` comparison at 8000, and only where the base solve at 8000 is
    finite. A point whose task stopped at a coarser rung by the stop rule
    counts as passing at every finer `N_riccati` (the stop rule's claim).
    Inversion: `U_max` = 250/√T rounded (790, 500, 350, 250, 180) with 504,
    320, 224, 160 and 112 nodes, the scoping spacing of 1.5625 per node to the
    nearest multiple of 8; the node count doubled at the same `U_max`; `U_max`
    doubled with the node count doubled, compared with the doubled-node
    setting; the `U_max` comparison at every rung at its own `N_riccati`. Stop
    rule: a rung is final when every quote at every ξ₀ meets all three
    comparisons or fails only the `U_max` comparison with the raised-`U_max`
    quote unchanged from the rung before, more steps then being unable to
    change the outcome; comparison solves are skipped where the base solve is
    not finite. Pilot points: H {0.02, 0.05, 0.10, 0.25, 0.48} × ν {0.05, 0.35,
    0.50, 0.65, 1.00} × ρ {0.00, −0.35, −0.70, −0.90, −0.99}, 125 triples, 625
    tasks, each at ξ₀ {0.02, 0.04, 0.07, 0.10, 0.16, 0.25}; the 16 corners of
    the widest candidate (ξ₀ 0.02 to 0.25) and D38's point are lattice points;
    order: the corners, D38's triple, then H ascending. Candidates: boxes with
    distinct lattice bounds in ν, ρ and ξ₀, H always full. Cost: D59's
    quantity, formed from the record as the median, over the triples priced at
    the resolution's `N_riccati` at every maturity of the grid, of the summed
    seconds of their base solves, with the pool running (a resolution with no
    such triple is not admissible). Pool: four; `--pool` defaults to it and the
    launch line records the value used. Confirming points: eight per attempt,
    uniform in the candidate with H over its full range, `default_rng(61)` then
    `default_rng(62)`, counted only once all are priced (an incomplete attempt
    is resumed, its draw being deterministic); confirmed when the candidate
    taken with them included equals, in bounds, grid and resolution, the one
    they were drawn for; two attempts; if the first fails the points join the
    pilot points and the order is applied again; if the second fails, no
    candidate passes. Tie in share: the larger share of ν, then of ρ, then of
    ξ₀, then the smaller lower bound on ξ₀, on ν and on ρ in turn, shares
    compared to nine decimal places (equal lattice widths differ in floating
    point). Seeds 61 and 62 are neither the sealed seed nor the probe's.
  - **What they rest on.** The scoping output, read as follows; the numbers are
    from `output/p5_g0_pilot_quotes.csv`. (i) The log10 modulus of the
    characteristic function at the top node of (400, 256) is a median 2.01
    times that of (200, 128) (2.00 to 2.02 by (ξ₀, T); 1.3 to 3.3 over tasks):
    the modulus falls about exponentially in u, so the `U_max` a quote needs is
    inversely proportional to its decay rate; that rate is proportional to ξ₀
    and grows as about T^0.46 (median −2.6 at T = 0.10 to −10.2 at T = 2.00 at
    ξ₀ 0.04), so `U_max` ∝ 1/√T keeps the modulus about even across maturities.
    (ii) At each task's top rung the `U_max` comparison was met on 99.5% of
    finite pairs whose lower side had a log10 modulus below −3 (97.5% to 99.9%
    by sub-bin), on 88% between −3 and −2, 29% between −2 and −1 and 2% above
    −1. (iii) The slow decays are at ρ near −1 and ν near 1 at short maturity:
    at ξ₀ 0.04 and T = 0.10 the log10 modulus at the top node is −0.6 to −1.9
    at ρ −0.99, −2.1 to −5.3 at ρ −0.70 and −3.0 to −7.5 at ρ 0 (medians over
    ν, ranges over H), and −1.2 at ν 1.00 against −3.2 at ν 0.35 (medians over
    H and ρ); at ξ₀ 0.01 and 0.001 every value is a quarter and a fortieth of
    these, so at T = 0.10 they need a `U_max` of about 1,100 and 11,000 and are
    not priced. (iv) Doubling the node count cost a median 1.24 to 1.36 times
    by rung at 128 nodes (1.0 to 1.7 by task); at the main run's counts, timed
    alone at `N_riccati` 2000 and D38's triple on battery power, a base solve
    at T = 0.10 with 504 nodes cost 1.8 times the scoping base (200, 128) and
    doubling the nodes 1.4 to 1.7 times. Where the first finite rung rose when
    `U_max` doubled (91 of 300 tasks; 177 unchanged, 26 never finite at (400,
    256)), it rose 4 times at H ≤ 0.10 and 2 times at 0.25 as the median,
    against 2^(1/(H+½)) = 3.8 to 2.0 in D59. (v) The ladder stops at 8000 on
    cost: scaled from the scoping run's pool-running medians (775 s for one
    triple's base solves over five maturities at 16000, at 128 nodes) by the
    node-count ratios above (1.3 to 1.4 per doubling), 16000 is predicted to
    cost about 1,000 to 1,100 s with the pool, near the 1,200 core-second
    limit, and a rung at 16000 with its comparison solves about four times a
    task's cost for ten more of the 300 scoping pairs predicted to pass; 16000
    is therefore tried only as the partner, and whether 8000 is admissible is
    the run's own measurement.
  - **Prediction, not a result.** Applying the decay rates of the scoping run
    to these settings with the truncation target a log10 modulus at or below
    −3.5, and bounding the step count a solve needs from the first finite rung
    at (200, 128) and at (400, 256) scaled by (U_max ratio)^(1/(H+½)), 154 of
    the scoping run's 300 (triple, ξ₀) pairs are predicted to pass at the
    ladder's top of 8000 (164 at 16000), and the largest box containing D38's
    point predicted to pass on five maturities is ν [0.05, 0.35] × ρ [−0.70, 0]
    × ξ₀ [0.04, 0.25] in both cases, the ν ceiling coming from H 0.02 and 0.05
    at ν ≥ 0.65, where the step count needed at T ≥ 0.25 exceeds the ladder or
    no rung was finite at T = 2.00 in the scoping run (at ρ 0 for ν 0.65); ν
    0.50 is in the lattice so that ceiling can land between 0.35 and 0.65. The
    step-count bound is loose: the scoping ladder began at 1000, so a point
    finite there may need far fewer steps, and bounding from the base rung
    alone gives 70 pairs at 8000 and no box containing D38's point. At `U_max`
    = 320/√T the step count fails broadly; at 200/√T about the same box is
    predicted. The box, the grid and the resolution are taken from the main
    run's output by the order of line 140, not from this prediction.
  - **What the script does.** `p5_g0_main.py` reuses the scoping script's
    pricing path with the settings above; without a flag, and before `--run`,
    the identity gate with `model_smile_cf_T` at the scoping base runs and
    stops the script otherwise, then the design is printed, the three settings
    timed at each maturity alone, and a bound and a labelled guess given;
    `--run` prices the 625 tasks on the pool, resumable, writing
    `output/p5_g0_main_tasks.jsonl`, `output/p5_g0_main.json` and
    `output/p5_g0_main_quotes.csv`; `--criterion` applies the order of line 140
    to the record and draws the pending attempt's confirming points into
    `output/p5_g0_main_criterion.json`; `--confirm` prices them; `--criterion`
    is then applied again. The criterion is a reading of the record: it
    compares pass flags across points, not quotes. No flag changes a point or a
    numerical setting. `p5_g0_pilot.py` gains one keyword argument
    (`cost_summary(ladder=...)`) and the docstring that names it, nothing else;
    the scoping run's provenance is the commit hash in its output. Nine
    mechanics tests in `test_p5_g0_main.py`, including the criterion's order,
    tie-break and confirmation states on hand-built records;
    `CONTRIBUTING.md`'s count becomes 289.
  - **Cost.** Scaled from the scoping run's launch-line timing (1.05 s for the
    three base solves at (200, 128) and `N_riccati` 2000, alone, on mains) by
    the node-count ratios in (iv) and as N², a task that exhausts the ladder
    costs about 1,000 s alone at T = 0.10 and 300 to 400 s at the other
    maturities, the partner at 16000 included; the bound for 625 tasks is about
    60 core-hours alone and the guess (H ≥ 0.25 stopping by 4000, H 0.10 by
    8000, half of H ≤ 0.05 exhausting) about 40; with the pool's slowdown of
    2.2 to 2.6 times, about a day and a half of wall for the bound and a day
    for the guess on four workers. The script prints its own figures from
    solves timed at launch. The confirmation stage is 40 tasks per attempt.
  - **Not decided.** The box, the grid and the resolution; the resolution at
    which the emulator is built. The prediction above decides nothing. The
    range of H is not narrowed. ξ₀ 0.001 and 0.01 are not priced, which bounds
    the candidates' lower ξ₀ at 0.02 by construction; that choice is a setting
    of the main run, declared here as made on seen output.
  - **Draft.** Line 140: the bold OPEN span for the main run and the sentence
    after it are replaced by the settings as values and one OPEN item, the
    emulator's resolution. Line 177 stays unticked: its second and third
    clauses are met at this commit, but its first, the gate tolerances
    declared, is not, G1 to G3 still being OPEN. No line is added or removed;
    179 lines; "OPEN" occurs 16 times before and after.
  - **Counts.** 15 items remain OPEN. No checklist box is ticked.
  - The main run, its declaration and the confirmation are owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D62** *(2026-10-10)* **P5 G0 main run made, confirmed and declared; the
  box, the grid and the resolution written in as values (three OPEN items
  close; nothing frozen).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md` (line 15 reworded; one bullet added
  under "What was known before freezing", line 22; the Parameters line, the
  Grid line and line 141 now carry values; the checklist box for the grid, the
  box and the resolution ticked). The main run was made by the author on his
  machine from `p5_g0_main.py` as committed in #188 (`99aa4d5`), with the
  working tree clean at `47e3bb6`, at the settings written into line 140 by
  D61: one launch, 2026-10-10 07:42:41 UTC, 625 tasks, 0 errors, 641.3 min of
  wall, 42.7 core-hours on a pool of 4; the criterion applied; a second launch
  at 18:30:48 UTC for the 40 confirming tasks of attempt 1 (seed 61; 8.0 min);
  the criterion applied again at 18:48:29 UTC, returning confirmed. Nothing
  below is a judgement: the box, the grid and the resolution are what the order
  of line 140, written before the run, returned. v1 remains the registered
  protocol. The decision sheet is not updated.
  - **What the criterion returned.** Box: H [0.02, 0.48], not narrowed, by
    rule; ν [0.05, 0.35]; ρ [−0.90, −0.70]; ξ₀ [0.04, 0.25]. Grid: the five
    maturities; ten candidates passed on five, all with ν [0.05, 0.35] and ρ
    [−0.90, −0.70] and differing only in their ξ₀ range, so the four-maturity
    grid was not reached. Resolution: `N_riccati` 8000 at every maturity, with
    `U_max` 790, 500, 350, 250 and 180 and 504, 320, 224, 160 and 112 nodes.
    Cost: 288 core-seconds per (H, ν, ρ) over the grid, the median over the 54
    triples priced at 8000 at every maturity (the 53 pilot triples and one
    confirming point whose partner solves were at 8000), with the pool running,
    against the limit of 1,200; the summary's median over the 53 pilot triples
    alone is 290 s. Share of the proposed ranges: 0.0538 (ν 0.316 × ρ 0.202 ×
    ξ₀ 0.843); no tie. Confirmation: the eight points of attempt 1, uniform in
    the candidate with H over its full range, all met every comparison at every
    maturity, at 2000 except one point at 4000 at every maturity and another at
    4000 at T = 2.00; the candidate taken with them included was the same in
    bounds, grid and resolution.
  - **What the run showed.** 413 of the 750 (point, ξ₀) pairs met every
    comparison at every maturity at or below 8000. By ν: 100% at 0.05, 67% at
    0.35, 42% at 0.50, 38% at 0.65, 29% at 1.00. By ρ: 22% at −0.99, 62% at
    −0.90, 71% at −0.70, 64% at −0.35, 56% at 0. By H: 27%, 32%, 55%, 81% and
    80% from 0.02 to 0.48. By ξ₀: 46% at 0.02, 55% to 59% at 0.04 to 0.25. Of
    the 3,750 (task, ξ₀) cases at each task's top rung, by the precedence base
    solve, doubled-`U_max` solve, base quote, then the comparisons: 2,197
    passed; 792 failed because the doubled-`U_max` solve was not finite within
    the ladder; 330 because the base solve was not; 225 on the `U_max`
    comparison, 95 of them because the raised-`U_max` or doubled-node quote
    itself had no finite implied vol; 204 on a base quote with no finite
    implied vol at the inversion stage; 2 on the node comparison. At ν ≤ 0.35
    and ξ₀ ≥ 0.04: ρ −0.35 and 0 failed only through the doubled-`U_max` solve
    (10 to 15 of 100 cases per maturity, all at H 0.02 and 0.05); ρ −0.99
    through the inversion stage (7 to 13 of 50 per maturity) and the `U_max`
    comparison (8 to 12); ν 0.50 inside the ρ band only through the
    doubled-`U_max` solve (10 to 20 of 50 per maturity, at H ≤ 0.05); ξ₀ 0.02
    inside the box on the `U_max` comparison (2 to 4 of 20 per maturity).
    Inside the box the largest `U_max` difference was 6.4e-7; outside it 0.038.
    D38's point passed at 2000 at T = 0.10 and 0.25 and at 4000 at T ≥ 0.50;
    the resolution 8000 is forced at each maturity by 10 to 20 of the box's 100
    (point, ξ₀) pairs, all at ν 0.35 and H 0.02 or 0.05. Stop rule: 317 tasks
    final by all comparisons met, 39 by truncation only, 269 climbed to 8000
    without a final rung (93 of the 125 at H 0.02, none at H 0.48 with ν 0.05).
    Cost, median seconds of the finite base solve with the pool running at
    2000, 4000, 8000 and the partner 16000 (the criterion file's cost table): T
    = 0.10: 5.7, 21.9, 86.1, 338; 0.25: 4.2, 16.7, 67.6, 267; 0.50: 3.3, 13.1,
    53.2, 212; 1.00: 2.8, 11.2, 46.0, 186; 2.00: 2.4, 8.9, 36.8, 154; against
    the launch line's solo timing at 2000 (2.78 s at T = 0.10, 1.05 at 2.00)
    the pool's slowdown was about 2 times. The sum of the per-maturity medians
    at 16000 is 1,156 s, which bears out D61's "near the limit".
  - **Prediction against outcome.** D61 predicted, over the scoping run's 300
    (triple, ξ₀) pairs, 154 passing at 8000 from its step-count bound and 70
    from the base rung alone, and ν [0.05, 0.35] × ρ [−0.70, 0] × ξ₀ [0.04,
    0.25] as the largest box containing D38's point. The main run's 750 pairs
    are a different lattice; on the 180 pairs common to both, 99 passed. The
    box taken differs in ρ: it is the band −0.90 to −0.70 (ρ −0.90 was not in
    the scoping lattice, so nothing was predicted about it), and ρ −0.35 and 0,
    predicted to pass, did not, through the doubled-`U_max` solve, which the
    prediction's step-count bound, scaled from the first finite rung at (400,
    256), did not catch at H 0.02 and 0.05. The ν ceiling at 0.35 lies in the
    range D61 left open (0.35 to 0.65), ν 0.50 falling through the same solve.
    The ξ₀ floor is as predicted.
  - **What the box's edges rest on.** A reading, labelled as such. The ρ and ν
    edges are set by the check's own requirement that the `U_max` comparison be
    made, which needs the doubled-`U_max` solve to stay finite within the
    ladder: at ρ 0 and ν ≤ 0.35 the base quotes at 8000 meet the step and node
    comparisons, and the solve at twice `U_max` does not stay finite at H 0.02
    and 0.05. The ξ₀ floor and the ρ end near −1 are set by truncation and by
    quotes with no finite implied vol at the inversion stage. Recovering ρ near
    0 or ν above 0.35 would mean changing the check after seeing output, which
    line 140 allows only as a declared judgement and which is not done.
  - **Not decided.** Every other OPEN item. The hypotheses, the priors of H
    (its range is unchanged), the noise levels, the emulator's form and design,
    and the bin shares are untouched; what D59 anticipated for a narrower box
    (the prior-knowledge wording, the sealed draw, which stays uniform on the
    box, the bin shares and the hypothesis numbers) is to be re-read in the
    next sitting.
  - **Draft.** Line 15: "G0's scoping run is declared below (D60); the main run
    is to be declared here before freezing." becomes "G0's scoping run (D60)
    and its main run and confirmation (D62) are declared below." One bullet
    added after the scoping bullet (new line 22), so the draft goes from 179 to
    180 lines and the decision sheet's references to lines from 22 on are off
    by two; the sheet is not updated (D59). Parameters line: the ranges become
    the box's values with the proposal kept on record, and the OPEN span
    becomes a note on what set the edges. Grid line: the OPEN span becomes the
    five maturities as the grid, the probe's and D41's record kept. Line 141
    (140 before the bullet): the OPEN item for the emulator's resolution
    becomes 8000 at every maturity with the inversion settings above.
    Checklist: the box for the grid, the box and the resolution is ticked; the
    box for the prior-knowledge section stays unticked, its check against the
    ROADMAP entries not having been made.
  - **Files.** `output/p5_g0_main_tasks.jsonl` (14,830,944 bytes: a header, two
    launch lines, 625 task and 40 confirm records), `output/p5_g0_main.json`
    (3,413,470), `output/p5_g0_main_quotes.csv` (25,661,232; 80,108 rows) and
    `output/p5_g0_main_criterion.json` (13,659), committed unmodified, added by
    name as the folder is git-ignored. The numbers above were read from them by
    a script in a scratch folder, not in the repository.
  - **Counts.** 12 items remain OPEN; "OPEN" occurs 13 times. One checklist box
    is ticked; 11 are not.
  - The re-read of the passages a narrower box touches, and the emulator
    sitting, are owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D63** *(2026-10-10)* **P5 v2 draft: re-read after the box narrowed; the
  prior-knowledge section checked against its sources (one checklist box ticks;
  no item closes; nothing frozen).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md` (lines 7, 24, 139 and 148 reworded
  in place; line 169 ticked). D62 listed as owed, for the narrower box, the
  re-read of the prior-knowledge wording, the sealed test draw, the bin shares
  and the hypothesis numbers, and attributed that list to D59; D59's entry
  fixes only that the range of H is not narrowed, so that the priors of H need
  no restating, and the list was a working note of that sitting that entered
  the record with D62. Checklist line 169 asks for the prior-knowledge section
  to be checked against the ROADMAP entries it cites and for G0's pilot to be
  declared in it. Both are done here. Nothing was run; no OPEN item is decided.
  v1 remains the registered protocol. The decision sheet is not updated.
  - **The citation check.** Every figure in lines 17 to 20 was read against its
    source. D37 (line 17): H the flat eigen-direction, degenerate with ν
    (correlation −0.82), a 62% spread at 0.1 vol point of noise; D37 items (2)
    and (3). D38 (line 18): the point θ = (0.10, 0.35, −0.70, 0.04) (D37's
    truth and `TRUTH` in `layer4_calibrate_surface.py`; D38's entry does not
    restate it), the 25-quote grid (`NOISE_VUS`, five points at five
    maturities, "trim strikes 7→5"), ten draws, spreads of about 10%, 29% and
    48% at 0.1, 0.3 and 0.5 vol point, κ kept fixed (item (4)); D38 items (3)
    and (4) and caveats (a) and (b); the standard deviations 0.010, 0.029 and
    0.048 are those percentages of H = 0.10, as the sentence says. Line 19: H
    rails to its lower bound (D39, D41, D42, each at the fitted optimum); the
    overflow at low H, high ν and long maturity (D39 caveat (2); D41 phase 0).
    D39: IV-RMSE 0.888 vol point on the reduced span. The 1.1 vol point on the
    full one-year span is not in the ROADMAP; it is in
    `OVERLEAF/P1/calibration_paper_revised.tex`, line 284 ("root-mean-square
    error of about 1.1 percentage points on the full one-year span"), and the
    sentence attributes it to the P1 paper. D42: IV-RMSE 1.63 on ETH; ν 0.713
    (BTC 0.714 in D42's comparison, 0.71 in D41). D41: the one-year tenor the
    least H-informative (H-sensitivity 64.5 at the short end to 5.2 at one
    year; D42 65.7 to 5.9); the H 0.02, ν 0.63 overflow at T 0.986 non-finite
    at `N_riccati` 6000 and finite at 8000. The code comment:
    `calibrate_btc.py` lines 62 to 64 (T about 0.989, overflow at 8000, finite
    from 8200). The probe (line 20): D55, 1.67 s at D38's point and 1.58 s
    mean, 24 points with 9 failing, the four follow-ups. Line 15 was read
    against D59. Lines 21 and 22 were written from the committed output in D60
    and D62 and read then. Every figure holds; no sentence of the section
    changes. The second clause of line 169 is met by lines 21 and 22; the box
    is ticked.
  - **Read and unchanged.** The observable (line 45) is defined without
    reference to the box. The priors (lines 58 to 64) are written for H in
    [0.02, 0.48], which the order did not narrow; "all other parameters uniform
    on the box" reads over the box as written in; the means 0.12 and 0.32, the
    standard deviations 0.054 and 0.062 and the 0.20 of the shift ratio (line
    108) are unchanged. The reference (line 68) is under the uniform prior on
    that box. The contraction c (lines 79 to 89) is defined from the posterior
    and the prior of H; its definition does not depend on the box's width, its
    values do, and they are measured at run time; the cut-offs stay OPEN. The
    noise levels (line 56) are read: 1.0 vol point is motivated by the live
    misfit, a regime line 24 now says v2 does not speak to, which is no
    contradiction and is for the noise sitting. The sealed draw (line 93) has
    not been made and is made on the box as written in; nothing is restated.
    The bin shares (lines 99 to 101) are measured at run time, and the 10% stop
    rule, OPEN, is what handles a bin left thin by a narrower box. The
    hypotheses (lines 117 to 128) are in units of H or of posterior mass and
    name no range; OPEN as before. Gates G1 to G3 are unchanged.
  - **Updated wording.** Line 7: the pilot "is to be run" and its script and
    output "are to be committed" become "was run" (D60 to D62) and "are
    committed". Line 24: one sentence added, that the box lies at ν ≤ 0.35 and
    ρ in [−0.90, −0.70], that the live BTC and ETH optima (ν 0.71 from line 19;
    ρ −0.31 and −0.30 from D41 and D42) lie outside it so nothing in v2 speaks
    to that regime, and that D38's point, the one point at which H's scatter
    under noise has been measured on this grid, sits on three of the box's
    faces (ν 0.35, ρ −0.70, ξ₀ 0.04). Line 139: "G0's pilot is run before
    freezing" becomes "was run before freezing (D60 to D62)" and "fixes values
    of the design" becomes "fixed the box, the grid and the resolution". Line
    148: the surface in scope is "on the stated grid over the box under
    Objects". No line is added or removed.
  - **For the noise sitting.** A reading, labelled as such; nothing is set, and
    neither effect is measured. Two effects pull against each other. The
    H-signal sits at the short end of the skew's term structure (D38's span
    sweep; D41's H-sensitivity at one fitted point), and in the model the skew
    is smaller at smaller ν, so over ν ≤ 0.35 the flat regime may be reached at
    lower noise than over the proposed box. The uniform prior on ν [0.05, 0.35]
    also cuts off the H↔ν valley (correlations −0.82 in D37, −0.85 in D38,
    −0.88 to −0.91 in D41, −0.86 in D42), which tightens the posterior on H and
    moves pairs out of the flat bin. The 10% stop rule covers either case.
  - **Two facts the narrower box makes visible.** The live BTC and ETH optima
    (ν 0.71 to 0.714, ρ −0.30 to −0.31, H railed at 0.02; D41, D42) lie outside
    the box in ν and ρ, so v2's claims, which are for the box under Objects,
    say nothing about the regime those fits occupy; real surfaces are out of
    scope in any case (line 150). D38's point, the one point at which H's
    scatter under noise has been measured on this grid, lies on three of the
    box's faces, so D38's figures describe the box's edge, not its interior.
  - **Counts.** 12 items remain OPEN; "OPEN" occurs 13 times. Two checklist
    boxes are ticked; 10 are not.
  - The emulator sitting is owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D64** *(2026-10-10)* **P5 v2 draft: emulator form, design and gate written
  in (two OPEN items close; one checklist box ticks; nothing frozen; nothing
  built).** Recorded in `docs/protocols/P5_protocol_v2_draft.md` (line 52
  reworded; line 54, the OPEN item for the emulator, now the design; line 142,
  G1's OPEN ratio, now 5 with a refinement rule; line 172 ticked). The design
  is a choice of the author, made on the assistant's lean and declared as such;
  it rests on the record (the box, the grid and the resolution of D62, the cost
  of 288 core-seconds per (H, ν, ρ) over the grid measured there, the tolerance
  of D59 and the noise levels proposed) and on one piece of reasoning about the
  model, labelled below. Nothing was measured or built: line 7 keeps every
  emulator, reference posterior and network until after freezing. v1 remains
  the registered protocol. The decision sheet is not updated.
  - **The design.** Form: a tensor-product Chebyshev interpolant of G on the
    box, one interpolant per quote (35), in (H, ν, ρ, ξ₀) mapped affinely to
    [−1, 1]⁴, Chebyshev nodes of the first kind, barycentric evaluation;
    deterministic, no training, no seed, exact at the nodes. Nodes: 16 in H, 8
    in ν, 6 in ρ, 12 in ξ₀: 9,216 training surfaces at 768 (H, ν, ρ) triples,
    the ξ₀ nodes sharing each triple's three Riccati solves per maturity.
    Training data: priced by the path of `p5_g0_main.py` at `N_riccati` 8000
    with `U_max` 790, 500, 350, 250 and 180 and 504, 320, 224, 160 and 112
    nodes, each surface's strikes anchored on its own ATM quote at that
    setting; a training quote that is not finite stops the build, the box
    having passed G0 on its pilot and confirming points, not everywhere.
    Held-out set: 200 points uniform on the box from `default_rng(73)` (neither
    the sealed seed, nor the probe's 7, nor the confirmation's 61 and 62),
    priced the same way; e is the 99th percentile over their 7,000 quotes of
    |G̃ − G|, with the maximum and the per-maturity 99th percentiles beside it.
    G1's ratio: 5, so e ≤ s / 5, which is 0.02 vol point at s = 0.1, the figure
    D59 took as the basis of the pricer's tolerance of 0.01 vol point.
    Refinement: if e exceeds s / 5 at the smallest noise level under Noise (0.1
    as proposed), the node count is doubled once in the dimension whose
    Chebyshev coefficient tail is largest, the tail of a dimension being the
    largest absolute Chebyshev coefficient, over all 35 quotes and all indices
    in the other three dimensions, at that dimension's highest index (the
    coefficients are the cosine transform of the node values, so the rule needs
    no further pricing), and the build repeated at the new nodes, declared as
    such; no second refinement; Q1 is then claimed at the noise levels G1
    allows and at no other, as line 51 already provides. Evaluation: 9,216 node
    values per quote, 35 barycentric sums per surface, so the reference (G2)
    and the calibrators' training data cost nothing in pricing.
  - **What it rests on.** A reading, labelled as such. At a fixed standardised
    quote the implied vol is a smooth function of (H, ν, ρ, ξ₀) on the interior
    of the box: the price comes from a characteristic function whose dependence
    on the parameters is analytic, discretised at fixed settings, and inverted
    by a smooth map. Chebyshev interpolation of a function analytic on a
    neighbourhood of the box converges geometrically in the node count, at a
    rate set by the distance to its nearest singularity; for a function that is
    merely smooth the rate is algebraic. Neither statement is measured here;
    the held-out check under G1 measures the error, and the refinement rule is
    the provision if it is larger than reasoned. The main run's lattice (125
    points at six ξ₀) is not a test of smoothness and was not used as one.
    Against a neural-network emulator, the choice is that G̃ should be a fixed
    object with no training randomness, so that Q2's gap is the calibrators'
    alone (line 50), and that its error should be set by a node count checked
    on held-out points rather than by an optimiser. The risk is smoothness near
    H = 0.02 at the shortest maturity; the nodes cluster at the box's ends,
    where that risk sits.
  - **Cost.** 768 training triples and 200 held-out points, 968 sets of solves
    at the 288 core-seconds per (H, ν, ρ) over the grid measured in D62 with
    the pool running: about 77 core-hours, 19 h on four workers. A refinement
    repeats the training part at the doubled count, the first-kind nodes not
    being nested: at most 1,536 triples, about 123 core-hours more, about 2,500
    sets of solves in all. The cost limit under G0 presumed about 4,000 such
    sets.
  - **Not decided.** The noise levels (line 56), the reference's sampler and
    acceptance (lines 68 and 143), the test-set size (line 94), training
    adequacy (line 144), the Beta parameters, the cut-offs, the hypothesis
    numbers. The emulator's script and build follow freezing.
  - **Draft.** Line 52: the pricing path of the training surfaces is
    `p5_g0_main.py`'s at the settings under Gates; the earlier sentence on
    `calibrate_btc._cached_cf` and `surface_model` is replaced. Line 54: the
    OPEN item becomes the design. Line 142: "OPEN — the ratio" becomes 5 with
    the refinement rule. Line 172 ticked. No line is added or removed; 180
    lines.
  - **Counts.** 10 items remain OPEN; "OPEN" occurs 11 times. Three checklist
    boxes are ticked; 9 are not.
  - The noise sitting is owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D65** *(2026-10-10)* **P5 v2 draft: noise levels set at 0.1, 0.3 and 0.5
  vol point (one OPEN item closes; one checklist box ticks; nothing frozen).**
  Recorded in `docs/protocols/P5_protocol_v2_draft.md` (line 56, the OPEN item
  for the three levels, now values; lines 141 and 142, two words that said
  "proposed"; line 171 ticked). The choice is the author's on the assistant's
  lean, declared as such; it rests on the record (D38's measured scatter of H
  under noise, the priors' standard deviations, the bin cut-offs as proposed,
  D63's reading of the box) and on arithmetic from those figures, labelled
  below as a reading. Nothing was measured. v1 remains the registered protocol.
  The decision sheet is not updated.
  - **The decision.** s in {0.1, 0.3, 0.5} vol point, independent Gaussian on
    each quote, as a design factor; 1.0 vol point, proposed as the size of the
    live misfit, is dropped. Three levels, as proposed, so the count of
    experiments (two calibrators, two priors, three levels, the seeds of line
    75) is unchanged.
  - **What it rests on.** A reading, labelled as such. 0.1, 0.3 and 0.5 are the
    three levels at which H's scatter under noise has been measured in this
    repository: D38, at θ = (0.10, 0.35, −0.70, 0.04), spreads of about 10%,
    29% and 48% of H, standard deviations of roughly 0.010, 0.029 and 0.048
    (the record's own caveats: ten draws, strikes anchored once on the target,
    approximate figures). Taken as if they were posterior standard deviations
    and read against the priors' (0.054 under A, 0.062 under B), the
    contraction c = 1 − SD²_post / SD²_prior of line 81 would be about 0.97,
    0.71 and 0.21 under A and 0.97, 0.78 and 0.40 under B: one level in each of
    the proposed bins (identified at or above 0.9, partial from 0.5 to 0.9,
    flat below 0.5). D38's point lies on the box's face ν = 0.35, which by the
    model's skew scaling with ν (D63's reading; not measured) is the face of
    most information about H in ν; it also lies on the face ρ = −0.70, the
    smallest |ρ| in the box, and on the ξ₀ floor, so by the same reading it
    bounds the box's contraction in neither direction. The levels are placed by
    their reading at that one point; where the bins fall elsewhere in the box,
    Stage 1 measures (line 99). D63's two-way reading (the flat regime reached
    at lower noise over ν ≤ 0.35; the valley cut by the uniform prior on ν)
    stands and is not resolved here. At 1.0 vol point, beyond the range D38
    measured, the scatter extrapolated from its roughly linear growth with the
    noise (0.010, 0.029, 0.048 at 0.1, 0.3, 0.5) would exceed the prior's
    standard deviation at D38's point, so every pair would be flat: a level
    that adds pairs to one bin and tests nothing in the others. Its motivation,
    the live misfit (0.9 vol point on BTC's reduced span, D39; 1.63 on ETH,
    D42), belongs to a regime D63 recorded as outside the box and outside v2's
    claims (line 150). The reading equates an optimiser ensemble's scatter
    under D38's anchoring with a posterior standard deviation under v2's
    observable; it is a basis for choosing levels, not a prediction of the bin
    shares, which Stage 1 measures (line 99), and the cut-offs it uses are
    still OPEN (line 89). The main run's lattice measured the pricer's
    convergence, not H's identifiability, and is not used here.
  - **Unchanged.** The tolerance under G0, a tenth of the smallest noise level,
    still 0.1 (line 141); G1's ratio, e ≤ s / 5, 0.02 vol point at 0.1 (line
    142); the 10% stop rule for a thin bin (line 101, OPEN); the hypotheses,
    scored per bin and pooled over levels (line 126); "three levels" under
    Scope (line 148).
  - **Not decided.** The contraction cut-offs and the 10% floor, the
    reference's sampler and G2's tolerance, training adequacy and G3's
    tolerance, the test-set size, the Beta parameters, the hypothesis numbers,
    the prior-work section.
  - **Draft.** Line 56: the OPEN span becomes the levels as values with the
    reason, 1.0 named as dropped. Line 141: "the smallest noise level proposed
    under Noise" and "what G1 as proposed allows" lose "proposed". Line 142:
    "(0.1 as proposed)" becomes "(0.1)". Line 171 ticked. No line is added or
    removed; 180 lines.
  - **Counts.** 9 items remain OPEN; "OPEN" occurs 10 times. Four checklist
    boxes are ticked; 8 are not.
  - The reference sitting (G2) is owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D66** *(2026-10-11)* **P5 v2 draft: the exact reference and gate G2 written
  in (two OPEN items close; one checklist box ticks; nothing frozen; nothing
  run).** Recorded in `docs/protocols/P5_protocol_v2_draft.md` (line 68, the
  OPEN item for the sampler, chains, length, R-hat ceiling and effective-sample
  floor, now values; line 143, G2's OPEN tolerance, now values with its pilot
  surfaces and grids; line 174 ticked). The design is a choice of the author on
  the assistant's lean, declared as such; it rests on what the draft now fixes
  (the box, the emulator of D64, the noise levels of D65, the priors, the
  contraction of line 81, the measures of lines 107 to 110) and on reasoning
  labelled below. Nothing was measured or run; the reference's script is
  written after freezing (line 7). The number of test θ (line 94) stays OPEN:
  it is set from the cost G2 measures. v1 remains the registered protocol. The
  decision sheet is not updated.
  - **The design.** Target: the posterior of θ = (H, ν, ρ, ξ₀) on the box under
    the likelihood Gaussian with the known s on the 35 quotes of the
    observable, log L = −Σ(y − G̃(θ))² / (2 s²), and the uniform box prior; the
    H-marginal is what the measures use. Sampler: adaptive random-walk
    Metropolis with a Gaussian proposal in the box mapped affinely to [0, 1]⁴;
    one generator per pair, `default_rng([83, i, j])` for the i-th test θ and
    the j-th noise level, serving the starts, the proposals and the acceptance
    draws; four chains from points uniform on the box; the proposal covariance
    2.38²/4 = 1.416 times 1/12 times the identity at the start (the variance of
    the uniform start) and, every 500 of the 10,000 burn-in steps per chain,
    reset to 1.416 times the empirical covariance of that chain's steps so far
    with a floor of 1e-10 on the diagonal; 40,000 sampling steps per chain with
    the proposal fixed; a proposal outside the box rejected. Acceptance, on the
    sampling steps only with each chain split into halves: the classical
    between-within split R-hat of Gelman and others (not rank-normalised) at
    most 1.01 on each of H, ν, ρ, ξ₀; the effective sample size of H over the
    eight half-chains, by the same between-within estimator, at least 2,000; a
    failing run repeated once with both lengths doubled from fresh starts of
    the same generator; a second failure recorded and declared as a reference
    failure for that (surface, noise level) pair. Priors A and B:
    self-normalised importance weights proportional to the prior's density of H
    at each pooled sampling step (ν, ρ and ξ₀ being uniform under all three
    priors), Kish's effective sample size (Σw)²/Σw² at least 1,000 under each,
    else a direct run under that prior with the same sampler and acceptance;
    means, variances, quantiles and interval probabilities as weighted means
    and weighted quantiles of the pooled sampling steps. Reported per pair: the
    posterior means of H under A and B (Ĥ_exact of lines 107 and 108), the
    posterior variance of H under A (c, line 81), the equal-tailed 90% interval
    under A, the posterior probability under A of any given interval (m, line
    110), the acceptance rate, R-hat and the effective sample sizes. G2: 12
    pilot surfaces, four points uniform on the box from `default_rng(79)` at
    the three noise levels with one noise draw each from `default_rng(89)`,
    none a test surface (line 95); on each, the MCMC posterior of H against a
    brute-force grid posterior on G̃ by quadrature: a cell-centred tensor grid
    of 32 points per dimension (1,048,576 cells) over the smallest axis-aligned
    box containing the four chains' sampling-phase samples, each side extended
    by 20% and clipped to the box, the posterior being the prior-A-weighted
    likelihood summed over cells; at the largest noise level a second grid, as
    a check against a mode the chains missed, of 48 cell-centred points in each
    of H, ν and ρ over the whole box with ξ₀ on the 32 points of the first grid
    (3,538,944 cells); tolerance under prior A on every pilot surface: the
    posterior mean of H within 0.001 and c within 0.01. Seeds 79, 83 and 89 are
    none of the sealed seed, the probe's 7, the confirmation's 61 and 62, or
    the emulator's 73.
  - **What it rests on.** A reading, labelled as such. The posterior is on a
    four-dimensional box with a smooth likelihood of 35 barycentric sums over
    9,216 nodes each (D64). Its shape at the smallest noise is narrow in ξ₀
    (D37's sensitivity ordering, ξ₀ far above ρ, ν and H) and elongated along
    the H↔ν valley, whose correlation was measured at the truth (D37, D38) and
    at fitted optima (D41, D42): −0.82 to −0.91 on full surfaces, −0.92 on
    D37's put wing. An adapted Gaussian proposal follows such a shape; a
    uniform grid over the whole box does not resolve the ξ₀ direction at any of
    the three noise levels (at 0.5 vol point, 48 cells over ξ₀ are each several
    times the posterior's width there), which is why the grid check is placed
    on the sampled support, and the whole-box grid, a check against a missed
    mode, keeps ξ₀ on that support. The thresholds are conventional, chosen
    here and not derived: R-hat 1.01, effective sample sizes 2,000 and 1,000,
    four chains. The tolerances are set so that reference error cannot move a
    verdict or a bin: 0.001 on the mean of H is a tenth of H-a's expected value
    of 0.01 (line 121); 0.01 on c is a hundredth of its range and a tenth of
    the narrowest proposed bin (identified, 0.1 wide). None of this is measured
    here; G2 measures it.
  - **Cost.** Arithmetic, not a measurement: a chain step is one evaluation of
    G̃, 35 barycentric sums (D64), so a run of 4 × 50,000 steps is 200,000
    evaluations, of the order of seconds to a minute per posterior on the
    author's machine; the grid check is 1,048,576 evaluations per pilot surface
    and 3,538,944 for the whole-box grid, minutes each, made 12 times and 4
    times. The cost per posterior is measured at G2 and written in with the
    test-set size (line 94); by this arithmetic the test-set size will be set
    by the calibrators' cost, not the reference's. Line 94 is unchanged here;
    if G2 bears this out, the amendment at line 94 should name the cost that
    binds.
  - **Not decided.** The number of test θ (line 94), training adequacy and G3's
    tolerance (line 144), the Beta parameters (line 66), the contraction
    cut-offs (line 89) and the 10% floor (line 101), the hypothesis numbers
    (line 117), the calibrators' architecture and training (line 75), the
    prior-work section. The reference's script and its runs follow freezing.
  - **Draft.** Line 68: the OPEN span becomes the design. Line 143: the OPEN
    tolerance becomes the pilot surfaces, the grids and the tolerance. Line 174
    ticked. No line is added or removed; 180 lines.
  - **Counts.** 7 items remain OPEN; "OPEN" occurs 8 times. Five checklist
    boxes are ticked; 7 are not.
  - The training-adequacy sitting (G3) is owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D67** *(2026-10-11)* **P5 v2 draft: gate G3 written in (one OPEN item
  closes; one checklist box ticks; nothing frozen; nothing run).** Recorded in
  `docs/protocols/P5_protocol_v2_draft.md` (line 144, G3's OPEN tolerance, now
  values; line 178 ticked; two clauses of line 68 widened so that the reference
  serves pilot surfaces and reports the posterior mean of ξ₀). The values are a
  choice of the author on the assistant's lean, declared as such; they rest on
  the draft as it stands (H-a's expected value at line 121, the reference of
  line 68 and its pilot-surface procedure, the noise levels of D65, the seeds
  already named) and on reasoning labelled below. Nothing was trained, measured
  or run. v1 remains the registered protocol. The decision sheet is not
  updated.
  - **The values.** Pilot surfaces: 48, from 16 points uniform on the box
    (`default_rng(97)`) at the three noise levels with one noise draw each
    (`default_rng(101)`), none a test surface (line 95), the exact reference
    computed on each by the procedure of line 68, whose generator now takes i
    from the surface's index in the pilot set; a reference failure removes that
    surface from the gate's set and is declared. Each check runs, per
    calibrator, over the 16 surfaces at that calibrator's noise level. ξ₀
    recovery: the slope of the least-squares line of the calibrator's point
    estimate of ξ₀ on the exact posterior mean of ξ₀ under the calibrator's
    training prior, above 0.95; the regressor is the exact posterior mean and
    not the truth because a faithful posterior mean regressed on the truth has
    a slope below one by shrinkage, by an amount that grows with the noise, so
    a check against the truth could fail an adequately trained calibrator at
    0.5 vol point for a reason unrelated to training; line 68 now reports the
    posterior means of ξ₀ under A and B. Each calibrator is to output a point
    estimate of ξ₀ beside Ĥ. Stability: the median over the 16 surfaces of
    abs(d), with d = Ĥ_net,P − Ĥ_exact,P for the calibrator's training prior P,
    at training size 2n within 0.002, in absolute value, of that at n, a fifth
    of H-a's expected bound of 0.01. Ladder: training sets of 10,000 × 2^k
    surfaces, k = 0 to 4 (10,000 to 160,000), each the first 10,000 × 2^k
    surfaces of a single draw of 320,000 from the training prior and the noise
    level, the two calibrators at a prior and noise level sharing the draw,
    with the first of the training seeds to be declared at line 75 (none is yet
    named); n is adequate when both checks hold at n, the stability check
    needing the run at 2n, which is made for the check only; the training-set
    size is the smallest adequate n. Budget: that draw of 320,000 surfaces per
    prior and noise level, and 4 h of wall per training run on the author's
    machine with the pool of four, a run being the whole calibrator (all five
    members of the ensemble); a rung n fails if its run at n or its check run
    at 2n exceeds either budget; if no rung passes, P5 stops, as line 144 says.
    Seeds 97 and 101 are none of the sealed seed, the probe's 7, the
    confirmation's 61 and 62, the emulator's 73, G2's pilot 79 and 89, or the
    reference's 83.
  - **What it rests on.** A reading, labelled as such. The stability tolerance
    is set against the smallest number in the hypotheses table, so that an
    inadequately trained calibrator cannot move H-a's verdict by its training
    alone. Forty-eight pilot surfaces give each noise level sixteen, which is
    few; the gate is a check of adequacy, not a measurement of fidelity, which
    Stage 2 makes on the test set. The factor of two is what line 144 already
    fixed ("a doubling"). The budget is chosen, not derived: the emulator makes
    surfaces free (D64), so the binding cost is the network's training on a
    four-core laptop (D60's record of the machine: an i7-1165G7 with 4 cores
    and no accelerator recorded), which G3 measures. The ξ₀ check is kept, with
    the exact posterior mean as its regressor, because ξ₀ is the parameter the
    record finds identified wherever it has assessed it: at D38's point (D37,
    D38) and on live BTC (D39); on ETH, D42 reports ξ₀ interior to its bound
    and does not assess it. A calibrator that cannot recover ξ₀ has not learned
    the surface.
  - **A consequence for the architecture sitting.** Each calibrator must output
    a point estimate of ξ₀ beside Ĥ (the ensemble by a second output of each
    member; the ratio estimator by its posterior mean of ξ₀), and the training
    seeds must be named: the architecture, the optimiser, the stopping rule,
    the ξ₀ output and the number and values of the training seeds stay OPEN at
    line 75, and that sitting is to provide them.
  - **Not decided.** The architecture and training procedure (line 75), the
    Beta parameters (line 66), the contraction cut-offs (line 89), the 10%
    floor (line 101), the number of test θ (line 94), the hypothesis numbers
    (line 117), the prior-work section.
  - **Draft.** Line 144: the OPEN tolerance becomes the pilot surfaces, the two
    checks with their tolerances, the ladder, the seed and the budget. Line 68:
    the reference's generator takes i from the θ's index in the set in
    question, test or pilot, and the posterior means of ξ₀ under A and B join
    what is reported. Line 178 ticked: all four gate tolerances are declared
    (D59, D64, D66, D67), the scoping script was committed before the scoping
    run (D60) and the main run's settings were written in before the main run
    (D61). No line is added or removed; 180 lines.
  - **Counts.** 6 items remain OPEN; "OPEN" occurs 7 times. Six checklist boxes
    are ticked; 6 are not.
  - The architecture sitting (line 75) is owed next.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D68** *(2026-10-11)* **P5 v2 draft: the calibrators' architecture and
  training written in (one OPEN item closes; one checklist box ticks; nothing
  frozen; nothing built).** The architecture sitting owed by D67. Recorded in
  `docs/protocols/P5_protocol_v2_draft.md`: line 75, the OPEN architecture,
  optimiser, stopping rule, ratio-estimation variant, posterior sample count
  and number of training seeds, becomes values; line 175 is ticked; the two
  phrases of line 144 that pointed "below" to line 75 now point to The
  calibrators. The values are a choice of the author on the assistant's lean,
  declared as such. No network was built, trained or timed; nothing is frozen.
  - **The design.** One trunk for both calibrators: a multilayer perceptron,
    three hidden layers of 128 units, ReLU, no normalisation or dropout,
    PyTorch's default initialisation, declared without search. Inputs are the
    35 quotes standardised by the training set's per-quote mean and standard
    deviation; θ is mapped to [0, 1]⁴ as the reference maps it. The ensemble:
    five heteroscedastic members with four outputs each, the means of H and ξ₀
    and two variances of the form softplus(raw) + 1e-6, trained by Gaussian
    negative log-likelihood on the whole training set, differing by
    initialisation and shuffling, without the adversarial step; Ĥ and the ξ₀
    estimate are the means over members; the total variance of H is the mean of
    the members' variances plus the variance of their means, in box units, the
    two components reported separately, and line 72's interval is Ĥ ± 1.645
    times its square root. The ratio estimator: the binary-classifier variant,
    39 inputs (quotes and mapped θ), one logit read as log r(θ, y), trained by
    binary cross-entropy on each batch's joint pairs against the marginal pairs
    made by a cyclic shift of θ within the batch; its posterior under the
    training prior, density ∝ P(H) · r, is sampled by the reference's own
    sampler (line 68) with log P(H) + logit as the target, the same chains,
    lengths, adaptation, acceptance and repeat, generator `default_rng([107, p,
    q, i, j])`; Ĥ and the ξ₀ estimate are posterior means over the accepted
    run's pooled steps, 160,000 or 320,000 after the repeat, the interval
    equal-tailed; a sampler failure after the repeat is the calibrator's
    failure on the pair at that seed, excluded from every measure using that
    network's output, S and E at that seed included, and counted per bin, and a
    bin where a calibrator's failed pairs exceed line 101's floor, as a share
    of the bin's pairs, is untestable for that calibrator. Optimiser: Adam,
    learning rate 0.001, β (0.9, 0.999), no weight decay, batch 256 with a
    partial final batch kept, constant rate. Stopping: a validation set of
    4,000 surfaces per prior and noise level from `default_rng([113, p, j])`,
    shared by both calibrators, all seeds and all rungs; each calibrator's
    training objective evaluated on it after every epoch, in its drawn order in
    batches of 256 with the last of 160 kept and the marginal pairs formed as
    in training; stop at 50 epochs without improvement or at 1,000 epochs; best
    parameters kept; quote noise fixed per training surface; the rule acts per
    ensemble member and on the one ratio estimator. A non-finite loss stops the
    run and is declared: at G3 the rung fails; after G3 one retraining from
    seed q = 6, which takes the failed seed's place in the pairing, then the
    cell stands with four seeds. Seeds: five per calibrator, prior and noise
    level, q = 1 to 5 (q, not k, which line 144 uses for the rung), a seed
    being the whole calibrator; seed q fixes the training draw
    `default_rng([109, p, j, q])` (G3's draw of 320,000 is q = 1) and torch's
    generator, seeded with 109,000,000 + 10,000·p + 1,000·j + 10·q + m, m the
    member and 0 for the ratio estimator. Stage 2's measures are formed per
    seed, the A and B networks of the same q paired for S and E, and the seed
    floor is abs(S) over the ten unordered pairs of seeds at the same prior.
    The leaf's requirements file pins torch and numpy when the scripts are
    committed; CPU, thread count declared; bitwise reproduction across machines
    not claimed.
  - **What it rests on.** A reading, labelled as such. One trunk keeps the two
    calibrators like for like, and a fixed modest size, about 38,000 weights
    for 35 inputs and a four-dimensional target, is declared without search, as
    line 150 requires. Of the sources the decision sheet
    (`docs/protocols/P5_v2_decision_sheet.md`, item 8) read at source, the
    benchmark of Miller, Weniger and Forré used 128 hidden units in three
    residual blocks, Lakshminarayanan, Pritzel and Blundell one hidden layer of
    50 ReLU units, and the sbi library's default classifier 50 hidden features
    (its depth not shown on the pages the sheet read), so 128 units is the
    largest width among them and three layers lies between the one of
    Lakshminarayanan, Pritzel and Blundell and the three residual blocks of
    Miller, Weniger and Forré; no claim is made that it is best. The binary
    classifier estimates the ratio itself, which the sheet records as Miller,
    Weniger and Forré's point against the multi-class variant, and is the
    nearest standard variant to the binary heads of Brigo, Huser and Leonte;
    the balanced variant is conservative by design and the telescoping one
    would need its own implementation and calibration step. The ratio
    estimator's posterior is sampled with the reference's own sampler rather
    than summed on a grid for D66's reason: a whole-box grid does not resolve
    the posterior in ξ₀ at any of the three noise levels. Sampler error of the
    reference's accepted size then enters Ĥ_net: by the sheet's arithmetic
    about 0.0005 in the posterior mean of H in the Identified bin at 1,000
    effective draws, less at the 2,000 the reference requires, a twentieth of
    H-a's 0.01, which is scored in that bin. Where the sampler fails, the pair
    is dropped at that seed, so a bin's measures are made on survivors; the
    per-bin failure counts are reported so that a reader can see where that
    happened, and if failures gather in the Flat bin, H-d's m is measured on
    the surfaces that remained. Early stopping on a held-out loss touches no H
    output; the patience of 50 is the value the sheet records for Brigo, Huser
    and Leonte, generous because D43 found fixed budgets under-training. Adam
    at 0.001 is Layer 3's setting (`layer3_deep_hedging.py`, D40's engine);
    batch 256, the validation size and the epoch cap are chosen. The arXiv
    pages of Hermans, Begy and Louppe (1903.04057) and Lakshminarayanan,
    Pritzel and Blundell (1612.01474) were re-read today: the first describes a
    learned likelihood-to-evidence ratio embedded in MCMC (its abstract; the
    description of it as a binary classifier is, as the sheet records, Miller,
    Weniger and Forré's), the second deep ensembles, whose member details are
    as the sheet records from that paper's body.
  - **Cost.** An estimate, unmeasured. A training step of this network at batch
    256 on the author's CPU is of the order of a millisecond or two, so an
    epoch at the top rung's check run (320,000 surfaces, 1,250 steps) is about
    two seconds, 1,000 epochs about 35 minutes per member and under three hours
    for the five, inside G3's 4 h; G3 measures it. The ratio estimator's
    posterior needs 200,000 network evaluations in batches of four chains,
    seconds per surface; with N test θ there are 3N × 2 × 5 such posteriors,
    15,000 at N = 500 and 30,000 at N = 1,000, and the chains of many surfaces
    can share one batch.
  - **Consequences for other sittings.** Line 94's rule sets the number of test
    θ from the reference's measured cost at G2; it should count the ratio
    estimator's sampling beside it, and that is left to the test-θ sitting. The
    pairing of seeds written at line 75 defines S, E and the seed floor of
    lines 108, 109 and 111 per seed, which the hierarchical bootstrap of line
    132 then resamples; those lines are unchanged in wording. The failure rule
    borrows line 101's floor for a second purpose; when line 101 is settled the
    two uses are to be reconciled deliberately, not by reference. The
    validation set is shared across seeds, so the seed floor leaves out
    validation-set variation and understates between-run variability by that
    much; a reading.
  - **Not decided.** The Beta parameters (line 66), the contraction cut-offs
    (line 89), the number of test θ (line 94), the 10% floor (line 101), the
    hypothesis numbers (line 117), the prior-work section.
  - **Draft.** Line 75: the OPEN item becomes the architecture, the inputs, the
    two calibrators' outputs and losses, the ratio estimator's posterior and
    its sampler, the optimiser, the stopping rule, the failure rule and the
    seeds. Line 144: "defined with the architecture below" becomes "defined
    under The calibrators" and "with the first of the training seeds to be
    declared below" becomes "the draw being that of training seed q = 1 under
    The calibrators", line 75 being above line 144. Line 175 ticked. No line is
    added or removed; 180 lines.
  - **Counts.** 5 items remain OPEN; "OPEN" occurs 6 times. Seven checklist
    boxes are ticked; 5 are not.
  - The Beta parameters (line 66) are owed next: the decision sheet puts its
    item 6 before the cut-offs and the floor.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D69** *(2026-10-11)* **P5 v2 draft: the Beta parameters written in (prior A
  accepted, prior B given A's variance; one OPEN item closes; one checklist box
  ticks; nothing frozen; nothing run).** The sitting the decision sheet's item
  6 puts before the cut-offs and the floor. Recorded in
  `docs/protocols/P5_protocol_v2_draft.md`: line 66, the OPEN Beta parameters,
  becomes values with their reasons; lines 62 and 64 carry prior B's new
  parameters and standard deviation; line 173 is ticked. The values are a
  choice of the author on the assistant's lean, declared as such. Nothing about
  the pricer, the emulator or any posterior was computed; nothing is frozen.
  - **The values.** Prior A: 0.02 + 0.46 · Beta(2.5, 9), as proposed (mean
    0.12, standard deviation 0.054, mode 0.093, central 90% [0.046, 0.220],
    2.0% of its mass at H ≥ 0.25). Prior B: 0.02 + 0.46 · Beta(235/23, 376/69),
    about Beta(10.217, 5.449), the scaled Beta with v1's mean of 0.32 and A's
    variance; on the unit interval mean 15/23 and variance 36/2645, so a + b =
    47/3 exactly (standard deviation 0.054, mode 0.330, central 90% [0.227,
    0.403], 10.6% of its mass below 0.25, 1.1e-4 at or below 0.12). The
    proposal's Beta(7.5, 4) had standard deviation 0.062. The two priors are
    equal in variance and differ in skew, 0.72 and −0.30, as any scaled-Beta
    pair with these means on this support must. The gap of 0.20 is 3.73
    standard deviations of either prior; the densities cross at H = 0.2197 and
    the overlap coefficient is 0.090 (0.114 under the proposal). Var_prior(H)
    in the contraction is declared the analytic variance of A, 0.00288 exactly.
    Every figure in this bullet is arithmetic on the Beta density, done today
    in exact rationals and with scipy's beta; in code the parameters are
    floats, at which the two variances agree to about 1e-16.
  - **What it rests on.** A reading, labelled as such. v1 registered the
    support, the scaled-Beta family, the means 0.12 and 0.32, the 0.20 gap and
    priors that "differ only in where they put their mass" (v1, Decision 2);
    equal variances make that literal, where the proposal's pair differed in
    spread as well (variance ratio 4/3, the sheet's arithmetic). A is kept
    because its 0.054 is the figure line 24 and the bins use: line 24's
    "identifies H well against a prior of standard deviation 0.054", D65's
    reading of the noise levels under A (c about 0.97, 0.71 and 0.21 at D38's
    point), and the cut-offs at line 89, which are fractions of A's variance;
    only B moves. A defines the bins and m because one prior must, for c to be
    one number per pair, and v1 made A the home prior (its P4 control is at
    home on test H from prior A); line 66 now says so. Under a Gaussian
    approximation to the likelihood in H, equal prior variances make the
    reference's own shift ratio 1 − c, so at the Identified bin's edge it is
    0.10 wherever the data sit; under the proposal it ran from 0.065 to 0.118
    at c = 0.9 as the likelihood's centre moves across the range of H (the
    sheet's "about 0.07 to 0.12", recomputed today). That bears on H-c, whose
    expected median S_net in the Identified bin is "above 0.10": under the
    proposal a faithful calibrator could have cleared it on the reference's own
    shift alone; with equal variances 0.10 is the reference's own value at the
    edge, not a margin above it, which the hypothesis-numbers sitting inherits
    as the cleaner yardstick. The real priors are skewed and c uses the exact
    posterior, so the reference's S will depart from 1 − c, most where a
    posterior sits in one prior's tail; the size is a Stage 1 fact. Half the
    uniform test θ lie where A has 2% of its mass, and 2.8e-6 of it lies above
    H = 0.40, which the sheet asks the author to be content with; it is
    accepted as the design: v1 chose a neutral test set so that neither network
    has home advantage, and the swap asks what a calibrator does where its
    prior and the data disagree. For the reference's reweighting from the
    uniform prior (line 68), a completely flat posterior would give an
    effective-sample fraction of 0.39 under A and 0.42 under the new B, 0.48
    under the proposal's B (the sheet's idealised case, recomputed); the
    direct-run fallback stands. Not taken: a mirror pair, which would move a
    registered mean; a wider or narrower common width; perturbing one prior in
    place of the swap; training once on the uniform prior and reweighting,
    which is not the draft's question.
  - **What changes elsewhere.** Nothing in wording. D63 and D65 cite B's
    standard deviation as 0.062, as it then stood, and are left as the record;
    D65's contraction reading under A is unchanged, and its figures under B
    (0.97, 0.78, 0.40) would now equal A's. Line 24 reads against A's 0.054 and
    is unchanged; line 56 reads against both priors' and still places one level
    in each bin, B's figures under D65's reading now equalling A's. The exact
    parameters are unround; the rounded Beta(10.22, 5.45) would break the
    equality of variances by about 0.02%, so the exact values are written and
    the rounded ones given as "about".
  - **Not decided.** The contraction cut-offs (line 89), the number of test θ
    (line 94), the 10% floor (line 101), the hypothesis numbers (line 117), the
    prior-work section.
  - **Draft.** Line 62: prior B's parameters. Line 64: prior B's standard
    deviation, 0.062 to 0.054. Line 66: the OPEN item becomes the values, what
    each prior is, the gap, the equal-variance consequence for the reference's
    shift ratio, the 2% point accepted, why A defines the bins, the analytic
    Var_prior(H) and what was not taken. Line 173 ticked. No line is added or
    removed; 180 lines.
  - **Counts.** 4 items remain OPEN; "OPEN" occurs 5 times. Eight checklist
    boxes are ticked; 4 are not.
  - The contraction cut-offs (line 89) are owed next, with the 10% floor (line
    101) in view, as the sheet's items 9 and 11.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D70** *(2026-10-11)* **P5 v2 draft: the contraction cut-offs accepted and
  the 10% floor written in (two OPEN items close; one checklist box ticks;
  nothing frozen; nothing run).** The sitting at which the decision sheet's
  items 9 and 11 are decided, with the noise levels (D65) and the priors (D69)
  already in place. Recorded in `docs/protocols/P5_protocol_v2_draft.md`: line
  89, the OPEN cut-offs, becomes the accepted values with their meaning,
  conventions and what was not taken; line 101, the OPEN floor, becomes a rule;
  line 75's per-calibrator failure rule is reconciled with it; one clause of
  line 126 is corrected; line 176 is ticked. The values are a choice of the
  author on the assistant's lean, declared as such. No posterior or contraction
  has been computed; nothing is frozen.
  - **The values.** Cut-offs 0.9 and 0.5, as proposed. The floor: fewer than
    10% of the scored pairs pooled over the noise levels, or fewer than 100
    pairs, makes a bin's hypotheses untestable for both calibrators, declared
    from the reference alone before any network is scored, a fourth outcome
    beside the three verdicts of line 133; the pairs counted are those on which
    the reference was accepted, and the number of distinct θ in each bin is
    reported beside the count; nothing is adjusted to fill a bin, and bin
    shares seen on pilot surfaces decide nothing. Conventions at line 89: c per
    pair from the pooled sampling steps of the reference's accepted run under A
    (importance-weighted, or the direct run under A where line 68 requires
    one), the weighted variance Σw(H − H̄)²/Σw against the analytic
    Var_prior(H); not clipped, a negative c filed under Flat and counted; each
    pair binned once by its c as computed, the pairs within 0.02 of a cut-off
    reported per bin; each hypothesis also re-scored twice, with both cut-offs
    moved together by +0.05 (0.95 and 0.55) and by −0.05 (0.85 and 0.45), the
    floor applied to each shifted bin, reported and deciding nothing. Line 89's
    "c depends only on the observed surface and the reference" now names prior
    A and says "on no network". Line 75's per-calibrator failure rule, which
    D68 left borrowing "the share at line 101" to be reconciled once the floor
    was settled, now applies the same two conditions per calibrator: failed
    pairs above 10% of the bin's pairs, or leaving fewer than 100. Line 126's
    "the reference itself has S near one" in the flat bin becomes "a large S,
    above 0.5 on the Gaussian reading ... and approaching one as c falls to
    zero".
  - **What it rests on.** A reading, labelled as such. In posterior standard
    deviations under A (0.054), c ≥ 0.9 is at most 0.017 and c ≥ 0.5 at most
    0.038, by 0.054 × √(1 − c). On a Gaussian reading of the likelihood in H, c
    = 0.5 is where the data and the prior carry equal precision, so Flat is
    where the prior outweighs the data, and c = 0.9 is where the data carry
    nine times the prior's precision; 0.5 is the one value with a meaning of
    its own on that reading (every other c is a precision ratio c/(1 − c), 0.75
    three times and 0.95 nineteen), and 0.9 is a round convention, a posterior
    variance a tenth of the prior's, kept as proposed and guarded by the
    re-scoring. With the priors' variances equal (D69) the reference's own
    shift ratio on that reading is 1 − c, 0.10 at the Identified edge and 0.50
    at the Flat edge, so in the Flat bin it runs from 0.5 towards one, which is
    why line 126's "near one" is corrected; the sheet's item 9 table gave 0.07
    to 0.12 and 0.39 to 0.56 at those edges under the proposal's unequal
    variances. D65's reading of D38's scatter places the three noise levels one
    per bin under these cut-offs (c about 0.97, 0.71 and 0.21 at D38's point),
    so the cut-offs and the levels agree as proposed. The sheet read
    Betancourt, Schad, Betancourt and Vasishth, and BayesFlow's diagnostic at
    source: all define contraction by this variance ratio, none proposes
    cut-offs, Schad, Betancourt and Vasishth describe contraction below 0.5 as
    weak and values approaching one as very strong in a worked example, and
    BayesFlow's implementation clips negative values to zero; the draft does
    not clip, so that nothing is hidden, and counts instead. The Monte Carlo
    error in c, (1 − c)·√(2/(n − 1)) for a roughly Gaussian posterior (the
    sheet's formula), is about 0.004 at c = 0.9 and 0.022 at c = 0.5 at the
    1,000 effective draws under A that line 68's Kish floor requires, and 0.003
    and 0.016 at 2,000, line 68's figure for a direct run; so the 0.02 window
    around each cut-off, a round figure above G2's tolerance, is of the order
    of the error at the lower edge and between four and six times it at the
    upper; G2 checks c to 0.01 on its pilot surfaces. The ±0.05 re-scoring is
    added so that a verdict's dependence on the edges is visible without
    deciding anything; it and the 100-pair count are chosen, not derived. The
    floor is of pooled pairs because the verdicts are pooled over noise levels
    (line 126); an absolute count is added because a share alone says nothing
    about the bootstrap of a median while the number of test θ is open: the 10%
    is 0.3N pairs at N test θ (60 at N = 200, 150 at 500, 300 at 1,000) and
    below N = 334 the count governs. The floor counts pairs, not θ, because the
    verdicts and line 132's bootstrap are over pairs within seed; 100 pairs may
    be as few as 34 θ, which is why the distinct θ per bin are reported. The
    sheet records Hesterberg's warning that percentile bootstrap intervals
    under-cover in small samples, with no minimum given. A share near 10% has a
    binomial 95% half-width of about 3.4 points on 300 pairs treated as
    independent (the three pairs sharing a θ are not), and the rule applies to
    the observed share. Not taken: equal-occupancy bins, v1's quintiles of
    CRB_H, because the hypotheses' numbers refer to a fixed meaning of each bin
    and the floor handles occupancy; two bins, which would remove the bin H-b
    is scored in; hypotheses on c as a continuous variable, which line 134's
    dose-response already reports; a divergence-based contraction, a different
    quantity; a per-level floor, which would multiply the decisions the pooled
    verdict does not need.
  - **What changes elsewhere.** The hypothesis-numbers sitting (line 117)
    inherits the reference's own S of 0.10 at the Identified edge and 0.50 at
    the Flat edge on the Gaussian reading, and the ±0.05 re-scoring beside each
    verdict. Line 75's failure rule, line 89's first sentence and line 126's
    clause are the only wording changed outside the two OPEN spans; lines 83 to
    87 (the bins), 93 to 95, 132 to 135 and 143 are unchanged. D63's "The 10%
    stop rule covers either case" (its reading for the noise sitting) now
    refers to the rule as written.
  - **Not decided.** The number of test θ (line 94), the hypothesis numbers
    (line 117), the prior-work section.
  - **Draft.** Line 89: the first sentence reworded; the OPEN item becomes the
    accepted cut-offs, their meaning, the reference's shift ratio at the edges,
    D65's placement, the sources via the sheet, the conventions, the re-scoring
    and what was not taken. Line 101: the OPEN floor becomes the rule with its
    reasons. Line 75: the per-calibrator failure rule applies the floor's two
    conditions. Line 126: one clause. Line 176 ticked. No line is added or
    removed; 180 lines.
  - **Counts.** 2 items remain OPEN; "OPEN" occurs 3 times. Nine checklist
    boxes are ticked; 3 are not.
  - The hypothesis numbers (line 117) are owed next; the number of test θ (line
    94) waits on G2's measured cost.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D71** *(2026-10-11)* **P5 v2 draft: the hypothesis numbers written in (H-c
  scored on excess, H-d made one-sided, H-a and H-b accepted; one OPEN item
  closes; one checklist box ticks; nothing frozen; nothing run).** The sitting
  the decision sheet's item 12 asks for. Recorded in
  `docs/protocols/P5_protocol_v2_draft.md`: line 117's OPEN becomes "(D71)";
  rows 123 and 124 of the table change; the reasons for every number are
  written under the table at line 126; the definitions the sheet found missing
  are written at lines 107, 109, 111, 126, 128, 132 and 134, and line 101's
  phrase on the bootstrap follows line 132; line 177 is ticked. The bets are
  the author's on the assistant's lean, declared as such. Nothing about any
  posterior, verdict or interval was computed; nothing is frozen.
  - **The table.** H-a, median abs(d) in Identified: expected below 0.01,
    refuted above 0.03, unchanged. H-b, median E in Partial: expected above
    0.15, refuted below 0.05, unchanged. H-c, now median E in Identified (was
    S_net): expected above 0.05 and above the seed floor, refuted below 0.02
    (were 0.10 and 0.05). H-d, median m in Flat, now one-sided: expected below
    0.80, refuted above 0.85 (was outside [0.80, 0.95] and inside [0.85,
    0.93]).
  - **Why each number.** A reading, labelled as such, written at line 126. H-a:
    0.01 is 0.59 of the largest posterior standard deviation the Identified bin
    allows (0.017) and 0.19 of prior A's; a faithful calibrator's d carries the
    reference's sampler error, about 0.0005 (D68), and training noise G3 holds
    to 0.002 (D67), so the expectation leaves five times that room; 0.03 is
    1.77 posterior standard deviations at the edge. The 0.01 is kept because
    G2's 0.001 (D66), G3's 0.002 (D67) and D68's sampler-error reading are
    built on it. H-b: through E = (d_B − d_A)/0.20, which the sheet's item 12
    derives from the draft's definitions and which is now written at line 109,
    0.05 and 0.15 are differences of 0.010 and 0.030 in H between the two
    networks' mean errors, H-a's two scales; on the Gaussian reading a network
    returning its prior mean scores E = c, 0.5 to 0.9 across the Partial bin
    (the sheet's arithmetic), so 0.15 is a modest leak, and a network within
    0.01 of the reference under both priors cannot exceed 0.10. H-c: moved from
    S_net to E for two reasons the sheet records: S_net is the one measure that
    does not compare with the reference, against line 113; and on the Gaussian
    reading the reference's own shift ratio is 1 − c, 0.10 at the Identified
    edge, so the old bet "S_net above 0.10" would have been met by Bayes' rule
    alone at that edge. With equal prior variances (D69) that reading holds for
    a Gaussian likelihood and Gaussian priors; D69 records that the skewed
    Betas will depart from it, by an amount that is a Stage 1 fact. The new
    numbers are H-a's scale in E units: 0.05 is 0.010 in H where the data carry
    at least nine times the prior's precision; 0.02 is 0.004 in H, the scale of
    the per-pair sampler error of the quantities in E, about what can be told
    from zero on a pair (a scale, not the error of a median); the seed floor
    stays as a second floor. The gap from 0.02 to 0.05 is narrow against the
    seed-noise scale of 0.0095, so Inconclusive is a likely outcome, which line
    126 says. The direction of H-b and H-c now has its mechanism written at
    line 126 as a reading: the test θ are uniform while each network trains
    under A or B (the sheet's arithmetic: about 46% of test H above A's 99th
    percentile, 32% below B's 1st); the sheet's reading of Schmitt, Bürkner,
    Köthe and Radev, that amortised posterior errors grow with the distance
    from the training simulations' typical set; and the identity, which makes E
    positive whenever each network is pulled towards its own prior mean, at low
    H through d_B and at high H through d_A. So H-c is in effect the bet that
    the B-trained network is not faithful at H-a's scale where H is identified,
    and H-b that the pull grows as the data weaken. v1's P3 bet the total shift
    ratio below 0.1 in the most identifiable quintile; v2 bets on an excess
    over Bayes' rule, a different quantity, which is why the number differs
    from v1's; why a leak is expected at all is the mechanism above. H-d: made
    one-sided towards overconfidence, the failure the sheet records Hermans and
    co-authors reporting for simulation-based inference algorithms, ratio
    estimation among them, and the one that matters to a user of the interval;
    the sheet's conversions, recomputed today, give m = 0.80 as a network
    interval 0.78 times the exact width and 0.85 as 0.875 times (Gaussian
    posterior, same centre), with a centre off by half a posterior standard
    deviation at the right width scoring 0.858; the old two-sided band let
    intervals too wide and too narrow both confirm. An interval wholly above
    0.95 is Refuted on this row, which bets on overconfidence only, so the null
    now also requires H-d's interval within [0.85, 0.95], and the share of
    pairs above 0.95 is reported. The ensemble's central Gaussian interval on a
    skewed posterior scores 0.9225 in the limit where the posterior is prior A
    (the sheet's limiting case, recomputed), so H-d is an easier Refuted for
    the ensemble than for the ratio estimator, whose equal-tailed interval has
    no such offset; the asymmetry is declared at line 126.
  - **Definitions added.** Line 107: d_P per training prior, Ĥ_exact,P the
    exact posterior mean, d alone d_A. Line 109: the A and B networks of the
    same seed paired, and the identity for E. Line 111: the seed floor as a
    number, per prior the median over the bin's pairs and the ten seed pairs of
    abs(S), the larger prior's value the floor, with the scale 0.954 τ/0.20,
    that is 4.77 τ, the sheet's illustration (0.0095 at τ = 0.002, G3's
    stability tolerance taken as a scale, since G3 bounds a median's shift
    under doubling and does not measure seed scatter; the median of abs(N(0,
    2τ²)) is 0.6745 × √2 × τ); the floor is a fixed number per bin, recomputed
    on each shifted bin of D70's re-scoring, and a seed pair with a failed
    network is left out for that pair. Line 126: a training-seed index is one
    unit across the three noise levels; the per-level breakdown carries no
    verdict; the eight verdicts are read individually at 95% without
    adjustment, said in the report; no verdict is pooled across hypotheses or
    calibrators, the null being read off the four verdicts and not a fifth;
    v1's sentences that a result inside the gap is reported as inconclusive,
    not stretched into a confirmation, and that the numbers are falsification
    thresholds, not targets, restored near-verbatim. Line 128: the null per
    calibrator is all four hypotheses Refuted with H-d's interval within [0.85,
    0.95], an Inconclusive or untestable outcome leaving it unproven; "within
    tolerance" removed. Line 132: the hierarchical form, "over training seeds
    and, within seed, pairs", which D68 and D70 refer to, is replaced by
    crossed resampling on the sheet's reading of Owen's pigeonhole bootstrap,
    since every seed is scored on the same pairs; the point estimate is the
    median over the bin's accepted (pair, seed) cells; each replicate resamples
    seed indices and pair indices independently, a cell entering with the
    product of its multiplicities; 2,000 replicates; percentile interval; the
    seed count and the percentile form stand, so D68 is not reopened; the
    five-seed narrowness is declared from the sheet's reading of Hesterberg's
    Table 5, which is for a mean of five values and not this statistic, and not
    corrected; the five per-seed medians are reported beside each interval; the
    reference's Monte Carlo error is not resampled. Line 101's "over pairs
    within seed" becomes "over pairs and seeds" to match. Line 134: the
    dose-response per calibrator, pooled, over the accepted pairs with the
    median over seeds of abs(d), its interval by the same bootstrap, expected
    negative, descriptive.
  - **What changes elsewhere.** Line 113, "every comparison is per pair against
    the reference", now holds for every row. Line 75's seed pairing and seed
    floor are used as written. The row that carries G2's and G3's tolerances
    (H-a) is unchanged, so D66, D67 and D68 stand. Changing the seed count or
    the interval type would reopen D68 and was not done; the bootstrap's
    resampling scheme is changed, as above.
  - **Not decided.** The number of test θ (line 94, set at G2 after freezing);
    the prior-work section, the author's; the OSF amendment.
  - **Draft.** Lines 101, 107, 109, 111, 117, 123, 124, 126, 128, 132 and 134
    as above; line 177 ticked. No line is added or removed; 180 lines.
  - **Counts.** 1 item remains OPEN; "OPEN" occurs 2 times. Ten checklist boxes
    are ticked; 2 are not.
  - What remains before freezing: the prior-work section, which is the author's
    to write from the papers; the OSF amendment; the number of test θ is set at
    G2 after freezing, as line 94 says.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D72** *(2026-10-11)* **P5: the emulator script for gate G1 committed (code
  and tests; nothing built; nothing frozen).** `p5_g1_emulator.py` and
  `test_p5_g1_emulator.py` implement line 54's emulator and line 142's gate as
  written, on the G0 scripts' primitives. The build is G1 and is made after
  freezing (lines 7 and 54); this entry records the script, which lines 7 and
  54 allow to be committed before its output. Nothing was priced at the
  declared settings: the tests run the pricer only at cheap settings on D38's
  point and on a small box around it, as G0's identity gate did, and compare
  nothing across parameter values.
  - **What the script does.** The literals of lines 39, 54, 141 and 142: the
    box (read back from `output/p5_g0_main_criterion.json`'s taken candidate in
    a test), nodes 16 × 8 × 6 × 12, `N_riccati` 8000 with `p5_g0_main.BASE_T`'s
    `U_max` and node counts per maturity, the held-out draw `default_rng(73)`
    of 200 points uniform on the box, the ratio 5 and the three noise levels.
    Chebyshev nodes of the first kind in ascending order, Berrut and
    Trefethen's barycentric weights, a vectorised evaluation of the
    four-dimensional tensor-product interpolant (`evaluate_many`, chunked, for
    G2's chains and the calibrators' training draws), the cosine transform to
    Chebyshev coefficients along each axis and the tail rule of line 142, read
    as the coefficients of the full tensor-product expansion: the largest
    absolute coefficient at a dimension's highest index over all quotes and all
    coefficient indices in the other three dimensions, the lowest index on a
    tie (line 142's "indices in the other three dimensions" could also be read
    as node indices of a one-axis transform; the two differ, and the full
    expansion is what is meant here). `price_triple` prices one (H, ν, ρ) at
    the written settings through the pilot's `solve_cf`, `make_cf`, `quote` and
    `price_setting` with `allow_anchor=True`, three Riccati solves per maturity
    shared across ξ₀ and strikes, the strikes anchored on each surface's own
    ATM quote; the 35 quotes are stored by maturity then z, ascending (line
    45). Tasks: 768 node triples with the 12 ξ₀ nodes and 200 held-out points,
    a resumable JSONL record in the G0 scripts' shape (header with git state,
    launches, tasks, errors), a worker pool of four, and the stop rule of line
    54: a task with any quote that is not finite is recorded with its stage and
    the launch stops, the pending tasks cancelled and those already running
    recorded as they finish; a later launch on that record refuses, so the stop
    survives a resume; the same rule is applied to a held-out point, which the
    draft does not say and this entry declares. Resume semantics, also
    declared: a launch appends to the record and skips labels already present;
    a task that raises is recorded as an error and tried again; a launch whose
    nodes, box, resolution, settings or stage differ from the record's header
    refuses. `--assemble` refuses while any task is missing or not finite, else
    writes the node values and nodes to `output/p5_g1_emulator.npz`, one row
    per held-out quote to `output/p5_g1_heldout.csv`, and
    `output/p5_g1_report.json` with e (numpy's percentile at 99, linear
    interpolation), the maximum, the per-maturity 99th percentiles, G1's three
    verdicts, the four tails and the refinement decision with its dimension.
    `--refine` runs only when the base report calls for it, doubles the named
    dimension and prices the refined triples to a second record; `--assemble
    --stage refined` writes the refined outputs, the held-out set reused from
    the base record (drawn once, the same points at the same settings; declared
    here); the refined report carries no refinement decision, there being no
    second one. `--estimate` prints D62's figure; without a flag the script
    prints the design, runs the pilot's identity gate at the scoping base
    (`N_riccati` 1000, `U_max` 200, 128 nodes, as G0's scripts did) and holds.
    `load_emulator` returns G̃ for the later scripts.
  - **The tests.** Nine, in `test_p5_g1_emulator.py`: the literals against the
    draft and D62's record; node, weight and exactness properties; a function
    entire in each variable recovered to 1e-8 on 9 × 7 × 5 × 8 nodes and a
    tensor-product polynomial of lower degree to rounding, a point outside the
    box refused; the coefficient transform on a known Chebyshev series and the
    tail rule on a hand-built array; the pricing path equal to the pilot's
    `run_task` quotes and anchored strikes, to `p5_g0_main.run_task_main`'s
    base rung and to the wrapper's `model_smile_cf_T`, bit for bit, at
    `N_riccati` 200 on D38's point (G0's identity gate), with the base settings
    equal to `settings_for(T)["A"]` at every maturity; the held-out draw equal
    to `default_rng(73).uniform` over the box; a quote that is not finite
    recorded as a status that stops the launch, refuses a resume and makes
    `assemble` refuse; a toy build on nodes 2 × 2 × 2 × 2 at `N_riccati` 200 on
    a small box around D38's point, with resume, assembly, exactness at a node,
    the held-out error computed with the record's box and the output schema;
    the G1 verdicts and the refined stage's absent decision on hand-built
    numbers. The whole suite collects 298 tests (CONTRIBUTING updated from
    289). One implementation detail: a point within 64·eps of a node in unit
    coordinates takes the node's value, since a node mapped to the box and back
    lands a few ulp off on a narrow box; the interpolant is unchanged beyond
    that scale. An independent review of the code found three faults, fixed
    before the commit: the held-out error was evaluated with the module's box
    rather than the record's; a resume after a recorded stop would have gone on
    pricing; and the `.npz` file handle was not closed.
  - **Cost.** D62's figure, not measured here: 968 sets of Riccati solves at
    about 288 core-seconds each, about 77 core-hours, about 19 h on four
    workers; a refinement repeats the training part at the doubled count, at
    most 1,536 triples, about 123 core-hours more, as D64 recorded.
  - **Not written.** The reference's script (G2), the calibrators (G3), any
    change to the draft. The build, its output and G1's verdict follow
    freezing; the commit that holds the output is to be named in the draft's
    declaration of G1 when it is made.
  - The prior-work section and the OSF amendment remain the author's before
    freezing; the number of test θ (line 94) is set at G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D73** *(2026-10-11)* **P5: the reference script for gate G2 committed (code
  and tests; nothing run on the emulator; nothing frozen).**
  `p5_g2_reference.py` and `test_p5_g2_reference.py` implement line 68's exact
  reference, line 143's gate and the references line 144 needs on G3's pilot
  surfaces, as written, on the emulator of `p5_g1_emulator.py`. Nothing was run
  on the real emulator, which does not exist until G1 is built after freezing;
  the tests use toy emulators in memory, linear maps whose posterior is a known
  Gaussian, and compare nothing across parameter values of the pricer.
  - **What the script does.** The literals of lines 62, 66, 68, 81, 93, 143 and
    144: the seeds 83 (one generator per pair, `default_rng([83, i, j])`), 79
    and 89 (G2's four pilot points and their noise), 97 and 101 (G3's sixteen),
    20261002 (the sealed θ draw); four chains; 10,000 burn-in and 40,000
    sampling steps; adaptation every 500; 2.38²/4, 1/12, the floor 1e-10; R-hat
    at most 1.01, the effective sample size of H at least 2,000, Kish's at
    least 1,000; the grids of 32 and 48 points, the 20% extension, the
    tolerances 0.001 and 0.01; the priors A = Beta(2.5, 9) and B = Beta(235/23,
    376/69) on 0.02 + 0.46x, and the analytic Var_prior(A) = 0.46² × 36/2645.
    The sampler: adaptive random-walk Metropolis in the box mapped to [0, 1]⁴,
    the four chains vectorised, a proposal with any coordinate outside the cube
    rejected without evaluation, the others evaluated in one call of the
    emulator. The diagnostics: the classical split R-hat of Gelman and others,
    √(var⁺/W) with var⁺ = (n − 1)/n·W + B/n on the chains split into halves,
    and the between-within effective sample size by the variogram estimator,
    ρ_t = 1 − V_t/(2 var⁺) summed to the first odd T at which ρ_{T+1} + ρ_{T+2}
    < 0 (the initial positive sequence), n_eff = mn/(1 + 2Σρ_t), the variogram
    computed exactly by FFT. `reference_pair`: the uniform run, its repeat with
    doubled lengths, Kish's effective sample size under A and B from the
    weights ∝ prior(H), the direct runs where it falls short, the weighted mean
    and variance of H, the equal-tailed 90% interval, the means of ξ₀, c
    against the analytic prior variance, the acceptance rate, R-hat and the
    effective sample sizes (of every parameter, H's being the one the
    acceptance uses), the support box of the pooled sampling samples, and the
    seconds. G2's grids: the cell-centred 32⁴ grid over the sampled support
    extended by 20% per side and clipped to the box, the prior-A-weighted
    likelihood summed over cells in chunks that never hold more than 8,192
    cells' quotes; at the largest noise level the 48³ grid in H, ν and ρ over
    the whole box with ξ₀ on the first grid's 32 points; each compared with the
    pair's figures under A at the written tolerance. Modes, mutually exclusive:
    `--pilot` (G2: the 12 references, a pool of four over pairs, the grid
    checks over the same pool, the report `output/p5_g2_report.json` with the
    gate's verdict and the cost per posterior and per grid), `--g3-pilot` (the
    48 references), `--test --n N --noise-seed S` (the sealed set, refused
    without both before anything is loaded), resumable JSONL records
    (`output/p5_g2_pilot.jsonl`, `output/p5_g3_pilot_refs.jsonl`,
    `output/p5_ref_test.jsonl`) with the stored samples beside them, one file
    per array written atomically as each pair finishes and a consolidated
    compressed archive written once at the end; `load_references` and
    `interval_prob` for the Stage 2 scripts; no flag prints the design and
    holds.
  - **Declared, beyond the draft's words.** The four starts are drawn as one
    block before any step; within a step the generator draws the proposal
    normals, then the acceptance uniforms, whether or not a proposal lies
    inside the box. The empirical covariance at an adaptation includes the
    chain's start; the floor of 1e-10 is applied to the diagonal after the
    2.38²/4 scaling; the adaptation at step 10,000 is the sampling proposal; a
    floored covariance that is still not positive definite (a chain that moved
    along a line) has 1e-10 × I added, tenfold at a time, until it is. "Fresh
    starts of the same generator" is read as new uniform starts from the
    continuing stream, not a re-seeding; the direct runs under A, then B,
    continue the same stream, each with its own single repeat with doubled
    lengths. The sampler and the diagnostics work in unit coordinates, to which
    R-hat and the effective sample size are invariant. The noise draws are
    `default_rng(seed).standard_normal((n, 3, 35))`, so y[i, j] = G̃(θ_i) + s_j
    ε[i, j]. A weighted quantile is the smallest sample value at which the
    cumulative normalised weight reaches q; intervals are closed. The H samples
    of each accepted run (the uniform run and any direct run) are stored as
    16-bit integers on the box's range of H, a resolution of 7e-6, so that the
    posterior probability under A of any interval can be computed later from
    the dequantised samples with the weights recomputed; the scalar summaries
    are computed on the exact samples before quantisation. The effective-sample
    truncation is the rule as Gelman and others state it, the first odd T with
    ρ_{T+1} + ρ_{T+2} < 0, which starts the test at the pair (ρ_2, ρ_3). The
    sealed test set is the first N rows of `default_rng(20261002)`'s uniform
    draw on the box, nested in N; the test set's noise seed is an argument with
    no default, recorded in the header, until line 94's amendment declares it
    with N; the three sets index the pair generator `default_rng([83, i, j])`
    by their own i, as line 68 says. One tolerance serves both grids. The
    support box is that of the accepted uniform run's pooled sampling samples.
    The whole-box grid's ξ₀ axis is the first grid's, so it checks for a mode
    missed in H, ν or ρ, as D66 reasoned. A reference failure among the 12
    pilot pairs fails G2, which line 143 leaves unsaid. The default emulator is
    the base stage's archive; a refined emulator is passed with `--emulator`,
    and the header records the emulator's stage, commit and node counts. The
    cost per posterior is wall time inside a worker with the pool running,
    reported with the pool size. Beyond line 68's list, the record also holds
    the variance and the interval under B, every attempt's diagnostics, the
    support box and Kish's sizes. Resume: a launch appends to the record and
    skips pairs already present whose samples are stored, doing again a pair
    recorded without them; a launch whose set, seeds, lengths or emulator
    (path, stage, commit, nodes) differ from the record's header refuses; the
    report refuses if a record's surface is not the one the current emulator
    gives; a pair that raises is recorded as an error and tried again; a
    reference failure is a recorded status, not an error.
  - **The tests.** Eight, in `test_p5_g2_reference.py`: the literals, with the
    two priors' variances equal to 36/2645 and the seeds distinct from those
    already used; R-hat and the effective sample size on synthetic chains
    (independent normals, an AR(1) process with the known ratio (1 − φ)/(1 +
    φ), chains shifted apart, the variogram on a hand-computed case); the
    weighted mean, variance, quantiles, Kish's size, the interval probability,
    the quantisation, the contraction and the prior densities on hand-built
    numbers; the sampler on a linear emulator with short chains, each
    parameter's mean within four Monte Carlo standard errors of the Gaussian
    posterior's, the acceptance rate between 0.1 and 0.6, determinism under the
    same generator, the out-of-box rejection and the adapted covariance; the
    repeat with doubled lengths and the reference failure, with R-hat forced to
    1.5; the reweighting to A against 400,000 exact Gaussian draws reweighted
    the same way, the direct-run path forced by an unreachable Kish floor, the
    grid check on the sampled support against the same yardstick, the
    extension's clipping and the whole-box grid's shape; the draws and the
    sealed set's nesting, the test mode's refusal without its arguments, the
    refusal without an emulator, the no-flag hold; the record, resume, header
    check and report on a toy pilot of two surfaces at one level with pool 1, a
    pair recorded without its samples done again, a surface from another
    emulator refused. The pool path loads the emulator from a file per worker
    and cannot be exercised with an in-memory toy; the tests say so. The whole
    suite collects 306 tests (CONTRIBUTING updated from 298). An independent
    review of the code found four faults, fixed before the commit: the samples
    archive was rewritten whole after every pair, not atomically, and after the
    record line, so a crash could leave a pair recorded without its samples and
    the rewrite would have cost hours on the sealed set; a resume did not check
    the emulator's identity; and the quantile helper could index past the end
    when the cumulative weight fell short of one by rounding.
  - **Cost.** D66's arithmetic, not measured here: 200,000 emulator evaluations
    per posterior, seconds to a minute; 1,048,576 cells per grid check and
    3,538,944 for the whole-box grid, minutes each. G2 measures the cost per
    posterior, which line 94's amendment then uses.
  - **Not written.** The calibrators (G3), the draft, any run on the emulator.
    G2's run, its output and its verdict follow the build of G1, after
    freezing; the commit that holds the output is to be named in the draft's
    declaration when it is made.
  - The calibrators' script (G3) is owed next among the scripts; the prior-work
    section and the OSF amendment remain the author's before freezing.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D74** *(2026-10-11)* **P5: the calibrators' script for gate G3 committed
  (code and tests in the torch leaf; nothing trained on the emulator; nothing
  frozen).** `p5_g3_calibrators.py`, `test_p5_g3_calibrators.py` and
  `requirements-p5.txt` implement lines 72, 73 and 75 (the two calibrators,
  their architecture and training, D68) and line 144 (G3, D67) as written, on
  the emulator of `p5_g1_emulator.py` and the references of
  `p5_g2_reference.py`. The leaf is isolated as line 154 requires: torch is a
  guarded import, the core stays torch-free, the venv `.venv-p5` is recreated
  from the requirements file and ignores itself. Nothing was trained on the
  real emulator, which does not exist until G1 is built after freezing; the
  tests use toy emulators in memory and short runs.
  - **What the script does.** The literals of line 75: the trunk of three
    hidden layers of 128 ReLU units with PyTorch's default initialisation; five
    members; Adam at 0.001 with β (0.9, 0.999) and no weight decay; batch 256
    with a partial final batch kept; patience 50 and the cap of 1,000 epochs;
    the variance floor 1e-6; 4,000 validation surfaces; the seeds 109, 113 and
    107 and the torch seed 109,000,000 + 10,000 p + 1,000 j + 10 q + m; and of
    line 144: the ladder 10,000 × 2^k from one draw of 320,000, the slope 0.95,
    the stability tolerance 0.002, the budgets of 320,000 surfaces and 4 h of
    wall. Draws: one generator per (prior, level, seed), `default_rng([109, p,
    j, q])`, drawing a 320,000 block in the order H (the prior's Beta, scaled
    to the box), then (ν, ρ, ξ₀) uniform, then ε; a run at n takes the first n
    rows, so the ladder's sets are nested exactly and every seed's set at n is
    the head of its block; surfaces y = G̃(θ) + s ε from the emulator in
    chunks; the validation block from `default_rng([113, p, j])` the same way.
    Inputs standardised by the per-quote mean and standard deviation over the
    run's n surfaces (standard deviation floored at 1e-12), stored with the
    run; θ mapped to [0, 1]⁴. The ensemble member: four outputs, the Gaussian
    negative log-likelihood summed over H and ξ₀ in mapped coordinates and
    averaged over the batch, the variance softplus(raw) + 1e-6; Ĥ and the ξ₀
    estimate the means over members mapped back; the two variance components of
    H in box units (the mapped variance times 0.46²) reported separately; the
    interval Ĥ ± 1.645 times the square root of their sum; nothing clipped. The
    ratio estimator: 39 inputs, one logit; each batch's B joint pairs against
    the B marginal pairs made by the cyclic shift of θ by one within the batch;
    binary cross-entropy over the 2B; its posterior by the reference's own
    sampler (the same chains, lengths, adaptation, acceptance and single
    repeat) with the target log P(H) + logit(θ, y), the generator
    `default_rng([107, p, q, i, j])`; Ĥ and ξ₀ the pooled means, the
    equal-tailed 90% interval, the diagnostics, or a failure after the repeat.
    Training: `torch.manual_seed` before construction and before the shuffles;
    epochs of shuffled batches; the validation objective after every epoch in
    drawn order in batches of 256 with the final 160 kept and the marginal
    pairs formed as in training; stop after 50 epochs without improvement or at
    1,000; the best state kept; per member. `--g3`: per calibrator, prior and
    level, the pilot surfaces at that level whose reference was accepted (a
    failed reference removes its surface, recorded), the runs at n_k and 2n_k
    (the run at 2n_k is the next rung's run except at the top, where 320,000 is
    the check run only), the two checks, the budgets, the smallest passing n,
    none passing reported as P5 stopping; every run's networks and standardiser
    saved under `output/p5_g3/`, every run and rung recorded in
    `output/p5_g3_ladder.jsonl`, the verdict in `output/p5_g3_report.json`;
    resumable, a run in the record with its networks saved being reused.
    `--train`: seeds q = 2 to 5 at n* per cell, q = 1's run reused; a
    non-finite loss retrained once from seed value 6, which takes the failed
    seed's place. `--score SET`: every calibrator at every (prior, level, seed)
    on a set's surfaces at its level, per pair the outputs of line 75, to
    `output/p5_scores_<set>.jsonl`; Stage 2's measures, bins and verdicts are
    the next script's work. No flag prints the design and holds; the modes
    refuse without torch, the emulator or the references.
  - **Declared, beyond the draft's words.** Indices: p is 1 for A and 2 for B
    and j is 1, 2, 3 for the levels, as line 75 states, in every generator and
    torch seed of this script; i counts a set's θ from 0 as the reference's
    records do; the reference's own records carry j from 0 (D73), and the two
    are reconciled by the level's value, not its index. "Seeds 2 to 5 draw the
    first n only" is read as every seed drawing the whole 320,000 block and a
    run taking its head, since the block draws all H before any ν and a direct
    draw of n would differ from the head in (ν, ρ, ξ₀) and ε; the head reading
    is the one consistent with q = 1 and with line 144's nesting. The wall
    budget is a deadline for the whole calibrator's run, started before the
    emulator prices the training and validation surfaces, checked after each
    epoch's validation, so a run can overshoot by one epoch; a member past it
    stops with the status over_budget, the remaining members are not trained,
    and the rung fails, as it does on a non-finite loss; the ladder goes on to
    the next rung, whose run at n already exists as this rung's check run.
    After G3 the wall budget is not applied, line 144's budget being the
    gate's, and the retraining from seed value 6 follows a non-finite loss
    only, as line 75 says; a second failure in a cell leaves its slot empty and
    is recorded. The member's torch seed does not depend on n, so at every rung
    a member starts from the same initialisation and the rungs differ by their
    data. The thread count is an argument, default 4, recorded in every record;
    the networks run in float32 and the sampler in float64; Adam's other
    settings are PyTorch's defaults; the variance over members is the
    population variance; the standardiser's standard deviation is floored at
    1e-12; bitwise reproduction across machines is not claimed (line 75). The
    ratio estimator's posterior passes the sampler a callable target that
    already holds log P(H); its R-hat and effective-sample thresholds are the
    reference's; at G3 a pair on which it failed is excluded from both checks
    at the run where it failed, the survivors at n and 2n possibly differing,
    both counts recorded. The ensemble's ξ₀ estimate is the mean of the
    members' ξ₀ means. The validation objective is the surface-weighted mean
    over its batches. On resume the ladder's checks are recomputed from the
    saved networks. Each run is saved as its state dictionaries and a JSON of
    its metadata, including the standardiser, the members' epochs, best
    validation losses, seconds and seeds; a scored pair carries the seed's slot
    as q and the run's label, which names seed 6 where it replaced a slot.
    `requirements-p5.txt` pins torch and numpy at the versions the scripts were
    committed with, as line 75 requires. `p5_g2_reference.py` gained one
    backward-compatible parameter, a callable target on its sampler functions,
    so that the ratio estimator's posterior uses the same code; its eight tests
    are unchanged and pass. Line 164's commitment to commit the trained
    networks is met by the saved files, to be force-added by name after the
    runs, as the G0 outputs were.
  - **The tests.** Ten, in `test_p5_g3_calibrators.py`: five run in the core,
    torch-free environment (the literals and the torch seeds' uniqueness; the
    draw's order, nesting and surfaces, the standardiser and the batches; G3's
    checks and the budgets on hand-built numbers; the pilot surfaces read from
    reference records with a failure removed; the hold and the refusals) and
    five need torch and skip without it (the models' shapes and parameter
    counts 38,148 and 38,273, the member loss and the batch construction on
    hand-built numbers; the training loop on a toy linear emulator, learning,
    determinism under the same seed, the over_budget status and the nonfinite
    status on both the training and the validation path; the ensemble's outputs
    recomputed by hand in box units, the overflow-safe variance, saving and
    loading, and the ratio estimator's posterior through the reference's
    sampler with a hand-built quadratic logit whose posterior under A is known,
    the generator's arguments pinned; a toy G3 ladder end to end with toy
    references and resume; after a forced pass, the seeds at n*, the one
    retraining from seed 6 with a second failure leaving its slot empty, and
    the scoring of a set with resume). In `.venv-p5` all ten pass; in the core
    interpreter five pass and five skip. The whole suite collects 316 tests
    (CONTRIBUTING updated from 306, with a sentence on the leaf's environment).
    An independent review of the code found six faults, fixed before the
    commit: a non-finite validation loss would have crashed the record instead
    of recording the declared failure; the prediction-time softplus overflowed
    in float32 where training's did not; a second failed seed in a cell would
    have received the same seed-6 network; the end-to-end test never reached
    the seeds and the scoring; the requirements file did not pin; and a
    `--pool` option did nothing while the scoring's thread count was neither
    set nor recorded.
  - **Cost.** Arithmetic, not a measurement: an epoch at 320,000 surfaces is
    1,250 steps of batch 256; D68's estimate of a millisecond or two per step
    puts 1,000 epochs at about 35 minutes per member, so a run at the top of
    the ladder is within the 4 h budget line 144 sets for the whole calibrator
    and the ladder is bounded by roughly twice the top per cell; early stopping
    usually cuts it; the ratio estimator's posterior is 200,000 network
    evaluations per pair, seconds. G3 measures all of it.
  - **Not written.** Stage 2's analysis (d, S, E, m, the bins, the floor, the
    verdicts, the bootstrap); the draft; any run on the emulator. G3's run, its
    output and its verdict follow G1 and G2 after freezing.
  - The Stage 2 analysis script is owed next among the scripts; the prior-work
    section and the OSF amendment remain the author's before freezing.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D75** *(2026-10-11)* **P5: the Stage 1 and Stage 2 analysis script
  committed (code and tests; nothing run on real outputs; nothing frozen).**
  `p5_stage2.py` and `test_p5_stage2.py` implement lines 79 to 89 (the bins and
  their conventions, D70), 97 to 101 (Stage 1 and the floor), 105 to 113 (the
  measures and the seed floor), 115 to 128 (the hypotheses, their conventions
  and the null, D71) and 130 to 134 (the bootstrap, the verdict rule, the
  dose-response) as written, on the records of `p5_g2_reference.py` and the
  scores of `p5_g3_calibrators.py`. The analysis is committed before any result
  exists, which is what a pre-registration asks; the tests use synthetic
  records in the scripts' formats.
  - **What the script does.** Stage 1: each accepted pair binned by c under A
    at the cut-offs 0.9 and 0.5 by the table's inequalities, a negative c under
    Flat and counted, the pairs within 0.02 of a cut-off counted, the shares
    pooled and per level, the distinct θ per bin, the floor of 10% of the
    accepted pairs pooled or 100 making a bin untestable for both calibrators,
    a reference failure removing its pair and counted; `output/p5_stage1.csv`
    with one row per pair and `output/p5_stage1.png` (the distribution of c per
    level and c against true H and ν) when matplotlib is present. Stage 2: per
    calibrator, seed slot q and pair, d_A, d_B, S_net from the A- and B-trained
    networks of the same slot, S_exact, E, and m from the stored H samples
    through the reference script's `weights_for` and `interval_prob`; line 75's
    exclusions (a failed network's pair excluded at that seed from every
    measure that uses it, S and E included, the other prior's measures kept);
    the per-calibrator failure floor (failed pairs above 10% of the bin's pairs
    or leaving fewer than 100); the seed floor per prior and bin as the median
    over the bin's pairs and the unordered seed pairs of abs(S), the larger
    prior's value, a pair with a failed network left out;
    `output/p5_stage2_cells.csv` with one row per cell. The statistic is the
    median over the bin's (pair, seed) cells; the crossed bootstrap resamples
    the seed indices (as many as the calibrator has) and the bin's pair indices
    independently with replacement, a cell entering with the product of its
    multiplicities, and takes the weighted median; 2,000 replicates; the 2.5
    and 97.5 percentiles; the per-seed medians beside each interval. The
    verdicts by line 133 on each one-sided row, H-c's interval also wholly
    above the seed floor to be Confirmed, H-d's over-wide share reported;
    Untestable where the bin is under Stage 1's floor or the calibrator's
    failure floor; the null per calibrator (all four Refuted and H-d's interval
    within [0.85, 0.95], else unproven); the per-level breakdown without
    verdicts; the re-scoring at the cut-offs moved by +0.05 and by −0.05 with
    the floors and the seed floor recomputed on the shifted bins, reported and
    deciding nothing; the multiplicity sentence in the report; the
    dose-response per calibrator, pooled, Spearman's correlation of c with the
    median over seeds of abs(d_A), its interval by the same bootstrap,
    descriptive. Everything is written to `output/p5_stage2_report.json` with
    the inputs' headers and the git state; no flag prints the design and holds;
    `--run` refuses without the three input files.
  - **Declared, beyond the draft's words.** The bootstrap's generators are
    `default_rng([127, calibrator index, row index])`, one stream per
    (calibrator, hypothesis) so that each row is self-contained, the
    dose-response using row index 9; 127 is a seed the draft does not name and
    none already used; the draws are, per replicate, the pair indices then the
    seed indices. The point estimate, the per-seed medians and every replicate
    use the same median: the smallest value whose cumulative weight reaches
    half the total (the lower weighted median; with unit weights and an even
    count the lower of the two middles). The bootstrap runs on the live cells:
    a pair with no accepted cell and a seed column with none are not resampled
    (line 132's "the four a cell has been left with"), the live counts and any
    replicate without a live cell being reported. The percentile interval is
    the 2.5th and 97.5th percentiles of the replicates by numpy's default
    interpolation. A pair's samples are read from the store under the keys the
    reference script wrote, `<pair>_u`, `_A`, `_B`, mapped to the names
    `weights_for` expects, and m is computed on the dequantised samples with
    the weights recomputed, as D73 declared; m is the mass of the A-trained
    network's interval, line 110 naming no prior. A seed slot with no score
    rows at all for a (calibrator, prior, level) is vacant, line 75's cell left
    with four seeds: its cells are excluded and not counted as failures; a row
    missing while its slot has rows at that level is an incomplete scoring and
    refused, as are colliding task labels. A calibrator's failed pair, for line
    75's floor, is a pair on which any of its networks failed at any seed; the
    per-cell count is reported beside it, and both counts sit on every
    hypothesis row. Line 126's share of pairs with m above 0.95 is the share of
    pairs whose median m over seeds exceeds 0.95, the share of cells beside it.
    The near-cut-off counts of line 89 are split by the side of the cut-off,
    which is the bin. The re-scoring block holds the shifted bins' intervals,
    floors and seed floors beside the verdicts, as lines 89 and 111 ask, the
    shifted cut-offs rounded so that no verdict rests on floating-point noise
    at a cut-off. In the dose-response a pair with no accepted seed is left out
    and counted, the per-pair median is the lower median over seeds, the
    correlation in a replicate is computed on the pairs expanded by their
    multiplicities (ties from the repeats taking average ranks), and a
    replicate without variation or with fewer than three pairs is dropped and
    counted. The null's "H-d's interval within [0.85, 0.95]" is read with
    closed ends. Levels are matched by their index, never by a float. A seed
    slot is one unit across levels and priors, so a seed-6 replacement sits in
    the slot it replaced, per (prior, level) cell, as line 75 arranges. The
    inputs are checked: an accepted reference without its stored samples is
    refused, a truncated score line is skipped as the producer's reader skips
    it while two well-formed rows under one label are refused, references and
    scores of different sets are refused, and the number of pairs the record
    should hold (three per test θ) is reported beside the number found. The
    prior-A weights are computed once per pair and the dose-response's per-pair
    medians are vectorised, so a run at N = 1,000 test θ takes minutes; the
    store is read into memory, about 1 GB at 3,000 pairs. Line 135's
    exploratory items are not implemented here; they belong to a later,
    declared script. The figure is descriptive and decides nothing.
  - **The tests.** Eight, in `test_p5_stage2.py`: the literals against the
    draft and the seed's novelty; the binning at the written and the shifted
    cut-offs, the window counts, negative c, the floor by share and by count
    (the count governing below N = 334), the distinct θ; the measures on
    hand-built numbers with line 109's identity, m against a direct weighted
    count on a toy sample store, line 75's exclusions on a pair with a failed
    network, the failure floor; the seed floor on hand-built estimates; the
    weighted median, the crossed bootstrap (a constant statistic, a dead seed
    column and a dead pair left out, determinism under the seed, the
    multiplicities, the point and the replicates on one median convention) and
    the per-seed medians; the verdict rule on every row including H-c's floor
    clause, H-d's one-sided form, the over-wide interval and the ends of the
    gap; the vacant slot and the incomplete scoring; an end-to-end run on
    synthetic reference and score files in the scripts' formats, a
    prior-dependent bias planted off the Identified bin and over-narrow
    intervals in the Flat bin, with one reference failure and one network
    failure, checking the counts, the four verdicts in the planted directions
    (H-a Confirmed, H-b Confirmed, H-c Refuted, H-d Confirmed), the re-scoring
    through the binning, the dose-response's sign, the CSVs, the figure,
    determinism and the hold and refusal paths; and a thin set on which every
    bin is under the floor and every row is Untestable. The whole suite
    collects 324 tests (CONTRIBUTING updated from 316). Two independent reviews
    of the code, one on the statistics against lines 79 to 135 and one on the
    software against the producing scripts' formats, found faults fixed before
    the commit: a vacant seed slot was read as a failure of every pair and
    tripped line 75's floor; the point estimate and the replicates used
    different median conventions; a dead seed column was still resampled; the
    over-wide share was over cells, not pairs; the re-scoring wrote verdicts
    without the recomputed floors; the near-cut-off counts were not per bin;
    the shifted cut-offs carried floating-point noise; colliding score labels
    would have dropped rows silently; the prior-A weights were recomputed for
    every cell and the dose-response looped over pairs in Python, about an hour
    at the real sizes between them; a missing sample surfaced as a bare error;
    and the collision guard conflicted with the producer's tolerance for a
    truncated line.
  - **Not written.** Line 135's exploratory analyses; any run on real outputs,
    which follow the sealed set's references and scoring after freezing; the
    draft.
  - Every script the draft needs now exists: G1's build, G2's reference, G3's
    calibrators and the analysis. What remains before freezing is the author's:
    the prior-work section (line 158) and the OSF amendment; line 94's
    amendment follows G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D76** *(2026-10-11)* **P5: the prior-work section drafted from sources read
  at their addresses (docs only; nothing frozen; the checklist box left for the
  author).** Line 158 of `docs/protocols/P5_protocol_v2_draft.md`, the
  placeholder under Relation to prior work, is replaced by the section: deep
  calibration of rough volatility (v1's papers re-read), the pricer's
  foundations and the correlation range, identifiability of H from a surface,
  Bayesian prior-sensitivity analysis, unfaithful and misspecified amortised
  posteriors, the novelty claim limited to (i) and (ii) as the placeholder
  fixed it, and a list of the sources read with how each was read. Nothing else
  in the draft changes; the box "Prior-work section written from the papers
  themselves" (line 179 before this change, 214 after it) stays unticked.
  - **What the section says.** Horvath, Muguruza and Tomas learn the pricing
    map and calibrate through it with an optimiser, attaching no uncertainty,
    and Bayer and Stemper, read as an abstract, pair an optimiser with a
    network on the implied-volatility map; Bayer, Horvath, Muguruza, Stemper
    and Tomas add an MCMC analysis through the surrogate for rough Bergomi, no
    prior distribution being given in the text and no second prior, Fisher
    information or Cramér–Rao bound appearing; Brigo, Huser and Leonte learn
    the rough Heston posterior by telescoping ratio estimation under a uniform
    prior on a box (H in (0.05, 0.45), ρ in (−0.95, −0.10)) from a 63-quote
    surface, attribute each parameter's contraction to cells of the surface,
    describe reweighting to a second prior at inference time without an
    experiment found applying it, check coverage by simulation from the
    training prior and use no exact reference. El Euch and Rosenbaum's Theorem
    4.1 is stated for ρ in (−1/√2, 1/√2] while the box runs to −0.90; Abi
    Jaber, Larsson and Pulido's Theorem 7.1 covers ρ in [−1, 1] and is stated
    to generalise some of El Euch and Rosenbaum's main results; labelled as a
    reading of the theorem statements. Hager and Kreher's analyticity of the
    Riccati solution in H is recorded as a statement about the map, not its
    inverse. Gelman and co-authors' Section 6.3 lists the routes to prior
    sensitivity without a spacing rule; Kallioinen and co-authors'
    power-scaling is a local perturbation with a 0.05 threshold; Elsemüller and
    co-authors amortise over a family of power-scaled priors without a
    reference posterior. Hermans and co-authors' overconfidence and the
    prior-returning estimator's nominal coverage, Delaunoy and co-authors'
    balancing and its collapse to the prior, Lueckmann and co-authors'
    reference posteriors by rejection sampling and classifier tests, Talts and
    co-authors' within-prior check, and the misspecification strand (Schmitt
    and co-authors, a shifted prior among the cases detected; Cannon, Ward and
    Schmon; Ward and co-authors) are set out with what each tests, and P5's
    difference stated: the forward model well specified and shared by the
    calibrators and the reference, the prior shifted by a stated amount, the
    measure an exact posterior per surface, stratified by contraction.
    Lakshminarayanan, Pritzel and Blundell is the ensemble's source, a
    non-Bayesian alternative. The claim closes on (i) and (ii) and nothing
    else.
  - **Read at source.** Eighteen sources at their arXiv addresses on 11 October
    2026, listed in the section with what was read of each: the abstract page
    for every one (identifier, version, date, any journal reference or
    comment); the HTML text for Horvath, Muguruza and Tomas (to Section 4.2),
    Bayer, Horvath, Muguruza, Stemper and Tomas (whole; the figures are
    images), Brigo, Huser and Leonte (whole), Gelman and co-authors (Sections
    5.9 and 6.3), Elsemüller and co-authors (to Section 5), Hermans and
    co-authors, Delaunoy and co-authors, Lueckmann and co-authors, Talts and
    co-authors, Schmitt and co-authors (to Section 4), Cannon, Ward and Schmon,
    Ward and co-authors and Lakshminarayanan, Pritzel and Blundell (whole); the
    PDF text, extracted locally, for El Euch and Rosenbaum (Theorem 4.1), Abi
    Jaber, Larsson and Pulido (Section 7) and Kallioinen and co-authors (whole,
    the HTML rendering being absent); the abstract alone for Bayer and Stemper
    and Hager and Kreher. Four findings against v1 are recorded in the section
    rather than silently carried, each labelled as a reading: Bayer and
    co-authors' text gives no prior distribution, v1's "uniform prior on a
    truncated box" being a reading of the training ranges, and the values of H
    v1 reads from Figures 6 and 7 are not stated in the text and were not
    checked, the figures being images; the same paper defines a Jacobian of the
    pricing map for its optimiser (Section 2), so v1's and D54's "none computes
    a Jacobian" holds only as none computing one to ask what the surface
    identifies; Horvath, Muguruza and Tomas report error in implied-volatility
    space in the sections read, not v1's "price-space error"; and the arXiv
    page of Bayer and co-authors shows no journal reference, so v1's "published
    2025" is not re-verified. Hernandez (2016) is on SSRN, which refused the
    request; it stays on v1's record and is marked not re-read. The identifier
    first tried for Ward and co-authors resolved to an unrelated paper and was
    corrected from the arXiv listing. The verification log with every
    statement's pinpoint was kept outside the repository.
  - **Declared, beyond the draft's words.** The section is drafted by Claude
    from the sources, not from the author's own reading, which the placeholder
    named as its condition; the draft's Reporting line already discloses AI
    drafting with the commitments the author's on freezing, and the box (line
    214 after this change) is left for the author to tick after reading the
    listed sources. The section's first paragraph says both. A sources list is
    added inside the section, in the decision sheet's format, because the draft
    has no reference list and the sheet is not modified; the sheet's Sources
    remain the record for the other sections. Two sources beyond the sheet's
    are added (Cannon, Ward and Schmon; Ward and co-authors), both read whole
    and both on the misspecification topic the placeholder required. The
    correlation-range note and the reweighting note the sheet left for this
    section are carried in. Dates of reading are given so that a later version
    of any source is distinguishable from what was read.
  - **Not written.** The tick at line 214; the OSF amendment; any change to v1,
    the decision sheet or the rest of the draft; any run.
  - What remains before freezing: the author's reading of the sources and the
    tick at line 214; the OSF amendment; line 94's amendment after G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D77** *(2026-10-11)* **P5: the OSF amendment specified: what OSF allows,
  the design (a new registration of the frozen v2 file and an update to APSF7
  pointing to it), the statement to paste, and the order of acts at freeze
  (docs only; nothing frozen; nothing on OSF touched).** The draft's legend
  (line 11) says the record is the OSF registration amended with this version,
  and D55 left the amendment as the author's act at freeze; nothing in the
  repository said what it consists of. `docs/protocols/P5_v2_amendment.md` now
  does, as a working aid that decides nothing. The draft is not changed by one
  line.
  - **What was verified at OSF, read-only, on 11 October 2026.** Registration
    `apsf7` (DOI 10.17605/OSF.IO/APSF7) is an Open-Ended Registration of the
    project `gdsv7`, registered 2 October 2026 at 19:08 UTC, public, review
    state accepted, unmodified since registration (API
    `/v2/registrations/apsf7/`), with one schema response, the original,
    approved; the registration's `revision_state` field read `approved`, then
    `in_progress` while a draft update the author had started existed, then
    `approved` again after the author deleted that draft the same day, nothing
    having been submitted. Its responses are a summary, naming the attached
    protocol as the authoritative document and citing commit `13bca7e`, and one
    uploaded file, `P5_protocol.md`, with recorded SHA-256
    5b7a67e5ca83b62308907b699f0587fe1febc4894a8335ce3b8acfe3e7890132 (API
    `/v2/schema_responses/6ac001210231cc8223157561/`). The git blob of
    `docs/protocols/P5_protocol.md` at `13bca7e` and at `main` hashes to that
    value exactly; the working copy differs only by the CRLF endings
    `core.autocrlf` applies on checkout. The commit (18:22 UTC) preceded the
    registration by 45 minutes. The Programme row's sentence that the
    registered file is byte-identical to the committed protocol therefore
    holds, of the blob. From OSF Support's "Welcome to Registrations &
    Preregistrations!" (the update sections and the FAQ, paraphrased): an admin
    contributor can start an update from the registration's overview, the
    fields prefilled, at least one changed, a justification required, every
    admin having 48 hours to approve or cancel before automatic approval; files
    cannot be added or removed in an update and documents cannot be uploaded or
    modified after submission; a document may be linked from a public project,
    a newly submitted public registration or an outside repository; updates are
    described as for events outside the researcher's control or unexpected
    anomalies; a withdrawn registration cannot be reinstated. The
    registration's own page is a browser application and showed nothing to a
    plain fetch; the API was read instead.
  - **The design, and why.** "Amended with this version" cannot be done by
    attaching v2 to `apsf7`. The frozen v2 file is to be registered anew, as an
    Open-Ended Registration from the same project with the amendment statement
    as its summary, so that a copy sits on an independent server with a
    timestamp and a hash as v1's does; and `apsf7` is to receive an update
    whose summary gains one paragraph pointing to the new registration, the
    same paragraph as its justification, so that a reader of v1's record finds
    v2 and so that v1 stays the record of what was registered on 2 October
    2026, not withdrawn. An update alone would leave v2 with no registered
    copy; a new registration alone would leave `apsf7` silent. The file
    registered is the git blob of the freeze commit (LF endings), exported with
    `git show`, so that OSF's recorded hash can be checked against the blob as
    v1's was. The file keeps its name, `P5_protocol_v2_draft.md`, which sixteen
    decision entries cite by name and four more by line; the Status line, not
    the name, is to say it is frozen. These are proposals in a working aid; the
    acts are the author's and none has been made.
  - **What the file holds.** The facts above with their addresses; the design;
    the amendment statement in the Reporting line's three heads, ready to
    paste: what changed from v1 as thirteen items, each naming the decision
    entries that fixed it (the question, D55; the frozen emulator, D55 and D64;
    the observable, grid, box and resolution from G0, D58, D59, D61 and D62;
    three noise levels, D65; the Beta parameters, D69; the exact reference and
    G2, D66; the calibrators and G3, D67 and D68; the contraction cut-offs,
    D70; the four hypotheses and the null, D71; the crossed bootstrap, the
    re-scoring and the analysis script, D71, D70 and D75, the verdict rule
    being kept from v1; three v1 clauses dropped without counterpart, the
    control at home, the optional cleaner noise level and the reason for the
    0.20 gap, from the decision sheet's list; the prior-work section, D76; the
    scripts committed before freezing, D59 to D62 and D72 to D75), why (D55's
    reason), and what had been seen (the 3 October probe, D55; G0's scoping and
    main runs, D60 to D62; nothing else, to be re-verified at the freeze commit
    by a search of the tracked files and the output directory); a proposed
    title for the new registration; the paragraph for the update to `apsf7`;
    the order of acts at freeze in five steps, each marked as the author's or
    the repository's (the tick at line 214 after the author's reading; the
    freeze commit changing only the Status line, the legend and the OSF box's
    words; the new registration; the update; the post-freeze commit that adds
    the v2 DOI to line 7, ticks the OSF box and records the hashes); an empty
    record table; and what is not Claude's to do on OSF. Placeholders in angle
    brackets stand for the freeze commit, the hash, the date, the registration
    id and the DOI.
  - **Declared.** The order mirrors v1's (commit first, OSF second), so that
    the registered file is the committed blob. The OSF box is left unticked at
    the freeze commit and ticked only after the acts, because the file
    registered must be the frozen one and a tick before the act would assert
    what had not happened; the post-freeze commit is a dated amendment changing
    two lines and saying so. The thirteen items of what changed are drawn from
    the decision sheet's "Differences from v1", from D55 and from the entries
    named; of the sheet's five clauses said to have no counterpart in v2, two
    are stale, the draft now carrying "no code may be tuned to reach them"
    (line 126, D71) and a Null paragraph (line 128), and the other three are
    listed as dropped. The working aid carries no AI-assistance line in its
    Status, as the decision sheet does not; the line inside the statement is
    for OSF, where v1's registration carries one. OSF's own pages are
    paraphrased, not quoted. The registration's `revision_state` values are
    reported as read, the draft update's existence and deletion on the author's
    word.
  - **Not written.** Any line of the draft; v1; the decision sheet; anything on
    OSF; the freeze; the tick at line 214.
  - What remains before freezing: the author's reading of the sources and the
    tick at line 214; then the order of acts in the new file, the OSF acts
    being the author's; line 94's amendment after G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D78** *(2026-10-11)* **P5 v2 draft: the prior-work box ticked on the
  author's confirmation that the eighteen sources were read (one checklist box
  ticks; docs only; nothing frozen).** Line 214 of
  `docs/protocols/P5_protocol_v2_draft.md`, "Prior-work section written from
  the papers themselves", is ticked. The author confirmed in the sitting of 11
  October 2026 that all eighteen sources listed under Relation to prior work
  had been read; the section (D76) was drafted from those sources and left the
  box for that confirmation. No other line of the draft changes; the one OPEN
  item (line 94) and the box "OSF registration amended" (line 215) stand. This
  is step 1 of the order of acts in `docs/protocols/P5_v2_amendment.md` (D77).
  - **Not written.** The freeze commit; anything on OSF; any change to the
    section's text, to v1 or to the decision sheet.
  - What remains before freezing: the freeze commit on the author's word, then
    the OSF acts and the post-freeze commit in the order D77's file sets out;
    line 94's amendment after G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D79** *(2026-10-12)* **P5 v2 draft: D78's confirmation withdrawn and the
  prior-work box unticked; the AI-use line redrafted as a declaration (docs
  only; nothing frozen; nothing on OSF).** Two corrections on the author's
  instruction before any freeze. First, D78 over-claimed: the author had read
  the abstracts of most of the eighteen sources, not their full texts, and the
  confirmation it recorded is withdrawn; line 214 of
  `docs/protocols/P5_protocol_v2_draft.md` returns to unticked, and the author
  reads the sources in full before confirming again. D78 stays in the log as
  written, the log being append-only; this entry is its correction. Second,
  line 200, the one-line AI disclosure under Reporting and disclosure, is
  redrafted as a declaration of generative-AI use in line with the University
  of Oxford's Policy for using generative AI in research (version 1.1, 7
  September 2026, read on 5 October 2026 as recorded below), under the heads
  the author asked for (tool and version, dates, what the tool did, what the
  author did, checks, limitations, records), and after the form of the Oxford
  Mathematical Institute's examples for research students ("Declaration of
  Generative AI Use - Examples",
  https://www.maths.ox.ac.uk/system/files/attachments/GenAI_PGR_declaration_examples.pdf,
  read on 12 October 2026: a compliance statement, what is the candidate's own,
  the tools with provider, model and date, the uses task by task, what the tool
  was not used for, a confidentiality statement, retention of prompts and
  outputs), so that a reader sees what the tool did and did not do and that the
  prose is not presented as the author's own writing. The form is borrowed with
  two departures the line itself states: no compliance statement, this being a
  research record and not a thesis, and the opposite of the examples' sentence
  that the final prose is the candidate's own. The freeze commit planned on 12
  October was not made; its uncommitted edits were discarded. The order of acts
  in `docs/protocols/P5_v2_amendment.md` stands at step 1 again.
  - **The declaration, and what each sentence rests on.** The tool, Claude
    (Anthropic) through Claude Code, with Claude Fable 5.1 named for the
    sittings of 11 and 12 October 2026, the model this session reports for
    itself and nothing in the repository shows; earlier sittings are dated in
    the log without a model version, which is why none is claimed for them. The
    uses listed are those the log records: the drafting of the document's text
    from the author's decisions (D55 to D71, D76, the last saying the section
    is drafted by Claude from the sources); the literature search and reading
    at source (D76; D54's "literature search" names no searcher and is read as
    the tool's by its closing line); the candidate values proposed by the tool
    and accepted or changed by the author, which D64 to D71 record as "a choice
    of the author on the assistant's lean" and D58, D59 and D61 as "as
    proposed"; the scripts and tests (D59 to D62, D72 to D75), the test suites
    and numerical checks run in the sittings on the author's machine (D69, D70,
    D72 to D75) and the independent review passes (D73 to D77); the
    decision-log entries, every one from D54 closing with the AI-assistance
    line; and the 3 October probe (D55). The author's part rests on the same
    record: the values chosen by the author (the entries above), each sitting's
    change confirmed by the author in the sitting before its commit, a
    confirmation that lives in the session transcript and in each entry's
    closing line rather than as a quoted word in the log; the G0 pilot run by
    the author on his machine (D60, D62); the reading of the sources an act the
    author performs himself, outstanding. The transcripts: Claude Code keeps
    session files under the author's user directory (four exist for this
    repository on 12 October 2026, the current session's opened on 31 August
    2026) and prunes them after 30 days by default, so the line says retention
    is the author's to arrange; the 3 October probe ran in a claude.ai chat
    (D55), and which other sittings, if any, were held there rather than in
    Claude Code is not recorded, so the line speaks of each by where it was
    held. The confidentiality sentence rests on the sittings having used the
    repository's committed files, which are public, and the author's own words.
    The sentence that the prose is AI-drafted and author-confirmed and not
    presented as the author's own writing is the point of the redraft. The
    University's research policy and the MPLS Division's page for research
    students were read on 5 October 2026 in a sitting whose reading is recorded
    in the assistant's notes outside the repository, not in the log: on that
    reading the University's page says a declaration of substantive use could
    include the tool's name, version, date and use, and the Division's page
    asks for the tool's name and version, its publisher and a brief description
    of use, and does not permit substantive AI-written prose in a thesis; this
    document is a research record, not a thesis or an application text, and the
    line says what it is. The papers' "Use of generative AI" sections, aligned
    with the Division's format on 25 July 2026 (commit `5a48243`), say AI gave
    drafting and editing support for manuscript text with the final wording
    approved by the author; they are not changed by this entry, and the
    difference of wording between them and this line is noted here for the
    author.
  - **Matched elsewhere.** The OSF summary text in
    `docs/protocols/P5_v2_amendment.md` carried the old one-line form; it now
    points to the declaration in the document. The first sentence of the
    prior-work section (line 158), which names the tool and the date of its
    reading, stands, and its clause that the box is left unticked is true
    again.
  - **Not written.** The freeze; anything on OSF; v1; the decision sheet; the
    papers' declarations; any run.
  - What remains before freezing: the author's reading of the eighteen sources
    in full and the tick at line 214 on his confirmation; then the order of
    acts in D77's file from step 2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D80** *(2026-10-22)* **P5 v2 draft: the prior-work box ticked on the
  author's confirmation that the full texts of the eighteen sources were read
  (one checklist box ticks; docs only; nothing frozen).** Line 214 of
  `docs/protocols/P5_protocol_v2_draft.md`, "Prior-work section written from
  the papers themselves", is ticked. D79 had withdrawn D78's confirmation, the
  abstracts having been read in most cases, and returned the box to unticked;
  the author confirmed on 22 October 2026 that the full texts of all eighteen
  sources listed under Relation to prior work have now been read. No other line
  of the draft changes; the one OPEN item (line 94) and the box "OSF
  registration amended" (line 215) stand. This is step 1 of the order of acts
  in `docs/protocols/P5_v2_amendment.md` (D77).
  - **Two clauses this tick makes stale, left to the freeze commit.** Line 158
    says the prior-work box is one "which this draft leaves unticked", and the
    declaration at line 200 says the author's reading is "outstanding at the
    time of writing". Both described the draft before this commit. They are
    corrected in the freeze commit (issue #226), which already planned the
    change to line 158, so that this commit touches line 214 alone, as the
    author asked; the interval is declared here.
  - **The model of this sitting.** This sitting ran on Claude Opus 5.5, the
    model the session reports for itself and nothing in the repository shows.
    The declaration at line 200 names Claude Fable 5.1 for the sittings of 11
    and 12 October 2026 only; the freeze commit is the place to extend it to
    the later sittings.
  - **Not written.** The freeze commit; anything on OSF; any change to the
    section's text, to v1 or to the decision sheet.
  - What remains before freezing: the freeze commit on the author's word (issue
    #226), then the OSF acts and the post-freeze commit in the order D77's file
    sets out; line 94's amendment after G2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D81** *(2026-10-22)* **P5 v2 draft: pre-freeze fixes after an independent
  review (the null, the extrapolation confound, the reweighting floor, G2's
  scope, failure and tolerances, the test set fixed at N = 400, the gate
  failures, G3's training size, precedence and binding, the declarations; docs
  and code; nothing frozen; nothing run on real outputs).**
  - **Summary.** The review of 22 October 2026 found v2 not ready to freeze,
    needing no redesign: a faithful calibrator could never meet the null;
    full-set verdicts mix extrapolation with prior dependence; the reweighting
    floor, G2, the test set's size, the gates' failure paths, precedence and
    some declarations were loose or missing. The author decided: a secondary
    analysis on well-covered pairs, the limitation stated; N = 400, G2 a go or
    no-go on cost; one retry at G1 and G2, then P5 stops, G3 per cell; G3 at
    the largest affordable n; G2's tolerances at least four Monte Carlo
    standard errors. The draft, the amendment file, the scripts and the tests
    now match. Left: the second review, eight boxes to re-tick, the freeze.
  - **The review.** An independent review, made at the author's request on 22
    October 2026 in a claude.ai chat on Claude Opus 5.5 on a clone of `main` at
    `b11063b` (a referee's read, a code-against-protocol audit and that
    sitting's own checks, all read-only; no P5 script, test or pricer was run,
    the audit's only computations being counts of the rows and fields of the
    committed G0 output), found v2 not ready to freeze and needing no redesign.
    Its record, `P5_v2_prefreeze_review_2026-10-22.md`, is committed unchanged
    at the repository's root. This entry records the author's decisions on it
    and every change made in answer, item by item, in
    `docs/protocols/P5_protocol_v2_draft.md`,
    `docs/protocols/P5_v2_amendment.md`, the four P5 scripts still to be run,
    their tests and `requirements-p5.txt`, with what two independent read-only
    readers of those changes found in this sitting and what was changed for it.
    Seven checklist boxes are reopened and a box for the independent review is
    added; a second, shorter review of these changes and the freeze follow.
  - **The author's decisions.** Given in the author's brief of 22 October 2026
    as the author's, made on the assistant's lean. D1, the confound: both a
    secondary analysis on well-covered pairs and the limitation stated. D2, the
    test set: N = 400, fixed now; G2 becomes a go or no-go on cost. D3, gate
    failures: one pre-specified retry for G1 and G2, then P5 stops; in G3
    passing cells proceed and failed cells are untestable. D4, G3's size: the
    largest n on the ladder that passes and fits the budget. After the hold of
    this sitting, also the author's, on the assistant's lean: D5, G2's
    tolerances: each becomes the larger of its absolute value and four Monte
    Carlo standard errors (B4); and "separates" becomes "limits" in lines 66,
    126, 136 and 170, as the text reader proposed, the word having been the
    assistant's, not the author's. The review's proposals on the null and the
    reweighting floor were taken as the review put them; its proposal on
    precedence (the scripts at the freeze commit with the readings of D72 to
    D75, the hashes of the six scripts and both requirements files, and the
    numpy version, listed in the frozen text) with the brief's additions:
    scipy's version, D81's readings, the listing under Status (placed inside
    the Status line, so that no line is renumbered) and the scripts' refusal of
    other versions.
  - **B1, the null.** Line 128: the null stands for a calibrator when H-a is
    Confirmed, H-b, H-c and H-d are Refuted and H-d's interval lies wholly
    within [0.85, 0.95]; as written before, all four Refuted, a faithful
    calibrator could never meet it. The analysis script reads the null so; a
    test plants a faithful calibrator (d near 0, E near 0, m near 0.90), on
    which the null stands, and an unfaithful one, on which it does not.
  - **B2, the extrapolation confound (D1).** The coverage covariate, per pair
    and per prior P, is the mass P puts on the equal-tailed 90% interval of H
    under the exact posterior under P; a pair is well covered when both masses
    are at least 0.001 (line 99; the reference reports both, line 68). Before
    it was written in, the arithmetic the author asked for, on the priors only:
    for a Gaussian posterior centred at the true H, 49.7% of uniform test H is
    well covered at the Identified edge's standard deviation (0.017; H from
    0.120 to 0.348) and 64.9% at the Flat edge's (0.038; 0.085 to 0.383), 597
    and 779 of the 1,200 pairs at N = 400, both above the 300 below which the
    author asked the work to stop. The secondary analysis, a new bullet under
    Analysis plan, re-scores H-a to H-d on the well-covered pairs with the same
    measures, thresholds, floors, bootstrap and verdict rule, reported beside
    the primary verdicts and labelled secondary; the null is read on the
    primary verdicts only. The limitation is stated in line 66 (replacing "that
    is the design and not a defect", with the training counts at the largest
    training size: about 3,150 of an A-trained network's surfaces where half
    the test θ lie, H at or above 0.25, thinning to about 15 above 0.36, and
    about 18 of a B-trained network's below 0.12, those two regions holding 48%
    of the test θ), in the reading of H-b and H-c (line 126, with the surfaces
    within 0.034 of the truth: about 1,700 at H = 0.30 and 10 at 0.40 under A,
    about 1,300 at 0.15 and 60 at 0.10 under B) and in novelty claim (ii) (line
    170); every count by arithmetic on the priors.
  - **B3, the reweighting floor.** Line 68: reweighting is accepted under a
    prior when (Kish's (Σw)²/Σw² / the number of pooled sampling steps) × the
    effective sample size of H is at least 1,000, else a direct run. Lines 89
    and 126, which rest on 1,000 effective draws under A and on the sampler
    error that follows from them, hold literally under this rule and are
    unchanged on that point.
  - **B4, G2 (D2, D3).** Line 143: the tolerance adds the posterior mean of H
    under B within 0.001 by the route the pair used, and A's 5% and 95%
    quantiles of H within 0.002, the grids computing the posterior under B and
    A's quantiles from the same likelihood evaluations (the quantile read from
    each H cell's mass spread uniformly over the cell). The arithmetic behind
    0.002, written in as a reading: at the Identified edge a quantile error of
    0.002 moves one tail's mass by 0.012, so m by at most 0.024 (0.873 to 0.922
    for a nominal 0.90), and at the Flat edge, where H-d is scored, by at most
    0.011. Two fixed pilot points, H = 0.04 and H = 0.46 at the box's centre in
    ν, ρ and ξ₀ (0.20, −0.80, 0.145), noise `default_rng(139)`: 18 pilot
    surfaces. On failure, one retry with burn-in and sampling doubled on fresh
    surfaces (points `default_rng(131)`, noise `default_rng(137)`, the fixed
    points' noise `default_rng(149)`); if it passes, the doubled lengths serve
    every later run; if it fails, P5 stops. The go or no-go: the projected cost
    of the 1,200 test pairs' references against 1,344 core-hours (G0's two
    weeks of four cores); the same budget and rule at G3 for scoring both
    calibrators. The tolerance is checked on every pilot surface on its first
    grid and, at the largest noise level, on the second as well; a reference
    failure on any pilot surface fails the gate, D73's reading, now written in.
    D5, the author's decision after this sitting's hold: each tolerance is the
    larger of its absolute value and four Monte Carlo standard errors of the
    MCMC figure, the error from the effective sample size n_eff of the run the
    figure came from ((Kish / pooled steps) × the ESS of H for a reweighted
    figure, the ESS of H for a direct run) by line 89's Gaussian reading: the
    posterior standard deviation over √n_eff for a mean, (1 − c)·√(2 / (n_eff −
    1)) for c, √(q(1 − q) / n_eff) / (φ(z_q) / the posterior standard
    deviation) for the q quantile; on every comparison, the whole-box grid's
    included (lines 89 and 143, with a reading of why four). What prompted it,
    by the same reading: under the absolute values alone every one of the 18
    surfaces had to meet c within 0.01, and at c ≈ 0.21 (the 0.5 level at D38's
    point) c's error is about 0.035 at 1,000 effective draws, 0.011 at 10,000
    and 0.008 at 20,000, so the six surfaces at that level met it together with
    probability about 0.0001, 0.06 and 0.25 (at c ≈ 0.5, 0.002, 0.36 and 0.76),
    the new 0.002 and 0.001 behaving alike near the floors; and under D3 a
    second failure stops P5. Under the rule, recomputed by the same reading:
    each comparison passes with probability at least 0.99994, exactly that
    where four standard errors govern; the six c checks at either of those
    levels pass together with probability 0.9996 at every n_eff; all 90
    comparisons on the first grids pass with probability at least 0.994 (0.992
    counting the second grid's 30 as if independent, which they are not,
    sharing the MCMC figures). The grid's own quadrature error is outside the
    rule: under about 0.0008 on the quantiles, by arithmetic on a Gaussian. The
    price is resolution where n_eff is low: at 1,000 effective draws the
    tolerance on a mean is 0.0022 at the Identified edge and 0.0048 at the Flat
    edge, and on a 5% quantile 0.0045 and 0.0102, so that a bias of 0.005 in a
    Flat-edge mean passes about 44% of the time and one of 0.010 almost never;
    at 10,000 effective draws the absolute values govern at the Identified
    edge.
  - **B5, the test set (D2).** Lines 93 and 94: N = 400 test θ, 1,200 pairs; θ
    the first 400 rows of `default_rng(20261002).uniform` over the box, shape
    (400, 4), columns H, ν, ρ, ξ₀; noise
    `default_rng(20261022).standard_normal((400, 3, 35))`, indexed by θ, level
    (0.1, 0.3, 0.5) and quote; the OPEN marker removed; the test set's
    references computed only after G2 has passed. Line 101 updated for N = 400
    (the 10%, 120 pairs, governs; a share near 10% uncertain by about ±1.7
    points on 1,200 pairs, where it said ±3.4 on 300); line 11 says no item is
    open (line 7, the Status line, never named one).
  - **Gate failures (D3; the review's non-blocking item 6 and the audit's first
    finding of code not in the protocol).** G1 (lines 54, 142): a training or
    held-out point with a quote that is not finite is priced once more at
    `N_riccati` 16000 with the same inversion settings, its quotes then taken
    from that pricing (the text's addition: the brief named the re-pricing, not
    where the quotes come from), a training point being one surface, so that
    the other surfaces of its triple keep their quotes; a quote still not
    finite stops P5; the refinement on e > s/5 stays the build's one
    refinement. G3 (line 144): each (calibrator, prior, level) cell passes or
    fails on its own; a failed cell's seeds are not trained; every measure
    needing a failed network loses that level for that calibrator, the other
    levels pooled as before, with line 75's floors and the seed floor taken on
    their pairs; a hypothesis left with no level for a calibrator is untestable
    for it; P5 stops only if every cell of both calibrators fails.
  - **G3's training size (D4; the review's non-blocking item 1).** Line 144:
    the largest n on the ladder at which both checks hold and the runs at n and
    2n are within the budget, every rung being run.
  - **B6, precedence and binding.** A second new bullet under Analysis plan:
    wherever the document is silent, the procedure is that of the P5 scripts at
    the freeze commit, with the readings D72 to D75 and D81 declare; their
    SHA-256 hashes, with those of the two requirements files, and the numpy and
    scipy versions the runs use (with torch's for the calibrators), are
    appended to the Status line at freezing, so that no line is renumbered; any
    change after freezing is a dated amendment. Written in explicitly: the
    bootstrap's seed and streams, `default_rng([127, c, r])`, r = 9 for the
    dose-response (line 132); m from the 90% interval of the network trained
    under A (line 110); the ensemble's variance over members with divisor 5,
    the variance of the equally weighted mixture (line 75); a pair failed for a
    calibrator if any of its networks failed on it (line 75); a failed direct
    run under A or B a reference failure (line 68); the generators' index
    bases, the reference's i and j from 0 and the ratio estimator's i from 0
    and j from 1 (lines 68, 75); the training draw's layout as D74 reads it
    (line 75); G1's tail rule as D72 reads it, the full tensor-product
    expansion and ties to the first dimension (line 142).
  - **B7, the declarations.** Under What was known before freezing: line 15
    announces them and declares the P5 test suites, the G1 test that builds a 2
    × 2 × 2 × 2 toy interpolant from the pricer at `N_riccati` 200 on a small
    box near D38's point included, and line 7 is qualified accordingly; line 20
    declares the second probe of 3 October 2026, from the review's record; line
    21 the scoping run's classical-Heston known answers (up to 0.098 in vol, 29
    to 30 of 350 rows lacking an implied vol on one side, D60); line 22 that
    the main run computed no known answer at its adopted settings, and that the
    G0 output holds quotes on v2's observable at the five values of H in every
    lattice cell inside the box (8,680 rows with a finite base quote at all
    five, the review's count), the noise levels, prior B, the cut-offs and the
    hypothesis numbers having been fixed after it was committed, with the log's
    record that it was not used for identifiability and was read by scripts
    outside the repository.
  - **G0's test-suite condition (the review's non-blocking item 5).** The whole
    suite was run on 22 October 2026 on the author's machine, in the core
    interpreter (Python 3.14.3, numpy 2.4.4, scipy 1.17.1), at commit `b11063b`
    before any change of this entry: 317 passed and 7 skipped (the torch
    tests), of 324 collected, in 14 minutes. Line 141 says the condition was
    not recorded at G0 and was checked on this date.
  - **Stale text and small corrections (the review's non-blocking items 2, 3,
    4, 7, 8 and 9).** Line 126: 32% becomes 36%, B's 1st percentile being at H
    = 0.1875 under the adopted B. Line 141: the emulator design is no longer
    said to be not yet fixed. Lines 158 and 200: the clauses D80 left for the
    freeze commit are corrected here, so that the freeze commit changes only
    what the amendment file's step 2 lists; line 200 names the model of each
    later sitting (this one and D80's on Claude Opus 5.5 in Claude Code; the
    review on Claude Opus 5.5 in a claude.ai chat) and adds the review and the
    second probe to what the tool did. Line 50: Q2 is exact up to the Monte
    Carlo error of the reference and, for the ratio estimator, of its own
    sampler. Line 89: Fisher information and CRB_H are no longer promised as
    descriptive companions; nothing computes them. Line 101: reference failures
    are reported by noise level and by true (H, ν), and above 10% of a level's
    test pairs (more than 40 of its 400) at any level P5 stops at Stage 1. The
    amendment file: the summary's open-item sentence dropped; a bullet for
    these changes in the change list; the items on the reference, the
    hypotheses and code before freezing brought into line; "What had been seen"
    brought into line with the declarations; steps 1 and 2 of the order of acts
    rewritten, step 2 carrying the hash list; one row added to the record.
  - **Layout.** The two new bullets under Analysis plan take the place of the
    blank lines around the Gates heading, so that lines 1 to 135 and every line
    from 141 on keep their numbers, which the scripts cite; the Gates heading
    moves from line 137 to 139 and the Pilots paragraph from 139 to 140, which
    no script cites; the checklist gains one line before the OSF box, which
    becomes line 216; the draft grows from 215 to 216 lines. Neither later pass
    (the readers', the author's answers after the hold) adds or removes a line.
  - **The checklist.** Reopened, unticked for the author to re-tick once each
    can be defended: the prior-knowledge section (declarations added); the
    emulator and its gate (the re-pricing, the tail rule); the Beta parameters
    (line 66's statement of the confound); the reference sampler and acceptance
    (the reweighting rule, the retry, the failure and index rules); the
    architectures and training (the divisor, the failed-pair rule, the draw's
    layout, the generator's base, the per-cell gate, the largest n); the
    hypotheses table (the null, the limitation, the 36%); the gate tolerances
    (G2's tolerance and failure path, G1's and G3's rules, G0's test-suite
    condition). The criterion: a box is reopened where the text it vouches for
    changed in substance, which reopens the Beta parameters although they did
    not move, line 66 now stating the confound and the training counts. Kept
    ticked, as their substance did not move: the grid and observable; the noise
    levels; the contraction cut-offs (line 89 loses only the descriptive
    companions and gains a pointer to G2's tolerance); the prior-work section
    (line 158's clause is a fact about the box, and the limitation added to
    claim (ii) concerns P5's own design, not the reading of a source). Added,
    unticked: an independent review before freezing, with findings and fixes
    recorded. These are proposals; the boxes are the author's.
  - **Code and tests.** `p5_g1_emulator.py`: the one re-pricing at 16000 in the
    pool's worker, the first pricing kept in the log, the stop; the shared
    version check. `p5_g2_reference.py`: the 90% interval under B and both
    coverage masses per pair; the (Kish / steps) × ESS rule; G2's added
    tolerance, each tolerance at least four Monte Carlo standard errors (D5),
    the fixed points, the retry (`--pilot-retry`), the go or no-go and the
    status that `--g3-pilot`, `--test` and G3 read; N = 400 and the noise seed
    as `--test`'s defaults. `p5_g3_calibrators.py`: per-cell pass or fail, the
    largest n within budget with each network predicted once, the scoring
    cost's projection, seeds trained for passing cells only, the ratio
    estimator's sampler at the reference's lengths. `p5_stage2.py`: the null; a
    Stage-1-only run (`--stage1`) from the references alone, before any score,
    which the scoring of the test set and Stage 2 require and check by a digest
    of the reference records (below); the coverage shares; the secondary
    analysis; G3's per-cell losses read from G3's report and checked against
    the scores; the reference-failure report and stop. Each script's
    command-line entry point, and not its import, records numpy and scipy in
    its output header and refuses versions other than the author's on 22
    October 2026 (the core interpreter numpy 2.4.4 and scipy 1.17.1 for G1, G2
    and Stage 1 and 2; the P5 leaf numpy 2.5.4 and scipy 1.18.1 for G3) unless
    `--allow-other-versions`, which is recorded; `requirements-p5.txt` pins
    scipy 1.18.1 beside numpy and torch, so that a leaf rebuilt from it passes
    that check. Nineteen new tests: thirteen written with the changes, five for
    the findings of this sitting's code reader (below) and one for D5; all
    nineteen fail on `b11063b`'s scripts, the last six and the five earlier new
    tests they strengthened also fail on the D81 scripts as they stood before
    the readers, and all pass on these. Four existing tests were adapted
    because the behaviour they pinned is what this entry changes: G2's grid
    pass rule, G2's refusal of `--test` without a noise seed, and the two Stage
    2 end-to-end runs, which now run Stage 1 first and carry a G3 report; their
    other assertions are unchanged. A fault the new tests found is fixed: the
    dose-response failed when a calibrator had no accepted cell at all, which
    the per-cell rule makes possible.
  - **This sitting's readers.** Two independent read-only readers were run on
    these changes before this entry was finished, one on the code and tests
    against the brief, one on the text against the brief, the review and the
    log. Taken up from the code reader: the calibrator's floors were taken over
    every level although a hypothesis pools only the levels G3 kept, so a
    verdict could rest on about 20 pairs, and they are now taken on the kept
    levels' pairs, G3's reason given first when no level is left; the first
    version of the test of the test set's definition drew the sealed θ and
    noise (What was run), and it now checks the definitions through a recorder
    that draws nothing; a run under `--allow-other-versions` was not recorded
    when it resumed a record, and now G2's launch lines carry the versions,
    G3's records gain a line for such a run and its report the versions, and
    G1's assembly records them; `--stage1` failed on an empty bin, now reported
    as none well covered of none, with no share; nothing stopped the test set
    being scored before Stage 1, and now the scoring requires Stage 1's output
    for its references and records that output's digest, which Stage 2 checks,
    the digest being of the reference records rather than of the file, so that
    a re-run of the references that only appends a launch line leaves Stage 1
    standing; an interrupted scoring could be read as empty seed slots, and now
    the scores' header lists the networks to be scored and Stage 2 refuses a
    record short of them; G1's re-pricing took every surface of a re-priced
    triple from the second pricing, now only the surfaces with a quote not
    finite (line 54's "point"); G3's predictions ran at one torch thread on a
    resumed run and at `--threads` on a fresh one, the projection counting both
    at `--threads`, and torch is now set to `--threads` before predicting; a
    level with no accepted pilot reference stopped the ensemble's prediction
    with an error, and its cell now fails; a passing cell whose scoring cost
    could not be projected counted as zero, and the projection is now
    incomplete and no go; G2's report took its lengths from the call, now from
    the record, a different call refused; `requirements-p5.txt` left scipy
    unpinned (above). Smaller: Stage 2 checks Stage 1 and G3's report before
    reading the stored samples, no longer rewrites Stage 1's CSV and figure,
    and reads no null on the shifted re-scoring; G1's header records the
    re-pricing a launch applies, and its assembly counts both pricings'
    seconds; G3's report names cells in one form. Kept: G2's header keeps
    `burn` and `sample` at the written values, an existing test pinning them,
    its `lengths` recording the lengths in force; the existing G3 ladder test's
    bound on the number of rungs, still true. Not tested: G3's use of G2's
    lengths in its entry point, which needs torch and an emulator file. Taken
    up from the text reader: the review's three passes named; lines 66 and 126
    said "almost no" training surfaces where that holds only at the far end,
    now reworded with counts; line 15 says that the G0 and G1 tests run the
    pricer at cheap settings at D38's point (G0's pilot test also at (0.02,
    1.00, −0.99)) and declares the sealed draws; line 22 says what the review
    counted and cites D65 for identifiability and D64 for smoothness; lines 41
    and 141 say which 3 October probe; line 54's clause on G0's box is back in
    place; lines 68 and 75 point to G2's doubled lengths and line 75 names the
    scipy pin; line 99 gains a colon; line 101 says of which test pairs the
    failure share is taken; line 136 says how the secondary verdicts are read;
    line 137 places the hash list inside the Status line with torch's version
    and says which versions are refused; line 141's record of the suite is
    reworded; line 143 says the second grid serves the largest noise level
    only; line 144 takes the floors on the kept levels' pairs; line 200 names
    the review's three passes and the chats; in the amendment file, "What had
    been seen" gains the timing of D65 and D69 to D71, the scripts outside the
    repository, the sealed draws and the review's counts, step 1 provides for
    recording the second review and step 2 places the hash list inside the
    Status line; and this entry is corrected where its first draft overreached
    (where the decisions were made, precedence as the review put it, lines 7
    and 11, the layout's line numbers). The two points the readers left for the
    author were decided after the hold: G2's tolerances against Monte Carlo
    error (D5, under B4), and the text reader's point that "separates"
    overstates, a pair being well covered with as few as about 160 of a
    network's training surfaces in its interval at the largest training size
    and about 10 at the smallest, so that the restriction limits extrapolation
    rather than removing it ("limits", under The author's decisions).
  - **What was run.** Tests only, and arithmetic on the priors and on a
    Gaussian. No P5 script was run from its command line. The tests ran the
    scripts' functions on toy maps and synthetic records, with the exceptions
    line 15 declares: the pricer at cheap settings, the G1 test's toy
    interpolant, and the first version of one new G2 test, which called the
    functions that draw the sealed test θ and noise and compared their output
    with a second draw, in memory, in each run of the G2 tests on the D81 code
    until this sitting's code reader found it, the whole suite's run included;
    that test passed each time, so pytest printed none of the values, and
    nothing was saved; it was then replaced (above). The whole suite at
    `b11063b` (above); the thirteen first new tests against `b11063b`'s
    scripts, all failing; the six P5 test files on the D81 code before the
    readers, 60 passed and 5 skipped in the core interpreter, and the
    calibrators' 12 passed in `.venv-p5`; the whole suite on that code, 330
    passed and 7 skipped of 337, in 15 minutes; after the readers' fixes, the
    eighteen new tests then written against `b11063b`'s scripts, all failing,
    and the five new and the four strengthened against the D81 scripts as they
    stood before the fixes, all failing; the six P5 test files, 65 passed and 5
    skipped in the core interpreter, the calibrators' 14 passed in `.venv-p5`,
    and the whole suite, 335 passed and 7 skipped of 342, in 36 minutes; after
    D5, its test and the grid test it strengthened against both earlier sets of
    scripts, all failing; on the final code, the whole suite, 336 passed and 7
    skipped (the torch tests), in 35 minutes, the six P5 test files within it
    66 passed and 5 skipped, and the calibrators' 14 passed in `.venv-p5`; 343
    tests collected, CONTRIBUTING updated from 324. The readers ran the six P5
    test files with the sealed-draw test deselected (the code reader: 59 passed
    and 5 skipped in the core interpreter, the calibrators' 12 passed in
    `.venv-p5`), a script of the code reader's own on synthetic Stage 2
    records, a check of torch's default thread count, and arithmetic on the
    priors and on a Gaussian. Two scratch scripts outside the repository called
    the scripts' functions on the tests' own inputs only: one ran G2's sampler
    and grid check on the G2 tests' toy linear map, as those tests do, to
    compare the grid's quantiles and mean under B with direct draws from that
    map's Gaussian posterior; one scored the Stage 2 tests' synthetic faithful
    records, to read H-d's interval while the faithful test was written.
  - **Declared, beyond the brief's words.** Precedence names D81 beside D72 to
    D75, as this entry's code readings need the same standing; the hash list
    includes the two requirements files, as the review proposed. The version
    check applies to the four scripts still to be run; the two G0 scripts,
    which have run and whose output headers record numpy and scipy, are not
    changed. The ratio estimator's sampler follows the retry's doubled lengths
    (line 75's "the same lengths"). G3's pilot references, like the test set's,
    wait for G2 to pass with a go. A cost above the budget at G2 stops P5
    without a retry, doubled lengths only raising the cost. The projection is
    the median seconds of one run of the lengths in force times the mean runs
    per pair, the seconds a pool worker's with one BLAS thread, as G0 read
    them; the scoring's seconds are counted at the scoring process's torch
    threads. Stage 1 refuses a record short of the pairs its header declares;
    Stage 2 refuses unless its Stage 1 declarations stand as made. The
    secondary analysis uses the primary's bootstrap streams and recomputes the
    floors within the well-covered set, at the written cut-offs only. The
    reference-failure share is of a level's test pairs, N from the record's
    header, and "exceed" is strictly above; the failure report's grid is five
    bins of true H by three of ν. G3's report keeps `passed` for every cell
    passing and carries the new rule in `p5_stops`. Line 143 writes in D73's
    reading that a reference failure on any pilot surface fails the gate; line
    54 says where the re-priced quotes come from, a training point read as one
    surface; lines 101 and 144 carry this sitting's readings (Stage 1 required
    by the scoring of the test set as well as by Stage 2; the floors on the
    kept levels' pairs); line 75 names the scipy pin; line 200 adds, under What
    the author did, the four decisions; step 2 of the amendment file names
    torch's version beside numpy's and scipy's. The scoring of the test set
    records Stage 1's digest and the networks it is to score; a projection that
    cannot be completed is no go; a run under the override is recorded in a
    record it resumes. The hash list binds the scripts as they stand at the
    freeze commit: `p5_g0_pilot.py` changed after the scoping run (`99aa4d5`, a
    `ladder` argument whose default keeps the old behaviour), so its listed
    hash will not be that of the version the scoping output's header names
    (`d394b08`); G0's runs are bound by their records. D5's errors use the MCMC
    figures' own posterior standard deviations and c; an n_eff below 2, which
    the acceptance criteria exclude, is taken as 2; the rule widens the
    tolerance for the sampler's error only, the grid's quadrature error staying
    outside it.
  - **Not written.** Anything on OSF; the freeze; v1; the decision sheet; any
    run on real outputs; the review's non-blocking item 10 (the percentile
    bootstrap over five seeds runs narrow), which the draft already declares
    and the author's brief did not take up; the audit's note that no code
    downstream reads G1's restriction on Q1, a reporting gap; and D72's
    sentence that the tests compare nothing across parameter values, which the
    toy interpolant contradicts and line 15 now replaces, the log being
    append-only.
  - What remains before freezing: the second, shorter review of these changes,
    recorded as step 1 of the amendment file's order of acts provides; the
    author's re-ticking of the seven reopened boxes and the review box; then
    the order of acts from step 2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D82** *(2026-10-23)* **P5 v2 draft: the second pre-freeze review's fixes
  (G1 binding what follows, the forward model in the hash list, a rule for
  disagreement, G3's complete pilot record, exact declarations, fixed values
  and overrides, no record written over, G2's full cost; docs and code; nothing
  frozen; nothing run on real outputs).**
  - **Summary.** The second review, of PR #234, found D81's answers in place
    and v2 ready to freeze after small fixes, with no redesign. The author
    decided: #234 merged as reviewed and the fixes in D82; both review records
    in `docs/protocols/`; the registration to hold the files the freeze binds;
    D5 kept, its cost stated. Now G1's outcome binds what follows and G̃'s hash
    travels with every record; the forward model is hashed and runs after
    freezing need a clean tree at the hashes; the text governs where it speaks;
    G3 needs a complete pilot record; the declarations are exact; N, the noise
    seed and the overrides are fixed and no record is written over. Left: that
    review's check of this diff, the author's re-ticking, the freeze.
  - **The review and its record.** The second, shorter review called for by
    step 1 of the amendment file's order of acts, made late on 22 October 2026
    (UTC) in a claude.ai chat configured for Claude Opus 5.5, on PR #234 at
    head `3ddf92a`: an independent referee, a code-against-protocol audit and
    the session's own checks, all read-only; no P5 script, test or pricer was
    run and no P5 output opened, the only computations being the referee's
    standalone arithmetic and a hash of the first record. Verdict: freeze after
    small fixes; no redesign; five changes needed before freezing (items 1 to
    5) and eight worth making in the same round (items 6 to 13). Its record,
    `docs/protocols/P5_v2_second_review_2026-10-22.md` (SHA-256 `740daca1`…,
    checked against the hash the author gave), is added unchanged, and the
    first, `docs/protocols/P5_v2_prefreeze_review_2026-10-22.md` (`29e73ceb`…),
    is moved beside it with `git mv`, unchanged; draft lines 200 and 215 cite
    the new path. Before this sitting the author merged #234 as reviewed
    (`f5a8257`), which brings in `3598211`, `64bdd4d` and `3ddf92a` and nothing
    else since `b11063b`, the merge's tree equal to `3ddf92a`'s; the branch was
    then deleted locally and on origin.
  - **The author's decisions.** Made on 23 October 2026, each on the
    assistant's lean, and the author's. 1, order: #234 merged as reviewed, the
    fixes in this entry with the record, so that what was reviewed and what was
    merged stay the same. 2, records: both review records in `docs/protocols/`,
    the first moved with `git mv`, the second added beside it, both unchanged.
    3, OSF: the new registration holds, beside the protocol, the files the
    freeze binds (the six P5 scripts, the forward-model modules, both
    requirements files and `ROADMAP.md`), each as its git blob at the freeze
    commit. 4, D5 stands, its full cost stated in the text.
  - **Item 1, G1 binds what follows.** `require_g1` (`p5_g1_emulator.py:216`)
    returns G̃'s path, the refined build when the base report calls for the
    refinement and the base build otherwise; it refuses, citing lines 54 and
    142, when the base stage stopped on a quote still not finite after the
    re-pricing (`stage_stopped`, :186), when the base report or emulator is
    missing or is not a complete assembly of that stage, and when the
    refinement called for stopped, is missing or is incomplete. From the
    command line G2 and G3 take G̃ from it and refuse an `--emulator` naming
    another file (`emulator_for_run`, :245; `p5_g2_reference.py:1090`;
    `p5_g3_calibrators.py:872`); functions called from tests may still be given
    an emulator. The emulator's path and the SHA-256 of its `.npz`
    (`emulator_info`, :158) are recorded in every reference record's header
    (G2's pilot, the retry, G3's pilot, the test set; `p5_g2_reference.py:709`,
    a resume refusing a changed file, :779), in G2's reports (:971) and status
    (:1023), in G3's record and report, in the training record, in the scores'
    header and in Stage 1's and Stage 2's reports (`p5_stage2.py:424`, :859).
    G3 refuses a G3-pilot record made with another emulator than G̃
    (`require_pilot`, `p5_g3_calibrators.py:553`); Stage 1 (the references
    against G2's hash in their header, `p5_stage2.py:398`), the scoring of the
    test set (G̃, the references, G3's report and Stage 1,
    `p5_g3_calibrators.py:805`) and Stage 2 (the references, the scores, G3's
    report and Stage 1's output, `p5_stage2.py:859`) refuse unless every hash
    they read is present and all are equal (`require_same_emulator`,
    `p5_g1_emulator.py:163`). Draft line 142 gains the sentence: if the build
    is repeated, the refined build is G̃ for everything after G1, the scripts
    refusing any other, and a stop at the refined stage stops P5 as at the
    base. Tests on hand-built toy builds and synthetic records: refinement
    called for, the refined path returned; a stopped stage refused; a
    mismatched hash refused, in G1, G3 and Stage 2.
  - **Item 2, the hash list covers the forward model.** The module-level
    imports of the six scripts, found by parsing their source with `ast`
    (nothing imported), reach five repository modules, `rough_heston_cf.py`,
    `layer4_calibrate_surface.py`, `layer4_calibrate.py`, `rough_heston.py` and
    `roughvol_core.py`, the five the review expected and no other. Two imports
    sit inside functions: `p5_g3_calibrators.score_set` imports `p5_stage2`,
    already listed; `layer4_calibrate.model_smile_lift` imports
    `rough_heston_lifted` and `layer4_convergence`, and by grep that function's
    only caller is `exp_lift`, an experiment in the same file
    (`layer4_calibrate.py:292–302`) that no P5 code calls, P5 taking only
    `KAPPA_FIXED` from that module and `layer4_calibrate_surface` never naming
    the function; the two are excluded and draft line 137 says why. Line 137
    and step 2 of the amendment file list the five, as hashes of git blobs
    (LF), with the record table's row; line 137 also says that every run after
    freezing is made from a clean tree, one with no modified tracked file (the
    headers' clean-tree field, which ignores untracked output), at a commit
    where each listed file has its listed hash, any other run being declared
    and its output not used.
  - **Item 3, a rule for disagreement.** Draft line 137 now opens: where this
    document speaks, it governs; where it is silent, the scripts at the freeze
    commit govern, as D72 to D75, D81 and D82 read them, a later entry
    superseding an earlier one; a disagreement between text and script found
    after freezing is corrected by a dated amendment before any run it affects.
    Line 144 adds the third stop the code makes, a passing cell whose scoring
    cost cannot be projected counting as no go; line 101 reads "before any
    network scores the test set", as amendment line 46 and the code have it.
    The binding statements under Objects, Test sets, Gates and Analysis plan
    were then read against the scripts once more (a reading, this sitting's):
    beyond the two the review named, the text disagreed with the code on the
    seeds a cell may be left with (lines 75 and 132; item 13), on G3's budget,
    which line 143 called the same as G2's (item 11), and on the count of G2's
    comparisons (item 9); each is corrected in the text, and no other was
    found.
  - **Item 4, G3 refuses an incomplete pilot record.** `require_pilot`
    (`p5_g3_calibrators.py:553`) refuses unless all 48 G3-pilot pairs (16
    surfaces at 3 levels) are recorded, accepted or failed, in a record made
    with G̃; the command line calls it for G3, the training of seeds and the
    scoring of the G3 set (:879), and line 144 says so. Tested on a synthetic
    record.
  - **Item 5, the declarations made exact.** An inventory of what the P5 test
    suites compute was made by reading the six test files, not running them:
    the sampler, grids and reweighting on toy linear maps and the analysis on
    synthetic records; the pricer at cheap settings, the G0 tests at D38's (H,
    ν, ρ) at ξ₀ 0.02, 0.04 and 0.25 and both G0 test files at (0.02, 1.00,
    −0.99), the G1 tests at D38's (H, ν, ρ) with ξ₀ in [0.04, 0.06]; two toy
    interpolants from the pricer, the 2 × 2 × 2 × 2 build on the small box
    (`test_p5_g1_emulator.py:157–188`) and the 2 × 1 × 1 × 2 build at D38's (H,
    ν, ρ) with ξ₀ near 0.043 and 0.057 (D81's test, now at :256–313); the
    design's unsealed draws in memory on toy maps (73; 79, 89 and 139; 131, 137
    and 149; 97 and 101; `[83, i, j]` for small i and j; prefixes of 109 and
    113); the committed G0 criterion file read by one G1 test; small networks
    with the design's trunk trained on a toy linear map by the torch tests.
    Rewritten from it: draft line 7 (no emulator, reference posterior or
    trained network of the design before freezing; the tests build toy ones);
    line 15 (the inventory; nothing printed or committed, pytest 9.1.1,
    installed in both environments, keeping by default the `tmp_path` folders
    of its last three sessions, `tmp_path_retention_count` 3 with the policy
    all, read from its installed source; D81's own runs; the second review;
    this sitting's runs; both records by their new paths); line 22 (the first
    review read the G0 output's fields, row counts and the criterion file's
    settings; the second opened none of it; both records by their new paths);
    line 200 (the second review's sitting and model, this sitting's, the
    decisions, D1 to D5 and "limits" after the first review and four after the
    second, the second review among what the tool did, both reviews' chats
    among the records); amendment lines 45 and 50, both records cited by their
    paths, so that "and nothing else" holds.
  - **Item 6, fixed values and overrides.** `--test` refuses an `--n` other
    than 400 or a `--noise-seed` other than 20261022 before any draw
    (`p5_g2_reference.py:1078`), its help saying so. Draft line 137 states when
    the version override may be used: only when a pinned version cannot be
    installed on the author's machine, each use declared in a dated amendment
    before the run's output is read. G3's command line also checks torch,
    2.14.1 with the local build tag aside (`LEAF_TORCH`,
    `p5_g1_emulator.py:104`; `version_check`'s `torch_expected`, :119;
    `p5_g3_calibrators.py:860`).
  - **Item 7, no record is overwritten.** Every report a P5 script writes
    refuses to write over an existing file (`refuse_overwrite`,
    `p5_g1_emulator.py:177`): G1's report, CSV and emulator (:676), G2's
    reports (`p5_g2_reference.py:950`, after the record's own checks and before
    any grid), G3's report (`p5_g3_calibrators.py:608`, before any training),
    the list of trained seeds (:899), Stage 1's report and CSV
    (`p5_stage2.py:393`) and Stage 2's report and cells CSV (:863, before the
    stored samples are read). A re-run needs the old file moved aside by hand,
    the move declared; line 141 extends "every run made is declared" from G0 to
    every stage after freezing.
  - **Item 8, resumed runs.** A resumed G3 compares the record's header with
    the call (the ladder, the draw, the validation size, the sampler's lengths,
    the emulator's hash), and a resumed training the passing cells and the
    emulator's hash, before any saved network is reused (`_check_resume`,
    `p5_g3_calibrators.py:543`, used at :620 and :738), as G1's launch does.
  - **Item 9, G2's text (D5's full cost).** By standalone arithmetic (`math`
    and `statistics` only), on line 89's Gaussian reading, the tolerance in
    force at 1,000 and 2,000 effective draws: on c, 0.090 and 0.063 at c = 0.5
    and 0.018 and 0.013 at c = 0.9, four standard errors governing in each,
    which allows a posterior standard deviation off by up to 8.9% and 6.3% and
    so moves m by up to about 0.030 and 0.021 (a reading, through the
    interval's width); on the 5% and 95% quantiles, 0.0045 and 0.0032 at the
    Identified edge (posterior standard deviation 0.017) and 0.0102 and 0.0072
    at the Flat edge (0.038), which by line 143's reading moves one tail's mass
    by about 0.028 and 0.019 and m by up to about 0.055 and 0.039 at either
    edge; 120 comparisons pass together with probability 0.992 on a correct
    sampler; a bias at the tolerance in force passes half the time, one a
    standard error beyond it about 16%. Written in: line 89, "above G2's
    absolute tolerance on c" and the tolerance on c at 1,000 effective draws;
    line 132, what G2 bounds (it checks the reference's Monte Carlo error
    against grid posteriors, which at the floors catches gross errors and does
    not bound it to H-d's precision); line 143, 120 comparisons in place of
    about 90, the bias sentence corrected, the full cost above, that G2 at four
    standard errors is a check for gross errors (a missed mode, an unmixed
    chain, a bug) and does not certify the reference at the floors to the
    precision H-d's numbers, 0.05 apart, need, that the effective sample size
    of H serves the variance and the quantiles as well as the mean, that the
    standard deviation and c in the formulas are the MCMC figures' own, and
    that the grid's quadrature error is outside the rule. D5 itself is
    unchanged.
  - **Item 10, "well covered".** Line 136: a mass of 0.001 is about 10 of a
    network's training surfaces at n = 10,000 and about 160 at n = 160,000, the
    ends of G3's ladder. Line 134 and Stage 2: the dose-response is also
    reported on the well-covered pairs, deciding nothing (`p5_stage2.py:891`,
    on the primary's bootstrap stream, row 9).
  - **Item 11, stops.** Line 197: when P5 stops at G1, G2, G3 or Stage 1, what
    is reported is the stop, its reason and the records up to it, in a dated
    decision-log entry; no hypothesis is scored, and nothing is re-run or tuned
    beyond the retries the text specifies. Line 143: G3 has a budget of its own
    of 1,344 core-hours, separate from G2's, as the code has it.
  - **Item 12, the retry's streams.** Lines 68 and 143: the retry's pairs are
    keyed `[83, i, j]` as the first attempt's, i counted within the retry's
    set, so that its chains at the fixed points start where the first attempt's
    did and draw the same proposals, only the noise differing; this errs
    towards stopping. No change to the code.
  - **Item 13, small items.** Line 141: what the characteristic function's
    modulus decided in D61 (`U_max` = 250/√T and the ξ₀ floor) replaces
    "decides nothing". Lines 75 and 132 and the training's docstring
    (`p5_g3_calibrators.py:720`): each failure after the one retraining leaves
    a slot empty, so a third leaves three seeds. `p5_stage2.py:66`: the
    docstring's streams read `[127, calibrator, row]`. The re-priced G1 record
    gains `N_by_xi0`, per surface the N its quotes come from, and a note that
    the re-priced surfaces take their quotes, strikes and ATM vols from
    `per_T_reprice` (`p5_g1_emulator.py:466`). Amendment step 1 names draft
    lines 15 and 22 beside line 200, and records that the second review was
    made and that its session checks this diff after the merge.
  - **Decision 3, in the amendment file.** The design: the registration holds
    the protocol and the files it binds, each the git blob at the freeze commit
    (LF) exported with `git show <freeze-commit>:<file>`. OSF's help, read on
    23 October 2026 at
    https://help.osf.io/article/229-select-a-registration-template (read-only;
    the address https://help.osf.io/article/158-create-a-preregistration served
    the same page): a registration made from a project or component archives
    every file in its storage with it, up to 5 GB in all; files cannot be added
    or removed in an update, and documents cannot be uploaded or changed after
    submission; the page does not say whether one question of the form takes
    several files. Proposed, therefore (a reading of the help): a component of
    `gdsv7` holding exactly the exported files, and the registration made from
    it, so that it archives them and nothing else. The summary names the bound
    files; step 2's hash list adds the five modules and says that the hashes
    are of git blobs with LF endings and that the hash of `p5_g0_pilot.py` is
    not that of the version the scoping run used (D81); step 2 exports and
    hashes every bound file, `ROADMAP.md` included; step 3 makes the component
    and registers it; step 5 compares the hash OSF reports for each file with
    its blob's; the record table gains the modules, a row for the exported
    files' hashes and the OSF row for every file. The change list gains a
    bullet for this entry.
  - **Code and tests.** In the four scripts, as itemised above. Five new tests,
    each failing on `main`'s scripts (`f5a8257`) and passing on these: G1's
    `require_g1` and its companions on hand-built toy builds; G2's fixed test
    set and G̃ before any draw or run, with the hash through the records, the
    resume and the report; G3's complete pilot record made with G̃; G3's
    resumes and its report not written over; Stage 2's hashes where records
    meet and the well-covered dose-response. Three earlier tests were
    strengthened, each failing on `main`'s scripts and passing on these (D81's
    per-surface re-pricing test: `N_by_xi0` and the assembly not written over;
    D81's test-set scoring test: the hashes; D81's Stage 1 test: Stage 2 not
    written over). Existing tests adapted, because what they did is what this
    entry refuses: a re-run onto an existing report now moves the old one aside
    first, as a declared re-run must (the end-to-end Stage 2 test, the toy G3
    ladder test, both older than D81, and D81's Stage 1 and G3 cost tests);
    D81's hand-off test stands in G̃'s path for the command line; the Stage 2
    fixture's headers carry a stand-in emulator hash, as every real record
    will; D81's test-set scoring test gives its stand-in Stage 1 output a hash.
    The two existing tests that check a refusal of `--test` now replace the
    sealed set's drawing functions with ones that fail if called, as the brief
    requires of a test that checks a refusal. `CONTRIBUTING.md` counts 348.
  - **What was run.** No P5 script was run from its command line; the tests ran
    the scripts' functions on toy maps, synthetic records and hand-built files,
    and the pricer at cheap settings, as line 15 declares; no test drew from
    the sealed streams, the drawing functions replaced by failing ones wherever
    a test reaches `--test`. In the core interpreter (Python 3.14.3, numpy
    2.4.4, scipy 1.17.1): the G1, G2 and G3 test files, 39 passed and 5 skipped
    (the torch tests), and the Stage 2 file, 15 passed, while the changes were
    written; the five new and three strengthened tests against `main`'s
    scripts, all eight failing; the whole suite on the final code, 341 passed
    and 7 skipped (the torch tests) of 348, in 17 minutes, the six P5 files
    within it 71 passed and 5 skipped. In `.venv-p5` (numpy 2.5.4, scipy
    1.18.1, torch 2.14.1+cpu): the calibrators' tests, 16 passed. Beside them:
    standalone arithmetic for item 9 (`math` and `statistics`, nothing from the
    repository); a parse of the six scripts' and five modules' imports with
    `ast`, which imported nothing; greps; pytest's version and its `tmp_path`
    defaults read from its installed source; OSF's help read with a fetch. No
    scratch script called the scripts' functions; no committed P5 output was
    opened.
  - **Declared, beyond the brief's words.** Readings of this sitting, each
    marked here. `require_g1` reads "incomplete" as a report that is not a
    complete assembly of its stage (its stage named and its node and training
    counts consistent) or an emulator file missing, the record's own
    completeness being assemble's check. `emulator_for_run` compares paths, so
    a copy of G̃ elsewhere is refused. G3's command line also checks G3's
    report against G̃ before training, and the scoring of the G3 set compares
    G̃, G3's report and the G3-pilot references. Stage 1 compares the
    references' hash with G2's as their header carries it, and a missing hash
    anywhere is refused rather than passed. The overwrite refusals come after a
    script's own record checks, so that those refusals keep their messages, and
    before any computation; G1's covers its emulator file too. A resumed G3
    compares neither threads nor versions (threads change timing, not the
    design; a version change is the override's, recorded); torch's local build
    tag is set aside. The amendment's change list gains a bullet for this
    entry, and its step 1 records the state of the second review. The
    re-reading's three further disagreements (item 3). The second review's "the
    four decisions" in D81's own text are stale, as below.
  - **Earlier entries now stale, not edited.** D81 says the first record "is
    committed unchanged at the repository's root" (ROADMAP 3368); it now sits
    in `docs/protocols/`, unchanged. D81 says that line 200 adds "the four
    decisions" (ROADMAP 3745); line 200 now names five after the first review,
    with "limits", and four after the second.
  - **The checklist.** No box is ticked or unticked. The review box (line 215)
    gains the second record and this entry, unticked. A ticked box whose
    section changed: the contraction cut-offs (line 211), whose paragraph, line
    89, now says "G2's absolute tolerance on c" and gives the tolerance in
    force at 1,000 effective draws; unticking it is the author's call. The grid
    (205), the noise levels (206) and the prior-work section (214) are
    untouched. No draft line was added or removed, so no draft line number
    changed.
  - **Not written.** Anything on OSF; the freeze; v1; the decision sheet; D81
    and every earlier entry; any tick; any run of a P5 script from its command
    line; the review's points outside the brief (the outputs force-added by
    hand, minor; G1's restriction on Q1 still unread by code, D81's reporting
    gap).
  - What remains before freezing: the second review's session checks this
    entry's diff against items 1 to 13 after the merge; the author re-ticks the
    seven reopened boxes and the review box; then the order of acts from step
    2.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D83** *(2026-10-23)* **P5 v2 draft: the check of D82 and the eight boxes
  ticked on the author's word (a rounding; the check declared; docs only;
  nothing frozen).**
  - **Summary.** Before #236's merge, the second review's session checked D82's
    diff read-only: items 1 to 13 done; two small points, a rounding (0.090 for
    0.089, draft lines 89 and 143) and step 1's "after the merge", both
    corrected here; the check declared. The eight boxes are ticked on the
    author's word (`fd58e8b`). Left: steps 2 to 5, from the freeze commit.
  - **The check.** Made on 23 October 2026 by the session that made the second
    review (D82), read-only, on D82's diff at #236's head `df1394b`, before the
    merge; the merge, `ba5acef`, has the same tree (`443ec9e`), checked here.
    It read the diff, the files at that head and the OSF help page the
    amendment file cites (its line 14); it ran nothing and opened no P5 output.
    The repository holds no other record of it; this entry records it from the
    author's brief. It found items 1 to 13 done as specified, in text and code,
    and two small points. The tolerance in force on c at c = 0.5 and 1,000
    effective draws is 4 × 0.5 × √(2/999) = 0.08949, 0.089 to three places, as
    line 143's 8.9% already has it; lines 89 and 143 gave 0.090, a rounding of
    0.0895, and now give 0.089, nothing else on them changing. Step 1 of the
    amendment file had the check made after the merge; it now says before, on
    the head whose tree the merge equals. The check is declared in draft lines
    15 and 200 and in the amendment file's line 51.
  - **The author's ticks.** Boxes 204, 207 to 210, 212, 213 and 215, the seven
    boxes D81 reopened and the review box, are ticked in `fd58e8b`, made in
    this sitting on the author's word: his edit in GitHub's editor had not
    reached the repository, no commit on any branch of origin having touched
    the draft since D82's, and the author then asked for the ticks to be made
    here. That commit changes those eight boxes and nothing else. Box 211, the
    cut-offs, stays ticked by the author's choice: D82 did not change the
    cut-offs, and on line 89 only corrected a justification and added figures.
    The rounding touches the sections of boxes 211 and 213 and is a rounding
    only. The ticks are their own commit; this entry follows in the same pull
    request.
  - **Stale, not edited.** D82's "0.090" (ROADMAP line 3933) and its plan to
    make the check "after the merge" (lines 3981 and 4078). The second review's
    record gives 0.090 too (its line 197); records stay as committed.
  - **Declared, beyond the brief's words.** Step 1 asks an entry with fixes to
    bring draft lines 15, 22 and 200 and the review box up to date. Line 200
    also gains this sitting in its list of models ("D82's and D83's sittings,
    on 23 October 2026, ran on Claude Opus 5.5 in Claude Code"), the model the
    session reports for itself; line 22 stays true as it is, the check having
    opened no P5 output; line 215 keeps the text the author ticked. The brief
    had the author ticking the boxes in GitHub's editor; they were ticked here,
    as above.
  - **Not written.** The freeze; anything on OSF; D82 and every earlier entry;
    v1; the decision sheet. Nothing of P5 was run in this sitting and no P5
    output opened; CI runs the tests on the pull request's head.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D84** *(2026-10-23)* **P5 v2 draft: prior work and the novelty claim before
  the freeze (Bayer and co-authors' ridge and Appendix A; claim (ii) limited to
  rough-volatility calibration; the ratio estimator's framing; docs only;
  nothing frozen).**
  - **Summary.** After D83 the second review's session checked the literature
    behind the novelty claim, read-only: claim (i) stands; claim (ii) was too
    broad; the prior-work section lacked Bayer and co-authors' ridge and
    Appendix A; the ratio estimator lacked its framing. Four text changes
    follow, on the author's decisions. Left: the freeze.
  - **The check.** Made on 23 October 2026, after D83, by the session that made
    the second review (D82), with two read-only research agents on the web and
    the session's own reading of the load-bearing passages at source, Bayer and
    co-authors' at the address the section gives; it ran nothing. The
    repository holds no other record of it; this entry records it from the
    author's brief.
  - **What it found.** Claim (i) stands, the closest works being Brigo, Huser
    and Leonte (arXiv 2609.31570) and Bayer and co-authors (arXiv 1908.08806).
    Claim (ii) was too broad as written. The prior-work section left out Bayer
    and co-authors' ridge sentence (Section 5.3) and their Appendix A. The
    framing of the ratio estimator was missing.
  - **The four changes.** 1, draft line 160: after "the figures being images
    that were not read.", a sentence on Section 5.3's ridge and one on Appendix
    A; line 164's item on Bayer and co-authors gains the ridge (Section 5.3;
    Figure 6). 2, draft lines 168 and 170: claim (ii) is limited to the
    calibration of a rough volatility model, and line 170 gains a sentence on
    general simulation-based-inference work, which the section does not survey
    and claim (ii) does not extend to; amendment line 44, the Prior work
    bullet, is limited likewise and gains D84. 3, draft line 75: the ratio the
    classifier learns at its optimum depends on the training prior only through
    a factor in y, so an exact ratio gives the exact posterior under P and any
    excess prior dependence beyond Monte Carlo error is error in the learned
    ratio. It agrees with the code: `nre_loss` (`p5_g3_calibrators.py:235`)
    trains on balanced joint and rolled pairs under binary cross-entropy, and
    `nre_target` (:400) samples log P(H) + logit. 4, draft line 200: the
    literature check under what the tool did, and D84's sitting, on Claude Opus
    5.5, the model the session reports for itself, in the list of sittings and
    models.
  - **The author's decisions.** Made on 23 October 2026, each on the
    assistant's lean, and the author's. 1: the four changes, before the freeze.
    2: no reweighted ratio estimator (trained under A, reweighted to B) as a
    comparator, which would be a design change at the freeze; it may come later
    as an exploratory analysis labelled as not pre-registered. 3: the general
    simulation-based-inference works are not cited in the protocol; this entry
    names them as pointers instead.
  - **Pointers, not cited in the protocol.** PriorGuide, Yang and co-authors,
    arXiv 2510.13763: the session read its abstract, and an agent reported from
    the text its evaluation against reference posteriors under shifted priors.
    Srivastava, Hazra and Dubbaku, arXiv 2609.25145: read at the level of the
    abstract. Betancourt, arXiv 1803.08393, whose posterior-contraction
    diagnostic is a precedent for stratifying by contraction: read only through
    an agent's report. This sitting checked each number, its authors and its
    title on its arXiv page.
  - **Boxes.** None is ticked or unticked. Box 214 (prior work) stays ticked:
    the new sentences on Bayer and co-authors come from a source the author
    read in full before D80; line 170's new sentence rests on the pointers
    above as read there, cites none of them (decision 3) and so carries no
    pinpoint, and the limits on claim (ii) only narrow it. Change 3 touches the
    section of box 210 (architectures and training), adding framing and
    changing nothing in the design. Unticking either box is the author's call.
  - **Declared, beyond the brief's words.** This sitting read Section 5.3 and
    Appendix A of arXiv 1908.08806v1, the only version, through arXiv's HTML
    rendering (ar5iv) on 23 October 2026, and an independent read-only reader
    checked the text against the brief, that rendering, the arXiv PDF, the code
    and the arXiv pages. Line 160 keeps two qualifiers of Section 5.3 that the
    brief left out (the level sets are in the (η, H) and (η, ρ) plots; on SPX
    the effect cannot be confirmed at first glance), and follows Appendix A
    where the brief did not: there the market scenarios that cannot all be
    trained on are why the route does not generalise to every setting, the
    poorly understood inverse map is why it is harder to justify for risk
    management, and the failure out of sample is on surfaces rough Bergomi did
    not generate, the network calibrating sharply on its own (Figure 8). "Since
    the sources above were read" became "beyond the sources above", the
    section's sources having been read on 11 October 2026 and both pointers to
    that work, PriorGuide (October 2025) and Srivastava, Hazra and Dubbaku
    (September 2026), being older. Box 214's reason is narrowed to the
    sentences on Bayer and co-authors. On line 75 the density sentence ends at
    its parenthesis, so that the new sentences follow it, and the sampler
    clause opens "The ratio estimator's posterior is sampled"; "up to its
    sampler's Monte Carlo error" became "up to Monte Carlo error", E and m
    carrying the reference's error too, and the next sentence carries the
    qualifier; "is largest" became "is expected to be largest", tied to the
    extrapolation the secondary analysis limits (lines 136 and 170). Line 200's
    clause takes a verb and names the session ("later that day in that session,
    made a read-only check") instead of repeating the date.
  - **Not written.** The freeze; anything on OSF; D83 and every earlier entry;
    v1; the decision sheet; a reweighted ratio estimator; any change to draft
    lines 15 and 22 or amendment line 51, which declare what had been run and
    seen of P5 before freezing, the check having been of the literature and
    having run nothing (line 200 declares it). Nothing of P5 was run in this
    sitting and no P5 output opened; CI runs the tests on the pull request's
    head.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D85** *(2026-10-23)* **P5 v2 draft: D84's three decisions declared in line
  200, and boxes 210 and 214 kept ticked (one clause; docs only; nothing
  frozen).** The author asked that line 200's account of what the author did
  list D84's three decisions, as it lists those made after the two reviews, and
  kept boxes 210 and 214 ticked. #238 was merged at 09:06 UTC on 23 October
  2026, before the clause could be pushed to it, so it comes in this entry.
  - **The change.** Draft line 200, under what the author did: "after the
    second, four (D82)" is followed by "; after the check of the literature,
    three (D84)". The same line's list of sittings and models gains D85's
    sitting, on Claude Opus 5.5, the model the session reports for itself, as
    step 1 of the amendment file asks and as D83 and D84 did. No other line of
    the draft, and nothing in the amendment file, changes; the draft stays at
    216 lines.
  - **The boxes.** The author keeps boxes 210 (architectures and training) and
    214 (prior work) ticked, on 23 October 2026: D84 added framing to the
    first's section and a fuller account of a paper he had read to the
    second's, and nothing in the design changed. No box is ticked or unticked.
  - **Not written.** The freeze; anything on OSF; D84 and every earlier entry;
    v1; the decision sheet. Nothing of P5 was run in this sitting; CI runs the
    tests on the pull request's head.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D86** *(2026-10-23)* **P5 v2: the post-freeze commit (registered on OSF as
  `695fw`; the registration in the Status line, the last box ticked and the
  record filled; docs only).**
  - **Summary.** Version 2, frozen at `59ca589`, is registered on OSF as
    `695fw` (DOI 10.17605/OSF.IO/695FW), archiving the 15 exported files, each
    equal to its blob; APSF7 now points to it. This commit adds the
    registration to line 7, ticks the last box and fills the amendment file's
    record. Nothing of P5 has been run.
  - **The changes.** Draft line 7 gains one sentence at its end: "Registered on
    OSF on 23 October 2026 as https://doi.org/10.17605/OSF.IO/695FW; the
    registered file is this file at the freeze commit, before this sentence and
    the tick of the last box were added (D86)." Line 216's box, "OSF
    registration amended", is ticked, nothing else on the line changing. No
    other line moved: the draft stays at 216 lines, and the protocol's blob at
    the freeze commit, the registered file, still hashes to `41255dbd`…, as
    listed below. In the amendment file the value cells of the table under
    Record are filled: the freeze commit, the protocol's hash, the freeze date,
    the registration's id, DOI and date, the update's approval, and pointers to
    the hashes below; nothing else in that file changes.
  - **The registration.** OSF `695fw`, an Open-Ended Registration of the
    component `n4x9j` (titled "P5 protocol v2: frozen files", inside `gdsv7`,
    by the author's account, the component not being publicly readable),
    registered at 18:11:59 UTC on 23 October 2026, approved by the author,
    public and not embargoed; DOI 10.17605/OSF.IO/695FW. Its title and summary
    are as the amendment file proposed, the summary equal, character for
    character, to that file's lines 29 to 53 at the freeze commit, each with
    its newline, with the three placeholders filled (the freeze commit, the
    protocol's hash, 23 October 2026) and the backticks removed. MIT; the five
    subjects of v1; nine tags. This sitting's checks 1 to 4, GET requests to
    OSF's public API only (the registration, its identifiers, its subjects and
    the files of its archive), found it public and not pending approval,
    embargoed, withdrawn or archiving, registered from `n4x9j` under licence
    `563c1cf88c5e4a3877f9e972` (MIT License) on 2026-10-23T18:11:59.269108,
    with that DOI, title and summary, and one folder, "Archive of OSF Storage",
    holding exactly 15 files, each with OSF's size and SHA-256 equal to its git
    blob's at the freeze commit:
    - `P5_protocol_v2_draft.md`, 102,963 bytes,
      `41255dbd288ecd03325701ab3018b716268380f4335eac858768dcfde42b9a80`
    - `p5_g0_pilot.py`, 46,221 bytes,
      `5a592624559c5dad30a3a8ef80deff1d0196e37427460ffc9985459f9d7970cd`
    - `p5_g0_main.py`, 39,975 bytes,
      `fa8706dc4028ebceaf8ad7f646030f25f2768a2877869e751109e75dad85fbc1`
    - `p5_g1_emulator.py`, 45,062 bytes,
      `2c81a92aa584e8edae02d2e82e089842259da4d9c9a36d9b36a481b725d2b04c`
    - `p5_g2_reference.py`, 62,681 bytes,
      `507674572a5f8487fbb37bad386356dfcad119c1e1b9a32000319bb7171e065d`
    - `p5_g3_calibrators.py`, 52,924 bytes,
      `c3b5e4298fb3f5e771cd8c9f95ac64b283ca6609ea61fd1cb7662ac16897dc31`
    - `p5_stage2.py`, 62,493 bytes,
      `5f196a44153fd4db994359329c0f0246b7dbda41127a4d8bd11ebcf9e56d1159`
    - `rough_heston_cf.py`, 13,784 bytes,
      `6b8fb570e444378db153fb83cc64269bb657d169772baf655364a8198fd7bb2f`
    - `layer4_calibrate_surface.py`, 13,347 bytes,
      `7776be7f0d2035a682369b1cacca478bd00289e493554db02a54bab2833d9331`
    - `layer4_calibrate.py`, 15,033 bytes,
      `c627572722ac4b2775c69b0013b2c5dd3fc4a4c845f845a4d6308b684c947597`
    - `rough_heston.py`, 11,455 bytes,
      `659062ff1027907279bcf5b3f02e47de37d075be9ae0f144a6472d65ab06006d`
    - `roughvol_core.py`, 7,023 bytes,
      `7fa98797451364dec119672a99d8a4c25da44287e8f1f2193702849391666982`
    - `requirements.txt`, 40 bytes,
      `48061e7fa7686b876b0221aaf663c6576c4aacc7227c1bb192d6250c5d037494`
    - `requirements-p5.txt`, 1,117 bytes,
      `85a2d7e9f279641ac8ca7663fd91ee2c87701452d10ba5634fbfc314ace12dd9`
    - `ROADMAP.md`, 384,843 bytes,
      `d16aed2e627dab6749a5084ab0a758c508638c82994efe25909286dbd3d40f47`
  - **The update to APSF7.** Schema response `6adba890da2bf2b3c80a0949`,
    submitted at 18:41:48 UTC and approved at 18:41:54 UTC (its last
    modification) on 23 October 2026: the amendment file's paragraph for the
    update, with the freeze date, `695FW`, the DOI's address, the freeze commit
    and the protocol's hash filled in and the backticks removed, placed at the
    top of the summary, v1's text unchanged below it after a blank line, and
    entered as the justification; only the summary changed
    (`updated_response_keys`); the uploaded file, `P5_protocol.md`, keeps its
    SHA-256 `5b7a67e5`…; APSF7 is public and not withdrawn. Check 5, GET
    requests only, read this from APSF7's schema responses: two, both approved,
    the earlier `6ac001210231cc8223157561` last modified on 2 October 2026.
  - **How it went on OSF.** The author's account, from his screens and from OSF
    pages read in his browser and checked by the session that guided him; only
    `695fw`, its files and APSF7 can be read publicly. All times UTC, on 23
    October 2026. A first component named "P5 protocol v2: frozen files" was
    created at 13:19 and received the 15 files at 13:49; a draft from it, on
    the OSF Preregistration template (created at 14:01:28), was never
    submitted. A second component, `usjn7`, was created empty at 14:21:06 and
    registered as `phkqc` at 14:47:14; no files having been archived, the
    author cancelled it within OSF's approval window at 15:12. Both of these
    components have since been deleted by the author. A third component,
    `n4x9j`, was created at 16:35:25 and registered as `n45wa` at 16:57:00
    before any file had been uploaded to it; the author cancelled that
    submission at 17:39:33. The 15 exported files were uploaded to `n4x9j`
    between 17:49:38 and 17:49:45, their SHA-256 values, read from OSF's API
    before resubmission, all equal to the blobs'; the same draft was then
    resubmitted and became `695fw`, whose public record gives 17:49:45 as its
    last modification. Neither `phkqc` nor `n45wa` was approved or made public;
    each went back to a draft when it was cancelled.
  - **Who did what.** The OSF acts were the author's. The second review's
    session guided them and checked them read-only: the component's file list
    before resubmission, as pasted by the author, and the public API after
    approval. This sitting re-read the public API, also read-only.
  - **The box.** Box 216 is ticked on the author's word.
  - **Not written.** Any P5 run, which none of this starts; any other box; v1;
    the decision sheet; D85 and every earlier entry. The places in the
    repository that still describe v2 as a draft or unregistered, or v1 as the
    registered protocol, are listed for the author and not changed.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D87** *(2026-10-23)* **P5 v2: the living documents that still called v2 a
  draft brought up to date (the ROADMAP's P5 paragraph; the status lines of the
  amendment file and the decision sheet; docs only).**
  - **Summary.** After D86, three living documents still called v2 a draft or
    v1 the registered protocol. The ROADMAP's P5 paragraph is replaced, and the
    status lines of the amendment file and the decision sheet each gain a
    sentence on what has happened since. The frozen draft and the files it
    binds are untouched; nothing of P5 has been run.
  - **The changes.** 1, `ROADMAP.md`, the P5 bullet under Publication seeds
    (lines 4394 to 4406 before this entry was added): replaced. The new
    paragraph gives v1's freeze and registration (APSF7), v2's freeze at
    `59ca589` and registration as `695fw` with the files it binds, APSF7
    pointing to it, v2's question and design in one sentence, that G0 alone has
    been run and G1 is next, and the targets of the P5 seed, unchanged. It
    gives no date for G1, by design, so the build timing that v1's paragraph
    carried over from the seed (its earliest build, summer 2027) is no longer
    stated. Each claim was checked against the frozen draft and the log: the
    two-step optimiser through G̃ is among the exploratory, labelled analyses
    (draft line 135); G0 is the only gate run, `output/` holding only its seven
    files and the log recording no later run; the targets are the seed's as it
    stood before v1 (`13bca7e^`). 2, `docs/protocols/P5_v2_amendment.md` line
    7: its text kept, one sentence appended ("Since written: …"), giving the
    freeze, the registration, the update to APSF7 and the filled record table
    (D86, D87). 3, `docs/protocols/P5_v2_decision_sheet.md` line 5: its text
    kept, one sentence appended ("Since written: …"), giving that every OPEN
    item was decided and every box ticked, and the freeze and the registration
    superseding v1, to which APSF7 points (D86, D87).
  - **Left as they are, by design.** The frozen draft's own wording, its file
    name, "this draft" and line 11's "amended with this version", which only a
    dated amendment could change; v1's Status line; `requirements-p5.txt`'s
    reference to the draft's file, a registered file; log entries D55 to D86
    and the review records, which are history.
  - **Not written.** Anything on OSF; any P5 run; any box; D86 and every
    earlier entry.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D88** *(2026-10-25)* **P5 v2: gate G1 run, assembled and recorded (G̃ the
  base build; Q1 claimed at all three noise levels; no refinement; the output
  committed).**
  - **Summary.** G1 passed at all three noise levels with no refinement (e =
    0.000021; smallest bound 0.0002). G̃ is the base build, SHA-256
    `c3adb2e2d01eeb4cf9105ce0011329c66de3cebfbf4ea970af1c306e4a2f4add`,
    committed with its record, held-out table and report. Nothing after G1 has
    been run.
  - **The pre-flight.** Held on 23 October 2026, read-only, in a Claude Code
    sitting: a clean tree at `3db18b5`; the 13 bound files' blobs at `HEAD`
    hashing to line 7's values, and `git diff --quiet 59ca589 HEAD` over them
    exiting 0; `py -3` the core interpreter (Python 3.14.3, numpy 2.4.4, scipy
    1.17.1) with no virtual environment active; no G1 output; 8 logical CPUs
    and 11.5 GB free on C:. The hold mode, `py -3 p5_g1_emulator.py` with no
    flag (run with Python's bytecode writing off, so that nothing was written),
    gave an identity difference of 0.0 over 35 quotes in 41 s and the estimate
    of 968 sets, 77.4 core-hours and 19.4 h on 4 workers, pricing nothing of
    the design. The author's first two attempts to start the run never reached
    the script: they were made from the wrong folder, and the pasted prompt
    text made PowerShell's `PS` alias run.
  - **The run.** The author's command, `py -3 p5_g1_emulator.py --run`, in his
    own terminal. The record's header and its one launch line: commit
    `3db18b5`, `clean_tree` true, Python 3.14.3 with numpy 2.4.4 and scipy
    1.17.1, checked against the pinned versions with no override, Windows 11,
    `cpu_count` 8, pool 4, launched at 05:10:45 UTC on 24 October 2026. The
    identity check at launch gave 0.0 in 22 s (the run's output, as the author
    reports it). One launch: 968 tasks (768 node triples, 200 held-out points),
    0 errors, every quote finite, no point re-priced (no `first_attempt` line).
    The run reported 1,156.0 min of wall time against the estimate's 19.4 h,
    the record's launch and last task (00:26:44 UTC on 25 October) agreeing;
    the task seconds sum to 277,139 s, 77.0 core-hours (222,883.7 for the
    triples, 54,255.3 for the held-out points).
  - **The assembly.** `py -3 p5_g1_emulator.py --assemble`, run by the author
    at 00:45:21 UTC on 25 October 2026 from `3db18b5` with a clean tree (the
    report's `utc_assembled`, `assembled_commit` and `assembled_clean_tree`).
    e, the 99th percentile of the absolute error over the 7,000 held-out quotes
    (numpy's default, linear interpolation), is 0.000021 (2.0945e-05, 0.0021
    vol point) against the bound s/5 = 0.0002 at s = 0.1 vol point, about a
    tenth of it; the bounds at 0.3 and 0.5 vol point, 0.0006 and 0.001, are
    wider still. Q1 is claimed at all three levels. The maximum, 0.000210,
    reported beside e as line 54 requires, exceeds the smallest level's bound
    by 5%; the gate is on the 99th percentile, so it decides nothing. The
    per-maturity 99th percentiles rise with T: 5.9e-06 at T = 0.1, 7.7e-06 at
    0.25, 1.8e-05 at 0.5, 2.7e-05 at 1.0 and 2.8e-05 at 2.0. The tails: H
    2.0e-10, ν 4.1e-05, ρ 9.3e-06 and ξ₀ 6.5e-07; ν's is the largest, the
    dimension a refinement would have doubled. No refinement: `refine` is
    false, and `require_g1()` returns the base build.
  - **This sitting's checks.** Read-only, Python reading the files as they are,
    `output/` unchanged (its listing, sizes and times), nothing priced, built,
    sampled or trained. 1, the record: one header and one launch line, 968
    distinct task lines (768 train, 200 heldout), no `stop` or `task_error`
    line, every task finite, no `first_attempt` line, header and launch as
    above. 2, the report: stage base, nodes (16, 8, 6, 12), `n_train` 768,
    `n_heldout` 200, the values above, `repriced_points` empty,
    `seconds_train_total` 222,883.7 and `seconds_heldout_total` 54,255.3, one
    launch. 3, the held-out table: 7,000 rows; the 99th percentile and the
    maximum of the absolute `diff` equal the report's `e_p99` and `max_abs`
    exactly, and every `diff` is exactly `G_tilde` − `G`. 4, the emulator:
    `values` (16, 8, 6, 12, 35), all finite, stage base; the box equals draft
    line 39's and the node coordinates are its first-kind Chebyshev nodes,
    exactly; `load_emulator()` at the 200 held-out points reproduces `G_tilde`
    exactly. 5, `require_g1()` returns `output/p5_g1_emulator.npz`, whose bytes
    on disk hash to G̃'s SHA-256 above. The four committed blobs, the text
    files' with LF line endings and their working copies CRLF, as with G0's
    output, the `.npz` byte for byte the file on disk:
    - `output/p5_g1_tasks.jsonl`, 22,487,574 bytes,
      `19f684b63e3eba96058610020d6f7331f00d4815556a2f069778272883e36f4c`
    - `output/p5_g1_emulator.npz`, 2,584,804 bytes,
      `c3adb2e2d01eeb4cf9105ce0011329c66de3cebfbf4ea970af1c306e4a2f4add`
    - `output/p5_g1_heldout.csv`, 1,086,349 bytes,
      `cbd75656d39a65e663eca0f9801110820bcb87300c517d4b2ceb1db6e4052a56`
    - `output/p5_g1_report.json`, 2,350 bytes,
      `959d959dbc52cb125094bfa20b9f0c526e17efb3dd8cb01e7b0c17adc9b212d5`
  - **Two tests updated.** On this entry's first commit CI failed two tests in
    `test_p5_g2_reference.py`, which expected the refusal "no emulator" because
    `output/` held no G1 build; they now point `p5_g1_emulator.paths` at an
    empty temporary folder for that refusal, with one new assertion that, G̃
    being committed, a file other than G̃ is refused before any load, and the
    failing run drew nothing sealed, both tests blocking sealed draws and their
    refusals coming before any load.
  - **What was seen.** G1's output only: the node values and the held-out
    errors, read by the assembly and by this sitting's checks. No reference
    posterior, network or test surface exists yet, and nothing about the
    hypotheses was looked at.
  - **Who did what.** The author ran the build and the assembly in his own
    terminal. The pre-flight was held in a Claude Code session on Claude Opus
    5.5, the model that session's transcript records for all 39 of its messages
    between 22:15 and 22:50 UTC on 23 October 2026; this sitting, in the same
    session, also ran on Claude Opus 5.5. Line 200 is frozen, so the log
    carries this.
  - **Not written.** G2 and everything after it; any change to the draft, v1,
    the decision sheet or earlier entries.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D89** *(2026-10-25)* **P5 v2: gate G2 run and recorded (it failed on its
  first attempt and on its retry, so P5 v2 stops at G2, line 143; the output
  committed).**
  - **Summary.** G2 failed on its first attempt and on its retry, so P5 v2
    stops (line 143). Every pair whose reference run was accepted matched the
    grid within tolerance, and the projected costs, 48.2 and 53.6 core-hours,
    were far within the budget of 1,344. Nothing after G2 has been run, and the
    sealed test set has never been drawn.
  - **Attempt 1.** The author's command, `py -3 p5_g2_reference.py --pilot`.
    The record's header and its first launch line: commit `7e97ac8`,
    `clean_tree` true, numpy 2.4.4 and scipy 1.17.1, checked against the pinned
    versions with no override, G̃'s SHA-256, lengths 10,000 and 40,000, seeds
    79, 89 and 139 (line 143), pool 4, launched at 01:52:58 UTC on 25 October
    2026. 4 of the 18 pairs were done and 14 raised
    `TypeError("reference_pair() got an unexpected keyword argument
    'emulator_path'")`. The cause is in `p5_g2_reference.py`'s `_worker` (line
    692): at a pool above 1, `run_set` passes `emulator_path` in each task's
    arguments, and `_worker` removes it only in the call that first loads the
    emulator, so every later pair in a worker raised at the call to
    `reference_pair`, before computing anything. The record shows the sequence:
    the first pair finished at 01:57:04 UTC and all 14 errors are recorded in
    the same second, while the three other pairs then running finished at
    01:57:25, 01:58:05 and 02:00:58; the 4 pairs done, the first four
    submitted, were each a worker's first and ran as written. Every pair's
    generator is `default_rng([83, i, j])` and every process runs one BLAS
    thread (`p5_g2_reference.py`, lines 88 and 89), so the pool changes no
    result. Windows then restarted, at 03:12 UTC, with no launch running; the
    terminal output of the first launch was lost, and the record and samples
    were intact (checked read-only at the time). The completion, `py -3
    p5_g2_reference.py --pilot --pool 1`, a path that passes no `emulator_path`
    (`p5_g2_reference.py`, lines 862 and 863), launched at 06:43:21 UTC from
    `7e97ac8` with a clean tree and `n_done_before` 4: the 14 remaining pairs,
    no errors, the last finished at 07:15:03 UTC. The record then holds 18
    pairs, 9 `ok` and 9 `reference_failure`, three of the failures from the
    first launch: all six at s = 0.001 (g2_0000_0, g2_0001_0, g2_0002_0,
    g2_0003_0, g2_0004_0 and g2_0005_0), two at 0.003 (g2_0000_1 and g2_0003_1)
    and one at 0.005 (g2_0003_2). Each failure ran twice, at 10,000 and 40,000
    and then at 20,000 and 80,000, and was accepted at neither (line 68: R-hat
    at most 1.01 for every parameter and an effective size of H of at least
    2,000): over their 18 runs, acceptance 0.14% to 4.2%, R-hat as high as
    14.2, and an effective size of H from 4 to 1,596. The 9 `ok` pairs all
    passed on the first grid, and the 5 at s = 0.005 on the whole-box grid as
    well; two comparisons exceeded their fixed value and passed on the Monte
    Carlo term of the tolerance (D81): g2_0000_2's contraction c on the
    whole-box grid, 0.01838 against 0.01 (in force 0.0845); g2_0000_2's 5%
    quantile of H on the whole-box grid, 0.002603 against 0.002 (in force
    0.00797). The report, written at 08:27:37 UTC from `7e97ac8` with a clean
    tree: `passed` false, 9 reference failures, a projection of 48.2 core-hours
    for the test set's 1,200 pairs (a median of 37.2 s per unit of 50,000 steps
    over 49 runs, times 3.89 units per pair), `go` true. The projection mixes
    the first launch's timings (10 runs, a median of 85.3 s per unit, four
    workers running) with the completion's (39 runs, 34.4 s); from the
    completion's runs alone it would be 44.6 core-hours, and at the first
    launch's median 110.6, both far within 1,344, so the mix changes nothing.
  - **The retry.** The author's command, `py -3 p5_g2_reference.py
    --pilot-retry --pool 1`, at doubled lengths on fresh surfaces: the header's
    `lengths` 20,000 and 80,000 (its `burn` and `sample` keep the written
    values, 10,000 and 40,000), seeds 131, 137 and 149 (line 143), each pair
    keyed `default_rng([83, i, j])`; one launch, at 08:40:48 UTC from `7e97ac8`
    with a clean tree and the versions as above; 18 pairs, no errors, the last
    finished at 09:29:36 UTC. 13 `ok` and 5 `reference_failure`, all five at s
    = 0.001 (g2r_0000_0, g2r_0001_0, g2r_0003_0, g2r_0004_0 and g2r_0005_0):
    over their 10 runs, at 20,000 and 80,000 and then at 40,000 and 160,000,
    acceptance 0.41% to 3.1%, R-hat as high as 25.9 (g2r_0004_0, the fixed
    point at H = 0.04), and an effective size of H from 4 to 1,817. The 13 `ok`
    pairs all passed on the first grid, and the 6 at s = 0.005 on the whole-box
    grid as well; two comparisons exceeded their fixed value and passed on the
    Monte Carlo term: g2r_0003_1's mean of H on the first grid, 0.001277
    against 0.001 (in force 0.0016); g2r_0003_1's 95% quantile of H on the
    first grid, 0.002435 against 0.002 (in force 0.00338). The report, written
    at 11:07:31 UTC from `7e97ac8` with a clean tree: `passed` false, 5
    reference failures, a projection of 53.6 core-hours (a median of 65.8 s per
    unit of 100,000 steps over 36 runs, times 2.44 units per pair), `go` true.
  - **What the two attempts show,** a reading labelled as such. The failures
    concentrate where the posterior is sharpest: 6 of 6 and then 5 of 6 at s =
    0.001, 2 of 6 and then none at 0.003, 1 of 6 and then none at 0.005. The
    acceptance rates and R-hat show chains that did not adapt, not chains that
    mixed slowly: the uniform-prior runs at s = 0.001 accepted 0.14% to 2.1% of
    their proposals on the first attempt and 0.41% to 3.1% on the retry,
    against 4.1% to 12.9% and 9.2% to 12.8% at s = 0.005, and R-hat reached
    14.2 and 25.9. Doubling the lengths cleared the two larger noise levels but
    not the smallest.
  - **The stop.** Under line 143, P5 v2 stops: G3, G3's pilot references, the
    test set's references, the calibrators and the scoring are not run.
    `g2_status()` returns `p5_stops` true, "G2 failed its checks on the first
    attempt and on the retry: P5 stops (line 143)". The two `go` values decide
    nothing, line 143 taking the go from a passing attempt. This entry is the
    dated decision-log entry that line 197 asks for on a stop: the stop, its
    reason and the records up to it, with no hypothesis scored and nothing
    re-run or tuned beyond the retry. No OSF act has been taken yet; a notice
    of the stop on `695fw` is the author's, and follows this entry.
  - **This sitting's checks.** Read-only, Python reading the files as they are,
    with `g2_status()` and `require_g1()` the only functions of the scripts
    called beside helpers that build paths or hash files; `output/` unchanged
    (its listing, sizes and times); nothing sampled, priced, built, trained or
    drawn. 1, attempt 1's record: one header and two launch lines as above,
    each launch from `7e97ac8` with a clean tree and no override; 18 distinct
    pair records, 9 `ok` and 9 `reference_failure`; 14 `task_error` lines, all
    the TypeError, all between the two launches, each for a pair with a later
    record; the first launch's 4 pairs the first four submitted; every pair's
    generator `default_rng([83, i, j])`; no other line. 2, its report: as
    above, its statuses equal to the record's. 3, the retry's record and
    report: as above, no `task_error`. 4, `g2_status()` as above. 5, the
    samples: each consolidated archive holds exactly the arrays its record
    implies for the `ok` pairs, each pair's `_u` array and one per direct run
    (19 and 23), each of n_chains × sample entries of its accepted run, and
    equals the union of its `_parts` files array for array (`uint16`); the
    `_parts` folders are not committed. 6, G̃: both headers carry G̃'s SHA-256,
    and `require_g1()` returns `output/p5_g1_emulator.npz`, whose bytes hash to
    it. No `output/p5_g3_pilot_refs*` or `output/p5_ref_test*` exists. The six
    committed blobs, the text files' with LF line endings and their working
    copies CRLF, as with G1's output, the two `.npz` byte for byte the files on
    disk:
    - `output/p5_g2_pilot.jsonl`, 50,081 bytes,
      `80d265f850c75d04c64bc153ea42c292d583eb059b78bfc20d4bc67907e1ddae`
    - `output/p5_g2_pilot_samples.npz`, 1,120,357 bytes,
      `d27d619c5cf350e93745c286dd86ba084878799f5de4a72156ea2daf1e4a6271`
    - `output/p5_g2_report.json`, 57,797 bytes,
      `1756d30fd4ab23322289da17c57ef42427072132769dc82f386af220ef0b7bb4`
    - `output/p5_g2r_pilot.jsonl`, 45,183 bytes,
      `7c4dcc2c6ca477c230fc8e56ca8fb937e632076ac0698bc54cc2eefb8df668f5`
    - `output/p5_g2r_pilot_samples.npz`, 1,861,426 bytes,
      `869837b7c21ccc8a4bd44cafd50b372193c0cebf87079a1dd658b96fa20bb822`
    - `output/p5_g2r_report.json`, 59,463 bytes,
      `a4966b55e89dc9c5ee6471c79eb590b3352207c3ba84eb15ce55cfd99ad6305b`
  - **What was seen.** G2's pilot surfaces and the posteriors on them: the
    means and contractions of H for the passing pairs, and the diagnostics of
    the failing ones. None is a test surface, no network exists, and nothing
    about the hypotheses was looked at.
  - **Who did what.** The author ran attempt 1, its completion and the retry.
    The G2 pre-flight (25 October 2026, to 01:50 UTC), the read-only checks
    during the runs and this sitting were held in one Claude Code session on
    Claude Opus 5.5, the model its transcript records for every message on 25
    October; the author confirmed that no other sitting is to be named.
  - **Not written.** Any continuation of P5, which would be a new version
    registered before any run; any change to the draft, v1, the decision sheet
    or earlier entries.
  - Drafted with AI assistance; every decision confirmed by the author.
- **D90** *(2026-10-25)* **P5 v2: the stop notice posted on OSF (`695fw`
  updated; docs only).**
  - **Summary.** The author updated `695fw` on 25 October 2026 with a notice of
    P5 v2's stop at G2 (D89), placed at the top of its summary above v2's
    unchanged summary and entered as the justification, as for APSF7 (D86). The
    registration stays public and is not withdrawn. No v3 work precedes this
    entry.
  - **The update.** Schema response `6ade09278faa7545598a0050`, created at
    13:50:31 UTC, submitted at 13:56:39 UTC and approved at 13:56:56 UTC on 25
    October 2026 (its submit and approve actions; the approval is its last
    modification). Only the summary changed (`updated_response_keys`); the
    response's uploaded-file entries are unchanged. The paragraph, as it stands
    at the top of the summary:
    > Stopped on 25 October 2026 at gate G2 (protocol line 143). The exact
    > reference's sampler failed its acceptance criteria on G2's pilot surfaces
    > on the first attempt (9 of 18 pairs) and on the one retry the protocol
    > allows, at doubled lengths (5 of 18), the failures concentrated at the
    > lowest noise level, 0.1 volatility point. As line 197 requires, the stop,
    > its reason and the records up to it are reported in decision-log entry
    > D89 of ROADMAP.md, merged at commit
    > dc54c0e0a2fccfee106601d48e928ba20f72ac28 of
    > https://github.com/Michaellumor/roughvollab. G2's records are committed
    > under output/: p5_g2_pilot.jsonl, p5_g2_pilot_samples.npz and
    > p5_g2_report.json for the first attempt, and p5_g2r_pilot.jsonl,
    > p5_g2r_pilot_samples.npz and p5_g2r_report.json for the retry, with their
    > SHA-256 listed in D89. No hypothesis was scored: the sealed test set was
    > never drawn, no network was trained, and nothing after G2 was run. This
    > registration remains the record of what was registered on 23 October 2026
    > and is not withdrawn. A version 3 of the protocol is being developed. Its
    > changed reference sampler is designed on the pilot surfaces already seen,
    > which it will declare as such. It will be registered separately before
    > any of its gates is run, and this registration will then be updated to
    > point to it.
  - **The readback.** GET requests only, OSF's public API: `695fw` has two
    schema responses, both approved, the original `6adba36f29130ee140f7c6d4`
    (23 October 2026) and the update. The update's summary is the paragraph, a
    blank line and v2's summary exactly, with one newline added at the end; its
    justification is the paragraph with three apostrophes typographic (’)
    rather than straight ('). Neither difference changes the text. A first
    draft of the update, which replaced the summary with the paragraph instead
    of placing the paragraph above it, was seen on its review page and not
    submitted.
  - **What was seen.** Nothing new: the notice restates D89.
  - **Who did what.** The author posted and approved the update on OSF. The
    notice's text and its placement were drafted, and the update read back, in
    the Claude Code session of D89, on Claude Opus 5.5.
  - **Not written.** Any v3 work (the v3 plan follows as D91); any change to
    the draft, v1, the decision sheet or earlier entries.
  - Drafted with AI assistance; every decision confirmed by the author.
---

## Publication seeds

Four papers, submission-shaped (`OVERLEAF/P1`–`P4`; the August 2026 prep arc is
recorded as D50–D51). The identification question is
deliberately split across two independent routes (P1 from option prices, P3 from
realised variance); P2 prices; P4 supplies the convergence foundation.

- **P1 — "Identifying roughness from an option surface: from a single smile to a
  live crypto market."** Calibrates rough-Heston (El Euch–Rosenbaum CF) to an IV
  surface and asks whether *H* is identifiable from the object one actually fits.
  Verdict: a single smile cannot identify *H* (flat direction, cond ≈ 6×10⁵); a
  multi-maturity surface tightens but does not cure it (H-spread 106%→10% at clean
  data, degrading under noise); on a live Deribit BTC surface *H* is non-identifiable
  outright (rails to boundary, |flat[H]| = 0.118). Model-level corollary: rough-Heston
  under-produces crypto's crash-fear put tail even at maximal roughness. Target:
  arXiv q-fin.PR/ST → SIURO. *(Q4; Layer 4 calibration.)*
- **P2 — "Turbocharged versus multilevel Monte Carlo for rough-volatility Asian
  options."** For arithmetic-Asian options under rough Bergomi, **MLMC does not earn
  its place** — conditioning pays as single-grid "turbocharging," not multilevel
  (κ-invariant std-MC / conditional-MLMC = 0.41–0.45 < 1), with κ=1 sharpening the
  winner ~1.3–1.5×; antithetic coupling refuted. Positioned as *confirming* Giles's
  complexity theory (β ≈ 2H < γ lands in the regime the theory itself flags as
  unfavourable), not competing with it. Absorbs the former standalone P1 baseline
  (naive-MLMC pathwise bound, β ≈ 2H across H ∈ [0.05, 0.35]), per D23. Target:
  arXiv q-fin.CP → JCF. *(Q2; Layer 1b.)*
- **P3 — "When is volatility roughness identifiable? A simulation-grounded audit of
  Hurst estimation from realised variance, with application to cryptocurrency."**
  Identifiability map over (η, Δ) + real-asset placement (BTC/ETH/SPX all
  non-identified at calibrated η̂); three estimators with intrinsic small-H biases
  (GJR & Cont–Das up, MF-DFA down). Content-complete (LaTeX, ~10pp, 2 figures,
  2 tables, 16 verified refs, AI-use statement). Target: arXiv q-fin.ST → SIURO.
  *(Q1; Layer 1c + Phase B.)*
- **P4 — "Weak convergence is faster than strong: measuring the weak order of a
  hybrid scheme."** Measures the *weak* order of the direct rough-Heston
  discretisation against the Gaussian-volatility weak-rate theory (Gassiat; FSW;
  BHT; BFN; BJP — none of which covers square-root Volterra dynamics): weak
  convergence far outpaces the throttled strong rate (H), reproducing the proved
  phase structure at H=1/6, with the sub-transition exponent stated as an
  observation above the theory line; plus a characterised negative result on the
  rank-one-noise lift as a weak-order surrogate (situated against Bayer–Breneis).
  Underpins P2's single-grid conclusion. Title/subtitle update to match the
  reframe still pending. Target: arXiv q-fin.CP / math.NA. *(Foundational;
  Layer 4 convergence.)*

- **P5 (protocol v1 frozen 2026-10-02; v2 frozen and registered 2026-10-23) —
  "Does a learned calibrator know what it cannot know?"** Pre-registered. v1,
  `docs/protocols/P5_protocol.md`, was frozen before any P5 code existed and
  registered on OSF as https://doi.org/10.17605/OSF.IO/APSF7. v2,
  `docs/protocols/P5_protocol_v2_draft.md`, supersedes it: frozen at `59ca589`
  and registered as https://doi.org/10.17605/OSF.IO/695FW with the files it
  binds, each byte-identical to its blob; APSF7 points to it and is not
  withdrawn (D55, D77, D86). v2 asks whether a learned calibrator, a deep
  ensemble or a neural ratio estimator, is more prior-driven than the exact
  posterior under its own training prior, scoring each against that posterior
  on a frozen Chebyshev emulator of the rough Heston surface, with gates G0 to
  G3, three noise levels and two priors on H; a two-step optimiser through the
  emulator remains an exploratory, labelled analysis. G0 and G1 passed (D62,
  D88); v2 stopped at G2 on 25 October 2026, its reference sampler failing to
  converge at the lowest noise level on both attempts (D89); a continuation
  would be a new, separately registered version. Targets unchanged: arXiv
  q-fin.CP, then SIFIN, Quantitative Finance or ACM ICAIF.

- **Do the P-measure and Q-measure roughness readings agree?** Realised-variance
  estimation reads H under the physical measure; surface calibration reads it
  under the pricing measure. Pathwise regularity is invariant under equivalent
  Girsanov-type measure changes, so a disagreement between the two readings is
  an estimator artefact, not a measure effect. Open question: do the two
  readings agree once *both* identifiability regions are accounted for?
  Requires the P-side audit and the Q-side audit as inputs. No date.

- Long-range: MLMC for market-risk measures (nested estimation) — aligns with the
  PhD direction.

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
