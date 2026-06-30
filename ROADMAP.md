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
| `layer3_deep_hedging.py` | Deep-hedging engine — Buehler-style direct policy optimization, CVaR objective, self-computed signatures; isolated torch venv, deletion-safe leaf. Finds: deep beats delta under frictions, roughness adds no hedging edge beyond it | ✅ built (7 tests, isolated venv; core torch-free) | 2026-06-30 |
| `rough_heston_cf.py` · `rough_heston_lifted.py` · `layer4_calibrate*.py` · `deribit_surface.py` | Rough-Heston convergence + Markovian lift (O(N·n) vs O(n²)) + high-ν pricing + calibration engine (single-smile → multi-maturity surface → live Deribit BTC); see the Layer-4 narrative + spec §5/§8 | ✅ built (D31–D39) | 2026-06-29 |
| `docs/gate_checks/` | Gate-check specs + recorded verdicts (index) | ✅ living | 2026-06-26 |
| `ROADMAP.md` | This file — project memory | living document | 2026-06-27 |

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

## Layer 1c — Roughness-estimator audit (Phase A + B complete; P3 drafted)

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

## Layer 2 — Market frictions (execution baseline built; rough-friction spec)

**Status — execution arc built (negative result).** The Almgren–Chriss
execution baseline is **built and validated** in `execution_alpha.py`: gate
**G-X1** reproduces the analytic AC frontier to **0.7%** in the Markovian limit
(the environment's "BS anchor"). A cheap causal vol-reactive execution probe
(`execution_alpha_phase1.py`) was then run as a kill-switch: on the matched-risk
efficient frontier it is **~5 s.e. worse** than AC with no edge that grows with
roughness → **no executable execution edge under linear impact; deep RL not
pursued.** See decisions **D24–D26** (including the look-ahead artifact caught
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
direct policy optimization (Buehler-style) with a CVaR objective; baselines: BS delta,
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
---

## Publication seeds

- **P1 (absorbed into P2, per D23):** the naive-MLMC pathwise-bound result
  (β ≈ 2H across H ∈ [0.05, 0.35] with exact κ=0 coupling; naive MLMC ≤
  standard MC in the practical regime; why the Asian average fails to smooth
  the Volterra error) is **not published standalone** — it became the baseline
  section of P2 rather than its own note. Figures exist (§2, §3, §4).
- **P2 (concluded, D20–D23):** *"Turbocharged versus multilevel Monte Carlo
  for rough-volatility Asian options."* Verdict: for arithmetic-Asian options
  under rough Bergomi, **MLMC does not earn its place** — conditioning pays as
  single-grid "turbocharging," not multilevel (κ-invariant std-MC /
  conditional-MLMC = 0.41–0.45 < 1), with κ=1 sharpening the winner ~1.3–1.5×;
  antithetic coupling refuted. Absorbs P1's baseline.
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
