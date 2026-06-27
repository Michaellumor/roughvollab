# RoughVolLab

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20research-orange)

**An open-source research programme on rough stochastic volatility — built on the principle of honest measurement over confident numbers.**

RoughVolLab interrogates the rough-volatility paradigm rather than assuming it. The paradigm holds that log-volatility behaves as a fractional process with a small Hurst exponent (H ≈ 0.1). This project asks **three questions** about that claim and reports the honest answer to each — two of which are negative, and deliberately so. It is an independent research programme by a mathematics undergraduate at the University of Salford, built to publication standard, where every numerical claim is backed by a committed, reproducible run.

> The unifying question: *is what we're seeing real, or an artefact of how we looked?*

### Three questions, three honest answers

**1. Is the roughness identifiable from data? — *Often, no.***
Roughness is measured through a noisy, discretely-sampled proxy of latent volatility. Rather than adjudicating whether measured roughness is genuine, RoughVolLab asks a prior question: *for what region of vol-of-vol and sampling parameters is the Hurst exponent identifiable at all?* Using a verified rough-Bergomi simulator, three roughness estimators (structure-function regression, a model-free *p*-variation index, and multifractal DFA) are characterised across the parameter space and formalised into an *identifiability map*. **Finding:** an identifiable region exists, but it is narrow: even multifractal DFA — the most favourable of the three estimators — recovers roughness across only ~30% of the parameter grid (concentrated at fine sampling), and the regime real assets occupy (BTC, ETH, S&P 500, whose calibration forces high vol-of-vol) falls **outside** it, where the inversion is non-identified. This reframes the "rough vs artefact" debate as a question of what the data can support. *(Layer 1c + Phase B, below.)*

**2. Can it be priced cheaply? — *Yes, but not with the fashionable tool.***
Pricing options under the rough model needs large Monte Carlo simulations, and Multilevel Monte Carlo (MLMC) is the celebrated cost-cutting technique. **Does it pay here?** **Finding:** for arithmetic-Asian options under rough Bergomi, **MLMC does not earn its place.** A conditional ("turbocharged") *standard* Monte Carlo estimator is the method of choice — and conditioning works best as single-grid standard MC, *not* bolted onto the multilevel machinery (the decisive κ-invariant ratio std-MC / conditional-MLMC = 0.41–0.45 < 1). Exact near-cell integration (κ=1) sharpens the winner further (~1.3–1.5× cheaper) without changing the convergence rate. *(Layer 1b / P2, below.)*

**3. Can the structure be traded? — *No.***
If volatility has exploitable texture, could a reinforcement-learning agent time its execution to it and beat the classical Almgren–Chriss schedule? **Finding:** **no exploitable execution edge under linear impact.** A causal vol-reactive policy, compared on the matched-risk efficient frontier, is ~5 standard errors *worse* than Almgren–Chriss, with no advantage that grows with roughness — so deep RL was not pursued. The first run produced a convincing illusion (a look-ahead artifact); it was caught by a built-in sanity gate and corrected, and the honest negative recorded. *(Layer 2 execution arc, below.)*

### The discipline

Every claim follows the same gate-check: **state the mechanism → commit a falsifiable prediction → build/run → verify against a known answer.** Comparisons are pinned to matched accuracy / matched risk so no method wins by being sloppy, and nothing is declared "tested" without a test that names it. **Negative results are first-class outcomes** — two of the three headline findings are negative, and the value is in having earned them rather than assumed otherwise. The full prediction-and-result history lives in [`ROADMAP.md`](ROADMAP.md); the gate-check specs and recorded verdicts live in [`docs/gate_checks/`](docs/gate_checks/).

---

## Structure

| File | Layer | Status |
|------|-------|--------|
| `roughvol_core.py` | Shared rough-path engine (κ=0 Volterra), pinned by tests | ✅ 18 tests pass |
| `layer1_rough_vol.py` | fBm simulation, hybrid scheme, Hurst estimation | ✅ complete |
| `layer1b_mlmc_asian.py` | MLMC Asian pricing + complexity; opt-in antithetic / conditional / κ=1 estimator flags (P2) | ✅ complete (v0.1) + P2 |
| `layer1b_kappa1.py` | Exact near-cell (κ=1) Volterra module + coarse coupler | ✅ G-H1 / G-H2 pass |
| `layer1c_roughness_audit.py` | Roughness-estimator audit (GJR + Cont–Das + MF-DFA + corruption ladder Rungs 1–5: RV-proxy mirage + envelope; microstructure noise + subsampling; jumps + bipower; finite-sample; calendar/day-of-week seasonality) | ✅ estimators + full ladder |
| `identifiability_map.py` | Layer 1c capstone — identifiability map over (η, Δ): classifier, phase diagram, per-asset η-calibration & placement (the P3 deliverable) | ✅ 15 tests pass |
| `paper_outputs.py` | Reproducibility script — one command regenerates the P3 figures (bias curves + identifiability map with asset overlay) and prints every paper number | ✅ reuses tested modules |
| `execution_alpha.py` · `execution_alpha_phase1.py` | Execution-alpha arc (Layer 2): rough-Bergomi execution env + Almgren–Chriss + naive + causal vol-heuristic kill-switch probe | ✅ Phase 0–1 (kill-switch fired) |
| `layer2_frictions.py` | Almgren–Chriss + rough-market execution (spec: `layer2_piece1_gate_check.md`) | ✅ AC baseline built & validated in `execution_alpha.py` (G-X1, 0.7%) — dedicated `layer2_frictions.py` module not yet split out |
| `layer3_deep_hedging.py` | Deep-hedging engine — path signatures, actor–critic, CVaR objective (distinct from the Layer 2 execution arc) | 📋 Planned — still unbuilt; spec in `ROADMAP.md` |
| `layer4_convergence.py` | Convergence theorems, SPX calibration, diagnostics | 🔜 Still ahead — needs spec before code |
| `binance_data.py` · `kline_verifier.py` · `rv_series.py` | Phase B data layer: download + SHA-verify Binance klines → log-RV proxy | ✅ 66 tests pass |
| `estimate_h.py` · `interpret_h.py` | Phase B analysis: 3 estimators + de-bias vs the Rung-1 envelope | ✅ 21 tests pass |
| `equity_data.py` | Equity arm: free daily OHLC → range-based log-variance (Rung-5 gap leg) | ✅ 6 tests; run on SPX |
| `docs/gate_checks/` | Gate-check specs + recorded verdicts (the audit index) | ✅ living |

Project memory — layer specs, conventions, the dated decisions log, and all
measured results — lives in [`ROADMAP.md`](ROADMAP.md). Read it first.

The architecture is a layered stack — a shared core, the simulation layer, the
three research arcs, then the two layers still ahead — with current build status:

```mermaid
flowchart TB
    CORE["roughvol core — rough-Bergomi engine<br/>✅"]
    L1["Layer 1 — Simulation<br/>fBm, hybrid scheme, Hurst estimation<br/>✅ complete"]

    subgraph ARCS["The three research arcs — complete · two of three negative"]
        direction LR
        L1B["Layer 1b · Arc 2 — Pricing<br/>“Can we price it cheaply?”<br/>✅ MLMC doesn't pay; conditional MC wins"]
        L1C["Layer 1c · Arc 1 — Identifiability<br/>“Is the roughness real?”<br/>✅ Non-identified for real assets (BTC/ETH/SPX)"]
        L2["Layer 2 · Arc 3 — Execution<br/>“Can we trade it?”<br/>✅ No execution edge — kill-switch fired"]
    end

    L3["Layer 3 — Deep-hedging engine<br/>path signatures, actor–critic, CVaR<br/>📋 planned · still unbuilt"]
    L4["Layer 4 — Convergence and calibration<br/>convergence study, SPX calibration, diagnostics<br/>🔜 spec in progress"]

    CORE --> L1
    L1 --> ARCS
    ARCS --> L3
    L3 --> L4

    classDef done fill:#eef7f2,stroke:#1D9E75;
    classDef spec fill:#fbf3e6,stroke:#BA7517;
    classDef planned fill:#f4f4f2,stroke:#9a9994,stroke-dasharray: 5 5;

    class CORE,L1,L1B,L1C,L2 done;
    class L3 planned;
    class L4 spec;
```

---

## Quick start

```bash
git clone https://github.com/Michaellumor/roughvollab.git
cd roughvollab
pip install -r requirements.txt
python layer1_rough_vol.py
```

> `pip install roughvollab` via PyPI coming once the core modules are stable.

---

## First results — Layer 1b (June 2026)

With an exact MLMC coupling (κ=0 hybrid scheme, coarse path generated from
pairwise-summed fine Brownian increments), the measured level-variance
decay rate β tracks the pathwise bound 2H across the roughness spectrum:

| H | measured β | pathwise bound 2H |
|---|---|---|
| 0.05 | 0.13 | 0.10 |
| 0.10 | 0.23 | 0.20 |
| 0.20 | 0.42 | 0.40 |
| 0.35 | 0.72 | 0.70 |

**The bound is tight** — the Asian time-average buys no extra decay,
because the Volterra strong error acts as a slowly-decaying common factor
that averaging cannot cancel. With β ≈ 2H ≪ γ = 1 this is the worst Giles
regime, and at ε = 0.025 naive MLMC costs *more* than standard Monte Carlo
(cost ratio ≈ 0.6×). That negative result motivated the **P2 estimator
programme** — now concluded (decisions D20–D23). Its verdict: the antithetic
coupling is **refuted** (β unchanged, ~10% costlier at matched accuracy), while
**conditional Monte Carlo wins as single-grid "turbocharging," not multilevel**
(the κ-invariant ratio std-MC / conditional-MLMC = 0.41–0.45 < 1), with exact
near-cell integration (κ=1) sharpening it ~1.3–1.5× without changing β. The
honest headline: for rough-Bergomi Asian options, **MLMC does not earn its
place** — conditional standard MC on the κ=1 variance path is the method of
choice.

![Measured beta vs Hurst exponent](layer1b_beta_vs_H.png)

### Layer 1c — estimator audit (three estimators, a sharper finding)

Three independent roughness estimators are built on the same validated
engine and run on clean simulated paths with *known* Hurst exponent (the
Rung-0 oracle check): the Gatheral-Jaisson-Rosenbaum structure-function
estimator, the Cont-Das model-free *p*-variation estimator, and MF-DFA. All
three recover H across the roughness range — but they **disagree in the sign
of their small-H bias**. GJR and Cont-Das *over*-estimate roughness as
H → 0 (positive bias, roughly +0.06 to +0.07 at H = 0.05); MF-DFA
*under*-estimates (negative bias, about −0.02), and its bias is intrinsic
rather than finite-sample.

That the *direction* of the error depends on which estimator is used — on
perfect data, before any market microstructure noise enters — is concrete
evidence that small-H roughness measurements are estimator-dependent, which
speaks directly to the "fact or artefact?" debate.

The first corruption-ladder rung (the realized-volatility proxy) makes this
sharper still. Spot volatility is unobservable, so in practice it is
estimated from high-frequency price returns as realized variance over
windows. Feeding a **genuinely smooth** process (true H = 0.5) through that
proxy, all three estimators report **rough** H (≈ 0.05–0.16 at a 32-return
window — the empirical H ≈ 0.1 signature) — even though the underlying
volatility has no roughness at all. A control confirms the estimators read
the *true* smooth signal correctly (≈ 0.5), so the spurious roughness is
purely an artefact of the proxy construction; its severity is governed by
the sampling window (smaller windows → more spurious roughness).

A second corruption rung (microstructure noise) poisons the **price itself**
before any return is taken — modelling the bid-ask bounce as Y = X + η.
Differencing gives an MA(1) structure with negative autocorrelation, which
reads as roughness, so adding noise drags the estimate **down** toward
spurious roughness (a different mechanism from the proxy, with the same
outcome — they compound). The artefact grows with the noise-to-signal ratio
and afflicts smooth and rough paths alike; subsampling the price series
(taking every k-th tick) dilutes the tick-independent noise relative to the
persistent signal and partly recovers the estimate.

A third rung adds price **jumps** (compound Poisson) to a smooth null. A jump
is a local singularity; roughness is global; through a finite window the
estimators cannot tell them apart, so jumps too are misread as roughness and
collapse the estimate. **Bipower variation** (Barndorff-Nielsen–Shephard) —
pairing adjacent absolute returns so an isolated jump meets a clean
neighbour — partly recovers it. The fourth rung is **finite sample**: with
clean data but few observations, the estimate is biased — and here the
result is estimator-dependent. GJR and Cont–Das carry a roughly constant
upward bias and never fabricate false roughness from small samples, but
MF-DFA suffers a genuine downward finite-sample drift, reading an ultra-rough
process as even rougher. Unlike the other rungs, finite-sample bias has **no
mitigation** — financial history is finite — which bears directly on anyone
measuring H ≈ 0.1 from a few years of daily data. Together the four rungs map
how proxy estimation, microstructure noise, jumps, and finite samples each
distort measured roughness; see [`ROADMAP.md`](ROADMAP.md).

---

## Phase B — real-data pipeline & finding (complete)

A five-stage, fully-tested pipeline takes raw exchange data to a de-biased
roughness estimate: `binance_data.py` (download + SHA-256 verify) →
`kline_verifier.py` (data-quality diagnostics) → `rv_series.py` (log-RV proxy,
byte-identical to the Layer 1c Rung-1 object) → `estimate_h.py` (GJR + Cont–Das
+ MF-DFA, with trust signals and cross-estimator disagreement) → `interpret_h.py`
(de-bias an observed Ĥ against a *matched* Rung-1 bias envelope, recovering the
implied **true** H and flagging where the inversion is ill-posed). Runbook:
[`run_phaseb.md`](run_phaseb.md). 87 tests.

**Finding** (BTCUSDT + ETHUSDT, 2019–2025, 2,557 daily observations; full
write-up in [`PHASE_B_FINDINGS.md`](PHASE_B_FINDINGS.md)): the apparent
ultra-roughness of crypto volatility (GJR Ĥ ≈ 0.08) is real and
sampling-invariant, but **not identifiable** as a property of the latent
volatility. It is seen only by the estimator that *assumes* a rough model; the
model-free Cont–Das cannot resolve it and MF-DFA is unphysical; microstructure
noise is ruled out by the sampling sweep; and de-biasing is non-identified at
the vol-of-vol the data itself selects (calibrating η to the observed RV
variability forces η ≥ 1.5, exactly the regime where rough and smooth are
observationally equivalent through the proxy). An empirical demonstration, on
crypto, of the Cont–Das / Rogers position — with the model dependence calibrated
away rather than assumed.

The same wall holds across asset classes. An equity arm (`equity_data.py`,
Garman–Klass range variance on free S&P 500 daily OHLC) reads SPX as **less
rough** than crypto (GJR Ĥ ≈ 0.13 vs ≈ 0.08) — the expected direction for a
calendar/gap effect — yet de-biasing SPX is **non-identified too**. So the
roughness reading resists identification whether the calendar is continuous
(crypto) or gapped (equity). This is suggestive rather than a clean isolation
(it mixes a calendar difference with a range-vs-RV proxy difference); the clean
isolation is the simulated Rung 5 in `layer1c_roughness_audit.py`.

---

## Execution alpha — Layer 2 (Phase 0–1: the kill-switch)

Can the rough structure be *traded*? The execution arc asks whether a
reinforcement-learning agent could time its liquidation to the rough vol path
and beat the classical **Almgren–Chriss** schedule. **Phase 0**
(`execution_alpha.py`) builds the rough-Bergomi execution environment with
Almgren–Chriss and naive baselines, validated by gate **G-X1**: in the Markovian
limit the simulated AC frontier matches the closed-form optimum to **0.7%**, so
the environment can be trusted to evaluate a policy. **Phase 1**
(`execution_alpha_phase1.py`) is a cheap causal vol-reactive heuristic — the
kill-switch. On the matched-risk efficient frontier it is **~5 s.e. worse** than
Almgren–Chriss (gap −0.025, sign-stable across seeds), with no edge that grows
with roughness. **Verdict: under linear impact, rough-volatility structure
offers no executable execution edge** — so deep RL is not pursued. The probe's
first run produced a look-ahead artifact (a spurious "edge" *larger* in the
Markovian limit); it was caught by the H-sanity gate and corrected to a causal
schedule, with the precondition `E[inv_pnl] ≈ 0` proven before re-reading the
result. Spec and verdicts: [`docs/gate_checks/`](docs/gate_checks/).

![Execution-alpha audit dashboard](roughvollab_alpha_audit.png)

---

## Key references

Papers whose methods are implemented in the current code:

- Gatheral, Jaisson & Rosenbaum (2018). *Volatility is rough.* Quantitative Finance. — RFSV model and the structure-function roughness estimator (Layers 1, 1c).
- Bayer, Friz & Gatheral (2016). *Pricing under rough volatility.* Quantitative Finance. — the rough Bergomi model priced in Layer 1b.
- Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics. — the κ=0 hybrid scheme in `roughvol_core.py`.
- Giles (2008). *Multilevel Monte Carlo path simulation.* Operations Research. — the MLMC method underpinning Layer 1b.
- Almgren & Chriss (2001). *Optimal execution of portfolio transactions.* Journal of Risk. — the optimal-liquidation baseline in the Layer-2 execution arc.
- Cont & Das (2024). *Rough volatility: fact or artefact?* Sankhya B. — the normalised p-variation estimator and the "spurious roughness" critique that Layer 1c audits.

Planned layers (not yet implemented — listed to indicate direction):

- Buehler, Gonon, Teichmann & Wood (2019). *Deep hedging.* Quantitative Finance. — basis for the planned Layer-3 RL hedging engine (still unbuilt; distinct from the Layer-2 execution arc above).
- El Euch & Rosenbaum (2019). *The characteristic function of rough Heston models.* Mathematical Finance. — for rough Heston pricing and calibration (Layer 4).

---

## Citation

If you use RoughVolLab in your research, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be minted at the first
tagged release. A BibTeX entry is provided below for convenience:

```bibtex
@software{roughvollab2026,
  author    = {Michael Lumor},
  title     = {RoughVolLab: Simulation, pricing, and optimal control
               under rough stochastic volatility},
  year      = {2026},
  url       = {https://github.com/Michaellumor/roughvollab},
  note      = {Independent research software,
               University of Salford}
}
```

---

## Licence

MIT — see [`LICENSE`](LICENSE) for full terms.
Code is free to use, modify, and distribute with attribution.
Theoretical results (proofs, theorems) accompanying published papers
remain under standard academic copyright until journal assignment.

---

*An independent research programme in applied mathematics, built to
publication standard. Results are released incrementally as modules are
completed — see [`ROADMAP.md`](ROADMAP.md) for what is measured so far.*
