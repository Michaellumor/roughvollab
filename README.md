# RoughVolLab

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20research-orange)

**RoughVolLab** is an open-source Python platform for simulation, pricing, and optimal control under rough stochastic volatility. Built on the mathematical foundation that volatility is rough — that is, driven by fractional Brownian motion with Hurst exponent H ≈ 0.1 rather than standard Brownian motion — the library provides a unified, pedagogically structured codebase spanning four research layers: (1) exact and fast O(N log N) simulation of fractional Brownian motion and rough volatility models (rough Bergomi, rough Heston); (2) multilevel Monte Carlo pricing of path-dependent derivatives including Asian options, with a rigorous complexity analysis under rough dynamics; (3) non-linear market friction modelling including Almgren-Chriss market impact and rough execution slippage; and (4) a risk-aware reinforcement learning hedging engine using path signature features to handle the non-Markovian state space that rough volatility induces. No existing open-source tool covers this full stack. RoughVolLab is designed for three audiences: students encountering rough volatility for the first time, researchers needing reproducible baselines, and practitioners building production-grade rough vol implementations. The codebase is an independent research programme by a mathematics undergraduate at the University of Salford, built to publication standard and designed to grow into doctoral research — every module is individually citable, and every numerical claim is backed by a committed, reproducible run.

---

## Structure

| File | Layer | Status |
|------|-------|--------|
| `roughvol_core.py` | Shared rough-path engine (κ=0 Volterra), pinned by tests | ✅ 18 tests pass |
| `layer1_rough_vol.py` | fBm simulation, hybrid scheme, Hurst estimation | ✅ complete |
| `layer1b_mlmc_asian.py` | MLMC Asian option pricing, complexity under roughness | ✅ complete (v0.1) |
| `layer1c_roughness_audit.py` | Roughness-estimator audit (GJR + oracle gate done; §2–4 next) | 🔄 §1 complete |
| `layer2_frictions.py` | Almgren-Chriss, rough slippage, Markov breakdown | 🔜 coming |
| `layer3_rl_hedging.py` | Path signatures, actor-critic, CVaR deep hedging | 🔜 coming |
| `layer4_convergence.py` | Convergence theorems, SPX calibration, diagnostics | 🔜 coming |

Project memory — layer specs, conventions, the dated decisions log, and all
measured results — lives in [`ROADMAP.md`](ROADMAP.md). Read it first.

Each layer is mapped to the undergraduate and postgraduate mathematics it
draws on, with current build status:

![RoughVolLab module-to-layer map](roughvollab_module_map.png)

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
(cost ratio ≈ 0.6×). That negative result is the point: it quantifies why
rough volatility needs specialised estimators, and motivates the antithetic
and conditional-MC couplings on the roadmap.

![Measured beta vs Hurst exponent](layer1b_beta_vs_H.png)

### Layer 1c — estimator audit (first finding)

Building the roughness-estimator audit on the same validated engine, the
Gatheral-Jaisson-Rosenbaum structure-function estimator was run on clean
simulated paths with *known* Hurst exponent (the Rung-0 oracle check). It
recovers H across the roughness range, but with a systematic positive
finite-lag bias that grows as H → 0 — roughly +0.06 at H = 0.05, falling to
near-zero by H = 0.3. This is a real property of the estimator on perfect
data, before any market microstructure noise enters; quantifying it, and
how it interacts with the noise of estimating volatility from returns, is
the goal of Layer 1c (see [`ROADMAP.md`](ROADMAP.md)).

---

## Key references

Papers whose methods are implemented in the current code:

- Gatheral, Jaisson & Rosenbaum (2018). *Volatility is rough.* Quantitative Finance. — RFSV model and the structure-function roughness estimator (Layers 1, 1c).
- Bayer, Friz & Gatheral (2016). *Pricing under rough volatility.* Quantitative Finance. — the rough Bergomi model priced in Layer 1b.
- Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics. — the κ=0 hybrid scheme in `roughvol_core.py`.
- Giles (2008). *Multilevel Monte Carlo path simulation.* Operations Research. — the MLMC method underpinning Layer 1b.
- Cont & Das (2022). *Rough volatility: fact or artefact?* — the normalised p-variation estimator and the "spurious roughness" critique that Layer 1c audits.

Planned layers (not yet implemented — listed to indicate direction):

- Buehler, Gonon, Teichmann & Wood (2019). *Deep hedging.* Quantitative Finance. — basis for the RL hedging engine (Layer 3).
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
