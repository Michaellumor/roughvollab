# RoughVolLab

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20research-orange)

**RoughVolLab** is an open-source Python platform for simulation, pricing, and optimal control under rough stochastic volatility. Built on the mathematical foundation that volatility is rough — that is, driven by fractional Brownian motion with Hurst exponent H ≈ 0.1 rather than standard Brownian motion — the library provides a unified, pedagogically structured codebase spanning four research layers: (1) exact and fast O(N log N) simulation of fractional Brownian motion and rough volatility models (rough Bergomi, rough Heston); (2) multilevel Monte Carlo pricing of path-dependent derivatives including Asian options, with a rigorous complexity analysis under rough dynamics; (3) non-linear market friction modelling including Almgren-Chriss market impact and rough execution slippage; and (4) a risk-aware reinforcement learning hedging engine using path signature features to handle the non-Markovian state space that rough volatility induces. No existing open-source tool covers this full stack. RoughVolLab is designed for three audiences: students encountering rough volatility for the first time, researchers needing reproducible baselines, and practitioners building production-grade rough vol implementations. The codebase is an independent research programme by a mathematics undergraduate at the University of Salford, built to publication standard and designed to grow into doctoral research — every module is individually citable, and every numerical claim is backed by a committed, reproducible run.

---

## Structure

| File | Layer | Status |
|------|-------|--------|
| `layer1_rough_vol.py` | fBm simulation, hybrid scheme, Hurst estimation | ✅ complete |
| `layer1b_mlmc_asian.py` | MLMC Asian option pricing, complexity under roughness | ✅ complete (v0.1) |
| `layer2_frictions.py` | Almgren-Chriss, rough slippage, Markov breakdown | 🔜 coming |
| `layer3_rl_hedging.py` | Path signatures, actor-critic, CVaR deep hedging | 🔜 coming |
| `layer4_convergence.py` | Convergence theorems, SPX calibration, diagnostics | 🔜 coming |

Project memory — layer specs, conventions, the dated decisions log, and all
measured results — lives in [`ROADMAP.md`](ROADMAP.md). Read it first.

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

## Key references

- Gatheral, Jaisson & Rosenbaum (2018). *Volatility is rough.* Quantitative Finance.
- Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics.
- Giles (2008). *Multilevel Monte Carlo path simulation.* Operations Research.
- Buehler et al. (2019). *Deep hedging.* Quantitative Finance.
- El Euch & Rosenbaum (2019). *The characteristic function of rough Heston models.* Mathematical Finance.

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
