---
title: 'RoughVolLab: a verification-first Python laboratory for rough stochastic volatility'
tags:
  - Python
  - quantitative finance
  - rough volatility
  - stochastic volatility
  - Monte Carlo methods
  - model calibration
  - reproducible research
authors:
  - name: Michael Lumor
    orcid: 0009-0000-0326-3891
    affiliation: 1
affiliations:
  - name: Department of Mathematics, University of Salford, United Kingdom
    index: 1
date: 4 July 2026
bibliography: paper.bib
---

# Summary

Rough volatility — the observation that realised-volatility paths behave like fractional processes with a Hurst exponent well below one half [@gatheral2018] — has become one of the most active programmes in quantitative finance [@bayer2016; @roughvolbook2024]. **RoughVolLab** is an open-source Python laboratory for working inside that paradigm. A single, heavily tested path generator — the $\kappa=0$ hybrid-scheme Volterra engine of @bennedsen2017 — underlies four layers of functionality: estimating roughness from realised-variance series; pricing path-dependent options by multilevel and conditional Monte Carlo; benchmarking execution schedules against the Almgren–Chriss solution; training and evaluating deep-hedging policies; and calibrating a rough Heston model to option surfaces via its characteristic function, including a live-market pipeline for Deribit Bitcoin options. The library ships with 254 pytest tests, a browser-based interactive tour of six explorers, a plain-English guide for non-specialists, and a development log in which every design decision was recorded — with its predicted outcome — before the result was known.

# Statement of need

The numerics of rough volatility are delicate. The driving kernel $(t-s)^{H-1/2}$ is singular, discretisation and normalisation choices interact, and errors typically fail *silently*, producing plausible-looking paths whose statistics are quietly wrong. RoughVolLab exists because of exactly such a failure: an early engine normalised the Volterra variance with the continuum formula $t^{2H}$ rather than the variance the discrete scheme actually produces, biasing everything built on it while the paths still looked fine. The corrected engine, `roughvol_core.py`, carries a pinned regression test that fails if the empirical variance ever diverges from the discrete formula, and the entire library draws its paths from this one module.

Publicly available rough-volatility code is largely a collection of single-paper artefacts — notebooks accompanying individual publications, lightly tested and rarely maintained. To the author's knowledge there is no maintained, tested, documented laboratory that spans estimation, pricing, execution, hedging and calibration over one verified engine, so cross-layer questions (does an estimator's bias survive realistic data corruption? does a variance-reduction trick survive a change of model?) currently require re-implementation each time. RoughVolLab targets three audiences: students meeting rough volatility for the first time, for whom the interactive tour and guide teach the paradigm with no installation required; researchers who need a verified baseline engine, estimator suite and calibration stack; and practitioners prototyping against live option data.

# Functionality

- **Core engine** (`roughvol_core.py`): $\kappa=0$ hybrid-scheme simulation of the Volterra process and rough Bergomi asset–variance paths [@bennedsen2017], with the discrete-variance compensator that keeps forward variance exact, and pinned tests for both.
- **Estimation** (Layer 1c): three roughness estimators — the structure-function estimator of @gatheral2018, the model-free $p$-variation estimator of @cont2024, and MF-DFA — audited on ground-truth paths, plus a four-rung "corruption ladder" that degrades ideal log-variance data towards realistic proxies so estimator behaviour can be attributed rung by rung.
- **Pricing** (Layer 1b): Asian-option pricing under rough Bergomi with an exactly coupled multilevel Monte Carlo estimator [@giles2008] and conditional single-grid ("turbocharged") estimators in the spirit of @mccrickerd2018, with cost–accuracy accounting.
- **Execution** (Layer 2): causal volatility-reactive liquidation schedules benchmarked on a matched-risk frontier against the Almgren–Chriss solution [@almgren2001], guarded by an explicit look-ahead sanity gate.
- **Hedging** (Layer 3): direct policy optimisation for hedging under rough dynamics with CVaR objectives, following @buehler2019, in an isolated optional environment whose tests skip automatically when PyTorch is absent.
- **Calibration** (Layer 4): the rough Heston characteristic function [@eleuch2019] with Fourier inversion pricing, a Markovian lift [@bayerbreneis2023], weak-order convergence tooling, single-smile and multi-maturity surface calibration, and a live Deribit BTC fetch–clean–calibrate pipeline with committed market snapshots for reproducibility.

# Verification-first design

Beyond the pinned engine test, the library is built to make silent failure hard: estimators are validated against paths whose roughness is known by construction; pricing estimators are cross-checked against independent characteristic-function references; and the execution layer's first apparent "edge" was caught by its own sanity gate as a look-ahead artefact and recorded as such. The `ROADMAP.md` development log registers each decision (D1–D46) with its predicted outcome before the experiment is run, so the repository documents not only what the software does but how its correctness was established.

# Acknowledgements

This software was developed with substantial assistance from Anthropic's Claude (large language model) for implementation and drafting, under the author's direction; the research questions, modelling decisions and verification discipline are the author's, and all behaviour is checked against known answers by the test suite. The author thanks Dr Sabine von Hünerbein (University of Salford) for guidance and encouragement.

# References
