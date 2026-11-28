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
date: 25 November 2026
bibliography: paper.bib
---

# Summary

Rough volatility — the observation that realised-volatility paths behave like fractional processes with a Hurst exponent well below one half [@gatheral2018] — has become one of the most active programmes in quantitative finance [@bayer2016; @roughvolbook2024]. **RoughVolLab** is an open-source Python laboratory for research inside that paradigm, built to ask whether roughness is identifiable, priceable, tradeable and hedgeable rather than to assume it. One heavily tested simulation engine, the $\kappa=0$ hybrid-scheme Volterra engine of @bennedsen2017, underlies the estimation, pricing, execution and hedging layers: Hurst-exponent estimation from realised variance; pricing of path-dependent options by multilevel and conditional Monte Carlo; execution benchmarked against the Almgren–Chriss schedule; and deep hedging with path-signature features and a CVaR objective. A separate characteristic-function pricer underlies the calibration layer, which fits the rough Heston model to option surfaces, including a live Deribit pipeline, and the tooling of a pre-registered study (P5) that scores two neural calibrators against exact reference posteriors. The library ships with 378 tests, of which the 369 that do not need PyTorch run in continuous integration; a browser-based interactive tour of six explorers and of the P5 study; a plain-English guide for non-specialists; and a dated, append-only development log that records each significant design decision with its rationale, its result and, where one was made, its prediction.

# Statement of need

The numerics of rough volatility are delicate. The driving kernel $(t-s)^{H-1/2}$ is singular, discretisation and normalisation choices interact, and errors typically fail *silently*, producing plausible-looking paths whose statistics are quietly wrong. RoughVolLab's shared engine exists because of exactly such a failure: an early engine normalised the Volterra variance with the continuum formula $t^{2H}$ rather than the variance the discrete scheme actually produces, biasing everything built on it while the paths still looked fine. The corrected engine, `roughvol_core.py`, carries a pinned regression test that fails if the empirical variance departs from the discrete formula by more than Monte Carlo error. The estimation, execution and hedging layers draw their paths from this module, and the pricing layer and the rough Heston simulator reuse its Volterra weights.

Research on the paradigm keeps raising questions that cross layers. Does an estimator's bias survive realistic data corruption? Does a variance-reduction trick survive a change of model? Can a parameter that fits an option surface be recovered from it at all? Answering them needs estimation, simulation, pricing and calibration code that agree with one another and are each checked against known answers. RoughVolLab targets three audiences: students meeting rough volatility for the first time, for whom the interactive tour and guide teach the paradigm with no installation required; researchers who need a verified baseline engine, estimator suite and calibration stack; and practitioners and model-risk teams prototyping against live option data.

# State of the field

Open-source code for rough volatility exists, but mostly as single models inside larger libraries or as companion code to single papers. For rough Bergomi, the code accompanying @mccrickerd2018 implements the hybrid scheme with turbocharged variance reduction as one module with notebooks, without a test suite, and was last changed in 2018 [@roughbergomi]. For rough Heston, PyFENG offers a Monte Carlo scheme and characteristic-function routines within a general derivative-pricing package [@pyfeng], and QuantLib added a calibratable rough Heston model, with a Fourier pricing engine and a fractional ODE solver, in its release 1.44 of October 2026 [@quantlib]. For estimation, MFDFA [@mfdfa] and nolds [@nolds] compute detrended-fluctuation and Hurst-exponent statistics for a given series, and a reference implementation of deep hedging is published as an illustrative sample [@deephedging].

RoughVolLab builds on the same methods rather than replacing these tools. What it adds is the integration and the verification around them: Hurst estimators audited on rough-volatility paths whose roughness is known by construction, then degraded step by step towards the realised-variance proxies used in practice; Monte Carlo pricers checked against an independent characteristic-function reference, and pricing estimators compared at matched accuracy; calibrators scored against exact reference posteriors; and all layers under one test suite, so that a result in one layer can be checked against another. To the author's knowledge no maintained package covers these layers together. The layers were built and verified together, as one experimental record, from June 2026; splitting them across general libraries in other languages or under other licences would lose the cross-layer checks that are the point of the laboratory.

# Software design

RoughVolLab is organised as layers of plain Python modules over two numerical cores, and its design choices trade convenience for checkability.

- **One engine, one pricer.** The engine (`roughvol_core.py`) was extracted verbatim from the pricing layer's validated path code after the earlier engine's normalisation bug, so that one pinned test guards every simulation that imports it; its variance compensator uses the discrete variance the scheme produces, which keeps forward variance exact at every grid point. Calibration runs instead on a characteristic-function pricer (`rough_heston_cf.py`): the characteristic function of @eleuch2019, obtained from a fractional Riccati equation, with Gil-Pelaez Fourier inversion, gated on reproducing the classical Heston price at $H=1/2$. A native rough Heston simulator reuses the engine's Volterra weights, a Markovian lift [@abijaber2019] built on the kernel approximation of @bayerbreneis2023 serves as an optional simulation forward model, and weak-order convergence tooling measures both simulators against the pricer.
- **Layers.** Estimation (Layer 1c) audits the structure-function estimator of @gatheral2018, the $p$-variation estimator of @cont2024 and MF-DFA on ground-truth paths, through a five-rung corruption ladder (realised-variance proxy, microstructure noise, jumps, finite samples, calendar effects) and an identifiability map over vol-of-vol and sampling interval. Pricing (Layer 1b) compares an exactly coupled multilevel Monte Carlo estimator [@giles2008] with conditional single-grid estimators [@mccrickerd2018]. Execution (Layer 2) benchmarks causal volatility-reactive schedules on a matched-risk frontier against @almgren2001, behind a look-ahead sanity gate. Hedging (Layer 3) trains hedging policies with path-signature features under a CVaR objective [@buehler2019]. Calibration (Layer 4) fits single smiles and multi-maturity surfaces, with committed Deribit snapshots.
- **P5's tooling.** A tensor-product Chebyshev emulator of the pricer, exact reference posteriors by adaptive random-walk Metropolis on that emulator, and two neural calibrators, a deep ensemble [@lakshminarayanan2017] and a neural ratio estimator [@hermans2020], each trained under two priors on $H$.
- **A light core.** The core depends only on NumPy, SciPy and Matplotlib. PyTorch is confined to two optional environments (Layer 3 and P5), each with its own requirements file, pinned for P5; the tests that need it skip when it is absent, so the core suite runs without it.
- **Gates before results.** Results are produced through gate checks: state the mechanism, set a falsifiable prediction, build and run, and verify against a known answer (specifications and verdicts in `docs/gate_checks/` and the development log). Comparisons are made at matched accuracy or matched risk, and runs, market snapshots and seeds are committed, so that every reported number can be regenerated from the repository.

# Research impact statement

RoughVolLab is the computational basis of the author's research programme on rough volatility, and its public outputs were produced with it. The preprint on identifying roughness from an option surface [@lumor2026p1] uses the calibration layer, including the live Deribit pipeline and its committed snapshots. The preprint on when roughness is identifiable from realised variance [@lumor2026p3] uses the estimation layer, the corruption ladder and the identifiability map. The preprint on the weak order of a direct rough Heston discretisation [@lumor2026p4] uses the rough Heston simulator, the Markovian lift, the weak-order convergence tooling and the characteristic-function pricer as its reference. The pre-registered study P5 [@lumor2026p5], registered before its test set was used or any of its calibrators trained, with what had already been run and seen declared in its protocol, uses the characteristic-function pricer, the Chebyshev emulator, the exact reference posteriors and the two calibrators. The committed run records, market snapshots and seeds make the reported numbers reproducible from the repository, and the interactive tour, published at roughvollab.com with the plain-English guide, lets students and newcomers drive the main results in the browser without installing anything.

# AI usage disclosure

Generative AI was used substantially throughout this project. The only AI tool used was Anthropic's Claude Code, from the project's start on 12 June 2026. The local session records, which begin on 22 June 2026, show Claude Code versions 2.1.181 to 2.1.325 and the following Claude models writing to the repository: Claude Opus 4.8 (June–July 2026), Claude Fable 5 (July–September 2026), Claude Opus 5 (August–October 2026), Claude Fable 5.1 (October–November 2026) and Claude Opus 5.5 (October–November 2026). The models wrote and refactored the library code and its tests; ran the test suites and numerical checks on the author's machine; searched and discussed the literature; proposed candidate analyses and parameter values, among them P5's numbers, thresholds and priors, each accepted, changed or discarded by the author; drafted the documentation, the interactive tour and guide, the development-log entries and this paper; and, as separate read-only reviewer agents, checked edits and citations against their sources. Correctness is checked by the test suite (378 tests, of which the 369 that do not need PyTorch run in continuous integration), by gate checks against known answers recorded in `docs/gate_checks/` and the development log, and by checking each citation in this paper against the publisher's page, the DOI registration record or, for software, the repository.

Claude drafted the text and wrote the code from the author's decisions. The author decided the research questions, the methods and the interpretation of every result, checked every technical claim, approved the final text, and takes full responsibility for the contents of this work.

# Acknowledgements

The author thanks Dr Sabine von Hünerbein (University of Salford) for guidance and encouragement. This work received no specific funding.

# References
