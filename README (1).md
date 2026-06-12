# RoughVolLab

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20research-orange)
![DOI](https://zenodo.org/badge/latestdoi/roughvollab)

**RoughVolLab** is an open-source Python platform for simulation, pricing, and optimal control under rough stochastic volatility. Built on the mathematical foundation that volatility is rough — that is, driven by fractional Brownian motion with Hurst exponent H ≈ 0.1 rather than standard Brownian motion — the library provides a unified, pedagogically structured codebase spanning four research layers: (1) exact and fast O(N log N) simulation of fractional Brownian motion and rough volatility models (rough Bergomi, rough Heston); (2) multilevel Monte Carlo pricing of path-dependent derivatives including Asian options, with a rigorous complexity analysis under rough dynamics; (3) non-linear market friction modelling including Almgren-Chriss market impact and rough execution slippage; and (4) a risk-aware reinforcement learning hedging engine using path signature features to handle the non-Markovian state space that rough volatility induces. No existing open-source tool covers this full stack. RoughVolLab is designed for three audiences: students encountering rough volatility for the first time, researchers needing reproducible baselines, and practitioners building production-grade rough vol implementations. The codebase accompanies a PhD research programme at the University of Manchester / University of Oxford and will grow alongside published results — every paper will have a corresponding, citable module.

---

## Structure

| File | Layer | Status |
|------|-------|--------|
| `layer1_rough_vol.py` | fBm simulation, hybrid scheme, Hurst estimation | ✅ complete |
| `layer1b_mlmc_asian.py` | MLMC Asian option pricing, complexity theorem | 🔄 in progress |
| `layer2_frictions.py` | Almgren-Chriss, rough slippage, Markov breakdown | 🔜 coming |
| `layer3_rl_hedging.py` | Path signatures, actor-critic, CVaR deep hedging | 🔜 coming |
| `layer4_convergence.py` | Convergence theorems, SPX calibration, diagnostics | 🔜 coming |

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

## Key references

- Gatheral, Jaisson & Rosenbaum (2018). *Volatility is rough.* Quantitative Finance.
- Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics.
- Giles (2008). *Multilevel Monte Carlo path simulation.* Operations Research.
- Buehler et al. (2019). *Deep hedging.* Quantitative Finance.
- El Euch & Rosenbaum (2019). *The characteristic function of rough Heston models.* Mathematical Finance.

---

## Citation

If you use RoughVolLab in your research, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff). A BibTeX entry is provided below for convenience:

```bibtex
@software{roughvollab2025,
  author    = {Michael Lumor},
  title     = {RoughVolLab: Simulation, pricing, and optimal control
               under rough stochastic volatility},
  year      = {2025},
  url       = {https://github.com/Michaellumor/roughvollab},
  note      = {Active research software accompanying PhD thesis,
               University of Manchester / University of Oxford}
}
```

---

## Licence

MIT — see [`LICENSE`](LICENSE) for full terms.
Code is free to use, modify, and distribute with attribution.
Theoretical results (proofs, theorems) accompanying published papers
remain under standard academic copyright until journal assignment.

---

*Developed as part of a PhD research programme in Applied Mathematics.
Results will be released incrementally as papers are submitted.*
