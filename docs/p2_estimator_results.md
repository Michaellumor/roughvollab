# P2 — Improved estimators for rough-volatility MLMC

*Results note, paper-ready prose for `p2_paper.tex`. Both estimators are
opt-in flags in `layer1b_mlmc_asian.py` (`antithetic=`, `conditional=`); the
gate-checks live in `p2_antithetic_verify.py` and `p2_conditional_verify.py`.*

## Motivation

Layer 1b established the central negative result for naive coupled MLMC under
rough volatility: the variance-decay rate tracks the pathwise bound
β ≈ 2H ≪ γ = 1, the worst Giles regime, because the Volterra strong error acts
as a slowly-decaying common factor across the path that the time-average
cannot cancel. The cost crossover never materialises (standard MC is not
convincingly beaten). P2 asks whether either of the two classical
variance-reduction routes — antithetic coupling or conditional Monte Carlo —
repairs this. The answer is instructive: **antithetic does not, conditional
does — but only as a single-level estimator, not as a multilevel one.**

All experiments use the SPX-tame defaults (H = 0.10, η = 1.5, ρ = −0.7,
ξ₀ = 0.04, n₀ = 32) and an arithmetic-Asian call; seeds 7 / 11 / 23 (rates),
11 / 99 / 1234 (cost), matching the baseline.

---

## 1. Antithetic coupling (Giles–Szpruch) — a documented negative result

We form the antithetic correction
`Yₗ = ½(P_f + P_fa) − P_c`, where `P_fa` recomputes the fine payoff from the
paired fine Brownian increments swapped within each coarse step. Because the
coarse increment is the pairwise sum, it is invariant under the swap, so `P_c`
and the telescoping identity are unchanged and the estimator is exact
(`E[Yₗ]` unchanged to Monte-Carlo noise at every level).

**The textbook antithetic coupling rescales the level variance but does not
pay for itself here.** The variance-decay rate is unmoved — β stays glued to
2H across the roughness sweep (β = 0.120, 0.219, 0.418, 0.726 for
H = 0.05, 0.10, 0.20, 0.35, identical to the naive fit to three decimals and
nowhere near the 4H a rate improvement would require) — confirming that the
antithetic swap cannot touch the leading common-factor Volterra error. It
delivers only a modest constant variance reduction of ~1.45× (1.49, 1.47,
1.43, 1.37 across the same H, *decreasing* with roughness), well short of the
2× a leading-order pre-check suggests, because the orthogonal-noise and
nonlinear parts of Var(Yₗ) are not halved by the swap. Against that sits a
cost-per-sample of 2.5/1.5 = 1.67× (a second fine path per coupled level), so
the net efficiency is ~1.45/1.67 ≈ 0.87× — **antithetic is slightly worse than
naive**, and at matched finest level the adaptive cost is ~9–11% higher.

A methodological caveat worth recording: a free-running adaptive driver can
make antithetic appear up to 7× *cheaper*, but only by selecting a coarser
finest level L (2 versus 5 at ε = 0.05) than the naive run. Since the two
estimators have identical bias, the correct L is shared; the apparent saving
is a stopping-rule artifact that flips sign across seeds (cheaper on seed 11,
dearer on seeds 99 and 1234). All comparisons here therefore hold L fixed and
use the Giles optimal cost `(2/ε²)(Σₗ √(VₗCₗ))²`.

---

## 2. Conditional Monte Carlo (geometric control variate) — variance, not multilevel

We use the conditional control-variate estimator
`P_cond = arith − (geom − E[geom | W])`, where `geom` is the geometric-average
Asian payoff on the same grid and weights, and `W` is the variance-driving
Brownian path (the `dW1` increments). Conditional on `W`, the discrete
log-prices are Gaussian, so the geometric average is lognormal and
`E[geom | W]` is a closed-form Black–Scholes expression — this integrates the
orthogonal driver W⊥ out of the control exactly. The control has zero
conditional mean, so `P_cond` is unbiased for the arithmetic-Asian price with
bias identical to the plain payoff. (The closed form was validated against
brute-force conditional MC to within Monte-Carlo noise, z < 1.5.)

**Conditioning is a powerful single-level variance reducer that the multilevel
difference largely wastes.** It cuts the single-level (full-payoff) variance by
~4.2× (measured 4.16×, ~4.21× on the fine grids) at essentially no extra cost
(the path simulation is unchanged; only an O(n) closed-form post-process is
added, a measured ~1.28× per-path constant). But the *level-difference*
variance falls by only ~3.2× (measured 3.19×, ~3.30× deep), because the
coupling has already cancelled much of the common variance that conditioning
targets. Conditioning therefore helps the single-level estimator strictly more
than it helps MLMC.

The consequence is decisive at matched accuracy (finest level L\* shared by all
four estimators, ε ∈ {0.1, 0.05}, three seeds):

| ε | L\* | naive-MLMC | cond-MLMC | naive-stdMC | **cond-stdMC** |
|---|----|-----------|-----------|-------------|----------------|
| 0.10 | 2 | 1.40×10⁶ | 5.3×10⁵ | 7.8×10⁵ | **2.4×10⁵** |
| 0.05 | 5 | 4.6×10⁷ | 1.8×10⁷ | 2.5×10⁷ | **7.5×10⁶** |

The ranking is **cond-stdMC < cond-MLMC < naive-stdMC < naive-MLMC** at every
ε and seed. The key ratio `cost(cond-stdMC)/cost(cond-MLMC) ≈ 0.42–0.45 < 1`
is stable across seeds (no sign flip) and is *independent of the conditional
cost overhead κ*, since both estimators pay it per path. **Conditional standard
Monte Carlo at the finest grid is the cheapest of the four; conditional MLMC
does not beat it.**

---

## 3. Synthesis

Under rough volatility the variance route that pays is *conditioning*, not
*multilevel*. The β ≈ 2H ≪ γ pathology that defeats naive MLMC also defeats
both improved couplings *as multilevel schemes*: the antithetic swap leaves the
rate untouched and loses on cost, and conditioning — though it removes W⊥
exactly — buys less on the level differences (3.2×) than on the single level
(4.2×), so once conditioned, the residual level-difference variance still
decays too slowly to repay refinement. The practical recommendation for
arithmetic-Asian pricing under rough Bergomi at these accuracies is therefore
**plain conditional standard Monte Carlo at the finest grid**, not MLMC in any
of the three couplings tested. This sharpens, rather than overturns, the
Layer 1b message: rough volatility needs specialised *estimators*, and the
specialisation that works is conditional MC, applied single-level.

*Reproduce:* `python p2_antithetic_verify.py`, `python p2_conditional_verify.py`
(≈1–4 min each; `--quick` for a fast pass).
