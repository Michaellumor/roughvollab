# P5 Protocol — Does a learned calibrator know what it cannot know?

Michael Lumor

## Status

Frozen. This document is the pre-registration as of the commit that adds it to the repository, which precedes any P5 code, any trained network and any surrogate.

The git commit timestamp is the record. It proves the predictions predate the data, which is the whole of what pre-registration claims.

Every item marked **DECISION** was confirmed by the author before freezing; the checklist at the end records each confirmation. After freezing, changes are dated amendments, never silent edits.

Drafted with AI assistance (Claude), disclosed as in the programme's papers. The predictions become the author's commitment on confirmation: pre-registration rests on that timing, not on who typed the words.

## The question

When a deep neural calibrator is trained to recover rough Heston parameters from implied-volatility surfaces, is its estimate of the Hurst exponent H driven by the data — or, where the data carry little information about H, by the distribution it was trained on?

The question is stated neutrally on purpose. The direction the author expects goes in the predictions, where it can be refuted.

It sharpens the author's first framing. That framing asked whether the network reports *spurious confidence*. Thinking through how such a network is trained (Decision 1) shows the likelier failure is subtler: an answer that looks honestly uncertain, but whose content is the training prior's rather than the data's. The author's own phrase names it — the network *substitutes training-distribution priors for data identifiability*.

## Why it is open

P1 found H non-identifiable from live Deribit BTC and ETH surfaces. H is the flat direction of the calibration Jacobian, degenerate with the volatility-of-volatility; it rails to its boundary, and two equally good fits sit at different parameters. That was found empirically, not proved.

Deep calibration has taken three routes. Hernandez (2016) maps quotes straight to parameters with a feed-forward network, for simple models, assuming the inverse is well defined and attaching no uncertainty. Horvath, Muguruza and Tomas (2021) reject direct inversion for rough volatility: a network learns the forward pricing map, a deterministic optimiser fits through it, and success is judged by speed and price-space error.

Bayer, Horvath, Muguruza, Stemper and Tomas (2019; published 2025) consolidate that line, including Bayer and Stemper (2018), and add Bayesian inference. An MCMC sampler draws the posterior through the network surrogate, under a fixed uniform prior on a truncated box. For rough Bergomi the posterior concentrates on the true H in synthetic tests (their Figure 6), and near H ≈ 0.07 on SPX (Figure 7).

They read this as confirming that the surrogate is accurate enough for calibration — a validation of the network, not an investigation of identifiability. Under a uniform prior the posterior is the likelihood, truncated, so their bell on the true H is genuine evidence that H is identified in that setting.

Brigo, Huser and Leonte (2026) take the Bayesian route, and are the state of the art in prior-conditional inference. Neural ratio estimation learns the posterior of the rough Heston parameters under a pre-specified prior-predictive distribution. Posterior samples give credible intervals for exotics, and Hellinger-SHAP attributes the prior-to-posterior contraction to maturity–moneyness cells.

Both Bayesian results hold under their stated priors, and P5 does not contest them. What none of these papers asks is what a posterior or a point estimate means when the prior is *not* the truth — the situation on any real market — or how the reported uncertainty compares with the model's Fisher information. A posterior can be correct under its prior and still be prior-driven along a flat direction; only a test outside the prior can tell which.

Bayer et al.'s finding sits in real tension with P1, where H is not identified on Deribit crypto surfaces. The likeliest reading is that identifiability of H is setting-specific: it depends on the model, the quote grid and the noise. P5 therefore does not assume H is flat — its quintiles measure where it is.

If a calibrator returns its prior along a flat direction, a practitioner receives the prior presented as a market reading. That is a model-risk failure, invisible to any check run inside the training prior.

**DECISION — literature recorded.** Search done; the novelty claim is narrowed from "unmeasured" to the two gaps above. Confirmed from the author's reading: neither Bayesian paper varies its prior (Bayer et al., Sections 3.2.1 and 4.2.1; Brigo et al.), and none of the four papers computes a Jacobian, Fisher information or Cramér–Rao bound.

## The objects

Three objects, and the experiment is a comparison between the second and the third.

**1. The forward model.** Rough Heston, priced through RoughVolLab's characteristic-function engine, already verified for P1 and P4. Parameters follow P1's parameterisation.

**Free parameters — confirmed from P1's code** (`layer4_calibrate.py`, `layer4_calibrate_surface.py`). Four are calibrated: H in \[0.02, 0.49\], ν in \[0.05, 1.00\], ρ in \[−0.99, 0.00\], and the flat forward variance ξ₀ in \[0.001, 0.25\], which sets both V0 and the long-run level. Mean reversion κ is fixed at 0.30, since the code notes a single maturity cannot identify it. P5 inherits exactly this.

**2. The ground truth: the information the surface carries.** With J(θ) the Jacobian of the quoted implied vols with respect to the parameters, and quote noise of variance s² on each quote, the Fisher information is:

```latex
I(\theta) = \frac{1}{s^{2}}\, J(\theta)^{\top} J(\theta)
```

The Cramér–Rao bound then limits any *unbiased* estimator of H:

```latex
\operatorname{Var}(\hat{H}) \;\ge\; \big[ I(\theta)^{-1} \big]_{HH} =: \mathrm{CRB}_H(\theta)
```

CRB\_H is this protocol's measure of flatness: large where the surface says little about H. One caveat carries the whole design. A *biased* estimator can beat the bound — and a network pulled towards its training prior is exactly such an estimator.

**3. The learned calibrators.** Two networks trained on synthetic surfaces to recover the parameters: a point-and-variance ensemble, and a posterior estimator.

**DECISION — one-step calibrator.** P5 studies the one-step design, because prior substitution is a property of networks trained to output parameters. The two-step design calibrates by optimisation, so it inherits P1's flat objective directly. It may be run as a baseline: it should wander along the flat direction as P1 did, where the one-step network should snap to its prior. The baseline is included, as an exploratory comparison.

**DECISION — a second calibrator: a neural ratio estimator of the posterior.** Trained on the same simulations and the same two priors, it is the method family of Brigo, Huser and Leonte, so the comparison is like-for-like. It is built on RoughVolLab's own pricer with standard simulation-based-inference tooling, not their code, and passes the same G3 gate before it is scored.

Its confidence is the posterior itself: the 90% credible interval and posterior variance of H. Under its training prior it is calibrated by construction, so the standard simulation-based-calibration check should pass at home. That makes it the cleanest test of P5's question — whether calibration at home survives a change of prior.

## Decision 1 — what the network's confidence means

**DECISION — a deep ensemble of heteroscedastic members (Lakshminarayanan, Pritzel and Blundell, 2017), reporting its two components separately.** Each member outputs a mean and a variance for every parameter, trained by Gaussian negative log-likelihood; the ensemble's total variance is the mean of the members' variances plus the variance of their means.

| Source | What it measures | Along a flat direction it likely shows | In P5 |
| --- | --- | --- | --- |
| Heteroscedastic head | Noise the network believes in (aleatoric) | Roughly the training prior's width — honest *relative to the prior* | Yes, reported alone |
| Deep-ensemble spread | Disagreement between members (epistemic) | Possibly little: members sharing a prior can agree on the prior's answer | Yes, reported alone |
| Inverse-map Jacobian | How far Ĥ moves when the surface moves | A geometric sensitivity, not a probability | Diagnostic only |
| Test-time dropout | Approximate Bayesian spread | Known to be poorly calibrated | No |
| Round-trip re-pricing | Whether θ̂ reproduces the surface | Nothing: a wrong H re-prices well, which is what flatness *means* | No — blind by construction |

One consequence reshapes the whole study. A head trained by likelihood learns the posterior mean and variance *under its training prior*. So along a flat direction it should not look overconfident at all: it should report wide, prior-shaped uncertainty and centre on the prior's mean. Its failure is not false precision but **dependence on the prior** — which coverage on in-distribution data cannot detect, and which the experiments below are built to expose.

## Decision 2 — the training prior and the prior-swap pair

**DECISION — two training priors over H that differ only in where they put their mass.** Both have the same support, so neither network ever meets an H it could not have been trained on. Every other parameter's prior is identical between them.

|  | Prior A | Prior B |
| --- | --- | --- |
| Support for H | \[0.02, 0.48\] | \[0.02, 0.48\] |
| Shape | Scaled Beta, mass low | Scaled Beta, mass high |
| Mean of H | 0.12 | 0.32 |
| Other parameters | P1's box, uniform | identical to A |

The gap between the means, 0.20, is the yardstick for the prior-swap experiment. It is wide enough that a prior-driven answer cannot hide inside noise, and both means sit inside the range P1 found plausible.

**DECISION — a neutral test distribution.** Test surfaces are drawn uniformly over the shared support, from neither prior. Neither network has home advantage, so any difference between their answers on the same surface is attributable to the prior and nothing else.

**DECISION — training-set size is set by a pilot, not guessed.** It is the smallest size at which the identifiable parameters are recovered to saturation (gate G3 below). Fixing it in advance by a round number would be the kind of untested choice this protocol exists to avoid.

## Decision 3 — the noise model

**DECISION — independent Gaussian noise added to each implied-vol quote, with standard deviation s of one tenth of a volatility point.** That is the level P1 found enough to corrupt H, so P5 starts where P1 left off.

The same s is used in three places: training, testing, and the Fisher information. The network and the Cramér–Rao bound then see identical noise, so any gap between them is about the estimator, not the noise.

**DECISION — the quote grid is P1's.** Same strikes and maturities, so P5's flat direction is P1's flat direction.

A second, tenfold-cleaner noise level is optional. It would show whether the prior's pull weakens as the surface gets sharper — but it doubles every experiment, so it is deferred unless the first level's results call for it.

## Design — the experiments

Four confirmatory experiments and one exploratory one. Every result is read across **flatness quintiles**: test surfaces ranked by CRB\_H and split into five equal groups, from most identifiable to flattest. The question is never whether the network works on average, but how its behaviour changes as the surface stops carrying information about H.

Every experiment runs on both calibrators. For the posterior estimator, Ĥ is the posterior mean and its intervals are 90% credible intervals; "network A" and "network B" mean either calibrator trained on prior A or prior B.

1. **E1 — Tracking.** On the neutral test set, regress network A's Ĥ on the true H within each quintile. A slope near one means the answer follows the data; a slope near zero means it ignores the data and returns a constant.
2. **E2 — The Cramér–Rao test.** At each fixed test θ, draw many independent noise realisations and measure the spread of Ĥ. Compare it with CRB\_H. An unbiased estimator cannot fall below the bound, so spread below it is a direct signature of bias.
3. **E3 — Prior swap.** Feed the same test surfaces to networks A and B. Measure the shift ratio — the difference in their answers divided by the 0.20 gap between their priors' means. A ratio near zero means the data decide; near one, the prior does.
4. **E4 — Coverage under prior shift.** Score network A's 90% intervals twice: on test H drawn from prior A, and on test H drawn from prior B's region. A prior-dependent network should look well calibrated at home and fail away from it, and fail most where the surface is flattest.
5. **E5 — Exploratory: which confidence notices.** Correlate the ensemble spread, and separately the heteroscedastic variance, with CRB\_H across the test set. Reported, but not used to judge success.

The two-step baseline goes through E1 and E2 only, as an exploratory comparison: reported, but not used to judge any prediction. It should track poorly in flat quintiles, without snapping to a prior.

## Predictions

The author expects the network to be data-driven where the surface is informative and prior-driven where it is flat. Each prediction below commits to that in a number, and names the result that would refute it.

|  | Measure | Most identifiable quintile | Flattest quintile | Refuted if (flattest) |
| --- | --- | --- | --- | --- |
| **P1 Tracking** (E1) | Slope of Ĥ on true H | above 0.9 | below 0.3 | slope above 0.7 |
| **P2 Below the bound** (E2) | Median of spread² ÷ CRB\_H | at least 0.8 | below 0.3 | median above 0.7 |
| **P3 Prior swap** (E3) | Median shift ratio | below 0.1 | above 0.6 | median below 0.3 |
| **P4 Coverage away** (E4) | 90% interval coverage, test H from prior B | at least 85% | below 70% | coverage at least 85% |

P4 also commits to a control: at home, on test H from prior A, coverage stays between 85% and 95% in **every** quintile. If the network fails at home too, the failure is not prior dependence but something cruder, and P4's away result cannot be read as evidence for it.

The gap between each expected value and its refutation threshold is deliberate. A result inside it is reported as **inconclusive**, not stretched into a confirmation.

**DECISION — every number in this table.** These are the protocol's commitments, and the author must be able to say why each sits where it does before freezing. They are falsification thresholds, not targets: no code may be tuned to reach them.

**DECISION — the table applies to both calibrators, with the same thresholds.** For the posterior estimator the home control in P4 is the sharper test, since it should hold by construction. A pass at home and a failure away is P5's central result: calibration that survives only inside the prior it was trained on.

## Null hypothesis

The network is data-driven everywhere: its estimate of H follows the surface in every quintile, its spread never falls below the Cramér–Rao bound, swapping its prior moves nothing, and its coverage holds wherever the test H is drawn from.

If all four predictions are refuted, the null stands — and that is itself a publishable finding. It would not mean the network beats P1: an unbiased estimator along a flat direction is simply very noisy, consistent with the bound. It would mean one-step deep calibration carries no hidden prior, which practitioners would want to know. The protocol commits to reporting that outcome as fully as the expected one.

## Feasibility gates — declared pilots

Three gates run before any confirmatory experiment. They are declared here as pilots: their outputs exist only to establish that the comparison is possible, and none of them enters the results. Each has a stop rule.

1. **G1 — The ground truth exists.** On P1's own test points, the Fisher information's smallest-eigenvalue direction must be dominated by H and the volatility-of-volatility, as P1 found. If it is not, the comparison has no ground truth, and P5 stops until the discrepancy with P1 is understood.
2. **G2 — The pricer is right.** RoughVolLab's characteristic-function engine passes its existing verification tests. Already established for P1 and P4; re-run, not rebuilt.
3. **G3 — The network can learn the easy part.** Trained on prior A, it must recover the variance level with a tracking slope above 0.95 in every quintile. If it cannot learn a parameter the surface fixes cleanly, its behaviour on H tells us nothing. G3 also fixes the training-set size (Decision 2).

**DECISION — the confirmatory test set is sealed now.** Its random seed is 20261002, in the repository's date-style convention; pilots draw their surfaces from different seeds. No pilot ever sees a surface the confirmatory experiments will be scored on.

## Analysis plan

Each prediction gets one of three verdicts, decided by a 95% bootstrap interval over test surfaces rather than by the point estimate:

- **Confirmed** — the whole interval lies on the expected side of the expected value.
- **Refuted** — the whole interval lies beyond the refutation threshold.
- **Inconclusive** — anything else.

A point estimate that clears a threshold while its interval straddles it is inconclusive. That rule is the guard against reading noise as a result.

The quintile boundaries are computed from CRB\_H on the sealed test set before any network is scored, so the bins cannot be drawn around the answer.

All four confirmatory predictions are reported for both calibrators, whatever their verdict. Nothing is dropped for being unflattering, and the exploratory E5 and the two-step baseline are labelled as such wherever they appear.

**DECISION — ensemble size: five members**, the setting of Lakshminarayanan et al. **DECISION — noise realisations per test point in E2**: set in the G3 pilot, as the smallest number at which the spread estimate stops moving.

## Scope

**In scope:** synthetic rough Heston surfaces on P1's grid; two calibrators — a deep ensemble and a neural-ratio posterior — each trained on both priors, plus an exploratory two-step baseline; and the four experiments above.

**Out of scope, deliberately:** real market surfaces (a later step, once synthetic behaviour is understood); other rough models; calibration speed; architecture search beyond one declared baseline; and any method for *correcting* prior dependence. P5 measures the problem. Fixing it is GEN 3's work, and a protocol that tried both would test neither cleanly.

## Reporting commitments and disclosure

- Every confirmatory prediction is reported with its verdict and its interval, including refutations and inconclusive results.
- Any departure from this protocol after freezing is a dated amendment in the repository, stating what changed, why, and whether any data had been seen.
- Pilot outputs are reported as pilots and never pooled with confirmatory results.
- Code, seeds and trained networks are committed, so every number can be regenerated.
- AI assistance in drafting this protocol, in the code and in any paper is disclosed in the form the programme's papers already use.

## Before freezing — decisions to confirm

Tick each only when you can explain the choice in your own words. Any you cannot defend, change or question first.

- [x] Literature searched; novelty claim stands, narrowed, or P5 stopped. Hernandez (2016), Horvath et al. (2021), Brigo, Huser and Leonte (2026) and Lakshminarayanan et al. (2017) checked against the papers themselves.
- [x] Free parameters confirmed from P1's code.
- [x] One-step calibrators; two-step baseline included, as an exploratory comparison.
- [x] Confidence measure: a five-member ensemble of heteroscedastic members, both components reported.
- [x] Prior pair: support \[0.02, 0.48\], means 0.12 and 0.32, all else identical.
- [x] Neutral, uniform test distribution.
- [x] Noise: Gaussian, one tenth of a vol point per quote, on P1's grid.
- [x] Every threshold in the predictions table understood and accepted.
- [x] Sealed test-set seed written into this document.
- [x] Training-set size and noise realisations left to the pilots, as stated.
- [x] Posterior calibrator: neural ratio estimation on RoughVolLab's pricer, with standard tooling — not Brigo et al.'s code.
- [x] Bayer et al. (2019/2025) read — it consolidates Bayer and Stemper (2018) — and the literature section updated.

When every box is ticked, export this document to markdown and commit it to the repository **before any P5 code exists**. That commit is the pre-registration.
