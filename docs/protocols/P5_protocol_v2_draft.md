# P5 Protocol v2 — Is a learned calibrator more prior-driven than Bayes' rule requires?

Michael Lumor

## Status

DRAFT. Supersedes v1 (frozen 2 Oct 2026, `docs/protocols/P5_protocol.md`, OSF https://doi.org/10.17605/OSF.IO/APSF7). At the time of this amendment no P5 code, pilot or trained network exists.

Reason for the amendment: v1's predictions followed from Bayes' rule and the parameter box, so an exact posterior would have confirmed them as readily as a network. v2 moves the question to the part that is uncertain, and fixes three things v1 left undefined against the repository: which grid, which noise level, and whether the pricer can afford the design.

Items marked **OPEN** are proposals. They become commitments only when the author can defend each and freezes the document. The record is the OSF registration, amended with this version; the git commit is a copy.

## What was known before freezing

The predictions below are bets made with this knowledge, and it is declared so a reader can discount accordingly.

- **D37, single smile (T = 1).** At one parameter point, H is the flat direction, degenerate with ν. H scatters by 62% at 0.1 vol point of noise.
- **D38, synthetic surface.** One parameter point, θ = (H 0.10, ν 0.35, ρ −0.70, ξ₀ 0.04), a 25-quote grid (five of D38's seven moneyness points, at each of its five maturities), a ten-draw optimiser ensemble. H scatter is about 10%, 29% and 48% of H at 0.1, 0.3 and 0.5 vol point: standard deviations of roughly 0.010, 0.029 and 0.048. κ is not identifiable on the surface either (EXP 4), so it stays fixed.
- **D39, D41, D42, live Deribit BTC and ETH.** H rails to its lower bound. Fit error is about 0.9 vol point on BTC's reduced span (D39), about 1.1 on its full one-year span (P1 paper), and 1.63 on ETH (D42). On the full-span BTC and ETH surfaces the one-year tenor is the least informative about H at each fitted optimum (D41, D42).
- **Pricer probe, 3 Oct 2026.** Timing and finiteness only; no Jacobian, Fisher information or posterior was computed. It was run by Claude in a claude.ai chat sandbox on a shallow clone of the repository, not on the author's machine, and the script is not in the repository (D55). At the default knobs (`N_riccati` 1000, 128 nodes) one 35-quote surface took about 1.6 s on one core with the per-maturity cache. Of 24 points drawn uniformly from the box (seed 7), 9 had at least one non-finite maturity. Four failing points were followed up, at the ATM quote only: all four were finite at T ≤ 0.25 with `N_riccati` 1000; T = 0.5 needed 2000 in two and T = 1 needed 4000 in two; T = 2 was finite at 2000 in two and still non-finite at 4000 in the other two.

Two consequences shape v2. At 0.1 vol point the surface probably identifies H well against a prior of standard deviation 0.054, so a single noise level would leave little flat regime to study. And the pricer is too slow and too fragile over the box to generate training sets or exact posteriors directly.

## The question

Where a surface carries little information about H, the exact posterior returns the prior. That is Bayes' rule, not a defect, and v2 does not test it.

The question is the gap between a learned calibrator and that exact posterior:

> Does an amortised calibrator for rough Heston reproduce the exact posterior of H under its own training prior, or does it lean on the prior more than the exact posterior does, and does the gap grow as the surface loses information about H?

- **Q1 (model, no network).** On a stated grid, at stated noise levels, how much does the surface tell us about H across the parameter box?
- **Q2 (network).** Given Q1, how far is each calibrator's answer from the exact posterior's answer on the same surface?

## Objects

**Parameters.** H in [0.02, 0.48], ν in [0.05, 1.00], ρ in [−0.99, 0.00], ξ₀ in [0.001, 0.25], from `layer4_calibrate_surface.py` with the H upper bound lowered from 0.49. κ fixed at 0.30, per D38 EXP 4. **OPEN — the box may be shrunk at gate G0** if the pricer cannot be made finite and converged over all of it at affordable cost. Any change is made before the emulator is built.

**Grid.** v1 said "P1's grid"; the repository has three (D37 single smile, D38 synthetic surface, D39 live Deribit). v2 uses D38's: maturities {0.10, 0.25, 0.50, 1.00, 2.00}, seven standardised moneyness points {−2, −1, −0.5, 0, 0.5, 1, 2}, 35 quotes. **OPEN — whether to drop T = 2.00.** In the 3 October probe its ATM quote was finite at `N_riccati` 2000 in two of the four failing points followed up and still non-finite at 4000 in the other two. D41 found the one-year tenor, the longest it priced, the least informative about H on live BTC at the fitted optimum; what T = 2.00 adds about H has not been measured in this repository.

**Observable.** D38 anchors strikes once from the target's noiseless ATM vol, which hands a Bayesian observer the exact ATM term structure. v2 must close that leak. **OPEN — one of:**

1. *Self-standardised (proposed).* Quote i is the implied vol at θ's own standardised moneyness. The forward map is G: θ → 35 numbers. Coherent, leak-free, keeps P1's box.
2. *Fixed absolute grid.* Fixed log-moneyness and maturities, as in the deep-calibration literature, with ξ₀ narrowed so every quote is priceable. Closer to deployed networks, further from P1.

**The model is a frozen emulator.** G̃ approximates G over the box and is built once from the pricer. The calibrators' training data and the reference posterior both use G̃. Consequences, stated plainly:

- Q2 is exact: network and reference face the same model, so their gap is the network's alone.
- Q1 describes G̃. It is claimed for rough Heston only at noise levels where gate G1 shows the emulator error is small against s.
- The pricer run uses `calibrate_btc._cached_cf`-style memoisation; `surface_model` re-solves the Riccati per strike.

**OPEN — emulator form, training-point count and design.**

**Noise.** Independent Gaussian on each quote, as a design factor: s in {0.1, 0.3, 1.0} vol point. 0.1 is P1's level; 1.0 is the size of the live misfit. Noise level is how the design reaches the flat regime, by construction. Well-specified noise only; no claim about real quotes. **OPEN — the three levels.**

**Priors.** Same support; all other parameters uniform on the box.

| | Prior A | Prior B |
| --- | --- | --- |
| H | 0.02 + 0.46 · Beta(2.5, 9) | 0.02 + 0.46 · Beta(7.5, 4) |
| Mean of H | 0.12 | 0.32 |
| SD of H | 0.054 | 0.062 |

**OPEN — the Beta parameters.**

**The exact reference.** Per scored surface and noise level: MCMC on G̃ under the uniform box prior; posteriors under A and B by importance reweighting, with a direct re-run where effective sample size falls below the floor. **OPEN — sampler, chains, length, R-hat ceiling, effective-sample floor.**

**The calibrators.** Each trained per prior and per noise level.

1. Deep ensemble of five heteroscedastic members, Gaussian negative log-likelihood. Ĥ is the ensemble mean; the 90% interval is the central Gaussian interval at the ensemble's total variance.
2. Neural ratio estimator. Ĥ is the posterior mean; the 90% interval is equal-tailed.

**OPEN — architecture, optimiser, stopping rule, ratio-estimation variant, posterior sample count, and the number of training seeds (proposed: five per calibrator, prior and noise level).**

## Stratification

Each scored (surface, noise level) pair is binned by the exact posterior's contraction of H under prior A:

c = 1 − Var_post(H | y) / Var_prior(H)

| Bin | Contraction |
| --- | --- |
| Identified | c ≥ 0.9 |
| Partial | 0.5 ≤ c < 0.9 |
| Flat | c < 0.5 |

c depends only on the observed surface and the reference. Fisher information and CRB_H are reported as descriptive companions and define nothing. **OPEN — the cut-offs.**

## Test sets

- θ uniform on the box, seed 20261002 (retained from v1); one noise draw per θ per noise level, seeds declared.
- **OPEN — the number of test θ.** Set from the reference's measured cost at G2, written in as an amendment before any network is trained.
- Pilots use other seeds and never touch the sealed set.

## Stage 1 — the identifiability map (Q1)

From the reference alone: the distribution of c per noise level, the share of pairs in each bin, and c against true (H, ν).

Descriptive, with one stop rule: a bin holding fewer than 10% of pairs makes its hypotheses untestable, and they are declared so, not adjusted. **OPEN — the 10% floor.**

## Stage 2 — fidelity to the exact posterior (Q2)

Per pair, with subscripts for the training prior:

- **Mean error.** d = Ĥ_net,A − Ĥ_exact,A.
- **Shift ratio.** S = (Ĥ_B − Ĥ_A) / 0.20, for the network and for the reference.
- **Excess prior dependence.** E = S_net − S_exact. Positive means the network leans on its prior more than Bayes' rule requires.
- **Interval mass.** m = exact posterior probability, under prior A, of the network's 90% interval. Faithful is 0.90.
- **Seed floor.** S between two networks trained on the same prior with different seeds.

Every comparison is per pair against the reference, so nothing rests on frequentist coverage within strata.

## Hypotheses

Named H-a to H-d to avoid collision with papers P1–P4. Directions and numbers are the author's bets and are **OPEN**.

| | Measure | Bin | Expected | Refuted if |
| --- | --- | --- | --- | --- |
| **H-a Faithful when informed** | median of abs(d) | Identified | below 0.01 | above 0.03 |
| **H-b Excess prior dependence** | median E | Partial | above 0.15 | below 0.05 |
| **H-c Leak into the identified regime** | median S_net | Identified | above 0.10 and above the seed floor | below 0.05 |
| **H-d Interval infidelity** | median m | Flat | outside [0.80, 0.95] | inside [0.85, 0.93] |

In the flat bin the reference itself has S near one, so excess prior dependence is not testable there; only interval fidelity is. Each hypothesis is scored per calibrator, pooled over noise levels, with the per-level breakdown reported. No aggregate verdict.

**Null.** Both calibrators match the reference within tolerance in every bin. Then the finding is that amortised calibration is faithful here and its prior dependence is exactly Bayes' rule; the practical recommendation becomes "report the contraction next to the estimate".

## Analysis plan

- 95% hierarchical bootstrap over training seeds and, within seed, pairs.
- **Confirmed**: whole interval on the expected side of the expected value. **Refuted**: whole interval beyond the refutation threshold. **Inconclusive**: otherwise.
- Pre-specified dose-response: Spearman correlation of c with abs(d), with interval.
- Exploratory, labelled: fidelity against training-prior density at true H; which ensemble variance component tracks c; a two-step optimiser through G̃.

## Gates

Pilots. No pilot output enters the results.

1. **G0 — pricer over the box.** Existing verification tests pass. On pilot points across the box, including its corners, every quote is finite and changes by less than a stated tolerance when `N_riccati` doubles. Cost per surface is measured. If this fails, the box or the grid is shrunk here. **OPEN — tolerance.**
2. **G1 — emulator.** On held-out pricer points, the 99th-percentile absolute implied-vol error e is reported. Q1 is claimed for rough Heston only at noise levels with e ≤ s / 5. **OPEN — the ratio.**
3. **G2 — reference.** On pilot surfaces the MCMC posterior of H matches a brute-force grid posterior on G̃. Cost per posterior is measured and fixes the test-set size. **OPEN — tolerance.**
4. **G3 — training adequacy.** Per calibrator, prior and noise level: ξ₀ recovered with slope above 0.95, and abs(d) on pilot surfaces stable to a doubling of the training set. Training-set size is the smallest that passes. If none within budget passes, P5 stops; architecture is not tuned against H outputs. **OPEN — tolerance.**

## Scope

In: an emulated rough Heston surface on the stated grid, well-specified noise at three levels, two calibrators, two priors, an exact reference.

Out: real surfaces, misspecified noise, other models, speed, architecture search, any correction method.

## Repository placement

An isolated leaf, as Layer 3 is: its own requirements file and virtual environment, the core staying free of torch.

## Relation to prior work

To be written from the author's own reading before freezing. Beyond v1's papers it must cover Bayesian prior-sensitivity analysis and the simulation-based-inference literature on unfaithful and misspecified amortised posteriors. The novelty claim is limited to (i) the identifiability map on a stated grid and (ii) calibrator fidelity measured against an exact posterior, stratified by contraction.

## Reporting and disclosure

- Every hypothesis reported with verdict and interval, for both calibrators.
- Any later change is a dated amendment stating what changed, why, and what had been seen.
- Code, seeds, emulator, reference posteriors and trained networks committed.
- Drafted with AI assistance (Claude); the commitments are the author's on freezing.

## Before freezing

- [ ] Prior-knowledge section checked against the ROADMAP entries it cites.
- [ ] Grid and observable convention chosen.
- [ ] Noise levels accepted or changed.
- [ ] Emulator form and gate declared.
- [ ] Beta parameters accepted or changed.
- [ ] Reference sampler and acceptance criteria declared.
- [ ] Architectures and training procedure declared.
- [ ] Contraction cut-offs accepted or changed.
- [ ] Every number in the hypotheses table defensible in the author's own words.
- [ ] Gate tolerances declared.
- [ ] Prior-work section written from the papers themselves.
- [ ] OSF registration amended.
