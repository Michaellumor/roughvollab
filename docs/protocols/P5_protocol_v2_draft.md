# P5 Protocol v2 — Is a learned calibrator more prior-driven than Bayes' rule requires?

Michael Lumor

## Status

DRAFT. Supersedes v1 (frozen 2 Oct 2026, `docs/protocols/P5_protocol.md`, OSF https://doi.org/10.17605/OSF.IO/APSF7). When this draft was first recorded no P5 code, pilot, network or surrogate existed in the repository (D55, 3 Oct 2026). G0's pilot of the pricer is to be run before freezing (see Gates), so v2, unlike v1, will not be frozen ahead of all P5 code: the pilot's script and output are to be committed, and what the pilot shows is to be declared below, before freezing. No emulator, reference posterior or trained network is built before freezing.

Reason for the amendment: v1's predictions followed from Bayes' rule and the parameter box, so an exact posterior would have confirmed them as readily as a network. v2 moves the question to the part that is uncertain, and fixes three things v1 left undefined against the repository: which grid, which noise level, and whether the pricer can afford the design.

Items marked **OPEN** are proposals. They become commitments only when the author can defend each and freezes the document. The record is the OSF registration, amended with this version; the git commit is a copy.

## What was known before freezing

The predictions below are bets made with this knowledge, and it is declared so a reader can discount accordingly. G0's scoping run is declared below (D60); the main run is to be declared here before freezing. The tolerance, the comparisons, the cost limit and the criterion of G0 under Gates, and its list of what the pilot computes, were written before any such run: from reading the code, from arithmetic on recorded figures and from values this draft proposes or presumes (D59). The main run's settings are not among them.

- **D37, single smile (T = 1).** At one parameter point, H is the flat direction, degenerate with ν. H scatters by 62% at 0.1 vol point of noise.
- **D38, synthetic surface.** One parameter point, θ = (H 0.10, ν 0.35, ρ −0.70, ξ₀ 0.04), a 25-quote grid (five of D38's seven moneyness points, at each of its five maturities), a ten-draw optimiser ensemble. H scatter is about 10%, 29% and 48% of H at 0.1, 0.3 and 0.5 vol point: standard deviations of roughly 0.010, 0.029 and 0.048. κ is not identifiable on the surface either (EXP 4), so it stays fixed.
- **D39, D41, D42, live Deribit BTC and ETH.** H rails to its lower bound. Fit error is about 0.9 vol point on BTC's reduced span (D39), about 1.1 on its full one-year span (P1 paper), and 1.63 on ETH (D42). On the full-span BTC and ETH surfaces the one-year tenor is the least informative about H at each fitted optimum (D41, D42). The full-span fits put ν at 0.71 on both (D41, D42). The overflow recorded on live BTC is at low H and high ν at long maturity (D39, D41): at H = 0.02 and ν = 0.63 the one-year BTC tenor (T = 0.986) was non-finite at `N_riccati` 6000 and finite at 8000 (D41); with T pinned at about 0.989, a comment in `calibrate_btc.py` records an overflow at 8000 and the railed optimum finite from 8200.
- **Pricer probe, 3 Oct 2026.** Timing and finiteness only; no Jacobian, Fisher information or posterior was computed. It was run by Claude in a claude.ai chat sandbox on a shallow clone of the repository, not on the author's machine, and the script is not in the repository (D55). At the default knobs (`N_riccati` 1000, 128 nodes) one 35-quote surface took about 1.6 s on one core with the per-maturity cache. Of 24 points drawn uniformly from the box (seed 7), 9 had at least one non-finite maturity. Four failing points were followed up, at the ATM quote only: all four were finite at T ≤ 0.25 with `N_riccati` 1000; T = 0.5 needed 2000 in two and T = 1 needed 4000 in two; T = 2 was finite at 2000 in two and still non-finite at 4000 in the other two.
- **G0 scoping run, 9 Oct 2026 (UTC).** Made by the author on his machine from `p5_g0_pilot.py` as committed (`1c6fc13`, tree clean at `d394b08`), on a pool of four workers: 300 tasks, no errors, 13.9 h of wall (D60); no random draws, so no seed. Settings as committed: base inversion `U_max` 200 with 128 nodes, the repository's own; comparison settings (200, 256) and (400, 256); ladder `N_riccati` 1000, 2000, 4000, 8000, 16000, climbed per point and maturity until a rung met all three comparisons at tolerance 0.0001 on its 35 quotes; 60 (H, ν, ρ) lattice triples containing the 16 corners of the proposed box and D38's point, each at ξ₀ {0.001, 0.01, 0.04, 0.10, 0.25}; D38's grid, strikes standardised on each point's own ATM quote at the first rung where it was finite under the base setting, then fixed for that task; the comparison solves skipped at a rung whose base solve was not finite. Cost, with the pool running: the base solves of one (H, ν, ρ) over the five maturities took a median 4.0, 12.3, 44.4, 190 and 775 s at `N_riccati` 1000 to 16000, the largest 856 s at 16000, all under the cost limit at `U_max` 200. Riccati stage: finite from `N_riccati` 1000 at T = 0.10 everywhere but two points; at T = 1.00 and 2.00 the steps needed rise with ν and fall with H and |ρ|; six (point, maturity) pairs were non-finite through 16000, all at ν ≥ 0.65 and H ≤ 0.05 at T ≥ 1.00; at (400, 256), 26 of the 283 with a finite base solve at 16000 were, and at (200, 256) none. Inversion stage: at ξ₀ 0.001, 290 of 2,100 quotes at the top rung had no finite implied vol (a price at or below 1e-12, or the inversion failed), mostly at z = +1 and +2; the modulus of the characteristic function at the top node, just below u = 200, was still 10^−0.1 to 10^−0.3 there (medians over tasks), 10^−2.6 to 10^−10 at ξ₀ 0.04 and below 10^−16 at ξ₀ 0.25. Comparisons: `N_riccati` doubled was met on every finite pair at 8000 against 16000; the node count doubled, at the top rung, on all but 5 of 9,777; `U_max` raised, at the top rung, on 9%, 48%, 78%, 92% and 98% of finite pairs at ξ₀ 0.001, 0.01, 0.04, 0.10 and 0.25, rising with T and falling with ν; at D38's point it was met at every maturity for ξ₀ ≥ 0.10, missed by 0.00014 at ξ₀ 0.04 and T = 0.10, and by up to 0.0148 at ξ₀ 0.01. All three comparisons were met on all 35 quotes in 11 of 300 tasks, all at ν 0.05 and T ≥ 1.00; the other 289 climbed to 16000. Known answers: Black–Scholes through the base setting was off by up to 0.0058 in vol (0.0014 at the money at T = 0.10, ξ₀ 0.001, the D59 arithmetic) and by 2.2e-6 at (400, 256). No pass or fail; nothing is decided by it; the main run's settings are to be written in with this output in view (D60).

Two consequences shape v2. At 0.1 vol point the surface probably identifies H well against a prior of standard deviation 0.054, so a single noise level would leave little flat regime to study. And the pricer is too slow and too fragile over the box to generate training sets or exact posteriors directly. The first consequence rests on D38's figures, which were measured with strikes anchored once on the target and not on v2's observable.

## The question

Where a surface carries little information about H, the exact posterior returns the prior. That is Bayes' rule, not a defect, and v2 does not test it.

The question is the gap between a learned calibrator and that exact posterior:

> Does an amortised calibrator for rough Heston reproduce the exact posterior of H under its own training prior, or does it lean on the prior more than the exact posterior does, and does the gap grow as the surface loses information about H?

- **Q1 (model, no network).** On a stated grid, at stated noise levels, how much does the surface tell us about H across the parameter box?
- **Q2 (network).** Given Q1, how far is each calibrator's answer from the exact posterior's answer on the same surface?

## Objects

**Parameters.** H in [0.02, 0.48], ν in [0.05, 1.00], ρ in [−0.99, 0.00], ξ₀ in [0.001, 0.25], from `layer4_calibrate_surface.py` with the H upper bound lowered from 0.49. κ fixed at 0.30, per D38 EXP 4. **OPEN — the box.** The ranges above are the proposal. This repository does not show whether the pricer can be made finite and converged over all of them at affordable cost, and no coordinates are recorded for the failing points of the 3 October probe (D55). G0's pilot is run on these ranges before freezing (see Gates). What it showed is declared under "What was known before freezing", and the box is then written in here as values on which G0's check has passed: these ranges or narrower ones. The priors of H below and the 0.20 that separates their means are written for this range of H; a different range would need them restated before freezing. After freezing no gate changes the box.

**Grid.** v1 said "P1's grid"; the repository has three (D37 single smile, D38 synthetic surface, D39 live Deribit). v2 uses D38's: maturities {0.10, 0.25, 0.50, 1.00, 2.00}, seven standardised moneyness points {−2, −1, −0.5, 0, 0.5, 1, 2}, 35 quotes. **OPEN — whether to drop T = 2.00.** It has not been shown that T = 2.00 cannot be priced, and it is not known to be uninformative. In the 3 October probe its ATM quote was finite at `N_riccati` 2000 in two of the four failing points followed up and still non-finite at 4000 in the other two; no result above 4000 is recorded for T = 2.00. D41 found the same symptom at the one-year tenor to be a limit of resolution, and the cost of the Riccati solve grows as the square of `N_riccati`. D41 found the one-year tenor, the longest it priced, the least informative about H on live BTC at the fitted optimum; what T = 2.00 adds about H has not been measured in this repository, and G0's pilot does not measure it. G0's pilot is run on all five maturities before freezing (see Gates). What it showed is declared under "What was known before freezing", and the grid is then written in here as values on which G0's check has passed, with T = 2.00 or without it. If T = 2.00 is left out, an unmeasured amount of information is given up: D38's figures above were measured with T = 2.00 in the grid, and D38 advised five or more maturities for a market run; without T = 2.00 v2 would have four. After freezing no gate changes the grid.

**Observable.** D38 anchors strikes once from the target's noiseless ATM vol, which hands a Bayesian observer the exact ATM term structure. v2 closes that leak by the first of two options:

1. *Self-standardised (chosen).* The model is priced with spot S0 and a zero interest rate, as in `layer4_calibrate_surface.py`, so strike S0 is at the money. For maturity T and standardised moneyness point z of the grid above, the quote is the model implied vol at strike S0·exp(z·σ_ATM(θ, T)·√T), where σ_ATM(θ, T) is θ's own noiseless implied vol at strike S0 for that maturity. The forward map is G: θ → one number for each quote of that grid, ordered by maturity and then by z, both ascending. The reference and the networks take as input the noisy quotes in that order and nothing else that depends on θ; the strikes and the noiseless σ_ATM are withheld, and that is what closes the leak. The noisy quote at z = 0 is σ_ATM plus noise. Implied vols are decimals: one vol point is 0.01.
2. *Fixed absolute grid (not chosen).* Fixed log-moneyness and maturities, as in part of the deep-calibration literature, with ξ₀ narrowed so every quote is priceable. It would change the box, and no grid values or ξ₀ range were proposed. The chosen observable is further from those published networks that are trained on a fixed strike grid: no claim is made about quotes at fixed strikes, and D38's figures, measured with strikes anchored once on the target, do not carry over to the chosen observable as measured.

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
- Pilots, G0's before freezing included, use other seeds and never touch the sealed set.

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

Pilots. No pilot output enters the results. G0's pilot is run before freezing and G1 to G3 after freezing; G0's output fixes values of the design and is declared under "What was known before freezing".

1. **G0 — pricer over the box.** Existing verification tests pass: the repository's test suite, run on the author's machine. On pilot points across the box, including its corners, every quote is finite and changes by less than the tolerance in each comparison below. Cost is measured. G0 is run before freezing, as a pilot on the author's machine, from a script committed to the repository with its output; it is not the 3 October probe. *Tolerance.* 0.01 vol point (0.0001), absolute, on every quote and in each comparison separately. That is a tenth of the smallest noise level proposed under Noise and half of what G1 as proposed allows the emulator at that level. *Comparisons.* Each quote is compared with itself under refinement of each numerical setting in turn: `N_riccati` doubled; the node count doubled; and `U_max` raised, both pricings at an `N_riccati` high enough for the quote at the raised `U_max` to stay finite. Doubling `N_riccati` alone does not show an error of the inversion: both pricings use the same nodes and the same `U_max`, so such an error is, to leading order, the same in both. The comparisons are checks of convergence, not a bound on the pricer's error. *What the pilot computes.* The existing tests; timing; whether each quote is finite and, if not, at which stage of the calculation it failed; the comparisons above; the inversion applied to characteristic functions for which a reference price is available (Black–Scholes and classical Heston); and the modulus of the characteristic function at the highest node, which is recorded and decides nothing. Beyond the existing tests it computes nothing that compares quotes across parameter values: no Jacobian, Fisher information, posterior, calibration or emulator. *Two runs.* A scoping run is made first, on the box and the grid proposed under Objects, its 16 corners included. It has no pass or fail: it measures cost on the author's machine and, at each point and maturity it prices, whether and from which `N_riccati` among those it tries the quotes are finite, and which comparisons are met. Its points and the values of `N_riccati`, node count and `U_max` it tries are those of the script as committed before it is run. The main run follows; its settings are written in here before it is run, with the scoping run's output in view, and are declared as such. *Cost limit.* A resolution is admissible on a grid if, and only if, the Riccati solves for one (H, ν, ρ) over that grid's maturities, at the node count from which the comparisons start, take at most 1,200 core-seconds, as the median over the points priced, on the author's machine with the pilot's worker pool running. That is two weeks of four cores for 4,000 such sets of solves; it presumes an emulator design of that order, which is not yet fixed (see Objects). *Criterion.* The box, the grid and the resolution are taken from the main run's output by the order that follows. The range of H is not narrowed. A candidate is a box whose bounds on ν, ρ and ξ₀ are values priced in the main run and whose corners are all pilot points. A candidate passes on a grid at a resolution if that resolution is admissible on that grid and, at every pilot point in the candidate, its corners included, every quote of that grid is finite and meets every comparison, there and at every finer resolution checked (the `U_max` comparison at the `N_riccati` used for it). The candidates considered are the passing ones that contain D38's point (H 0.10, ν 0.35, ρ −0.70, ξ₀ 0.04), or all passing ones if none contains it. If any of them passes on five maturities, the grid is the five and only those are kept; otherwise the grid is the four without T = 2.00. Of those kept, the one keeping the largest share of the proposed ranges of ν, ρ and ξ₀ (the product of the three fractions) is taken, at the coarsest resolution at which it passes on that grid. It is then confirmed on fresh uniform points inside it, tested as pilot points are; the fresh points count as pilot points from then on, and if the candidate then no longer passes on that grid at that resolution the order is applied again. If no candidate passes, the check has passed on no box and grid and the draft is not frozen as it stands; anything adopted after that is declared as a judgement made on seen output. Pilot seeds are stated with each run; none is 20261002, the sealed seed, or 7, the probe's. *The main run's settings*, written in with the scoping run's output in view (D61) and before the main run was made; the script is `p5_g0_main.py` as committed. Resolutions: one `N_riccati` per maturity, from the ladder 2000, 4000, 8000 climbed per point and maturity, with a base solve at 16000 made only for the `N_riccati` comparison at 8000 and only where the base solve at 8000 is finite (16000 is not a resolution: it is tried as the partner alone, on cost, D61); a resolution is the vector of per-maturity values, and a candidate's resolution is the per-maturity smallest at which it passes, a point whose task stopped at a coarser rung by the stop rule counting as passing at every finer `N_riccati`. Inversion: `U_max` = 250/√T, rounded to 790, 500, 350, 250 and 180 at T = 0.10, 0.25, 0.50, 1.00 and 2.00, with 504, 320, 224, 160 and 112 nodes (the scoping run's spacing, to the nearest multiple of 8); the node count doubled at the same `U_max`; `U_max` doubled with the node count doubled, compared with the doubled-node setting; the `U_max` comparison made at every rung at the rung's own `N_riccati`. A task stops at the first rung at which every quote at every ξ₀ either meets all three comparisons or fails only the `U_max` comparison with the raised-`U_max` quote unchanged from the rung before; the comparison solves are skipped at a rung whose base solve is not finite. Pilot points: H {0.02, 0.05, 0.10, 0.25, 0.48} × ν {0.05, 0.35, 0.50, 0.65, 1.00} × ρ {0.00, −0.35, −0.70, −0.90, −0.99}, each at ξ₀ {0.02, 0.04, 0.07, 0.10, 0.16, 0.25}; the 16 corners of the widest candidate (ξ₀ 0.02 to 0.25) and D38's point are among them; ξ₀ 0.001 and 0.01 are not priced: by the reading of the scoping run's modulus in D61 they would need a `U_max` of about 11,000 and 1,100 at T = 0.10. Candidates: boxes with distinct lattice bounds in each of ν, ρ and ξ₀, H always [0.02, 0.48]. Cost: as defined above, formed from the run's own record as the median, over the triples priced at the resolution's `N_riccati` at every maturity of the grid, of the summed seconds of their base solves (a resolution with no such triple is not admissible). Worker pool: four. Confirming points: eight, uniform in the candidate, from numpy's `default_rng(61)`, counted only once all are priced; two confirmations attempted, the second from `default_rng(62)` on the candidate the order then gives; if both fail, no candidate passes. Tie in share: the larger share of ν, then of ρ, then of ξ₀, then the smaller lower bound on ξ₀, then on ν, then on ρ, shares compared to nine decimal places. **OPEN — after the main run: the resolution (`N_riccati`) at which the emulator is built.** It is written in after the main run. Every run made, whatever it showed, is declared under "What was known before freezing" with the settings it used, and anything set or changed after pilot output has been seen is declared as such. The box, the grid and the resolution are then written in as values on which this check has passed, and the emulator is built at that resolution and with the `U_max` and node count from which the comparisons start. After freezing no gate changes the box, the grid or the resolution; any later change is a dated amendment (see Reporting and disclosure).
2. **G1 — emulator.** On held-out pricer points, the 99th-percentile absolute implied-vol error e is reported. Q1 is claimed for rough Heston only at noise levels with e ≤ s / 5. **OPEN — the ratio.**
3. **G2 — reference.** On pilot surfaces the MCMC posterior of H matches a brute-force grid posterior on G̃. Cost per posterior is measured and fixes the test-set size. **OPEN — tolerance.**
4. **G3 — training adequacy.** Per calibrator, prior and noise level: ξ₀ recovered with slope above 0.95, and abs(d) on pilot surfaces stable to a doubling of the training set. Training-set size is the smallest that passes. If none within budget passes, P5 stops; architecture is not tuned against H outputs. **OPEN — tolerance.**

## Scope

In: an emulated rough Heston surface on the stated grid, well-specified noise at three levels, two calibrators, two priors, an exact reference.

Out: real surfaces, quotes at fixed strikes, misspecified noise, other models, speed, architecture search, any correction method.

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

- [ ] Prior-knowledge section checked against the ROADMAP entries it cites, and G0's pilot declared in it.
- [ ] Grid and observable convention chosen (the observable is chosen; the grid, the box and the resolution are written in after G0's pilot).
- [ ] Noise levels accepted or changed.
- [ ] Emulator form and gate declared.
- [ ] Beta parameters accepted or changed.
- [ ] Reference sampler and acceptance criteria declared.
- [ ] Architectures and training procedure declared.
- [ ] Contraction cut-offs accepted or changed.
- [ ] Every number in the hypotheses table defensible in the author's own words.
- [ ] Gate tolerances declared; G0's scoping script committed before the scoping run, and the main run's settings written in before the main run.
- [ ] Prior-work section written from the papers themselves.
- [ ] OSF registration amended.
