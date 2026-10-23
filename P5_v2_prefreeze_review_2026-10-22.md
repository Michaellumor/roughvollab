# P5 v2 — pre-freeze review, 22 October 2026

## Status

A record of review, not a decision. Nothing in the protocol, the code, the decision log or on OSF was changed by it.

## How it was done

Requested by the author on 22 October 2026 ("can you check if v2 is in good shape to be frozen"). Carried out in a claude.ai chat session on Claude Opus 5.5, on a fresh clone of `main` at `b11063b` (after D80). Three passes:

1. **Independent referee.** A fresh Claude subagent that had not seen the discussion behind v2, asked to judge only the documents (the v2 draft, v1, the amendment file, the decision sheet, D55–D80). Read-only; forbidden to run any P5 script, test or the pricer.
2. **Code-versus-protocol audit.** A second fresh subagent, same constraints. It parsed the committed G0 outputs for their fields and row counts only, and computed nothing from them.
3. **The session's own checks.** Arithmetic on the two priors as defined (no P5 data); the G0 criterion file against the box, grid and resolution the draft declares; and verification against the source of the load-bearing claims in both reports.

No P5 script, test or pricer was run in any pass.

## Verdict

**Not ready to freeze; no redesign needed.** Four of the referee's blocking issues change binding rules or code; the rest are text. The author's decisions are needed on the items listed under "Decisions" below before the edits can be made.

## Verified against the source in this session

- Referee B1: `p5_stage2.py:503` sets the null as all four hypotheses Refuted, H-a included, matching draft line 128.
- Referee B2: at n = 160,000, an A-trained network sees about 15 training surfaces with H > 0.36 (0.4 with H > 0.40), a B-trained one about 18 with H < 0.12 (2.3 with H < 0.10); those two regions hold 47.8% of uniform test H.
- Referee B3: `kish()` in `p5_g2_reference.py` is computed on the importance weights over all pooled sampling steps, with no adjustment for autocorrelation.
- Audit: the reference's generator takes the noise-level index from `enumerate` (from 0); the ratio estimator's takes `LEVEL_INDEX` (from 1); draft line 75 defines j as 1, 2, 3.
- Audit: `test_p5_g1_emulator.py:157` builds a 2×2×2×2 interpolant from the real pricer at `N_riccati` 200 on a small box around D38's point.
- Audit: `p5_stage2.py` `run` refuses without the scores file, so Stage 1 cannot be run alone.
- Audit: in `p5_g3_calibrators.py` any failing cell sets `p5_stops`, and seed training is refused unless the whole gate passed.
- Audit: the ensemble's variance over members uses `ddof=0`, the divisor 5. This is the variance of the equally weighted mixture, as in Lakshminarayanan and co-authors; the draft does not say so.
- The G0 criterion file reproduces the box ν [0.05, 0.35], ρ [−0.90, −0.70], ξ₀ [0.04, 0.25], share 0.0538, `N_riccati` 8000 at every maturity, 288 core-seconds over 54 triples, ten candidates passing on five maturities, no tie, confirmed on attempt 1.

## The session's own findings

- **Draft line 126, stale figure.** "32% below B's 1st" is the share for the superseded prior B, Beta(7.5, 4) (31.9%). Under the adopted B, Beta(235/23, 376/69), the 1st percentile is at H = 0.1875 and the share is **36.4%**. Every other figure checked in lines 66, 89, 111 and 126 reproduces: both priors' means, standard deviations, skews, modes, 90% intervals and tail masses, the crossing at H = 0.2197, 3.73 standard deviations, m = 0.9225, the width ratios 0.779 and 0.875, 0.858, and 0.954τ/0.20 = 0.00954 at τ = 0.002.
- **An undeclared run before freezing, made by Claude in this chat on 3 October 2026.** A second timing-and-finiteness probe of `layer4_calibrate_surface.py`, ATM quote only, at default knobs (`N_riccati` 1000, 128 nodes). (a) 24 points uniform on v1's box [0.02, 0.49] × [0.05, 1.00] × [−0.99, 0.00] × [0.001, 0.25] from `default_rng(20261003)`, at T ∈ {0.08, 0.25, 0.5, 1.0, 2.0}: 7 had at least one non-finite maturity, all at low H and high ν; D38's point, as a control, was finite at every maturity. (b) D39's two calibrated points, (0.02, 0.64, −0.37, 0.23) and (0.02, 0.54, −0.46, 0.23), at T ∈ {11/365, 0.1, 0.2, 0.35, 0.49}: non-finite from T = 0.2 and from T = 0.35 respectively at the default, finite at every maturity at `N_riccati` 4000. Two earlier attempts in the same sitting failed on bugs in the probe itself and priced nothing. No Jacobian, Fisher information or posterior was computed. Draft line 20 declares only the other 3 October probe, and the amendment statement's "and nothing else" (amendment line 49) does not yet hold.

## Decisions the author needs to make before the edits

1. **The null (referee B1).** Proposed: the null stands for a calibrator when H-a is Confirmed, H-b, H-c and H-d are Refuted, and H-d's interval lies within [0.85, 0.95]. Code and a test with a planted faithful calibrator to follow.
2. **The extrapolation confound (referee B2).** Whether to pre-register a per-pair coverage covariate and a secondary re-scoring on adequately covered pairs, or to state in the hypotheses and novelty claim (ii) that full-set verdicts include extrapolation outside the training support — or both.
3. **The reweighting floor (referee B3).** Proposed: (Kish / pooled steps) × ESS of H ≥ 1,000, with lines 89 and 126 corrected.
4. **G2's scope and failure (referee B4).** Whether to add the posterior mean under B and A's 5% and 95% quantiles to the tolerance, pilot θ near both ends of H, and what follows a failure.
5. **The test set (referee B5; audit).** A compute budget in core-hours and a formula for N; a minimum below which P5 stops; a cap; the noise seed; and the rule that N is fixed before any reference beyond G2's 12 is computed.
6. **Gate failures (audit; referee non-blocking 6).** What follows a stopped G1 build, and whether a partial G3 pass (some of the twelve cells) continues or stops P5.
7. **G3's training size (referee non-blocking 1).** The smallest stable n, as drafted, or the largest n within budget.
8. **Precedence and hashes (referee B6; audit).** Proposed: the scripts at the freeze commit, with the readings D72 to D75 declare, are the plan wherever the text is silent; the SHA-256 of the six scripts and both requirements files, and the numpy version, are listed in the frozen text.

## Report 1 — the independent referee (verbatim)

Paths are those of the reviewing session's clone.

**Verdict: not ready.** There are seven blocking issues. None needs a redesign, but four of them change binding rules or code.

Files: draft = /home/claude/rvl/docs/protocols/P5_protocol_v2_draft.md, v1 = …/P5_protocol.md, amendment = …/P5_v2_amendment.md, sheet = …/P5_v2_decision_sheet.md, ROADMAP = /home/claude/rvl/ROADMAP.md. The scripts are in /home/claude/rvl/.

### 2. Blocking issues

**B1. A faithful calibrator can never satisfy the null.**
- **What is wrong.** draft:128 says the null stands when "all four … are Refuted". H-a bets on fidelity: it is Refuted when the median of abs(d) is above 0.03 (draft:121), which means the calibrator is *unfaithful* where H is identified.
- A faithful calibrator (d ≈ 0, E ≈ 0, m ≈ 0.90) scores H-a Confirmed and H-b, H-c, H-d Refuted. So the stated conclusion, "faithful here … exactly Bayes' rule", can never be reached. Only a calibrator that is equally wrong under both priors can reach it.
- **Where it came from.** It was carried over from v1:151, where all four predictions bet against fidelity.
- **In the code.** p5_stage2.py:503–505 implements it as written. No test plants a faithful calibrator; the only null assertions are that it does not stand (test_p5_stage2.py:269, 319).
- **Fix.** The null stands when H-a is Confirmed, H-b, H-c and H-d are Refuted, and H-d's interval lies within [0.85, 0.95]. Change the code to match and add a test with a faithful calibrator.

**B2. The design, not the calibrators, supplies the direction of H-b and H-c (and to some extent H-a and H-d).** This is the v1 failure in a subtler form.
- **What v2 fixed.** An exact posterior gives E ≡ 0 (draft:109), so Bayes' rule alone no longer confirms these rows.
- **What it did not fix.** Test θ are uniform on H in [0.02, 0.48] (draft:93), but P_A(H > 0.40) = 2.8e-6 and P_B(H < 0.10) = 1.4e-5.
  - At G3's largest training size, n = 160,000 (draft:144), that is about 0.4 and about 2 training surfaces. Each of those tails holds 17.4% of the test θ.
  - For H > 0.36 or H < 0.12 the counts are about 15 and about 18 surfaces, and those regions hold 48% of the test θ.
- **Why that matters.** In those regions a positive E is the likely way any amortised fit fails when it has no nearby training data. The draft's own reading of Schmitt et al. already reports this kind of failure (draft:168: "a shifted or rescaled prior being among the cases detected"). The draft also calls H-c "in effect the bet that the B-trained network is not faithful … where H is identified" (draft:126).
- A Confirmed verdict therefore cannot tell "leans on its prior more than Bayes' rule requires" (draft:32) apart from "has no training data there".
- **A second confound.** If c varies with true H, the bins differ in their mix of H and so in coverage. That confounds the Partial-versus-Identified contrast and the dose-response (draft:134).
- **How the draft handles it.** draft:66 accepts this as "the design and not a defect". draft:168 sets P5 apart from "robustness to data the simulator did not generate" without noting that, for these θ, the test data are effectively of that kind.
- **Fix.** Pre-register a coverage covariate for each pair, for example each training prior's mass in the exact posterior's central 90% interval. Then do one of two things:
  - re-score H-a to H-d, with the same thresholds, on the pairs that have adequate coverage under both priors, as a pre-specified secondary analysis; or
  - state in the hypotheses and in novelty claim (ii) that verdicts on the full set include extrapolation outside the training support.

  The exploratory item at draft:135 is keyed to true H and decides nothing, so it does not cover this.

**B3. The reweighting floor does not give the effective sample size the error budgets assume.**
- **What is wrong.** draft:68 accepts reweighting when Kish's (Σw)²/Σw² is at least 1,000. That figure is computed over all 160,000 pooled, autocorrelated sampling steps (p5_g2_reference.py:330–331, 404–410).
- The chain floor allows an effective sample size (ESS) for H of 2,000, which means an autocorrelation time of up to about 80. The weighted estimator's effective size is then roughly Kish × ESS_H / 160,000, which is about 12 when both floors are just met.
- The text treats it as "the 1,000 effective draws under A", and two passages rest on that figure:
  - draft:89: the Monte Carlo error of c and the 0.02 window;
  - draft:126: H-a's 0.0005 sampler error and H-c's 0.02 scale.
- The sheet noted that Kong's formula ignores autocorrelation (sheet:494, 509). Reweighting is the default route to c and to both posterior means in S_exact.
- **Fix.** Require an effective size of at least 1,000 for the weighted chain, for example (Kish / steps) × ESS_H ≥ 1,000, or a variogram ESS of the weighted series of H. Correct draft:89 and draft:126.

**B4. G2 does not check what H-b, H-c and H-d use, and has no failure path.**
- **What is wrong.** draft:143 checks only the posterior mean of H and c under prior A (p5_g2_reference.py:516–520), on 12 surfaces drawn from 4 θ.
- E contains Ĥ_exact,B, so an error δ in it moves E by δ/0.20. An unchecked bias of 0.004 in H is H-c's whole refutation threshold (draft:123).
- m, which H-d uses, depends on the tail mass under A, and that is not checked either. Yet draft:132 says "G2 bounds it".
- No consequence of failing G2 is stated (also flagged at sheet:1290). D73 adds one only in code: any reference failure fails G2 (ROADMAP:2684–2685).
- **Fix.**
  - Add to G2's tolerance the posterior mean under B (by whichever route each pair used) and the 5% and 95% quantiles under A.
  - Include pilot θ near both ends of H, where reweighting is hardest.
  - State what happens on failure: either P5 stops, or one declared change is made and G2 is re-run on fresh pilot surfaces.

**B5. The one OPEN item has no rule, and the test-set noise seed is not declared.**
- **The number of test θ.** draft:94 sets N "from the reference's measured cost at G2" with no budget, formula, minimum or maximum.
  - N decides which bins can be tested: below N = 334, the 100-pair count governs rather than the 10% share (draft:101).
  - The rule names the reference's cost, but the log expects the calibrators' cost to be the one that binds (ROADMAP:1971–1974, 2172–2174).
- **The noise seed.** draft:93 says the noise seeds are "declared", but none appears in the draft. The generator and draw order for θ are also missing (sheet:1318). Both exist only in code (p5_g2_reference.py:441–454), and the noise seed is an argument with no default (ROADMAP:2678–2680).
- **Ordering.** "Before any network is trained" still allows G3's 48 pilot references, or test-set references, to be computed and their c values seen before N is fixed.
- **Fix.**
  - Write in now a compute budget in core-hours and a formula for N.
  - Set a minimum N below which P5 stops.
  - Name the generator and draw order, and the test-set noise seed.
  - State that N is fixed before any reference beyond G2's 12 is computed.

**B6. Choices that can change verdicts exist only in code, with no rule for which document governs.**
- The draft names no G1, G2, G3 or Stage-2 script. D72 to D75 fix readings that go "beyond the draft's words" (ROADMAP:2927–2975), including:
  - the bootstrap seed, `default_rng([127, …])` (p5_stage2.py:103); draft:132 states no seed;
  - the lower weighted median;
  - the failure floor read as "a pair failed at any seed" (p5_stage2.py:317–328); draft:75 does not say whether failures count at any seed or per seed;
  - m computed from the A-trained network (ROADMAP:2943–2944), where draft:110 says only "the network's";
  - the null band read with closed ends.
- Each of these can flip a borderline verdict or make a bin untestable. With no precedence rule, a reading can be chosen after the data are in.
- **Fix.** Add one sentence to the Analysis plan: the scripts at the freeze commit, with the readings D72 to D75 declare, are the plan wherever the text is silent. Write the bootstrap seed into draft:132 and "A-trained" into draft:110.

**B7. "What was known before freezing" is incomplete.**
- **(i) The G0 output already holds data on how quotes vary with H.** The committed G0 output holds quotes on v2's own observable and grid at all five values of H for each lattice cell (ν, ρ, ξ₀, T, z). Inside the box, p5_g0_main_quotes.csv has 8,680 rows with a finite base quote across all five H values; I counted rows and computed nothing else.
  - That is the raw material for H-sensitivity and for how the bins will fill. "Computes nothing that compares quotes across parameter values" (draft:141) describes the script, not what its output makes possible.
  - The noise levels (D65), prior B (D69), the cut-offs (D70) and the hypothesis numbers (D71) were all fixed after this output was committed (D62).
  - The log says the output was not used for identifiability (ROADMAP:1870–1871) and that it was read by scripts kept outside the repository (ROADMAP:1392–1394, 1651–1652). draft:15 should say all of this.
- **(ii) A scoping-run result is missing.** draft:21 omits the scoping run's classical-Heston known answers: errors up to 0.098 in vol, and 29 to 30 of 350 rows with no implied vol (ROADMAP:1366–1367). These lie outside the adopted box, but draft:141 promises to declare every run "whatever it showed". Also, the main run computed no known answer at the inversion settings that were adopted (p5_g0_main.py has no such step).
- **(iii) Script and test runs are undeclared.** draft:7 says "No emulator … is built before freezing" and declares only G0's scripts. In fact the G1 to Stage-2 test suites were run (draft:200). The G1 test also builds a 2×2×2×2 interpolant from the real pricer at `N_riccati` 200 on a small box around D38's point (test_p5_g1_emulator.py:157–186). This reveals nothing about outcomes, but it should be declared.
- **Fix.** These are text changes only.

### 3. Non-blocking issues

1. **G3 can certify an under-trained network** (draft:144; p5_g3_calibrators.py:448, 546–548). The check limits how much the median of abs(d) changes when the training set doubles, not how large it is. A learner that improves less than 10% per doubling passes with abs(d) ≈ 0.02, twice H-a's expected bound. Taking the smallest such n favours H-b and H-c and works against H-a. Use the largest n within budget, or also require a level.
2. **draft:126 has a stale figure.** "32% below B's 1st" was computed for the earlier prior Beta(7.5, 4) (sheet:912; ROADMAP:2437–2438). With the adopted B, the 1st percentile is at H = 0.187, so 36% of test H lies below it.
3. **draft:141 has a stale clause.** It says the emulator design is "not yet fixed", but draft:54 fixes it at 968 sets of solves.
4. **draft:158 and draft:200 are stale after D80.**
   - draft:158 still says "which this draft leaves unticked".
   - draft:200 still says the reading is "outstanding at the time of writing", and names a model only for the 11–12 October sittings.
   - D80 plans to fix both lines in the freeze commit (ROADMAP:3329–3340), but amendment:62 says the freeze commit changes nothing else. Reconcile the two.
5. **G0's test-suite condition has no recorded result.** "Existing verification tests pass" (draft:141) is not recorded in D60 to D62 or in draft:21–22. Record the commit, the test count and the outcome, or drop the condition.
6. **G1's failure path and tail rule are underspecified.**
   - A non-finite training quote "stops the build" (draft:54), but nothing says what happens next. A non-finite held-out quote is handled only in code (ROADMAP:2550–2551).
   - "Indices in the other three dimensions" (draft:142) is ambiguous; D72 resolves it as coefficient indices in code (ROADMAP:2537–2539). Write that into the text.
7. **Reference failures are dropped with no floor** (draft:101). They will cluster in ridge-shaped and flat posteriors, so the bins are scored on the pairs that survived. Report failures by noise level and by true (H, ν), and add a stop rule.
8. **"Q2 is exact" overstates** (draft:50). The gap also contains the reference's MCMC error and, for the ratio estimator, its own MCMC error.
9. **Fisher information and CRB_H are promised but undefined** (draft:89). It is not said how they are computed or at which θ, and p5_stage2.py does not compute them. Specify them or drop them.
10. **The percentile bootstrap over five seeds runs narrow** (draft:132). The draft declares this but does not correct it. A narrow interval makes both Confirmed and Refuted too easy to reach. Consider seed-level t intervals or more seeds.

### 4. Checked and found sound

- An exact posterior scores E ≡ 0, so H-b and H-c come out Refuted for it. v1's core flaw is removed.
- **G0's order of events holds in git.**
  - Scoping script merged at 08:49 UTC on 9 Oct; the scoping run launched at 09:48.
  - Main-run settings merged at 07:28 UTC on 10 Oct; the main run launched at 07:42.
  - Both output headers record those commits with a clean working tree.
  - The criterion file reproduces the box, the 0.0538 share, `N_riccati` 8000 at every maturity, 288 s over 54 triples, no tie, and confirmation on attempt 1.
- **The prior and threshold arithmetic reproduces.** This covers the tail masses, the 90% intervals, the crossing at 0.2197, skews of 0.72 and −0.30, 3.73 standard deviations, Var_prior = 0.00288, m = 0.9225, the 0.78 and 0.875 width ratios, 0.858, and 0.954τ/0.20. The one exception is the 32% in non-blocking item 2.
- **v1's registered file matches.** The v1 file at commit 13bca7e hashes to the SHA-256 that the amendment reports from OSF (5b7a67e5…).
- **No outcome objects exist yet.** The tracked `output/` folder holds only G0 files. The G2, G3 and Stage-2 scripts refuse to run without an emulator. The only exception is the cheap test build in B7(iii).

I ran no P5 script, test or pricer and modified no file. I read code, committed outputs and git metadata, and did arithmetic on the Beta priors only.

## Report 2 — the code-versus-protocol audit (verbatim)

### P5 v2 audit: does the code match the protocol?

On numbers the code matches the protocol, and no CI run can have produced a P5 result. But freezing the protocol text alone would not bind the analysis: three items are fixed by neither the text nor the code, and about a dozen choices that can change a result exist only in the code and the decision log.

I read all of the protocol, D58 to D80, the decision sheet's gap lists, the amendment file, the six scripts, the six test files, both requirements files and the CI workflow. I ran nothing except a short script that parsed the committed G0 files for their fields and row counts. Nothing was modified. "L" below is a line of `P5_protocol_v2_draft.md`; "RM" is a line of `ROADMAP.md`.

#### 1. Constants, thresholds, seeds, settings and rules

No numeric mismatch. Every value the protocol states is in the code with the same value:
- **G0 scoping run (already run):** `p5_g0_pilot.py:85-103` matches L21.
- **G0 main run:** `p5_g0_main.py:66-90, 128-148, 314-441` matches L141. That covers the U_max and node table, the ladder 2000/4000/8000 with partner 16000, the lattice, the proposed ranges, seeds 61/62, 8 points, 2 attempts, the 1,200 s cost limit, pool 4, the stop rule, and the criterion order and tie-break.
- **Box, κ 0.30, S0/r, grid, quote order, strike anchoring:** `p5_g1_emulator.py:70, 248-250` and `p5_g0_pilot.py:77-80, 241-243` match L39, L41 and L45.
- **Emulator and G1:** `p5_g1_emulator.py:71-77, 96-122, 438-452` match L54 and L142. Nodes 16/8/6/12; N_riccati 8000 with D62's settings; 200 held-out points from `default_rng(73)`; e ≤ s/5; refine if e > 0.0002.
- **Noise and priors:** noise levels `p5_g1_emulator.py:77` match L56. Priors A and B and Var_prior 0.00288, `p5_g2_reference.py:101-102`, match L62 and L66.
- **Exact reference:** `p5_g2_reference.py:88-97, 205-307, 365-424` match L68. Seed 83; 4 chains; 10,000 burn-in and 40,000 sampling steps; adaptation every 500; the 2.38²/4 scale, the 1/12 start and the 1e-10 floor; split R-hat ≤ 1.01; ESS of H ≥ 2,000; one doubled repeat; Kish ≥ 1,000.
- **G2 and test seeds:** `:89, 98-99, 466-521, 732-785` match L143. G3 pilot seeds 97/101 (`:90`) match L144. The sealed seed 20261002 (`:91, 451-454`) matches L93.
- **Calibrators:** `p5_g3_calibrators.py:94-110, 130-156, 191-287, 359-409, 582-626` match L72 to L75.
  - Trunk, members, loss, the ±1.645 interval, the ratio estimator's construction, Adam, batch 256, validation 4,000, patience 50, cap 1,000.
  - Seeds `[109,p,j,q]`, `[113,p,j]` and `[107,p,q,i,j]`, the torch seed formula and the retrain seed 6.
- **G3:** `:101-104, 433-455, 497-560` match L144. torch and numpy are pinned (`requirements-p5.txt:15,17`) as L75 requires.
- **Stage 1 and 2:** `p5_stage2.py:97-116, 196-226, 264-346, 406-553, 587-595` match L81 to L89, L101, L107 to L111 and L121 to L134. Cut-offs, window, ±0.05 re-scoring, both floors, 0.20, the measures, seed floor, H-a to H-d, the null band, 0.95, the 2,000-replicate crossed bootstrap, verdict rule and dose-response.

**One conflict: the index base in the seeded random-number generators.**
- L68 seeds the reference with `default_rng([83, i, j])` "for the i-th θ … the j-th noise level", and L75 defines j as 1, 2, 3.
- The code uses 0-based i and 0-based j for the reference (`p5_g2_reference.py:387`, indices from `:690-692`).
- It uses 0-based i and 1-based j for the ratio estimator's posterior (`p5_g3_calibrators.py:398`, `:107`).
- This is declared only in D74 (RM 2798-2802). It introduces no bias, but someone implementing from the frozen text would draw different chains.

#### 2. In the protocol, not in the code

- **G0's test-suite condition.** L141 requires the existing tests to pass on the author's machine and lists them first among what the pilot computes. Neither G0 script runs them, and no record exists: not in D60 to D62 (RM 1278-1657), not in the output headers or launch lines. Whether it was met cannot be established from the repository.
- **Fisher information and CRB_H** (L89, "descriptive companions"): no P5 script computes them. The evaluation point, noise level and derivative method would be chosen after freezing.
- **Test-set noise seed.** L93 says "seeds declared"; none is.
  - The code takes it as a required argument with no default (`p5_g2_reference.py:796, 810-811`).
  - D73 defers it to the L94 amendment (RM 2678-2679), but L94 and `P5_v2_amendment.md:29` name only the number of test θ.
  - The decision sheet flagged this at its lines 750 and 1268. It is still open.
- **Number of test θ.** Open by design (L94; `--n` has no default, `:795`). Neither text nor code has a rule turning G2's measured cost into N (sheet line 777).
- **Stage 1 declared before scoring.** L101 says the untestable verdict is declared from the reference alone, before any network is scored. `p5_stage2.py` has no Stage-1-only path: `run` refuses without the scores file (`:571-577`).
- **G1's restriction on Q1.** L51 and L142 claim Q1 for rough Heston only at noise levels G1 allows. No downstream code reads G1's verdicts; Stage 1 and 2 always use all three levels (`p5_stage2.py:92, 222-225`). This is a reporting gap only.
- **Gate failures have no stated consequence** for a failed G2 (L143) or a stopped G1 build (L54).
  - The code records both (`p5_g2_reference.py:773`; `p5_g1_emulator.py:347-352`).
  - Nothing downstream checks the G2 report; `p5_g3_calibrators.py:695-701` reads only the references.
  - Either event would leave a decision to be made after freezing, with pilot output in view.
- **Minor:**
  - The exploratory items (L135) are not implemented (D75, RM 3011).
  - Committing outputs and networks (L199) is a manual force-add, because `output/` is git-ignored.

#### 3. In the code, not in the protocol

These are ordered by how far each could move a gate, a bin or a verdict. Most are declared in D72 to D75 but not in the protocol.

1. **One failing G3 cell stops all of P5.** If any of the 12 (calibrator, prior, level) cells fails, the gate fails and seed training is refused everywhere (`p5_g3_calibrators.py:549-553, 592-593`).
   - L144 says only "If none within budget passes, P5 stops". The sheet flagged the partial case (its line 1309).
   - D74 says only "none passing reported as P5 stopping" (RM 2786-2787).
2. **A "failed pair" means failed at any seed.** For the per-calibrator floor, a pair counts as failed if any of the calibrator's 10 networks failed on it (`p5_stage2.py:323`). L75 does not say this, and it decides when the ratio estimator is untestable (D75, RM 2948-2950).
3. **The ensemble's epistemic variance divides by 5, not 4** (`p5_g3_calibrators.py:374`). L75 says only "the variance over members". Dividing by 4 would multiply that component by 1.25, which changes the interval width, and so m and H-d for the ensemble (D74, RM 2820-2821).
4. **m uses the A-trained network's interval** (`p5_stage2.py:301-306`). L110 says "the network's 90% interval". This is the only reading consistent with "Faithful is 0.90", so it is a wording gap, not a real fork (D75, RM 2943-2944).
5. **Bootstrap seed 127.** Streams are `[127, calibrator, row]`, with row 9 for the dose-response (`p5_stage2.py:103, 469, 527`). L132 names no seed (D75, RM 2927-2931). Through Monte Carlo error it can decide an interval that sits near a threshold.
6. **A failed direct run under A or B drops the whole pair as a reference failure** (`p5_g2_reference.py:412-417`; c is only set at `:422`). This removes the pair from Stage 1 and from H-a and H-d even when only B failed and c under A was available. L68 defines failure only for the uniform-prior run.
7. **Training-draw layout.** Each block draws all H, then (ν, ρ, ξ₀), then ε; every seed draws the full 320,000 and a run takes the first n (`p5_g3_calibrators.py:138-152`). L75's "seeds 2 to 5 draw the first n only" also allows a reading that gives different data (D74, RM 2802-2806).
8. **G1 refinement dimension.** The Chebyshev tail is read on the full tensor-product expansion, with ties going to the lowest index (`p5_g1_emulator.py:195-205`). L142 is ambiguous, and D72 says the two readings differ (RM 2537-2539). This picks the dimension refined, and so the final emulator.
9. **G1 error and stop rule.** e uses numpy's linear-interpolated 99th percentile (`:450`; D72, RM 2558); this matters only if e lands near 0.0002. A non-finite held-out quote also stops the build (`:347-352`), where L54 says only "a training quote" (RM 2550-2551).
10. **G2 extensions** (L143 is silent; D73, RM 2681-2685):
    - a reference failure on any of the 12 pilot surfaces fails G2 (`p5_g2_reference.py:751-754`);
    - the whole-box grid must also meet the tolerance (`:725-728`);
    - the grid's support is taken from the uniform-prior run even when the prior-A figures come from a direct run (`:401-402, 494`).
11. **Median and quantile conventions** (each matters only at a threshold):
    - point estimates, per-seed medians and bootstrap replicates use the lower weighted median (`p5_stage2.py:352-362`);
    - the seed floor (`:344`) and G3's stability check (`p5_g3_calibrators.py:446-447`) use numpy's midpoint median instead;
    - the percentile interval uses numpy's interpolation (`p5_stage2.py:432`), and verdict inequalities are strict (`:445-452`);
    - reference quantiles take the smallest sample reaching the cumulative weight, with closed intervals (`p5_g2_reference.py:322-337`);
    - m is computed from H samples stored at 16-bit precision (7e-6 resolution) with recomputed weights (`:104, 344-349`; `p5_stage2.py:275-279`).
12. **Sampler internals.** The chain's start is included in the adapted covariance, the floor is applied after scaling, and jitter is multiplied tenfold until the matrix is positive definite (`p5_g2_reference.py:164-182, 216-222`; D73).
13. **Training internals** (D74):
    - improvement must be strict, with no minimum change;
    - validation loss is the surface-weighted mean over batches;
    - the 4 h deadline starts before surfaces are priced and is checked once per epoch, which can decide a G3 budget near the limit;
    - other optimiser settings are PyTorch defaults, in float32, and the standardiser's SD is floored at 1e-12 (`p5_g3_calibrators.py:173, 244, 268-283, 302-303`).

    In G3, a ratio-estimator failure on a pilot surface excludes that surface only at the run where it failed (`:437-444, 458-471`).
14. **Environment.**
    - G1, G2 and Stage 2 run in the core environment, where numpy and scipy are unpinned (`requirements.txt:1-3`).
    - The P5 leaf leaves scipy unpinned (`requirements-p5.txt:16`).
    - G0 ran numpy 2.4.4 (output headers), while the leaf pins 2.5.4.
    - numpy does not promise identical generator streams across releases. So the sealed θ, the noise, the chains and the training draws are fixed by seed only together with the numpy version used at run time.
15. **Already spent at G0.** "Unchanged" was read as within the 1e-4 tolerance (`p5_g0_main.py:145`); the cost includes partner-only rungs and one confirming point (`:298-311`); confirming points are rounded (`:440`). All were declared in D61 and D62 and bear only on the box, which is already fixed.

#### 4. What the tests exercise, and what CI could have produced

| Test file | What it exercises |
|---|---|
| `test_p5_g0_pilot.py` | The real rough-Heston pricer, but only at D38's (H, ν, ρ) and at (0.02, 1.00, −0.99), chosen to overflow. Toy settings: N_riccati ≤ 400, ≤ 64 nodes, U_max ≤ 800, 3 strikes (`:23, 50-148`). Plus a Black–Scholes check (`:151-159`). Nothing is compared across θ. |
| `test_p5_g0_main.py` | The same real-pricer use at D38's (H, ν, ρ), N_riccati ≤ 800 (`:45-78`). The criterion and confirmation logic run on hand-built fake records (`:86-244`). |
| `test_p5_g1_emulator.py` | Chebyshev maths on analytic functions (`:43-91, 191-208`); the real pricer at D38's point (`:94-109, 126-155`); the held-out coordinates (`:112-123`). One test builds a toy emulator from the real pricer (`:157-188`). |
| `test_p5_g2_reference.py` | All MCMC, R-hat, ESS, reweighting and grid checks run on `LinearEm`, a random linear map (`:25-41, 109-189, 228-267`). The sealed set is never drawn (`:205` only inspects source). The real entry points exit before loading an emulator (`:209-213`). |
| `test_p5_g3_calibrators.py` | The numpy-only tests use `LinearEm` or hand-built numbers. The torch tests (`:117-307`) skip without torch (`:23`) and train only on `LinearEm` or a hand-built logit. |
| `test_p5_stage2.py` | Synthetic records only (`:36-93, 196-241`). |

The toy emulator test (`test_p5_g1_emulator.py:157-188`) works like this:
- It uses 2×2×2×2 nodes on a small box near D38's point (H 0.08–0.12, ν 0.30–0.35, ρ −0.75 to −0.70, ξ₀ 0.04–0.06).
- It prices at N_riccati 200, U_max 100 and 32 nodes, on 2 maturities × 3 strikes.
- It computes the toy emulator's held-out error and Chebyshev tails into a temporary directory. With two nodes, those tails are secant sensitivities of six toy quotes across the box. Nothing is printed or kept.

**Could CI have produced a P5 result? No.** No CI run can have produced a P5 posterior contraction, calibrator output or fidelity measure.
- CI installs only `requirements.txt` and pytest (`.github/workflows/tests.yml:22-26`, unchanged since 5 Jul 2026), so the torch tests skip there.
- Every MCMC in the tests targets a toy linear map.
- No emulator file exists to load.
- The Stage 2 tests use synthetic records.

The one exception is the toy emulator build above, which uses the real pricer across parameter values. It is not the P5 emulator and is not on v2's grid, box or resolution. It does contradict the claim that the tests "compare nothing across parameter values" (`p5_g1_emulator.py:22-24`; D72, RM 2521-2523). Strictly, it is an emulator of rough Heston built before freezing, which L7 says does not happen.

Outside P5, the same CI run also computes a finite-difference Jacobian, its conditioning and a calibration at D38's point, using D38's strike convention (`test_layer4_calibrate.py:60-83`; `test_layer4_calibrate_surface.py:53-76`). That is the D37/D38 knowledge already declared at L17-L18, and the tests check it with assertions rather than printing it.

In this checkout, no G1, G2, G3, reference, scores or Stage 2 output exists in `output/`, tracked or untracked.

#### 5. The committed G0 outputs

What the files contain:
- **`output/p5_g0_pilot_tasks.jsonl`** (302 lines): a header, one launch line and 300 task records.
  - The header holds the settings, the lattice, commit d394b08, a clean tree, numpy 2.4.4, scipy 1.17.1 and seed null.
  - The launch line holds solo timings and 1,750 known-answer rows (Black–Scholes and classical Heston, no Riccati solve).
  - Each task record holds, per (H, ν, ρ, T): the ATM vol, anchoring rung and seven strikes per ξ₀; per rung, solve seconds, status and failing stage; seven implied-vol quotes per ξ₀ and setting, with statuses; the characteristic function's modulus at the top node; and the three convergence comparisons with their per-quote differences.
- **`output/p5_g0_main_tasks.jsonl`** (668 lines): a header (commit 47e3bb6, seeds 61/62), two launch lines, 625 task records and 40 confirm records. Same shape, plus attempt, candidate, partner, final rung and stop reason.
- **The two `.json` files** are summaries: cost tables, stop reasons, and per-task flags, timings and ATM vols, with no quote values.
- **The two `_quotes.csv` files** have one row per (point, ξ₀, T, z, rung): 51,345 and 80,108 rows.
- **`p5_g0_main_criterion.json`**: 1,500 candidates, cost medians per (T, N), the ten passing boxes on each grid, the box taken, no tie, and the state "confirmed".

**Is any of it an outcome quantity?** No. No field is a Jacobian, Fisher information, CRB, posterior, contraction, calibration or emulator.

**Could it be turned into one? Yes.**
- The quotes are noiseless model values on v2's own observable, at settings that met G0's convergence checks.
- Inside the frozen box they cover 100 lattice points (5 H × 2 ν × 2 ρ × 5 ξ₀) and the 8 confirming points: 9,282 rows, 1,092 of them at N_riccati 8000. The scoping run adds 30 in-box points on the ρ −0.70 face.
- From them one could form secant derivatives in H and ξ₀, and in ν and ρ at the box's two faces only. That gives a Fisher information J^T J / s², and from it an approximate CRB_H or contraction at each noise level, or a crude grid posterior. It would be coarse, but it is the kind of quantity Stage 1 measures.
- Nothing in the repository shows this was done. L141, D64 and D65 say the lattice was not used for this (RM 1797-1798, 1870-1871). But D60 and D62 read the outputs with scripts kept outside the repository (RM 1392-1394, 1651-1652), so that cannot be checked from here.

One reading across parameter values was made and declared. In D61, the characteristic function's modulus across ξ₀, T, ρ and ν (RM 1451-1467) set U_max = 250/√T and dropped ξ₀ 0.001 and 0.01, which fixed the ξ₀ floor (L141; RM 1529-1533). That is a numerical property of the pricer, not information about H, but it sits uneasily with L141's "recorded and decides nothing".

#### Bottom line

Every value the protocol states matches the code; the only conflict is the 0-based index in the seeded generators, and no CI run can have produced a P5 result. Freezing the text as it stands still would not bind the whole analysis. Three things are fixed by neither text nor code and would be decided after freezing, with pilot output (pilot contractions included) in view: the sealed test set's noise seed (L93 says declared; none is), what follows a failed G2 or a stopped G1 build, and the rule turning G2's cost into N. About a dozen choices that can move a gate, a bin or a verdict exist only in the code and D72 to D75. The main ones are one failing G3 cell stopping P5, the any-seed failure rule, the ensemble's divide-by-5 variance, a failed run under prior B dropping the pair, the bootstrap seed, the G1 tail and percentile readings, and the median conventions. Before freezing: write in the noise seed and the two gate consequences, fix the i/j wording, and qualify L7 for the CI toy emulator. Then have the frozen protocol and the OSF registration name the commit and file hashes of the six scripts and both requirements files, plus the core numpy version, as part of the registered design. With that, freezing binds the analysis this code will perform; without it, the text binds the numbers but not the procedure.
