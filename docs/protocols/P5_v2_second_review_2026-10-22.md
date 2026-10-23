# P5 v2 — second pre-freeze review, 22 October 2026

## Status

A record of review, not a decision. Nothing in the protocol, the code, the decision log or on OSF was changed by it.

## How it was done

This is the second, shorter review called for by step 1 of the order of acts in `docs/protocols/P5_v2_amendment.md`. It covers the changes D81 made after the first review held the same day (`P5_v2_prefreeze_review_2026-10-22.md`). It was carried out late on 22 October 2026 (UTC) in a claude.ai chat session configured for Claude Opus 5.5; the two readers ran on the session's model. The object of review was pull request #234 (`docs/p5-d81-prefreeze-fixes`, head `3ddf92a`, three commits on `main` at `b11063b`). It was fetched from GitHub into a clone, and the PR head was checked out for the readers in the session's scratch folder. Paths in the reports are relative to that checkout, and line numbers are at `3ddf92a`. Three passes:

1. **Independent referee.** A fresh Claude subagent that had not seen the discussion behind v2 or D81. It was asked to judge the changed documents against the first review: the draft, the first review's record, D81 and the amendment file, with v1, the decision sheet and D55–D80 for background.
2. **Code-versus-protocol audit.** A second fresh subagent. It checked the four changed scripts and their tests against the draft and D81, for mismatches, for bugs, for any draw of the sealed test set, and for what the tests compute.
3. **The session's own checks.** Verification against the source of the load-bearing claims in both reports, and the checks listed below.

Both readers were read-only. They were forbidden to run or import any P5 script, test or the pricer, to open the committed P5 output, and to draw random numbers from any seed the protocol names. Standalone arithmetic was allowed if it imported nothing from the repository and read no repository file, and every such computation had to be listed.

- The referee ran three snippets: the four-standard-error tolerances and their pass probabilities; the quadrature error of a grid quantile for a Gaussian; and the priors' tail masses, percentiles and well-covered range of H. It also printed the numpy and scipy versions.
- The auditor ran no Python. It summed counts of test functions with shell arithmetic and listed the names of the tracked files under `output/`.
- The session hashed the committed first-review record and read files and git metadata.

No P5 script, test or pricer was run in any pass, and no P5 output was opened.

## Verdict

**Freeze after small fixes; no redesign.** Both readers found D81's answers to the first review in place, in text and code:

- the null;
- the reweighting floor;
- G2's scope, tolerances, retry and go or no-go;
- the test set at 400, with its noise seed;
- G3 per cell, at the largest affordable n;
- Stage 1 alone, with its digest;
- the secondary analysis;
- the version guards.

The numbers agree across D81, the draft and the amendment file, and no committed test draws the sealed test set. Five things must change before freezing. Eight more are cheap now and are best made in the same round, because the scripts are hashed at the freeze and any later change to them is an amendment.

## Verified against the source in this session

- **Audit M1.**
  - `p5_g2_reference.py:141` and `p5_g3_calibrators.py:129` set the default `--emulator` to the base build.
  - No script reads G1's report.
  - `require_g2` (`p5_g2_reference.py:1024–1030`) checks G2's status alone.
  - The headers of G3 and of the scores (`p5_g3_calibrators.py:655–669`, `:753–757`) carry no emulator.
  - G2's report does refuse a record whose surfaces differ from the current emulator's (`p5_g2_reference.py:933–935`). That keeps G2 consistent with itself, but it does not choose between the base and the refined build.
- **Audit M3.** `pilot_surfaces` (`p5_g3_calibrators.py:426–441`) takes whatever pairs the G3-pilot record holds. G3 records the count and the removed references but compares them with nothing.
- **Referee blocking 1.** Draft:137 says what governs where the text is silent, and nothing about where text and script disagree. Draft:144 says "P5 stops only if" two conditions hold, but `p5_g3_calibrators.py:629–637` also stops when the scoring cost of a passing cell cannot be projected.
- **Referee blocking 2.** `p5_g0_pilot.py` imports `layer4_calibrate_surface`, `layer4_calibrate` and `rough_heston_cf`. The first two import `rough_heston`, which imports `roughvol_core`, and `layer4_calibrate.py:79–80` imports two more modules inside a function. None of these is in the hash list of draft:137 and amendment:63.
- **Referee blocking 3; audit M2.**
  - The new test at `test_p5_g1_emulator.py:256–310` launches a 2 × 1 × 1 × 2 build from pricer surfaces at D38's (H, ν, ρ) (`:266–267`, `:277–282`).
  - `test_p5_g0_main.py:69` prices (0.02, 1.00, −0.99).
  - The G0 tests price D38's (H, ν, ρ) at ξ₀ 0.02, 0.04 and 0.25; D38's own ξ₀ is 0.04 (`p5_g0_pilot.py:101`).
  - ROADMAP 3710–3719 records the readers' runs and two scratch scripts, which amendment:50's "and nothing else" leaves out.
  - Draft:200 says "the four decisions", where D81 records five.
  - The first review's referee "read code, committed outputs and git metadata" (first record, line 161), and its audit parsed the G0 files (lines 280–300). Amendment:50 says only that the first review "read the documents and the code".
- **Referee N1.**
  - Draft:143's "about 90 comparisons" counts five on each of the 18 pilot surfaces. The second grid's five on each of the six surfaces at the largest level bring the total to 120.
  - At n_eff = 1,000, four standard errors of c at c = 0.5 are 4 × 0.5 × √(2/999) = 0.089, against the absolute tolerance of 0.01.
- The committed first-review record is byte-identical to the file sent to the author (SHA-256 `29e73ceb7f033835104a480f0e93b3f5844a83f4469ee4ff64092d9968bc5c31`).
- D81's record of the author's decisions (ROADMAP 3377–3395) matches the answers the author gave in the first review's sitting: D1 to D4 at its four questions, and D5 and "limits" after the hold.

## The session's own findings

- **The first review's record sits at the repository's root.** The prompt for D81 asked for `docs/protocols/`. Every reference cites the bare file name, which resolves at the root, so nothing is broken. But the record belongs beside the protocol. Proposed: move it there, with this one.
- **Four standard errors cost more than the mean's figure shows.**
  - At the effective-size floors, G2 detects an error in c near the lower cut-off only above about 0.06 to 0.09.
  - It detects an error in a tail quantile at the Flat edge only above about 0.007 to 0.010. By draft:143's own reading, an error of that size can move m by about 0.04 to 0.055; H-d's numbers are 0.05 apart.
  - At four standard errors, G2 remains a check for gross errors: a missed mode, a chain that has not mixed, a bug. Those are the ways an MCMC reference that passes R-hat and its effective-size floor realistically fails, and the second grid is there to catch them.
  - What G2 no longer does is certify the reference, at the floors, to the precision H-d's numbers need. The text should say so where it now implies the opposite (draft:89, :132, :143). D5 itself need not change.
- **The retry's streams.** The audit's bug B3 follows draft:68 exactly. Because the streams are shared, the retry at a fixed point starts its chains where the first attempt did. A miss tied to the starts therefore repeats, and P5 stops. That errs towards stopping, not towards passing. Proposed: say so in the text, with no change to the code.

## What must change before freezing

1. **G1's outcome binds what follows** (audit M1 and bug B1).
   - A `require_g1()` used by G2 and G3 by default. It refuses when G1's build stopped or is incomplete, and it selects the refined build when the base report called for refinement.
   - The emulator's path and SHA-256 recorded in the headers of G2, G3 and the scores; Stage 2 refuses when they differ.
   - One sentence at draft:142: the refined build, once made, is G̃ for everything after G1.
2. **The hash list covers the forward model** (referee blocking 2).
   - Add every repository module the six scripts import at module level, directly or through another module: `rough_heston_cf.py`, `layer4_calibrate_surface.py`, `layer4_calibrate.py`, `rough_heston.py`, `roughvol_core.py`, and any others found beside them. The two modules imported inside a function are to be included, or their exclusion stated.
   - Say at draft:137 that every run after freezing is made from a clean tree at a commit where each listed file has its listed hash, as the commit and clean-tree fields of the output headers show.
3. **A rule for disagreement** (referee blocking 1).
   - At draft:137: where this document speaks, it governs. Where it is silent, the scripts at the freeze commit govern, as D72 to D75 and D81 read them, a later entry superseding an earlier one. A disagreement found after freezing is corrected by a dated amendment before any run it affects.
   - The two disagreements already found: draft:144 is to add the third stop (a passing cell whose scoring cost cannot be projected); draft:101 is to read "before any network scores the test set", as amendment:46 and the code have it (audit M5).
4. **G3 refuses an incomplete pilot record** (audit M3 and bug B2): it does not run unless all 48 pairs are in the record, accepted or failed.
5. **The declarations made exact** (referee blocking 3, audit M2, referee N8). The places to change are draft:7, :15, :22 and :200, and amendment:45 and :50:
   - two toy interpolants, not one;
   - the pricer at D38's (H, ν, ρ) at ξ₀ 0.02, 0.04 and 0.25 (G0's tests) and at ξ₀ in [0.04, 0.06] (G1's), instead of "at D38's point";
   - the overflow point, in both G0 test files;
   - the design's own pilot, held-out, training and validation draws made in memory by the tests (none of them sealed), and the G0 criterion file read by one test;
   - pytest's folders of recent sessions, which keep the toy files, so that "nothing kept" becomes "nothing printed or committed";
   - D81's readers' runs and its two scratch scripts, in "What had been seen";
   - the first review, as having read the committed G0 output too;
   - five decisions, not four, at draft:200;
   - this review, in the same places, as step 1 of the amendment file says.

## Worth making in the same round

6. **Fixed values and overrides** (audit M4, referee N3). `--test` refuses an `--n` other than 400 and a `--noise-seed` other than 20261022. The conditions for the version override are stated at draft:137, and G3 checks torch's version.
7. **No record is overwritten** (audit bug B4, referee N5). Stage 1's report refuses to be written over, and so does G2's first report, on which the retry and the go rule depend. Draft:141's "every run made is declared" is extended from G0 to every stage after freezing.
8. **Resumed runs** (audit bug B5). A resumed G3 or seed training compares the record's header with the current call before reusing saved networks.
9. **G2's text** (referee N1 and N2; see the session's findings).
   - State the cost of four standard errors for c and the quantiles, as well as for the mean.
   - Draft:89's "a round figure above G2's tolerance" to read "above G2's absolute tolerance on c".
   - Draft:132's "G2 bounds it" softened.
   - Draft:143's 90 corrected to 120, and its "a bias … larger than the tolerance in force still fails it" corrected.
10. **What "well covered" means at the training sizes in use** (referee, on the first review's B2). A mass of 0.001 is about 160 training surfaces at n = 160,000 and about 10 at n = 10,000. The dose-response is also to be reported on the well-covered pairs.
11. **What is reported when P5 stops** at G2, G3 or Stage 1; and G3's budget named as its own 1,344 core-hours (referee N6).
12. **The retry's streams** stated in the text (audit bug B3; the session's finding).
13. **Small items.**
    - Draft:141's "decides nothing" against D61's use of the modulus (referee N7).
    - A third seed failure (referee N11).
    - The docstring at `p5_stage2.py:63` (audit M6).
    - The top-level `N` of the re-priced G1 record (audit bug B6).
    - Step 1 of the amendment file to name draft:15 and :22 beside line 200 (referee N9).

## Decisions for the author

1. **Order.** Proposed: merge #234 as reviewed, then make these changes in a new entry, D82, with this record, so that what was reviewed and what was merged stay the same thing. The alternative is to push them onto #234 first.
2. **The two review records.** Proposed: both in `docs/protocols/`. The alternative is both at the root.
3. **OSF** (referee N4).
   - Proposed: attach to the new registration, beside the protocol, the files the freeze binds, so that the registry holds the plan the text defers to. These are the six scripts, the forward-model modules, the two requirements files, and ROADMAP.md at the freeze commit.
   - Say in the Status line that the hashes are of git blobs with LF endings, and that `p5_g0_pilot.py`'s hash is not that of the version the scoping run used.
   - Whether the registration form takes several files is to be read from OSF's help when the amendment file is updated.
4. **D5.** Proposed: it stands, with its full cost stated. The alternative is to reopen it.

## Report 1 — the independent referee (verbatim)

**1. Verdict:** freeze after small fixes.

**2. First review's issues** (draft = `docs/protocols/P5_protocol_v2_draft.md` at 3ddf92a)

Referee, blocking:
- B1 null: resolved. draft:128; p5_stage2.py:630–640; test_p5_stage2.py:357–383; amendment:41.
- B2 extrapolation: partly. Done: covariate (draft:99, reported :68), secondary analysis (:136), limitation stated where verdicts are read (:66, :126 under the table, :136, :170). Not done: the review's second confound (bins differ in their mix of true H; the dose-response at :134 is not re-scored on covered pairs). The draft also never says how weak "well covered" is: a mass of 0.001 is about 160 training surfaces at n = 160k and about 10 at 10k (ROADMAP:3680–3684).
- B3 reweighting floor: resolved. :68; p5_g2_reference.py:466–473.
- B4 G2 scope and failure: resolved. :143. See N1 on how much the 4-SE rule costs in resolution.
- B5 test set: resolved. :93–94, :101; budget at :143. Minimum and cap are moot.
- B6 precedence: partly. :137, :132, :110. No conflict rule and an incomplete hash list (Blocking 1, 2).
- B7 declarations: resolved in substance (:7, :15, :20–22) but not exact (Blocking 3).

Referee, non-blocking:
- NB1 resolved (:144). NB2 resolved (:126). NB3 resolved (:141).
- NB4 resolved (:158, :200, amendment:63), but :200 is now stale (Blocking 3d).
- NB5 resolved as a declaration (:141). The condition cannot be met after the fact, and the text says so.
- NB6 resolved (:54, :142). NB7 resolved (:101). NB8 resolved (:50). NB9 resolved (:89, amendment:38).
- NB10 not fixed, by choice. It is already declared at :132, next to the interval rule.

Audit:
- Generator index bases: resolved (:68, :75).
- G0 test-suite condition: resolved (:141).
- Fisher information and CRB: resolved (:89).
- Noise seed and the rule for N: resolved (:93–94).
- Stage 1 run alone: resolved (:101; p5_stage2.py:184–198, 376–421; p5_g3_calibrators.py:744–746).
- G1's restriction on Q1 not read by code: not fixed (D81 calls it a reporting gap); the text states the restriction at :51 and :142.
- G1 and G2 failure consequences: resolved (:54, :142, :143).
- Exploratory items: moot (labelled, :135).
- Outputs force-added by hand: not addressed; minor.
- §3 items 1–8: resolved (:144; :75 for items 2, 3 and 7; :110; :132; :68; :142).
- §3 item 9: held-out stop resolved at :54; percentile convention moot by :137.
- §3 item 10: resolved at :143; the grid's support is left to :137.
- §3 items 11–13: moot by :137 (see N4).
- §3 item 14 (environment): partly. Version guard at p5_g1_emulator.py:97–120, scipy pinned, versions listed at freeze (amendment:63). The override has no condition and torch is not checked (N3).
- §3 item 15: moot.
- §5 (D61 used the modulus, yet the text says it "decides nothing"): not fixed (N7).

Session's own findings:
- 32%: resolved (:126).
- Second probe: resolved (:20, :200, amendment:50).

Decisions 1–8 are implemented at :128; :66/:99/:126/:136/:170; :68; :143; :93–94; :54/:142/:144; :144; :137. The numbers agree across D81, the draft and the amendment (seeds, fixed points, 1,344, N = 400, seed 20261022, the tolerances, 16000, the versions). The one drift is the count of decisions (Blocking 3d).

**3. BLOCKING**

1. **draft:137 has no rule for a conflict.**
   - The scripts bind "wherever this document is silent". Nothing says what governs where the text and a script disagree. Nothing ranks the D72–D75 readings against D81, although some are superseded: D73's noise seed "an argument with no default" (ROADMAP:2678) and "12 pilot pairs" (:2684); D74's "none passing reported as P5 stopping" (:2787).
   - Disagreements exist now. :144 says "P5 stops only if" two conditions hold; the script adds a third, an incomplete projection counting as no go (p5_g3_calibrators.py:523, 633–637).
   - :101 says "before any network is scored". Read literally, that forbids G3's pilot scoring before Stage 1. The code (p5_g3_calibrators.py:744–746) and amendment:46 require Stage 1 only before the test set is scored.
   - Why it blocks: a conflict found after data leaves the choice of reading open.
   - Fix: one sentence. Where the document speaks, it governs. Where it is silent, the scripts govern, and a later log entry supersedes an earlier one. A disagreement found after freezing is corrected by a dated amendment before any run it affects. Also reword :101 and :144.

2. **draft:137 and amendment:63: the hash list does not bind the forward model.**
   - p5_g0_pilot.py:69–72 imports the pricer and the design constants from three files outside the list: layer4_calibrate_surface.py (TS, VUS, S, R, the strike functions), layer4_calibrate.py (KAPPA_FIXED) and rough_heston_cf.py (the Riccati solver, Gil–Pelaez pricing, bs_iv). These in turn import rough_heston.py.
   - G1 prices through them, and the grid constants reach every script through p5_g0_pilot. They define G (so G̃ and every posterior), the grid, κ, S0 and r.
   - A later edit to any of them changes every result while every listed hash still matches.
   - Fix: hash every repository module the six scripts import, directly or indirectly. Alternatively, require each post-freeze run at the freeze commit with a clean tree (the output headers already record both) and say so in :137.

3. **The declarations are not exact.** None of these bears on outcomes; each is one clause in step 1's docs commit.
   - (a) amendment:50 ends "and nothing else", and draft:15 lists the same items. Both omit runs that D81 records (ROADMAP:3711–3719): two scratch scripts that called P5 functions (G2's sampler and grid check on the toy map; Stage 2 scoring of synthetic records), the code reader's own Stage 2 script, and a check of torch's thread count.
   - (b) draft:7, :15 and amendment:45, :50 say the tests build "one" toy interpolant. D81's new test_p5_g1_emulator.py:256–310 assembles a second one, 2×1×1×2, from pricer quotes (:294).
   - (c) draft:15 says "G0's pilot test also at (0.02, 1.00, −0.99)". test_p5_g0_main.py:69 prices that point too, and amendment:50 omits the point entirely.
   - (d) draft:200 says the author "made the four decisions". D81 records five, D1–D5, with D5 taken after the hold (ROADMAP:3384–3388). D81's own line 3745 is also stale.
   - (e) amendment:50 says the first review "read the documents and the code". By its own record (review:12–13, 161) it also read the committed G0 output: the criterion file's values, and a count that depended on which quotes were finite.

**4. NON-BLOCKING**

- **N1. The cost of the 4-SE rule is understated** (draft:143 "Why four" and "Why 0.002"; :89; :132).
  - At 1,000 effective draws, the tolerance in force at the Flat edge is 0.090 on c (not 0.01) and 0.010 on the 5% quantile (not 0.002). By the draft's own reading, that quantile tolerance can move m by about 0.055, which is H-d's whole 0.05 gap. The draft states the price only for the mean.
  - :89 says 0.02 is "a round figure above G2's tolerance". That is false at the lower cut-off whenever four standard errors govern.
  - :132 says "G2 bounds it", which overstates.
  - :143 says a bias "larger than the tolerance in force still fails it". D81's own figure contradicts this: a bias of 0.005 against a tolerance of 0.0048 passes 44% of the time.
  - :143's "about 90 comparisons" is 120 (90 on the first grids, 30 on the second).
  - The grid's quadrature error is outside the rule; D81 says so (ROADMAP:3470), the draft does not.
- **N2. The standard-error formulas are correct** under the independent-draws Gaussian reading: sd/√n, (1−c)·√(2/(n−1)), √(q(1−q)/n)·sd/φ(z_q). The n_eff definitions are unambiguous and match p5_g2_reference.py:620–642. Two caveats:
  - The effective sample size of H, which is the one for the mean, is also used for the variance and the tail quantiles.
  - The text does not say that the sd and c in the formulas are the MCMC figures' own (they are, :635–641).
- **N3. Ways to deviate have no stated condition.**
  - :137 allows the version override without saying when it may be used.
  - `--test` accepts a different `--n` and `--noise-seed` "recorded" (p5_g2_reference.py:1042–1044), although :93 fixes both.
  - torch's version is not checked at run time.
- **N4. Auditing from the text alone (question f).** The binding plan now includes about 5,000 lines of code plus the D72–D75 and D81 readings. The OSF record will hold only the protocol file (amendment:19).
  - Attach the scripts, both requirements files and ROADMAP.md at the freeze commit to the new registration.
  - State in the Status line that the hashes are of git blobs with LF endings; the working copies are CRLF (amendment:13).
  - State that p5_g0_pilot.py's listed hash will not match the version the scoping run used (ROADMAP:3750–3753).
- **N5. Re-runs after freezing.** "Every run made … is declared" (:141) covers only G0. Extend it to G1–G3, the test references, Stage 1 and the scoring. The retry and go rules key on the first G2 report (p5_g2_reference.py:991–1021), which a re-run could overwrite.
- **N6. Stops.** The text does not say what is reported when P5 stops at G2, G3 or Stage 1 (:101, :143–144, :197). G3's "same budget" is a separate 1,344 core-hours in the code (p5_g3_calibrators.py:494–524); say so.
- **N7.** :141 says the modulus "decides nothing", but D61 set the main run's U_max and the ξ₀ lattice from it, as the same line records.
- **N8. Two more wording points in :15.**
  - "Nothing from it is kept": the repository has no pytest configuration, so pytest's defaults very likely keep the last three sessions' temporary folders, toy build files included. I did not check the author's pytest version.
  - "Toy maps and synthetic records" omits that the tests draw the actual G1, G2 (first attempt and retry) and G3 pilot sets, and the first rows of the training and validation draws, in memory. One G1 test also reads the committed criterion file (test_p5_g1_emulator.py:29). None of this is sealed.
- **N9.** Step 1 of the order of acts (amendment:62) updates "What had been seen" after this review, but not the draft's :15 and :22.
- **N10. D81's summary (ROADMAP:3352–3361).**
  - It is accurate on the findings and on D1–D5.
  - It omits the null, reweighting and precedence/version-guard decisions, the change of "separates" to "limits", and the sealed-draw incident.
  - "Now match" overstates (Blocking 1, 3).
  - "One retry at G1" is a re-pricing of a non-finite point, not a retry of the gate.
- **N11.** A third seed failure leaves three seeds in the code (p5_g3_calibrators.py:699–720); :75 and :132 speak only of five or four.

**5. Checked and found sound**

- The null can be met by a faithful calibrator and not by an unfaithful one, and code and tests match :128. Neither the secondary nor the shifted re-scoring reads a null.
- Every figure D81 and the draft quote reproduces by arithmetic, to rounding: the 4-SE and pass-probability figures, the 0.002 reading, ±1.7 points, and the new prior counts (3,152; 1,711; 9.5; 1,338; 59.8; 0.1875 and 36.4%; 49.7% and 64.9%, i.e. 597 and 779 pairs). Grid-quantile quadrature error for a Gaussian is at most 0.00075, consistent with D81's "about 0.0008".
- The sealed-draw incident is declared consistently at :15, amendment:50 and in D81. The replacement test draws nothing (test_p5_g2_reference.py:415–447, 502–529). Only 3ddf92a touches tests, so no committed version ever drew from the sealed streams.
- The per-cell G3 rule, Stage 1 before scoring (checked by a digest of the references), the reference-failure stop, the secondary floors and the go/no-go are implemented as the text states.
- The amendment's line citations (11, 200, 215, 7, 216) and D81's layout claim hold.

**6. Run other than reading, grep, sed, wc, git**

- Plain commands: `ls`, `awk` (to print line ranges), `which python3`, and `python3 -I -c` printing the scipy and numpy versions.
- Snippet 1 (scipy.stats.norm, math): the 4-SE tolerances at n_eff 1,000–20,000; c's standard error and pass probabilities at c 0.21 and 0.5; 0.99994 raised to the k-th power; the pass rates for a biased mean; the 0.002 reading; ±1.7 and ±3.4 points.
- Snippet 2 (numpy, scipy.stats.norm): the quantile error of a deterministic 32-cell grid for Gaussians with sd 0.017, 0.038 and 0.050.
- Snippet 3 (scipy.stats.beta, scipy.optimize.brentq): the priors' tail masses times 160,000, the mass within ±0.034, the percentiles, and the well-covered range of H.

All snippets ran from the scratchpad with `python3 -I`. They imported nothing from the repository, read no repository file and drew no random numbers. No P5 script, test, pricer or output was run or opened.

## Report 2 — the code-versus-protocol audit (verbatim; headings one level down)

**Second review of PR #234 (D81 code at 3ddf92a) against the v2 draft**

All files were read in full at the head, read-only. Paths are relative to `/tmp/claude-0/-home-claude/3c180cf6-6ffc-51f4-8411-725f861f1348/scratchpad/pr234/`. "draft" means `docs/protocols/P5_protocol_v2_draft.md` and "amend" means `docs/protocols/P5_v2_amendment.md`.

Verdict: three BLOCKING items, one of them text-only. Everything else D81 wrote is implemented as written.

### 1. Protocol against code

**What matches**
- **Reference (draft:68):** generator `[83,i,j]` with i and j from 0 (p5_g2_reference.py:448, 832–839). Reweighting floor (Kish/pooled)×ESS_H ≥ 1000 (:462, 470–471). A failed direct run is a reference failure (:474–480). Coverage covariate (:381–399, 486).
- **G2 (draft:143):**
  - The 18 surfaces, fixed points and seeds 79/89/139 and 131/137/149 (:115–118, 528–538).
  - Under B and under A's quantiles from one set of likelihood evaluations, with the cell-spread quantile (:561–593).
  - The second grid at the largest level only (:943).
  - Tolerance max(abs, 4·MCSE), using the MCMC figure's own sd and c. n_eff is (Kish/n)·ESS when reweighted and the direct run's ESS when direct; formulas sd/√n, (1−c)√(2/(n−1)), √(q(1−q)/n)/(φ(z_q)/sd) (:620–662).
  - A reference failure fails the gate (:939–942).
  - Retry at doubled lengths; P5 stops on a second failure (:991–1030, 1070–1087).
  - Go/no-go at 1,344 core-hours over 1,200 pairs (:879–901, 960–961, 1016–1019). G3 pilot and test references only after a go (:1076–1077).
- **Test set (draft:93):** 400 rows; `(400,3,35)` noise from 20261022 (:120–122, 515–525, 1105).
- **Ratio estimator (draft:75):** `[107,p,q,i,j]` with j from 1 (p5_g3_calibrators.py:409, 118). Lengths come from G2's status (:820–821). Ensemble divisor 5 (:385). Training draw layout (:149–163).
- **G3 (draft:144):** pass or fail per cell; largest passing n, every rung run (:596–628); scoring projected against the budget (:494–524, 629–638); seeds trained for passing cells only (:683–694).
- **Stage 1 and 2:**
  - Stage 1 alone, with the digest required by the test scoring and by Stage 2 (p5_stage2.py:173–198, 376–421, 824–837; p5_g3_calibrators.py:744–757).
  - Failure stop above 40 of 400 (p5_stage2.py:350–373).
  - Null (:630–640).
  - Bootstrap `[127,c,r]`, with r=9 for the dose-response (:660, 723).
  - m from the A-trained network (:464–469).
  - Lost G3 levels with floors on the kept levels (:620–627, 662–682, 772–797).
  - Secondary analysis with the same streams and floors recomputed (:864–870).
- **Versions:** refused unless the override is given, and recorded (p5_g1_emulator.py:97–119 and each `main(check_versions=True)`); requirements-p5.txt:15–17.

**Mismatches**

- **M1 — BLOCKING. G1's outcome is not carried downstream.**
  - The text: draft:54 and 142 say a quote still not finite after the re-pricing "stops P5". draft:142 says the build is "repeated at the new nodes" if e > s/5.
  - The code enforces the stop only inside G1's own record (p5_g1_emulator.py:386–390, 416–423). The refined build is written to `output/p5_g1_refined_emulator.npz` (:122–127, 581–584).
  - G2 and G3 default to the base emulator (p5_g2_reference.py:141, 671–679, 1045; p5_g3_calibrators.py:129, 796, 813–815). Neither reads G1's report.
  - `require_g2` (p5_g2_reference.py:1024–1030) does not check that the G2 report, the test references and G3 used the same emulator.
  - G3's header and report, and the scores header, record no emulator at all (p5_g3_calibrators.py:574–578, 655–669, 753–757). So Stage 2 cannot check draft:50's claim that network and reference face the same model.
  - Result: which emulator is G̃ is decided by the `--emulator` flag after G1's output is seen, and a written stop can be ignored (scenarios in §2).

- **M2 — BLOCKING (text only). The declarations of what the tests compute are incomplete.** draft:15, amend:45 and amend:50 are wrong in four places (details in §4):
  - a second toy interpolant built by a new G1 test is not declared;
  - "at D38's point" is not exact;
  - the overflow point is priced by both G0 test files, not only the pilot test;
  - D81's scratch computations on toy inputs are left out of amend:50, which says "and nothing else".

- **M3 — BLOCKING (only if the G3 pilot references are incomplete).** draft:144 says G3 uses the 16 pilot surfaces at each level, with failures removed and declared. G3 reads whatever the G3-pilot record holds and never checks that all 48 pairs are present (p5_g3_calibrators.py:426–441, 816–824). G2's report (p5_g2_reference.py:929–935) and Stage 1 (p5_stage2.py:390–393) do refuse an incomplete record.

- **M4 — NON-BLOCKING.** draft:93 fixes N=400 and the noise seed, but `--n` and `--noise-seed` still accept other values. Such values are only recorded in the header (p5_g2_reference.py:1042–1044, 1105–1109), and Stage 1 accepts any `n_points` (p5_stage2.py:390–393). The text governs, but nothing refuses or flags a run that departs from it.

- **M5 — NON-BLOCKING (wording).** draft:101 says Stage 1 comes "before any network is scored". G3 necessarily scores networks on pilot surfaces first. The code (p5_g3_calibrators.py:744) and amend:46 say "before any network scores the test set"; draft:101 should match them.

- **M6 — NON-BLOCKING.** The docstring at p5_stage2.py:63 says the bootstrap uses `default_rng(127)`. draft:132 and the code say `[127,c,r]`.

- Declared and not done: no script applies G1's restriction on Q1 (draft:142; D81 lists this under "Not written").

### 2. Bugs

- **B1 (M1) — BLOCKING.** Two scenarios:
  - **Refinement ignored.** The base G1 report says refine (e = 0.0003). The refined emulator is built. `p5_g2_reference.py --pilot` with no `--emulator` then builds G2, the G3 pilot set, the test references and G3 training on the base emulator. Passing `--emulator` to some scripts and not others mixes emulators, and nothing shows it, because G3 and the scores record no emulator.
  - **Stop ignored.** A refined node is still non-finite at 16000, so the text says P5 stops. The base `.npz` is still at the default path and G2 runs normally.
  - **Fix:** add a `require_g1()` that returns the refined path when the base report calls for refinement and refuses on a stopped or incomplete stage. Use it as the default in G2 and G3. Record the emulator's path and SHA-256 in the G2, G3 and scores headers, and have Stage 2 compare them. Add one sentence to draft:142 saying the refined build is G̃ from then on.

- **B2 (M3) — BLOCKING (conditional).**
  - **Scenario:** `--g3-pilot` is interrupted after 30 of the 48 pairs (the script is built to be resumed). The operator then runs `--g3`. Each cell is gated on about 10 surfaces, `removed_references` stays empty, and the slope and stability verdicts can change.
  - **Fix:** refuse unless all 48 pairs are recorded.

- **B3 — NON-BLOCKING. The G2 retry reuses the first attempt's random streams.**
  - The retry's pairs use `[83,i,j]` with i restarting at 0 (p5_g2_reference.py:448, 835–837), and the fixed points sit at i=4 and 5 (:535). The retry's chains therefore start from the same four points and draw the same innovations as the first attempt. At the fixed points θ is also identical; only the noise differs.
  - **Scenario:** a fixed-point pair fails because all four chains start in one basin. The retry is likely to repeat the same miss.
  - This is literal to draft:68, which says "the i-th θ of the set in question". Either write it into the text or give the retry its own key.
  - The same key sharing joins G2 pilot pairs with test pairs (i<6), G3 pilot pairs with test pairs (i<16), and `[107,p,1,i,j]` between G3's pilot posteriors and the test scoring at q=1. All of this is harmless.

- **B4 — NON-BLOCKING. Stage 1's record of timing and its digest.**
  - `run_stage1` silently overwrites its report (p5_stage2.py:417–420), so a re-run after scoring loses the evidence that Stage 1 came first.
  - `refs_digest` (:173–181) includes the volatile `utc` and `seconds` fields (p5_g2_reference.py:487, 690–691). If one sample file is lost and that pair is recomputed with an identical result, the digest still changes. Stage 2 and the scoring then refuse, forcing a full re-score. This fails safe but is fragile.

- **B5 — NON-BLOCKING.** When G3 or the seed training resumes, it reuses saved networks without comparing its header to the current call (p5_g3_calibrators.py:543–554, 574–578). It cannot even compare the emulator, since none is recorded.

- **B6 — NON-BLOCKING.** A re-priced G1 record keeps the first pricing's `N` (8000) and its per-maturity strikes and ATM vols at the top level, with the 16000 detail under `per_T_reprice` (p5_g1_emulator.py:337–340). Someone reading the strikes from `per_T` would get the wrong ones.

**Checked and clean**
- Every RNG key against the index base it is fed:
  - `[83,i,j]`: i and j from 0;
  - `[107,…,j]`, `[109,p,j,q]`, `[113,p,j]` and the torch seed: j from 1;
  - the conversions j−1 and j+1 between files (p5_g3_calibrators.py:430, 762; p5_stage2.py:215, 225).
- ddof and axes: ensemble, adapted covariance, the between/within estimator, the Standardiser.
- Empty bins and division by zero: Stage 1, the failure and seed floors, projections, `mc_errors` (n_eff floored at 2).
- NaN: JSON is written with `allow_nan=False`, comparisons that meet a NaN fail closed, the bootstrap drops NaN replicates.
- Budgets: 1344 = 14·24·4; 4 h; 320,000; 400·5 items per cell.
- Units: noise levels in decimals; the G1 bound 0.0002.

### 3. The sealed test set (checked by reading only)

- `git grep` finds 20261002 and 20261022 only in p5_g2_reference.py (constants at :120–121, docstrings, and `test_points` / `test_noise` at :515–525) and in the tests as literal values.
- **test_p5_g2_reference.py, the only places the sealed seeds could be drawn:**
  - Lines 432–433 are a docstring.
  - Lines 50, 354, 439, 445–446, 457 and 527 are constants, expected log tuples or a header dictionary.
  - The only calls to `test_points()` and `test_noise()` (443–444) run inside a `monkeypatch.context()` that replaces `np.random.default_rng` with `_NoDraw` (442). R looks that name up at call time (p5_g2_reference.py:507, 512). The assertions at 445–447 confirm the recorder was hit and returned zeros.
  - `main(["--test"])` at 211 is refused by `require_g2` (the report is redirected to a missing file at 209) before `test_points` (:1105).
  - At 412, `get_emulator("no_such_file.npz")` exits (:1078) before the draw.
  - At 525, `R.test_points` and `R.noise_draws` are replaced by recorders (502–510).
- **Other files:**
  - No P5 module draws at import; the module tops hold constants only.
  - There is no conftest.py, pytest.ini or pyproject.
  - The other test files use the sealed seeds only as excluded values or labels (test_p5_g1:38, test_p5_g3:51, test_p5_stage2:32 and :210, test_p5_g0_*).
  - The pricer modules' own `default_rng(seed)` calls never receive these seeds.
- **The first version of the G2 test:** within the PR, the test file is changed only by 3ddf92a, so the first version was never committed and CI never ran it.

### 4. What the tests compute; declarations; CI

**test_p5_g1_emulator.py** (12 tests, all run in CI):
- Analytic Chebyshev checks.
- Reads G0's committed criterion JSON.
- Draws G1's 200 held-out points (seed 73) in memory.
- Real pricer at N_riccati 200, 32/64 nodes, U_max 100–400 and three strikes:
  - D38's point at T 0.25 (path identity);
  - D38's (H, ν, ρ) at ξ₀ 0.05 and T 0.10 (:126–155 and the new :212–254);
  - the declared 2×2×2×2 toy build (:157–188).
- **New, not declared** (:256–310): a 2×1×1×2 interpolant assembled from pricer surfaces at D38's (H, ν, ρ) and ξ₀ ≈ 0.043 and 0.057 (T 0.10). It is assessed on two held-out surfaces whose ξ₀ comes from seed 73 in [0.04, 0.06]; e and the tails are computed into tmp_path.

**test_p5_g2_reference.py** (14, all in CI):
- MCMC, grids and reweighting on LinearEm (a random linear map) and on synthetic chains.
- Draws the design's pilot θ and noise (seeds 79, 89, 97, 101, 131, 137, 139, 149) and the generator [83,0,0], on toy maps only.
- No pricer and no real emulator.

**test_p5_g3_calibrators.py** (14: 9 run in CI, 5 marked `_needs_torch` skip):
- The numpy tests use LinearEm, hand-built numbers and stand-ins (prefixes of the [109,…] and [113,…] draws).
- The torch tests train small networks with the design's trunk on LinearEm surfaces (n ≤ 800) and run the ratio estimator's sampler on a hand-built logit.

**test_p5_stage2.py** (14, all in CI): synthetic reference, sample, score and G3-report files only.

None of the tests runs MCMC on rough-Heston surfaces or uses a real emulator of the design.

**Declarations (M2).** draft:15 says "the G0 and G1 tests run the pricer … at D38's point (G0's pilot test also at (0.02, 1.00, −0.99)) and one G1 test builds a 2×2×2×2 interpolant". Corrections needed:
- The G0 tests price D38's (H, ν, ρ) at ξ₀ 0.02, 0.04 and 0.25 (test_p5_g0_main.py:48, 54; test_p5_g0_pilot.py:110).
- The G1 tests price at ξ₀ in [0.04, 0.06].
- The overflow point is priced by both G0 test files (test_p5_g0_main.py:69–70).
- A second toy interpolant exists (above).
- amend:45 ("only a small toy interpolant") and amend:50 ("one small toy interpolant … and nothing else") repeat the error. amend:50 also omits the overflow point and D81's scratch scripts on the tests' toy inputs (ROADMAP:3714–3719).

**Wording (NON-BLOCKING).**
- draft:7 reads as saying no reference posterior or trained network was built before freezing. The tests computed toy ones, so it should say "of the design".
- "Nothing … kept": pytest keeps the tmp_path directories of recent sessions by default, so the toy `.npz` files and reports stay on the author's machine for a while.

**CI** (.github/workflows/tests.yml): 66 P5 tests run, including the real-pricer G0 and G1 tests and all 19 new ones, which need no torch. The five G3 torch tests skip. The workflow comment at :30–31 mentions only the deep-hedging skips.

**Counts agree.** Test functions went from 293 to 312 (+19 = 3+6+4+6, no parametrisation in the P5 files). Parametrised extras elsewhere add 31, giving 324 → 343, which matches CONTRIBUTING.md:30 and D81. The P5 files hold 71 tests (66 passed + 5 skipped); test_p5_g3 holds 14; 7 skips in total (5 G3 + 2 Layer 3). The four adapted existing tests match D81.

### 5. Code choices the text does not make

| Choice | Where | Judgement |
|---|---|---|
| G1 re-pricing re-anchors strikes at 16000; finite surfaces of the triple keep their 8000 quotes; record fields as in B6 | p5_g1_emulator.py:323–342 | Harmless; worth one clause |
| Version check is an exact string match, from the command line only | p5_g1_emulator.py:106–119 | Harmless |
| Retry reuses the reference streams; fixed points placed at i=4,5 | p5_g2_reference.py:448, 535 | Write into the text (B3) |
| Pilot noise layout (n,3,35) | p5_g2_reference.py:510–512 | Harmless; binds through precedence |
| Go/no-go median over all runs, doubled runs normalised; seconds cover the sampler and acceptance only | p5_g2_reference.py:879–901 | Harmless (D81 declares it) |
| MCSE from the MCMC sd and c; n_eff floored at 2; tolerance check closed (≤) | p5_g2_reference.py:629–658 | Harmless (declared) |
| Coverage mass of a zero-width interval = 0 (not well covered) | p5_g2_reference.py:389–390 | Harmless |
| `--n` and `--noise-seed` overridable | p5_g2_reference.py:1042–1044 | Problematic; should refuse (M4) |
| Scoring cost: cores = torch threads; ensemble cost per surface taken from a batch of ≤16; reference cost excluded | p5_g3_calibrators.py:494–524 | Harmless; worth a sentence in line 143 |
| An incomplete projection counts as no go | p5_g3_calibrators.py:518–523 | Harmless (declared) |
| n* may sit above a failing rung | p5_g3_calibrators.py:622–625 | Consistent with "largest" |
| A replacement seed uses q=6 in its sampler key | p5_g3_calibrators.py:409 | Harmless |
| More than one retraining failure per cell leaves several slots empty | p5_g3_calibrators.py:705–720 | Harmless; text silent beyond the second |
| Digest of references, launch lines excluded | p5_stage2.py:173–181 | Declared; see B4 |
| Stage 2 compares only the untestable flags with Stage 1's output | p5_stage2.py:851–853 | Harmless |
| Failure report uses a 5×3 grid; "exceed" read as strict | p5_stage2.py:146–147, 368 | Harmless (declared) |
| Stage 1 report overwritable | p5_stage2.py:417–420 | Should refuse (B4) |
| No emulator recorded by G3 or in the scores | p5_g3_calibrators.py:574–578, 753–757 | Problematic (M1) |

### What I ran

Beyond reading files, grep, sed, wc and git:
- `ls`; `cat` and `head` to read files;
- `sort`, `uniq` and `echo` to format grep output;
- bash integer arithmetic to sum `grep -c '^\s*def test_'` over the test files at HEAD and at b11063b;
- `git ls-files output/`, which listed names only.

No Python, no script, test, import or pricer was run; no `output/p5_*` file was opened; no random numbers were drawn.

### Bottom line

The code at 3ddf92a matches D81's text closely, and the first review's findings are fixed: the null, the reweighting floor, G2's tolerances and failure path, N=400 and its noise seed, G3 per cell at the largest n, Stage 1 alone with its digest, the secondary analysis and the version guards. Three things must change before freezing:

1. **G1's outcome must bind the later scripts.** If the base report calls for refinement, the refined emulator must be the one used; a stopped G1 build must stop P5; and the emulator must be recorded in G2, G3 and the scores, with Stage 2 checking they agree. This needs a small code change and one sentence in draft:142. Without it, the choice of G̃ stays open after G1's output and a written stop can be bypassed.
2. **The test declarations in draft:15 and amend:45 and 50 must be corrected:** the second toy interpolant, the ξ₀ values, both G0 tests, and the scratch computations.
3. **G3 must refuse an incomplete G3-pilot reference record.**

`--n` / `--noise-seed`, the retry's streams, and the overwriting of Stage 1 are worth fixing at the same time but do not block. After these, the second review's own record (amend step 1) can be written and the boxes re-ticked.
