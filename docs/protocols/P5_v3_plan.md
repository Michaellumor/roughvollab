# P5 v3: what changes, what does not, and how it gets frozen

A working aid for the author, prepared on 25 October 2026 after G2's stop (D89). It decides nothing. The questions under Decisions are the author's, and are answered in the decision-log entry that commits this file; the file is not edited to record the answers. A lean is given with each question. Once the author agrees it, the file is committed as `docs/protocols/P5_v3_plan.md` before any v3 work, so that its date precedes everything it plans.

*How it was written.* Line numbers are the frozen draft's, unchanged since `59ca589`, read at `dc54c0e`. The figures are D88's and D89's as committed. Nothing was run, sampled or drawn.

## Why v3

v2's gate G2 failed on its first attempt (9 of 18 pilot pairs were reference failures) and on its one retry at doubled lengths (5 of 18). Under line 143, P5 v2 stops (D89).

The failures concentrate where the posterior is sharpest:
- at s = 0.1 vol point (0.001), 6 of 6 failed, then 5 of 6;
- at 0.3 (0.003), 2 of 6, then none;
- at 0.5 (0.005), 1 of 6, then none.

What the record shows (D89):
- **Acceptance.** The failed pairs' runs accepted 0.14% to 4.2% of their proposals on the first attempt, and 0.41% to 3.1% on the retry.
- **R-hat** reached 14.2, then 25.9.
- **The effective size of H** ran from 4 to 1,596, then from 4 to 1,817, against a floor of 2,000.
- **Even at 0.5 vol point,** where most pairs passed, the uniform-prior runs accepted only 4.1% to 12.9% of proposals, then 9.2% to 12.8%.

Every pair whose reference run was accepted matched the brute-force grids within tolerance. The projected cost of the test set's references was 48.2 and 53.6 core-hours, against a budget of 1,344. The reference method, the emulator and the grid checks held.

D89 reads this, as a reading labelled as such, as chains that did not adapt rather than chains that mixed slowly. The sampler starts its four chains uniformly on the box, with a first proposal of 2.38²/4 · (1/12) · I. Every 500 burn-in steps it adapts to the covariance of each chain's whole history, the uniform start included. The reading therefore points at the start and the first proposal, and does so most strongly where the posterior is narrowest.

G2 also exposed a bug. With more than one worker, `_worker` in `p5_g2_reference.py` passes `emulator_path` into `reference_pair` on every call after a worker's first, and the call fails (D89).

## What v3 changes (exhaustively)

1. **The sampler's start and first proposal, by candidate (A) below.** Everything else about the sampler stays: the adaptation, the lengths, the acceptance, the repeat, the reweighting and the direct runs, and the storage.
   - **Where the sampler is used.** The exact reference uses it (line 68), in the uniform run and the direct runs alike. The ratio estimator's posterior also uses it (line 75), through `_run_with_repeat` with a `target`. Whether and how the change reaches the ratio estimator is Decision 3.
   - **The retry's start clause goes.** Lines 68 and 143 lose the clause that the retry's chains at the fixed points start where the first attempt's did, and draw the same proposals (D82): under (A), the start and the first proposal depend on each surface's y. The pair key `default_rng([83, i, j])`, with i counted within each set, stays, so only the generator's stream is shared between attempts.
   - **The go rule.** Line 143's go rule gains a clause: a run's core-seconds include its optimiser.
   - **The constants are recorded.** The constants of (A) go into the header's `lengths` dict. A resume is checked against that dict, so a launch with other constants is refused, as one with other lengths is.
   - **Docstring and tests.** The script's docstring, and the tests' assertions on the start and the first proposal, change with it.
2. **The pool bug.**
   - **The fix.** `_worker` and `_grid_job` both call `get_emulator(kw.pop("emulator_path", None))` on every call. `get_emulator` returns the cached object when the path matches or is None, and reloads otherwise. `_grid_job` has the same pattern today; it is harmless there only because it reads `kw` by key.
   - **The audit.** The other pool paths (`p5_g0_pilot.launch`, `p5_g0_main.launch_main`, `p5_g1_emulator.launch`) carry no per-process cache, and pass their keyword dicts unchanged. `p5_g3_calibrators.py` parallelises through torch threads and `p5_stage2.py` has no process pool.
   - **A new test runs both pool paths** of `p5_g2_reference.py`, `launch` and `g2_report`. Its set-up:
     - two workers and at least four pairs, so that some worker takes a second pair;
     - a small emulator file holding every key the loader reads, written to a temporary folder and loaded in the parent as `main` loads G̃;
     - short lengths and relaxed acceptance through `task_kwargs`;
     - `_no_sealed_draws` active.

     It asserts three things: that `launch` returns the number of pairs; that the record holds no `task_error` line; and that every record equals a pool-1 run on the same jobs. It must pass under both start methods: spawn (the author's Windows, Python 3.14) and fork (CI's Linux, Python 3.12, where workers inherit the parent's cache). A second launch, with the cache cleared, exercises the loading branch.
3. **G2's seeds.** The draft names six fresh seeds: points, noise and fixed-point noise, for the first attempt and for the retry.
   - None of them is 7, 61, 62, 73, 79, 83, 89, 97, 101, 107, 109, 113, 127, 131, 137, 139, 149, 20261002 or 20261022.
   - The tests' exclusion set gains v2's six.
   - The fixed points, H = 0.04 and 0.46 at the box's centre, keep their θ, with fresh noise.
4. **G1 is not re-run.** G̃, built and passed under v2 (D88), is adopted, with its SHA-256 bound in v3's Status line.
   - **Its inputs are unchanged:** the pricer, the box, the nodes, the resolution and the held-out rule.
   - **A rebuild would not match.** The build has no training and no seed (line 54), but no rebuild is recorded. A rebuild would not reproduce the file byte for byte either, because the `.npz` records the commit of the record that built it (`p5_g1_emulator.py`, line 690).
   - **The reason to adopt.** Adopting G̃ by its hash keeps one emulator, with one hash, across v2 and v3.
5. **Where v3's records go.** Under new names set in the script: `output/p5v3_g2_*`, `p5v3_g2r_*`, `p5v3_g3_pilot_refs*` and `p5v3_ref_test*`. v2's committed G2 files stay where they are, as history; line 141 forbids writing over a record (Decision 9).
6. **What v3 declares, and the lines that change with no change of rule.**
   - **Line 7, the Status:** v3, what it supersedes, the new hash list, G̃'s hash and the versions. The sentence saying that nothing of the design is built before freezing is replaced, since G̃, v2's G2 posteriors and the development study all exist.
   - **Line 9:** the reason for this amendment.
   - **Line 15, what had been seen:** it gains G1's output (D88), both of G2's attempts (D89), the development study, and the seed list for the tests.
   - **Lines 54 and 140:** G1 was run under v2.
   - **Line 137:** the entries that read the scripts gain v3's.
   - **Line 142:** G1's outcome under v2 (passed, no refinement, G̃ the base build; D88).
   - **Line 143:** the seeds, the retry clause and the go rule's clause (items 1 and 3).
   - **Line 200:** v3's sittings and models.
   - **The checklist:**
     - boxes 204 (prior knowledge), 209 (the reference sampler), 210 (line 75's sampler, under Architecture and training), 213 (the gate tolerances, since line 143 changes) and 215 (the review, so that it cites v3's review record) are reopened, and re-ticked on the author's word before the freeze;
     - box 216 is reworded at the freeze commit to "the act that follows this commit", as in v2, and ticked in the post-freeze commit.
   - **`CONTRIBUTING.md`:** its collected test count changes with each new test.

## What v3 does not change

These stay exactly as registered in v2:
- the question, Q1 and the hypotheses with their thresholds;
- the priors, the cut-offs and the bins;
- the calibrators, their architecture and training, except the start and first proposal of the ratio estimator's sampler, if Decision 3 extends (A) to it;
- G0; G1's rule as text, its v2 outcome adopted (item 4); and G3's rule;
- G2's tolerances, its one retry at doubled lengths and its budget (its seeds and the retry's start clause change, by items 1 and 3);
- the sealed test set (N = 400, seeds 20261002 and 20261022);
- the scoring, the analysis and the prior-work section.

If any of these changes, v3 is no longer the narrow fix it claims to be. A reviewer would rightly call that moving the goalposts after seeing pilot posteriors.

## The candidate sampler (A): anchored starts and a preconditioned first proposal

This runs once per sampler run, before the chains: in each call of `run_chains`, that is, the uniform run, its repeat, each direct run and its repeat. The run's generator is the pair's `default_rng([83, i, j])`, continuing as in v2. Its draws come in this order:
1. one block of 16 × 4 uniform draws, for the optimiser's starts;
2. one block of 4 × 4 standard normals, for the chains' start perturbations;
3. the proposals and acceptance draws, as in v2.

The steps are these.

1. **Find the modes.** Minimise the negative log-target in unit coordinates x, from 16 starting points uniform on [0, 1]⁴.
   - **The residuals** are r(x) = (y − G̃(θ(x)))/s, so that ½Σr² = −log L.
   - **The solver** is `scipy.optimize.least_squares`, with method 'trf', bounds [0, 1]⁴ and jac '2-point'. Its tolerances are declared in the draft, and its exit status is recorded for each start.
   - **The Jacobian.** The solver returns J = ∂r/∂x at each solution, by finite differences, since the emulator has no derivative routine.
   - **For a direct run under prior A or B,** each solution kept by step 2 is then polished by L-BFGS-B on the full negative log-target, −log L − log P(x_H). The gradient of the first term is Jᵀr; that of the second is analytic.
   - **The H bounds of a direct run** stay ε = 1e-6 inside the faces, because both priors' densities vanish there.
2. **Keep the modes that matter.** Take the solutions in order of log-target, from the best. A solution is distinct when its unit-coordinate distance to every solution already kept exceeds 0.01.
   - Keep up to four distinct solutions whose Laplace mass, log f(x_k) + ½ log det Σ_k, is within log 10⁴ of the best's. Any mode dropped then holds under 10⁻⁴ of the mass.
   - Along a flat valley these are start regions for the chains, not modes, and their number m is reported as such.
3. **Compute a local covariance at each kept solution:**
   - for the uniform run, Σ = (JᵀJ)⁻¹, which is already a covariance in unit coordinates. It is formed from J's singular values (V S⁻² Vᵀ) rather than by inverting JᵀJ;
   - for a direct run, Σ_P = (JᵀJ + diag(−∂² log P/∂x_H², 0, 0, 0))⁻¹, at its solution.

   In both cases the eigenvalues are clipped to [1e-10, 1/12].
4. **Start the chains.** Chain k starts at kept solution k mod m, plus a draw from N(0, 4Σ_s), clipped to the box (to [ε, 1 − ε] in H, for a direct run).
   - **The perturbation cap.** Σ_s is Σ with its eigenvalues capped at 0.0025, so that a direction flat at the optimum does not start a chain on a far face.
   - **Near a face,** Σ is the unconstrained Gauss–Newton covariance. Proposals outside the box are rejected, as in v2, and the adaptation corrects the truncation; the study's acceptance over the first 500 burn-in steps shows how far.
   - **Several modes.** Chains at different solutions let R-hat see a second mode that they do not cross. Such a pair fails, as G2 already requires. (A) reports such a posterior rather than sampling it; sampling it is what (B) is for.
   - **One mode.** With m = 1 the chains start together, and the 16 optimiser starts take over the search for other modes.
5. **Set the first proposal** to 2.38²/4 · Σ of the chain's solution, in place of 2.38²/4 · (1/12) · I.
6. **Leave everything else as in v2:**
   - the adaptation every 500 burn-in steps;
   - the lengths, and the acceptance (R-hat ≤ 1.01 on every parameter, and an effective size of H ≥ 2,000);
   - the one repeat at doubled lengths, which re-optimises from 16 fresh starts, since (A) runs once per sampler run;
   - the reweighting rule and the direct runs;
   - the quantised storage.

The optimiser's seconds fall inside each run's timed window, so `project_cost` keeps its form.

**The ratio estimator's posterior (Decision 3).**
- **Why the general route is dropped.** The ratio estimator's log-target is log P(H) plus the logit of a ReLU network evaluated in float32, which is piecewise linear in θ. Its finite-difference Hessian is zero or arbitrary, and the default finite-difference gradient step lies below float32's resolution. The general route first proposed for it, L-BFGS-B with finite differences and a finite-difference Hessian, is therefore dropped.
- **The gradient route.** Modes are found by L-BFGS-B from 16 starts, with exact gradients from torch autograd on a float64 copy of the network.
  - Σ is the exact reference's Σ_P for the same pair and prior, written into the pair's reference record, since the reference precedes the scoring of every pair.
  - Σ is a preconditioner only, so the stationary distribution is unchanged.
  - This route changes `p5_g3_calibrators.py` and the record.
- **The alternative** keeps v2's start and first proposal for the ratio estimator. Its sampler may then fail at 0.1 vol point as v2's reference did, and line 75 charges any such failure to the calibrator.

**The fallback, (B),** is a sequential Monte Carlo sampler with adaptive likelihood tempering. It is more robust to several modes, but it is not a narrow change. Because it has no chains, it rewrites:
- line 68's acceptance, repeat and Kish floor;
- line 143's retry at doubled lengths, its Monte Carlo term (n_eff, by the chains' effective size) and the first grid's support box;
- the stored samples, and Stage 2's reading of them;
- line 75's sampler for the ratio estimator, and its failure definition;
- the cost projection, which is in steps.

Under (B), the list of what v3 does not change would no longer hold, and the decisions below would be revisited. (B) is used only if (A) fails the development study, and that would be declared.

## The development study (before any v3 text is final)

All of the study is declared, and all of it uses only what has already been seen.

- **Its standing.**
  - Line 197 closes v2: nothing of v2 is re-scored, re-run or tuned, and its records stand.
  - The study is v3's development, made on surfaces v2 has seen, before v3 is frozen. It is declared under line 198 as part of what had been seen, and the stop notice on `695fw` says so.
  - D89 says a continuation would be "a new version registered before any run". That is read as before any run of v3's gates, and the study runs no gate.
- **Its code.** The study script sits outside the frozen scripts.
  - It calls `p5_g2_reference.py`, as committed after the pool fix, for everything but the start and first proposal: `reference_pair`, `_run_with_repeat`, `accept`, the reweighting and direct runs, `grid_check`, `whole_box_check` and `g2_report`. Only `run_chains` is replaced, by the candidate's.
  - Its record carries the gate records' header (commit, clean tree, versions checked against the pinned ones, and G̃'s SHA-256) and the constants of (A).
  - The script and its settings are committed before the run, as G0's scoping script was (box 213).
- **Surfaces.** v2's 36 pilot pairs: 18 from attempt 1 and 18 from the retry, with each `y` read from v2's committed records.
  - Each pair is keyed `default_rng([83, i, j])`, with i counted within its v2 set. The 36 pairs therefore share 18 generator streams, each g2 pair with its g2r namesake (line 143; D82). The start randomness is drawn 18 times, not 36, and the paired surfaces differ only through y.
  - No new seed is drawn, and nothing touches the test set.
- **Runs.** Candidate (A) is run on all 36 pairs at v2's base lengths, with its one repeat, and with the direct runs under A and B wherever the reweighting rule calls for them.
  - Then come G2's grid checks on every accepted pair, with the whole-box check at s = 0.5 vol point, all under v2's tolerance rule.
  - The study runs at pool 4, after the pool fix.
- **The ratio estimator's route.** If Decision 3 extends (A) to the ratio estimator, the study also runs its route on stand-in targets, so that it does not reach G3 untested:
  - the reference's own log-target, passed as a callable `target`, which must give accepted runs within G2's tolerances of the least-squares route's;
  - the same target, displaced in H.
- **The verdict on (A).**
  - **Pass:** all 36 pairs accepted, every comparison within tolerance, and the projected cost of the test set's references, the optimiser's seconds included, within the budget. (A) is then adopted.
  - **Fail:** any pair not accepted after its repeat, or any comparison outside tolerance on an accepted pair. The report says which kind of failure it was:
    - non-adaptation: acceptance below 1%, or R-hat above 1.1, on a run;
    - slow mixing: R-hat within 1.01, but an effective size of H below 2,000;
    - a grid miss: the pair's first grid is then recomputed in the study at 64 points per dimension, to separate quadrature error from sampler error.
- **If (A) fails.** At most one declared revision of (A) is allowed, or else a move to (B), and then the same study is run again. A revision may change only the constants of steps 1 to 5, never the lengths, the acceptance or the tolerances. If the second study also fails, no v3 draft is written and the stop is declared.
- **What the study reports.** Per pair:
  - the acceptance over the sampling phase, and over each chain's first 500 burn-in steps (before the first adaptation, which tests the start and first proposal directly);
  - R-hat and the effective sizes;
  - m, and the number of starts that reached each kept solution;
  - Σ's eigenvalues at each kept solution, and how many were clipped;
  - the ratio of the sampling proposal's covariance to the accepted run's sample covariance, about 2.38²/4 if the adaptation is clean;
  - how many start coordinates were clipped to a face;
  - the optimiser's seconds, and the run's seconds;
  - the grid comparisons;
  - at every noise level, on the whole-box grid of line 143, the share of the posterior mass under A that lies outside the first grid's box. This is a diagnostic, compared with nothing.

  It also reports the 14 pairs v2 failed, compared with what v2's sampler did on them at the lengths each ran: for the first attempt's nine, one and two times the study's lengths; for the retry's five, two and four times.

v3's own G2 then checks the chosen design on surfaces it has not seen: four fresh θ, plus the two fixed points with fresh noise. The study will have seen the fixed points' θ at twelve pairs.

## Order of acts

1. **D89 merged; the stop notice.** On OSF, the author posts an update to `695fw` saying that v2 stopped at G2. The notice gives the reason, D89 and the record files, and says that a v3 is being developed on the seen pilot surfaces.
2. **D90.** The stop notice is recorded: its schema-response id and its submitted and approved times, read from the public API as D86 did.
3. **D91.** This plan is committed, in docs only. The entry records the author's answers to the decisions below. With them it records the list of what v3 does not change and the study's verdict rule, which then bind the study.
4. **D92.** The pool fix in `p5_g2_reference.py` and the pool-path test, in their own commit and entry, with `CONTRIBUTING.md`'s count.
5. **D93.** The development study. Its script and settings are committed before the run; the author runs it at pool 4; its output is committed and declared.
6. **D94.** The v3 draft, `docs/protocols/P5_protocol_v3_draft.md`: v2 with exactly the changes above, and the hypotheses, priors and everything else byte-identical where unchanged. With it come:
   - the adopted sampler in `p5_g2_reference.py` (and in `p5_g3_calibrators.py`, if Decision 3 takes the gradient route), with its tests and the docstring;
   - a transcription check: the written-in sampler is re-run on the study's pairs at the study's lengths, and must reproduce the study's records exactly. The same generator and one BLAS thread make the chains deterministic;
   - `docs/protocols/P5_v3_amendment.md`, carrying v3's OSF summary (what changed, why, and what had been seen), its title, and the paragraph that points `695fw` to v3.
7. **D95.** An independent, read-only review of the sampler change, the code and the v3 text against this plan, from a session that wrote none of v3. Its record is committed to `docs/protocols/` and cited at box 215. Its fixes are checked by the review's session before the merge, as D83's were.
8. **The freeze commit,** with no entry of its own, as `59ca589` had none. Its Status line carries the new hash list, G̃'s hash and the versions, and box 216 is reworded to "the act that follows this commit".
9. **OSF.** v3 is registered from a component holding exactly its files (Decision 10), giving a new ID and DOI. `695fw` is updated to point to it.
10. **D96 and D97.** D96 is the post-freeze commit, as D86 was: the registration, box 216 ticked, and the record. D97 then updates the living documents, as D87 did.
11. **v3's G2, then G3 and onwards.** These run from a clean tree at the frozen hashes, with any pool size, and write their records under v3's names.

## Decisions for the author

1. **The sampler.** (A) first, with (B) as a declared fallback. *Lean: yes.* (A) is the narrowest change that addresses what G2 showed. (B) is not narrow, as stated above.
2. **Where (A) runs.** Once per sampler run, inside `run_chains`, as written above; or once per `_run_with_repeat` call, with the repeat reusing the optima. *Lean: once per sampler run.* It is the simplest to write, test and cost: the repeat gets an independent search, and the optimiser's time falls inside each run's seconds.
3. **The ratio estimator's posterior.** The gradient route, or v2's start and first proposal kept for it. *Lean: the gradient route, run on stand-in targets in the study.* Keeping v2's start risks failures at 0.1 vol point that line 75 would charge to the calibrator.
4. **G1.** Adopt G̃ without re-running G1. *Lean: yes.* Its inputs are unchanged and its hash is committed. A rebuild would cost about 19 hours of pricing and give a file with another hash.
5. **G2.** Fresh seeds, the retry's start clause dropped, and the tolerances, the one retry at doubled lengths and the budget unchanged. *Lean: yes.*
6. **Order.** The pool fix comes before the study, in its own commit (D92), and D91's answers bind the study. *Lean: yes.* The study can then run at pool 4, and the fix is reviewed on its own.
7. **Review.** An independent review before the freeze, plus D94's transcription check. *Lean: yes.* v2's two reviews were read-only and neither read the pool path, and no test ran it. The pool-path test covers that gap; the review is for the sampler change.
8. **The stop notice on `695fw`,** posted before any v3 work, with its own entry. *Lean: yes.* A dated public stop makes v3's provenance clean.
9. **v3's records** go under new names, with v2's G2 files left in place. *Lean: yes.*
10. **The OSF component** for v3 holds the 15 files of v2's kind at the freeze commit, plus G̃, whose hash the Status line binds. The G2 records and the study's record are cited by path and blob hash in the summary. *Lean: yes.*
11. **Timing.**
    - **The study** runs during term, and is mostly machine time.
    - **The freeze** comes in November, and G2 in December.
    - **The test set's references** take about 12 to 14 hours of wall time on four workers, by v2's projections of 48.2 and 53.6 core-hours.
    - **G3 and the training** get no date until G3's first rung has shown a training time. G3's ladder (line 144) trains at most six sizes per cell, from 10,000 to 320,000 surfaces, over twelve cells, at up to 4 hours of wall time each: up to 288 hours, before the seeds and the scoring.

    *Lean: yes. Nothing here is urgent.*
