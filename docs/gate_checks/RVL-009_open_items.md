# RVL-009 — open items from the gate-check reconstruction

*Dated 2026-09-30.* This is the record of **questions the reconstruction could
not answer from the code**. Five gate-check specifications in this directory
were rebuilt from their driver scripts under RVL-009 (`docs/AUDIT_TAIL.md`); the
originals were chat-only and are not recoverable. Where a driver fixes a
threshold, a seed or a sample count without recording *why*, the question is
written down here rather than answered with a plausible reconstruction.

Every item carries a `file:line` citation into the driver, engine or record
document it concerns, so the question can be taken straight to the code.

A gap left visible is worth more to the audit trail than a rationale invented to
close it. Nothing here is a defect: these are unanswered questions about intent,
not findings against the code.

---

## p2_antithetic_build_and_verify.md

1. **Where 1.45, 0.87 and 2.0 come from — partly recoverable.** *Recoverable:* the ~1.45× variance factor is measured with a per-H breakdown and the 0.87× efficiency is derived arithmetically as 1.45/1.67, both in `docs/p2_estimator_results.md:40-46`; the 1.45-versus-2.5/1.5 comparison is also in the engine at `layer1b_mlmc_asian.py:348-349`. *Not recoverable:* the **2.0** "linearised guess" / "leading-order pre-check" — every artifact labels it and none contains it. Also outside the repo: a superseded chat-era original of this document survives at `C:/Users/micha/Downloads/p2_antithetic_build_and_verify.md` (7,071 bytes, dated 22 June), which `README.md:105-106` points at. The drivers were first committed the same day, about eight hours later (2026-06-22 14:43 vs 23:04), so the pre-registration ordering holds by hours, not months. It presents 1.45 / 0.87 as findings of an earlier sandbox run to be re-verified on-machine — which establishes them as pre-registered *relative to the driver runs* — but it supplies no derivation of the 2.0 pre-check either. **It must not be transcribed into the repo:** its §1 per-H variance ratios differ from the recorded run (1.48 vs 1.49 at H = 0.05), its G-A4 method is the free-running adaptive cost ratio that both drivers refute as an L-selection artifact (§4), it directs the reader to the dead `p2_coupling_gate_check.md`, and it is written as an operator prompt rather than a spec. (`p2_antithetic_gatecheck.py:157-163`, `p2_antithetic_verify.py:400-410`)

2. **Why the consistency gate is 0.51.** Production prints the check against `< 1` (`layer1b_mlmc_asian.py:540-541`) and its docstring asks only for ≈0 within Monte-Carlo noise; elsewhere in the project the telescoping bar is `< 1` (`gh2_kappa1_coupler.py:135`, `README.md:25`). Only the antithetic drivers harden to 0.51, with no comment. Whether 0.51 is a measured naive baseline plus margin, or an arbitrary "half", is not recoverable from code. (`p2_antithetic_gatecheck.py:51-54`, `p2_antithetic_verify.py:232-233`)

3. **Why the two drivers commit different bands.** P1: 0.05 vs 0.10 on `abs(Δβ)`. P2: `[1.30, 1.60]` vs `[1.25, 1.65]`. P3: `≤ 1.0` vs `≤ 1.001`. Both centre on the same predictions; no rationale for the differing widths, nor for which is canonical, appears anywhere. (`p2_antithetic_gatecheck.py:147-162`, `p2_antithetic_verify.py:391-421`)

4. **The G-A2 tolerance form** `0.6·(2H) + 0.05` — a 60 % relative band plus an absolute floor — and the verify script's near-4H band `0.3·(2H)`. Generous enough that β could sit some way off 2H and still pass; why these widths is unstated. (`p2_antithetic_gatecheck.py:147-148`, `p2_antithetic_verify.py:387-388`)

5. **The G-A1(a) noise band** `3·exp(η²v[-1]/2)/√N`. The factor 3 and the use of `v[-1]` (the worst-case `t = T` variance) for a bound applied at every `t` are both unexplained. (`p2_antithetic_gatecheck.py:44-45`, `p2_antithetic_verify.py:219-220`)

6. **Seeds 1 and 123**, and the choices `n = 256`, `N = 60,000` (seed 1's engine-level forward-variance check) and `nb = 64` / `n_f = 8` (seed 123's mechanical construction pre-gate). Not in the documented 7/23/11 allocation and not justified in code. (`p2_antithetic_gatecheck.py:38-39`, `p2_antithetic_verify.py:185-186`, `p2_antithetic_verify.py:212-214`)

7. **The two drivers' G-A4 numbers are not strictly like-for-like, for two separate reasons.** (a) `Lmax` on `_choose_L`: gatecheck caps at 6, so `L*` never exceeds the pilot depth and no variance extrapolation happens; verify leaves the default 9 and extrapolates (`_ext_VC`). Whether the cap is a considered choice or an artefact of reusing the pilot depth is not stated. (b) α: gatecheck floors α at 0.5 (line 124), verify feeds the unfloored pilot α into `_choose_L` (lines 106, 326, 347) and floors only inside `adaptive2`. Since `_choose_L` divides by `2^α − 1`, the two drivers can select different `L*` from the same pilot. No file comments on either difference. (`p2_antithetic_gatecheck.py:124-128`, `p2_antithetic_verify.py:295-302`, `p2_antithetic_verify.py:326-347`)

8. **Unused imports and a stale docstring.** `_level_cost_coef` is imported but never called in `p2_antithetic_gatecheck.py` (line 27); `_bs_call` (line 37) and `import layer1b_mlmc_asian as L` (line 35) are unused in `p2_antithetic_verify.py`, whose header (lines 6-7) also claims to import `mlmc_asian_level`, which it does not. Harmless. No in-file use and no comment; the intent behind any of them is unknown. (`p2_antithetic_gatecheck.py:26-27`, `p2_antithetic_verify.py:35-37`)

9. **Both `gate_a1()` return values are discarded** (§1.6). Whether G-A1 was meant to gate the summary and the wiring was forgotten, or whether it was always advisory, cannot be determined from the code. (`p2_antithetic_gatecheck.py:54-54`, `p2_antithetic_gatecheck.py:185-185`, `p2_antithetic_verify.py:246-246`, `p2_antithetic_verify.py:451-451`)

10. **No mechanism statement in either driver.** The drivers give the falsifier ("refute if ~4H") but never say why 4H is the alternative or which error term the swap was expected to cancel. The only account in the repo is the asserted one in `docs/p2_estimator_results.md:38-43` (§10). (`p2_antithetic_gatecheck.py:147-149`, `p2_antithetic_verify.py:384-391`)

11. **Whether the two-driver redundancy was designed for integration-independence.** Gatecheck states it "end-to-end-checks the integrated flag" (lines 4-5); verify states non-invasiveness (lines 5-8) and "independent verification" (line 2). That the pair was built so the result would not depend on the integration being correct is a reading of the two headers, not a stated purpose. (`p2_antithetic_gatecheck.py:1-15`, `p2_antithetic_verify.py:1-22`)

12. **Variance-factor discrepancy across artifacts (pre-existing).** `README.md:22,37` records ≈1.44×; `layer1b_mlmc_asian.py:349` and `docs/p2_estimator_results.md:40` record ~1.45×. Both describe the same measured quantity. Which is the corrected value is not determinable from the repo; this document quotes the README's 1.44× in §9 and flags the other. (`layer1b_mlmc_asian.py:348-349`)

13. **G-A3 has no per-level PASS/FAIL.** Only the pooled mean across all (H, l) is tested (§1.4). Whether per-level scatter was judged uninformative, or the gate was simply never tightened, is not stated. (`p2_antithetic_gatecheck.py:155-156`, `p2_antithetic_verify.py:398-404`)

14. **Free-running coverage differs between drivers.** Gatecheck runs the confounded free-running comparison at all three ε (line 106); verify runs it at ε ∈ {0.10, 0.05} only (line 336). Since the output is discarded for the verdict in both, the asymmetry is probably immaterial — but no file says so. (`p2_antithetic_gatecheck.py:101-115`, `p2_antithetic_verify.py:336-341`)

---

## p2_conditional_gate_check.md

*These are recorded as gaps rather than rationalised. A reconstruction with marked holes is worth more to the audit trail than a plausible invention.*

1. **Why `z < 3.5` (unit test) and `z < 4` (unbiasedness)?** — Two different thresholds for two Gaussian-`z` checks, neither justified in the code. The unit test takes a max over 4 paths and the unbiasedness check a max over 7 levels, so a multiplicity correction would push the *other* way. Unexplained. (`p2_conditional_verify.py:119-121`, `p2_conditional_verify.py:230`)

2. **Why is unbiasedness advisory rather than a G-C4 clause?** — Its exclusion from the conjunction at line 318 is verified; that the exclusion was deliberate is not recorded anywhere, and no reason is reconstructed here. (`p2_conditional_verify.py:318`)

3. **Why `n = 48` in the unit test?** — Not a grid the MLMC ladder ever uses (`n0·2^l` gives 32, 64, …), so the closed form is validated off-ladder. No reason given. (`p2_conditional_verify.py:106`)

4. **Why level `l = 3` and `N = 8000` for the κ timing?** — κ may drift with `n` (FFT versus `O(n)` terms); one level is measured and applied to all. Since the headline ratio is κ-invariant (§4), the choice only affects the naive-column comparisons, but that trade-off is not stated. (`p2_conditional_verify.py:164-182`)

5. **Why is the standard-MC cost charged the bare `n0·2^{L*}` while every MLMC level is charged `1.5×`** — (lines 248–249 versus line 232)? Arguably correct — single-level MC simulates one grid — but the asymmetry is never defended, and it is the one modelling choice that moves the naive-column comparisons. (`p2_conditional_verify.py:232`, `p2_conditional_verify.py:248-249`)

6. **Why the seeds 11 / 99 / 1234, and why three?** — Seeds 7 / 23 / 11 serve the antithetic thread (`p2_antithetic_gatecheck.py:49,66,108,122`) and 5 / 11 / 23 the κ=1 adoption thread (`gh4_kappa1_conditional.py:109`), so the project reuses no fixed triple. No provenance. (`p2_conditional_verify.py:209`)

7. **Why the *value* `0.5` in the `α` floor `max(0.5, …)`?** — The floor itself is not unexplained: it mirrors the production selector, whose comment (`layer1b_mlmc_asian.py:638-640`) attributes it to Giles' practice and to guarding a noise-dominated or negative fitted slope from corrupting the `2^α − 1` bias test. What remains unexplained is the choice of `0.5` specifically, which feeds `L*` through `2^α − 1`. (`p2_conditional_verify.py:217`)

8. **Why `ε ∈ {0.10, 0.05}`** — here, when the κ=1 adoption thread reports at ε ∈ {0.05, 0.025} (`gh4_kappa1_conditional.py:133`, `gh4_kappa1_cost.py:26`)? The two cost comparisons are therefore not stated at a common accuracy. (`p2_conditional_verify.py:235`)

9. **The mechanism behind the §5 prediction, and the origin of the `~4.2x` / `~3.2x` reference figures — both undetermined.** — The natural mechanism (that `W_perp` contributes a *level-independent* share of single-level variance, so the coupled level difference has less left to remove) appears nowhere: not in `p2_conditional_verify.py`, its docstring, `layer1b_mlmc_asian.py`, or the docs. Only the second half — that the coupling has already cancelled much of the common variance — has an in-repo source (`docs/p2_estimator_results.md:76-79`). Likewise, the hard-coded `[predicted ~4.2x]` / `[predicted ~3.2x]` annotations (lines 278, 280) have no derivation in the driver. `ROADMAP.md:739` records the *directional* prediction as pre-registered, but `docs/p2_estimator_results.md:72-77` records the measured values as 4.16× (4.21× deep) and 3.19× (3.30× deep) — matching the annotations to within rounding, which is equally consistent with a prior stated before the run and with a back-fill after it. Undetermined; this spec asserts neither. (`p2_conditional_verify.py:277-280`)

10. **Fact about ordering, with no stated reason.** — The driver runs this check first: `assert unit_test_Egeom(PARAMS, args.quick)` at line 337 precedes `gate_c4(args.quick)` at line 338, and a failure aborts before any gate number is produced. The driver offers no justification for that ordering, and none is reconstructed here. (`p2_conditional_verify.py:337-338`)

---

## p2_conditional_build_and_verify.md

*These are points the code fixes without stating a reason, and no repo artifact supplies one. They are left open rather than rationalised:*

1. **Derivation of the `2/ε²` prefactor and the `eps/√2` tolerance** — The derivation of the `2/ε²` prefactor (`:200`, `:248-249`) and of the paired `eps/√2` bias tolerance (`:193`). Both are consistent with a standard MSE split, but the code and docstrings state only the formulae; no split is named anywhere in the repo. (`p2_conditional_verify.py:193,200,248-249`)

2. **Provenance of the predicted `~4.2×` / `~3.2×` factors** — The provenance of the predicted `~4.2×` / `~3.2×` factors. They exist only as literal text in print strings and are never tested numerically. (`p2_conditional_verify.py:278-280,290-293`)

3. **Unit-test grid `n = 48` and the `3.5` vs `4` thresholds** — Why the `Egeom` unit test uses `n = 48`, off the production grid; and why its threshold is `z < 3.5` while the §5 unbiasedness check uses `z < 4`. (`p2_conditional_verify.py:106,119-121,230`)

4. **Unbiasedness advisory, `signflip` excluded from `overall`** — Why the unbiasedness check is advisory rather than gating, when `docs/gate_checks/README.md`:46 lists unbiasedness as part of the gate; and why `signflip` is printed but excluded from `overall`. (`p2_conditional_verify.py:228-230,305,312,318`)

5. **`measure_kappa` constants, and κ\* charged at every level** — Why `measure_kappa` fixes `l = 3`, `N = 8000`, `reps = 8`, and whether a single κ\* measured at one level is expected to hold at all levels (`Cc = kappa·Cn`, `:233`) when the post-process is O(n) and the FFT O(n log n). (`p2_conditional_verify.py:164-173,233`)

6. **Choice of seeds 11 / 99 / 1234, 0 and 7** — Why the seeds are 11 / 99 / 1234, `measure_kappa` uses seed 0 and the unit test seed 7. (`p2_conditional_verify.py:105,166,209`)

7. **Restriction of ε and `Lmax`** — Why ε is restricted to `{0.10, 0.05}` and `Lmax` to 6. (`p2_conditional_verify.py:185,208,235`)

8. **`_choose_L` scan start, imposing `L* ≥ 2`** — Why `_choose_L` starts its scan at `Lc = 2`, imposing `L* ≥ 2` — deliberate floor or artefact of the `min(3, Lc)` tail window. (`p2_conditional_verify.py:190-192`)

9. **Inline re-implementation vs the engine's `_cond_asian_payoff`** — Why the driver re-implements the path build and payoffs inline rather than calling the engine's `_cond_asian_payoff`, and therefore whether G-C4 is intended to certify the production path at all. (`p2_conditional_verify.py:48,52-97`)

10. **`beta_naive` / `beta_cond` fitted but never read** — Why `beta_naive` / `beta_cond` are fitted and printed but never read. (`p2_conditional_verify.py:218-221`)

11. **Magnitude of the `1e-300` `sigG` floor** — The `1e-300` literal in the `sigG` floor (`:92`). Its *effect* is verified (§3); the choice of magnitude is unexplained. (`p2_conditional_verify.py:92`)

12. **Which headline ratio is correct, `0.41–0.45` or `≈0.42–0.45`** — Which of `0.41–0.45` (`docs/gate_checks/README.md`) and `≈0.42–0.45` (`docs/p2_estimator_results.md`:90) is the correct headline. The driver stores neither, and the two records disagree. (`p2_conditional_verify.py:253,257-258`)

13. **Which z the figure `≈1.5` belongs to** — Which z the figure `≈1.5` belongs to — the §5 unbiasedness check (`docs/gate_checks/README.md`:46) or the §4 closed-form unit test (`docs/p2_estimator_results.md`:69). The two records disagree and the driver stores neither. (`p2_conditional_verify.py:118-121,227-230`)

---

## gh1_kappa1_fine_path_spec.md

1. **Why 1%** in G-H1a is not stated anywhere in the code. (`√(2/N) ≈ 0.41%` for a Gaussian variance
   estimator at N=120_000, so the bar sits near 2.4 standard errors — but that arithmetic is this
   reconstruction's, not the script's.) (`gh1_kappa1_finepath.py:31,46`)
2. **G-H1b's PASS constants `2×`, `+5`, `4×` are unexplained.** The additive `+5` in particular makes
   the correct-compensator bar loose in absolute terms; whether that slack is deliberate tolerance
   or an arbitrary cushion cannot be told from the code. Note that the multiplicity explanation could
   only ever apply to the **correct-compensator** leg: the trap leg's bias is constant in t (§3), so
   its `max_t` selects noise, not a worst-case time. (`gh1_kappa1_finepath.py:94-96`)
3. **Seeds 100 / 7 / 2 / 11 have no stated provenance**, and the README index records this thread's
   seeds as *"fixed internal (1/2/7/100/999)"* (`README.md:24`) — which neither includes 11 nor
   matches the four literals in the driver; 1 and 999 appear nowhere in it. One of the two records is
   wrong; the code cannot say which. (`gh1_kappa1_finepath.py:38,69,109,134`)
4. **Why `n=256` for G-H1a/b but `n=128` for G-H1c/d** is not stated.
   (`gh1_kappa1_finepath.py:31,64,107,131`)
5. **The hard-coded `pred ~0.9996` / `~0.8529` are the *analytic* ratios** (§2 reproduces them
   exactly) but are printed against the *empirical* columns, which come out at 0.9973 / 0.8520 —
   a 0.23% / 0.11% gap, far outside the analytic residual and far inside the 1% gate. Whether they
   were intended as a target for the analytic column or as a loose eyeball bar for the empirical one
   is ambiguous; likewise whether the printed, ungated κ=0 column was ever meant to be checked (§4).
   And the code cannot date the literals at all, so "committed before measuring" is an inference.
   (`gh1_kappa1_finepath.py:44-53`)
6. **The recorded correct-compensator `z = 2.86` does not reproduce** (§6, Discrepancy 1): the
   committed seed gives 5.4, on both the current and the 2026-06-22 engine, while every other
   recorded figure reproduces. Neither the driver nor the README records a run date, a log artifact
   or an environment — the driver writes no output file — so the source of the recorded value cannot
   be identified. Unresolved. (`gh1_kappa1_finepath.py:69,96`; record at
   `docs/gate_checks/README.md:61`)
7. **Whether "≥1e5 paths" in the G-H1a banner is a committed minimum or a description** of the
   N=120_000 actually used. (`gh1_kappa1_finepath.py:29,31`)
8. **G-H1d's cell choice `{1, 32, 64, 127}` and its exclusion of cell 0**, and the asymmetric 2%/3%
   Var/Cov tolerances, have no comment. Index-independence (§7) makes the choice moot, but it leaves
   open whether a positional check was *intended* and quietly does not bite.
   (`gh1_kappa1_finepath.py:143,147`)
9. **The two gate-numbering schemes (§0) are unreconciled in the record.** Of the design note's §8
   G-H1 contents, only **coupling tightness** was renumbered into the driver/README scheme, as
   **G-H2b**; **cross-covariance was not renumbered at all** — it left the gate ladder and lives as a
   unit check in `kappa1_coupling_design_check.py:127`; and the design note's **G-H3 (variance
   factor)** has no counterpart anywhere in the drivers or the README. Whether this was deliberate
   rescoping once the coarse coupler became its own driver, or drift, is visible in the record but
   its reason is not. (`kappa1_hybrid_coupling_design.md:157-178`)
10. **No run date for the recorded PASS.** The driver was committed 2026-06-22 (`d778074`) and
    `README.md:12` attributes its numbers to *"this session's driver runs"*, but no run date and no
    log artifact is stored anywhere in the repo. (`gh1_kappa1_finepath.py:154-174` — `main()` prints
    to stdout only; no `open(`/`savefig`/`to_csv` anywhere in the file)

---

## gh4_kappa1_adoption_spec.md

1. **Why `Var_k1/Var_k0 < 1.6`** — No derivation accompanies 1.6. It is not obviously the break-even penalty implied by a one-level grid saving at the recorded ~1.08× overhead, and the driver does not say what it is meant to exclude. **Undetermined.** (`gh4_kappa1_conditional.py:150`)

2. **Why ε ∈ {0.05, 0.025}** — Not stated. Both values also appear in the P2 ε lists — which additionally include 0.10 (`p2_antithetic_gatecheck.py:101`, `p2_antithetic_verify.py:343`) and 0.2/0.1 (`p2_baseline_regen.py:94`) — so the *pair as such* does not appear there, and neither gh4 driver references them. **Undetermined.** (`gh4_kappa1_conditional.py:133`, `gh4_kappa1_cost.py:26`)

3. **Why the `(2/ε²)` prefactor** — It is exactly the path count for a statistical standard error of `ε/√2`, the same `ε/√2` used as the bias threshold in 2a (`gh4_kappa1_conditional.py:134`), so the two constants are mutually consistent with an equal bias/noise split of an RMSE budget. **Neither file states that argument**; the consistency is an observation about the two literals, not a recorded rationale. (`gh4_kappa1_cost.py:88-89`)

4. **Why `n_fine = 2048`** — `N = 150_000` and `B = 2000` in 2a versus `N = 200_000`, `B = 40_000` in 2b. The 2a docstring justifies *anchoring* rather than using an independent proxy, but not the choice of 2048, and nothing checks that 2048 is converged. The small 2a batch is arithmetically consistent with the memory footprint of three `(B, 2048)` arrays (`gh4_kappa1_conditional.py:60-62`) against 2b's widest `(B, 64)`, but no reason is recorded. **Undetermined.** (`gh4_kappa1_conditional.py:51`, `gh4_kappa1_cost.py:45`)

5. **Why seed `1` for `unbiased_check`, `0` for the cost timings, and `z < 3`** — for the unbiasedness pass mark. Unexplained. `z < 3` is a conventional 3σ screen; the file does not say so, and the standard error it is applied to is the uncentred second moment rather than the variance of the difference (`gh4_kappa1_conditional.py:44`). **Undetermined.** (`gh4_kappa1_conditional.py:31`, `gh4_kappa1_cost.py:33`, `gh4_kappa1_conditional.py:48`)

6. **Whether the conservative `se` in `unbiased_check` was intended.** — The formula `sqrt(E[(a−c)²]/N)` is conservative as arithmetic (§3); the file records no intent. **Undetermined.** (`gh4_kappa1_conditional.py:44`)

7. **The independent proxy is gone.** — The 2a docstring argues against a naive n=2048 proxy on noise grounds, and the index records seeds "5 / 11 / 23 (+proxy 999)" (`docs/gate_checks/README.md:26`). Seed 999 appears in **neither** driver as read. Either the proxy arm was removed after the argument was made, or it lived in a variant not in the repo. The audit trail cannot say which. **Undetermined.** (`gh4_kappa1_conditional.py:6-8`)

8. **No falsification criterion is attached to the printed bias-ratio prediction.** — The `~0.65x` literal (`gh4_kappa1_conditional.py:127`) is printed and never compared; the only coded conditions in 2a are `cond_i` (the `n*` comparison, `gh4_kappa1_conditional.py:136-149`) and `cond_ii` (`var_ratio < 1.6`, `gh4_kappa1_conditional.py:150`). What a bias ratio of, say, 1.0 would have done to the gate is not expressed in the code. **Undetermined.** (`gh4_kappa1_conditional.py:127`)

9. **No mechanism is recorded for the variance ratio.** — κ=1 both adds the fresh nearest-cell term `sig_perp * Z` **and** swaps the kernel (`g_hyb`, with `g[0]` zeroed) and the compensator (`v_k1`) (`layer1b_mlmc_asian.py:208–212`). Neither driver attributes the ~1.13× to any of these. **Undetermined.** (`layer1b_mlmc_asian.py:208-212`)

10. **No reason is recorded for the per-seed stability clause** — in gate (i) (`gh4_kappa1_conditional.py:138-148`). As index context only — not as the driver's stated reason — the README records a structurally similar artifact in the antithetic thread ("The free-running adaptive driver's apparent 'win' is an L-selection artifact (flips sign across seeds)", `docs/gate_checks/README.md:38–39`), and 2b's docstring uses a sign-flip argument for cost ratios (`gh4_kappa1_cost.py:11–12`). Neither driver mentions the antithetic thread; the word does not occur in either file. Whether that experience motivated this clause is **undetermined.** (`gh4_kappa1_conditional.py:138-148`)

11. **2b has no programmatic verdict, and the 2a→2b chain is manual.** — 2b prints a predicted band and the table, never a computed PASS/FAIL, and 2a's `run2b` return value is discarded at the call site (`gh4_kappa1_conditional.py:170`). Whether the hand-off is a deliberate human-in-the-loop step or an unfinished chain is not determinable from the code. **Undetermined.** (`gh4_kappa1_cost.py:102-106`, `gh4_kappa1_conditional.py:170`)

12. **Whether excluding Gaussian generation from the timed region was deliberate** — (isolating the payoff kernel) **or an oversight** (`gh4_kappa1_cost.py:33–39`). The direction of the resulting bias is clear (§9); the intent is not recorded. **Undetermined.** (`gh4_kappa1_cost.py:33-39`)

13. **Why gate (ii) averages the variance ratio over all four grids** — rather than evaluating it at the adopted `n*` pair (`gh4_kappa1_conditional.py:117, :126, :150`). No rationale is recorded. **Undetermined.** (`gh4_kappa1_conditional.py:117`, `gh4_kappa1_conditional.py:126`, `gh4_kappa1_conditional.py:150`)

14. **A `z ≈ 1.5` figure floats in the doc set and belongs to neither driver here.** — `docs/gate_checks/README.md:46` attributes `z≈1.5` to the conditional-MC unbiasedness check; `docs/p2_estimator_results.md:69` attaches `z < 1.5` to the closed-form pre-gate validation instead — two different checks, both of which exist in `p2_conditional_verify.py` (`:105-122` and `:223-230`), so both records may be individually correct. The observable fact is that the same ~1.5 figure is attached to two different checks in two documents. Neither gh4 driver stores either value — 2a's screen is `z < 3` on a different quantity (§3). Flagged here only so the numbers are not conflated; which record is right is **undetermined** and is not this spec's to settle. (`gh4_kappa1_conditional.py:48`)
