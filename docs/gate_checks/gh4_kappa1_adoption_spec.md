# Gate-check spec — κ=1 adoption (G-H4, steps 2a and 2b)

> **Driver committed:** 2026-06-22 (`d778074`) · **Spec reconstructed:** 2026-09-29
> from the driver scripts under RVL-009. The original specification was
> chat-only and is not recoverable. Predictions and gates below are as
> recorded in the code; this document does not attest when they were
> written.

**Status:** run and recorded. 2a gate **PASS**; 2b costed; verdict **ADOPT for the
conditional standard-MC path only — not for MLMC** (index of record:
`docs/gate_checks/README.md`, κ=1 adoption row). Recorded cost ratio `k1/k0` =
**0.79× (ε=0.05)** and **0.68× (ε=0.025)** — the ε=0.05 figure falls **outside** the
band predicted in the driver (0.6–0.75); see §8. · **Drivers:**
`gh4_kappa1_conditional.py` (2a), `gh4_kappa1_cost.py` (2b).
**Date:** neither driver records one; the run is logged under `ROADMAP.md` D22
(2026-06-23). · **Depends on:** `layer1b_mlmc_asian.py` — `PARAMS`,
`_cond_asian_payoff`, `_paths_from_increments`.

*Reconstructed from the two drivers named above. The code is authoritative: every
threshold, seed and sample count below is copied from it. Where the drivers do not
record **why** a constant has its value, §10 says so rather than supplying a
rationale.*

> Reconstruction discipline: this spec was written **after** the drivers ran, so it
> cannot claim the pre-registration that a spec written before the build earns. Facts
> are cited to the code; recorded run outcomes are cited to the artifact that holds
> them (`docs/gate_checks/README.md`, `ROADMAP.md`, the 2b docstring) and are marked as
> such. Nothing is inferred into a reason. Negative and off-prediction results are
> reported at the same volume as the wins.

---

G-H1a–d (fine path, `gh1_kappa1_finepath.py`) and G-H2a–c (coarse coupler,
`gh2_kappa1_coupler.py`) established that the κ=1 hybrid variance path is correct (fine
path closes the Volterra variance gap; the coarse coupler telescopes and is tight).
Correct is not the same as worth adopting. κ=1 buys a smaller weak error per grid
point and pays for it in per-path work: one extra `(N, n)` standard-normal array `Z`
and **two extra multiplies and two extra adds per cell** —
`W_near = c_near*dW1 + sig_perp*Z`, then `W_tilde = sqrt(2H)*(W_near + far)`
(`layer1b_mlmc_asian.py:209–211`) — plus a per-call `volterra_weights_kappa1(n,H,T)`
build on top of `volterra_weights(n,H,T)`, which it calls internally (`:178`) before
adding the kernel copy, the `g[0]` zeroing and the `v_k1`/`c_near`/`sig_perp` scalars
(`:179-186`); neither builder is cached. The FFT convolution
is the same size in both schemes (`:210` vs `:203`); the κ=1 design note also states
the FFT is unchanged (`kappa1_hybrid_coupling_design.md:203–205`).

G-H4 asks the only question that matters for production: **at matched accuracy, is
κ=1 cheaper than κ=0 for P2's winning estimator, the conditional standard-MC path?**
Step 2a measures whether κ=1 clears the bias budget on a coarser grid; step 2b prices
that saving against the measured overhead. The gate is written so that a negative
answer closes the adoption question rather than inviting a rematch.

---

## 0. Scope discipline — what these two drivers do, and what they do NOT

**IN.** Single-grid (standard-MC) conditional payoffs only, via
`_cond_asian_payoff(dW1, dW2, n, p, kappa, Z)` at one `n` at a time, at the
`PARAMS` defaults (H=0.10, η=1.50, ρ=−0.70, ξ₀=0.04, S₀=K=100, T=1, r=0;
`PARAMS["n0"]=32` is unused here — both drivers pass `n` explicitly). 2a additionally
calls `_paths_from_increments(..., "asian", kappa=1, Z=Z)` — the plain arithmetic
payoff — inside `unbiased_check` only (`gh4_kappa1_conditional.py:39`).

**OUT, and this is load-bearing for reading the verdict:**

- **No MLMC.** Neither driver calls `mlmc_asian_level` (`layer1b_mlmc_asian.py:329`),
  neither imports `layer1b_kappa1.py`, and no level difference `Y_l` is ever formed.
  The κ=1 coarse coupler is not exercised here at all.
- **No absolute bias.** "Truth" is *defined* as κ=1 at `n_fine=2048`. Every bias
  reported is a difference against that anchor, not against the continuum price.
- **No re-derivation in 2b.** Step 2b hard-codes 2a's answer in `NSTAR`
  (`{0.05: (32, 16), 0.025: (64, 32)}`, `gh4_kappa1_cost.py:26`) and its variance
  finding (≈1.13×) in its docstring. The two steps are chained by hand: 2a's `main()`
  returns `run2b` and nothing consumes it (`gh4_kappa1_conditional.py:170`).

Consequence: these drivers can only ever support **half** of a split adoption
decision — the standard-MC half. See §8.

---

## 1. The question, and why a naive proxy will not answer it

The whole gate turns on comparing two biases of order 10⁻²–10⁻¹ currency units.
The obvious design — price both schemes against an independent high-`n` run — is
rejected in the 2a docstring on a measurement-noise argument: a naive n=2048 proxy has
"s.e. ~0.003, comparable to the biases, and flips the gate" (`:6–7`). A gate that flips
on proxy noise is not a gate. So 2a measures the *differences* directly, with the
randomness shared, and keeps only one high-`n` batch.

---

## 2. The bias estimator — an exact decomposition of coupled differences

With `P0(n)`, `P1(n)` the conditional payoff at grid `n` under κ=0 and κ=1, and
truth taken as `E[P1(2048)]`, `measure()` accumulates

```
d2048      = E[P1(2048) − P0(2048)]                         # same-grid κ1−κ0 at the anchor
bias_k0(n) = E[P0(n) − P0(2048)] − d2048
bias_k1(n) = E[P0(n) − P0(2048)] + E[P1(n) − P0(n)] − d2048
```

This is an **identity, not an approximation**: by linearity the three terms telescope,
so `bias_k0(n) = E[P0(n)] − E[P1(2048)]` and `bias_k1(n) = E[P1(n)] − E[P1(2048)]`
exactly. The coupling buys variance on the estimator, nothing else — which is the
point, since the noise is what was going to decide the gate.

What is shared and what is fresh:

- `dW1f`, `dW2f` are drawn at `n_fine=2048`; the grid-`n` increments are
  **block-sums** of them (`dW1f.reshape(nb, n, bf).sum(2)`, `bf = 2048 // n`), so
  the κ=0 refinement difference is exactly coupled — no coarse coupler involved.
- The nearest-cell residuals are **not** coarse-grained: `Zf` at 2048 and `Zn` at
  each `n` are independent draws. So `P1(n) − P0(n)` is a *same-grid pair* sharing
  `dW` with a fresh `Z`, exactly as the docstring states (`:12–13`).

---

## 3. Unbiasedness pre-check — the control variate under κ=1

Before any bias measurement, `unbiased_check()` **reports** — advisory only; see below
— whether switching the variance path to κ=1 has broken the geometric control variate.
At `n=128`, `N=300_000`, seed `1`, batch `40_000`, it forms the paired difference
between the κ=1 conditional payoff and the κ=1 plain arithmetic payoff
(`_paths_from_increments(..., kappa=1, Z=Z)`) on identical `dW1, dW2, Z`, and prints

```
[unbiasedness @n=128] kappa=1 cond … vs naive arith …  |diff|=… z=…  OK / CHECK
```

with `OK` iff `z < 3` (`:48`).

**This is not an assertion and not a guard.** The function contains no `assert`,
returns `None`, and its output is not consumed at the call site (`:106`). A `CHECK`
verdict does not halt 2a and gates nothing downstream. (The distinction matters in
this doc set: `kappa1_hybrid_coupling_design.md:164–170` specifically credits the
coupler gates with *asserts* that a marginal check is blind to. There is no analogue
here.)

The standard error is conservative as a matter of arithmetic: it uses the raw second
moment of the paired difference, `sqrt(E[(a−c)²]/N)`, without subtracting the mean
(`:44`), so `se ≥ sd/√N`.
The check tests *mean-preservation of the control at κ=1*; it does not check that κ=1
is unbiased for the continuum, which is G-H1's job.

---

## 4. Step 2a — what is measured

- Grids `[16, 32, 64, 128]`; anchor `n_fine = 2048`.
- Seeds `[5, 11, 23]`, `N = 150_000` paths per seed, batch `B = 2000`.
- Per grid, per seed: `bias_k0`, `bias_k1`, and the single-grid variance ratio
  `vratio = Var(P1(n)) / Var(P0(n))` (population form, `E[X²] − E[X]²`).
- Reported as seed means with the across-seed standard deviation in **parentheses**
  (`:123`). That spread is `np.std` with the default `ddof=0` over the three seeds
  (`:115–116`) — a population sd of three numbers, not an unbiased estimate; read it
  as an indicator, not as a standard error.
- Also printed: the ratio `bias_k1/bias_k0` per grid, suppressed to `nan` when
  `|bias_k0| ≤ 1e-5` (`:122`).
- Then two headline scalars: the **bias ratio averaged over those grids with
  `|bias_k0| > 1e-5`** (`:125` — the same filter as the printed column, *not* an
  average over all four grids) and the variance ratio averaged over all four grids
  (`:126`).

---

## 5. The committed prediction (printed in the driver, before the gate logic runs)

The predictions are literals in the 2a output, printed at `:127–128`, i.e. before the
gate block at `:131` onward. The code alone cannot establish **when** they were
written relative to any run; no commit history is cited here. What follows is
therefore "the driver's stated predictions", not a pre-registration claim.

- **bias ratio ≈ 0.65×** — `"=> bias ratio (avg)  ~ {…}x   (predicted ~0.65x)"`.
  Neither driver compares the measured bias ratio against this or any other value:
  the printed ratio is never tested (see §10).
- **variance ratio ≈ 1.10×** — `"=> variance ratio    ~ {…}x   (predicted ~1.10x)"`.
  This prediction is also untested as such; what *is* coded is a separate, looser
  screen, `var_ratio < 1.6` (§6(ii)). Neither driver states a mechanism for the
  variance ratio being above 1 (see §10).
- **step 2b: net cost ratio `k1/k0` ≈ 0.6–0.75**, i.e. κ=1 ≈1.3–1.7× cheaper for
  the conditional std-MC path. The 2b banner spans two print calls and ends with a
  colon (`gh4_kappa1_cost.py:103–104`):

  ```
  VERDICT (predicted: grid halves; net cost ratio k1/k0 ~0.6-0.75,
  i.e. kappa=1 ~1.3-1.7x cheaper for the conditional std-MC path):
  ```

Note what is **not** predicted: no claim is made about β, about level variance,
or about MLMC. The expected win is a constant-factor one, sourced from the bias
constant.

---

## 6. Gate-check bar — G-H4 step 2a (PASS/FAIL, committed in code)

The admissible bias at tolerance ε is `thr = eps / np.sqrt(2.0)`
(`gh4_kappa1_conditional.py:134`) — `0.0354` at ε=0.05, `0.0177` at ε=0.025. The code
carries the formula and no comment; for what the `√2` is doing, see §10. `n_star()`
walks the grid list in ascending order and returns the **first** `n` with
`|bias(n)| ≤ thr`, or `None` (`:91–95`).

**(i) Grid saving, robust across seeds.** For each ε ∈ {0.05, 0.025}:

```
coarser  ⟺  n*_k1 ≤ n*_k0 // 2          (both non-None)
stable   ⟺  every seed in {5,11,23} reproduces the seed-averaged coarser flag
gate(ε)  ⟺  coarser AND stable
(i) PASS ⟺  gate(0.05) OR gate(0.025)
```

Printed as `"(i)  grid reduced >=1 level AND stable across seeds: PASS|FAIL"`, with
the per-seed `(n*_k0, n*_k1)` pairs and a `STABLE` / `FLIPS->noise` tag shown
alongside (`:138–148`). The stability clause is in the code; **no reason for it is
recorded in either driver** — see §10.

**(ii) Variance penalty bounded.** `var_ratio < 1.6`, on the grid-averaged ratio
(`:150`). Printed as `"(ii) Var_k1/Var_k0 < 1.6: {…}  PASS|FAIL"`.

**Both must hold** (`run2b = cond_i and cond_ii`, `:155`). On PASS the driver prints
`"GATE PASSES -> run STEP 2b (cost) for eps [...]"` and lists the qualifying ε.
On FAIL it prints, verbatim (`:161–163`):

```
GATE FAILS -> STOP.  kappa=1 does not buy a coarser bias-grid
(robustly, across seeds) for the conditional standard-MC path.
Adoption question closes as a clean NEGATIVE: no production role.
```

The FAIL branch therefore closes the question in the driver's own words rather than
inviting a re-tune.

---

## 7. Gate-check bar — G-H4 step 2b (matched-accuracy cost)

Step 2b is **documented** as running only because 2a's gate passed (`gh4_kappa1_cost.py:3–7`).
Nothing in the code enforces that: `main()` runs unconditionally under `__main__`,
reads nothing from 2a, and takes `NSTAR` from a module-level literal — see §0 and
§10. It compares each scheme **at its own `n*`**; the docstring justifies only the
*measurement* ("the per-path wall cost MEASURED empirically for each scheme at its own
n\*", `:9–10`), not the choice of grid.

```
cost(scheme) = (2/ε²) · Var(P_cond at n*) · c(n*)
```

- **`Var`** from `cond_var()`: `N = 200_000`, batch `B = 40_000`, per seed in
  `{5, 11, 23}`, population variance of the conditional payoff.
- **`c(n)`** from `per_path_cost()`: wall seconds per path, **measured, not
  assumed** — `batch = 20_000`, 2 untimed warm-up calls, then `reps = 12` timed
  calls of `_cond_asian_payoff`, divided by `reps · batch`; the reported `c` is
  the **median of 3** such timings (`:74–75`). Printed per grid in microseconds
  together with the measured overhead `c_k1/c_k0`. Grids timed: `{16, 32, 64}` — the
  union of the `NSTAR` pairs (`:71`).
- **Robustness.** The ratio `k1/k0` is recomputed per seed; the driver tags the row
  `"SIGN FLIPS -> inside noise"` when `(r.min() < 1) != (r.max() < 1)`, otherwise
  `"stable"` (`:95`).

**Bar — reconstructed, not present in the code.** 2b computes **no PASS/FAIL string of
its own**: it prints the cost table, the per-seed ratios, the flip tag, and the
*predicted* band, then stops. The only threshold the code uses is the unity reference
inside the sign-flip test (`:95`). Read against that, the bar this spec records is:
adopt only if the seed-averaged `k1/k0 < 1` with no sign flip, at an ε that passed 2a.
The 2b verdict is taken by a human reading the table against that bar.

---

## 8. The decision rule that separates adoption from rejection

Adopt κ=1 as the variance-path scheme **for a named estimator**, not globally.
For estimator *E*, adopt iff all three hold:

1. **Grid saving** — κ=1 meets *E*'s bias budget `ε/√2` at least one refinement
   level coarser than κ=0, with the same verdict on every seed (§6(i)).
2. **Bounded variance penalty** — `Var_k1/Var_k0 < 1.6` (§6(ii)); i.e. the extra
   paths needed cannot eat the grid saving.
3. **Net cost** — matched-accuracy cost ratio `k1/k0 < 1`, with the per-path
   overhead measured at each scheme's own `n*` and no sign flip across seeds (§7).

Clause 1 is only evaluable for an estimator whose cost is set by a *single* grid.
For MLMC it is the wrong currency entirely: MLMC cost is `(2/ε²)(Σ√(V_l C_l))²`,
governed by β and the per-level variance `V_l`, and **neither driver computes a
level difference**. So 2a/2b can return `ADOPT` for the conditional standard-MC
path and say nothing whatever about MLMC. Per the index of record, the negative half
of the split verdict is owned by **G-H2c** — "β unchanged, larger level variance — see
G-H2c" (`docs/gate_checks/README.md`, κ=1 adoption row). Any future claim of the form
"κ=1 helps/hurts MLMC" must cite that, never these two drivers.

*Do not cite "G-H3" for this.* Under the driver/README scheme there is no G-H3. Under
the design note's older scheme (§8 of `kappa1_hybrid_coupling_design.md`), G-H3 is a
**planned** per-level variance-factor gate (`:173`, "expected modest") with **no
recorded verdict anywhere in the repo**. See §9.

**Recorded outcome** — from the index of record (`docs/gate_checks/README.md:78–90`),
`ROADMAP.md:740` (D22) and the 2b docstring. *Not computed in the source read for this
spec, and neither driver was executed for it:*

- Gate passes: `n*` halves 32→16 at ε=0.05 and 64→32 at ε=0.025, identical across all
  three seeds.
- Bias ratio **0.59×** against a predicted ~0.65× (`ROADMAP.md:740`); the README
  records it as "~0.5–0.7× at the grids that matter" (`:81`). This is the prediction
  that landed.
- `Var_k1/Var_k0 ≈ 1.13 < 1.6`; measured per-path overhead ≈1.08×.
- Cost ratio `k1/k0` = **0.79× (ε=0.05)** and **0.68× (ε=0.025)**, stable across
  seeds. The ε=0.025 figure lands inside the predicted 0.6–0.75 band. **The ε=0.05
  figure misses it on the expensive side** — 0.79 > 0.75, i.e. 1.27× cheaper against a
  predicted floor of 1.3×. The index and ROADMAP both record the achieved result as
  "~1.3–1.5× cheaper", narrower than the predicted "~1.3–1.7× cheaper"
  (`gh4_kappa1_cost.py:104`).

Verdict: **ADOPT for conditional standard MC only; not for MLMC** — with the ε=0.05
band miss recorded above as part of the result, not rounded into it.

---

## 9. Honest scope note

- **"Truth" is κ=1 at n=2048, not the continuum.** Neither driver checks that
  2048 is itself converged. A residual κ=1 discretisation error `δ` at the anchor
  shifts `bias_k0` and `bias_k1` by the *same additive amount*, so it cancels in
  their **difference** — and in nothing else. It does **not** protect the ratio
  (`(b1+δ)/(b0+δ) ≠ b1/b0` unless `δ=0` or `b1=b0`), and it does **not** protect
  `n*`: `n_star` thresholds `|bias| ≤ thr` (`:91–95`), and a common offset can move
  the two series across that threshold by different numbers of grid levels — which is
  exactly what gate (i) reads. So neither of the two headline readings the gate rests
  on (the bias ratio, `:125`; the `n*` comparison, `:135–136`) is anchored to the
  continuum, and nor is any absolute reading of the bias column.
- **Gate-id collision across the doc set.** The drivers and
  `docs/gate_checks/README.md` use G-H1a–d (fine path), G-H2a–c (coupler: a =
  telescoping, b = coupling tightness, c = rate β≈2H) and G-H4 (adoption, this spec).
  `kappa1_hybrid_coupling_design.md` §8 uses an older, incompatible scheme: G-H1 =
  coarse-law exactness + coupling tightness + cross-covariance, G-H2 = rate, G-H3 =
  variance factor, G-H4 = Giles cost. The two are **not** reconciled in the repo and
  this spec does not reconcile them; it records the discrepancy and uses the
  driver/README scheme throughout. A reader following a "G-H2" or "G-H3" citation must
  first establish which scheme the citing document is on.
- **The timed cost excludes the Gaussians.** In `per_path_cost` the `dW1`, `dW2`
  and `Z` arrays are drawn *before* `time.perf_counter()` and reused across all
  reps (`gh4_kappa1_cost.py:33–39`). The measured overhead therefore prices the extra
  κ=1 arithmetic but **not** the extra `(N, n)` standard-normal draw that κ=1
  requires. This biases the measured overhead **downwards**, i.e. in favour of
  adoption. The design note expected the κ=1 overhead to be "dominated by the extra
  Gaussians" (`kappa1_hybrid_coupling_design.md:203–205`), which is precisely the term
  the timing omits. The size of the effect is not quantified anywhere.
- **The variance gate is not evaluated at `n*`.** `cond_ii` uses the mean of
  `vratio` over all four grids `{16,32,64,128}` (`:126, :150`), not the ratio at the
  adopted pair. 2b then uses the correct per-`n*` variances in the cost formula, so
  the cost verdict is unaffected; only the 1.6 screen is coarser than it looks.
- **`n_star` assumes monotone bias.** It returns the first passing entry in
  ascending grid order, which is the coarsest admissible grid *only if* `|bias|`
  decreases in `n`. No monotonicity check is performed.
- **Structural ceiling on the gate.** If κ=0 already clears the budget at the
  coarsest grid (`n*_k0 = 16`), then `n*_k0 // 2 = 8` is not in `grids`, `n*_k1`
  cannot satisfy `n1a <= n0a // 2` (`:136`), and clause (i) must FAIL regardless of how
  accurate κ=1 is. The grid list (`:108`) bounds the detectable saving to 16…128.
- **Sequential RNG consumption in `measure`.** `Zn` is drawn *inside* the grid
  loop, after the block-sums (`:70`), so the random stream seen by any one grid depends
  on which other grids are present in `grids` and in what order. Reproduction
  requires the exact list `[16, 32, 64, 128]`, not merely the seed.
- **2a's results header hard-codes `H=0.10`** as a string literal (`:119`) rather than
  reading `p["H"]`; it would silently misreport if `PARAMS` changed.

---

## 10. Open items — intent not recoverable from the drivers

*Questions this reconstruction could not answer from the code are recorded in [`RVL-009_open_items.md`](RVL-009_open_items.md).*
