# Antithetic MLMC coupling — gate spec (G-A1…G-A4) and build-and-verify record

> **Driver committed:** 2026-06-22 (`d778074`) · **Spec reconstructed:** 2026-09-29
> from the driver scripts under RVL-009. The original specification was
> chat-only and is not recoverable. Predictions and gates below are as
> recorded in the code; this document does not attest when they were
> written.

**Status:** **REFUTED** (documented negative result) · **Gates:** G-A1, G-A2, G-A3, G-A4 · **Drivers:** `p2_antithetic_gatecheck.py` (runs the gates *through* production `layer1b_mlmc_asian.py`), `p2_antithetic_verify.py` (standalone re-implementation, independent of the production flag) · **Engine flag:** `antithetic=` in `mlmc_asian_level` / `estimate_rates` / `mlmc_run`; the κ=0 naive coupling is and stays the default · **Seeds:** 7 / 23 / 11 (+1 for the engine-level forward-variance check and 123 for the mechanical construction pre-gate)

> **Reconstruction note — read this before citing the document.** The original build-and-verify
> doc was never in the repo (`docs/gate_checks/README.md:105-106`: "the original was in Downloads,
> not the repo"). This file is rebuilt **from the two driver scripts**, so every threshold, seed and
> sample count below is copied out of code. Where the code fixes a number but gives no reason for
> it, the reason is recorded as missing in §11 rather than supplied. The drivers attribute three
> pre-registered figures ("~1.45", "efficiency ~0.87x", "the linearised guess said 2.0") to *this*
> document; one of the three is reproduced in the repo and one is derived there — see open item 1
> (§11) for what is and is not recoverable, and for the status of the superseded chat-era original
> that still exists outside the repo.

> **Terminology warning — "REFUTED" is about the hypothesis, not the gates.** The recorded thread
> verdict is REFUTED because the *claim under test* — that a Giles–Szpruch antithetic swap repairs
> the β<γ pathology of naive MLMC under rough vol — fails. The drivers' own three predictions
> (P1/P2/P3) predict that failure, so a healthy run prints `HOLDS` on all three. A driver line
> reading `REFUTED` means a **prediction** missed, which is the opposite signal. Do not read the
> two senses of the word as the same event. (The reconciliation of the two usages is this
> document's reading of `README.md:32-33` against the drivers' verdict lines; no file states it.)

---

## 0. Scope discipline — what this is, and is NOT

**IN.** One question, asked once: at a **matched finest level L**, does the antithetic coupling
`Y = ½(P_f + P_fa) − P_c` buy more variance than the extra fine path costs, for the arithmetic
Asian payoff under rough Bergomi? Three sub-questions follow from it — does the swap change the
MLMC rate β (G-A2), by what constant factor does it cut level variance (G-A3), and what is the
matched-accuracy cost (G-A4) — plus one correctness precondition (G-A1: the swap must not
introduce bias).

**OUT, deliberately:**
- *Rescuing MLMC.* If β is unchanged this thread does not, and is not intended to, fix
  β<γ. That is what the κ=1 and conditional-MC threads address separately.
- *Composition with other estimators.* Production raises `ValueError` if `antithetic` and
  `conditional` are both set, and refuses `kappa=1` with either (`layer1b_mlmc_asian.py:362-372`).
  The three estimators are independent axes; nothing here measures a combination.
- *Non-Asian payoffs.* Both drivers run `payoff="asian"` throughout.
- *A free-running cost comparison as evidence.* It is computed and printed by both drivers, and
  in both it is explicitly labelled diagnostic only (§4).
- *Judging G-A1 in the summary line.* As built, G-A1 cannot fail the run (§1.6) — an
  implementation weakness recorded here, not a design choice this document endorses.

---

## 1. Gate spec — G-A1 … G-A4

*This section is the antithetic gate spec.* Both driver headers refer the reader to a
`p2_coupling_gate_check.md` (`p2_antithetic_gatecheck.py:8-9`, `p2_antithetic_verify.py:10-11`).
**That filename is dead.** No such file exists anywhere, and `README.md:107-109` records it as
"a dead filename the build doc's prompt refers to; **not a real file**", with the gate spec
belonging in `p2_antithetic_build_and_verify.md` §1. It must not be created, and must not be
cited as if live. The gate definitions live here.

### 1.1 Mechanism under test

The Layer 1b engine simulates the κ=0 (optimal-discretisation Riemann) Volterra scheme, so the
coarse path is a deterministic function of the fine Brownian increments — coarse increment =
pairwise sum — and the MLMC coupling is exact. Giles–Szpruch antithetic MLMC replaces the fine
leg of the level correction with the average of the path and its **within-pair swapped** twin:

```
naive        Y_l = P_f − P_c
antithetic   Y_l = ½(P_f + P_fa) − P_c
```

`P_fa` is built by reversing the two fine increments inside each coarse step. Their **sum is
invariant**, so `P_c` — and hence the coupling and the level's mean — is untouched.

Operationally, what the drivers test is the **rate**: whether β moves from ≈2H toward ≈4H. That
is the whole of the mechanism statement in either file — `refute if ~4H`
(`p2_antithetic_gatecheck.py:149`) and `predicted beta_anti ~ beta_naive ~ 2H (NOT 4H)`
(`p2_antithetic_verify.py:384`). **Neither driver says why 4H would be the alternative, nor what
error term the swap is supposed to cancel.** A qualitative account exists in the thread's results
writeup but not in the drivers; see §10 and open item 10 (§11).

### 1.2 G-A1 — bias-free / exact coupling

Three checks, all committed as PASS/FAIL before measurement:

- **(a) Forward-variance identity** `E[V_t] = ξ₀` for every `t`. Measured directly from the
  variance path — `g, v = volterra_weights(n, H, T)`, `W̃ = √(2H)·fftconvolve(dW₁, g)`,
  `V = ξ₀·exp(ηW̃ − ½η²v)` — as `max_t |E[V_t]/ξ₀ − 1|` over **N = 60,000** paths on an
  **n = 256** grid at the default `PARAMS` (H = 0.10, η = 1.50, ρ = −0.70, ξ₀ = 0.04, S₀ = K = 100,
  T = 1, r = 0, n₀ = 32; `layer1b_mlmc_asian.py:140-150`), **rng seed 1**.
  **Bar:** below the Monte-Carlo band `3·exp(η²·v[-1]/2)/√N`. Prints `OK` / `CHECK`.
  Both drivers run this identically in full mode, and both mark it as an engine property rather
  than a coupling one — verify calls it "engine property, coupling-independent" (line 210),
  gatecheck "a property of the variance path, identical for naive and antithetic (the swap does
  not touch the Volterra law)" (lines 36-37). It is in G-A1 as a precondition, not as a
  discriminator.
- **(b) Telescoping consistency** for the antithetic estimator. With `a_l = E[P_f]` at level `l`
  and `m_l = E[Y_l]`, the driver forms
  `|a_l − a_{l−1} − m_l| / [3(√V^P_l + √V^P_{l−1} + √V_l)/√N]` and takes the max over `l ≥ 1`.
  Pilot: **L = 5, N = 16,000, seed 7**, naive and antithetic run paired off the same seed.
  **Bar: < 0.51** for the antithetic estimator. Production's `estimate_rates` docstring
  (`layer1b_mlmc_asian.py:499-509`) asks only for `a_l − a_{l−1} − m_l ≈ 0` within Monte-Carlo
  noise; the figure `< 1` appears not in the docstring but in the function's verbose print
  (`layer1b_mlmc_asian.py:540-541`). Both drivers harden the antithetic gate to 0.51.
- **(c) Per-level mean equality** — `p2_antithetic_verify.py` only. For every `l = 0…5`,
  `|E[Y_l]^naive − E[Y_l]^anti| < 2·√((V^naive_l + V^anti_l)/N)` (line 238). This is the check that
  the swap introduces no bias *at the level where it acts*, and the production-side driver does
  not have it.

### 1.3 G-A2 — rate (β against 2H)

Sweep **H ∈ {0.05, 0.10, 0.20, 0.35}** (everything else at `PARAMS`), pilot **L = 5,
N = 12,000 per estimator, seed 23**, naive and antithetic paired off the same seed. β is the OLS
slope of `−log₂Var(Y_l)` on `l = 1…5`. Compare `β_anti` against `2H` and against `β_naive`.

**Bar** — the two drivers commit slightly different bands and both are recorded:

| Driver | β condition |
|---|---|
| `p2_antithetic_gatecheck.py` | `abs(β_anti − β_naive) < 0.05` **and** `abs(β_anti − 2H) < 0.6·(2H) + 0.05`, at every H (lines 147-148) |
| `p2_antithetic_verify.py` | `abs(β_anti − β_naive) < 0.10` **and** `not(abs(β_anti − 4H) < 0.3·(2H))`, at every H (lines 388, 391) |

The verify script's form is the sharper statement of the falsifier: it tests β against the **4H**
alternative explicitly and tags each H as `~2H`, `~4H!` or `?`.

### 1.4 G-A3 — variance reduction factor

At each H from the G-A2 sweep, the per-level ratio `Var(Y_l)^naive / Var(Y_l)^anti` for
`l = 1…5` (the coupled levels; `l = 0` carries no coupling). Both drivers print all 20 ratios
(4 H × 5 levels) and reduce them to one pooled mean.

**Bar:** the pooled mean over all (H, l) must land in `[1.30, 1.60]` (gatecheck, line 156) or
`[1.25, 1.65]` (verify, line 404). There is **no per-level threshold** — a single level out of
band cannot fail G-A3; only the pooled mean can. G-A3 therefore has no independent PASS/FAIL of
its own beyond the P2 band in §6.

### 1.5 G-A4 — cost at matched finest level

The load-bearing gate. Cost is Giles' optimal continuous-N cost at a **fixed, shared** finest
level `L*`:

```
cost(ε, L*) = (2/ε²) · ( Σ_{l=0}^{L*} √(V_l · C_l) )²      C_l = n₀ · 2^l · coef(l)
coef(0) = 1.0     coef(l≥1) = 1.5 (naive) | 2.5 (antithetic)
standard-MC reference:  2·V₀/ε² · n₀ · 2^{L*}     (V₀ = the full-payoff variance at level 0)
```

`L*` comes from the Giles bias test applied to the **naive** pilot's level means only —
`max_{o<3} |m_{L−o}|·2^{−αo} / (2^α − 1) ≤ ε/√2`, smallest `L` from 2 upward that passes. Sharing
`L*` is justified in code by the identical-bias property: `_choose_L`'s docstring (verify:281-282)
states it is "Identical for naive and anti because their level means are equal in expectation."

**The two drivers do not use the same α, and the difference is not cosmetic** — `_choose_L`
divides by `2^α − 1` (verify:290, gatecheck:87), so a different α can select a different `L*`:

| Driver | α fed to `_choose_L` | `Lmax` passed |
|---|---|---|
| `p2_antithetic_gatecheck.py` | re-derived **with a 0.5 floor**: `max(0.5, −slope of log₂|m_l| on l = 1…6)` (line 124) | 6 (line 128; the function's own default is 9, line 81) |
| `p2_antithetic_verify.py` | the **unfloored** pilot α straight from `estimate_rates2` — `−slope of log₂|m_l| on l = 1…L_pilot` (line 106), taken at line 326 and passed at line 347 | default 9 (line 347 passes no `Lmax`) |

The 0.5 floor does appear in `p2_antithetic_verify.py`, but only inside `adaptive2`
(lines 137-150), i.e. on the pinned cross-check — not on the analytic `L*` that decides the
verdict. This compounds the like-for-like problem recorded as open item 7 (§11).

Accuracy targets **ε ∈ {0.10, 0.05, 0.025}**. Pilot **L = 6, N = 12,000, seed 11**.

**Bar:** mean over ε of the matched-L efficiency `1 / (cost_anti/cost_naive)` must be
**≤ 1.0** (gatecheck, line 162) / **≤ 1.001** (verify, line 421) — i.e. antithetic must not come
out cheaper once L is held equal. `p2_antithetic_verify.py` additionally reports a
**coupled-levels-only** efficiency, `(Σ_{l=1}^{L*}√(V^a C^a) / Σ_{l=1}^{L*}√(V^n C^n))²` inverted
(lines 352-355), and cross-checks the analytic number against its own adaptive driver **pinned**
to `Lmin = Lmax = L*` with α and β fixed from the pilot, reporting both pinned prices
(lines 357-360, 373-374).

### 1.6 Gate plumbing — a known weakness

`gate_a1()` in `p2_antithetic_gatecheck.py` returns a boolean (line 54) that `main()` discards
(line 185). `gate_a1()` in `p2_antithetic_verify.py` returns a dict containing `bias_ok`
(line 246), which is passed to `verdicts()` as the argument `a1` (line 451) and then never read.
**Consequence: no G-A1 failure can change the printed SUMMARY line in either driver** — G-A1 is
advisory as implemented, and a reader must inspect its `OK` / `CHECK` lines directly. The only
hard assertion in either driver is the mechanical pre-gate (§2), raised as
`assert gate_construction(), "construction exactness FAILED"` (verify:447).

---

## 2. The construction — and the mechanical pre-gate

Given fine increments `dW1, dW2` of shape `(nb, n_f)` and `n_c = n_f // 2`:

```
coarse   dW1_c = dW1.reshape(nb, n_c, 2).sum(axis=2)                    # pairwise sum
swapped  dW1_s = dW1.reshape(nb, n_c, 2)[:, :, ::-1].reshape(nb, n_f)   # reverse within pair
```

and identically for `dW2`. Both the orthogonal driver **and** the Volterra driver are swapped —
the antithetic twin is a full path, not a variance-path-only variant. The level then returns
`out[0] = Y`, `out[1] = P_f`, exactly as the naive branch does.

`p2_antithetic_verify.py` gates the construction itself *before* any gate runs
(`gate_construction`, lines 180-201, **rng seed 123, nb = 64, n_f = 8, n_c = 4**):

- **(i)** coarse increment invariant under the swap: `max|Σpair − Σpair_swapped| < 1e-12`;
- **(ii)** the swap is a **non-trivial involution** — `dW_s ≠ dW` and applying it twice returns
  `dW` — which rules out the degenerate case where the "antithetic" path is the original path and
  the whole experiment silently measures nothing.

Both are hard-asserted. This pre-gate touches no engine code: it draws raw
`rng.standard_normal((64, 8))`, reshapes, reverses, sums and compares — no `volterra_weights`, no
`_paths_from_increments`, no payoff. It is pure array mechanics.

`p2_antithetic_gatecheck.py` has no equivalent pre-gate, and **no reason for the omission is given
anywhere in the file** (its docstring, lines 1-15, does not mention it). Worth recording as a
consequence: production's own swap (`layer1b_mlmc_asian.py:405-406`) is asserted by neither
driver — verify asserts its own re-implementation of the swap (lines 69-70), and gatecheck
exercises production's without checking it.

**RNG pairing.** In both drivers the antithetic branch consumes *exactly* the same random numbers
as the naive branch — `dW1` and `dW2` are drawn identically, and `P_fa` is computed from
already-drawn increments, so no extra draws occur. Naive and antithetic at the same seed therefore
see the same Brownian paths, and every comparison in G-A1…G-A4 is paired. Batch size
(`max(200, min(batch, 2_560_000 // n_f))`) depends only on `n_f`, so the pairing survives batching.

---

## 3. The cost asymmetry (2.5 vs 1.5) — the crux of G-A4

The antithetic level evaluates **three** paths where the naive level evaluates two:

```
antithetic   P_f (n_f) + P_fa (n_f) + P_c (n_f/2)  =  2.5 · n_f
naive        P_f (n_f)             + P_c (n_f/2)  =  1.5 · n_f
```

so `coef(l≥1) = 2.5` against `1.5`, and `coef(0) = 1.0` for both (level 0 has no coupling).
`p2_antithetic_verify.py` states the trap in its header: "If you forget the extra fine path,
G-A4 is silently rigged in antithetic's favour." Production charges the same 2.5 via
`_level_cost_coef(l, antithetic)` (`layer1b_mlmc_asian.py:317-326`), and both drivers use the
matching coefficient — the verify script through its own `_cost_coef` (lines 81-85), the gatecheck
script through the `c` array that production's `estimate_rates` already builds
(`layer1b_mlmc_asian.py:521`).

This fixes the bar antithetic has to clear: the variance factor must exceed **2.5/1.5 ≈ 1.67**
for the swap to pay at matched L. A factor of ~1.45 loses, and the size of the loss is G-A4's
answer. The comparison is recorded in the engine itself —
`layer1b_mlmc_asian.py:348-349`, "variance factor ~1.45x < cost factor 2.5/1.5" — and the
arithmetic `2.5/1.5 = 1.67×` is written out in `docs/p2_estimator_results.md:44`.

---

## 4. The L-selection artifact — why the free-running driver cannot decide G-A4

Both drivers also run the adaptive driver **free** (each estimator picks its own `L`), and both
discard the result for the verdict. The reason is stated in code
and is not subtle: the antithetic swap leaves the level means unchanged, so the bias — and
therefore the honest finest level — is **identical** for the two estimators. Any `L` difference the
stopping rule produces is stopping noise, and because cost is roughly geometric in `L`, an
estimator that happens to stop one level early looks dramatically cheaper while being *more
biased*.

`p2_antithetic_gatecheck.py` prints (lines 116-117):

> `^ NOTE: naive and anti pick DIFFERENT L (identical bias => this is a stopping-rule artifact, not a real cost difference). Honest below.`

`p2_antithetic_verify.py` heads the same block `[free-running L, CONFOUNDED — for diagnosis only]`
(line 335) and closes its cost verdict with (lines 426-428):

> `NOTE: the free-running driver can make anti look cheaper, but only by stopping at a coarser (more biased) L — an artifact, not a win.`

**The sign-flip claim is not measured by either driver.** `README.md:38-39` records the artifact as
flipping sign across seeds, and `docs/p2_estimator_results.md:52-53` gives the detail (cheaper on
seed 11, dearer on seeds 99 and 1234). Both antithetic drivers hardcode seed 11 for G-A4
(gatecheck:108-109, verify:337-338), so neither runs the replication that claim rests on.

**The identical-bias rule — always compare at matched finest level — is the methodological output
of this thread**, and it is carried forward: `kappa1_hybrid_coupling_design.md` §8 ("Gate-check
plan (G-H1 … G-H4)", line 157) restates it in its own words at lines 159-160 — "always compare at
matched finest level L (identical-bias rule), never via the free-running driver". It quotes no text
from either driver, and the causal direction (this thread as the source) is this document's reading
of the cross-reference, not a stated attribution.

**A gate-id numbering discrepancy, recorded rather than reconciled.** `docs/gate_checks/README.md`
and the κ=1 drivers use: **G-H1a–d** for the fine path (`gh1_kappa1_finepath.py:6-11`),
**G-H2a–c** for the coarse coupler (`gh2_kappa1_coupler.py:12-15`; G-H2a telescoping, G-H2b
coupling tightness, G-H2c rate β≈2H), and **G-H4** for adoption (`gh4_kappa1_cost.py:65`).
`kappa1_hybrid_coupling_design.md` §8 uses an **older, incompatible** scheme in which G-H1 bundles
coarse-law exactness with coupling tightness and cross-covariance, G-H2 is the rate, G-H3 the
variance factor and G-H4 the Giles cost (lines 162-178). This document uses the driver/README
scheme throughout. The design note's §8 is cited above only for the matched-L rule it restates;
its gate ids are not the live ones.

---

## 5. The committed prediction (hard-coded in the driver)

Three predictions, hard-coded as acceptance bands in both drivers' `verdicts()`:

- **P1 — no rate change.** `β_anti ≈ β_naive ≈ 2H` across H ∈ {0.05, 0.10, 0.20, 0.35}.
  *Falsifier:* β moves toward **4H**. This is the prediction that matters: 4H would mean the swap
  restores β > γ and rescues MLMC under rough vol.
- **P2 — variance factor ≈ 1.45, not 2.0.** The per-level variance reduction is a modest constant
  well short of the leading-order guess. The drivers label 2.0 as "the linearised guess" /
  "the 2.0 leading-order guess" and 1.45 as the committed figure.
  *Falsifier:* the pooled mean leaves `[1.30, 1.60]` / `[1.25, 1.65]` — reported as `DIFFERS`,
  not `REFUTED`, in both drivers.
- **P3 — net worse on cost.** At matched L, antithetic efficiency **≈ 0.87×** — below 1, i.e. the
  swap does not pay for itself, and the matched-L cost ratio must **not** improve.
  *Falsifier:* mean matched-L efficiency > 1.

One of the three figures is independently reproduced in the repo, one is derived there, and one is absent:
`docs/p2_estimator_results.md:40-46` records the ~1.45× variance factor with its per-H breakdown
(1.49, 1.47, 1.43, 1.37, *decreasing* with roughness), sets it against the cost-per-sample
2.5/1.5 = 1.67×, and derives the efficiency arithmetically as ~1.45/1.67 ≈ 0.87×; the same
1.45-versus-2.5/1.5 comparison sits in the engine at `layer1b_mlmc_asian.py:348-349`. The **2.0**
figure is the exception: it is labelled "a leading-order pre-check" / "the linearised guess" in
every artifact that mentions it, and the pre-check itself appears nowhere (open item 1, §11).

---

## 6. Gate-check bar (PASS/FAIL committed before any measurement)

| Gate | Quantity | Bar | Where |
|---|---|---|---|
| G-A1(a) | `max_t abs(E[V_t]/ξ₀ − 1)` | `< 3·exp(η²v[-1]/2)/√N`, N = 60,000, n = 256, seed 1 | both |
| G-A1(b) | antithetic telescoping consistency | `< 0.51`, L = 5, N = 16,000, seed 7 | both |
| G-A1(c) | `abs(E[Y_l]^naive − E[Y_l]^anti)` | `< 2√((V^n_l+V^a_l)/N)` for all l = 0…5 | verify only |
| PRE | coarse-sum invariance under swap | `< 1e-12`; swap non-identity **and** involution; hard `assert` | verify only |
| G-A2 | β_anti vs 2H and vs β_naive | `abs(Δβ) < 0.05` ∧ `abs(β_a − 2H) < 0.6(2H)+0.05` (gatecheck); `abs(Δβ) < 0.10` ∧ not-near-4H at `0.3(2H)` (verify) | both |
| G-A3 | pooled `Var^naive/Var^anti`, l = 1…5, 4 H values | mean ∈ `[1.30, 1.60]` (gatecheck) / `[1.25, 1.65]` (verify) | both |
| G-A4 | mean matched-L efficiency `1/(cost_a/cost_n)`, ε ∈ {0.10, 0.05, 0.025} | `≤ 1.0` (gatecheck) / `≤ 1.001` (verify) | both |

**Verdict strings — these are the gate definitions.** Quoted exactly; `{…}` marks an interpolated
measurement.

`p2_antithetic_gatecheck.py`:

```
[P1] beta_anti ~ 2H, no rate change (refute if ~4H):
     => HOLDS | REFUTED
[P2] variance ratio ~1.45 (linearised guess said 2.0): measured mean {m} (range {lo}-{hi})
     => HOLDS | DIFFERS
[P3] antithetic net WORSE on cost (efficiency ~0.87x; ratio should NOT improve):
     => matched-L mean efficiency {e}x <= 1: HOLDS | REFUTED
        (free-running driver's apparent win is the L-selection artifact)
SUMMARY: P1 {…} | P2 {…} | P3 {…}
```

`p2_antithetic_verify.py`:

```
[P1] RATE: predicted beta_anti ~ beta_naive ~ 2H (NOT 4H)
     => HOLDS: antithetic gives NO rate change; beta tracks 2H, not 4H.
[P2] VARIANCE FACTOR: predicted ~1.45 (NOT the 2.0 leading-order guess)
     => HOLDS: variance reduction is a modest constant (~{m}x), well short of 2x.
[P3] COST: predicted antithetic is net slightly WORSE (efficiency ~0.87x; adaptive cost should NOT improve)
     (matched finest level L; bias identical so L is shared)
     coupled-levels-only efficiency ~{e_c}x  (this is the doc's ~0.87x number)
     => HOLDS: all-levels efficiency {e}x <= 1  -> antithetic does NOT pay for itself.
SUMMARY:  P1(rate) {…}   |   P2(var~1.45) {…}   |   P3(net-worse) {…}
```

Note the two cost measures the verify script keeps apart: **all-levels** efficiency is the verdict
quantity (`p3_ok = me_all <= 1.001`, line 421); **coupled-levels-only** (l ≥ 1) is labelled in code
as "the doc's ~0.87x number" (lines 351, 422-423). The level-0 term enters both sums identically —
at `l = 0` the antithetic branch is the naive branch (`Y_0 = P_f`), the two runs share a seed, and
`coef(0) = 1.0` either way — so it is a common addend pulling the all-levels ratio toward 1. That
follows from the two cost formulas (lines 305-308 vs 352-355); it is not stated in code.

---

## 7. Build and verify — the two drivers

Each driver states its own purpose, and they differ:

**`p2_antithetic_gatecheck.py` — through production.** Its docstring (lines 1-5) says it runs the
four gates "THROUGH the production functions in layer1b_mlmc_asian.py", producing the
naive-vs-antithetic comparison table and "end-to-end-check[ing] the integrated flag". It drives the
gates purely through the `antithetic=` flag on production's own `estimate_rates` and `mlmc_run`
(imports, lines 26-27). Its G-A4 **`L*`-selection** is capped at `Lmax = 6` (line 128;
`_choose_L`'s own default is 9, line 81) — the cap is on level selection, not on the pilot, which
is `L = 6` by its own argument (lines 122-123). Since the cap equals the pilot depth, `L*` never
runs past measured levels and no variance extrapolation is needed. α is fitted from the naive
pilot only.

**`p2_antithetic_verify.py` — standalone.** Its stated reason is temporal non-invasiveness: it
imports the production engine and "re-implements ONLY the pieces that need the antithetic flag,
so the production file is left untouched until the numbers are approved" (lines 5-8), under the
title "independent verification" (line 2). It still takes from
the engine — `PARAMS`, `_paths_from_increments` and `volterra_weights`; everything else it
re-implements: the level estimator (`mlmc_level`), the cost coefficient (`_cost_coef`), the rate
pilot (`estimate_rates2`) and the adaptive driver (`adaptive2`). It adds the mechanical pre-gate,
the per-level bias check G-A1(c), variance/cost **extrapolation past the pilot depth**
(`_ext_VC`: `V_l = V_{L_p}·2^{−β(l−L_p)}`, `L_max = 9`, β taken per-estimator, lines 295-302 and
328-329), the coupled-levels-only cost measure, and a pinned-driver cross-check. `adaptive2`
mirrors `mlmc_run`: `N₀ = 2,000`, `Lmin = 2`, `Lmax = 9`, sample rule
`N_l = ⌈(2/ε²)√(V_l/C_l)·Σ√(V_k C_k)⌉`, bias test `tail.max()/(2^α − 1) > ε/√2` to grow L, floors
`α ≥ 0.5` and `β ≥ 0.1`. It has a `--quick` smoke mode with reduced settings (§8).

The pair is redundant by construction, and that redundancy does buy independence from the
production integration — but **that is this document's inference, not a purpose either file
states** (open item 11, §11).

---

## 8. Seeds, sample counts and reproduction

| Gate | Seed | Full-mode settings | `--quick` (verify only) |
|---|---|---|---|
| PRE (construction) | 123 | nb = 64, n_f = 8, n_c = 4 | same |
| G-A1(a) forward variance | 1 | N = 60,000, n = 256, H = 0.10 | N = 20,000 |
| G-A1(b,c) coupling | 7 | L = 5, N = 16,000 | L = 4, N = 8,000 |
| G-A2 + G-A3 sweep | 23 | L = 5, N = 12,000, H ∈ {0.05, 0.10, 0.20, 0.35} | N = 6,000, H ∈ {0.10, 0.30} |
| G-A4 cost | 11 | pilot L = 6, N = 12,000; ε ∈ {0.10, 0.05, 0.025} | pilot L = 5, N = 8,000; ε ∈ {0.10, 0.05} |

Seeds 7 / 23 / 11 are the baseline allocation recorded in `README.md:40-41`; seed **1** (the
engine-level forward-variance check) and seed **123** (the mechanical construction pre-gate) are
extra internal fixtures and appear in neither the index nor any doc.

```
python p2_antithetic_gatecheck.py          # through production
python p2_antithetic_verify.py             # standalone, full
python p2_antithetic_verify.py --quick     # standalone, smoke pass
```

Run from the repository root (the drivers are root-level by convention). Both reconfigure stdout
to UTF-8 and print a wall-clock total. Result writeup: `../p2_estimator_results.md`.

---

## 9. Recorded outcome — REFUTED

From the `README.md` index (`README.md:22`, `:36-39` — the recorded run of this session's drivers).
Every number in this section is index-sourced, not read off a driver run recorded here:

- **β unchanged and tracking 2H, not 4H** — identical to naive, `0.120 / 0.219 / 0.418 / 0.726`
  across H ∈ {0.05, 0.10, 0.20, 0.35}. **P1 holds; the repair hypothesis is refuted.**
- **Variance factor ≈ 1.44×** — a modest constant, below the 2.5/1.5 ≈ 1.67 cost factor.
  **P2 holds.** *Artifact discrepancy:* `README.md:22,37` records ≈1.44× for this quantity, while
  `layer1b_mlmc_asian.py:349` and `docs/p2_estimator_results.md:40` both record ~1.45×. The value
  quoted here is the README's. The disagreement is pre-existing and is recorded, not reconciled
  (open item 12, §11).
- **~9–11 % costlier at matched L**, efficiency **≈ 0.91×** (all-levels). **P3 holds.** The
  pre-registered 0.87× corresponds to the coupled-levels-only measure, which the verify driver
  labels as such (lines 422-423); the headline all-levels figure is 0.91× (`README.md:22`). The
  free-running driver's apparent win is the L-selection artifact of §4.

So all three committed predictions hold and the estimator is rejected. Antithetic coupling
remains an opt-in flag in production, off by default, with the negative result recorded in
`mlmc_asian_level`'s docstring — "Verified net slightly WORSE here (variance factor ~1.45x <
cost factor 2.5/1.5); see p2_antithetic_verify.py" (`layer1b_mlmc_asian.py:348-349`).
`_level_cost_coef`'s docstring (`layer1b_mlmc_asian.py:318-323`) records only the 1.5 / 2.5 flop
counting under the γ = 1 model (plus the conditional estimator's measured ~1.3× note); it does
**not** contain the "WORSE" verdict. `grep -n "WORSE\|worse" layer1b_mlmc_asian.py` returns line
348 and nothing else.

---

## 10. Honest scope note — what this does NOT establish

- **Not a general refutation of Giles–Szpruch.** The construction is exact and the pre-gate
  confirms it; what fails is the *economics* for this payoff, this scheme and this cost model.
  The drivers measure that the swap does not steepen β; **neither driver analyses the surviving
  error term**, and neither states what error the swap was expected to cancel. A qualitative
  account is offered in the thread's results writeup — `docs/p2_estimator_results.md:38-43`: the
  swap "cannot touch the leading common-factor Volterra error", and the reduction falls short of
  2× "because the orthogonal-noise and nonlinear parts of Var(Yₗ) are not halved by the swap" —
  but it is asserted there, not derived, and it is outside the drivers this spec covers.
- **One payoff, one parameter set.** Arithmetic Asian at `PARAMS` (with H swept for G-A2/G-A3
  only). Nothing is measured at the Bayer–Friz–Gatheral SPX calibration that
  `layer1b_mlmc_asian.py:137` notes as the realistic case.
- **γ = 1 cost model.** Cost is counted as flops in units of `n_f`, per the `_level_cost_coef`
  γ = 1 model, not wall-clock. The 2.5 and 1.5 coefficients are *derived by counting path
  evaluations*, not measured — unlike the conditional-MC and κ=1 threads, which time their
  per-path overhead empirically (`layer1b_mlmc_asian.py:320-322`, "a small measured constant
  (~1.3x)"; `README.md:83-84`, "overhead measured ~1.08×, not assumed"). If the swapped fine path
  were materially cheaper or dearer than the original in practice (cache behaviour, FFT reuse),
  G-A4's margin would move.
- **The ~9–11 % margin is not a wide one.** It is the mean of three ε values from one seed per
  gate, against a band (`≤ 1.0`) that has no slack. No multi-seed replication of the cost number
  exists in either driver. The qualitative conclusion — no rate change, factor below 1.67 — is
  robust; the precise percentage is not a tight number.
- **G-A1 is advisory as built** (§1.6). The run can print `CHECK` on the bias precondition and
  still summarise three holding predictions.

---

## 11. Open items and reconstruction gaps

*Questions this reconstruction could not answer from the code are recorded in [`RVL-009_open_items.md`](RVL-009_open_items.md).*
