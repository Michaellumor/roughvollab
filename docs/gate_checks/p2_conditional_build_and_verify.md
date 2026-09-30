# Build-and-verify spec — conditional MC (geometric control variate), G-C4

> **Driver committed:** 2026-06-22 (`d778074`) · **Spec reconstructed:** 2026-09-29
> from the driver scripts under RVL-009. The original specification was
> chat-only and is not recoverable. Predictions and gates below are as
> recorded in the code; this document does not attest when they were
> written.

**Status:** built & measured · **Driver:** `p2_conditional_verify.py` (standalone, `--quick` flag)
**Engine touched:** none — imports only `PARAMS`, `volterra_weights`, `_paths_from_increments` from `layer1b_mlmc_asian.py`
**Gate bar:** chat-only; the intended filename `p2_conditional_gate_check.md` is listed in `docs/gate_checks/README.md` under *"Specs referenced but not in the repo"* and **no such file exists** — this document does **not** restate the G-C4 thresholds · **Write-up:** `../p2_estimator_results.md` §2
**Verdict of record:** **CONFIRMED** — conditional *standard* MC cheapest; conditional MLMC does **not** beat it, ratio **0.41–0.45** (`docs/gate_checks/README.md`:23,50; the write-up `docs/p2_estimator_results.md`:90 records the same quantity as **≈0.42–0.45** — the two records differ at the lower end and neither figure is derivable from code, `key_ratio = c_cs/c_cm` at `:253` is printed, not stored) · **Seeds:** 11 / 99 / 1234

*Reconstructed from the driver. Every constant below is copied from the code.
Where the code fixes a number without saying why, this document says so rather
than supplying a motive.*

---

## 0. Scope discipline — what this is, and is NOT

**IN:** how the estimator is *built* (closed form, array algebra, code path), how
the verification run is *organised* (pilot levels, seeds, paired sampling, cost
model, the measured overhead κ\*), and how it is *reproduced*.

**OUT:** the G-C4 bar itself. The pass criteria were given in chat and were
never committed: `docs/gate_checks/README.md`:110 records
`p2_conditional_gate_check.md` as a spec *to add*, and the file is absent from
the repo. It must not be cited as if it were live. §8 below therefore records
the verdict *strings the code prints* — in the absence of a committed spec those
strings are the only operational gate definition, and a reconstruction that
paraphrases them is lying by omission.

**ALSO OUT:** this driver is **κ=0 only** (κ here is the Volterra scheme axis —
see the naming note in §6). It never passes a `kappa` argument and never draws a
near-cell Gaussian. The κ=1 variant of the same estimator is a separate thread
(G-H4 adoption, drivers `gh4_kappa1_conditional.py` / `gh4_kappa1_cost.py`; its
spec `gh4_kappa1_adoption_spec.md` is likewise chat-only and not in the repo).
The production *level* function refuses the combination — `if kappa == 1 and
(antithetic or conditional): raise ValueError("kappa=1 is a separate scheme
axis; …")` in `mlmc_asian_level` (`layer1b_mlmc_asian.py`:370-372). The refusal
is at that level only: the payoff function `_cond_asian_payoff` does take
`kappa`/`Z` and documents κ=1 support (`:265`, docstring `:280-285`), and
`gh4_kappa1_conditional.py`:40,64,72 calls it with `kappa=1` directly on a
single grid, bypassing `mlmc_asian_level`.

**Gate-id scheme.** The ids used here are the driver/README scheme
(`docs/gate_checks/README.md`): G-C4 for this thread, G-A1..G-A4 antithetic,
G-H1a..d the κ=1 fine path (`gh1_kappa1_finepath.py`), G-H2a..c the κ=1 coarse
coupler (`gh2_kappa1_coupler.py`: G-H2a telescoping, G-H2b coupling tightness,
G-H2c rate β≈2H), G-H4 adoption. **Discrepancy on record:**
`docs/gate_checks/kappa1_hybrid_coupling_design.md` §8 (`:157`, "Gate-check plan
(G-H1 … G-H4)") uses an older, incompatible numbering — G-H1 = coarse-law
exactness + coupling tightness + cross-covariance, G-H2 = rate, G-H3 = variance
factor, G-H4 = Giles cost. Where this document cites that design note (§6) it
cites the *section*, not its gate ids; the two schemes are not reconciled here.

---

## 1. The estimator as built

A **conditional geometric-Asian control variate**, in the driver's own notation:

```
P_cond = arith − ( geom − E[geom | W] ) = arith − geom + E[geom | W]
```

`W = dW1` is the variance-driving Brownian path. `geom` is the geometric-average
Asian payoff on the **same grid and the same trapezoidal weights** as `arith`.
Conditional on `W` the discrete log-prices are Gaussian, so the geometric
average is lognormal and `E[geom | W]` is closed form — which integrates the
orthogonal driver `W⊥` out of the control *exactly* rather than by regression.

Two consequences the build relies on, both asserted in the driver docstring
(`:8-13`, `:22-23`) rather than tested:

- The control has **zero conditional mean given `W`**, hence zero unconditional
  mean, hence `P_cond` is unbiased for the arithmetic-Asian price **at every
  grid level**; the docstring states the discretisation bias is *identical* to
  naive arithmetic MLMC.
- Therefore all four estimators compared in §6 share one bias curve, and a cost
  comparison at a **shared finest level L\*** is legitimate.

What the driver actually checks is weaker: the §5 z-test on the control's own
sample mean. It never compares the two bias curves level by level. Recorded as
a scope hole (§10), not as a defect.

No control coefficient is fitted: the coefficient is pinned at 1 by
construction. Nothing in the driver estimates or optimises one.

---

## 2. Code path — `_cond_payoffs`, and what it deliberately does not reuse

`_cond_payoffs(dW1, dW2, n, p)` (`p2_conditional_verify.py:52`) returns the
triple `(arith, geom, Egeom)` on one grid of `n` steps. It **re-implements** the
path construction inline instead of calling the engine's `_simulate_paths`:

```
g, v      = volterra_weights(n, H, T)                        # engine, imported
W_tilde   = sqrt(2H) · fftconvolve(dW1, g)[:, :n]
V_left[:,0]  = xi0
V_left[:,1:] = xi0 · exp(eta·W_tilde[:,:-1] − ½·eta²·v[:-1])  # left-point V
dW_S      = rho·dW1 + sqrt(1−rho²)·dW2
dlogS     = (r − ½V_left)·dt + sqrt(V_left)·dW_S
logS      = [0, cumsum(dlogS)] ;  S = S0·exp(logS)
```

The compensator uses the engine's **exact discrete** `v` (not `t^{2H}`); the
engine annotates the same line "exact forward variance E[V]=xi0"
(`layer1b_mlmc_asian.py`:236). Both averages use the trapezoidal rule on the
level's own grid:

```
A  = (½S_0 + Σ_{1..n−1} S_k + ½S_n) / n          → arith = e^{−rT}(A − K)⁺
LG = (½logS_0 + Σ_{1..n−1} logS_k + ½logS_n) / n → geom  = e^{−rT}(S0·e^{LG} − K)⁺
```

At `PARAMS` defaults `r = 0.0`, so `disc = exp(−r·T) = 1.0` — the discount
factor is a no-op in the recorded run and is **not** exercised by this gate.

**Why the duplication matters.** The engine already ships the same estimator as
`_cond_asian_payoff` behind the `conditional=True` flag of `mlmc_asian_level`.
The driver does not call it. G-C4 therefore verifies the *standalone*
re-implementation; that the production flag computes the same thing is
plausible from side-by-side reading of the two functions but is **not** asserted
anywhere in this driver. Recorded as a scope hole, not as a defect. The
docstring calls the arrangement "NON-INVASIVE" (`:15`) and gives no further
reason for the duplication.

---

## 3. The load-bearing algebra — `E[geom | W]`

Given `W`, `LG` is Gaussian with

```
mu_j  = (r − ½V_j)·dt + sqrt(V_j)·rho·dW1_j            # E[dlogS_j | W]
muG   = trapezoidal average of cumsum(mu)
Wbar_j = 1 − (1 + 2j)/(2n)                             # j = 0 … n−1
sigG²  = (1 − rho²)·dt · Σ_j V_j · Wbar_j²
```

`Wbar_j` is the **cumulative trapezoidal weight on the steps strictly after step
`j`**, and it is exact, not an approximation: with `w_0 = w_n = 1/2n` and
`w_k = 1/n` inside, `Σ_{k>j} w_k = (2n − 1 − 2j)/(2n)`, which is the coded
expression. `sigG²` is then the variance of the `W⊥` contribution to `LG`, and
the conditional expectation is a closed-form call on the forward
`F = S0·exp(muG + ½sigG²)`:

```
d1 = (log(F/K) + ½sigG²)/sigG,  d2 = d1 − sigG
Egeom = e^{−rT} · ( F·Φ(d1) − K·Φ(d2) )
```

with `Φ = scipy.stats.norm.cdf`.

`sigG` is floored: `sigG = sqrt(max(sigG², 1e-300))` (`:92`). What the floor
does is checkable: at `sigG² = 0` it gives `sigG = 1e-150`, so
`d1 = log(F/K)/1e-150` saturates `norm.cdf` at 1, 0 or (at `F = K`) 0.5, and
`Egeom` collapses to `disc·max(F − K, 0)` — the intrinsic value.
Confirmed numerically: `(F,K) = (110,100) → 10.0`, `(90,100) → 0.0`,
`(100,100) → 0.0`, each equal to the intrinsic. The degenerate limit is
therefore handled correctly, not papered over.

It is also unreachable at the defaults: the `j = 0` term alone bounds
`sigG² ≥ (1 − rho²)·dt·xi0·(1 − 1/2n)² > 0` with `rho = −0.70`, `xi0 = 0.04`.
Never exercised by a test.

---

## 4. The unit test that guards the closed form

`unit_test_Egeom` (`:101`) is a hard gate: `main` wraps it in
`assert unit_test_Egeom(PARAMS, args.quick), "Egeom closed form FAILED"`, so the
run aborts before G-C4 if it fails.

Method: seed **7**, grid **n = 48**, **4** fixed `dW1` paths. Each path is
replicated `M` times (`M = 200_000` full, `60_000` quick), fresh `dW2` per path,
and the brute-force conditional mean `geom.mean()` is compared with the
closed-form `Egeom[0]`:

```
z = |closed-form − brute-force| / (geom.std()/sqrt(M)) ,   pass iff z < 3.5
```

printed per path as `OK` / `FAIL`. Three things are worth flagging: `n = 48` is
**not** a production grid (`n0·2^l ∈ {32, 64, 128, …}`), the threshold is `3.5`
where the §5 unbiasedness check uses `4`, and the four `z` values are combined
by `ok &=` — i.e. **any** path failing aborts the whole run.

---

## 5. Run organisation

`estimate_cond_rates(L, N, p, seed, batch=5000)` (`:126`) is the pilot. One
`np.random.default_rng(seed)` per call, drawn sequentially over levels
`l = 0 … L` — so levels are **not** independent streams and there are no common
random numbers across levels, but they are bit-reproducible given the seed.

Per level: `n_f = n0·2^l`, batch capped by `b = max(200, min(batch,
2_560_000 // n_f))` — the same peak-memory cap the engine uses. Fine increments
`dW1, dW2 ~ N(0, dt_f)`; coarse increments by **pairwise summation**
(`dW1.reshape(nb, n_c, 2).sum(axis=2)`), the exact κ=0 MLMC coupling. At `l = 0`
there is no coarse level and `Y = P_f`.

**Everything is paired.** `arith` and `P_cond` are computed from the *same*
increments on the same call, so every variance ratio and every mean difference
below is a paired comparison, not two independent runs. Recorded per level:

| Field | Meaning |
|---|---|
| `Vn`, `Vc` | Var of the level difference `Y_l`, naive / conditional |
| `VPn`, `VPc` | Var of the single-level fine payoff `P_L`, naive / conditional |
| `mn`, `mc` | `E[Y_l]` (feeds the bias/α fit) |
| `an`, `ac` | `E[P_f]`, `E[P_cond]` (feeds the unbiasedness check) |
| `sd_ctrl` | `std(P_f − P_cond)` = std of the control `geom − E[geom|W]` |

Sample counts: `N = 14_000` full, `8_000` with `--quick`; pilot depth
`Lpil = 6` (finest grid `32·64 = 2048`); seeds `[11, 99, 1234]`. `N` is the same
at every level — this is a pilot for rates and variances, **not** a Giles
`N_l`-allocated MLMC run.

**Unbiasedness check** (`:223-230`): because `an − ac = mean(geom − Egeom)`
identically, the test is a one-sample z on the control's own mean,
`z = max_l |an − ac| / (sd_ctrl/√N)`, flagged `OK` if `z < 4` else `CHECK`.
Computed from the **seed-11 pilot only**. Crucially it is *printed, not
enforced* — §8. `docs/gate_checks/README.md`:46 records `z ≈ 1.5` for this
check; note that `docs/p2_estimator_results.md`:69 attaches a `z < 1.5` to the
§4 closed-form unit test instead.

**Rate fits** (`:217-219`), from the seed-11 naive means, levels 1…6:
`alpha = max(0.5, −slope of log2|mn|)`, `beta_n`, `beta_c` = −slope of
`log2(Vn)`, `log2(Vc)`. Only `alpha` feeds a decision (the L\* bias test); the
two βs are printed and never read again.

---

## 6. The cost model and the measured κ\*

**Naming.** `κ` is overloaded in this codebase: the Volterra scheme axis
(κ ∈ {0,1}, §0) and the conditional per-path cost overhead. The driver reuses
the name for both (`measure_kappa` vs the `kappa` argument of `_volterra`).
The repo has **no** such convention: `docs/p2_estimator_results.md`:91-92 reads
"*independent of the conditional cost overhead κ*", where the trailing `*` closes the
markdown emphasis opened at "*independent" and renders as an italic κ, not κ\*.
(`docs/gate_checks/README.md`:50 writes the same property as "κ-invariant".) This
document therefore introduces **κ\*** itself, to resolve the collision, and
this document follows that convention: **κ\* = the measured cost overhead**,
bare κ = the scheme axis.

Per-sample work in units of `n_f`, γ=1 model, hard-coded in the driver at
`:232-233` rather than read from the engine's `_level_cost_coef`:

```
Cn = n0·2^l · (1.0 if l == 0 else 1.5)
Cc = kappa · Cn
```

The driver carries **no comment** on either line and states no reading of the
coefficients. The `1.5 = P_f + P_c` gloss comes from the engine:
`_level_cost_coef`'s docstring, `layer1b_mlmc_asian.py`:318-319 — *"Naive: P_f
(n_f) + P_c (n_f/2) = 1.5"*. `Cc` applies κ\* at **all** levels including
`l = 0`.

Four matched-accuracy costs per (ε, seed), with the `2/ε²` prefactor hard-coded
at `:200` (`_giles_cost`) and `:248-249`:

```
naive-MLMC   (2/ε²)(Σ_{l≤L*} √(Vn_l·Cn_l))²
cond-MLMC    (2/ε²)(Σ_{l≤L*} √(Vc_l·Cc_l))²
naive-stdMC  (2/ε²)·VPn[L*]·(n0·2^{L*})
cond-stdMC   (2/ε²)·VPc[L*]·(kappa·n0·2^{L*})
```

Neither the code nor the docstring derives the factor 2, or the paired
`eps/np.sqrt(2.0)` tolerance in `_choose_L` (`:193`). The docstring (`:24-25`)
states only the formulae.

The single-level costs correctly carry coefficient 1.0 (no coarse path), not
1.5.

**κ\* is measured, not assumed.** `measure_kappa` (`:164`) times `_cond_payoffs`
against the production `_paths_from_increments(..., "asian")` on identical
increments: seed **0**, level **l = 3** (`n = 256`), `N = 8000`, one warm-up
call each, then **8** timed repetitions; `kappa = tc / tn`. The engine's
`_level_cost_coef` docstring records the measured constant as **≈1.3×**
(`layer1b_mlmc_asian.py`:321) and explicitly declines to charge it in the
engine's own cost model, deferring to this driver. (`docs/p2_estimator_results.md`:75
records the same constant as ~1.28×; the ≈1.3× figure quoted here is the engine
docstring's.) Called **once**, outside the seed loop — one κ\* for all seeds and
both ε.

**What κ\* does and does not isolate.** The driver's inline path build is
operation-for-operation the engine's κ=0 `_simulate_paths` body
(`layer1b_mlmc_asian.py`:216-249, via the κ=0 branch of `_volterra` at
`:201-204`): same `volterra_weights` call, same `fftconvolve` slice, same
`V_left` recursion, same `dW_S`/`dlogS`/`cumsum`/`exp`; and
`_paths_from_increments` adds exactly the trapezoidal arithmetic payoff the
driver computes at `:76-77`. So κ\* ≈ (build + arith + conditional
post-process)/(build + arith): it isolates the `geom`/`muG`/`sigG²`/closed-form
block plus call overhead, which is what it claims to measure. The standing risk
is not contamination but **drift** — the duplicate can diverge from the engine
silently, and nothing tests them against each other (§10).

The headline ratio `cond-stdMC / cond-MLMC` is **κ\*-invariant** — both pay κ\*
per path — which is why it is the quantity the verdict is written on (docstring
`:27-28`); the ratios against the *naive* estimators are not κ\*-invariant and
are reported as secondary.

**L\* discipline.** `_choose_L(m, alpha, eps, Lmax=6)` (`:185`) applies the
Giles remaining-bias test to the **naive** level means only:

```
tail = [ |m[Lc−o]|·2^{−alpha·o} for o in 0..min(3,Lc)−1 ]
accept Lc iff  tail.max() / (2^alpha − 1) ≤ eps / √2
```

scanning `Lc = 2 … 6` — so **L\* ≥ 2 by construction**, and since
`len(m) − 1 = 6 = Lmax` the function's extrapolation branch (`l > Lp`) is
unreachable dead code in this driver. The selected level is then imposed on all
four estimators, on the identical-bias argument of §1. `ε ∈ {0.10, 0.05}`; the
docstring commits to `L* ≈ 2` and `≈ 5` respectively, and the run of record
returns exactly those (`docs/p2_estimator_results.md`:86-87).

This is the deliberate correction to the G-A4 antithetic artefact
(`p2_antithetic_build_and_verify.md` §1 — chat-only, not in the repo): the
docstring states it as *"Do NOT let the adaptive driver pick L (it manufactures
phantom cost ratios by choosing different finest levels for estimators with
identical bias)"* (`:19-21`). Same rule as
`docs/gate_checks/kappa1_hybrid_coupling_design.md` §8 (`:159-160`) — **always
compare at matched finest level** (that section's gate ids use the older
numbering; see §0).

---

## 7. Predicted outcome (as recorded in the driver)

The driver carries two prediction *labels* inline, in the print strings of
`variance_factors` and `verdicts`:

- **[V1] single-level variance reduction ~4.2×** — printed as
  `[predicted ~4.2x]` (`:278`).
- **[V2] level-difference variance reduction ~3.2×** — printed as
  `[predicted ~3.2x]` (`:280`).
- **The ordering is the claim, not the magnitudes:** the only thing the driver
  tests is that conditioning helps the single-level estimator *more* than it
  helps the level difference. Hence conditional **standard** MC at the finest
  grid is expected to be the cheapest of the four, and conditional MLMC not to
  pay for itself.
- **Refute if** `cond_std/cond_mlmc ≥ 1` on any seed, or if the cheapest
  estimator is ever not `cond-stdMC`. These are the two coded criteria
  (`all_lt1`, `cheapest_always_condstd` at `:306-307`, folded into `overall` at
  `:318`). The docstring adds a third as *intent* — *"flag any 'win' that flips
  sign as an artifact"* (`:29`) — and `signflip` is computed and printed at
  `:305,312`, but it never reaches `overall`; it is **not** an enforced
  refutation criterion (§8 (iii)).

*Mechanism, attributed:* the write-up `docs/p2_estimator_results.md`:76-78
explains the ordering by saying the coupling has already cancelled much of the
common variance that conditioning targets. That is the write-up's
interpretation. The driver prints no reason and tests no mechanism — it tests
only the ordering.

`4.2` and `3.2` are **hard-coded labels**: no line of the driver tests the
measured factors against them. The only numeric comparison V1/V2 make is
`sl[-3:].mean() > ld[-3:].mean()` — deep levels 4–6 (`:289,296`); the
full-array means printed alongside them (`:277-280`) never enter a decision.
The provenance of `4.2`/`3.2` is not in the code — treat them as the prior recorded for
display, not as a bar.

Measured for the record (`../p2_estimator_results.md` §2, `:73-76`): 4.16×
single-level (≈4.21× deep), 3.19× level-difference (≈3.30× deep).

---

## 8. What the run actually asserts (PASS/FAIL as coded)

No committed spec holds the bar (§0). What follows is the *mechanism* — the
verdict strings, quoted, because they are the gate as executed.

`variance_factors` (`:263`) averages ratios **across seeds, per level** (mean of
ratios, not ratio of means): `sl[l] = mean_s VPn/VPc`, `ld[l] = mean_s Vn/Vc`
for `l ≥ 1`. "Deep levels" everywhere means `[-3:]` — levels 4, 5, 6.

`verdicts` (`:284`) then prints:

```
[V1] single-level variance reduction ~4.2x: measured ~{sl_m:.2f}x (deep levels)
[V2] level-diff   variance reduction ~3.2x: measured ~{ld_m:.2f}x (deep levels)
     => single-level reduces MORE than level-diff ({sl_m:.2f} > {ld_m:.2f}): CONFIRMED|REFUTED

[V3] conditional STANDARD MC is cheapest & cond-MLMC does NOT beat it (cond_std/cond_mlmc < 1):
     eps={eps} L*={Lstar}  cond_std/cond_mlmc per seed = [...]  cheapest={set}  (SIGN FLIP!)|(stable)
     => cond_std/cond_mlmc < 1 on every seed: CONFIRMED|REFUTED;  cheapest is conditional-stdMC throughout: YES|NO
```

where `sl_m, ld_m = sl[-3:].mean(), ld[-3:].mean()` (`:289`), and the summary,
gated on `overall = (sl_m > ld_m) and all_lt1 and cheapest_always_condstd`:

```
SUMMARY: prediction CONFIRMED|NOT fully confirmed — conditioning helps single-level more than
level-diff, conditional standard MC is the cheapest of the four, and conditional MLMC does not
pay for itself.
```

**Three gaps to be explicit about.** (i) The **unbiasedness check does not
gate** — `z ≥ 4` prints `CHECK` and the run continues; `overall` never reads it.
Only the §4 `Egeom` unit test can abort the run. (ii) The `4.2×` / `3.2×`
magnitudes do not gate either, as above. (iii) `signflip` is computed and
printed but not folded into `overall`.

---

## 9. Reproduction

```
python p2_conditional_verify.py            # full:  N = 14_000, unit-test M = 200_000
python p2_conditional_verify.py --quick    # quick: N =  8_000, unit-test M =  60_000
```

Run from the repo root (the driver imports `layer1b_mlmc_asian` by module name).
Output is **stdout only** — no figure, no CSV, no JSON; `sys.stdout.reconfigure(
encoding="utf-8")` is attempted and silently ignored on failure. Total wall time
is printed at the end; `../p2_estimator_results.md`:113-114 records ≈1–4 min
**each** for the two P2 drivers. Blocks printed, in order: the `[G-C4]` banner
with the estimator line and `mode=`, the `E[geom|W]` unit test, the G-C4
header, `kappa`/`alpha`/`beta_naive`/`beta_cond`, the unbiasedness line, one
cost table per ε, the per-level variance factors, and the verdicts.

**Run mode of the archived numbers.** No run log is committed. Both records give
the reproduction command without `--quick` — `docs/gate_checks/README.md`:51
(*"**Run:** `python p2_conditional_verify.py` · seeds 11/99/1234"*) and
`docs/p2_estimator_results.md`:113-114 (*"`python p2_conditional_verify.py`
(≈1–4 min each; `--quick` for a fast pass)"*) — so FULL is the documented run.
Neither record states the mode of the archived 0.41–0.45 itself; that it was
FULL is an inference from the documented command, not a recorded fact. The
banner prints `mode=FULL|QUICK` at run time (`:334-335`), so a rerun can be
diffed against the recorded figures.

---

## 10. Honest scope note — what this does not establish

- **Single point in parameter space.** `PARAMS` only: `H = 0.10`, `eta = 1.50`,
  `rho = −0.70`, `xi0 = 0.04`, `S0 = K = 100`, `T = 1`, `r = 0`, `n0 = 32`. No
  H-sweep, unlike the antithetic G-A2. The conclusion is not shown to be robust
  in `H`, in moneyness, or at `r ≠ 0`.
- **Two accuracies, one option.** `ε ∈ {0.10, 0.05}`, arithmetic Asian only.
  Nothing is claimed at tighter ε, where a deeper L\* could in principle move
  the crossover.
- **The production flag is not exercised.** §2 — the driver verifies its own
  `_cond_payoffs`, not `mlmc_asian_level(conditional=True)`; no test compares
  the two.
- **Identical bias is asserted, not verified.** §1 — the docstring's shared-bias
  argument underwrites the shared-L\* design, but the driver only z-tests the
  control's mean; it never compares the naive and conditional bias curves level
  by level.
- **γ=1 work model, hand-copied.** The `1.0 / 1.5` coefficients are literals in
  the driver, not a call into `_level_cost_coef`; they can drift from the engine
  silently, and nothing tests them against it.
- **The inline path build can drift too.** §6 — it is today operation-for-
  operation the engine's κ=0 `_simulate_paths`, but nothing asserts that it
  stays so.
- **Rates come from one seed.** `alpha`, `beta_n`, `beta_c` and hence L\* are
  fitted on seed 11 alone; only the costs and variance factors are swept over
  all three seeds.
- **A cost model, not a timing.** Except for κ\*, all four costs are analytic
  Giles/standard-MC formulae evaluated on pilot variances — no end-to-end run to
  a target ε was timed.

*Questions this reconstruction could not answer from the code are recorded in [`RVL-009_open_items.md`](RVL-009_open_items.md).*
