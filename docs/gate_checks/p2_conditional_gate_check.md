# Gate-check spec — conditional MC for the arithmetic Asian (geometric control variate)

> **Driver committed:** 2026-06-22 (`d778074`) · **Spec reconstructed:** 2026-09-29
> from the driver scripts under RVL-009. The original specification was
> chat-only and is not recoverable. Predictions and gates below are as
> recorded in the code; this document does not attest when they were
> written.

**Status:** driver `p2_conditional_verify.py` **BUILT & MEASURED** (G-C4 **CONFIRMED**; seeds 11 / 99 / 1234 — D21, 2026-06-23) · **Engine:** non-invasive — imports `PARAMS`, `volterra_weights`, `_paths_from_increments` from `layer1b_mlmc_asian.py` and adds only the conditional pieces
**Depends on:** `layer1b_mlmc_asian.py` (κ=0 Volterra machinery, `PARAMS`, `_paths_from_increments`) · **Sibling:** `p2_conditional_build_and_verify.md` carries the build and verification narrative; **this file is the gate half only** — the gates, their thresholds and the PASS/FAIL bar

> The thread this documents followed the project gate-check discipline: state the mechanism,
> fix a falsifiable prediction in code, then validate against a known answer. **This document
> records that sequence from the driver; it does not enact it.** Here the known answer is a
> brute-force conditional Monte-Carlo estimate of `E[geom | W]`, and the adjudication is
> against a bar fixed in the code.
> The load-bearing discipline in this thread is the *matched finest level* rule: the antithetic
> thread (G-A4) established that a free-running adaptive driver manufactures phantom cost
> ratios by handing different finest levels `L` to estimators whose bias is identical (driver
> docstring, `p2_conditional_verify.py:18-23`). This gate pins `L*` instead. A win that only
> exists because of `L`-selection is not a win.

---

## 0. Scope discipline — what this spec is, and is NOT

**IN:** the four-way matched-accuracy cost comparison **G-C4** (naive MLMC, conditional MLMC,
naive standard MC, conditional standard MC), the two variance verdicts **V1** / **V2**, and the
**unbiasedness** check on the control variate. Thresholds, verdict strings and the PASS/FAIL
bar.

**NOT here:**
- The build narrative — the derivation of `E[geom | W]`, the array layout of `_cond_payoffs`,
  and the run log. That is `p2_conditional_build_and_verify.md`.
- **Any rate claim.** The driver computes `beta_naive` and `beta_cond` by polyfit from the
  seed-11 pilot and prints them once (lines 218–221); **no gate is placed on either, and the
  driver states no expected value for either.** The production docstring characterises the
  conditional estimator as a constant-factor variance reduction ("a large constant-factor
  single-level variance reduction (~4x at the defaults)", `layer1b_mlmc_asian.py:277`), not a
  repair of the β<γ pathology. Compare the antithetic thread, which was **REFUTED** on exactly
  the rate point.
- **κ=1.** `_cond_payoffs` builds the variance path with the κ=0 optimal-discretisation
  convolution only; it takes no `kappa` argument. Whether κ=1 should drive the conditional
  estimator is a separate question, settled later by G-H4 (`gh4_kappa1_adoption_spec.md`).

The driver emits **stdout only** — no CSV, no JSON, no figure. The printed verdict lines *are*
the gate record, so §6 quotes them verbatim. Two notes on matching this document to that
stdout:

- The driver labels its verdict blocks `[V1]`, `[V2]` and `[V3]` (lines 290, 292, 298).
  `[V3]` carries clauses 2 and 3 of G-C4 below; the driver has no printed id for the
  pre-gate unit test (see §6). `docs/gate_checks/README.md:23` likewise lists this thread's
  gates as "G-C4 (+ variance verdicts V1/V2, unbiasedness)" and does not mention V3.
- The closing SUMMARY prose at lines 320–323 ("conditioning helps single-level more than
  level-diff, conditional standard MC is the cheapest of the four, and conditional MLMC does
  not pay for itself") is printed **unconditionally**. Only the `CONFIRMED` /
  `NOT fully confirmed` token varies. The trailing clause is not a verdict and must not be
  quoted as one.

---

## 1. The estimator

For the arithmetic Asian under rough Bergomi, with `W = dW1` the Brownian driver of the
variance path:

```
P_cond = arith − ( geom − E[geom | W] )
       = arith − geom + E[geom | W]
```

a **conditional geometric-Asian control variate**. Both averages use the *same* trapezoidal
weights on the level's own grid: `arith` averages `S`, `geom` averages `logS` and exponentiates
(`_cond_payoffs`, lines 76–83).

`geom − E[geom|W]` has zero conditional mean given `W`, hence zero unconditional mean, so
`P_cond` is unbiased for the arithmetic-Asian price **at every grid level**. Its discretisation
bias is therefore *identical* to the naive arithmetic payoff on the same grid — which is what
licenses the shared-`L*` rule of §3. The control removes the orthogonal driver `W_perp`
exactly, rather than approximately, because the conditional mean is closed form (driver
docstring, lines 11–13).

---

## 2. The load-bearing object — the closed-form `E[geom | W]`

Conditional on the variance path, the log-prices are Gaussian, so the trapezoidal log-average
`LG` is Gaussian and `geom` is a Black-type expectation. The driver computes, per path:

```
mu_j   = (r − ½V_j)·dt + √V_j·ρ·dW1_j                  # E[dlogS_j | W]
muG    = trapezoidal average of the cumulated mu
Wbar_j = 1 − (1 + 2j)/(2n)                             # cumulative trapezoidal weight above step j
sigG²  = (1 − ρ²)·dt·Σ_j V_j·Wbar_j²
F      = S0·exp(muG + ½sigG²)
E[geom|W] = e^{−rT}·( F·Φ(d1) − K·Φ(d2) ),   d1 = (log(F/K) + ½sigG²)/sigG,  d2 = d1 − sigG
```

(lines 86–96). `Wbar_j` is the *exact* coefficient of `dlogS_j` in `LG` under the trapezoidal
weights (that coefficient is `(n − j − 0.5)/n`), so `sigG²` is exact, not an approximation —
the whole estimator rests on this. `sigG²` is floored at `1e-300` inside the square root only
(line 92).

---

## 3. Matched `L*` — the artifact this gate exists to avoid

**Rule:** do *not* let the adaptive driver pick `L`. Fix one `L*` per ε and give it to all four
estimators. Their bias is identical (§1), so any difference in finest level is an artifact of
the selector, not a property of the estimator.

`L*` is chosen by `_choose_L` from the **seed-11 naive** level-diff means `mn` only
(lines 185–195, the comment at line 214, and the call at line 238). For `Lc = 2 … Lmax`, with
`offs = 0,1,2` (truncated at `Lc`):

```
tail = max_o |mn[Lc − o]|·2^{−α·o}
accept Lc  iff  tail / (2^α − 1) ≤ ε/√2
```

returning the first accepting `Lc`, else `Lmax`. `α` is the fitted weak-rate exponent
(`−polyfit` slope of `log2|mn[1:]|` against levels 1…6) **floored at 0.5** (line 217).
`Lmax = 6`, equal to the pilot depth `Lpil = 6`, so `L*` can never exceed the measured range.
ε ∈ {0.10, 0.05} (line 235). The driver's own docstring annotates the expected outcome as
"~2 at eps=0.1, ~5 at eps=0.05" — an annotation, not a threshold; the printed `L*` is
authoritative.

---

## 4. Honest cost — κ is measured, and the headline ratio is κ-invariant

The conditional estimator does extra `O(n)` work per path. Charging it zero would rig the
comparison, so the per-path overhead **κ is measured, not assumed** (`measure_kappa`,
lines 164–182): at level `l = 3` (`n = 256`), `N = 8000` paths, `rng` seed 0, both routines
warmed up, then 8 timed repetitions each — κ = wall(`_cond_payoffs`) / wall(`_paths_from_increments`).
This is deliberate: `_level_cost_coef` in `layer1b_mlmc_asian.py` does **not** charge the
conditional overhead ("it is not charged here; fold it in explicitly for exact cost
comparisons", lines 320–323), so the driver folds it in.

Per-level work (lines 232–233, 246–249):

```
Cn[l] = n0·2^l · (1.0 if l == 0 else 1.5)          # naive: fine + coarse path
Cc[l] = κ·Cn[l]                                     # conditional: same two paths, κ× the work
naive-MLMC   = (2/ε²)·(Σ_{l≤L*} √(Vn[l]·Cn[l]))²    # Giles optimal
cond-MLMC    = (2/ε²)·(Σ_{l≤L*} √(Vc[l]·Cc[l]))²
naive-stdMC  = (2/ε²)·Var(P_naive at L*)·(n0·2^{L*})
cond-stdMC   = (2/ε²)·Var(P_cond  at L*)·(κ·n0·2^{L*})
```

**The headline ratio `cond-stdMC / cond-MLMC` is exactly κ-invariant.** Verifiable from the
arithmetic: `Cc = κ·Cn` makes the Giles sum scale as `√κ` and its square as `κ`, and the
standard-MC cost carries one factor of `κ`. So the gate's central number cannot be moved by a
mistimed κ — only the *cross-estimator* comparisons against the naive column can.

---

## 5. The committed prediction (hard-coded in the driver)

Conditioning on `W` removes the orthogonal driver `W_perp` from the control exactly (driver
docstring, lines 11–13). The MLMC level difference `Y_l = P_f − P_c` benefits less, "because
the coupling has already cancelled much of the common variance that conditioning targets"
(`docs/p2_estimator_results.md:76-79`). So:

> **Prediction.** Conditioning helps the **single-level** variance *more* than it helps the
> **level difference**. Therefore it improves standard MC more than it improves MLMC, and the
> cheapest of the four estimators at matched `L*` is **conditional standard MC** — with
> conditional MLMC failing to beat it.

The prediction is recorded as pre-registered in `ROADMAP.md:739` (D21): "pre-check predicted
the removed share is *higher* single-level than level-difference". The driver itself carries
only the hard-coded `[predicted …]` annotations below; the pre-registration is not provable
from code alone.

| quantity | predicted | falsified if |
|---|---|---|
| single-level variance reduction (V1) | large, reference ≈4.2× at deep levels | — (reported, not gated) |
| level-diff variance reduction (V2) | smaller, reference ≈3.2× at deep levels | — (reported, not gated) |
| V1 vs V2 ordering | single-level reduces **more** than level-diff | `sl ≤ ld` at deep levels |
| `cond-stdMC / cond-MLMC` | `< 1` on every seed, at both ε | any seed with ratio `≥ 1` |
| cheapest of the four | `cond-stdMC`, at every seed and ε | any other estimator cheapest |
| stability | no sign flip across seeds 11 / 99 / 1234 | the ratio straddles 1 |

β is **not** in this table: the driver commits no prediction about it (§0).

The 4.2× / 3.2× figures are printed as `[predicted ~4.2x]` / `[predicted ~3.2x]` reference
annotations (lines 277–280, 290–293). **Neither is a threshold.** The gate is placed on the
*ordering* `sl > ld`, not on either magnitude — a point worth stating plainly, because the
printed layout invites the opposite reading.

---

## 6. Gate-check bar (PASS/FAIL committed before any measurement)

Pilot configuration, all levels `l = 0 … 6` on grids `n = 32·2^l` (so 32 … 2048), `N` paths per
level, defaults `H = 0.10`, `η = 1.50`, `ρ = −0.70`, `ξ₀ = 0.04`, `S₀ = K = 100`, `T = 1`,
`r = 0`, `n0 = 32` (`layer1b_mlmc_asian.py:140-150`):

| | FULL | `--quick` |
|---|---|---|
| `N` per level (`estimate_cond_rates`) | 14 000 | 8 000 |
| `M` brute-force paths (unit test) | 200 000 | 60 000 |

Seeds **11 / 99 / 1234**, one independent pilot each; seed 11 is the *reference* pilot from
which `α`, `β`, `L*` and the unbiasedness `z` are taken. **Thresholds do not relax in `--quick`
mode** — only the sample counts shrink, so `--quick` is a strictly noisier run against the same
bar.

**Pre-gate: closed-form `E[geom|W]` unit test (hard abort).** The driver calls this
`unit_test_Egeom` and prints it as
`UNIT TEST — closed-form E[geom|W] matches brute-force conditional MC` (lines 101, 103). **It
carries no gate id**: neither the driver nor `docs/gate_checks/README.md:23` assigns one, and
this spec does not mint one. Four fixed variance paths (`rng` seed 7, `n = 48`, `n_paths = 4`);
for each, `M` brute-force conditional draws of `dW2`. Compare the closed form against the
brute-force mean via `z = |cf − bf| / se`. **PASS iff `z < 3.5` on all four paths**
(lines 119, 121). This is an `assert`, not a report: failure aborts the run with
`"Egeom closed form FAILED"` (line 337) and no gate number is produced.

**G-C4 — matched-accuracy cost, naive vs conditional.** PASS requires **all three** of:

1. **V1 > V2 ordering.** Deep-level (last three levels) mean variance ratios satisfy
   `sl_m > ld_m`, where `sl` is `Var(P_naive)/Var(P_cond)` per level and `ld` is
   `Var(Y_naive)/Var(Y_cond)` per level, each averaged across the three seeds.
   Verdict string: `=> single-level reduces MORE than level-diff (...): CONFIRMED` / `REFUTED`.
2. **Ratio below 1 everywhere.** `cond-stdMC / cond-MLMC < 1` for **every** seed at **both**
   ε ∈ {0.10, 0.05}. Verdict string: `=> cond_std/cond_mlmc < 1 on every seed: CONFIRMED` /
   `REFUTED`. A sign flip across seeds is flagged `(SIGN FLIP!)` versus `(stable)`; the flag is
   diagnostic only — any flip already fails this clause.
3. **Cheapest is `cond-stdMC` throughout**, at every seed and both ε. Verdict string:
   `cheapest is conditional-stdMC throughout: YES` / `NO`.

All three → `SUMMARY: prediction CONFIRMED`. Any one failing → `NOT fully confirmed`.

**Unbiasedness — advisory, not a clause of G-C4.** On the seed-11 pilot,
`z = max_l |E[arith] − E[P_cond]| / s.e.`, with `s.e. = std(geom − E[geom|W]) / √N` — the
*paired* control-variate standard error, which is the right yardstick because the two means
share their paths. Printed as:

```
control-variate unbiasedness: max_l |E[arith]-E[P_cond]|/s.e. = <z>  (mean-zero control -> identical bias)   OK | CHECK
```

`OK` iff `z < 4`, else `CHECK` (line 230). **It does not enter the PASS/FAIL conjunction**
(line 318): a `CHECK` prints but cannot by itself fail the gate.

> **MEASURED (2026-06-23, D21).** Gate outcome **CONFIRMED** (seeds 11 / 99 / 1234, per
> `docs/gate_checks/README.md:23` and `ROADMAP.md:739`): conditional **standard** MC is the
> cheapest of the four; conditional MLMC does **not** beat it,
> `cond-stdMC / cond-MLMC` ≈ **0.41–0.45** — below 1, stable across seeds, and κ-invariant by
> §4. Conditioning helps the single level more than the level difference, exactly as predicted
> in §5. Writeup: `../p2_estimator_results.md`.
>
> *Two provenance caveats on that blockquote.* (i) The ratio range **0.41–0.45** is quoted from
> `docs/gate_checks/README.md:23,50`; `docs/p2_estimator_results.md:90` — the writeup this spec
> points readers to — records **0.42–0.45** instead. The driver stores no value, so neither is
> checkable from code. (ii) No unbiasedness `z` is quoted here.
> `docs/gate_checks/README.md:46` records "unbiasedness (z≈1.5)", but
> `docs/p2_estimator_results.md:69` attaches a `z < 1.5` to the closed-form pre-gate instead.
> The two records disagree, the driver stores neither, and the `OK`/`CHECK` flag is likewise
> not recorded anywhere. The verdict above rests on the index's CONFIRMED entry, not on a `z`.

**Run:** `python p2_conditional_verify.py` (FULL) · `python p2_conditional_verify.py --quick`.
From the repo root; the driver imports from `layer1b_mlmc_asian.py`.

---

## 7. Scope and honest boundaries

- **A constant-factor result, not a rate result.** The gate says conditional standard MC is
  cheapest at matched accuracy for *these* parameters. It says nothing about β, and nothing
  about whether MLMC is repairable under rough vol. The driver reports `beta_naive` /
  `beta_cond` for information and predicts nothing about them.
- **One parameter point.** `H = 0.10`, ATM (`S₀ = K = 100`), `r = 0`, `T = 1`, arithmetic Asian
  only. No H-sweep, no strike sweep, no maturity sweep — unlike G-A2
  (`p2_antithetic_gatecheck.py:62`, `H_list = [0.05, 0.10, 0.20, 0.35]`) and G-H1a
  (`gh1_kappa1_finepath.py:35`, `for H in (0.05, 0.10, 0.20)`), this thread never sweeps `H`.
  The conclusion is not established off this point.
  *Gate-id note.* The ids above follow the **driver / `docs/gate_checks/README.md` scheme**,
  in which G-H1a…d are the κ=1 **fine path** (`gh1_kappa1_finepath.py`) and G-H2a…c the κ=1
  **coarse coupler** (`gh2_kappa1_coupler.py`: G-H2a telescoping, G-H2b coupling tightness,
  G-H2c rate β≈2H, at the single default `H`). `docs/gate_checks/kappa1_hybrid_coupling_design.md`
  §8 uses an **older, incompatible** numbering (G-H1 coarse-law exactness/tightness/covariance,
  G-H2 rate swept over `H∈{0.05,0.10,0.20,0.35}`, G-H3 variance factor, G-H4 Giles cost). The
  discrepancy is recorded here, not reconciled.
- **`L*` is inherited from the naive estimator's bias test.** That is the identical-bias rule of
  §3 and it is the right discipline, but it does mean `L*` is only as good as the seed-11 `mn`
  fit and the `α` floor of 0.5.
- **Cost is a model, not a wall-clock measurement** — `(2/ε²)·…` Giles/standard-MC formulae with
  per-level work in units of grid points. Only κ is empirical. The 1.5 coefficient charges
  fine + coarse work and mirrors `_level_cost_coef` (`layer1b_mlmc_asian.py:317-326`), which
  the driver re-states inline at line 232 rather than calling.
- **The pilot variances are estimated at the same `N` on every level**, including the finest
  (`n = 2048`). No level gets extra paths, so the deep-level ratios that decide V1/V2 are the
  noisiest numbers in the run. The gate is placed on an *ordering* and on a ratio recorded at
  ≈0.41–0.45; no design rationale connecting the two is stated in the driver.
- **`_cond_payoffs` (`p2_conditional_verify.py:52-97`) is a standalone re-implementation of the
  production `_cond_asian_payoff` (`layer1b_mlmc_asian.py:265-314`)**, including the path build
  that `_simulate_paths` (`layer1b_mlmc_asian.py:216`) provides. The `mu` / `M` / `muG` / `jj` /
  `Wbar` / `sigG2` / `sigG` / `F` / `d1` / `Egeom` algebra of §2 appears line for line in both.
  The driver gives no reason for the duplication. Both arms of the comparison ride on it: the
  "naive" arithmetic payoff in `estimate_cond_rates` is `_cond_payoffs`'s own `arith`, not
  `_paths_from_increments` (which the driver imports but uses only in `measure_kappa`). A
  future change to either the production Volterra path or the production conditional payoff
  would not propagate here, and the pre-gate unit test would not catch it — it compares the
  closed form only against the driver's *own* `geom` (lines 116–117).

---

## 8. Open items — intent not recoverable from the driver

*Questions this reconstruction could not answer from the code are recorded in [`RVL-009_open_items.md`](RVL-009_open_items.md).*
