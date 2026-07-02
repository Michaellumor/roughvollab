# RoughVolLab Error Audit — 2026-07-02

**Baseline:** commit `3f59638` · `python -m pytest -q` → **252 passed, 2 skipped
(torch absent → layer-3 skips), 6 warnings, 1234.53s (20:34)**. The 6 warnings
are overflow `RuntimeWarning`s from `rough_heston_cf.py:184` raised inside
tests that deliberately exercise the guarded overflow path — expected.

**How this audit was made:** a multi-agent deep read of every module produced
~50 candidate issues; each candidate relevant to correctness was then
**adversarially re-verified against the current code** (refute-first), with
runtime measurements where cheap. Five candidates were refuted and are listed
at the bottom so future audits don't rediscover them. **No fixes have been
applied** — this file is the worklist.

---

## How to use this file (humans and AI models)

- One issue per `### RVL-NNN` block. Every block has the same 9 fields.
- **Lifecycle:** `Status: OPEN → IN_PROGRESS → FIXED → VERIFIED-FIXED`.
  Update the Status field in place and append a dated `- **Fix log:**` line.
  Never delete a block; never renumber.
- **Fix protocol:** one issue = one commit. Run the issue's **Acceptance
  test**, then the full suite (`python -m pytest -q`, expect the baseline
  above ± the fix), then append a dated decision entry to `ROADMAP.md`'s
  decisions log (append-only).
- **Severity rubric (the project owner's):**
  - **HIGH** — produces wrong numbers in results/figures/papers, or breaks a
    real code path.
  - **MEDIUM** — wrong in edge cases, doc-vs-code mismatch, masked test
    failures, misleading contracts.
  - **LOW** — cosmetic: stale metadata, naming, duplication, doc precision.
- **Confidence:**
  - `VERIFIED (runtime)` — reproduced with a measurement this session; the
    reproduction command is in Evidence.
  - `VERIFIED (static)` — confirmed by direct code reading of the cited lines
    this session (quoted in Evidence).
  - `PLAUSIBLE` — reported by one static reviewer pass, **not** independently
    re-verified. Re-verify before fixing; if refuted, move it to the refuted
    table instead of deleting.
- Environment note for any runtime work: set `PYTHONIOENCODING=utf-8`
  (modules print Greek glyphs; Windows cp1252 consoles crash otherwise).
  torch is NOT installed (Python 3.14; `.venv-layer3` never created) — a
  standing environment limitation, not an issue block.

## Summary

| ID | Sev | Confidence | Status | File | One-liner |
|----|-----|-----------|--------|------|-----------|
| RVL-001 | HIGH | VERIFIED (runtime) | OPEN | layer1_rough_vol.py:257 | fbm_hybrid over-subtracts near-diagonal kernel terms → Var(B^H_1)≈0.91, internally inconsistent |
| RVL-002 | HIGH | VERIFIED (runtime) | OPEN | layer1_rough_vol.py:439 | rBergomi compensator uses continuum t^{2H} → E[V_t] up to 65% low |
| RVL-003 | MEDIUM | VERIFIED (static) | OPEN | interpret_h.py:187 | monotonicity check bounds steps, not trend — shallow humps pass as invertible |
| RVL-004 | MEDIUM | VERIFIED (static) | OPEN | layer4_calibrate_surface.py:101 | calibrate_surface drops `weights`; patched only by wrapper in calibrate_btc.py |
| RVL-005 | MEDIUM | VERIFIED (static) | OPEN | rough_heston_cf.py:219 | core CF has no overflow/NaN guard; only the lifted wrapper guards it |
| RVL-006 | MEDIUM | VERIFIED (static) | OPEN | layer1c_roughness_audit.py:34 | GJR bias constants in prose (+0.05/+0.005) vs code (+0.062/+0.006) |
| RVL-007 | MEDIUM | VERIFIED (static) | OPEN | layer1b_mlmc_asian.py:~225 | volterra_weights lacks the 0<H<1 guard its twin in roughvol_core has |
| RVL-008 | MEDIUM | VERIFIED (static) | OPEN | test_rv_series.py:74 | importorskip silently skips the load-bearing round-trip test |
| RVL-009 | MEDIUM | VERIFIED (runtime) | OPEN | docs/gate_checks/README.md:100 | five referenced spec files are absent from the repo |
| RVL-010 | MEDIUM | PLAUSIBLE | OPEN | rh_beta_gate.py:34 | hardcoded β reference table can drift from re-runs |
| RVL-011 | MEDIUM | PLAUSIBLE | OPEN | gh4_kappa1_conditional.py:149 | κ=1 adoption verdict rests on 3 seeds with sign-flipping cost ratios |
| RVL-012 | MEDIUM | PLAUSIBLE | OPEN | rough_heston.py:82 | default ν=0.20 IS the validated ceiling; no runtime warning |
| RVL-013 | MEDIUM | PLAUSIBLE | OPEN | execution_alpha_phase1.py:62 | boolean-mask indexing in _causal_schedule may misalign shapes |
| RVL-014 | MEDIUM | PLAUSIBLE | OPEN | execution_alpha_phase1.py:67 | final-step forced liquidation can be overridden by the clip |
| RVL-015 | MEDIUM | PLAUSIBLE | OPEN | estimate_h.py:171 | catch-all except masks estimator crashes as NaN |
| RVL-016 | LOW | VERIFIED (static) | OPEN | layer1_rough_vol.py:200 | docstring restricts H∈(0,½); kernel and core allow (0,1) |
| RVL-017 | LOW | VERIFIED (static) | OPEN | layer1_rough_vol.py:1 | module header doesn't warn about L1-1 / point to roughvol_core |
| RVL-018 | LOW | VERIFIED (static) | OPEN | test_layer1c.py:31 | test name promises positivity; asserts ordering (docstring is honest) |
| RVL-019 | LOW | VERIFIED (static) | OPEN | CITATION.cff:32 | stale date/version vs ROADMAP state |
| RVL-020 | LOW | UNCERTAIN | OPEN | run_phaseb.md:27 | "78 passed" vs README's "87 tests" — likely different scopes; reconcile |
| RVL-021 | LOW | PLAUSIBLE | OPEN | layer4_calibrate.py:247 | printed EXP labels out of order (1,3,2,4) vs docstring |
| RVL-022 | LOW | PLAUSIBLE | OPEN | binance_data.py/deribit_surface.py | duplicated _http_get with inconsistent error handling |
| RVL-023 | LOW | PLAUSIBLE | OPEN | module_map.py | hardcoded status pills — regenerate or the flagship PNG goes stale |
| RVL-024 | LOW | PLAUSIBLE | OPEN | layer1b_mlmc_asian.py:749 | honest-negative headline lives only in print output, not the docstring |
| RVL-025 | LOW | VERIFIED (static) | OPEN | README.md (κ=1 row) | κ=1 MLMC at l>0 is intentionally NotImplemented; README phrasing could oversell |
| RVL-026 | LOW | PLAUSIBLE | OPEN | test_rough_heston_cf.py:136 | N_riccati=2000 pinned empirically; convergence table only in __main__ |
| RVL-027 | LOW | PLAUSIBLE | OPEN | rough_kernel_soe.py:68 | BB tuning constants (α=1.6, β=0.4275) uncited, no ablation |
| RVL-028 | LOW | PLAUSIBLE | OPEN | test_rough_heston_lifted.py | no lifted variant of the non-anticipation (causality) test |
| RVL-029 | LOW | PLAUSIBLE | OPEN | interpret_h.py:98 / estimate_h.py:94 | _FLAT_SLOPE=0.25 and _MF_QS grid hardcoded, underivations undocumented |
| RVL-030 | LOW | PLAUSIBLE | OPEN | calibrate_btc.py:69 | per-maturity N_riccati step h=1.1e-4 empirically pinned, no sensitivity note |
| RVL-031 | LOW | PLAUSIBLE | OPEN | test_interpret_h.py:127 | de-bias recovery never tested in the collapse-zone window regime |
| RVL-032 | LOW | PLAUSIBLE | OPEN | layer3_deep_hedging.py:235,338,354 | tensor-device handling, eta==0 ignores H, hardcoded gate tolerance |
| RVL-033 | LOW | PLAUSIBLE | OPEN | p2_baseline_regen.py:78 | two coexisting H=0.10 baselines; no note which is canonical |
| RVL-034 | LOW | PLAUSIBLE | OPEN | deribit/kline/rv data layer | four documented invariant fragilities (see block) |
| RVL-035 | LOW | PLAUSIBLE | OPEN | test_layer3_deep_hedging.py:18 | torch skips reduce coverage visibility (currently ALL skipped on this machine) |

---

## HIGH

### RVL-001 — fbm_hybrid over-subtracts near-diagonal kernel contributions
- **Severity:** HIGH
- **Confidence:** VERIFIED (runtime, 2026-07-02)
- **Status:** OPEN
- **Location:** `layer1_rough_vol.py:257-264` (root cause), function `fbm_hybrid` (:175)
- **Evidence:** Measured this session (n=128, H=0.1, 4000 paths, `np.random.seed(7)`):
  **Var(B^H_1) = 0.9058 ± 0.0203**. The compensator downstream assumes 1.0; the
  module's own Riemann–Liouville kernel (`t^{H-1/2}/Γ(H+½)`, :216) implies a
  continuum variance of **2.255** at H=0.1 — the measurement matches *neither*,
  so the implementation is internally inconsistent (exactly as ROADMAP L1-1
  records; it measured 0.89 at 3000 paths). Root cause by code reading: the FFT
  convolution (:232-254) only ever adds point-kernel terms at lags ≥ b
  (`kernel_fft_vec[b+i]`), but the correction loop (:258-262) subtracts
  `kernel((t_k − j·dt))·dW[j]` for **all** k ≥ j — every near-diagonal term with
  k−j < b is subtracted without ever having been added. Control: the trusted
  engine gives Var(W̃_1) = 0.8445 ± 0.0189 vs its own discrete v_n = 0.8311
  (ratio 1.016, consistent).
  Repro: `python <scratch>/verify_l1_1.py` (script preserved in the session
  scratchpad; 10 lines — import `fbm_hybrid`, simulate, compare variances).
- **Impact:** Every plot/number produced by Layer 1's simulation path (Sections
  2–4 of `layer1_rough_vol.py`, `output/section2_hybrid.png` etc.) carries the
  wrong process law. Fatal for pricing — which is why `roughvol_core.py` exists
  and why Layers 1b/1c/4 deliberately do NOT import Layer 1. No published paper
  number depends on it (papers use core/1b/1c/4), but the repo's advertised
  quick-start (`python layer1_rough_vol.py`) showcases the buggy engine.
- **Fix plan (from ROADMAP L1-1, reuse verbatim):** rewrite `fbm_hybrid`
  against the κ=0 weights in `roughvol_core.volterra_weights` (vectorised,
  FFT), normalise so Var(W̃_t) matches the discrete formula; alternatively fix
  the correction loop bound to `for k in range(j+b, n)` and normalise the
  kernel convention, then reconcile with RVL-002.
- **Acceptance test:** empirical Var(W̃_T) within MC noise of the discrete v_n
  (|ratio − 1| < 4·s.e.), and `max_t |E[V_t]/ξ₀ − 1|` within 3 s.e. at
  N=100k — these are exactly Layer 1b §1's checks; reuse them. Add both as a
  new `test_layer1_rough_vol.py` so the fix is pinned.
- **Related:** RVL-002 (same fix campaign), RVL-016, RVL-017, ROADMAP issue L1-1.

### RVL-002 — rough_bergomi_paths uses the continuum compensator t^{2H}
- **Severity:** HIGH
- **Confidence:** VERIFIED (runtime, 2026-07-02)
- **Status:** OPEN
- **Location:** `layer1_rough_vol.py:439` (`- 0.5 * eta**2 * t[k+1]**(2*H)`)
- **Evidence:** Measured this session (n=128, H=0.1, η=1.9, 8000 paths,
  `np.random.seed(11)`): **E[V_t]/ξ₀ = 0.803 at T, dipping to 0.350 early in
  the path** (exact answer: 1.0 at every t). The trusted engine's
  `rough_bergomi_paths` under the same settings stays in [0.880, 1.062]
  (MC noise around 1). The bias is worst early because Var(B^H_t) of the
  hybrid process is furthest from t^{2H} there. Repro: same script as RVL-001.
- **Impact:** All Layer-1 rBergomi output has a large, t-dependent forward
  variance bias — cosmetic for path *plots*, fatal for any pricing or
  H-estimation built on it. Isolated by design (nothing downstream imports it).
- **Fix plan:** switch the compensator to the discrete variance
  `v_i = 2H·dt·cumsum(g²)` of whatever kernel RVL-001's fix lands on
  (i.e. copy `roughvol_core.rough_bergomi_paths`'s construction, :176-178).
- **Acceptance test:** `max_t |E[V_t]/ξ₀ − 1|` within 3 s.e. at N=100k, η=1.9.
- **Related:** RVL-001 (fix together), ROADMAP issue L1-1.

---

## MEDIUM

### RVL-003 — bias-curve monotonicity check bounds steps, not the trend
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (static, 2026-07-02)
- **Status:** OPEN
- **Location:** `interpret_h.py:173-187` (`_is_monotone`)
- **Evidence:** The check is `np.all(d >= -tol) or np.all(d <= tol)` with
  tol=0.012 on the per-step differences. A curve that declines by ≤0.012 per
  step for 20 steps (net −0.24 — decisively non-monotone) passes as
  "monotone increasing". The check constrains individual steps, never the
  cumulative trend.
- **Impact:** `_is_monotone` gates the de-biasing inversion: a shallow hump
  misclassified as monotone would let the inversion return a unique H where
  the honest answer is "multivalued". In practice the measured bias-curve
  humps are steep (per-step declines ≫ 0.012), so no known published number
  is wrong — but the guard doesn't guard the regime it exists for.
- **Fix plan:** additionally require net-trend consistency, e.g.
  `(m[-1]-m[0])` sign must match the step-majority sign and
  `max(cummax(m)-m) <= k·tol` (drawdown bound), or fit isotonic regression
  and bound the residual.
- **Acceptance test:** unit test feeding a synthetic shallow hump
  (e.g. 30 steps of −0.008 after a rise) → must classify NOT monotone;
  plus a genuinely monotone noisy curve (wobble < tol) → monotone.
- **Related:** RVL-029, RVL-031.

### RVL-004 — calibrate_surface() silently drops the `weights` parameter
- **Severity:** MEDIUM (would be HIGH if any published run had used it unwrapped)
- **Confidence:** VERIFIED (static, 2026-07-02)
- **Status:** OPEN
- **Location:** `layer4_calibrate_surface.py:101-106`
- **Evidence:** `calibrate_surface`'s signature has no `weights`; it calls
  `calibrate(...)` (which does accept `weights`) without forwarding one.
  `calibrate_btc.py:118-126` defines `calibrate_surface_weighted`, commented
  verbatim: *"The engine-gap fix: calibrate_surface drops `weights`; the
  underlying `calibrate` applies them."* — so the production BTC/ETH runs
  (D39–D42) went through the wrapper and are unaffected.
- **Impact:** API trap: any future caller using the engine directly with the
  intention of weighted calibration silently gets unweighted fits.
- **Fix plan:** add `weights=None` passthrough to `calibrate_surface`; retire
  the wrapper (or keep it as a thin alias); add the parameter to its docstring.
- **Acceptance test:** weighted vs unweighted synthetic surface calibration
  produce different theta when weights are non-uniform; wrapper and engine
  give identical results for identical inputs.
- **Related:** none.

### RVL-005 — rough_heston_cf() has no overflow/NaN guard
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (static; overflow path observed in baseline warnings)
- **Status:** OPEN
- **Location:** `rough_heston_cf.py:219-226` (assembly), `:184` (the ψ² term that overflows)
- **Evidence:** The fractional Riccati's quadratic term overflows at small H +
  high ν + low N_riccati — the pytest baseline shows the RuntimeWarnings from
  `:184` raised in tests of the *guarded wrapper*
  (`rough_heston_lifted._cf_price_guarded`), which catches the resulting NaN.
  The core `rough_heston_cf` itself propagates NaN/inf silently to any direct
  caller.
- **Impact:** direct users of the CF (e.g. a new calibration script that
  doesn't copy the wrapper) get silent NaN prices instead of an error.
- **Fix plan:** early-exit in `_frac_riccati` when `~np.isfinite(psi)` appears
  (raise or return NaN with a warning), or validate/floor `N_riccati` per
  (H, ν, T) as calibrate_btc.py's per-maturity schedule already does.
- **Acceptance test:** a known-overflow combo (see
  `test_rough_heston_lifted.py:76-85`) raises/warns instead of silently
  returning NaN from the unguarded function.
- **Related:** RVL-026, RVL-030.

### RVL-006 — GJR bias constants: prose vs code disagree by ~24%
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (static, 2026-07-02)
- **Status:** OPEN
- **Location:** `layer1c_roughness_audit.py:34-35` (prose: "≈ +0.05 at H=0.05
  … ≈ +0.005 at H=0.3") vs the code's constants (~:416: `0.062…0.006`).
- **Evidence:** direct read of both sites this session. The prose hedges with
  "≈", but +0.05 vs +0.062 is a 24% understatement of a number that P3
  discusses; ROADMAP quotes the prose values in places.
- **Impact:** anyone quoting the header instead of the code/tests understates
  the estimator's bias; no test is wrong (tests use the code constants).
- **Fix plan:** update the module-header numbers to the measured constants and
  add "(see ORACLE_TOLERANCE / section1 output for exact values)".
- **Acceptance test:** grep finds a single consistent pair of constants across
  module header, ROADMAP, and P3 text.
- **Related:** RVL-018.

### RVL-007 — layer1b volterra_weights lacks the 0<H<1 input guard
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (static; two independent passes quoted both sites)
- **Status:** OPEN
- **Location:** `layer1b_mlmc_asian.py:~225` vs `roughvol_core.py:74-75`
- **Evidence:** `roughvol_core.volterra_weights` raises
  `ValueError` unless 0<H<1 (and special-cases H=½); the algorithmically
  identical copy in layer1b has no guard. Downstream imports use the core copy
  (guarded), so exposure is direct calls into layer1b only.
- **Impact:** nonsense H (e.g. 0, negative, ≥1) produces NaN/complex weights
  silently in the layer1b path.
- **Fix plan:** copy the two guard lines from core (or import
  `volterra_weights` from core and delete the duplicate — preferable, one
  engine).
- **Acceptance test:** `pytest` case asserting `ValueError` for H∈{0,1,-0.1}
  via the layer1b entry point.
- **Related:** RVL-022 (same "duplicate drift" species).

### RVL-008 — importorskip masks failure of the load-bearing round-trip test
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (static; quoted by two passes)
- **Status:** OPEN
- **Location:** `test_rv_series.py:74` (`pytest.importorskip("layer1c_roughness_audit")`)
- **Evidence:** `test_matches_phase_a_rung1` validates that the Phase-B RV
  construction is byte-identical to Layer 1c's definition. If
  `layer1c_roughness_audit` ever fails to import (syntax error, bad refactor),
  this test SKIPS instead of FAILING — a green suite with the core validation
  silently disabled.
- **Impact:** masked breakage risk; no current failure (module imports fine —
  the full suite passes 252).
- **Fix plan:** replace with a plain `import` (the module is a first-party
  file in the same repo, not an optional dependency); keep importorskip only
  for genuinely optional deps (torch).
- **Acceptance test:** deliberately break the import locally → suite must go
  red, not yellow.
- **Related:** RVL-035.

### RVL-009 — five spec files referenced by the gate-check index are absent
- **Severity:** MEDIUM
- **Confidence:** VERIFIED (runtime file-system check, 2026-07-02)
- **Status:** OPEN
- **Location:** `docs/gate_checks/README.md:100-113` ("chat-only — to add")
- **Evidence:** recursive search of the repo finds none of:
  `p2_antithetic_build_and_verify.md`, `p2_conditional_gate_check.md`,
  `p2_conditional_build_and_verify.md`, `gh1_kappa1_fine_path_spec.md`,
  `gh4_kappa1_adoption_spec.md`. The index itself flags them as chat-only, so
  this is a known gap — but the audit trail for four gate threads currently
  lives nowhere in the repo.
- **Impact:** the gate-check discipline's promise ("specs live in
  docs/gate_checks/") is unfulfilled for those threads; a referee following
  the index hits dead ends.
- **Fix plan:** reconstruct the five specs from ROADMAP D-entries + driver
  docstrings (each driver documents its gates), or amend the index to point
  at the ROADMAP entries as the canonical record.
- **Acceptance test:** every file named in the index exists in the repo.
- **Related:** RVL-023.

### RVL-010 — hardcoded L1B_BETA reference table can drift
- **Severity:** MEDIUM · **Confidence:** PLAUSIBLE · **Status:** OPEN
- **Location:** `rh_beta_gate.py:34`
- **Evidence (single static pass):** the β(H) reference values the gate
  compares against are hardcoded from one layer1b run; re-running layer1b with
  different N/seed shifts them, and nothing regenerates the table.
- **Impact:** gate could pass/fail for stale reasons. **Fix plan:** regenerate
  the table via a `--regen` path or pin it to the committed run's seed and say
  so. **Acceptance test:** gate docstring names the exact generating command.
- **Related:** RVL-033.

### RVL-011 — κ=1 adoption verdict rests on 3 seeds with sign flips
- **Severity:** MEDIUM · **Confidence:** PLAUSIBLE · **Status:** OPEN
- **Location:** `gh4_kappa1_conditional.py:138-163`
- **Evidence (single static pass):** per-seed cost ratios flip sign across
  seeds {5,11,23} while the printed VERDICT claims a clean NEGATIVE; the
  committed acceptance criterion in the gate docs should be checked against
  what the code actually asserts.
- **Impact:** a publishable claim ("κ=1 conditional gives no cost win") may be
  noise-dominated at this seed count. **Fix plan:** re-run with ≥10 seeds or
  state the uncertainty in the verdict line. **Acceptance test:** verdict text
  reports a CI, not a point estimate.
- **Related:** ROADMAP D-entries on κ=1 adoption.

### RVL-012 — default ν sits exactly on the validated ceiling, unguarded
- **Severity:** MEDIUM · **Confidence:** PLAUSIBLE · **Status:** OPEN
- **Location:** `rough_heston.py:82-84` (PARAMS, "VALIDATED CEILING" comment)
- **Evidence (single static pass):** docstrings document ν ≤ 0.20 as the
  validated range; the shipped default is ν = 0.20 and no runtime warning
  fires above it.
- **Impact:** a user nudging ν upward gets silently unvalidated output (β
  collapse at ν≥0.25 is documented behaviour). **Fix plan:** `warnings.warn`
  when ν > 0.20 in `rough_heston_paths` / lifted equivalents (the lifted path
  is validated to 0.40 — different threshold). **Acceptance test:** pytest
  `pytest.warns` at ν=0.30 explicit scheme, no warning at 0.15.
- **Related:** RVL-005.

### RVL-013 / RVL-014 — execution-alpha Phase-1 schedule edge cases
- **Severity:** MEDIUM · **Confidence:** PLAUSIBLE · **Status:** OPEN
- **Location:** `execution_alpha_phase1.py:62` (mask indexing), `:67`
  (final-step liquidation vs clip).
- **Evidence (single static pass):** (13) `phi[pos]` with
  `dh_ac[pos]/x_ac[:-1][pos]` may misalign if shapes differ off-by-one;
  (14) the forced `g=1.0` at the last step can be overridden by
  `clip(phi*(1+theta*z),0,1)` when `1+theta*z<0`, leaving inventory uncleared.
- **Impact:** Phase-1 produced a *negative* result, so an edge-case bug here
  would strengthen-or-weaken an already-null finding rather than flip a
  positive one — but the kill-switch numbers (~5 s.e. worse than AC) deserve a
  clean implementation. **Fix plan:** assert shapes before masking; apply the
  final-step override AFTER the clip. **Acceptance test:** unit test with an
  adversarial θ·z < −1 path must end with zero inventory.
- **Related:** docs/gate_checks/execution_rl_gate_check.md.

### RVL-015 — catch-all exception handler masks estimator crashes
- **Severity:** MEDIUM · **Confidence:** PLAUSIBLE · **Status:** OPEN
- **Location:** `estimate_h.py:164-171` (`_run_one`)
- **Evidence (single static pass):** any Exception from an estimator becomes
  NaN + generic note; degenerate-data NaNs (intended) and genuine bugs
  (unintended) are indistinguishable downstream.
- **Impact:** debugging cost + a real estimator bug could masquerade as
  "non-identified data". **Fix plan:** catch the specific numeric exceptions
  (FloatingPointError, LinAlgError, ValueError) and log the traceback at
  debug level; let others raise. **Acceptance test:** an injected TypeError
  inside an estimator propagates.
- **Related:** RVL-003.

---

## LOW

### RVL-016 — docstring restricts H ∈ (0,½)
`layer1_rough_vol.py:200`. VERIFIED (read this session). The kernel formula
works on (0,1) (core allows it, and Layer 1c needs H=0.5+ nulls). **Fix:**
docstring edit alongside RVL-001. **Acceptance:** doc states (0,1) with the
H=½ special case noted.

### RVL-017 — Layer-1 header doesn't warn about L1-1
`layer1_rough_vol.py:1-19`. VERIFIED (roughvol_core:17-24 documents the bug;
the buggy module itself doesn't). **Fix:** add a header warning + pointer to
`roughvol_core.py` for correctness-critical use. **Acceptance:** header names
L1-1/RVL-001 and the trusted alternative.

### RVL-018 — test name promises positivity, asserts ordering
`test_layer1c.py:31-41`. VERIFIED (read this session). The docstring honestly
states the ordering contract ("If this ordering ever reverses…"), so only the
*name* overpromises. **Fix:** rename to
`test_gjr_bias_ordering_grows_as_H_shrinks` or add
`assert biases[0.05] > 0`. **Acceptance:** name matches assertions.

### RVL-019 — CITATION.cff stale
`CITATION.cff:32` (date 2026-06-12, version 0.1.0-research) vs ROADMAP now at
D44+/2026-07-01. VERIFIED (static). **Fix:** bump `date-released` when the
next tagged release/paper ships; keep version until a real tag. **Acceptance:**
date matches the tagged commit at release time.

### RVL-020 — "78 passed" vs "87 tests" (docs)
`run_phaseb.md:27` says expect 78; README's Phase-B table sums differently
(66+21=87). UNCERTAIN — likely different scopes (runbook counts the 4 pipeline
modules; README adds interpret_h). **Fix:** state the scope next to each
number. **Acceptance:** both docs name their file lists; a fresh run matches.

### RVL-021 — EXP print labels out of order
`layer4_calibrate.py:247,261,274`: prints "EXP 1/3/2" (docstring order is
cf → ident → noise → lift). PLAUSIBLE. **Fix:** renumber the prints.

### RVL-022 — duplicated `_http_get`
`binance_data.py` vs `deribit_surface.py`: same helper, different error
handling (404 handling differs). PLAUSIBLE. **Fix:** either unify in a tiny
shared helper or add a one-line comment in each noting the intentional
difference. (The repo's flat-module style may prefer the comment.)

### RVL-023 — module_map.py hardcodes status pills
Regeneration discipline: statuses are edited by hand; the flagship
`roughvollab_module_map.png` can silently go stale (Layer-3/Layer-4 statuses
have changed since some regenerations). PLAUSIBLE. **Fix:** regenerate after
every status-changing commit (add to the ROADMAP session checklist), or derive
statuses from the ROADMAP status board.

### RVL-024 — the honest-negative headline is print-only
`layer1b_mlmc_asian.py:749`: the module's central finding (naive MLMC loses)
is absent from the module docstring. PLAUSIBLE. **Fix:** one sentence in the
header. (README/ROADMAP already carry it.)

### RVL-025 — κ=1 README phrasing vs NotImplementedError
`layer1b_mlmc_asian.py:387-390` raises a clean, well-messaged
`NotImplementedError` for κ=1 at l>0 — VERIFIED this session, and judged
**works-as-designed** (κ=1 was adopted for the single-grid conditional
estimator only; the message points at the design docs). The only issue is the
README table row "opt-in antithetic / conditional / κ=1 estimator flags",
which could read as "κ=1 works in MLMC". **Fix:** README cell append
"(κ=1: fine path/single-grid only)". **Acceptance:** README matches the guard.

### RVL-026 — N_riccati=2000 pinned without in-test rationale
`test_rough_heston_cf.py:136`. PLAUSIBLE. **Fix:** comment citing the
convergence table (module `__main__` output) next to the constant.

### RVL-027 — SOE (Bayer–Breneis) tuning constants uncited
`rough_kernel_soe.py:68-95` (α=1.6, β=0.4275 "chosen empirically").
PLAUSIBLE. **Fix:** cite the ablation (or run + commit a small one) in the
docstring; parametrise with documented defaults.

### RVL-028 — no lifted-scheme non-anticipation test
`test_rough_heston.py:95-107` has the causality test for the explicit scheme;
the lifted simulator has no equivalent. PLAUSIBLE. **Fix:** parametrise the
test over both simulators.

### RVL-029 — undocumented magic thresholds in the H pipeline
`interpret_h.py:98` (`_FLAT_SLOPE=0.25`), `estimate_h.py:94`
(`_MF_QS=[-4,-2,2,4]`). PLAUSIBLE. **Fix:** docstring the derivation/choice;
expose as parameters with defaults.

### RVL-030 — per-maturity N_riccati step pinned empirically
`calibrate_btc.py:69` (h=1.1e-4; overflow ceiling N=8200 at T≈1). PLAUSIBLE
(the D41 ROADMAP entry documents the choice; the code doesn't). **Fix:**
comment linking D41 + the precheck that validates it per-run (which exists:
`precheck()`).

### RVL-031 — de-bias recovery untested in the collapse zone
`test_interpret_h.py:127` uses window 576 (clean regime) only. PLAUSIBLE.
**Fix:** add a collapse-zone case asserting the honest outcome
(multivalued/collapsed classification), not recovery.

### RVL-032 — layer-3 minor robustness items
`layer3_deep_hedging.py:235` (torch device handling in `pnl_from_deltas`),
`:338` (η=0 ignores H — benign GBM limit), `:354` (gate tolerance 0.06
hardcoded). PLAUSIBLE — cannot runtime-verify here (torch not installed).
**Fix:** device-normalise inputs; docstring the η=0 semantics; name the
tolerance's origin.

### RVL-033 — two coexisting H=0.10 baselines
`p2_baseline_regen.py:78` (seed-23 L=5 N=12k vs seed-7 L=6 N=20k). PLAUSIBLE.
**Fix:** one line declaring which is canonical for the paper numbers.

### RVL-034 — data-layer invariant fragilities (documented, latent)
All PLAUSIBLE, all currently safe: `deribit_surface.py:186` (online fetch
without snapshot pins T to run-clock), `deribit_surface.py:224` (vega-floor
computation is filter-order dependent), `kline_verifier.py:219` (2^53
timestamp guard only at load), `rv_series.py:307` (silent thinning of
misaligned bars if the verifier is bypassed). **Fix:** assertions/comments at
the named sites. These document invariants the current pipeline already
respects.

### RVL-035 — torch skips reduce coverage visibility
`test_layer3_deep_hedging.py:18`: on this machine ALL torch tests skip
(torch not installed; Python 3.14 wheel availability unverified). VERIFIED in
the baseline (2 skipped). **Fix:** when working on Layer 3, create
`.venv-layer3` per `requirements-layer3.txt` (check torch cp314 Windows wheels
first) and run its suite there; CI note in README.

---

## Refuted candidates (do not re-report)

| Candidate | Why refuted | Evidence |
|-----------|-------------|----------|
| `p2_antithetic_verify.py:69` — "antithetic coupling never computes `dW2_s`; coupling only on dW1" (was rated critical) | **False.** Lines 69-71 compute BOTH `dW1_s` and `dW2_s` (pairwise swap) and pass both to `_paths_from_increments(dW1_s, dW2_s, …)`. The published antithetic-refuted conclusion (D20-D23) stands on a correctly-built estimator. | Direct read, 2026-07-02 |
| `layer1b_mlmc_asian.py:362` — "antithetic coarse-path invariance is an undocumented subtlety" | The docstring at :358-363 documents it explicitly ("the coarse increment (their sum) is invariant, so P_c and the coupling are unchanged"). | Direct read, 2026-07-02 |
| "`layer1c_roughness_audit.py` has NO dedicated test file" (was rated critical by a static pass) | `test_layer1c.py` IS its dedicated test file — it imports `gjr_hurst`, `pvariation_hurst`, `mfdfa_hurst`, `realized_log_variance`, tolerance constants, etc. directly from the module. | Import list read, 2026-07-02 |
| "README figures `docs/guide/roughvollab_map_v2.png` / `roughvollab_module_map.png` may not exist" | Both files exist at the referenced paths. | File-system check, 2026-07-02 |
| "`test_layer4_convergence.py` does not exist / module untested" | The file exists (root, 4.7 kB) and its 8 tests ran in the baseline. | File listing + baseline run, 2026-07-02 |

---

*Generated 2026-07-02 by an adversarial audit session (multi-agent deep read →
refute-first verification → this report). Baseline suite re-run after all
documentation deliverables were added: expected unchanged (docs-only session).*
