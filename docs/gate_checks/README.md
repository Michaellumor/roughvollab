# Gate-check index — Layer 1b estimators (P2) and the κ=1 hybrid

This is an **index**, not a rewrite of the specs. For each gate-check thread it
records *why* the piece was built, *what* it had to satisfy (gate IDs), the
one-line verdict with its headline number, and the seeds that produced it.

- **Engine is read-only here.** All estimators are opt-in flags in
  `layer1b_mlmc_asian.py` (`antithetic=`, `conditional=`, `kappa=`); the κ=0
  naive path is and stays the default. The κ=1 coarse coupler lives in the
  separate module `layer1b_kappa1.py` and is **not** wired into the production
  default. Drivers are left in the repo root (run from there).
- **Verdicts/numbers below are from this session's driver runs** (re-runnable
  via the listed commands). Where a number is not in a file readable here it is
  marked “see spec”.
- Spec files that were only ever given in chat are listed as **“chat-only — to
  add”** in the final section; do not treat their absence as “no spec”.

## Index

| Thread | Spec | Driver(s) | Gate IDs | Verdict (headline) | Seeds |
|---|---|---|---|---|---|
| Antithetic coupling | `p2_antithetic_build_and_verify.md` §1 (build-and-verify doc — to be added) | `p2_antithetic_gatecheck.py` (via production), `p2_antithetic_verify.py` (standalone) | G-A1, G-A2, G-A3, G-A4 | **REFUTED** — β unchanged ≈2H (not 4H); variance factor ≈1.44×; **~9–11% costlier** at matched L (eff ≈0.91×) | 7 / 23 / 11 |
| Conditional MC (geometric control variate) | chat-only (`p2_conditional_gate_check.md`, `p2_conditional_build_and_verify.md`) | `p2_conditional_verify.py` | G-C4 (+ variance verdicts V1/V2, unbiasedness) | **CONFIRMED** — conditional *standard* MC cheapest; conditional MLMC does **not** beat it, ratio **0.41–0.45** | 11 / 99 / 1234 |
| κ=1 fine path | chat-only (`gh1_kappa1_fine_path_spec.md`) | `gh1_kappa1_finepath.py` (engine: `kappa=1` in `layer1b_mlmc_asian.py`) | G-H1a, G-H1b, G-H1c, G-H1d | **PASS** — variance gap closed **0.853 → 0.9996** (analytic; 0.9973 emp.); compensator unbiased; BS z=0.64 | fixed internal (1/2/7/100/999) |
| κ=1 coarse coupler | `kappa1_hybrid_coupling_design.md` (this dir) | `gh2_kappa1_coupler.py` (gate), `kappa1_coupling_design_check.py` (covariance de-risk), `layer1b_kappa1.py` (coupler engine) | G-H2a, G-H2b, G-H2c | **PASS** — coupling tightness **0.001 vs 0.39–0.58 (552× separation)**; telescoping <1; β≈2H | 11 |
| κ=1 adoption | chat-only (`gh4_kappa1_adoption_spec.md`) | `gh4_kappa1_conditional.py` (2a), `gh4_kappa1_cost.py` (2b) | G-H4 step 2a, step 2b | **ADOPT for conditional-std-MC only** — ~1.3–1.5× cheaper (k1/k0 = 0.79× @ε=0.05, 0.68× @ε=0.025); **not** for MLMC | 5 / 11 / 23 (+proxy 999) |
| Baseline regeneration | n/a (regeneration driver, not a gate) | `p2_baseline_regen.py` | n/a | β-sweep reproduces **0.120/0.219/0.418/0.726 bit-for-bit**; costratio 0.63 (<1) | 7 / 11 / 23 (+1/2) |

## Thread detail

### Antithetic coupling — REFUTED (documented negative result)
- **Why:** test whether a Giles–Szpruch antithetic swap repairs the β<γ
  pathology of naive MLMC under rough vol.
- **Gates:** G-A1 bias-free (E[V_t]=ξ₀; telescoping consistency <0.51) · G-A2
  rate (β vs 2H over H∈{0.05,0.10,0.20,0.35}) · G-A3 variance factor · G-A4 cost.
- **Result:** β identical to naive and tracks 2H (0.120/0.219/0.418/0.726);
  variance factor ≈1.44× < cost factor 2.5/1.5 ⇒ net worse at matched finest
  level. The free-running adaptive driver’s apparent “win” is an L-selection
  artifact (flips sign across seeds).
- **Run:** `python p2_antithetic_gatecheck.py` · seeds 7 (estimate_rates), 23
  (H-sweep), 11 (adaptive). Writeup: `../p2_estimator_results.md`.

### Conditional MC (geometric control variate) — CONFIRMED
- **Why:** P2’s second estimator; P_cond = arith − (geom − E[geom|W]), W the
  variance path; E[geom|W] closed-form lognormal.
- **Gates:** unbiasedness (z≈1.5) · single-level variance ↓ ~4.2× (V1) ·
  level-diff variance ↓ ~3.2× (V2) · G-C4 matched-accuracy cost.
- **Result:** conditioning helps single-level more than the level difference, so
  conditional **standard** MC at the finest grid is cheapest; conditional MLMC
  cost/cond-std-MC ratio 0.41–0.45 (<1, stable, κ-invariant).
- **Run:** `python p2_conditional_verify.py` · seeds 11/99/1234.

### κ=1 fine path — PASS (G-H1)
- **Why:** the BLP hybrid integrates the nearest (singular) kernel cell exactly;
  closes the kappa=0 Volterra variance gap. Opt-in `kappa=1` flag, fine path
  only (coarse coupler deferred).
- **Gates:** G-H1a variance gap (Var(W̃_T) vs continuum, swept over H) · G-H1b
  forward variance (κ=1 compensator; the wrong κ=0 compensator gives a 0.19
  systematic bias — the trap) · G-H1c BS anchor (η=0) · G-H1d near-cell law.
- **Result:** variance gap 0.853 → 0.9996 (analytic), 0.9973 empirical, within
  1% of continuum for all H; forward variance unbiased (z=2.86 vs trap z=64);
  BS z=0.64; near-cell Var/Cov within 0.05–0.67%.
- **Run:** `python gh1_kappa1_finepath.py`.

### κ=1 coarse coupler — PASS (G-H2)
- **Why:** exact MLMC coupling of the κ=1 nearest-cell Gaussian across levels
  (split + conditional resampling). Spec/design + gate plan in
  `kappa1_hybrid_coupling_design.md` (this dir).
- **Gates:** G-H2a unbiased (telescoping consistency) · G-H2b coupling tightness
  (the assert that a marginal check is blind to) · G-H2c rate β≈2H.
- **Result:** per-cell covariance validated to 1e-11
  (`kappa1_coupling_design_check.py`); coupling tightness 0.001 (correct) vs
  0.39–0.58 (deliberately sub-cell-swapped) = 552× separation; consistency
  0.389; β=0.186.
- **Run:** `python gh2_kappa1_coupler.py` · `python kappa1_coupling_design_check.py`
  · seed 11.

### κ=1 adoption (G-H4) — ADOPT for conditional-std-MC only
- **Why:** does κ=1’s smaller weak-error buy a coarser bias-grid that repays its
  per-path overhead, for the conditional standard-MC estimator (P2’s winner)?
- **Gates:** step 2a — bias ratio (~0.5–0.7× at the grids that matter), variance
  ratio (1.13×), gate = (grid ≥1 level coarser, stable across seeds) ∧ (Var
  ratio <1.6). step 2b — matched-accuracy cost with empirically-timed per-path
  cost (overhead measured ~1.08×, not assumed).
- **Result:** gate passes (n* halves 32→16, 64→32, identical across all seeds);
  cost ratio k1/k0 = 0.79× (ε=0.05) and 0.68× (ε=0.025), stable. κ=1 is **not**
  worth it for MLMC (β unchanged, larger level variance — see G-H2c) but **is**
  worth it as the variance-path scheme for conditional standard MC.
- **Run:** `python gh4_kappa1_conditional.py` (2a) → `python gh4_kappa1_cost.py`
  (2b) · seeds 5/11/23.

### Baseline regeneration — provenance for p2_paper.tex
- **Why:** regenerate the κ=0 naive baseline macros/table on this machine,
  through the production functions, to replace sandbox placeholders.
- **Result:** β-sweep reproduces 0.120/0.219/0.418/0.726 bit-for-bit;
  validation (Var(W̃_T)=0.853, BS z=0.77, telescoping 0.386); cost table
  L=[2,3,5,7] monotone, std-MC/MLMC ratio 0.63 at ε=0.025; Asian price 4.2143.
- **Run:** `python p2_baseline_regen.py` · seeds 7/11/23 (+1/2 validation).

## Specs referenced but not in the repo (chat-only — to add)

These were provided as inline specs in chat; the drivers reconstruct them. Drop
the .md into `docs/gate_checks/` to complete the audit trail:

- `p2_antithetic_build_and_verify.md` — antithetic build **and the gate spec
  (§1)** for G-A1..G-A4 (the original was in Downloads, not the repo).
- `p2_coupling_gate_check.md` — a dead filename the build doc's prompt refers
  to; **not a real file**. The antithetic gate spec lives in
  `p2_antithetic_build_and_verify.md` §1, not a separate file.
- `p2_conditional_gate_check.md`, `p2_conditional_build_and_verify.md` —
  conditional MC build + G-C gates.
- `gh1_kappa1_fine_path_spec.md` — κ=1 fine-path build + G-H1a..d.
- `gh4_kappa1_adoption_spec.md` — κ=1 adoption G-H4 (step 2a/2b).

## Related files (not gate specs)
- `../p2_estimator_results.md` — paper-ready writeup for the antithetic +
  conditional threads; relocated to `docs/` (a result artifact, not a gate spec).
- `../../layer2_piece1_gate_check.md` — belongs to a **separate Layer 2
  workstream** (Almgren–Chriss execution), not these P2/κ=1 gate-checks; left
  exactly where it is.
