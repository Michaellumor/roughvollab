# Gate-check spec — Layer 4: weak convergence of the κ=0 hybrid scheme

**Status:** spec (not yet built) · **Module:** `layer4_convergence.py` + `rough_heston.py` (both planned)
**Date:** 2026-06-27 · **Depends on:** `roughvol_core.py` (κ=0 Volterra machinery, reused) · builds a native rough-Heston simulator + CF reference

> Format follows the project gate-check discipline: **state the mechanism → commit a
> falsifiable prediction *before* measuring → validate against a known answer.** Negative
> results are first-class. Comparisons are pinned to a non-circular reference so the study
> cannot measure its own self-consistency and call it convergence.

---

## 0. What is already settled (and therefore NOT what this study does)

The **strong** convergence order of the κ=0 hybrid scheme is **established and empirically
corroborated** — it is *not* an open question:

- **Stated:** strong rate `O(n^{−H})` for the optimal-discretisation Volterra scheme
  (Bennedsen–Lunde–Pakkanen 2017), recorded in `layer1b_mlmc_asian.py`.
- **Measured (tight):** the Giles level-variance decay `β = 2 × (strong order) = 2H` was
  measured against the exact κ=0 MLMC coupling and tracks 2H across H ∈ {0.05, 0.10, 0.20, 0.35}
  with equality (the pathwise bound is *achieved*, not merely bounded).

Re-measuring the strong order would be redundant. **Layer 4 measures the WEAK order**, which the
repo's own notes flag as unmeasured ("α unmeasurable at this N"; a large-N weak-rate study is
listed as a pending extension). The weak order is the bias in `E[payoff]` — the quantity that
actually governs pricing accuracy — and it is the genuine open question.

---

## 1. Mechanism

The κ=0 hybrid scheme discretises the rough Volterra process on an n-step grid. Two errors:
- **Strong (pathwise) error** — order `H` — settled (above).
- **Weak error** — the bias `|E[g(S_n^Δ)] − E[g(S_true)]|` in a payoff functional `g`. Weak error
  typically converges *faster* than strong error (cancellation of pathwise fluctuations in
  expectation), but for this scheme under roughness the weak rate **α is unknown**.

The study refines Δt (n = 2^k steps) and measures how fast the *priced* bias decays — i.e. α.

---

## 2. The non-circularity problem and its solution (load-bearing)

Weak error requires a **known true price** to measure bias against. A "convergence study" that
compares coarse simulations against a *fine simulation* is **circular** — it measures
self-consistency, not convergence to truth. This study therefore measures bias against a
**simulation-free semi-analytic reference**:

**Reference = the rough-Heston characteristic function (El Euch–Rosenbaum 2019).** Rough Heston
admits a characteristic function via a fractional Riccati equation, so European option prices are
obtained by **Fourier inversion without any simulation** — a genuine known-truth. This same CF
machinery is the **SPX calibration tool** (Section 5), so one component serves both roles.

**Sanity anchor:** at **H = ½** the scheme reduces to classical Heston, where the closed form is
unambiguous — a gate-check point the measured prices must match exactly (cf. the Markovian-limit
gate G-X1 used in the execution arc).

### 2a. Same model on both sides — native rough-Heston simulator (chosen approach)

The engine `roughvol_core.py` simulates rough **Bergomi**; the CF reference is rough **Heston**.
Rather than test the Bergomi scheme against a Heston reference (which would require a carry-over
assumption that the convergence order is model-independent), Layer 4 **builds a native rough-Heston
simulator** and measures *its* weak order against *its own* CF reference — **same model on both
sides, no carry-over assumption to defend.** This is the rigorous choice.

**Consequence (scope):** Layer 4 is therefore **two deliverables**, not one — a new rough-Heston
simulator **and** the convergence study built on it. The simulator is not throwaway scaffolding:
it is the same engine the SPX calibration uses (§5), and rough Heston (El Euch–Rosenbaum's affine
structure) is a reusable capability for the lab beyond this study. The rough-Heston variance
process is itself the rough Volterra object (CIR-style, driven by the same `H−½` kernel), so the
simulator reuses `roughvol_core.py`'s κ=0 Volterra machinery for the kernel discretisation but
adds the variance/price dynamics specific to rough Heston — a sibling engine, not a trivial port.

**The shared κ=0 discretisation means the strong-order result (O(n^{−H}), β = 2H) is expected to
carry to the rough-Heston simulator** — but this is now a *checkable prediction within Layer 4*
(re-measure β on the new simulator as a build-validation gate), not an untested assumption.

---

## 3. The committed prediction (falsifiable, set BEFORE measuring)

**The precise theoretical weak order α for this κ=0 variant is NOT sourced** (the repo does not
state it; it must be derived from BLP/El Euch–Rosenbaum or measured). This spec therefore commits
a **directional** claim rather than inventing a number:

> **Prediction.** The weak order satisfies **α > H** (weak converges strictly faster than the
> strong order H), with **expectation α ≈ 1** for a smooth payoff (the classical weak rate, if
> roughness does not bottleneck the bias).

**Refutation / outcomes (all publishable):**
- **PASS:** measured α is significantly **> H** and consistent with the ≈ 1 expectation → the
  scheme prices accurately faster than its pathwise error suggests (the useful, expected result).
- **PARTIAL:** α is **> H but < 1** (e.g. roughness-limited weak rate) → a quantified
  roughness penalty on the weak order — a real finding.
- **FAIL (and the most interesting):** α ≈ H (weak ≈ strong, no speed-up) → the roughness
  bottleneck hits the *weak* rate too, contradicting the classical expectation. Non-obvious and
  publishable.

**To-be-verified before commit:** check BLP 2017 / El Euch–Rosenbaum 2019 for a stated weak order
of the hybrid scheme; if a sourced value exists, replace the directional claim with the precise
predicted α (the strong order was pinned to BLP this way — the weak order deserves the same).

---

## 4. Method

1. Choose a smooth European payoff `g` (call) with a CF reference price `P_true` (Fourier
   inversion of the rough-Heston CF) at fixed (H, ξ₀, η, ρ, T, K).
2. For n = 2^k, k = k_min … k_max: simulate `M` paths (M large — target ≥ 10⁶ per level per the
   pending large-N extension), price `P_n = mean(g)`, record bias `b_n = |P_n − P_true|`.
   - Monte Carlo error must be **driven below the discretisation bias** at every n (else the slope
     measures MC noise, not weak order) — use a fixed common random-number stream and report the
     MC standard error alongside b_n; only fit where b_n ≫ MC s.e.
3. Fit `log b_n` vs `log Δt`; the slope is the measured weak order **α** (with CI).
4. Repeat across an **H-sweep** (H ∈ {0.05, 0.10, 0.20, 0.35, 0.50}) to characterise α(H) and
   to hit the H = ½ closed-form anchor.

---

## 5. SPX calibration (relevance, not a separate exercise)

Calibrate rough-Heston (H, ξ₀, η, ρ) to an **SPX option surface** via the CF, then **run the
convergence study at the calibrated parameters** — answering *"does the scheme converge at the
predicted weak rate in the regime real markets actually occupy?"* This connects Layer 4 to the
identifiability finding: SPX calibration forces **high vol-of-vol** (η large), which is plausibly
where the weak rate is **slowest** and the bias constant **largest**. So the SPX run is not a
formality — it tests the scheme in the hardest, most realistic regime, and an honest caveat
(calibration mixes real risk-premia/term-structure effects) is recorded.

---

## 6. Diagnostics

- **Primary:** log-log bias-vs-Δt plot, fitted slope α + confidence interval, MC-s.e. band
  overlaid (to evidence that the fit region is bias-dominated, not noise-dominated).
- **H-sweep panel:** α(H) across the spectrum, with the strong-order line (H) drawn for contrast —
  the visual demonstration of whether weak beats strong, and by how much.
- **Richardson extrapolation check:** confirms the fitted order is self-consistent and yields a
  higher-accuracy reference price as a cross-check on the CF.
- **H = ½ gate:** measured price must match the closed-form Heston value to within MC error.
- **Reproducibility harness:** seeded, one-command regeneration of every figure and number
  (mirrors `paper_outputs.py`) — the convergence claims must be reproducible by `python
  layer4_convergence.py`.

---

## 7. Gate-check bar (PASS/FAIL committed before any measurement)

> **PASS** iff: (i) the H = ½ case matches the closed-form Heston reference within MC error;
> (ii) the fit region is bias-dominated (b_n ≫ MC s.e.); (iii) the measured weak order α is
> significantly **> H** across the H-sweep; and (iv) α is consistent with the stated expectation
> (≈ 1, or the sourced value if §3 verification supplies one).
> **FAIL** → α ≈ H (no weak speed-up) or α inconsistent with prediction. **A FAIL is a result**,
> not a defect: it would establish that roughness bottlenecks the weak rate of the hybrid scheme —
> a publishable characterisation of the scheme under roughness.

---

## 8. Scope and honest boundaries

- Measures the **κ=0** scheme. κ=1 (`layer1b_kappa1.py`) improves the error *constant*, not the
  rate (β = 2H either way), so the *order* result is κ-invariant; a κ=1 run would sharpen the
  constant only.
- Layer 4 is **two deliverables**: a native **rough-Heston simulator** (reusing `roughvol_core.py`'s
  κ=0 Volterra machinery, adding rough-Heston variance/price dynamics) and the convergence study
  built on it. The simulator carries a **build-validation gate**: re-measure β on it and confirm
  β = 2H (the shared κ=0 discretisation should reproduce the established strong order) *before*
  trusting it for the weak-order study. Same model on both sides → **no carry-over assumption**.
- Weak order is measured for a **smooth** payoff first; non-smooth payoffs (digitals, barriers)
  typically degrade the weak rate and are a deliberate **later** extension, not part of the
  first falsifiable claim (no fishing across payoffs to find a flattering α).
- Independent of the unbuilt **Layer 3** (deep-hedging engine): this is a Layer-1-numerics
  capstone and proceeds now.

---

## 9. Open item before build

Run the §3 verification: does BLP 2017 or El Euch–Rosenbaum 2019 state a **weak** convergence
order for the hybrid scheme? If yes → commit that precise α and tighten the gate bar. If no →
proceed with the directional claim (α > H, expect ≈ 1) and let the study *measure* α as the
contribution.
