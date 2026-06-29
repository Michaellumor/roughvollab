# Gate-check spec — Layer 4: weak convergence of the κ=0 hybrid scheme

**Status:** `rough_heston.py` simulator **built & validated for ν ≤ 0.20** (QE positivity; β=2H gate PASS, 2026-06-27) · `rough_heston_cf.py` CF reference **built & certified** (D30) · `layer4_convergence.py` weak-order study **BUILT & MEASURED** (α ≫ H banked; PARTIAL@H=0.05, borderline@H=0.10, PASS@H=0.20 — D31, 2026-06-27) · **Module:** `layer4_convergence.py` + `rough_heston.py` + `rough_heston_cf.py`
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

> **Status — BUILT & CERTIFIED (2026-06-27, `rough_heston_cf.py`).** The CF reference exists:
> El Euch–Rosenbaum fractional Riccati (Diethelm–Ford–Freed FABM) + Gil-Pelaez inversion, with an
> Albrecher little-trap classical-Heston CF as the independent anchor. Two-stage validated (inverter
> vs Black–Scholes to 1e-13; H=½ pipeline == closed-form Heston to ~1e-6). **Certified** at the
> rough regime (H=0.10, ν=0.20): convergence **order ≈ 1.60 = 1+H+½**, `err ≈ 0.68·(T/N_riccati)^1.60`;
> to reach reference error ≤ X use N_riccati ≥ {1e-5: 1040, 1e-6: 4370, 1e-7: 18400}. Cost O(N²) per
> CF-set (fine at ν≤0.20; sum-of-exp acceleration needed only for ≤1e-7 refs or calibration loops).
> See ROADMAP **D30**.

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

**Validated regime (2026-06-27).** α is measured for **ν ≤ 0.20** — the range where the
rough-Heston simulator is validated (β=2H holds; the priced bias is scheme-independent). At
ν ≳ 0.25 the explicit scheme degrades on *both* axes (§5; §8 boundary finding), so α is **not**
claimed there; that regime needs the multifactor-lift extension.

**MEASURED (2026-06-27, D31) — the prediction is now resolved.** The committed directional claim
is confirmed and refined against the CF reference (OTM call K=110, ν=0.20; dual estimators —
absolute `|E[Pₙ]−P_CF|` and CRN-coupled — gated to agree, plus Romano–Touzi conditional MC for
variance reduction):

- **α ≫ H at every H — robust and banked** across both estimators and all fit windows: the strong
  order O(n^{−H}) does **not** bound the weak/pricing rate (the headline result).
- **PASS @ H = 0.20** — α ≈ 1.0 ≈ 1 (weak converges classically; reproducible `--sweep`
  prec-weighted 1.01).
- **PARTIAL @ H = 0.05** — α ≈ 0.74, robustly **< 1** (8σ on the tight absolute estimator,
  window-stable 0.74–0.77): a genuine roughness penalty on the *weak* rate at the rough end.
- **Borderline-pending-finer-grids @ H = 0.10** — α ≈ 0.84–0.95, leans < 1 but **not
  distinguishable from 1**; the spread is ~±0.1 *systematic* (pre-asymptotic window-sensitivity,
  not statistical), so more paths at these coarse grids will not resolve it — finer grids (the §5
  multifactor lift) would.

So the §3 outcome is **PASS for H ≥ 0.20, PARTIAL at H = 0.05, borderline mid-range — α > H
everywhere, no FAIL.** The H=½ anchor recovered ≈0.89/0.94 ATM, ≈1.14/1.09 OTM (inside the
literature range [0.6, 1] at ν_F = 0.60, arXiv:2106.10926). Full record + caveats: ROADMAP **D31**.

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

## 5. SPX calibration — in-range validation + an explicit out-of-range EXTENSION

**Measured boundary (2026-06-27).** The explicit hybrid Volterra–Euler simulator is validated only
for **ν ≤ 0.20** (`rough_heston.py`; §8 boundary finding). Real rough-Heston **SPX/index calibration
forces high vol-of-vol (ν ≈ 0.3–0.4)** — *above* that ceiling — where (i) the β-coupling breaks and
(ii) the priced bias becomes scheme- and n-dependent (2.4–5.3% at ν=0.4). So the SPX run **cannot**
be done with this simulator without contaminating the weak-order measurement. This splits §5 in two:

- **In range (ν ≤ 0.20):** the weak-order study runs and α is a legitimate **method-validation**
  result in a stylised / moderate-vol-of-vol regime. This is what Layer 4's first brick delivers.
- **Out of range (high-ν SPX) — explicit Layer 4 EXTENSION:** the **market-relevant** SPX-calibrated
  weak-order claim requires a **multifactor Markovian-lift simulator** (Abi Jaber–El Euch: rough
  kernel ≈ sum of exponentials → Markovian factors + per-factor QE/Alfonsi), which holds positivity
  *and* the coupling at high ν. It is **its own brick with its own build-validation gate** — scoped
  here as a first-class extension, NOT a silent caveat. The honest framing: until that extension
  exists, the market-relevant claim is open.

**STATUS UPDATE (2026-06-28, D32) — the lift's kernel foundation is BUILT (brick 4a).**
`rough_kernel_soe.py` delivers the sum-of-exponentials kernel approximation K(t) ≈ Σ wᵢ e^(−γᵢ t)
that underpins the Markovian lift, with both literature constructions source-pinned (AJ–EE
arXiv:1801.10359; Bayer–Breneis arXiv:2108.05048) and **selected on evidence** via Gate A (closed-form
L² kernel error, simulation-free): AJ–EE realizes its pinned rate n^(−4H/5) *exactly* but the uniform
mesh is impractical; **Bayer–Breneis (superpolynomial exp(−c√N)) is SELECTED.** Conservative
**N_factors(rel-L² ≤ 1e-3) ≈ 130 (H=0.20), 250–520 (H=0.10), 512–1024 (H=0.05)** (global α=1.6, not
per-H tuned → per-H §4.2 tuning only improves it). Economic justification: the lift is **O(N·n)** vs
the Volterra **O(n²)** ⇒ it wins when N<n, i.e. at the fine grids (n ≳ 256–1024) needed to resolve
brick-3's borderline H=0.10. **Still open:** the lifted *simulator* (4b) + its β=2H/H=½ build gates
and high-ν positivity (4b/4c). See ROADMAP **D32**.

**STATUS UPDATE (2026-06-28, D33) — the lifted SIMULATOR is BUILT & GATED (brick 4b).**
`rough_heston_lifted.py` (N OU factors, O(N·n) per path) passes **both** brick-1 gates at ν≤0.20:
GATE C (H=½ price = CF, unbiased) and **GATE B (β=2H): 0.066/0.175/0.397/0.699, max|β−2H|=0.034,
monotone** — reproducing brick-1's own rough-Heston QE β (0.070/0.167/0.384/0.737, D29). Positivity
was selected on evidence (the brick-1 discipline): truncation **collapses** β at H≤0.10/ν=0.20 (near-0
13%/9%), implicit-drift ≈ truncation, naive-Alfonsi is GATE-C-biased — the validated scheme is
**brick-1 Andersen QE *ported* to the aggregate** (with an effective increment so the factors still
reconstruct the QE'd V). The β over-steepening was diagnosed as a *truncation artifact* (gone under
QE). Rate needs only ~tens of factors. **Still open:** high-ν (ν≈0.3–0.4, GATE D), the weak-order
re-run on the lift (resolving H=0.10), SPX calibration — brick 4c+. See ROADMAP **D33**.

**STATUS UPDATE (2026-06-28, D34) — high-ν GATE D done (brick 4c): (B) PARTIAL, a SPLIT boundary.**
Ran Gate D at ν∈{0.25,0.30,0.40}, H∈{0.05,0.10,0.20} with the brick-2 CF as a *high-ν known-answer*
(exact at any ν — verified). **PRICING WIN (headline):** lifted-qe prices converge to the CF —
**<0.5% at H≥0.10 to ν=0.40** (qe−CF→0 as n grows; H=0.10: 1.57→0.92→0.48%) — vs the explicit
scheme's 2.4–5.3% scheme-dependent bias; **truncation diverges 2–9%** (plateaus, not converging),
confirming QE is the fix. The high-ν SPX-relevant **pricing boundary is broken to ν≈0.40** (one
caveat: a ~2% residual remains at the extreme H=0.05/ν=0.40 corner — drops 3.44→2.03% with n but
flattens). **β-RATE BOUNDARY:** the MLMC variance-rate β=2H is clean to **ν≤0.30** (max|β−2H| 0.052/
0.056), degrades at ν=0.40 (H=0.20 dev 0.095) under 16–34% near-0; qe still vastly beats trunc (which
collapses to 0.016). **The split (the finding):** price-boundary ν≈0.40, β-rate-boundary ν≤0.30 —
moved *differently*, not forced to (A)/(C). **Still open (4d+):** the H=0.05/ν=0.40 price residual,
the weak-order α re-run on the lift (resolving H=0.10), SPX calibration. See ROADMAP **D34**.

**STATUS UPDATE (2026-06-28, D35) — the lift does NOT preserve the WEAK ORDER (§4/§6); H=0.10 stays open.**
Attempting to resolve brick-3's borderline H=0.10 by re-running the weak-order (α) study on the lift
hit a **known-answer-gate STOP**: Gate 1 (H=½ anchor) passed (lifted a_cpl=1.000), but **Gate 2 FAILED**
— the lifted α is systematically *below* brick-3's explicit α at the resolved cells (H=0.20: 0.74 vs
1.01; H=0.05: 0.40 vs 0.74, coupled-confirmed). An N-sweep shows a_cpl **flat at 0.854 across
N=150→300→600** → the perturbation is **NOT** the SOE-kernel truncation but **intrinsic to the
shared-ΔW noise envelope** (the O(N·n) approximation): fine for β=2H (a rate) and prices (a level,
validated 4b/4c), but it **perturbs the weak ORDER** (a finer quantity), N-independently. Corollary:
the lift's O(N·n) cost win *is* the envelope, which is exactly what perturbs the rate — an exact-noise
lift (O(N²)) might preserve it but forfeits the cost win. **So the cheap lift cannot measure the weak
order; brick-3's H=0.10 borderline REMAINS OPEN** — resolving it needs the **explicit sim at finer n**
(O(n²)) or an **exact-noise lift** (O(N²)), neither cheap. **Explicit path feasibility-probed
(2026-06-28, run-by-name this session):** at n=256 it hits a **memory-vs-SNR vise** — the M×n_max
increment/recursion allocation is **~2–4 GB at M=150k**, where the **n=256 level is noise-dominated**
(absolute b/se≈4.3 → excluded from the fit; coupled Y/se≈2; the two estimators disagree **0.58 vs
1.30** → α **unresolved**), and the **M≳1M** needed to resolve it **exceeds memory (M=300k OOMs)** —
the SAME wall brick-3 hit at n≤128, now confirmed at n=256. So the explicit path needs a
**streaming/chunked harness** (batch paths, accumulate the bias estimators, free per-batch) to break
the M×n_max ceiling and reach M≳1M — a bounded refactor, **not done**; and even then whether α
converges or keeps drifting is unknown. **H=0.10 is therefore unresolvable by the current explicit
harness at feasible memory** — a streaming harness is the prerequisite to even attempt it. (Banked:
the `sim`-callback generalization of the weak-order harness — any simulator now plugs in.) See
ROADMAP **D35**.

**STATUS UPDATE (2026-06-28) — H=0.05/ν=0.40 corner residual is a REAL SCHEME FLOOR, strike-amplified (sharpens D34's open caveat).**
D34 left it ambiguous ("partly discretization plus a small real residual"). An **n=512 price-vs-CF probe
(conditional MC + CRN, M=50k; run by name this session)** resolves it — a **real scheme floor, not
finite-n** (CF verified reliable at this corner: U_max=200==400; K=100→6.42733, K=110→1.96676):
- **ATM (K=100):** qe−CF = **2.81 / 2.53 / 2.19%** at n=128/256/512 — drops only **~10–13%/doubling**
  (pure discretization at α≈0.74 would give ~40%/doubling: 2.81→1.7→1.0); CRN n→2n change −0.29 / −0.34%.
  A **~2% floor**, not vanishing.
- **OTM (K=110): far larger** — **26.1 / 23.2 / 20.75%** at n=128/256/512 (~2.5%/doubling). The lift
  **mis-prices the OTM tail by ~20%** at this corner (CF reliable, conditional MC bias-preserving → real).
- **Mechanism:** the price-level fingerprint of D35's envelope perturbation — the QE/shared-ΔW envelope
  perturbs the variance dynamics; **negligible at H=0.10** (4c: ATM bias → 0.48% at n=256), **material at
  H=0.05 ATM** (~2%), **amplified in the tail-sensitive OTM price** (~20%). One mechanism, strike-dependent
  magnitude.
- **Feasibility:** n=512 was reachable (M=50k, ~1.3 GB) because this is a **price-LEVEL** check — unlike
  D35's weak-order slope-fit, which OOM'd at the same n (the price-vs-slope distinction).
- **Open question this surfaces (for SPX):** 4c validated only **ATM** pricing (clean to ν=0.40 at H≥0.10).
  The **OTM tail at high ν is untested at H≥0.10** — does the tail-amplification affect only the H=0.05
  extreme, or also the **SPX-relevant H≥0.10 OTM wings** at high ν? Matters for smile calibration.

See ROADMAP **D34**.

**STATUS UPDATE (2026-06-28, D36) — OTM-smile gate: lift's high-ν OTM smile validated, SPX-viable on the put wing.**
The OTM-smile gate (SPX prerequisite) compared the lifted vs CF **implied-vol smile** at H≈0.10, ν∈{0.30,0.40}
(one conditional-MC run → analytic BS across strikes; vega-reliability mask). **(B) WING DIVERGENCE,
ASYMMETRIC:** ATM + the entire OTM **put wing are CLEAN** (lifted−CF ≤ ~0.2pp; ATM confirms 4c), while the
OTM **call wing is over-priced ~0.8pp (ν=0.30) → ~1.2pp (ν=0.40)**, peaking vu≈1–1.5 then easing (not a
deep-OTM cliff). So thread-2's tail-amplification reaches H=0.10 **mildly, call-wing only** (the
low-integrated-variance tail where D35's envelope perturbation shows). **SPX viability:** equity skew lives
in the **put wing** (the CF's steep downside skew, ~28% vs ~13% IV) — the lift is clean there + ATM → **SPX
put-wing/ATM skew calibration is VIABLE**; the OTM call wing needs exclusion or a documented ~1pp caveat
(it grows with ν; tested ν=0.30–0.40 — the real SPX-implied ν sets the caveat's size). This validates the
lift's OTM-smile pricing at H≈0.10/high-ν — the gate that precedes any SPX calibration. See ROADMAP **D36**.

**STATUS UPDATE (2026-06-28, D37) — SPX calibration engine built + validated; H unidentifiable from one smile.**
`layer4_calibrate.py` fits θ=[H,ν,ρ,ξ₀] to an IV smile via least_squares (TRF, IV space) against the CF
(ξ₀→V0=θ, kappa fixed — single maturity can't pin mean-reversion). Validated against synthetic CF
known-answers (real-market SPX is a later step on the user's machine). Four experiments: **(1) CF→CF gate —
EXACT** recovery (0.00%, IV-RMSE 0.0000, both grids); **(2) identifiability** — ξ₀/ρ/ν IDENTIFIED, **H WEAK**
(flat JᵀJ direction; H~ν degenerate −0.82/−0.92; cond 6.2e5 full / 2.4e7 put — worse); **(3) noise-robustness**
— ξ₀/ρ rock-solid, ν stable, **H blows up** (62% spread at 0.1pp noise → 153% at 0.5pp); **(4) lift→CF** — the
lift's ~1pp call-wing bias is **absorbed into H** (+166% full), ν/ρ/ξ₀ stay CLEAN on the FULL grid (~5%,
RMSE 0.018pp); the put-wing-only is WORSE (ν+18%, ρ+12%) — conditioning beats wing-cleanliness for
calibration. **Unifying finding:** H is the weak parameter everywhere (both noise and the lift's bias corrupt
H, the flat eigen-direction) — the calibration-context echo of the project's H-identifiability theme.
**SPX guidance:** calibrate with the CF (exact); from one maturity trust ξ₀/ρ/ν not H (fix it / add
maturities → **multi-maturity surface is the motivated next step**); if the lift must be used, calibrate
ν/ρ/ξ₀ on the full grid (bias quarantined in H), don't restrict to the put wing. See ROADMAP **D37**.

**STATUS UPDATE (2026-06-29, D38) — multi-maturity surface: the term structure makes H usable at low-noise data.**
`layer4_calibrate_surface.py` extends D37 to a joint IV surface across maturities (imports D37's
calibrate/residuals/IdentReport; adds T-parameterised smiles, √T-scaled per-T strike grids, a stacked
residual, a dimension-agnostic identifiability report). Tests whether the term structure identifies H, which
a single smile structurally cannot (D37: H flat, cond 6e5, H~ν −0.84). **Verdict: (B) IMPROVES STRONGLY,
PARTIAL.** Surface CF→CF recovers H exactly (noise-free); cond improves **~100×** (8.97e5→8.5e3) but H stays
the flat direction (|flat[H]|≈0.89, H~ν −0.85 — not structurally broken). **Noise-robustness (the payoff,
matched knobs):** H-spread ~106%→~10% at σ=0.1pp (~10×), ~190%→~29% at 0.3pp, ~194%→~48% at 0.5pp; ν/ρ/ξ₀
tight throughout. So the surface makes **H usable at clean data (~H±0.01)** — vs hopeless from one smile —
but the gain SHRINKS under noise because the soft H~ν direction persists (mitigates, doesn't cure → (B) not
(A)). **κ: keep fixed** (freeing it inflates cond 550×, |flat[κ]|=0.96 — degenerate; with flat ξ₀ the drift
cancels in expectation). **Real-market guidance (span sweep):** use **≥5 maturities incl. a short (~1-month)
tenor** (the short end gives the biggest cond jump — the T^(H−1/2) signal); more/wider keeps tightening; keep
κ fixed. **Caveats:** the single-smile baseline is knob/ensemble-sensitive (62/96/106% across runs) so the
comparison uses this run's matched-knob baseline; 10-draw spreads are APPROXIMATE (sampling error) — the
order-of-magnitude tightening is robust, the exact factors are not. (Infra: the noise-ensemble first stalled
2h+ on BLAS pool-oversubscription — NOT convergence; fixed by pinning BLAS=1/worker + pool=physical-cores,
baked into the module.) See ROADMAP **D38**.

**STATUS UPDATE (2026-06-29, D39) — real-market run: the engine calibrated to LIVE Deribit BTC options.**
`deribit_surface.py` (fetch+clean) + `calibrate_btc.py` (driver) calibrate the D38 engine to a live BTC
option surface (63 pts / 5 maturities, 11d→6mo, put-wing+ATM, forward-normalised, snapshotted). Conventions
pinned from the live API (IV in %, moneyness via the forward, inverse settlement irrelevant in IV space).
**Fit:** H=0.0201 (railed to LB), ν=0.636, ρ=−0.371, ξ₀=0.230, **IV-RMSE 0.888pp** — rough-Heston fits the
gross smile, ξ₀/ρ/ν plausible (ν high = D34 regime, ρ put-skew, ξ₀≈ATM-var). **Finding 1 — H NON-IDENTIFIED**
(3 diagnostics: Jacobian |flat[H]|=0.99 / cond 5.2e4 / corr(H,ν)=−0.96; H at-LB in 100% of bootstrap draws;
cross-run ν/ρ instability at equal RMSE) — the **option-calibration angle confirms the observational-
equivalence wall** (Phase B Ĥ≈0.08, D37/D38), now from two independent angles. **Finding 2 — the model
UNDER-PRODUCES crypto's extreme put-tail** (undershoots the deep-put wing + long-tenor curvature even at
maximal roughness; real crypto's left tail is steeper than rough-Heston → jumps/steeper-kernel motivation),
which COMPOUNDS Finding 1 (H rails to the bound straining for a tail it can't reach). **Caveats:** the
warm-start bootstrap's H-std≈0 is "railed to the boundary", NOT tight identification (the jacobian is the
evidence); the reduced ≤6mo surface (the 1yr tenor overflows the CF even at N=4000) drops H-span and
compounds the non-identification (open: does full span via per-maturity N_riccati tighten H?); ξ₀ is the one
robustly-identified parameter. **Infra:** a 13× Riccati-cache (memoise the strike-independent CF per
maturity — bit-identical IVs) made it tractable. SPX = the unchanged-engine transfer for later; ETH = the
comparison follow-up. See ROADMAP **D39**.

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

> **Scope:** measured at **ν ≤ 0.20** (the validated simulator range; the high-ν SPX-calibrated
> regime is a separate multifactor-lift extension, §5). **PASS** iff: (i) the H = ½ case matches the
> closed-form Heston reference within MC error; (ii) the fit region is bias-dominated (b_n ≫ MC
> s.e.); (iii) the measured weak order α is significantly **> H** across the H-sweep; and (iv) α is
> consistent with the stated expectation (≈ 1, or the sourced value if §3 verification supplies one).
> **FAIL** → α ≈ H (no weak speed-up) or α inconsistent with prediction. **A FAIL is a result**,
> not a defect: it would establish that roughness bottlenecks the weak rate of the hybrid scheme —
> a publishable characterisation of the scheme under roughness.

> **MEASURED (2026-06-27, D31).** Gate outcome: **(i) PASS** — H=½ matches closed-form Heston to
> ~1e-6, recovered weak order ≈0.89–1.14 inside the literature range [0.6, 1]; **(ii) PASS** — fit
> regions are bias-dominated (b_n ≫ MC s.e.; Romano–Touzi conditional MC widened the window);
> **(iii) PASS** — α significantly **> H** at every H (the banked headline); **(iv) MIXED** — α is
> consistent with ≈1 for H ≥ 0.20 (PASS), shows a quantified roughness penalty α ≈ 0.74 < 1 at
> H = 0.05 (PARTIAL — a result, not a defect), and is borderline (≈0.84–0.95, *pre-asymptotic*
> window-sensitivity) at H = 0.10. **No FAIL** — α never collapses to H. The mid-range borderline
> and a definitive α(H) curve are **pending finer grids** (the §5 multifactor-lift extension).
> Full record + caveats: ROADMAP **D31**.

---

## 8. Scope and honest boundaries

- Measures the **κ=0** scheme. κ=1 (`layer1b_kappa1.py`) improves the error *constant*, not the
  rate (β = 2H either way), so the *order* result is κ-invariant; a κ=1 run would sharpen the
  constant only.
- Layer 4 is **two deliverables**: a native **rough-Heston simulator** (`rough_heston.py`, reusing
  `roughvol_core.py`'s κ=0 Volterra machinery, adding rough-Heston variance/price dynamics) and the
  convergence study built on it. The simulator's **build-validation gate** (re-measure β, confirm
  β = 2H) is **PASSED with the QE positivity scheme for ν ≤ 0.20** (`rh_beta_gate.py`, 2026-06-27):
  β = 0.070 / 0.167 / 0.384 / 0.737 across H ∈ {.05,.10,.20,.35}, max|β−2H| = 0.037, consistent with
  the layer1b reference (0.13/0.23/0.42/0.72). Same model on both sides → **no carry-over assumption**.
- **Boundary finding (2026-06-27, a result in its own right).** The explicit hybrid Volterra–Euler
  scheme's strong-order *coupling* (β) and weak-order *priced bias* **both** degrade as V→0 events
  exceed ~10% of samples (ν ≳ 0.25): β collapses (the V=0 clip/branch fires at *different* times on
  the fine vs coarse MLMC grids → the coupling breaks) **and** the priced bias becomes scheme- and
  n-dependent (qe vs truncation 2.4–5.3% at ν=0.4, halving as n doubles; <0.5% at ν=0.15). The
  scheme is **validated for ν ≤ 0.20**; high-ν needs the multifactor lift (§5; its kernel foundation
  is now BUILT — `rough_kernel_soe.py`, D32). Three positivity
  schemes were compared on evidence — **QE** (chosen, smallest E[V] bias +4%, best β), full
  **truncation** (β collapses, E[V] +13%), **reflection** (rejected: E[V] +158%, β worst). This
  characterises where the method stops being trustworthy — a genuine property, not a config limit.
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
