# Gate-check spec — κ=1 hybrid FINE path (G-H1a…G-H1d)

> **Driver committed:** 2026-06-22 (`d778074`) · **Spec reconstructed:** 2026-09-29
> from the driver scripts under RVL-009. The original specification was
> chat-only and is not recoverable. Predictions and gates below are as
> recorded in the code; this document does not attest when they were
> written.

**Status:** driver `gh1_kappa1_finepath.py` **BUILT & RUN** — recorded verdict **PASS**, all four
sub-gates green · **Engine:** opt-in `kappa=1` flag in `layer1b_mlmc_asian.py`; κ=0 is and stays the
default · **Sibling:** `kappa1_hybrid_coupling_design.md` (this dir) covers the **COARSE** coupler
(gated **G-H2a/b/c** by `gh2_kappa1_coupler.py`) — this file is the **FINE** path only.
Cross-reference, no overlap.
Where the script's intent is not recoverable from the code it is marked open in §8's extraction
rather than guessed (`docs/AUDIT_TAIL.md:17`).

**The question (not a claim).** The κ=0 optimal-discretisation scheme understates the discrete
Volterra variance `Var(W̃_T)` by **14.7%** at H=0.10 — a *known, quantified* defect, not a bug. The
BLP hybrid (κ=1) integrates the nearest — most singular — kernel cell exactly. Does that close the
gap on the fine path, and does it do so **without** introducing a bias somewhere the
marginal-variance check is blind to? G-H1 is a *correctness* gate, not a performance gate: it asks
whether the κ=1 fine path is right, never whether it is cheaper. Cost and adoption are G-H4's
question, whose drivers are `gh4_kappa1_conditional.py` (step 2a) and `gh4_kappa1_cost.py` (step 2b);
its spec `gh4_kappa1_adoption_spec.md` is **chat-only and not in the repo**
(`docs/gate_checks/README.md:113`), so it cannot be cited as a live document.

---

## 0. Scope discipline — what this is, and is NOT

**IN.** The κ=1 **fine path** — the per-cell exact nearest-cell Gaussian, the compensator that goes
with it, and the four checks that the resulting `W̃` and the asset built on it are correct.

**OUT, deliberately:**

- **The coarse coupler.** `mlmc_asian_level` refuses κ=1 above level 0 outright
  (`layer1b_mlmc_asian.py:373–377`, `NotImplementedError`: *"kappa=1 coarse coupler is not wired yet
  — fine path only (l=0). See kappa1_hybrid_coupling_design.md / gh2_kappa1_coupler.py."*). The split
  + conditional-resampling construction and the **coupling-tightness** assert are the sibling design
  note's business, gated by `gh2_kappa1_coupler.py` as **G-H2a** (telescoping consistency
  `a_l − a_{l−1} − E[Y_l] < 1`), **G-H2b** (coupling tightness) and **G-H2c** (rate β≈2H)
  — `gh2_kappa1_coupler.py:12–15`, verdict list at `:135–137`, index at
  `docs/gate_checks/README.md:69–70`.
- **The cross-covariance check is not a G-H2 sub-gate at all.** `Cov(ΔW^c_j, W^c_{j,1})` is verified
  as a unit check in the de-risk script `kappa1_coupling_design_check.py:127` (*"coarse cross-cov
  Cov(dWc,Wc1): target …"*), recorded at `README.md:71–72`. It is neither in this driver nor in
  `gh2_kappa1_coupler.py`.
- **Composition with the other estimators.** κ=1 is a third, independent scheme axis; the engine
  raises on any combination (`layer1b_mlmc_asian.py:370–372`: *"kappa=1 is a separate scheme axis;
  combine with neither antithetic nor conditional"*).
- **Any MLMC or cost claim.** The driver never calls `mlmc_asian_level`. It imports and exercises
  `_volterra`, `_paths_from_increments`, `volterra_weights`, `volterra_weights_kappa1` and `_bs_call`
  directly (`gh1_kappa1_finepath.py:23–24`), so **no** level machinery, no telescoping, no `Var(Y_l)`
  is under test here.

**Note on gate numbering — two incompatible schemes are in the record.** The sibling design note's
§8 (`kappa1_hybrid_coupling_design.md:157–178`) uses an **older** scheme:

| Design note §8 | Driver / `docs/gate_checks/README.md` |
|---|---|
| **G-H1** = coarse-law exactness + coupling tightness + cross-covariance | **G-H1a–d** = the FINE path (`gh1_kappa1_finepath.py`) |
| **G-H2** = rate | **G-H2a–c** = the COARSE coupler (`gh2_kappa1_coupler.py`) |
| **G-H3** = variance factor | *(no counterpart; `G-H3` occurs nowhere but the design note)* |
| **G-H4** = Giles cost | **G-H4** = adoption (step 2a/2b) |

This spec uses the **driver/README scheme throughout**. The design note's §8 numbering is recorded
here as a live discrepancy in the audit trail, not silently reconciled; §8 carries it as an open
item.

---

## 1. Mechanism — what `kappa=1` actually changes

`volterra_weights_kappa1(n, H, T)` (`layer1b_mlmc_asian.py:163–186`) returns four objects and nothing
else:

- `g_hyb` — the κ=0 kernel with the **nearest weight zeroed**, `g_hyb[0] = 0.0` (line 180).
- `v_k1` — the exact discrete variance of the κ=1 scheme,
  `v_k1 = v0 − 2H·dt·g₁² + dt^{2H}` (line 181): the nearest-cell Riemann term is *subtracted* and the
  exact nearest-cell contribution `dt^{2H}` *added*. The k≥2 tail inside `v0` is untouched.
- `c_near = Cov(W_{i,1}, ΔW_i)/dt`, with `Cov = dt^{H+½}/(H+½)` (lines 182, 184).
- `sig_perp`, the residual standard deviation, `sig_perp² = max(dt^{2H}/(2H) − dt^{2H}/(H+½)², 0)`
  (lines 183, 185) — strictly positive for H < ½.

The path build (`_volterra`, lines 205–212) is then three lines:

```
W_near  = c_near * dW1 + sig_perp * Z        # exact nearest cell  (line 209)
far     = fftconvolve(dW1, g_hyb)[:, :n]     # k>=2 Riemann tail   (line 210)
W_tilde = sqrt(2H) * (W_near + far)          #                     (line 211)
```

`Z` is a fresh N(0,1) array, one per path per cell, and is **mandatory** — `_volterra` raises
`ValueError("kappa=1 needs Z (fresh N(0,1) nearest-cell residuals)")` without it (lines 206–207).
Two consequences matter for the gates. First, `c_near` and `sig_perp` are **scalars**, not vectors:
the nearest-cell law is identical in every cell by construction. Second, `Z` enters the *variance*
path only — the asset driver stays `dW_S = ρ·ΔW₁ + √(1−ρ²)·ΔW₂` (line 243), unchanged.

---

## 2. The gap being closed (the analytic target)

`v_k0` and `v_k1` are deterministic functions of `(n, H, T)`, so the target is exact, not estimated.
Recomputed from the engine's own closed forms during this reconstruction, at **n=256, T=1**:

| H | κ=0 `v_k0[n]/T^{2H}` | κ=1 `v_k1[n]/T^{2H}` | κ=1 residual `1 − v_k1/T^{2H}` | corr(W_{i,1}, ΔW_i) |
|---|---|---|---|---|
| 0.05 | 0.6151 | 0.9995 | 4.63×10⁻⁴ | 0.5750 |
| 0.10 | **0.8529** | **0.9996** | 4.47×10⁻⁴ | 0.7454 |
| 0.20 | 0.9798 | 0.9998 | 1.89×10⁻⁴ | 0.9035 |

The correlation column is `√(2H)/(H+½)`, dt-independent, derived from lines 182–183; at H=0.10 it
reproduces the ≈0.745 quoted in the sibling note's §6 (`kappa1_hybrid_coupling_design.md:131`) — the
same number from the other direction.

**Why `v_k1` is exact rather than approximate.** Zeroing `g_hyb[0]` (line 180) makes the near term
and the far tail use **disjoint** increments — `W_near` at cell *i* consumes `dW1[:, i]` (plus the
independent `Z`), while `far` at cell *i* consumes only indices `< i`. The two are therefore
independent, and `2H·(Cov²/dt + sig_perp²) + (v0 − 2H·dt·g₁²)` collapses to exactly line 181's
`v_k1` (checked here to 1.1×10⁻¹⁶). That independence is what makes §2 a closed-form target and not
an estimate.

Two things to read off. The gap is **worst where the model is roughest** (38.5% at H=0.05, 2.0% at
H=0.20), which is exactly the regime the project cares about. And κ=1 does **not** reach 1.0 — a
residual of **1.9–4.6×10⁻⁴** survives across the swept H, because only the nearest cell was made
exact and the k≥2 Riemann tail inside `v0` is still an approximation (line 181). A gate demanding
`= 1` would be wrong.

---

## 3. The compensator trap — why G-H1b exists at all

The variance path is `V_t = ξ₀·exp(η·W̃_t − ½η²·v_t)`, and `E[V_t] = ξ₀` holds **only if `v_t` is the
discrete variance of the `W̃` actually simulated**. Swap the κ=1 path in and leave the κ=0
compensator `v_k0` in place and the code still runs, still produces plausible prices, and is
silently biased. The engine's docstring names it: *"using v_k0 here is the silent-bias trap that
G-H1b guards against"* (lines 197–198).

**Size of the trap.** The bias is `E[V_t]/ξ₀ − 1 = exp(½η²(v_k1 − v_k0)) − 1`, **not** the variance
gap of §2. At `PARAMS` (H=0.10, η=1.50) and n=256, `v_k1 − v_k0 = 0.146612` — and this difference is
*constant in i*, since line 181 adds the same `dt^{2H} − 2H·dt·g₁²` at every index. So the bias is

```
exp(0.5 · 1.50² · 0.146612) − 1 = 0.179321,  uniform across all 256 points
```

larger than the §2 variance gap (0.1471) and η-dependent, which the variance gap is not. One
consequence for §5: because the trap's bias is flat in t, G-H1b's `max_t` cannot be locating a
worst-case time point on the trap leg — for that leg the max only selects the noisiest point.

G-H1b therefore runs **three** schemes, not two: κ=0 baseline, κ=1 with the correct `v_k1`, and κ=1
with the wrong `v_k0` — the trap deliberately built and measured; the driver's own comment is
*"the trap is caught"* (lines 94–95). *(Inferred, not stated in the source: a gate that checked only
the correct branch could not distinguish "unbiased" from "the test is insensitive". The three-scheme
construction is in the code; this reading of why is this reconstruction's.)*

---

## 4. The committed prediction (hard-coded in the driver)

The prediction is not prose here — it is a literal in the print statement
(`gh1_kappa1_finepath.py:52–53`):

```
headline (n=256, H=0.10): kappa1 = {r1:.4f} (pred ~0.9996), kappa0 = {r0:.4f} (pred ~0.8529)
```

*The code cannot establish **when** those literals were written.* They are the analytic ratios and
§2 reproduces them exactly, so they **could** have been fixed before any run; nothing in the repo
shows that they were. Treat "committed in advance" as an inference, not a recorded fact (§8).

- **κ=1 closes the gap to within 1% of the continuum `T^{2H}`, at every H in {0.05, 0.10, 0.20}.**
  *Falsified if any H misses the 1% band* — the sweep is conjunctive, not best-of-three
  (`ok &= within`, line 47). This is the only clause the driver gates on.
- **κ=0's `pred ~0.8529` is printed, not tested.** The κ=0 column is computed and displayed
  (lines 44–50) and the literal appears in the headline print, but `ok` accumulates the κ=1 band
  alone (line 47). No code path treats a κ=0 deviation as anything at all.
- **The correct compensator is unbiased and the wrong one is not.** Both halves are gated: the PASS
  conjunction at line 96 fails if either the correct leg drifts or the trap is not caught.
- **η=0 reproduces Black–Scholes.** *(The further reading — that this exists to stop κ=1's extra
  Gaussian leaking into the asset driver — is inferred from lines 237–244, not stated anywhere in
  the driver. See §7.)*

Nothing here predicts a rate or a cost improvement. On the sibling note's §7 reasoning the strong
rate is O(n^{−H}) independent of κ (`kappa1_hybrid_coupling_design.md:140`), so none is expected —
and none is claimed by this gate.

---

## 5. Gate-check bar (PASS/FAIL committed before any measurement)

All four sub-gates are in one driver, `python gh1_kappa1_finepath.py`; the overall verdict string
**`ALL GREEN`** / **`NOT all green`** is printed at **line 172** (`allg` is computed at line 168).
κ=0 and κ=1 are run **side by side on the same `dW1`** in **G-H1a, G-H1b and G-H1c** (lines 41–42,
75–76, 114–115), so in those three the difference is the scheme, not the draw. **G-H1d has no κ=0
leg**: it never calls `_volterra`, taking only `c_near, sig_perp` from `volterra_weights_kappa1`
(line 133) and forming `W_near` directly (line 137) — it exercises the κ=1 nearest-cell constants
alone, against closed forms, with nothing to compare side by side. Every seed is a fixed literal;
none is swept.

**G-H1a — variance gap (the headline).** Banner: *"variance gap: Var(W~_T)/T^{2H} (>=1e5 paths) PASS
= k1 within 1%"*. `n=256`, `N=120_000`, `rng = default_rng(100)` re-created per H, H swept over
{0.05, 0.10, 0.20} (lines 31–42). Empirical `W[:, −1].var()/T^{2H}` and analytic `v[−1]/T^{2H}` are
both printed; the gate reads the **empirical** one.
**PASS:** `abs(r1e − 1.0) < 0.01` for **all three** H (`ok &= within`, lines 46–47). A secondary
line is printed but does not gate: *"=> does Var(W~_T) clear 0.99 under kappa=1?"* (lines 54–55).
**The three H are not three independent tests.** `rng` is re-created with seed 100 *inside* the loop
(line 38) and `dt = T/n` does not depend on H, so `dW1` and `Z` are **bit-identical** across all
three H (verified here). The three results share one draw set and their deviations are positively
correlated — material, given that the 1% bar sits at only ~2.4 standard errors on this
reconstruction's arithmetic (§8).

**G-H1b — forward variance / the trap.** Banner: *"forward variance: max_t |E[V_t]/xi0 - 1| (correct
vs WRONG compensator)"*. `PARAMS` (H=0.10, η=1.50, ξ₀=0.04), `n=256`, `Ntot=200_000` accumulated in
batches of `B=40_000`, `rng = default_rng(7)` (lines 63–70). Reports, per scheme,
`err = max_t |E[V_t]/ξ₀ − 1|` and `z = max_t |E[V_t] − ξ₀|/se` for `kappa0 (baseline)`,
`kappa1 CORRECT comp.` and `kappa1 WRONG comp. (v_k0)`.
**PASS** (line 96) is the exact conjunction

```
res["k1c"][1] < 2 * res["k0"][1] + 5   and   res["k1w"][1] > 4 * res["k1c"][1]
```

— i.e. the correct compensator's z is *relative to the κ=0 baseline's* z, not to an absolute bar, and
the trap must additionally be caught by a factor 4. Verdict string: *"=> kappa1 correct compensator
unbiased (z~baseline), wrong one a systematic bias: PASS/CHECK"* (lines 97–98).

**G-H1c — Black–Scholes anchor.** Banner: *"BS anchor: eta=0 European price vs Black-Scholes PASS = z
< 2"*. `PARAMS` with `eta=0.0`, `n=128`, `N=120_000`, `rng = default_rng(2)`; European payoff via
`_paths_from_increments(..., "european", κ)` for κ=0 and κ=1 on shared `dW1`, `dW2`; reference
`_bs_call(S0=100, K=100, T=1, r=0, σ=√ξ₀=0.2)` (lines 106–115).
**PASS:** `z = |mc − bs|/se < 2`, applied to the κ=1 leg only
(`okc &= z < 2 if name == "kappa1" else True`, line 120).
**The two legs are numerically identical, so the clause cannot bite.** At η=0,
`V_left[:, 1:] = xi0*exp(0·W̃ − 0) ≡ ξ₀` irrespective of κ (`layer1b_mlmc_asian.py:237–240`), and both
legs are handed the same `dW1`, `dW2` (lines 114–115); the payoff therefore cannot depend on κ.
Confirmed by running it: `np.array_equal(q0, q1)` is `True`, `max|q0 − q1| = 0.0`, and the driver
prints the same `MC 7.9900 ± 0.0381` and the same `z = 0.64` on both lines. The κ=0 line is a
**duplicate**, not an independent control, and whether it is gated is immaterial to the verdict.

**G-H1d — near-cell law.** Banner: *"near-cell law: Var(W_{i,1}), Cov(W_{i,1},dW_i) vs closed
forms"*. H=0.10, `n=128`, `N=200_000`, `rng = default_rng(11)`; builds
`W_near = c_near*dW1 + sig_perp*Z` and compares against `var_cf = dt^{2H}/(2H)` and
`cov_cf = dt^{H+½}/(H+½)` at cells `i ∈ {1, 32, 64, 127}` (lines 131–143).
**PASS:** `rel.err(Var) < 0.02` **and** `rel.err(Cov) < 0.03` at every one of the four cells (line
147). Verdict string: *"=> near-cell law matches closed forms within MC noise: PASS/CHECK"*.

---

## 6. Recorded outcome — PASS, with two transcription discrepancies

**As recorded** in `docs/gate_checks/README.md:53–63` (the run record for this thread; note that
`docs/AUDIT_TAIL.md:44–45` reserves the phrase *system of record* for itself):

- Variance gap **0.853 → 0.9996** analytic, **0.9973** empirical; within 1% of the continuum at all
  three H.
- Forward variance **unbiased, z = 2.86**, against the wrong-compensator trap at **z = 64** and a
  **0.19** bias.
- BS anchor **z = 0.64**.
- Near-cell Var/Cov within **0.05–0.67%**, inside the 2%/3% bars.

**As reproduced** during this reconstruction (`python gh1_kappa1_finepath.py` at HEAD,
no uncommitted changes to tracked files, numpy 2.4.4 / scipy 1.17.1, wall 21s) — verdict **ALL GREEN**, all four
sub-gates PASS:

| Quantity | Record | Re-run | |
|---|---|---|---|
| G-H1a κ=1 empirical, H=0.10 | 0.9973 | 0.9973 | ✔ |
| G-H1a κ=1 empirical, H=0.05 / 0.20 | — | 0.9978 / 0.9970 | — |
| G-H1b trap `err` | 0.19 | 0.1892 | ✔ |
| G-H1b trap `z` | 64 | 64.5 | ✔ |
| G-H1b **correct-compensator `z`** | **2.86** | **5.4** | **✘** |
| G-H1b baseline κ=0 `z` | — | 4.2 | — |
| G-H1c `z` (both legs) | 0.64 | 0.64 | ✔ |
| G-H1d Var rel.err | 0.05–0.67% | 0.052–0.666% | ✔ |
| G-H1d **Cov rel.err** | *(covered by the same 0.05–0.67%)* | **0.002–0.999%** | **✘** |

**Discrepancy 1 — the correct-compensator z does not reproduce.** The record's 2.86 comes out as 5.4
at the committed seed 7. The driver is byte-identical to its commit (`git diff d778074 HEAD --
gh1_kappa1_finepath.py` is empty), and re-running G-H1b against the engine *as of that commit* also
gives 5.4, so the three 2026-07-09 engine commits (`67ee685`, `8ce5aa9`, `e895b42`) do not explain it.
The driver writes **no file** — no figure, no CSV, no log — so nothing from the recorded run
survives to compare against. **Unresolved**
— see §8. The verdict is unaffected: `5.4 < 2·4.2 + 5 = 13.4` and `64.5 > 4·5.4 = 21.6`, so line
96 still passes. The margin does change: the trap-catch factor is **≈12** as reproduced, against the
**≈22** implied by the recorded 2.86. Both clear the required 4.

**Discrepancy 2 — the near-cell range covers Var only.** `0.05–0.67%` is exactly the **Var** column
(0.052% … 0.666%). The **Cov** column runs 0.002% … **0.999%**, so the recorded range understates
it. Both remain well inside the 2% / 3% bars and the sub-gate passes either way.

**Discrepancy 3 — none.** The record's `0.19` for the trap is *not* a mis-transcription of the
closed-form 0.17932 of §3: the driver prints `err`, a **max over 256 noisy points**, whose
expectation sits above the flat systematic value, and it prints **0.1892**. What is loose in the
record is only the *label* — 0.19 is a measured max-over-t error, not "the systematic bias".

**Verdict: PASS**, all four sub-gates, reproduced. The κ=1 fine path is correct and the compensator
that goes with it is the κ=1 one.

---

## 7. Honest scope note — what these four gates do NOT establish

- **G-H1c cannot fail because of a wrong `W̃`.** At η=0 the compensated exponential collapses to 1,
  so `V_left ≡ ξ₀` (`layer1b_mlmc_asian.py:237–240`), the log-Euler step is exact for constant
  variance, and the Volterra path is computed and then annihilated by η=0 — which is why the κ=0 and
  κ=1 legs are bitwise identical (§5). Nothing about the Volterra scheme's accuracy is tested.
  Reading it as a pricing-accuracy check would overclaim. *(Inferred, not stated in the driver: what
  the gate can still catch is `Z` leaking into the asset driver — a leak would enter `dW_S` at line
  243, break that identity and move the κ=1 price off BS. The identity holding is the evidence that
  no such leak exists in this build; the purpose attributed to the gate is this reconstruction's.)*
- **G-H1d is not a positional test.** `c_near` and `sig_perp` are scalars, so `W_near` is i.i.d.
  across cells by construction; the four indices are four independent replications of one law (disjoint columns of
  `dW1`/`Z`),
  not a check of anchoring or indexing. Cell 0 is not tested and nothing distinguishes it. The
  sub-cell index errors the sibling note warns about (`kappa1_hybrid_coupling_design.md:190`,
  *"Index carefully"*) are invisible to G-H1d — they are **G-H2b**'s business, which is exactly what
  `gh2_kappa1_coupler.py:7–10` says the tightness assert exists to catch.
- **G-H1d's closed forms are not independently derived.** The driver's `var_cf` and `cov_cf`
  (lines 138–139) are the same two expressions the engine uses (lines 182–183). The gate confirms the
  *decomposition* `c_near = Cov/dt`, `sig_perp² = Var − Cov²/dt` is implemented correctly **given**
  those target moments; it does not re-derive the moments from the Itô isometry. A wrong target
  moment would pass.
- **G-H1b checks the compensator identity, not the production variance array.** It builds `V` from
  every column of `W̃` with the full `v` (lines 77–79), whereas `_simulate_paths` uses the
  left-endpoint shift `W̃[:, :-1]`, `v[:-1]` (lines 239–240). The identity
  `E[exp(ηW̃_t − ½η²v_t)] = 1` is what is verified; the off-by-one assembly of `V_left` is not.
- **No cost, no rate, no MLMC.** Nothing here supports "κ=1 is better". The measured statement is
  "κ=1's fine path is correct". Whether that correctness is worth paying for is G-H4, which
  **adopts κ=1 for conditional standard MC only and not for MLMC** (`README.md:86–88`).
- **Single-seed sub-gates, and G-H1a's three H share one draw set.** Each sub-gate runs one fixed
  seed, and within G-H1a the three H values reuse bit-identical `dW1`/`Z` (§5). No seed-stability or
  sign-flip check is performed, so the margins in §6 carry no cross-seed error bar — which is also
  why §6's Discrepancy 1 cannot be diagnosed from the repo.

---

## 8. Open items (reconstruction)

*Questions this reconstruction could not answer from the code are recorded in [`RVL-009_open_items.md`](RVL-009_open_items.md).*
