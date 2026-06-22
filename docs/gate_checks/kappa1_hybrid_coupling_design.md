# κ=1 hybrid MLMC coupling — design (ROADMAP item 3)

*Design-before-coding note. The load-bearing covariance algebra is validated
numerically in `kappa1_coupling_design_check.py` (closed forms match
quadrature isometry to ~1e-11; coarse marginal preserved to 1e-12).*

## 1. Goal

The Layer 1b engine simulates the Volterra process with the **κ=0**
optimal-discretisation (Riemann) scheme: every kernel cell is approximated by
`g(b_k)·ΔW`, so W̃ is a pure convolution of the Brownian increments and the
MLMC coupling is trivially exact (coarse increment = pairwise sum). The **κ=1
hybrid** scheme (Bennedsen–Lunde–Pakkanen 2017) instead integrates the
*nearest* — most singular — kernel cell exactly, which leaves the strong rate
at O(n^{-H}) but slashes the error constant; it is the standard high-accuracy
rough-vol simulator. ROADMAP item 3 flags the obstacle: the exact nearest-cell
Gaussian "needs the correlated nearest-cell Gaussian to be coupled across
levels — non-trivial." This note solves that coupling and specifies the build.

## 2. The κ=1 fine path

With `α = H−½`, grid `t_i = i·dt`, the hybrid Volterra value is

```
W̃_{t_i} = √(2H) [ W_{i,1}  +  Σ_{k=2}^{i} g(b_k)·ΔW_{i-k+1} ]
```

i.e. the κ=0 convolution with its **nearest weight `g(b_1)` removed** and
replaced by the exact nearest-cell integral

```
W_{i,1} = ∫_{t_{i-1}}^{t_i} (t_i − s)^α dW_s.
```

Per cell we therefore draw the **bivariate** Gaussian `(ΔW_i, W_{i,1})` with

```
Var(ΔW)   = h,                         h = dt
Var(W_1)  = h^{2H}/(2H)
Cov(ΔW,W_1) = h^{H+½}/(H+½)
```

(all exact Itô isometries). The plain increments `ΔW` still drive the k≥2
Riemann tail **and** the asset log-Euler step `dW_S = ρ·ΔW + √(1−ρ²)·ΔW₂`; the
exact `W_{i,1}` enters **only** the variance path. So κ=1 touches the Volterra
convolution alone — the asset coupling and the orthogonal driver are unchanged.

## 3. The coupling problem

For MLMC we need a coarse path (grid `2h`) that (a) has the **exact** κ=1 coarse
law — else the telescoping `E[P^L]=E[P^0]+ΣE[Y_l]` is biased — and (b) is
tightly coupled to the fine path. The plain increments coarse-grain trivially
(`ΔW^c_j = ΔW^f_{2j-1}+ΔW^f_{2j}`), and that handles the asset and the k≥2 tail.
The only hard object is the **coarse nearest-cell integral**

```
W^c_{j,1} = ∫_{t_{2j-2}}^{t_{2j}} (t_{2j} − s)^α dW_s,
```

anchored at the coarse endpoint `t_{2j}` and spanning two fine sub-cells. It is
*not* a deterministic function of the fine `(ΔW^f, W^f_{·,1})` — hence "non-
trivial."

## 4. The solution — exact split + conditional resampling

Split the coarse integral at the midpoint `t_{2j-1}`:

```
W^c_{j,1} = ∫_{t_{2j-2}}^{t_{2j-1}} (t_{2j}−s)^α dW_s  +  ∫_{t_{2j-1}}^{t_{2j}} (t_{2j}−s)^α dW_s
          =        I₁^{(j)}                            +        W^f_{2j,1}.
```

**Key fact:** the second term is *exactly* the fine nearest-cell integral of
cell `2j` (same anchor `t_{2j}`, same interval `[t_{2j-1},t_{2j}]`) — so it is
**reused verbatim** from the fine path. The two terms are over disjoint
intervals, hence independent, and their variances add to the exact coarse
marginal:

```
Var(I₁) + Var(W^f_{2j,1}) = h^{2H}(2^{2H}−1)/(2H) + h^{2H}/(2H) = (2h)^{2H}/(2H).  ✓ (checked to 1e-12)
```

Only `I₁^{(j)} = ∫_{first subcell} (t_{2j}−s)^α dW_s` (kernel anchored one cell
*away*, so non-singular) must be generated. The fine path summarises that
sub-cell by `(ΔW^f_{2j-1}, W^f_{2j-1,1})`. Generate `I₁` from its **Gaussian
conditional law** given that summary, using the 3×3 covariance of
`(ΔW, W_1, I₁)` on a sub-cell:

```
Var(I₁)    = h^{2H}(2^{2H}−1)/(2H)
Cov(ΔW,I₁) = h^{H+½}(2^{H+½}−1)/(H+½)
Cov(W_1,I₁)= h^{2H}·C(H),   C(H)=∫_0^1 w^α(1+w)^α dw   (one quadrature constant)
```

Then with `x=(ΔW,W_1)`, `β = Σ_xx^{-1} Σ_{x,I₁}`, `σ_c² = Var(I₁) − Σ_{I₁,x}β`:

```
I₁ = β·x + σ_c·Z,   Z ~ N(0,1) fresh.
```

This makes `W^c_{j,1} = I₁ + W^f_{2j,1}` have the **exact** coarse law (so the
estimator is unbiased) while sharing the conditional mean `β·x` with the fine
path. Validation across H∈{0.07,0.10,0.20}: `σ_c² > 0` always, and the fine
summary explains **99.4–99.8% of Var(I₁)** — the residual fresh noise is
0.2–0.6%, so the κ=1 coupling is nearly as tight as the κ=0 one.

C(H) is a single scalar per H (1.558, 1.486, 1.299 at H=0.07/0.10/0.20),
precomputed by quadrature; everything else is closed form.

## 5. Why it is unbiased

Unbiased telescoping requires only that the coarse path have the correct
coarse-level marginal law — it does **not** require the coarse path to be a
deterministic function of the fine increments (that was sufficient for κ=0, not
necessary). Here the coarse `(ΔW^c, W^c_{·,1})` is constructed to have exactly
the κ=1 coarse covariance (plain increments sum exactly; `W^c_{j,1}` has the
exact split + conditionally-correct `I₁`), so `E[P^c_l] = E[P^f_{l-1}]` and the
sum telescopes to the κ=1 fine price. The fresh `Z` per coarse cell costs a
little coupling tightness (0.2–0.6% of one cell's variance) but introduces **no
bias**. The gate-check below verifies this directly.

## 6. Alternative considered and rejected

*κ=1 fine / κ=0 coarse* (let the coarse stay a pure convolution, skip the
nearest-cell treatment on the coarse level). Rejected: the coarse path is then a
κ=0 law, which differs from the κ=1 law of the fine at the level below, so
`E[P^c_l] ≠ E[P^f_{l-1}]` and the telescoping sum no longer collapses to the κ=1
price — a silent bias. Note the difference is *not* in the nearest-cell marginal
variance — BLP's optimal `b₁` is chosen so `g(b₁)²·dt = dt^{2H}/(2H)` matches it
exactly — but in the **correlation structure**: κ=0 forces
`corr(nearest-cell, ΔW) = 1`, whereas κ=1 has `≈0.745`, which changes the joint
law and every downstream convolution term (the discrete `Var(W̃_t)` then differs
by tens of percent, especially at early `t`). The coarse level **must** be κ=1.

## 7. Predicted outcome (state before measuring)

Following the antithetic/conditional pattern, the honest prediction is **a
constant-factor result, not a rate change**:

- **β ≈ 2H, unchanged.** The hybrid's strong rate is O(n^{-H}) independent of κ
  (BLP), so the level variance still decays like 2^{-2Hl}. κ=1 does **not**
  rescue the β<γ pathology. *Refute if β jumps toward 4H.*
- **Weak error / bias constant improves markedly** (κ=1 is far more accurate per
  grid point), so the adaptive driver should reach a given ε at a **coarser
  finest level L** than κ=0 — this is where any cost win would come from, not
  from β.
- **Level-variance constant improves**, but the coupling is already 99.5%
  tight, so the per-level variance *ratio* κ0/κ1 is expected to be modest
  (order 1–2×), unlike conditional MC's 4×.

Net expectation: κ=1 is the *least unfavourable* of the three estimators for
the cost crossover — it attacks the actual common Volterra error rather than
orthogonal noise — but probably still does not flip β. The interesting,
genuinely-open question is whether the bias-constant improvement (fewer levels)
moves the matched-accuracy cost crossover even with β unchanged.

## 8. Gate-check plan (G-H1 … G-H4)

Mirror the antithetic/conditional gates; **always compare at matched finest
level L** (identical-bias rule), never via the free-running driver.

- **G-H1 (exact / unbiased):** κ=1 coarse law correct — check `Var(W̃^c_i)`
  against the exact double-kernel isometry; telescoping consistency `< 1`; with
  η=0 the European price still matches Black–Scholes. **Two extra asserts the
  marginal check is blind to** (a wrong sub-cell index passes the marginal but
  breaks the coupling): (i) **coupling-tightness** — `Var(Y_l)/Var(P_f) ≲ 0.01`
  (or directly `Var(W̃_f − W̃_c)` tiny at coarse points); the correct build is
  ~0.001, a sub-cell swap gives ~0.8. (ii) **cross-covariance** —
  `Cov(ΔW^c_j, W^c_{j,1}) = (2h)^{H+½}/(H+½)` (validated in the design check);
  catches anchor/conditioning errors the `W^c_{j,1}` variance alone misses.
- **G-H2 (rate):** sweep H∈{0.05,0.10,0.20,0.35}; fit β for κ=0 and κ=1 vs 2H.
  Prediction: β_{κ1} ≈ β_{κ0} ≈ 2H.
- **G-H3 (variance factor):** per-level Var(Y_{κ0})/Var(Y_{κ1}); expected modest.
- **G-H4 (cost):** Giles optimal cost `(2/ε²)(Σ√(VₗCₗ))²` at matched L\* for
  κ=0 vs κ=1, plus standard MC. **Honest cost:** κ=1 adds one correlated
  Gaussian per cell on the fine path and one conditional draw per coarse cell;
  measure the per-path overhead and fold into Cₗ (as κ was measured for
  conditional MC).

## 9. Implementation sketch

- `volterra_weights_hybrid(n,H,T)` → existing `g,v` **plus** the per-cell pair
  covariance `(h, Var W_1, Cov)` and the convolution kernel with `g[0]:=0`.
- `_simulate_paths_kappa1(dW1, W1diag, dW2, n, p)` → like `_simulate_paths` but
  `W̃ = √(2H)(W1diag + conv(dW1, g_with_g0_zeroed))`; needs the nearest-cell
  draws `W1diag` passed in (so the level coupler controls them).
- `mlmc_asian_level(..., kappa=1)`: draw fine `(ΔW1, W1diag^f)` jointly; build
  coarse `ΔW1^c` by summation and the coarse nearest-cell draw by reusing the
  **right** sub-cell (whose right endpoint *is* the coarse endpoint) and
  conditionally resampling `I₁` on the **left** sub-cell.  **Index carefully:**
  with 0-indexed fine arrays a coarse cell `c` spans fine cells `[2c, 2c+1]`, so

  ```
  W1diag^c[c] = I₁[c] + W1diag^f[2c+1]          # reuse the RIGHT sub-cell (2c+1)
  I₁[c] = β · (ΔW1^f[2c], W1diag^f[2c]) + σ_c·Z  # condition on the LEFT (2c)
  ```

  (In the 1-indexed math of §4 the same statement is `W^c_{j,1}=I₁+W^f_{2j,1}`,
  reuse cell `2j`, condition on `2j-1`.) Reusing the *wrong* sub-cell still
  passes a marginal-variance check but destroys the coupling (`Var(Y_l)` stops
  shrinking) — see the G-H1 tightness assertion below. Asset uses plain
  increments throughout. `out[1]=P_f` as usual.
- Cost coefficient: extend `_level_cost_coef` with a `kappa1` branch once the
  overhead is measured (expected ≈1.1–1.3×, dominated by the extra Gaussians,
  the FFT being unchanged).
- Validation hooks: reuse `kappa1_coupling_design_check.py` as a unit test of
  the per-cell algebra before MLMC wiring.

## 10. Risks

- **Silent bias from a wrong covariance.** Mitigated: the per-cell algebra is
  validated to 1e-11, and G-H1 adds a direct `Var(W̃^c)` marginal check plus the
  telescoping consistency test (a coarse-law error shows up immediately).
- **Numerical edge:** `σ_c²` is small (~1e-3·h^{2H}); use `max(σ_c²,0)` and the
  Cholesky / linear-solve forms shown to avoid catastrophic cancellation.
- **Scope creep:** κ=1 composes with neither antithetic nor conditional in this
  design; treat as a third independent estimator axis.

**Recommendation:** proceed to code G-H1 first (fine κ=1 path + marginal
validation) in isolation, confirm `Var(W̃)` against the exact continuum and the
BS anchor, *then* wire the coarse coupler and run G-H2–G-H4. The math is
de-risked; the remaining risk is implementation, which G-H1 isolates.
