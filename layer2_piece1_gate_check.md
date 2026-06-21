# Layer 2 · Piece 1 — Classical Almgren–Chriss · gate-check spec

**Status:** spec · not yet built — acceptance is all 7 gates green (§5) · **Module:** `layer2_frictions.py` (Piece 1 portion) · **Tests:** `test_layer2_frictions.py`

This is a *gate-check* document in the project's sense: it fixes what gets built, the
known answer it is validated against, and the tests that confirm it — **before** any
code. Every closed-form target below was verified numerically (see §4 note); these are
checked ground truths, not recollections.

---

## 1. Purpose & place

Almgren–Chriss has a **closed-form solution**, so it is Layer 2's ground-truth anchor —
the exact analogue of "known true *H*" in Layer 1's corruption ladder. Piece 1 builds the
cost machinery and the optimal-execution solver for the **classical** case only: constant
volatility, linear impact, deterministic schedule. Nothing rough yet.

The gate is the same shape as Layer 1: a **general numerical optimiser** (the tool that
will generalise to the rough, no-closed-form cases later) must **reproduce the AC closed
form** to tolerance. Once that is locked, Pieces 2–4 perturb from a validated base instead
of from thin air.

---

## 2. The known answer (what we validate against)

### 2.1 Model

Liquidate `X` shares over `[0, T]` in `N` equal intervals, `tau = T/N`, `t_k = k*tau`.

- Holdings `x_0 = X`, `x_N = 0`; trades `n_k = x_{k-1} - x_k` for `k = 1..N` (so `sum n_k = X`).
- Arithmetic price with drift removed: `S_k = S_{k-1} + sigma*sqrt(tau)*xi_k - gamma*n_k`,
  `xi_k ~ N(0,1)` i.i.d. (`gamma` = linear **permanent** impact).
- Execution price per share in interval `k`: `S~_k = S_{k-1} - eta*(n_k/tau)`
  (`eta` = linear **temporary** impact; fixed spread term `eps` dropped — pure quadratic case).
- `sigma` is **absolute** volatility ($/share/√time), not relative.

### 2.2 Closed-form cost (any deterministic trajectory)

```
tau     = T / N
eta_t   = eta - 0.5 * gamma * tau           # "eta-tilde";  REQUIRE eta_t > 0

E[IS]   = 0.5 * gamma * X**2  +  (eta_t / tau) * sum_{k=1..N}   n_k**2
V[IS]   = sigma**2 * tau      *               sum_{k=1..N-1} x_k**2
```

`E[IS]` = expected implementation shortfall; `V[IS]` = its variance. The `0.5*gamma*X**2`
permanent-impact term is trajectory-independent (a sunk cost); the `-0.5*gamma*tau` in
`eta_t` is the standard cross-term correction (you pay half your own permanent impact).
`eta_t > 0` is the well-posedness / convexity condition.

### 2.3 Closed-form optimum (minimise `E[IS] + lam*V[IS]`)

```
k_t2    = lam * sigma**2 / eta_t                         # urgency^2 (continuous limit)
kappa   = (1/tau) * arccosh( 1 + 0.5 * k_t2 * tau**2 )    # discrete decay rate
x_k     = X * sinh(kappa*(T - t_k)) / sinh(kappa*T)       # k = 0..N      <-- THE trajectory

lam = 0  ->  kappa = 0  ->  x_k = X*(T - t_k)/T           # linear; n_k = X/N uniform
```

The **efficient frontier** is the locus of `(V[IS], E[IS])` as `lam` sweeps `[0, inf)`:
`lam = 0` gives the slowest sensible schedule (max `V`, min `E`); `lam -> inf` liquidates
immediately (`V -> 0`, max `E`). It is monotone (`E` falls as `V` rises) and convex.

### 2.4 Hand-checkable corollaries (the test anchors)

```
linear trajectory:    E = 0.5*gamma*X**2 + eta_t*X**2 / T
                      V = sigma**2 * tau * (X**2 / N**2) * (N-1)*N*(2N-1)/6
immediate liq (n_1=X): E = 0.5*gamma*X**2 + (eta_t/tau)*X**2 ,   V = 0
```

These are exact and independently computable — they are what Gate 1 checks against.

---

## 3. What we build (module API)

`layer2_frictions.py`, Piece-1 surface (numpy + scipy only, repo style — `__future__`
annotations, `__all__`, type hints, "why" comments):

| object | contract |
|---|---|
| `ACParams` (frozen dataclass) | fields `X, T, N, sigma, gamma, eta, lam`; derived `tau`, `eta_t`; **rejects `eta_t <= 0`** on construction |
| `expected_shortfall(x, p)` | `E[IS]` for holdings array `x` (length `N+1`, `x[0]=X`, `x[-1]=0`) — §2.2 |
| `shortfall_variance(x, p)` | `V[IS]` — §2.2 |
| `mv_objective(x, p)` | `E[IS] + lam*V[IS]` |
| `ac_kappa(p)` | `kappa` (returns `0.0` when `lam == 0`) |
| `ac_optimal_trajectory(p)` | the **analytical sinh** trajectory — §2.3 |
| `optimal_trajectory_numeric(p)` | the optimum found by **an independent route**: solve the convex quadratic in the interior holdings (exact linear solve, *not* the sinh formula) |
| `efficient_frontier(p, lambdas)` | arrays `(V, E)` over a `lam` grid |

**Critical design point.** `ac_optimal_trajectory` (closed form) and
`optimal_trajectory_numeric` (quadratic solve) must be computed by **genuinely different
routes**, or Gate 2 is a tautology that proves nothing. The numeric solver is also the
piece that *generalises*: in Pieces 3–4 the sinh formula no longer exists, but the same
"minimise the cost objective" solver carries over. Piece 1 is where you prove that solver
is correct, against the one case where the answer is known.

---

## 4. The gates (`test_layer2_frictions.py`)

Parametrise every gate over a grid of `(sigma, gamma, eta, lam, N, T)` — not one lucky
config. All targets below were confirmed numerically before this spec was written
(headline agreements: sinh vs numeric optimum `5e-7` under BFGS → machine precision with an
exact solve; `lam=0` vs linear `1e-16`; both cost formulas exact; frontier monotone;
2000-draw perturbation test clean).

| gate | checks | target | tol |
|---|---|---|---|
| **G1** | `E[IS]`, `V[IS]` on linear & immediate-liquidation trajectories | §2.4 corollaries | `1e-10` rel |
| **G2** | `optimal_trajectory_numeric` == `ac_optimal_trajectory` | identical optimum, two routes | `1e-8` (exact solve) |
| **G3** | `lam = 0` ⇒ linear trajectory | `x_k = X*(T-t_k)/T` | `1e-12` |
| **G4** | trajectory structure | `x[0]=X`, `x[-1]=0`, monotone non-increasing, convex (front-loaded) for `lam>0` | — |
| **G5** | efficient frontier | `V` strictly down & `E` strictly up as `lam` rises; `lam→0` endpoint matches the §2.4 linear closed form | — |
| **G6** | independent optimality | random interior perturbations of `ac_optimal_trajectory` raise `mv_objective` | over N draws |
| **G7** | validity guard | constructing `ACParams` with `eta - 0.5*gamma*tau <= 0` raises | — |

---

## 5. Acceptance

Piece 1 is **done** when all seven gates are green across the parametrised grid — and not
before. No rough variance path, no Monte-Carlo simulator, no Piece 2, until the classical
anchor is locked. One validated thing at a time.

---

## 6. Forward link (why this is the anchor, not busywork)

Piece 2 replaces the constant-`sigma` arithmetic walk with a **rough-Bergomi variance
path** from `roughvol_core` (the engine Layer 1 already uses) and runs the schedule against
it by Monte-Carlo. Its gate will be: **in the constant-volatility limit, the simulated
shortfall's empirical mean and variance reproduce §2.2's `E[IS]` and `V[IS]`.**

So Piece 1's formulas *are* Piece 2's validation target — the same closed forms, re-checked
against simulation. That is how the chain stays anchored all the way to the non-Markovian
result: every new layer is pinned, in some limit, to something already proven.

---

## 7. Reference

Almgren, R. & Chriss, N. (2001). *Optimal execution of portfolio transactions.* Journal of
Risk 3(2): 5–40. — Source of the closed-form trajectory, cost, and frontier in §2.
Rough-impact link (Piece 3 onward): Gatheral, Jaisson & Rosenbaum (2018).
