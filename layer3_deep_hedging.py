"""
layer3_deep_hedging.py — deep-hedging engine (Layer 3, EXPLORATORY).
=====================================================================
Project: RoughVolLab. The deep-HEDGING engine (distinct from the Layer-2 execution
arc). The sharpened, on-theme question: deep hedging beats delta under frictions in
ANY model (a generic Buehler frictions effect) — so does the ROUGH structure add a
hedging edge BEYOND the frictions edge? The experiment is a CONTRAST: the deep-vs-delta
CVaR edge on a ROUGH market (H=0.10) vs the SAME edge on a SMOOTH/Markovian control
(H=0.5, same generator/params — only H differs). The roughness-specific result is
(rough-edge − smooth-edge). Prior (Layer 2 = no execution edge; D37–D39 = H barely
identifiable): the roughness increment is MODEST/ABSENT. A tight positive is a red flag.

╔══════════════════════════════════════════════════════════════════════════════════╗
║ STRICT ISOLATION (non-negotiable) — Layer 3 is a downstream LEAF.                  ║
║  • It IMPORTS the validated core (roughvol_core, rough_heston_cf) to GENERATE      ║
║    paths; NOTHING in the core imports Layer 3.                                     ║
║  • Its tests (test_layer3_deep_hedging.py) are a SEPARATE suite that NEVER gates   ║
║    the core CI — the core's full suite stays torch-free.                           ║
║  • torch is a GUARDED optional import (the repo's first ML dep), installed ONLY in ║
║    .venv-layer3. Run Layer 3 with that venv's python.                             ║
║  • DELETION-SAFE: delete this module / nuke the venv → the core is untouched.      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Refs: Buehler, Gonon, Teichmann & Wood (2019) deep hedging; Kidger & Lyons (signatures
for ML practice). The signature is self-computed (signatory is unmaintained on py3.14)
and verified against a hand-computed toy answer before it is trusted.
"""
import math
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- guarded optional ML dependency (the repo's first; isolated to .venv-layer3) --- #
try:
    import torch
    _HAS_TORCH = True
except Exception:                                       # pragma: no cover - env-dependent
    torch = None
    _HAS_TORCH = False


def require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "Layer 3 needs PyTorch — run with the isolated env: "
            "`.venv-layer3/Scripts/python ...` (the core env stays torch-free).")


# --------------------------------------------------------------------------- #
# Truncated path signature (self-computed; numpy-only — no torch, no signatory)
# --------------------------------------------------------------------------- #
# The signature of a piecewise-linear path is the ordered tensor product (Chen) of the
# segment exponentials exp_⊗(Δ) = (1, Δ, Δ⊗Δ/2!, …). We keep it as a list of level
# arrays [scalar, (d,), (d,d), …] truncated at `depth`, then flatten levels 1..depth.
def _exp_sig(delta, depth):
    out = [np.array(1.0)]
    term = np.array(1.0)
    for m in range(1, depth + 1):
        term = np.multiply.outer(term, delta)           # Δ^{⊗m}
        out.append(term / math.factorial(m))
    return out


def _chen(A, B, depth):
    """Truncated tensor-algebra product: (A⊗B)_m = Σ_{i=0}^{m} A_i ⊗ B_{m−i}."""
    out = []
    for m in range(depth + 1):
        s = np.multiply.outer(A[0], B[m])
        for i in range(1, m + 1):
            s = s + np.multiply.outer(A[i], B[m - i])
        out.append(np.asarray(s))
    return out


def signature(path, depth):
    """Truncated signature of a (L, d) sample path, flattened over levels 1..depth.
    Length = Σ_{m=1}^{depth} d^m. Level 0 (the constant 1) is dropped."""
    path = np.asarray(path, float)
    d = path.shape[1]
    incs = np.diff(path, axis=0)
    sig = [np.array(1.0)] + [np.zeros((d,) * m) for m in range(1, depth + 1)]
    for delta in incs:
        sig = _chen(sig, _exp_sig(delta, depth), depth)
    return np.concatenate([np.asarray(sig[m]).ravel() for m in range(1, depth + 1)])


def signature_dim(d, depth):
    return sum(d ** m for m in range(1, depth + 1))


# --------------------------------------------------------------------------- #
# Phase-0 verification — KNOWN ANSWERS, before anything is trusted
# --------------------------------------------------------------------------- #
def verify_signature_toy(verbose=True):
    """Verify the signature implementation against hand-computed known answers."""
    ok = True
    # (1) straight line 0 -> (a,b) in R^2: sig = exp_⊗(Δ) exactly.
    a, b = 0.7, -0.3
    sig = signature([[0.0, 0.0], [a, b]], depth=3)
    lvl1 = np.array([a, b])
    lvl2 = np.outer([a, b], [a, b]) / 2.0
    lvl3 = np.multiply.outer(np.outer([a, b], [a, b]), [a, b]) / 6.0
    expect = np.concatenate([lvl1, lvl2.ravel(), lvl3.ravel()])
    e1 = float(np.max(np.abs(sig - expect)))
    ok &= e1 < 1e-12
    # (2) reparam/Chen invariance: splitting the straight line into 2 collinear segments
    #     gives the IDENTICAL signature.
    sig2 = signature([[0.0, 0.0], [0.4 * a, 0.4 * b], [a, b]], depth=3)
    e2 = float(np.max(np.abs(sig2 - sig)))
    ok &= e2 < 1e-12
    # (3) level-1 = endpoint − start (always); signed-area = ½(S^{12} − S^{21}) for a
    #     triangle 0->(1,0)->(1,1)->0 has area ½ -> antisymmetric level-2 part = area.
    tri = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    st = signature(tri, depth=2)
    lvl1_t = st[:2]; s12, s21 = st[2 + 1], st[2 + 2]     # level-2 is [s11,s12,s21,s22]
    area = 0.5 * (s12 - s21)
    e3 = float(abs(lvl1_t[0]) + abs(lvl1_t[1]))          # closed loop -> endpoint==start
    ok &= e3 < 1e-12 and abs(area - 0.5) < 1e-12
    if verbose:
        print(f"  signature toy checks: exp-tensor err={e1:.1e}  reparam err={e2:.1e}  "
              f"loop-endpoint={e3:.1e}  signed-area={area:.6f} (exp 0.5)  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "signature toy verification FAILED"
    return ok


def torch_autograd_sanity(verbose=True):
    """Trust the autodiff stack: d/dx (Σ xᵢ²) = 2x at a known point."""
    require_torch()
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    (x.pow(2).sum()).backward()
    err = float((x.grad - 2 * x.detach()).abs().max())
    if verbose:
        print(f"  torch autograd sanity: ∇‖x‖² err={err:.1e} (torch {torch.__version__}) "
              f"-> {'PASS' if err < 1e-6 else 'FAIL'}")
    assert err < 1e-6, "torch autograd sanity FAILED"
    return True


# --------------------------------------------------------------------------- #
# Market: GBM complete-market limit (Gate 1) + rough/smooth (Gate 2 contrast)
# --------------------------------------------------------------------------- #
S0_, K_, T_, XI0_ = 100.0, 100.0, 1.0, 0.04             # ATM call; ξ₀=0.04 -> σ_BS=0.2


def simulate_market(H, eta, n_paths, N, *, seed, xi0=XI0_, rho=-0.7, S0=S0_, T=T_):
    """(t, S, V). eta==0 -> EXACT GBM (constant vol √xi0; the complete-market limit where
    BS-delta is the exact hedge — Gate 1). eta>0 -> rough_bergomi_paths at the given H
    (Gate 2: rough H=0.10 vs smooth Markovian H=0.5)."""
    if eta == 0.0:
        rng = np.random.default_rng(seed)
        dt = T / N
        sig = math.sqrt(xi0)
        dW = rng.standard_normal((n_paths, N)) * math.sqrt(dt)
        logS = np.concatenate([np.zeros((n_paths, 1)),
                               np.cumsum(-0.5 * sig * sig * dt + sig * dW, axis=1)], axis=1)
        S = S0 * np.exp(logS)
        V = np.full((n_paths, N + 1), xi0)
        t = np.linspace(0.0, T, N + 1)
        return t, S, V
    from roughvol_core import rough_bergomi_paths
    return rough_bergomi_paths(N, H, n_paths, T=T, eta=eta, rho=rho, xi0=xi0, S0=S0, r=0.0,
                               rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------- #
# Causal features: running signature of the (t,S,V) path so far ++ absolute state.
# The signature is translation-invariant (path SHAPE / history); BS-delta needs the
# ABSOLUTE (t, S_t, √V_t), so that is always appended. simple mode = absolute-only
# (the Markovian control); sig mode = signature(history) ++ absolute (adds path memory).
# Both are CAUSAL: feature at step k uses path[0:k+1] only.
# --------------------------------------------------------------------------- #
def _running_signatures(aug, depth):
    """aug: (n_paths, L, d). Returns (n_paths, L, sigdim) — signature of aug[:, :k+1]."""
    b, L, d = aug.shape
    sig = [np.ones((b, 1))] + [np.zeros((b, d ** m)) for m in range(1, depth + 1)]
    out = np.zeros((b, L, signature_dim(d, depth)))                  # t=0 -> zeros (empty path)
    for k in range(1, L):
        delta = aug[:, k, :] - aug[:, k - 1, :]
        # exp_⊗(delta), batched
        es = [np.ones((b, 1))]; term = np.ones((b, 1))
        for m in range(1, depth + 1):
            term = (term[:, :, None] * delta[:, None, :]).reshape(b, -1)
            es.append(term / math.factorial(m))
        # Chen: sig <- sig ⊗ es
        new = []
        for m in range(depth + 1):
            acc = (sig[0][:, :, None] * es[m][:, None, :]).reshape(b, -1)
            for i in range(1, m + 1):
                acc = acc + (sig[i][:, :, None] * es[m - i][:, None, :]).reshape(b, -1)
            new.append(acc)
        sig = new
        out[:, k, :] = np.concatenate([sig[m] for m in range(1, depth + 1)], axis=1)
    return out


def build_features(t, S, V, *, mode="sig", depth=3, N=None):
    """Causal features at rebalancing steps 0..N-1. Returns (n_paths, N, fdim).
    Absolute state [t/T, S/S0, √V] is ALWAYS included; sig mode prepends the running
    signature of the (t,S,V) path."""
    b, L = S.shape
    N = (L - 1) if N is None else N
    tt = np.broadcast_to(t / t[-1], (b, L))
    absstate = np.stack([tt, S / S0_, np.sqrt(np.maximum(V, 0.0))], axis=-1)   # (b,L,3)
    if mode == "simple":
        feats = absstate
    elif mode == "sig":
        aug = np.stack([np.broadcast_to(t, (b, L)), S, V], axis=-1)            # raw (t,S,V) path
        rs = _running_signatures(aug, depth)
        feats = np.concatenate([rs, absstate], axis=-1)
    else:
        raise ValueError(mode)
    return feats[:, :N, :]                                                     # decide δ at steps 0..N-1


# --------------------------------------------------------------------------- #
# P&L (causal, self-financing), CVaR (Rockafellar–Uryasev), BS-delta baseline
# --------------------------------------------------------------------------- #
def bs_delta(S, K, tau, sigma):
    """Φ(d₁) — call delta. tau, sigma scalars or arrays; tau→0 handled (→ 1{S>K})."""
    from scipy.special import ndtr
    S = np.asarray(S, float); tau = np.maximum(np.asarray(tau, float), 1e-12)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return ndtr(d1)


def pnl_from_deltas(deltas, S, *, premium, K, cost_c):
    """deltas (n_paths, N) held over [t_k, t_{k+1}); terminal close at T. Returns P&L per path.
    P&L = premium + Σ δ_k (S_{k+1}-S_k) − (S_N−K)+ − cost_c·Σ S_k|δ_k−δ_{k-1}| (open+rebal+close)."""
    xp = torch if (_HAS_TORCH and isinstance(deltas, torch.Tensor)) else np
    n, N = deltas.shape
    Sn = S if xp is np else torch.as_tensor(S, dtype=deltas.dtype)
    gains = (deltas * (Sn[:, 1:N + 1] - Sn[:, :N])).sum(axis=1) if xp is np \
        else (deltas * (Sn[:, 1:N + 1] - Sn[:, :N])).sum(dim=1)
    payoff = xp.clip(Sn[:, N] - K, 0.0, None) if xp is np else torch.clamp(Sn[:, N] - K, min=0.0)
    prev = (xp.zeros((n, 1)) if xp is np else torch.zeros((n, 1), dtype=deltas.dtype))
    d_aug = xp.concatenate([prev, deltas], axis=1) if xp is np else torch.cat([prev, deltas], dim=1)
    turn = (xp.abs(d_aug[:, 1:] - d_aug[:, :-1]) if xp is np else (d_aug[:, 1:] - d_aug[:, :-1]).abs())
    cost_rebal = (cost_c * (Sn[:, :N] * turn)).sum(axis=1) if xp is np \
        else (cost_c * (Sn[:, :N] * turn)).sum(dim=1)
    cost_close = cost_c * Sn[:, N] * (xp.abs(deltas[:, -1]) if xp is np else deltas[:, -1].abs())
    return premium + gains - payoff - cost_rebal - cost_close


def cvar_loss(pnl, alpha, w):
    """Rockafellar–Uryasev CVaR of the LOSS L=−P&L: w + 1/(1−α)·E[(L−w)₊] (min over net,w)."""
    L = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(L - w, min=0.0).mean()


def cvar_np(pnl, alpha):
    """Sample CVaR of the loss (mean of the worst (1−α) tail) — for reporting."""
    L = np.sort(-np.asarray(pnl))
    k = max(1, int(round((1.0 - alpha) * len(L))))
    return float(L[-k:].mean())


# --------------------------------------------------------------------------- #
# Policy (Buehler-style direct optimisation: one shared MLP, NO value network)
# --------------------------------------------------------------------------- #
def _make_policy(fdim, hidden=32):
    require_torch()
    return torch.nn.Sequential(
        torch.nn.Linear(fdim, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1))


def train_policy(feats, S, *, premium, K, cost_c, alpha=0.95, epochs=300, lr=1e-3,
                 batch=4096, seed=0, hidden=32, verbose=True):
    """Direct CVaR (Rockafellar–Uryasev) minimisation of terminal hedging P&L. Features are
    z-scored on the TRAIN set. Returns (policy, (mean,std,fdim))."""
    require_torch()
    torch.manual_seed(seed)
    b, N, fdim = feats.shape
    mean = feats.reshape(-1, fdim).mean(0); std = feats.reshape(-1, fdim).std(0) + 1e-8
    F = torch.as_tensor((feats - mean) / std, dtype=torch.float32)
    St = torch.as_tensor(S, dtype=torch.float32)
    policy = _make_policy(fdim, hidden)
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(policy.parameters()) + [w], lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(b)
        for i in range(0, b, batch):
            idx = perm[i:i + batch]
            deltas = policy(F[idx]).squeeze(-1)              # (batch, N)
            pnl = pnl_from_deltas(deltas, St[idx], premium=premium, K=K, cost_c=cost_c)
            loss = cvar_loss(pnl, alpha, w)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % max(1, epochs // 8) == 0:
            with torch.no_grad():
                pnl = pnl_from_deltas(policy(F).squeeze(-1), St, premium=premium, K=K, cost_c=cost_c)
            print(f"    epoch {ep+1}/{epochs}: CVaR95={cvar_np(pnl.numpy(), alpha):.4f}  "
                  f"meanPnL={float(pnl.mean()):+.4f}  std={float(pnl.std()):.4f}", flush=True)
    return policy, (mean, std, fdim)


def policy_deltas(policy, feats, norm):
    mean, std, _ = norm
    F = torch.as_tensor((feats - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return policy(F).squeeze(-1).numpy()


# --------------------------------------------------------------------------- #
# Causality guard (the look-ahead trap that bit Layer 2): a CAUSAL hedge of a
# martingale underlying (r=0) has E[Σ δ_k ΔS_k] ≈ 0; a look-ahead hedge breaks it.
# --------------------------------------------------------------------------- #
def assert_causal(deltas, S, *, label="", z_max=4.0, verbose=True):
    n, N = deltas.shape
    gains = (deltas * (S[:, 1:N + 1] - S[:, :N])).sum(axis=1)
    m = float(gains.mean()); se = float(gains.std(ddof=1) / math.sqrt(n)); z = abs(m) / max(se, 1e-12)
    ok = z < z_max
    if verbose:
        print(f"  causality [{label}]: E[Σδ·ΔS]={m:+.4f} (s.e.{se:.4f}, z={z:.1f}) "
              f"-> {'CAUSAL' if ok else 'LOOK-AHEAD!'}")
    return ok, z


# --------------------------------------------------------------------------- #
# GATE 1 — recover BS-delta in the frictionless complete-market (GBM) limit
# --------------------------------------------------------------------------- #
def gate1_recover_delta(mode="sig", depth=3, n_train=20000, n_test=20000, N=20,
                        epochs=300, seed=0, tol=0.06):
    from rough_heston_cf import bs_call
    sigma = math.sqrt(XI0_); premium = bs_call(S0_, K_, T_, 0.0, sigma)
    print(f"=== GATE 1: recover BS-delta, GBM limit (mode={mode}, depth={depth}, N={N}, "
          f"train={n_train}) | σ_BS={sigma:.3f} premium={premium:.4f} ===", flush=True)
    t, S, V = simulate_market(0.5, 0.0, n_train, N, seed=seed)            # eta=0 -> exact GBM
    feats = build_features(t, S, V, mode=mode, depth=depth, N=N)
    policy, norm = train_policy(feats, S, premium=premium, K=K_, cost_c=0.0,
                                epochs=epochs, seed=seed)
    tt, St, Vt = simulate_market(0.5, 0.0, n_test, N, seed=seed + 777)    # disjoint test
    dl = policy_deltas(policy, build_features(tt, St, Vt, mode=mode, depth=depth, N=N), norm)
    bsd = np.stack([bs_delta(St[:, k], K_, T_ - tt[k], sigma) for k in range(N)], axis=1)
    err = np.abs(dl - bsd)
    pnl_dl = pnl_from_deltas(dl, St, premium=premium, K=K_, cost_c=0.0)
    pnl_bs = pnl_from_deltas(bsd, St, premium=premium, K=K_, cost_c=0.0)
    print(f"  learned δ vs Φ(d₁): mean|Δ|={err.mean():.4f}  max|Δ|={err.max():.4f}  "
          f"(over {n_test}×{N} states)")
    print(f"  hedged P&L std: learned={pnl_dl.std():.4f}  BS-delta={pnl_bs.std():.4f}  "
          f"(GBM frictionless → ~0)")
    print(f"  CVaR95(loss):   learned={cvar_np(pnl_dl,0.95):.4f}  BS-delta={cvar_np(pnl_bs,0.95):.4f}")
    assert_causal(dl, St, label="learned policy")
    ratio = float(pnl_dl.std() / max(pnl_bs.std(), 1e-9))
    print(f"  P&L-std ratio learned/BS = {ratio:.2f}  (delta is variance-optimal frictionless → target ~1; <1 would be a red flag)")
    ok = (float(err.mean()) < tol) and (ratio < 1.20)
    print(f"  -> GATE 1 {'PASS' if ok else 'FAIL'}  (mean|δ−Φ(d₁)|<{tol} AND P&L-std ratio<1.20)", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# GATE 2 — under frictions, does deep beat delta on CVaR? + the ROUGH-vs-SMOOTH
# contrast (the on-theme question: roughness-edge = rough-edge − smooth-edge)
# --------------------------------------------------------------------------- #
def hedge_edge(H, eta, seed, *, cost_c, alpha=0.95, mode="sig", depth=3,
               n_train=15000, n_test=20000, N=20, epochs=150, hidden=32):
    """Train the deep policy on the (H, eta) market WITH frictions; return test-set
    CVaR(deep) vs CVaR(delta-baseline σ=√ξ₀). edge = CVaR(delta) − CVaR(deep) (>0 = deep wins)."""
    from rough_heston_cf import bs_call
    sigma0 = math.sqrt(XI0_); premium = bs_call(S0_, K_, T_, 0.0, sigma0)
    t, S, V = simulate_market(H, eta, n_train, N, seed=seed)
    feats = build_features(t, S, V, mode=mode, depth=depth, N=N)
    policy, norm = train_policy(feats, S, premium=premium, K=K_, cost_c=cost_c,
                                alpha=alpha, epochs=epochs, seed=seed, hidden=hidden, verbose=False)
    tt, St, Vt = simulate_market(H, eta, n_test, N, seed=seed + 777)
    dl = policy_deltas(policy, build_features(tt, St, Vt, mode=mode, depth=depth, N=N), norm)
    bsd = np.stack([bs_delta(St[:, k], K_, T_ - tt[k], sigma0) for k in range(N)], axis=1)
    pnl_deep = pnl_from_deltas(dl, St, premium=premium, K=K_, cost_c=cost_c)
    pnl_delta = pnl_from_deltas(bsd, St, premium=premium, K=K_, cost_c=cost_c)
    cvd, cvb = cvar_np(pnl_deep, alpha), cvar_np(pnl_delta, alpha)
    return dict(cv_deep=cvd, cv_delta=cvb, edge=cvb - cvd)


def gate2_contrast(seeds, *, eta=1.0, cost_c=0.005, alpha=0.95, mode="sig", depth=3,
                   n_train=15000, n_test=20000, N=20, epochs=150):
    print(f"=== GATE 2 + CONTRAST: deep vs delta CVaR{int(alpha*100)} under frictions "
          f"(c={cost_c}, eta={eta}, mode={mode}, {len(seeds)} seeds) ===", flush=True)
    print(f"  edge = CVaR(delta) − CVaR(deep)  (>0 = deep wins); roughness-edge = rough − smooth", flush=True)
    re, se_, incr = [], [], []
    for s in seeds:
        r = hedge_edge(0.10, eta, s, cost_c=cost_c, alpha=alpha, mode=mode, depth=depth,
                       n_train=n_train, n_test=n_test, N=N, epochs=epochs)
        m = hedge_edge(0.50, eta, s, cost_c=cost_c, alpha=alpha, mode=mode, depth=depth,
                       n_train=n_train, n_test=n_test, N=N, epochs=epochs)
        re.append(r["edge"]); se_.append(m["edge"]); incr.append(r["edge"] - m["edge"])
        print(f"  seed {s:3d}: ROUGH deep={r['cv_deep']:.3f} delta={r['cv_delta']:.3f} edge={r['edge']:+.3f} "
              f"| SMOOTH deep={m['cv_deep']:.3f} delta={m['cv_delta']:.3f} edge={m['edge']:+.3f} "
              f"| Δ(rough−smooth)={r['edge']-m['edge']:+.3f}", flush=True)
    re, se_, incr = np.array(re), np.array(se_), np.array(incr)
    def stat(x): return x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    rm, rse = stat(re); sm, sse = stat(se_); im, ise = stat(incr)
    print(f"\n  ROUGH  edge (deep beats delta): {rm:+.3f} ± {rse:.3f}  [>0 in {int((re>0).sum())}/{len(re)} seeds]", flush=True)
    print(f"  SMOOTH edge (deep beats delta): {sm:+.3f} ± {sse:.3f}  [>0 in {int((se_>0).sum())}/{len(se_)} seeds]", flush=True)
    print(f"  ★ ROUGHNESS INCREMENT (rough−smooth): {im:+.3f} ± {ise:.3f}  (z={abs(im)/max(ise,1e-9):.1f}, "
          f"sign>0 in {int((incr>0).sum())}/{len(incr)} seeds)", flush=True)
    verdict = ("MODEST/ABSENT — roughness adds little hedging edge beyond frictions (PREDICTED)"
               if abs(im) / max(ise, 1e-9) < 2.0 or abs(im) < 0.5 * abs(rm)
               else "SIGNIFICANT — scrutinise for a bug (leakage/unfair baseline) before believing it")
    print(f"  => roughness-specific hedging edge: {verdict}", flush=True)
    return dict(rough=re, smooth=se_, incr=incr)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--time1", action="store_true", help="time ONE training (runtime estimate)")
    ap.add_argument("--mode", default="sig", choices=["sig", "simple"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cost", type=float, default=0.01)
    a = ap.parse_args()
    if a.time1:
        t0 = time.time()
        r = hedge_edge(0.10, 1.0, 0, cost_c=0.005, mode=a.mode, epochs=a.epochs,
                       n_train=a.n, n_test=20000)
        dt = time.time() - t0
        print(f"ONE rough training+eval ({a.mode}, n={a.n}, epochs={a.epochs}): {dt:.0f}s  edge={r['edge']:+.3f}")
        print(f"GATE 2 estimate (2 markets × {a.seeds} seeds = {2*a.seeds} trainings): ~{2*a.seeds*dt/60:.0f} min")
    elif a.gate2:
        gate2_contrast(list(range(a.seeds)), mode=a.mode, epochs=a.epochs, n_train=a.n, cost_c=a.cost)
    elif a.gate1:
        gate1_recover_delta(mode=a.mode, n_train=a.n, n_test=a.n, epochs=a.epochs)
    else:
        print(f"Layer 3 Phase-0 verification (torch available: {_HAS_TORCH})")
        print(f"  signature dim (d=3, depth=3) = {signature_dim(3, 3)}  | depth=4 = {signature_dim(3, 4)}")
        verify_signature_toy()
        if _HAS_TORCH:
            torch_autograd_sanity()
        print("Phase 0 OK.")
