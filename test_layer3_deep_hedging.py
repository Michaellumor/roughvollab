"""
test_layer3_deep_hedging.py — SEPARATE Layer-3 suite (deep hedging).
====================================================================
STRICT ISOLATION: this suite is NOT part of the core CI. Run it with the isolated env
(`.venv-layer3/Scripts/python -m pytest test_layer3_deep_hedging.py`). Torch-dependent
tests SKIP if torch is absent, so the file is harmless if accidentally collected by the
core (torch-free) python — but the core's "full suite" command must NOT include it.

Numpy-only tests (signature, BS-delta, the causality guard, CVaR) run anywhere; the
training/gate tests require torch.
"""
import math
import numpy as np
import pytest

import layer3_deep_hedging as L

_needs_torch = pytest.mark.skipif(not L._HAS_TORCH, reason="torch not in this env (run in .venv-layer3)")


# ---- signature (numpy, no torch) ----
def test_signature_toy_known_answers():
    assert L.verify_signature_toy(verbose=False)


def test_signature_dim():
    assert L.signature_dim(3, 3) == 39 and L.signature_dim(3, 4) == 120
    assert L.signature(np.array([[0.0, 0.0], [1.0, 2.0]]), 3).shape == (2 + 4 + 8,)  # d=2


# ---- BS-delta vs analytic ----
def test_bs_delta_atm_and_limits():
    from scipy.special import ndtr
    sig = 0.2
    atm = L.bs_delta(100.0, 100.0, 1.0, sig)                 # Φ(½σ√T)
    assert abs(atm - ndtr(0.5 * sig)) < 1e-12
    assert L.bs_delta(200.0, 100.0, 1e-9, sig) > 0.999       # deep ITM, τ→0 → 1
    assert L.bs_delta(50.0, 100.0, 1e-9, sig) < 1e-3         # deep OTM, τ→0 → 0


# ---- ★ the causality guard MUST fire on a look-ahead policy (the Layer-2 trap) ----
def test_causality_guard_fires_on_lookahead():
    t, S, V = L.simulate_market(0.5, 0.0, 20000, 20, seed=1)   # GBM (martingale, r=0)
    N = 20
    # causal hedge (BS-delta uses S_k only) -> E[Σδ·ΔS]≈0, z<4
    sig = math.sqrt(L.XI0_)
    causal = np.stack([L.bs_delta(S[:, k], L.K_, L.T_ - t[k], sig) for k in range(N)], axis=1)
    ok_c, z_c = L.assert_causal(causal, S, label="causal", verbose=False)
    # look-ahead hedge (uses the FUTURE increment) -> E[Σ(ΔS)₊]>0, z huge
    peek = (S[:, 1:N + 1] > S[:, :N]).astype(float)
    ok_p, z_p = L.assert_causal(peek, S, label="peek", verbose=False)
    assert ok_c and z_c < 4.0, f"causal hedge flagged (z={z_c})"
    assert (not ok_p) and z_p > 4.0, f"look-ahead NOT caught (z={z_p})"


# ---- CVaR (Rockafellar–Uryasev sample form) ----
def test_cvar_np_known():
    losses_pnl = -np.arange(1.0, 101.0)                      # losses 1..100; worst 5% = 96..100
    assert abs(L.cvar_np(losses_pnl, 0.95) - 98.0) < 1e-9    # mean(96..100)=98


# ---- torch-gated ----
@_needs_torch
def test_torch_autograd_sanity():
    assert L.torch_autograd_sanity(verbose=False)


@_needs_torch
def test_gate1_tiny_recovers_delta_direction():
    """A tiny GBM training run must move the policy TOWARD BS-delta (sanity, not the full
    gate): the learned δ correlates strongly with Φ(d₁)."""
    import torch
    torch.manual_seed(0)
    N = 16
    t, S, V = L.simulate_market(0.5, 0.0, 4000, N, seed=3)
    from rough_heston_cf import bs_call
    sig = math.sqrt(L.XI0_); premium = bs_call(L.S0_, L.K_, L.T_, 0.0, sig)
    feats = L.build_features(t, S, V, mode="simple", N=N)
    policy, norm = L.train_policy(feats, S, premium=premium, K=L.K_, cost_c=0.0,
                                  epochs=60, seed=0, verbose=False)
    dl = L.policy_deltas(policy, feats, norm)
    bsd = np.stack([L.bs_delta(S[:, k], L.K_, L.T_ - t[k], sig) for k in range(N)], axis=1)
    corr = np.corrcoef(dl.ravel(), bsd.ravel())[0, 1]
    assert corr > 0.9, f"learned δ correlation with Φ(d₁) only {corr:.2f}"
