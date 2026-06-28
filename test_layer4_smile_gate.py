"""
test_layer4_smile_gate.py — mechanics tests for the OTM-smile gate (SPX prereq).
Guard the BS-IV machinery (parity, round-trip, vega) and that the smile gate runs
with an anchored ATM. The measured smile result (lifted vs CF, the (B) wing
divergence) lives in ROADMAP D36.
"""
import numpy as np

from rough_heston_cf import bs_call, bs_put, bs_iv, bs_vega


def test_bs_put_call_parity():
    for K in (80.0, 100.0, 120.0):
        c = bs_call(100.0, K, 1.0, 0.0, 0.25); p = bs_put(100.0, K, 1.0, 0.0, 0.25)
        assert abs((c - p) - (100.0 - K)) < 1e-10


def test_bs_iv_round_trip():
    """bs_iv must recover the input vol across strikes (OTM inversion incl. parity)."""
    for K in (70.0, 85.0, 100.0, 115.0, 130.0):
        px = bs_call(100.0, K, 1.0, 0.0, 0.23)
        assert abs(bs_iv(px, 100.0, K, 1.0, 0.0) - 0.23) < 1e-6


def test_bs_iv_below_intrinsic_is_nan():
    assert np.isnan(bs_iv(0.0, 100.0, 90.0, 1.0, 0.0))


def test_bs_vega_positive_peaks_near_atm():
    v = [bs_vega(100.0, K, 1.0, 0.0, 0.20) for K in (80.0, 100.0, 120.0)]
    assert all(x > 0 for x in v) and v[1] >= v[0] and v[1] >= v[2]


def test_smile_gate_runs_and_atm_anchored():
    """Gate runs end-to-end; ATM IV diff is anchored (~0, loose at small M)."""
    from layer4_smile_gate import smile_gate
    out = smile_gate(H=0.10, nu_list=(0.40,), N=60, n=128, M=20000,
                     vus=(-1.0, -0.5, 0.0, 0.5, 1.0), plot=False)
    vus, iv_cf, iv_lift, diff, rel = out[0.40]
    assert np.isfinite(iv_cf).all() and np.isfinite(iv_lift).all()
    atm = abs(diff[np.argmin(np.abs(vus))])
    assert atm < 1.0, f"ATM IV diff {atm:.2f}pp not anchored"
