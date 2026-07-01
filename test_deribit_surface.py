"""
test_deribit_surface.py — offline mechanics tests for the Deribit fetcher/cleaner.
A FAKE fetcher (in-memory canned JSON) keeps tests independent of the live, ever-changing
chain. Guards: HTTP-layer parsing + error handling, parse-by-field, forward-normalization,
each cleaning filter, the de-dup OTM rule, stacking-order/lockstep, snapshot round-trip.
The measured real-market result lives in ROADMAP D39.
"""
import json
import numpy as np

import deribit_surface as D
from deribit_surface import OptionQuote


# ---- HTTP layer (injectable fetcher) ----
def test_api_get_parses_result_and_raises_on_error():
    ok = lambda url, t: json.dumps({"result": {"x": 1}}).encode()
    assert D._api_get("m", {}, fetcher=ok) == {"x": 1}
    err = lambda url, t: json.dumps({"error": {"code": -1, "message": "no"}}).encode()
    try:
        D._api_get("m", {}, fetcher=err); assert False, "should raise"
    except D.DeribitError:
        pass


# ---- parse from FIELDS (not the instrument name) ----
def _inst(name="BTC-31JUL26-60000-P", strike=60000.0, opt="put", exp_ms=2_592_000_000):
    return {"instrument_name": name, "strike": strike, "option_type": opt,
            "expiration_timestamp": exp_ms}


def _ob(mark=44.0, bid=43.5, ask=44.5, vega=70.0, delta=-0.45, fwd=60000.0, spot=59900.0, oi=100.0):
    return {"mark_iv": mark, "bid_iv": bid, "ask_iv": ask, "underlying_price": fwd,
            "index_price": spot, "open_interest": oi, "greeks": {"vega": vega, "delta": delta},
            "stats": {"volume": 5.0}}


def test_parse_instrument_uses_fields_and_computes_T():
    q = D.parse_instrument(_inst(), _ob(), now_ms=0)
    assert q.strike == 60000.0 and q.opt_type == "put"           # from fields, not name
    assert abs(q.T_years - 2_592_000_000 / 1000 / (365 * 86400)) < 1e-9   # ~30d -> 0.0822
    assert q.mark_iv == 44.0 and q.vega == 70.0 and q.delta == -0.45


def test_forward_normalization():
    q = OptionQuote("x", 0, 0.08, 54000.0, "put", 60000.0, 59900.0, 40.0, 39.5, 40.5, 60.0, -0.3, 100.0, 5.0)
    kept, _ = D.clean([q])
    assert kept and abs(kept[0].K_norm - 90.0) < 1e-9            # 100*54000/60000


def _q(strike=60000.0, opt="put", fwd=60000.0, mark=44.0, bid=43.5, ask=44.5, vega=70.0,
       delta=-0.45, oi=100.0, T=0.08):
    return OptionQuote("BTC-X", 1, T, strike, opt, fwd, fwd - 100, mark, bid, ask, vega, delta, oi, 5.0)


def test_clean_each_filter():
    cases = {
        "illiquid_oi":   _q(oi=1.0),
        "not_two_sided": _q(bid=None),
        "wide_spread":   _q(bid=40.0, ask=50.0),                 # 10pt spread > 5
        "low_vega":      _q(vega=1.0),
        "outside_region":_q(delta=-0.02),                        # 2Δ put, outside [−0.5,−0.1] & ATM
        "mark_iv_insane":_q(mark=300.0),
    }
    for reason, q in cases.items():
        kept, drops = D.clean([q])
        assert kept == [] and reason in drops, f"{reason}: kept={len(kept)} drops={drops}"
    keep = _q(delta=-0.40)                                       # 40Δ put, in region
    kept, drops = D.clean([keep])
    assert len(kept) == 1 and not drops


def test_clean_dedup_keeps_otm_put():
    p = _q(strike=54000.0, opt="put", delta=-0.30)               # K_norm=90 (<100 -> put is OTM)
    c = _q(strike=54000.0, opt="call", delta=0.45)               # same strike, call (ITM)
    kept, drops = D.clean([p, c])
    assert len(kept) == 1 and kept[0].opt_type == "put" and drops.get("dup_strike") == 1


def test_to_grids_lockstep_sorted_and_normalized():
    qs = [_q(strike=66000.0, opt="call", delta=0.40), _q(strike=54000.0, opt="put", delta=-0.30),
          _q(strike=60000.0, opt="put", delta=-0.50)]
    kept, _ = D.clean(qs)
    grids, target, weights, meta = D.to_grids_target_weights(kept)
    T = next(iter(grids))
    K = grids[T]
    assert np.all(np.diff(K) > 0)                                 # sorted ascending by K_norm
    assert len(K) == len(target[T]) == len(weights[T])            # lockstep
    assert np.isclose(K[0], 90.0) and np.isclose(K[-1], 110.0)    # 54k/60k/66k -> 90/100/110
    assert np.allclose(target[T], 0.44)                           # mark_iv 44% -> 0.44 decimal


def test_stacking_order_matches_engine_concatenation():
    """Target must stack as np.concatenate([... for T in sorted(grids)]) — the engine's order."""
    qs = [_q(strike=60000.0, opt="put", delta=-0.50, T=0.50),
          _q(strike=54000.0, opt="put", delta=-0.30, T=0.08),
          _q(strike=60000.0, opt="put", delta=-0.50, T=0.08)]
    kept, _ = D.clean(qs)
    grids, target, weights, _ = D.to_grids_target_weights(kept)
    stacked = np.concatenate([target[T] for T in sorted(grids)])
    assert stacked.shape[0] == sum(len(grids[T]) for T in grids)
    # first block is the smallest T
    T0 = sorted(grids)[0]
    assert np.allclose(stacked[:len(grids[T0])], target[T0])


def test_snapshot_round_trip(tmp_path):
    insts = [_inst()]
    tickers = {"BTC-31JUL26-60000-P": _ob()}
    path = D.save_snapshot("BTC", insts, tickers, out_dir=str(tmp_path))
    i2, t2, meta = D.load_snapshot(path)
    assert i2 == insts and t2 == tickers and meta["currency"] == "BTC"


# ---- T pinned to the snapshot capture time (reproducibility, not the run clock) ----
def test_clean_from_snapshot_pins_T_to_fetched_utc(tmp_path, monkeypatch):
    """T must be measured from the snapshot's fetched_utc, NOT the run clock, so a saved
    surface re-cleans to identical maturities on any later date."""
    cap = "20260101T000000Z"
    cap_ms = D._parse_stamp_ms(cap)
    exp_ms = cap_ms + 30 * 86400 * 1000                          # expiry exactly 30 days after capture
    inst = {"instrument_name": "BTC-31JAN26-60000-P", "strike": 60000.0,
            "option_type": "put", "expiration_timestamp": exp_ms}
    snap = tmp_path / "BTC_20260101T000000Z.json"
    snap.write_text(json.dumps({"fetched_utc": cap, "currency": "BTC",
                                "instruments": [inst], "tickers": {inst["instrument_name"]: _ob(delta=-0.40)}}))
    monkeypatch.setattr(D.time, "time", lambda: 9_999_999_999.0)  # far-future run clock must NOT affect T
    grids, target, weights, meta = D.clean_from_snapshot(str(snap), want_expiries=("31JAN26",), verbose=False)
    assert len(grids) == 1
    T = next(iter(grids))
    assert abs(T - 30.0 / 365.0) < 1e-9                          # pinned to capture, not drifted by the clock


def test_clean_from_snapshot_requires_fetched_utc(tmp_path):
    """A snapshot with no capture stamp must RAISE, not silently fall back to the run clock."""
    inst = {"instrument_name": "BTC-31JAN26-60000-P", "strike": 60000.0,
            "option_type": "put", "expiration_timestamp": 1_800_000_000_000}
    snap = tmp_path / "BTC_nostamp.json"
    snap.write_text(json.dumps({"currency": "BTC", "instruments": [inst],
                                "tickers": {inst["instrument_name"]: _ob()}}))
    try:
        D.clean_from_snapshot(str(snap), want_expiries=("31JAN26",), verbose=False)
        assert False, "should raise when fetched_utc is missing"
    except ValueError:
        pass
