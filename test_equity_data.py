"""
test_equity_data.py
===================
Equity-arm tests. All run on SYNTHETIC OHLC — no network. They check that the
range estimators recover a known variance, that the parser tolerates Yahoo-style
extra columns, and (the important one) that the built CSV is read correctly by
estimate_h.py's own loader, so the equity series plugs into the existing
estimator/de-bias pipeline unchanged.
"""

import sys

import numpy as np
import pytest

from equity_data import (build_equity_rv_csv, garman_klass_log_var,
                         parkinson_log_var, parse_ohlc_csv)


# ── synthetic OHLC from driftless GBM days with a known daily variance ──────

def _simulate_ohlc(n_days, daily_var, n_intra=100, seed=0):
    """Return (o,h,l,c) price arrays for `n_days` days, each an intraday GBM
    path with total open→close variance `daily_var`."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(daily_var / n_intra)
    o, h, l, c = [], [], [], []
    base = 100.0
    for _ in range(n_days):
        steps = rng.normal(0.0, s, n_intra)
        p = np.concatenate(([0.0], np.cumsum(steps)))     # log-price, open at 0
        price = base * np.exp(p)
        o.append(price[0]); c.append(price[-1])
        h.append(price.max()); l.append(price.min())
        base = price[-1]
    return (np.array(o), np.array(h), np.array(l), np.array(c))


# ───────────────────────────────── estimators ─────────────────────────────

def test_garman_klass_recovers_known_variance():
    V = 4e-4                                               # ~2% daily vol
    # fine intraday grid so the sampled H/L approach the true continuous range
    # (real daily OHLC records the true high/low, so this is the realistic case)
    o, h, l, c = _simulate_ohlc(4000, V, n_intra=1500, seed=1)
    var = np.exp(garman_klass_log_var(o, h, l, c))
    rel = abs(var.mean() - V) / V
    assert rel < 0.12, f"GK mean var {var.mean():.2e} vs true {V:.2e} (rel {rel:.2f})"


def test_parkinson_recovers_known_variance():
    V = 4e-4
    o, h, l, c = _simulate_ohlc(4000, V, n_intra=1500, seed=2)
    var = np.exp(parkinson_log_var(h, l))
    rel = abs(var.mean() - V) / V
    assert rel < 0.12, f"Parkinson mean var {var.mean():.2e} vs true {V:.2e}"


# ─────────────────────────────────── parser ───────────────────────────────

def test_parser_tolerates_yahoo_columns(tmp_path):
    """Yahoo CSVs carry an extra 'Adj Close'; the parser must still work."""
    p = tmp_path / "ohlc.csv"
    p.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2020-01-02,100,102,99,101,101,1000\n"
        "2020-01-03,101,103,100,102,102,1200\n"
        "2020-01-06,102,102,98,99,99,1500\n")
    d = parse_ohlc_csv(p)
    assert d["o"].size == 3
    assert d["date_ms"][0] < d["date_ms"][1] < d["date_ms"][2]
    assert np.allclose(d["c"], [101, 102, 99])


def test_parser_drops_bad_rows(tmp_path):
    p = tmp_path / "ohlc.csv"
    p.write_text(
        "Date,Open,High,Low,Close\n"
        "2020-01-02,100,102,99,101\n"
        "2020-01-03,0,0,0,0\n"            # non-positive → dropped
        "bad,row,here,x,y\n"             # unparsable → dropped
        "2020-01-06,102,104,101,103\n")
    d = parse_ohlc_csv(p)
    assert d["o"].size == 2


# ─────────────────────── pipeline compatibility (the key test) ────────────

def test_build_csv_is_readable_by_estimate_h_loader(tmp_path):
    """The built series must load through estimate_h.load_log_rv_csv — proving
    the equity output drops into the existing estimator/de-bias pipeline."""
    o, h, l, c = _simulate_ohlc(60, 4e-4, seed=3)
    ohlc = tmp_path / "spy.csv"
    rows = ["Date,Open,High,Low,Close"]
    base_day = 1577923200  # 2020-01-02 UTC, seconds
    for i in range(o.size):
        ts = base_day + i * 86400
        from datetime import datetime, timezone
        ds = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append(f"{ds},{o[i]:.4f},{h[i]:.4f},{l[i]:.4f},{c[i]:.4f}")
    ohlc.write_text("\n".join(rows) + "\n")

    out = tmp_path / "spy_rv.csv"
    build_equity_rv_csv(ohlc, out, estimator="gk")

    from estimate_h import load_log_rv_csv          # the real downstream reader
    log_rv, log_bv, ms = load_log_rv_csv(out)
    assert log_rv.size == o.size
    assert np.all(np.isfinite(log_rv))
    assert ms is not None and ms.size == o.size


def test_build_rejects_unknown_estimator(tmp_path):
    p = tmp_path / "ohlc.csv"
    p.write_text("Date,Open,High,Low,Close\n2020-01-02,100,102,99,101\n")
    with pytest.raises(ValueError):
        build_equity_rv_csv(p, tmp_path / "o.csv", estimator="nope")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
