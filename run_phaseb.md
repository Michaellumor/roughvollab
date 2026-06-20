# RoughVolLab — Phase B Runbook

The repeatable path from raw exchange archives to a Hurst estimate on real
volatility. Run from the repo root in **PowerShell**. Each stage is one script;
the output of one feeds the next:

```
binance_data.py  →  kline_verifier.py  →  rv_series.py  →  estimate_h.py
   download           verify              build RV proxy     estimate H
```

---

## Setup (once)

```powershell
cd $HOME\Documents\roughvollab
python -m pip install numpy scipy matplotlib pytest
```

Confirm the suite is green before trusting any output:

```powershell
python -m pytest test_binance_data.py test_kline_verifier.py test_rv_series.py test_estimate_h.py -v
```

Expect **78 passed**. If `kline_verifier.py` errors on importing `canonical_ms`,
you are on the old copy — replace it with the current one.

---

## 1 — Download

```powershell
python binance_data.py --symbol BTCUSDT --interval 1m --start 2019-01 --end 2025-12 --out data
```

- Pulls monthly 1-minute archives from `data.binance.vision`, verifies each
  against its SHA-256 `.CHECKSUM`, extracts CSVs to
  `data\spot\klines\BTCUSDT\1m\`.
- **Restartable** — months already on disk are skipped, so just re-run if the
  connection drops.
- This range crosses the **2025-01-01 ms→µs timestamp switch**; it is handled
  downstream, no action needed.
- Add `data\` to `.gitignore` — raw archives are not committed (only the small
  processed RV series is).
- First time, do a 3-month smoke (`--start 2024-01 --end 2024-03`) to confirm
  the network path before the long pull.

---

## 2 — Verify

```powershell
python kline_verifier.py data\spot\klines\BTCUSDT\1m\*.csv --interval 1m
```

Diagnostic, **not** pass/fail. Read: coverage %, gap count + locations,
duplicates, grid alignment, price/OHLC sanity, and the mixed-unit (ms+µs) note
if the range crosses 2025. `is_clean = True` means no corruption — gaps are
*reported but not "dirty"* (a missing minute is a calendar fact, not an error).

> Use **backslashes** (PowerShell). The path has **no `monthly` segment** —
> it is `data\spot\klines\...`.

---

## 3 — Build the RV series

```powershell
python rv_series.py data\spot\klines\BTCUSDT\1m\*.csv --sampling 5m --rv-bar 1d --symbol BTCUSDT --out data\processed\btc_rv.csv
```

- Produces **log-realized-variance** — the Rung-1 proxy the estimators consume
  — one observation per UTC day.
- `--sampling` is the knob: `5m` is the noise-mitigated standard, `1m`
  maximises microstructure noise, coarser = fewer returns per bar. You are
  meant to sweep it (stage 4 does).
- `--out` writes the small, committable processed CSV.
- The report flags a series too short for H estimation: aim for ≥ ~250 and
  comfortably ≥ ~1000 daily points (i.e. **years** of history).

---

## 4 — Estimate H

```powershell
# from the processed CSV
python estimate_h.py data\processed\btc_rv.csv --symbol BTCUSDT --chunks 4

# from klines, with the sampling sweep
python estimate_h.py data\spot\klines\BTCUSDT\1m\*.csv --sampling 5m --sweep 1m,5m,15m --symbol BTCUSDT
```

Runs GJR, Cont–Das, and MF-DFA. Read the output in this order:

1. **Did each estimator survive?** `nan` or `OUTSIDE (0,1)` means the series is
   too short for that estimator — do not interpret the number.
2. **Trust signal per estimator** — GJR monofractal R² (≈1 is good), Cont–Das
   critical power p\* found, MF-DFA multifractal width Δh(q).
3. **Disagreement** — the spread, and whether the estimators agree on
   rough-vs-smooth. Straddling 0.5 *is the finding*: the estimators carry
   opposite-sign small-H biases (GJR / Cont–Das up, MF-DFA down).
4. **Sampling sweep** — Ĥ drifting steadily as sampling gets finer is the proxy
   manufacturing roughness (Rung 1). A flat row is the credible case.
5. **Sub-window stability** — large chunk-to-chunk spread means roughness drifts
   in time, or the chunks are too short to estimate.

---

## Reading the result honestly

- Everything here runs on the **RV proxy**, not true spot variance. Phase A
  (`layer1c_roughness_audit`) proved this proxy can manufacture roughness from a
  smooth path. Read any Ĥ against those simulated bias maps before claiming the
  roughness is real.
- A single Ĥ is a trap. The deliverable is *what can be concluded at stated
  confidence* — the spread, the sweep, the stability — not one number.
- **Length gates everything.** On 91 daily points the estimators return noise
  (and say so). Years of history is the precondition for an inferential result.

---

## File map

| file | role |
|---|---|
| `binance_data.py` | download + checksum-verify Binance public-data archives |
| `kline_verifier.py` | diagnostic data-quality report; `canonical_ms` (shared ms→µs normalisation) |
| `rv_series.py` | klines → log-RV proxy (Rung-1 twin), calendar-bucketed, gap-aware |
| `estimate_h.py` | run the 3 estimators + trust signals + disagreement + sweep + stability |
| `test_*.py` | the matching test suites (78 tests total) |
