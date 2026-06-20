"""
test_binance_data.py — tests for the Binance public-data downloader
===================================================================
Two layers: (1) the pure URL / date / checksum logic, pinned against
known-good data.binance.vision paths; (2) the download orchestration
(verify -> extract -> status), driven by an in-memory FAKE fetcher so the
full path is exercised without touching the network (the sandbox can't reach
data.binance.vision; the real download runs locally).

Run:  pytest test_binance_data.py -v
"""

import io
import zipfile

import pytest

from binance_data import (
    BASE_URL,
    NotFound,
    build_url,
    build_path,
    month_range,
    day_range,
    expected_files,
    sha256_bytes,
    parse_checksum,
    download_range,
    download_klines,
    summarize,
)


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def make_zip_bytes(member_name: str, text: str) -> bytes:
    """A real .zip (in memory) containing one CSV member."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, text)
    return buf.getvalue()


class FakeFeed:
    """Maps URLs to bytes; raises NotFound for anything not in the store."""
    def __init__(self, store: dict[str, bytes]):
        self.store = store
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.calls.append(url)
        if url not in self.store:
            raise NotFound(url)
        return self.store[url]


CSV_BODY = (
    "1704067200000,42000.00,42100.00,41950.00,42050.00,1.5,"
    "1704067259999,63075.0,10,0.8,33640.0,0\n"
)


# ──────────────────────────────────────────────────────────────────────────
# URL / path construction
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,expected", [
    (dict(data_type="klines", symbol="BTCUSDT", date_str="2024-01", interval="1m"),
     "/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"),
    (dict(data_type="klines", symbol="BTCUSDT", date_str="2024-01-15",
          interval="1m", period="daily"),
     "/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-15.zip"),
    (dict(data_type="aggTrades", symbol="BTCUSDT", date_str="2024-01"),
     "/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"),
    (dict(data_type="trades", symbol="ETHUSDT", date_str="2024-03"),
     "/data/spot/monthly/trades/ETHUSDT/ETHUSDT-trades-2024-03.zip"),
    (dict(data_type="klines", symbol="BTCUSDT", date_str="2024-01",
          interval="1m", market="futures/um"),
     "/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"),
])
def test_build_path_known_cases(kwargs, expected):
    assert build_path(**kwargs) == expected
    assert build_url(**kwargs) == BASE_URL + expected


def test_symbol_is_upper_cased():
    assert "/BTCUSDT/" in build_path("klines", "btcusdt", "2024-01", interval="1m")


def test_kline_requires_interval():
    with pytest.raises(ValueError):
        build_path("klines", "BTCUSDT", "2024-01")          # no interval


def test_aggtrades_rejects_interval():
    with pytest.raises(ValueError):
        build_path("aggTrades", "BTCUSDT", "2024-01", interval="1m")


def test_bad_market_and_period_rejected():
    with pytest.raises(ValueError):
        build_path("klines", "BTCUSDT", "2024-01", interval="1m", market="forex")
    with pytest.raises(ValueError):
        build_path("klines", "BTCUSDT", "2024-01", interval="1m", period="yearly")


# ──────────────────────────────────────────────────────────────────────────
# date enumeration
# ──────────────────────────────────────────────────────────────────────────

def test_month_range_spans_year_boundary():
    assert month_range("2024-11", "2025-02") == \
        ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_month_range_single_and_reversed():
    assert month_range("2024-06", "2024-06") == ["2024-06"]
    with pytest.raises(ValueError):
        month_range("2024-06", "2024-05")


def test_day_range_handles_leap_day():
    # 2024 is a leap year — Feb 29 must appear.
    assert day_range("2024-02-27", "2024-03-01") == \
        ["2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01"]


def test_expected_files_count_and_shape():
    files = expected_files("BTCUSDT", "2024-01", "2024-03", interval="1m")
    assert len(files) == 3
    d, url = files[0]
    assert d == "2024-01" and url.endswith("BTCUSDT-1m-2024-01.zip")


# ──────────────────────────────────────────────────────────────────────────
# checksums
# ──────────────────────────────────────────────────────────────────────────

def test_checksum_roundtrip_and_tamper():
    data = make_zip_bytes("x.csv", CSV_BODY)
    digest = sha256_bytes(data)
    checksum_file = f"{digest}  BTCUSDT-1m-2024-01.zip\n"
    assert parse_checksum(checksum_file) == digest
    # a single flipped byte must change the digest
    tampered = data[:-1] + bytes([data[-1] ^ 0x01])
    assert sha256_bytes(tampered) != digest


def test_parse_checksum_handles_bytes_and_blank():
    assert parse_checksum(b"abc123  file.zip") == "abc123"
    assert parse_checksum("") == ""


# ──────────────────────────────────────────────────────────────────────────
# orchestration (offline, via FakeFeed)
# ──────────────────────────────────────────────────────────────────────────

def _seed_store(url: str, body: str = CSV_BODY, good_checksum: bool = True):
    """Build a FakeFeed store for one kline file, with member name matching
    the CSV name the loader will look for."""
    member = url.rsplit("/", 1)[-1][:-4] + ".csv"   # BTCUSDT-1m-2024-01.csv
    zip_bytes = make_zip_bytes(member, body)
    digest = sha256_bytes(zip_bytes) if good_checksum else "0" * 64
    return {url: zip_bytes, url + ".CHECKSUM": f"{digest}  {member[:-4]}.zip\n"}


def test_download_extracts_and_verifies(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    feed = FakeFeed(_seed_store(url))
    res = download_range("klines", "BTCUSDT", "2024-01", "2024-01",
                         interval="1m", out_dir=tmp_path, fetcher=feed)
    assert len(res) == 1
    r = res[0]
    assert r.status == "downloaded"
    assert r.checksum_verified is True
    assert r.path.endswith("BTCUSDT-1m-2024-01.csv")
    from pathlib import Path
    assert Path(r.path).exists()
    assert Path(r.path).read_text() == CSV_BODY


def test_checksum_mismatch_blocks_extraction(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    feed = FakeFeed(_seed_store(url, good_checksum=False))
    res = download_range("klines", "BTCUSDT", "2024-01", "2024-01",
                         interval="1m", out_dir=tmp_path, fetcher=feed)
    r = res[0]
    assert r.status == "checksum_failed"
    assert r.checksum_verified is False
    # NOTHING should have been extracted from a corrupt archive
    assert not list(tmp_path.rglob("*.csv"))


def test_missing_file_reports_not_found(tmp_path):
    # store is empty -> every fetch raises NotFound
    feed = FakeFeed({})
    res = download_range("klines", "BTCUSDT", "2099-01", "2099-01",
                         interval="1m", out_dir=tmp_path, fetcher=feed)
    assert res[0].status == "not_found"


def test_missing_checksum_downloads_unverified(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    member = "BTCUSDT-1m-2024-01.csv"
    store = {url: make_zip_bytes(member, CSV_BODY)}     # no .CHECKSUM entry
    feed = FakeFeed(store)
    res = download_range("klines", "BTCUSDT", "2024-01", "2024-01",
                         interval="1m", out_dir=tmp_path, fetcher=feed)
    r = res[0]
    assert r.status == "downloaded"
    assert r.checksum_verified is None          # reported, not assumed OK


def test_existing_file_is_skipped(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    feed = FakeFeed(_seed_store(url))
    first = download_range("klines", "BTCUSDT", "2024-01", "2024-01",
                           interval="1m", out_dir=tmp_path, fetcher=feed)
    assert first[0].status == "downloaded"
    feed2 = FakeFeed(_seed_store(url))
    second = download_range("klines", "BTCUSDT", "2024-01", "2024-01",
                            interval="1m", out_dir=tmp_path, fetcher=feed2)
    assert second[0].status == "exists"
    assert feed2.calls == []                    # no fetch attempted


def test_keep_zip_writes_archive(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    feed = FakeFeed(_seed_store(url))
    download_range("klines", "BTCUSDT", "2024-01", "2024-01", interval="1m",
                   out_dir=tmp_path, keep_zip=True, fetcher=feed)
    assert list(tmp_path.rglob("*.zip"))


def test_mixed_range_summary(tmp_path):
    # three months: one present, one 404, one present
    store = {}
    for m in ("2024-01", "2024-03"):
        store.update(_seed_store(build_url("klines", "BTCUSDT", m, interval="1m")))
    feed = FakeFeed(store)
    res = download_range("klines", "BTCUSDT", "2024-01", "2024-03",
                         interval="1m", out_dir=tmp_path, fetcher=feed)
    summary = summarize(res, printout=False)
    assert summary["total"] == 3
    assert summary["counts"]["downloaded"] == 2
    assert summary["counts"]["not_found"] == 1


def test_download_klines_wrapper(tmp_path):
    url = build_url("klines", "BTCUSDT", "2024-01", interval="1m")
    feed = FakeFeed(_seed_store(url))
    res = download_klines("BTCUSDT", "1m", "2024-01", "2024-01",
                          out_dir=tmp_path, fetcher=feed)
    assert res[0].status == "downloaded"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
