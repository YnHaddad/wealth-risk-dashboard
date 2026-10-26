"""Reference data: instruments, portfolios, models, benchmarks, limits, look-through.

All of it lives in data/*.csv so it can be read and edited without touching code.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MARKET = DATA / "market"
CASH = "CASH"

SECTORS = ["Information Technology", "Communication Services", "Consumer Discretionary", "Consumer Staples",
           "Energy", "Financials", "Health Care", "Industrials", "Materials", "Real Estate", "Utilities"]
REGIONS = ["North America", "Europe", "Japan", "Asia Pacific ex-Japan", "Emerging Markets", "Global"]
ASSET_CLASSES = ["Equity", "Fixed Income", "Real Assets", "Cash & Equivalents"]

# Historical stress windows (peak-to-trough of the S&P 500 unless noted).
STRESS_WINDOWS = [
    ("gfc", "Global Financial Crisis", "2008-09-12", "2009-03-09", "Lehman collapse to the market bottom"),
    ("q4_2018", "Q4 2018 sell-off", "2018-09-20", "2018-12-24", "Fed tightening and growth scare"),
    ("covid", "Covid crash", "2020-02-19", "2020-03-23", "Pandemic shock"),
    ("rates_2022", "2022 rate shock", "2022-01-03", "2022-10-12", "Fastest Fed hiking cycle in decades; stocks and bonds fell together"),
]
EXTRA_TICKERS = ["SPY", "XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU"]


def _read(name: str) -> list[dict]:
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x, default=None):
    try:
        return float(x) if x not in ("", None) else default
    except ValueError:
        return default


@lru_cache(maxsize=1)
def instruments() -> dict[str, dict]:
    out = {}
    for r in _read("instruments.csv"):
        r["duration"] = _num(r["duration"])
        out[r["symbol"]] = r
    return out


@lru_cache(maxsize=1)
def lookthrough() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in _read("lookthrough.csv"):
        out.setdefault(r["symbol"], {}).setdefault(r["dimension"], {})[r["bucket"]] = float(r["weight"])
    return out


@lru_cache(maxsize=1)
def portfolios() -> dict[str, dict]:
    out = {}
    for r in _read("portfolios.csv"):
        r["initial_aum"] = float(r["initial_aum"])
        out[r["id"]] = r
    return out


@lru_cache(maxsize=1)
def models() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in _read("models.csv"):
        out.setdefault(r["portfolio"], {})[r["symbol"]] = float(r["target_weight"])
    return out


@lru_cache(maxsize=1)
def benchmarks() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in _read("benchmarks.csv"):
        b = out.setdefault(r["portfolio"], {"label": r["label"], "weights": {}})
        b["weights"][r["symbol"]] = float(r["weight"])
    return out


@lru_cache(maxsize=1)
def limits() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _read("limits.csv"):
        out.setdefault(r["portfolio"], []).append(
            {"rule": r["rule"], "min": _num(r["min"]), "max": _num(r["max"]), "severity": r["severity"]})
    return out


def all_market_tickers() -> list[str]:
    syms = set(instruments())
    for r in instruments().values():
        for p in (r["stress_proxy"] or "").split("|"):
            if p:
                syms.add(p)
    return sorted(syms | set(EXTRA_TICKERS))


def classify(symbol: str) -> dict:
    """Exposure breakdown of one unit of weight in `symbol`:
    {"asset_class": {...}, "sector": {...}, "region": {...}} (equity sector/region only)."""
    if symbol == CASH:
        return {"asset_class": {"Cash & Equivalents": 1.0}, "sector": {}, "region": {}}
    r = instruments()[symbol]
    lt = lookthrough().get(symbol, {})
    ac = r["asset_class"]
    out = {"asset_class": {ac: 1.0}, "sector": {}, "region": {}}
    if r["sector"] == "look-through":
        out["sector"] = dict(lt.get("sector", {}))
    elif r["sector"]:
        out["sector"] = {r["sector"]: 1.0}
    if r["region"] == "look-through":
        out["region"] = dict(lt.get("region", {}))
    elif r["region"]:
        out["region"] = {r["region"]: 1.0}
    if ac != "Equity" and not (ac == "Real Assets" and r["sector"]):
        out["sector"] = {}
    return out
