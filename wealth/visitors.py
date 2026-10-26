"""Visitor trades: anyone with the dashboard link can place a trade in a model portfolio.

Every trade passes a pre-trade compliance check before it is booked:
  * the symbol is in the approved universe, the quantity is a whole number of shares,
  * no short selling and no overdraft (cash must stay >= 0 after the trade and costs),
  * hard mandate limits (single stock, single fund, asset-class ranges) may not be newly
    breached or made worse; soft limits only produce warnings,
  * one trade may not exceed 25% of the portfolio; at most 100 visitor trades per day.
Trades execute at the latest market price (fetched at booking time) with 2 bps commission
(min $5). They live in data/visitor_trades.csv, separate from the simulated book, and expire
after 7 days so the demo resets itself.

    PAYLOAD='{"portfolio":"growth","side":"buy","symbol":"NVDA","qty":100}' python -m wealth.visitors --apply
    python -m wealth.visitors --expire
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from . import refdata

FILE = refdata.DATA / "visitor_trades.csv"
FIELDS = ["trade_id", "date", "portfolio", "symbol", "side", "quantity", "price", "amount", "fees", "reason", "source"]
DAILY_CAP = 100
MAX_TICKET = 0.25
TTL_DAYS = 7
CASH = refdata.CASH


class Rejected(ValueError):
    pass


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(rows: list[dict]) -> None:
    with FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def last_prices() -> dict[str, float]:
    with (refdata.MARKET / "close.csv").open() as f:
        rows = list(csv.reader(f))
    head, last = rows[0], rows[-1]
    return {s: float(v) for s, v in zip(head[1:], last[1:]) if v}


def live_price(symbol: str, fallback: float) -> tuple[float, str]:
    try:
        from .market import live_quotes
        q = live_quotes([symbol]).get(symbol)
        if q and q.get("price"):
            return float(q["price"]), "live"
    except Exception:  # noqa: BLE001
        pass
    return fallback, "last close"


def book(pid: str, extra: list[dict] | None = None) -> tuple[dict[str, float], float]:
    """Units and cash for a portfolio from the simulated ledger plus visitor trades."""
    qty: dict[str, float] = defaultdict(float)
    cash = 0.0
    for t in _read(refdata.DATA / "trades.csv") + _read(FILE) + (extra or []):
        if t["portfolio"] != pid:
            continue
        q = float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
        qty[t["symbol"]] += q
        cash -= q * float(t["price"]) + float(t["fees"])
    for f in _read(refdata.DATA / "cash_flows.csv"):
        if f["portfolio"] == pid:
            cash += float(f["amount"])
    return {s: q for s, q in qty.items() if abs(q) > 1e-9}, cash


def measures(qty: dict[str, float], cash: float, px: dict[str, float]) -> dict:
    tv = cash + sum(q * px[s] for s, q in qty.items())
    w = {s: q * px[s] / tv for s, q in qty.items()}
    ins = refdata.instruments()
    ac = defaultdict(float)
    for s, v in w.items():
        for k, x in refdata.classify(s)["asset_class"].items():
            ac[k] += v * x
    ac["Cash & Equivalents"] += cash / tv
    stocks = {s: v for s, v in w.items() if ins[s]["kind"] == "stock"}
    funds = {s: v for s, v in w.items() if ins[s]["kind"] == "etf"}
    return {"tv": tv, "w": w, "cash": cash, "cash_w": cash / tv, "asset_class": ac,
            "max_stock": max(stocks.items(), key=lambda kv: kv[1]) if stocks else ("", 0.0),
            "max_fund": max(funds.items(), key=lambda kv: kv[1]) if funds else ("", 0.0)}


def rule_value(rule: str, m: dict):
    if rule == "max_single_stock":
        return m["max_stock"][1], m["max_stock"][0]
    if rule == "max_single_fund":
        return m["max_fund"][1], m["max_fund"][0]
    if rule == "min_cash":
        return m["cash_w"], ""
    if rule == "cash":
        return m["cash_w"], ""
    name = {"equity": "Equity", "fixed_income": "Fixed Income", "real_assets": "Real Assets"}.get(rule)
    if name:
        return m["asset_class"].get(name, 0.0), ""
    return None, ""


def outside(v, lo, hi) -> float:
    """Distance outside [lo, hi] (0 when inside)."""
    if v is None:
        return 0.0
    if hi is not None and v > hi:
        return v - hi
    if lo is not None and v < lo:
        return lo - v
    return 0.0


def fmt_range(lo, hi) -> str:
    if lo is None:
        return f"max {hi:.0%}"
    if hi is None:
        return f"min {lo:.0%}"
    return f"{lo:.0%} to {hi:.0%}"


def check(entry: dict, price: float) -> dict:
    pid = entry.get("portfolio")
    if pid not in refdata.portfolios():
        raise Rejected("unknown portfolio")
    sym = str(entry.get("symbol", "")).upper().strip()
    ins = refdata.instruments().get(sym)
    if not ins or ins["kind"] == "benchmark":
        raise Rejected(f"{sym or 'symbol'} is not in the approved universe")
    side = entry.get("side")
    if side not in ("buy", "sell"):
        raise Rejected("side must be buy or sell")
    try:
        q = float(entry.get("qty"))
    except (TypeError, ValueError):
        raise Rejected("quantity must be a number")
    if q <= 0 or q != math.floor(q) or q > 10_000_000:
        raise Rejected("quantity must be a positive whole number of shares")
    px = last_prices()
    px[sym] = price
    qty, cash = book(pid)
    before = measures(qty, cash, px)
    if side == "sell" and q > qty.get(sym, 0.0) + 1e-9:
        raise Rejected(f"the portfolio holds {qty.get(sym, 0.0):,.0f} {sym}; short selling is not allowed")
    notional = q * price
    fees = round(max(5.0, notional * 2e-4), 2)
    if notional > MAX_TICKET * before["tv"]:
        raise Rejected(f"a single trade can't exceed 25% of the portfolio (${MAX_TICKET * before['tv']:,.0f})")
    q2 = dict(qty)
    q2[sym] = q2.get(sym, 0.0) + (q if side == "buy" else -q)
    if abs(q2[sym]) < 1e-9:
        q2.pop(sym)
    cash2 = cash - (notional if side == "buy" else -notional) - fees
    if cash2 < 0:
        raise Rejected(f"not enough cash: the trade needs ${notional + fees:,.0f}, the portfolio has ${cash:,.0f}")
    after = measures(q2, cash2, px)
    problems, warnings = [], []
    for lim in refdata.limits()[pid]:
        v0, _ = rule_value(lim["rule"], before)
        v1, who = rule_value(lim["rule"], after)
        if v1 is None:
            continue
        d0, d1 = outside(v0, lim["min"], lim["max"]), outside(v1, lim["min"], lim["max"])
        if d1 > d0 + 1e-12:
            msg = (f"{lim['rule'].replace('_', ' ')} would be {v1:.1%}" + (f" ({who})" if who else "")
                   + f", outside its limit ({fmt_range(lim['min'], lim['max'])})")
            (problems if lim["severity"] == "hard" else warnings).append(msg)
    if problems:
        raise Rejected("blocked by pre-trade compliance: " + "; ".join(problems))
    return {"portfolio": pid, "symbol": sym, "side": side, "qty": q, "price": price, "notional": notional,
            "fees": fees, "warnings": warnings, "weight_after": after["w"].get(sym, 0.0)}


def apply(entry: dict) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = _read(FILE)
    if sum(1 for r in rows if r["date"] == today) >= DAILY_CAP:
        raise Rejected("the demo's daily trade limit is reached; try again tomorrow")
    sym = str(entry.get("symbol", "")).upper().strip()
    fallback = last_prices().get(sym)
    if fallback is None:
        raise Rejected(f"{sym or 'symbol'} is not in the approved universe")
    price, src = live_price(sym, fallback)
    res = check(entry, round(price, 4))
    n = sum(1 for r in rows if r["date"] == today) + 1
    row = {"trade_id": f"V-{today.replace('-', '')}-{n:03d}", "date": today, "portfolio": res["portfolio"],
           "symbol": res["symbol"], "side": res["side"], "quantity": f"{res['qty']:.0f}",
           "price": f"{res['price']:.4f}", "amount": f"{res['notional']:.2f}", "fees": f"{res['fees']:.2f}",
           "reason": f"Visitor trade ({src} price)" + (" — " + "; ".join(res["warnings"]) if res["warnings"] else ""),
           "source": "visitor"}
    rows.append(row)
    _write(rows)
    res["trade_id"], res["price_source"] = row["trade_id"], src
    return res


def expire(days: int = TTL_DAYS) -> int:
    rows = _read(FILE)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    keep = [r for r in rows if r["date"] >= cutoff]
    if len(keep) != len(rows) or not FILE.exists():
        _write(keep)
    return len(rows) - len(keep)


REQUESTS = refdata.DATA / "visitor_requests.json"


def record_request(req_id: str, ok: bool, message: str) -> None:
    """Outcome of each dashboard order, read back by /api/status (last 300 kept)."""
    try:
        log = json.loads(REQUESTS.read_text()) if REQUESTS.exists() else {}
    except json.JSONDecodeError:
        log = {}
    log[req_id] = {"ok": ok, "message": message, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    log = dict(sorted(log.items(), key=lambda kv: kv[1]["at"])[-300:])
    REQUESTS.write_text(json.dumps(log, indent=0) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--expire", action="store_true")
    ap.add_argument("--id", default="", help="request id from the dashboard (records the outcome)")
    a = ap.parse_args(argv)
    if a.expire:
        print(f"expired {expire()} visitor trades")
        return 0
    try:
        entry = json.loads(os.environ.get("PAYLOAD", "{}"))
        if not isinstance(entry, dict):
            raise Rejected("malformed order")
        res = apply(entry)
    except (Rejected, json.JSONDecodeError) as e:
        msg = str(e)
        msg = msg[0].upper() + msg[1:] if msg else "Rejected"
        print(f"::warning title=Trade rejected::{msg}")
        if a.id:
            record_request(a.id, False, msg)
        return 0 if a.id else 1
    side = "Bought" if res["side"] == "buy" else "Sold"
    msg = (f"{side} {res['qty']:,.0f} {res['symbol']} at ${res['price']:,.2f} ({res['price_source']} price) "
           f"in {refdata.portfolios()[res['portfolio']]['name']}; now {res['weight_after']:.1%} of the portfolio")
    if res["warnings"]:
        msg += ". Soft-limit warning: " + "; ".join(res["warnings"])
    print(f"::notice title=Trade booked::{msg}")
    if a.id:
        record_request(a.id, True, msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
