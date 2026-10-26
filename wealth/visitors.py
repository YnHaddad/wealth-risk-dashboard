"""Visitor orders: anyone with the dashboard link can trade the model portfolios.

Order types
  * market: fills at once at the latest market price.
  * limit (day order): fills at once if the market is already at or better than the limit
    (at the market price). Otherwise it rests as a working order; every 15-minute refresh
    checks the 15-minute bars since it was placed and fills it at the limit price if the
    market traded through it. Unfilled orders expire after the session's close.

Every order passes a pre-trade compliance check, at submission and again at fill:
  * the symbol is in the approved universe; quantity is a whole number of shares,
  * no short selling and no overdraft; cash and shares held by other working orders are
    reserved,
  * hard mandate limits may not be newly breached or made worse (soft limits only warn),
  * one order may not exceed 25% of the portfolio; limits more than 20% away from the
    market are rejected as fat-finger errors; at most 100 visitor orders per day.
Fills carry 2 bps commission (min $5). Visitor trades live in data/visitor_trades.csv,
apart from the simulated book, and expire after 7 days so the demo resets itself.

    PAYLOAD='{"portfolio":"growth","side":"buy","symbol":"NVDA","qty":100,"type":"limit","limit":200}' \\
        python -m wealth.visitors --apply --id abc123
    python -m wealth.visitors --match      # fill working limit orders (each refresh)
    PAYLOAD='{"cancel":"O-20261026-001"}' python -m wealth.visitors --apply --id abc124
    python -m wealth.visitors --expire     # nightly: expire day orders and 7-day-old trades
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
ORDERS = refdata.DATA / "visitor_orders.csv"
REQUESTS = refdata.DATA / "visitor_requests.json"
FIELDS = ["trade_id", "date", "portfolio", "symbol", "side", "quantity", "price", "amount", "fees", "reason", "source"]
ORDER_FIELDS = ["order_id", "created", "portfolio", "symbol", "side", "quantity", "type", "limit", "status",
                "updated", "fill_price", "trade_id", "note"]
DAILY_CAP = 100
MAX_TICKET = 0.25
MAX_LIMIT_DEVIATION = 0.20
TTL_DAYS = 7
SESSION_CLOSE_UTC = 21  # day orders placed before 21:00 UTC expire at that night's run
CASH = refdata.CASH


class Rejected(ValueError):
    pass


# --------------------------------------------------------------------------- io
def _read(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


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


# --------------------------------------------------------------------------- book and checks
def book(pid: str) -> tuple[dict[str, float], float]:
    """Units and cash for a portfolio from the simulated ledger plus visitor trades."""
    qty: dict[str, float] = defaultdict(float)
    cash = 0.0
    for t in _read(refdata.DATA / "trades.csv") + _read(FILE):
        if t["portfolio"] != pid:
            continue
        q = float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
        qty[t["symbol"]] += q
        cash -= q * float(t["price"]) + float(t["fees"])
    for f in _read(refdata.DATA / "cash_flows.csv"):
        if f["portfolio"] == pid:
            cash += float(f["amount"])
    return {s: q for s, q in qty.items() if abs(q) > 1e-9}, cash


def fee(notional: float) -> float:
    return round(max(5.0, notional * 2e-4), 2)


def reservations(pid: str, exclude: str = "") -> tuple[float, dict[str, float]]:
    """Cash (buys, at the limit plus commission) and shares (sells) held by working orders."""
    cash, shares = 0.0, defaultdict(float)
    for o in _read(ORDERS):
        if o["portfolio"] != pid or o["status"] != "working" or o["order_id"] == exclude:
            continue
        q, lim = float(o["quantity"]), float(o["limit"])
        if o["side"] == "buy":
            cash += q * lim + fee(q * lim)
        else:
            shares[o["symbol"]] += q
    return cash, shares


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
    if rule in ("min_cash", "cash"):
        return m["cash_w"], ""
    name = {"equity": "Equity", "fixed_income": "Fixed Income", "real_assets": "Real Assets"}.get(rule)
    if name:
        return m["asset_class"].get(name, 0.0), ""
    return None, ""


def outside(v, lo, hi) -> float:
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


def parse(entry: dict) -> dict:
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
    kind = entry.get("type", "market")
    if kind not in ("market", "limit"):
        raise Rejected("order type must be market or limit")
    limit = None
    if kind == "limit":
        try:
            limit = round(float(entry.get("limit")), 4)
        except (TypeError, ValueError):
            raise Rejected("limit price must be a number")
        if not limit > 0:
            raise Rejected("limit price must be positive")
    return {"portfolio": pid, "symbol": sym, "side": side, "qty": q, "type": kind, "limit": limit}


def check(o: dict, price: float, exclude_order: str = "") -> dict:
    """Compliance for executing (or reserving) `o` at `price`."""
    pid, sym, side, q = o["portfolio"], o["symbol"], o["side"], o["qty"]
    px = last_prices()
    px[sym] = price
    qty, cash = book(pid)
    res_cash, res_shares = reservations(pid, exclude=exclude_order)
    before = measures(qty, cash, px)
    free_shares = qty.get(sym, 0.0) - res_shares.get(sym, 0.0)
    if side == "sell" and q > free_shares + 1e-9:
        held = qty.get(sym, 0.0)
        extra = f" ({res_shares[sym]:,.0f} already in working sell orders)" if res_shares.get(sym) else ""
        raise Rejected(f"the portfolio holds {held:,.0f} {sym}{extra}; short selling is not allowed")
    notional = q * price
    fees = fee(notional)
    if notional > MAX_TICKET * before["tv"]:
        raise Rejected(f"a single order can't exceed 25% of the portfolio (${MAX_TICKET * before['tv']:,.0f})")
    q2 = dict(qty)
    q2[sym] = q2.get(sym, 0.0) + (q if side == "buy" else -q)
    if abs(q2[sym]) < 1e-9:
        q2.pop(sym)
    cash2 = cash - (notional if side == "buy" else -notional) - fees
    if cash2 - res_cash < 0:
        avail = cash - res_cash
        extra = f" after ${res_cash:,.0f} reserved for working orders" if res_cash else ""
        raise Rejected(f"not enough cash: the order needs ${notional + fees:,.0f}, the portfolio has "
                       f"${max(avail, 0):,.0f} available{extra}")
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
    return {**o, "price": price, "notional": notional, "fees": fees, "warnings": warnings,
            "weight_after": after["w"].get(sym, 0.0)}


# --------------------------------------------------------------------------- booking
def _book_fill(res: dict, reason: str) -> str:
    rows = _read(FILE)
    today = _now().strftime("%Y-%m-%d")
    n = sum(1 for r in rows if r["date"] == today) + 1
    tid = f"V-{today.replace('-', '')}-{n:03d}"
    rows.append({"trade_id": tid, "date": today, "portfolio": res["portfolio"], "symbol": res["symbol"],
                 "side": res["side"], "quantity": f"{res['qty']:.0f}", "price": f"{res['price']:.4f}",
                 "amount": f"{res['notional']:.2f}", "fees": f"{res['fees']:.2f}",
                 "reason": reason + (" — " + "; ".join(res["warnings"]) if res["warnings"] else ""),
                 "source": "visitor"})
    _write_rows(FILE, FIELDS, rows)
    return tid


def _new_order_id(orders: list[dict]) -> str:
    today = _now().strftime("%Y%m%d")
    n = sum(1 for o in orders if o["order_id"].startswith(f"O-{today}")) + 1
    return f"O-{today}-{n:03d}"


def _order_row(o: dict, oid: str, status: str, note: str = "", fill_price=None, tid="") -> dict:
    now = _iso(_now())
    return {"order_id": oid, "created": now, "portfolio": o["portfolio"], "symbol": o["symbol"], "side": o["side"],
            "quantity": f"{o['qty']:.0f}", "type": o["type"], "limit": "" if o["limit"] is None else f"{o['limit']:.4f}",
            "status": status, "updated": now, "fill_price": "" if fill_price is None else f"{fill_price:.4f}",
            "trade_id": tid, "note": note}


def apply(entry: dict) -> dict:
    if entry.get("cancel"):
        return cancel(str(entry["cancel"]))
    today = _now().strftime("%Y-%m-%d")
    orders = _read(ORDERS)
    if sum(1 for o in orders if o["created"][:10] == today) >= DAILY_CAP:
        raise Rejected("the demo's daily order limit is reached; try again tomorrow")
    o = parse(entry)
    price, src = live_price(o["symbol"], last_prices()[o["symbol"]])
    price = round(price, 4)
    oid = _new_order_id(orders)
    if o["type"] == "limit":
        if abs(o["limit"] / price - 1) > MAX_LIMIT_DEVIATION:
            raise Rejected(f"limit ${o['limit']:,.2f} is more than 20% from the market (${price:,.2f}); "
                           "check for a typo")
        marketable = (o["side"] == "buy" and price <= o["limit"]) or (o["side"] == "sell" and price >= o["limit"])
        if not marketable:
            check(o, o["limit"])  # reserve at the limit: must be affordable and compliant there
            orders.append(_order_row(o, oid, "working", note=f"market ${price:,.2f} ({src}) when placed"))
            _write_rows(ORDERS, ORDER_FIELDS, orders)
            name = refdata.portfolios()[o["portfolio"]]["name"]
            return {"status": "working", "order_id": oid, **o, "market": price,
                    "message": f"Limit order {oid} working: {o['side']} {o['qty']:,.0f} {o['symbol']} at "
                               f"${o['limit']:,.2f} or better in {name} (market ${price:,.2f}). It fills if the "
                               f"price reaches the limit today and expires after the close."}
    res = check(o, price)
    label = "Market order" if o["type"] == "market" else f"Limit ${o['limit']:,.2f}, marketable"
    tid = _book_fill(res, f"Visitor {label.lower()} ({src} price)")
    orders.append(_order_row(o, oid, "filled", note=label, fill_price=price, tid=tid))
    _write_rows(ORDERS, ORDER_FIELDS, orders)
    side = "Bought" if o["side"] == "buy" else "Sold"
    msg = (f"{side} {o['qty']:,.0f} {o['symbol']} at ${price:,.2f} ({src} price) in "
           f"{refdata.portfolios()[o['portfolio']]['name']}; now {res['weight_after']:.1%} of the portfolio")
    if res["warnings"]:
        msg += ". Soft-limit warning: " + "; ".join(res["warnings"])
    return {"status": "filled", "order_id": oid, "trade_id": tid, **res, "message": msg}


def cancel(order_id: str) -> dict:
    orders = _read(ORDERS)
    for o in orders:
        if o["order_id"] == order_id:
            if o["status"] != "working":
                raise Rejected(f"order {order_id} is already {o['status']}")
            o["status"], o["updated"], o["note"] = "cancelled", _iso(_now()), "Cancelled from the dashboard"
            _write_rows(ORDERS, ORDER_FIELDS, orders)
            return {"status": "cancelled", "order_id": order_id,
                    "message": f"Cancelled {order_id}: {o['side']} {float(o['quantity']):,.0f} {o['symbol']} "
                               f"at ${float(o['limit']):,.2f}"}
    raise Rejected(f"order {order_id} not found")


def match(bars: dict | None = None) -> list[str]:
    """Fill working limit orders whose limit traded since they were placed (fill at the limit)."""
    orders = _read(ORDERS)
    working = [o for o in orders if o["status"] == "working"]
    if not working:
        return []
    if bars is None:
        from .market import intraday_bars
        bars = intraday_bars(sorted({o["symbol"] for o in working}))
    notes = []
    for o in sorted(working, key=lambda r: r["created"]):
        since = datetime.fromisoformat(o["created"])
        sb = [b for b in bars.get(o["symbol"], []) if b["end"] > since]
        lim = float(o["limit"])
        hit = any((b["low"] <= lim) if o["side"] == "buy" else (b["high"] >= lim) for b in sb)
        if not hit:
            continue
        od = {"portfolio": o["portfolio"], "symbol": o["symbol"], "side": o["side"], "qty": float(o["quantity"]),
              "type": "limit", "limit": lim}
        try:
            res = check(od, lim, exclude_order=o["order_id"])
        except Rejected as e:
            o["status"], o["updated"], o["note"] = "cancelled", _iso(_now()), f"Not filled at the limit: {e}"
            notes.append(f"{o['order_id']} cancelled at fill: {e}")
            continue
        tid = _book_fill(res, f"Visitor limit order {o['order_id']} filled at ${lim:,.2f}")
        o["status"], o["updated"], o["fill_price"], o["trade_id"] = "filled", _iso(_now()), f"{lim:.4f}", tid
        notes.append(f"{o['order_id']} filled: {o['side']} {res['qty']:,.0f} {o['symbol']} at ${lim:,.2f}")
    _write_rows(ORDERS, ORDER_FIELDS, orders)
    return notes


def expire(days: int = TTL_DAYS) -> tuple[int, int]:
    """Expire day orders placed before today's close, and visitor trades/orders older than `days`."""
    now = _now()
    cutoff_close = now.replace(hour=SESSION_CLOSE_UTC, minute=0, second=0, microsecond=0)
    if now < cutoff_close:
        cutoff_close -= timedelta(days=1)
    orders = _read(ORDERS)
    n_exp = 0
    for o in orders:
        if o["status"] == "working" and datetime.fromisoformat(o["created"]) < cutoff_close:
            o["status"], o["updated"], o["note"] = "expired", _iso(now), "Day order expired unfilled"
            n_exp += 1
    old = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    orders = [o for o in orders if o["created"][:10] >= old or o["status"] == "working"]
    _write_rows(ORDERS, ORDER_FIELDS, orders)
    rows = _read(FILE)
    keep = [r for r in rows if r["date"] >= old]
    _write_rows(FILE, FIELDS, keep)
    return n_exp, len(rows) - len(keep)


def record_request(req_id: str, ok: bool, message: str, status: str = "") -> None:
    """Outcome of each dashboard request, read back by /api/status (last 300 kept)."""
    try:
        log = json.loads(REQUESTS.read_text()) if REQUESTS.exists() else {}
    except json.JSONDecodeError:
        log = {}
    log[req_id] = {"ok": ok, "message": message, "status": status, "at": _iso(_now())}
    log = dict(sorted(log.items(), key=lambda kv: kv[1]["at"])[-300:])
    REQUESTS.write_text(json.dumps(log, indent=0) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--match", action="store_true")
    g.add_argument("--expire", action="store_true")
    ap.add_argument("--id", default="", help="request id from the dashboard (records the outcome)")
    a = ap.parse_args(argv)
    if a.expire:
        n_exp, n_old = expire()
        print(f"expired {n_exp} day orders; removed {n_old} visitor trades older than {TTL_DAYS} days")
        return 0
    if a.match:
        for n in match():
            print(f"::notice title=Limit order::{n}")
        return 0
    try:
        entry = json.loads(os.environ.get("PAYLOAD", "{}"))
        if not isinstance(entry, dict):
            raise Rejected("malformed order")
        res = apply(entry)
    except (Rejected, json.JSONDecodeError) as e:
        msg = str(e)
        msg = msg[0].upper() + msg[1:] if msg else "Rejected"
        print(f"::warning title=Order rejected::{msg}")
        if a.id:
            record_request(a.id, False, msg, "rejected")
        return 0 if a.id else 1
    print(f"::notice title=Order {res['status']}::{res['message']}")
    if a.id:
        record_request(a.id, True, res["message"], res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
