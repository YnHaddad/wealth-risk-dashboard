"""Simulated portfolio history: deterministic, incremental, on real market prices.

Each portfolio is funded on its inception date and bought at its model weights. After that,
on every trading day:
  * dividends are credited to cash on the ex-date (held units x amount per share),
  * scheduled client contributions and withdrawals arrive (income mandates draw monthly),
  * new cash is invested and withdrawals are funded by trading toward the model,
  * on the first trading day of each quarter, holdings that drifted beyond tolerance
    are rebalanced back to target,
  * a few dated model changes swap one holding for another (data/model_changes.csv).

Trades execute at that day's close, in whole shares, with commission of 2 bps (min $5).
Randomness (contribution sizes) comes from a seed per portfolio and month, so re-running
never changes the past and each nightly run only appends new days.

Quantities and prices are stored in split-adjusted units (consistent with the price cache).
When a new split appears, earlier ledger rows for that symbol are re-based.

    python -m wealth.simulate            # extend to the latest close in the cache
    python -m wealth.simulate --rebuild  # regenerate from inception
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import date

import numpy as np
import pandas as pd

from . import market, refdata

CASH = refdata.CASH
TRADES = refdata.DATA / "trades.csv"
FLOWS = refdata.DATA / "cash_flows.csv"
META = refdata.DATA / "sim_meta.json"
CHANGES = refdata.DATA / "model_changes.csv"
TRADE_FIELDS = ["trade_id", "date", "portfolio", "symbol", "side", "quantity", "price", "amount", "fees", "reason", "source"]
FLOW_FIELDS = ["flow_id", "date", "portfolio", "type", "amount", "symbol", "note", "source"]
FEE_BPS, FEE_MIN = 2.0, 5.0
INVEST_TRIGGER = 0.01    # invest when cash exceeds target by 1% of the portfolio
RAISE_TRIGGER = 0.5      # top up cash when it falls below half its target
MIN_TICKET = 10_000      # smallest discretionary trade ($)
INCEPTION_DEFAULT = "2024-01-02"


# --------------------------------------------------------------------------- helpers
def fee(notional: float) -> float:
    return round(max(FEE_MIN, abs(notional) * FEE_BPS / 1e4), 2)


def rng_for(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def read_csv(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def num(x) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def model_changes() -> list[dict]:
    return sorted(read_csv(CHANGES), key=lambda r: r["date"])


def model_on(pid: str, day: str) -> dict[str, float]:
    """Model weights in force on `day`, undoing later model changes."""
    m = dict(refdata.models()[pid])
    for ch in reversed(model_changes()):
        if ch["portfolio"] == pid and ch["date"] > day:
            w = m.pop(ch["to_symbol"], 0.0)
            m[ch["from_symbol"]] = m.get(ch["from_symbol"], 0.0) + w
    return m


# --------------------------------------------------------------------------- client flows
def scheduled_flows(pid: str, month_first_days: dict[str, str]) -> dict[str, list[tuple[str, float, str]]]:
    """{date: [(type, amount, note)]} for a portfolio. month_first_days: 'YYYY-MM' -> first trading day."""
    out: dict[str, list] = {}

    def add(month, kind, amount, note):
        d = month_first_days.get(month)
        if d:
            out.setdefault(d, []).append((kind, round(amount, -3), note))

    months = sorted(month_first_days)
    for m in months:
        y, mo = map(int, m.split("-"))
        r = rng_for(pid, m)
        if pid == "growth":
            if mo in (2, 5, 8, 11):
                add(m, "contribution", r.uniform(200_000, 600_000), "Quarterly client contribution")
            if m == "2025-06":
                add(m, "withdrawal", 1_500_000, "Client withdrawal (property purchase)")
        elif pid == "equity_income":
            add(m, "withdrawal", 45_000, "Monthly income distribution to client")
            if m == "2024-06":
                add(m, "contribution", 1_000_000, "Additional client funding")
        elif pid == "balanced":
            if mo in (3, 9):
                add(m, "contribution", r.uniform(300_000, 800_000), "Semi-annual client contribution")
            if m == "2026-03":
                add(m, "withdrawal", 2_000_000, "Client withdrawal")
        elif pid == "conservative_income":
            add(m, "withdrawal", 25_000, "Monthly income distribution to client")
        elif pid == "multi_asset":
            if m == "2024-09":
                add(m, "contribution", 2_000_000, "Additional client funding")
            if m == "2025-11":
                add(m, "withdrawal", 750_000, "Client withdrawal")
            if m == "2026-05":
                add(m, "contribution", 500_000, "Client contribution")
    return out


# --------------------------------------------------------------------------- state
class Book:
    def __init__(self, pid: str):
        self.pid = pid
        self.qty: dict[str, float] = {}
        self.cash = 0.0
        self.trades: list[dict] = []
        self.flows: list[dict] = []
        self.seq = 0

    # -- ledger writers
    def _tid(self, day: str) -> str:
        self.seq += 1
        return f"S-{self.pid[:3].upper()}-{day.replace('-', '')}-{self.seq:03d}"

    def flow(self, day, kind, amount, symbol="", note=""):
        self.cash += amount
        self.flows.append({"flow_id": self._tid(day).replace("S-", "F-"), "date": day, "portfolio": self.pid,
                           "type": kind, "amount": f"{amount:.2f}", "symbol": symbol, "note": note,
                           "source": "simulated"})

    def trade(self, day, sym, qty, price, reason):
        if qty == 0:
            return
        notional = qty * price
        f = fee(notional)
        self.qty[sym] = self.qty.get(sym, 0.0) + qty
        if abs(self.qty[sym]) < 1e-9:
            self.qty.pop(sym)
        self.cash -= notional + f
        self.trades.append({"trade_id": self._tid(day), "date": day, "portfolio": self.pid, "symbol": sym,
                            "side": "buy" if qty > 0 else "sell", "quantity": num(abs(qty)), "price": num(price),
                            "amount": f"{abs(notional):.2f}", "fees": f"{f:.2f}", "reason": reason,
                            "source": "simulated"})

    # -- valuation
    def value(self, px: dict[str, float]) -> float:
        return self.cash + sum(q * px[s] for s, q in self.qty.items())

    def weights(self, px) -> dict[str, float]:
        tv = self.value(px)
        w = {s: q * px[s] / tv for s, q in self.qty.items()}
        w[CASH] = self.cash / tv
        return w


def replay(pid: str, trades: list[dict], flows: list[dict]) -> Book:
    b = Book(pid)
    for t in trades:
        if t["portfolio"] != pid:
            continue
        q = float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
        b.qty[t["symbol"]] = b.qty.get(t["symbol"], 0.0) + q
        if abs(b.qty[t["symbol"]]) < 1e-9:
            b.qty.pop(t["symbol"])
        b.cash -= q * float(t["price"]) + float(t["fees"])
    for f in flows:
        if f["portfolio"] == pid:
            b.cash += float(f["amount"])
    b.seq = sum(1 for t in trades if t["portfolio"] == pid) + sum(1 for f in flows if f["portfolio"] == pid)
    return b


# --------------------------------------------------------------------------- trading rules
def tolerance(target: float) -> float:
    return max(0.005, 0.20 * target)


def buy_qty(value: float, price: float) -> int:
    return int(math.floor(value / price))


def invest_excess(b: Book, day, px, model, reason):
    """Spend cash above the cash target on the most underweight holdings."""
    tv = b.value(px)
    excess = b.cash - model.get(CASH, 0.0) * tv
    if excess < INVEST_TRIGGER * tv:  # let dividends build up; invest in batches
        return
    w = b.weights(px)
    gaps = {s: (t - w.get(s, 0.0)) * tv for s, t in model.items() if s != CASH and t - w.get(s, 0.0) > 0}
    total_gap = sum(gaps.values())
    if total_gap <= 0:  # nothing underweight: spread pro rata to target
        gaps = {s: t * tv for s, t in model.items() if s != CASH}
        total_gap = sum(gaps.values())
    budget = excess * 0.995
    for s, g in sorted(gaps.items(), key=lambda kv: -kv[1]):
        amt = budget * g / total_gap
        if amt >= MIN_TICKET:
            b.trade(day, s, buy_qty(amt, px[s]), px[s], reason)


def raise_cash(b: Book, day, px, model, reason):
    """Restore cash to target by selling overweights first, then pro rata."""
    tv = b.value(px)
    target = model.get(CASH, 0.0) * tv
    if b.cash >= RAISE_TRIGGER * target and b.cash >= 0:  # buffer still adequate
        return
    need = target - b.cash
    w = b.weights(px)
    over = {s: (w[s] - model.get(s, 0.0)) * tv for s in b.qty if w[s] - model.get(s, 0.0) > 0}
    for s, amt in sorted(over.items(), key=lambda kv: -kv[1]):
        if need <= 0:
            break
        sell = min(amt, need)
        q = min(b.qty[s], math.ceil(sell / px[s]))
        b.trade(day, s, -q, px[s], reason)
        need -= q * px[s]
    if need > 0:
        held = sum(b.qty[s] * px[s] for s in b.qty)
        for s in list(b.qty):
            q = min(b.qty[s], math.ceil(need * (b.qty[s] * px[s] / held) / px[s]))
            b.trade(day, s, -q, px[s], reason)


def rebalance(b: Book, day, px, model, reason="Quarterly rebalance"):
    tv = b.value(px)
    w = b.weights(px)
    # sells first (out-of-model holdings and overweights), then buys
    for s in sorted(b.qty):
        t = model.get(s, 0.0)
        if s not in model or w[s] - t > tolerance(t):
            q = b.qty[s] - (buy_qty(t * tv, px[s]) if s in model else 0)
            if q > 0:
                b.trade(day, s, -q, px[s], reason if s in model else "Model change: exit")
    w = b.weights(px)
    for s, t in model.items():
        if s == CASH:
            continue
        if t - w.get(s, 0.0) > tolerance(t):
            q = buy_qty(t * tv, px[s]) - b.qty.get(s, 0.0)
            if q > 0 and q * px[s] < b.cash:
                b.trade(day, s, q, px[s], reason)


# --------------------------------------------------------------------------- splits
def rebase_splits(trades: list[dict], meta: dict) -> list[str]:
    applied = set(meta.get("splits_applied", []))
    notes = []
    splits = pd.read_csv(market.SPLITS) if market.SPLITS.exists() else pd.DataFrame(columns=["date", "symbol", "ratio"])
    for _, r in splits.iterrows():
        key = f"{r['symbol']}:{r['date']}"
        if key in applied:
            continue
        ratio = float(r["ratio"])
        n = 0
        for t in trades:
            if t["symbol"] == r["symbol"] and t["date"] < r["date"]:
                t["quantity"] = num(float(t["quantity"]) * ratio)
                t["price"] = num(float(t["price"]) / ratio)
                n += 1
        applied.add(key)
        if n:
            notes.append(f"re-based {n} rows of {r['symbol']} for {ratio:g}:1 split on {r['date']}")
    meta["splits_applied"] = sorted(applied)
    return notes


# --------------------------------------------------------------------------- main loop
def run(rebuild: bool = False) -> dict:
    close = market.load_wide(market.CLOSE).ffill()
    divs = pd.read_csv(market.DIVS)
    meta = {} if rebuild or not META.exists() else json.loads(META.read_text())
    trades = [] if rebuild else read_csv(TRADES)
    flows = [] if rebuild else read_csv(FLOWS)
    if rebuild:  # every split already in the cache is reflected in its prices
        if market.SPLITS.exists():
            sp = pd.read_csv(market.SPLITS)
            meta["splits_applied"] = sorted(f"{r.symbol}:{r.date}" for r in sp.itertuples())
    notes = rebase_splits([t for t in trades if t.get("source") == "simulated"], meta)

    days = [d.strftime("%Y-%m-%d") for d in close.index if not np.isnan(close.loc[d].get("SPY", np.nan))]
    month_first = {}
    for d in days:
        month_first.setdefault(d[:7], d)
    quarter_first = {month_first[m] for m in month_first if int(m[5:]) in (1, 4, 7, 10)}
    divs_by_day: dict[str, list] = {}
    for r in divs.itertuples():
        divs_by_day.setdefault(r.date, []).append((r.symbol, float(r.amount)))
    changes_by_day: dict[str, list] = {}
    for ch in model_changes():
        changes_by_day.setdefault(ch["date"], []).append(ch)

    start_after = meta.get("simulated_through", "")
    books = {}
    for pid, p in refdata.portfolios().items():
        b = replay(pid, trades, flows)
        b.inception = p["inception"]
        b.schedule = scheduled_flows(pid, month_first)
        books[pid] = b

    last = start_after
    for day in days:
        if day <= start_after:
            continue
        px = {s: float(v) for s, v in close.loc[day].items() if not np.isnan(v)}
        for pid, b in books.items():
            if day < b.inception:
                continue
            model = model_on(pid, day)
            if not b.qty and abs(b.cash) < 1e-9:  # first trading day on or after inception
                b.flow(day, "contribution", refdata.portfolios()[pid]["initial_aum"], note="Initial funding")
                tv = b.value(px)
                for s, t in model.items():
                    if s != CASH:
                        b.trade(day, s, buy_qty(t * tv, px[s]), px[s], "Initial allocation")
                continue
            # dividends on the ex-date to holders at the previous close
            for sym, amt in divs_by_day.get(day, []):
                q = b.qty.get(sym, 0.0)
                if q > 0:
                    b.flow(day, "dividend", q * amt, symbol=sym, note=f"{num(q)} units x ${amt:.4f}")
            for kind, amt, note in b.schedule.get(day, []):
                b.flow(day, kind, amt if kind == "contribution" else -amt, note=note)
            for ch in changes_by_day.get(day, []):
                if ch["portfolio"] == pid and ch["from_symbol"] in b.qty:
                    s_from = ch["from_symbol"]
                    b.trade(day, s_from, -b.qty[s_from], px[s_from], f"Model change: {ch['reason']}")
            if day in quarter_first or any(ch["portfolio"] == pid for ch in changes_by_day.get(day, [])):
                rebalance(b, day, px, model)
            raise_cash(b, day, px, model, "Raise cash for client withdrawal")
            invest_excess(b, day, px, model, "Invest new cash")
            if b.cash < 0:
                raise RuntimeError(f"{pid} cash negative on {day}: {b.cash:.2f}")
        last = day

    for b in books.values():
        trades.extend(b.trades)
        flows.extend(b.flows)
    trades.sort(key=lambda r: (r["date"], r["portfolio"], r["trade_id"]))
    flows.sort(key=lambda r: (r["date"], r["portfolio"], r["flow_id"]))
    write_csv(TRADES, TRADE_FIELDS, trades)
    write_csv(FLOWS, FLOW_FIELDS, flows)
    meta.update({"simulated_through": last or start_after, "fee_bps": FEE_BPS, "fee_min": FEE_MIN})
    META.write_text(json.dumps(meta, indent=1) + "\n")
    new_t = sum(len(b.trades) for b in books.values())
    new_f = sum(len(b.flows) for b in books.values())
    print(f"simulated through {meta['simulated_through']}: +{new_t} trades, +{new_f} cash flows"
          + ("; " + "; ".join(notes) if notes else ""))
    return {"through": meta["simulated_through"], "trades": new_t, "flows": new_f}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    run(rebuild=a.rebuild or not TRADES.exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())
