"""Analytics engine: valuation, performance, attribution, risk, exposures, limits, stress tests,
rebalancing and income for every portfolio, plus a firm-level roll-up.

Conventions (all USD):
  * Holdings are valued at the daily close (unadjusted); trades execute at their recorded price.
  * Daily time-weighted return  r_t = (MV_t - F_t) / MV_{t-1} - 1, with external client flows F_t
    treated as arriving at the close. Dividends are income, not flows.
  * Instrument total return     r_i,t = (P_t + D_t) / P_{t-1} - 1 (D on the ex-date).
  * Contribution to return      CTR_i = sum_t w_i,t-1 * r_i,t  (arithmetic; costs and timing in a residual).
  * Brinson-Fachler by asset class, daily, summed over the period; geometric linking residual shown.
  * Risk on current weights with up to 2 years of daily total returns: historical VaR/ES (1-day),
    ex-ante volatility, tracking error and beta vs the benchmark, risk contribution per holding.
  * Benchmarks are daily-rebalanced blends of index ETFs (total return).
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import market, refdata

CASH = refdata.CASH
RISK_DAYS = 504
PERIODS = ["1M", "3M", "YTD", "1Y", "SI"]


# =========================================================================== helpers
def _r(x, n=6):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return round(float(x), n)
    return x


def xirr(flows: list[tuple[date, float]]) -> float | None:
    if len(flows) < 2:
        return None
    t0 = flows[0][0]

    def npv(r):
        return sum(a / (1 + r) ** ((t - t0).days / 365.25) for t, a in flows)
    lo, hi = -0.95, 5.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        m = (lo + hi) / 2
        if npv(lo) * npv(m) <= 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def _hist(x: np.ndarray, bins: int = 36):
    """Histogram on the 0.25%-99.75% range; tail days are counted in the end bins."""
    lo, hi = np.quantile(x, 0.0025), np.quantile(x, 0.9975)
    c, e = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    return c.tolist(), [float(v) for v in e]


def asset_class(sym: str) -> str:
    if sym == CASH:
        return "Cash & Equivalents"
    return refdata.instruments()[sym]["asset_class"]


def period_start(label: str, days: pd.DatetimeIndex, inception: pd.Timestamp) -> pd.Timestamp:
    end = days[-1]
    if label == "1M":
        target = end - pd.DateOffset(months=1)
    elif label == "3M":
        target = end - pd.DateOffset(months=3)
    elif label == "YTD":
        target = pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1)
    elif label == "1Y":
        target = end - pd.DateOffset(years=1)
    else:
        target = inception
    prior = days[days <= target]
    start = prior[-1] if len(prior) else days[0]
    return max(start, inception)


# =========================================================================== market inputs
class Market:
    def __init__(self, quotes: dict | None = None):
        self.close = market.load_wide(market.CLOSE).ffill()
        self.adj = market.load_wide(market.ADJ).ffill()
        d = pd.read_csv(market.DIVS)
        d["date"] = pd.to_datetime(d["date"])
        self.divs_long = d
        div = d.pivot_table(index="date", columns="symbol", values="amount", aggfunc="sum")
        self.div = div.reindex(self.close.index).fillna(0.0).reindex(columns=self.close.columns, fill_value=0.0)
        self.stress = pd.read_csv(market.STRESS)
        self.quotes = quotes or {}
        self.last_close_day = self.close.index[-1]
        live_day = None
        if self.quotes:
            ts = max(pd.Timestamp(q["time"]) for q in self.quotes.values())
            ld = pd.Timestamp(ts.tz_convert("America/New_York").date()) if ts.tzinfo else ts.normalize()
            if ld > self.last_close_day:
                live_day = ld
        self.live_day = live_day
        if live_day is not None:  # add a provisional row priced at the latest quotes
            row_c = self.close.iloc[-1].copy()
            row_a = self.adj.iloc[-1].copy()
            for s, q in self.quotes.items():
                if s in row_c.index and q.get("price"):
                    ratio = q["price"] / row_c[s] if row_c[s] else 1.0
                    row_c[s] = q["price"]
                    row_a[s] = row_a[s] * ratio
            self.close.loc[live_day] = row_c
            self.adj.loc[live_day] = row_a
            self.div.loc[live_day] = 0.0
        self.tr = ((self.close + self.div) / self.close.shift(1) - 1).fillna(0.0)  # total return, unadjusted + divs
        self.adj_ret = self.adj.pct_change().fillna(0.0)
        self.asof = self.close.index[-1]

    def price(self, sym: str) -> float:
        return float(self.close[sym].iloc[-1])

    def ttm_div(self, sym: str) -> float:
        d = self.divs_long
        cutoff = self.last_close_day - pd.Timedelta(days=365)
        return float(d[(d.symbol == sym) & (d.date > cutoff)].amount.sum())


# =========================================================================== one portfolio
class PortfolioAnalytics:
    def __init__(self, pid: str, mk: Market, trades: pd.DataFrame, flows: pd.DataFrame):
        self.pid = pid
        self.meta = refdata.portfolios()[pid]
        self.mk = mk
        self.trades = trades[trades.portfolio == pid].copy()
        self.flows = flows[flows.portfolio == pid].copy()
        incep = pd.Timestamp(self.meta["inception"])
        self.days = mk.close.index[mk.close.index >= incep]
        self.inception = self.days[0]
        self._build_series()

    # ------------------------------------------------------------------ daily book
    def _map_day(self, s: pd.Series) -> pd.Series:
        """Map ledger dates to valuation days (first valuation day on or after; else the last)."""
        idx = self.days
        pos = idx.searchsorted(pd.to_datetime(s))
        pos = np.minimum(pos, len(idx) - 1)
        return pd.Series(idx[pos], index=s.index)

    def _build_series(self):
        mk, days = self.mk, self.days
        t = self.trades
        t["vday"] = self._map_day(t["date"]) if len(t) else pd.Series(dtype="datetime64[ns]")
        t["sq"] = t["quantity"].astype(float) * np.where(t["side"] == "buy", 1, -1)
        self.symbols = sorted(set(t["symbol"]))
        dq = t.pivot_table(index="vday", columns="symbol", values="sq", aggfunc="sum").reindex(days).fillna(0.0)
        dq = dq.reindex(columns=self.symbols, fill_value=0.0)
        Q = dq.cumsum()
        Q[Q.abs() < 1e-9] = 0.0
        self.Q = Q
        t["cash"] = -(t["sq"] * t["price"].astype(float)) - t["fees"].astype(float)
        trade_cash = t.groupby("vday")["cash"].sum().reindex(days).fillna(0.0)
        fees = t.groupby("vday")["fees"].sum().reindex(days).fillna(0.0).astype(float)
        f = self.flows
        f["vday"] = self._map_day(f["date"]) if len(f) else pd.Series(dtype="datetime64[ns]")
        f["amt"] = f["amount"].astype(float)
        ext = f[f["type"].isin(["contribution", "withdrawal"])].groupby("vday")["amt"].sum().reindex(days).fillna(0.0)
        allf = f.groupby("vday")["amt"].sum().reindex(days).fillna(0.0)
        cash = (allf + trade_cash).cumsum()
        px = mk.close.reindex(index=days, columns=self.symbols)
        mv_pos = (Q * px).sum(axis=1)
        self.cash = cash
        self.mv = mv_pos + cash
        self.ext = ext
        self.fees = fees
        prev = self.mv.shift(1)
        r = (self.mv - ext) / prev - 1
        r.iloc[0] = 0.0
        self.ret = r.fillna(0.0)
        # weights at the previous close and contributions
        wprev = (Q.shift(1).fillna(0.0) * px.shift(1)).div(prev, axis=0).fillna(0.0)
        self.wprev = wprev
        tr = mk.tr.reindex(index=days, columns=self.symbols).fillna(0.0)
        contrib = wprev * tr
        contrib.iloc[0] = 0.0
        self.contrib = contrib
        self.cost_drag = (fees / prev).fillna(0.0)
        self.cash_wprev = (cash.shift(1) / prev).fillna(0.0)
        # benchmark (daily-rebalanced, total return via adjusted closes)
        bw = refdata.benchmarks()[self.pid]["weights"]
        br = mk.adj_ret.reindex(index=days, columns=list(bw)).fillna(0.0)
        self.bench_w = bw
        self.bench_ret_by = br
        b = (br * pd.Series(bw)).sum(axis=1)
        b.iloc[0] = 0.0
        self.bret = b
        self.twr_index = (1 + self.ret).cumprod()
        self.bench_index = (1 + self.bret).cumprod()

    # ------------------------------------------------------------------ snapshot
    def current_weights(self) -> dict[str, float]:
        q = self.Q.iloc[-1]
        px = self.mk.close.iloc[-1]
        tv = float(self.mv.iloc[-1])
        w = {s: float(q[s] * px[s] / tv) for s in self.symbols if abs(q[s]) > 1e-9}
        w[CASH] = float(self.cash.iloc[-1] / tv)
        return w

    def holdings(self) -> list[dict]:
        """Average-cost book per symbol with realized and unrealized P&L."""
        book = {}
        for r in self.trades.sort_values(["date", "trade_id"]).itertuples():
            b = book.setdefault(r.symbol, {"qty": 0.0, "cost": 0.0, "realized": 0.0, "fees": 0.0, "first": r.date})
            q = float(r.quantity) * (1 if r.side == "buy" else -1)
            p = float(r.price)
            if q > 0:
                b["qty"] += q
                b["cost"] += q * p
            else:
                avg = b["cost"] / b["qty"] if b["qty"] else 0.0
                b["realized"] += -q * (p - avg)
                b["qty"] += q
                b["cost"] = avg * b["qty"]
                if abs(b["qty"]) < 1e-9:
                    b["qty"], b["cost"] = 0.0, 0.0
            b["fees"] += float(r.fees)
        divs = self.flows[self.flows["type"] == "dividend"].groupby("symbol")["amt"].sum()
        tv = float(self.mv.iloc[-1])
        prev_close = self.mk.close.iloc[-2] if len(self.mk.close) > 1 else self.mk.close.iloc[-1]
        model = model_weights(self.pid, self.days[-1])
        out = []
        for s, b in book.items():
            px = self.mk.price(s)
            mv = b["qty"] * px
            ins = refdata.instruments()[s]
            out.append({
                "symbol": s, "name": ins["name"], "kind": ins["kind"], "asset_class": ins["asset_class"],
                "sub_class": ins["sub_class"], "sector": ins["sector"] if ins["sector"] != "look-through" else "Diversified",
                "region": ins["region"] if ins["region"] != "look-through" else "Diversified",
                "qty": b["qty"], "avg_cost": b["cost"] / b["qty"] if b["qty"] else None, "price": px,
                "value": mv, "weight": mv / tv, "target": model.get(s, 0.0),
                "unrealized": mv - b["cost"], "realized": b["realized"], "dividends": float(divs.get(s, 0.0)),
                "fees": b["fees"], "day_change": b["qty"] * (px - float(prev_close[s])),
                "open": b["qty"] > 1e-9, "since": b["first"],
                "visitor": bool((self.trades[(self.trades.symbol == s)]["source"] == "visitor").any()),
            })
        out.sort(key=lambda h: (-h["value"], h["symbol"]))
        return out

    # ------------------------------------------------------------------ performance
    def performance(self) -> dict:
        days = self.days
        res = {}
        for p in PERIODS:
            s = period_start(p, days, self.inception)
            m = (days > s)
            r, b = self.ret[m], self.bret[m]
            tw = float((1 + r).prod() - 1)
            bw = float((1 + b).prod() - 1)
            yrs = (days[-1] - s).days / 365.25
            ann = (lambda x: (1 + x) ** (1 / yrs) - 1 if yrs > 1 else None)
            res[p] = {"start": s.strftime("%Y-%m-%d"), "twr": tw, "bench": bw, "excess": tw - bw,
                      "twr_ann": ann(tw), "bench_ann": ann(bw), "years": yrs}
        flows = [(self.inception.date(), -float(self.mv.iloc[0]))]
        for d, a in self.ext.items():
            if d != self.inception and a:
                flows.append((d.date(), -float(a)))
        flows.append((days[-1].date(), float(self.mv.iloc[-1])))
        res["SI"]["mwr_ann"] = xirr(flows)
        return res

    def ctr(self, period: str) -> dict:
        s = period_start(period, self.days, self.inception)
        m = self.days > s
        c = self.contrib[m].sum()
        total = float((1 + self.ret[m]).prod() - 1)
        arith = float(self.ret[m].sum())
        by_sym = {k: float(v) for k, v in c.items() if abs(v) > 1e-9}
        by_class = defaultdict(float)
        for k, v in by_sym.items():
            by_class[asset_class(k)] += v
        costs = -float(self.cost_drag[m].sum())
        resid = arith - sum(by_sym.values()) - costs
        return {"by_symbol": by_sym, "by_class": dict(by_class), "costs": costs, "residual": resid,
                "arith_total": arith, "twr": total}

    def brinson(self, period: str) -> dict:
        """Brinson-Fachler by asset class, daily, summed over the period."""
        s = period_start(period, self.days, self.inception)
        m = self.days > s
        classes = refdata.ASSET_CLASSES
        sym_cls = {sym: asset_class(sym) for sym in self.symbols}
        wp = {c: pd.Series(0.0, index=self.days) for c in classes}
        cp = {c: pd.Series(0.0, index=self.days) for c in classes}
        for sym in self.symbols:
            wp[sym_cls[sym]] = wp[sym_cls[sym]] + self.wprev[sym]
            cp[sym_cls[sym]] = cp[sym_cls[sym]] + self.contrib[sym]
        wp["Cash & Equivalents"] = wp["Cash & Equivalents"] + self.cash_wprev
        bcls = {sym: asset_class(sym) if sym in refdata.instruments() else "Equity" for sym in self.bench_w}
        wb = {c: sum(w for sym, w in self.bench_w.items() if bcls[sym] == c) for c in classes}
        rb_c = {}
        for c in classes:
            members = [sym for sym in self.bench_w if bcls[sym] == c]
            if wb[c] > 0:
                rb_c[c] = sum(self.bench_ret_by[sym] * self.bench_w[sym] for sym in members) / wb[c]
            else:
                rb_c[c] = None
        rb = self.bret
        out = {}
        tot = {"allocation": 0.0, "selection": 0.0, "interaction": 0.0}
        for c in classes:
            wpc = wp[c]
            rpc = (cp[c] / wpc.where(wpc.abs() > 1e-9)).fillna(0.0)
            rbc = rb_c[c] if rb_c[c] is not None else rpc  # no benchmark in class: allocation vs total bench only
            if rb_c[c] is None:
                alloc = (wpc - 0.0) * (rbc - rb)
                sel = pd.Series(0.0, index=self.days)
                inter = pd.Series(0.0, index=self.days)
            else:
                alloc = (wpc - wb[c]) * (rbc - rb)
                sel = wb[c] * (rpc - rbc)
                inter = (wpc - wb[c]) * (rpc - rbc)
                rpc = rpc.where(wpc.abs() > 1e-9, rbc)
            a, se, it = float(alloc[m].sum()), float(sel[m].sum()), float(inter[m].sum())
            out[c] = {"w_port": float(wpc[m].mean()), "w_bench": wb[c],
                      "r_port": float((1 + rpc[m]).prod() - 1) if wpc[m].abs().sum() > 0 else None,
                      "r_bench": float((1 + rb_c[c][m]).prod() - 1) if rb_c[c] is not None else None,
                      "allocation": a, "selection": se, "interaction": it, "total": a + se + it}
            tot["allocation"] += a
            tot["selection"] += se
            tot["interaction"] += it
        active_arith = float((self.ret[m] - rb[m]).sum())
        costs = -float(self.cost_drag[m].sum())
        total_effects = tot["allocation"] + tot["selection"] + tot["interaction"]
        tw = float((1 + self.ret[m]).prod() - 1)
        bw = float((1 + rb[m]).prod() - 1)
        return {"classes": out, "totals": tot, "costs": costs,
                "other": active_arith - total_effects - costs,
                "active_arith": active_arith, "active_geo": tw - bw}

    # ------------------------------------------------------------------ risk
    def _ret_matrix(self, syms: list[str]) -> pd.DataFrame:
        r = self.mk.adj_ret[syms].iloc[-RISK_DAYS:]
        return r.fillna(0.0)

    def risk(self) -> dict:
        w = self.current_weights()
        syms = [s for s in w if s != CASH]
        wv = np.array([w[s] for s in syms])
        R = self._ret_matrix(syms)
        port = R.values @ wv
        bsyms = list(self.bench_w)
        bR = self.mk.adj_ret[bsyms].iloc[-RISK_DAYS:].fillna(0.0)
        bench = bR.values @ np.array([self.bench_w[s] for s in bsyms])
        tv = float(self.mv.iloc[-1])

        def var_es(x, q):
            cut = np.quantile(x, 1 - q)
            return -cut, -x[x <= cut].mean()
        v95, e95 = var_es(port, 0.95)
        v99, e99 = var_es(port, 0.99)
        sd = port.std(ddof=1)
        bsd = bench.std(ddof=1)
        cov = np.cov(R.values.T)
        sig = math.sqrt(max(wv @ cov @ wv, 1e-18))
        mrc = cov @ wv / sig
        rc = wv * mrc  # contribution to daily vol; sums to sig
        rc_sym = {s: float(rc[i] / sig) for i, s in enumerate(syms)}
        rc_cls = defaultdict(float)
        for s, v in rc_sym.items():
            rc_cls[asset_class(s)] += v
        active = port - bench
        beta = float(np.cov(port, bench)[0, 1] / bench.var(ddof=1)) if bsd > 0 else None
        # realized (last 1Y) from TWR
        r1 = self.ret[self.days > period_start("1Y", self.days, self.inception)]
        b1 = self.bret[self.days > period_start("1Y", self.days, self.inception)]
        rf_daily = self.mk.adj_ret["BIL"].reindex(r1.index).fillna(0.0)
        ann = lambda x: float((1 + x).prod() ** (252 / max(len(x), 1)) - 1)
        rvol = float(r1.std(ddof=1) * math.sqrt(252))
        rf = ann(rf_daily)
        down = r1[r1 < 0]
        dd_dev = float(math.sqrt((down ** 2).sum() / max(len(r1), 1)) * math.sqrt(252))
        idx = self.twr_index
        dd = idx / idx.cummax() - 1
        hh = {s: v for s, v in w.items() if s != CASH}
        hhi = sum(v * v for v in hh.values())
        top10 = sum(sorted(hh.values(), reverse=True)[:10])
        return {
            "lookback_days": int(len(R)),
            "var95": float(v95), "es95": float(e95), "var99": float(v99), "es99": float(e99),
            "var95_usd": float(v95 * tv), "es95_usd": float(e95 * tv), "var99_usd": float(v99 * tv),
            "param_var95": float(1.645 * sd),
            "vol_exante": float(sd * math.sqrt(252)), "bench_vol": float(bsd * math.sqrt(252)),
            "te_exante": float(active.std(ddof=1) * math.sqrt(252)), "beta": beta,
            "vol_realized_1y": rvol,
            "te_realized_1y": float((r1 - b1).std(ddof=1) * math.sqrt(252)),
            "sharpe_1y": (ann(r1) - rf) / rvol if rvol > 0 else None,
            "sortino_1y": (ann(r1) - rf) / dd_dev if dd_dev > 0 else None,
            "rf_1y": rf,
            "max_drawdown": float(dd.min()), "max_drawdown_date": dd.idxmin().strftime("%Y-%m-%d"),
            "current_drawdown": float(dd.iloc[-1]),
            "hhi": hhi, "effective_n": 1 / hhi if hhi else None, "top10": top10,
            "risk_contrib": dict(sorted(rc_sym.items(), key=lambda kv: -kv[1])),
            "risk_contrib_class": dict(rc_cls),
            "var_hist": _hist(port)[0], "var_hist_edges": _hist(port)[1],
            "best_day": float(port.max()), "worst_day": float(port.min()),
        }

    # ------------------------------------------------------------------ exposures
    def exposures(self) -> dict:
        w = self.current_weights()
        out = {"asset_class": defaultdict(float), "sector": defaultdict(float), "region": defaultdict(float),
               "sub_class": defaultdict(float), "credit": defaultdict(float)}
        for s, v in w.items():
            cl = refdata.classify(s)
            for dim in ("asset_class", "sector", "region"):
                for k, x in cl[dim].items():
                    out[dim][k] += v * x
            if s != CASH:
                ins = refdata.instruments()[s]
                out["sub_class"][ins["sub_class"] if ins["kind"] == "etf" else "Single stocks"] += v
                if ins["credit"]:
                    out["credit"][ins["credit"]] += v
        bench = {"asset_class": defaultdict(float), "sector": defaultdict(float), "region": defaultdict(float)}
        for s, v in self.bench_w.items():
            cl = refdata.classify(s)
            for dim in bench:
                for k, x in cl[dim].items():
                    bench[dim][k] += v * x
        fi = {s: v for s, v in w.items() if s != CASH and refdata.instruments()[s]["duration"] is not None}
        fi_w = sum(fi.values())
        dur_fi = sum(v * refdata.instruments()[s]["duration"] for s, v in fi.items()) / fi_w if fi_w else None
        dur_port = sum(v * refdata.instruments()[s]["duration"] for s, v in fi.items())
        yields = {}
        for s in w:
            if s == CASH:
                continue
            px = self.mk.price(s)
            yields[s] = self.mk.ttm_div(s) / px if px else 0.0
        port_yield = sum(w[s] * y for s, y in yields.items())
        fi_yield = sum(fi[s] * yields[s] for s in fi) / fi_w if fi_w else None
        eq = sum(v for s, v in w.items() if s != CASH and refdata.instruments()[s]["asset_class"] == "Equity")
        eq_stocks = sum(v for s, v in w.items() if s != CASH and refdata.instruments()[s]["kind"] == "stock")
        return {
            "portfolio": {k: dict(v) for k, v in out.items()},
            "benchmark": {k: dict(v) for k, v in bench.items()},
            "fixed_income": {"weight": fi_w, "duration": dur_fi, "duration_contribution": dur_port,
                             "distribution_yield": fi_yield},
            "distribution_yield": port_yield, "yields": yields,
            "equity_split": {"equity": eq, "stocks": eq_stocks, "funds": eq - eq_stocks},
        }

    # ------------------------------------------------------------------ limits
    def limits(self, exp: dict, risk: dict) -> list[dict]:
        w = self.current_weights()
        ins = refdata.instruments()
        stocks = {s: v for s, v in w.items() if s != CASH and ins[s]["kind"] == "stock"}
        funds = {s: v for s, v in w.items() if s != CASH and ins[s]["kind"] == "etf"}
        sec = exp["portfolio"]["sector"]
        reg = exp["portfolio"]["region"]
        ac = exp["portfolio"]["asset_class"]
        cr = exp["portfolio"]["credit"]

        def top(d):
            return max(d.items(), key=lambda kv: kv[1]) if d else ("-", 0.0)
        out = []
        for lim in refdata.limits()[self.pid]:
            rule = lim["rule"]
            label, value, detail = rule, None, ""
            if rule == "max_single_stock":
                s, value = top(stocks); label, detail = "Largest single stock", s
            elif rule == "max_single_fund":
                s, value = top(funds); label, detail = "Largest single fund", s
            elif rule == "max_sector":
                s, value = top(sec); label, detail = "Largest sector (look-through)", s
            elif rule == "cash":
                value, label = w[CASH], "Uninvested cash"
            elif rule == "min_cash":
                value, label = w[CASH], "No overdraft (cash ≥ 0)"
            elif rule in ("equity", "fixed_income", "real_assets"):
                name = {"equity": "Equity", "fixed_income": "Fixed Income", "real_assets": "Real Assets"}[rule]
                value, label = ac.get(name, 0.0), f"{name} allocation"
            elif rule == "emerging_markets":
                value, label = reg.get("Emerging Markets", 0.0), "Emerging markets exposure"
            elif rule == "high_yield_of_portfolio":
                value, label = cr.get("High Yield", 0.0), "High-yield bonds"
            elif rule == "duration":
                value, label = exp["fixed_income"]["duration"], "Bond duration (years)"
            elif rule == "tracking_error":
                value, label = risk["te_exante"], "Tracking error (ex-ante)"
            elif rule == "var_95_1d":
                value, label = risk["var95"], "1-day VaR 95%"
            status = "ok"
            if value is not None:
                lo, hi = lim["min"], lim["max"]
                if (hi is not None and value > hi + 1e-12) or (lo is not None and value < lo - 1e-12):
                    status = "breach"
                elif hi is not None and hi > 0 and value > 0.9 * hi:
                    status = "warning"
                elif lo is not None and lo > 0 and value < 1.1 * lo:
                    status = "warning"
            out.append({"rule": rule, "label": label, "detail": detail, "value": value, "min": lim["min"],
                        "max": lim["max"], "severity": lim["severity"], "status": status,
                        "unit": "years" if rule == "duration" else "pct"})
        return out

    # ------------------------------------------------------------------ rebalancing
    def rebalancing(self) -> dict:
        w = self.current_weights()
        model = model_weights(self.pid, self.days[-1])
        tv = float(self.mv.iloc[-1])
        rows, trades = [], []
        for s in sorted(set(w) | set(model)):
            cur, tgt = w.get(s, 0.0), model.get(s, 0.0)
            tol = max(0.005, 0.20 * tgt)
            rows.append({"symbol": s, "weight": cur, "target": tgt, "drift": cur - tgt, "tolerance": tol,
                         "outside": abs(cur - tgt) > tol})
            if s != CASH and abs(cur - tgt) > tol:
                px = self.mk.price(s)
                q = math.floor(abs(tgt - cur) * tv / px) * (1 if tgt > cur else -1)
                if q:
                    trades.append({"symbol": s, "side": "buy" if q > 0 else "sell", "qty": abs(q), "price": px,
                                   "notional": abs(q) * px, "est_cost": max(5.0, abs(q) * px * 2e-4),
                                   "from": cur, "to": cur + q * px / tv,
                                   "reason": "not in model" if tgt == 0 else ("overweight" if q < 0 else "underweight")})
        rows.sort(key=lambda r: -abs(r["drift"]))
        net = sum(t["notional"] * (1 if t["side"] == "buy" else -1) for t in trades)
        return {"rows": rows, "trades": trades, "net_cash": -net,
                "turnover": sum(t["notional"] for t in trades) / tv if tv else 0.0,
                "max_drift": max((abs(r["drift"]) for r in rows), default=0.0)}

    # ------------------------------------------------------------------ stress
    def stress(self) -> list[dict]:
        w = self.current_weights()
        tv = float(self.mv.iloc[-1])
        ins = refdata.instruments()
        st = self.mk.stress
        out = []
        bw = self.bench_w
        for key, name, start, end, desc in refdata.STRESS_WINDOWS:
            win = st[st.scenario == key].pivot_table(index="date", columns="symbol", values="adj").sort_index()
            win = win[(win.index >= start) & (win.index <= end)]
            proxies = {}

            def series_for(sym):
                chain = [sym] + [p for p in (ins.get(sym, {}).get("stress_proxy") or "").split("|") if p]
                for c in chain:
                    if c in win.columns:
                        col = win[c]
                        if col.first_valid_index() is not None and col.first_valid_index() <= win.index[min(3, len(win) - 1)]:
                            if c != sym:
                                proxies[sym] = c
                            return col.ffill()
                proxies[sym] = "SPY"
                return win["SPY"].ffill()
            cols = {s: series_for(s) for s in w if s != CASH}
            rets = {s: float(c.iloc[-1] / c.dropna().iloc[0] - 1) for s, c in cols.items()}
            daily = pd.DataFrame({s: c.pct_change().fillna(0.0) for s, c in cols.items()})
            path = (1 + daily.values @ np.array([w[s] for s in daily.columns])).cumprod()
            port = float(sum(w[s] * r for s, r in rets.items()))
            bcols = {s: series_for(s) for s in bw}
            bench = float(sum(bw[s] * (c.iloc[-1] / c.dropna().iloc[0] - 1) for s, c in bcols.items()))
            contrib = sorted(((s, w[s] * r) for s, r in rets.items()), key=lambda kv: kv[1])
            out.append({"key": key, "name": name, "type": "historical", "start": start, "end": end,
                        "description": desc, "portfolio": port, "pnl": port * tv, "benchmark": bench,
                        "worst_point": float(path.min() - 1) if len(path) else None,
                        "worst": [{"symbol": s, "contribution": c} for s, c in contrib[:5]],
                        "proxied": proxies})
        # hypothetical shocks
        R = self.mk.adj_ret.iloc[-RISK_DAYS:]
        spy = R["SPY"]
        beta = {s: float(R[s].cov(spy) / spy.var()) for s in w if s != CASH}
        bbeta = {s: float(R[s].cov(spy) / spy.var()) for s in bw}

        def shock(fn, weights):
            return {s: fn(s) for s in weights if s != CASH}
        dur = lambda s: ins[s]["duration"] or 0.0
        credit = lambda s: ins[s]["credit"]
        sub = lambda s: ins[s]["sub_class"]
        scenarios = [
            ("eq_down_20", "Equity sell-off −20%", "S&P 500 −20%; every holding moves by its 2-year beta to the S&P 500.",
             lambda s, b: -0.20 * b[s]),
            ("rates_up_100", "Rates +100 bp", "Parallel +1% in yields; bonds lose duration × 1%. Equities unchanged.",
             lambda s, b: -dur(s) * 0.01 if dur(s) else 0.0),
            ("credit_event", "Credit event",
             "Spreads widen (IG +100 bp, HY +300 bp, EM +250 bp, preferreds +200 bp); equities −10% × beta.",
             lambda s, b: (-dur(s) * {"Investment Grade": .01, "High Yield": .03, "EM Sovereign": .025,
                                      "Preferred": .02}.get(credit(s), 0.0)) if dur(s) else -0.10 * b[s]),
            ("stagflation", "Stagflation",
             "Equities −15% × beta, yields +150 bp (TIPS +50 bp real), gold +15%, commodities +20%.",
             lambda s, b: (0.15 if sub(s) == "Gold" else 0.20 if sub(s) == "Broad Commodities" else
                           -dur(s) * (0.005 if sub(s) == "Inflation-Linked" else 0.015) if dur(s) else -0.15 * b[s])),
        ]
        for key, name, desc, fn in scenarios:
            rets = {s: fn(s, beta) for s in w if s != CASH}
            port = sum(w[s] * r for s, r in rets.items())
            bench = sum(bw[s] * fn(s, bbeta) for s in bw)
            contrib = sorted(((s, w[s] * r) for s, r in rets.items()), key=lambda kv: kv[1])
            out.append({"key": key, "name": name, "type": "hypothetical", "description": desc, "portfolio": port,
                        "pnl": port * tv, "benchmark": bench,
                        "worst": [{"symbol": s, "contribution": c} for s, c in contrib[:5]], "proxied": {}})
        return out

    # ------------------------------------------------------------------ income
    def income(self) -> dict:
        f = self.flows
        d = f[f["type"] == "dividend"].copy()
        d["month"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
        by_month = d.groupby("month")["amt"].sum()
        cutoff = (self.days[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        ttm = float(d[d["date"] > cutoff]["amt"].sum())
        q = self.Q.iloc[-1]
        proj = {s: float(q[s] * self.mk.ttm_div(s)) for s in self.symbols if q[s] > 0}
        wd = f[(f["type"] == "withdrawal") & (f["date"] > cutoff)]
        withdrawals = wd["amt"].sum() if len(wd) >= 6 else 0.0  # coverage only for regular income draws
        tv = float(self.mv.iloc[-1])
        return {"ttm": ttm, "projected_12m": sum(proj.values()), "projected_yield": sum(proj.values()) / tv,
                "by_month": [{"month": k, "amount": float(v)} for k, v in by_month.items()],
                "top_payers": sorted(({"symbol": s, "amount": v} for s, v in proj.items()), key=lambda r: -r["amount"])[:8],
                "withdrawals_ttm": -float(withdrawals), "coverage": ttm / -float(withdrawals) if withdrawals else None}

    # ------------------------------------------------------------------ assemble
    def report(self) -> dict:
        perf = self.performance()
        exp = self.exposures()
        risk = self.risk()
        lim = self.limits(exp, risk)
        hold = self.holdings()
        ctr = {p: self.ctr(p) for p in ("YTD", "1Y", "SI")}
        bri = {p: self.brinson(p) for p in ("YTD", "1Y", "SI")}
        tv = float(self.mv.iloc[-1])
        flows_ext = self.flows[self.flows["type"].isin(["contribution", "withdrawal"])]
        ytd0 = f"{self.days[-1].year}-01-01"
        t = self.trades.sort_values(["date", "trade_id"], ascending=False)
        blotter = [{"id": r.trade_id, "date": r.date, "symbol": r.symbol, "side": r.side, "qty": float(r.quantity),
                    "price": float(r.price), "amount": float(r.amount), "fees": float(r.fees), "reason": r.reason,
                    "source": r.source} for r in t.head(60).itertuples()]
        step = max(1, len(self.days) // 400)
        idx = list(range(0, len(self.days), step))
        if idx[-1] != len(self.days) - 1:
            idx.append(len(self.days) - 1)
        hist = {"dates": [self.days[i].strftime("%Y-%m-%d") for i in idx],
                "value": [round(float(self.mv.iloc[i]), 0) for i in idx],
                "twr": [round(float(self.twr_index.iloc[i]), 6) for i in idx],
                "bench": [round(float(self.bench_index.iloc[i]), 6) for i in idx],
                "net_flows": [round(float(self.ext.iloc[: i + 1].sum()), 0) for i in idx]}
        return {
            "id": self.pid, "name": self.meta["name"], "strategy": self.meta["strategy"],
            "description": self.meta["description"], "inception": self.inception.strftime("%Y-%m-%d"),
            "benchmark": refdata.benchmarks()[self.pid]["label"], "benchmark_weights": self.bench_w,
            "value": tv, "cash": float(self.cash.iloc[-1]),
            "day_change": float(self.mv.iloc[-1] - self.mv.iloc[-2] - self.ext.iloc[-1]),
            "day_return": float(self.ret.iloc[-1]),
            "net_flows_si": float(flows_ext["amt"].sum()),
            "net_flows_ytd": float(flows_ext[flows_ext["date"] >= ytd0]["amt"].sum()),
            "performance": perf, "ctr": ctr, "brinson": bri, "risk": risk, "exposures": exp, "limits": lim,
            "holdings": hold, "rebalancing": self.rebalancing(), "stress": self.stress(), "income": self.income(),
            "history": hist, "blotter": blotter,
            "model": model_weights(self.pid, self.days[-1]),
        }


def model_weights(pid: str, day: pd.Timestamp) -> dict[str, float]:
    from .simulate import model_on
    return model_on(pid, day.strftime("%Y-%m-%d"))


# =========================================================================== firm level
def load_ledger(include_visitors: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(refdata.DATA / "trades.csv", dtype=str)
    vt = refdata.DATA / "visitor_trades.csv"
    if include_visitors and vt.exists():
        v = pd.read_csv(vt, dtype=str)
        if len(v):
            t = pd.concat([t, v[t.columns]], ignore_index=True)
    f = pd.read_csv(refdata.DATA / "cash_flows.csv", dtype=str).fillna("")
    t = t.fillna("")
    for c in ("quantity", "price", "amount", "fees"):
        t[c] = t[c].astype(float)
    f["amount"] = f["amount"].astype(float)
    return t, f


def run(quotes: dict | None = None) -> dict:
    mk = Market(quotes)
    trades, flows = load_ledger()
    reports = {}
    pas = {}
    for pid in refdata.portfolios():
        pa = PortfolioAnalytics(pid, mk, trades, flows)
        pas[pid] = pa
        reports[pid] = pa.report()
    # firm roll-up
    days = mk.close.index[mk.close.index >= min(pa.inception for pa in pas.values())]
    step = max(1, len(days) // 400)
    idx = list(range(0, len(days), step))
    if idx[-1] != len(days) - 1:
        idx.append(len(days) - 1)
    aum = {pid: [round(float(pa.mv.reindex(days).fillna(0).iloc[i]), 0) for i in idx] for pid, pa in pas.items()}
    total = sum(r["value"] for r in reports.values())
    ins = refdata.instruments()
    universe = [{"symbol": s, "name": r["name"], "kind": r["kind"], "asset_class": r["asset_class"],
                 "sector": r["sector"], "region": r["region"], "duration": r["duration"], "credit": r["credit"],
                 "price": float(mk.close[s].iloc[-1]),
                 "classify": refdata.classify(s)}
                for s, r in ins.items() if r["kind"] != "benchmark"]
    visitors = trades[trades["source"] == "visitor"]
    return {
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices_asof": (max(q["time"] for q in quotes.values()) if quotes else
                        mk.last_close_day.strftime("%Y-%m-%d") + "T20:00:00+00:00"),
        "price_mode": "live" if mk.live_day is not None else "close",
        "last_close": mk.last_close_day.strftime("%Y-%m-%d"),
        "firm": {"aum": total, "portfolios": len(reports),
                 "day_change": sum(r["day_change"] for r in reports.values()),
                 "net_flows_ytd": sum(r["net_flows_ytd"] for r in reports.values()),
                 "breaches": sum(1 for r in reports.values() for l in r["limits"] if l["status"] == "breach"),
                 "warnings": sum(1 for r in reports.values() for l in r["limits"] if l["status"] == "warning"),
                 "history": {"dates": [days[i].strftime("%Y-%m-%d") for i in idx], "aum": aum},
                 "visitor_trades": int(len(visitors))},
        "portfolios": reports,
        "universe": universe,
        "limits": refdata.limits(),
        "sectors": refdata.SECTORS, "regions": refdata.REGIONS, "asset_classes": refdata.ASSET_CLASSES,
    }
