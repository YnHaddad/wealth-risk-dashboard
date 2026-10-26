"""Market data: daily closes, adjusted closes, dividends, splits and stress-window history.

Cached under data/market/ so builds work (and stay reproducible) even if the price feed
hiccups. Source: Yahoo Finance via yfinance (unofficial, delayed). Used for a demo only.

    python -m wealth.market --full      # rebuild the whole cache (first run)
    python -m wealth.market --update    # refresh the last few weeks into the cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import refdata

log = logging.getLogger(__name__)
HISTORY_START = "2023-01-01"
CLOSE = refdata.MARKET / "close.csv"
ADJ = refdata.MARKET / "adj_close.csv"
DIVS = refdata.MARKET / "dividends.csv"
SPLITS = refdata.MARKET / "splits.csv"
STRESS = refdata.MARKET / "stress_windows.csv"


def _yf():
    import yfinance as yf
    return yf


def _download(tickers: list[str], **kw) -> pd.DataFrame:
    yf = _yf()
    df = yf.download(tickers, auto_adjust=False, progress=False, threads=True, group_by="column", **kw)
    if df is None or df.empty:
        raise RuntimeError("empty download")
    return df


def _wide(df: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    x = df[field]
    if isinstance(x, pd.Series):
        x = x.to_frame(tickers[0])
    x = x.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    x.index.name = "date"
    return x.reindex(columns=[t for t in tickers if t in x.columns]).sort_index()


def fetch_history(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _download(tickers, start=start)
    return _wide(df, "Close", tickers), _wide(df, "Adj Close", tickers)


def fetch_actions(tickers: list[str], since: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    yf = _yf()
    divs, splits = [], []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            for ix, v in tk.dividends.items():
                d = pd.Timestamp(ix).tz_localize(None).strftime("%Y-%m-%d")
                if d >= since and v > 0:
                    divs.append((d, t, float(v)))
            for ix, v in tk.splits.items():
                d = pd.Timestamp(ix).tz_localize(None).strftime("%Y-%m-%d")
                if d >= since and v > 0:
                    splits.append((d, t, float(v)))
        except Exception as e:  # noqa: BLE001
            log.warning("actions failed for %s: %s", t, e)
    return (pd.DataFrame(divs, columns=["date", "symbol", "amount"]).sort_values(["date", "symbol"]),
            pd.DataFrame(splits, columns=["date", "symbol", "ratio"]).sort_values(["date", "symbol"]))


def fetch_stress(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for key, _, start, end, _ in refdata.STRESS_WINDOWS:
        s = (date.fromisoformat(start) - timedelta(days=10)).isoformat()
        e = (date.fromisoformat(end) + timedelta(days=5)).isoformat()
        try:
            _, adj = fetch_history(tickers, s)
            adj = adj.loc[:e]
            long = adj.reset_index().melt(id_vars="date", var_name="symbol", value_name="adj").dropna()
            long.insert(0, "scenario", key)
            frames.append(long)
        except Exception as e:  # noqa: BLE001
            log.warning("stress window %s failed: %s", key, e)
    out = pd.concat(frames)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["adj"] = out["adj"].round(4)
    return out


def _save_wide(df: pd.DataFrame, path) -> None:
    out = df.round(4).copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.to_csv(path, index_label="date")


def load_wide(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return df.sort_index()


def full_refresh() -> None:
    tickers = refdata.all_market_tickers()
    refdata.MARKET.mkdir(parents=True, exist_ok=True)
    close, adj = fetch_history(tickers, HISTORY_START)
    _save_wide(close, CLOSE)
    _save_wide(adj, ADJ)
    divs, splits = fetch_actions(tickers, HISTORY_START)
    divs.to_csv(DIVS, index=False)
    splits.to_csv(SPLITS, index=False)
    fetch_stress(tickers).to_csv(STRESS, index=False)
    print(f"cache: {len(close)} days x {close.shape[1]} tickers; {len(divs)} dividends; {len(splits)} splits")


def update_cache(days: int = 30) -> None:
    """Re-download the last `days` days and overwrite those rows (prices get revised, and
    a split re-bases history, in which case the whole series is re-fetched)."""
    tickers = refdata.all_market_tickers()
    close, adj = load_wide(CLOSE), load_wide(ADJ)
    missing = [t for t in tickers if t not in close.columns]
    start = (date.today() - timedelta(days=days)).isoformat()
    new_c, new_a = fetch_history(tickers, start)
    divs, splits = fetch_actions(tickers, HISTORY_START)
    old_splits = pd.read_csv(SPLITS) if SPLITS.exists() else pd.DataFrame(columns=["date", "symbol", "ratio"])
    resplit = set(splits["symbol"]) - set(old_splits["symbol"]) | set(missing)
    if resplit:  # history for these tickers is re-based (or new): fetch it whole
        log.info("re-fetching full history for %s", sorted(resplit))
        rc, ra = fetch_history(sorted(resplit), HISTORY_START)
        for t in rc.columns:
            close = close.reindex(close.index.union(rc.index))
            adj = adj.reindex(adj.index.union(ra.index))
            close[t] = rc[t]
            adj[t] = ra[t]
    close = pd.concat([close[close.index < new_c.index.min()], new_c]).sort_index()
    adj = pd.concat([adj[adj.index < new_a.index.min()], new_a]).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    adj = adj[~adj.index.duplicated(keep="last")]
    _save_wide(close, CLOSE)
    _save_wide(adj, ADJ)
    divs.to_csv(DIVS, index=False)
    splits.to_csv(SPLITS, index=False)
    if not STRESS.exists():
        fetch_stress(tickers).to_csv(STRESS, index=False)
    print(f"cache updated to {close.index.max().date()} ({close.shape[1]} tickers)")


def live_quotes(tickers: list[str]) -> dict[str, dict]:
    """Latest intraday price per ticker (15-minute bars, delayed)."""
    out: dict[str, dict] = {}
    try:
        df = _download(tickers, period="5d", interval="15m", prepost=False)
        c = df["Close"]
        if isinstance(c, pd.Series):
            c = c.to_frame(tickers[0])
        for t in c.columns:
            s = c[t].dropna()
            if len(s):
                ts = pd.Timestamp(s.index[-1])
                ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
                out[t] = {"price": float(s.iloc[-1]), "time": ts.isoformat()}
    except Exception as e:  # noqa: BLE001
        log.warning("live quotes failed: %s", e)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--update", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.full or not CLOSE.exists():
        full_refresh()
    else:
        update_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
