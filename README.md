# Portfolio Desk: sample multi-asset wealth dashboard

A working demo of a wealth-management portfolio desk. It runs five model portfolios across
the risk spectrum, each with its own benchmark and mandate limits. It covers:

- performance and attribution
- risk
- exposures and compliance
- stress tests
- rebalancing
- income

**The data is part real, part simulated.** Positions, clients and trades are simulated.
Prices, dividends and splits are real market data (Yahoo Finance, delayed). The dashboard
refreshes every 15 minutes during US market hours, and anyone with the link can place a
trade.

> Sample data only. Not investment advice. Not affiliated with any firm.

## The five portfolios

Every portfolio's equity sleeve is split **half ETFs, half single stocks**.

| Portfolio | Strategy | Mix (model) | Benchmark |
|---|---|---|---|
| Growth | Long-term capital growth, quality/tech tilt | 90% equity (5 ETFs + 15 large caps), 8% bonds, 2% cash | 90% MSCI ACWI / 10% US Agg |
| Equity Income | Dividend income from equities | 80% equity (dividend, covered-call, intl dividend ETFs + 12 dividend payers), 18% preferreds/credit, 2% cash | 70% US High Dividend / 15% Intl Dividend / 15% US Agg |
| Balanced | Classic 60/40 | 60% equity, 35% bonds incl. TIPS, 3% gold, 2% cash | 60% ACWI / 40% US Agg |
| Conservative Income | Capital preservation and yield | 25% defensive equity, 65% short/intermediate high-quality bonds, 10% T-bills and cash | 25% ACWI / 65% US Agg / 10% T-Bills |
| Global Multi-Asset | Diversified all-weather | 50% global equity (incl. 9 non-US leaders), 25% bonds (TIPS, EM, Agg), 23% real assets (gold, commodities, REITs, infrastructure) | 50% ACWI / 30% Agg / 10% TIPS / 5% Gold / 5% Commodities |

Starting AUM is $82M (initial funding on 2 Jan 2024).

## How it works

```
data/instruments.csv      approved universe: 48 stocks + 26 ETFs (+ benchmark proxies), classification
data/lookthrough.csv      approximate sector/region look-through for equity ETFs
data/portfolios.csv       mandates          data/models.csv       model weights
data/benchmarks.csv       benchmark blends  data/limits.csv       mandate limits (hard/soft)
data/model_changes.csv    dated model swaps
data/market/              cached market data (closes, adjusted closes, dividends, splits, crisis windows)
data/trades.csv           simulated trade ledger      data/cash_flows.csv   client flows and dividends
data/visitor_trades.csv   trades placed from the dashboard (expire after 7 days)

wealth/market.py      fetch / update market data            wealth/simulate.py   simulated history (incremental)
wealth/analytics.py   all analytics                         wealth/visitors.py   visitor trades + pre-trade compliance
wealth/build.py       writes site/index.html                functions/api/       Cloudflare Pages Functions (trade ticket)
```

The GitHub Actions workflows:

- **refresh.yml** runs every 15 minutes (13:00–21:45 UTC, weekdays). It takes live quotes,
  rebuilds the page, deploys it to Cloudflare Pages and checks that the site is serving it.
- **nightly.yml** runs after the US close. It updates the market cache, extends the
  simulation by a day and expires old visitor trades.
- **visitor-trade.yml** is started by the dashboard's trade ticket. It runs the compliance
  check, books the trade and triggers a rebuild.

## Simulation

The simulation is deterministic: there's a seed per portfolio and month, so re-running it
never changes the past.

Each portfolio is funded on 2 Jan 2024 and bought at its model weights. After that, the
simulation applies these rules:

- **Dividends** are credited to cash on the ex-date.
- **Client flows:** Equity Income and Conservative Income pay their clients a monthly income.
  Growth and Balanced receive periodic contributions. There are a few one-off withdrawals.
- **Cash:** new cash is invested in batches once it exceeds the cash target by more than 1%
  of the portfolio. Cash is topped up when it falls below half its target, selling
  overweight positions first.
- **Rebalancing:** on the first trading day of each quarter, positions outside tolerance are
  traded back to target. Tolerance is ±20% of the target weight, with a minimum of 0.5 points.
- **Model changes:** four dated changes swap one holding for another.
- **Execution:** at the close, in whole shares, with 2 bp commission (minimum $5).
- **Splits:** quantities and prices are kept in split-adjusted units. A new split re-bases
  the earlier ledger rows for that stock.

## Methodology

| Measure | Definition |
|---|---|
| Time-weighted return | Daily: r = (MV_t − F_t) / MV_{t−1} − 1, where client flows F arrive at the close. Dividends are income, not flows. Periods chain-link. |
| Money-weighted return | IRR of client flows plus ending value |
| Benchmark | Daily-rebalanced blend of index ETFs, total return |
| Contribution to return | Σ w_{i,t−1} · r_{i,t} with total returns (price plus dividend on the ex-date). Trading costs are shown separately; the residual is zero. |
| Attribution | Brinson-Fachler by asset class, daily, summed over the period (allocation, selection, interaction). Effects plus costs equal the arithmetic active return. |
| VaR / Expected Shortfall | Historical, 1-day, at 95% and 99%, on current weights × up to 504 days of daily total returns |
| Volatility, tracking error, beta | Ex-ante, on the same return history. Realized versions use the last 12 months of time-weighted returns. |
| Sharpe / Sortino | Last 12 months. The risk-free rate is the T-bill ETF's return. |
| Risk contribution | w_i (Σw)_i / σ_p, the share of portfolio volatility from each holding |
| Concentration | HHI, effective number of holdings (1/HHI), top-10 weight |
| Exposures | Asset class, sector and region of the whole portfolio. Equity ETFs are looked through with approximate, rounded weights. |
| Fixed income | Weighted duration (approximate fund durations); yield is the trailing 12-month distributions over price |
| Historical stress | Each current holding's actual total return over the crisis window. Holdings without history then use a sector or asset-class proxy, and these are flagged. |
| Hypothetical stress | Equity −20% (2-year betas to the S&P 500); rates +100 bp (−duration × 1%); a credit event (spread durations); stagflation |
| Income | Dividends received over the trailing 12 months. The projection is today's units × trailing-12-month dividend per share. |

## Visitor trades

The **New trade** button needs no login, by design.

**The browser previews the compliance check before an order is sent.** A workflow then
re-checks the order in Python before booking it. These are the rules:

- **Approved universe:** the symbol must be one of the approved instruments.
- **Whole shares only.**
- **No shorting and no overdraft.**
- **Hard limits:** an order is blocked if it would newly breach a hard limit, or worsen an
  existing breach. Hard limits cover single stocks, single funds and asset-class ranges.
  Soft limits only warn.
- **Trade size:** at most 25% of the portfolio per trade.
- **Daily cap:** at most 100 visitor trades per day.

**Booking:** trades execute at the latest market price, are marked "visitor" everywhere,
and expire after 7 days, so the book resets.

## Running locally

```bash
pip install -r requirements.txt
python -m wealth.market --update        # refresh market data (needs internet)
python -m wealth.simulate               # extend the simulation
python -m wealth.build --offline        # writes site/index.html at the last close
```

## Setup (owner)

Add these repository secrets:

| Secret | Purpose |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Account · Cloudflare Pages · Edit |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account |
| `GH_LOG_TOKEN` | Fine-grained GitHub token, this repo only: **Actions read/write**, **Contents read**. Enables the trade ticket. |

After adding or changing `GH_LOG_TOKEN`, run *Actions → Refresh dashboard → Run workflow*
once to push it to Cloudflare.

## Limitations

- **Approximate inputs:** ETF look-through weights and bond durations are approximate, rounded
  estimates. Treat sector, region and duration figures as indicative.
- **Price feed:** yfinance is an unofficial client of Yahoo Finance. Quotes are delayed, and a
  failed fetch falls back to the cached close.
- **USD only:** all instruments are US-listed in USD (ADRs for non-US companies). There's no
  FX layer.
- **Gross returns:** no management fees are deducted, and uninvested cash earns no interest
  (the T-bill ETF does).
