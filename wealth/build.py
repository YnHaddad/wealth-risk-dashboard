"""Build the dashboard: live quotes -> analytics -> site/index.html (one self-contained page).

    python -m wealth.build            # with live quotes (GitHub Actions)
    python -m wealth.build --offline  # last close only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import analytics, market, refdata

TEMPLATE = refdata.ROOT / "templates" / "dashboard.html"


def render(data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"), default=str, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA_JSON__", blob)
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def clean(o):
    """Replace NaN/inf with None so the JSON is valid."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(refdata.ROOT / "site"))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    quotes = {}
    if not a.offline:
        tickers = sorted(s for s, r in refdata.instruments().items()) + ["SPY"]
        quotes = market.live_quotes(tickers)
    data = clean(analytics.run(quotes))
    out = render(data, Path(a.out))
    f = data["firm"]
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"AUM ${f['aum']:,.0f}; prices {data['price_mode']} as of {data['prices_asof']}; "
          f"{len(quotes)} live quotes; {f['breaches']} breaches, {f['warnings']} warnings; "
          f"{f['visitor_trades']} visitor trades")
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=Dashboard built::AUM ${f['aum']:,.0f}; prices {data['price_mode']} "
              f"({len(quotes)} live quotes); {f['breaches']} breaches, {f['warnings']} warnings; "
              f"{f['visitor_trades']} visitor trades")
    return 0


if __name__ == "__main__":
    sys.exit(main())
