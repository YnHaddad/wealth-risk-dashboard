// POST /api/log — public trade ticket for the sample dashboard (no login by design).
// Validates the order's shape, then starts the "Visitor trade" workflow, which runs the
// authoritative pre-trade compliance check in Python, books the trade at the latest market
// price and rebuilds the dashboard. Needs Pages secrets GH_LOG_TOKEN and GH_REPO.

const PORTFOLIOS = ["growth", "equity_income", "balanced", "conservative_income", "multi_asset"];
const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });

export async function onRequestPost({ request, env }) {
  if (!env.GH_LOG_TOKEN || !env.GH_REPO) return json(503, { error: "Trading isn't switched on for this site yet." });
  const origin = request.headers.get("Origin");
  if (!origin || new URL(origin).host !== new URL(request.url).host) return json(403, { error: "Cross-origin request refused." });
  if (!(request.headers.get("Content-Type") || "").includes("application/json")) return json(415, { error: "Send JSON." });
  if (Number(request.headers.get("Content-Length") || 0) > 2000) return json(413, { error: "Order too large." });
  let e;
  try { e = await request.json(); } catch { return json(400, { error: "Body is not valid JSON." }); }
  const order = {
    portfolio: String(e.portfolio || ""),
    side: String(e.side || ""),
    symbol: String(e.symbol || "").trim().toUpperCase(),
    qty: Number(e.qty),
  };
  if (!PORTFOLIOS.includes(order.portfolio)) return json(400, { error: "Pick a portfolio." });
  if (!["buy", "sell"].includes(order.side)) return json(400, { error: "Side must be buy or sell." });
  if (!/^[A-Z0-9.-]{1,10}$/.test(order.symbol)) return json(400, { error: "Enter a ticker symbol." });
  if (!Number.isInteger(order.qty) || order.qty <= 0 || order.qty > 10000000) return json(400, { error: "Enter a whole number of shares." });

  const id = `${Date.now().toString(36)}-${crypto.randomUUID().slice(0, 6)}`;
  const r = await fetch(`https://api.github.com/repos/${env.GH_REPO}/actions/workflows/visitor-trade.yml/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_LOG_TOKEN}`, Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "portfolio-desk", "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main", inputs: { id, payload: JSON.stringify(order) } }),
  });
  if (!r.ok) return json(502, { error: `Couldn't reach the trading backend (HTTP ${r.status}). Try again in a minute.` });
  return json(202, { id });
}
