// GET /api/status            -> { configured }
// GET /api/status?id=<id>    -> progress of a dashboard order:
//   { configured, log: {status, conclusion, ok, message} | null, refresh: {status, conclusion} | null }

const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });

function gh(env, path, accept = "application/vnd.github+json") {
  return fetch(`https://api.github.com/repos/${env.GH_REPO}${path}`, {
    headers: { Authorization: `Bearer ${env.GH_LOG_TOKEN}`, Accept: accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "portfolio-desk" },
  });
}
async function runs(env, wf) {
  const r = await gh(env, `/actions/workflows/${wf}/runs?per_page=15`);
  if (!r.ok) throw new Error(`GitHub HTTP ${r.status}`);
  return (await r.json()).workflow_runs || [];
}
function rebuildState(list, since) {
  const after = list.filter((r) => r.created_at >= since);
  if (!after.length) return null;
  if (after.some((r) => r.conclusion === "success")) return { status: "completed", conclusion: "success" };
  if (after.some((r) => r.status !== "completed")) return { status: "in_progress", conclusion: null };
  return { status: "completed", conclusion: after[0].conclusion };
}

export async function onRequestGet({ request, env }) {
  const configured = Boolean(env.GH_LOG_TOKEN && env.GH_REPO);
  const id = new URL(request.url).searchParams.get("id");
  if (!configured || !id) return json(200, { configured });
  if (!/^[a-z0-9-]{6,40}$/.test(id)) return json(400, { configured, error: "bad id" });
  try {
    const run = (await runs(env, "visitor-trade.yml")).find((r) => r.display_title === `Order ${id}`);
    let log = null, refresh = null;
    if (run) {
      log = { status: run.status, conclusion: run.conclusion, ok: null, message: null };
      if (run.status === "completed") {
        const f = await gh(env, "/contents/data/visitor_requests.json?ref=main", "application/vnd.github.raw");
        if (f.ok) {
          const rec = (await f.json())[id];
          if (rec) { log.ok = rec.ok; log.message = rec.message; }
        }
        if (log.ok === null && run.conclusion !== "success") {
          log.ok = false;
          log.message = "The order couldn't be processed right now. Please try again.";
        }
        if (log.ok === false) log.conclusion = "failure";
        if (log.ok && run.conclusion === "success") refresh = rebuildState(await runs(env, "refresh.yml"), run.created_at);
      }
    }
    return json(200, { configured, log, refresh });
  } catch (err) {
    return json(502, { configured, error: String(err.message || err) });
  }
}
