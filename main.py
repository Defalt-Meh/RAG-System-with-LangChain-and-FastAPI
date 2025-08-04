"""
Main application entrypoint for the RAG API.

Principles:
- Keep FastAPI app assembly here (metadata, middleware, routers).
- Keep domain logic out of the web layer (lives in rag.py, etc.).
- Fail fast on misconfig, but never crash the server for non-critical extras.
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Your version string; bump on meaningful changes.
APP_NAME = "RAG API"
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# Router with /query endpoint(s)
from endpoints import router  # noqa: E402

logger = logging.getLogger("uvicorn.error")


# -----------------------------------------------------------------------------
# Optional: lightweight app startup to warm caches / load index.
# If rag.init() doesn’t exist, we degrade gracefully.
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle.
    Use this to pre-load vector indexes or models so the first request isn't slow.
    """
    try:
        # Deliberately optional to keep web layer decoupled.
        from rag import init as rag_init  # type: ignore

        try:
            await rag_init()
            logger.info("RAG backend initialized successfully.")
        except TypeError:
            # Backward compatibility: allow sync init()
            rag_init()
            logger.info("RAG backend initialized successfully (sync).")
        except Exception as e:
            # Do not crash the app; serve 503 at query-time instead.
            logger.warning(f"RAG backend initialization failed: {e}")
    except Exception:
        # No init available; that’s acceptable.
        logger.info("No rag.init() found; skipping backend warm-up.")
    yield
    # Place any async cleanup here if needed (closing DB, flushing traces, etc.)


# -----------------------------------------------------------------------------
# App assembly
# -----------------------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    summary="Retrieval-Augmented Generation over your documents.",
    description=(
        "A minimal but production-conscious RAG service. "
        "Use `/docs` for interactive exploration."
    ),
    lifespan=lifespan,
)

# CORS: allow local dev tools and simple frontends. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers (kept thin and testable)
app.include_router(router)


# -----------------------------------------------------------------------------
# Convenience endpoints (DX and uptime checks)
# -----------------------------------------------------------------------------
@app.get("/", tags=["Meta"], summary="Service info")
def root():
    """
    Quick pointer for humans. Avoids 404s on `/`.
    """
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/healthz",
        "query_endpoint": "/query/",
        "ui": "/ui",
    }


@app.get("/healthz", tags=["Meta"], summary="Liveness/Readiness probe")
def healthz():
    """
    Basic liveness check. Extend to verify vector store/model readiness if needed.
    Keep this fast and allocation-free.
    """
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ui", tags=["Meta"], summary="Simple UI", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RAG Console — 40K Codex</title>
<style>
  /* ---------- Warhammer 40K Theme ---------- */
  :root {
    --bg: #0a0b0d;           /* void black */
    --surface: #121417;      /* gunmetal */
    --surface-2: #1a1d22;    /* deeper panel */
    --text: #e6e6e6;         /* pale parchment */
    --muted: #a2a6ad;        /* steel gray */
    --accent: #c8aa6e;       /* sanctified gold */
    --accent-2: #7a1d1d;     /* dried crimson */
    --border: #2a2f37;       /* riveted edge */
    --ok: #59d39c;           /* lumen green */
    --err: #ef4444;
    --shadow: 0 10px 28px rgba(0,0,0,.45);
    --ring: 0 0 0 2px rgba(200,170,110,.25);
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f7f7f5; --surface: #ffffff; --surface-2: #fafafa;
      --text: #111215; --muted: #686c73; --accent: #b18d4d; --accent-2: #8b1f1f;
      --border: #e6e6e8; --shadow: 0 10px 28px rgba(0,0,0,.1); --ring: 0 0 0 2px rgba(177,141,77,.22);
    }
  }
  [data-theme="light"] {
    --bg: #f7f7f5; --surface: #ffffff; --surface-2: #fafafa; --text: #111215; --muted: #686c73;
    --accent: #b18d4d; --accent-2: #8b1f1f; --border: #e6e6e8; --shadow: 0 10px 28px rgba(0,0,0,.1); --ring: 0 0 0 2px rgba(177,141,77,.22);
  }
  [data-theme="dark"] {
    --bg: #0a0b0d; --surface: #121417; --surface-2: #1a1d22; --text: #e6e6e6; --muted: #a2a6ad;
    --accent: #c8aa6e; --accent-2: #7a1d1d; --border: #2a2f37; --shadow: 0 10px 28px rgba(0,0,0,.45); --ring: 0 0 0 2px rgba(200,170,110,.25);
  }

  /* ---------- Layout & Components ---------- */
  html,body { height:100%; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font: 16px/1.45 ui-sans-serif, -apple-system, system-ui, Segoe UI, Roboto, "Helvetica Neue", Arial;
  }
  .container { max-width: 960px; margin: 32px auto; padding: 0 16px; }

  .header {
    display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:18px;
  }
  .brand { display:flex; align-items:center; gap:12px; }
  .logo {
    width:40px; height:40px; border-radius:10px;
    background: radial-gradient(60% 60% at 50% 40%, rgba(200,170,110,.55), transparent 70%),
                linear-gradient(135deg, rgba(200,170,110,.9), rgba(122,29,29,.85));
    box-shadow: var(--shadow);
    position: relative;
  }
  /* minimalist Aquila impression */
  .logo:before, .logo:after {
    content:""; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:26px; height:2px; background: #221d11; opacity:.35;
  }
  .logo:before { transform: translate(-50%,-50%) rotate(18deg); }
  .logo:after  { transform: translate(-50%,-50%) rotate(-18deg); }

  h1 { font-size:20px; margin:0; letter-spacing:.3px; }
  .meta { color:var(--muted); font-size:13px; }

  .card {
    background:var(--surface);
    border:1px solid var(--border); border-radius:14px; padding:16px; box-shadow: var(--shadow);
  }
  .controls { display:grid; grid-template-columns: 1fr auto auto; gap:10px; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }

  input[type="text"] {
    width:100%; padding:12px 12px; border-radius:10px;
    border:1px solid var(--border); background:var(--surface-2); color:var(--text); outline:none;
  }
  input[type="text"]:focus { box-shadow: var(--ring); border-color: var(--accent); }
  input::placeholder { color:var(--muted); }

  select, .chk {
    padding:10px; border-radius:10px; border:1px solid var(--border);
    background:var(--surface-2); color:var(--text);
  }

  button {
    padding:12px 16px; border-radius:10px; border:1px solid var(--accent);
    background: linear-gradient(180deg, var(--accent), #9c7d46);
    color:#0d0e10; cursor:pointer; font-weight:700; letter-spacing:.2px;
  }
  button.secondary {
    background: var(--surface-2); color: var(--text); border-color: var(--border);
  }
  button:disabled { opacity:.6; cursor:wait; }

  .samples { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }
  .chip {
    padding:6px 10px; border-radius:999px; border:1px solid var(--border);
    background:var(--surface-2); color:var(--muted); cursor:pointer; user-select:none;
  }
  .chip:hover { border-color: var(--accent); color: var(--text); }

  .answer { margin-top:16px; }
  .title { font-size:14px; color:var(--muted); margin:0 0 6px; }
  .answer-box {
    white-space:pre-wrap; background:var(--surface-2); border:1px solid var(--border);
    padding:14px; border-radius:12px; min-height:72px;
  }
  .sources { margin-top:12px; display:grid; gap:8px; }
  .source {
    background:var(--surface-2); border:1px dashed var(--border);
    border-radius:10px; padding:10px; font-size:14px;
  }

  .status { display:flex; gap:10px; align-items:center; color:var(--muted); font-size:13px; margin-top:10px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--ok); display:inline-block; }

  .err { border-color: rgba(239,68,68,.5) !important; }
  .footer { margin-top:18px; color:var(--muted); font-size:12px; text-align:center; }

  .spinning { animation: spin 1s linear infinite; display:inline-block; }
  @keyframes spin { from { transform: rotate(0) } to { transform: rotate(360deg) } }
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="logo" aria-hidden="true"></div>
        <div>
          <h1>RAG Codex: M41</h1>
          <div class="meta">Query your compendium • <span id="mode">mode: ?</span></div>
        </div>
      </div>
      <div class="row">
        <button id="theme" class="secondary" title="Toggle theme (t)">🌓 Theme</button>
        <a class="secondary" href="/docs" style="text-decoration:none;display:inline-block;padding:12px 16px;border-radius:10px;border:1px solid var(--border);">API Docs</a>
      </div>
    </div>

    <div class="card">
      <div class="controls">
        <input id="q" type="text" placeholder="Ask the Codex… (Press Enter to submit)" />
        <select id="topk" title="Chunks to retrieve">
          <option>4</option><option>3</option><option>5</option><option>6</option><option>8</option>
        </select>
        <label class="row" style="justify-content:flex-end; gap:6px;">
          <input id="trace" class="chk" type="checkbox" /> trace
        </label>
      </div>

      <!-- 40K-flavored samples -->
      <div class="samples">
        <span class="chip">Who is Abaddon the Despoiler?</span>
        <span class="chip">Explain the Astronomican in two sentences.</span>
        <span class="chip">What is Exterminatus and when is it used?</span>
        <span class="chip">Compare Eldar and Drukhari in one paragraph.</span>
      </div>

      <div class="row" style="margin-top:10px;">
        <button id="go">Query</button>
        <button id="copy" class="secondary" title="Copy answer (⌘/Ctrl+C)">Copy</button>
        <button id="clear" class="secondary">Clear</button>
      </div>

      <div class="status" id="status" style="display:none;">
        <span class="spinning">⛭</span> <span>Consulting the archives…</span>
      </div>

      <div class="answer">
        <p class="title">Answer</p>
        <div id="answer" class="answer-box"></div>
      </div>

      <div class="answer">
        <p class="title">Sources</p>
        <div id="sources" class="sources"></div>
      </div>

      <div class="status">
        <span class="dot"></span>
        <span id="meta">latency: – • request_id: –</span>
      </div>
    </div>

    <div class="footer">Tip: Press <span class="kbd">Enter</span> to ask, <span class="kbd">⌘/Ctrl</span> + <span class="kbd">C</span> to copy. Theme remembers your last choice.</div>
  </div>

<script>
(function(){
  const $ = id => document.getElementById(id);
  const q = $("q"), topk = $("topk"), trace = $("trace");
  const go = $("go"), copyBtn = $("copy"), clearBtn = $("clear");
  const answer = $("answer"), sources = $("sources"), meta = $("meta");
  const status = $("status"), modeSpan = $("mode");
  const themeBtn = $("theme");

  // theme
  const applyTheme = (t) => document.documentElement.setAttribute("data-theme", t);
  let theme = localStorage.getItem("theme") || "";
  if (theme) applyTheme(theme);
  themeBtn.onclick = () => { theme = (theme==="dark"?"light":"dark"); localStorage.setItem("theme", theme); applyTheme(theme); };
  document.addEventListener("keydown", e => { if (e.key.toLowerCase() === "t") themeBtn.click(); });

  // samples
  document.querySelectorAll(".chip").forEach(ch => ch.addEventListener("click", () => { q.value = ch.textContent; q.focus(); }));

  // ask
  async function ask(){
    const text = q.value.trim();
    if (!text) { q.classList.add("err"); setTimeout(()=>q.classList.remove("err"), 600); return; }

    setLoading(true);
    try {
      const url = "/query/?" + new URLSearchParams({
        query: text, top_k: topk.value, trace: trace.checked ? "true" : "false"
      });
      const r = await fetch(url);
      const raw = await r.text();
      if (!r.ok) {
        answer.classList.add("err"); answer.textContent = `Error ${r.status}: ${raw}`;
        sources.innerHTML = "";
        meta.textContent = "latency: – • request_id: –";
        return;
      }
      const data = JSON.parse(raw);
      modeSpan.textContent = "mode: " + (data.mode || "?");
      answer.classList.remove("err");
      answer.textContent = (data.answer || "").trim();
      meta.textContent = `latency: ${data.latency_ms ?? "?"} ms • request_id: ${data.request_id || "-"}`;
      sources.innerHTML = (data.sources || []).map(s =>
        `<div class="source"><b>${(s.source||"unknown")}</b><div>${escapeHtml(s.snippet||"")}</div></div>`
      ).join("") || `<div class="source">No sources returned.</div>`;
    } catch (e) {
      answer.classList.add("err"); answer.textContent = "Request failed: " + e;
      sources.innerHTML = "";
      meta.textContent = "latency: – • request_id: –";
    } finally { setLoading(false); }
  }

  function setLoading(on){
    go.disabled = on; copyBtn.disabled = on; clearBtn.disabled = on;
    status.style.display = on ? "flex" : "none";
  }

  function escapeHtml(s){ return s.replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;", '"':"&quot;","'":"&#039;" }[m])); }

  // copy / clear
  copyBtn.onclick = async () => {
    const txt = answer.textContent.trim();
    if (!txt) return;
    try { await navigator.clipboard.writeText(txt); copyBtn.textContent = "Copied"; setTimeout(()=>copyBtn.textContent="Copy", 900); } catch {}
  };
  clearBtn.onclick = () => { q.value=""; answer.textContent=""; sources.innerHTML=""; meta.textContent="latency: – • request_id: –"; q.focus(); };

  // shortcuts
  q.addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
  document.addEventListener("keydown", e => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "c") { e.preventDefault(); copyBtn.click(); }
  });

  // initial focus + mode fetch
  q.focus();
  fetch("/healthz").then(r=>r.ok && (modeSpan.textContent = modeSpan.textContent.replace("?", "free/openai")));
  go.onclick = ask;
})();
</script>
</body>
</html>
    """

