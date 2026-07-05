"""
Lumen backend - FastAPI entrypoint.

Hosts:
  - GET  /health  : lightweight liveness probe
  - WS   /ws      : single WebSocket endpoint, one connection per session
  - GET  /*       : the frontend (../frontend), mounted last so the routes
                    above take priority. Serving the page same-origin is what
                    makes wss://<same-host>/ws work behind any tunnel/host.

The Session class (api/session.py) holds per-connection state. The Router
(api/router.py) parses incoming messages and dispatches to handlers. The FSM
(fsm/task_fsm.py) is authoritative for task state - the client maintains no
state of its own beyond a "connected" indicator.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Open http://localhost:8000 in a browser - that loads the frontend and the
frontend opens the WebSocket back to the same origin automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

import os

from api.session import Session


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that tells the browser not to cache the frontend.

    Mobile browsers (especially iOS Safari) hold onto cached JS aggressively
    and ignore standard revalidation on WebSocket / MP3 flows. During
    iteration a stale ``app.js`` will silently reintroduce bugs the fix
    already removed - we've been bitten by exactly this. Serving the frontend
    with ``Cache-Control: no-cache, no-store, must-revalidate`` means every
    page load fetches the current source. It's a few extra KBs on the wire,
    fine for our use.
    """

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lumen.main")

app = FastAPI(title="Lumen Backend", version="0.1.0")


@app.on_event("startup")
def _maybe_preload_navigation() -> None:
    """Opt-in warmup: LUMEN_NAV_PRELOAD=1 loads the navigation models and
    pre-synthesizes the fixed guidance phrases in a background thread at
    startup, so the FIRST "navigate to..." answers immediately instead of
    downloading/initializing for up to a minute. Off by default - development
    restarts (--reload) shouldn't pay the load on every code change."""
    if os.getenv("LUMEN_NAV_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
        from services import navigation
        log.info("LUMEN_NAV_PRELOAD set - warming navigation models + TTS cache...")
        navigation.preload_in_background()

# CORS: the frontend is served from a different port (typically 8080) during
# local dev, so we allow cross-origin requests. WebSocket connections aren't
# subject to CORS, but the /health probe is.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe. Returns immediately - does not touch ML models."""
    return {"status": "ok", "service": "lumen-backend", "version": "0.1.0"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """One WebSocket per user session.

    The Session object owns the FSM state, latest frame, audio buffer, and
    task context. When the connection closes (either gracefully or via error),
    the session is dropped.
    """
    await ws.accept()
    session = Session(ws)
    log.info("Session %s connected from %s", session.id, ws.client)

    try:
        await session.send_session_hello()
        await session.send_initial_state()
        await session.run()
    except WebSocketDisconnect:
        log.info("Session %s disconnected cleanly", session.id)
    except Exception:
        log.exception("Session %s crashed; closing connection", session.id)
        try:
            await ws.close(code=1011)  # internal error
        except Exception:
            pass
    finally:
        await session.cleanup()
        log.info("Session %s ended", session.id)


# ---------- frontend (mounted last so /health and /ws win) ----------

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.is_dir():
    # html=True makes GET / serve index.html, and unknown paths under /
    # fall back to index.html only if they don't exist (handy for SPAs).
    app.mount("/", _NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    log.info("Serving frontend from %s (no-cache headers)", FRONTEND_DIR)
else:
    log.warning("Frontend directory not found at %s; serving API only", FRONTEND_DIR)
