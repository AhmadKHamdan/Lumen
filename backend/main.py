"""
Lumen backend - FastAPI entrypoint.

Hosts:
  - GET  /health  : lightweight liveness probe
  - WS   /ws      : single WebSocket endpoint, one connection per session

The Session class (api/session.py) holds per-connection state. The Router
(api/router.py) parses incoming messages and dispatches to handlers. The FSM
(fsm/task_fsm.py) is authoritative for task state - the client maintains no
state of its own beyond a "connected" indicator.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.session import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lumen.main")

app = FastAPI(title="Lumen Backend", version="0.1.0")

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
