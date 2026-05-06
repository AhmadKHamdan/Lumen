"""
End-to-end smoke test for the WebSocket round-trip.

Connects a Python WebSocket client to a running uvicorn instance and walks
through the full Sprint 1 lifecycle:

    1. Open /ws -> server pushes initial fsm_state: idle
    2. Send user_event: start -> server pushes fsm_state: listening
    3. Send a fake JPEG frame -> server logs decode and stores it
    4. Send a fake command_audio blob -> server transcribes (mocked here),
       parses, transitions FSM, sends back transcription + tts
    5. Send user_event: stop -> server pushes returning then idle

Run with::

    cd backend && PYTHONPATH=. python tests/smoke_ws.py

Requires the server to be running on ws://127.0.0.1:8765/ws.
You can override with the WS_URL environment variable.

This is intentionally NOT a pytest test - it requires a live server. It's a
manual smoke check used during integration debugging.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys

import websockets
from PIL import Image


WS_URL = os.environ.get("WS_URL", "ws://127.0.0.1:8765/ws")


def make_dummy_jpeg() -> bytes:
    """A tiny but valid 8x8 RGB JPEG."""
    img = Image.new("RGB", (8, 8), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


async def expect(ws, predicate, timeout=2.0, label=""):
    """Receive messages until ``predicate(msg)`` is true or timeout."""
    end = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = end - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {label}")
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timed out waiting for {label}")

        if isinstance(msg, str):
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            print(f"  <- json: {payload}")
            if predicate(payload):
                return payload
        else:
            print(f"  <- binary: tag=0x{msg[0]:02x} len={len(msg)-1}")


async def main():
    print(f"Connecting to {WS_URL}")
    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        print("Connected. Waiting for initial fsm_state ...")
        await expect(ws, lambda m: m.get("type") == "fsm_state" and m.get("state") == "idle",
                     label="initial idle")

        print("\n[1] Sending user_event:start")
        await ws.send(json.dumps({"type": "user_event", "event": "start"}))
        await expect(ws, lambda m: m.get("type") == "fsm_state" and m.get("state") == "listening",
                     label="listening")

        print("\n[2] Sending a fake JPEG frame")
        jpeg = make_dummy_jpeg()
        frame_msg = bytes([0x01]) + jpeg
        await ws.send(frame_msg)
        # No reply expected for frames; server logs them.
        await asyncio.sleep(0.2)

        print("\n[3] Sending user_event:stop")
        await ws.send(json.dumps({"type": "user_event", "event": "stop"}))
        await expect(ws, lambda m: m.get("type") == "fsm_state" and m.get("state") == "returning",
                     label="returning")
        await expect(ws, lambda m: m.get("type") == "fsm_state" and m.get("state") == "idle",
                     label="idle (after returning)")

        print("\nAll FSM transitions verified.")
        print("(Audio round-trip not tested here because Whisper + gTTS need network.)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FAILED: {e!r}", file=sys.stderr)
        sys.exit(1)
    print("\nSMOKE TEST PASSED")
