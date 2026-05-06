// WebSocket client.
//
// Opens a single persistent connection to the backend, dispatches incoming
// messages by type, and exposes send helpers for JSON control frames and
// tagged binary media frames.
//
// Wire format (see docs/protocol.md):
//   - text frames: JSON with a `type` field
//   - binary frames: 1-byte tag prefix + payload bytes
//        0x01 = frame (client -> server, JPEG)
//        0x02 = command_audio (client -> server, WebM/Opus)
//        0x03 = tts (server -> client, MP3)

export const TAG_FRAME = 0x01;
export const TAG_COMMAND_AUDIO = 0x02;
export const TAG_TTS = 0x03;

export class WSClient {
  /**
   * @param {string} url e.g. "ws://localhost:8000/ws"
   */
  constructor(url) {
    this.url = url;
    this.ws = null;
    this._handlers = {};         // type -> listener(payload)
    this._binHandlers = {};      // tag  -> listener(bytes)
    this._connHandlers = [];     // listener(state) where state in {connecting, connected, disconnected}
    this._retried = false;       // one retry on error per Sprint 1 spec
  }

  // ---------- subscriptions ----------

  /** Listen to a JSON message type. e.g. on("fsm_state", payload => ...) */
  on(type, fn) {
    this._handlers[type] = fn;
  }

  /** Listen to a binary tag (e.g. TAG_TTS) with raw Uint8Array body. */
  onBinary(tag, fn) {
    this._binHandlers[tag] = fn;
  }

  /** Connection-state change listener. */
  onConnectionChange(fn) {
    this._connHandlers.push(fn);
  }

  // ---------- lifecycle ----------

  connect() {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      return;
    }
    this._setConn("connecting");

    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      console.error("WS construct failed", e);
      this._setConn("disconnected");
      return;
    }
    this.ws.binaryType = "arraybuffer";

    this.ws.addEventListener("open", () => {
      console.log("WS open ->", this.url);
      this._retried = false;
      this._setConn("connected");
    });

    this.ws.addEventListener("close", (ev) => {
      console.log("WS close", ev.code, ev.reason);
      this._setConn("disconnected");
      // One retry on unclean close (per v1 spec - no fancy state recovery).
      if (!this._retried && ev.code !== 1000) {
        this._retried = true;
        console.log("WS attempting one retry...");
        setTimeout(() => this.connect(), 1000);
      }
    });

    this.ws.addEventListener("error", (ev) => {
      console.warn("WS error", ev);
    });

    this.ws.addEventListener("message", (ev) => this._onMessage(ev));
  }

  close() {
    this._retried = true;  // suppress retry on manual close
    if (this.ws) {
      try { this.ws.close(1000, "client_close"); } catch (_) {}
      this.ws = null;
    }
    this._setConn("disconnected");
  }

  // ---------- send ----------

  /** Send a JSON control message. */
  sendJson(payload) {
    if (!this._readyToSend()) return false;
    try {
      this.ws.send(JSON.stringify(payload));
      return true;
    } catch (e) {
      console.error("WS sendJson failed", e);
      return false;
    }
  }

  /**
   * Send a tagged binary payload. `body` may be ArrayBuffer, Uint8Array,
   * or Blob. Blob is converted to ArrayBuffer first (async).
   */
  async sendBinary(tag, body) {
    if (!this._readyToSend()) return false;

    let bodyBuf;
    if (body instanceof Blob) {
      bodyBuf = await body.arrayBuffer();
    } else if (body instanceof ArrayBuffer) {
      bodyBuf = body;
    } else if (ArrayBuffer.isView(body)) {
      bodyBuf = body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
    } else {
      console.error("sendBinary: unsupported body type", body);
      return false;
    }

    const out = new Uint8Array(bodyBuf.byteLength + 1);
    out[0] = tag & 0xff;
    out.set(new Uint8Array(bodyBuf), 1);

    try {
      this.ws.send(out.buffer);
      return true;
    } catch (e) {
      console.error("WS sendBinary failed", e);
      return false;
    }
  }

  // ---------- internals ----------

  _readyToSend() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  _onMessage(ev) {
    if (typeof ev.data === "string") {
      let payload;
      try {
        payload = JSON.parse(ev.data);
      } catch (e) {
        console.warn("WS got non-JSON text:", ev.data);
        return;
      }
      const type = payload && payload.type;
      const handler = this._handlers[type];
      if (handler) {
        try { handler(payload); }
        catch (e) { console.error("handler for", type, "threw", e); }
      } else {
        console.log("WS unhandled JSON type:", type, payload);
      }
      return;
    }

    if (ev.data instanceof ArrayBuffer) {
      const view = new Uint8Array(ev.data);
      if (view.byteLength === 0) {
        console.warn("WS empty binary frame");
        return;
      }
      const tag = view[0];
      const body = view.slice(1);
      const handler = this._binHandlers[tag];
      if (handler) {
        try { handler(body); }
        catch (e) { console.error("binary handler for", tag, "threw", e); }
      } else {
        console.log("WS unhandled binary tag:", tag, "(", body.byteLength, "bytes)");
      }
      return;
    }

    console.warn("WS unknown message type:", ev.data);
  }

  _setConn(state) {
    for (const fn of this._connHandlers) {
      try { fn(state); }
      catch (e) { console.error("conn handler threw", e); }
    }
  }
}
