// Top-level coordinator. Wires the buttons in index.html to the WS client,
// the camera/mic, the audio playback queue, and the wake lock.

import { WSClient, TAG_FRAME, TAG_COMMAND_AUDIO, TAG_TTS } from "./ws_client.js";
import { MediaController } from "./media.js";
import { AudioQueue } from "./audio_queue.js";
import { WakeLockManager } from "./wakelock.js";

// Backend URL. Default to localhost:8000 - override via ?backend= in the URL.
function backendURL() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("backend");
  if (override) return override;
  // Build ws:// or wss:// based on the page protocol if we're on https,
  // otherwise default to ws:// to localhost.
  if (window.location.protocol === "https:") {
    return `wss://${window.location.host.replace(/:\d+$/, "")}:8000/ws`;
  }
  return "ws://localhost:8000/ws";
}

// ---------- DOM refs ----------

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnPTT = document.getElementById("btn-ptt");
const videoEl = document.getElementById("video-preview");
const connIndicator = document.getElementById("conn-indicator");
const stateIndicator = document.getElementById("state-indicator");
const lastTranscription = document.getElementById("last-transcription");
const lastError = document.getElementById("last-error");

// ---------- module instances ----------

const ws = new WSClient(backendURL());
const media = new MediaController(videoEl);
const audioQueue = new AudioQueue();
const wakeLock = new WakeLockManager();

let active = false;  // true between Start and Stop

// ---------- WS subscriptions ----------

ws.onConnectionChange((state) => {
  console.log("conn:", state);
  connIndicator.textContent = state;
  connIndicator.className = `status-pill conn-${state}`;
});

ws.on("fsm_state", (msg) => {
  const state = msg.state || "idle";
  console.log("fsm:", state);
  stateIndicator.textContent = state.replace(/_/g, " ");
  stateIndicator.className = `status-pill state-${state}`;
});

ws.on("transcription", (msg) => {
  const text = msg.text || "";
  const conf = msg.confidence != null ? Math.round(msg.confidence * 100) : null;
  console.log("transcript:", text, "conf:", conf);
  lastTranscription.textContent = text
    ? (conf != null ? `"${text}" (${conf}% confidence)` : `"${text}"`)
    : "";
  lastError.textContent = "";  // clear any prior error
});

ws.on("error", (msg) => {
  console.warn("server error:", msg);
  lastError.textContent = msg.message || "Server error.";
});

ws.onBinary(TAG_TTS, (bytes) => {
  console.log(`TTS clip received: ${bytes.byteLength} bytes`);
  audioQueue.enqueue(bytes);
});

// ---------- media wiring ----------

media.onFrame(async (jpegBlob) => {
  if (!active) return;
  await ws.sendBinary(TAG_FRAME, jpegBlob);
});

media.onAudioBlob(async (audioBlob) => {
  if (!active) return;
  console.log(`Sending PTT audio: ${audioBlob.size} bytes`);
  await ws.sendBinary(TAG_COMMAND_AUDIO, audioBlob);
});

// ---------- buttons ----------

btnStart.addEventListener("click", async () => {
  if (active) return;
  lastError.textContent = "";
  lastTranscription.textContent = "";

  try {
    await media.start();
  } catch (e) {
    console.error("media.start failed", e);
    lastError.textContent = e.message === "camera-permission-denied"
      ? "Camera/microphone permission denied. Reload and try again."
      : "Could not start camera or microphone.";
    return;
  }

  await wakeLock.acquire();
  ws.connect();

  // Wait briefly for the WebSocket to open before sending start.
  // (sendJson silently no-ops if not yet open, but we want the start event
  // to actually reach the server.)
  await waitForOpen();
  ws.sendJson({ type: "user_event", event: "start" });

  media.startFrameLoop();
  active = true;

  btnStart.disabled = true;
  btnStop.disabled = false;
  btnPTT.disabled = false;
});

btnStop.addEventListener("click", async () => {
  if (!active) return;
  active = false;

  ws.sendJson({ type: "user_event", event: "stop" });
  media.stopFrameLoop();
  await media.stop();
  await wakeLock.release();
  ws.close();

  btnStart.disabled = false;
  btnStop.disabled = true;
  btnPTT.disabled = true;
  btnPTT.classList.remove("recording");
});

// PTT: pointerdown to start recording, pointerup/pointercancel to stop.
function attachPTT(btn) {
  const onDown = (ev) => {
    if (!active) return;
    ev.preventDefault();
    media.startRecording();
    btn.classList.add("recording");
    try { btn.setPointerCapture(ev.pointerId); } catch (_) {}
  };
  const onUp = (ev) => {
    if (!active) return;
    ev.preventDefault();
    media.stopRecording();
    btn.classList.remove("recording");
    try { btn.releasePointerCapture(ev.pointerId); } catch (_) {}
  };
  btn.addEventListener("pointerdown", onDown);
  btn.addEventListener("pointerup", onUp);
  btn.addEventListener("pointercancel", onUp);
  btn.addEventListener("pointerleave", (ev) => {
    if (btn.classList.contains("recording")) onUp(ev);
  });
}

attachPTT(btnPTT);

// ---------- helpers ----------

function waitForOpen(timeoutMs = 3000) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const check = () => {
      if (ws.ws && ws.ws.readyState === WebSocket.OPEN) return resolve(true);
      if (Date.now() - t0 > timeoutMs) {
        console.warn("waitForOpen: timed out");
        return resolve(false);
      }
      setTimeout(check, 50);
    };
    check();
  });
}

// Surface unhandled rejections in the DOM for visibility during dev.
window.addEventListener("unhandledrejection", (ev) => {
  console.error("unhandled rejection", ev.reason);
});
