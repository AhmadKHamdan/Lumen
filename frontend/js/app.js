// Top-level coordinator. Wires the buttons in index.html to the WS client,
// the camera/mic, the audio playback queue, and the wake lock.

import { WSClient, TAG_FRAME, TAG_COMMAND_AUDIO, TAG_TTS } from "./ws_client.js";
import { MediaController } from "./media.js";
import { AudioQueue } from "./audio_queue.js";
import { WakeLockManager } from "./wakelock.js";
import { MotionMonitor } from "./motion.js";
import { HeadingMonitor } from "./heading.js";

// ---------- browser TTS fallback ----------
//
// Used when the server can't reach us (WS closed, error before Start), or when
// a server-side error message is worth speaking aloud. The primary voice
// channel is still gTTS + AudioQueue; this is the safety net.
function speakLocally(text, {rate = 1.0, volume = 1.0} = {}) {
  if (!text) return;
  const synth = window.speechSynthesis;
  if (!synth) return;   // very old browser
  try {
    synth.cancel();     // stop any previous local speech
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = rate;
    utter.volume = volume;
    utter.lang = "en-US";
    synth.speak(utter);
  } catch (e) {
    console.warn("speechSynthesis failed", e);
  }
}


// Backend URL.
//
// Strategy:
//   1. Honour an explicit ?backend= URL override (handy for local dev with the
//      page opened as file:// pointed at a remote backend).
//   2. Otherwise, derive the WebSocket URL from the page itself: same host,
//      same port, scheme swapped to ws:// (http) or wss:// (https). This is
//      what makes the app work unchanged behind Cloudflare Tunnel, Azure,
//      ngrok, etc. - whatever serves the page also serves /ws.
//   3. Fallback (file:// or unknown protocol): the local backend.
function backendURL() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("backend");
  if (override) return override;

  const proto = window.location.protocol;
  if (proto === "https:" || proto === "http:") {
    const wsProto = proto === "https:" ? "wss" : "ws";
    return `${wsProto}://${window.location.host}/ws`;
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
const motion = new MotionMonitor((state) => {
  // Only meaningful while a session is active. Server still tolerates late
  // messages, but no point sending noise otherwise.
  if (!active) return;
  ws.sendJson({ type: "motion", state });
});
const heading = new HeadingMonitor((degrees) => {
  // Compass heading for the navigation task's 360 room scan. Same rule:
  // only send while a session is active.
  if (!active) return;
  ws.sendJson({ type: "heading", degrees });
});

let active = false;  // true between Start and Stop

// ---------- session persistence ----------
//
// Server assigns a session id on every WS accept and sends it via a
// `session_hello` message. We stash it in localStorage. On our NEXT connect
// (after a drop, reload, or reopen), we send `resume` with the previous id;
// if the server still has an active task saved for that id, it re-fires the
// task and announces "Resuming search for your cup."
const SESSION_ID_KEY = "lumen.session_id";
let sessionId = null;
try { sessionId = window.localStorage.getItem(SESSION_ID_KEY); }
catch (_) { /* private mode / storage blocked */ }

// ---------- WS subscriptions ----------

ws.onConnectionChange((state) => {
  console.log("conn:", state);
  connIndicator.textContent = state;
  connIndicator.className = `status-pill conn-${state}`;
  // Only announce unexpected disconnects, and only if the user is mid-session
  // (active). A clean stop shouldn't be announced.
  if (state === "disconnected" && active) {
    speakLocally("Connection lost. Trying to reconnect.");
  }
  if (state === "connected" && sessionId) {
    // Ask the server to restore whatever task we had before the drop. If
    // the server has no live state for this id (fresh install or TTL
    // expired), it silently no-ops - harmless.
    ws.sendJson({ type: "resume", id: sessionId });
  }
});

ws.on("session_hello", (msg) => {
  if (msg && msg.id) {
    sessionId = msg.id;
    try { window.localStorage.setItem(SESSION_ID_KEY, sessionId); }
    catch (_) {}
    console.log("session id:", sessionId);
  }
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
  // No local speech here. The server follows most user-facing errors with a
  // gTTS clip; layering browser TTS on top produced the "two voices"
  // double-up. Errors that don't get a gTTS clip are visual-only.
});

ws.onBinary(TAG_TTS, (bytes) => {
  console.log(`TTS clip received: ${bytes.byteLength} bytes`);
  audioQueue.enqueue(bytes);
});

// Haptic feedback for reach cues, completion, and cancel. Server sends a
// pattern name; we map to a navigator.vibrate() millisecond sequence.
// iOS Safari has no Vibration API, so vibrate() is undefined - we degrade
// silently and the voice cues remain the primary channel.
const HAPTIC_PATTERNS = {
  short:   [40],           // reach "almost" - one brief tap
  medium:  [80],           // generic ack
  long:    [200],          // fingertip touched the target
  success: [40, 60, 40],   // "got it"
  cancel:  [120],          // task_abort
};
ws.on("haptic", (msg) => {
  const pattern = HAPTIC_PATTERNS[msg.pattern];
  if (!pattern) return;
  if (typeof navigator.vibrate !== "function") return;  // iOS
  try { navigator.vibrate(pattern); }
  catch (e) { console.warn("navigator.vibrate failed", e); }
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

  // Unlock audio output for the page WHILE STILL INSIDE THE CLICK GESTURE.
  // Mobile browsers (iOS Safari, Android Chrome) silently reject .play()
  // calls on audio that arrives outside a user-gesture call stack - which
  // is exactly what TTS clips are once they come back from the server.
  // Priming here, before any await, registers our intent to play audio.
  audioQueue.prime();

  try {
    await media.start();
  } catch (e) {
    console.error("media.start failed", e);
    const errText = e.message === "camera-permission-denied"
      ? "Camera or microphone permission denied. Please reload and allow access."
      : "Could not start the camera or microphone.";
    lastError.textContent = errText;
    // Server can't TTS this - the WS isn't even open yet. Speak locally.
    speakLocally(errText);
    return;
  }

  await wakeLock.acquire();
  // Start the device-motion monitor from inside the gesture - iOS 13+
  // requires that for permission. Fire-and-forget; failure just means the
  // server won't see motion updates and falls back to plain debouncing.
  motion.start().catch((e) => console.warn("motion.start failed", e));
  // Compass heading, same gesture requirement on iOS. Failure is fine - the
  // navigation scan falls back to a compass-less single pass.
  heading.start().catch((e) => console.warn("heading.start failed", e));
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
  motion.stop();
  heading.stop();
  ws.close();

  btnStart.disabled = false;
  btnStop.disabled = true;
  btnPTT.disabled = true;
  btnPTT.classList.remove("recording");
});

// Tap-anywhere PTT. Any pointer-down anywhere on the page (except on
// interactive controls that would clash - Start, Stop, form fields, links)
// starts recording; the matching pointer-up stops it. The visible PTT button
// is kept as a large tactile target and visual "recording" indicator, but a
// blind user no longer has to hunt for it - any part of the screen works.
//
// We attach at document level so a drag that leaves the original tap target
// still fires pointerup and stops recording cleanly.
function attachTapAnywherePTT() {
  const EXCLUDED_SELECTOR =
    "#btn-start, #btn-stop, a, input, textarea, select, [data-no-ptt]";
  let recordingViaTap = false;

  const onDown = (ev) => {
    if (!active) return;
    // Don't hijack taps that would activate Start / Stop or other controls.
    if (ev.target.closest && ev.target.closest(EXCLUDED_SELECTOR)) return;
    // Only left-button / primary pointer.
    if (ev.pointerType === "mouse" && ev.button !== 0) return;
    ev.preventDefault();
    media.startRecording();
    btnPTT.classList.add("recording");
    recordingViaTap = true;
  };

  const onUp = (ev) => {
    if (!recordingViaTap) return;
    if (ev && ev.preventDefault) ev.preventDefault();
    media.stopRecording();
    btnPTT.classList.remove("recording");
    recordingViaTap = false;
  };

  document.addEventListener("pointerdown", onDown);
  document.addEventListener("pointerup", onUp);
  document.addEventListener("pointercancel", onUp);
  // Safety net: if the tab loses focus mid-recording (e.g. iOS home button),
  // stop cleanly instead of leaving the mic hot.
  window.addEventListener("blur", () => onUp(null));
}

attachTapAnywherePTT();

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
