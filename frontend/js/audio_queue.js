// Audio playback queue.
//
// Receives MP3 blobs from the server and plays them sequentially through a
// single, persistent <audio> element. Never overlapping - if a clip is still
// playing when a new one arrives, the new one waits in the queue.
//
// iOS gotcha (read this before "simplifying" prime()):
// ---------------------------------------------------
// iOS Safari (and iOS Chrome, which uses WebKit) enforces audio policy at
// TWO independent layers:
//   (1) Web Audio API - unlocked by the FIRST AudioContext call inside any
//       user gesture.
//   (2) HTMLAudioElement - unlocked PER ELEMENT by the first .play() call
//       on that element inside a user gesture. A FRESH `new Audio()` after
//       the gesture is locked again, even if other elements are unlocked.
//
// Our TTS clips arrive a second or more after the user tapped Start - well
// outside any gesture. The previous implementation created a `new Audio()`
// per clip, and every one of those was rejected silently on iPhone. The fix
// is to keep ONE persistent <audio> element, prime it with a silent WAV
// inside the Start gesture, then reuse the same element (by swapping its
// .src) for every subsequent TTS clip. iOS treats follow-up .play() calls
// on a primed element as continuations of the original gesture-authorised
// playback.

export class AudioQueue {
  constructor() {
    this._queue = [];          // Array<Uint8Array>
    this._playing = false;
    this._primed = false;
    this._audioCtx = null;
    this._element = null;      // single persistent HTMLAudioElement, set on prime()
  }

  /**
   * Unlock audio output for the page. MUST be called synchronously from a
   * user-gesture handler (the Start button tap). Idempotent.
   *
   * Two prongs:
   *   (a) Web Audio context resume + a 1-sample silent buffer.
   *   (b) A persistent <audio> element whose first playback is a silent WAV,
   *       kicked off here so the element is gesture-authorised. We reuse
   *       this element for every actual TTS clip (see _playOne).
   */
  prime() {
    if (this._primed) return;

    // (a) Web Audio unlock (covers desktop and modern Android Chrome).
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) {
        if (!this._audioCtx) this._audioCtx = new Ctx();
        if (this._audioCtx.state === "suspended" && this._audioCtx.resume) {
          this._audioCtx.resume().catch(() => {});
        }
        const buf = this._audioCtx.createBuffer(1, 1, 22050);
        const src = this._audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(this._audioCtx.destination);
        src.start(0);
      }
    } catch (e) {
      console.warn("Web Audio prime failed", e);
    }

    // (b) HTMLAudio unlock (the part iOS actually needs).
    try {
      if (!this._element) {
        this._element = new Audio();
        this._element.preload = "auto";
        // playsInline so iOS doesn't try to fullscreen anything (relevant for
        // <video>, harmless on <audio> but required by some old Safari builds).
        this._element.setAttribute("playsinline", "");
      }
      this._element.src = _silentWavDataUrl();
      const p = this._element.play();
      if (p && typeof p.then === "function") {
        p.then(() => { try { this._element.pause(); } catch (_) {} })
         .catch((e) => console.warn("HTMLAudio prime play() rejected", e));
      }
    } catch (e) {
      console.warn("HTMLAudio prime failed", e);
    }

    this._primed = true;
    console.log("AudioQueue primed (Web Audio + HTMLAudio)");
  }

  /**
   * Append an MP3 clip (Uint8Array). Starts playback if idle.
   * @param {Uint8Array} mp3Bytes
   */
  enqueue(mp3Bytes) {
    if (!mp3Bytes || mp3Bytes.byteLength === 0) {
      console.warn("AudioQueue.enqueue: empty bytes");
      return;
    }
    this._queue.push(mp3Bytes);
    if (!this._playing) {
      this._drain();
    }
  }

  /** Drop any pending clips. Currently-playing clip keeps playing. */
  clear() {
    this._queue.length = 0;
  }

  // ---------- internals ----------

  async _drain() {
    if (this._playing) return;
    this._playing = true;

    while (this._queue.length > 0) {
      const bytes = this._queue.shift();
      try {
        await this._playOne(bytes);
      } catch (e) {
        console.error("AudioQueue: failed to play clip", e);
        // Continue with next clip rather than getting stuck.
      }
    }

    this._playing = false;
  }

  _playOne(bytes) {
    return new Promise((resolve) => {
      const blob = new Blob([bytes], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);

      // Prefer the gesture-primed persistent element. On iOS, falling back
      // to a fresh `new Audio()` here is exactly what was silently failing.
      const audio = this._element || new Audio();

      const cleanup = () => {
        URL.revokeObjectURL(url);
        audio.onended = null;
        audio.onerror = null;
        resolve();
      };
      audio.onended = cleanup;
      audio.onerror = (e) => {
        console.warn("Audio playback error", e);
        cleanup();
      };

      audio.src = url;
      const p = audio.play();
      if (p && typeof p.catch === "function") {
        p.catch((e) => {
          console.warn("audio.play() rejected", e);
          cleanup();
        });
      }
    });
  }
}


// ---------- helpers ----------

/**
 * Build a data: URL for a minimal valid silent PCM WAV (46 bytes total: a
 * 44-byte header plus one zero-valued 16-bit sample). Constructed at runtime
 * so we don't have to inline a magic base64 string that could be subtly wrong.
 * iOS Safari plays WAV from data URLs.
 */
function _silentWavDataUrl() {
  const bytes = new Uint8Array(46);
  const dv = new DataView(bytes.buffer);
  // "RIFF" <size-8> "WAVE"
  bytes.set([0x52, 0x49, 0x46, 0x46], 0);
  dv.setUint32(4, 38, true);
  bytes.set([0x57, 0x41, 0x56, 0x45], 8);
  // "fmt " <16> PCM mono 44.1kHz 16-bit
  bytes.set([0x66, 0x6d, 0x74, 0x20], 12);
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);       // PCM
  dv.setUint16(22, 1, true);       // mono
  dv.setUint32(24, 44100, true);   // sample rate
  dv.setUint32(28, 88200, true);   // byte rate = sr * ch * bps/8
  dv.setUint16(32, 2, true);       // block align = ch * bps/8
  dv.setUint16(34, 16, true);      // bits/sample
  // "data" <2> + one zero sample (already zero-initialised)
  bytes.set([0x64, 0x61, 0x74, 0x61], 36);
  dv.setUint32(40, 2, true);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return "data:audio/wav;base64," + btoa(bin);
}
