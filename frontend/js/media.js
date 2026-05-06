// Camera + microphone capture, push-to-talk recording, and 5-FPS frame loop.
//
// Owns the MediaStream from getUserMedia. Three things hang off the stream:
//   1. The hidden <video> element (so the browser plays the camera through
//      the canvas downstream).
//   2. A MediaRecorder for the PTT audio button (Opus in WebM container).
//   3. A frame loop that draws the video to an offscreen canvas at 5 FPS,
//      JPEG-encodes each frame, and hands the blob to a callback.
//
// Caller wires the MediaRecorder result and frame blobs into the WSClient.

const FRAME_INTERVAL_MS = 200;       // 5 FPS
const FRAME_QUALITY = 0.7;
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const PREFERRED_AUDIO_MIME = "audio/webm;codecs=opus";

export class MediaController {
  /**
   * @param {HTMLVideoElement} videoEl
   */
  constructor(videoEl) {
    this.videoEl = videoEl;
    this.stream = null;
    this.recorder = null;
    this.recordedChunks = [];
    this._frameTimer = null;
    this._canvas = null;
    this._ctx = null;
    this._onFrame = null;        // (Blob) => Promise<void>
    this._onAudioBlob = null;    // (Blob) => Promise<void>
  }

  // ---------- public API ----------

  /** Acquire camera + mic. Returns true on success. */
  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: CAPTURE_WIDTH }, height: { ideal: CAPTURE_HEIGHT } },
        audio: true,
      });
    } catch (e) {
      console.error("getUserMedia failed", e);
      throw new Error("camera-permission-denied");
    }

    this.videoEl.srcObject = this.stream;
    this.videoEl.classList.add("visible");
    try {
      await this.videoEl.play();
    } catch (e) {
      console.warn("video.play() failed (may autoplay anyway)", e);
    }

    // Set up offscreen canvas for frame capture
    this._canvas = document.createElement("canvas");
    this._canvas.width = CAPTURE_WIDTH;
    this._canvas.height = CAPTURE_HEIGHT;
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: false });

    // Pick a MediaRecorder mime type the browser supports
    const mime = this._pickAudioMime();
    if (!mime) {
      throw new Error("audio-recorder-unsupported");
    }
    this._audioMime = mime;

    // Set up the audio MediaRecorder. We keep the audio-only stream by
    // pulling tracks from the same getUserMedia stream.
    const audioStream = new MediaStream(this.stream.getAudioTracks());
    this.recorder = new MediaRecorder(audioStream, { mimeType: mime });
    this.recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) this.recordedChunks.push(ev.data);
    };
    this.recorder.onstop = async () => {
      const blob = new Blob(this.recordedChunks, { type: mime });
      this.recordedChunks = [];
      console.log(`PTT recording stopped: ${blob.size} bytes (${mime})`);
      if (this._onAudioBlob) {
        try { await this._onAudioBlob(blob); }
        catch (e) { console.error("onAudioBlob handler threw", e); }
      }
    };
    return true;
  }

  /** Stop everything and release tracks. */
  async stop() {
    this.stopFrameLoop();
    if (this.recorder && this.recorder.state !== "inactive") {
      try { this.recorder.stop(); } catch (_) {}
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        try { track.stop(); } catch (_) {}
      }
    }
    this.stream = null;
    this.recorder = null;
    if (this.videoEl) {
      this.videoEl.srcObject = null;
      this.videoEl.classList.remove("visible");
    }
  }

  /** Register a callback for each captured JPEG frame. */
  onFrame(fn) { this._onFrame = fn; }

  /** Register a callback for each finalized PTT audio blob. */
  onAudioBlob(fn) { this._onAudioBlob = fn; }

  // ---------- frame loop ----------

  startFrameLoop() {
    if (this._frameTimer) return;
    if (!this.stream) {
      console.warn("startFrameLoop called without stream");
      return;
    }
    this._frameTimer = setInterval(() => this._captureFrame(), FRAME_INTERVAL_MS);
    console.log("Frame loop started @", 1000 / FRAME_INTERVAL_MS, "FPS");
  }

  stopFrameLoop() {
    if (this._frameTimer) {
      clearInterval(this._frameTimer);
      this._frameTimer = null;
      console.log("Frame loop stopped");
    }
  }

  async _captureFrame() {
    if (!this.videoEl || !this._ctx) return;
    if (this.videoEl.readyState < this.videoEl.HAVE_CURRENT_DATA) return;

    try {
      this._ctx.drawImage(this.videoEl, 0, 0, this._canvas.width, this._canvas.height);
    } catch (e) {
      console.warn("drawImage failed", e);
      return;
    }

    return new Promise((resolve) => {
      this._canvas.toBlob(
        (blob) => {
          if (blob && this._onFrame) {
            this._onFrame(blob).catch((e) => console.error("onFrame threw", e));
          }
          resolve();
        },
        "image/jpeg",
        FRAME_QUALITY,
      );
    });
  }

  // ---------- PTT ----------

  startRecording() {
    if (!this.recorder) {
      console.warn("startRecording: no recorder");
      return;
    }
    if (this.recorder.state === "recording") return;
    this.recordedChunks = [];
    try {
      this.recorder.start();
      console.log("PTT recording started");
    } catch (e) {
      console.error("recorder.start failed", e);
    }
  }

  stopRecording() {
    if (!this.recorder) return;
    if (this.recorder.state !== "recording") return;
    try {
      this.recorder.stop();
    } catch (e) {
      console.error("recorder.stop failed", e);
    }
  }

  get audioMime() { return this._audioMime; }

  // ---------- internals ----------

  _pickAudioMime() {
    const candidates = [
      PREFERRED_AUDIO_MIME,
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const m of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) {
        return m;
      }
    }
    return null;
  }
}
