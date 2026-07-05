// Coarse device-motion classifier.
//
// Listens to DeviceMotionEvent (accelerometer) and every ~200 ms decides
// whether the phone is "still", "moving", or "walking". Emits a callback
// only on state CHANGE (not every sample) so the WS traffic stays bounded.
//
// Honest scope note: we cannot recover true depth from pure IMU on a
// handheld phone - parallax needs translation and rotation gives no depth.
// What we CAN detect is whether the user is holding the phone still vs.
// moving/walking, and the server uses that as a signal to bypass the
// spatial debounce so guidance keeps up when the user has actually moved.
//
// iOS 13+ Safari (and iOS Chrome, which uses WebKit) requires an explicit
// permission call from inside a user gesture. We call requestPermission
// from the Start button handler.

export class MotionMonitor {
  /**
   * @param {(state: "still"|"moving"|"walking") => void} onStateChange
   */
  constructor(onStateChange) {
    this.onStateChange = onStateChange;
    this.currentState = "still";
    this._samples = [];     // {t, dev}
    this._boundHandler = (ev) => this._onSample(ev);
    this._enabled = false;
  }

  /**
   * Attach the accelerometer listener. Handles iOS 13+ permission prompt.
   * Returns true if we're now sampling, false otherwise (permission denied,
   * browser lacks the API, etc.).
   */
  async start() {
    if (typeof DeviceMotionEvent === "undefined") {
      console.log("MotionMonitor: DeviceMotionEvent not supported");
      return false;
    }
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      try {
        const result = await DeviceMotionEvent.requestPermission();
        if (result !== "granted") {
          console.log("MotionMonitor: permission not granted");
          return false;
        }
      } catch (e) {
        // Not called from a gesture, or already denied - just skip.
        console.warn("MotionMonitor.requestPermission failed", e);
        return false;
      }
    }
    window.addEventListener("devicemotion", this._boundHandler);
    this._enabled = true;
    return true;
  }

  stop() {
    if (this._enabled) {
      window.removeEventListener("devicemotion", this._boundHandler);
      this._enabled = false;
    }
    this._samples.length = 0;
    if (this.currentState !== "still") {
      this.currentState = "still";
      this.onStateChange("still");
    }
  }

  // ---------- internals ----------

  _onSample(ev) {
    const a = ev.accelerationIncludingGravity;
    if (!a || a.x == null) return;

    // Vector magnitude minus gravity ~ raw motion. Not physically clean
    // (we don't isolate gravity direction), but robust enough for a
    // three-way still / moving / walking bucket.
    const mag = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    const dev = Math.abs(mag - 9.81);

    const now = Date.now();
    this._samples.push({ t: now, dev });
    // Keep only the last ~1 second of samples.
    const cutoff = now - 1000;
    while (this._samples.length && this._samples[0].t < cutoff) {
      this._samples.shift();
    }

    // Mean deviation over the window.
    if (this._samples.length < 5) return;   // wait for a bit of history
    const mean = this._samples.reduce((s, x) => s + x.dev, 0) / this._samples.length;

    // Buckets tuned empirically:
    //   still   : holding phone as steady as a normal human can
    //   moving  : reaching, panning the camera, small position changes
    //   walking : rhythmic accelerations from stepping
    let next;
    if (mean < 0.35) next = "still";
    else if (mean < 1.5) next = "moving";
    else next = "walking";

    if (next !== this.currentState) {
      this.currentState = next;
      try { this.onStateChange(next); }
      catch (e) { console.warn("MotionMonitor callback threw", e); }
    }
  }
}
