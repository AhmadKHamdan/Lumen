// Phone compass heading monitor.
//
// Listens to DeviceOrientationEvent and reports the compass heading in
// degrees (0-360, clockwise) at a bounded rate. The navigation task uses it
// to track the guided 360 room scan and to anchor door/indicator bearings;
// without a compass (laptop, permission denied) the server falls back to a
// single-pass scan, so this failing is never fatal.
//
// iOS 13+ Safari requires an explicit permission call from inside a user
// gesture - same dance as MotionMonitor. On iOS we prefer
// webkitCompassHeading (true heading, already clockwise); elsewhere alpha is
// counter-clockwise from north, so we flip it.

export class HeadingMonitor {
  /**
   * @param {(degrees: number) => void} onHeading - throttled heading callback
   */
  constructor(onHeading) {
    this.onHeading = onHeading;
    this.current = null;      // latest raw reading (degrees CW, or null)
    this._lastSent = null;    // last value delivered to the callback
    this._lastSentAt = 0;
    this._enabled = false;
    this._boundHandler = (ev) => this._onReading(ev);
  }

  /**
   * Attach the orientation listener. Handles the iOS 13+ permission prompt;
   * call from inside a user gesture. Returns true if we're now sampling.
   */
  async start() {
    if (typeof DeviceOrientationEvent === "undefined") {
      console.log("HeadingMonitor: DeviceOrientationEvent not supported");
      return false;
    }
    if (typeof DeviceOrientationEvent.requestPermission === "function") {
      try {
        const p = await DeviceOrientationEvent.requestPermission();
        if (p !== "granted") {
          console.log("HeadingMonitor: permission denied");
          return false;
        }
      } catch (e) {
        console.warn("HeadingMonitor: permission request failed", e);
        return false;
      }
    }
    window.addEventListener("deviceorientation", this._boundHandler, true);
    this._enabled = true;
    return true;
  }

  stop() {
    if (!this._enabled) return;
    window.removeEventListener("deviceorientation", this._boundHandler, true);
    this._enabled = false;
    this.current = null;
    this._lastSent = null;
  }

  _onReading(ev) {
    let deg = null;
    if (typeof ev.webkitCompassHeading === "number") {
      deg = ev.webkitCompassHeading;          // iOS: true heading, clockwise
    } else if (typeof ev.alpha === "number") {
      deg = (360 - ev.alpha) % 360;           // others: alpha is CCW -> make CW
    }
    if (deg === null || Number.isNaN(deg)) return;
    this.current = deg;

    // Throttle: only bother the server when the phone actually turned
    // (>= 2 degrees) and at most ~5 times a second. The nav loop reads the
    // *latest* value each tick, so dropped intermediates are harmless.
    const now = performance.now();
    const moved = this._lastSent === null
      || Math.abs(((deg - this._lastSent + 540) % 360) - 180) >= 2;
    if (moved && now - this._lastSentAt >= 200) {
      this._lastSent = deg;
      this._lastSentAt = now;
      this.onHeading(deg);
    }
  }
}
