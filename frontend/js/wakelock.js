// Screen Wake Lock wrapper.
//
// Acquired on Start to keep the phone screen from auto-locking during a
// task (which would freeze MediaRecorder and getUserMedia). Released on
// Stop. Browsers without Wake Lock support degrade gracefully.

export class WakeLockManager {
  constructor() {
    this._sentinel = null;
  }

  get supported() {
    return typeof navigator !== "undefined" && "wakeLock" in navigator;
  }

  async acquire() {
    if (!this.supported) {
      console.warn("Wake Lock API not supported on this browser");
      return false;
    }
    if (this._sentinel) {
      // Already held
      return true;
    }
    try {
      this._sentinel = await navigator.wakeLock.request("screen");
      this._sentinel.addEventListener("release", () => {
        console.log("Wake lock released (system event)");
        this._sentinel = null;
      });
      console.log("Wake lock acquired");
      return true;
    } catch (e) {
      console.warn("Wake lock request failed", e);
      this._sentinel = null;
      return false;
    }
  }

  async release() {
    if (this._sentinel) {
      try {
        await this._sentinel.release();
        console.log("Wake lock released");
      } catch (e) {
        console.warn("Wake lock release failed", e);
      }
      this._sentinel = null;
    }
  }
}
