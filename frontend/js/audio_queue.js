// Audio playback queue.
//
// Receives MP3 blobs from the server and plays them sequentially through a
// single <audio> element. Never overlapping - if a clip is still playing
// when a new one arrives, the new one waits in the queue.
//
// We create one Audio element per clip (rather than reusing one) because
// browsers are touchier about reusing elements after a Blob URL is revoked,
// and the GC overhead is negligible for short MP3s.

export class AudioQueue {
  constructor() {
    this._queue = [];          // Array<Uint8Array>
    this._playing = false;
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
        // Continue with next clip rather than getting stuck
      }
    }

    this._playing = false;
  }

  _playOne(bytes) {
    return new Promise((resolve) => {
      const blob = new Blob([bytes], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

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

      audio.play().catch((e) => {
        // Autoplay policies on iOS Safari and some Android Chrome configs
        // require a user gesture before .play() works. Once the user has
        // tapped Start the page is "active" and play should succeed.
        console.warn("audio.play() rejected", e);
        cleanup();
      });
    });
  }
}
