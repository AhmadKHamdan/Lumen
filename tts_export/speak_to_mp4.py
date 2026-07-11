"""
Standalone TTS export — same voice as Lumen.

Synthesizes speech with the EXACT same engine/voice the app uses
(gTTS, lang="en", slow=False — see backend/services/tts_service.py)
and saves it as an .mp4 audio file.

Usage:
    python speak_to_mp4.py                       -> speaks TEXT below, saves OUT
    python speak_to_mp4.py "Any sentence here"   -> speaks the argument
    python speak_to_mp4.py "Hello" hello.mp4     -> custom output name

Requires:  pip install gtts     (already installed for the backend)
MP4 conversion uses ffmpeg if available; otherwise falls back to the
bundled imageio-ffmpeg (pip install imageio-ffmpeg). If neither exists,
the raw MP3 is saved instead and you're told about it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gtts import gTTS

# ----------------------------------------------------------------------
TEXT = "Good, keep scanning."   # <-- change this text
OUT = "output.mp4"                        # <-- change the output filename
# ----------------------------------------------------------------------

LANG = "en"   # same voice parameters as backend/services/tts_service.py
SLOW = False


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def synthesize_mp3(text: str) -> bytes:
    """Identical gTTS call to the app's tts_service.synthesize()."""
    if not text or not text.strip():
        raise ValueError("empty text")
    import io
    buf = io.BytesIO()
    gTTS(text=text, lang=LANG, slow=SLOW).write_to_fp(buf)
    mp3 = buf.getvalue()
    if not mp3:
        raise RuntimeError("gTTS returned empty bytes")
    return mp3


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    out = Path(sys.argv[2] if len(sys.argv) > 2 else OUT)

    print(f"Synthesizing ({LANG}, slow={SLOW}): {text!r}")
    mp3 = synthesize_mp3(text)

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        fallback = out.with_suffix(".mp3")
        fallback.write_bytes(mp3)
        print(f"ffmpeg not found -> saved raw MP3 instead: {fallback}")
        print("Install ffmpeg (or `pip install imageio-ffmpeg`) to get .mp4 output.")
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3)
        tmp_path = tmp.name
    try:
        # MP3 -> AAC in an MP4 container.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_path,
             "-c:a", "aac", "-b:a", "192k", str(out)],
            check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
