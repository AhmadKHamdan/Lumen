"""
Batch-generate the demo audio: one numbered MP3 per Lumen line, in the same
voice as the app (gTTS, lang="en", slow=False — see backend/services/tts_service.py).

Usage:
    python make_demo_audio.py            -> writes ./demo_audio/01_*.mp3 ...

Edit LINES below if the video needs different directions or step counts.
MP3s drop straight onto the video timeline; if you want .mp4 clips instead,
run each file through speak_to_mp4.py.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from gtts import gTTS

OUT_DIR = Path(__file__).parent / "demo_audio"

# (slug, exact spoken line) — order matches demo_script.md
LINES = [
    ("navigating",     "Navigating to the kitchen."),
    ("scan_room1",     "Looking for the kitchen. Let's scan the room — slowly turn "
                       "to your right, all the way around."),
    ("nudge_90",       "Good, keep scanning."),
    ("scan1_done",     "You're back where you started — scan complete. I found a "
                       "door on your left. Turn toward it and point your camera "
                       "at it, so I can guide you in precisely."),
    ("door_callout",   "The door is about 5 steps ahead. Let me check the path ahead."),
    ("chair_warning",  "There's a chair in your path. Step to your left, where "
                       "it's clear."),
    ("path_clear",     "Okay, the way ahead is clear. The door is about 4 steps "
                       "ahead. Walk forward slowly, with your hand out in front "
                       "of you until you feel the door."),
    ("at_the_door",    "You're right at the door. Open "
                       "it, walk through the doorway."),
    ("through",        "You're through. Now slowly turn to your right, all the "
                       "way around, so I can scan this room."),
    ("nudge_90_b",     "Good, keep scanning."),
    ("scan2_fridge",   "You're back where you started — scan complete. I noticed "
                       "a fridge on your right. Let's double-check — turn that "
                       "way and point the camera at it."),
    ("arrival",        "I can see a fridge — we've reached the kitchen."),
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for i, (slug, text) in enumerate(LINES, start=1):
        buf = io.BytesIO()
        gTTS(text=re.sub(r"\s+", " ", text), lang="en", slow=False).write_to_fp(buf)
        path = OUT_DIR / f"{i:02d}_{slug}.mp3"
        path.write_bytes(buf.getvalue())
        print(f"{path.name}  <- {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"\nDone: {len(LINES)} clips in {OUT_DIR}")


if __name__ == "__main__":
    main()
