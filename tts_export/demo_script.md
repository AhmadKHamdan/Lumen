# Lumen demo script — "Take me to the kitchen" (chair before the door)

Every line below is **exactly** what the current code speaks, in order.
Generate the audio with `python make_demo_audio.py` (one numbered MP3 per line),
then lay the clips onto the video.

Adjustable per the video: the door direction ("on your right"), the step counts
(5 / 4), and the fridge direction — swap the words, nothing else changes.

---

**Omar:** "Take me to the kitchen."

1. **Command confirmed** — immediately after the command:
   > Navigating to the kitchen.

2. **Scan instruction** — nav task starts (first frame with compass):
   > Looking for the kitchen. Let's scan the room — slowly turn to your right, all the way around.

3. **Nudge at ~90° of the turn:**
   > Good, keep scanning.

4. **Nudge at ~270° of the turn:**
   > Keep scanning.

5. **Scan complete** — back at the start direction; one door found, no kitchen signs (Omar faces the door after this line):
   > You're back where you started — scan complete. I found a door on your right. Turn toward it and point your camera at it, so I can guide you in precisely.

6. **Door call-out** — Omar is now facing the door (facing it is a silent handoff; this speaks right after):
   > The door is about 5 steps ahead. Let me check the path ahead.

7. **Obstacle** — the chair is confirmed in the walking lane (~1 s after the call-out):
   > There's a chair in your path. Step to your left, where it's clear.

8. **Path clears** — Omar side-stepped left; clear verdict + re-orientation in ONE utterance:
   > Okay, the way ahead is clear. The door is about 4 steps ahead. Walk forward slowly, with your hand out in front of you until you feel the door.

9. **At the door** — the door fills the camera view:
   > You're right at the door. Reach out with your hand, open it, walk through the doorway, and take two or three steps into the room.

10. **Through the doorway** — ~2 s after the door leaves the view while walking:
    > You're through. Now slowly turn to your right, all the way around, so I can scan this room.

11. **Nudge at ~90° of the second turn:**
    > Good, keep scanning.

12. **Nudge at ~270° of the second turn:**
    > Keep scanning.

13. **Second scan complete** — fridge sighted on his right, needs confirmation (Omar turns right after this line):
    > You're back where you started — scan complete. I noticed a fridge on your right. Let's double-check — turn that way and point the camera at it.

14. **Arrival** — Omar faces the fridge (silent handoff), Lumen re-confirms it (~2 s), then:
    > I can see a fridge — we've reached the kitchen.

---

### Optional filler lines (only if the video needs them)

- If the confirmation in step 14 drags on screen:
  > Keep the camera there, panning slowly, while I confirm the kitchen.
- If Omar hesitates before facing the door (repeats every ~6 s while not facing it):
  > Turn slowly to your right to face the door.
- Stronger variant of line 13 (used when the scan evidence is already strong):
  > You're back where you started — scan complete. I've seen signs of the kitchen on your right. Turn that way, and let's make sure we've reached it.
