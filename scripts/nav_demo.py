"""Interactive demo for the Sprint 4 Navigation task (Parts 1 + 2).

Run from the repo root:

    python scripts/nav_demo.py

Simulates the full dialog flow without real Whisper/gTTS/WebSocket/YOLO.
Every TTS line is printed; every FSM event is printed; detections are
hand-crafted via the /detect command.

Commands at the user> prompt:
    /state          Print the current NavigationContext.
    /detect <cls> [region] [distance] [conf]
                    Inject ONE synthetic detection (default: center, near, 0.85).
                    Region: left | center | right (default center).
                    Distance: near | medium | far (default near).
    /detect_two <a> <b>
                    Inject a frame with two classes at once (for C7 testing).
                    Both default to center+near.
    /detect_off     Inject one frame with NO detections (a "miss").
    /timeout        Fast-forward to fire the 30s waypoint-prompt timeout
                    OR the 60s detection-timeout, whichever is active.
    /skip           Send "skip" to the manager (only meaningful in recovery).
    /reset          Cancel current task and start a fresh one.
    /quit           Exit the demo entirely.

Anything else typed is treated as the user's voice transcription.

Try these scenarios (use /reset between them):

  A. Happy reached-flow:
     >  kitchen          (destination)
     >  the chair
     >  /detect chair
     >  /detect chair
     -> "You've reached the chair."

  B. Temporal smoothing — single hit doesn't trigger:
     >  kitchen
     >  the chair
     >  /detect chair    (only one frame; not enough)
     >  /state           (notice: still active, not reached)

  C. Partial recognition:
     >  kitchen
     >  doorway then turn left
     -> "I didn't recognize: turn left."  + "Looking for doorway."

  D. Unknown landmark:
     >  kitchen
     >  the unicorn
     -> "I don't know that landmark. What's the landmark?"

  E. Disambiguation:
     >  living room
     >  the seat
     >  /detect_two chair couch
     >  /detect_two chair couch
     -> "I see a chair and a couch. Which would you like me to find?"
     >  chair
     -> "Looking for the chair."

  F. Detection timeout + skip:
     >  kitchen
     >  the chair
     >  /timeout
     -> "I haven't spotted the chair yet. Say 'skip' to move on..."
     >  /skip
     -> "What's the next landmark?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from navigation import (  # noqa: E402
    NavigationTaskManager,
    WAYPOINT_PROMPT_TIMEOUT_SECONDS,
    LANDMARK_DETECTION_TIMEOUT_SECONDS,
)


ACTION_CONTINUE = "continue"
ACTION_NEW_TASK = "new_task"
ACTION_QUIT = "quit"


class PrintTTS:
    def synthesize(self, text: str) -> bytes:
        print(f"  [TTS]  {text!r}")
        return b""


def send_tts_noop(_audio: bytes) -> None:
    pass


class PrintFSM:
    def handle_event(self, event: str, payload=None) -> None:
        print(f"  [FSM]  event={event!r} payload={payload!r}")


def show_state(mgr: NavigationTaskManager) -> None:
    if not mgr.is_active():
        print("  [STATE] no active task")
        return
    c = mgr.context
    print(f"  [STATE] destination={c.destination!r}")
    print(f"          awaiting_waypoint={c.awaiting_waypoint}")
    print(f"          awaiting_disambiguation={c.awaiting_disambiguation}")
    print(f"          awaiting_recovery={mgr._awaiting_recovery}")
    print(f"          retry_count={c.retry_count}")
    print(f"          frame_processing_enabled={c.frame_processing_enabled}")
    print(f"          detection_history={list(c.detection_history)}")
    print(f"          waypoints ({len(c.waypoints)}):")
    for i, wp in enumerate(c.waypoints):
        marker = " <-- current" if i == c.current_index else ""
        locked = f" locked={wp.locked_class}" if wp.locked_class else ""
        print(
            f"            [{i}] raw={wp.raw_text!r} norm={wp.normalized_text!r} "
            f"classes={wp.target_classes} status={wp.status.value}{locked}{marker}"
        )


def _make_detection(parts: list[str]) -> dict:
    """Build a detection dict from CLI-style args.

    Multi-word class names like 'dining table' can be passed by either
    using an underscore ('dining_table') or quoting separately is not
    supported in the simple split-on-space input parsing — use underscore.
    The underscore is converted back to a space.
    """
    cls = parts[0].replace("_", " ")
    region = parts[1] if len(parts) > 1 else "center"
    distance = parts[2] if len(parts) > 2 else "near"
    try:
        conf = float(parts[3]) if len(parts) > 3 else 0.85
    except ValueError:
        conf = 0.85
    return {
        "class_name": cls,
        "confidence": conf,
        "region": region,
        "distance_category": distance,
        "bbox": [100, 100, 400, 400],
    }


def start_task(mgr: NavigationTaskManager) -> bool:
    try:
        destination = input("User says 'navigate to ___': ").strip()
    except EOFError:
        print()
        return False
    if destination in ("/quit", ""):
        return False
    print()
    print(f"--> FSM transitions to NavigationActive (destination={destination!r})")
    mgr.on_navigation_command(destination)
    print()
    return True


def handle_command(mgr: NavigationTaskManager, line: str) -> str:
    if line == "/quit":
        return ACTION_QUIT

    if line == "/state":
        show_state(mgr)
        return ACTION_CONTINUE

    if line == "/reset":
        if mgr.is_active():
            print("  [DEMO] /reset — cancelling current task")
            mgr.cancel()
        return ACTION_NEW_TASK

    if line == "/skip":
        mgr.on_user_response("skip")
        if mgr.is_active():
            show_state(mgr)
        return ACTION_CONTINUE if mgr.is_active() else ACTION_NEW_TASK

    if line == "/timeout":
        # Fire whichever timer is currently active.
        if not mgr.is_active():
            print("  [DEMO] no active task")
            return ACTION_CONTINUE
        ctx = mgr.context
        if ctx.awaiting_waypoint or mgr._awaiting_recovery:
            fake_now = ctx.last_prompt_at + WAYPOINT_PROMPT_TIMEOUT_SECONDS + 1
            print(f"  [DEMO] fast-forwarding waypoint-prompt timer by "
                  f"{WAYPOINT_PROMPT_TIMEOUT_SECONDS + 1}s")
        else:
            fake_now = (ctx.current_waypoint_started_at or 0) + \
                LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
            print(f"  [DEMO] fast-forwarding detection timer by "
                  f"{LANDMARK_DETECTION_TIMEOUT_SECONDS + 1}s")
        mgr.check_timeout(now=fake_now)
        return ACTION_NEW_TASK if not mgr.is_active() else ACTION_CONTINUE

    if line.startswith("/detect_off"):
        if not mgr.is_active():
            print("  [DEMO] no active task")
            return ACTION_CONTINUE
        print("  [DEMO] injecting empty frame (a miss)")
        mgr.on_detections([])
        show_state(mgr)
        return ACTION_CONTINUE

    if line.startswith("/detect_two"):
        parts = line.split()[1:]
        if len(parts) < 2:
            print("  [DEMO] usage: /detect_two <classA> <classB>")
            return ACTION_CONTINUE
        a, b = parts[0], parts[1]
        det_a = _make_detection([a, "center", "near"])
        det_b = _make_detection([b, "right", "medium"])  # offset so reached
                                                          # wouldn't fire on B
        print(f"  [DEMO] injecting frame with {a} (center,near) + {b} (right,med)")
        mgr.on_detections([det_a, det_b])
        show_state(mgr)
        return ACTION_CONTINUE

    if line == "/here":
        if mgr.is_active():
            print("  [DEMO] simulating user saying \"I'm here\"")
            mgr.on_user_response("I'm here")
            if mgr.is_active():
                show_state(mgr)
        else:
            print("  [DEMO] no active task")
        return ACTION_CONTINUE if mgr.is_active() else ACTION_NEW_TASK

    if line == "/done":
        if mgr.is_active():
            print("  [DEMO] simulating user saying \"done\"")
            mgr.on_user_response("done")
        else:
            print("  [DEMO] no active task")
        return ACTION_CONTINUE if mgr.is_active() else ACTION_NEW_TASK

    if line == "/status":
        if not mgr.is_active():
            print("  [STATUS] inactive")
        else:
            from pprint import pformat
            print("  [STATUS]", pformat(mgr.get_status(), width=80))
        return ACTION_CONTINUE

    if line.startswith("/obstacle"):
        parts = line.split()[1:]
        if not parts:
            print("  [DEMO] usage: /obstacle <class> [region] [distance] [conf]")
            print("         e.g. /obstacle person   (person, center, near, 0.85)")
            return ACTION_CONTINUE
        if not mgr.is_active():
            print("  [DEMO] no active task")
            return ACTION_CONTINUE
        det = _make_detection(parts)
        print(f"  [DEMO] injecting obstacle: {det['class_name']} "
              f"({det['region']}, {det['distance_category']}, "
              f"conf={det['confidence']})")
        mgr.on_detections([det])
        show_state(mgr)
        return ACTION_CONTINUE

    if line.startswith("/detect"):
        parts = line.split()[1:]
        if not parts:
            print("  [DEMO] usage: /detect <class> [region] [distance] [conf]")
            return ACTION_CONTINUE
        det = _make_detection(parts)
        if not mgr.is_active():
            print("  [DEMO] no active task")
            return ACTION_CONTINUE
        print(f"  [DEMO] injecting detection: {det['class_name']} "
              f"({det['region']}, {det['distance_category']}, "
              f"conf={det['confidence']})")
        mgr.on_detections([det])
        show_state(mgr)
        return ACTION_CONTINUE

    # Anything else: pass to manager as a transcription.
    mgr.on_user_response(line)
    if mgr.is_active():
        show_state(mgr)
    return ACTION_CONTINUE


def main() -> None:
    mgr = NavigationTaskManager(
        tts_service=PrintTTS(),
        send_tts=send_tts_noop,
        fsm=PrintFSM(),
    )

    print("Lumen Navigation — Parts 1+2+3+4+5 demo")
    print("Commands:")
    print("  /state /status                       inspect current state")
    print("  /detect <cls> [region] [dist] [conf] inject a detection")
    print("  /detect_two <a> <b>                  inject 2 classes at once")
    print("  /detect_off                          inject a 'no detection' frame")
    print("  /obstacle <cls> [region] [dist]      inject an obstacle")
    print("  /here                                simulate 'I'm here'")
    print("  /done                                simulate 'done'")
    print("  /skip                                simulate 'skip'")
    print("  /timeout                             fast-forward the active timer")
    print("  /reset                               cancel and restart")
    print("  /quit                                exit the demo")
    print("-" * 60)

    while True:
        if not start_task(mgr):
            break

        action = ACTION_CONTINUE
        while mgr.is_active() and action == ACTION_CONTINUE:
            try:
                line = input("user> ").strip()
            except EOFError:
                print()
                action = ACTION_QUIT
                break
            action = handle_command(mgr, line)

        print()
        print("Task ended. Final state:")
        show_state(mgr)
        print()

        if action == ACTION_QUIT:
            break

    print("Goodbye.")


if __name__ == "__main__":
    main()
