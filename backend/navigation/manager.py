"""NavigationTaskManager — orchestrates a single Navigation task.

This is the only class the FSM / WebSocket layer needs to talk to for
Navigation. It owns the NavigationContext lifecycle: entry from FSM,
waypoint-collection dialog, timeout watching, voice cancellation, and
clean handoff back to the FSM when the task ends.

Dependencies are injected so the module can be unit-tested in isolation
before partners' Sprint 1–3 code is ready:
- tts_service:  object with .synthesize(text: str) -> bytes  (Omar's API).
- send_tts:     callable(bytes) -> None — pushes an MP3 down the WebSocket.
- fsm:          object with .handle_event(event, payload) (FSM contract).
- logger:       optional logging.Logger.

Decisions implemented here:
- 2  Single NavigationTaskManager class.
- 9  FSM is already in NavigationActive when on_navigation_command runs.
- 10 frame_processing_enabled stays False until first waypoint added.
- 11 Voice "stop"/"cancel" recognised in addition to user_event cancel.
- 12 Up to 2 re-prompts on unrecognised waypoint, then cancel.
- 13 30-second timeout for an unanswered waypoint prompt.
- 14 Audible confirmation after a waypoint is stored.
- 15 Confirmation phrasing: "Looking for [waypoint]".

User overrode the diagram-based phrasing in decision 7 — we use
"Where should I take you first?" instead.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol

from .context import NavigationContext, Waypoint, WaypointStatus
from .detector import LandmarkDetector
from .landmark_map import resolve as resolve_landmark
from .parser import (
    parse_waypoint,
    is_cancellation_phrase,
    is_arrival_phrase,
    is_yes,
    is_no,
    is_done_phrase,
)


# ---- Configuration constants (single source of truth) ----

WAYPOINT_PROMPT_TEXT = "Where should I take you first?"
WAYPOINT_REPROMPT_TEXT = "I didn't catch that. What's the landmark?"
WAYPOINT_CONFIRMATION_TEMPLATE = "Looking for {waypoint}."
WAYPOINT_CANCELLED_TEXT = "Navigation cancelled."
WAYPOINT_TIMEOUT_TEXT = "Cancelling navigation. No landmark received."
WAYPOINT_GIVEUP_TEXT = "I'm having trouble understanding. Cancelling navigation."

# Part 2 phrases.
WAYPOINT_UNKNOWN_TEXT = "I don't know that landmark. What's the landmark?"   # A4
WAYPOINT_PARTIAL_RECOGNITION_TEMPLATE = "I didn't recognize: {dropped}."     # B4
WAYPOINT_REACHED_TEMPLATE = "You've reached the {waypoint}."                 # E5 callback
NEXT_WAYPOINT_PROMPT_TEXT = "What's the next landmark?"                      # G3
DETECTION_TIMEOUT_TEXT = (                                                   # G2
    "I haven't spotted the {waypoint} yet. "
    "Say 'skip' to move on, 'stop' to cancel, or describe a different landmark."
)
DISAMBIGUATION_PROMPT_TEMPLATE = (                                           # C7 sub 8
    "I see a {a} and a {b}. Which would you like me to find?"
)
DISAMBIGUATION_REPROMPT_TEXT = (
    "Sorry, please name one. Which would you like me to find?"               # C7 sub 2
)
DISAMBIGUATION_CONFIRMATION_TEMPLATE = "Looking for the {waypoint}."         # C7 sub 9

# Part 3 — obstacle warning phrases. Escalation ladder per Issue 3 resolution.
# Level 0 = mild ("Chair ahead."), 1 = reminder ("Chair still ahead."),
# 2 = cautionary ("Chair still in path, please be careful.").
# E1 = A: class-specific phrasing; "obstacle" is the fallback noun if a class
# isn't worth naming.
OBSTACLE_WARNING_LEVEL0_TEMPLATE = "{noun} ahead."
OBSTACLE_WARNING_LEVEL1_TEMPLATE = "{noun} still ahead."
OBSTACLE_WARNING_LEVEL2_TEMPLATE = "{noun} still in path, please be careful."
# A3: person is its own phrasing.
OBSTACLE_PERSON_LEVEL0_TEMPLATE = "Person ahead."
OBSTACLE_PERSON_LEVEL1_TEMPLATE = "Person still ahead."
OBSTACLE_PERSON_LEVEL2_TEMPLATE = "Person still in path, please be careful."

# Part 4 — guidance phrasings.
# B1 = A: "on your left" / "ahead" / "on your right"
# B2 = A (option A revised): far -> "far ahead"; medium -> "ahead";
# near -> "right in front of you"
# B3+B4: combine class + region + distance into one cue.
# B5: full utterance on region-only change.
# B6 = C + Interpretation 1: "getting closer" template for distance-only
# change in same region (with class name preserved).

# Region phrasings ('center' is rendered as 'ahead' which conflicts with
# the distance "ahead" — we resolve this by treating center+distance as a
# single phrase: e.g. "right in front of you", "ahead", "far ahead" already
# include the directional sense for centered objects).
def _region_phrase(region: str) -> str:
    return {
        "left": "on your left",
        "right": "on your right",
        "center": "ahead",
    }.get(region, "ahead")

def _distance_phrase(distance: str) -> str:
    return {
        "far": "far away",
        "medium": "approaching",
        "near": "right in front of you",
    }.get(distance, "")

# Templates assembled at call time in _on_guidance — kept as functions
# rather than strings because the natural English combination depends on
# whether the region is center or not.
GUIDANCE_LOST_SIGHT_TEMPLATE = "Lost sight of the {waypoint}. Looking for it."

# Part 5 phrases.
NAVIGATION_COMPLETE_TEXT = "Navigation complete."
WHATS_NEXT_PROMPT_TEXT = (
    "What's next? Say a landmark, 'done' to finish, or 'stop' to cancel."
)
COMPLETION_TIMEOUT_TEXT = "Navigation finished."
# B1 (keep confirmation): when user says "I'm here" but detector disagrees,
# describe what we see and ask. The {observation} slot is filled at runtime
# based on current detection state.
ARRIVAL_CONFIRMATION_TEMPLATE = (
    "{observation} Are you at the {waypoint}? Say 'yes' to confirm or 'no' to keep going."
)
ARRIVAL_CONFIRMATION_NO_OBSERVATION_TEMPLATE = (
    "I don't see the {waypoint} yet. Are you sure you've reached it? "
    "Say 'yes' to confirm or 'no' to keep going."
)
ARRIVAL_CONFIRMED_DECLINED_TEXT = "OK, continuing to look."

WAYPOINT_PROMPT_TIMEOUT_SECONDS = 30.0   # Decision 13.
MAX_WAYPOINT_REPROMPTS = 2               # Decision 12.
MAX_DISAMBIGUATION_REPROMPTS = 1         # C7 sub 2.


# ---- Dependency Protocols (let tests pass simple stubs) ----

class TTSService(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class FSM(Protocol):
    def handle_event(self, event: str, payload: Optional[dict] = None) -> None: ...


# ---- Manager ----

class NavigationTaskManager:
    """Orchestrates one Navigation task. One instance per Session."""

    def __init__(
        self,
        tts_service: TTSService,
        send_tts: Callable[[bytes], None],
        fsm: FSM,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._tts = tts_service
        self._send_tts = send_tts
        self._fsm = fsm
        self._log = logger or logging.getLogger(__name__)
        self.context: Optional[NavigationContext] = None
        # Part 2: detector is constructed when a Navigation task starts and
        # discarded when it ends.
        self._detector: Optional[LandmarkDetector] = None
        # G3: when the detection-timeout fires we enter a sub-state where
        # "skip" / a new landmark / "stop" are the valid replies. Tracked
        # via the awaiting_recovery flag.
        self._awaiting_recovery: bool = False

    # ---------------------------------------------------------------
    # Entry from FSM
    # ---------------------------------------------------------------

    def on_navigation_command(self, destination: str) -> None:
        """Called by the FSM's NavigationActive entry hook.

        Per decision 9, FSM is already in NavigationActive when this
        runs. We initialise the context, leave frame processing OFF
        (decision 10), and prompt the user for the first landmark.

        Part 5 F2: if a task is already active, cancel it cleanly first
        before starting the new one. This handles the user re-issuing a
        "navigate to" command mid-task.
        """
        if self.context is not None:
            self._log.info(
                "nav_restart_existing_task old=%r new=%r",
                self.context.destination, destination,
            )
            # Cancel the existing task but don't re-enter the FSM event —
            # we're about to start a new task in the same FSM state.
            self._silent_teardown()
        self.context = NavigationContext(destination=(destination or "").strip())
        self.context.task_started_at = time.time()
        self._log.info(
            "nav_started destination=%r awaiting_waypoint=True",
            self.context.destination,
        )
        self._prompt_for_waypoint(WAYPOINT_PROMPT_TEXT)

    def _silent_teardown(self) -> None:
        """Tear down current context/detector without speaking or emitting
        an FSM event. Used when re-issuing a navigate command (F2)."""
        if self.context is not None:
            self.context.cancel_remaining()
            self.context.frame_processing_enabled = False
        self.context = None
        self._detector = None
        self._awaiting_recovery = False

    # ---------------------------------------------------------------
    # User input handling
    # ---------------------------------------------------------------

    def on_user_response(self, transcription: str) -> None:
        """Called for every transcription that arrives while in NavigationActive.

        Routing priority (top to bottom):
          1. is_cancellation_phrase            -> cancel.
          2. awaiting_arrival_confirmation     -> yes/no (Part 5).
          3. awaiting_next_or_done             -> done / new landmark / stop (Part 5).
          4. awaiting_disambiguation           -> resolve the C7 prompt.
          5. _awaiting_recovery (G3 detection-timeout dialog) -> skip/cancel/new.
          6. is_arrival_phrase (mid-task)      -> "I'm here" override (Part 5).
          7. awaiting_waypoint                 -> waypoint candidate(s).
          8. otherwise                         -> log and ignore.
        """
        if self.context is None:
            self._log.warning("on_user_response called with no active context; ignoring")
            return

        text = (transcription or "").strip()
        self._log.debug(
            "nav_user_response text=%r awaiting_wp=%s awaiting_disamb=%s "
            "awaiting_recovery=%s awaiting_arrival_conf=%s awaiting_next=%s",
            text,
            self.context.awaiting_waypoint,
            self.context.awaiting_disambiguation,
            self._awaiting_recovery,
            self.context.awaiting_arrival_confirmation,
            self.context.awaiting_next_or_done,
        )

        # 1) Voice cancel ALWAYS wins.
        if is_cancellation_phrase(text):
            self._log.info("nav_voice_cancel text=%r", text)
            self.cancel()
            return

        # 2) Arrival-confirmation yes/no (Part 5 B1 keep-confirmation).
        if self.context.awaiting_arrival_confirmation:
            self._handle_arrival_confirmation(text)
            return

        # 3) "What's next?" dialog after a reached with empty queue (Part 5).
        if self.context.awaiting_next_or_done:
            self._handle_next_or_done(text)
            return

        # 4) Disambiguation pending? Try to interpret the answer.
        if self.context.awaiting_disambiguation:
            self._handle_disambiguation_response(text)
            return

        # 5) Recovery sub-state (after detection timeout, G2/G3).
        if self._awaiting_recovery:
            self._handle_recovery_response(text)
            return

        # 6) "I'm here" mid-task override (Part 5 B1).
        # Only meaningful if we have an active waypoint to apply it to.
        if (
            is_arrival_phrase(text)
            and not self.context.awaiting_waypoint
            and self.context.current_waypoint() is not None
        ):
            self._handle_arrival_override()
            return

        # 7) Waypoint collection.
        if self.context.awaiting_waypoint:
            # If user said "I'm here" while we're awaiting their first
            # landmark, B3 says: re-prompt — there's nothing to arrive at.
            if is_arrival_phrase(text):
                self._log.info("nav_arrival_during_collection — re-prompting")
                self._speak(
                    "There's no destination yet. " + WAYPOINT_PROMPT_TEXT
                )
                self.context.last_prompt_at = time.time()
                return
            self._handle_waypoint_response(text)
            return

        # 8) Fall-through.
        self._log.debug("nav_user_response ignored (no matching sub-state)")

    def _handle_waypoint_response(self, text: str) -> None:
        candidates = parse_waypoint(text)

        if not candidates:
            # Empty or all-filler — re-prompt or cancel per decision 12.
            self.context.retry_count += 1
            if self.context.retry_count > MAX_WAYPOINT_REPROMPTS:
                self._log.info("nav_waypoint_retries_exceeded — cancelling")
                self._speak(WAYPOINT_GIVEUP_TEXT)
                self.cancel(speak_cancel=False)
                return
            self._log.info(
                "nav_waypoint_reprompt count=%d", self.context.retry_count,
            )
            self._prompt_for_waypoint(WAYPOINT_REPROMPT_TEXT)
            return

        # B1 = A: resolve each candidate eagerly. A4 = A: drop unmappables.
        # B4 = C: announce which were dropped (only if we kept at least one).
        accepted: list[tuple[dict, list[str]]] = []
        dropped: list[str] = []
        for c in candidates:
            classes = resolve_landmark(c["normalized_text"])
            if classes:
                accepted.append((c, classes))
            else:
                dropped.append(c["normalized_text"])

        if not accepted:
            # B4 sub 1: all-unmappable -> treat as a re-prompt event.
            self.context.retry_count += 1
            if self.context.retry_count > MAX_WAYPOINT_REPROMPTS:
                self._log.info("nav_waypoint_retries_exceeded — cancelling")
                self._speak(WAYPOINT_GIVEUP_TEXT)
                self.cancel(speak_cancel=False)
                return
            self._log.info(
                "nav_waypoint_all_unmappable dropped=%s retry=%d",
                dropped, self.context.retry_count,
            )
            self._speak(WAYPOINT_UNKNOWN_TEXT)
            self.context.last_prompt_at = time.time()
            return

        # B4 sub 3: announce rejection FIRST, then "Looking for ...".
        if dropped:
            self._speak(
                WAYPOINT_PARTIAL_RECOGNITION_TEMPLATE.format(dropped=", ".join(dropped))
            )

        # Store every accepted candidate, in order.
        for cand, classes in accepted:
            wp = self.context.add_waypoint(
                cand["raw_text"], cand["normalized_text"], classes
            )
            self._log.info(
                "nav_waypoint_added raw=%r normalized=%r classes=%s status=%s",
                wp.raw_text, wp.normalized_text, classes, wp.status.value,
            )

        # Exit waypoint-collection sub-state.
        self.context.awaiting_waypoint = False
        self.context.retry_count = 0
        # Decision 10: enable frame processing now that we have a target.
        self.context.frame_processing_enabled = True

        # Part 2: spin up the detector for this task.
        self._detector = LandmarkDetector(
            context=self.context,
            on_reached=self._on_waypoint_reached,
            on_disambiguation_needed=self._on_disambiguation_needed,
            on_detection_timeout=self._on_detection_timeout,
            on_obstacle=self._on_obstacle_warning,  # Part 3
            on_guidance=self._on_guidance,          # Part 4
            logger=self._log,
        )

        # Decision 14 + 15: audible confirmation referencing the active waypoint.
        active = self.context.current_waypoint()
        if active is not None:
            self._speak(
                WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint=active.normalized_text)
            )

    # ---------------------------------------------------------------
    # Disambiguation dialog (C7)
    # ---------------------------------------------------------------

    def _handle_disambiguation_response(self, text: str) -> None:
        """User has been asked 'chair or couch?'. Try to parse their answer."""
        if self._detector is None:
            return
        choices = self.context.disambiguation_choices
        normalized = text.lower().strip()

        picked: Optional[str] = None
        for c in choices:
            if c in normalized:
                picked = c
                break

        if picked is None:
            # C7 sub 2: re-prompt once, then fall back to highest-confidence.
            self.context.disambiguation_retry_count += 1
            if self.context.disambiguation_retry_count > MAX_DISAMBIGUATION_REPROMPTS:
                self._log.info("nav_disambiguation_giveup — falling back")
                self._detector.disambiguate_fallback()
                self._announce_disambiguation_committed()
                return
            self._speak(DISAMBIGUATION_REPROMPT_TEXT)
            self.context.disambiguation_started_at = time.time()
            return

        # Got it.
        self._detector.disambiguate(picked)
        self._announce_disambiguation_committed()

    def _announce_disambiguation_committed(self) -> None:
        """C7 sub 9: tell the user which class we committed to."""
        wp = self.context.current_waypoint() if self.context else None
        if wp is None or wp.locked_class is None:
            return
        self._speak(
            DISAMBIGUATION_CONFIRMATION_TEMPLATE.format(waypoint=wp.locked_class)
        )

    # ---------------------------------------------------------------
    # Recovery dialog after detection timeout (G2 / G3)
    # ---------------------------------------------------------------

    def _handle_recovery_response(self, text: str) -> None:
        """User was just told 'I haven't spotted X — say skip, stop, or rename'."""
        t = text.lower().strip()

        # Skip path (G3 = A).
        if t.startswith("skip") or t == "skip it" or t == "move on":
            self._handle_skip()
            return

        # Otherwise treat the input as a new waypoint description.
        # (G2 sub-decision: 'ask for another landmark or re-search'.)
        # Reuse the waypoint-handling pipeline.
        self._awaiting_recovery = False
        self.context.awaiting_waypoint = True
        self._handle_waypoint_response(text)

    def _handle_skip(self) -> None:
        """G3 = A: skip current waypoint, advance to next; prompt if none."""
        self._log.info("nav_skip waypoint=%r", self.context.current_waypoint())
        self._awaiting_recovery = False
        nxt = self.context.skip_current()
        if self._detector is not None:
            self._detector.reset_for_new_waypoint()

        if nxt is not None:
            self._speak(
                WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint=nxt.normalized_text)
            )
        else:
            # No more queued waypoints — ask for a new one.
            self.context.awaiting_waypoint = True
            self.context.frame_processing_enabled = False
            self._prompt_for_waypoint(NEXT_WAYPOINT_PROMPT_TEXT)

    # ---------------------------------------------------------------
    # Part 5 — "What's next?" dialog (A2 = A)
    # ---------------------------------------------------------------

    def _handle_next_or_done(self, text: str) -> None:
        """User just heard 'What's next?'. They can:
          - say 'done' / 'finished' -> complete the task cleanly.
          - describe a new landmark -> append as next waypoint, advance to it.
          - say 'stop' (handled earlier in routing) -> cancel.
        """
        if is_done_phrase(text):
            self._log.info("nav_user_said_done")
            self.complete()
            return

        # Parse the user's landmark description with the same pipeline as
        # the initial waypoint collection, but bypass the speak/state flow
        # in _handle_waypoint_response — we manage everything inline here.
        candidates = parse_waypoint(text)
        if not candidates:
            # Empty / filler -> re-ask.
            self._speak(WAYPOINT_REPROMPT_TEXT)
            self.context.last_prompt_at = time.time()
            return

        # Resolve + filter mappables (same as Part 2 logic).
        accepted: list[tuple[dict, list[str]]] = []
        dropped: list[str] = []
        for c in candidates:
            classes = resolve_landmark(c["normalized_text"])
            if classes:
                accepted.append((c, classes))
            else:
                dropped.append(c["normalized_text"])

        if not accepted:
            # All unmappable — same dialog.
            self._speak(WAYPOINT_UNKNOWN_TEXT)
            self.context.last_prompt_at = time.time()
            return

        # Optional partial-recognition announcement.
        if dropped:
            self._speak(
                WAYPOINT_PARTIAL_RECOGNITION_TEMPLATE.format(
                    dropped=", ".join(dropped)
                )
            )

        # Append the new waypoint(s) directly to the context. We use
        # add_waypoint but then override the status flow because the
        # previous waypoint is REACHED and current_index needs to move.
        for cand, classes in accepted:
            wp = self.context.add_waypoint(
                cand["raw_text"], cand["normalized_text"], classes
            )
            # add_waypoint marks the FIRST appended as ACTIVE only if the
            # list was empty. Here it isn't, so the new ones are PENDING.
            self._log.info(
                "nav_waypoint_added_after_whats_next raw=%r classes=%s",
                wp.raw_text, classes,
            )

        # Advance past any reached waypoints to the first PENDING one.
        while (
            self.context.current_index < len(self.context.waypoints)
            and self.context.waypoints[self.context.current_index].status
                == WaypointStatus.REACHED
        ):
            self.context.advance()

        new_active = self.context.current_waypoint()
        if new_active is not None:
            new_active.status = WaypointStatus.ACTIVE
            self.context.current_waypoint_started_at = time.time()

        # Exit "what's next" sub-state, re-enable detection.
        self.context.awaiting_next_or_done = False
        self.context.awaiting_waypoint = False
        self.context.frame_processing_enabled = True
        if self._detector is not None:
            self._detector.reset_for_new_waypoint()

        # Announce the new active waypoint.
        if new_active is not None:
            self._speak(
                WAYPOINT_CONFIRMATION_TEMPLATE.format(
                    waypoint=new_active.normalized_text
                )
            )

    def complete(self, speak_complete: bool = True) -> None:
        """Successful end of the navigation task (Part 5 C1).

        Speaks the completion line, emits the FSM task_completed event with
        the summary payload (C2 = A), then tears down state.
        """
        if self.context is None:
            return
        summary = self.context.summary()
        self._log.info(
            "nav_completed destination=%r reached=%d skipped=%d duration=%.1f",
            summary["destination"],
            summary["waypoints_reached"],
            summary["waypoints_skipped"],
            summary["duration_seconds"] or 0.0,
        )
        if speak_complete:
            self._speak(NAVIGATION_COMPLETE_TEXT)
        # Same try/except pattern as cancel (F3 = A).
        try:
            self._fsm.handle_event(
                "task_completed",
                {"task": "navigation", "summary": summary},
            )
        except Exception:
            self._log.exception("fsm_event_dispatch_failed event=task_completed")
        self.context = None
        self._detector = None
        self._awaiting_recovery = False

    # ---------------------------------------------------------------
    # Part 5 — "I'm here" override + confirmation dialog (B1 keep)
    # ---------------------------------------------------------------

    def _handle_arrival_override(self) -> None:
        """User said 'I'm here' / 'got it' / etc. mid-task.

        Logic:
          - If the detector currently shows this waypoint as center+near
            (the reached criterion), accept silently and mark reached as
            if the detector fired (the detector probably will fire on the
            next frame anyway, but accepting here is responsive).
          - Otherwise, ask back with a description of what we currently see
            (or "I don't see X yet" if nothing visible). Wait for yes/no.
        """
        wp = self.context.current_waypoint()
        if wp is None:
            return

        # Build the description of current state.
        observation = self._build_arrival_observation(wp)
        if observation is None:
            # No clear observation -> use the "I don't see X yet" variant.
            text = ARRIVAL_CONFIRMATION_NO_OBSERVATION_TEMPLATE.format(
                waypoint=wp.locked_class or wp.normalized_text
            )
        else:
            text = ARRIVAL_CONFIRMATION_TEMPLATE.format(
                observation=observation,
                waypoint=wp.locked_class or wp.normalized_text,
            )

        self.context.awaiting_arrival_confirmation = True
        self.context.last_prompt_at = time.time()
        # Pause guidance/reached temporarily so we don't talk over the
        # confirmation. Obstacles can still fire (safety doesn't pause).
        # Frame processing remains enabled; the detector's gates already
        # know to skip reached/guidance when awaiting confirmation isn't
        # explicitly checked — we add that check below.
        self._log.info(
            "nav_arrival_override_asked waypoint=%r observation=%r",
            wp.normalized_text, observation,
        )
        self._speak(text)

    def _build_arrival_observation(self, wp: Waypoint) -> Optional[str]:
        """Describe the current detection state for the arrival-confirm prompt.

        Returns None if we have no recent observation worth reporting.
        Otherwise returns a short phrase like "I see the chair on your right."
        """
        from .obstacle_map import display_name
        # Inspect the most recent detection history. If the latest frame
        # had the waypoint's class, describe it.
        history = self.context.detection_history
        if not history:
            return None
        latest = history[-1]
        active = set(wp.active_classes())
        seen = active & latest
        if not seen:
            return None
        # Use the spoken state if known; otherwise just say "I see the X".
        cls = next(iter(seen))
        noun = display_name(cls)
        # Capitalize for sentence-start.
        noun_cap = noun if not noun.islower() else noun.capitalize()
        region = self.context.last_spoken_region
        distance = self.context.last_spoken_distance
        if region and distance:
            # Reuse the guidance phrasings.
            if region == "center":
                if distance == "near":
                    return f"I see the {noun} right in front of you."
                if distance == "medium":
                    return f"I see the {noun} ahead."
                return f"I see the {noun} far ahead."
            side = "left" if region == "left" else "right"
            if distance == "near":
                return f"I see the {noun} on your {side}, right in front of you."
            if distance == "medium":
                return f"I see the {noun} on your {side}."
            return f"I see the {noun} on your {side}, far away."
        # Fallback when last_spoken_* is empty (e.g., first frame).
        return f"I see the {noun} but not in front of you yet."

    def _handle_arrival_confirmation(self, text: str) -> None:
        """User answers the arrival-confirmation prompt (B1 keep)."""
        wp = self.context.current_waypoint() if self.context else None
        if wp is None:
            self.context.awaiting_arrival_confirmation = False
            return

        if is_yes(text):
            self._log.info("nav_arrival_confirmed waypoint=%r", wp.normalized_text)
            self.context.awaiting_arrival_confirmation = False
            # Mark reached the same way detector would have.
            wp.status = WaypointStatus.REACHED
            if self._detector is not None:
                # Make sure the detector knows reached fired so it doesn't
                # try to re-fire on subsequent frames.
                self._detector._reached_already_fired = True
            # Trigger the standard reached path (announce + advance OR ask
            # "what's next").
            self._on_waypoint_reached(wp)
            return

        if is_no(text):
            self._log.info("nav_arrival_declined waypoint=%r", wp.normalized_text)
            self.context.awaiting_arrival_confirmation = False
            self._speak(ARRIVAL_CONFIRMED_DECLINED_TEXT)
            return

        # Anything else: assume the user meant yes (most charitable reading
        # for an assistive interaction). Could re-prompt instead, but a
        # blind user who just said "I'm here" is unlikely to want a
        # multi-turn yes/no clarifier.
        self._log.info(
            "nav_arrival_ambiguous waypoint=%r text=%r — defaulting to yes",
            wp.normalized_text, text,
        )
        # Treat as yes.
        self.context.awaiting_arrival_confirmation = False
        wp.status = WaypointStatus.REACHED
        if self._detector is not None:
            self._detector._reached_already_fired = True
        self._on_waypoint_reached(wp)

    # ---------------------------------------------------------------
    # Part 5 — status query (F4 = A)
    # ---------------------------------------------------------------

    def get_status(self) -> dict:
        """Read-only snapshot of where the task is right now.

        Useful for UI display, debugging, and Sprint 5 evaluation logging.
        Safe to call when no task is active (returns a clear 'inactive' shape).
        """
        if self.context is None:
            return {"active": False}
        c = self.context
        wp = c.current_waypoint()
        return {
            "active": True,
            "destination": c.destination,
            "current_waypoint": wp.normalized_text if wp else None,
            "current_waypoint_status": wp.status.value if wp else None,
            "current_waypoint_classes": (
                wp.locked_class and [wp.locked_class] or (wp.target_classes if wp else [])
            ),
            "waypoint_count": len(c.waypoints),
            "waypoint_index": c.current_index,
            "awaiting_waypoint": c.awaiting_waypoint,
            "awaiting_disambiguation": c.awaiting_disambiguation,
            "awaiting_arrival_confirmation": c.awaiting_arrival_confirmation,
            "awaiting_next_or_done": c.awaiting_next_or_done,
            "awaiting_recovery": self._awaiting_recovery,
            "frame_processing_enabled": c.frame_processing_enabled,
            "last_spoken_region": c.last_spoken_region,
            "last_spoken_distance": c.last_spoken_distance,
            "last_spoken_class": c.last_spoken_class,
        }

    # ---------------------------------------------------------------
    # Detection input (Part 2 — called by partner Ahmad's frame handler)
    # ---------------------------------------------------------------

    def on_detections(self, detections: list[dict]) -> None:
        """Frame handler calls this once per frame (~5 FPS, C3 = A).

        Gated by is_frame_processing_enabled() — handler should consult
        the gate before calling, but we also no-op defensively here.

        Part 5 D3 = A: each dict is normalized to our canonical schema
        before being passed to the detector. Tolerates a few common key-
        name variations so partner-side schema changes don't break us.
        """
        if self._detector is None:
            return
        if not self.is_frame_processing_enabled():
            return
        normalized = [self._normalize_detection(d) for d in detections]
        # Filter out any None entries (malformed input).
        normalized = [d for d in normalized if d is not None]
        self._detector.on_frame_detections(normalized)

    @staticmethod
    def _normalize_detection(d: dict) -> Optional[dict]:
        """D3 = A: adapter that accepts a few common key-name variations.

        Canonical schema (what the detector expects):
            class_name, confidence, bbox, region, distance_category

        Tolerated synonyms:
            class -> class_name
            label -> class_name
            conf  -> confidence
            score -> confidence
            box   -> bbox
            distance -> distance_category

        Returns None if the dict has no recognizable class identifier.
        """
        if not isinstance(d, dict):
            return None
        out = dict(d)  # shallow copy; never mutate the caller's dict
        # Class name.
        if "class_name" not in out:
            if "class" in out:
                out["class_name"] = out["class"]
            elif "label" in out:
                out["class_name"] = out["label"]
            elif "name" in out:
                out["class_name"] = out["name"]
        if "class_name" not in out:
            return None
        # Confidence.
        if "confidence" not in out:
            if "conf" in out:
                out["confidence"] = out["conf"]
            elif "score" in out:
                out["confidence"] = out["score"]
        # Bbox.
        if "bbox" not in out and "box" in out:
            out["bbox"] = out["box"]
        # Distance category.
        if "distance_category" not in out and "distance" in out:
            # Only adopt if the value looks like a category, not a number.
            v = out["distance"]
            if isinstance(v, str):
                out["distance_category"] = v
        return out

    # ---------------------------------------------------------------
    # Detector callbacks
    # ---------------------------------------------------------------

    def _on_waypoint_reached(self, waypoint: Waypoint) -> None:
        """E5 = A callback: detector says current waypoint is reached.

        Part 5 behavior:
          - Announce the reach.
          - If more waypoints queued -> advance + announce next.
          - If queue empty -> ask "What's next?" (A2 = A). User can add
            another landmark, say "done" (complete), or "stop" (cancel).
        """
        self._log.info("manager_received_reached waypoint=%r", waypoint.normalized_text)
        spoken = waypoint.locked_class or waypoint.normalized_text
        self._speak(WAYPOINT_REACHED_TEMPLATE.format(waypoint=spoken))
        if self.context is None:
            return

        if self.context.has_more_waypoints():
            nxt = self.context.advance()
            if self._detector is not None:
                self._detector.reset_for_new_waypoint()
            if nxt is not None:
                self._speak(
                    WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint=nxt.normalized_text)
                )
            return

        # Queue empty -> Part 5 "What's next?" dialog.
        self.context.awaiting_next_or_done = True
        # Pause detection while waiting for the user; otherwise the detector
        # would keep checking the just-reached waypoint pointlessly.
        self.context.frame_processing_enabled = False
        self.context.last_prompt_at = time.time()
        self._speak(WHATS_NEXT_PROMPT_TEXT)

    def _on_disambiguation_needed(self, choices: list[str]) -> None:
        """C7 callback: detector saw multiple alias classes; ask the user."""
        if len(choices) < 2:
            return
        # Pause guidance; obstacle layer (Part 3) keeps running independently.
        a, b = choices[0], choices[1]
        self._speak(DISAMBIGUATION_PROMPT_TEMPLATE.format(a=a, b=b))

    def _on_detection_timeout(self) -> None:
        """G1/G2 callback: 60s without any matching detection. Offer skip."""
        wp = self.context.current_waypoint() if self.context else None
        if wp is None:
            return
        self._log.info("manager_detection_timeout waypoint=%r", wp.normalized_text)
        self._awaiting_recovery = True
        self._speak(DETECTION_TIMEOUT_TEXT.format(waypoint=wp.normalized_text))
        # Re-arm the prompt timer so the user has 30s to answer the
        # recovery question before we cancel.
        self.context.last_prompt_at = time.time()
        self.context.awaiting_waypoint = False  # recovery uses its own flag

    def _on_obstacle_warning(self, info: dict) -> None:
        """Part 3 callback: detector says an obstacle is in the user's path.

        Phrasing depends on:
          - is_person (A3)         -> person-specific phrasing
          - escalation_level (Issue 3) -> mild / reminder / cautionary

        D1 = A: this warning is spoken; the detector has already suppressed
        the reached announcement for this frame.
        """
        from .obstacle_map import display_name
        cls = info["class_name"]
        level = info.get("escalation_level", 0)
        is_person = info.get("is_person", False)

        if is_person:
            templates = (
                OBSTACLE_PERSON_LEVEL0_TEMPLATE,
                OBSTACLE_PERSON_LEVEL1_TEMPLATE,
                OBSTACLE_PERSON_LEVEL2_TEMPLATE,
            )
            text = templates[min(level, 2)]
        else:
            noun = display_name(cls)
            # Only auto-capitalize if the display_name didn't already define
            # casing (e.g. "TV" stays "TV"; "chair" becomes "Chair").
            if noun.islower():
                noun = noun.capitalize()
            templates = (
                OBSTACLE_WARNING_LEVEL0_TEMPLATE,
                OBSTACLE_WARNING_LEVEL1_TEMPLATE,
                OBSTACLE_WARNING_LEVEL2_TEMPLATE,
            )
            text = templates[min(level, 2)].format(noun=noun)

        self._log.info(
            "manager_obstacle_warning class=%r level=%d text=%r",
            cls, level, text,
        )
        self._speak(text)

    def _on_guidance(self, info: dict) -> None:
        """Part 4 callback: detector says a guidance cue is warranted.

        Composes the spoken text from the structured payload. Phrasing
        rules from B1-B6 + Interpretation 1:
          - first_cue / both_changed / class_change: full "Class region, distance"
          - region_changed: full again (B5 = A)
          - distance_changed in same region: "Class region, getting closer"
            (B6 = C, except when distance is now "near" -> use "right in
            front of you" so the user knows it's the final approach).
          - lost_sight: "Lost sight of the X."
        """
        cls = info["class_name"]
        change_type = info["change_type"]
        region = info.get("region")
        distance = info.get("distance_category")

        # Use locked or displayable class name (Title-case the first letter,
        # but preserve special cases like "TV").
        from .obstacle_map import display_name
        noun = display_name(cls)
        if noun.islower():
            noun_cap = noun.capitalize()
        else:
            noun_cap = noun  # e.g. "TV" stays "TV"

        if change_type == "lost_sight":
            text = GUIDANCE_LOST_SIGHT_TEMPLATE.format(waypoint=noun)
            self._log.info(
                "manager_guidance change=%r text=%r", change_type, text,
            )
            self._speak(text)
            return

        # Build the directional+distance phrase.
        rp = _region_phrase(region) if region else "ahead"
        dp = _distance_phrase(distance) if distance else ""

        if change_type == "distance_changed":
            # B6 = C: use "getting closer" for medium, and the full near
            # phrase ("right in front of you") for near, since near is the
            # course-correction moment (D4 = A).
            if distance == "near":
                if region == "center":
                    # Avoid "ahead, right in front of you" — collapse.
                    text = f"{noun_cap} right in front of you."
                else:
                    text = f"{noun_cap} {rp}, {dp}."
            elif distance == "far":
                # Distance went BACK to far — rare; speak full.
                if region == "center":
                    text = f"{noun_cap} far ahead."
                else:
                    text = f"{noun_cap} {rp}, {dp}."
            else:
                # medium -> "getting closer" if we're closing in, or
                # "moving away" if we drifted out. We don't track delta
                # direction in v1; "getting closer" matches the common case.
                text = f"{noun_cap} {rp}, getting closer."
        else:
            # first_cue, region_changed, both_changed, class_change all use
            # the full form.
            if distance == "medium" and region == "center":
                # Avoid double "ahead" ("ahead, approaching" is clunky).
                text = f"{noun_cap} ahead, getting closer."
            else:
                # Compose. If region is center, _region_phrase returns
                # "ahead"; combining with "right in front of you" reads
                # naturally as "right in front of you" without "ahead" repeat.
                if region == "center" and distance == "near":
                    text = f"{noun_cap} right in front of you."
                elif region == "center" and distance == "far":
                    text = f"{noun_cap} far ahead."
                elif region == "center":
                    text = f"{noun_cap} ahead, {dp}."
                else:
                    text = f"{noun_cap} {rp}, {dp}."

        self._log.info(
            "manager_guidance class=%r change=%r region=%r distance=%r text=%r",
            cls, change_type, region, distance, text,
        )
        self._speak(text)

    # ---------------------------------------------------------------
    # Timeout watch
    # ---------------------------------------------------------------

    def check_timeout(self, now: Optional[float] = None) -> bool:
        """Call periodically from the session loop. Returns True if a timeout fired.

        Watches:
          - waypoint-collection / recovery prompt (Part 1: 30s).
          - "What's next?" prompt (Part 5 A4: 30s -> auto-complete).
          - arrival-confirmation prompt (Part 5: 30s -> assume yes).
          - disambiguation prompt (Part 2: 10s).
          - landmark-detection (Part 2: 60s, evaluated by the detector).
        """
        if self.context is None:
            return False
        t = now if now is not None else time.time()
        fired = False

        # 1) Waypoint-prompt + recovery sub-state timeout.
        prompt_timeout_active = (
            self.context.awaiting_waypoint or self._awaiting_recovery
        )
        if prompt_timeout_active:
            elapsed = t - self.context.last_prompt_at
            if elapsed >= WAYPOINT_PROMPT_TIMEOUT_SECONDS:
                self._log.info("nav_prompt_timeout elapsed=%.1fs", elapsed)
                self._speak(WAYPOINT_TIMEOUT_TEXT)
                self.cancel(speak_cancel=False)
                return True

        # 1b) "What's next?" timeout -> auto-complete (Part 5 A4).
        if self.context.awaiting_next_or_done:
            elapsed = t - self.context.last_prompt_at
            if elapsed >= WAYPOINT_PROMPT_TIMEOUT_SECONDS:
                self._log.info("nav_whats_next_timeout — auto-completing")
                self._speak(COMPLETION_TIMEOUT_TEXT)
                self.complete(speak_complete=False)
                return True

        # 1c) Arrival-confirmation timeout -> default to yes (charitable).
        if self.context.awaiting_arrival_confirmation:
            elapsed = t - self.context.last_prompt_at
            if elapsed >= WAYPOINT_PROMPT_TIMEOUT_SECONDS:
                wp = self.context.current_waypoint()
                self._log.info("nav_arrival_confirm_timeout — defaulting to yes")
                self.context.awaiting_arrival_confirmation = False
                if wp is not None:
                    wp.status = WaypointStatus.REACHED
                    if self._detector is not None:
                        self._detector._reached_already_fired = True
                    self._on_waypoint_reached(wp)
                return True

        # 2/3) Detector timeouts (disambiguation + detection).
        if self._detector is not None:
            if self._detector.check_timeout(now=t):
                fired = True

        return fired

    # ---------------------------------------------------------------
    # Cancellation
    # ---------------------------------------------------------------

    def cancel(self, speak_cancel: bool = True) -> None:
        """Cancel the task and signal FSM to transition to ReturningToIdle."""
        if self.context is None:
            return
        if speak_cancel:
            self._speak(WAYPOINT_CANCELLED_TEXT)
        self.context.cancel_remaining()
        self.context.frame_processing_enabled = False
        self._log.info(
            "nav_cancelled destination=%r waypoints=%d",
            self.context.destination, len(self.context.waypoints),
        )
        # Event name matches the FSM contract: any active state -> ReturningToIdle.
        try:
            self._fsm.handle_event("task_cancelled", {"task": "navigation"})
        except Exception:
            self._log.exception("fsm_event_dispatch_failed event=task_cancelled")
        self.context = None
        self._detector = None
        self._awaiting_recovery = False

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _prompt_for_waypoint(self, text: str) -> None:
        assert self.context is not None
        self.context.last_prompt_at = time.time()
        self._speak(text)

    def _speak(self, text: str) -> None:
        """Synthesize TTS and push it over the WebSocket. Failures are logged, not raised.

        D2 = A: auto-detects async TTS implementations and runs them in
        the current event loop if there is one; otherwise raises a clear
        error rather than silently dropping. Most partners' code will be
        sync.
        """
        import asyncio
        import inspect

        try:
            audio = self._tts.synthesize(text)
            if inspect.iscoroutine(audio):
                # The TTS is async. Try to await it in a running loop;
                # otherwise run a new one.
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're called from inside an async context. The
                        # caller has to await the result themselves; the
                        # safest fallback is to schedule and not wait.
                        # In practice partners will configure the manager
                        # with a sync wrapper for the WebSocket path.
                        self._log.warning(
                            "async TTS detected inside running loop; "
                            "audio will be scheduled but not awaited here"
                        )
                        asyncio.ensure_future(audio)
                        return
                    audio = loop.run_until_complete(audio)
                except RuntimeError:
                    audio = asyncio.new_event_loop().run_until_complete(audio)
        except Exception:
            self._log.exception("tts_synthesize_failed text=%r", text)
            return
        try:
            self._send_tts(audio)
        except Exception:
            self._log.exception("tts_send_failed text=%r", text)

    # ---------------------------------------------------------------
    # Read-only accessors for the WebSocket / frame-processing layer
    # ---------------------------------------------------------------

    def is_frame_processing_enabled(self) -> bool:
        """The frame-handler should consult this before calling YOLO (decision 10)."""
        return self.context is not None and self.context.frame_processing_enabled

    def is_active(self) -> bool:
        return self.context is not None


# ----------------------------------------------------------------------
# Part 5 D4 = A: wiring helper for partners.
# ----------------------------------------------------------------------

def create_navigation_session(
    tts_service,
    fsm,
    send_tts: Callable[[bytes], None],
    logger: Optional[logging.Logger] = None,
) -> NavigationTaskManager:
    """One-line constructor for a NavigationTaskManager.

    Use from the WebSocket session setup code:

        from navigation import create_navigation_session

        nav = create_navigation_session(
            tts_service=omar_tts_service,
            fsm=omar_fsm,
            send_tts=lambda mp3: ws.send_bytes(mp3),
        )

    Equivalent to calling NavigationTaskManager(...) directly with kwargs;
    this helper exists mainly so partners have one import to remember.
    """
    return NavigationTaskManager(
        tts_service=tts_service,
        send_tts=send_tts,
        fsm=fsm,
        logger=logger,
    )
