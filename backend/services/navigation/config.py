"""All tunable constants for the Lumen exploration controller, grouped by concern.

This is the one place to adjust behavior. Every other module does
`from .config import *`, so the names here are referenced unqualified elsewhere.
No logic lives here — values only.
"""
from __future__ import annotations

# --- Detection / temporal confirmation ------------------------------------------
CONF = 0.45
WINDOW, MIN_HITS = 3, 2  # 2-of-3 temporal rule for DOORS (presence, not accumulation)
# Indicators are accumulated across the whole room-scan instead: an object seen in
# >= this many frames during one room visit counts as confirmed. A slow sweep then
# builds evidence rather than forgetting it after 3 frames. Arrival is declared
# straight from the scan (no second confirmation pass), so the bar is high.
INDICATOR_HITS = 4
# Motion gate: if the phone pans too fast the frame is blurred and useless, so we
# skip detection and coach the user to slow down. Tunable (small-image mean diff).
# 85, up from 60: normal handheld turning was tripping the gate constantly.
MOTION_MAX = 85.0
# A single blurred frame (autofocus hunt, exposure change) is NOT the user moving
# fast — only SPEAK "slow down" after this many consecutive too-fast frames.
# Blurred frames are still silently skipped for detection either way.
MOTION_COACH_FRAMES = 3

# --- Door perception stack -------------------------------------------------------
DOOR_CONF = 0.45
# Shape sanity check: doors aren't extremely wide. Rejecting very wide boxes cuts
# some wall/furniture false positives without hurting normal door recall.
DOOR_MAX_WH = 1.4  # max width/height ratio to still count as a door
# Blank walls fire as doors with door-level confidence. A real door has internal
# structure (frame, panel seams, hinges, handle) -> many edges; a blank wall is
# smooth -> almost none. Reject door boxes whose interior edge density is too low.
DOOR_EDGE_MIN = 0.030  # fraction of Canny-edge pixels inside the box (tunable)
# The strongest wall tell: the detector boxes the WHOLE scene (full width AND height)
# and calls it a door. A real door you'd route toward always leaves wall/floor margin
# around it -> it never fills the frame during the scan. Above this area fraction we
# treat a "door" as a whole-wall latch and drop it. Skipped once we're committed to
# approaching a door (go_door/face_target), where the box legitimately grows as we near it.
DOOR_MAX_FRAME_FRAC = 0.80
DOOR_DEBUG = True  # print each door candidate's conf/edge/fill + keep/reject to the terminal
# Second-opinion verification (4-class DoorDetect model: door/handle/cabinet/fridge door).
# Geometry can't separate a lace curtain from a door (curtains are edge-rich, tall,
# door-sized), so weak candidates must be corroborated SEMANTICALLY: either the
# verifier also sees a door there, or it sees a handle inside the box. Curtains have
# neither. Strong candidates (conf above DOOR_STRONG_CONF) pass on their own.
DOOR_STRONG_CONF = 0.60  # accept a door on primary-model confidence alone above this
VERIFY_CONF = 0.30       # verifier runs permissive; it only corroborates, never detects alone
VERIFY_IOU = 0.40        # verifier box overlapping a candidate this much = same object
# A fridge and a door look alike and both models see the same frame. If a door box
# and an object box overlap this much, treat them as the same thing and keep only
# the higher-confidence one (kills door<->refrigerator double-claims).
DOOR_OBJ_IOU = 0.5

# --- Distance-from-known-size (pinhole) ------------------------------------------
# A door is ~2 m tall, so its pixel height tells us roughly how far away it is.
# HFOV ~60 deg is typical for a phone webcam; focal length in pixels is derived
# per-frame from the image's long side.
DOOR_HEIGHT_M = 2.0
ASSUMED_HFOV_DEG = 60.0
STEP_LENGTH_M = 0.75  # average walking step
# One-shot empirical calibration for the whole distance chain (true HFOV, lens
# distortion, box looseness): stand at a KNOWN distance from a door, read the
# [dist] debug line, then set DIST_CAL = true_distance / reported_distance.
DIST_CAL = 1.0
HAND_REACH_STEPS = 3  # within this many steps, ask the user to reach out and feel for the door

# --- Pacing ----------------------------------------------------------------------
# Pacing/timeout constants are SECONDS of observed-frame time. The engine passes a
# clamped per-tick dt into the controller, so the timers mean the same thing at any
# loop rate, and a stall (TTS synthesis, slow inference) can neither rush a timeout
# nor fake an absence ("door gone for 2 s" needs 2 s of FRAMES without the door,
# not 2 s of wall clock while nothing was observed). Evidence thresholds
# (OBST_HITS, NEAR_STREAK, WALK_FRAMES_MIN, the 2-of-3 door rule, sighting minima)
# stay as consecutive-OBSERVATION counts - those are samples, not durations.
REPROMPT_SEC = 4.0     # seconds between spoken re-prompts
ROOM_SCAN_SEC = 10.0   # seconds for the go_indicator re-confirm before giving up
# The directed confirmation is a RE-check of evidence the 360 scan already flagged —
# it doesn't need the scan's own strictness. Fewer hits + a short settle keep the
# arrival snappy (~2 s of steady pointing) while still naming ALL the adjacent
# indicators (the fridge right next to the oven), not just whichever confirmed first.
CONFIRM_HITS = 3       # detection frames per class during go_indicator (scan keeps 4)
CONFIRM_SETTLE_SEC = 1.0  # extra settle so neighbours cross the bar too
DOOR_LOST_SEC = 12.0   # go_door seconds with no door re-confirmed -> full rescan

# --- Compass-driven 360 scan -----------------------------------------------------
# The user does ONE slow guided turn. The phone compass tracks rotation so we know
# when a full circle is done, and tags detections into fine BUCKETS sectors for the
# spatial summary. No rigid stop-and-hold per sector.
BUCKETS = 12                      # 30-deg sectors -> fine coverage, no gaps
BUCKET_DEG = 360.0 / BUCKETS
FULL_TURN_DEG = 350.0             # rotated this far = back to the start direction (full circle)
START_TOL = 25.0                  # within this many deg of the start heading = "back at start"
STILL_MAX = 12.0                  # (fallback path only) motion below = phone steady
FALLBACK_SCAN_SEC = 13.0          # (no-compass path) seconds of steady capture before finishing
FACE_TOL = 25.0                   # within this many deg of the door's heading = "facing it"
DOOR_CLUSTER_DEG = 30.0           # door sightings within this many deg = the SAME door
# Reliability gates for what the scan REPORTS and ACTS ON. Detection isn't as good
# as human eyes: a door or fridge that flickered for a frame or two is noise — never
# speak it, never navigate to it.
DOOR_MIN_SIGHTINGS = 3            # evidence UNITS for a real door: a verifier-corroborated
                                  # frame counts 2, a bare strong-conf frame counts 1 — so
                                  # 2 corroborated sweep-past frames suffice, 3 bare ones needed
OBJ_MIN_SIGHTINGS = 3             # an object mention (per direction) needs this many sightings

# --- Doorway transit -------------------------------------------------------------
DOOR_FILL_FRAC = 0.85  # door height (fraction of frame) meaning "you're at the doorway"
# The at-the-door instruction asks for a whole sequence (open, walk through, take 2-3
# steps in) — announcing "you're through" too early talks over the user mid-action,
# but waiting too long feels broken. ~2 s with our door out of view lands right after
# the steps in (far doors glimpsed in the NEW room don't reset this — see transit).
TRANSIT_GONE_SEC = 2.0  # seconds with our door gone after being at it -> walked through
# At arm's length a door is a flat panel: the detector still boxes it (huge box) but
# the edge gate rejects it (smooth interior). If we TRACKED an approach down to this
# distance, a saturated door box means "at the door" — walls can't fake that, because
# they were never confirmed as an approaching door first.
NEAR_DOOR_M = 2.0      # last confirmed distance below this = the approach reached the door
                       # (at-the-door readings saturate ~1.7-1.8 m; 3.5 was so loose that
                       # STANDING 3 steps away armed the latch without a single step)
APPROACH_FRAC = 0.8    # ...or the confirmed door grew to this frame-height fraction
                       # (genuine at-door frames read 0.94-1.0; a door 3 steps away
                       # already fills ~0.75, so 0.5 was trivially satisfied)
# The camera itself tells us whether the user MOVED: walking produces sustained
# frame-to-frame motion (> STILL_MAX), standing still reads near zero. "You're
# through" additionally requires this many movement frames after reaching the door —
# no movement, no transit, no matter what detection flickers do.
WALK_FRAMES_MIN = 4
# "You're right at the door" needs this many CONSECUTIVE qualifying frames. A single
# spiky loose box (86% of frame height while mid-sidestep, next frame 3.4 m) once
# armed the latch and spoke at-door instructions out of order with the obstacle flow.
NEAR_STREAK = 2
# New rooms throw full-frame door candidates too, which would hold the at-the-door
# latch forever (the user already walked through!). Cap how long the latch can hold
# without a properly confirmed door before we infer the transit happened. Sized well
# above TRANSIT_GONE_SEC so the failsafe never fires while the user is still standing at
# the door listening to the (long) walk-through instruction.
NEAR_HOLD_MAX_SEC = 13.0

# --- Journey limits (design doc §4.5 / §9.5) --------------------------------------
# Without SLAM the explorer has no map, so a wrong wing of a building could keep it
# walking forever. rooms_visited counts doorway transits per JOURNEY (not per room);
# at the cap Lumen checks in with the user instead of looping silently - "stop"
# (already a global voice command) is the escape hatch, continuing is the default.
ROOM_CAP = 4             # rooms searched before checking in with the user
ROOM_CAP_REMIND = 2      # ...then remind every this many further rooms
NAV_TIMEOUT_SEC = 300.0  # whole-journey hard stop (engine enforces, speaks first)

# --- Obstacle watchdog (Phase A: YOLO classes; the depth tripwire is Phase B) ----
# While the user WALKS to a door (go_door), warn about known objects standing in the
# lower-centre "walking lane". COCO can't see unnamed clutter (clothes piles, boxes)
# -> that's what Phase B's depth model adds; this is the named-object safety layer.
OBSTACLE_CLASSES = {"person", "chair", "couch", "bed", "dining table", "potted plant",
                    "backpack", "handbag", "suitcase", "bench", "tv", "dog", "cat"}
OBST_NAMES = {"dining table": "table", "potted plant": "plant", "tv": "TV"}  # out loud
CORRIDOR_X = (0.30, 0.70)  # central horizontal band = the lane the user walks into
OBST_BOTTOM_FRAC = 0.62    # box bottom must reach below this (near the floor / close)
OBST_MIN_H_FRAC = 0.18     # ignore tiny, far boxes
OBST_MIN_OVERLAP = 0.10    # min lane overlap (fraction of width) to count as "in the way"
                           # (0.04 let side furniture grazing the lane edge trigger stops)
OBST_HITS = 3              # consecutive frames before the FIRST alert (kills flicker)
OBST_CLEAR_HITS = 2        # consecutive clear frames before declaring the path clear
OBST_REPROMPT_SEC = 3.0    # seconds between repeated warnings while still blocked
# After the door call-out ("...let me check the path ahead"), obstacle speech is held
# this many frames so the call-out finishes playing, then the path verdict follows —
# either "the path is clear, walk..." or the obstacle warning. Detection still runs
# during the hold; only the SPEECH waits.
OBST_HOLDOFF_SEC = 2.0

# --- Unnamed-obstacle signal (OBST-3). The question was never "is there an object?"
# but "is the strip of floor I'm about to walk on clear?" — so the primary signal is
# now FLOOR SEGMENTATION: a semantic model labels every pixel, and the lane is
# blocked when it stops being mostly floor. Class-agnostic like the old depth
# tripwire, but grounded: a fan/bin BESIDE the path leaves the floor corridor
# continuous (silent), a chair IN the path punches a hole in it (stop).
# The depth tripwire is kept below as a one-line fallback for A/B comparison. ---
OBST_SIGNAL = "floor"      # "floor" (segmentation) | "depth" (old tripwire) | "off" (YOLO only)
# The lane judged for walkable floor: the strip just ahead of the feet, centre band.
FLOOR_LANE_ROWS = (0.66, 0.94)   # rows of the frame checked for floor
FLOOR_LANE_X = (0.38, 0.62)      # centre band (same width the depth lane used)
FLOOR_MIN_COVER = 0.60     # lane must be at least this fraction walkable, else blocked
# Pixel classes that count as walkable: floor/rug/carpet — plus DOOR, because in
# go_door the door is the destination we're walking toward, never an obstacle.
FLOOR_WALKABLE_LABELS = ("floor", "rug", "carpet", "door")

# --- Phase B fallback: depth tripwire (class-agnostic). Self-calibrates per frame:
# the bottom strip (floor at the feet) anchors "near", the top-centre strip anchors
# "far"; an obstacle = the centre lane reads much nearer than the side floor at the
# same height. Live testing showed it can't separate beside-path from in-path in
# cluttered rooms — kept only for A/B comparison via OBST_SIGNAL = "depth". ---
DEPTH_INPUT_W = 384        # downscale width for the depth net (~75 ms vs ~150 ms)
# The depth lane is DELIBERATELY small: the floor strip 1-2 steps ahead of the feet,
# in a narrow centre band. Live false alarms (a fan and a bin BESIDE the path, a far
# wall) all came from judging higher rows and wider columns — things the user would
# simply walk past. A real blocker mid-path floods this tight strip; side objects
# barely graze it.
DEPTH_LANE_ROWS = (0.68, 0.88)   # floor strip just ahead of the feet
DEPTH_LANE_X = (0.38, 0.62)      # narrower centre band than the YOLO corridor
DEPTH_NEAR_ROWS = (0.88, 1.00)   # bottom strip = floor at the feet (near anchor)
DEPTH_FAR_ROWS = (0.00, 0.30)    # top strip ahead (far anchor)
DEPTH_FLAT_MIN = 8.0       # if near/far anchors differ by less than this (of 255), the
                           # scene is too flat to judge (e.g. facing a near wall) -> skip
DEPTH_REL_MARGIN = 0.18    # a lane pixel this much nearer than its row's floor = intrusion
DEPTH_AREA_FRAC = 0.40     # near HALF the tight lane must be blocked before we alert
