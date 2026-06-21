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
MOTION_MAX = 60.0

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
REPROMPT = 12          # cycles between spoken re-prompts (~4 s)
ROOM_SCAN_CYCLES = 30  # frames for the go_indicator re-confirm before giving up
DOOR_LOST_CYCLES = 36  # go_door frames with no door re-confirmed -> full rescan (~12 s)

# --- Compass-driven 360 scan -----------------------------------------------------
# The user does ONE slow guided turn. The phone compass tracks rotation so we know
# when a full circle is done, and tags detections into fine BUCKETS sectors for the
# spatial summary. No rigid stop-and-hold per sector.
BUCKETS = 12                      # 30-deg sectors -> fine coverage, no gaps
BUCKET_DEG = 360.0 / BUCKETS
FULL_TURN_DEG = 350.0             # rotated this far = back to the start direction (full circle)
START_TOL = 25.0                  # within this many deg of the start heading = "back at start"
STILL_MAX = 12.0                  # (fallback path only) motion below = phone steady
FALLBACK_FRAMES = 40              # (no-compass path) usable frames before finishing
FACE_TOL = 25.0                   # within this many deg of the door's heading = "facing it"
DOOR_CLUSTER_DEG = 30.0           # door sightings within this many deg = the SAME door
# Reliability gates for what the scan REPORTS and ACTS ON. Detection isn't as good
# as human eyes: a door or fridge that flickered for a frame or two is noise — never
# speak it, never navigate to it.
DOOR_MIN_SIGHTINGS = 3            # a door cluster needs this many confirmed sightings to be real
OBJ_MIN_SIGHTINGS = 3             # an object mention (per direction) needs this many sightings

# --- Doorway transit -------------------------------------------------------------
DOOR_FILL_FRAC = 0.85  # door height (fraction of frame) meaning "you're at the doorway"
TRANSIT_GONE = 3       # cycles with no door after being at one -> user walked through
# At arm's length a door is a flat panel: the detector still boxes it (huge box) but
# the edge gate rejects it (smooth interior). If we TRACKED an approach down to this
# distance, a saturated door box means "at the door" — walls can't fake that, because
# they were never confirmed as an approaching door first.
NEAR_DOOR_M = 3.5      # last confirmed distance below this = the approach reached the door
APPROACH_FRAC = 0.5    # ...or the confirmed door grew to this frame-height fraction
                       # (distance-scale independent, so calibration can't break it)
# New rooms throw full-frame door candidates too, which would hold the at-the-door
# latch forever (the user already walked through!). Cap how long the latch can hold
# without a properly confirmed door before we infer the transit happened.
NEAR_HOLD_MAX = 20     # ~7 s at ~3 fps

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
OBST_MIN_OVERLAP = 0.04    # min lane overlap (fraction of width) to count as "in the way"
OBST_HITS = 2              # consecutive frames before the FIRST alert (kills flicker)
OBST_CLEAR_HITS = 2        # consecutive clear frames before declaring the path clear
OBST_REPROMPT = 9          # frames between repeated warnings while still blocked (~3 s)

# --- Phase B: depth tripwire (class-agnostic). We can't trust an absolute near/far
# convention from the model, so each frame self-calibrates: the bottom strip (floor at
# the feet) anchors "near", the top-centre strip anchors "far", and every region is
# scored 0 (far) .. 1 (near) between them. An obstacle = the centre lane reads much
# nearer than the side floor at the same height. Sign-agnostic and scale-free. ---
DEPTH_INPUT_W = 384        # downscale width for the depth net (~75 ms vs ~150 ms)
DEPTH_LANE_ROWS = (0.45, 0.85)   # mid-lower rows = the strip just ahead of the feet
DEPTH_NEAR_ROWS = (0.88, 1.00)   # bottom strip = floor at the feet (near anchor)
DEPTH_FAR_ROWS = (0.00, 0.30)    # top strip ahead (far anchor)
DEPTH_FLAT_MIN = 8.0       # if near/far anchors differ by less than this (of 255), the
                           # scene is too flat to judge (e.g. facing a near wall) -> skip
DEPTH_REL_MARGIN = 0.18    # a lane pixel this much nearer than its row's floor = intrusion
DEPTH_AREA_FRAC = 0.12     # this fraction of the centre lane intruding = a real obstacle
