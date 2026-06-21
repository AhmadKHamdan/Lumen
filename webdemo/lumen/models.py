"""Model loading + warmup. Importing this module loads the models (once) — same
timing as before, when importing the server did it. Other modules reference the
loaded models as `_model`, `_door_model`, `_verify_model`, `_names`, `_name_to_id`,
and `_depth_pipe`.

Models, in order of use:
- `_model`        YOLOv8m (COCO): goal indicators (fridge, oven, …) + obstacle classes.
- `_door_model`   custom single-class door detector (best.pt, mAP50 ~0.95).
- `_verify_model` 4-class DoorDetect (door/handle/cabinet/fridge door) — second opinion.
- `_depth_pipe`   Depth Anything V2 Small — class-agnostic obstacle tripwire (optional).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# yolov8m, not n: on a GPU it's ~18 ms/frame slower but far more reliable at
# spotting kitchen/bathroom indicators at angle and distance. Detection accuracy
# is the bottleneck here, not local inference time.
print("Loading YOLOv8m (COCO)...")
from ultralytics import YOLO  # noqa: E402
_model = YOLO("yolov8m.pt")
_names = _model.names
_name_to_id = {v: k for k, v in _names.items()}  # COCO class name -> id (for class filtering)

# Custom single-class door detector. Resolved relative to the repo root so it works
# regardless of the shell's cwd.
_door_path = _REPO_ROOT / "best.pt"
print(f"Loading door model: {_door_path.name} ...")
_door_model = YOLO(str(_door_path))

# 4-class DoorDetect verifier (door/handle/cabinet door/refrigerator door). Too low
# recall to be the primary detector, but ideal as a SECOND OPINION: corroborate weak
# door candidates and arbitrate door-vs-fridge claims. Optional — absent = old behavior.
_verify_path = (_REPO_ROOT / "door_training" / "runs" / "detect"
                / "door_yolov8s_4cls" / "weights" / "best.pt")
_verify_model = None
if _verify_path.exists():
    print("Loading 4-class door verifier...")
    _verify_model = YOLO(str(_verify_path))

# Phase B obstacle layer: monocular depth (Depth Anything V2 Small). Runs ONLY while
# walking to a door, and catches UNNAMED clutter (clothes piles, boxes, bins) that the
# YOLO class layer is blind to. Optional — if it can't load, Phase A still runs.
_depth_pipe = None
try:
    import torch  # noqa: E402
    from transformers import pipeline as _hf_pipeline  # noqa: E402
    print("Loading depth model (Depth Anything V2 Small)...")
    _depth_pipe = _hf_pipeline("depth-estimation",
                               model="depth-anything/Depth-Anything-V2-Small-hf",
                               device=0 if torch.cuda.is_available() else -1)
except Exception as e:  # noqa: BLE001 — any failure -> degrade to YOLO-only obstacles
    print(f"Depth model unavailable ({type(e).__name__}); obstacle watchdog will use "
          "YOLO classes only. To enable it: pip install transformers torch")

# Warm up models so the FIRST real frame isn't stalled by CUDA/kernel init
# (that lag is the long silence at the start of the demo).
print("Warming up models...")
_warm = np.zeros((480, 640, 3), dtype=np.uint8)
_model.predict(_warm, verbose=False)
_door_model.predict(_warm, verbose=False)
if _verify_model is not None:
    _verify_model.predict(_warm, verbose=False)
if _depth_pipe is not None:
    _depth_pipe(Image.fromarray(np.zeros((288, 384, 3), dtype=np.uint8)))
print("Ready.")
