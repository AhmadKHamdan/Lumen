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

from .config import OBST_SIGNAL, FLOOR_WALKABLE_LABELS

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

# Unnamed-obstacle signal (see config.OBST_SIGNAL). Only the selected model loads:
#   "floor" -> SegFormer-B0 (ADE20K semantic segmentation): is the walking lane floor?
#   "depth" -> Depth Anything V2 Small (relative depth): the older tripwire, fallback.
# Either is optional — if it can't load, the YOLO named-obstacle layer still runs.
_depth_pipe = None
_seg_infer = None          # callable: PIL image -> HxW ndarray of ADE20K class ids
_seg_walkable_ids = None   # ids whose label counts as walkable (floor/rug/door...)
if OBST_SIGNAL == "floor":
    try:
        import torch  # noqa: E402
        from transformers import (AutoImageProcessor,  # noqa: E402
                                  SegformerForSemanticSegmentation)
        print("Loading floor segmentation (SegFormer-B0, ADE20K)...")
        _seg_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        _seg_proc = AutoImageProcessor.from_pretrained(_seg_name)
        _seg_dev = "cuda" if torch.cuda.is_available() else "cpu"
        _seg_model = SegformerForSemanticSegmentation.from_pretrained(_seg_name).to(_seg_dev).eval()
        _seg_walkable_ids = np.array(
            [int(i) for i, n in _seg_model.config.id2label.items()
             if any(k in n.lower() for k in FLOOR_WALKABLE_LABELS)])

        def _seg_infer(pil_img):
            with torch.no_grad():
                inp = _seg_proc(images=pil_img, return_tensors="pt").to(_seg_dev)
                return _seg_model(**inp).logits.argmax(1)[0].cpu().numpy()
    except Exception as e:  # noqa: BLE001 — degrade to YOLO-only obstacles
        print(f"Floor segmentation unavailable ({type(e).__name__}); obstacle watchdog "
              "will use YOLO classes only. To enable it: pip install transformers torch")
elif OBST_SIGNAL == "depth":
    try:
        import torch  # noqa: E402
        from transformers import pipeline as _hf_pipeline  # noqa: E402
        print("Loading depth model (Depth Anything V2 Small)...")
        _depth_pipe = _hf_pipeline("depth-estimation",
                                   model="depth-anything/Depth-Anything-V2-Small-hf",
                                   device=0 if torch.cuda.is_available() else -1)
    except Exception as e:  # noqa: BLE001 — degrade to YOLO-only obstacles
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
if _seg_infer is not None:
    _seg_infer(Image.fromarray(np.zeros((640, 480, 3), dtype=np.uint8)))
print("Ready.")
