"""Navigation model loading + warmup (shared across sessions — models are stateless).

Unlike the webdemo prototype (which loaded everything at import), the backend loads
lazily via :func:`ensure_loaded` so that importing ``services.navigation`` — e.g. in
the unit tests, or at server startup — stays instant. The navigation engine calls
``ensure_loaded()`` (in a thread) the first time a navigation task starts; every
later task reuses the loaded singletons.

Other modules reference the loaded models late-bound (``models._model``, never
``from .models import _model``) so they see the post-load values.

Models, in order of use:
- ``_model``        YOLOv8m (COCO): goal indicators (fridge, oven, …) + obstacle classes.
- ``_door_model``   custom single-class door detector (best.pt, mAP50 ~0.95).
- ``_verify_model`` 4-class DoorDetect (door/handle/cabinet/fridge door) — second opinion.
- ``_seg_infer``    SegFormer-B0 floor segmentation — class-agnostic obstacle signal.
- ``_depth_pipe``   Depth Anything V2 Small — the older obstacle tripwire (fallback).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .config import FLOOR_WALKABLE_LABELS, OBST_SIGNAL

log = logging.getLogger("lumen.nav.models")

# backend/services/navigation/models.py -> repo root (weights live there / in
# door_training/, exactly where the training pipeline wrote them).
_REPO_ROOT = Path(__file__).resolve().parents[3]

_model = None
_names = None
_name_to_id = None       # COCO class name -> id (for class filtering)
_door_model = None
_verify_model = None
_depth_pipe = None
_seg_infer = None        # callable: PIL image -> HxW ndarray of ADE20K class ids
_seg_walkable_ids = None  # ids whose label counts as walkable (floor/rug/door...)

_loaded = False
_load_lock = threading.Lock()


def ensure_loaded() -> None:
    """Load + warm up all navigation models, once. Blocking (seconds on first call;
    call from a worker thread). Thread-safe; later calls are no-ops."""
    global _model, _names, _name_to_id, _door_model, _verify_model
    global _depth_pipe, _seg_infer, _seg_walkable_ids, _loaded
    with _load_lock:
        if _loaded:
            return

        # yolov8m, not n: on a GPU it's ~18 ms/frame slower but far more reliable at
        # spotting kitchen/bathroom indicators at angle and distance. Detection
        # accuracy is the bottleneck here, not local inference time. (Object
        # Allocation keeps its own YOLOv8n in yolo_service — CPU-friendly and its
        # noun list doesn't need the extra recall.)
        log.info("Loading YOLOv8m (COCO)...")
        from ultralytics import YOLO
        _model = YOLO("yolov8m.pt")
        _names = _model.names
        _name_to_id = {v: k for k, v in _names.items()}

        # Custom single-class door detector, resolved relative to the repo root so
        # it works regardless of the shell's cwd.
        door_path = _REPO_ROOT / "best.pt"
        log.info("Loading door model: %s ...", door_path)
        _door_model = YOLO(str(door_path))

        # 4-class DoorDetect verifier (door/handle/cabinet door/refrigerator door).
        # Too low recall to be the primary detector, but ideal as a SECOND OPINION:
        # corroborate weak door candidates and arbitrate door-vs-fridge claims.
        # Optional — absent = prototype's old behavior.
        verify_path = (_REPO_ROOT / "door_training" / "runs" / "detect"
                       / "door_yolov8s_4cls" / "weights" / "best.pt")
        if verify_path.exists():
            log.info("Loading 4-class door verifier...")
            _verify_model = YOLO(str(verify_path))
        else:
            log.warning("Door verifier weights not found at %s — running without "
                        "semantic verification.", verify_path)

        # Unnamed-obstacle signal (see config.OBST_SIGNAL). Only the selected model
        # loads:
        #   "floor" -> SegFormer-B0 (ADE20K): is the walking lane floor?
        #   "depth" -> Depth Anything V2 Small (relative depth): older tripwire.
        # Either is optional — if it can't load, the YOLO named-obstacle layer
        # still runs.
        if OBST_SIGNAL == "floor":
            try:
                import torch
                from transformers import (AutoImageProcessor,
                                          SegformerForSemanticSegmentation)
                log.info("Loading floor segmentation (SegFormer-B0, ADE20K)...")
                seg_name = "nvidia/segformer-b0-finetuned-ade-512-512"
                seg_proc = AutoImageProcessor.from_pretrained(seg_name)
                seg_dev = "cuda" if torch.cuda.is_available() else "cpu"
                seg_model = (SegformerForSemanticSegmentation
                             .from_pretrained(seg_name).to(seg_dev).eval())
                _seg_walkable_ids = np.array(
                    [int(i) for i, n in seg_model.config.id2label.items()
                     if any(k in n.lower() for k in FLOOR_WALKABLE_LABELS)])

                def _infer(pil_img):
                    with torch.no_grad():
                        inp = seg_proc(images=pil_img, return_tensors="pt").to(seg_dev)
                        return seg_model(**inp).logits.argmax(1)[0].cpu().numpy()

                _seg_infer = _infer
            except Exception as e:  # noqa: BLE001 — degrade to YOLO-only obstacles
                log.warning("Floor segmentation unavailable (%s); obstacle watchdog "
                            "will use YOLO classes only. To enable it: "
                            "pip install transformers torch", type(e).__name__)
        elif OBST_SIGNAL == "depth":
            try:
                import torch
                from transformers import pipeline as hf_pipeline
                log.info("Loading depth model (Depth Anything V2 Small)...")
                _depth_pipe = hf_pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1)
            except Exception as e:  # noqa: BLE001 — degrade to YOLO-only obstacles
                log.warning("Depth model unavailable (%s); obstacle watchdog will "
                            "use YOLO classes only.", type(e).__name__)

        # Warm up so the FIRST real frame isn't stalled by CUDA/kernel init
        # (that lag is the long silence at the start of a navigation task).
        log.info("Warming up navigation models...")
        warm = np.zeros((480, 640, 3), dtype=np.uint8)
        _model.predict(warm, verbose=False)
        _door_model.predict(warm, verbose=False)
        if _verify_model is not None:
            _verify_model.predict(warm, verbose=False)
        if _depth_pipe is not None:
            _depth_pipe(Image.fromarray(np.zeros((288, 384, 3), dtype=np.uint8)))
        if _seg_infer is not None:
            _seg_infer(Image.fromarray(np.zeros((640, 480, 3), dtype=np.uint8)))
        log.info("Navigation models ready.")
        _loaded = True


def is_loaded() -> bool:
    return _loaded
