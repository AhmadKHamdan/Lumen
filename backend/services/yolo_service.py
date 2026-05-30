"""
Object detection service (YOLOv8n via ultralytics).

Wraps the ultralytics ``YOLO("yolov8n.pt")`` detector. Designed to be
standalone and importable - run from a Python REPL.

Why YOLOv8n?
- Nano variant: ~6 MB weights, fastest of the YOLOv8 family, CPU-friendly.
- COCO-pretrained, so its 80 class names line up with the noun list the
  command parser already canonicalizes to ("cup", "cell phone", ...).
- ultralytics ships pre-built wheels (incl. Python 3.13) and pulls torch +
  torchvision + opencv automatically.

Model download
--------------
The ``yolov8n.pt`` weights auto-download (~6 MB) from the ultralytics CDN on
the first inference call and are cached under ``~/.config/Ultralytics`` (or
``%APPDATA%\\Ultralytics`` on Windows). Subsequent runs reuse the cache.

Lazy loading
------------
ultralytics + torch are heavy to import, so we defer ``from ultralytics import
YOLO`` until the first detect() call. That keeps ``import yolo_service`` cheap
(only stdlib + numpy at module load), which also lets the pure detection-parsing
helper ``_extract_detections`` be unit-tested without torch installed.

A ``detect()`` call returns a list of :class:`Detection`, each carrying the
COCO class label, the confidence, and the pixel-space bounding box
``(x1, y1, x2, y2)``.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger("lumen.yolo")

# Weights file. ultralytics resolves a bare name against its model zoo and
# downloads it on first use.
_MODEL_NAME = "yolov8n.pt"

# Default confidence floor. Detections below this are dropped before they
# ever reach the spatial-reasoning layer. 0.35 trades a few missed frames for
# far fewer phantom objects - important when guiding a blind user.
DEFAULT_CONF_THRESHOLD = 0.35

# Inference image size (square). 640 is the YOLOv8 default; smaller is faster
# but loses small/distant objects.
_IMGSZ = 640

_model = None  # ultralytics.YOLO - lazy-loaded
_model_lock = threading.Lock()


@dataclass(frozen=True)
class Detection:
    """One detected object in a single frame.

    Attributes
    ----------
    label : str
        COCO class name, e.g. ``"cup"`` or ``"cell phone"``.
    confidence : float
        Detector confidence in [0, 1].
    box : tuple[float, float, float, float]
        Pixel-space bounding box ``(x1, y1, x2, y2)`` with the origin at the
        top-left of the frame.
    """

    label: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def _get_model():
    """Lazy-load the YOLOv8n model on first call (downloads weights if needed)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info(
            "Loading YOLO model %r (downloads ~6MB from the ultralytics CDN "
            "on first run)", _MODEL_NAME,
        )
        # Local import keeps module import cost low until the first detect call.
        from ultralytics import YOLO
        _model = YOLO(_MODEL_NAME)
        log.info("YOLO model loaded; %d classes", len(_model.names))
    return _model


def class_names() -> dict[int, str]:
    """Return the model's ``{index: class_name}`` mapping (loads the model)."""
    return dict(_get_model().names)


def _extract_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: dict,
    conf_threshold: float,
    target_labels: Optional[set[str]] = None,
) -> list[Detection]:
    """Convert raw YOLO box arrays into a list of :class:`Detection`.

    Pure function (no torch / no model) so it can be unit-tested directly.

    Parameters
    ----------
    xyxy : np.ndarray, shape (N, 4)
        Boxes as ``(x1, y1, x2, y2)`` in pixels.
    conf : np.ndarray, shape (N,)
        Per-box confidence.
    cls : np.ndarray, shape (N,)
        Per-box class index (float or int).
    names : dict
        ``{index: class_name}`` mapping from the model.
    conf_threshold : float
        Drop boxes with confidence below this.
    target_labels : set[str] | None
        If given, keep only detections whose (lowercased) label is in this set.
    """
    out: list[Detection] = []
    n = int(xyxy.shape[0]) if xyxy.ndim == 2 else 0
    for i in range(n):
        c = float(conf[i])
        if c < conf_threshold:
            continue
        idx = int(cls[i])
        label = str(names.get(idx, str(idx)))
        if target_labels is not None and label.lower() not in target_labels:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        out.append(Detection(label=label, confidence=c, box=(x1, y1, x2, y2)))
    return out


def detect(
    frame_rgb: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    target_labels: Optional[Iterable[str]] = None,
) -> list[Detection]:
    """Run YOLOv8n on a single RGB frame.

    Parameters
    ----------
    frame_rgb : np.ndarray, shape (H, W, 3)
        RGB image (the format ``frame_handler`` stores on the session).
    conf_threshold : float
        Minimum confidence to keep a detection.
    target_labels : Iterable[str] | None
        Optional whitelist of COCO class names to keep (case-insensitive).
        Inference still runs over all classes; this just filters the output.

    Returns
    -------
    list[Detection]
        Possibly empty. Never raises on an empty / no-object frame.
    """
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
        return []

    model = _get_model()
    targets = {t.lower() for t in target_labels} if target_labels is not None else None

    # ultralytics accepts an HxWx3 numpy array directly. verbose=False keeps
    # the per-frame "0: 480x640 1 cup, 12ms" spam out of the logs.
    results = model.predict(frame_rgb, conf=conf_threshold, imgsz=_IMGSZ, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    # Move tensors to CPU numpy. ultralytics Boxes expose xyxy/conf/cls tensors.
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    return _extract_detections(xyxy, conf, cls, model.names, conf_threshold, targets)


def warm_up() -> None:
    """Eagerly load the model (and trigger the weight download) ahead of time.

    Optional - call at server start to move the one-time download + load cost
    off the first user command. Safe to call multiple times.
    """
    _get_model()
