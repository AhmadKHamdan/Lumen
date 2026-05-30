"""
Unit tests for yolo_service's pure helpers.

We deliberately do NOT load the YOLO model or import ultralytics/torch here.
The model-dependent ``detect()`` is exercised manually on hardware (the
blindfolded find-a-cup trial). What we can test in isolation is
``_extract_detections`` - the pure box-array -> Detection conversion - and the
``Detection`` geometry properties.

Run with::

    cd backend && python -m pytest tests/test_yolo_service.py
"""
from __future__ import annotations

import numpy as np
import pytest

from services.yolo_service import Detection, _extract_detections


# COCO-like index -> name map (subset).
NAMES = {0: "person", 41: "cup", 67: "cell phone", 39: "bottle"}


def test_detection_geometry_properties():
    d = Detection(label="cup", confidence=0.9, box=(10, 20, 30, 60))
    assert d.center_x == pytest.approx(20.0)
    assert d.center_y == pytest.approx(40.0)
    assert d.area == pytest.approx(20 * 40)


def test_extract_basic_two_boxes():
    xyxy = np.array([[0, 0, 10, 10], [20, 20, 40, 50]], dtype=float)
    conf = np.array([0.9, 0.8], dtype=float)
    cls = np.array([0, 41], dtype=float)
    dets = _extract_detections(xyxy, conf, cls, NAMES, conf_threshold=0.35)
    assert [d.label for d in dets] == ["person", "cup"]
    assert dets[1].box == (20.0, 20.0, 40.0, 50.0)


def test_extract_filters_by_confidence():
    xyxy = np.array([[0, 0, 10, 10], [20, 20, 40, 50]], dtype=float)
    conf = np.array([0.9, 0.10], dtype=float)
    cls = np.array([0, 41], dtype=float)
    dets = _extract_detections(xyxy, conf, cls, NAMES, conf_threshold=0.35)
    assert len(dets) == 1
    assert dets[0].label == "person"


def test_extract_filters_by_target_label():
    xyxy = np.array([[0, 0, 10, 10], [20, 20, 40, 50]], dtype=float)
    conf = np.array([0.9, 0.8], dtype=float)
    cls = np.array([0, 41], dtype=float)
    dets = _extract_detections(
        xyxy, conf, cls, NAMES, conf_threshold=0.35, target_labels={"cup"},
    )
    assert len(dets) == 1
    assert dets[0].label == "cup"


def test_extract_target_label_is_case_insensitive():
    xyxy = np.array([[20, 20, 40, 50]], dtype=float)
    conf = np.array([0.8], dtype=float)
    cls = np.array([67], dtype=float)
    dets = _extract_detections(
        xyxy, conf, cls, NAMES, conf_threshold=0.35, target_labels={"cell phone"},
    )
    assert len(dets) == 1
    assert dets[0].label == "cell phone"


def test_extract_empty_input():
    xyxy = np.empty((0, 4), dtype=float)
    conf = np.empty((0,), dtype=float)
    cls = np.empty((0,), dtype=float)
    dets = _extract_detections(xyxy, conf, cls, NAMES, conf_threshold=0.35)
    assert dets == []


def test_extract_unknown_class_index_falls_back_to_string():
    xyxy = np.array([[0, 0, 5, 5]], dtype=float)
    conf = np.array([0.9], dtype=float)
    cls = np.array([999], dtype=float)
    dets = _extract_detections(xyxy, conf, cls, NAMES, conf_threshold=0.35)
    assert dets[0].label == "999"
