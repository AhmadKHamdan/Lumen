"""M1 verification on a still photo — no live camera needed.

Snap a photo with the Windows Camera app (reliable), then run the FULL M1
pipeline on it: YOLOv8n (COCO) detection -> navigation.goal_map arrival check.
This proves semantic arrival works on a real image without fighting OpenCV's
flaky live-camera access on this laptop.

Usage:
    python scripts/m1_image_test.py "C:\\path\\to\\photo.jpg" --goal kitchen

Tip: Windows Camera app saves to  C:\\Users\\<you>\\Pictures\\Camera Roll\\
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2  # noqa: E402

from navigation import (  # noqa: E402
    resolve_goal,
    indicators_for,
    evaluate_arrival,
    arrival_phrase,
)

CONF_THRESHOLD = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a photo (jpg/png)")
    ap.add_argument("--goal", default="kitchen")
    ap.add_argument("--show", action="store_true", help="display the annotated image")
    args = ap.parse_args()

    goal = resolve_goal(args.goal)
    if goal is None:
        print(f"Unknown goal {args.goal!r}.")
        return
    primary, secondary = indicators_for(goal)
    indicator_set = set(primary) | set(secondary)

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    print(f"Goal: {goal!r}  (primary={primary}, secondary={secondary})\n")
    print("Loading YOLOv8n (COCO)...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    names = model.names

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    res = model.predict(img, verbose=False, conf=CONF_THRESHOLD)[0]
    detected = {}
    for box in res.boxes:
        cls = names[int(box.cls[0])]
        conf = float(box.conf[0])
        detected[cls] = max(detected.get(cls, 0.0), conf)
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        is_ind = cls in indicator_set
        color = (0, 200, 0) if is_ind else (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if is_ind else 1)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if is_ind else 1)

    print("Detected objects:")
    for c, cf in sorted(detected.items(), key=lambda kv: -kv[1]):
        tag = "  <-- indicator" if c in indicator_set else ""
        print(f"  {c:16} {cf:.2f}{tag}")

    result = evaluate_arrival(goal, detected.keys())
    print()
    if result["arrived"]:
        print("ARRIVAL:", arrival_phrase(goal, result))
    else:
        print(f"NOT ARRIVED — matched primary={result['matched_primary']}, "
              f"secondary={result['matched_secondary']} "
              f"(need >=1 primary or >=2 secondary). Keep exploring.")

    out = img_path.with_name(img_path.stem + "_m1.jpg")
    cv2.imwrite(str(out), img)
    print(f"\nAnnotated image saved: {out}")
    if args.show:
        cv2.imshow("M1 image test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
