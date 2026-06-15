"""Convert DoorDetect → 4-class YOLO labels (door / handle / cabinet door / refrigerator door).

DoorDetect ships YOLO/Darknet labels with 4 classes (obj.names):
    0 door   1 handle   2 cabinet door   3 refrigerator door
We KEEP ALL FOUR (ids unchanged). Teaching the model 'refrigerator door' and
'cabinet door' explicitly is what stops fridges/cupboards being mistaken for a
navigable door; 'handle' is a corroboration signal. In the app, only class 0
(door) is treated as a navigable door.

This step is model-size-agnostic (it only makes labels). The model (yolov8s/n/m)
is chosen later in the training command.

Input:   <src>/images/*  and  <src>/labels/*.txt
Output:  <out>/images/dd_*  and  <out>/labels/dd_*.txt   (filenames prefixed
         'dd_' so they never collide with other sources when merged).

Images with no boxes become negative samples (empty label file) — kept only with
--keep-negatives. A few negatives help suppress false positives.

Run:
    python convert_doordetect.py --src "C:/Users/OHussain/Downloads/DoorDetect-Dataset" --out converted/doordetect
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# obj.names order is preserved as the class ids.
CLASS_NAMES = ["door", "handle", "cabinet door", "refrigerator door"]
KEEP_CLASS_IDS = {0, 1, 2, 3}  # keep all four
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _winlong(p) -> str:
    """Absolute path with the Windows \\\\?\\ prefix so >260-char paths work."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DoorDetect-Dataset root (has images/ labels/)")
    ap.add_argument("--out", required=True, help="output dir (images/ labels/ created inside)")
    ap.add_argument("--keep-negatives", action="store_true",
                    help="also copy images with no labelled boxes (empty labels)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir, lbl_dir = src / "images", src / "labels"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    kept = neg = skipped = 0
    idx = 0
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        lines: list[str] = []
        try:
            content = Path(_winlong(lbl)).read_text()
        except OSError:
            content = ""  # no/unreadable label -> negative sample
        for line in content.splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if cid in KEEP_CLASS_IDS:
                lines.append(f"{cid} " + " ".join(parts[1:]))  # ids unchanged
        if not lines and not args.keep_negatives:
            continue
        name = f"dd_{idx:05d}"  # short sequential name (originals can exceed MAX_PATH)
        try:
            shutil.copy(_winlong(img), _winlong(out / "images" / (name + img.suffix)))
        except OSError:
            skipped += 1
            continue
        (out / "labels" / (name + ".txt")).write_text("\n".join(lines))
        idx += 1
        kept += 1 if lines else 0
        neg += 0 if lines else 1

    print(f"DoorDetect (4-class): {kept} labelled images + {neg} negatives "
          f"({skipped} unreadable skipped) -> {out}")
    print(f"classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
