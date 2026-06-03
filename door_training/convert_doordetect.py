"""Convert DoorDetect → single-class 'door' YOLO labels.

DoorDetect ships YOLO/Darknet labels with 4 classes (obj.names):
    0 door   1 handle   2 cabinet door   3 refrigerator door
We keep ONLY class 0 (door) and remap it to class 0 of a single-class dataset.

Input:   <src>/images/*  and  <src>/labels/*.txt
Output:  <out>/images/dd_*  and  <out>/labels/dd_*.txt   (filenames prefixed
         'dd_' so they never collide with DeepDoors2 when merged).

Images with no door box become negative samples (empty label file) — kept only
with --keep-negatives. A few negatives help suppress false positives.

Run:
    python convert_doordetect.py --src ../DoorDetect-Dataset --out converted/doordetect
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _winlong(p) -> str:
    """Absolute path with the Windows \\\\?\\ prefix so >260-char paths work."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s

DOOR_CLASS_IDS = {0}  # obj.names: door == 0
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DoorDetect-Dataset root (has images/ labels/)")
    ap.add_argument("--out", required=True, help="output dir (images/ labels/ created inside)")
    ap.add_argument("--keep-negatives", action="store_true",
                    help="also copy images with no door (empty labels)")
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
            if parts and int(float(parts[0])) in DOOR_CLASS_IDS:
                lines.append("0 " + " ".join(parts[1:]))
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

    print(f"DoorDetect: {kept} door images + {neg} negatives "
          f"({skipped} unreadable skipped) -> {out}")


if __name__ == "__main__":
    main()
