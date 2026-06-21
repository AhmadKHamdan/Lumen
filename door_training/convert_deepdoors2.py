"""Convert DeepDoors2 (detection/segmentation subset) → single-class 'door' YOLO.

DeepDoors2 masks colour door / door-frame pixels (192,224,192) and background
(0,0,0). We threshold on that colour, take the bounding box of each connected
component, normalise it, and write a single-class (0 = door) YOLO label.

Expected input (original gasparramoa/DeepDoors2 detection subset):
    <src>/Images/*       RGB images (480x640)
    <src>/Annotations/*  colour masks, same filename stem
Folder names vary between mirrors — override with --images / --masks if needed.
Filenames are prefixed 'dd2_' so they never collide with DoorDetect when merged.

Run:
    python convert_deepdoors2.py --src /path/to/DeepDoors2/Detection --out converted/deepdoors2
    # or, if folders are named differently:
    python convert_deepdoors2.py --images <imgs> --masks <masks> --out converted/deepdoors2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# DeepDoors2 masks are binary: pure-black background + a single door colour.
# The README claims door == (192,224,192) but the actual Drive release uses the
# PASCAL-VOC class-1 colour (128,0,0). To be robust to either, we treat ANY
# non-(near-)black pixel as door rather than matching a specific colour.
BG_THRESH = 40                    # pixels with max channel <= this are background
MIN_AREA_FRAC = 0.005             # drop boxes smaller than 0.5% of the frame
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def door_binary(mask_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask of door pixels = anything that isn't black background."""
    return (mask_bgr.max(axis=2) > BG_THRESH).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="DeepDoors2 detection subset root (Images/ Annotations/)")
    ap.add_argument("--images", help="images folder (overrides --src/Images)")
    ap.add_argument("--masks", help="masks folder (overrides --src/Annotations)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = Path(args.images) if args.images else Path(args.src) / "Images"
    msk_dir = Path(args.masks) if args.masks else Path(args.src) / "Annotations"
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    n = boxes = skipped = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        mp = masks.get(img.stem)
        if mp is None:
            skipped += 1
            continue
        mask = cv2.imread(str(mp))
        if mask is None:
            skipped += 1
            continue
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(door_binary(mask), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        lines: list[str] = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw * bh < MIN_AREA_FRAC * w * h:
                continue
            cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        shutil.copy(img, out / "images" / ("dd2_" + img.name))
        (out / "labels" / ("dd2_" + img.stem + ".txt")).write_text("\n".join(lines))
        n += 1
        boxes += len(lines)

    print(f"DeepDoors2: {n} images, {boxes} door boxes ({skipped} skipped, no mask) -> {out}")


if __name__ == "__main__":
    main()
