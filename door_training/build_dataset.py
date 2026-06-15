"""Merge converted door datasets → final YOLO dataset with a train/val split.

Pools image+label pairs from one or more converted dirs (each with images/ and
labels/), shuffles deterministically, splits, and writes data.yaml.

Run (after the two convert_*.py steps):
    python build_dataset.py --inputs converted/doordetect converted/deepdoors2 \
                            --out doors_dataset --val-frac 0.2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="converted dirs, each containing images/ and labels/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for d in args.inputs:
        d = Path(d)
        for img in (d / "images").iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = d / "labels" / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))

    random.Random(args.seed).shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        shutil.copy(img, out / "images" / split / img.name)
        shutil.copy(lbl, out / "labels" / split / (img.stem + ".txt"))

    (out / "data.yaml").write_text(
        "# IMPORTANT: after moving this folder to another machine, "
        "change 'path:' to its new absolute location.\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 4\n"
        f"names: ['door', 'handle', 'cabinet door', 'refrigerator door']\n"
    )

    print(f"Merged {len(pairs)} pairs -> {out}  (train={len(pairs) - n_val}, val={n_val})")
    print("Now train (4 GB GPU -> yolov8s, batch 8; drop to batch 4 if it OOMs):")
    print(f"  yolo detect train model=yolov8s.pt data={out / 'data.yaml'} "
          f"epochs=100 imgsz=640 batch=8 device=0 patience=20 name=door_yolov8s_4cls")


if __name__ == "__main__":
    main()
