# Door Detection Model — Training & Results

## Objective
Train a single-class **`door`** detector to fill the one architectural gap in the
navigation pipeline: the base YOLOv8n (COCO) has no door class. The trained model
adds door detection so the system can use doorways as navigation targets.

## Datasets
Two public door datasets, merged and reduced to a single `door` class:

| Dataset | Contribution | Notes |
|---|---|---|
| **DoorDetect** (MiguelARD) | 386 door images + 827 hard-negatives | Originally 4 classes; kept only `door`. Negatives (handles/cabinets) added to reduce false positives. Diverse human/web viewpoints. |
| **DeepDoors2** (gasparramoa) | 3,000 images, 3,106 door boxes | Detection/segmentation subset. Segmentation masks converted to bounding boxes. Real-world blur/occlusion. |

**Merged dataset:** 4,213 images, single class `door`, split **3,371 train / 842 val** (80/20).

Mask-to-box note: DeepDoors2 masks were converted by treating any non-black pixel as
door (the masks use colour `(128,0,0)`, not the `(192,224,192)` stated in its README).

## Training setup
- **Model:** YOLOv8n, fine-tuned from COCO-pretrained weights (transfer learning; 319/355 layers transferred).
- **Image size:** 640 · **Batch:** 8 · **Optimizer:** AdamW (auto) · **AMP:** on.
- **Epochs:** best checkpoint selected at **epoch 39**.
- **Classes:** 1 (`door`).

## Results (best checkpoint)

| Metric | Value |
|---|---|
| **mAP@50** | **0.946** |
| **mAP@50-95** | **0.829** |
| **Precision** | **0.968** |
| **Recall** | **0.892** |

For reference, shorter runs scored lower (8 epochs → mAP50 0.86; 20 epochs → mAP50 0.93),
confirming the ~40-epoch run as the best of the set.

## Interpretation
- **mAP@50 0.95 / recall 0.89** is a strong single-class result — the model reliably
  localises doors and finds ~89% of them.
- **Precision 0.97** means very few false doors, helped by the hard-negative images.
- This comfortably exceeds the ≥0.6 mAP@50 target set for the task.

## Output
- Trained weights: **`best.pt`** — the deployable door detector.
- Integrated into the navigation pipeline as a conditionally-loaded model (runs only
  when the active target is a door).
