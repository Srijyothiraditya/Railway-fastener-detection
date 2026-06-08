"""
=============================================================
  YOLOv8 Segmentation Training Script — Version 2
  Project : Railway Fastener + Sleeper Detection
  Classes : Fastener (0), Sleeper (1)
  Dataset : 984 train / 41 val / 41 test
=============================================================

USAGE:
    python train_v2.py                     # CPU training
    python train_v2.py --device 0          # GPU training (recommended)
    python train_v2.py --resume            # Resume from last checkpoint
"""

import os
import sys
import argparse
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
DATASET_DIR = Path("dataset2")
RUNS_DIR    = Path("runs_v2")
RUN_NAME    = "fastener_sleeper_v1"
EPOCHS      = 100
IMGSZ       = 640
BATCH       = 8         # reduce to 4 if you get memory errors
PATIENCE    = 20
LR0         = 0.01
LRF         = 0.001
# ────────────────────────────────────────────────────────────


def write_data_yaml(dataset_dir: Path):
    yaml_content = f"""path: {dataset_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Fastener', 'Sleeper']
"""
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✅ data.yaml written")
    return yaml_path


def train(device="cpu", resume=False):
    from ultralytics import YOLO

    dataset_dir = DATASET_DIR.resolve()
    yaml_path   = write_data_yaml(dataset_dir)

    train_imgs = list((dataset_dir / "train" / "images").glob("*.jpg"))
    val_imgs   = list((dataset_dir / "valid" / "images").glob("*.jpg"))
    test_imgs  = list((dataset_dir / "test"  / "images").glob("*.jpg"))

    print("\n" + "="*60)
    print("  Railway Track Condition — YOLOv8-seg Training v2")
    print("="*60)
    print(f"  Classes  : Fastener (0), Sleeper (1)")
    print(f"  Train    : {len(train_imgs)} images")
    print(f"  Val      : {len(val_imgs)} images")
    print(f"  Test     : {len(test_imgs)} images")
    print(f"  Epochs   : {EPOCHS}  |  Batch: {BATCH}  |  Device: {device}")
    print("="*60 + "\n")

    # Load model — try pretrained weights first
    try:
        model = YOLO("yolov8n-seg.pt")
        print("✅ Loaded pretrained yolov8n-seg.pt")
    except Exception:
        model = YOLO("yolov8n-seg.yaml")
        print("⚠️  Training from scratch (no pretrained weights)")

    results = model.train(
        data         = str(yaml_path),
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        device       = device,
        name         = RUN_NAME,
        project      = str(RUNS_DIR),
        patience     = PATIENCE,
        lr0          = LR0,
        lrf          = LRF,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        mosaic       = 0.5,
        degrees      = 15.0,
        fliplr       = 0.5,
        hsv_h        = 0.015,
        hsv_s        = 0.4,
        hsv_v        = 0.4,
        scale        = 0.3,
        translate    = 0.1,
        save         = True,
        plots        = True,
        workers      = 4 if device != "cpu" else 2,
        resume       = resume,
        verbose      = True,
    )

    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Best model  : {best_pt}")
    print(f"  Results dir : {save_dir}")
    print(f"{'='*60}\n")

    # Validate on test set
    print("Running validation on test split...")
    best_model   = YOLO(str(best_pt))
    test_results = best_model.val(
        data   = str(yaml_path),
        split  = "test",
        imgsz  = IMGSZ,
        device = device,
        plots  = True,
    )

    box  = test_results.box
    mask = test_results.seg

    print(f"\n{'─'*60}")
    print("  TEST RESULTS (best.pt)")
    print(f"{'─'*60}")
    print(f"  Box  mAP@50     : {box.map50:.4f}")
    print(f"  Box  mAP@50-95  : {box.map:.4f}")
    print(f"  Mask mAP@50     : {mask.map50:.4f}")
    print(f"  Mask mAP@50-95  : {mask.map:.4f}")
    print(f"{'─'*60}\n")

    # Export to ONNX
    try:
        print("Exporting to ONNX...")
        best_model.export(format="onnx", imgsz=IMGSZ, simplify=True)
        print(f"✅ ONNX saved next to best.pt")
    except Exception as e:
        print(f"⚠️  ONNX export skipped: {e}")

    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(device=args.device, resume=args.resume)
