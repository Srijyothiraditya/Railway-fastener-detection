"""
=============================================================
  Railway Track Condition Assessment
  Project : Fastener + Sleeper Detection
  Output  : 3 track condition categories

  CATEGORY 1 — GOOD     : Sleeper present, fasteners on BOTH sides
  CATEGORY 2 — WARNING  : Sleeper present, fastener on ONE side only
  CATEGORY 3 — CRITICAL : Sleeper present, NO fasteners at all

=============================================================

USAGE:
    # Single image:
    python assess_condition.py --source "path\\to\\image.jpg"

    # Entire folder:
    python assess_condition.py --source "path\\to\\folder"

    # With GPU:
    python assess_condition.py --source "path\\to\\image.jpg" --device 0

    # Custom confidence threshold:
    python assess_condition.py --source "path\\to\\image.jpg" --conf 0.3
"""

import argparse
import sys
import os
import json
import csv
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
# Update this path after training completes
MODEL_PATH  = r"runs_v2\fastener_sleeper_v1\weights\best.pt"
CONF        = 0.25
IOU         = 0.5
IMGSZ       = 640

CLASS_FASTENER = 0
CLASS_SLEEPER  = 1

# How far from centre line to consider left vs right (fraction of image width)
# A fastener is "left side" if its centre x < sleeper_centre_x - SIDE_MARGIN * image_width
# A fastener is "right side" if its centre x > sleeper_centre_x + SIDE_MARGIN * image_width
SIDE_MARGIN = 0.05
# ────────────────────────────────────────────────────────────


def get_box_centre(box_xyxy):
    """Return (cx, cy) centre of a bounding box given [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def assess_single_image(result, img_width):
    """
    Analyse detections in one image and return condition category.

    Returns dict with:
        category    : 1, 2, or 3
        label       : human-readable label
        description : explanation
        sleeper_found   : bool
        fasteners_left  : int
        fasteners_right : int
        total_fasteners : int
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "category"       : 0,
            "label"          : "UNKNOWN",
            "description"    : "No detections found in this image",
            "sleeper_found"  : False,
            "fasteners_left" : 0,
            "fasteners_right": 0,
            "total_fasteners": 0,
        }

    # Separate fasteners and sleepers
    fastener_boxes = []
    sleeper_boxes  = []

    for box in boxes:
        cls  = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cx, cy = get_box_centre(xyxy)

        if cls == CLASS_FASTENER:
            fastener_boxes.append({"xyxy": xyxy, "cx": cx, "cy": cy, "conf": conf})
        elif cls == CLASS_SLEEPER:
            sleeper_boxes.append({"xyxy": xyxy, "cx": cx, "cy": cy, "conf": conf})

    # ── No sleeper found ────────────────────────────────────
    if len(sleeper_boxes) == 0:
        return {
            "category"       : 0,
            "label"          : "UNKNOWN",
            "description"    : "No sleeper detected — cannot assess track condition",
            "sleeper_found"  : False,
            "fasteners_left" : len(fastener_boxes),
            "fasteners_right": 0,
            "total_fasteners": len(fastener_boxes),
        }

    # Use the highest-confidence sleeper as reference
    sleeper_boxes.sort(key=lambda b: b["conf"], reverse=True)
    sleeper = sleeper_boxes[0]
    sleeper_cx = sleeper["cx"]

    margin = img_width * SIDE_MARGIN

    # Count fasteners on left and right of sleeper centre
    left_fasteners  = [f for f in fastener_boxes if f["cx"] < sleeper_cx - margin]
    right_fasteners = [f for f in fastener_boxes if f["cx"] > sleeper_cx + margin]

    n_left  = len(left_fasteners)
    n_right = len(right_fasteners)
    n_total = len(fastener_boxes)

    has_left  = n_left  > 0
    has_right = n_right > 0

    # ── Determine category ───────────────────────────────────
    if has_left and has_right:
        category    = 1
        label       = "GOOD"
        description = (f"Sleeper present. Fasteners on BOTH sides "
                       f"(left: {n_left}, right: {n_right}). Track is secure.")

    elif has_left or has_right:
        side        = "LEFT" if has_left else "RIGHT"
        missing     = "RIGHT" if has_left else "LEFT"
        count       = n_left if has_left else n_right
        category    = 2
        label       = "WARNING"
        description = (f"Sleeper present. Fastener(s) on {side} side only ({count} found). "
                       f"{missing} side fastener is MISSING. Inspection required.")

    else:
        category    = 3
        label       = "CRITICAL"
        description = (f"Sleeper present but NO fasteners detected on either side. "
                       f"IMMEDIATE inspection required.")

    return {
        "category"       : category,
        "label"          : label,
        "description"    : description,
        "sleeper_found"  : True,
        "fasteners_left" : n_left,
        "fasteners_right": n_right,
        "total_fasteners": n_total,
        "sleeper_cx"     : round(sleeper_cx, 1),
    }


def print_result(image_name, assessment, show_separator=True):
    """Print a clean formatted result for one image."""
    cat  = assessment["category"]
    lbl  = assessment["label"]
    desc = assessment["description"]

    # Colour codes for terminal
    colors = {0: "\033[90m", 1: "\033[92m", 2: "\033[93m", 3: "\033[91m"}
    reset  = "\033[0m"
    color  = colors.get(cat, "")

    icons = {0: "❓", 1: "✅", 2: "⚠️ ", 3: "🚨"}
    icon  = icons.get(cat, "")

    print(f"\n  Image    : {image_name}")
    print(f"  {icon}  Category {cat} — {color}{lbl}{reset}")
    print(f"  Detail   : {desc}")
    if assessment["sleeper_found"]:
        print(f"  Counts   : Left fasteners={assessment['fasteners_left']}  "
              f"Right fasteners={assessment['fasteners_right']}  "
              f"Total={assessment['total_fasteners']}")
    if show_separator:
        print("  " + "─"*56)


def run(source, device="cpu", conf=CONF, save_images=True):
    from ultralytics import YOLO

    # Resolve model path
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        # Try to find it automatically
        candidates = sorted(Path(".").glob("runs_v2/**/weights/best.pt"))
        if candidates:
            model_path = candidates[-1]
            print(f"✅ Found model: {model_path}")
        else:
            print(f"❌ Model not found at {MODEL_PATH}")
            print("   Run train_v2.py first to train the model.")
            sys.exit(1)

    model = YOLO(str(model_path))

    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ Source not found: {source}")
        sys.exit(1)

    # Run prediction
    print(f"\n{'='*60}")
    print(f"  Railway Track Condition Assessment")
    print(f"{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Source : {source}")
    print(f"  Conf   : {conf}")
    print(f"{'─'*60}")

    results = model.predict(
        source      = str(source_path),
        conf        = conf,
        iou         = IOU,
        imgsz       = IMGSZ,
        device      = device,
        save        = save_images,
        save_txt    = True,
        show_labels = True,
        show_conf   = True,
        project     = "runs_v2",
        name        = "condition_assessment",
        verbose     = False,
    )

    # ── Assess each image ────────────────────────────────────
    all_assessments = []
    summary = {1: 0, 2: 0, 3: 0, 0: 0}

    for r in results:
        img_name   = Path(r.path).name
        img_width  = r.orig_shape[1]   # original image width in pixels
        assessment = assess_single_image(r, img_width)
        assessment["image"] = img_name

        print_result(img_name, assessment)
        all_assessments.append(assessment)
        summary[assessment["category"]] += 1

    # ── Summary ──────────────────────────────────────────────
    total = len(all_assessments)
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {total} image(s) assessed")
    print(f"{'─'*60}")
    print(f"  ✅  Category 1 — GOOD     : {summary[1]:3d} images  ({100*summary[1]//max(total,1)}%)")
    print(f"  ⚠️   Category 2 — WARNING  : {summary[2]:3d} images  ({100*summary[2]//max(total,1)}%)")
    print(f"  🚨  Category 3 — CRITICAL : {summary[3]:3d} images  ({100*summary[3]//max(total,1)}%)")
    if summary[0]:
        print(f"  ❓  Category 0 — UNKNOWN  : {summary[0]:3d} images  (no sleeper detected)")
    print(f"{'='*60}\n")

    # ── Save results to CSV ──────────────────────────────────
    csv_path = Path("runs_v2/condition_assessment/assessment_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "category", "label", "description",
            "sleeper_found", "fasteners_left", "fasteners_right", "total_fasteners"
        ])
        writer.writeheader()
        for a in all_assessments:
            writer.writerow({k: a.get(k, "") for k in [
                "image", "category", "label", "description",
                "sleeper_found", "fasteners_left", "fasteners_right", "total_fasteners"
            ]})

    print(f"  📄 Results CSV saved to: {csv_path}")
    if save_images:
        print(f"  🖼️  Output images saved to: runs_v2/condition_assessment/")

    return all_assessments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Railway Track Condition Assessment")
    parser.add_argument("--source", required=True,
                        help="Path to image or folder of images")
    parser.add_argument("--device", default="cpu",
                        help="Device: 'cpu' or '0' for GPU")
    parser.add_argument("--conf", type=float, default=CONF,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save output images")
    args = parser.parse_args()

    run(
        source      = args.source,
        device      = args.device,
        conf        = args.conf,
        save_images = not args.no_save,
    )
