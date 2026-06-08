from ultralytics import YOLO
import time

# Load best.pt from Drive and export to ONNX
model = YOLO('/content/drive/MyDrive/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print("ONNX exported")


#runtime

import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path

# Load ONNX model
onnx_path = '/content/drive/MyDrive/best.onnx'
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print(f"Provider being used: {session.get_providers()}")
print(f"Input name: {session.get_inputs()[0].name}")
print(f"Input shape: {session.get_inputs()[0].shape}")
print()

# Create a dummy image (random pixels, same as 640x640 track image)
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
input_name = session.get_inputs()[0].name

# Warmup run (first run is always slower)
session.run(None, {input_name: dummy_input})
print("Warmup done")

# Time 20 inference runs
times = []
for i in range(20):
    start = time.time()
    outputs = session.run(None, {input_name: dummy_input})
    end = time.time()
    times.append((end - start) * 1000)  # convert to milliseconds

avg   = np.mean(times)
mn    = np.min(times)
mx    = np.max(times)
fps   = 1000 / avg

print(f"Results over 20 runs:")
print(f"  Average : {avg:.2f} ms")
print(f"  Fastest : {mn:.2f} ms")
print(f"  Slowest : {mx:.2f} ms")
print(f"  FPS     : {fps:.1f} frames per second")
print()

# Now test on a real image
import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path
from IPython.display import Image as IPImage, display
import os

ONNX_PATH = '/content/drive/MyDrive/best.onnx'
TEST_DIR  = '/content/output_v2/dataset2/test/images'
OUT_DIR   = '/content/onnx_test_results'
CONF      = 0.25
CLASSES   = ['Fastener', 'Sleeper']
COLORS    = {'Fastener': (0, 255, 0), 'Sleeper': (255, 165, 0)}

os.makedirs(OUT_DIR, exist_ok=True)

session    = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print(f"Provider: {session.get_providers()[0]}\n")

test_imgs = list(Path(TEST_DIR).glob('*.jpg'))
print(f"Total test images: {len(test_imgs)}\n")

def run_inference(image_path):
    orig_img       = cv2.imread(str(image_path))
    orig_h, orig_w = orig_img.shape[:2]
    scale_x        = orig_w / 640
    scale_y        = orig_h / 640

    img = cv2.resize(orig_img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0

    start   = time.time()
    outputs = session.run(None, {input_name: img})
    elapsed = (time.time() - start) * 1000

    preds     = outputs[0][0].T
    boxes_out = []
    for pred in preds:
        x, y, w, h   = pred[0], pred[1], pred[2], pred[3]
        class_scores = pred[4:4+len(CLASSES)]
        cls_id       = int(np.argmax(class_scores))
        conf         = float(class_scores[cls_id])
        if conf < CONF:
            continue
        x1 = max(0, int((x - w/2) * scale_x))
        y1 = max(0, int((y - h/2) * scale_y))
        x2 = min(orig_w, int((x + w/2) * scale_x))
        y2 = min(orig_h, int((y + h/2) * scale_y))
        boxes_out.append((cls_id, conf, x1, y1, x2, y2))

    # NMS
    boxes_out = sorted(boxes_out, key=lambda x: x[1], reverse=True)
    kept = []
    while boxes_out:
        best = boxes_out.pop(0)
        kept.append(best)
        def iou(a, b):
            ix1 = max(a[2], b[2]); iy1 = max(a[3], b[3])
            ix2 = min(a[4], b[4]); iy2 = min(a[5], b[5])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = (a[4]-a[2])*(a[5]-a[3]) + (b[4]-b[2])*(b[5]-b[3]) - inter
            return inter/union if union > 0 else 0
        boxes_out = [b for b in boxes_out if b[0] != best[0] or iou(best, b) < 0.5]

    result_img = orig_img.copy()
    for cls_id, conf, x1, y1, x2, y2 in kept:
        cls_name = CLASSES[cls_id]
        color    = COLORS[cls_name]
        label    = f"{cls_name} {conf:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(result_img, (x1, y1-28), (x1+len(label)*11, y1), color, -1)
        cv2.putText(result_img, label, (x1+2, y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return result_img, kept, elapsed

# Run on all test images
all_times   = []
total_fast  = 0
total_sleep = 0

for i, img_path in enumerate(test_imgs):
    result_img, detections, ms = run_inference(img_path)

    out_path = f"{OUT_DIR}/{img_path.name}"
    cv2.imwrite(out_path, result_img)

    n_fast  = sum(1 for d in detections if d[0] == 0)
    n_sleep = sum(1 for d in detections if d[0] == 1)
    total_fast  += n_fast
    total_sleep += n_sleep
    all_times.append(ms)

    img_cx  = result_img.shape[1] / 2
dead    = result_img.shape[1] * 0.01
left_f  = sum(1 for d in detections if d[0]==0 and (d[2]+d[4])/2 < img_cx - dead)
right_f = sum(1 for d in detections if d[0]==0 and (d[2]+d[4])/2 > img_cx + dead)
has_sleep = n_sleep > 0

if has_sleep and left_f > 0 and right_f > 0:
    cat = "✅ GOOD"
elif has_sleep and (left_f > 0 or right_f > 0):
    cat = "⚠️  WARNING"
elif has_sleep:
    cat = "🚨 CRITICAL"
else:
    cat = "❓ UNKNOWN"

print(f"[{i+1:2d}/41] {img_path.name[:40]}  |  {ms:6.1f}ms  |  F:{n_fast} S:{n_sleep}  |  {cat}")

print()
print("="*60)
print(f"  SUMMARY")
print("="*60)
print(f"  Total images     : {len(test_imgs)}")
print(f"  Avg time         : {np.mean(all_times):.2f} ms")
print(f"  Fastest          : {np.min(all_times):.2f} ms")
print(f"  Slowest          : {np.max(all_times):.2f} ms")
print(f"  Avg FPS          : {1000/np.mean(all_times):.1f}")
print(f"  Total Fasteners  : {total_fast}")
print(f"  Total Sleepers   : {total_sleep}")
print(f"  Output images    : {OUT_DIR}")
print("="*60)

# Show first 3 results
print("\nSample results:")
for img_path in list(Path(OUT_DIR).glob('*.jpg'))[:3]:
    print(f"\n{img_path.name}")
    display(IPImage(str(img_path)))

