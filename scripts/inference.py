from ultralytics import YOLO
import time

# Load best.pt from Drive and export to ONNX
model = YOLO('/content/drive/MyDrive/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print("ONNX exported")
