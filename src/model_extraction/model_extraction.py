from ultralytics import YOLO

model = YOLO("src/model_extraction/yolov8n-face.pt")
model.export(format="onnx", opset=13)