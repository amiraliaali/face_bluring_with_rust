# Face Anonymiation with Rust

## Demo Video
Here is a demo video.

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/amiraliaali/face_bluring_with_rust/blob/main/demo_video.gif" width="900" height="500" />
</div>

The video demonstrates the pipeline detecting faces in a video and anonymizing them by overlaying black rectangles.

## Overview
This project implements a real-time face anonymization pipeline using Rust. The workflow consists of:
1. Model Export:
We start with a YOLOv8 face detector implemented in Python and export it to the ONNX format. This allows the model to be used outside Python in a Rust environment with ONNX Runtime.

2. Rust Pipeline:
Using the ONNX model, the Rust pipeline:
- Reads a video frame by frame using OpenCV.
- Detects faces using the YOLOv8 ONNX model.
- Applies Non-Maximum Suppression (NMS) to remove overlapping detections.
- Anonymizes detected faces by drawing filled black rectangles over them.
- Writes the processed frames back into a new video.

The entire pipeline is optimized for speed and efficiency, using multithreading and ONNX runtime graph optimizations.

## How it Works
1- Loading the ONNX Model:
Rust loads the YOLOv8 face detector from yolov8n-face.onnx using onnxruntime. The session is configured with optimizations and multithreading for faster inference.
2- Reading the Video:
OpenCV reads the input video frame by frame. Each frame is resized to 640x640 for model compatibility, and pixel values are normalized to [0,1].
3- Running Inference:
The frame is converted into a 4D tensor and passed through the YOLOv8 ONNX model. The output is a set of predicted bounding boxes and confidence scores for detected faces.
4- Non-Maximum Suppression (NMS):
To avoid multiple overlapping boxes on the same face, NMS filters predictions based on Intersection over Union (IoU) thresholds.
5- Anonymizing Faces:
For each detected face, a black rectangle is drawn using OpenCV. The rectangle completely covers the face, preserving privacy.
6- Writing the Video:
The processed frames are saved into a new video with the same resolution as the original.

## Project Structure
```
📂 project_root
├── src/
  ├── model_extraction
    ├── model_extraction.py        # A simple script to extract yolo v8 face detector to its ONNX
    ├── yolov8n-face.onnx        # the extracted onnx
    └── yolov8n-face.pt        # The weights of the model downloaded from https://github.com/akanametov/yolo-face
  └── pipeline        # All the rust files
└── setup.txt        # For building onnxruntime binaries from scratch

```