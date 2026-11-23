# Detector for the browser version

yolo26s-736x1280.onnx is YOLO26s by Ultralytics, exported to ONNX for the page in hf/video-space/, which runs at https://huggingface.co/spaces/amreleraqi09/possession-from-video.

- Input: images, one 736x1280 RGB picture, letterboxed with grey (114) and scaled to 0-1.
- Output: up to 300 detections, each x1, y1, x2, y2, score, class, in input pixels. The page keeps COCO classes 0 (person) and 32 (sports ball).

The workflow .github/workflows/export-detector.yml makes the file with Ultralytics 8.4.216:

    YOLO("yolo26s.pt").export(format="onnx", imgsz=(736, 1280), opset=17, simplify=True, nms=False)

To make it again, run that workflow from the Actions tab.

The weights belong to Ultralytics and are licensed under AGPL-3.0, not under this repository's MIT license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
