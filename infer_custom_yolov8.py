"""infer_custom_yolov8.py

Load a fine-tuned YOLOv8 model (best.pt) and run inference.

Usage (example):
    python infer_custom_yolov8.py \
        --weights path/to/runs/detect/vehicle_finetune/weights/best.pt \
        --source uploads/example.jpg

This script shows how to:
- load the fine-tuned `best.pt`
- run prediction
- save annotated images to an output folder

Notes:
- If you pass a video/webcam source, YOLO will handle it.
- For basic class names, Ultralytics uses `model.names` from the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with fine-tuned YOLOv8")
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to your trained best.pt",
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image/video path, or webcam index (e.g. 0)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results to disk.",
    )
    p.add_argument(
        "--project",
        type=str,
        default="outputs_custom_infer",
        help="Output directory for saved predictions.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Subfolder name under project.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    model = YOLO(str(weights_path))  # <-- loads best.pt instead of yolov8n.pt

    # Run prediction
    # If --save is set, Ultralytics will save annotated outputs.
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        project=args.project if args.save else None,
        name=args.name if args.save else None,
        verbose=False,
    )

    # Print a simple summary for the first image/frame
    # (For videos, `results` contains a list per frame.)
    if not results:
        print("[INFO] No results returned.")
        return

    r0 = results[0]
    # boxes: xyxy, conf, cls
    if r0.boxes is not None and len(r0.boxes) > 0:
        names = model.names  # class idx -> name
        for box in r0.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {names.get(cls_id, cls_id)} conf={conf:.3f}")
    else:
        print("[INFO] No detections.")


if __name__ == "__main__":
    main()

