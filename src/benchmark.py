"""
Benchmark utility — measures pure YOLOv8 inference FPS on a given device,
separate from the full tracking pipeline. Useful for reporting the
"30 FPS GPU-accelerated inference" claim with real numbers.

Usage:
    python src/benchmark.py --source path/to/video.mp4 --model yolov8n.pt
"""

import argparse
import time

import cv2
import torch
from ultralytics import YOLO


def benchmark(source: str, model_path: str, num_frames: int = 150):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Benchmarking on device: {device}")

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Warm-up (first inference includes model/CUDA init overhead)
    ret, frame = cap.read()
    if ret:
        model.predict(frame, device=device, verbose=False)

    frame_times = []
    count = 0

    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        model.predict(frame, device=device, verbose=False)
        t1 = time.time()

        frame_times.append(t1 - t0)
        count += 1

    cap.release()

    if not frame_times:
        print("[WARN] No frames processed.")
        return

    avg_time = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print(f"[RESULT] Frames benchmarked : {len(frame_times)}")
    print(f"[RESULT] Avg inference time : {avg_time * 1000:.2f} ms")
    print(f"[RESULT] Avg FPS            : {fps:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 inference FPS benchmark")
    parser.add_argument("--source", type=str, required=True,
                         help="Webcam index or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--frames", type=int, default=150,
                         help="Number of frames to benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.source, args.model, args.frames)
