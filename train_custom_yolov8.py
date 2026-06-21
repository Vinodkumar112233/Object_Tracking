"""train_custom_yolov8.py

Fine-tune a YOLOv8 model on a custom YOLO-format dataset (images/ + labels/).

Usage (example):
    python train_custom_yolov8.py \
        --data path/to/data.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --project outputs_custom \
        --run_name vehicle_custom

Expected dataset structure (YOLO):
    dataset_root/
      images/
        train/*.jpg
        val/*.jpg
        test/*.jpg        (optional)
      labels/
        train/*.txt
        val/*.txt
        test/*.txt        (optional)

Your `data.yaml` must point to these folders.

Notes:
- This script uses transfer learning by starting from `yolov8n.pt`.
- Ultralytics automatically saves checkpoints under:
    runs/detect/<run_name>/...
  and the best weights are available as:
    runs/detect/<run_name>/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 for custom classes")

    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to your data.yaml (YOLO dataset definition).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Starting weights (e.g., yolov8n.pt). Transfer learning is used.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (increase for more data).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (imgsz x imgsz).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (reduce if you run out of GPU memory).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader worker processes.",
    )
    p.add_argument(
        "--project",
        type=str,
        default="outputs_custom",
        help="Ultralytics project directory (runs will be created under it).",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default="vehicle_finetune",
        help="Name of the training run (subfolder).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string for Ultralytics (e.g. '0' for GPU0, 'cpu'). Leave empty for auto.",
    )
    p.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    p.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor.",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Weight decay (optimizer regularization).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    # Load YOLO model starting point (transfer learning)
    model = YOLO(args.model)

    # Train
    # - ultralytics will save:
    #     <project>/detect/<run_name>/weights/best.pt
    #   (exact path depends on Ultralytics version, but best.pt is under weights/)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device else None,
        project=args.project,
        name=args.run_name,
        # Reasonable defaults for a small custom dataset
        # (you can tweak these after the first run)
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        # Save best checkpoint based on val metrics
        save="best",
        # Keep training output deterministic-ish
        seed=0,
        # Mixed precision helps on modern GPUs
        amp=True,
    )

    # Print a friendly hint to locate best.pt
    # Ultralytics typically exposes run directory via results.save_dir.
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        best_pt = Path(save_dir) / "weights" / "best.pt"
        print(f"\n[INFO] Training complete. Best weights: {best_pt}")
        if best_pt.exists():
            print("[INFO] best.pt exists ✅")
        else:
            print("[WARN] best.pt not found where expected; check runs directory.")
    else:
        print("\n[INFO] Training complete. Check the project/run folder for weights/best.pt")


if __name__ == "__main__":
    main()

