# Real-Time Object Detection & Tracking (YOLOv8 + DeepSORT)

A real-time multi-object detection and tracking system built with **YOLOv8** for detection and **DeepSORT** for tracking. Each detected object is assigned a persistent unique ID that follows it across frames, with GPU-accelerated inference for high FPS performance.

## ✨ Features

- **Real-time object detection** using YOLOv8 (Ultralytics)
- **Multi-object tracking** with DeepSORT — persistent unique IDs per object across frames
- **GPU acceleration** via CUDA (auto-detects GPU, falls back to CPU)
- Works with **webcam input** or **video files**
- Optional **output video saving** with bounding boxes + track IDs drawn
- Live **FPS counter** overlay
- Standalone **benchmark script** to measure raw inference FPS

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | DeepSORT (deep-sort-realtime) |
| Video I/O | OpenCV |
| Acceleration | PyTorch + CUDA |
| Language | Python 3.10+ |

## 📁 Project Structure

```
object-tracking-yolov8/
├── src/
│   ├── tracker.py        # Main detection + tracking pipeline
│   └── benchmark.py       # Standalone FPS benchmark script
├── tests/
│   └── test_tracker.py    # Smoke tests for the pipeline
├── assets/                 # Sample input videos (gitignored)
├── outputs/                 # Saved output videos (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vinodkumar112233/object-tracking-yolov8.git
cd object-tracking-yolov8
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run via the `ultralytics` package — no manual download needed.

### 4. Run on webcam

```bash
python src/tracker.py --source 0
```

### 5. Run on a video file

```bash
python src/tracker.py --source assets/sample_video.mp4 --output outputs/result.mp4
```

### 6. Run headless (no display window, e.g. on a server)

```bash
python src/tracker.py --source assets/sample_video.mp4 --output outputs/result.mp4 --no-show
```

## ⚙️ Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or path to video file |
| `--model` | `yolov8n.pt` | YOLOv8 model weights (n/s/m/l/x variants) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--output` | `None` | Path to save annotated output video |
| `--no-show` | `False` | Disable live display window |

## 📊 Benchmarking

To measure raw inference FPS on your hardware:

```bash
python src/benchmark.py --source assets/sample_video.mp4 --model yolov8n.pt --frames 150
```

This reports average inference time and FPS, separate from tracking + drawing overhead — useful for comparing CPU vs GPU performance.

> **Note:** ~30 FPS performance is achievable on a CUDA-enabled GPU (e.g. NVIDIA GTX 1650 or better) using the `yolov8n` (nano) model. CPU-only inference will be significantly slower — use `yolov8n.pt` for the best CPU performance.

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

## 🔍 How It Works

1. **Detection** — Each frame is passed through YOLOv8, returning bounding boxes, confidence scores, and class labels.
2. **Tracking** — Detections are passed to DeepSORT, which uses a Kalman filter for motion prediction and a deep appearance-embedding network for re-identification, assigning consistent IDs across frames even through brief occlusions.
3. **Visualization** — Bounding boxes and track IDs are drawn on each frame in real time, with an FPS counter for performance monitoring.

## 📌 Future Improvements

- [ ] Multi-class filtering (track only specific object classes)
- [ ] Web dashboard (Flask/Streamlit) for live stream viewing
- [ ] Export tracking analytics (object counts, dwell time, trajectories) to CSV
- [ ] Support for larger YOLOv8 variants with TensorRT optimization

## 👤 Author

**Vinod Kumar Barnana**
B.Tech, Artificial Intelligence & Machine Learning — KIET Kakinada
[LinkedIn](https://linkedin.com/in/vinod-barnana-891923334) • [GitHub](https://github.com/Vinodkumar112233)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
