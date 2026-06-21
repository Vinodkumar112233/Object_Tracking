import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from src.tracker import ObjectTracker


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
OUTPUT_DIR = APP_ROOT / "outputs_web"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = Flask(__name__, template_folder=str(APP_ROOT / "templates"), static_folder=str(APP_ROOT / "static"))


@dataclass
class RunState:
    mode: str = "idle"  # idle | camera | video
    stop_event: threading.Event = threading.Event()
    worker: threading.Thread | None = None
    current_job_id: str | None = None


state = RunState()


def _frame_generator(job_id: str):
    """Yield multipart MJPEG frames for the *current* job."""
    # Each job has a dedicated tracker loop writing into a queue-less shared generator scope.
    # We keep it simple: use a local VideoCapture and produce frames directly.
    # To prevent stale jobs from feeding frames, validate job_id periodically.

    # NOTE: This generator is only created after job start; stopping is handled by stop_event.
    yield from ()


def _encode_mjpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


def _run_camera(job_id: str, model: str, conf: float, device: str | None):
    tracker = ObjectTracker(model_path=model, confidence_threshold=conf, device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    try:
        while not state.stop_event.is_set() and state.current_job_id == job_id:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            detections = tracker.detect(frame)
            tracks = tracker.track(frame, detections)
            frame = tracker.draw_tracks(frame, tracks, detections)

            # store latest frame for streaming
            with _latest_frame_lock:
                global _latest_frame
                _latest_frame = frame

            time.sleep(0.001)
    finally:
        cap.release()


def _run_video(job_id: str, video_path: str, model: str, conf: float, device: str | None, output_path: str | None):
    tracker = ObjectTracker(model_path=model, confidence_threshold=conf, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Optional output recording
    writer = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    try:
        while not state.stop_event.is_set() and state.current_job_id == job_id:
            ret, frame = cap.read()
            if not ret:
                break

            detections = tracker.detect(frame)
            tracks = tracker.track(frame, detections)
            frame = tracker.draw_tracks(frame, tracks, detections)

            with _latest_frame_lock:
                global _latest_frame
                _latest_frame = frame

            if writer:
                writer.write(frame)

            # Keep stream responsive
            time.sleep(0.001)
    finally:
        cap.release()
        if writer:
            writer.release()


_latest_frame = None
_latest_frame_lock = threading.Lock()


def _stream_mjpeg(job_id: str):
    """Continuously stream latest annotated frame as MJPEG."""
    # Wait until first frame becomes available
    while state.current_job_id == job_id:
        if state.stop_event.is_set():
            break

        with _latest_frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.02)
            continue

        data = _encode_mjpeg(frame)
        if data is None:
            time.sleep(0.02)
            continue

        yield data

    # When job ends, stop streaming
    return


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "mode": state.mode})


@app.route("/video_feed")
def video_feed():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    return Response(_stream_mjpeg(job_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global _latest_frame

    payload = request.get_json(force=True) if request.is_json else {}
    model = payload.get("model", "yolov8n.pt")
    conf = float(payload.get("conf", 0.4))
    device = payload.get("device")  # optional

    # Stop previous job
    state.stop_event.set()
    if state.worker and state.worker.is_alive():
        state.worker.join(timeout=2)

    state.stop_event = threading.Event()
    state.mode = "camera"
    state.current_job_id = str(uuid.uuid4())
    job_id = state.current_job_id

    with _latest_frame_lock:
        _latest_frame = None

    def worker():
        try:
            _run_camera(job_id=job_id, model=model, conf=conf, device=device)
        except Exception as e:
            # Surface error by keeping job_id but clearing frames
            with _latest_frame_lock:
                global _latest_frame
                _latest_frame = None
            print(f"[ERROR] camera job failed: {e}")
        finally:
            if state.current_job_id == job_id:
                state.mode = "idle"

    state.worker = threading.Thread(target=worker, daemon=True)
    state.worker.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.route("/upload", methods=["POST"])
def upload():
    global _latest_frame

    if "video" not in request.files:
        return jsonify({"error": "Missing video file field 'video'"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    payload = request.form
    model = payload.get("model", "yolov8n.pt")
    conf = float(payload.get("conf", 0.4))
    device = payload.get("device")  # optional

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    input_name = f"upload_{uuid.uuid4().hex}{ext}"
    input_path = str(UPLOAD_DIR / input_name)

    file.save(input_path)

    # Stop previous job
    state.stop_event.set()
    if state.worker and state.worker.is_alive():
        state.worker.join(timeout=2)

    state.stop_event = threading.Event()
    state.mode = "video"
    state.current_job_id = str(uuid.uuid4())
    job_id = state.current_job_id

    with _latest_frame_lock:
        _latest_frame = None

    # Optional output recording (same name, outputs_web)
    output_name = f"tracked_{Path(input_name).stem}.mp4"
    output_path = str(OUTPUT_DIR / output_name)

    def worker():
        try:
            _run_video(
                job_id=job_id,
                video_path=input_path,
                model=model,
                conf=conf,
                device=device,
                output_path=output_path,
            )
        except Exception as e:
            with _latest_frame_lock:
                global _latest_frame
                _latest_frame = None
            print(f"[ERROR] video job failed: {e}")
        finally:
            if state.current_job_id == job_id:
                state.mode = "idle"

    state.worker = threading.Thread(target=worker, daemon=True)
    state.worker.start()

    return jsonify({"ok": True, "job_id": job_id, "output": output_path})


@app.route("/stop", methods=["POST"])
def stop():
    state.stop_event.set()
    state.mode = "idle"
    return jsonify({"ok": True})


if __name__ == "__main__":
    # For local testing.
    # Disable the reloader to prevent Werkzeug/Windows socket teardown
    # while the MJPEG streaming response is still active.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


