"""
Real-Time Object Detection and Tracking System
Using YOLOv8 for detection and DeepSORT for multi-object tracking.

Author: Vinod Kumar Barnana
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    """YOLOv8 detection + DeepSORT tracking with professional display overlays.

    Features added:
      1) LABEL + CONFIDENCE SCORE + track id
      2) MOTION TRAIL per track id
      3) COUNTING LINE (configurable, counts each track id once)
      4) FPS / STATS overlay
      5) CLASS-WISE SUMMARY panel
      6) CSV export of per-frame tracking analytics
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        max_age: int = 30,
        device: str | None = None,
        # --- Feature configs ---
        trail_length: int = 20,
        # Counting line defaults: computed on first processed frame (midline).
        line_start: tuple[int, int] | None = None,
        line_end: tuple[int, int] | None = None,
        # Label stabilization / vehicle remap
        # - Stabilize class per track to prevent frequent label flips.
        # - Optionally filter to COCO vehicle-like classes.
        stabilize_track_class: bool = True,
        vehicle_only: bool = True,
        vehicle_label_remap: bool = True,
        min_conf_for_class_update: float = 0.35,
        # CSV export
        export_csv: bool = True,
        csv_path: str | None = None,
        csv_append: bool = True,
        csv_flush_every_n: int = 30,
        # Panel placement
        stats_overlay_pos: tuple[int, int] = (10, 30),  # top-left text baseline
        summary_panel_anchor: str = "top_right",  # top_right | bottom_right
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
        )

        self.class_names = self.model.names

        # --- Label stabilization / filtering state ---
        self.stabilize_track_class = bool(stabilize_track_class)
        self.vehicle_only = bool(vehicle_only)
        self.vehicle_label_remap = bool(vehicle_label_remap)
        self.min_conf_for_class_update = float(min_conf_for_class_update)

        # track_id -> best class info observed so far
        self.track_best_class: dict[int, dict[str, object]] = {}

        # COCO vehicle-like classes in YOLOv8 (dataset-dependent, so we check membership at runtime)
        self._vehicle_like_classes = {
            "car",
            "motorcycle",
            "bus",
            "truck",
            "bicycle",
            "train",
        }

        # Remap intent: normalize confusing vehicle subclasses into a smaller stable set.
        #
        # Why: YOLO class predictions can frequently confuse “car/jeep/truck/suv/…”,
        # especially when appearance is partially occluded. We force a consistent
        # naming bucket so the *track* label is stable and user-friendly.
        self._vehicle_remap = {
            # cars (and common car-like variants)
            "car": "car",
            "jeep": "car",
            "sedan": "car",
            "suv": "car",
            "wagon": "car",
            "limousine": "car",
            "convertible": "car",
            # motorcycles & scooters
            "motorcycle": "motorcycle",
            "scooter": "motorcycle",
            # trucks
            "truck": "truck",
            "lorry": "truck",
            # buses
            "bus": "bus",
            # bicycles
            "bicycle": "bicycle",
            "bike": "bicycle",
            # trains
            "train": "train",
        }


        # --- Feature state ---
        self.trail_length = int(trail_length)
        self.trails: dict[int, list[tuple[int, int]]] = {}  # track_id -> last N centers


        self.line_start = line_start
        self.line_end = line_end
        self._line_ready = False

        self.counted_ids: set[int] = set()  # ensure each track id is counted once
        self.class_counts: dict[str, int] = {}  # class_name -> running count

        # FPS/stats/frame tracking
        self.frame_idx = 0
        self.start_time = None
        self.last_fps_update_time = None
        self._fps = 0.0

        self.stats_overlay_pos = stats_overlay_pos
        self.summary_panel_anchor = summary_panel_anchor

        # CSV export (open once per ObjectTracker lifetime)
        self.export_csv = export_csv
        self.csv_path = csv_path or str(Path("outputs") / "tracking_export.csv")
        self.csv_append = csv_append
        self.csv_flush_every_n = int(csv_flush_every_n)
        self._csv_file = None
        self._csv_writer = None
        self._csv_rows_since_flush = 0

    def _reset_run_state(self):
        """Reset per-run state.

        Called lazily from draw_tracks/run so that repeated uses of the same
        ObjectTracker instance (if any) start clean.
        """
        self.trails.clear()
        self.counted_ids.clear()
        self.class_counts.clear()
        self.track_best_class.clear()


        self.frame_idx = 0
        self.start_time = None
        self.last_fps_update_time = None
        self._fps = 0.0

        self._line_ready = False

        # CSV writer lifecycle is per-tracker lifetime; do not reset file handles here.

    def _ensure_csv_opened(self, header: list[str]):
        if not self.export_csv:
            return
        if self._csv_file is not None:
            return

        # Ensure output directory exists
        csv_path = Path(self.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.csv_append else "w"
        file_exists = csv_path.exists() and mode == "a"
        import csv

        self._csv_file = open(csv_path, mode, newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(header)
            self._csv_file.flush()

    def _close_csv(self):
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            self._csv_file = None
            self._csv_writer = None


    def detect(self, frame):
        """Run YOLOv8 inference on a single frame and return detections
        formatted for DeepSORT: [[x, y, w, h], confidence, class_id]."""
        results = self.model.predict(
            frame, conf=self.confidence_threshold, device=self.device, verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls_id))

        return detections

    def track(self, frame, detections):
        """Update DeepSORT tracker with current frame detections and
        return list of confirmed tracks."""
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return [t for t in tracks if t.is_confirmed()]

    def _maybe_filter_and_remap_class(self, class_name: str | None) -> str | None:
        if class_name is None:
            return None

        # Normalize casing just in case.
        class_name = str(class_name).strip()

        if self.vehicle_only:
            if class_name in self._vehicle_like_classes:
                # Optional remap into stable buckets.
                if self.vehicle_label_remap:
                    return self._vehicle_remap.get(class_name, class_name)
                return class_name
            # Drop non-vehicle-like classes.
            return None

        if self.vehicle_label_remap:
            return self._vehicle_remap.get(class_name, class_name)

        return class_name

    def _get_stable_class_for_track(
        self,
        track_id: int,
        current_class_name: str | None,
        current_conf: float | None,
    ) -> tuple[str | None, float | None]:
        """Return stabilized (class_name, confidence) for a track.

        Avoids frequent label flips by keeping the best-confidence class seen
        for this track. If vehicle_only filtering filtered out the current class,
        it won’t overwrite the stable label.
        """
        if not self.stabilize_track_class:
            return current_class_name, current_conf

        if track_id not in self.track_best_class:
            # Initialize when we first get a valid class.
            if current_class_name is None:
                return None, None
            return current_class_name, current_conf

        best = self.track_best_class[track_id]
        best_name = best.get("class_name")
        best_conf = best.get("conf")

        if current_class_name is None:
            # No new usable class this frame.
            return best_name, best_conf

        conf_val = float(current_conf) if current_conf is not None else 0.0
        if conf_val < self.min_conf_for_class_update:
            # Don’t allow low-confidence mismatches to override.
            return best_name, best_conf

        # Update if better confidence (or if best was None).
        best_conf_val = float(best_conf) if best_conf is not None else -1.0
        if best_name is None or conf_val >= best_conf_val:
            self.track_best_class[track_id] = {"class_name": current_class_name, "conf": conf_val}
            return current_class_name, conf_val

        return best_name, best_conf

    def _match_track_to_detection(self, track_center, detections):

        """Associate a DeepSORT track with the most likely current YOLO detection.


        DeepSORT tracks don't carry class/confidence in this wiring, so we
        cheaply match by nearest detection center.

        Returns: (class_id, class_name, confidence) or (None, None, None)
        """
        if not detections:
            return None, None, None

        cx, cy = track_center
        best = None
        best_dist2 = None

        for det in detections:
            (x1, y1, w, h), conf, cls_id = det
            dcx = x1 + w / 2.0
            dcy = y1 + h / 2.0
            dist2 = (dcx - cx) ** 2 + (dcy - cy) ** 2
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best = (cls_id, self.class_names.get(cls_id, str(cls_id)), conf)

        if best is None:
            return None, None, None

        cls_id, class_name, conf = best
        class_name = self._maybe_filter_and_remap_class(class_name)

        # If filtered out, treat as no match for class purposes.
        if class_name is None:
            return None, None, None

        return cls_id, class_name, conf


    def draw_tracks(self, frame, tracks, detections):
        """Draw all tracking overlays for the current frame.

        This method integrates the 6 requested features and is called by
        the Flask app for both webcam + uploaded video.
        """
        if self.start_time is None:
            # Initialize timing when we see the first frame.
            self.start_time = time.time()

        self._reset_if_line_not_ready(frame)

        frame_h, frame_w = frame.shape[:2]
        self.frame_idx += 1

        # Stats: number of currently being tracked objects
        active_count = len(tracks)

        # Prepare reusable timestamp for the entire frame (for CSV rows)
        timestamp = time.time()

        # Build per-track info and apply counting/trails/csv
        track_info = {}

        for track in tracks:
            track_id = int(track.track_id)
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(r), int(b)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            color = self._color_for_id(track_id)

            cls_id, class_name, conf = self._match_track_to_detection((cx, cy), detections)

            # Stabilize class label per track to avoid frequent flips.
            stable_class_name, stable_conf = self._get_stable_class_for_track(
                track_id=track_id,
                current_class_name=class_name,
                current_conf=conf,
            )

            track_info[track_id] = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "color": color,
                "class_name": stable_class_name or "unknown",
                "conf": float(stable_conf) if stable_conf is not None else 0.0,
            }


            # Motion trail update
            self.draw_trails(frame, track_id, (cx, cy))

            # Counting line update (needs prev center)
            self.check_line_crossing(track_id, track_info[track_id])

            # CSV export row(s)
            self.export_to_csv(
                frame_number=self.frame_idx,
                timestamp=timestamp,
                track_id=track_id,
                class_name=track_info[track_id]["class_name"],
                confidence=track_info[track_id]["conf"],
                x1=track_info[track_id]["x1"],
                y1=track_info[track_id]["y1"],
                x2=track_info[track_id]["x2"],
                y2=track_info[track_id]["y2"],
            )

        # Draw boxes + labels
        for track_id, info in track_info.items():
            self._draw_box_and_label(frame, track_id, info)

        # FPS / frame stats overlay
        fps = self._update_fps()
        self.draw_stats_overlay(frame, fps=fps, frame_idx=self.frame_idx, active_count=active_count)

        # Class-wise summary panel
        self.draw_summary_panel(frame)

        # Flush CSV occasionally to limit IO overhead
        self._maybe_flush_csv()

        return frame

    def _reset_if_line_not_ready(self, frame):
        """Initialize counting line to midline on first frame if not provided."""
        if self._line_ready:
            return

        frame_h, frame_w = frame.shape[:2]

        if self.line_start is None:
            self.line_start = (0, int(frame_h * 0.5))
        if self.line_end is None:
            self.line_end = (frame_w, int(frame_h * 0.5))

        self._line_ready = True

    def _update_fps(self):
        if self.last_fps_update_time is None:
            self.last_fps_update_time = self.start_time
            return 0.0

        now = time.time()
        elapsed = now - self.last_fps_update_time
        if elapsed <= 0:
            return self._fps

        # Compute instantaneous-ish FPS using total frames since start.
        total_elapsed = now - self.start_time if self.start_time else 0
        if total_elapsed > 0:
            self._fps = self.frame_idx / total_elapsed
        self.last_fps_update_time = now
        return self._fps

    def _draw_box_and_label(self, frame, track_id: int, info: dict):
        """Feature 1: draw filled label with class + confidence + track id."""
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        color = info["color"]
        class_name = info["class_name"]
        conf = info["conf"]

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f} | ID {track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Ensure label background is within image bounds
        y1_label_top = max(0, y1 - th - 8)

        cv2.rectangle(frame, (x1, y1_label_top), (x1 + tw + 4, y1_label_top + th + 4), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1_label_top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_trails(self, frame, track_id: int, center: tuple[int, int]):
        """Feature 2: update and draw motion trail behind the object."""
        pts = self.trails.get(track_id)
        if pts is None:
            pts = []
            self.trails[track_id] = pts

        pts.append(center)
        if len(pts) > self.trail_length:
            pts.pop(0)

        # Draw fading dots (simple and fast)
        # Oldest -> lightest, newest -> most visible
        n = len(pts)
        if n == 0:
            return

        color = self._color_for_id(track_id)
        for i, (x, y) in enumerate(pts):
            # fade factor: 0..1
            alpha = i / max(1, n - 1)
            # Map alpha to brightness
            b = int(50 + 205 * alpha)
            dot_color = (int(color[0] * alpha + (255 * (1 - alpha)) * 0.2),
                         int(color[1] * alpha + (255 * (1 - alpha)) * 0.2),
                         int(color[2] * alpha + (255 * (1 - alpha)) * 0.2))
            cv2.circle(frame, (x, y), 2, dot_color, -1)

    def _point_line_side(self, point: tuple[int, int], line_start: tuple[int, int], line_end: tuple[int, int]):
        """Signed area / side test for a point relative to a directed line."""
        (x, y) = point
        (x1, y1) = line_start
        (x2, y2) = line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def check_line_crossing(self, track_id: int, info: dict):
        """Feature 3: increment class counter when center crosses the line once per track."""
        if track_id in self.counted_ids:
            return

        # Need previous center for this track
        pts = self.trails.get(track_id, [])
        if len(pts) < 2:
            return

        prev = pts[-2]
        curr = pts[-1]

        if self.line_start is None or self.line_end is None:
            return

        side_prev = self._point_line_side(prev, self.line_start, self.line_end)
        side_curr = self._point_line_side(curr, self.line_start, self.line_end)

        # Cross if sign changes (with tolerance)
        if side_prev == 0:
            return
        if side_curr == 0:
            return

        if (side_prev > 0 and side_curr < 0) or (side_prev < 0 and side_curr > 0):
            class_name = info["class_name"]
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            self.counted_ids.add(track_id)

            # (Optional) could draw a highlight here; keeping it off for performance.


    def _draw_counting_line(self, frame, color=(0, 255, 255)):
        if frame is None:
            return
        if self.line_start is None or self.line_end is None:
            return
        cv2.line(frame, self.line_start, self.line_end, color, 2)

    def draw_stats_overlay(self, frame, fps: float, frame_idx: int, active_count: int):
        """Feature 4: FPS / frame number / active tracked objects (top-left)."""
        x, y = self.stats_overlay_pos
        lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_idx}",
            f"Tracking: {active_count}",
        ]

        for i, txt in enumerate(lines):
            y_i = y + i * 22
            cv2.putText(
                frame,
                txt,
                (x, y_i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def draw_summary_panel(self, frame):
        """Feature 5: running class-wise summary panel."""
        if not self.class_counts:
            # Still draw panel header lightly
            pass

        panel_items = sorted(self.class_counts.items(), key=lambda kv: kv[1], reverse=True)
        panel_items = panel_items[:10]

        pad = 10
        font_scale = 0.6
        line_height = 22

        # Compute panel size
        text_lines = ["Counts:"] + [f"{cls}: {cnt}" for cls, cnt in panel_items]
        (max_tw, _), _ = cv2.getTextSize(
            max(text_lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        panel_w = max_tw + pad * 2
        panel_h = pad + line_height * len(text_lines) + pad

        frame_h, frame_w = frame.shape[:2]

        if self.summary_panel_anchor == "bottom_right":
            x2 = frame_w - 5
            y2 = frame_h - 5
            x1 = x2 - panel_w
            y1 = y2 - panel_h
        else:
            # top_right default
            x2 = frame_w - 5
            y1 = 5
            x1 = x2 - panel_w
            y2 = y1 + panel_h

        x1 = max(0, x1)
        y1 = max(0, y1)

        panel_bg = (40, 40, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), panel_bg, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        for i, txt in enumerate(text_lines):
            y_i = y1 + pad + i * line_height
            cv2.putText(
                frame,
                txt,
                (x1 + pad, y_i),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def export_to_csv(
        self,
        frame_number: int,
        timestamp: float,
        track_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ):
        """Feature 6: append analytics rows to a CSV file (file opened once)."""
        if not self.export_csv:
            return

        header = [
            "frame_number",
            "track_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "timestamp",
        ]

        self._ensure_csv_opened(header)
        if self._csv_writer is None:
            return

        self._csv_writer.writerow(
            [
                frame_number,
                track_id,
                class_name,
                f"{confidence:.6f}",
                x1,
                y1,
                x2,
                y2,
                f"{timestamp:.6f}",
            ]
        )
        self._csv_rows_since_flush += 1

    def _maybe_flush_csv(self):
        if not self.export_csv:
            return
        if self._csv_file is None:
            return
        if self._csv_rows_since_flush >= self.csv_flush_every_n:
            self._csv_file.flush()
            self._csv_rows_since_flush = 0


    @staticmethod
    def _color_for_id(track_id):
        """Deterministic color per track ID for visual consistency."""
        idx = int(track_id) if str(track_id).isdigit() else hash(track_id)
        rng = (idx * 47) % 255
        return (int((rng * 37) % 255), int((rng * 17) % 255), int((rng * 89) % 255))

    def run(self, source=0, output_path: str | None = None, show: bool = True):
        """Main loop: read frames, detect, track, draw, display/save."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        frame_count = 0
        start_time = time.time()

        # Reset overlays/counters/timing for this run.
        self._close_csv()  # close any previous CSV file handles
        self._reset_run_state()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break


                detections = self.detect(frame)
                tracks = self.track(frame, detections)
                frame = self.draw_tracks(frame, tracks, detections)

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

                if writer:
                    writer.write(frame)

                if show:
                    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

            # Best-effort flush/close for CSV export.
            self._maybe_flush_csv()
            self._close_csv()

            total_time = time.time() - start_time

            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"[INFO] Processed {frame_count} frames in {total_time:.2f}s "
                  f"(avg {avg_fps:.2f} FPS)")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT Object Tracking")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: webcam index (0) or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default=None,
                         help="Path to save output video (e.g. outputs/result.mp4)")
    parser.add_argument("--no-show", action="store_true",
                         help="Disable live display window (useful on servers)")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    tracker = ObjectTracker(model_path=args.model, confidence_threshold=args.conf)
    tracker.run(source=source, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
