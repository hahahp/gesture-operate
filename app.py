#!/usr/bin/env python3
"""Real-time MediaPipe gesture recognition plus a tiny air-drawing demo."""

import argparse
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# MediaPipe's 21-point hand skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

GESTURE_HINTS = {
    "Pointing_Up": "DRAW",
    "Closed_Fist": "CLEAR",
    "Victory": "NEXT COLOR",
    "Open_Palm": "PAUSE",
}


class LatestResult:
    """Thread-safe handoff from MediaPipe's callback to the UI loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None

    def update(self, result) -> None:
        with self._lock:
            self._result = result

    def get(self):
        with self._lock:
            return self._result


class AirCanvas:
    """Small gesture-controlled drawing layer."""

    COLORS = (
        (80, 220, 255),   # yellow
        (255, 120, 80),   # blue
        (120, 255, 120),  # green
        (255, 100, 220),  # pink
    )

    def __init__(self) -> None:
        self.layer: Optional[np.ndarray] = None
        self.last_point: Optional[Tuple[int, int]] = None
        self.color_index = 0
        self.last_gesture = ""

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.COLORS[self.color_index]

    def ensure_size(self, frame: np.ndarray) -> None:
        if self.layer is None or self.layer.shape != frame.shape:
            self.layer = np.zeros_like(frame)
            self.last_point = None

    def apply(self, frame: np.ndarray, gesture: str, point: Optional[Tuple[int, int]]) -> None:
        self.ensure_size(frame)

        if gesture == "Closed_Fist":
            self.layer.fill(0)
            self.last_point = None
        elif gesture == "Victory":
            if self.last_gesture != "Victory":
                self.color_index = (self.color_index + 1) % len(self.COLORS)
            self.last_point = None
        elif gesture == "Pointing_Up" and point is not None:
            if self.last_point is not None:
                cv2.line(self.layer, self.last_point, point, self.color, 8, cv2.LINE_AA)
            self.last_point = point
        else:
            self.last_point = None

        self.last_gesture = gesture

    def clear(self) -> None:
        if self.layer is not None:
            self.layer.fill(0)
        self.last_point = None

    def next_color(self) -> None:
        self.color_index = (self.color_index + 1) % len(self.COLORS)

    def composite(self, frame: np.ndarray) -> np.ndarray:
        if self.layer is None:
            return frame
        mask = np.any(self.layer != 0, axis=2)
        output = frame.copy()
        output[mask] = cv2.addWeighted(frame, 0.25, self.layer, 0.75, 0)[mask]
        return output


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    print("Downloading the official MediaPipe gesture model...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with partial_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        partial_path.replace(model_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def draw_hand(frame: np.ndarray, landmarks) -> Tuple[int, int]:
    height, width = frame.shape[:2]
    points = [
        (
            max(0, min(width - 1, int(landmark.x * width))),
            max(0, min(height - 1, int(landmark.y * height))),
        )
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (90, 210, 255), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 7 if index == 8 else 4
        color = (255, 90, 180) if index == 8 else (80, 255, 160)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    return points[8]


def add_hud(
    frame: np.ndarray,
    gesture: str,
    score: float,
    handedness: str,
    fps: float,
    color: Tuple[int, int, int],
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (430, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    title = gesture if gesture else "Show your hand"
    if gesture:
        title = "%s  %.0f%%" % (gesture, score * 100)
    cv2.putText(frame, title, (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    action = GESTURE_HINTS.get(gesture, "GESTURE DETECTED" if gesture else "")
    details = "  ".join(part for part in (handedness, action) if part)
    cv2.putText(frame, details, (30, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Point: draw | Fist: clear | Victory: color",
        (30, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Q / Esc: quit    C: clear    Space: color    %.0f FPS" % fps,
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )


def create_recognizer(model_path: Path, result_callback=None):
    running_mode = (
        mp.tasks.vision.RunningMode.LIVE_STREAM
        if result_callback is not None
        else mp.tasks.vision.RunningMode.IMAGE
    )
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def open_camera(camera_index: int):
    if sys.platform == "darwin":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def check_camera(camera_index: int) -> int:
    camera = open_camera(camera_index)
    try:
        if not camera.isOpened():
            print(
                "Cannot open camera %d. Check macOS camera permissions." % camera_index,
                file=sys.stderr,
            )
            return 2

        # A few warm-up reads avoid reporting an empty first frame on some webcams.
        frame = None
        for _ in range(10):
            ok, candidate = camera.read()
            if ok and candidate is not None:
                frame = candidate
                break
        if frame is None:
            print("Camera opened, but no frame could be read.", file=sys.stderr)
            return 3

        height, width = frame.shape[:2]
        print("OK: camera %d returned a %dx%d frame." % (camera_index, width, height))
        return 0
    finally:
        camera.release()


def run_camera(model_path: Path, camera_index: int, mirror: bool) -> int:
    latest = LatestResult()

    def on_result(result, _output_image, _timestamp_ms) -> None:
        latest.update(result)

    camera = open_camera(camera_index)
    if not camera.isOpened():
        camera.release()
        print(
            "Cannot open camera %d. On macOS, allow camera access for your terminal "
            "in System Settings > Privacy & Security > Camera." % camera_index,
            file=sys.stderr,
        )
        return 2

    canvas = AirCanvas()
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    last_timestamp_ms = -1

    try:
        with create_recognizer(model_path, on_result) as recognizer:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera opened, but no frame could be read.", file=sys.stderr)
                    return 3

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                recognizer.recognize_async(mp_image, timestamp_ms)

                gesture = ""
                score = 0.0
                handedness = ""
                index_tip = None
                result = latest.get()
                if result is not None and result.hand_landmarks:
                    index_tip = draw_hand(frame, result.hand_landmarks[0])
                    for hand_landmarks in result.hand_landmarks[1:]:
                        draw_hand(frame, hand_landmarks)

                    if result.gestures and result.gestures[0]:
                        category = result.gestures[0][0]
                        gesture = category.category_name or ""
                        score = float(category.score or 0.0)
                    if result.handedness and result.handedness[0]:
                        handedness = result.handedness[0][0].category_name or ""

                # Reject weak classifications but still show detected landmarks.
                if score < 0.55 or gesture == "None":
                    gesture = ""
                    score = 0.0

                canvas.apply(frame, gesture, index_tip)
                frame = canvas.composite(frame)

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = instant_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * instant_fps)
                add_hud(frame, gesture, score, handedness, smoothed_fps, canvas.color)

                cv2.imshow("MediaPipe Gesture Playground", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    canvas.clear()
                elif key == ord(" "):
                    canvas.next_color()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="path to a .task model")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera preview")
    parser.add_argument("--check", action="store_true", help="verify dependencies/model without opening a camera")
    parser.add_argument("--camera-check", action="store_true", help="read one camera frame and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_model(args.model)
        if args.check:
            with create_recognizer(args.model) as recognizer:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                blank_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank_frame)
                recognizer.recognize(blank_image)

            callback_completed = threading.Event()

            def on_check_result(_result, _output_image, _timestamp_ms) -> None:
                callback_completed.set()

            with create_recognizer(args.model, on_check_result) as live_recognizer:
                live_recognizer.recognize_async(blank_image, time.monotonic_ns() // 1_000_000)
                if not callback_completed.wait(timeout=3):
                    raise RuntimeError("LIVE_STREAM callback did not complete within 3 seconds")

            print("OK: IMAGE and LIVE_STREAM inference both completed.")
            return 0
        if args.camera_check:
            return check_camera(args.camera)
        return run_camera(args.model, args.camera, not args.no_mirror)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("Error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
