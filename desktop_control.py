#!/usr/bin/env python3
"""Control the Windows or macOS desktop with a spatial hand halo."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from desktop_core import (
    ActionKind,
    DesktopGestureEngine,
    HandObservation,
    InteractionFrame,
    ScreenArea,
)
from desktop_platform import DryRunBackend, create_backend, dispatch_action


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"


class ObservationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observation: Optional[HandObservation] = None
        self._updated_at = -1.0

    def update(self, observation: Optional[HandObservation]) -> None:
        with self._lock:
            self._observation = observation
            self._updated_at = time.monotonic()

    def get(self) -> Tuple[Optional[HandObservation], float]:
        with self._lock:
            return self._observation, self._updated_at


def observation_from_result(result, width: int, height: int, preferred_hand: str) -> Optional[HandObservation]:
    if result is None or not result.hand_landmarks:
        return None

    selected = 0
    preferred = preferred_hand.lower()
    labels = []
    for index in range(len(result.hand_landmarks)):
        label = ""
        if result.handedness and index < len(result.handedness) and result.handedness[index]:
            label = result.handedness[index][0].category_name or ""
        labels.append(label)
    if preferred != "any":
        for index, label in enumerate(labels):
            if label.lower() == preferred:
                selected = index
                break

    landmarks = result.hand_landmarks[selected]

    def pixel(index: int) -> Tuple[float, float]:
        return landmarks[index].x * width, landmarks[index].y * height

    thumb = pixel(4)
    index_tip = pixel(8)
    middle_tip = pixel(12)
    palm_a = pixel(5)
    palm_b = pixel(17)
    palm_width = max(
        ((palm_a[0] - palm_b[0]) ** 2 + (palm_a[1] - palm_b[1]) ** 2) ** 0.5,
        1.0,
    )

    def ratio(target: Tuple[float, float]) -> float:
        return (
            ((thumb[0] - target[0]) ** 2 + (thumb[1] - target[1]) ** 2) ** 0.5
            / palm_width
        )

    gesture = ""
    if result.gestures and selected < len(result.gestures) and result.gestures[selected]:
        gesture = result.gestures[selected][0].category_name or ""
    return HandObservation(
        x=float(landmarks[8].x),
        y=float(landmarks[8].y),
        index_pinch_ratio=ratio(index_tip),
        middle_pinch_ratio=ratio(middle_tip),
        gesture=gesture,
        handedness=labels[selected],
    )


class CameraWorker:
    """Captures frames without ever displaying the camera preview."""

    def __init__(self, camera_index: int, model_path: Path, mirror: bool, preferred_hand: str) -> None:
        self.camera_index = camera_index
        self.model_path = model_path
        self.mirror = mirror
        self.preferred_hand = preferred_hand
        self.store = ObservationStore()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="gesture-camera", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _run(self) -> None:
        camera = None
        try:
            import cv2
            import mediapipe as mp

            from app import create_recognizer, ensure_model, open_camera

            ensure_model(self.model_path)
            camera = open_camera(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError("Cannot open camera %d" % self.camera_index)

            frame_size = [1280, 720]

            def on_result(result, _output_image, _timestamp_ms) -> None:
                self.store.update(
                    observation_from_result(
                        result,
                        frame_size[0],
                        frame_size[1],
                        self.preferred_hand,
                    )
                )

            last_timestamp = -1
            with create_recognizer(self.model_path, on_result) as recognizer:
                while not self.stop_event.is_set():
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        raise RuntimeError("Camera opened, but no frame could be read")
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]
                    frame_size[:] = [width, height]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = time.monotonic_ns() // 1_000_000
                    if timestamp <= last_timestamp:
                        timestamp = last_timestamp + 1
                    last_timestamp = timestamp
                    recognizer.recognize_async(image, timestamp)
        except Exception as error:
            self.error = str(error)
        finally:
            self.store.update(None)
            if camera is not None:
                camera.release()


def run_self_check() -> int:
    screen = ScreenArea(0, 0, 1920, 1080)
    engine = DesktopGestureEngine(screen, active=True)
    backend = DryRunBackend()
    neutral = HandObservation(0.5, 0.5, 1.0, 1.0)
    index = HandObservation(0.5, 0.5, 0.25, 1.0)
    moved = HandObservation(0.65, 0.58, 0.25, 1.0)
    middle = HandObservation(0.55, 0.5, 1.0, 0.3)

    frames = [
        engine.update(neutral, 0.0),
        engine.update(index, 0.1),
        engine.update(neutral, 0.18),
        engine.update(index, 0.3),
        engine.update(neutral, 0.38),
        engine.update(index, 0.6),
        engine.update(moved, 0.7),
        engine.update(neutral, 0.8),
        engine.update(middle, 1.0),
        engine.update(neutral, 1.1),
    ]
    for frame in frames:
        for action in frame.actions:
            dispatch_action(backend, action)

    names = [name for name, _position in backend.events]
    required = {"primary_click", "secondary_click", "drag_begin", "drag_move", "drag_end"}
    if not required.issubset(names):
        raise RuntimeError("Gesture self-check missed actions: %s" % sorted(required - set(names)))

    paused = DesktopGestureEngine(screen, active=True, open_palm_seconds=1.0)
    paused.update(HandObservation(0.5, 0.5, 1.0, 1.0, "Open_Palm"), 2.0)
    result = paused.update(HandObservation(0.5, 0.5, 1.0, 1.0, "Open_Palm"), 3.1)
    if result.active or not any(action.kind == ActionKind.ACTIVE_CHANGED for action in result.actions):
        raise RuntimeError("Open-palm pause self-check failed")

    print("OK: mapping, click, double-pinch timing, drag, secondary pinch and pause state passed.")
    return 0


def run_desktop(args) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from desktop_overlay import SpatialHaloOverlay

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Gesture Operate")
    application.setQuitOnLastWindowClosed(False)
    overlay = SpatialHaloOverlay(application)
    engine = DesktopGestureEngine(overlay.screen_area(), active=args.active)
    backend = create_backend(dry_run=args.dry_run)
    worker = CameraWorker(args.camera, args.model, not args.no_mirror, args.hand)
    worker.start()
    error_reported = False

    def tick() -> None:
        nonlocal error_reported
        if worker.error is not None:
            if not error_reported:
                print("Error: %s" % worker.error, file=sys.stderr)
                error_reported = True
            application.quit()
            return

        observation, updated_at = worker.store.get()
        now = time.monotonic()
        if now - updated_at > 0.28:
            observation = None
        frame: InteractionFrame = engine.update(observation, now)
        for action in frame.actions:
            if action.kind != ActionKind.ACTIVE_CHANGED:
                dispatch_action(backend, action)
        if frame.feedback in ("primary", "secondary", "double", "drag_begin", "drag_end"):
            overlay.add_pulse(frame.feedback, frame.position)
        backend.set_cursor_hidden(
            frame.active and frame.visible and not args.show_system_cursor and not args.dry_run
        )
        overlay.set_interaction(frame)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(16)

    def cleanup() -> None:
        timer.stop()
        worker.stop()
        backend.close()

    application.aboutToQuit.connect(cleanup)
    signal.signal(signal.SIGINT, lambda _signal, _frame: application.quit())
    print("Gesture Operate is running without a camera preview.")
    print("Hold an open palm for one second to pause/resume. Press Ctrl+C to quit.")
    if args.dry_run:
        print("Dry-run mode: the halo is live, but no desktop input will be sent.")
    return application.exec()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="path to gesture model")
    parser.add_argument("--hand", choices=("any", "left", "right"), default="any", help="preferred controlling hand")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera input")
    parser.add_argument("--active", action="store_true", help="start active instead of safely paused")
    parser.add_argument("--dry-run", action="store_true", help="show feedback without sending desktop input")
    parser.add_argument("--show-system-cursor", action="store_true", help="do not hide the native pointer while tracking")
    parser.add_argument("--check", action="store_true", help="run a headless, side-effect-free logic check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            return run_self_check()
        return run_desktop(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("Error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
