#!/usr/bin/env python3
"""A camera-driven spatial canvas controlled by fingertip pinches."""

import argparse
import math
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app import DEFAULT_MODEL_PATH, HAND_CONNECTIONS, create_recognizer, ensure_model, open_camera


WINDOW_NAME = "Spatial Canvas // MediaPipe"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def point(point_value: np.ndarray) -> Tuple[int, int]:
    return int(point_value[0]), int(point_value[1])


def vector_angle(first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    return math.atan2(float(delta[1]), float(delta[0]))


@dataclass
class HandState:
    key: str
    cursor: np.ndarray
    landmarks: List[Tuple[int, int]]
    pinched: bool
    just_pinched: bool
    just_released: bool
    pinch_ratio: float


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    color: Tuple[int, int, int]
    life: float


class ResultStore:
    """Thread-safe bridge between MediaPipe's callback and the render loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None
        self._timestamp = -1

    def update(self, result, timestamp: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp = timestamp

    def get(self):
        with self._lock:
            return self._result, self._timestamp


class PinchTracker:
    """Converts landmarks into stable cursors and pinch events."""

    def __init__(self) -> None:
        self.cursors: Dict[str, np.ndarray] = {}
        self.pinch_states: Dict[str, bool] = {}

    @staticmethod
    def expanded_cursor(landmark, width: int, height: int) -> np.ndarray:
        # Expand the central 86% of the camera view to the whole screen.
        x = clamp((landmark.x - 0.07) / 0.86, 0.0, 1.0) * width
        y = clamp((landmark.y - 0.07) / 0.86, 0.0, 1.0) * height
        return np.array([x, y], dtype=np.float32)

    def update(self, result, width: int, height: int) -> List[HandState]:
        if result is None or not result.hand_landmarks:
            for key in self.pinch_states:
                self.pinch_states[key] = False
            return []

        output: List[HandState] = []
        seen = set()
        for index, landmarks in enumerate(result.hand_landmarks):
            label = "Hand %d" % (index + 1)
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                label = result.handedness[index][0].category_name or label
            key = label if label not in seen else "%s %d" % (label, index + 1)
            seen.add(key)

            raw_cursor = self.expanded_cursor(landmarks[8], width, height)
            previous_cursor = self.cursors.get(key, raw_cursor)
            cursor = previous_cursor * 0.64 + raw_cursor * 0.36
            self.cursors[key] = cursor

            thumb_tip = np.array([landmarks[4].x * width, landmarks[4].y * height])
            index_tip = np.array([landmarks[8].x * width, landmarks[8].y * height])
            palm_a = np.array([landmarks[5].x * width, landmarks[5].y * height])
            palm_b = np.array([landmarks[17].x * width, landmarks[17].y * height])
            palm_width = max(float(np.linalg.norm(palm_a - palm_b)), 1.0)
            pinch_ratio = float(np.linalg.norm(thumb_tip - index_tip)) / palm_width

            was_pinched = self.pinch_states.get(key, False)
            # Two thresholds provide hysteresis and prevent noisy double-clicks.
            is_pinched = pinch_ratio < (0.62 if was_pinched else 0.42)
            self.pinch_states[key] = is_pinched

            screen_landmarks = [
                (
                    int(clamp(landmark.x, 0.0, 1.0) * width),
                    int(clamp(landmark.y, 0.0, 1.0) * height),
                )
                for landmark in landmarks
            ]
            output.append(
                HandState(
                    key=key,
                    cursor=cursor,
                    landmarks=screen_landmarks,
                    pinched=is_pinched,
                    just_pinched=is_pinched and not was_pinched,
                    just_released=was_pinched and not is_pinched,
                    pinch_ratio=pinch_ratio,
                )
            )
        return output


class SpatialCard:
    def __init__(
        self,
        title: str,
        states: Tuple[str, ...],
        center: Tuple[float, float],
        size: Tuple[float, float],
        accent: Tuple[int, int, int],
        icon: str,
    ) -> None:
        self.title = title
        self.states = states
        self.center = np.array(center, dtype=np.float32)
        self.base_size = np.array(size, dtype=np.float32)
        self.accent = accent
        self.icon = icon
        self.scale = 1.0
        self.angle = 0.0
        self.velocity = np.zeros(2, dtype=np.float32)
        self.angular_velocity = 0.0
        self.state_index = 0
        self.pulse = 0.0

    @property
    def subtitle(self) -> str:
        return self.states[self.state_index % len(self.states)]

    @property
    def size(self) -> np.ndarray:
        return self.base_size * self.scale

    def corners(self) -> np.ndarray:
        half = self.size / 2.0
        local = np.array(
            [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]],
            dtype=np.float32,
        )
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        return local @ rotation.T + self.center

    def contains(self, target: np.ndarray) -> bool:
        delta = target - self.center
        cosine, sine = math.cos(-self.angle), math.sin(-self.angle)
        local_x = delta[0] * cosine - delta[1] * sine
        local_y = delta[0] * sine + delta[1] * cosine
        half = self.size / 2.0
        return abs(float(local_x)) <= half[0] and abs(float(local_y)) <= half[1]

    def activate(self) -> None:
        self.state_index = (self.state_index + 1) % len(self.states)
        self.pulse = 1.0

    def update(self, dt: float, width: int, height: int, grabbed: bool) -> None:
        self.pulse = max(0.0, self.pulse - dt * 1.7)
        if grabbed:
            return
        self.center += self.velocity * dt
        self.angle += self.angular_velocity * dt
        damping = math.pow(0.88, dt * 60.0)
        self.velocity *= damping
        self.angular_velocity *= damping

        half = self.size / 2.0
        if self.center[0] - half[0] < 22:
            self.center[0] = half[0] + 22
            self.velocity[0] = abs(self.velocity[0]) * 0.55
        elif self.center[0] + half[0] > width - 22:
            self.center[0] = width - half[0] - 22
            self.velocity[0] = -abs(self.velocity[0]) * 0.55
        if self.center[1] - half[1] < 90:
            self.center[1] = half[1] + 90
            self.velocity[1] = abs(self.velocity[1]) * 0.55
        elif self.center[1] + half[1] > height - 52:
            self.center[1] = height - half[1] - 52
            self.velocity[1] = -abs(self.velocity[1]) * 0.55


class SpatialScene:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cards: List[SpatialCard] = []
        self.selected: Optional[SpatialCard] = None
        self.hovered: Optional[SpatialCard] = None
        self.mode = "IDLE"
        self.owner_key: Optional[str] = None
        self.drag_offset = np.zeros(2, dtype=np.float32)
        self.grab_origin = np.zeros(2, dtype=np.float32)
        self.grab_distance = 0.0
        self.transform_pair = ("", "")
        self.transform_distance = 1.0
        self.transform_angle = 0.0
        self.transform_scale = 1.0
        self.transform_card_angle = 0.0
        self.transform_offset = np.zeros(2, dtype=np.float32)
        self.last_midpoint = np.zeros(2, dtype=np.float32)
        self.trails: Dict[str, Deque[np.ndarray]] = {}
        self.particles: List[Particle] = []
        self.reset()

    def reset(self) -> None:
        card_width = clamp(self.width * 0.205, 210, 290)
        card_height = clamp(self.height * 0.225, 145, 190)
        size = (card_width, card_height)
        self.cards = [
            SpatialCard("PHOTOS", ("MEMORIES / 24", "NEON ALBUM", "LIVE CAPTURE"), (self.width * 0.23, self.height * 0.34), size, (255, 205, 65), "photo"),
            SpatialCard("MUSIC", ("AMBIENT / PAUSED", "PLAYING / 92 BPM", "SYNTHWAVE / LIVE"), (self.width * 0.52, self.height * 0.29), size, (220, 80, 255), "music"),
            SpatialCard("NOTES", ("IDEAS / 07", "PINNED / TODAY", "CAPTURED / 12"), (self.width * 0.74, self.height * 0.56), size, (95, 255, 145), "notes"),
            SpatialCard("PORTAL", ("DEPTH / 0.8", "ORBIT / ACTIVE", "LINK / READY"), (self.width * 0.38, self.height * 0.70), size, (70, 150, 255), "portal"),
        ]
        self.selected = None
        self.hovered = None
        self.mode = "IDLE"
        self.owner_key = None
        self.trails.clear()
        self.particles.clear()

    def pick(self, target: np.ndarray) -> Optional[SpatialCard]:
        for card in reversed(self.cards):
            if card.contains(target):
                return card
        return None

    def burst(self, location: np.ndarray, color: Tuple[int, int, int], count: int) -> None:
        for _ in range(count):
            angle = random.random() * math.tau
            speed = random.uniform(45.0, 190.0)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append(Particle(location.astype(np.float32).copy(), velocity, color, random.uniform(0.35, 0.8)))

    def start_drag(self, hand: HandState, card: SpatialCard) -> None:
        self.selected = card
        self.cards.remove(card)
        self.cards.append(card)
        self.mode = "DRAG"
        self.owner_key = hand.key
        self.drag_offset = card.center - hand.cursor
        self.grab_origin = hand.cursor.copy()
        self.grab_distance = 0.0
        card.velocity.fill(0)
        card.angular_velocity = 0.0
        self.burst(hand.cursor, card.accent, 10)

    def start_transform(self, first: HandState, second: HandState) -> None:
        if self.selected is None:
            return
        self.mode = "TRANSFORM"
        self.transform_pair = (first.key, second.key)
        self.transform_distance = max(float(np.linalg.norm(second.cursor - first.cursor)), 20.0)
        self.transform_angle = vector_angle(first.cursor, second.cursor)
        self.transform_scale = self.selected.scale
        self.transform_card_angle = self.selected.angle
        midpoint = (first.cursor + second.cursor) / 2.0
        self.transform_offset = self.selected.center - midpoint
        self.last_midpoint = midpoint.copy()
        self.burst(midpoint, self.selected.accent, 16)

    def release(self, location: Optional[np.ndarray], allow_tap: bool = True) -> None:
        if self.selected is not None and allow_tap and self.grab_distance < 30:
            self.selected.activate()
        if location is not None and self.selected is not None:
            self.burst(location, self.selected.accent, 13)
        self.selected = None
        self.owner_key = None
        self.mode = "IDLE"

    def update(self, hands: List[HandState], dt: float) -> None:
        hand_map = {hand.key: hand for hand in hands}
        active = [hand for hand in hands if hand.pinched]
        for hand in hands:
            self.trails.setdefault(hand.key, deque(maxlen=14)).append(hand.cursor.copy())
        for key in list(self.trails):
            if key not in hand_map and self.trails[key]:
                self.trails[key].popleft()

        self.hovered = None
        for hand in hands:
            self.hovered = self.pick(hand.cursor)
            if self.hovered is not None:
                break

        if self.selected is None:
            for hand in hands:
                if hand.just_pinched:
                    card = self.pick(hand.cursor)
                    if card is not None:
                        self.start_drag(hand, card)
                        break
        elif self.mode == "DRAG":
            owner = hand_map.get(self.owner_key or "")
            if owner is None or not owner.pinched:
                self.release(owner.cursor if owner is not None else None)
            else:
                target = owner.cursor + self.drag_offset
                delta = target - self.selected.center
                self.selected.velocity = delta / max(dt, 1e-3)
                self.selected.center += delta * 0.48
                self.grab_distance = max(self.grab_distance, float(np.linalg.norm(owner.cursor - self.grab_origin)))
                others = [hand for hand in active if hand.key != owner.key]
                if others:
                    self.start_transform(owner, others[0])
        elif self.mode == "TRANSFORM":
            pair = [hand_map.get(key) for key in self.transform_pair]
            pair = [hand for hand in pair if hand is not None and hand.pinched]
            if len(pair) == 2:
                first, second = pair
                distance = max(float(np.linalg.norm(second.cursor - first.cursor)), 20.0)
                current_angle = vector_angle(first.cursor, second.cursor)
                midpoint = (first.cursor + second.cursor) / 2.0
                self.selected.scale = clamp(self.transform_scale * distance / self.transform_distance, 0.58, 1.7)
                self.selected.angle = self.transform_card_angle + current_angle - self.transform_angle
                self.selected.center += (midpoint + self.transform_offset - self.selected.center) * 0.42
                self.selected.velocity = (midpoint - self.last_midpoint) / max(dt, 1e-3)
                self.selected.angular_velocity = (current_angle - self.transform_angle) / max(dt, 1e-3) * 0.08
                self.last_midpoint = midpoint.copy()
                self.grab_distance = max(self.grab_distance, 40.0)
            elif len(pair) == 1:
                owner = pair[0]
                self.mode = "DRAG"
                self.owner_key = owner.key
                self.drag_offset = self.selected.center - owner.cursor
                self.grab_origin = owner.cursor.copy()
                self.grab_distance = 40.0
            else:
                self.release(None, allow_tap=False)

        for card in self.cards:
            card.update(dt, self.width, self.height, card is self.selected)
        alive: List[Particle] = []
        for particle in self.particles:
            particle.life -= dt
            if particle.life > 0:
                particle.position += particle.velocity * dt
                particle.velocity *= math.pow(0.94, dt * 60.0)
                alive.append(particle)
        self.particles = alive

    @staticmethod
    def draw_icon(frame: np.ndarray, card: SpatialCard, center: Tuple[int, int]) -> None:
        x, y = center
        color = card.accent
        if card.icon == "photo":
            cv2.rectangle(frame, (x - 29, y - 22), (x + 29, y + 20), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x + 15, y - 10), 6, color, -1, cv2.LINE_AA)
            mountains = np.array([[x - 24, y + 15], [x - 7, y - 5], [x + 5, y + 7], [x + 17, y - 1], [x + 26, y + 15]])
            cv2.polylines(frame, [mountains], False, color, 2, cv2.LINE_AA)
        elif card.icon == "music":
            cv2.line(frame, (x + 12, y - 25), (x + 12, y + 10), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x + 12, y - 25), (x + 32, y - 19), color, 4, cv2.LINE_AA)
            cv2.circle(frame, (x, y + 15), 10, color, -1, cv2.LINE_AA)
        elif card.icon == "notes":
            for offset in (-17, 0, 17):
                cv2.line(frame, (x - 27, y + offset), (x + 29, y + offset), color, 3, cv2.LINE_AA)
        else:
            cv2.ellipse(frame, (x, y), (35, 15), -18, 0, 360, color, 2, cv2.LINE_AA)
            cv2.ellipse(frame, (x, y), (15, 35), 28, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)

    def draw_card(self, frame: np.ndarray, card: SpatialCard, now: float) -> None:
        corners = card.corners().astype(np.int32)
        is_selected = card is self.selected
        is_hovered = card is self.hovered
        pulse = card.pulse * (0.55 + 0.45 * math.sin(now * 12.0))

        glow = frame.copy()
        cv2.polylines(glow, [corners], True, card.accent, 14 if is_selected else 8, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.13 + pulse * 0.15, frame, 0.87 - pulse * 0.15, 0, frame)
        panel = frame.copy()
        cv2.fillConvexPoly(panel, corners, (35, 31, 44) if is_selected else (24, 23, 34), cv2.LINE_AA)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
        cv2.polylines(frame, [corners], True, card.accent, 4 if is_selected else (3 if is_hovered else 2), cv2.LINE_AA)
        for corner in corners:
            cv2.circle(frame, tuple(corner), 4 if is_selected else 3, card.accent, -1, cv2.LINE_AA)

        center = point(card.center)
        icon_center = (center[0] - int(card.size[0] * 0.27), center[1] - 8)
        self.draw_icon(frame, card, icon_center)
        text_x = center[0] - int(card.size[0] * 0.02)
        cv2.putText(frame, card.title, (text_x, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72 * card.scale, (244, 246, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, card.subtitle, (text_x, center[1] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * card.scale, card.accent, 1, cv2.LINE_AA)

    @staticmethod
    def draw_hand(frame: np.ndarray, hand: HandState) -> None:
        if not hand.landmarks:
            return
        overlay = frame.copy()
        for start, end in HAND_CONNECTIONS:
            cv2.line(overlay, hand.landmarks[start], hand.landmarks[end], (255, 195, 85), 2, cv2.LINE_AA)
        for index, landmark in enumerate(hand.landmarks):
            cv2.circle(overlay, landmark, 6 if index in (4, 8) else 3, (240, 95, 255) if index in (4, 8) else (110, 240, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    @staticmethod
    def draw_cursor(frame: np.ndarray, hand: HandState) -> None:
        center = point(hand.cursor)
        color = (235, 90, 255) if hand.pinched else (90, 245, 255)
        radius = 10 if hand.pinched else 16
        glow = frame.copy()
        cv2.circle(glow, center, radius + 11, color, 3, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.28, frame, 0.72, 0, frame)
        cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
        if hand.pinched:
            cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
        else:
            cv2.line(frame, (center[0] - 24, center[1]), (center[0] - 10, center[1]), color, 1, cv2.LINE_AA)
            cv2.line(frame, (center[0] + 10, center[1]), (center[0] + 24, center[1]), color, 1, cv2.LINE_AA)
            cv2.line(frame, (center[0], center[1] - 24), (center[0], center[1] - 10), color, 1, cv2.LINE_AA)
            cv2.line(frame, (center[0], center[1] + 10), (center[0], center[1] + 24), color, 1, cv2.LINE_AA)
        strength = clamp(1.0 - (hand.pinch_ratio - 0.25) / 0.7, 0.0, 1.0)
        cv2.ellipse(frame, center, (radius + 6, radius + 6), -90, 0, int(360 * strength), color, 2, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, hands: List[HandState], fps: float, mouse_mode: bool, now: float) -> np.ndarray:
        trail_layer = frame.copy()
        for trail in self.trails.values():
            trail_points = list(trail)
            for index in range(1, len(trail_points)):
                strength = index / max(len(trail_points) - 1, 1)
                cv2.line(trail_layer, point(trail_points[index - 1]), point(trail_points[index]), (255, int(120 + strength * 110), int(70 + strength * 150)), max(1, int(strength * 5)), cv2.LINE_AA)
        cv2.addWeighted(trail_layer, 0.34, frame, 0.66, 0, frame)

        for particle in self.particles:
            cv2.circle(frame, point(particle.position), max(1, int(4 * clamp(particle.life / 0.8, 0, 1))), particle.color, -1, cv2.LINE_AA)
        for card in self.cards:
            self.draw_card(frame, card, now)
        for hand in hands:
            self.draw_hand(frame, hand)
            self.draw_cursor(frame, hand)

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 18), (535, 83), (13, 15, 24), -1)
        cv2.rectangle(overlay, (self.width - 220, 18), (self.width - 20, 83), (13, 15, 24), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
        cv2.putText(frame, "SPATIAL CANVAS", (38, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (245, 247, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "MEDIAPIPE // INTERACTION LAB", (38, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 190, 90), 1, cv2.LINE_AA)
        cv2.putText(frame, "%3.0f FPS" % fps, (self.width - 190, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 247, 255), 1, cv2.LINE_AA)
        label = "MOUSE PREVIEW" if mouse_mode else "%d HAND%s" % (len(hands), "" if len(hands) == 1 else "S")
        cv2.putText(frame, label, (self.width - 190, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 230, 255), 1, cv2.LINE_AA)

        if self.mode == "TRANSFORM":
            status = "TWO-HAND TRANSFORM // SCALE + ROTATE"
        elif self.mode == "DRAG" and self.selected is not None:
            status = "GRABBED // MOVE AND RELEASE %s" % self.selected.title
        elif self.hovered is not None:
            status = "PINCH TO GRAB // %s" % self.hovered.title
        elif not hands:
            status = "SHOW A HAND TO ENTER THE SPACE"
        else:
            status = "MOVE INDEX FINGER // PINCH THUMB + INDEX"
        text_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(frame, status, (max(20, (self.width - text_width) // 2), self.height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 232, 245), 1, cv2.LINE_AA)
        return frame


def cinematic_background(camera_frame: np.ndarray, now: float) -> np.ndarray:
    height, width = camera_frame.shape[:2]
    blurred = cv2.GaussianBlur(camera_frame, (0, 0), 7.0)
    tint = np.zeros_like(blurred)
    tint[:, :, 0], tint[:, :, 1], tint[:, :, 2] = 42, 20, 34
    frame = cv2.addWeighted(blurred, 0.34, tint, 0.66, 0)
    grid = frame.copy()
    horizon = int(height * 0.60)
    for y in range(horizon, height, max(24, height // 18)):
        cv2.line(grid, (0, y), (width, y), (75, 62, 92), 1, cv2.LINE_AA)
    for x in range(-width, width * 2, max(90, width // 10)):
        cv2.line(grid, (width // 2, horizon), (x, height), (70, 58, 88), 1, cv2.LINE_AA)
    cv2.addWeighted(grid, 0.34, frame, 0.66, 0, frame)
    sweep_x = int((now * 70) % (width + 300)) - 150
    sweep = frame.copy()
    cv2.line(sweep, (sweep_x, 0), (sweep_x + 180, height), (100, 80, 140), 2, cv2.LINE_AA)
    cv2.addWeighted(sweep, 0.14, frame, 0.86, 0, frame)
    return frame


def synthetic_background(width: int, height: int, now: float) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    frame[:, :, 0] = np.clip(28 + gradient * 34, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(12 + gradient * 12, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(25 + gradient * 20, 0, 255).astype(np.uint8)
    for index in range(28):
        x = int((index * 173 + now * (8 + index % 5)) % width)
        y = int((index * 97) % height)
        cv2.circle(frame, (x, y), 1 + index % 2, (95, 75, 125), -1, cv2.LINE_AA)
    return cinematic_background(frame, now)


class MousePreview:
    def __init__(self, width: int, height: int) -> None:
        self.position = np.array([width / 2, height / 2], dtype=np.float32)
        self.pressed = False
        self.was_pressed = False

    def callback(self, _event, x: int, y: int, flags: int, _parameter) -> None:
        self.position[:] = (x, y)
        self.pressed = bool(flags & cv2.EVENT_FLAG_LBUTTON)

    def hand(self) -> HandState:
        state = HandState("Mouse", self.position.copy(), [], self.pressed, self.pressed and not self.was_pressed, self.was_pressed and not self.pressed, 0.25 if self.pressed else 1.0)
        self.was_pressed = self.pressed
        return state


def configure_window(fullscreen: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)


def handle_key(key: int, scene: SpatialScene, fullscreen: bool) -> Tuple[bool, bool]:
    if key in (ord("q"), 27):
        return False, fullscreen
    if key == ord("r"):
        scene.reset()
    elif key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if not fullscreen:
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    return True, fullscreen


def run_mouse_preview(fullscreen: bool) -> int:
    width, height = 1280, 720
    scene = SpatialScene(width, height)
    mouse = MousePreview(width, height)
    configure_window(fullscreen)
    cv2.setMouseCallback(WINDOW_NAME, mouse.callback)
    previous = time.perf_counter()
    fps = 0.0
    running = True
    while running:
        now = time.perf_counter()
        dt = clamp(now - previous, 1.0 / 240.0, 0.05)
        previous = now
        fps = (1.0 / dt) if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1
        frame = synthetic_background(width, height, now)
        hands = [mouse.hand()]
        scene.update(hands, dt)
        scene.draw(frame, hands, fps, True, now)
        cv2.imshow(WINDOW_NAME, frame)
        running, fullscreen = handle_key(cv2.waitKey(1) & 0xFF, scene, fullscreen)
    cv2.destroyAllWindows()
    return 0


def run_camera_demo(model_path: Path, camera_index: int, fullscreen: bool) -> int:
    camera = open_camera(camera_index)
    if not camera.isOpened():
        camera.release()
        print(
            "Cannot open camera. Allow Codex or Terminal in System Settings > Privacy & Security > Camera.\n"
            "Tip: run with --mouse to preview without a camera.",
            file=sys.stderr,
        )
        return 2

    store = ResultStore()

    def on_result(result, _output_image, timestamp: int) -> None:
        store.update(result, timestamp)

    tracker = PinchTracker()
    scene: Optional[SpatialScene] = None
    hands: List[HandState] = []
    processed_timestamp = -1
    last_input_timestamp = -1
    previous = time.perf_counter()
    fps = 0.0
    configure_window(fullscreen)
    try:
        with create_recognizer(model_path, on_result) as recognizer:
            running = True
            while running:
                ok, raw_frame = camera.read()
                if not ok:
                    print("Camera opened, but no frame could be read.", file=sys.stderr)
                    return 3
                raw_frame = cv2.flip(raw_frame, 1)
                height, width = raw_frame.shape[:2]
                if scene is None:
                    scene = SpatialScene(width, height)

                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = time.monotonic_ns() // 1_000_000
                if timestamp <= last_input_timestamp:
                    timestamp = last_input_timestamp + 1
                last_input_timestamp = timestamp
                recognizer.recognize_async(media_image, timestamp)

                result, result_timestamp = store.get()
                if result_timestamp != processed_timestamp:
                    hands = tracker.update(result, width, height)
                    processed_timestamp = result_timestamp

                now = time.perf_counter()
                dt = clamp(now - previous, 1.0 / 240.0, 0.05)
                previous = now
                fps = (1.0 / dt) if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1
                frame = cinematic_background(raw_frame, now)
                scene.update(hands, dt)
                scene.draw(frame, hands, fps, False, now)
                cv2.imshow(WINDOW_NAME, frame)
                running, fullscreen = handle_key(cv2.waitKey(1) & 0xFF, scene, fullscreen)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def self_check() -> int:
    width, height = 1280, 720
    scene = SpatialScene(width, height)
    origin = scene.cards[0].center.copy()
    scene.update([HandState("Test", origin, [], True, True, False, 0.25)], 1.0 / 60.0)
    moved = origin + np.array([80, 40], dtype=np.float32)
    scene.update([HandState("Test", moved, [], True, False, False, 0.25)], 1.0 / 60.0)
    scene.update([HandState("Test", moved, [], False, False, True, 1.0)], 1.0 / 60.0)
    frame = synthetic_background(width, height, 1.0)
    scene.draw(frame, [], 60.0, True, 1.0)
    if frame.shape != (height, width, 3) or scene.selected is not None:
        raise RuntimeError("Spatial scene self-check failed")
    print("OK: spatial scene rendered and grab/drag/release completed.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="path to gesture model")
    parser.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    parser.add_argument("--mouse", action="store_true", help="hold left mouse button to simulate a pinch")
    parser.add_argument("--check", action="store_true", help="run a headless interaction/render check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            return self_check()
        if args.mouse:
            return run_mouse_preview(args.fullscreen)
        ensure_model(args.model)
        return run_camera_demo(args.model, args.camera, args.fullscreen)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("Error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
