"""Platform-neutral gesture state machine for desktop spatial control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


Point = Tuple[float, float]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


@dataclass(frozen=True)
class ScreenArea:
    """A rectangle in virtual-desktop coordinates."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class HandObservation:
    """The small, serializable subset of a MediaPipe result we need."""

    x: float
    y: float
    index_pinch_ratio: float
    middle_pinch_ratio: float
    gesture: str = ""
    handedness: str = ""
    landmarks: Tuple[Point, ...] = ()


class ActionKind(str, Enum):
    PRIMARY_CLICK = "primary_click"
    SECONDARY_CLICK = "secondary_click"
    DRAG_BEGIN = "drag_begin"
    DRAG_MOVE = "drag_move"
    DRAG_END = "drag_end"
    ACTIVE_CHANGED = "active_changed"


@dataclass(frozen=True)
class DesktopAction:
    kind: ActionKind
    position: Optional[Point] = None
    active: Optional[bool] = None


@dataclass(frozen=True)
class InteractionFrame:
    position: Optional[Point]
    visible: bool
    active: bool
    mode: str
    feedback: Optional[str]
    actions: Tuple[DesktopAction, ...]
    landmarks: Tuple[Point, ...] = ()
    gesture: str = ""
    handedness: str = ""


class PinchLatch:
    """Hysteresis for one thumb/finger pinch."""

    def __init__(self, enter: float, release: float) -> None:
        self.enter = enter
        self.release = release
        self.active = False

    def update(self, ratio: float) -> Tuple[bool, bool, bool]:
        previous = self.active
        threshold = self.release if previous else self.enter
        self.active = ratio < threshold
        return self.active, self.active and not previous, previous and not self.active

    def reset(self) -> bool:
        was_active = self.active
        self.active = False
        return was_active


class DesktopGestureEngine:
    """Turns hand observations into deliberate desktop actions.

    A short index pinch clicks on release. Moving while pinched promotes the
    gesture to a drag. A thumb/middle-finger pinch performs the secondary
    action. Holding an open palm toggles the controller between active and
    paused states.
    """

    def __init__(
        self,
        screen: ScreenArea,
        *,
        active: bool = False,
        margin: float = 0.07,
        drag_distance: float = 24.0,
        double_click_seconds: float = 0.42,
        open_palm_seconds: float = 1.0,
    ) -> None:
        self.screen = screen
        self.active = active
        self.margin = margin
        self.drag_distance = drag_distance
        self.double_click_seconds = double_click_seconds
        self.open_palm_seconds = open_palm_seconds

        self.index_latch = PinchLatch(0.42, 0.62)
        self.middle_latch = PinchLatch(0.48, 0.68)
        self.pinch_owner: Optional[str] = None
        self.position: Optional[Point] = None
        self.index_origin: Optional[Point] = None
        self.dragging = False
        self.last_primary_time = -100.0
        self.last_primary_position: Optional[Point] = None
        self.open_palm_started: Optional[float] = None
        self.open_palm_latched = False

    def map_position(self, normalized: Point) -> Point:
        usable = max(0.01, 1.0 - 2.0 * self.margin)
        nx = clamp((normalized[0] - self.margin) / usable, 0.0, 1.0)
        ny = clamp((normalized[1] - self.margin) / usable, 0.0, 1.0)
        return (
            self.screen.left + nx * max(1, self.screen.width - 1),
            self.screen.top + ny * max(1, self.screen.height - 1),
        )

    def smooth_position(self, target: Point) -> Point:
        if self.position is None:
            self.position = target
            return target
        motion = distance(self.position, target)
        alpha = clamp(0.25 + motion / 240.0, 0.25, 0.72)
        self.position = (
            self.position[0] + (target[0] - self.position[0]) * alpha,
            self.position[1] + (target[1] - self.position[1]) * alpha,
        )
        return self.position

    def map_landmarks(self, observation: HandObservation, anchor: Point) -> Tuple[Point, ...]:
        """Map the detected skeleton to the desktop and keep its index tip on the cursor."""
        if not observation.landmarks:
            return ()
        mapped = tuple(self.map_position(point) for point in observation.landmarks)
        index_tip = mapped[8] if len(mapped) > 8 else self.map_position((observation.x, observation.y))
        offset_x = anchor[0] - index_tip[0]
        offset_y = anchor[1] - index_tip[1]
        return tuple((point[0] + offset_x, point[1] + offset_y) for point in mapped)

    def _toggle_from_open_palm(self, gesture: str, now: float) -> Optional[DesktopAction]:
        if gesture == "Open_Palm":
            if self.open_palm_started is None:
                self.open_palm_started = now
            if not self.open_palm_latched and now - self.open_palm_started >= self.open_palm_seconds:
                self.open_palm_latched = True
                self.active = not self.active
                return DesktopAction(ActionKind.ACTIVE_CHANGED, active=self.active)
        else:
            self.open_palm_started = None
            self.open_palm_latched = False
        return None

    def _cancel_gesture(self, actions: List[DesktopAction]) -> None:
        if self.dragging and self.position is not None:
            actions.append(DesktopAction(ActionKind.DRAG_END, self.position))
        self.index_latch.reset()
        self.middle_latch.reset()
        self.pinch_owner = None
        self.index_origin = None
        self.dragging = False

    def update(self, observation: Optional[HandObservation], now: float) -> InteractionFrame:
        actions: List[DesktopAction] = []
        feedback: Optional[str] = None

        if observation is None:
            self._cancel_gesture(actions)
            self.open_palm_started = None
            self.open_palm_latched = False
            return InteractionFrame(self.position, False, self.active, "idle", feedback, tuple(actions))

        position = self.smooth_position(self.map_position((observation.x, observation.y)))
        landmarks = self.map_landmarks(observation, position)
        toggle = self._toggle_from_open_palm(observation.gesture, now)
        if toggle is not None:
            self._cancel_gesture(actions)
            actions.append(toggle)
            feedback = "resumed" if self.active else "paused"

        index_active, index_started, index_released = self.index_latch.update(
            observation.index_pinch_ratio
        )
        middle_active, middle_started, middle_released = self.middle_latch.update(
            observation.middle_pinch_ratio
        )

        # Lock a pinch to the finger that started it. If both thresholds are
        # crossed in the same frame, the closer fingertip wins. The losing
        # latch is cancelled silently, so a noisy pose cannot emit two clicks.
        if self.pinch_owner == "index":
            self.middle_latch.reset()
            middle_active = middle_started = middle_released = False
        elif self.pinch_owner == "middle":
            self.index_latch.reset()
            index_active = index_started = index_released = False
        elif index_active and middle_active:
            if observation.index_pinch_ratio <= observation.middle_pinch_ratio:
                self.pinch_owner = "index"
                self.middle_latch.reset()
                middle_active = middle_started = middle_released = False
            else:
                self.pinch_owner = "middle"
                self.index_latch.reset()
                index_active = index_started = index_released = False
        elif index_started:
            self.pinch_owner = "index"
        elif middle_started:
            self.pinch_owner = "middle"

        if not self.active:
            self._cancel_gesture(actions)
            mode = "paused"
            return InteractionFrame(
                position,
                True,
                self.active,
                mode,
                feedback,
                tuple(actions),
                landmarks,
                observation.gesture,
                observation.handedness,
            )

        if index_started:
            self.index_origin = position
            feedback = "primary_armed"

        if index_active and self.index_origin is not None:
            if not self.dragging and distance(self.index_origin, position) >= self.drag_distance:
                self.dragging = True
                actions.append(DesktopAction(ActionKind.DRAG_BEGIN, self.index_origin))
                feedback = "drag_begin"
            if self.dragging:
                actions.append(DesktopAction(ActionKind.DRAG_MOVE, position))

        if index_released:
            if self.dragging:
                actions.append(DesktopAction(ActionKind.DRAG_END, position))
                feedback = "drag_end"
            elif self.index_origin is not None:
                click_position = self.index_origin
                actions.append(DesktopAction(ActionKind.PRIMARY_CLICK, click_position))
                is_double = (
                    now - self.last_primary_time <= self.double_click_seconds
                    and self.last_primary_position is not None
                    and distance(click_position, self.last_primary_position) <= 42.0
                )
                feedback = "double" if is_double else "primary"
                self.last_primary_time = now
                self.last_primary_position = click_position
            self.index_origin = None
            self.dragging = False
            self.pinch_owner = None

        if middle_started:
            feedback = "secondary_armed"
        if middle_released:
            actions.append(DesktopAction(ActionKind.SECONDARY_CLICK, position))
            feedback = "secondary"
            self.pinch_owner = None

        if self.dragging:
            mode = "drag"
        elif middle_active:
            mode = "secondary_pinch"
        elif index_active:
            mode = "primary_pinch"
        else:
            mode = "tracking"
        return InteractionFrame(
            position,
            True,
            self.active,
            mode,
            feedback,
            tuple(actions),
            landmarks,
            observation.gesture,
            observation.handedness,
        )
