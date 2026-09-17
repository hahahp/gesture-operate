"""Native desktop input adapters for Windows and macOS."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

from desktop_core import DesktopAction, Point


class DesktopBackend(ABC):
    """Small platform boundary used by the gesture controller."""

    @abstractmethod
    def primary_click(self, position: Point) -> None:
        raise NotImplementedError

    @abstractmethod
    def secondary_click(self, position: Point) -> None:
        raise NotImplementedError

    @abstractmethod
    def begin_drag(self, position: Point) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_drag(self, position: Point) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_drag(self, position: Point) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_cursor_hidden(self, hidden: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class DryRunBackend(DesktopBackend):
    """Records actions without touching the real desktop."""

    def __init__(self) -> None:
        self.events: List[Tuple[str, Point]] = []
        self.cursor_hidden = False

    def _record(self, name: str, position: Point) -> None:
        self.events.append((name, position))

    def primary_click(self, position: Point) -> None:
        self._record("primary_click", position)

    def secondary_click(self, position: Point) -> None:
        self._record("secondary_click", position)

    def begin_drag(self, position: Point) -> None:
        self._record("drag_begin", position)

    def move_drag(self, position: Point) -> None:
        self._record("drag_move", position)

    def end_drag(self, position: Point) -> None:
        self._record("drag_end", position)

    def set_cursor_hidden(self, hidden: bool) -> None:
        self.cursor_hidden = hidden

    def close(self) -> None:
        self.cursor_hidden = False


class WindowsBackend(DesktopBackend):
    """Win32 input adapter.

    The native cursor is hidden while a hand is controlling the spatial halo.
    Win32 still receives conventional mouse events, which is the most broadly
    compatible fallback for custom-drawn and legacy applications.
    """

    LEFT_DOWN = 0x0002
    LEFT_UP = 0x0004
    RIGHT_DOWN = 0x0008
    RIGHT_UP = 0x0010

    def __init__(self) -> None:
        import ctypes

        self.ctypes = ctypes
        self.user32 = ctypes.windll.user32
        self.dragging = False
        self.cursor_hidden = False
        try:
            self.user32.SetProcessDPIAware()
        except AttributeError:
            pass

    @staticmethod
    def _point(position: Point) -> Tuple[int, int]:
        return int(round(position[0])), int(round(position[1]))

    def _move(self, position: Point) -> None:
        x, y = self._point(position)
        if not self.user32.SetCursorPos(x, y):
            raise OSError("SetCursorPos failed")

    def _button(self, flag: int) -> None:
        self.user32.mouse_event(flag, 0, 0, 0, 0)

    def primary_click(self, position: Point) -> None:
        self._move(position)
        self._button(self.LEFT_DOWN)
        self._button(self.LEFT_UP)

    def secondary_click(self, position: Point) -> None:
        self._move(position)
        self._button(self.RIGHT_DOWN)
        self._button(self.RIGHT_UP)

    def begin_drag(self, position: Point) -> None:
        if self.dragging:
            return
        self._move(position)
        self._button(self.LEFT_DOWN)
        self.dragging = True

    def move_drag(self, position: Point) -> None:
        if self.dragging:
            self._move(position)

    def end_drag(self, position: Point) -> None:
        if not self.dragging:
            return
        self._move(position)
        self._button(self.LEFT_UP)
        self.dragging = False

    def set_cursor_hidden(self, hidden: bool) -> None:
        if hidden == self.cursor_hidden:
            return
        # ShowCursor uses a display counter. Bound the loop so another process
        # cannot leave this controller spinning on a corrupted counter.
        if hidden:
            for _ in range(32):
                if self.user32.ShowCursor(False) < 0:
                    break
        else:
            for _ in range(32):
                if self.user32.ShowCursor(True) >= 0:
                    break
        self.cursor_hidden = hidden

    def close(self) -> None:
        if self.dragging:
            self._button(self.LEFT_UP)
            self.dragging = False
        self.set_cursor_hidden(False)


class MacOSBackend(DesktopBackend):
    """Quartz input adapter for macOS."""

    def __init__(self, request_accessibility: bool = True) -> None:
        import ApplicationServices
        import Quartz

        self.ax = ApplicationServices
        self.q = Quartz
        self.dragging = False
        self.cursor_hidden = False
        self.last_position: Point = (0.0, 0.0)
        self.last_click_at = -100.0
        self.last_click_position: Point = (-10000.0, -10000.0)
        if request_accessibility:
            options = {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
            if not ApplicationServices.AXIsProcessTrustedWithOptions(options):
                print(
                    "Grant Accessibility permission in System Settings > "
                    "Privacy & Security > Accessibility.",
                    file=sys.stderr,
                )

    def _event(self, event_type, position: Point, button, click_count: int = 1) -> None:
        self.last_position = position
        event = self.q.CGEventCreateMouseEvent(None, event_type, position, button)
        if event is None:
            raise RuntimeError("CGEventCreateMouseEvent failed")
        self.q.CGEventSetIntegerValueField(
            event, self.q.kCGMouseEventClickState, click_count
        )
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def primary_click(self, position: Point) -> None:
        now = time.monotonic()
        close = (
            abs(position[0] - self.last_click_position[0]) <= 42
            and abs(position[1] - self.last_click_position[1]) <= 42
        )
        count = 2 if close and now - self.last_click_at <= 0.42 else 1
        self._event(self.q.kCGEventLeftMouseDown, position, self.q.kCGMouseButtonLeft, count)
        self._event(self.q.kCGEventLeftMouseUp, position, self.q.kCGMouseButtonLeft, count)
        self.last_click_at = now
        self.last_click_position = position

    def secondary_click(self, position: Point) -> None:
        self._event(self.q.kCGEventRightMouseDown, position, self.q.kCGMouseButtonRight)
        self._event(self.q.kCGEventRightMouseUp, position, self.q.kCGMouseButtonRight)

    def begin_drag(self, position: Point) -> None:
        if self.dragging:
            return
        self._event(self.q.kCGEventLeftMouseDown, position, self.q.kCGMouseButtonLeft)
        self.dragging = True

    def move_drag(self, position: Point) -> None:
        if self.dragging:
            self._event(self.q.kCGEventLeftMouseDragged, position, self.q.kCGMouseButtonLeft)

    def end_drag(self, position: Point) -> None:
        if not self.dragging:
            return
        self._event(self.q.kCGEventLeftMouseUp, position, self.q.kCGMouseButtonLeft)
        self.dragging = False

    def set_cursor_hidden(self, hidden: bool) -> None:
        if hidden == self.cursor_hidden:
            return
        display = self.q.CGMainDisplayID()
        if hidden:
            self.q.CGDisplayHideCursor(display)
        else:
            self.q.CGDisplayShowCursor(display)
        self.cursor_hidden = hidden

    def close(self) -> None:
        if self.dragging:
            self.end_drag(self.last_position)
        self.set_cursor_hidden(False)


def create_backend(*, dry_run: bool = False) -> DesktopBackend:
    if dry_run:
        return DryRunBackend()
    if sys.platform == "win32":
        return WindowsBackend()
    if sys.platform == "darwin":
        return MacOSBackend()
    raise RuntimeError("Desktop control currently supports Windows and macOS only")


def dispatch_action(backend: DesktopBackend, action: DesktopAction) -> None:
    if action.position is None:
        return
    if action.kind.value == "primary_click":
        backend.primary_click(action.position)
    elif action.kind.value == "secondary_click":
        backend.secondary_click(action.position)
    elif action.kind.value == "drag_begin":
        backend.begin_drag(action.position)
    elif action.kind.value == "drag_move":
        backend.move_drag(action.position)
    elif action.kind.value == "drag_end":
        backend.end_drag(action.position)
