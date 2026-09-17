"""Transparent, click-through spatial halo rendered across the desktop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QWidget

from desktop_core import InteractionFrame, Point, ScreenArea


@dataclass
class Pulse:
    position: Point
    kind: str
    started: float


def virtual_desktop_geometry(application: QApplication) -> QRect:
    screens = application.screens()
    if not screens:
        return QRect(0, 0, 1280, 720)
    geometry = screens[0].geometry()
    for screen in screens[1:]:
        geometry = geometry.united(screen.geometry())
    return geometry


class SpatialHaloOverlay(QWidget):
    """A non-activating overlay that only paints interaction feedback."""

    def __init__(self, application: QApplication) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(None, flags)
        self.desktop_geometry = virtual_desktop_geometry(application)
        self.setGeometry(self.desktop_geometry)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame = InteractionFrame(None, False, False, "paused", None, ())
        self.pulses: List[Pulse] = []
        self.status_changed_at = time.monotonic()
        self.show()

    def screen_area(self) -> ScreenArea:
        rect = self.desktop_geometry
        return ScreenArea(rect.left(), rect.top(), rect.width(), rect.height())

    def set_interaction(self, frame: InteractionFrame) -> None:
        if frame.active != self.frame.active:
            self.status_changed_at = time.monotonic()
        self.frame = frame
        self.update()

    def add_pulse(self, kind: str, position: Optional[Point]) -> None:
        if position is None:
            return
        self.pulses.append(Pulse(position, kind, time.monotonic()))
        self.update()

    def _local(self, position: Point) -> QPointF:
        return QPointF(
            position[0] - self.desktop_geometry.left(),
            position[1] - self.desktop_geometry.top(),
        )

    @staticmethod
    def _accent(mode: str) -> QColor:
        if mode in ("secondary_pinch", "secondary"):
            return QColor(196, 92, 255)
        if mode == "paused":
            return QColor(150, 158, 176)
        return QColor(76, 226, 255)

    def _draw_halo(self, painter: QPainter, now: float) -> None:
        frame = self.frame
        if not frame.visible or frame.position is None:
            return
        center = self._local(frame.position)
        accent = self._accent(frame.mode)
        pinched = frame.mode in ("primary_pinch", "secondary_pinch", "drag")
        radius = 13.0 if pinched else 22.0
        breathing = 1.5 * (1.0 + math.sin(now * 4.2))
        radius += breathing if not pinched else 0.0

        glow = QRadialGradient(center, radius + 24.0)
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 72))
        glow.setColorAt(0.46, QColor(accent.red(), accent.green(), accent.blue(), 34))
        glow.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, radius + 24.0, radius + 24.0)

        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 72 if pinched else 16))
        painter.setPen(QPen(accent, 3.0))
        painter.drawEllipse(center, radius, radius)
        if frame.mode == "drag":
            painter.setBrush(accent)
            painter.drawEllipse(center, 5.0, 5.0)
        elif frame.mode == "secondary_pinch":
            painter.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
            painter.drawText(
                QRectF(center.x() - 14, center.y() - 14, 28, 28),
                Qt.AlignmentFlag.AlignCenter,
                "•••",
            )

    def _draw_pulses(self, painter: QPainter, now: float) -> None:
        alive: List[Pulse] = []
        for pulse in self.pulses:
            age = now - pulse.started
            if age >= 0.55:
                continue
            alive.append(pulse)
            progress = age / 0.55
            center = self._local(pulse.position)
            secondary = pulse.kind == "secondary"
            color = QColor(196, 92, 255) if secondary else QColor(76, 226, 255)
            color.setAlpha(int(220 * (1.0 - progress)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 3.0 * (1.0 - progress) + 1.0))
            radius = 13.0 + progress * 44.0
            painter.drawEllipse(center, radius, radius)
            if pulse.kind == "double":
                painter.drawEllipse(center, radius + 10.0, radius + 10.0)
        self.pulses = alive

    def _draw_status(self, painter: QPainter, now: float) -> None:
        active = self.frame.active
        recently_changed = now - self.status_changed_at < 3.5
        if active and not recently_changed:
            width, height = 18.0, 18.0
            label = ""
        else:
            width, height = 268.0, 38.0
            label = "ACTIVE  •  OPEN PALM TO PAUSE" if active else "PAUSED  •  HOLD OPEN PALM TO START"

        center_x = self.width() / 2.0
        rect = QRectF(center_x - width / 2.0, 18.0, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 18, 27, 198))
        painter.drawRoundedRect(rect, height / 2.0, height / 2.0)
        dot = QColor(76, 226, 255) if active else QColor(148, 154, 170)
        if label:
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(rect.left() + 19, rect.center().y()), 4.5, 4.5)
            painter.setPen(QColor(238, 241, 248))
            painter.setFont(QFont("Sans Serif", 9, QFont.Weight.DemiBold))
            painter.drawText(
                QRectF(rect.left() + 31, rect.top(), rect.width() - 38, rect.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label,
            )
        else:
            painter.setBrush(dot)
            painter.drawEllipse(rect.center(), 4.0, 4.0)

    def paintEvent(self, _event) -> None:
        now = time.monotonic()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_halo(painter, now)
        self._draw_pulses(painter, now)
        self._draw_status(painter, now)
