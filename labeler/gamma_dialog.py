from __future__ import annotations
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

DEFAULT_CTRL: List[Tuple[int, int]] = [(0, 0), (128, 128), (255, 255)]


def compute_lut(ctrl: List[Tuple[int, int]]) -> np.ndarray:
    """Return a 256-element uint8 LUT from (x_in, y_out) control points via cubic spline."""
    pts = sorted(ctrl, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    cs = CubicSpline(xs, ys, bc_type="natural")
    return np.clip(cs(np.arange(256, dtype=float)), 0, 255).astype(np.uint8)


class GammaCurveWidget(QWidget):
    """
    Interactive curve editor — 3 control points with fixed X (0 / 128 / 255).
    Drag a dot vertically to reshape the tone-mapping curve.
    """

    curve_changed = pyqtSignal(object)   # emits np.ndarray (256-element LUT)

    _MARGIN = 24
    _DOT_R  = 6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ctrl: List[Tuple[int, int]] = list(DEFAULT_CTRL)
        self._lut: np.ndarray = compute_lut(self._ctrl)
        self._dragging: int = -1

    # ── public ────────────────────────────────────────────────────────────────

    def set_control_points(self, pts: List[Tuple[int, int]]) -> None:
        self._ctrl = [tuple(p) for p in pts]  # type: ignore[misc]
        self._lut = compute_lut(self._ctrl)
        self.update()

    def control_points(self) -> List[Tuple[int, int]]:
        return list(self._ctrl)

    def lut(self) -> np.ndarray:
        return self._lut.copy()

    def reset_default(self) -> None:
        self.set_control_points(list(DEFAULT_CTRL))
        self.curve_changed.emit(self._lut)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _cw(self) -> int:
        return self.width()  - 2 * self._MARGIN

    def _ch(self) -> int:
        return self.height() - 2 * self._MARGIN

    def _to_w(self, ix: int, iy: int) -> QPointF:
        m = self._MARGIN
        return QPointF(m + ix / 255.0 * self._cw(),
                       m + (1.0 - iy / 255.0) * self._ch())

    def _iy_from_wy(self, wy: float) -> int:
        m = self._MARGIN
        return max(0, min(255, int(round((1.0 - (wy - m) / self._ch()) * 255))))

    def _near_dot(self, pos: QPointF) -> int:
        """Return index of control point within click radius, or -1."""
        for i, (cx, cy) in enumerate(self._ctrl):
            pt = self._to_w(cx, cy)
            if (pos.x() - pt.x()) ** 2 + (pos.y() - pt.y()) ** 2 < (self._DOT_R + 6) ** 2:
                return i
        return -1

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self._MARGIN
        cw, ch = self._cw(), self._ch()

        # Background & border
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QPen(QColor("#333"), 1))
        for i in range(1, 4):
            p.drawLine(m + i * cw // 4, m, m + i * cw // 4, m + ch)
            p.drawLine(m, m + i * ch // 4, m + cw, m + i * ch // 4)
        p.setPen(QPen(QColor("#555"), 1))
        p.drawRect(m, m, cw, ch)

        # Identity diagonal
        p.setPen(QPen(QColor("#444"), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(self._to_w(0, 0).x()),   int(self._to_w(0, 0).y()),
                   int(self._to_w(255, 255).x()), int(self._to_w(255, 255).y()))

        # Tone curve
        path = QPainterPath()
        for ix in range(256):
            pt = self._to_w(ix, int(self._lut[ix]))
            path.moveTo(pt) if ix == 0 else path.lineTo(pt)
        p.setPen(QPen(QColor("#4af"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Control points
        for i, (cx, cy) in enumerate(self._ctrl):
            pt = self._to_w(cx, cy)
            r = self._DOT_R
            is_mid = (i == 1)
            p.setPen(QPen(QColor("#fff"), 1.5))
            p.setBrush(QBrush(QColor("#4af") if is_mid else QColor("#888")))
            p.drawEllipse(pt, float(r), float(r))

        # Axis labels
        p.setPen(QPen(QColor("#777"), 1))
        fm = p.fontMetrics()
        for label, ix, iy in [("0", 0, 0), ("128", 128, 0), ("255", 255, 0)]:
            x = int(self._to_w(ix, 0).x()) - fm.horizontalAdvance(label) // 2
            p.drawText(x, self.height() - 4, label)

        p.end()

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = self._near_dot(event.position())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging < 0:
            return
        cx = self._ctrl[self._dragging][0]
        iy = self._iy_from_wy(event.position().y())
        self._ctrl[self._dragging] = (cx, iy)
        self._lut = compute_lut(self._ctrl)
        self.update()
        self.curve_changed.emit(self._lut)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = -1


class GammaCurveDialog(QDialog):
    """Floating dialog that hosts the gamma curve editor."""

    lut_changed = pyqtSignal(object)   # forwarded from widget

    def __init__(self, ctrl: List[Tuple[int, int]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gamma Curve")
        self.setMinimumSize(360, 420)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)

        self._widget = GammaCurveWidget()
        self._widget.set_control_points(ctrl)
        self._widget.curve_changed.connect(self.lut_changed)

        hint = QLabel("Drag dots to adjust tone curve  ·  G = toggle on/off")
        hint.setStyleSheet("color: #888; font-size: 11px;")

        btn_reset = QPushButton("Reset to Default")
        btn_reset.clicked.connect(self._widget.reset_default)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)

        row = QHBoxLayout()
        row.addWidget(btn_reset)
        row.addStretch()
        row.addWidget(btn_close)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self._widget, 1)
        lay.addLayout(row)

    def control_points(self) -> List[Tuple[int, int]]:
        return self._widget.control_points()
