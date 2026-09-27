from __future__ import annotations
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

# Two interior points rather than one: a single mid point can only bow the
# curve one way, so S-shapes — lift the shadows, hold the highlights — were out
# of reach. The interior X positions are the defaults only; they are draggable.
DEFAULT_CTRL: List[Tuple[int, int]] = [(0, 0), (85, 85), (170, 170), (255, 255)]

# Closest two control points may sit on the X axis. The spline needs strictly
# increasing X, and points nearer than this are impossible to separate by mouse.
MIN_X_GAP = 6


def compute_lut(ctrl: List[Tuple[int, int]]) -> np.ndarray:
    """Return a 256-element uint8 LUT from (x_in, y_out) control points via cubic spline.

    Tolerates unsorted input and duplicate X — a saved curve from an older build
    or a hand-edited setting should shape the image, not raise.
    """
    by_x = {int(x): int(y) for x, y in sorted(ctrl, key=lambda p: p[0])}
    xs = np.array(sorted(by_x), dtype=float)
    if len(xs) < 2:
        return np.arange(256, dtype=np.uint8)
    ys = np.array([by_x[int(x)] for x in xs], dtype=float)
    cs = CubicSpline(xs, ys, bc_type="natural")
    return np.clip(cs(np.arange(256, dtype=float)), 0, 255).astype(np.uint8)


def serialize_ctrl(ctrl: List[Tuple[int, int]]) -> str:
    """Control points as "x,y;x,y;…" for QSettings."""
    return ";".join(f"{int(x)},{int(y)}" for x, y in ctrl)


def parse_ctrl(raw: object) -> List[Tuple[int, int]]:
    """Inverse of serialize_ctrl. Returns [] for anything unusable."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    pts: List[Tuple[int, int]] = []
    for chunk in raw.split(";"):
        x, _, y = chunk.partition(",")
        try:
            pts.append((int(x), int(y)))
        except ValueError:
            return []
    return pts if len(pts) >= 2 else []


class GammaCurveWidget(QWidget):
    """
    Interactive curve editor — four control points.

    The two endpoints keep X at 0 and 255 (they are the black and white points,
    so moving them along X would mean clipping, not tone mapping) and move only
    vertically. The two interior points move freely in both axes, which is what
    lets the curve take an S or an inverted S rather than only bowing.
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

    def _ix_from_wx(self, wx: float) -> int:
        m = self._MARGIN
        return max(0, min(255, int(round((wx - m) / self._cw() * 255))))

    def _is_endpoint(self, i: int) -> bool:
        return i == 0 or i == len(self._ctrl) - 1

    def _clamp_x(self, i: int, ix: int) -> int:
        """Keep point i strictly between its neighbours, by at least MIN_X_GAP.

        Without this the spline gets a repeated or out-of-order X and either
        raises or folds the curve back on itself.
        """
        low = self._ctrl[i - 1][0] + MIN_X_GAP
        high = self._ctrl[i + 1][0] - MIN_X_GAP
        if low > high:                       # no room left — pin to the middle
            return (self._ctrl[i - 1][0] + self._ctrl[i + 1][0]) // 2
        return max(low, min(high, ix))

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

        # Control points — the draggable interior ones stand out from the
        # endpoints, which only move vertically.
        for i, (cx, cy) in enumerate(self._ctrl):
            pt = self._to_w(cx, cy)
            r = float(self._DOT_R)
            interior = not self._is_endpoint(i)
            if i == self._dragging:
                p.setPen(QPen(QColor("#fff"), 2.0))
                p.setBrush(QBrush(QColor("#7cf")))
                r += 1.5
            else:
                p.setPen(QPen(QColor("#fff"), 1.5))
                p.setBrush(QBrush(QColor("#4af") if interior else QColor("#888")))
            p.drawEllipse(pt, r, r)

        # Readout for the point being dragged — with X free, the dot alone no
        # longer says where it sits.
        if self._dragging >= 0:
            dx, dy = self._ctrl[self._dragging]
            p.setPen(QPen(QColor("#ccc"), 1))
            p.drawText(m + 6, m + 16, f"in {dx}  →  out {dy}")

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
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging < 0:
            return
        i = self._dragging
        iy = self._iy_from_wy(event.position().y())
        if self._is_endpoint(i):
            ix = self._ctrl[i][0]            # black/white point: vertical only
        else:
            ix = self._clamp_x(i, self._ix_from_wx(event.position().x()))
        self._ctrl[i] = (ix, iy)
        self._lut = compute_lut(self._ctrl)
        self.update()
        self.curve_changed.emit(self._lut)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = -1
        self.update()                        # drop the drag readout


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

        hint = QLabel("가운데 두 점은 좌우로도 움직입니다  ·  양 끝점은 상하만  ·  G = on/off")
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
