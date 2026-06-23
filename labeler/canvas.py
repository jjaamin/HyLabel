from __future__ import annotations
import math
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

from .mask_manager import MaskManager

SNAP_DIST = 14      # view-space pixels for polygon first-point snap
CP_SNAP   = 12      # view-space pixels for control-point grab


class Mode(Enum):
    IDLE  = auto()
    PAN   = auto()
    DRAW  = auto()
    BRUSH = auto()


# ──────────────────────────────────────────────────────────────────────────────
# Mask overlay item
# ──────────────────────────────────────────────────────────────────────────────

class _MaskOverlayItem(QGraphicsItem):
    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self._w = w
        self._h = h
        self._img = QImage(w, h, QImage.Format.Format_ARGB32)
        self._img.fill(Qt.GlobalColor.transparent)
        self.setZValue(5)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.drawImage(0, 0, self._img)

    def refresh_region(self, rgba: np.ndarray, x: int, y: int) -> None:
        data = np.ascontiguousarray(rgba)
        h, w = data.shape[:2]
        sub = QImage(data.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        p = QPainter(self._img)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(x, y, sub)
        p.end()
        self.update(QRectF(x, y, w, h))

    def fill_all(self, rgba: np.ndarray) -> None:
        data = np.ascontiguousarray(rgba)
        h, w = data.shape[:2]
        self._img = QImage(data.data, w, h, w * 4,
                           QImage.Format.Format_RGBA8888).copy()
        self.update()

    def clear(self) -> None:
        self._img.fill(Qt.GlobalColor.transparent)
        self.update()


# ──────────────────────────────────────────────────────────────────────────────
# Canvas
# ──────────────────────────────────────────────────────────────────────────────

class ImageCanvas(QGraphicsView):
    """
    Signals
    -------
    annotation_committed()   Enter commits pending → MaskManager
    stroke_finished()        brush mouse release
    mode_changed(str)        "idle" | "pan" | "draw" | "brush"
    brush_size_changed(int)
    edit_changed()           brush or point-drag modified an existing annotation
    edit_cleared()           edit mode exited
    """

    annotation_committed = pyqtSignal(int)   # ann_id
    stroke_finished      = pyqtSignal()
    mode_changed         = pyqtSignal(str)
    brush_size_changed   = pyqtSignal(int)
    edit_changed         = pyqtSignal()
    edit_cleared         = pyqtSignal()
    undo_record          = pyqtSignal(object)  # dict pushed to window undo stack

    def __init__(self, parent=None) -> None:
        scene = QGraphicsScene()
        super().__init__(scene, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[_MaskOverlayItem] = None
        self._brush_ring: Optional[QGraphicsEllipseItem] = None

        self._mask_manager: Optional[MaskManager] = None
        self._cat_colors: Dict[int, Tuple[int, int, int]] = {}
        self._pending_mask: Optional[np.ndarray] = None

        # annotation edit state
        self._edit_ann_id: int = -1
        self._edit_mask: Optional[np.ndarray] = None

        # contour / control points
        self._contour_overlay: Optional[_MaskOverlayItem] = None  # pixel boundary
        self._cp_contours: List[List[Tuple[float, float]]] = []
        self._cp_path_items: List[QGraphicsPathItem] = []   # drag preview only
        self._cp_dot_items: List[List[QGraphicsEllipseItem]] = []
        self._dragging_cp: Tuple[int, int] = (-1, -1)

        self._mode = Mode.IDLE
        self._active_cat_id: int = -1
        self._brush_size: int = 20
        self._painting = False
        self._stroke_erase = False

        # polygon draft
        self._draft_pts: list = []
        self._draft_path: Optional[QGraphicsPathItem] = None
        self._draft_line: Optional[QGraphicsLineItem] = None
        self._draft_dot: Optional[QGraphicsEllipseItem] = None

        # gamma / faint-mode
        self._original_pixmap: Optional[QPixmap] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_enabled: bool = False
        self._faint_mode: bool = False

    # ── faint / gamma public API ─────────────────────────────────────────────

    def set_faint_mode(self, faint: bool) -> None:
        self._faint_mode = faint
        if self._overlay_item is not None:
            self._overlay_item.setOpacity(0.12 if faint else 1.0)

    def set_gamma_lut(self, lut: np.ndarray) -> None:
        self._gamma_lut = lut
        self._apply_pixmap_gamma()

    def set_gamma_enabled(self, enabled: bool) -> None:
        self._gamma_enabled = enabled
        self._apply_pixmap_gamma()

    def _apply_pixmap_gamma(self) -> None:
        if self._pixmap_item is None or self._original_pixmap is None:
            return
        if self._gamma_enabled and self._gamma_lut is not None:
            pm = self._gamma_apply(self._original_pixmap, self._gamma_lut)
        else:
            pm = self._original_pixmap
        self._pixmap_item.setPixmap(pm)

    def _gamma_apply(self, pixmap: QPixmap, lut: np.ndarray) -> QPixmap:
        img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(h * img.bytesPerLine())
        arr = np.frombuffer(ptr, dtype=np.uint8).copy().reshape(h, w, 4)
        # Format_RGB32 (little-endian): byte order is B, G, R, FF
        arr[:, :, 0] = lut[arr[:, :, 0]]
        arr[:, :, 1] = lut[arr[:, :, 1]]
        arr[:, :, 2] = lut[arr[:, :, 2]]
        out = QImage(arr.data, w, h, img.bytesPerLine(), QImage.Format.Format_RGB32)
        return QPixmap.fromImage(out.copy())

    # ── undo public API ───────────────────────────────────────────────────────

    def has_draft_points(self) -> bool:
        return bool(self._draft_pts)

    def undo_draw_point(self) -> None:
        """Remove the last polygon vertex (called by window on Ctrl+Z in DRAW mode)."""
        if not self._draft_pts:
            return
        self._draft_pts.pop()
        if not self._draft_pts:
            self._cancel_draw()
        else:
            self._update_draft_path()
            if self._draft_line:
                self.scene().removeItem(self._draft_line)
                self._draft_line = None

    def restore_pending_mask(self, mask: np.ndarray) -> None:
        """Restore pending mask from an undo snapshot."""
        if self._pending_mask is not None:
            self._pending_mask[:] = mask
            self._refresh_overlay_full()

    def refresh_edit_contour(self) -> None:
        """Refresh control-point dots after an external mask change (undo)."""
        if self._edit_ann_id >= 0:
            self._show_contour()

    # ── public API ────────────────────────────────────────────────────────────

    def load_image(self, path: str) -> Tuple[int, int]:
        self._cancel_draw()
        self._painting = False
        self._pending_mask = None
        self._edit_ann_id = -1
        self._edit_mask = None
        self._clear_contour()

        pixmap = QPixmap(path)
        self._original_pixmap = pixmap
        scene = self.scene()
        for item in (self._pixmap_item, self._overlay_item,
                     self._contour_overlay, self._brush_ring):
            if item is not None:
                scene.removeItem(item)

        self._pixmap_item = scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._apply_pixmap_gamma()
        w, h = pixmap.width(), pixmap.height()
        scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self._overlay_item = _MaskOverlayItem(w, h)
        scene.addItem(self._overlay_item)
        if self._faint_mode:
            self._overlay_item.setOpacity(0.12)

        self._contour_overlay = _MaskOverlayItem(w, h)
        self._contour_overlay.setZValue(8)   # above mask (5), below draft (20)
        scene.addItem(self._contour_overlay)

        self._brush_ring = QGraphicsEllipseItem()
        self._brush_ring.setZValue(100)
        self._brush_ring.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
        self._brush_ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._brush_ring.hide()
        scene.addItem(self._brush_ring)

        self._mask_manager = None
        return w, h

    def set_mask_manager(self, mgr: Optional[MaskManager],
                          cat_colors: Dict[int, Tuple[int, int, int]]) -> None:
        self._mask_manager = mgr
        self._cat_colors = cat_colors
        self._edit_ann_id = -1
        self._edit_mask = None
        self._clear_contour()
        if mgr:
            self._pending_mask = np.zeros((mgr.height, mgr.width), dtype=np.uint8)
        else:
            self._pending_mask = None
        if self._overlay_item is None:
            return
        if mgr:
            self._overlay_item.fill_all(mgr.full_rgba(cat_colors))
        else:
            self._overlay_item.clear()
        if self._contour_overlay:
            self._contour_overlay.clear()

    def update_cat_colors(self, cat_colors: Dict[int, Tuple[int, int, int]]) -> None:
        self._cat_colors = cat_colors
        self._refresh_overlay_full()

    def refresh_overlay(self) -> None:
        self._refresh_overlay_full()

    def set_active_category(self, cat_id: int) -> None:
        self._active_cat_id = cat_id

    def set_mode(self, mode: Mode) -> None:
        if mode == self._mode:
            return
        if self._mode == Mode.DRAW:
            self._cancel_draw()
        self._mode = mode
        self._painting = False
        if mode == Mode.PAN:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._apply_cursor()
        self.mode_changed.emit(mode.name.lower())

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(50, size))

    def fit_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ── annotation edit API ───────────────────────────────────────────────────

    def set_edit_annotation(self, ann_id: int, mask: np.ndarray) -> None:
        """Enter edit mode: brush and point-drag will modify this annotation."""
        self._discard_pending()
        self._edit_ann_id = ann_id
        self._edit_mask = mask
        self._show_contour()

    def clear_edit_annotation(self) -> None:
        if self._edit_ann_id < 0:
            return
        self._edit_ann_id = -1
        self._edit_mask = None
        self._clear_contour()
        self.edit_cleared.emit()

    @property
    def current_mode(self) -> str:
        return self._mode.name.lower()

    # ── Qt events ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._mode == Mode.DRAW:
            if event.button() == Qt.MouseButton.LeftButton:
                self._draw_click(self.mapToScene(event.position().toPoint()))
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_draw()
            return

        # Control-point drag takes priority over brushing
        if event.button() == Qt.MouseButton.LeftButton and self._cp_contours:
            sp = self.mapToScene(event.position().toPoint())
            ci, pi = self._find_control_point(sp)
            if ci >= 0:
                # Save undo snapshot before contour drag
                if self._edit_mask is not None:
                    self.undo_record.emit({
                        "type": "edit_stroke",
                        "ann_id": self._edit_ann_id,
                        "mask": self._edit_mask.copy(),
                    })
                self._dragging_cp = (ci, pi)
                # Switch to drag-preview polygon; hide pixel boundary
                for item in self._cp_path_items:
                    item.setVisible(True)
                if self._contour_overlay:
                    self._contour_overlay.clear()
                return

        if self._mode == Mode.BRUSH:
            if event.button() == Qt.MouseButton.LeftButton:
                self._save_brush_undo()
                self._painting = True
                self._stroke_erase = False
                self._do_paint(self.mapToScene(event.position().toPoint()))
            elif event.button() == Qt.MouseButton.RightButton:
                self._save_brush_undo()
                self._painting = True
                self._stroke_erase = True
                self._do_paint(self.mapToScene(event.position().toPoint()))
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.position().toPoint())

        # Control-point dragging
        ci, pi = self._dragging_cp
        if ci >= 0:
            self._move_control_point(ci, pi, sp)
            return

        if self._mode == Mode.DRAW and self._draft_pts:
            last = self._draft_pts[-1]
            if self._draft_line is None:
                pen = QPen(QColor("#FFFF00"), 1, Qt.PenStyle.DashLine)
                self._draft_line = self.scene().addLine(
                    last.x(), last.y(), sp.x(), sp.y(), pen)
                self._draft_line.setZValue(25)
            else:
                self._draft_line.setLine(last.x(), last.y(), sp.x(), sp.y())
            if self._draft_dot and len(self._draft_pts) >= 3:
                near = self._view_dist(sp, self._draft_pts[0]) < SNAP_DIST
                self._draft_dot.setBrush(
                    QBrush(QColor("#FFFF00")) if near else QBrush(Qt.BrushStyle.NoBrush))

        if self._mode == Mode.BRUSH:
            self._move_brush_ring(sp)
            if self._painting:
                self._do_paint(sp)

        # Cursor: show SizeAllCursor near control points
        if self._cp_contours:
            near_cp = self._find_control_point(sp)[0] >= 0
            if near_cp:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self._apply_cursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        ci, pi = self._dragging_cp
        if ci >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_cp = (-1, -1)
            self._commit_contour_edit()
            self._show_contour()  # restores pixel boundary, hides drag polygon
            return

        if self._mode == Mode.BRUSH:
            if self._painting and event.button() in (
                    Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._painting = False
                if self._edit_ann_id >= 0:
                    self._show_contour()  # refresh contour after brush stroke
                self.stroke_finished.emit()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._mode == Mode.DRAW and event.button() == Qt.MouseButton.LeftButton:
            if self._draft_pts:
                self._draft_pts.pop()
            if len(self._draft_pts) >= 3:
                self._complete_draw()
            else:
                self._cancel_draw()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._edit_ann_id >= 0:
                self.clear_edit_annotation()
                self.set_mode(Mode.IDLE)
                return
            if self._mode == Mode.DRAW:
                self._cancel_draw()
            self._discard_pending()
            self.set_mode(Mode.IDLE)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._edit_ann_id >= 0:
                # In draw mode with draft points → new polygon, exit edit first
                if self._mode == Mode.DRAW and len(self._draft_pts) >= 3:
                    self.clear_edit_annotation()
                else:
                    return
            if self._mode == Mode.DRAW:
                if len(self._draft_pts) >= 3:
                    self._complete_draw()
                else:
                    self._cancel_draw()
            self._commit_pending()
        elif key == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._mode != Mode.PAN:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                if self._mode != Mode.BRUSH:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key == Qt.Key.Key_F:
            self.fit_view()
        elif key == Qt.Key.Key_BracketLeft:
            self.set_brush_size(self._brush_size - 1)
            self.brush_size_changed.emit(self._brush_size)
        elif key == Qt.Key.Key_BracketRight:
            self.set_brush_size(self._brush_size + 1)
            self.brush_size_changed.emit(self._brush_size)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._mode != Mode.PAN:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self._apply_cursor()
        super().keyReleaseEvent(event)

    def enterEvent(self, event) -> None:
        if self._mode == Mode.BRUSH and self._brush_ring:
            self._brush_ring.show()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._brush_ring:
            self._brush_ring.hide()
        super().leaveEvent(event)

    # ── contour / control points ──────────────────────────────────────────────

    def _show_contour(self) -> None:
        """Pixel-perfect boundary overlay + draggable control point dots."""
        self._clear_contour()
        if self._edit_mask is None or self._contour_overlay is None:
            return

        contours = MaskManager.extract_cp_contours(self._edit_mask)
        if not contours:
            return
        self._cp_contours = contours

        dot_pen = QPen(QColor("#FFFF00"), 1.5)
        dot_pen.setCosmetic(True)
        dot_brush = QBrush(QColor("#FFFF00"))
        drag_pen = QPen(QColor("#FFFF00"), 1, Qt.PenStyle.DashLine)
        drag_pen.setCosmetic(True)
        CP_R = 5

        for cp_pts in contours:
            # Drag-preview polygon — hidden until a point is dragged
            path = QPainterPath()
            path.moveTo(cp_pts[0][0], cp_pts[0][1])
            for x, y in cp_pts[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
            path_item = self.scene().addPath(
                path, drag_pen, QBrush(Qt.BrushStyle.NoBrush))
            path_item.setZValue(32)
            path_item.setVisible(False)
            self._cp_path_items.append(path_item)

            # Control point dots at pixel centres (cp_pts already have +0.5)
            dots: List[QGraphicsEllipseItem] = []
            for x, y in cp_pts:
                dot = QGraphicsEllipseItem(-CP_R, -CP_R, CP_R * 2, CP_R * 2)
                dot.setPen(dot_pen)
                dot.setBrush(dot_brush)
                dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                dot.setPos(x, y)
                dot.setZValue(35)
                self.scene().addItem(dot)
                dots.append(dot)
            self._cp_dot_items.append(dots)

    def _clear_contour(self) -> None:
        if self._contour_overlay is not None:
            self._contour_overlay.clear()
        for item in self._cp_path_items:
            self.scene().removeItem(item)
        for dots in self._cp_dot_items:
            for dot in dots:
                self.scene().removeItem(dot)
        self._cp_path_items = []
        self._cp_dot_items = []
        self._cp_contours = []
        self._dragging_cp = (-1, -1)

    def _find_control_point(self, sp: QPointF) -> Tuple[int, int]:
        """Return (contour_idx, point_idx) of nearest control point within CP_SNAP, or (-1,-1)."""
        for ci, pts in enumerate(self._cp_contours):
            for pi, (x, y) in enumerate(pts):
                if self._view_dist(sp, QPointF(x, y)) < CP_SNAP:
                    return (ci, pi)
        return (-1, -1)

    def _move_control_point(self, ci: int, pi: int, sp: QPointF) -> None:
        if not self._mask_manager:
            return
        x = max(0.5, min(float(sp.x()), float(self._mask_manager.width  - 1) + 0.5))
        y = max(0.5, min(float(sp.y()), float(self._mask_manager.height - 1) + 0.5))
        self._cp_contours[ci][pi] = (x, y)

        # Update drag-preview polygon
        pts = self._cp_contours[ci]
        path = QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            path.lineTo(px, py)
        path.closeSubpath()
        self._cp_path_items[ci].setPath(path)

        self._cp_dot_items[ci][pi].setPos(x, y)

    def _commit_contour_edit(self) -> None:
        """Re-rasterize all edited contours into the annotation mask."""
        if self._edit_mask is None:
            return
        self._edit_mask[:] = 0
        for pts in self._cp_contours:
            if len(pts) >= 3:
                MaskManager.fill_polygon_on(self._edit_mask, pts)
        self._refresh_overlay_full()
        self.edit_changed.emit()

    # ── overlay helpers ───────────────────────────────────────────────────────

    def _refresh_overlay_region(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._mask_manager or not self._overlay_item:
            return
        pm = self._pending_mask[y1:y2, x1:x2] if self._pending_mask is not None else None
        rgba = self._mask_manager.rgba_region(
            x1, y1, x2, y2, self._cat_colors, pm, self._active_cat_id)
        self._overlay_item.refresh_region(rgba, x1, y1)

    def _refresh_overlay_full(self) -> None:
        if not self._mask_manager or not self._overlay_item:
            return
        rgba = self._mask_manager.full_rgba(
            self._cat_colors, self._pending_mask, self._active_cat_id)
        self._overlay_item.fill_all(rgba)

    # ── pending mask ──────────────────────────────────────────────────────────

    def _commit_pending(self) -> None:
        if self._pending_mask is None or not self._pending_mask.any():
            return
        if self._mask_manager is None or self._active_cat_id < 0:
            return
        ann_id = self._mask_manager.add_annotation(self._active_cat_id, self._pending_mask)
        self._pending_mask[:] = 0
        self._refresh_overlay_full()
        self.annotation_committed.emit(ann_id)

    def _discard_pending(self) -> None:
        if self._pending_mask is not None and self._pending_mask.any():
            self._pending_mask[:] = 0
            self._refresh_overlay_full()

    def _save_brush_undo(self) -> None:
        """Emit a snapshot of the current mask state before a brush stroke begins."""
        if self._edit_ann_id >= 0 and self._edit_mask is not None:
            self.undo_record.emit({
                "type": "edit_stroke",
                "ann_id": self._edit_ann_id,
                "mask": self._edit_mask.copy(),
            })
        elif self._pending_mask is not None:
            self.undo_record.emit({
                "type": "pending_brush",
                "mask": self._pending_mask.copy(),
            })

    # ── polygon draw ──────────────────────────────────────────────────────────

    def _draw_click(self, sp: QPointF) -> None:
        if len(self._draft_pts) >= 3:
            if self._view_dist(sp, self._draft_pts[0]) < SNAP_DIST:
                self._complete_draw()
                return
        self._draft_pts.append(sp)
        self._update_draft_path()

    def _update_draft_path(self) -> None:
        path = QPainterPath()
        if self._draft_pts:
            path.moveTo(self._draft_pts[0])
            for pt in self._draft_pts[1:]:
                path.lineTo(pt)
        pen = QPen(QColor("#FFFF00"), 2)
        if self._draft_path is None:
            self._draft_path = self.scene().addPath(
                path, pen, QBrush(Qt.BrushStyle.NoBrush))
            self._draft_path.setZValue(20)
            if self._draft_pts:
                fp = self._draft_pts[0]
                self._draft_dot = self.scene().addEllipse(
                    fp.x() - 5, fp.y() - 5, 10, 10,
                    QPen(QColor("#FFFF00"), 2), QBrush(Qt.BrushStyle.NoBrush))
                self._draft_dot.setZValue(21)
        else:
            self._draft_path.setPath(path)

    def _complete_draw(self) -> None:
        if self._active_cat_id < 0 or len(self._draft_pts) < 3:
            self._cancel_draw()
            return
        pts = [(p.x(), p.y()) for p in self._draft_pts]
        self._clear_draft()
        if self._mask_manager and self._overlay_item and self._pending_mask is not None:
            x1, y1, x2, y2 = MaskManager.fill_polygon_on(self._pending_mask, pts)
            if x2 > x1 and y2 > y1:
                self._refresh_overlay_region(x1, y1, x2, y2)
        self.set_mode(Mode.IDLE)

    def _cancel_draw(self) -> None:
        self._draft_pts.clear()
        self._clear_draft()

    def _clear_draft(self) -> None:
        for obj in (self._draft_path, self._draft_line, self._draft_dot):
            if obj is not None:
                self.scene().removeItem(obj)
        self._draft_path = None
        self._draft_line = None
        self._draft_dot = None

    # ── brush ─────────────────────────────────────────────────────────────────

    def _do_paint(self, sp: QPointF) -> None:
        if self._mask_manager is None or self._overlay_item is None:
            return

        ix, iy = int(round(sp.x())), int(round(sp.y()))
        w, h = self._mask_manager.width, self._mask_manager.height
        r = self._brush_size
        if ix + r < 0 or ix - r >= w or iy + r < 0 or iy - r >= h:
            return

        if self._edit_ann_id >= 0 and self._edit_mask is not None:
            # Edit mode: write directly to the existing annotation's mask
            if not self._stroke_erase:
                x1, y1, x2, y2 = MaskManager.paint_circle_on(self._edit_mask, ix, iy, r)
            else:
                x1, y1, x2, y2 = MaskManager.erase_circle_on(self._edit_mask, ix, iy, r)
            if x2 > x1 and y2 > y1:
                rgba = self._mask_manager.rgba_region(x1, y1, x2, y2, self._cat_colors)
                self._overlay_item.refresh_region(rgba, x1, y1)
            self.edit_changed.emit()
        else:
            # Normal mode: write to pending mask
            if self._pending_mask is None:
                return
            if not self._stroke_erase and self._active_cat_id < 0:
                return
            if not self._stroke_erase:
                x1, y1, x2, y2 = MaskManager.paint_circle_on(self._pending_mask, ix, iy, r)
            else:
                x1, y1, x2, y2 = MaskManager.erase_circle_on(self._pending_mask, ix, iy, r)
            if x2 > x1 and y2 > y1:
                self._refresh_overlay_region(x1, y1, x2, y2)

    def _move_brush_ring(self, sp: QPointF) -> None:
        if self._brush_ring is None:
            return
        r = self._brush_size
        self._brush_ring.setRect(sp.x() - r, sp.y() - r, 2 * r, 2 * r)
        self._brush_ring.show()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_cursor(self) -> None:
        if self._mode == Mode.BRUSH:
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif self._mode == Mode.DRAW:
            self.setCursor(Qt.CursorShape.CrossCursor)
            if self._brush_ring:
                self._brush_ring.hide()
        elif self._mode == Mode.PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self._brush_ring:
                self._brush_ring.hide()
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._brush_ring:
                self._brush_ring.hide()

    def _view_dist(self, a: QPointF, b: QPointF) -> float:
        va = self.mapFromScene(a)
        vb = self.mapFromScene(b)
        return math.hypot(va.x() - vb.x(), va.y() - vb.y())
