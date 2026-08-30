from __future__ import annotations
import math
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

from .mask_manager import MaskManager

SNAP_DIST = 14      # view-space pixels for polygon first-point snap
CP_SNAP   = 12      # view-space pixels for control-point grab
LASSO_SNAP_DIST = 5  # view-space pixels for lasso magnetic start-point snap

# Mask overlay opacity per faint level, cycled by the V key. Level 0 is the
# normal labelling view; the rest fade the overlay so the image underneath can
# be checked, one step stronger and one step weaker than the single faint
# setting this replaced (0.22).
FAINT_LEVELS = (0.8, 0.30, 0.08)


class Mode(Enum):
    IDLE   = auto()
    SELECT = auto()
    PAN    = auto()
    DRAW   = auto()
    LASSO  = auto()
    BRUSH  = auto()
    MAGIC  = auto()


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
# Control-point dots
# ──────────────────────────────────────────────────────────────────────────────

class _PointsItem(QGraphicsItem):
    """Draws every control-point dot of a contour set in ONE scene item.

    One QGraphicsEllipseItem per point does not scale: with CHAIN_APPROX_NONE a
    single annotation carries thousands of points, and each item flagged
    ItemIgnoresTransformations forces the view to recompute its transform on
    every pan/zoom. Here the dots are kept as plain coordinates and painted in a
    single pass, sized in device pixels so they stay constant on screen.
    """

    # Dot diameter tracks the zoom (radius_px is the size at 1:1). No floor on
    # the low end: a dense CHAIN_APPROX_NONE contour has a point on every
    # boundary pixel, so a fixed minimum width reads as a solid band tracing
    # the shape once zoomed out far enough — thinner is more correct there,
    # not a bug to guard against. MAX_DOT_PX still caps the high end so a dot
    # can't swallow the image when zoomed in close.
    MIN_DOT_PX = 0.0
    MAX_DOT_PX = 14.0

    def __init__(self, radius_px: float, color: QColor, z: int) -> None:
        super().__init__()
        self._contours: List[List[Tuple[float, float]]] = []
        self._polys: List[QPolygonF] = []      # cached, rebuilt only on change
        self._brects: List[QRectF] = []        # per-contour bounds, for culling
        self._radius_px = radius_px
        self._color = color
        self._rect = QRectF()
        self.setZValue(z)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ── data ──────────────────────────────────────────────────────────────
    def set_contours(self, contours: List[List[Tuple[float, float]]]) -> None:
        self._contours = contours
        self._rebuild()

    def clear(self) -> None:
        self._contours = []
        self._rebuild()

    def move_point(self, ci: int, pi: int, x: float, y: float) -> None:
        self._contours[ci][pi] = (x, y)
        if ci < len(self._polys):
            self._polys[ci][pi] = QPointF(x, y)
        self.update()

    def _rebuild(self) -> None:
        self.prepareGeometryChange()
        self._polys = []
        self._brects = []
        for c in self._contours:
            poly = QPolygonF([QPointF(x, y) for x, y in c])
            self._polys.append(poly)
            self._brects.append(poly.boundingRect())
        if self._brects:
            union = self._brects[0]
            for br in self._brects[1:]:
                union = union.united(br)
            # Pad generously: dot radius is in screen px, so its scene-space
            # extent grows without bound as the view zooms out.
            self._rect = union.adjusted(-32.0, -32.0, 32.0, 32.0)
        else:
            self._rect = QRectF()
        self.update()

    # ── style ─────────────────────────────────────────────────────────────
    def set_style(self, radius_px: float, color: QColor) -> None:
        self._radius_px = radius_px
        self._color = color
        self.update()

    # ── painting ──────────────────────────────────────────────────────────
    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._polys:
            return
        # A round-capped cosmetic pen draws each point as a dot whose diameter
        # is the pen width in *device* pixels, and the whole contour goes out in
        # one C++ call. Iterating points in Python to drawEllipse() each one
        # costs ~5us apiece, which at full contour density is tens of ms/frame.
        #
        # The pen stays cosmetic even though the dots grow with zoom: switching
        # to a scene-unit pen makes Qt stroke every point individually, which
        # measured 39ms/frame against 20ms here, and it loses the culling win at
        # high zoom. Scaling the width instead keeps the fast path.
        scale = abs(painter.transform().m11()) or 1.0
        width = min(self.MAX_DOT_PX,
                    max(self.MIN_DOT_PX, self._radius_px * 2.0 * scale))
        pen = QPen(self._color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setCosmetic(True)
        painter.setPen(pen)
        pad = width / scale
        exposed = option.exposedRect.adjusted(-pad, -pad, pad, pad)
        for poly, brect in zip(self._polys, self._brects):
            if exposed.intersects(brect):
                painter.drawPoints(poly)


def _disc_footprint_path(radius: int) -> QPainterPath:
    """Outline of exactly the pixels the brush fills at this radius.

    The old cursor was a smooth ellipse, which sits half a pixel off the
    staircase the brush actually paints — fine for a rough stroke, useless when
    placing a boundary pixel. This traces the real filled/empty edges instead,
    so the preview is the result.

    Returned centred on (0, 0) in scene units, so the caller only has to
    setPos() the snapped pixel centre as the mouse moves. Cheap enough to build
    on every brush-size change (0.2ms at the maximum radius) and free after.
    """
    fp = MaskManager.brush_footprint(radius)
    filled = np.pad(fp, 1)

    path = QPainterPath()
    # A cell boundary exists wherever a filled pixel meets an empty one. Pixel
    # (px, py) spans scene [px, px+1), so edge coordinates are already integral.
    off = fp.shape[0] // 2 + 1        # footprint centre + the 1px pad

    horiz = filled[1:, :] != filled[:-1, :]
    for y, x in zip(*np.nonzero(horiz)):
        path.moveTo(x - off, y + 1 - off)
        path.lineTo(x + 1 - off, y + 1 - off)

    vert = filled[:, 1:] != filled[:, :-1]
    for y, x in zip(*np.nonzero(vert)):
        path.moveTo(x + 1 - off, y - off)
        path.lineTo(x + 1 - off, y + 1 - off)

    return path


def _disc_fill_path(radius: int) -> QPainterPath:
    """Closed fill region for the same footprint _disc_footprint_path() traces.

    That one is disjoint edge segments — fine to stroke, but filling it draws
    nothing (a straight segment has no interior) and combining the two edge
    networks in one path would additionally stroke every fill rectangle's own
    border, striping the disc with unwanted lines. Kept separate instead.

    The footprint's rows are each a single contiguous run of filled pixels, so
    the exact filled region is just one rectangle per row — cheaper than a full
    boundary trace, and unambiguous since it uses the same "pixel (px, py)
    spans scene [px, px+1)" convention directly rather than round-tripping
    through contour extraction's pixel-center points.
    """
    fp = MaskManager.brush_footprint(radius)
    off = fp.shape[0] // 2
    path = QPainterPath()
    for y in range(fp.shape[0]):
        xs = np.nonzero(fp[y])[0]
        if xs.size == 0:
            continue
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        path.addRect(x0 - off, y - off, x1 - x0, 1)
    return path


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
    brush_size_step(int)     Shift+wheel: +1 / -1, same step curve as [ / ]
    edit_changed()           brush or point-drag modified an existing annotation
    edit_cleared()           edit mode exited
    """

    annotation_committed = pyqtSignal(int)   # ann_id
    stroke_finished      = pyqtSignal()
    mode_changed         = pyqtSignal(str)
    brush_size_step      = pyqtSignal(int)
    edit_changed         = pyqtSignal(int)   # ann_id
    edit_cleared         = pyqtSignal()
    undo_record          = pyqtSignal(object)  # dict pushed to window undo stack
    magic_requested      = pyqtSignal(object, object)  # (points, labels) → window runs SAM
    select_requested     = pyqtSignal(float, float, bool)  # scene x, y, ctrl-held → window picks a label

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

        self._img_w = 0
        self._img_h = 0
        self._updating_rect = False   # guards setSceneRect → resize reentry

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[_MaskOverlayItem] = None
        self._brush_ring: Optional[QGraphicsPathItem] = None
        self._brush_fill: Optional[QGraphicsPathItem] = None

        self._mask_manager: Optional[MaskManager] = None
        self._cat_colors: Dict[int, Tuple[int, int, int]] = {}
        self._pending_mask: Optional[np.ndarray] = None

        # annotation edit state
        self._edit_ann_id: int = -1
        self._edit_mask: Optional[np.ndarray] = None
        self._edit_cat_id: int = -1

        # contour / control points
        self._contour_overlay: Optional[_MaskOverlayItem] = None  # unused; kept idle for now
        self._cp_contours: List[List[Tuple[float, float]]] = []
        self._cp_arrays: List[np.ndarray] = []   # same data, (n,2) float for hit-test
        self._cp_path_items: List[QGraphicsPathItem] = []   # drag preview only
        self._cp_dots: Optional[_PointsItem] = None
        self._dragging_cp: Tuple[int, int] = (-1, -1)
        self._class_dots: Optional[_PointsItem] = None  # read-only class overview
        self._contours_visible: bool = True   # toggled by X

        self._mode = Mode.IDLE
        self._active_cat_id: int = -1
        self._brush_size: int = 20
        self._painting = False
        self._stroke_erase = False
        # Pre-stroke state, kept so the outline can be spliced at stroke end
        # instead of the authored vertices being thrown away wholesale.
        self._stroke_old_mask: Optional[np.ndarray] = None
        self._stroke_old_polygons: Optional[List] = None

        # polygon draft
        self._draft_pts: list = []
        self._draft_path: Optional[QGraphicsPathItem] = None

        # Lasso: freehand outline, filled on release
        self._lasso_pts: List[QPointF] = []
        self._lasso_item: Optional[QGraphicsPathItem] = None
        self._lasso_rubber: Optional[QGraphicsPathItem] = None
        self._lasso_start_dot: Optional[QGraphicsEllipseItem] = None
        self._lasso_path = QPainterPath()
        self._lasso_drawing = False
        self._lasso_straight = 0.0   # min scene px between vertices; 0 = free
        self._draft_line: Optional[QGraphicsLineItem] = None
        self._draft_dot: Optional[QGraphicsEllipseItem] = None
        self._pending_polygons: Optional[List[List[List[float]]]] = None

        # magic wand state
        self._magic_pts: List[Tuple[float, float]] = []
        self._magic_lbls: List[int] = []
        self._magic_masks: Optional[np.ndarray] = None   # (3, H, W) bool
        self._magic_mask_idx: int = 0
        self._magic_dot_items: List[QGraphicsEllipseItem] = []

        # gamma / faint-mode
        self._original_pixmap: Optional[QPixmap] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_enabled: bool = False
        self._faint_level: int = 0

        self._last_mouse_scene_pos = QPointF(0.0, 0.0)

    # ── faint / gamma public API ─────────────────────────────────────────────

    @property
    def faint_level(self) -> int:
        return self._faint_level

    @property
    def contours_visible(self) -> bool:
        return self._contours_visible

    def set_contours_visible(self, visible: bool) -> None:
        """Show or hide every outline dot layer (X). Hidden costs nothing to
        paint, so this doubles as a way to speed up a dense image."""
        self._contours_visible = visible
        for item in (self._cp_dots, self._class_dots):
            if item is not None:
                item.setVisible(visible)

    def set_faint_level(self, level: int) -> None:
        """Select an overlay opacity from FAINT_LEVELS. Wraps, so callers can
        just pass level + 1 to advance."""
        self._faint_level = level % len(FAINT_LEVELS)
        if self._overlay_item is not None:
            self._overlay_item.setOpacity(FAINT_LEVELS[self._faint_level])

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
            # Both the pixels being removed and those being added must be
            # repainted, so refresh the union of the two extents.
            rect = MaskManager.union_bbox(
                MaskManager.compute_bbox(self._pending_mask),
                MaskManager.compute_bbox(mask))
            self._pending_mask[:] = mask
            self._refresh_overlay_rect(rect)

    def refresh_edit_contour(self) -> None:
        """Refresh control-point dots after an external mask change (undo)."""
        if self._edit_ann_id >= 0:
            self._show_contour()

    # ── public API ────────────────────────────────────────────────────────────

    def load_image(self, path: str) -> Tuple[int, int]:
        self._cancel_draw()
        self._cancel_lasso()
        self._painting = False
        self._pending_mask = None
        self._edit_ann_id = -1
        self._edit_mask = None
        self._edit_cat_id = -1
        self._clear_contour()
        self.clear_class_contours()

        pixmap = QPixmap(path)
        self._original_pixmap = pixmap
        scene = self.scene()
        for item in (self._pixmap_item, self._overlay_item, self._contour_overlay,
                     self._brush_ring, self._brush_fill):
            if item is not None:
                scene.removeItem(item)

        self._pixmap_item = scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._apply_pixmap_gamma()
        w, h = pixmap.width(), pixmap.height()
        self._img_w, self._img_h = w, h
        scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_scene_rect()

        self._overlay_item = _MaskOverlayItem(w, h)
        scene.addItem(self._overlay_item)
        self._overlay_item.setOpacity(FAINT_LEVELS[self._faint_level])

        self._contour_overlay = _MaskOverlayItem(w, h)
        self._contour_overlay.setZValue(8)   # above mask (5), below draft (20)
        scene.addItem(self._contour_overlay)

        self._brush_fill = QGraphicsPathItem()
        self._brush_fill.setZValue(99)       # under the ring's outline stroke
        self._brush_fill.setPen(QPen(Qt.PenStyle.NoPen))
        self._brush_fill.hide()
        scene.addItem(self._brush_fill)

        self._brush_ring = QGraphicsPathItem()
        self._brush_ring.setZValue(100)
        ring_pen = QPen(Qt.GlobalColor.white, 1.2)
        ring_pen.setCosmetic(True)          # stays legible at any zoom
        self._brush_ring.setPen(ring_pen)
        self._brush_ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._brush_ring.hide()
        scene.addItem(self._brush_ring)
        self._brush_fp_radius = -1          # radius the cached paths were built for
        self._brush_fill_color: Optional[Tuple[int, int, int]] = None

        self._mask_manager = None
        return w, h

    def set_mask_manager(self, mgr: Optional[MaskManager],
                          cat_colors: Dict[int, Tuple[int, int, int]]) -> None:
        self._mask_manager = mgr
        self._cat_colors = cat_colors
        self._edit_ann_id = -1
        self._edit_mask = None
        self._edit_cat_id = -1
        self._pending_polygons = None
        self._clear_contour()
        self.clear_class_contours()
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

    def refresh_overlay(self, rect: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Repaint the mask overlay. Pass the affected (x1,y1,x2,y2) when known —
        omitting it forces a full-image composite, which is far more expensive."""
        if rect is None:
            self._refresh_overlay_full()
        else:
            self._refresh_overlay_rect(rect)

    def set_active_category(self, cat_id: int) -> None:
        self._active_cat_id = cat_id

    def set_mode(self, mode: Mode) -> None:
        if mode == self._mode:
            return
        if self._mode == Mode.DRAW:
            self._cancel_draw()
        if self._mode == Mode.LASSO:
            self._cancel_lasso()
        if self._mode == Mode.MAGIC and mode != Mode.MAGIC:
            self.clear_magic(keep_pending=False)
        self._mode = mode
        self._painting = False
        if mode == Mode.PAN:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._apply_cursor()
        # Update control point dot appearance on mode change
        if self._edit_ann_id >= 0 and self._cp_dots is not None:
            self._cp_dots.set_style(*self._cp_dot_style())
        self.mode_changed.emit(mode.name.lower())

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(88, size))
        if self._mode == Mode.BRUSH and self._brush_ring and self._brush_ring.isVisible():
            self._move_brush_ring(self._last_mouse_scene_pos)

    def fit_view(self) -> None:
        if not self._pixmap_item:
            return
        # Drop the padding before fitting, then let _update_scene_rect() decide
        # whether this zoom level still needs any.
        self.scene().setSceneRect(QRectF(0, 0, self._img_w, self._img_h))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_scene_rect()

    def zoom_by(self, factor: float) -> None:
        """Zoom, then resize the scrollable area to match the new zoom level."""
        self.scale(factor, factor)
        self._update_scene_rect()

    # ── scrollable area ───────────────────────────────────────────────────────

    def _update_scene_rect(self) -> None:
        """Pad the scene so any image pixel can be scrolled to the canvas centre.

        Unpadded, the scroll range stops at the image border, so edge pixels can
        only ever sit against the side of the viewport — the worst place to
        label them. The padding is empty scene space: no pixmap, no overlay, and
        the mask stays exactly w×h, so a brush stroke or polygon that strays
        into it is simply clipped rather than interrupted.

        Only the axis that does not currently fit gets padded, which keeps the
        scrollbars away at fit-view.
        """
        if self._pixmap_item is None or self._updating_rect:
            return
        w, h = self._img_w, self._img_h
        vis = self.mapToScene(self.viewport().rect()).boundingRect()
        # Half a viewport of slack is exactly enough to bring a corner pixel to
        # the centre, and no more. The half-pixel tolerance keeps fit-view —
        # where vis matches the image to within rounding — on the zero branch.
        mx = vis.width()  / 2.0 if vis.width()  < w - 0.5 else 0.0
        my = vis.height() / 2.0 if vis.height() < h - 0.5 else 0.0
        rect = QRectF(-mx, -my, w + 2 * mx, h + 2 * my)
        if rect == self.sceneRect():
            return
        self._updating_rect = True
        try:
            self.scene().setSceneRect(rect)
        finally:
            self._updating_rect = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scene_rect()

    # ── annotation edit API ───────────────────────────────────────────────────

    def set_edit_annotation(self, ann_id: int, mask: np.ndarray) -> None:
        """Enter edit mode: brush and point-drag will modify this annotation."""
        self._discard_pending()
        self.clear_class_contours()
        self._edit_ann_id = ann_id
        self._edit_mask = mask
        # Cached rather than looked up per brush stroke: get_annotation() is a
        # linear scan, and _do_paint() runs it on every mouse move otherwise.
        ann = self._mask_manager.get_annotation(ann_id) if self._mask_manager else None
        self._edit_cat_id = ann.cat_id if ann is not None else -1
        self._show_contour()

    def clear_edit_annotation(self) -> None:
        if self._edit_ann_id < 0:
            return
        self._edit_ann_id = -1
        self._edit_mask = None
        self._edit_cat_id = -1
        self._clear_contour()
        self.edit_cleared.emit()

    @property
    def current_mode(self) -> str:
        return self._mode.name.lower()

    @property
    def is_editing(self) -> bool:
        return self._edit_ann_id >= 0

    # ── Qt events ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # Space-held temporary pan: defer to view regardless of current mode
        if (self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
                and self._mode != Mode.PAN):
            super().mousePressEvent(event)
            return

        if self._mode == Mode.DRAW:
            if event.button() == Qt.MouseButton.LeftButton:
                self._draw_click(self.mapToScene(event.position().toPoint()))
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_draw()
            return

        if self._mode == Mode.LASSO:
            if event.button() == Qt.MouseButton.LeftButton:
                self._start_lasso(self.mapToScene(event.position().toPoint()))
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_lasso()
            return

        if self._mode == Mode.MAGIC:
            sp = self.mapToScene(event.position().toPoint())
            if event.button() == Qt.MouseButton.LeftButton:
                self._magic_click(sp, 1)
            elif event.button() == Qt.MouseButton.RightButton:
                self._magic_click(sp, 0)
            return

        # Control-point drag (disabled in Brush mode — use brush to edit instead)
        if (event.button() == Qt.MouseButton.LeftButton
                and self._cp_contours and self._mode != Mode.BRUSH):
            sp = self.mapToScene(event.position().toPoint())
            ci, pi = self._find_control_point(sp)
            if ci >= 0:
                # Save undo snapshot before contour drag
                if self._edit_mask is not None:
                    self.undo_record.emit({
                        "type": "edit_stroke",
                        "ann_id": self._edit_ann_id,
                        "mask": self._edit_mask.copy(),
                        "polygons": self._edit_polygons_snapshot(),
                    })
                self._dragging_cp = (ci, pi)
                # Switch to drag-preview polygon; hide pixel boundary
                for item in self._cp_path_items:
                    item.setVisible(True)
                if self._contour_overlay:
                    self._contour_overlay.clear()
                return

        # Plain click with the arrow tool picks whatever label is under the
        # cursor. Sits below the control-point branch on purpose, so dragging a
        # vertex of the already-selected label still wins over re-picking.
        if self._mode == Mode.SELECT:
            if event.button() == Qt.MouseButton.LeftButton:
                sp = self.mapToScene(event.position().toPoint())
                additive = bool(event.modifiers()
                                & Qt.KeyboardModifier.ControlModifier)
                self.select_requested.emit(sp.x(), sp.y(), additive)
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
        self._last_mouse_scene_pos = sp

        # Space-held temporary pan
        if (self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
                and self._mode != Mode.PAN):
            if self._brush_ring:
                self._brush_ring.hide()
            if self._brush_fill:
                self._brush_fill.hide()
            super().mouseMoveEvent(event)
            return

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

        if self._mode == Mode.LASSO and self._lasso_drawing:
            alt_held = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self._extend_lasso(sp, magnetic=not alt_held)

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
        # Space-held temporary pan
        if (self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
                and self._mode != Mode.PAN):
            super().mouseReleaseEvent(event)
            return

        ci, pi = self._dragging_cp
        if ci >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_cp = (-1, -1)
            self._commit_contour_edit()
            self._show_contour()  # restores pixel boundary, hides drag polygon
            return

        if self._mode == Mode.LASSO:
            if self._lasso_drawing and event.button() == Qt.MouseButton.LeftButton:
                self._complete_lasso()
            return

        if self._mode == Mode.BRUSH:
            if self._painting and event.button() in (
                    Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._painting = False
                if self._edit_ann_id >= 0:
                    if self._mask_manager is not None:
                        # Settle the bbox after a stroke that may have erased.
                        self._mask_manager.recompute_bbox(self._edit_ann_id)
                    self._splice_stroke_outline()
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
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Window owns the step curve and the magic-mode variant, so route
            # this through the same path as [ / ] instead of duplicating it.
            self.brush_size_step.emit(1 if delta > 0 else -1)
            event.accept()
            return
        self.zoom_by(1.15 if delta > 0 else 1 / 1.15)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._edit_ann_id >= 0:
                self.clear_edit_annotation()
                self.set_mode(Mode.IDLE)
                return
            if self._mode == Mode.DRAW:
                self._cancel_draw()
            if self._mode == Mode.LASSO:
                self._cancel_lasso()
            if self._mode == Mode.MAGIC:
                self.clear_magic(keep_pending=False)
            self._discard_pending()
            self.set_mode(Mode.IDLE)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._mode == Mode.MAGIC:
                self._commit_pending()
                self.clear_magic(keep_pending=True)
                return
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
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key == Qt.Key.Key_F:
            self.fit_view()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._mode != Mode.PAN:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self._apply_cursor()
        super().keyReleaseEvent(event)

    def enterEvent(self, event) -> None:
        if self._mode == Mode.BRUSH:
            if self._brush_ring:
                self._brush_ring.show()
            if self._brush_fill:
                self._brush_fill.show()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._brush_ring:
            self._brush_ring.hide()
        if self._brush_fill:
            self._brush_fill.hide()
        super().leaveEvent(event)

    # ── contour / control points ──────────────────────────────────────────────

    def _edit_contour_source(self) -> List[List[Tuple[float, float]]]:
        """Control points to expose for dragging.

        Prefer the annotation's own polygon. Those are the authored
        coordinates, and moving one vertex must leave every other vertex
        bit-identical when saved — re-deriving them from the mask would replace
        the whole outline with a pixel staircase (a 20-point hand-drawn polygon
        comes back as ~300 points, none of them the originals).

        Fall back to the mask only when there is no polygon to preserve, i.e.
        after a brush or magic-wand edit, which sets original_polygons to None.
        """
        if self._mask_manager is not None and self._edit_ann_id >= 0:
            ann = self._mask_manager.get_annotation(self._edit_ann_id)
            if ann is not None and ann.original_polygons:
                polys = [[(float(x), float(y)) for x, y in poly]
                         for poly in ann.original_polygons if len(poly) >= 3]
                if polys:
                    return polys
        return MaskManager.extract_cp_contours(self._edit_mask)

    def _show_contour(self) -> None:
        """Pixel-perfect boundary overlay + draggable control point dots."""
        self._clear_contour()
        if self._edit_mask is None or self._contour_overlay is None:
            return

        contours = self._edit_contour_source()
        if not contours:
            return
        self._cp_contours = contours
        self._cp_arrays = [np.asarray(c, dtype=np.float64) for c in contours]

        drag_pen = QPen(QColor("#FFFF00"), 1, Qt.PenStyle.DashLine)
        drag_pen.setCosmetic(True)

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
        r, color = self._cp_dot_style()
        self._cp_dots = _PointsItem(r, color, z=35)
        self._cp_dots.setVisible(self._contours_visible)
        self.scene().addItem(self._cp_dots)
        self._cp_dots.set_contours(contours)

    def _cp_dot_style(self) -> Tuple[float, QColor]:
        """Dot appearance — dimmer and larger while the brush is active."""
        if self._mode == Mode.BRUSH:
            return 4.0, QColor(255, 255, 0, 110)
        return 3.0, QColor("#FFFF00")

    def _brush_target_color(self) -> Tuple[int, int, int]:
        """Class color the brush is currently painting with — the annotation
        being edited if one is active, otherwise the active class for a new
        pending shape. Used for the brush cursor's fill preview."""
        if self._edit_ann_id >= 0:
            return self._cat_colors.get(self._edit_cat_id, (255, 255, 0))
        return self._cat_colors.get(self._active_cat_id, (255, 255, 0))

    def _clear_contour(self) -> None:
        if self._contour_overlay is not None:
            self._contour_overlay.clear()
        for item in self._cp_path_items:
            self.scene().removeItem(item)
        if self._cp_dots is not None:
            self.scene().removeItem(self._cp_dots)
            self._cp_dots = None
        self._cp_path_items = []
        self._cp_contours = []
        self._cp_arrays = []
        self._dragging_cp = (-1, -1)

    def show_class_contours(self, masks: List[np.ndarray]) -> None:
        """Show read-only contour dots for all annotations of the selected class."""
        self.clear_class_contours()
        if not self._pixmap_item:
            return
        contours: List[List[Tuple[float, float]]] = []
        for mask in masks:
            contours.extend(MaskManager.extract_cp_contours(mask))
        if not contours:
            return
        self._class_dots = _PointsItem(3.0, QColor(255, 255, 255, 220), z=30)
        self._class_dots.setVisible(self._contours_visible)
        self.scene().addItem(self._class_dots)
        self._class_dots.set_contours(contours)

    def clear_class_contours(self) -> None:
        if self._class_dots is not None:
            self.scene().removeItem(self._class_dots)
            self._class_dots = None

    def _find_control_point(self, sp: QPointF) -> Tuple[int, int]:
        """Return (contour_idx, point_idx) of nearest control point within
        CP_SNAP, or (-1,-1).

        Runs on every mouse move, so it compares in scene space with numpy
        instead of calling mapFromScene() per point — with full-density contours
        that was two Qt calls per point per move.
        """
        if not self._cp_arrays:
            return (-1, -1)
        scale = abs(self.transform().m11()) or 1.0
        snap2 = (CP_SNAP / scale) ** 2      # view px → scene units
        sx, sy = sp.x(), sp.y()
        best = (-1, -1)
        best_d2 = snap2
        for ci, arr in enumerate(self._cp_arrays):
            if arr.size == 0:
                continue
            dx = arr[:, 0] - sx
            dy = arr[:, 1] - sy
            d2 = dx * dx + dy * dy
            pi = int(np.argmin(d2))
            if d2[pi] < best_d2:
                best_d2 = float(d2[pi])
                best = (ci, pi)
        return best

    def _move_control_point(self, ci: int, pi: int, sp: QPointF) -> None:
        if not self._mask_manager:
            return
        x = max(0.5, min(float(sp.x()), float(self._mask_manager.width  - 1) + 0.5))
        y = max(0.5, min(float(sp.y()), float(self._mask_manager.height - 1) + 0.5))
        self._cp_contours[ci][pi] = (x, y)
        self._cp_arrays[ci][pi, 0] = x
        self._cp_arrays[ci][pi, 1] = y

        # Update drag-preview polygon
        pts = self._cp_contours[ci]
        path = QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            path.lineTo(px, py)
        path.closeSubpath()
        self._cp_path_items[ci].setPath(path)

        if self._cp_dots is not None:
            self._cp_dots.move_point(ci, pi, x, y)

    def _commit_contour_edit(self) -> None:
        """Re-rasterize all edited contours into the annotation mask."""
        if self._edit_mask is None:
            return
        old_rect = None
        if self._mask_manager is not None:
            prev = self._mask_manager.get_annotation(self._edit_ann_id)
            if prev is not None:
                old_rect = prev.bbox
        self._edit_mask[:] = 0
        for pts in self._cp_contours:
            if len(pts) >= 3:
                MaskManager.fill_polygon_on(self._edit_mask, pts)
        if self._mask_manager is not None:
            # Mask was rebuilt from scratch — bbox must be recomputed before
            # any render, or culling would use the pre-edit extent.
            self._mask_manager.recompute_bbox(self._edit_ann_id)
            ann = self._mask_manager.get_annotation(self._edit_ann_id)
            self._refresh_overlay_rect(MaskManager.union_bbox(
                old_rect, ann.bbox if ann is not None else None))
        # Preserve edited contour points as polygon — no simplification on save
        if self._mask_manager is not None:
            ann = self._mask_manager.get_annotation(self._edit_ann_id)
            if ann is not None:
                ann.original_polygons = [
                    [[x, y] for x, y in pts]
                    for pts in self._cp_contours if len(pts) >= 3
                ]
        self.edit_changed.emit(self._edit_ann_id)

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

    def _refresh_overlay_rect(self,
                              rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Refresh a single region, clamped to the image. None is a no-op."""
        if rect is None or not self._mask_manager or not self._overlay_item:
            return
        x1, y1, x2, y2 = rect
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self._mask_manager.width, x2)
        y2 = min(self._mask_manager.height, y2)
        if x2 <= x1 or y2 <= y1:
            return
        self._refresh_overlay_region(x1, y1, x2, y2)

    # ── pending mask ──────────────────────────────────────────────────────────

    def _commit_pending(self) -> None:
        if self._pending_mask is None or not self._pending_mask.any():
            return
        if self._mask_manager is None or self._active_cat_id < 0:
            return
        rect = MaskManager.compute_bbox(self._pending_mask)
        ann_id = self._mask_manager.add_annotation(self._active_cat_id, self._pending_mask)
        if self._pending_polygons is not None:
            ann = self._mask_manager.get_annotation(ann_id)
            if ann is not None:
                ann.original_polygons = self._pending_polygons
            self._pending_polygons = None
        self._pending_mask[:] = 0
        self._refresh_overlay_rect(rect)
        self.annotation_committed.emit(ann_id)

    def _discard_pending(self) -> None:
        self._pending_polygons = None
        if self._pending_mask is not None and self._pending_mask.any():
            rect = MaskManager.compute_bbox(self._pending_mask)
            self._pending_mask[:] = 0
            self._refresh_overlay_rect(rect)

    def _edit_polygons_snapshot(self) -> Optional[List]:
        """Deep copy of the edited annotation's polygon, for the undo stack.

        The mask alone is not enough to undo an edit: original_polygons is what
        gets written on save, so restoring one without the other would leave the
        saved coordinates reflecting an edit the user already undid.
        """
        if self._mask_manager is None or self._edit_ann_id < 0:
            return None
        ann = self._mask_manager.get_annotation(self._edit_ann_id)
        if ann is None or ann.original_polygons is None:
            return None
        return [[[x, y] for x, y in poly] for poly in ann.original_polygons]

    def _save_brush_undo(self) -> None:
        """Emit a snapshot of the current mask state before a brush stroke begins."""
        if self._edit_ann_id >= 0 and self._edit_mask is not None:
            polygons = self._edit_polygons_snapshot()
            # Same snapshot feeds the splice at stroke end, which needs to know
            # both what the outline was and which pixels the stroke changed.
            self._stroke_old_mask = self._edit_mask.copy()
            self._stroke_old_polygons = polygons
            self.undo_record.emit({
                "type": "edit_stroke",
                "ann_id": self._edit_ann_id,
                "mask": self._edit_mask.copy(),
                "polygons": polygons,
            })
        elif self._pending_mask is not None:
            # Not editing an annotation, so there is no outline to splice.
            self._stroke_old_mask = None
            self._stroke_old_polygons = None
            self.undo_record.emit({
                "type": "pending_brush",
                "mask": self._pending_mask.copy(),
            })

    def _splice_stroke_outline(self) -> None:
        """Rebuild the edited outline at stroke end, keeping untouched vertices.

        _do_paint drops original_polygons as soon as the mask changes, because
        mid-stroke the mask is the only truthful description of the shape. Here
        the stroke is over, so the authored vertices the brush never reached can
        be put back and only the region it actually altered is left as a pixel
        staircase. Falls back to the mask-derived outline when the splice cannot
        be verified against the mask.
        """
        if (self._mask_manager is None or self._edit_ann_id < 0
                or self._edit_mask is None or self._stroke_old_mask is None):
            return
        ann = self._mask_manager.get_annotation(self._edit_ann_id)
        old_polys, self._stroke_old_polygons = self._stroke_old_polygons, None
        old_mask, self._stroke_old_mask = self._stroke_old_mask, None
        if ann is None or not old_polys:
            return
        ann.original_polygons = MaskManager.splice_polygons(
            old_polys, old_mask, self._edit_mask)

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
        self._pending_polygons = [[[p[0], p[1]] for p in pts]]
        self._clear_draft()
        if self._mask_manager and self._overlay_item and self._pending_mask is not None:
            x1, y1, x2, y2 = MaskManager.fill_polygon_on(self._pending_mask, pts)
            if x2 > x1 and y2 > y1:
                self._refresh_overlay_region(x1, y1, x2, y2)
        self.set_mode(Mode.IDLE)

    def _cancel_draw(self) -> None:
        self._draft_pts.clear()
        self._clear_draft()

    # ── lasso ─────────────────────────────────────────────────────────────────

    def set_lasso_straight(self, px: float) -> None:
        """Shortest segment the lasso will lay down, in image pixels.

        Image pixels rather than screen pixels: the setting then describes the
        label geometry itself, so the same value produces the same outline
        whatever the zoom. 0 disables it and the outline follows the cursor
        exactly, like a brush.
        """
        self._lasso_straight = max(0.0, float(px))

    def _start_lasso(self, sp: QPointF) -> None:
        self._cancel_lasso()
        self._lasso_drawing = True
        self._lasso_pts = [sp]
        pen = QPen(QColor("#FFFF00"), 2)
        pen.setCosmetic(True)
        # Keep the path here rather than reading it back from the item:
        # addPath() drops a path holding nothing but a moveTo, and the next
        # lineTo on the emptied path picks up an implicit moveTo(0, 0), which
        # draws the outline from the scene origin.
        self._lasso_path = QPainterPath()
        self._lasso_path.moveTo(sp)
        self._lasso_item = self.scene().addPath(
            self._lasso_path, pen, QBrush(Qt.BrushStyle.NoBrush))
        self._lasso_item.setZValue(20)
        rubber = QPen(QColor("#FFFF00"), 1, Qt.PenStyle.DashLine)
        rubber.setCosmetic(True)
        self._lasso_rubber = self.scene().addPath(
            QPainterPath(), rubber, QBrush(Qt.BrushStyle.NoBrush))
        self._lasso_rubber.setZValue(21)
        # Start-point marker, same color as the outline. ItemIgnoresTransformations
        # (as the magic-wand include/exclude dots use) keeps it a fixed screen
        # size, matching the view-space snap radius it represents.
        self._lasso_start_dot = QGraphicsEllipseItem(
            -LASSO_SNAP_DIST, -LASSO_SNAP_DIST,
            LASSO_SNAP_DIST * 2, LASSO_SNAP_DIST * 2)
        self._lasso_start_dot.setPen(QPen(QColor("#FFFF00"), 2))
        self._lasso_start_dot.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._lasso_start_dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._lasso_start_dot.setPos(sp)
        self._lasso_start_dot.setZValue(22)
        self.scene().addItem(self._lasso_start_dot)

    def _extend_lasso(self, sp: QPointF, magnetic: bool = True) -> None:
        """Take the cursor position, but only as a new vertex once it is far
        enough from the last one — that gap is what stays perfectly straight.

        magnetic=True (Alt not held) snaps the effective position to the
        start point once the cursor is within LASSO_SNAP_DIST of it, so the
        outline closes exactly rather than leaving a hairline gap. Alt bypasses
        this for a start point the user genuinely wants to draw near.
        """
        if not self._lasso_drawing or self._lasso_item is None:
            return
        near_start = (magnetic and len(self._lasso_pts) >= 2
                     and self._view_dist(sp, self._lasso_pts[0]) < LASSO_SNAP_DIST)
        eff = self._lasso_pts[0] if near_start else sp
        last = self._lasso_pts[-1]
        d = math.hypot(eff.x() - last.x(), eff.y() - last.y())
        if d > 0.0 and d >= self._lasso_straight:
            self._lasso_pts.append(eff)
            self._lasso_path.lineTo(eff)
            self._lasso_item.setPath(self._lasso_path)
        # The rubber band shows where the cursor is even while the outline is
        # holding its line, so a large setting does not look like a freeze.
        if self._lasso_rubber is not None:
            tip = QPainterPath()
            tip.moveTo(self._lasso_pts[-1])
            tip.lineTo(eff)
            tip.lineTo(self._lasso_pts[0])
            self._lasso_rubber.setPath(tip)
        if self._lasso_start_dot is not None:
            self._lasso_start_dot.setBrush(
                QBrush(QColor("#FFFF00")) if near_start else QBrush(Qt.BrushStyle.NoBrush))

    def _complete_lasso(self) -> None:
        """Close the outline and fill it into the pending mask (Enter commits)."""
        pts = [(p.x(), p.y()) for p in self._lasso_pts]
        self._cancel_lasso()
        if self._active_cat_id < 0 or len(pts) < 3:
            return
        # Same handoff the polygon tool makes: leaving edit mode here is what
        # lets the following Enter reach _commit_pending().
        if self._edit_ann_id >= 0:
            self.clear_edit_annotation()
        self._pending_polygons = [[[x, y] for x, y in pts]]
        if self._mask_manager and self._overlay_item and self._pending_mask is not None:
            x1, y1, x2, y2 = MaskManager.fill_polygon_on(self._pending_mask, pts)
            if x2 > x1 and y2 > y1:
                self._refresh_overlay_region(x1, y1, x2, y2)
        self.set_mode(Mode.IDLE)

    def _cancel_lasso(self) -> None:
        self._lasso_drawing = False
        self._lasso_pts = []
        self._lasso_path = QPainterPath()
        for obj in (self._lasso_item, self._lasso_rubber, self._lasso_start_dot):
            if obj is not None:
                self.scene().removeItem(obj)
        self._lasso_item = None
        self._lasso_rubber = None
        self._lasso_start_dot = None

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
                # Painting can extend the mask — bbox must grow with it or the
                # new pixels would be culled from rendering.
                self._mask_manager.grow_bbox(self._edit_ann_id, (x1, y1, x2, y2))
            else:
                x1, y1, x2, y2 = MaskManager.erase_circle_on(self._edit_mask, ix, iy, r)
                # Erasing can only shrink the true extent; a stale-larger bbox
                # is harmless, so recompute once at stroke end instead.
            if x2 > x1 and y2 > y1:
                rgba = self._mask_manager.rgba_region(x1, y1, x2, y2, self._cat_colors)
                self._overlay_item.refresh_region(rgba, x1, y1)
            # Brush invalidates polygon precision — must extract from mask on save
            ann = self._mask_manager.get_annotation(self._edit_ann_id)
            if ann is not None:
                ann.original_polygons = None
            self.edit_changed.emit(self._edit_ann_id)
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
        if r != self._brush_fp_radius:
            # Shape only depends on the radius, so rebuild it just on resize.
            self._brush_ring.setPath(_disc_footprint_path(r))
            if self._brush_fill is not None:
                self._brush_fill.setPath(_disc_fill_path(r))
            self._brush_fp_radius = r
        # Translucent class-color fill under the outline, so the footprint
        # reads as "this area, this class" before the click lands rather than
        # just an outline shape. The outline stays exactly as before — a thin
        # fixed-width white ring with no brush of its own.
        if self._brush_fill is not None:
            color = self._brush_target_color()
            if color != self._brush_fill_color:
                self._brush_fill.setBrush(QBrush(QColor(*color, 110)))
                self._brush_fill_color = color
            self._brush_fill.setPos(int(round(sp.x())), int(round(sp.y())))
            self._brush_fill.show()
        # Snap to the same pixel _do_paint() will use, or the preview would sit
        # up to half a pixel away from what actually gets painted.
        self._brush_ring.setPos(int(round(sp.x())), int(round(sp.y())))
        self._brush_ring.show()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_cursor(self) -> None:
        if self._mode == Mode.BRUSH:
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif self._mode in (Mode.DRAW, Mode.LASSO, Mode.MAGIC):
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._hide_brush_cursor()
        elif self._mode == Mode.PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._hide_brush_cursor()
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._hide_brush_cursor()

    def _hide_brush_cursor(self) -> None:
        if self._brush_ring:
            self._brush_ring.hide()
        if self._brush_fill:
            self._brush_fill.hide()

    # ── magic wand ────────────────────────────────────────────────────────────

    def _magic_click(self, sp: QPointF, label: int) -> None:
        if self._mask_manager is None:
            return
        x = max(0, min(int(round(sp.x())), self._mask_manager.width  - 1))
        y = max(0, min(int(round(sp.y())), self._mask_manager.height - 1))
        self._magic_pts.append((x, y))
        self._magic_lbls.append(label)

        color = QColor("#00e64d") if label == 1 else QColor("#ff3333")
        dot = QGraphicsEllipseItem(-6, -6, 12, 12)
        dot.setPen(QPen(color, 2))
        dot.setBrush(QBrush(color))
        dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        dot.setPos(x + 0.5, y + 0.5)
        dot.setZValue(50)
        self.scene().addItem(dot)
        self._magic_dot_items.append(dot)

        self.magic_requested.emit(list(self._magic_pts), list(self._magic_lbls))

    def set_magic_preview(self, masks: np.ndarray, mask_idx: int) -> None:
        """Called by window with SAM output. masks: (3,H,W) bool-like."""
        self._magic_masks = masks
        self._magic_mask_idx = max(0, min(mask_idx, len(masks) - 1))
        self._update_magic_preview()

    def set_magic_mask_idx(self, idx: int) -> None:
        if self._magic_masks is None:
            return
        self._magic_mask_idx = max(0, min(idx, len(self._magic_masks) - 1))
        self._update_magic_preview()

    def _update_magic_preview(self) -> None:
        if self._magic_masks is None or self._pending_mask is None:
            return
        mask = self._magic_masks[self._magic_mask_idx]
        old = MaskManager.compute_bbox(self._pending_mask)
        self._pending_mask[:] = 0
        self._pending_mask[mask > 0] = 255
        new = MaskManager.compute_bbox(self._pending_mask)
        self._refresh_overlay_rect(MaskManager.union_bbox(old, new))

    def clear_magic(self, keep_pending: bool = False) -> None:
        self._magic_pts = []
        self._magic_lbls = []
        self._magic_masks = None
        for dot in self._magic_dot_items:
            self.scene().removeItem(dot)
        self._magic_dot_items = []
        if not keep_pending and self._pending_mask is not None and self._pending_mask.any():
            self._pending_mask[:] = 0
            self._refresh_overlay_full()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _view_dist(self, a: QPointF, b: QPointF) -> float:
        va = self.mapFromScene(a)
        vb = self.mapFromScene(b)
        return math.hypot(va.x() - vb.x(), va.y() - vb.y())
