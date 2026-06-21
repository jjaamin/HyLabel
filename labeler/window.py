from __future__ import annotations
import os
from typing import Dict, Optional

from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

from .canvas import ImageCanvas, Mode
from .mask_manager import MaskManager
from .models import Project
from . import coco_io

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _color_icon(hex_color: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(hex_color))
    return QIcon(pm)


def _hex_to_rgb(hex_color: str):
    c = QColor(hex_color)
    return (c.red(), c.green(), c.blue())


def _zoom_icon(plus: bool, size: int = 21) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#404040")
    s = float(size)

    cx, cy = s * 0.36, s * 0.36
    r = s * 0.29
    pen = QPen(color, max(1.0, s * 0.09))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(int(cx - r), int(cy - r), max(2, int(r * 2)), max(2, int(r * 2)))

    pen_h = QPen(color, max(1.0, s * 0.12))
    pen_h.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen_h)
    p.drawLine(int(cx + r * 0.7), int(cy + r * 0.7), int(s * 0.93), int(s * 0.93))

    pen_s = QPen(color, max(1.0, s * 0.09))
    pen_s.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen_s)
    arm = s * 0.17
    p.drawLine(int(cx - arm), int(cy), int(cx + arm), int(cy))
    if plus:
        p.drawLine(int(cx), int(cy - arm), int(cx), int(cy + arm))

    p.end()
    return QIcon(pm)


def _fit_icon(size: int = 21) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#404040")
    s = float(size)

    img_m = max(1, int(s * 0.15))
    p.setPen(QPen(color, max(1.0, s * 0.08)))
    p.setBrush(QColor("#cce0f0"))
    p.drawRect(img_m, img_m, size - 2 * img_m, size - 2 * img_m)

    pen_a = QPen(color, max(1.0, s * 0.10))
    pen_a.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen_a)
    p.setBrush(Qt.BrushStyle.NoBrush)
    shaft, head = s * 0.22, s * 0.18

    tx, ty = s * 0.07, s * 0.07
    p.drawLine(int(tx + shaft), int(ty + shaft), int(tx), int(ty))
    p.drawLine(int(tx), int(ty), int(tx + head), int(ty))
    p.drawLine(int(tx), int(ty), int(tx), int(ty + head))

    bx, by = s * 0.93, s * 0.93
    p.drawLine(int(bx - shaft), int(by - shaft), int(bx), int(by))
    p.drawLine(int(bx), int(by), int(bx - head), int(by))
    p.drawLine(int(bx), int(by), int(bx), int(by - head))

    p.end()
    return QIcon(pm)


def _polygon_icon(size: int = 21) -> QIcon:
    import math
    from PyQt6.QtGui import QPainterPath

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#404040")
    s = float(size)

    cx, cy, r = s * 0.50, s * 0.52, s * 0.43
    pts = [
        (cx + r * math.cos(math.pi * (-0.5 + 2 * i / 5)),
         cy + r * math.sin(math.pi * (-0.5 + 2 * i / 5)))
        for i in range(5)
    ]

    path = QPainterPath()
    path.moveTo(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    p.setPen(QPen(color, max(1.0, s * 0.09)))
    p.setBrush(QColor("#d8e8f8"))
    p.drawPath(path)

    dr = max(1.2, s * 0.12)
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    for x, y in pts:
        p.drawEllipse(int(x - dr), int(y - dr), max(2, int(dr * 2)), max(2, int(dr * 2)))

    p.end()
    return QIcon(pm)


def _emoji_icon(symbol: str, size: int = 21) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    f = p.font()
    f.setPixelSize(int(size * 0.82))
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Label Editor")
        self.resize(1280, 800)

        self.project = Project()
        self.image_dir: str = ""
        self.save_path: Optional[str] = None
        self._settings = QSettings("LabelEditor", "LabelEditor")
        self._last_dir: str = self._settings.value("lastDir", "")
        self.current_img_ann = None
        self._modified = False

        # image_id → MaskManager  (in-place modified during editing)
        self._mask_managers: Dict[int, MaskManager] = {}

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        self._act_open_folder = QAction("Open &Folder…", self, shortcut="Ctrl+O")
        self._act_open_file   = QAction("Open &Image…", self)
        self._act_load_ann    = QAction("&Load Annotations…", self)
        self._act_save        = QAction("&Save", self, shortcut="Ctrl+S")
        self._act_save_as     = QAction("Save &As…", self, shortcut="Ctrl+Shift+S")
        for a in (self._act_open_folder, self._act_open_file, None,
                  self._act_load_ann, None,
                  self._act_save, self._act_save_as, None):
            fm.addSeparator() if a is None else fm.addAction(a)
        fm.addAction(QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close))

        vm = mb.addMenu("&View")
        self._act_zoom_in  = QAction("Zoom &In",   self, shortcut="=")
        self._act_zoom_out = QAction("Zoom &Out",  self, shortcut="-")
        self._act_fit      = QAction("&Fit Image", self, shortcut="F")
        for a in (self._act_zoom_in, self._act_zoom_out, self._act_fit):
            vm.addAction(a)

        # ── Left Vertical Toolbar ─────────────────────────────────────────────
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(21, 21))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.setStyleSheet("""
            QToolBar { spacing: 1px; padding: 3px 2px; }
            QToolButton {
                padding: 4px;
                min-width: 29px;
                min-height: 29px;
                border-radius: 4px;
            }
            QToolButton:checked {
                background-color: #3a7bd5;
            }
            QToolButton:hover {
                background-color: #d0d8e8;
            }
            QToolButton:checked:hover {
                background-color: #2e6bc0;
            }
        """)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)

        # Tool group (exclusive): Draw / Brush / Pan
        self._act_draw  = QAction(self, shortcut="D", checkable=True)
        self._act_brush = QAction(self, shortcut="B", checkable=True)
        self._act_hand  = QAction(self, shortcut="H", checkable=True)

        self._act_draw.setIcon(_polygon_icon())
        self._act_draw.setToolTip("Draw Polygon  (D)")
        self._act_brush.setIcon(_emoji_icon("🖌"))
        self._act_brush.setToolTip("Brush  (B)  —  LMB: paint  /  RMB: erase")
        self._act_hand.setIcon(_emoji_icon("🖐"))
        self._act_hand.setToolTip("Pan  (H)")

        self._act_zoom_in.setIcon(_zoom_icon(plus=True))
        self._act_zoom_in.setToolTip("Zoom In  (=)")
        self._act_zoom_out.setIcon(_zoom_icon(plus=False))
        self._act_zoom_out.setToolTip("Zoom Out  (-)")
        self._act_fit.setIcon(_fit_icon())
        self._act_fit.setToolTip("Fit Image  (F)")

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for a in (self._act_draw, self._act_brush):
            tool_group.addAction(a)
            tb.addAction(a)

        tb.addSeparator()
        tb.addAction(self._act_hand)
        tool_group.addAction(self._act_hand)
        tb.addAction(self._act_zoom_in)
        tb.addAction(self._act_zoom_out)
        tb.addAction(self._act_fit)

        # ── Layout ────────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # Center: canvas
        self.canvas = ImageCanvas()
        splitter.addWidget(self.canvas)

        # Right panel: brush size + classes + layers + images
        right = QWidget()
        right.setFixedWidth(230)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(6)

        # Brush size row
        size_widget = QWidget()
        sh = QHBoxLayout(size_widget)
        sh.setContentsMargins(4, 2, 4, 2)
        brush_lbl = QLabel("Brush:")
        brush_lbl.setStyleSheet("font-size: 13px;")
        sh.addWidget(brush_lbl)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 50)
        self._brush_slider.setValue(20)
        self._brush_slider.setFixedHeight(24)
        self._brush_slider.setToolTip("Brush / Eraser size  ( [ / ] to adjust )")
        sh.addWidget(self._brush_slider)
        self._brush_size_lbl = QLabel("20")
        self._brush_size_lbl.setStyleSheet("font-size: 13px;")
        self._brush_size_lbl.setFixedWidth(30)
        sh.addWidget(self._brush_size_lbl)
        rv.addWidget(size_widget)

        # Classes group
        cg = QGroupBox("Classes")
        cg.setStyleSheet("QGroupBox { font-weight: normal; }")
        cv = QVBoxLayout(cg)
        self._class_list = QListWidget()
        self._class_list.setMaximumHeight(180)
        cv.addWidget(self._class_list)
        ch = QHBoxLayout()
        self._btn_add_cls = QPushButton("+ Add")
        self._btn_rem_cls = QPushButton("− Remove")
        ch.addWidget(self._btn_add_cls)
        ch.addWidget(self._btn_rem_cls)
        cv.addLayout(ch)
        rv.addWidget(cg)

        # Labels group
        lg = QGroupBox("Labels")
        lg.setStyleSheet("QGroupBox { font-weight: normal; }")
        lav = QVBoxLayout(lg)
        self._label_list = QListWidget()
        self._label_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._label_list.setMaximumHeight(140)
        lav.addWidget(self._label_list)
        self._btn_clear_label = QPushButton("Delete Selected Label")
        lav.addWidget(self._btn_clear_label)
        rv.addWidget(lg)

        # Images section (below Clear Active Layer)
        img_lbl = QLabel("Images")
        img_lbl.setStyleSheet("margin-top: 4px;")
        rv.addWidget(img_lbl)
        self._img_list = QListWidget()
        rv.addWidget(self._img_list)

        splitter.addWidget(right)
        splitter.setSizes([1050, 230])

        # Status bar
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._lbl_mode   = QLabel("Mode: Idle")
        self._lbl_status = QLabel(
            "H=pan  D=draw  B=brush(LMB=paint/RMB=erase)  Enter=commit  Esc=cancel  [/]=brush size"
        )
        sb.addWidget(self._lbl_mode)
        sb.addPermanentWidget(self._lbl_status)

    def _connect_signals(self) -> None:
        self._act_open_folder.triggered.connect(self._open_folder)
        self._act_open_file.triggered.connect(self._open_file)
        self._act_save.triggered.connect(self._save)
        self._act_save_as.triggered.connect(self._save_as)
        self._act_load_ann.triggered.connect(self._load_annotations)
        self._act_hand.toggled.connect(self._on_tool_toggled)
        self._act_draw.toggled.connect(self._on_tool_toggled)
        self._act_brush.toggled.connect(self._on_tool_toggled)
        self._act_zoom_in.triggered.connect(lambda: self.canvas.scale(1.2, 1.2))
        self._act_zoom_out.triggered.connect(lambda: self.canvas.scale(1 / 1.2, 1 / 1.2))
        self._act_fit.triggered.connect(self.canvas.fit_view)

        self._brush_slider.valueChanged.connect(self._on_slider_changed)
        self.canvas.brush_size_changed.connect(self._sync_slider)

        self._btn_add_cls.clicked.connect(self._add_class)
        self._btn_rem_cls.clicked.connect(self._remove_class)
        self._class_list.currentRowChanged.connect(self._update_active_class)
        self._class_list.currentRowChanged.connect(self._update_class_bold)
        self._class_list.clicked.connect(self._on_class_clicked)

        self._btn_clear_label.clicked.connect(self._clear_active_label)

        self._img_list.currentRowChanged.connect(self._on_image_selected)

        self.canvas.annotation_committed.connect(self._on_annotation_committed)
        self.canvas.edit_changed.connect(self._mark_modified)
        self.canvas.edit_cleared.connect(self._on_edit_cleared)
        self.canvas.mode_changed.connect(self._on_mode_changed)
        self._label_list.currentRowChanged.connect(self._on_label_selected)

    # ── file operations ───────────────────────────────────────────────────────

    def _open_folder(self) -> None:
        if not self._confirm_discard():
            return
        folder = QFileDialog.getExistingDirectory(self, "Open Image Folder", self._last_dir)
        if not folder:
            return
        self._last_dir = folder
        self._settings.setValue("lastDir", folder)
        self._reset_project(folder)

        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        self._img_list.clear()
        for f in files:
            self._img_list.addItem(f)

        auto = os.path.join(folder, "annotations.json")
        if os.path.isfile(auto):
            try:
                proj, mgrs = coco_io.load_coco(auto, folder)
                self.project = proj
                self._mask_managers = mgrs
                self.save_path = auto
                self._refresh_class_list()
                self._lbl_status.setText("Loaded: annotations.json")
            except Exception:
                pass

        if files:
            self._img_list.setCurrentRow(0)
        self._update_title()

    def _open_file(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self._last_dir,
            filter="Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp)"
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("lastDir", self._last_dir)
        self._reset_project(os.path.dirname(path))
        self._img_list.clear()
        self._img_list.addItem(os.path.basename(path))
        self._img_list.setCurrentRow(0)
        self._update_title()

    def _save(self) -> None:
        if self.save_path is None:
            self._save_as()
            return
        self._do_save(self.save_path)

    def _save_as(self) -> None:
        default = os.path.join(self.image_dir, "annotations.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", default, filter="JSON (*.json)"
        )
        if not path:
            return
        self.save_path = path
        self._do_save(path)

    def _do_save(self, path: str) -> None:
        try:
            coco_io.save_coco(self.project, self._mask_managers, path, self.image_dir)
            self._modified = False
            self._update_title()
            self._lbl_status.setText(f"Saved → {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_annotations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load COCO Annotations", self.image_dir,
            filter="JSON (*.json)"
        )
        if not path:
            return
        try:
            proj, mgrs = coco_io.load_coco(path, self.image_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
            return
        self.project = proj
        self._mask_managers = mgrs
        self.save_path = path
        self._refresh_class_list()
        row = self._img_list.currentRow()
        if row >= 0:
            self._on_image_selected(row)
        self._lbl_status.setText(f"Loaded: {os.path.basename(path)}")

    # ── image navigation ──────────────────────────────────────────────────────

    def _on_image_selected(self, row: int) -> None:
        if row < 0:
            return

        name = self._img_list.item(row).text()
        path = os.path.join(self.image_dir, name)

        w, h = self.canvas.load_image(path)
        img_ann = self.project.get_or_create_image(path, w, h)
        self.current_img_ann = img_ann

        # Get or create MaskManager for this image
        mgr = self._mask_managers.get(img_ann.image_id)
        if mgr is None:
            mgr = MaskManager(w, h)
            self._mask_managers[img_ann.image_id] = mgr

        self.canvas.set_mask_manager(mgr, self._color_tuples())

        # Reset tool buttons to idle
        self._uncheck_all_tools()
        self.canvas.set_mode(Mode.IDLE)

        self._update_active_class()
        self._refresh_labels()
        self._lbl_status.setText(f"{name}  ({w}×{h})")

    # ── class management ──────────────────────────────────────────────────────

    def _add_class(self) -> None:
        existing = {c.name for c in self.project.categories}
        n = 1
        while f"Class{n}" in existing:
            n += 1
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:", text=f"Class{n}")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(c.name == name for c in self.project.categories):
            QMessageBox.warning(self, "Duplicate", f"'{name}' already exists.")
            return
        cat = self.project.add_category(name)
        item = QListWidgetItem(_color_icon(cat.color), cat.name)
        item.setData(Qt.ItemDataRole.UserRole, cat.id)
        self._class_list.addItem(item)
        self._class_list.setCurrentRow(self._class_list.count() - 1)
        self.canvas.update_cat_colors(self._color_tuples())
        self._refresh_labels()
        self._mark_modified()

    def _remove_class(self) -> None:
        row = self._class_list.currentRow()
        if row < 0:
            return
        cat_id = self._class_list.item(row).data(Qt.ItemDataRole.UserRole)
        in_use = any(mgr.has_any(cat_id) for mgr in self._mask_managers.values())
        if in_use:
            QMessageBox.warning(
                self, "In Use",
                "Cannot remove a class that has painted regions.\n"
                "Clear the layer first."
            )
            return
        self.project.categories = [c for c in self.project.categories if c.id != cat_id]
        self._class_list.takeItem(row)
        self.canvas.update_cat_colors(self._color_tuples())
        self._update_active_class(self._class_list.currentRow())
        self._refresh_labels()
        self._mark_modified()

    def _refresh_class_list(self) -> None:
        self._class_list.clear()
        for cat in self.project.categories:
            item = QListWidgetItem(_color_icon(cat.color), cat.name)
            item.setData(Qt.ItemDataRole.UserRole, cat.id)
            self._class_list.addItem(item)
        if self.project.categories:
            self._class_list.setCurrentRow(0)

    def _update_active_class(self, row: int = -1) -> None:
        if row < 0:
            row = self._class_list.currentRow()
        if row < 0 or row >= self._class_list.count():
            return
        cat_id = self._class_list.item(row).data(Qt.ItemDataRole.UserRole)
        self.canvas.set_active_category(cat_id)

    def _update_class_bold(self, row: int = -1) -> None:
        if row < 0:
            row = self._class_list.currentRow()
        for i in range(self._class_list.count()):
            item = self._class_list.item(i)
            font = item.font()
            font.setBold(i == row)
            item.setFont(font)

    # ── labels panel ──────────────────────────────────────────────────────────

    def _refresh_labels(self) -> None:
        """Update the Labels panel: show committed annotations as individual rows."""
        self.canvas.clear_edit_annotation()
        self._label_list.blockSignals(True)
        self._label_list.clear()
        if self.current_img_ann is None:
            self._label_list.blockSignals(False)
            return
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            self._label_list.blockSignals(False)
            return
        cat_map = {c.id: c for c in self.project.categories}
        cat_counts: Dict[int, int] = {}
        for ann in mgr.annotations():
            cat = cat_map.get(ann.cat_id)
            if cat is None:
                continue
            cat_counts[ann.cat_id] = cat_counts.get(ann.cat_id, 0) + 1
            n = cat_counts[ann.cat_id]
            icon = _color_icon(cat.color, 12)
            item = QListWidgetItem(icon, f"●  {cat.name}  #{n}")
            item.setData(Qt.ItemDataRole.UserRole, ann.ann_id)
            self._label_list.addItem(item)
        self._label_list.blockSignals(False)

    def _clear_active_label(self) -> None:
        """Remove the selected annotation from this image."""
        row = self._label_list.currentRow()
        if row < 0 or self.current_img_ann is None:
            return
        item = self._label_list.item(row)
        if item is None:
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            return
        mgr.remove_annotation(ann_id)
        self.canvas.refresh_overlay()
        self._refresh_labels()
        self._mark_modified()

    # ── tool toggling ─────────────────────────────────────────────────────────

    def _on_tool_toggled(self, checked: bool) -> None:
        if not checked:
            return
        # Hand tool: no category required
        if self._act_hand.isChecked():
            self.canvas.set_mode(Mode.PAN)
            self.canvas.setFocus()
            return
        # Drawing tools: require at least one class
        if not self.project.categories:
            QMessageBox.information(
                self, "No Classes", "Please add at least one class first."
            )
            self._uncheck_all_tools()
            return
        if self._act_draw.isChecked():
            self._update_active_class()
            self.canvas.set_mode(Mode.DRAW)
        elif self._act_brush.isChecked():
            self._update_active_class()
            self.canvas.set_mode(Mode.BRUSH)
        self.canvas.setFocus()

    def _uncheck_all_tools(self) -> None:
        for a in (self._act_hand, self._act_draw, self._act_brush):
            a.blockSignals(True)
            a.setChecked(False)
            a.blockSignals(False)

    # ── brush size ────────────────────────────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        self._brush_size_lbl.setText(str(value))
        self.canvas.set_brush_size(value)

    def _sync_slider(self, size: int) -> None:
        """Sync slider when brush size changed via [ / ] keys on canvas."""
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(size)
        self._brush_slider.blockSignals(False)
        self._brush_size_lbl.setText(str(size))

    # ── canvas signal handlers ────────────────────────────────────────────────

    def _on_annotation_committed(self) -> None:
        self._refresh_labels()
        self._mark_modified()

    def _on_class_clicked(self, _) -> None:
        self.canvas.clear_edit_annotation()
        self._clear_label_bold()

    def _on_label_selected(self, row: int) -> None:
        if row < 0 or self.current_img_ann is None:
            self.canvas.clear_edit_annotation()
            return
        item = self._label_list.item(row)
        if item is None:
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            return
        cat_map = {c.id: c for c in self.project.categories}
        cat_counts: Dict[int, int] = {}
        for ann in mgr.annotations():
            cat_counts[ann.cat_id] = cat_counts.get(ann.cat_id, 0) + 1
            if ann.ann_id == ann_id:
                cat = cat_map.get(ann.cat_id)
                n = cat_counts[ann.cat_id]
                self.canvas.set_edit_annotation(ann_id, ann.mask)
                self._uncheck_all_tools()
                self.canvas.set_mode(Mode.IDLE)
                self._lbl_mode.setText(
                    f"Editing: {cat.name if cat else '?'}  #{n}"
                    "   (drag points / B=brush  Esc=done)"
                )
                self._set_label_bold(row)
                self._clear_class_bold()
                break

    def _on_edit_cleared(self) -> None:
        self._label_list.blockSignals(True)
        self._label_list.clearSelection()
        self._label_list.blockSignals(False)
        self._clear_label_bold()
        self._on_mode_changed(self.canvas.current_mode)

    def _set_label_bold(self, row: int) -> None:
        for i in range(self._label_list.count()):
            item = self._label_list.item(i)
            f = item.font()
            f.setBold(i == row)
            item.setFont(f)

    def _clear_label_bold(self) -> None:
        for i in range(self._label_list.count()):
            item = self._label_list.item(i)
            f = item.font()
            f.setBold(False)
            item.setFont(f)

    def _clear_class_bold(self) -> None:
        self._class_list.blockSignals(True)
        for i in range(self._class_list.count()):
            item = self._class_list.item(i)
            f = item.font()
            f.setBold(False)
            item.setFont(f)
        self._class_list.blockSignals(False)

    def _on_mode_changed(self, mode_str: str) -> None:
        labels = {
            "idle":  "Mode: Idle",
            "pan":   "Mode: Pan  (drag to move image)",
            "draw":  "Mode: Draw  (double-click or snap to close)",
            "brush": "Mode: Brush  (LMB: paint  /  RMB: erase)",
        }
        self._lbl_mode.setText(labels.get(mode_str, f"Mode: {mode_str}"))
        if mode_str == "idle":
            self._uncheck_all_tools()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _color_tuples(self) -> dict:
        return {cat.id: _hex_to_rgb(cat.color) for cat in self.project.categories}

    def _reset_project(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.project = Project()
        self.save_path = None
        self._modified = False
        self._mask_managers = {}
        self.current_img_ann = None
        self._class_list.clear()
        self._label_list.clear()

    def _mark_modified(self) -> None:
        self._modified = True
        self._update_title()

    def _update_title(self) -> None:
        base = os.path.basename(self.image_dir) if self.image_dir else "Label Editor"
        marker = " *" if self._modified else ""
        self.setWindowTitle(f"Label Editor — {base}{marker}")

    def _confirm_discard(self) -> bool:
        if not self._modified:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes", "Discard unsaved changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
