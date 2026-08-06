from __future__ import annotations
import contextlib
import os
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QSize, QSettings, QItemSelectionModel
from PyQt6.QtGui import QAction, QActionGroup, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QSlider, QSplitter, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)

from .canvas import ImageCanvas, Mode
from .mask_manager import MaskManager
from .models import Project
from . import coco_io
from .gamma_dialog import GammaCurveDialog, compute_lut
from . import sam_worker

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


def _arrow_icon(size: int = 21) -> QIcon:
    from PyQt6.QtGui import QPainterPath

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)

    # Classic pointer: shaft down the left, notch, then the tail kicking right.
    path = QPainterPath()
    path.moveTo(s * 0.24, s * 0.09)
    path.lineTo(s * 0.24, s * 0.79)
    path.lineTo(s * 0.41, s * 0.63)
    path.lineTo(s * 0.53, s * 0.90)
    path.lineTo(s * 0.66, s * 0.83)
    path.lineTo(s * 0.54, s * 0.58)
    path.lineTo(s * 0.76, s * 0.56)
    path.closeSubpath()

    pen = QPen(QColor("#404040"), max(1.0, s * 0.08))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(QColor("#f5f5f5"))
    p.drawPath(path)

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


def _json_doc_icon(size: int = 14) -> QIcon:
    from PyQt6.QtGui import QPainterPath as _Path
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)

    dw, dh = s * 0.62, s * 0.80
    dx = (s - dw) / 2
    dy = (s - dh) / 2
    fold = dw * 0.30

    body = _Path()
    body.moveTo(dx, dy)
    body.lineTo(dx + dw - fold, dy)
    body.lineTo(dx + dw, dy + fold)
    body.lineTo(dx + dw, dy + dh)
    body.lineTo(dx, dy + dh)
    body.closeSubpath()
    p.setPen(QPen(QColor("#1D4ED8"), max(1.0, s * 0.07)))
    p.setBrush(QBrush(QColor("#BFDBFE")))
    p.drawPath(body)

    corner = _Path()
    corner.moveTo(dx + dw - fold, dy)
    corner.lineTo(dx + dw - fold, dy + fold)
    corner.lineTo(dx + dw, dy + fold)
    corner.closeSubpath()
    p.setPen(QPen(QColor("#1D4ED8"), max(1.0, s * 0.07)))
    p.setBrush(QBrush(QColor("#93C5FD")))
    p.drawPath(corner)

    lpen = QPen(QColor("#1E40AF"), max(1.0, s * 0.09))
    lpen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(lpen)
    lx1, lx2 = dx + dw * 0.18, dx + dw * 0.80
    for fy in (0.52, 0.65, 0.78):
        ly = dy + dh * fy
        p.drawLine(int(lx1), int(ly), int(lx2), int(ly))

    p.end()
    return QIcon(pm)


def _text_icon(symbol: str, size: int = 21) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    f = p.font()
    f.setPixelSize(int(size * 0.82))
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
    p.end()
    return QIcon(pm)


def _magic_wand_icon(size: int = 21) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)

    # Purple wand stick
    pen = QPen(QColor("#7C3AED"), max(2.0, s * 0.11))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(int(s*0.14), int(s*0.88), int(s*0.60), int(s*0.38))

    # Colorful sparkle rays from tip
    cx, cy = s * 0.68, s * 0.30
    arm  = s * 0.18
    arm2 = arm * 0.68
    rays = [
        ("#FF4D6D",  0,     -arm ),   # up    — pink
        ("#FFD700",  arm,    0   ),   # right — yellow
        ("#00D4AA",  0,      arm ),   # down  — teal
        ("#60A5FA", -arm,    0   ),   # left  — blue
        ("#F472B6", -arm2,  -arm2),   # up-left  — light pink
        ("#A3E635",  arm2,   arm2),   # down-right — lime
        ("#FB923C",  arm2,  -arm2),   # up-right — orange
        ("#C084FC", -arm2,   arm2),   # down-left — violet
    ]
    for color, dx, dy in rays:
        rpen = QPen(QColor(color), max(1.5, s * 0.09))
        rpen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(rpen)
        p.drawLine(int(cx + dx*0.30), int(cy + dy*0.30),
                   int(cx + dx),      int(cy + dy))

    # Gold star center
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#FFD700")))
    sr = max(2, int(s * 0.13))
    p.drawEllipse(int(cx - sr), int(cy - sr), sr * 2, sr * 2)

    p.end()
    return QIcon(pm)



class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HyLabel")
        self.resize(1280, 800)

        self.project = Project()
        self.image_dir: str = ""
        self.save_path: Optional[str] = None
        self._settings = QSettings("HyLabel", "HyLabel")
        self._last_dir: str = self._settings.value("lastDir", "")
        y0 = int(self._settings.value("gammaY0", 0))
        y1 = int(self._settings.value("gammaY1", 128))
        y2 = int(self._settings.value("gammaY2", 255))
        self._gamma_ctrl = [(0, y0), (128, y1), (255, y2)]
        self._gamma_dialog: Optional[GammaCurveDialog] = None
        self._sam_predictor: Optional[object] = None
        self._sam_img_path: str = ""
        self._sam_model_key: str = self._settings.value("samModel", sam_worker.MODEL_EDGESAM)
        if self._sam_model_key not in sam_worker.visible_models():
            self._sam_model_key = sam_worker.MODEL_EDGESAM
        self.current_img_ann = None
        self._modified = False
        self._pre_pan_action: Optional[QAction] = None
        self._shift_alone = False

        # image_id → MaskManager  (in-place modified during editing)
        self._mask_managers: Dict[int, MaskManager] = {}

        # undo stack: list of operation dicts (cleared on image change)
        self._undo_stack: List[dict] = []
        self._syncing_selection = False

        self._build_ui()
        self._connect_signals()
        self.canvas.set_gamma_lut(compute_lut(self._gamma_ctrl))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        self._act_open_folder = QAction("Open &Folder…", self, shortcut="Ctrl+O")
        self._act_open_file   = QAction("Open &Image…", self)
        self._act_load_ann    = QAction("&Load from Folder…", self)
        self._act_save        = QAction("&Save", self, shortcut="Ctrl+S")
        self._act_save_as     = QAction("Save to &Folder…", self, shortcut="Ctrl+Shift+S")
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
        vm.addSeparator()
        self._act_faint = QAction("Faint Label &View", self, shortcut="V", checkable=True)
        self._act_gamma = QAction("Apply &Gamma",  self, shortcut="G", checkable=True)
        self._act_gamma_curve = QAction("Gamma Curve Setting", self)
        vm.addAction(self._act_faint)
        vm.addAction(self._act_gamma)
        vm.addAction(self._act_gamma_curve)

        em = mb.addMenu("&Edit")
        self._act_undo = QAction("&Undo", self, shortcut="Ctrl+Z")
        em.addAction(self._act_undo)

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

        # Tool group (exclusive): Select / Draw / Brush / Magic / Pan
        self._act_select = QAction(self, shortcut="S", checkable=True)
        self._act_draw  = QAction(self, shortcut="D", checkable=True)
        self._act_brush = QAction(self, shortcut="B", checkable=True)
        self._act_magic = QAction(self, shortcut="M", checkable=True)
        self._act_hand  = QAction(self, checkable=True)

        self._act_select.setIcon(_arrow_icon())
        self._act_select.setToolTip(
            "Select  (S)  —  click a label on the canvas to select it"
        )
        self._act_draw.setIcon(_polygon_icon())
        self._act_draw.setToolTip("Draw Polygon  (D)")
        self._act_brush.setIcon(_text_icon("🖌", size=21))
        self._act_brush.setToolTip("Brush  (B)  —  LMB: paint  /  RMB: erase")
        self._act_magic.setIcon(_magic_wand_icon())
        self._act_magic.setToolTip(
            "AI Magic Wand  (M)  —  LMB: include point  /  RMB: exclude point  /  Enter: commit")
        self._act_hand.setIcon(_text_icon("🖐", size=21))
        self._act_hand.setToolTip("Pan  (H)")

        self._act_zoom_in.setIcon(_zoom_icon(plus=True))
        self._act_zoom_in.setToolTip("Zoom In  (=)")
        self._act_zoom_out.setIcon(_zoom_icon(plus=False))
        self._act_zoom_out.setToolTip("Zoom Out  (-)")
        self._act_fit.setIcon(_fit_icon())
        self._act_fit.setToolTip("Fit Image  (F)")

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for a in (self._act_select, self._act_draw, self._act_brush, self._act_magic):
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
        self._brush_slider.setRange(1, 88)
        self._brush_slider.setValue(20)
        self._brush_slider.setFixedHeight(24)
        self._brush_slider.setToolTip("Brush / Eraser size  ( [ / ] to adjust )")
        sh.addWidget(self._brush_slider)
        self._brush_size_lbl = QLabel("20")
        self._brush_size_lbl.setStyleSheet("font-size: 13px;")
        self._brush_size_lbl.setFixedWidth(30)
        sh.addWidget(self._brush_size_lbl)
        rv.addWidget(size_widget)

        # Model row (AI Magic Wand)
        model_widget = QWidget()
        mow = QHBoxLayout(model_widget)
        mow.setContentsMargins(4, 2, 4, 2)
        model_lbl = QLabel("AI Model:")
        model_lbl.setStyleSheet("font-size: 13px;")
        mow.addWidget(model_lbl)
        self._sam_model_combo = QComboBox()
        for key in sam_worker.visible_models():
            self._sam_model_combo.addItem(sam_worker.MODEL_INFO[key]["label"], key)
        idx = self._sam_model_combo.findData(self._sam_model_key)
        self._sam_model_combo.setCurrentIndex(max(0, idx))
        self._sam_model_combo.setToolTip("AI Magic Wand model")
        mow.addWidget(self._sam_model_combo, 1)
        rv.addWidget(model_widget)

        # Mask index row (AI Magic Wand)
        mask_widget = QWidget()
        mh = QHBoxLayout(mask_widget)
        mh.setContentsMargins(4, 2, 4, 2)
        mask_lbl = QLabel("Mask:")
        mask_lbl.setStyleSheet("font-size: 13px;")
        mh.addWidget(mask_lbl)
        self._mask_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_slider.setRange(0, 2)
        self._mask_slider.setValue(0)
        self._mask_slider.setFixedHeight(24)
        self._mask_slider.setEnabled(False)
        self._mask_slider.setToolTip("Select SAM mask candidate  (1 = smallest, 3 = largest)")
        mh.addWidget(self._mask_slider)
        self._mask_idx_lbl = QLabel("1/3")
        self._mask_idx_lbl.setStyleSheet("font-size: 13px;")
        self._mask_idx_lbl.setFixedWidth(30)
        mh.addWidget(self._mask_idx_lbl)
        rv.addWidget(mask_widget)

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
        self._label_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        lav.addWidget(self._label_list)
        self._btn_clear_label = QPushButton("Delete Selected Label")
        lav.addWidget(self._btn_clear_label)
        self._btn_merge_labels = QPushButton("Merge Selected Labels  (Home)")
        self._btn_merge_labels.setEnabled(False)
        self._btn_merge_labels.setToolTip(
            "Combine the selected labels into one.\n"
            "Ctrl+click to pick several, on the canvas or in this list."
        )
        lav.addWidget(self._btn_merge_labels)
        self._label_class_combo = QComboBox()
        self._label_class_combo.setEnabled(False)
        self._label_class_combo.setToolTip("Change class of selected label")
        lav.addWidget(self._label_class_combo)
        rv.addWidget(lg, 3)

        # Images section — deliberately the smaller of the two lists; labelling
        # works out of Labels, and Images is mostly for jumping between files.
        img_lbl = QLabel("Images")
        img_lbl.setStyleSheet("margin-top: 4px;")
        rv.addWidget(img_lbl)
        self._img_list = QListWidget()
        rv.addWidget(self._img_list, 1)

        splitter.addWidget(right)
        splitter.setSizes([1050, 230])

        # Window-level shortcuts (work regardless of focus)
        self._act_brush_dec = QAction(self, shortcut="[")
        self._act_brush_inc = QAction(self, shortcut="]")
        self._act_pan_toggle = QAction(self, shortcut="H")
        self._act_img_prev = QAction(self, shortcut="PgUp")
        self._act_img_next = QAction(self, shortcut="PgDown")
        self._act_class_prev = QAction(self, shortcut="Up")
        self._act_class_next = QAction(self, shortcut="Down")
        self._act_label_prev = QAction(self, shortcut="Left")
        self._act_label_next = QAction(self, shortcut="Right")
        # Delete has to be window-level too: selecting a label switches to the
        # brush tool, which hands focus to the canvas, so a key filter on the
        # label list never sees the keystroke.
        self._act_label_del = QAction(self, shortcut="Delete")
        self._act_label_merge = QAction(self, shortcut="Home")
        self.addAction(self._act_brush_dec)
        self.addAction(self._act_brush_inc)
        self.addAction(self._act_pan_toggle)
        self.addAction(self._act_img_prev)
        self.addAction(self._act_img_next)
        self.addAction(self._act_class_prev)
        self.addAction(self._act_class_next)
        self.addAction(self._act_label_prev)
        self.addAction(self._act_label_next)
        self.addAction(self._act_label_del)
        self.addAction(self._act_label_merge)

        # Status bar
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._lbl_mode   = QLabel("Mode: Idle")
        self._lbl_status = QLabel(
            "H=pan  D=draw  B=brush(LMB=paint/RMB=erase)  Enter=commit  Esc=cancel  [/]=brush size  "
            "PgUp/PgDn=image  ↑/↓=class  ←/→=label"
        )
        sb.addWidget(self._lbl_mode)
        sb.addPermanentWidget(self._lbl_status)

    def _connect_signals(self) -> None:
        self._act_undo.triggered.connect(self._handle_undo)
        self._act_open_folder.triggered.connect(self._open_folder)
        self._act_open_file.triggered.connect(self._open_file)
        self._act_save.triggered.connect(self._save)
        self._act_save_as.triggered.connect(self._save_as)
        self._act_load_ann.triggered.connect(self._load_annotations)
        self._act_select.toggled.connect(self._on_tool_toggled)
        self._act_hand.toggled.connect(self._on_tool_toggled)
        self._act_draw.toggled.connect(self._on_tool_toggled)
        self._act_brush.toggled.connect(self._on_tool_toggled)
        self._act_magic.toggled.connect(self._on_tool_toggled)
        self._act_zoom_in.triggered.connect(lambda: self.canvas.scale(1.2, 1.2))
        self._act_zoom_out.triggered.connect(lambda: self.canvas.scale(1 / 1.2, 1 / 1.2))
        self._act_fit.triggered.connect(self.canvas.fit_view)
        self._act_faint.triggered.connect(self._toggle_faint)
        self._act_gamma.triggered.connect(self._toggle_gamma)
        self._act_gamma_curve.triggered.connect(self._open_gamma_dialog)

        self._brush_slider.valueChanged.connect(self._on_slider_changed)
        self.canvas.brush_size_changed.connect(self._sync_slider)
        self._act_brush_dec.triggered.connect(lambda: self._adjust_size(-1))
        self._act_brush_inc.triggered.connect(lambda: self._adjust_size(+1))
        self._act_pan_toggle.triggered.connect(self._toggle_pan)
        self._act_img_prev.triggered.connect(lambda: self._step_list(self._img_list, -1))
        self._act_img_next.triggered.connect(lambda: self._step_list(self._img_list, +1))
        self._act_class_prev.triggered.connect(lambda: self._step_list(self._class_list, -1))
        self._act_class_next.triggered.connect(lambda: self._step_list(self._class_list, +1))
        self._act_label_prev.triggered.connect(lambda: self._step_list(self._label_list, -1))
        self._act_label_next.triggered.connect(lambda: self._step_list(self._label_list, +1))
        self._act_label_del.triggered.connect(self._clear_active_label)
        self._act_label_merge.triggered.connect(self._merge_selected_labels)
        self._mask_slider.valueChanged.connect(self._on_mask_slider_changed)
        self._sam_model_combo.currentIndexChanged.connect(self._on_sam_model_changed)

        self._btn_add_cls.clicked.connect(self._add_class)
        self._btn_rem_cls.clicked.connect(self._remove_class)
        self._class_list.currentRowChanged.connect(self._update_active_class)
        self._class_list.currentRowChanged.connect(self._update_class_bold)
        self._class_list.clicked.connect(self._on_class_clicked)

        self._btn_clear_label.clicked.connect(self._clear_active_label)
        self._label_class_combo.currentIndexChanged.connect(self._on_label_class_changed)

        self._img_list.currentRowChanged.connect(self._on_image_selected)

        self.canvas.annotation_committed.connect(self._on_annotation_committed)
        self.canvas.magic_requested.connect(self._on_magic_requested)
        self.canvas.select_requested.connect(self._on_canvas_select)
        self.canvas.undo_record.connect(self._push_undo)
        self.canvas.edit_changed.connect(self._on_edit_changed)
        self.canvas.edit_cleared.connect(self._on_edit_cleared)
        self.canvas.mode_changed.connect(self._on_mode_changed)
        # itemSelectionChanged, not currentRowChanged: with several rows picked
        # the current row alone does not describe the selection.
        self._label_list.itemSelectionChanged.connect(self._on_label_selection_changed)
        self._btn_merge_labels.clicked.connect(self._merge_selected_labels)

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

        if files and coco_io.has_labelme_annotations(folder, files):
            with self._loading_progress(f"{os.path.basename(folder)} 여는 중…") as tick:
                try:
                    proj, mgrs = coco_io.load_labelme(folder, files, progress=tick)
                    self.project = proj
                    self._mask_managers = mgrs
                    self.save_path = folder
                    self._refresh_class_list()
                except Exception:
                    pass

        self._refresh_image_icons()
        if files:
            self._img_list.setCurrentRow(0)
        self._update_title()

    @contextlib.contextmanager
    def _loading_progress(self, label: str):
        """Yield a progress(done, total) callback backed by a QProgressDialog.

        Reading a folder's annotations takes about 17ms per image, so a few
        hundred images is a multi-second freeze with no feedback. The dialog
        only appears once the work has already run past minimumDuration, which
        keeps small folders from flashing a box open and shut.

        No cancel button: aborting midway would leave the project half-loaded,
        and the point here is feedback rather than control.
        """
        dlg = QProgressDialog(label, None, 0, 100, self)
        dlg.setWindowTitle("HyLabel")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        def tick(done: int, total: int) -> None:
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()

        try:
            yield tick
        finally:
            dlg.close()

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
        target = self.save_path or self.image_dir
        if not target:
            self._save_as()
            return
        self._do_save(target)

    def _save_as(self) -> None:
        default = self.image_dir or self._last_dir
        directory = QFileDialog.getExistingDirectory(
            self, "Save Annotations — Select Folder", default
        )
        if not directory:
            return
        self._do_save(directory)

    def _do_save(self, directory: str) -> None:
        try:
            coco_io.save_labelme(self.project, self._mask_managers, directory)
            self.save_path = directory
            self._modified = False
            self._update_title()
            self._lbl_status.setText(f"Saved → {os.path.basename(directory)}/")
            self._refresh_image_icons()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_annotations(self) -> None:
        default = self.image_dir or self._last_dir
        directory = QFileDialog.getExistingDirectory(
            self, "Load Annotations — Select JSON Folder", default
        )
        if not directory:
            return
        files = [self._img_list.item(i).text() for i in range(self._img_list.count())]
        if not files:
            QMessageBox.information(self, "No Images", "Open an image folder first.")
            return
        try:
            with self._loading_progress(
                f"{os.path.basename(directory)} 어노테이션 불러오는 중…"
            ) as tick:
                proj, mgrs = coco_io.load_labelme(directory, files, progress=tick)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
            return
        self.project = proj
        self._mask_managers = mgrs
        self.save_path = directory
        self._refresh_class_list()
        row = self._img_list.currentRow()
        if row >= 0:
            self._on_image_selected(row)
        self._lbl_status.setText(f"Loaded from: {os.path.basename(directory)}/")
        self._refresh_image_icons()

    def _load_json_for_image(self, filename: str, w: int, h: int) -> Optional[MaskManager]:
        """Try to load JSON for a single image; merge categories into current project."""
        if not self.image_dir:
            return None
        try:
            proj, mgrs = coco_io.load_labelme(self.image_dir, [filename])
        except Exception:
            return None
        if not mgrs:
            return None
        for cat in proj.categories:
            if not any(c.name == cat.name for c in self.project.categories):
                self.project.add_category(cat.name)
        self._refresh_class_list()
        return next(iter(mgrs.values()))

    # ── image navigation ──────────────────────────────────────────────────────

    def _on_image_selected(self, row: int) -> None:
        if row < 0:
            return
        self._undo_stack.clear()
        self._sam_img_path = ""  # force re-encode on next magic click

        name = self._img_list.item(row).text()
        path = os.path.join(self.image_dir, name)

        w, h = self.canvas.load_image(path)
        img_ann = self.project.get_or_create_image(path, w, h)
        self.current_img_ann = img_ann

        # Get or create MaskManager for this image
        mgr = self._mask_managers.get(img_ann.image_id)
        if mgr is None:
            mgr = self._load_json_for_image(name, w, h) or MaskManager(w, h)
            self._mask_managers[img_ann.image_id] = mgr

        self.canvas.set_mask_manager(mgr, self._color_tuples())

        # Back to the arrow tool for the new image
        self._activate_select_tool()

        self._update_active_class()
        self._refresh_labels()
        self._lbl_status.setText(f"{name}  ({w}×{h})")
        self._update_title()

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
        self._update_label_class_combo()

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
        ann = mgr.get_annotation(ann_id)
        rect = ann.bbox if ann is not None else None
        if ann is not None:
            self._push_undo({
                "type": "ann_deleted",
                "ann_id": ann.ann_id,
                "cat_id": ann.cat_id,
                "mask": ann.mask.copy(),
                "index": mgr.annotation_index(ann_id),
                "polygons": (
                    [[[x, y] for x, y in p] for p in ann.original_polygons]
                    if ann.original_polygons is not None else None),
            })
        self.canvas.clear_edit_annotation()
        mgr.remove_annotation(ann_id)
        self.canvas.refresh_overlay(rect)
        self._refresh_labels()
        self._mark_modified()
        self._show_class_contours()

        # Removing an annotation clears the selection, which drops the user back
        # on the Classes panel. Move to the row that slid up into the deleted
        # one's place instead — or the new last row if we deleted the bottom one
        # — so deleting several in a row does not need a trip back to the list.
        # Done last so the currentRowChanged handler, which enters edit mode,
        # runs after _show_class_contours() and clears the class dots itself.
        remaining = self._label_list.count()
        if remaining:
            self._label_list.setCurrentRow(min(row, remaining - 1))

    def _merge_selected_labels(self) -> None:
        """Combine the selected annotations into the first one."""
        if self.current_img_ann is None:
            return
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            return
        ids = self._selected_ann_ids()
        anns = [a for a in (mgr.get_annotation(i) for i in ids) if a is not None]
        if len(anns) < 2:
            self._lbl_status.setText(
                "합칠 레이블을 2개 이상 선택하세요 (Ctrl+클릭).")
            return

        keep = anns[0]
        cat_map = {c.id: c for c in self.project.categories}
        other_cats = {a.cat_id for a in anns} - {keep.cat_id}
        if other_cats:
            # Merging across classes silently discards the others' class, so
            # make the surviving one explicit before doing it.
            keep_name = cat_map[keep.cat_id].name if keep.cat_id in cat_map else "?"
            names = ", ".join(sorted(
                cat_map[c].name for c in other_cats if c in cat_map))
            if QMessageBox.question(
                self, "Merge across classes",
                f"선택한 레이블의 클래스가 다릅니다 ({names} → {keep_name}).\n"
                f"합친 결과는 '{keep_name}' 으로 남습니다. 계속할까요?",
            ) != QMessageBox.StandardButton.Yes:
                return

        self._push_undo({
            "type": "anns_merged",
            "keep_id": keep.ann_id,
            "members": [{
                "ann_id": a.ann_id,
                "cat_id": a.cat_id,
                "mask": a.mask.copy(),
                "index": mgr.annotation_index(a.ann_id),
                "polygons": (
                    [[[x, y] for x, y in p] for p in a.original_polygons]
                    if a.original_polygons is not None else None),
            } for a in anns],
        })

        rect = keep.bbox
        union = keep.mask.copy()
        for a in anns[1:]:
            union |= a.mask
            rect = MaskManager.union_bbox(rect, a.bbox)
        self.canvas.clear_edit_annotation()
        keep.mask[:] = union
        # The merged outline no longer matches any authored polygon, so the mask
        # becomes the source of truth and save extracts the combined contour.
        keep.original_polygons = None
        mgr.recompute_bbox(keep.ann_id)
        for a in anns[1:]:
            mgr.remove_annotation(a.ann_id)

        self.canvas.refresh_overlay(rect)
        self._refresh_labels()
        self._mark_modified()
        row = self._row_of_ann(keep.ann_id)
        if row >= 0:
            self._label_list.setCurrentRow(row)
        self._lbl_status.setText(f"{len(anns)}개 레이블을 하나로 합쳤습니다.")

    # ── tool toggling ─────────────────────────────────────────────────────────

    def _on_tool_toggled(self, checked: bool) -> None:
        if not checked:
            return
        # Dispatch on the action that fired, not on isChecked(). The action
        # group has not finished unchecking the previous tool when this runs, so
        # polling the others reads a stale True and picks the wrong branch.
        action = self.sender()
        # Hand and Select need no category — neither creates an annotation.
        if action is self._act_hand:
            self.canvas.set_mode(Mode.PAN)
            self.canvas.setFocus()
            return
        if action is self._act_select:
            self._pre_pan_action = self._act_select
            self.canvas.set_mode(Mode.SELECT)
            self.canvas.setFocus()
            return
        # Drawing tools: require at least one class
        if not self.project.categories:
            QMessageBox.information(
                self, "No Classes", "Please add at least one class first."
            )
            self._uncheck_all_tools()
            return
        if action is self._act_draw:
            self._pre_pan_action = self._act_draw
            self._update_active_class()
            self.canvas.set_mode(Mode.DRAW)
        elif action is self._act_brush:
            self._pre_pan_action = self._act_brush
            self._update_active_class()
            self.canvas.set_mode(Mode.BRUSH)
        elif action is self._act_magic:
            if self.canvas.is_editing:
                self.canvas.clear_edit_annotation()
                self.canvas.set_mode(Mode.IDLE)
                return
            self._pre_pan_action = self._act_magic
            self._update_active_class()
            self.canvas.set_mode(Mode.MAGIC)
        self.canvas.setFocus()

    def _toggle_pan(self) -> None:
        if self._act_hand.isChecked():
            prev = self._pre_pan_action
            if prev is not None:
                prev.setChecked(True)
            else:
                self._uncheck_all_tools()
                self.canvas.set_mode(Mode.IDLE)
        else:
            self._act_hand.setChecked(True)

    def _uncheck_all_tools(self) -> None:
        for a in (self._act_select, self._act_hand, self._act_draw,
                  self._act_brush, self._act_magic):
            a.blockSignals(True)
            a.setChecked(False)
            a.blockSignals(False)

    def _activate_select_tool(self) -> None:
        """Fall back to the arrow tool — the resting state between actions.

        Checked without blocking signals so the action group registers it as the
        current tool; otherwise the group keeps treating a stale action as
        checked, which breaks exclusivity and lets S toggle Select back off.
        """
        self._act_select.setChecked(True)
        self._pre_pan_action = self._act_select
        # setChecked is a no-op when it was already checked, so no toggled
        # signal fires and the mode has to be set here.
        if self.canvas.current_mode != "select":
            self.canvas.set_mode(Mode.SELECT)

    @staticmethod
    def _step_list(list_widget: QListWidget, delta: int) -> None:
        count = list_widget.count()
        if count == 0:
            return
        row = max(0, min(count - 1, list_widget.currentRow() + delta))
        if row != list_widget.currentRow():
            list_widget.setCurrentRow(row)

    # ── brush size ────────────────────────────────────────────────────────────

    def _adjust_size(self, delta: int) -> None:
        if self.canvas.current_mode == "magic":
            self._mask_slider.setValue(self._mask_slider.value() + delta)
        else:
            cur = self._brush_slider.value()
            step = 1 if cur <= 30 else (2 if cur <= 60 else 4)
            self._brush_slider.setValue(cur + (step if delta > 0 else -step))

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

    def _on_annotation_committed(self, ann_id: int) -> None:
        if self.current_img_ann is not None:
            mgr = self._mask_managers.get(self.current_img_ann.image_id)
            if mgr is not None:
                ann = mgr.get_annotation(ann_id)
                if ann is not None:
                    self._push_undo({
                        "type": "ann_added",
                        "ann_id": ann_id,
                        "cat_id": ann.cat_id,
                        "mask": ann.mask.copy(),
                    })
                cat_order = [c.id for c in self.project.categories]
                mgr.sort_by_category_order(cat_order)
        self._refresh_labels()
        for i in range(self._label_list.count()):
            it = self._label_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == ann_id:
                self._label_list.scrollToItem(it)
                break
        self._mark_modified()
        self._show_class_contours()

    def _on_canvas_select(self, x: float, y: float, additive: bool) -> None:
        """Arrow-tool click: select whichever label covers this pixel.

        additive (Ctrl held) toggles the label in the current selection instead
        of replacing it, so several can be picked for merging.
        """
        if self.current_img_ann is None:
            return
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            return
        ix, iy = int(x), int(y)
        if not (0 <= ix < mgr.width and 0 <= iy < mgr.height):
            return

        hits = [a for a in mgr.annotations()
                if a.bbox is not None
                and a.bbox[0] <= ix < a.bbox[2] and a.bbox[1] <= iy < a.bbox[3]
                and a.mask[iy, ix]]
        if not hits:
            if not additive:     # plain click on empty space deselects
                self._label_list.clearSelection()
                self._label_list.setCurrentRow(-1)
            return
        # Overlapping annotations: prefer the smallest, so a label nested inside
        # a larger one stays reachable.
        target = min(hits, key=lambda a: int(a.mask.sum()))
        row = self._row_of_ann(target.ann_id)
        if row < 0:
            return

        item = self._label_list.item(row)
        if additive:
            item.setSelected(not item.isSelected())
            # NoUpdate: the plain setCurrentItem() overload applies
            # ClearAndSelect, which would wipe the very selection being built.
            self._label_list.setCurrentItem(
                item, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            self._label_list.setCurrentRow(row)
        # Selecting a label arms the brush; undo that so the arrow tool stays
        # active and the user can keep clicking from label to label.
        self._act_select.setChecked(True)

    def _row_of_ann(self, ann_id: int) -> int:
        for i in range(self._label_list.count()):
            item = self._label_list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == ann_id:
                return i
        return -1

    def _selected_ann_ids(self) -> List[int]:
        """Selected annotation ids, in list order."""
        return [self._label_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self._label_list.count())
                if self._label_list.item(i).isSelected()]

    def _on_label_selection_changed(self) -> None:
        """Route the selection: one row edits, several only highlight."""
        rows = [i for i in range(self._label_list.count())
                if self._label_list.item(i).isSelected()]
        self._btn_merge_labels.setEnabled(len(rows) >= 2)

        if len(rows) == 1:
            self._on_label_selected(rows[0])
            return

        # Zero or many: no single annotation to edit, so leave edit mode and let
        # _show_class_contours() outline whatever is selected.
        self._syncing_selection = True
        try:
            self.canvas.clear_edit_annotation()
        finally:
            self._syncing_selection = False
        self._clear_label_bold()
        self._update_label_class_combo()
        self._show_class_contours()
        if len(rows) >= 2:
            self._lbl_mode.setText(
                f"{len(rows)} labels selected  (Home = merge)")

    def _on_class_clicked(self, _) -> None:
        self.canvas.clear_edit_annotation()
        self._clear_label_bold()
        self._show_class_contours()

    def _show_class_contours(self) -> None:
        """Read-only outline dots on the canvas.

        A multi-selection takes priority: with two labels picked for merging the
        user needs to see *those*, not every label of the active class. One or
        no selection falls back to the class overview.
        """
        if self.current_img_ann is None:
            self.canvas.clear_class_contours()
            return
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            self.canvas.clear_class_contours()
            return

        selected = set(self._selected_ann_ids())
        if len(selected) >= 2:
            masks = [a.mask for a in mgr.annotations() if a.ann_id in selected]
            self.canvas.show_class_contours(masks)
            return

        row = self._class_list.currentRow()
        if row < 0:
            self.canvas.clear_class_contours()
            return
        cat_id = self._class_list.item(row).data(Qt.ItemDataRole.UserRole)
        masks = [ann.mask for ann in mgr.annotations() if ann.cat_id == cat_id]
        self.canvas.show_class_contours(masks)

    def _select_last_label(self) -> None:
        count = self._label_list.count()
        if count == 0:
            return
        # Signals left live so the selection handler runs and resets the
        # multi-select state (merge button, mode label) along with it.
        self._label_list.setCurrentRow(count - 1)

    def _on_label_selected(self, row: int) -> None:
        if row < 0 or self.current_img_ann is None:
            self.canvas.clear_edit_annotation()
            self._update_label_class_combo()
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
                self._act_brush.setChecked(True)
                self._lbl_mode.setText(
                    f"Editing: {cat.name if cat else '?'}  #{n}"
                    "   (drag points / B=brush  M=done)"
                )
                self._set_label_bold(row)
                self._clear_class_bold()
                break
        self._update_label_class_combo()

    def _on_edit_changed(self, ann_id: int) -> None:
        """Called when an annotation's mask is modified by brush or contour drag."""
        self._mark_modified()

    def _on_edit_cleared(self) -> None:
        # Skip the selection reset when we are the ones who ended edit mode:
        # picking a second label leaves single-edit, and wiping the list here
        # would destroy the multi-selection the user is building.
        if not self._syncing_selection:
            self._label_list.blockSignals(True)
            self._label_list.clearSelection()
            self._label_list.setCurrentRow(-1)
            self._label_list.blockSignals(False)
        self._clear_label_bold()
        self._update_label_class_combo()
        self._update_class_bold(self._class_list.currentRow())
        self._on_mode_changed(self.canvas.current_mode)
        self._show_class_contours()

    # ── undo ──────────────────────────────────────────────────────────────────

    def _push_undo(self, record: dict) -> None:
        self._undo_stack.append(record)
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _handle_undo(self) -> None:
        """Ctrl+Z dispatcher: polygon-point undo is canvas-local, rest via undo stack."""
        if self.canvas.current_mode == "draw" and self.canvas.has_draft_points():
            self.canvas.undo_draw_point()
        else:
            self._do_undo()

    def _do_undo(self) -> None:
        if not self._undo_stack:
            return
        record = self._undo_stack.pop()
        t = record["type"]

        if t == "pending_brush":
            self.canvas.restore_pending_mask(record["mask"])

        elif t == "edit_stroke":
            ann_id = record["ann_id"]
            if self.current_img_ann is None:
                return
            mgr = self._mask_managers.get(self.current_img_ann.image_id)
            if mgr is None:
                return
            ann = mgr.get_annotation(ann_id)
            if ann is not None:
                old = ann.bbox
                ann.mask[:] = record["mask"]
                # Restore the polygon alongside the mask — original_polygons is
                # what gets saved, so undoing one without the other would leave
                # the JSON reflecting an edit the user just undid.
                ann.original_polygons = record.get("polygons")
                mgr.recompute_bbox(ann_id)   # mask replaced wholesale
                self.canvas.refresh_overlay(
                    MaskManager.union_bbox(old, ann.bbox))
                self.canvas.refresh_edit_contour()
            self._mark_modified()

        elif t == "ann_added":
            ann_id = record["ann_id"]
            if self.current_img_ann is None:
                return
            mgr = self._mask_managers.get(self.current_img_ann.image_id)
            if mgr is None:
                return
            ann = mgr.get_annotation(ann_id)
            rect = ann.bbox if ann is not None else None
            self.canvas.clear_edit_annotation()
            mgr.remove_annotation(ann_id)
            self.canvas.refresh_overlay(rect)
            self._refresh_labels()
            self._mark_modified()

        elif t == "ann_deleted":
            if self.current_img_ann is None:
                return
            mgr = self._mask_managers.get(self.current_img_ann.image_id)
            if mgr is None:
                return
            mgr.restore_annotation(
                record["ann_id"], record["cat_id"],
                record["mask"], record.get("index"),
                record.get("polygons"),
            )
            restored = mgr.get_annotation(record["ann_id"])
            self.canvas.refresh_overlay(
                restored.bbox if restored is not None else None)
            self._refresh_labels()
            self._mark_modified()

        elif t == "anns_merged":
            if self.current_img_ann is None:
                return
            mgr = self._mask_managers.get(self.current_img_ann.image_id)
            if mgr is None:
                return
            self.canvas.clear_edit_annotation()
            merged = mgr.get_annotation(record["keep_id"])
            rect = merged.bbox if merged is not None else None
            mgr.remove_annotation(record["keep_id"])
            # Ascending index order, so each insert lands where it started.
            for m in sorted(record["members"], key=lambda d: d["index"]):
                mgr.restore_annotation(m["ann_id"], m["cat_id"], m["mask"],
                                       m["index"], m["polygons"])
                rect = MaskManager.union_bbox(
                    rect, mgr.get_annotation(m["ann_id"]).bbox)
            self.canvas.refresh_overlay(rect)
            self._refresh_labels()
            self._mark_modified()

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
            "select": "Mode: Select  (click a label to select  /  drag its points to edit)",
            "pan":   "Mode: Pan  (drag to move image)",
            "draw":  "Mode: Draw  (double-click or snap to close)",
            "brush": "Mode: Brush  (LMB: paint  /  RMB: erase)",
            "magic": "Mode: AI Magic Wand  (LMB: include  /  RMB: exclude  /  Enter: commit  /  Esc: reset)",
        }
        self._lbl_mode.setText(labels.get(mode_str, f"Mode: {mode_str}"))
        if mode_str == "idle":
            self._uncheck_all_tools()

    # ── AI magic wand (EdgeSAM / SAM2) ──────────────────────────────────────────

    def _on_sam_model_changed(self, combo_idx: int) -> None:
        key = self._sam_model_combo.itemData(combo_idx)
        if key is None or key == self._sam_model_key:
            return
        self._sam_model_key = key
        self._settings.setValue("samModel", key)
        self._sam_predictor = None
        self._sam_img_path = ""
        self._mask_slider.setEnabled(False)
        self.canvas.clear_magic(keep_pending=False)
        label = sam_worker.MODEL_INFO[key]["label"]
        self._lbl_status.setText(f"AI model switched to {label}")

    def _on_mask_slider_changed(self, idx: int) -> None:
        self._mask_idx_lbl.setText(f"{idx + 1}/3")
        self.canvas.set_magic_mask_idx(idx)

    def _on_magic_requested(self, points: object, labels: object) -> None:
        if not self._ensure_sam_loaded():
            return
        current_item = self._img_list.currentItem()
        if current_item is None:
            return
        img_path = os.path.join(self.image_dir, current_item.text())
        if img_path != self._sam_img_path:
            if not self._sam_encode_image(img_path):
                return
        self._lbl_status.setText("AI: predicting…")
        QApplication.processEvents()
        try:
            masks, scores = self._sam_predictor.predict(points, labels)  # type: ignore[union-attr]
            self.canvas.set_magic_preview(masks, self._mask_slider.value())
            self._mask_slider.setEnabled(True)
            score_str = "  ".join(f"{s:.2f}" for s in scores)
            self._lbl_status.setText(
                f"AI: {len(points)} point(s)  |  scores: [{score_str}]")
        except Exception as e:
            self._lbl_status.setText(f"SAM predict error: {e}")

    def _ensure_sam_loaded(self) -> bool:
        if self._sam_predictor is not None:
            return True
        key = self._sam_model_key
        label = sam_worker.MODEL_INFO[key]["label"]
        if not sam_worker.is_installed():
            QMessageBox.warning(
                self, "onnxruntime not installed",
                "onnxruntime is required for AI Magic Wand.\n\n"
                "Run:  pip install onnxruntime\n\n"
                "With an NVIDIA GPU, install onnxruntime-gpu instead "
                "(see README, install step 4).",
            )
            return False
        missing = sam_worker.missing_weights(key)
        if missing:
            QMessageBox.warning(
                self, "Weights not found",
                f"{label} ONNX weights not found.\n\n"
                f"Run:  python download_weights.py --model {key}\n\n"
                "Expected:\n" + "\n".join(f"  {p}" for p in missing),
            )
            return False
        self._lbl_status.setText(f"Loading {label} model…")
        QApplication.processEvents()
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self._sam_predictor = sam_worker.create_predictor(key)
            device = self._sam_predictor.device  # type: ignore[union-attr]
            self._lbl_status.setText(f"{label} loaded  ({device})")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        return True

    def _sam_encode_image(self, img_path: str) -> bool:
        import cv2
        import numpy as np
        self._lbl_status.setText("AI: encoding image…")
        QApplication.processEvents()
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise ValueError(f"Cannot read: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._sam_predictor.set_image(rgb)  # type: ignore[union-attr]
            self._sam_img_path = img_path
        except Exception as e:
            self._lbl_status.setText(f"SAM encode error: {e}")
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        return True

    # ── faint / gamma ─────────────────────────────────────────────────────────

    _FAINT_NAMES = ("원본", "흐리게", "더 흐리게")

    def _toggle_faint(self) -> None:
        """V cycles the overlay opacity 0 → 1 → 2 → 0."""
        level = (self.canvas.faint_level + 1) % len(self._FAINT_NAMES)
        self.canvas.set_faint_level(level)
        # The action is checkable, and Qt has already flipped it by the time we
        # get here. Re-drive it from the level so the tick means "not original"
        # rather than tracking its own two-state toggle.
        self._act_faint.setChecked(level != 0)
        self._lbl_status.setText(
            f"레이블 투명도: {self._FAINT_NAMES[level]}  ({level + 1}/3)")

    def _toggle_gamma(self) -> None:
        self.canvas.set_gamma_enabled(self._act_gamma.isChecked())

    def _open_gamma_dialog(self) -> None:
        if self._gamma_dialog is not None:
            self._gamma_dialog.show()
            self._gamma_dialog.raise_()
            self._gamma_dialog.activateWindow()
            return
        self._gamma_dialog = GammaCurveDialog(self._gamma_ctrl, self)
        self._gamma_dialog.lut_changed.connect(self._on_gamma_lut_changed)
        self._gamma_dialog.show()

    def _on_gamma_lut_changed(self, lut: object) -> None:
        if self._gamma_dialog is not None:
            self._gamma_ctrl = self._gamma_dialog.control_points()
            self._settings.setValue("gammaY0", self._gamma_ctrl[0][1])
            self._settings.setValue("gammaY1", self._gamma_ctrl[1][1])
            self._settings.setValue("gammaY2", self._gamma_ctrl[2][1])
        self.canvas.set_gamma_lut(lut)  # type: ignore[arg-type]

    # ── label class change ────────────────────────────────────────────────────

    def _update_label_class_combo(self) -> None:
        self._label_class_combo.blockSignals(True)
        self._label_class_combo.clear()
        row = self._label_list.currentRow()
        if row < 0 or self.current_img_ann is None:
            self._label_class_combo.setEnabled(False)
            self._label_class_combo.blockSignals(False)
            return
        item = self._label_list.item(row)
        if item is None:
            self._label_class_combo.setEnabled(False)
            self._label_class_combo.blockSignals(False)
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            self._label_class_combo.setEnabled(False)
            self._label_class_combo.blockSignals(False)
            return
        ann = mgr.get_annotation(ann_id)
        if ann is None:
            self._label_class_combo.setEnabled(False)
            self._label_class_combo.blockSignals(False)
            return
        self._label_class_combo.setEnabled(True)
        for cat in self.project.categories:
            self._label_class_combo.addItem(_color_icon(cat.color, 12), cat.name, cat.id)
        for i in range(self._label_class_combo.count()):
            if self._label_class_combo.itemData(i) == ann.cat_id:
                self._label_class_combo.setCurrentIndex(i)
                break
        self._label_class_combo.blockSignals(False)

    def _on_label_class_changed(self, idx: int) -> None:
        if idx < 0 or self.current_img_ann is None:
            return
        new_cat_id = self._label_class_combo.itemData(idx)
        if new_cat_id is None:
            return
        row = self._label_list.currentRow()
        if row < 0:
            return
        item = self._label_list.item(row)
        if item is None:
            return
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        mgr = self._mask_managers.get(self.current_img_ann.image_id)
        if mgr is None:
            return
        ann = mgr.get_annotation(ann_id)
        if ann is None or ann.cat_id == new_cat_id:
            return
        mgr.change_annotation_category(ann_id, new_cat_id)
        cat_order = [c.id for c in self.project.categories]
        mgr.sort_by_category_order(cat_order)
        self.canvas.refresh_overlay()
        self._refresh_labels()
        for i in range(self._label_list.count()):
            it = self._label_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == ann_id:
                self._label_list.setCurrentRow(i)
                break
        self._mark_modified()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_image_icons(self) -> None:
        json_dir = self.save_path or self.image_dir
        if not json_dir:
            return
        has_icon = _json_doc_icon()
        no_icon = QIcon()
        for i in range(self._img_list.count()):
            item = self._img_list.item(i)
            if item is None:
                continue
            stem = os.path.splitext(item.text())[0]
            exists = os.path.isfile(os.path.join(json_dir, stem + ".json"))
            item.setIcon(has_icon if exists else no_icon)

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
        if not self.image_dir:
            self.setWindowTitle("HyLabel")
            return
        base = os.path.basename(self.image_dir)
        item = self._img_list.currentItem()
        name = f"  /  {item.text()}" if item is not None else ""
        marker = " *" if self._modified else ""
        self.setWindowTitle(f"HyLabel — {base}{name}{marker}")

    def _confirm_discard(self) -> bool:
        if not self._modified:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "저장하지 않은 변경 사항이 있습니다.",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._save()
            return not self._modified  # False if save was itself cancelled
        return reply == QMessageBox.StandardButton.Discard

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.canvas.keyPressEvent(event)
            return
        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Shift:
                self._shift_alone = True
            else:
                self._shift_alone = False
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.canvas.keyReleaseEvent(event)
            return
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            if self._shift_alone:
                self._select_last_label()
            self._shift_alone = False
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
