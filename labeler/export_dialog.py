"""Dialog for File → Export for Training."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from .export_training import IMAGES_DIR, LABELS_DIR, CLASSES_FILE

_MAX_SIDE = 20000


class ExportTrainingDialog(QDialog):
    """Pick an output folder and the size the pair should be written at.

    Width and height are tied to the source aspect ratio while "비율 고정" is on,
    so editing one moves the other. The lock can be released for the odd case
    where a trainer wants a fixed square input and the distortion is acceptable.
    """

    def __init__(self, src_w: int, src_h: int, image_count: int,
                 mixed_sizes: bool = False, start_dir: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export for Training")
        self.setMinimumWidth(460)
        self._src_w = max(1, src_w)
        self._src_h = max(1, src_h)
        self._syncing = False

        self._dir_edit = QLineEdit(start_dir)
        self._dir_edit.setPlaceholderText("내보낼 폴더를 선택하세요")
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._pick_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse)

        self._width = QSpinBox()
        self._width.setRange(1, _MAX_SIDE)
        self._width.setValue(self._src_w)
        self._width.setSuffix(" px")
        self._height = QSpinBox()
        self._height.setRange(1, _MAX_SIDE)
        self._height.setValue(self._src_h)
        self._height.setSuffix(" px")
        self._lock = QCheckBox("비율 고정")
        self._lock.setChecked(True)
        self._reset = QPushButton("원본 크기")
        self._reset.clicked.connect(self._restore_source)

        self._width.valueChanged.connect(self._width_changed)
        self._height.valueChanged.connect(self._height_changed)

        size_box = QGroupBox("크기")
        form = QFormLayout(size_box)
        form.addRow("Width:", self._width)
        form.addRow("Height:", self._height)
        lock_row = QHBoxLayout()
        lock_row.addWidget(self._lock)
        lock_row.addStretch(1)
        lock_row.addWidget(self._reset)
        form.addRow("", self._wrap(lock_row))

        source = QLabel(f"원본: {self._src_w} × {self._src_h}  ·  이미지 {image_count}장")
        source.setStyleSheet("color: #888;")

        self._warn = QLabel(
            "이미지 크기가 서로 다릅니다. 모두 위 크기로 맞춰 저장되므로 "
            "일부는 비율이 달라질 수 있습니다.")
        self._warn.setWordWrap(True)
        self._warn.setStyleSheet("color: #c47f00;")
        self._warn.setVisible(mixed_sizes)

        self._auto_contrast = QCheckBox("Auto-contrast (8bit 변환 시 자동 대비)")
        self._auto_contrast.setToolTip(
            "끄면 비트 깊이 전체 범위를 그대로 8bit에 대응시킵니다 (모든 이미지에 동일한 변환).\n"
            "켜면 이미지마다 자기 데이터의 0.5~99.5 백분위로 늘립니다 — "
            "16bit 컨테이너에 12bit 데이터가 든 경우처럼 어둡게 나올 때 쓰세요.")

        layout = QLabel(
            f"출력 구조:\n"
            f"    {IMAGES_DIR}/<이름>.png      8bit 원본 이미지\n"
            f"    {LABELS_DIR}/<이름>.png      배경 0, 클래스는 Classes 순서대로 1, 2, 3 …\n"
            f"    {CLASSES_FILE}            인덱스 ↔ 클래스 이름")
        layout.setStyleSheet(
            "color: #888; font-family: Consolas, monospace; font-size: 11px;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._dir_edit.textChanged.connect(self._update_ok)
        self._update_ok()

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("출력 폴더"))
        outer.addLayout(dir_row)
        outer.addWidget(source)
        outer.addWidget(size_box)
        outer.addWidget(self._warn)
        outer.addWidget(self._auto_contrast)
        outer.addWidget(layout)
        outer.addWidget(buttons)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(inner) -> QLabel:
        from PyQt6.QtWidgets import QWidget
        holder = QWidget()
        holder.setLayout(inner)
        return holder

    def _pick_dir(self) -> None:
        start = self._dir_edit.text() or ""
        chosen = QFileDialog.getExistingDirectory(self, "Export Folder", start)
        if chosen:
            self._dir_edit.setText(chosen)

    def _update_ok(self) -> None:
        self._ok.setEnabled(bool(self._dir_edit.text().strip()))

    def _width_changed(self, value: int) -> None:
        if self._syncing or not self._lock.isChecked():
            return
        self._syncing = True
        self._height.setValue(max(1, round(value * self._src_h / self._src_w)))
        self._syncing = False

    def _height_changed(self, value: int) -> None:
        if self._syncing or not self._lock.isChecked():
            return
        self._syncing = True
        self._width.setValue(max(1, round(value * self._src_w / self._src_h)))
        self._syncing = False

    def _restore_source(self) -> None:
        self._syncing = True
        self._width.setValue(self._src_w)
        self._height.setValue(self._src_h)
        self._syncing = False

    # ── result ────────────────────────────────────────────────────────────────

    def out_dir(self) -> str:
        return self._dir_edit.text().strip()

    def target_size(self) -> Tuple[int, int]:
        return self._width.value(), self._height.value()

    def auto_contrast(self) -> bool:
        return self._auto_contrast.isChecked()
