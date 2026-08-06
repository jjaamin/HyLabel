import os
import sys

import cv2
from PyQt6.QtWidgets import QApplication
from .window import MainWindow


def _quiet_image_loaders() -> None:
    """Silence the per-file TIFF chatter both image readers produce.

    Vendor TIFFs carry private tags (65006-65027 on ours) that libtiff reports
    as "TIFFReadDirectory: Unknown field with tag ... encountered", once per tag
    per file. Nothing is wrong and there is nothing to act on.

    The two readers need separate handling: Qt's plugin logs through the Qt
    categories, while cv2.imread routes libtiff's warnings through OpenCV's own
    logger, which the Qt rule cannot reach. Raising OpenCV to ERROR keeps real
    failures visible — a missing file still returns None and bad arguments still
    raise.
    """
    os.environ.setdefault("QT_LOGGING_RULES", "qt.imageformats.tiff=false")
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def main() -> None:
    _quiet_image_loaders()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
