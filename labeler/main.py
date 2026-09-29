import os
import sys

# These have to be set before the libraries that read them load, so they sit
# above the imports rather than in _quiet_image_loaders() below. OpenCV reads
# OPENCV_LOG_LEVEL once, while its logger initialises on import.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Qt categories we have nothing to do about:
#
#   qt.imageformats.tiff  one line per private tag in a vendor TIFF, per file.
#
#   qt.qpa.window         "SetProcessDpiAwarenessContext() failed: Access is
#                         denied", printed when something else in the process
#                         has already fixed its DPI awareness before Qt starts.
#                         Corporate document-security software injects a DLL
#                         that does exactly that, and a process cannot change
#                         its DPI awareness twice, so Qt's attempt is refused.
#                         Qt carries on with the awareness already in force.
#
# Ours go first so anything the user set in their own environment is applied
# after, and wins.
_LOGGING_RULES = "qt.imageformats.tiff=false;qt.qpa.window.warning=false"
_existing = os.environ.get("QT_LOGGING_RULES", "")
os.environ["QT_LOGGING_RULES"] = (
    f"{_LOGGING_RULES};{_existing}" if _existing else _LOGGING_RULES)

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

    The environment variables above already do this job on any build. The call
    below is a second pass for the case where something imported cv2 before this
    module got to set them, and it is wrapped because not every OpenCV build
    exposes cv2.utils.logging — one that does not used to take the whole app
    down on startup over a log setting.
    """
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


def main() -> None:
    _quiet_image_loaders()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
