import os
import sys
from PyQt6.QtWidgets import QApplication
from .window import MainWindow


def main() -> None:
    os.environ.setdefault("QT_LOGGING_RULES", "qt.imageformats.tiff=false")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
