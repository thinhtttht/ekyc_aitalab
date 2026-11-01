"""Entry point for the eKYC Native Desktop GUI application (PyQt6)."""
from __future__ import annotations

import os
import sys

# Đảm bảo đường dẫn backend và desktop_app nằm trong sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
desktop_dir = os.path.join(root_dir, "desktop_app")

for d in [root_dir, backend_dir, desktop_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    # Bật hỗ trợ màn hình High-DPI
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("eKYC Proctor Desktop")
    app.setOrganizationName("eKYC Biometrics")

    # Đặt font chữ mặc định mượt mà
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
