"""Full-screen optical flashing overlay widget.
Inherits the exact full-screen illumination mechanism from the web application,
lighting up the entire monitor while transparently cutting out the face oval.
"""
from __future__ import annotations

from typing import Optional
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class FullScreenFlashOverlay(QWidget):
    """Lớp phủ phát quang toàn màn hình (Full-Screen Color Flashing) khoét rỗng khung Oval."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.flash_color: Optional[QColor] = None
        self.oval_rect: Optional[QRectF] = None
        self.status_text = ""
        self.hide()

    def set_flash(self, color: Optional[QColor], oval_rect: Optional[QRectF] = None, text: str = ""):
        """Thiết lập màu nháy toàn màn hình và toạ độ lỗ khoét Oval."""
        self.flash_color = color
        self.oval_rect = oval_rect
        self.status_text = text

        if color is not None:
            if self.parent():
                self.setGeometry(self.parent().rect())
            self.show()
            self.raise_()
        else:
            self.hide()
        self.update()

    def paintEvent(self, event):
        if self.flash_color is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Toàn bộ màn hình cửa sổ
        full_path = QPainterPath()
        full_path.addRect(QRectF(self.rect()))

        # Khoét rỗng khung Oval khuôn mặt
        if self.oval_rect is not None:
            oval_path = QPainterPath()
            oval_path.addEllipse(self.oval_rect)
            mask_path = full_path.subtracted(oval_path)
        else:
            mask_path = full_path

        # Chiếu sáng rực rỡ toàn bộ màn hình (alpha = 255: 100% đậm đặc bão hòa, không bị pha loãng)
        fill_brush = QColor(
            self.flash_color.red(),
            self.flash_color.green(),
            self.flash_color.blue(),
            255,
        )
        painter.fillPath(mask_path, fill_brush)

        # Viền Oval phát sáng công nghệ cao (Dual Chroma Glow Ring)
        if self.oval_rect is not None:
            # Vòng hào quang ngoại vi (Outer Neon Halo)
            outer_glow = QPen(
                QColor(self.flash_color.red(), self.flash_color.green(), self.flash_color.blue(), 160),
                10,
            )
            painter.setPen(outer_glow)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.oval_rect)

            # Viền lõi sắc nét (Inner Sharp Chroma Rim)
            inner_rim = QPen(
                QColor(255, 255, 255, 230),
                3,
            )
            painter.setPen(inner_rim)
            painter.drawEllipse(self.oval_rect)

        # Banner chỉ dẫn nổi bật ở đáy màn hình
        if self.status_text:
            painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            banner_w = 520
            banner_h = 46
            bx = (self.width() - banner_w) / 2
            by = self.height() - 75
            banner_rect = QRectF(bx, by, banner_w, banner_h)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 235))
            painter.drawRoundedRect(banner_rect, 23, 23)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(banner_rect, Qt.AlignmentFlag.AlignCenter, self.status_text)
