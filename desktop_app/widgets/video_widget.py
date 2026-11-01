"""Custom video display widget with clean biometric overlays, portrait oval, and stage animations."""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QFont,
    QBrush,
    QPainterPath,
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

import config as C


class VideoWidget(QWidget):
    """Widget for displaying video stream with clean, elegant eKYC overlays (Apple FaceID aesthetic)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 450)

        # Trạng thái Oval
        self.show_oval = True
        self.is_zoom = False
        self.oval_color = QColor(56, 189, 248)  # Cyan default
        self.oval_progress = 0.0  # 0.0 -> 1.0 (tiến trình vòng cung chạy quanh oval)
        self.guidance_text = ""
        self.stage_icon = "🎯"
        self.badge_text = ""
        self.badge_color = QColor(16, 185, 129)

        # Mũi tên hướng quay đầu ('left', 'right', None)
        self.turn_arrow: Optional[str] = None

        # Lớp chiếu màu quang học (Active Optical Color Flashing)
        self.flash_color: Optional[QColor] = None

        self._pixmap: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def set_oval_zoom(self, zoom: bool):
        """Chuyển đổi kích thước Oval: Normal (FQA) hoặc Phóng to (Zoom-in)."""
        self.is_zoom = zoom
        self.update()

    def get_oval_cfg(self) -> dict:
        return C.OVAL_ZOOM if self.is_zoom else C.OVAL_NORMAL

    def get_viewport_rect(self) -> QRectF:
        """Tính toán toạ độ QRectF của khung nhìn 4:5 bên trong VideoWidget."""
        target_aspect = 4.0 / 5.0
        avail_w = self.width() - 20
        avail_h = self.height() - 20

        if avail_w / max(1, avail_h) > target_aspect:
            dest_h = avail_h
            dest_w = int(dest_h * target_aspect)
        else:
            dest_w = avail_w
            dest_h = int(dest_w / target_aspect)

        dx = (self.width() - dest_w) // 2
        dy = (self.height() - dest_h) // 2
        return QRectF(dx, dy, dest_w, dest_h)

    def get_oval_window_rect(self) -> QRectF:
        """Trả về toạ độ QRectF của khung Oval trong hệ toạ độ cửa sổ MainWindow."""
        dest_rect = self.get_viewport_rect()
        oval_cfg = self.get_oval_cfg()
        cx = dest_rect.x() + dest_rect.width() * oval_cfg["cx"]
        cy = dest_rect.y() + dest_rect.height() * oval_cfg["cy"]
        rx = dest_rect.width() * oval_cfg["rx"]
        ry = dest_rect.height() * oval_cfg["ry"]
        local_pt = QPointF(cx - rx, cy - ry).toPoint()
        top_window = self.window()
        mapped_pt = self.mapTo(top_window, local_pt)
        return QRectF(float(mapped_pt.x()), float(mapped_pt.y()), rx * 2, ry * 2)

    def set_oval_state(self, color_name: str = "cyan", text: str = "", progress: float = 0.0, icon: str = "🎯"):
        """Cập nhật màu sắc viền oval, tiến trình và lời nhắc."""
        color_map = {
            "cyan": QColor(56, 189, 248),
            "green": QColor(16, 185, 129),
            "yellow": QColor(245, 158, 11),
            "amber": QColor(245, 158, 11),
            "violet": QColor(167, 139, 250),
            "purple": QColor(167, 139, 250),
            "red": QColor(239, 68, 68),
            "gray": QColor(148, 163, 184),
        }
        self.oval_color = color_map.get(color_name.lower(), QColor(56, 189, 248))
        self.guidance_text = text
        self.oval_progress = max(0.0, min(1.0, progress))
        self.stage_icon = icon
        self.update()

    def set_turn_arrow(self, direction: Optional[str]):
        """Hiển thị mũi tên chỉ dẫn quay đầu ('left', 'right', hoặc None)."""
        self.turn_arrow = direction
        self.update()

    def set_flash_color(self, qcolor: Optional[QColor]):
        """Thiết lập màu nhấp nháy quang học chống giả mạo."""
        self.flash_color = qcolor
        self.update()

    def set_landmarks(self, keypoints: Optional[dict] = None, parts_status: Optional[dict] = None):
        """Bỏ qua việc vẽ mốc khuôn mặt trên màn hình để mặt người dùng tự nhiên và rõ ràng."""
        pass

    def set_badge(self, text: str, color_name: str = "green"):
        """Đặt huy hiệu trạng thái góc trên."""
        color_map = {
            "green": QColor(16, 185, 129),
            "red": QColor(239, 68, 68),
            "yellow": QColor(245, 158, 11),
            "blue": QColor(56, 189, 248),
            "violet": QColor(167, 139, 250),
        }
        self.badge_text = text
        self.badge_color = color_map.get(color_name, QColor(16, 185, 129))
        self.update()

    def update_frame(self, frame_bgr: np.ndarray):
        """Chuyển đổi frame OpenCV BGR sang QPixmap với bộ nhớ liền mạch tuyệt đối."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        frame_bgr = np.ascontiguousarray(frame_bgr)
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_frame(self, frame_bgr: np.ndarray, face_result=None, show_oval: bool = True):
        """Tương thích cập nhật frame kèm tùy chọn hiển thị Oval."""
        self.show_oval = show_oval
        self.update_frame(frame_bgr)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Nền tối
        painter.fillRect(self.rect(), QColor(11, 17, 32))

        if self._pixmap is None:
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Đang kết nối camera...")
            return

        dest_rect = self.get_viewport_rect()
        viewport_path = QPainterPath()
        viewport_path.addRoundedRect(dest_rect, 24, 24)

        painter.save()
        painter.setClipPath(viewport_path)

        # 1. Vẽ camera chuẩn nét
        scaled_pixmap = self._pixmap.scaled(
            int(dest_rect.width()),
            int(dest_rect.height()),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(int(dest_rect.x()), int(dest_rect.y()), scaled_pixmap)

        # 2. Vẽ MẶT NẠ OVAL VÀ LỚP CHIẾU MÀU QUANG HỌC
        oval_cfg = self.get_oval_cfg()
        if self.show_oval:
            cx = dest_rect.x() + dest_rect.width() * oval_cfg["cx"]
            cy = dest_rect.y() + dest_rect.height() * oval_cfg["cy"]
            rx = dest_rect.width() * oval_cfg["rx"]
            ry = dest_rect.height() * oval_cfg["ry"]

            oval_rect = QRectF(cx - rx, cy - ry, rx * 2, ry * 2)
            oval_hole = QPainterPath()
            oval_hole.addEllipse(oval_rect)

            mask_path = viewport_path.subtracted(oval_hole)

            if self.flash_color is not None:
                flash_brush = QColor(
                    self.flash_color.red(),
                    self.flash_color.green(),
                    self.flash_color.blue(),
                    210,
                )
                painter.fillPath(mask_path, flash_brush)
            else:
                painter.fillPath(mask_path, QColor(9, 13, 22, 175))

            # Viền phát sáng Oval (Glow)
            glow_pen = QPen(QColor(self.oval_color.red(), self.oval_color.green(), self.oval_color.blue(), 50), 8)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(oval_rect)

            stroke_pen = QPen(self.oval_color, 3.2, Qt.PenStyle.SolidLine)
            stroke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(stroke_pen)
            painter.drawEllipse(oval_rect)

            # 3. VÒNG TIẾN ĐỘ PHÁT QUANG (Progress Ring xung quanh Oval)
            if self.oval_progress > 0.0:
                prog_pen = QPen(QColor(16, 185, 129), 4.5, Qt.PenStyle.SolidLine)
                prog_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(prog_pen)
                # Vẽ cung tiến độ xoay từ đỉnh Oval (90 độ)
                start_angle = 90 * 16
                span_angle = int(-360 * 16 * self.oval_progress)
                painter.drawArc(oval_rect, start_angle, span_angle)

            # 4. Mũi tên quay đầu trực quan
            if self.turn_arrow in ("left", "right"):
                self._draw_turn_arrow(painter, dest_rect, cx, cy, rx, ry, self.turn_arrow)

        # 5. DẢI HƯỚNG DẪN HIỆN ĐẠI DƯỚI ĐÁY KHUNG HÌNH (Floating Capsule)
        if self.guidance_text:
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            text_rect = QRectF(dest_rect.x() + 20, dest_rect.bottom() - 56, dest_rect.width() - 40, 42)

            # Nền mờ kính mờ cao cấp
            painter.setPen(QPen(QColor(self.oval_color.red(), self.oval_color.green(), self.oval_color.blue(), 90), 1))
            painter.setBrush(QColor(15, 23, 42, 235))
            painter.drawRoundedRect(text_rect, 21, 21)

            painter.setPen(self.oval_color)
            display_str = f"{self.stage_icon}  {self.guidance_text}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, display_str)

        # 6. Huy hiệu trạng thái góc trên
        if self.badge_text:
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            badge_rect = QRectF(dest_rect.right() - 170, dest_rect.top() + 18, 150, 32)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 220))
            painter.drawRoundedRect(badge_rect, 16, 16)

            painter.setBrush(self.badge_color)
            painter.drawEllipse(int(badge_rect.x() + 14), int(badge_rect.y() + 11), 10, 10)

            painter.setPen(QColor(248, 250, 252))
            text_sub = QRectF(badge_rect.x() + 32, badge_rect.y(), badge_rect.width() - 36, badge_rect.height())
            painter.drawText(text_sub, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.badge_text)

        painter.restore()

        # Viền ngoài của viewport
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(dest_rect, 24, 24)

    def _draw_turn_arrow(
        self,
        painter: QPainter,
        dest_rect: QRectF,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        direction: str,
    ):
        """Vẽ thẻ mũi tên phát quang chuyển động lớn chỉ hướng quay đầu rõ ràng."""
        is_left = (direction == "left")

        arrow_x = cx - rx - 55 if is_left else cx + rx + 55
        arrow_y = cy

        box_w, box_h = 82, 54
        box_rect = QRectF(arrow_x - box_w / 2, arrow_y - box_h / 2, box_w, box_h)

        # Nền bóng phát sáng màu tím ánh lam công nghệ cao
        painter.setPen(QPen(QColor(167, 139, 250, 220), 2))
        painter.setBrush(QColor(15, 23, 42, 240))
        painter.drawRoundedRect(box_rect, 16, 16)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(167, 139, 250))

        arrow = QPolygonF()
        if is_left:
            tip_x = box_rect.left() + 16
            tip_y = arrow_y
            arrow.append(QPointF(tip_x, tip_y))
            arrow.append(QPointF(tip_x + 18, tip_y - 12))
            arrow.append(QPointF(tip_x + 18, tip_y - 5))
            arrow.append(QPointF(tip_x + 46, tip_y - 5))
            arrow.append(QPointF(tip_x + 46, tip_y + 5))
            arrow.append(QPointF(tip_x + 18, tip_y + 5))
            arrow.append(QPointF(tip_x + 18, tip_y + 12))
        else:
            tip_x = box_rect.right() - 16
            tip_y = arrow_y
            arrow.append(QPointF(tip_x, tip_y))
            arrow.append(QPointF(tip_x - 18, tip_y - 12))
            arrow.append(QPointF(tip_x - 18, tip_y - 5))
            arrow.append(QPointF(tip_x - 46, tip_y - 5))
            arrow.append(QPointF(tip_x - 46, tip_y + 5))
            arrow.append(QPointF(tip_x - 18, tip_y + 5))
            arrow.append(QPointF(tip_x - 18, tip_y + 12))

        painter.drawPolygon(arrow)
