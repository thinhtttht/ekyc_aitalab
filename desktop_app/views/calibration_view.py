"""Head Pose Monitoring & Angle Sensitivity Configuration View.
Thay thế bước hiệu chuẩn 9 điểm bằng giám sát trực quan Góc xoay đầu (Landmark 3D Head Pose).
Xác định chính xác hướng nhìn ra ngoài (Quay Trái / Quay Phải / Cúi Đầu / Ngửa Đầu) theo góc quay thời gian thực.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from eye_tracker import EyeTracker, GazeResult
    from sound_effects import play_chime_ding, play_chime_step, play_chime_warning
except ImportError:
    from desktop_app.eye_tracker import EyeTracker, GazeResult
    from desktop_app.sound_effects import play_chime_ding, play_chime_step, play_chime_warning


class HeadPoseAngleBar(QWidget):
    """Thanh đo góc xoay thời gian thực có hiển thị vùng an toàn và vùng vượt ngưỡng."""

    def __init__(self, title: str, min_val: float, max_val: float, safe_thresh: float, parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.safe_thresh = safe_thresh
        self.current_val = 0.0

    def set_value(self, val: float, safe_thresh: Optional[float] = None):
        self.current_val = val
        if safe_thresh is not None:
            self.safe_thresh = safe_thresh
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Nền thanh đo
        painter.fillRect(0, 0, w, h, QColor("#0f172a"))

        # Vẽ tiêu đề & giá trị số
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        is_safe = abs(self.current_val) <= self.safe_thresh
        text_color = QColor("#38bdf8") if is_safe else QColor("#ef4444")
        painter.setPen(text_color)
        painter.drawText(0, 14, f"{self.title}: {self.current_val:+.1f}° (Ngưỡng: ±{self.safe_thresh:.0f}°)")

        # Vẽ trục tọa độ
        bar_y = 24
        bar_h = 16
        painter.fillRect(0, bar_y, w, bar_h, QColor("#1e293b"))

        # Vùng an toàn ở giữa (màu xanh lá trong suốt)
        center_x = w / 2.0
        range_span = self.max_val - self.min_val
        px_per_deg = w / max(1.0, range_span)

        safe_w = (self.safe_thresh * 2) * px_per_deg
        safe_x0 = center_x - safe_w / 2.0
        painter.fillRect(int(safe_x0), bar_y, int(safe_w), bar_h, QColor(16, 185, 129, 60))

        # Vạch tâm 0 độ
        painter.setPen(QPen(QColor("#94a3b8"), 2))
        painter.drawLine(int(center_x), bar_y - 2, int(center_x), bar_y + bar_h + 2)

        # Con trỏ giá trị hiện tại
        clamped_val = max(self.min_val, min(self.max_val, self.current_val))
        cursor_x = center_x + clamped_val * px_per_deg
        cursor_color = QColor("#10b981") if is_safe else QColor("#ef4444")
        painter.setPen(QPen(cursor_color, 3))
        painter.drawLine(int(cursor_x), bar_y - 3, int(cursor_x), bar_y + bar_h + 3)
        painter.setBrush(cursor_color)
        painter.drawEllipse(int(cursor_x) - 4, bar_y + bar_h // 2 - 4, 8, 8)


class HeadPoseView(QWidget):
    """Giao diện kiểm tra và tinh chỉnh góc quay đầu Landmark 3D Head Pose."""

    launch_fullscreen_requested = pyqtSignal()

    def __init__(self, eye_tracker: Optional[EyeTracker] = None, parent=None):
        super().__init__(parent)
        self.eye_tracker = eye_tracker or EyeTracker()
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        # === CỘT TRÁI: CAMERA VIEW VẼ TIA HƯỚNG ĐẦU 3D ===
        left_box = QFrame()
        left_box.setStyleSheet("background-color: #0f172a; border-radius: 12px; border: 1px solid #1e293b; padding: 12px;")
        left_layout = QVBoxLayout(left_box)
        left_layout.setSpacing(10)

        v_title = QLabel("📷 CAMERA GIÁM SÁT HƯỚNG QUAY ĐẦU (REAL-TIME)")
        v_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #38bdf8;")
        left_layout.addWidget(v_title)

        self.lbl_camera = QLabel()
        self.lbl_camera.setMinimumSize(540, 420)
        self.lbl_camera.setStyleSheet("background-color: #020617; border: 2px solid #334155; border-radius: 8px;")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_camera, stretch=1)

        # Banner trạng thái kiểm tra trực tiếp
        self.lbl_test_badge = QLabel("🟢 CHÍNH DIỆN: TẬP TRUNG LÀM BÀI")
        self.lbl_test_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_test_badge.setStyleSheet("""
            background-color: #064e3b;
            color: #6ee7b7;
            font-size: 14px;
            font-weight: 800;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #059669;
        """)
        left_layout.addWidget(self.lbl_test_badge)

        layout.addWidget(left_box, stretch=3)

        # === CỘT PHẢI: BẢNG ĐO GÓC & TÙY CHỈNH NGƯỠNG NHẠY ===
        right_box = QFrame()
        right_box.setStyleSheet("background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 16px;")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(14)

        t_title = QLabel("🧭 ĐO LƯỜNG GÓC XOAY ĐẦU LANDMARK 3D")
        t_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #38bdf8;")
        t_sub = QLabel(
            "Xác định hướng nhìn ra ngoài trực tiếp từ góc quay của đầu (Yaw/Pitch/Roll).\n"
            "Không cần hiệu chuẩn mắt 9 điểm phức tạp - chính xác và ổn định tuyệt đối."
        )
        t_sub.setStyleSheet("font-size: 11px; color: #94a3b8; line-height: 1.5;")
        t_sub.setWordWrap(True)
        right_layout.addWidget(t_title)
        right_layout.addWidget(t_sub)

        # ĐỒNG HỒ ĐO GÓC THỜI GIAN THỰC
        gauges_frame = QFrame()
        gauges_frame.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        g_layout = QVBoxLayout(gauges_frame)
        g_layout.setSpacing(12)

        # 1. Yaw (Góc quay Trái / Phải)
        self.gauge_yaw = HeadPoseAngleBar("Góc quay ngang (Yaw: Trái / Phải)", -50.0, 50.0, self.eye_tracker.yaw_threshold)
        self.gauge_yaw.setFixedHeight(48)
        # 2. Pitch (Góc cúi / ngửa)
        self.gauge_pitch = HeadPoseAngleBar("Góc cúi / ngửa (Pitch)", -40.0, 40.0, self.eye_tracker.pitch_down_threshold)
        self.gauge_pitch.setFixedHeight(48)
        # 3. Roll (Góc nghiêng đầu)
        self.gauge_roll = HeadPoseAngleBar("Góc nghiêng đầu (Roll)", -40.0, 40.0, self.eye_tracker.roll_threshold)
        self.gauge_roll.setFixedHeight(48)

        g_layout.addWidget(self.gauge_yaw)
        g_layout.addWidget(self.gauge_pitch)
        g_layout.addWidget(self.gauge_roll)
        right_layout.addWidget(gauges_frame)

        # KHUNG ĐIỀU CHỈNH NGƯỠNG ĐỘ NHẠY
        thresh_frame = QFrame()
        thresh_frame.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        th_layout = QVBoxLayout(thresh_frame)
        th_layout.setSpacing(10)

        cfg_title = QLabel("⚙️ CÀI ĐẶT NGƯỠNG GÓC CẢNH BÁO")
        cfg_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38bdf8;")
        th_layout.addWidget(cfg_title)

        # Slider Yaw
        self.lbl_yaw_cfg = QLabel(f"• Ngưỡng quay đầu Trái/Phải: ±{self.eye_tracker.yaw_threshold:.0f}°")
        self.lbl_yaw_cfg.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        self.slider_yaw = QSlider(Qt.Orientation.Horizontal)
        self.slider_yaw.setRange(10, 30)
        self.slider_yaw.setValue(int(self.eye_tracker.yaw_threshold))
        self.slider_yaw.valueChanged.connect(self._on_yaw_threshold_changed)
        th_layout.addWidget(self.lbl_yaw_cfg)
        th_layout.addWidget(self.slider_yaw)

        # Slider Pitch
        self.lbl_pitch_cfg = QLabel(f"• Ngưỡng cúi đầu xuống (nhìn tài liệu/phone): -{self.eye_tracker.pitch_down_threshold:.0f}°")
        self.lbl_pitch_cfg.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        self.slider_pitch = QSlider(Qt.Orientation.Horizontal)
        self.slider_pitch.setRange(10, 25)
        self.slider_pitch.setValue(int(self.eye_tracker.pitch_down_threshold))
        self.slider_pitch.valueChanged.connect(self._on_pitch_threshold_changed)
        th_layout.addWidget(self.lbl_pitch_cfg)
        th_layout.addWidget(self.slider_pitch)

        right_layout.addWidget(thresh_frame)

        # Nút Reset bộ lọc
        btn_reset = QPushButton("🔄 Đặt lại bộ lọc nhìn thẳng (Center Pose)")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                font-weight: 700;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        btn_reset.clicked.connect(self._on_reset_clicked)
        right_layout.addWidget(btn_reset)

        right_layout.addStretch()
        layout.addWidget(right_box, stretch=2)

    def _on_yaw_threshold_changed(self, val: int):
        self.eye_tracker.yaw_threshold = float(val)
        self.lbl_yaw_cfg.setText(f"• Ngưỡng quay đầu Trái/Phải: ±{val}°")
        self.gauge_yaw.safe_thresh = float(val)
        self.gauge_yaw.update()

    def _on_pitch_threshold_changed(self, val: int):
        self.eye_tracker.pitch_down_threshold = float(val)
        self.lbl_pitch_cfg.setText(f"• Ngưỡng cúi đầu xuống (nhìn tài liệu/phone): -{val}°")
        self.gauge_pitch.safe_thresh = float(val)
        self.gauge_pitch.update()

    def _on_reset_clicked(self):
        self.eye_tracker.filter_yaw.reset()
        self.eye_tracker.filter_pitch.reset()
        self.eye_tracker.filter_roll.reset()
        play_chime_ding()

    def on_frame_received(self, frame: np.ndarray):
        """Xử lý frame video camera, vẽ tia hướng nhìn và cập nhật đồng hồ đo góc."""
        if not self.isVisible() or frame is None:
            return

        w = self.lbl_camera.width() or 540
        h = self.lbl_camera.height() or 420
        gaze_res = self.eye_tracker.process_frame(frame, screen_size=(w, h))

        # Cập nhật thanh đo góc
        self.gauge_yaw.set_value(gaze_res.yaw)
        self.gauge_pitch.set_value(gaze_res.pitch)
        self.gauge_roll.set_value(gaze_res.roll)

        # Cập nhật banner trạng thái
        if gaze_res.is_looking_away:
            self.lbl_test_badge.setText(f"🚨 CẢNH BÁO: {gaze_res.looking_away_reason.upper()}!")
            self.lbl_test_badge.setStyleSheet("""
                background-color: #7f1d1d;
                color: #fecaca;
                font-size: 14px;
                font-weight: 800;
                padding: 10px;
                border-radius: 8px;
                border: 2px solid #ef4444;
            """)
        else:
            self.lbl_test_badge.setText(f"🟢 CHÍNH DIỆN: TẬP TRUNG (Yaw: {gaze_res.yaw:+.1f}° | Pitch: {gaze_res.pitch:+.1f}°)")
            self.lbl_test_badge.setStyleSheet("""
                background-color: #064e3b;
                color: #6ee7b7;
                font-size: 14px;
                font-weight: 800;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #059669;
            """)

        # Vẽ frame kèm tia nhìn 3D
        disp = frame.copy()
        if gaze_res.nose_ray_2d:
            p1, p2 = gaze_res.nose_ray_2d
            color = (0, 0, 255) if gaze_res.is_looking_away else (0, 255, 0)
            cv2.arrowedLine(disp, p1, p2, color, 3, tipLength=0.35)
            cv2.circle(disp, p1, 5, color, -1)

        # Vẽ HUD thông tin góc độ
        hud_txt = f"Yaw: {gaze_res.yaw:+.1f} | Pitch: {gaze_res.pitch:+.1f} | {gaze_res.head_direction_label}"
        bg_color = (0, 0, 180) if gaze_res.is_looking_away else (0, 80, 0)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), bg_color, -1)
        cv2.putText(disp, hud_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        disp_rgb = np.ascontiguousarray(disp_rgb)
        dh, dw, dch = disp_rgb.shape
        qimg = QImage(disp_rgb.data, dw, dh, dch * dw, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_camera.setPixmap(pix)


# Alias tương thích ngược để MainWindow không bị vỡ import
CalibrationView = HeadPoseView
FullScreenCalibrationWindow = QWidget  # Stub rỗng an toàn
