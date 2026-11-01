"""Enrollment 3D View - Quy trình đăng ký sinh trắc học 3D RealSense.
Kiểm tra liên tục độ sâu thực tế (3D Depth Liveness & Anti-Spoofing) trong suốt toàn bộ quá trình.
Chống giả mạo tuyệt đối trước ảnh in phẳng, màn hình điện thoại, máy tính bảng và mặt nạ 2D.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Backend path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import config as C
from database import get_user_repository
from depth_liveness import DepthLivenessDetector, DepthLivenessResult
from face_analyzer import FaceAnalyzer, evaluate_face
from feature_extractor import get_arcface_extractor
from sound_effects import (
    play_chime_ding,
    play_chime_step,
    play_chime_success,
    play_chime_tick,
    play_chime_warning,
)
from widgets.metrics_card import MetricsCard
from widgets.video_widget import VideoWidget

# Bảng màu quang phổ 100% bão hòa cho bước chớp màu quang học
SPECTRAL_COLOR_PALETTE = [
    (QColor(255, 0, 40), (255, 0, 40), "ĐỎ ĐẬM"),
    (QColor(0, 245, 75), (0, 245, 75), "XANH LỤC ĐẬM"),
    (QColor(0, 80, 255), (0, 80, 255), "XANH LAM ĐẬM"),
    (QColor(255, 200, 0), (255, 200, 0), "VÀNG ĐẬM"),
]


class Enrollment3DView(QWidget):
    """Giao diện đăng ký sinh trắc học 3D với RealSense Depth Liveness liên tục."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_repo = get_user_repository()
        self.extractor = get_arcface_extractor()
        self.analyzer = FaceAnalyzer(stream=True)
        self.depth_detector = DepthLivenessDetector()

        # Trạng thái quy trình:
        # 0: Chờ (Idle)
        # 1: Căn mặt vào Oval & Kiểm tra chiều sâu 3D (Frontal & 3D Depth)
        # 2: Xoay đầu sang Trái (Turn Left)
        # 3: Xoay đầu sang Phải (Turn Right)
        # 4: Cự ly gần (Zoom In)
        # 5: Chớp quang phổ & Chụp chân dung 3D (Active Flashing & Capture)
        # 6: Hoàn tất - Chờ lưu hồ sơ
        self.current_step = 0
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.baseline_face_h: Optional[float] = None
        self.hold_start_time: Optional[float] = None

        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.latest_depth_cmap: Optional[np.ndarray] = None

        self.captured_frame: Optional[np.ndarray] = None
        self.captured_depth_map: Optional[np.ndarray] = None
        self.captured_embedding: Optional[np.ndarray] = None
        self.captured_face_res = None
        self.captured_liveness_res: Optional[DepthLivenessResult] = None
        self.last_fps = 30.0

        # Cache kết quả suy luận AI để chống giật lag và duy trì 30 FPS mượt mà
        self._cached_face_res = None
        self._cached_liveness_res: Optional[DepthLivenessResult] = None
        self._last_ai_time = 0.0
        self._frame_count = 0

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # === CỘT TRÁI: CAMERA VIEW CHÍNH & DEPTH MINI-PREVIEW ===
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)

        # Video chính (RGB Color kèm khung hướng dẫn Oval sinh trắc học)
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(600, 480)
        left_layout.addWidget(self.video_widget, stretch=4)

        # Khung phụ: Bản đồ chiều sâu 3D Depth Map trực quan
        depth_preview_box = QFrame()
        depth_preview_box.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 6px;"
        )
        depth_layout = QHBoxLayout(depth_preview_box)
        depth_layout.setContentsMargins(8, 6, 8, 6)

        d_title = QLabel("🌐 Bản đồ 3D Depth Map:")
        d_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 12px;")
        depth_layout.addWidget(d_title)

        self.lbl_depth_preview = QLabel()
        self.lbl_depth_preview.setFixedSize(180, 110)
        self.lbl_depth_preview.setStyleSheet("background-color: #000; border-radius: 4px;")
        self.lbl_depth_preview.setScaledContents(True)
        depth_layout.addWidget(self.lbl_depth_preview)

        # Hộp cảnh báo độ sâu 3D trực quan
        self.lbl_spoof_alert = QLabel("Chờ tín hiệu camera 3D...")
        self.lbl_spoof_alert.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8; font-size: 13px; font-weight: 600; padding: 10px; border-radius: 6px;"
        )
        self.lbl_spoof_alert.setWordWrap(True)
        depth_layout.addWidget(self.lbl_spoof_alert, stretch=1)

        left_layout.addWidget(depth_preview_box, stretch=1)
        layout.addLayout(left_layout, stretch=3)

        # === CỘT PHẢI: BẢNG ĐIỀU HƯỚNG BẢO MẬT & TELEMETRY 3D ===
        right_panel = QFrame()
        right_panel.setStyleSheet(
            "background-color: #1e293b; border-radius: 12px; border: 1px solid #334155;"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(12)

        # Tiêu đề
        h_title = QLabel("🌐 ĐĂNG KÝ 3D REALSENSE")
        h_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;")
        h_sub = QLabel("Xác thực người thật liên tục qua cảm biến độ sâu hồng ngoại")
        h_sub.setStyleSheet("font-size: 11px; color: #94a3b8; margin-bottom: 4px;")
        right_layout.addWidget(h_title)
        right_layout.addWidget(h_sub)

        # Thanh tiến trình 5 bước
        step_box = QFrame()
        step_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        step_layout = QVBoxLayout(step_box)
        step_layout.setSpacing(6)

        self.lbl_step_title = QLabel("Bước 0: Sẵn sàng bắt đầu")
        self.lbl_step_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 5)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #334155; height: 10px; border-radius: 5px; }"
            "QProgressBar::chunk { background-color: #38bdf8; border-radius: 5px; }"
        )
        self.lbl_step_desc = QLabel("Bấm 'Bắt đầu đăng ký 3D' để khởi động quy trình.")
        self.lbl_step_desc.setStyleSheet("font-size: 12px; color: #cbd5e1;")
        self.lbl_step_desc.setWordWrap(True)

        step_layout.addWidget(self.lbl_step_title)
        step_layout.addWidget(self.progress_bar)
        step_layout.addWidget(self.lbl_step_desc)
        right_layout.addWidget(step_box)

        # BẢNG THÔNG SỐ ĐỘ SÂU 3D THỜI GIAN THỰC (REAL-TIME TELEMETRY)
        metrics_box = QFrame()
        metrics_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        metrics_grid = QGridLayout(metrics_box)
        metrics_grid.setSpacing(8)

        # Card 1: Khoảng cách cự ly
        self.card_dist = MetricsCard("Khoảng cách (Z)", "-- cm", "Chuẩn: 45-75 cm")
        # Card 2: Độ nhô sống mũi
        self.card_relief = MetricsCard("Độ lồi sống mũi", "-- mm", "Mặt thật: >10 mm")
        # Card 3: Độ cong phi phẳng
        self.card_curvature = MetricsCard("Độ cong 3D", "-- mm", "Mặt phẳng 2D: <4 mm")
        # Card 4: Điểm 3D Liveness
        self.card_liveness = MetricsCard("Điểm người thật 3D", "-- %", "Ngưỡng: >= 60%")

        metrics_grid.addWidget(self.card_dist, 0, 0)
        metrics_grid.addWidget(self.card_relief, 0, 1)
        metrics_grid.addWidget(self.card_curvature, 1, 0)
        metrics_grid.addWidget(self.card_liveness, 1, 1)
        right_layout.addWidget(metrics_box)

        # THÔNG TIN THÍ SINH CẦN LƯU
        form_frame = QFrame()
        form_frame.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setSpacing(6)

        lbl_uid = QLabel("Mã thí sinh / Số báo danh:")
        lbl_uid.setStyleSheet("font-size: 11px; font-weight: 600; color: #94a3b8;")
        self.txt_user_id = QLineEdit()
        self.txt_user_id.setPlaceholderText("VD: TS2026_01")
        self.txt_user_id.setStyleSheet("background-color: #1e293b; color: white; padding: 6px; border-radius: 4px;")

        lbl_name = QLabel("Họ và tên thí sinh:")
        lbl_name.setStyleSheet("font-size: 11px; font-weight: 600; color: #94a3b8;")
        self.txt_full_name = QLineEdit()
        self.txt_full_name.setPlaceholderText("VD: Nguyễn Văn A")
        self.txt_full_name.setStyleSheet("background-color: #1e293b; color: white; padding: 6px; border-radius: 4px;")

        form_layout.addWidget(lbl_uid)
        form_layout.addWidget(self.txt_user_id)
        form_layout.addWidget(lbl_name)
        form_layout.addWidget(self.txt_full_name)
        right_layout.addWidget(form_frame)

        # CÁC NÚT THAO TÁC
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.btn_start = QPushButton("▶️  Bắt đầu đăng ký 3D")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: 700; padding: 10px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_start.clicked.connect(self._on_start_clicked)

        self.btn_reset = QPushButton("🔄  Làm lại")
        self.btn_reset.setStyleSheet(
            "QPushButton { background-color: #475569; color: white; font-weight: 600; padding: 10px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #334155; }"
        )
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_reset)
        right_layout.addLayout(btn_box)

        self.btn_save = QPushButton("💾  Lưu hồ sơ sinh trắc học 3D")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; font-weight: 800; padding: 12px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #059669; }"
            "QPushButton:disabled { background-color: #334155; color: #64748b; }"
        )
        self.btn_save.clicked.connect(self._on_save_clicked)
        right_layout.addWidget(self.btn_save)

        right_layout.addStretch()
        layout.addWidget(right_panel, stretch=2)

    def on_frames_received(
        self,
        color_bgr: np.ndarray,
        raw_depth_mm: np.ndarray,
        depth_colormap: np.ndarray,
    ):
        """Xử lý đồng thời luồng RGB và Depth từ RealSense."""
        self.latest_color = color_bgr
        self.latest_depth_raw = raw_depth_mm
        self.latest_depth_cmap = depth_colormap
        self._frame_count += 1

        # Cập nhật khung hình mini preview Depth Map (cách frame và dùng nearest neighbor để tối ưu CPU)
        if depth_colormap is not None and self._frame_count % 2 == 0:
            small_d = cv2.resize(depth_colormap, (180, 110), interpolation=cv2.INTER_NEAREST)
            rgb_d = cv2.cvtColor(small_d, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_d.data, 180, 110, 180 * 3, QImage.Format.Format_RGB888)
            self.lbl_depth_preview.setPixmap(QPixmap.fromImage(q_img))

        if color_bgr is None:
            return

        frame = color_bgr.copy()

        now_t = time.time()
        # Chạy AI inference (MediaPipe + Depth Liveness) theo chu kỳ ~80ms (~12Hz) để video luôn mượt 30 FPS
        if now_t - self._last_ai_time >= 0.08 or self._cached_face_res is None:
            self._last_ai_time = now_t
            oval_cfg = self.video_widget.get_oval_cfg()
            self._cached_face_res = self.analyzer.analyze(frame, oval=oval_cfg)

            # PHÂN TÍCH CHIỀU SÂU SINH TRẮC HỌC 3D (CONTINUOUS 3D DEPTH LIVENESS)
            if self._cached_face_res.face_count == 1 and raw_depth_mm is not None:
                self._cached_liveness_res = self.depth_detector.analyze(raw_depth_mm, self._cached_face_res.bbox)
                self._update_telemetry_ui(self._cached_liveness_res)
            else:
                self._cached_liveness_res = None

            # Chạy máy trạng thái đăng ký 3D
            self._process_registration_flow(frame, self._cached_face_res, self._cached_liveness_res)
        else:
            # Giữa các nhịp AI, tiếp tục render video mượt mà 30 FPS với kết quả nhận diện đã cache
            if self.current_step == 6 and self.captured_frame is not None:
                self.video_widget.set_frame(self.captured_frame, face_result=self.captured_face_res, show_oval=False)
            else:
                self.video_widget.set_frame(frame, face_result=self._cached_face_res, show_oval=(self.current_step <= 5))

    def _update_telemetry_ui(self, res: DepthLivenessResult):
        """Cập nhật các chỉ số đo đạc chiều sâu lên giao diện."""
        # 1. Khoảng cách
        self.card_dist.set_value(f"{res.distance_cm:.1f} cm")
        if 45.0 <= res.distance_cm <= 75.0:
            self.card_dist.set_status("Tốt (Trong vùng eKYC)", True)
        else:
            self.card_dist.set_status("Cần điều chỉnh", False)

        # 2. Độ nhô sống mũi
        self.card_relief.set_value(f"+{res.nose_relief_mm:.1f} mm")
        self.card_relief.set_status("3D nổi rõ" if res.nose_relief_mm >= 10.0 else "Quá phẳng (2D)", res.nose_relief_mm >= 10.0)

        # 3. Độ cong phi phẳng
        self.card_curvature.set_value(f"{res.curvature_residual_mm:.1f} mm")
        self.card_curvature.set_status("Topography OK" if res.curvature_residual_mm >= 6.0 else "Mặt phẳng 2D", res.curvature_residual_mm >= 6.0)

        # 4. Liveness score
        score_pct = int(res.liveness_score * 100)
        self.card_liveness.set_value(f"{score_pct} %")
        self.card_liveness.set_status("NGƯỜI THẬT 3D" if res.is_real_3d else "NGHI VẤN GIẢ MẠO", res.is_real_3d)

        # Banner cảnh báo
        if res.is_real_3d:
            self.lbl_spoof_alert.setText(f"✅ XÁC NHẬN NGƯỜI THẬT 3D\nCự ly: {res.distance_cm:.1f}cm | Độ lồi mũi: +{res.nose_relief_mm:.1f}mm")
            self.lbl_spoof_alert.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-size: 12px; font-weight: 700; padding: 8px; border-radius: 6px; border: 1px solid #059669;"
            )
        else:
            self.lbl_spoof_alert.setText(f"🚨 {res.message.upper()}")
            self.lbl_spoof_alert.setStyleSheet(
                "background-color: #7f1d1d; color: #f87171; font-size: 12px; font-weight: 700; padding: 8px; border-radius: 6px; border: 1px solid #dc2626;"
            )

    def _process_registration_flow(
        self,
        frame: np.ndarray,
        face_res,
        liveness_res: Optional[DepthLivenessResult],
    ):
        """Xử lý từng bước đăng ký có kiểm tra chặt chẽ độ sâu 3D."""
        if self.current_step == 0:
            # Idle
            self.video_widget.set_frame(frame, face_result=face_res, show_oval=True)
            return

        if self.current_step == 6:
            # Đã chụp xong, chờ lưu
            self.video_widget.set_frame(self.captured_frame, face_result=self.captured_face_res, show_oval=False)
            return

        # KIỂM TRA CHẶN GIẢ MẠO 3D TRONG SUỐT QUÁ TRÌNH
        if liveness_res is not None and not liveness_res.is_real_3d:
            # Bị phát hiện ảnh phẳng 2D hoặc màn hình điện thoại
            self.video_widget.set_frame(frame, face_result=face_res, show_oval=True)
            self.lbl_step_desc.setText(f"❌ {liveness_res.message}!\nKhông thể dùng ảnh hoặc màn hình phẳng để đăng ký!")
            self.fqa_consecutive_frames = 0
            return

        # BƯỚC 1: Căn mặt vào Oval & Kiểm tra chiều sâu chuẩn
        if self.current_step == 1:
            eval_res = evaluate_face(face_res)
            # Yêu cầu nhìn thẳng, lọt oval, và xác nhận 3D người thật
            if eval_res.is_valid and liveness_res and liveness_res.is_real_3d:
                self.fqa_consecutive_frames += 1
                self.lbl_step_desc.setText(f"Giữ yên khuôn mặt... ({self.fqa_consecutive_frames}/12)")
                if self.fqa_consecutive_frames >= 12:
                    self.baseline_face_h = face_res.face_h
                    self._advance_to_step(2)
            else:
                self.fqa_consecutive_frames = max(0, self.fqa_consecutive_frames - 1)
                self.lbl_step_desc.setText(eval_res.message or "Căn mặt vào giữa hình Oval và nhìn thẳng.")

        # BƯỚC 2: Xoay đầu nhẹ sang TRÁI (Yaw < -12)
        elif self.current_step == 2:
            yaw = face_res.yaw or 0.0
            if yaw < -10.0:
                self.turn_hold_counter += 1
                self.lbl_step_desc.setText(f"Tốt! Giữ nguyên sang trái... ({self.turn_hold_counter}/8)")
                if self.turn_hold_counter >= 8:
                    self._advance_to_step(3)
            else:
                self.turn_hold_counter = 0
                self.lbl_step_desc.setText("Vui lòng quay đầu nhẹ sang TRÁI (hướng mặt về bên trái).")

        # BƯỚC 3: Xoay đầu nhẹ sang PHẢI (Yaw > +12)
        elif self.current_step == 3:
            yaw = face_res.yaw or 0.0
            if yaw > 10.0:
                self.turn_hold_counter += 1
                self.lbl_step_desc.setText(f"Tốt! Giữ nguyên sang phải... ({self.turn_hold_counter}/8)")
                if self.turn_hold_counter >= 8:
                    self._advance_to_step(4)
            else:
                self.turn_hold_counter = 0
                self.lbl_step_desc.setText("Vui lòng quay đầu nhẹ sang PHẢI (hướng mặt về bên phải).")

        # BƯỚC 4: Tiến gần camera (Zoom-in)
        elif self.current_step == 4:
            base_h = self.baseline_face_h or 0.35
            curr_h = face_res.face_h
            if curr_h >= base_h * 1.15:
                self.turn_hold_counter += 1
                self.lbl_step_desc.setText(f"Cự ly gần đạt chuẩn... ({self.turn_hold_counter}/8)")
                if self.turn_hold_counter >= 8:
                    self._advance_to_step(5)
            else:
                self.turn_hold_counter = 0
                self.lbl_step_desc.setText("Vui lòng nghiêng hoặc ghé người lại gần camera hơn một chút.")

        # BƯỚC 5: Căn đầu vào Oval & Chụp trích xuất ArcFace 512D trực tiếp (Người thật 3D + Đầu trong Oval)
        elif self.current_step == 5:
            is_real = bool(liveness_res is not None and liveness_res.is_real_3d)
            inside_oval = bool(
                face_res.detected
                and face_res.face_count == 1
                and not face_res.is_upside_down
                and face_res.oval_dist <= C.OVAL_INSIDE_TOL
                and abs(face_res.yaw or 0.0) <= 12.0
                and abs(face_res.pitch or 0.0) <= 15.0
            )

            if is_real and inside_oval:
                self.turn_hold_counter += 1
                self.lbl_step_desc.setText(f"✅ Người thật 3D & Đầu trong Oval! Giữ yên để chụp trích xuất... ({self.turn_hold_counter}/6)")
                if self.turn_hold_counter >= 6:
                    self._finalize_3d_capture()
                    return
            else:
                self.turn_hold_counter = max(0, self.turn_hold_counter - 1)
                if not is_real:
                    msg = liveness_res.message if liveness_res else "Chưa phát hiện chiều sâu 3D người thật"
                    self.lbl_step_desc.setText(f"⚠️ {msg}! Vui lòng ngồi đối diện camera RealSense.")
                elif not inside_oval:
                    if face_res.oval_dist > C.OVAL_INSIDE_TOL:
                        self.lbl_step_desc.setText("Vui lòng căn đầu và khuôn mặt lọt gọn vào giữa khung Oval.")
                    elif abs(face_res.yaw or 0.0) > 12.0:
                        self.lbl_step_desc.setText("Vui lòng nhìn thẳng vào camera, không quay đầu.")
                    else:
                        eval_res = evaluate_face(face_res)
                        self.lbl_step_desc.setText(eval_res.message or "Căn đầu vào khung Oval để chụp.")

        self.video_widget.set_frame(frame, face_result=face_res, show_oval=(self.current_step <= 5))

    def _advance_to_step(self, step: int):
        self.current_step = step
        self.progress_bar.setValue(step)
        play_chime_step()

        if step == 2:
            self.lbl_step_title.setText("Bước 2/5: Quay đầu sang TRÁI")
            self.lbl_step_desc.setText("Nghiêng đầu nhẹ sang TRÁI để kiểm tra chuyển động 3D.")
            self.turn_hold_counter = 0
        elif step == 3:
            self.lbl_step_title.setText("Bước 3/5: Quay đầu sang PHẢI")
            self.lbl_step_desc.setText("Nghiêng đầu nhẹ sang PHẢI để kiểm tra chuyển động 3D.")
            self.turn_hold_counter = 0
        elif step == 4:
            self.lbl_step_title.setText("Bước 4/5: Tiến lại gần (Zoom In)")
            self.lbl_step_desc.setText("Ghé khuôn mặt lại gần camera thêm khoảng 10-15cm.")
            self.turn_hold_counter = 0
        elif step == 5:
            self.lbl_step_title.setText("Bước 5/5: Căn mặt vào Oval & Chụp ArcFace 3D")
            self.lbl_step_desc.setText("Nhìn thẳng và căn đầu lọt gọn vào khung Oval để hệ thống tự động chụp trích xuất.")
            self.turn_hold_counter = 0

    def _finalize_3d_capture(self):
        """Chụp frame chuẩn khi đầu trong Oval, trích xuất ArcFace 512D và xác nhận 3D."""
        if self.latest_color is None:
            QMessageBox.warning(self, "Lỗi", "Không nhận được khung hình từ camera.")
            self._on_reset_clicked()
            return

        frame = self.latest_color.copy()
        oval_cfg = self.video_widget.get_oval_cfg()
        face_res = self.analyzer.analyze(frame, oval=oval_cfg, fresh=True)

        if face_res.face_count != 1:
            QMessageBox.warning(self, "Chụp thất bại", "Không tìm thấy khuôn mặt rõ ràng lúc chụp. Vui lòng thử lại.")
            self._on_reset_clicked()
            return

        # Bắt buộc đầu phải nằm trong khung Oval lúc chụp trích xuất
        if face_res.oval_dist > C.OVAL_INSIDE_TOL:
            self.lbl_step_desc.setText("⚠️ Đầu chưa lọt gọn trong khung Oval. Vui lòng căn chỉnh lại.")
            self.turn_hold_counter = 0
            return

        # Kiểm tra lần cuối với Depth
        liveness_res = None
        if self.latest_depth_raw is not None:
            liveness_res = self.depth_detector.analyze(self.latest_depth_raw, face_res.bbox)

        if liveness_res and not liveness_res.is_real_3d:
            QMessageBox.critical(self, "Từ chối hồ sơ", f"Cảnh báo giả mạo 3D: {liveness_res.message}!")
            self._on_reset_clicked()
            return

        # Trích xuất ArcFace 512D
        emb = self.extractor.extract_embedding(frame, landmarks_5pts=face_res.arcface_kps, bbox=face_res.bbox)
        if emb is None:
            QMessageBox.warning(self, "Lỗi trích xuất", "Không thể trích xuất đặc trưng khuôn mặt ArcFace.")
            self._on_reset_clicked()
            return

        self.captured_frame = frame
        self.captured_face_res = face_res
        self.captured_depth_map = self.latest_depth_raw.copy() if self.latest_depth_raw is not None else None
        self.captured_embedding = emb
        self.captured_liveness_res = liveness_res

        self.current_step = 6
        self.progress_bar.setValue(5)
        self.lbl_step_title.setText("✅ ĐÃ CHỤP THÀNH CÔNG!")
        self.lbl_step_desc.setText("Người thật 3D + ArcFace 512D đã được trích xuất thành công! Vui lòng nhập MSSV, Họ tên và bấm 'Lưu hồ sơ sinh trắc học 3D'.")
        self.btn_save.setEnabled(True)
        play_chime_success()

    def _on_start_clicked(self):
        """Bắt đầu phiên đăng ký 3D."""
        self.current_step = 1
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.progress_bar.setValue(1)
        self.lbl_step_title.setText("Bước 1/5: Căn mặt & Đo chiều sâu 3D")
        self.lbl_step_desc.setText("Nhìn thẳng vào giữa hình Oval và ngồi cách camera khoảng 50-70cm.")
        self.btn_save.setEnabled(False)
        play_chime_ding()

    def _on_reset_clicked(self):
        """Khôi phục về trạng thái ban đầu."""
        self.current_step = 0
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.progress_bar.setValue(0)
        self.lbl_step_title.setText("Bước 0: Sẵn sàng bắt đầu")
        self.lbl_step_desc.setText("Bấm 'Bắt đầu đăng ký 3D' để khởi động quy trình.")
        self.btn_save.setEnabled(False)
        self.captured_frame = None
        self.captured_embedding = None

    def _on_save_clicked(self):
        """Lưu hồ sơ vào cơ sở dữ liệu SQLite kèm metadata 3D."""
        uid = self.txt_user_id.text().strip()
        name = self.txt_full_name.text().strip()

        if not uid or not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ Mã thí sinh và Họ tên.")
            return

        if self.captured_frame is None or self.captured_embedding is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có ảnh chụp chân dung hợp lệ.")
            return

        # Mã hoá ảnh sang base64
        _, buf = cv2.imencode(".jpg", self.captured_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        b64_snap = base64.b64encode(buf).decode("utf-8")

        meta = {
            "branch": "3d_realsense",
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.captured_liveness_res:
            meta["depth_distance_cm"] = self.captured_liveness_res.distance_cm
            meta["nose_relief_mm"] = self.captured_liveness_res.nose_relief_mm
            meta["curvature_rms_mm"] = self.captured_liveness_res.curvature_residual_mm
            meta["liveness_score"] = self.captured_liveness_res.liveness_score

        try:
            self.user_repo.create_user(
                user_id=uid,
                full_name=name,
                embedding=self.captured_embedding,
                snapshot_b64=b64_snap,
                metadata_dict=meta,
            )
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã lưu thành công hồ sơ sinh trắc học 3D cho thí sinh: {name} ({uid})!",
            )
            self.txt_user_id.clear()
            self.txt_full_name.clear()
            self._on_reset_clicked()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi lưu dữ liệu", f"Không thể lưu vào cơ sở dữ liệu: {e}")
