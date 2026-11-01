"""Enrollment View - Interactive 5-stage eKYC flow with continuous anti-spoofing tracking,
per-stage timeouts, and escalating lockout penalties (1st: reset, 2nd: 1-min, 3rd: 10-min, 4th+: permanent).
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
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
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

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import config as C
from database import get_user_repository
from face_analyzer import FaceAnalyzer, evaluate_face
from feature_extractor import get_arcface_extractor
from optical_analyzer import OpticalAnalyzer
from security_guard import SecurityGuard
from sound_effects import play_chime_ding, play_chime_step, play_chime_success, play_chime_tick, play_chime_warning
from widgets.metrics_card import MetricsCard
from widgets.video_widget import VideoWidget

# Bảng màu quang phổ 100% bão hòa (Pure Spectral Wavelengths - Đậm đặc, không pha loãng)
SPECTRAL_COLOR_PALETTE = [
    (QColor(255, 0, 40), (255, 0, 40), "ĐỎ ĐẬM"),          # Pure Red (~630nm, hấp thụ hemoglobin)
    (QColor(0, 245, 75), (0, 245, 75), "XANH LỤC ĐẬM"),    # Pure Emerald Green (~525nm, phản xạ mao mạch)
    (QColor(0, 80, 255), (0, 80, 255), "XANH LAM ĐẬM"),    # Pure Royal Blue (~460nm, tán xạ lớp sừng)
    (QColor(255, 200, 0), (255, 200, 0), "VÀNG ĐẬM"),      # Pure Amber Yellow (~585nm, tương phản sắc độ)
    (QColor(220, 0, 255), (220, 0, 255), "TÍM ĐẬM"),       # Pure Magenta (~500/650nm, phân cực màn hình)
    (QColor(0, 230, 255), (0, 230, 255), "XANH LƠ ĐẬM"),   # Pure Cyan (~490nm, phổ kép lục-lam)
]


class EnrollmentView(QWidget):
    """Giao diện đăng ký sinh trắc học 5 bước chuẩn với cơ chế chống giả mạo liên tục và xử phạt leo thang."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_repo = get_user_repository()
        self.extractor = get_arcface_extractor()
        self.analyzer = FaceAnalyzer(stream=True)
        self.optical_analyzer = OpticalAnalyzer()
        self.security_guard = SecurityGuard()

        # Trạng thái quy trình:
        # 0: Sẵn sàng (Idle)
        # 1: Căn mặt vào Oval chuẩn (FQA Frontal)
        # 2: Quay đầu sang TRÁI (Turn Left)
        # 3: Quay đầu sang PHẢI (Turn Right)
        # 4: Đưa mặt lại gần camera (Zoom-in)
        # 5: Quét phản xạ quang học đa màu (Active Optical Color Flashing)
        # 6: Đã chụp thành công - Chờ lưu hồ sơ
        self.current_step = 0
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.baseline_face_h: Optional[float] = None
        self.hold_start_time: Optional[float] = None

        # Dữ liệu Color Flashing
        self.flash_colors = [
            (QColor(239, 68, 68), (239, 68, 68), "ĐỎ"),
            (QColor(16, 185, 129), (16, 185, 129), "XANH LÁ"),
            (QColor(59, 130, 246), (59, 130, 246), "XANH DƯƠNG"),
        ]
        self.flash_step = 0
        self.flash_frames: List[np.ndarray] = []
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._on_flash_timer_tick)

        self.latest_frame: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self.captured_embedding: Optional[np.ndarray] = None
        self.captured_face_res = None
        self.last_fps = 30.0

        # Trạng thái đệm chống nhiễu Color Flashing & chụp chân dung mượt mà
        self.flash_violation_frames = 0
        self.awaiting_final_capture = False
        self.final_capture_cooldown = 0.0
        self.final_capture_deadline = 0.0

        self._init_ui()
        self._check_lockout_state()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(20)

        # --- CỘT TRÁI: CAMERA VIEWPORT ---
        left_box = QVBoxLayout()
        left_box.setSpacing(12)

        self.video_widget = VideoWidget()
        left_box.addWidget(self.video_widget, stretch=1)

        # Thanh tiến trình tổng thể
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_box.addWidget(self.progress_bar)

        main_layout.addLayout(left_box, stretch=3)

        # --- CỘT PHẢI: BẢNG ĐIỀU KHIỂN & CÁC BƯỚC ---
        right_box = QVBoxLayout()
        right_box.setSpacing(12)

        # 1. BẢNG CẢNH BÁO KHÓA AN NINH (LOCKOUT BANNER)
        self.lockout_banner = QFrame()
        self.lockout_banner.setStyleSheet(
            "background-color: rgba(239, 68, 68, 0.18); border: 2px solid #ef4444; border-radius: 10px; padding: 12px;"
        )
        lockout_layout = QVBoxLayout(self.lockout_banner)
        lockout_layout.setSpacing(6)

        self.lbl_lockout_title = QLabel("🚫 TẠM KHÓA AN NINH THIẾT BỊ")
        self.lbl_lockout_title.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: 800;")
        lockout_layout.addWidget(self.lbl_lockout_title)

        self.lbl_lockout_timer = QLabel("Thời gian mở khóa: 00:00")
        self.lbl_lockout_timer.setStyleSheet("color: #f8fafc; font-size: 12px; font-weight: 600;")
        lockout_layout.addWidget(self.lbl_lockout_timer)

        btn_unlock_row = QHBoxLayout()
        btn_unlock_row.addStretch()
        self.btn_admin_unlock = QPushButton("🔑 Admin Mở khóa")
        self.btn_admin_unlock.setProperty("class", "btn-secondary")
        self.btn_admin_unlock.setStyleSheet(
            "background-color: #ef4444; color: #ffffff; border: none; font-weight: 700; padding: 6px 14px; border-radius: 6px;"
        )
        self.btn_admin_unlock.clicked.connect(self.prompt_admin_unlock)
        btn_unlock_row.addWidget(self.btn_admin_unlock)
        lockout_layout.addLayout(btn_unlock_row)

        self.lockout_banner.setVisible(False)
        right_box.addWidget(self.lockout_banner)

        # 2. Card các bước quy trình
        steps_card = QFrame()
        steps_card.setProperty("class", "card")
        steps_layout = QVBoxLayout(steps_card)
        steps_layout.setSpacing(8)

        steps_header = QHBoxLayout()
        title = QLabel("QUY TRÌNH ĐĂNG KÝ SINH TRẮC HỌC")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;")
        steps_header.addWidget(title)

        self.lbl_stage_timer = QLabel("")
        self.lbl_stage_timer.setStyleSheet(
            "color: #fbbf24; font-size: 12px; font-weight: 700; background: rgba(251, 191, 36, 0.12); padding: 2px 8px; border-radius: 10px;"
        )
        self.lbl_stage_timer.setVisible(False)
        steps_header.addWidget(self.lbl_stage_timer, alignment=Qt.AlignmentFlag.AlignRight)

        steps_layout.addLayout(steps_header)

        self.step_labels = [
            QLabel("1. Nhìn thẳng vào khung Oval"),
            QLabel("2. ⬅️ Quay đầu sang TRÁI của bạn"),
            QLabel("3. Quay đầu sang PHẢI của bạn ➡️"),
            QLabel("4. 🔍 Đưa mặt lại gần camera"),
            QLabel("5. 🌈 Quét quang học toàn màn hình (Color Flashing)"),
            QLabel("6. Nhập thông tin & Lưu hồ sơ"),
        ]
        for lbl in self.step_labels:
            lbl.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: 500; padding: 3px 0px;")
            steps_layout.addWidget(lbl)

        right_box.addWidget(steps_card)

        # 3. Instruction Alert Box
        self.alert_box = QFrame()
        self.alert_box.setStyleSheet(
            "background-color: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px;"
        )
        alert_layout = QVBoxLayout(self.alert_box)
        self.alert_text = QLabel("Bấm nút 'Bắt đầu đăng ký' để tiến hành thu thập khuôn mặt.")
        self.alert_text.setWordWrap(True)
        self.alert_text.setStyleSheet("color: #38bdf8; font-size: 13px; font-weight: 600;")
        alert_layout.addWidget(self.alert_text)
        right_box.addWidget(self.alert_box)

        # 4. Student Information Input Card
        info_card = QFrame()
        info_card.setProperty("class", "card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setSpacing(10)

        info_title = QLabel("THÔNG TIN THÍ SINH")
        info_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #f8fafc;")
        info_layout.addWidget(info_title)

        self.input_user_id = QLineEdit()
        self.input_user_id.setPlaceholderText("Mã sinh viên (MSSV) *")
        info_layout.addWidget(self.input_user_id)

        self.input_full_name = QLineEdit()
        self.input_full_name.setPlaceholderText("Họ và tên thí sinh *")
        info_layout.addWidget(self.input_full_name)

        self.input_notes = QLineEdit()
        self.input_notes.setPlaceholderText("Lớp / Phòng thi (Tuỳ chọn)")
        info_layout.addWidget(self.input_notes)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_start = QPushButton("Bắt đầu đăng ký")
        self.btn_start.setProperty("class", "btn-primary")
        self.btn_start.clicked.connect(self.start_enrollment)
        btn_layout.addWidget(self.btn_start)

        self.btn_reset = QPushButton("Làm lại")
        self.btn_reset.setProperty("class", "btn-secondary")
        self.btn_reset.clicked.connect(self.reset_flow)
        btn_layout.addWidget(self.btn_reset)

        info_layout.addLayout(btn_layout)

        self.btn_save = QPushButton("💾 Lưu hồ sơ vào CSDL")
        self.btn_save.setProperty("class", "btn-success")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_user_profile)
        info_layout.addWidget(self.btn_save)

        right_box.addWidget(info_card)

        # 5. Metrics Card
        self.metrics_card = MetricsCard()
        right_box.addWidget(self.metrics_card)

        right_box.addStretch()
        main_layout.addLayout(right_box, stretch=2)

    def _check_lockout_state(self) -> bool:
        """Kiểm tra và cập nhật giao diện nếu thiết bị đang bị khóa."""
        is_locked, rem_secs, msg = self.security_guard.is_locked()
        if is_locked:
            self.lockout_banner.setVisible(True)
            self.btn_start.setEnabled(False)
            if self.security_guard.is_permanent_locked:
                self.lbl_lockout_title.setText("🚫 THIẾT BỊ BỊ KHÓA VĨNH VIỄN DO GIAN LẬN")
                self.lbl_lockout_timer.setText("Chỉ Quản trị viên (Admin) mới có quyền mở khóa.")
            else:
                mins = rem_secs // 60
                secs = rem_secs % 60
                self.lbl_lockout_title.setText(f"⏳ TẠM KHÓA THIẾT BỊ (Lần vi phạm: {self.security_guard.violation_count})")
                self.lbl_lockout_timer.setText(f"Thời gian mở khóa sau: {mins:02d}:{secs:02d}")
            self.alert_text.setText(msg)
            self.alert_box.setStyleSheet(
                "background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;"
            )
            self.video_widget.set_oval_state("red", "Thiết bị tạm khóa do vi phạm an ninh", icon="🚫")
            return True
        else:
            self.lockout_banner.setVisible(False)
            self.btn_start.setEnabled(True)
            return False

    def prompt_admin_unlock(self):
        """Hộp thoại Admin mở khóa an ninh."""
        pin, ok = QInputDialog.getText(
            self,
            "Quản trị viên mở khóa",
            "Nhập mã PIN Quản trị viên (Admin PIN):",
            QLineEdit.EchoMode.Password,
        )
        if ok and pin:
            success, msg = self.security_guard.admin_unlock(pin)
            if success:
                play_chime_success()
                QMessageBox.information(self, "Thành công", msg)
                self.reset_flow()
                self._check_lockout_state()
            else:
                play_chime_warning()
                QMessageBox.critical(self, "Từ chối", msg)

    def start_enrollment(self):
        """Khởi động quy trình đăng ký tuần tự từng bước."""
        if self._check_lockout_state():
            return

        self.current_step = 1
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.baseline_face_h = None
        self.hold_start_time = None
        self.captured_frame = None
        self.captured_embedding = None
        self.flash_violation_frames = 0
        self.awaiting_final_capture = False
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(10)

        self.security_guard.start_stage(1)
        self.lbl_stage_timer.setVisible(True)

        self.video_widget.set_oval_zoom(False)
        self.video_widget.set_turn_arrow(None)
        self.video_widget.set_flash_color(None)
        self._update_step_ui()

    def reset_flow(self):
        """Đặt lại toàn bộ trạng thái về ban đầu."""
        self.flash_timer.stop()
        self.current_step = 0
        self.fqa_consecutive_frames = 0
        self.turn_hold_counter = 0
        self.baseline_face_h = None
        self.hold_start_time = None
        self.captured_frame = None
        self.captured_embedding = None
        self.flash_violation_frames = 0
        self.awaiting_final_capture = False
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_stage_timer.setVisible(False)

        self.security_guard.reset_spoof_streak()

        self.video_widget.set_oval_zoom(False)
        self.video_widget.set_turn_arrow(None)
        self.video_widget.set_flash_color(None)
        self.video_widget.set_oval_state("cyan", "Sẵn sàng đăng ký", progress=0.0, icon="🎯")
        self.video_widget.set_badge("")

        # Tắt full-screen overlay trên MainWindow nếu đang mở
        window = self.window()
        if hasattr(window, "set_flash_overlay"):
            window.set_flash_overlay(None)

        self.alert_text.setText("Bấm 'Bắt đầu đăng ký' để tiến hành thu thập khuôn mặt.")
        self.alert_box.setStyleSheet(
            "background-color: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px;"
        )
        self._update_step_ui()
        self._check_lockout_state()

    def _update_step_ui(self):
        """Đánh dấu bước đang thực hiện."""
        for idx, lbl in enumerate(self.step_labels):
            if idx + 1 == self.current_step:
                lbl.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 13px; padding: 3px 0px;")
            elif idx + 1 < self.current_step:
                lbl.setStyleSheet("color: #10b981; font-weight: 600; font-size: 13px; padding: 3px 0px;")
            else:
                lbl.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: 500; padding: 3px 0px;")

    @pyqtSlot(np.ndarray)
    def on_frame_received(self, frame_bgr: np.ndarray):
        """Xử lý từng khung hình video theo trạng thái máy eKYC với kiểm tra an ninh liên tục."""
        self.latest_frame = frame_bgr.copy()

        # 0. Nếu đang trong trạng thái bị khóa an ninh -> Cập nhật đồng hồ đếm ngược và chặn
        if self._check_lockout_state():
            self.video_widget.update_frame(frame_bgr)
            return

        # 1. Phân tích khuôn mặt thời gian thực bằng MediaPipe & MiniFASNet
        oval_cfg = self.video_widget.get_oval_cfg()
        face_res = self.analyzer.analyze(frame_bgr, oval=oval_cfg)

        # Cập nhật số liệu kỹ thuật FQA & PAD
        if face_res.detected:
            pad_prob = face_res.anti_spoof.real_prob if face_res.anti_spoof else 0.99
            is_real = face_res.anti_spoof.is_real if face_res.anti_spoof else True
            self.metrics_card.update_metrics(
                fps=self.last_fps,
                brightness=face_res.brightness_mean,
                sharpness=face_res.sharpness,
                yaw=face_res.yaw or 0.0,
                pad_prob=pad_prob,
                is_real=is_real,
            )

        # Nếu chưa bấm bắt đầu
        if self.current_step == 0:
            if not face_res.detected:
                self.video_widget.set_oval_state("gray", "Chưa phát hiện khuôn mặt", icon="👤")
            else:
                self.video_widget.set_oval_state("cyan", "Nhìn thẳng vào Oval", icon="🎯")
            self.video_widget.update_frame(frame_bgr)
            return

        # Nếu đang trong bước hoàn tất chụp ảnh chân dung (sau khi quét màu quang học xong)
        if getattr(self, "awaiting_final_capture", False):
            if time.time() < self.final_capture_cooldown:
                self.video_widget.set_oval_state("cyan", "Đang hoàn tất ảnh chân dung...", icon="📸")
                self.video_widget.update_frame(frame_bgr)
                return

            if face_res.detected and not face_res.is_upside_down:
                try:
                    self.captured_frame = frame_bgr.copy()
                    self.captured_embedding = self.extractor.extract_embedding(
                        self.captured_frame,
                        landmarks_5pts=face_res.arcface_kps,
                        bbox=face_res.bbox,
                    )
                    self.awaiting_final_capture = False
                    self.current_step = 6
                    self._update_step_ui()
                    self.progress_bar.setValue(100)
                    self.btn_save.setEnabled(True)
                    self.lbl_stage_timer.setVisible(False)

                    self.video_widget.set_oval_state("green", "ĐÃ VƯỢT QUA ĐẦY ĐỦ CÁC TẦNG BẢO MẬT eKYC!", progress=1.0, icon="✅")
                    self.video_widget.set_badge("ĐÃ DUYỆT 512D", "green")
                    play_chime_success()

                    self.alert_text.setText("Bước 6: Nhập MSSV và Họ tên, sau đó bấm 'Lưu hồ sơ vào CSDL'.")
                    self.alert_box.setStyleSheet(
                        "background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 12px;"
                    )
                    self.video_widget.update_frame(frame_bgr)
                    return
                except Exception as e:
                    print("Lỗi trích xuất ArcFace trên frame chân dung:", e)

            # Chờ người dùng nhìn thẳng trong vòng 4s, nếu quá thời gian mới quét lại nhẹ nhàng
            if time.time() > self.final_capture_deadline:
                self.awaiting_final_capture = False
                self._start_color_flashing()
            else:
                self.video_widget.set_oval_state("cyan", "Nhìn thẳng vào camera để hoàn tất lưu ảnh...", icon="📸")
            self.video_widget.update_frame(frame_bgr)
            return

        # Đang trong chuỗi Color Flashing
        if self.current_step == 5:
            # Nếu đang trong thời gian cooldown sau khi bị hủy do quay đầu
            if hasattr(self, "flash_cooldown_until") and time.time() < self.flash_cooldown_until:
                self.video_widget.update_frame(frame_bgr)
                return

            # Nếu timer chưa chạy (vừa hết cooldown) -> khởi động lại khi mặt đã tương đối thẳng
            if not self.flash_timer.isActive():
                yaw = abs(face_res.yaw or 0.0) if face_res.detected else 99.0
                pitch = abs(face_res.pitch or 0.0) if face_res.detected else 99.0
                oval_dist = face_res.oval_dist if face_res.detected else 99.0

                if face_res.detected and yaw <= 18.0 and pitch <= 18.0 and oval_dist <= 1.60:
                    self._start_color_flashing()
                else:
                    self.video_widget.set_oval_state("cyan", "Nhìn thẳng vào Oval để quét lại màu...", icon="🎯")
                self.video_widget.update_frame(frame_bgr)
                return

            # ĐANG QUÉT MÀU: KIỂM SOÁT ĐẦU VỚI BỘ LỌC CHỐNG NHIỄU KHUNG HÌNH (JITTER-FILTERED)
            yaw = abs(face_res.yaw or 0.0) if face_res.detected else 99.0
            pitch = abs(face_res.pitch or 0.0) if face_res.detected else 99.0
            oval_dist = face_res.oval_dist if face_res.detected else 99.0

            is_turning_head = (yaw > 22.0 or pitch > 20.0)
            is_outside = (oval_dist > 1.70)
            is_lost = (not face_res.detected)

            if is_lost or is_turning_head or is_outside:
                self.flash_violation_frames += 1
                # Chỉ hủy khi thực sự quay mặt hoặc lệch liên tục >= 8 frames (~0.25s), không hủy vì chớp nhiễu
                if self.flash_violation_frames >= 8:
                    if is_lost:
                        reason = "Mất khuôn mặt! Giữ yên đầu trong Oval."
                    elif is_turning_head:
                        dir_txt = "TRÁI" if (face_res.yaw or 0) > 0 else "PHẢI"
                        reason = f"Đầu bị quay {dir_txt} ({yaw:.0f}°)! Giữ thẳng đầu."
                    else:
                        reason = "Mặt bị trôi xa Oval! Giữ vị trí trung tâm."
                    self._abort_and_restart_color_flashing(reason)
            else:
                self.flash_violation_frames = 0

            self.video_widget.update_frame(frame_bgr)
            return

        # =================================================================
        # 2. THEO DÕI CHỐNG GIẢ MẠO XUYÊN SUỐT (CONTINUOUS ANTI-SPOOFING)
        # =================================================================
        if face_res.detected and face_res.anti_spoof:
            if not face_res.anti_spoof.is_real:
                # Tính thời gian delta giữa 2 frame
                dt = 1.0 / max(1.0, self.last_fps)
                is_escalated, warn_msg, penalty = self.security_guard.record_spoof_frame(self.current_step, dt)

                if is_escalated:
                    play_chime_warning()
                    self.reset_flow()
                    self._check_lockout_state()
                    QMessageBox.critical(self, "CẢNH BÁO VI PHẠM AN NINH", warn_msg)
                    return
                else:
                    self.video_widget.set_oval_state("red", warn_msg, icon="⚠️")
                    self.alert_text.setText(warn_msg)
                    self.video_widget.update_frame(frame_bgr)
                    return
            else:
                # Mặt thật -> reset bộ đếm giả mạo liên tục
                self.security_guard.reset_spoof_streak()

        # =================================================================
        # 4. XỬ LÝ THEO TỪNG GIAI ĐOẠN MÁY TRẠNG THÁI
        # =================================================================
        v = evaluate_face(
            face_res,
            require_oval=(self.current_step == 1),
            is_turning=(self.current_step in (2, 3)),
        )

        # -------------------------------------------------------------
        # BƯỚC 1: Căn mặt vào Oval chuẩn (FQA)
        # -------------------------------------------------------------
        if self.current_step == 1:
            if not v.all_ok:
                self.fqa_consecutive_frames = max(0, self.fqa_consecutive_frames - 1)
                self.video_widget.set_oval_state("yellow" if face_res.detected else "red", v.message, icon="🎯")
                self.alert_text.setText(f"Bước 1: {v.message}")
            else:
                self.fqa_consecutive_frames += 1
                prog = min(1.0, self.fqa_consecutive_frames / 6.0)
                self.progress_bar.setValue(int(prog * 20))
                self.video_widget.set_oval_state("green", "Khuôn mặt hợp lệ! Chuẩn bị quay đầu...", progress=prog, icon="✨")
                if self.fqa_consecutive_frames >= 6:
                    play_chime_step()
                    self.baseline_face_h = face_res.face_h
                    self.current_step = 2
                    self.turn_hold_counter = 0
                    self.security_guard.start_stage(2)  # Bắt đầu tính giờ Bước 2
                    self._update_step_ui()
                    self.video_widget.set_turn_arrow("left")
                    self.video_widget.set_oval_state("violet", "Quay đầu sang bên TRÁI của bạn", icon="⬅️")
                    self.alert_text.setText("Bước 2: ⬅️ Vui lòng quay đầu từ từ sang bên TRÁI của bạn (Góc ≥ +15°)")
                    self.progress_bar.setValue(25)

        # -------------------------------------------------------------
        # BƯỚC 2: Quay đầu sang TRÁI (Turn Left)
        # -------------------------------------------------------------
        elif self.current_step == 2:
            yaw = face_res.yaw or 0.0
            self.video_widget.set_turn_arrow("left")

            if yaw >= 15.0:
                self.turn_hold_counter += 1
                prog = min(1.0, self.turn_hold_counter / 3.0)
                self.video_widget.set_oval_state("green", f"Đã nhận diện quay TRÁI ({yaw:+.1f}°)", progress=prog, icon="⬅️")
                if self.turn_hold_counter >= 3:
                    play_chime_step()
                    self.current_step = 3
                    self.turn_hold_counter = 0
                    self.security_guard.start_stage(3)  # Bắt đầu tính giờ Bước 3
                    self._update_step_ui()
                    self.video_widget.set_turn_arrow("right")
                    self.video_widget.set_oval_state("violet", "Quay đầu sang bên PHẢI của bạn", icon="➡️")
                    self.alert_text.setText("Bước 3: Vui lòng quay đầu từ từ sang bên PHẢI của bạn ➡️ (Góc ≤ -15°)")
                    self.progress_bar.setValue(45)
            else:
                self.turn_hold_counter = 0
                self.video_widget.set_oval_state(
                    "violet", f"Quay sang TRÁI của bạn (Hiện tại: {yaw:+.1f}° / Mục tiêu: ≥ +15°)", icon="⬅️"
                )

        # -------------------------------------------------------------
        # BƯỚC 3: Quay đầu sang PHẢI (Turn Right)
        # -------------------------------------------------------------
        elif self.current_step == 3:
            yaw = face_res.yaw or 0.0
            self.video_widget.set_turn_arrow("right")

            if yaw <= -15.0:
                self.turn_hold_counter += 1
                prog = min(1.0, self.turn_hold_counter / 3.0)
                self.video_widget.set_oval_state("green", f"Đã nhận diện quay PHẢI ({yaw:+.1f}°)", progress=prog, icon="➡️")
                if self.turn_hold_counter >= 3:
                    play_chime_step()
                    self.current_step = 4
                    self.turn_hold_counter = 0
                    self.hold_start_time = None
                    self.security_guard.start_stage(4)  # Bắt đầu tính giờ Bước 4
                    self._update_step_ui()
                    self.video_widget.set_turn_arrow(None)
                    self.video_widget.set_oval_zoom(True)
                    self.video_widget.set_oval_state("amber", "Tiến gần camera hơn nữa", icon="🔍")
                    self.alert_text.setText("Bước 4: 🔍 Đưa mặt LẠI GẦN camera hơn để lấp đầy Oval phóng to.")
                    self.progress_bar.setValue(65)
            else:
                self.turn_hold_counter = 0
                self.video_widget.set_oval_state(
                    "violet", f"Quay sang PHẢI của bạn (Hiện tại: {yaw:+.1f}° / Mục tiêu: ≤ -15°)", icon="➡️"
                )

        # -------------------------------------------------------------
        # BƯỚC 4: Đưa mặt lại gần camera (Zoom-in / Close-up Check)
        # -------------------------------------------------------------
        elif self.current_step == 4:
            self.video_widget.set_turn_arrow(None)
            self.video_widget.set_oval_zoom(True)

            base_h = self.baseline_face_h or 0.35
            growth = face_res.face_h / max(0.01, base_h)
            growth_pct = int(growth * 100)

            # Cự ly hợp lệ nếu: chiều cao tăng >= 10% HOẶC mặt chiếm >= 38% khung hình HOẶC scale_ratio >= 0.62
            is_close_enough = (
                (growth >= getattr(C, "ZOOM_MIN_GROWTH", 1.10))
                or (face_res.face_h >= 0.38)
                or (face_res.scale_ratio >= 0.62)
            )
            head_straight = (
                face_res.detected
                and not face_res.is_upside_down
                and abs(face_res.yaw or 0.0) <= 20.0
                and abs(face_res.pitch or 0.0) <= 20.0
            )

            if not face_res.detected or not is_close_enough or not head_straight:
                self.hold_start_time = None
                prog = min(1.0, max(0.0, (growth - 1.0) / 0.10, (face_res.face_h - 0.28) / 0.10))
                self.video_widget.set_oval_state(
                    "amber", f"Tiến gần lại camera một chút ({growth_pct}% / 110%)", progress=prog, icon="🔍"
                )
            else:
                now = time.time()
                if self.hold_start_time is None:
                    self.hold_start_time = now

                elapsed = now - self.hold_start_time
                hold_target = getattr(C, "ZOOM_HOLD_SEC", 0.35)
                hold_pct = min(1.0, elapsed / hold_target)
                self.progress_bar.setValue(int(65 + hold_pct * 15))
                self.video_widget.set_oval_state(
                    "green", "Cự ly rất tốt! Giữ yên...", progress=hold_pct, icon="✨"
                )

                if elapsed >= hold_target:
                    play_chime_step()
                    self._start_color_flashing()

        # -------------------------------------------------------------
        # BƯỚC 6: Đã hoàn tất các tầng bảo mật
        # -------------------------------------------------------------
        elif self.current_step == 6:
            self.video_widget.set_oval_state("green", "Khuôn mặt đã sẵn sàng để lưu vào CSDL.", icon="✅")

        self.video_widget.update_frame(frame_bgr)

    def _abort_and_restart_color_flashing(self, reason: str):
        """Hủy chuỗi quét màu nếu thí sinh quay đầu hoặc lệch khung hình, yêu cầu nhìn thẳng để quét lại."""
        if self.flash_timer.isActive():
            self.flash_timer.stop()
        self.flash_step = 0
        self.flash_frames = []
        self.flash_violation_frames = 0
        self.awaiting_final_capture = False

        self.video_widget.set_flash_color(None)
        window = self.window()
        if hasattr(window, "set_flash_overlay"):
            window.set_flash_overlay(None)

        play_chime_warning()
        self.video_widget.set_oval_state("red", f"⚠️ {reason}", icon="⚠️")
        self.alert_text.setText(f"Bước 5: ⚠️ {reason} Vui lòng giữ yên khuôn mặt để hệ thống tự động quét lại.")
        self.alert_box.setStyleSheet(
            "background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 12px;"
        )
        self.flash_cooldown_until = time.time() + 0.8

    def _generate_random_flash_sequence(self, count: int = 4) -> List[Tuple[QColor, Tuple[int, int, int], str]]:
        """Sinh chuỗi màu thách thức ngẫu nhiên từ bảng màu thuần bão hòa cao (100% saturation, không pha loãng)."""
        import random
        pool = list(SPECTRAL_COLOR_PALETTE)
        random.shuffle(pool)
        seq = [pool[0]]
        for _ in range(count - 1):
            candidates = [c for c in pool if c[2] != seq[-1][2]]
            seq.append(random.choice(candidates))
        return seq

    def _start_color_flashing(self):
        """Bắt đầu chuỗi nhấp nháy màu quang học toàn màn hình (Active Optical Color Flashing)."""
        self.current_step = 5
        self._update_step_ui()
        self.video_widget.set_oval_zoom(False)
        self.flash_step = 0
        self.flash_frames = []
        self.flash_violation_frames = 0
        self.awaiting_final_capture = False
        self.security_guard.start_stage(5)

        # Sinh ngẫu nhiên chuỗi 4 màu quang phổ 100% đậm đặc, không pha loãng
        self.flash_colors = self._generate_random_flash_sequence(4)

        self.alert_text.setText("Bước 5: 🌈 Đang quét quang phổ đa màu bão hòa cao... Nhìn thẳng màn hình!")
        self.alert_box.setStyleSheet(
            "background-color: rgba(168, 85, 247, 0.15); border: 1px solid #a855f7; border-radius: 8px; padding: 12px;"
        )
        self.progress_bar.setValue(85)

        self._on_flash_timer_tick()
        self.flash_timer.start(200)  # 200ms mỗi xung màu (tốc độ cao, dứt khoát, đón đầu AWB)

    def _on_flash_timer_tick(self):
        """Xử lý từng bước màu quang học toàn màn hình (khoét rỗng khung Oval)."""
        if self.flash_step > 0 and self.latest_frame is not None:
            self.flash_frames.append(self.latest_frame.copy())

        if self.flash_step >= len(self.flash_colors):
            self.flash_timer.stop()
            self.video_widget.set_flash_color(None)
            window = self.window()
            if hasattr(window, "set_flash_overlay"):
                window.set_flash_overlay(None)
            self._evaluate_optical_flashing()
            return

        qcolor, rgb, name = self.flash_colors[self.flash_step]
        self.video_widget.set_flash_color(qcolor)
        self.video_widget.set_oval_state("cyan", f"Đang quét ánh sáng: {name}")

        play_chime_tick()

        window = self.window()
        if hasattr(window, "set_flash_overlay"):
            oval_rect = self.video_widget.get_oval_window_rect()
            banner_msg = f"Đang quét quang phổ [Màu {self.flash_step + 1}/{len(self.flash_colors)}: {name}] - Nhìn thẳng màn hình"
            window.set_flash_overlay(qcolor, oval_rect, banner_msg)

        self.flash_step += 1

    def _evaluate_optical_flashing(self):
        """Đánh giá phản xạ quang học và chuyển sang bước chụp chân dung tự nhiên."""
        expected_rgb = [item[1] for item in self.flash_colors[: len(self.flash_frames)]]

        optical_passed = True
        if len(self.flash_frames) >= 2:
            try:
                res = self.optical_analyzer.analyze_sequence(
                    frames_bgr=self.flash_frames,
                    expected_colors_rgb=expected_rgb,
                )
                print(
                    f"[GUI OPTICAL FLASH] passed={res.passed} | r={res.correlation_score:.3f} | amp={res.amplitude:.2f} | verdict={res.verdict}"
                )
                optical_passed = res.passed
            except Exception as e:
                print("Lỗi phân tích quang học:", e)

        # Chuyển tiếp êm ái sang chụp chân dung ở điều kiện sáng phòng bình thường (tránh ám màu màn hình)
        self.awaiting_final_capture = True
        self.final_capture_cooldown = time.time() + 0.25  # 250ms cho camera ổn định độ sáng phòng
        self.final_capture_deadline = time.time() + 4.0
        self.video_widget.set_oval_state("cyan", "Đang hoàn tất ảnh chân dung...", icon="📸")

    @pyqtSlot(float)
    def on_fps_updated(self, fps: float):
        self.last_fps = fps

    def save_user_profile(self):
        """Lưu hồ sơ sinh trắc học vào SQLite ekyc.db."""
        if self.captured_frame is None or self.captured_embedding is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có ảnh khuôn mặt đạt chuẩn để lưu!")
            return

        user_id = self.input_user_id.text().strip()
        full_name = self.input_full_name.text().strip()
        notes = self.input_notes.text().strip()

        if not user_id:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Mã sinh viên (MSSV)!")
            self.input_user_id.setFocus()
            return

        if not full_name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Họ và tên thí sinh!")
            self.input_full_name.setFocus()
            return

        try:
            _, buffer = cv2.imencode(".jpg", self.captured_frame)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            snapshot_b64 = f"data:image/jpeg;base64,{b64_str}"

            res = self.user_repo.save_user(
                full_name=full_name,
                embedding=self.captured_embedding,
                snapshot_b64=snapshot_b64,
                user_id=user_id,
                metadata={"notes": notes, "enrolled_via": "desktop_gui"},
            )

            play_chime_success()
            QMessageBox.information(
                self,
                "Đăng ký thành công",
                f"Đã lưu thành công hồ sơ sinh trắc học của thí sinh:\n"
                f"- MSSV: {user_id}\n"
                f"- Họ tên: {full_name}\n"
                f"- Vector ArcFace: 512 chiều (Đã chuẩn hoá L2)\n"
                f"- Cơ sở dữ liệu: backend/data/ekyc.db",
            )

            self.input_user_id.clear()
            self.input_full_name.clear()
            self.input_notes.clear()
            self.reset_flow()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi lưu Database", f"Không thể lưu hồ sơ vào cơ sở dữ liệu: {e}")
