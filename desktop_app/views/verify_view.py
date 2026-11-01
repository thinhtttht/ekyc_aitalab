"""Verify View - 1:1 and 1:N real-time biometric verification with Active Color Flashing for exam proctoring."""
from __future__ import annotations

import base64
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
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


class VerifyView(QWidget):
    """Giao diện xác thực 1:1 và nhận diện 1:N với quét quang phổ (Color Flashing) trước khi vào thi."""

    enter_exam_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_repo = get_user_repository()
        self.extractor = get_arcface_extractor()
        self.analyzer = FaceAnalyzer(stream=True)
        self.optical_analyzer = OpticalAnalyzer()

        self.current_frame: Optional[np.ndarray] = None
        self.last_fps = 30.0
        self.auto_verify_timer = QTimer(self)
        self.auto_verify_timer.timeout.connect(self._on_auto_verify_tick)
        self.last_verify_time = 0.0
        self.last_matched_candidate: Optional[dict] = None

        # Cấu hình quét quang phổ (Color Flashing) toàn màn hình
        self.is_flashing = False
        self.flash_step = 0
        self.flash_frames = []
        self.flash_violation_frames = 0
        self.flash_cooldown_until = 0.0
        self.flash_colors = list(SPECTRAL_COLOR_PALETTE[:4])
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._on_flash_timer_tick)

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(20)

        # --- CỘT TRÁI: CAMERA STREAM ---
        left_box = QVBoxLayout()
        left_box.setSpacing(12)

        self.video_widget = VideoWidget()
        self.video_widget.set_oval_state("cyan", "Nhìn thẳng vào Oval để quét sinh trắc học")
        left_box.addWidget(self.video_widget, stretch=1)

        # Thanh điều khiển nhanh dưới camera
        cam_ctrl_layout = QHBoxLayout()
        self.chk_auto_verify = QCheckBox("Tự động xác thực khi nhìn thẳng")
        self.chk_auto_verify.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        self.chk_auto_verify.toggled.connect(self._toggle_auto_verify)
        cam_ctrl_layout.addWidget(self.chk_auto_verify)

        cam_ctrl_layout.addStretch()

        self.btn_verify_now = QPushButton("🌈 Điểm danh & Quét quang học (Color Flashing)")
        self.btn_verify_now.setProperty("class", "btn-primary")
        self.btn_verify_now.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
                padding: 10px 18px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0ea5e9;
            }
        """)
        self.btn_verify_now.clicked.connect(self.verify_current_face)
        cam_ctrl_layout.addWidget(self.btn_verify_now)

        left_box.addLayout(cam_ctrl_layout)
        main_layout.addLayout(left_box, stretch=3)

        # --- CỘT PHẢI: BẢNG KẾT QUẢ & CẤU HÌNH ---
        right_box = QVBoxLayout()
        right_box.setSpacing(14)

        # Card cấu hình chế độ
        cfg_card = QFrame()
        cfg_card.setProperty("class", "card")
        cfg_layout = QVBoxLayout(cfg_card)
        cfg_layout.setSpacing(10)

        title = QLabel("CHẾ ĐỘ XÁC THỰC SINH TRẮC HỌC")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;")
        cfg_layout.addWidget(title)

        mode_layout = QHBoxLayout()
        lbl_mode = QLabel("Phương thức:")
        lbl_mode.setStyleSheet("color: #94a3b8; font-size: 12px;")
        mode_layout.addWidget(lbl_mode)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("🔍 Nhận diện 1:N (Tự động tìm kiếm thí sinh)", "1toN")
        self.combo_mode.addItem("🎯 Xác thực 1:1 (So khớp theo MSSV)", "1to1")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.combo_mode, stretch=1)
        cfg_layout.addLayout(mode_layout)

        # Chọn thí sinh cho chế độ 1:1
        self.student_select_row = QHBoxLayout()
        self.lbl_target = QLabel("Mã thí sinh:")
        self.lbl_target.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.student_select_row.addWidget(self.lbl_target)

        self.combo_target_user = QComboBox()
        self.combo_target_user.setEditable(True)
        self.combo_target_user.setPlaceholderText("Nhập hoặc chọn MSSV")
        self.student_select_row.addWidget(self.combo_target_user, stretch=1)

        self.btn_refresh_users = QPushButton("🔄")
        self.btn_refresh_users.setToolTip("Tải lại danh sách thí sinh")
        self.btn_refresh_users.setProperty("class", "btn-secondary")
        self.btn_refresh_users.clicked.connect(self.refresh_users_combo)
        self.student_select_row.addWidget(self.btn_refresh_users)

        cfg_layout.addLayout(self.student_select_row)

        # Slider ngưỡng Cosine Similarity
        thresh_layout = QHBoxLayout()
        self.lbl_thresh = QLabel(f"Ngưỡng Cosine: {C.DEFAULT_VERIFY_THRESHOLD:.2f}")
        self.lbl_thresh.setStyleSheet("color: #94a3b8; font-size: 12px;")
        thresh_layout.addWidget(self.lbl_thresh)

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(30, 80)
        self.slider_thresh.setValue(int(C.DEFAULT_VERIFY_THRESHOLD * 100))
        self.slider_thresh.valueChanged.connect(self._on_slider_changed)
        thresh_layout.addWidget(self.slider_thresh, stretch=1)

        cfg_layout.addLayout(thresh_layout)
        right_box.addWidget(cfg_card)

        # Card kết quả xác thực
        self.result_card = QFrame()
        self.result_card.setProperty("class", "card-highlight")
        res_layout = QVBoxLayout(self.result_card)
        res_layout.setSpacing(10)

        # Banner trạng thái to
        self.banner_verdict = QLabel("CHƯA CÓ KẾT QUẢ")
        self.banner_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_verdict.setStyleSheet(
            "background-color: #334155; color: #94a3b8; font-size: 14px; font-weight: 700; padding: 10px; border-radius: 6px;"
        )
        res_layout.addWidget(self.banner_verdict)

        # Khung so sánh 2 ảnh: Ảnh hồ sơ đăng ký vs Ảnh điểm danh
        compare_layout = QHBoxLayout()
        compare_layout.setSpacing(12)

        # Ảnh hồ sơ
        orig_box = QVBoxLayout()
        orig_title = QLabel("ẢNH HỒ SƠ ĐĂNG KÝ")
        orig_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_title.setStyleSheet("font-size: 10px; font-weight: 600; color: #94a3b8;")
        orig_box.addWidget(orig_title)

        self.lbl_img_original = QLabel()
        self.lbl_img_original.setFixedSize(110, 110)
        self.lbl_img_original.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px;")
        self.lbl_img_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_box.addWidget(self.lbl_img_original)
        compare_layout.addLayout(orig_box)

        # Mũi tên so sánh
        arrow_lbl = QLabel("⇄")
        arrow_lbl.setStyleSheet("font-size: 24px; color: #38bdf8;")
        arrow_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compare_layout.addWidget(arrow_lbl)

        # Ảnh hiện tại lúc điểm danh
        live_box = QVBoxLayout()
        live_title = QLabel("ẢNH CAMERA ĐIỂM DANH")
        live_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_title.setStyleSheet("font-size: 10px; font-weight: 600; color: #94a3b8;")
        live_box.addWidget(live_title)

        self.lbl_img_live = QLabel()
        self.lbl_img_live.setFixedSize(110, 110)
        self.lbl_img_live.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px;")
        self.lbl_img_live.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_box.addWidget(self.lbl_img_live)
        compare_layout.addLayout(live_box)

        res_layout.addLayout(compare_layout)

        # Chi tiết số liệu so khớp
        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(14)
        metrics_grid.setVerticalSpacing(6)

        self.lbl_res_id = self._add_result_row(metrics_grid, 0, "Mã sinh viên:", "--")
        self.lbl_res_name = self._add_result_row(metrics_grid, 1, "Họ và tên:", "--")
        self.lbl_res_sim = self._add_result_row(metrics_grid, 2, "Độ tương đồng:", "--%")
        self.lbl_res_conf = self._add_result_row(metrics_grid, 3, "Độ tự tin:", "--%")
        self.lbl_res_liveness = self._add_result_row(metrics_grid, 4, "Chống giả mạo PAD:", "--")
        self.lbl_res_optical = self._add_result_row(metrics_grid, 5, "Quang phổ (Flashing):", "--")
        self.lbl_res_latency = self._add_result_row(metrics_grid, 6, "Thời gian xử lý:", "-- ms")

        res_layout.addLayout(metrics_grid)

        # Nút chuyển sang phòng thi khi xác thực thành công
        self.btn_enter_exam = QPushButton("🎓 VÀO PHÒNG THI (HIỆU CHUẨN 9 ĐIỂM) →")
        self.btn_enter_exam.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 14px;
                font-weight: 800;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0ea5e9;
            }
        """)
        self.btn_enter_exam.clicked.connect(self._on_enter_exam_clicked)
        self.btn_enter_exam.hide()
        res_layout.addWidget(self.btn_enter_exam)

        right_box.addWidget(self.result_card)

        # Bảng Top-K ứng viên (chế độ 1:N)
        self.table_candidates = QTableWidget()
        self.table_candidates.setColumnCount(4)
        self.table_candidates.setHorizontalHeaderLabels(["Hạng", "MSSV", "Họ tên", "Tương đồng"])
        self.table_candidates.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_candidates.setMaximumHeight(140)
        self.table_candidates.setVisible(False)
        right_box.addWidget(self.table_candidates)

        right_box.addStretch()
        main_layout.addLayout(right_box, stretch=2)

        self.refresh_users_combo()

    def _add_result_row(self, grid: QGridLayout, row: int, label: str, val: str) -> QLabel:
        lbl_title = QLabel(label)
        lbl_title.setStyleSheet("color: #94a3b8; font-size: 12px;")
        grid.addWidget(lbl_title, row, 0)

        lbl_val = QLabel(val)
        lbl_val.setStyleSheet("color: #f8fafc; font-size: 12px; font-weight: 600;")
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(lbl_val, row, 1)
        return lbl_val

    def refresh_users_combo(self):
        """Cập nhật danh sách MSSV vào combobox từ SQLite."""
        self.combo_target_user.clear()
        users = self.user_repo.list_users()
        for u in users:
            self.combo_target_user.addItem(f"{u['user_id']} - {u['full_name']}", u["user_id"])

    def _on_mode_changed(self, idx: int):
        is_1to1 = (self.combo_mode.currentData() == "1to1")
        self.lbl_target.setVisible(is_1to1)
        self.combo_target_user.setVisible(is_1to1)
        self.btn_refresh_users.setVisible(is_1to1)
        self.table_candidates.setVisible(not is_1to1)

    def _on_slider_changed(self, val: int):
        thresh = val / 100.0
        self.lbl_thresh.setText(f"Ngưỡng Cosine: {thresh:.2f}")

    def _toggle_auto_verify(self, checked: bool):
        if checked:
            self.auto_verify_timer.start(1500)
        else:
            self.auto_verify_timer.stop()

    def _on_auto_verify_tick(self):
        if self.isVisible() and not self.is_flashing:
            self.verify_current_face(is_silent=True)

    @pyqtSlot(np.ndarray)
    def on_frame_received(self, frame_bgr: np.ndarray):
        """Xử lý từng khung hình video theo thời gian thực."""
        self.current_frame = frame_bgr.copy()
        face_res = self.analyzer.analyze(frame_bgr, oval=C.OVAL_NORMAL)

        # Nếu đang trong chuỗi Color Flashing: KIỂM SOÁT ĐẦU PHẢI GIỮ YÊN TUYỆT ĐỐI
        if self.is_flashing:
            now_t = time.time()
            if now_t < self.flash_cooldown_until:
                self.video_widget.update_frame(frame_bgr)
                return

            yaw = abs(face_res.yaw or 0.0) if face_res.detected else 99.0
            pitch = abs(face_res.pitch or 0.0) if face_res.detected else 99.0
            oval_dist = face_res.oval_dist if face_res.detected else 99.0

            is_turning_head = (yaw > 22.0 or pitch > 20.0)
            is_outside = (oval_dist > 1.70)
            is_lost = (not face_res.detected)

            if is_lost or is_turning_head or is_outside:
                self.flash_violation_frames += 1
                if self.flash_violation_frames >= 8:
                    if is_lost:
                        reason = "Mất khuôn mặt! Giữ yên đầu trong Oval."
                    elif is_turning_head:
                        dir_txt = "TRÁI" if (face_res.yaw or 0) > 0 else "PHẢI"
                        reason = f"Đầu bị quay {dir_txt} ({yaw:.0f}°)! Giữ thẳng đầu."
                    else:
                        reason = "Mặt bị trôi ra ngoài Oval! Giữ vị trí trung tâm."
                    self._abort_verify_color_flashing(reason)
            else:
                self.flash_violation_frames = 0

            self.video_widget.update_frame(frame_bgr)
            return

        # Khi không quét màu: hiển thị trạng thái Oval
        if not face_res.detected:
            self.video_widget.set_oval_state("gray", "Chưa phát hiện khuôn mặt", icon="👤")
        else:
            yaw = abs(face_res.yaw or 0.0)
            pitch = abs(face_res.pitch or 0.0)
            if yaw > 15.0 or pitch > 15.0 or face_res.oval_dist > 1.35:
                self.video_widget.set_oval_state("amber", "Vui lòng nhìn thẳng vào giữa Oval để quét sinh trắc", icon="🎯")
            else:
                self.video_widget.set_oval_state("cyan", "Nhìn thẳng vào Oval - Sẵn sàng quét màu", icon="✨")

        self.video_widget.update_frame(frame_bgr)

    @pyqtSlot(float)
    def on_fps_updated(self, fps: float):
        self.last_fps = fps

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

    def verify_current_face(self, is_silent: bool = False):
        """Khởi động quy trình xác thực sinh trắc học tích hợp quét quang phổ (Color Flashing)."""
        if self.is_flashing:
            return

        if self.current_frame is None:
            if not is_silent:
                QMessageBox.warning(self, "Lỗi Camera", "Chưa nhận được tín hiệu hình ảnh từ webcam!")
            return

        frame = self.current_frame.copy()
        face_res = self.analyzer.analyze(frame, oval=C.OVAL_NORMAL, fresh=True)

        if not face_res.detected or face_res.face_count == 0:
            if not is_silent:
                self._render_failure("KHÔNG PHÁT HIỆN MẶT", "Không tìm thấy khuôn mặt trước camera.", time.time())
            return

        yaw = abs(face_res.yaw or 0.0)
        pitch = abs(face_res.pitch or 0.0)
        if yaw > 16.0 or pitch > 16.0 or face_res.oval_dist > 1.40:
            self.video_widget.set_oval_state("amber", "Vui lòng đưa mặt vào giữa Oval và nhìn thẳng!", icon="🎯")
            if not is_silent:
                self.banner_verdict.setText("VUI LÒNG NHÌN THẲNG VÀO KHUNG OVAL")
                self.banner_verdict.setStyleSheet("background-color: #d97706; color: white; font-size: 13px; font-weight: 700; padding: 10px; border-radius: 6px;")
            return

        # BẮT ĐẦU CHUỖI COLOR FLASHING TOÀN MÀN HÌNH
        self._start_verify_color_flashing()

    def _start_verify_color_flashing(self):
        """Kích hoạt chuỗi Color Flashing toàn màn hình cho khâu xác thực điểm danh."""
        self.is_flashing = True
        self.flash_step = 0
        self.flash_frames = []
        self.flash_violation_frames = 0
        self.btn_verify_now.setEnabled(False)
        self.btn_enter_exam.hide()

        # Sinh ngẫu nhiên 4 màu thuần bão hòa cao (100% saturation)
        self.flash_colors = self._generate_random_flash_sequence(4)

        self.banner_verdict.setText("🌈 ĐANG QUÉT QUANG PHỔ ĐA MÀU BÃO HÒA CAO (COLOR FLASHING)...")
        self.banner_verdict.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; font-size: 13px; font-weight: 700; padding: 10px; border-radius: 6px;"
        )

        self._on_flash_timer_tick()
        self.flash_timer.start(200)  # 200ms mỗi xung màu (nhanh, dứt khoát)

    def _abort_verify_color_flashing(self, reason: str):
        """Hủy chuỗi quét màu trong Verify nếu người dùng quay đầu, bắt đầu lại khi đã nhìn thẳng."""
        if self.flash_timer.isActive():
            self.flash_timer.stop()
        self.is_flashing = False
        self.flash_step = 0
        self.flash_frames = []
        self.flash_violation_frames = 0
        self.btn_verify_now.setEnabled(True)

        self.video_widget.set_flash_color(None)
        window = self.window()
        if hasattr(window, "set_flash_overlay"):
            window.set_flash_overlay(None)

        play_chime_warning()
        self.video_widget.set_oval_state("red", f"⚠️ {reason}", icon="⚠️")
        self.banner_verdict.setText(f"⚠️ {reason.upper()}! GIỮ YÊN ĐẦU ĐỂ QUÉT LẠI")
        self.banner_verdict.setStyleSheet(
            "background-color: #991b1b; color: #ffffff; font-size: 13px; font-weight: 700; padding: 10px; border-radius: 6px;"
        )
        self.flash_cooldown_until = time.time() + 0.8

    def _on_flash_timer_tick(self):
        """Xử lý từng bước màu quang học toàn màn hình."""
        if self.flash_step > 0 and self.current_frame is not None:
            self.flash_frames.append(self.current_frame.copy())

        if self.flash_step >= len(self.flash_colors):
            self.flash_timer.stop()
            self.is_flashing = False
            self.btn_verify_now.setEnabled(True)
            self.video_widget.set_flash_color(None)
            window = self.window()
            if hasattr(window, "set_flash_overlay"):
                window.set_flash_overlay(None)

            # Đánh giá quang học và so khớp sau khi hoàn tất quét màu
            self._finalize_verification_after_flashing()
            return

        qcolor, rgb, name = self.flash_colors[self.flash_step]
        self.video_widget.set_flash_color(qcolor)
        self.video_widget.set_oval_state("cyan", f"Đang quét ánh sáng: {name}")

        play_chime_tick()

        window = self.window()
        if hasattr(window, "set_flash_overlay"):
            oval_rect = self.video_widget.get_oval_window_rect()
            banner_msg = f"Đang quét phản xạ quang học [Màu {self.flash_step + 1}/{len(self.flash_colors)}: {name}] - Giữ thẳng đầu"
            window.set_flash_overlay(qcolor, oval_rect, banner_msg)

        self.flash_step += 1

    def _finalize_verification_after_flashing(self):
        """Đánh giá phản xạ quang phổ và thực hiện đối sánh ArcFace 512D."""
        start_t = time.time()

        # 1. Kiểm tra phản xạ quang phổ
        expected_rgb = [item[1] for item in self.flash_colors[: len(self.flash_frames)]]
        optical_passed = True
        amp = 0.0
        corr = 0.0

        if len(self.flash_frames) >= 2:
            try:
                opt_res = self.optical_analyzer.analyze_sequence(
                    frames_bgr=self.flash_frames,
                    expected_colors_rgb=expected_rgb,
                )
                optical_passed = opt_res.passed
                amp = opt_res.amplitude
                corr = opt_res.correlation_score
                print(f"[VERIFY OPTICAL FLASH] passed={opt_res.passed} | r={corr:.3f} | amp={amp:.2f} | {opt_res.verdict}")
            except Exception as e:
                print("Lỗi phân tích quang học verify:", e)

        if not optical_passed:
            self._render_failure(
                "TỪ CHỐI QUANG HỌC (SPOOF)",
                "Bề mặt không phản xạ đúng quang phổ tự nhiên (Nghi vấn ảnh in/màn hình).",
                start_t,
                pad_val=0.1
            )
            self.lbl_res_optical.setText(f"Spoof (r={corr:.2f})")
            self.lbl_res_optical.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: 600;")
            return

        self.lbl_res_optical.setText(f"Đạt (r={corr:.2f}, A={amp:.1f})")
        self.lbl_res_optical.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600;")

        # 2. Chụp khung hình chuẩn xác và trích xuất vector ArcFace
        if self.current_frame is None:
            self._render_failure("LỖI CAMERA", "Mất tín hiệu camera.", start_t)
            return

        frame = self.current_frame.copy()
        face_res = self.analyzer.analyze(frame, oval=C.OVAL_NORMAL, fresh=True)

        if not face_res.detected or face_res.face_count == 0:
            self._render_failure("KHÔNG PHÁT HIỆN MẶT", "Không tìm thấy khuôn mặt sau khi quét màu.", start_t)
            return

        # Kiểm tra MiniFASNet PAD
        if face_res.anti_spoof and not face_res.anti_spoof.is_real:
            msg = f"Từ chối giả mạo: {face_res.anti_spoof.message}"
            self._render_failure("PHÁT HIỆN GIẢ MẠO", msg, start_t, pad_val=face_res.anti_spoof.real_prob)
            return

        try:
            query_emb = self.extractor.extract_embedding(
                frame,
                landmarks_5pts=face_res.arcface_kps,
                bbox=face_res.bbox,
            )
        except Exception as e:
            self._render_failure("LỖI ARCFACE", f"Không thể trích xuất vector: {e}", start_t)
            return

        self._display_live_image(frame, face_res.bbox)

        # 3. Đối sánh 1:1 hoặc 1:N
        threshold = self.slider_thresh.value() / 100.0
        mode = self.combo_mode.currentData()

        if mode == "1to1":
            target_id = self.combo_target_user.currentData() or self.combo_target_user.currentText().split(" - ")[0].strip()
            if not target_id:
                QMessageBox.warning(self, "Chưa chọn thí sinh", "Vui lòng chọn hoặc nhập MSSV cần xác thực 1:1!")
                return
            res = self.user_repo.match_user_1_to_1(query_emb, target_id, threshold=threshold)
            self._render_match_result(res, start_t, face_res, mode="1to1")
        else:
            res = self.user_repo.match_user_1_to_n(query_emb, threshold=threshold, top_k=5)
            self._render_match_result(res, start_t, face_res, mode="1toN")

    def _render_match_result(self, res: dict, start_t: float, face_res, mode: str):
        latency_ms = round((time.time() - start_t) * 1000, 1)
        is_match = res.get("is_match", False)
        sim = res.get("similarity", 0.0)
        conf = res.get("confidence_pct", 0.0)

        matched_user = res.get("matched_user") or res.get("target_user")

        if is_match and matched_user:
            play_chime_success()
            self.last_matched_candidate = matched_user
            self.btn_enter_exam.show()
            self.banner_verdict.setText("✓ XÁC THỰC THÀNH CÔNG (ĐẠT QUANG PHỔ & SINH TRẮC)")
            self.banner_verdict.setStyleSheet(
                "background-color: #059669; color: #ffffff; font-size: 14px; font-weight: 700; padding: 10px; border-radius: 6px;"
            )
            self.video_widget.set_badge("XÁC THỰC ĐẠT", "green")
            self.video_widget.set_oval_state("green", f"Đã xác nhận: {matched_user.get('full_name', '')}", progress=1.0, icon="✅")

            self._display_enrolled_image(matched_user.get("snapshot_b64"))
            self.lbl_res_id.setText(matched_user.get("user_id", "--"))
            self.lbl_res_name.setText(matched_user.get("full_name", "--"))
        else:
            play_chime_warning()
            self.last_matched_candidate = None
            self.btn_enter_exam.hide()
            self.banner_verdict.setText("✗ KHÔNG TRÙNG KHỚP / TỪ CHỐI")
            self.banner_verdict.setStyleSheet(
                "background-color: #dc2626; color: #ffffff; font-size: 14px; font-weight: 700; padding: 10px; border-radius: 6px;"
            )
            self.video_widget.set_badge("TỪ CHỐI", "red")
            self.video_widget.set_oval_state("red", "Không trùng khớp với hồ sơ thí sinh", progress=0.0, icon="❌")

            if matched_user:
                self._display_enrolled_image(matched_user.get("snapshot_b64"))
                self.lbl_res_id.setText(matched_user.get("user_id", "--"))
                self.lbl_res_name.setText(matched_user.get("full_name", "--"))
            else:
                self.lbl_res_id.setText("Không xác định")
                self.lbl_res_name.setText("Không tìm thấy")
                self.lbl_img_original.clear()

        sim_color = "#10b981" if is_match else "#ef4444"
        self.lbl_res_sim.setText(f"{sim * 100:.1f}%")
        self.lbl_res_sim.setStyleSheet(f"color: {sim_color}; font-size: 12px; font-weight: 700;")
        self.lbl_res_conf.setText(f"{conf:.1f}%")

        pad_val = face_res.anti_spoof.real_prob if face_res.anti_spoof else 0.99
        self.lbl_res_liveness.setText(f"Real ({pad_val * 100:.1f}%)")
        self.lbl_res_liveness.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600;")
        self.lbl_res_latency.setText(f"{latency_ms} ms")

        if mode == "1toN":
            candidates = res.get("candidates", [])
            self.table_candidates.setRowCount(len(candidates))
            for row, cand in enumerate(candidates):
                rank_item = QTableWidgetItem(f"#{row + 1}")
                rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_candidates.setItem(row, 0, rank_item)

                self.table_candidates.setItem(row, 1, QTableWidgetItem(cand["user_id"]))
                self.table_candidates.setItem(row, 2, QTableWidgetItem(cand["full_name"]))

                sim_item = QTableWidgetItem(f"{cand['similarity'] * 100:.1f}%")
                sim_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if cand.get("is_match", False):
                    sim_item.setForeground(Qt.GlobalColor.green)
                self.table_candidates.setItem(row, 3, sim_item)

    def _render_failure(self, title: str, message: str, start_t: float, pad_val: Optional[float] = None):
        latency_ms = round((time.time() - start_t) * 1000, 1)
        self.last_matched_candidate = None
        self.btn_enter_exam.hide()
        self.banner_verdict.setText(f"✗ {title}")
        self.banner_verdict.setStyleSheet(
            "background-color: #991b1b; color: #ffffff; font-size: 13px; font-weight: 700; padding: 10px; border-radius: 6px;"
        )
        self.video_widget.set_badge("CẢNH BÁO", "red")
        self.video_widget.set_oval_state("red", message)

        self.lbl_res_id.setText("--")
        self.lbl_res_name.setText(message)
        self.lbl_res_sim.setText("0.0%")
        self.lbl_res_conf.setText("0.0%")
        if pad_val is not None:
            self.lbl_res_liveness.setText(f"Spoof ({pad_val * 100:.1f}%)")
            self.lbl_res_liveness.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: 600;")
        else:
            self.lbl_res_liveness.setText("--")
        self.lbl_res_latency.setText(f"{latency_ms} ms")

    def _on_enter_exam_clicked(self):
        """Chuyển tiếp thí sinh sang màn hình hiệu chuẩn ánh mắt toàn màn hình & phòng thi."""
        if self.last_matched_candidate:
            self.enter_exam_requested.emit(self.last_matched_candidate)

    def _display_enrolled_image(self, b64_str: Optional[str]):
        if not b64_str:
            self.lbl_img_original.clear()
            return
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            pix = QPixmap()
            pix.loadFromData(img_bytes)
            self.lbl_img_original.setPixmap(
                pix.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        except Exception:
            self.lbl_img_original.clear()

    def _display_live_image(self, frame_bgr: np.ndarray, bbox: Optional[tuple]):
        try:
            crop = frame_bgr
            if bbox:
                x1, y1, x2, y2 = bbox
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.lbl_img_live.setPixmap(
                pix.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        except Exception:
            self.lbl_img_live.clear()
