"""Main Application Window for eKYC Biometric System."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent, QFont, QIcon
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from camera_thread import CameraThread
from eye_tracker import EyeTracker
from realsense_thread import RealSenseCaptureThread
from theme import DARK_STYLESHEET
from views.calibration_view import CalibrationView, FullScreenCalibrationWindow
from views.enrollment_view import EnrollmentView
from views.enrollment_3d_view import Enrollment3DView
from views.exam_view import ExamView
from views.realsense_view import RealSenseView
from views.users_view import UsersView
from views.verify_view import VerifyView
from views.verify_3d_view import Verify3DView
from widgets.flash_overlay import FullScreenFlashOverlay


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng Native Desktop eKYC."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống eKYC Sinh trắc học 2D/3D - Thi vấn đáp & Giám thị AI Eye Tracking (PyQt6)")
        self.resize(1300, 820)
        self.setMinimumSize(1024, 700)

        # Áp dụng Dark Theme hiện đại
        self.setStyleSheet(DARK_STYLESHEET)

        # Khởi tạo Camera Worker Thread 2D
        self.camera_thread = CameraThread(camera_index=0)

        # Khởi tạo RealSense Capture Thread 3D
        self.rs_capture_thread = RealSenseCaptureThread()

        # Khởi tạo Eye Tracker dùng chung với Head Pose
        self.eye_tracker = EyeTracker()
        self.current_candidate_info: dict = {}

        self._init_ui()
        self._init_camera_connections()

        # Lớp phủ phát sáng quang học toàn màn hình (khoét rỗng khung Oval)
        self.flash_overlay = FullScreenFlashOverlay(self)

        # Khởi động camera luồng nền
        self.camera_thread.start()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR) ---
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background-color: #0b1120; border-right: 1px solid #1e293b;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 20, 14, 20)
        side_layout.setSpacing(10)

        # Logo / Tiêu đề hệ thống
        logo_box = QVBoxLayout()
        logo_title = QLabel("🛡️ eKYC PROCTOR")
        logo_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #38bdf8; letter-spacing: 1px;")
        logo_sub = QLabel("Sinh trắc học & Giám sát thi")
        logo_sub.setStyleSheet("font-size: 11px; color: #64748b; font-weight: 500;")
        logo_box.addWidget(logo_title)
        logo_box.addWidget(logo_sub)
        side_layout.addLayout(logo_box)

        side_layout.addSpacing(20)

        # --- NHÓM 1: NHÁNH 2D WEBCAM TIÊU CHUẨN ---
        lbl_sec_2d = QLabel("NHÁNH 1: WEBCAM 2D")
        lbl_sec_2d.setStyleSheet("font-size: 10px; font-weight: 800; color: #64748b; letter-spacing: 0.5px; margin-top: 4px;")
        side_layout.addWidget(lbl_sec_2d)

        self.btn_nav_enroll = QPushButton("👤  Đăng ký 2D (Webcam)")
        self.btn_nav_enroll.setProperty("class", "nav-btn")
        self.btn_nav_enroll.setCheckable(True)
        self.btn_nav_enroll.setChecked(True)

        self.btn_nav_verify = QPushButton("🔍  Điểm danh 2D (Webcam)")
        self.btn_nav_verify.setProperty("class", "nav-btn")
        self.btn_nav_verify.setCheckable(True)

        side_layout.addWidget(self.btn_nav_enroll)
        side_layout.addWidget(self.btn_nav_verify)

        # --- NHÓM 2: NHÁNH 3D REALSENSE ĐỘ SÂU ---
        lbl_sec_3d = QLabel("NHÁNH 2: 3D REALSENSE")
        lbl_sec_3d.setStyleSheet("font-size: 10px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px; margin-top: 10px;")
        side_layout.addWidget(lbl_sec_3d)

        self.btn_nav_enroll_3d = QPushButton("🌐  Đăng ký 3D (RealSense)")
        self.btn_nav_enroll_3d.setProperty("class", "nav-btn")
        self.btn_nav_enroll_3d.setCheckable(True)

        self.btn_nav_verify_3d = QPushButton("🛡️  Điểm danh 3D (RealSense)")
        self.btn_nav_verify_3d.setProperty("class", "nav-btn")
        self.btn_nav_verify_3d.setCheckable(True)

        side_layout.addWidget(self.btn_nav_enroll_3d)
        side_layout.addWidget(self.btn_nav_verify_3d)

        # --- NHÓM 3: CÔNG CỤ & THI CỬ ---
        lbl_sec_tools = QLabel("CÔNG CỤ & THI CỬ")
        lbl_sec_tools.setStyleSheet("font-size: 10px; font-weight: 800; color: #c084fc; letter-spacing: 0.5px; margin-top: 10px;")
        side_layout.addWidget(lbl_sec_tools)

        self.btn_nav_realsense = QPushButton("📷  Test Cam 3D/2D")
        self.btn_nav_realsense.setProperty("class", "nav-btn")
        self.btn_nav_realsense.setCheckable(True)

        self.btn_nav_calib = QPushButton("🧭  Góc xoay đầu (Head Pose)")
        self.btn_nav_calib.setProperty("class", "nav-btn")
        self.btn_nav_calib.setCheckable(True)

        self.btn_nav_exam = QPushButton("📝  Phòng thi Mock")
        self.btn_nav_exam.setProperty("class", "nav-btn")
        self.btn_nav_exam.setCheckable(True)

        self.btn_nav_users = QPushButton("📋  Danh sách hồ sơ")
        self.btn_nav_users.setProperty("class", "nav-btn")
        self.btn_nav_users.setCheckable(True)

        side_layout.addWidget(self.btn_nav_realsense)
        side_layout.addWidget(self.btn_nav_calib)
        side_layout.addWidget(self.btn_nav_exam)
        side_layout.addWidget(self.btn_nav_users)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_group.addButton(self.btn_nav_enroll, 0)
        self.nav_group.addButton(self.btn_nav_verify, 1)
        self.nav_group.addButton(self.btn_nav_enroll_3d, 2)
        self.nav_group.addButton(self.btn_nav_verify_3d, 3)
        self.nav_group.addButton(self.btn_nav_realsense, 4)
        self.nav_group.addButton(self.btn_nav_calib, 5)
        self.nav_group.addButton(self.btn_nav_exam, 6)
        self.nav_group.addButton(self.btn_nav_users, 7)
        self.nav_group.idClicked.connect(self._on_nav_clicked)

        side_layout.addStretch()

        # Thông tin phiên bản & Cấu hình AI
        sys_info = QFrame()
        sys_info.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 10px;")
        sys_info_layout = QVBoxLayout(sys_info)
        sys_info_layout.setSpacing(4)

        t_lbl = QLabel("THÔNG TIN BẢO MẬT")
        t_lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #38bdf8;")
        m1 = QLabel("• 2D PAD: MiniFASNetV2")
        m1.setStyleSheet("font-size: 11px; color: #94a3b8;")
        m2 = QLabel("• 3D PAD: RealSense Depth")
        m2.setStyleSheet("font-size: 11px; color: #94a3b8;")
        m3 = QLabel("• ArcFace: ResNet50 (512D)")
        m3.setStyleSheet("font-size: 11px; color: #94a3b8;")
        m4 = QLabel("• Calib: Fullscreen 9-Point")
        m4.setStyleSheet("font-size: 11px; color: #94a3b8;")
        sys_info_layout.addWidget(t_lbl)
        sys_info_layout.addWidget(m1)
        sys_info_layout.addWidget(m2)
        sys_info_layout.addWidget(m3)
        sys_info_layout.addWidget(m4)

        side_layout.addWidget(sys_info)
        root_layout.addWidget(sidebar)

        # --- KHU VỰC NỘI DUNG CHÍNH (STACKED WIDGET) ---
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background-color: #0f172a;")

        self.enrollment_view = EnrollmentView()            # 0
        self.verify_view = VerifyView()                    # 1
        self.enrollment_3d_view = Enrollment3DView()        # 2
        self.verify_3d_view = Verify3DView()                # 3
        self.realsense_view = RealSenseView()              # 4
        self.calibration_view = CalibrationView(eye_tracker=self.eye_tracker)  # 5
        self.exam_view = ExamView(eye_tracker=self.eye_tracker)                # 6
        self.users_view = UsersView()                      # 7

        # Kết nối luồng luân chuyển sau khi xác thực
        self.verify_view.enter_exam_requested.connect(self._on_enter_exam_requested)
        self.verify_3d_view.enter_exam_requested.connect(self._on_enter_exam_requested)
        self.exam_view.exit_exam_requested.connect(self._on_exit_exam_requested)

        self.stacked_widget.addWidget(self.enrollment_view)       # 0
        self.stacked_widget.addWidget(self.verify_view)           # 1
        self.stacked_widget.addWidget(self.enrollment_3d_view)    # 2
        self.stacked_widget.addWidget(self.verify_3d_view)        # 3
        self.stacked_widget.addWidget(self.realsense_view)        # 4
        self.stacked_widget.addWidget(self.calibration_view)      # 5
        self.stacked_widget.addWidget(self.exam_view)             # 6
        self.stacked_widget.addWidget(self.users_view)            # 7

        root_layout.addWidget(self.stacked_widget, stretch=1)

        # --- THANH TRẠNG THÁI (STATUS BAR) ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status_cam = QLabel("Webcam: Đang kết nối...")
        self.lbl_status_cam.setStyleSheet("color: #94a3b8; font-size: 12px; margin-right: 15px;")
        self.status_bar.addWidget(self.lbl_status_cam)

        self.lbl_status_fps = QLabel("FPS: --")
        self.lbl_status_fps.setStyleSheet("color: #38bdf8; font-size: 12px; margin-right: 15px;")
        self.status_bar.addWidget(self.lbl_status_fps)

        self.lbl_status_db = QLabel("SQLite: ekyc.db (Sẵn sàng)")
        self.lbl_status_db.setStyleSheet("color: #10b981; font-size: 12px;")
        self.status_bar.addPermanentWidget(self.lbl_status_db)

    def _init_camera_connections(self):
        """Kết nối luồng camera 2D và RealSense 3D với các View."""
        self.camera_thread.status_changed.connect(self._on_cam_status_changed)
        self.camera_thread.fps_updated.connect(self._on_fps_updated)
        self.camera_thread.frame_ready.connect(self._dispatch_frame)

        self.rs_capture_thread.frames_ready.connect(self._dispatch_realsense_frame)

    def _dispatch_frame(self, frame):
        """Chuyển tiếp frame video 2D tới View đang hiển thị."""
        curr_idx = self.stacked_widget.currentIndex()
        if curr_idx == 0:
            self.enrollment_view.on_frame_received(frame)
        elif curr_idx == 1:
            self.verify_view.on_frame_received(frame)
        elif curr_idx == 5:
            self.calibration_view.on_frame_received(frame)
        elif curr_idx == 6:
            self.exam_view.on_frame_received(frame)

    def _dispatch_realsense_frame(self, color_bgr, raw_depth_mm, depth_colormap):
        """Chuyển tiếp luồng RealSense 3D (Color + Depth) tới View 3D tương ứng."""
        curr_idx = self.stacked_widget.currentIndex()
        if curr_idx == 2:
            self.enrollment_3d_view.on_frames_received(color_bgr, raw_depth_mm, depth_colormap)
        elif curr_idx == 3:
            self.verify_3d_view.on_frames_received(color_bgr, raw_depth_mm, depth_colormap)

    def _on_nav_clicked(self, view_id: int):
        prev_idx = self.stacked_widget.currentIndex()

        # Quản lý luồng phần cứng giữa 2 Nhánh (2D và 3D) để tránh tranh chấp cổng USB:
        # Nếu đang ở nhánh 3D (2, 3) và chuyển sang nhánh khác:
        if prev_idx in (2, 3) and view_id not in (2, 3):
            self.rs_capture_thread.stop()

        # Nếu đang ở tab Test RealSense (4) và chuyển đi nơi khác:
        if prev_idx == 4 and view_id != 4:
            self.realsense_view.stop_stream()

        # Khi chuyển VÀO nhánh 3D (2 hoặc 3):
        if view_id in (2, 3):
            if self.camera_thread.isRunning():
                self.camera_thread.stop()
            if hasattr(self, "realsense_view"):
                self.realsense_view.stop_stream()
            if not self.rs_capture_thread.isRunning():
                self.rs_capture_thread.start()

        # Khi chuyển VÀO tab Test RealSense (4):
        elif view_id == 4:
            if self.camera_thread.isRunning():
                self.camera_thread.stop()
            if self.rs_capture_thread.isRunning():
                self.rs_capture_thread.stop()
            self.realsense_view.start_stream()

        # Khi chuyển VÀO các tab 2D hoặc Công cụ khác (0, 1, 5, 6, 7):
        else:
            if self.rs_capture_thread.isRunning():
                self.rs_capture_thread.stop()
            if hasattr(self, "realsense_view"):
                self.realsense_view.stop_stream()
            if not self.camera_thread.isRunning():
                self.camera_thread.start()

        self.stacked_widget.setCurrentIndex(view_id)

        if view_id == 1:
            self.verify_view.refresh_users_combo()
        elif view_id == 3:
            self.verify_3d_view.refresh_users_combo()
        elif view_id == 7:
            self.users_view.load_users()

    def _on_enter_exam_requested(self, candidate_info: dict):
        """Khi thí sinh được xác thực thành công (2D hoặc 3D): Chuyển thẳng vào phòng thi với giám sát Landmark Head Pose."""
        self.current_candidate_info = candidate_info
        self.btn_nav_exam.setChecked(True)
        self.stacked_widget.setCurrentIndex(6)
        self.exam_view.start_exam(candidate_info)

    def _on_exit_exam_requested(self):
        """Khi nộp bài hoặc yêu cầu quay lại từ phòng thi."""
        self.btn_nav_verify.setChecked(True)
        self.stacked_widget.setCurrentIndex(1)
        self.verify_view.refresh_users_combo()

    def _on_cam_status_changed(self, msg: str, is_ok: bool):
        color = "#10b981" if is_ok else "#ef4444"
        self.lbl_status_cam.setText(f"Webcam: {msg}")
        self.lbl_status_cam.setStyleSheet(f"color: {color}; font-size: 12px; margin-right: 15px;")

    def _on_fps_updated(self, fps: float):
        self.lbl_status_fps.setText(f"FPS: {fps:.1f}")
        self.enrollment_view.on_fps_updated(fps)
        self.verify_view.on_fps_updated(fps)

    def set_flash_overlay(self, color, oval_rect=None, text=""):
        """Kích hoạt hoặc tắt lớp phủ nháy màu quang học toàn màn hình."""
        if hasattr(self, "flash_overlay"):
            self.flash_overlay.set_flash(color, oval_rect, text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "flash_overlay") and self.flash_overlay.isVisible():
            self.flash_overlay.setGeometry(self.rect())

    def closeEvent(self, event: QCloseEvent):
        """Dừng camera và giải phóng tài nguyên khi đóng ứng dụng."""
        if hasattr(self, "fs_calibration_window"):
            self.fs_calibration_window.close()
        if hasattr(self, "rs_capture_thread"):
            self.rs_capture_thread.stop()
        if hasattr(self, "realsense_view"):
            self.realsense_view.stop_stream()
        self.camera_thread.stop()
        event.accept()
