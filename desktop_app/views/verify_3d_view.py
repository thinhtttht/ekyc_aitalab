"""Verify 3D View - Xác thực & Điểm danh thí sinh bằng camera 3D RealSense.
Kiểm tra chống giả mạo chiều sâu 3D liên tục kết hợp so khớp khuôn mặt ArcFace 512D.
Chặn đứng hoàn toàn việc dùng ảnh in phẳng, điện thoại, máy tính bảng để điểm danh hộ.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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
from face_analyzer import FaceAnalyzer
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

SPECTRAL_COLOR_PALETTE = [
    (QColor(255, 0, 40), (255, 0, 40), "ĐỎ ĐẬM"),
    (QColor(0, 245, 75), (0, 245, 75), "XANH LỤC ĐẬM"),
    (QColor(0, 80, 255), (0, 80, 255), "XANH LAM ĐẬM"),
    (QColor(255, 200, 0), (255, 200, 0), "VÀNG ĐẬM"),
]


class Verify3DView(QWidget):
    """Giao diện xác thực & điểm danh 3D RealSense với bảo mật đa tầng."""

    enter_exam_requested = pyqtSignal(dict)  # candidate_info

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_repo = get_user_repository()
        self.extractor = get_arcface_extractor()
        self.analyzer = FaceAnalyzer(stream=True)
        self.depth_detector = DepthLivenessDetector()

        self.selected_user_id: Optional[str] = None
        self.selected_user_info: Optional[dict] = None
        self.target_embedding: Optional[np.ndarray] = None

        self.is_verifying = False
        self.consecutive_match_frames = 0
        self.match_threshold_frames = 5  # 5 frame liên tiếp đạt chuẩn 3D + trong Oval (~0.6s)
        self.verified_success = False

        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.latest_depth_cmap: Optional[np.ndarray] = None

        # Flash sequence
        self.flash_step = 0
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._on_flash_timer_tick)

        # AI inference caching & cadence control để duy trì 30 FPS mượt mà
        self._cached_face_res = None
        self._cached_liveness_res: Optional[DepthLivenessResult] = None
        self._last_ai_time = 0.0
        self._last_arcface_time = 0.0
        self._frame_count = 0

        self._init_ui()
        self.refresh_users_combo()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # === CỘT TRÁI: CAMERA VIEW CHÍNH & MINI DEPTH ===
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)

        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(600, 480)
        left_layout.addWidget(self.video_widget, stretch=4)

        # Khung phụ: Depth Map & Cảnh báo liveness
        depth_box = QFrame()
        depth_box.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 6px;"
        )
        depth_layout = QHBoxLayout(depth_box)
        depth_layout.setContentsMargins(8, 6, 8, 6)

        d_title = QLabel("🌐 Bản đồ 3D Depth Map:")
        d_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 12px;")
        depth_layout.addWidget(d_title)

        self.lbl_depth_preview = QLabel()
        self.lbl_depth_preview.setFixedSize(180, 110)
        self.lbl_depth_preview.setStyleSheet("background-color: #000; border-radius: 4px;")
        self.lbl_depth_preview.setScaledContents(True)
        depth_layout.addWidget(self.lbl_depth_preview)

        self.lbl_spoof_alert = QLabel("Chờ bắt đầu phiên xác thực...")
        self.lbl_spoof_alert.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8; font-size: 13px; font-weight: 600; padding: 10px; border-radius: 6px;"
        )
        self.lbl_spoof_alert.setWordWrap(True)
        depth_layout.addWidget(self.lbl_spoof_alert, stretch=1)

        left_layout.addWidget(depth_box, stretch=1)
        layout.addLayout(left_layout, stretch=3)

        # === CỘT PHẢI: BẢNG ĐIỀU HƯỚNG VÀ THÔNG SỐ XÁC THỰC ===
        right_panel = QFrame()
        right_panel.setStyleSheet(
            "background-color: #1e293b; border-radius: 12px; border: 1px solid #334155;"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(12)

        # Tiêu đề
        h_title = QLabel("🛡️ ĐIỂM DANH 3D REALSENSE")
        h_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;")
        h_sub = QLabel("Đối soát ArcFace 512D + Chống giả mạo độ sâu 3D liên tục")
        h_sub.setStyleSheet("font-size: 11px; color: #94a3b8; margin-bottom: 4px;")
        right_layout.addWidget(h_title)
        right_layout.addWidget(h_sub)

        # CHỌN THÍ SINH
        user_select_box = QFrame()
        user_select_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        sel_layout = QVBoxLayout(user_select_box)
        sel_layout.setSpacing(6)

        sel_lbl = QLabel("Chọn thí sinh cần điểm danh:")
        sel_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #cbd5e1;")
        self.combo_users = QComboBox()
        self.combo_users.setStyleSheet(
            "background-color: #1e293b; color: white; padding: 8px; border-radius: 6px; font-size: 13px;"
        )
        self.combo_users.currentIndexChanged.connect(self._on_user_selection_changed)

        sel_layout.addWidget(sel_lbl)
        sel_layout.addWidget(self.combo_users)
        right_layout.addWidget(user_select_box)

        # KHUNG THÔNG TIN THÍ SINH (Ảnh gốc + Tên)
        user_info_frame = QFrame()
        user_info_frame.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        info_layout = QHBoxLayout(user_info_frame)
        info_layout.setSpacing(12)

        self.lbl_profile_photo = QLabel()
        self.lbl_profile_photo.setFixedSize(70, 90)
        self.lbl_profile_photo.setStyleSheet(
            "background-color: #1e293b; border-radius: 6px; border: 1px solid #334155;"
        )
        self.lbl_profile_photo.setScaledContents(True)

        user_details = QVBoxLayout()
        user_details.setSpacing(4)
        self.lbl_user_name = QLabel("Chưa chọn thí sinh")
        self.lbl_user_name.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        self.lbl_user_id = QLabel("Mã thí sinh: --")
        self.lbl_user_id.setStyleSheet("font-size: 12px; color: #94a3b8;")
        self.lbl_enrolled_branch = QLabel("Nhánh: --")
        self.lbl_enrolled_branch.setStyleSheet("font-size: 11px; color: #38bdf8;")

        user_details.addWidget(self.lbl_user_name)
        user_details.addWidget(self.lbl_user_id)
        user_details.addWidget(self.lbl_enrolled_branch)
        user_details.addStretch()

        info_layout.addWidget(self.lbl_profile_photo)
        info_layout.addLayout(user_details, stretch=1)
        right_layout.addWidget(user_info_frame)

        # BẢNG TELEMETRY 3D THỜI GIAN THỰC
        telemetry_box = QFrame()
        telemetry_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        grid = QGridLayout(telemetry_box)
        grid.setSpacing(8)

        self.card_dist = MetricsCard("Khoảng cách (Z)", "-- cm", "Chuẩn: 45-75 cm")
        self.card_relief = MetricsCard("Độ lồi sống mũi", "-- mm", "Mặt thật: >10 mm")
        self.card_similarity = MetricsCard("Độ khớp khuôn mặt", "-- %", "Ngưỡng: >= 55%")
        self.card_liveness = MetricsCard("Điểm 3D Liveness", "-- %", "Mặt thật: >= 60%")

        grid.addWidget(self.card_dist, 0, 0)
        grid.addWidget(self.card_relief, 0, 1)
        grid.addWidget(self.card_similarity, 1, 0)
        grid.addWidget(self.card_liveness, 1, 1)
        right_layout.addWidget(telemetry_box)

        # TRẠNG THÁI TIẾN ĐỘ XÁC THỰC
        prog_box = QFrame()
        prog_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setSpacing(6)

        self.lbl_status_text = QLabel("Sẵn sàng. Chọn thí sinh và bấm 'Bắt đầu xác thực'.")
        self.lbl_status_text.setStyleSheet("font-size: 12px; font-weight: 600; color: #f8fafc;")
        self.lbl_status_text.setWordWrap(True)

        self.match_progress = QProgressBar()
        self.match_progress.setRange(0, self.match_threshold_frames)
        self.match_progress.setValue(0)
        self.match_progress.setStyleSheet(
            "QProgressBar { background-color: #334155; height: 10px; border-radius: 5px; }"
            "QProgressBar::chunk { background-color: #10b981; border-radius: 5px; }"
        )

        prog_layout.addWidget(self.lbl_status_text)
        prog_layout.addWidget(self.match_progress)
        right_layout.addWidget(prog_box)

        # CÁC NÚT ĐIỀU HƯỚNG
        self.btn_toggle_verify = QPushButton("▶️  Bắt đầu xác thực 3D")
        self.btn_toggle_verify.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: 700; padding: 12px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_toggle_verify.clicked.connect(self._on_toggle_verify)
        right_layout.addWidget(self.btn_toggle_verify)

        self.btn_enter_exam = QPushButton("🚪  Tiến hành vào phòng thi")
        self.btn_enter_exam.setEnabled(False)
        self.btn_enter_exam.setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; font-weight: 800; padding: 12px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #059669; }"
            "QPushButton:disabled { background-color: #334155; color: #64748b; }"
        )
        self.btn_enter_exam.clicked.connect(self._on_enter_exam_clicked)
        right_layout.addWidget(self.btn_enter_exam)

        right_layout.addStretch()
        layout.addWidget(right_panel, stretch=2)

    def refresh_users_combo(self):
        """Tải danh sách thí sinh từ cơ sở dữ liệu SQLite."""
        self.combo_users.clear()
        self.combo_users.addItem("-- Chọn thí sinh để điểm danh --", None)
        users = self.user_repo.list_users()
        for u in users:
            branch_tag = u.get("metadata", {}).get("branch", "2d_webcam") if isinstance(u.get("metadata"), dict) else "2d"
            tag = "🌐 3D" if "3d" in branch_tag else "📷 2D"
            self.combo_users.addItem(f"{tag} {u['full_name']} ({u['user_id']})", u)

    def _on_user_selection_changed(self, idx: int):
        data = self.combo_users.currentData()
        if data is None:
            self.selected_user_id = None
            self.selected_user_info = None
            self.target_embedding = None
            self.lbl_user_name.setText("Chưa chọn thí sinh")
            self.lbl_user_id.setText("Mã thí sinh: --")
            self.lbl_enrolled_branch.setText("Nhánh: --")
            self.lbl_profile_photo.clear()
            self._reset_match_state()
            return

        self.selected_user_id = data["user_id"]
        self.selected_user_info = data
        self.target_embedding = data["embedding"]

        self.lbl_user_name.setText(data["full_name"])
        self.lbl_user_id.setText(f"Mã: {data['user_id']}")
        branch = data.get("metadata", {}).get("branch", "2d_webcam") if isinstance(data.get("metadata"), dict) else "2d"
        self.lbl_enrolled_branch.setText(f"Đăng ký qua: {branch.upper()}")

        # Hiển thị ảnh chụp hồ sơ
        snap_b64 = data.get("snapshot_b64", "")
        if snap_b64:
            try:
                raw_bytes = base64.b64decode(snap_b64)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = rgb_img.shape[:2]
                q_img = QImage(rgb_img.data, w, h, w * 3, QImage.Format.Format_RGB888)
                self.lbl_profile_photo.setPixmap(QPixmap.fromImage(q_img))
            except Exception:
                self.lbl_profile_photo.clear()

        self._reset_match_state()

    def _reset_match_state(self):
        self.consecutive_match_frames = 0
        self.match_progress.setValue(0)
        self.verified_success = False
        self.btn_enter_exam.setEnabled(False)

    def _on_toggle_verify(self):
        if self.is_verifying:
            self.is_verifying = False
            self.btn_toggle_verify.setText("▶️  Bắt đầu xác thực 3D")
            self.btn_toggle_verify.setStyleSheet(
                "QPushButton { background-color: #0284c7; color: white; font-weight: 700; padding: 10px; border-radius: 6px; }"
                "QPushButton:hover { background-color: #0369a1; }"
            )
            self.lbl_status_text.setText("Đã tạm dừng xác thực.")
        else:
            if not self.selected_user_id:
                QMessageBox.warning(self, "Chưa chọn thí sinh", "Vui lòng chọn thí sinh trong danh sách trước.")
                return
            self.is_verifying = True
            self.btn_toggle_verify.setText("⏹️  Dừng xác thực")
            self.btn_toggle_verify.setStyleSheet(
                "QPushButton { background-color: #ef4444; color: white; font-weight: 700; padding: 10px; border-radius: 6px; }"
                "QPushButton:hover { background-color: #dc2626; }"
            )
            self._reset_match_state()
            play_chime_ding()

    def _on_flash_clicked(self):
        """Kích hoạt chớp màu quang phổ ngắn 800ms."""
        self.flash_step = 0
        self.flash_timer.start(200)
        self._on_flash_timer_tick()

    def _on_flash_timer_tick(self):
        main_win = self.window()
        if self.flash_step < len(SPECTRAL_COLOR_PALETTE):
            color, _, name = SPECTRAL_COLOR_PALETTE[self.flash_step]
            play_chime_tick()
            if hasattr(main_win, "set_flash_overlay"):
                main_win.set_flash_overlay(color, None, f"Quét xác thực quang phổ: {name}")
            self.flash_step += 1
        else:
            self.flash_timer.stop()
            if hasattr(main_win, "set_flash_overlay"):
                main_win.set_flash_overlay(None)

    def on_frames_received(
        self,
        color_bgr: np.ndarray,
        raw_depth_mm: np.ndarray,
        depth_colormap: np.ndarray,
    ):
        """Nhận và xử lý đồng thời luồng RGB và Depth từ RealSense."""
        self.latest_color = color_bgr
        self.latest_depth_raw = raw_depth_mm
        self.latest_depth_cmap = depth_colormap
        self._frame_count += 1

        # Cập nhật khung mini preview Depth Map (cách frame và dùng nearest neighbor để tối ưu CPU)
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

            # PHÂN TÍCH CHIỀU SÂU THỜI GIAN THỰC (DEPTH LIVENESS)
            if self._cached_face_res.face_count == 1 and raw_depth_mm is not None:
                self._cached_liveness_res = self.depth_detector.analyze(raw_depth_mm, self._cached_face_res.bbox)
                self._update_telemetry(self._cached_liveness_res)
            else:
                self._cached_liveness_res = None

            # SO KHỚP KHUÔN MẶT ARCFACE NẾU ĐANG BẬT XÁC THỰC
            if self.is_verifying and self._cached_face_res.face_count == 1:
                # ArcFace chạy nhịp ~120ms để không nghẽn CPU
                if now_t - self._last_arcface_time >= 0.12:
                    self._last_arcface_time = now_t
                    self._evaluate_match(frame, self._cached_face_res, self._cached_liveness_res)

        # Render video mượt mà 30 FPS trên giao diện
        self.video_widget.set_frame(frame, face_result=self._cached_face_res, show_oval=True)

    def _update_telemetry(self, res: DepthLivenessResult):
        self.card_dist.set_value(f"{res.distance_cm:.1f} cm")
        self.card_dist.set_status("Tốt" if 45 <= res.distance_cm <= 75 else "Chưa chuẩn", 45 <= res.distance_cm <= 75)

        self.card_relief.set_value(f"+{res.nose_relief_mm:.1f} mm")
        self.card_relief.set_status("3D nổi rõ" if res.nose_relief_mm >= 10.0 else "Phẳng (2D)", res.nose_relief_mm >= 10.0)

        score_pct = int(res.liveness_score * 100)
        self.card_liveness.set_value(f"{score_pct} %")
        self.card_liveness.set_status("NGƯỜI THẬT 3D" if res.is_real_3d else "GIẢ MẠO 2D", res.is_real_3d)

        if res.is_real_3d:
            self.lbl_spoof_alert.setText(f"✅ NGƯỜI THẬT 3D: Cự ly {res.distance_cm:.1f}cm | Sống mũi +{res.nose_relief_mm:.1f}mm")
            self.lbl_spoof_alert.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-size: 12px; font-weight: 700; padding: 8px; border-radius: 6px; border: 1px solid #059669;"
            )
        else:
            self.lbl_spoof_alert.setText(f"🚨 {res.message.upper()}")
            self.lbl_spoof_alert.setStyleSheet(
                "background-color: #7f1d1d; color: #f87171; font-size: 12px; font-weight: 700; padding: 8px; border-radius: 6px; border: 1px solid #dc2626;"
            )

    def _evaluate_match(
        self,
        frame: np.ndarray,
        face_res,
        liveness_res: Optional[DepthLivenessResult],
    ):
        """Kiểm tra điều kiện người thật 3D và tính độ khớp cosine ArcFace."""
        # 1. Bắt buộc vượt qua kiểm tra 3D Depth Liveness (Người thật 3D)
        if liveness_res is None or not liveness_res.is_real_3d:
            self.consecutive_match_frames = max(0, self.consecutive_match_frames - 2)
            self.match_progress.setValue(self.consecutive_match_frames)
            self.lbl_status_text.setText(f"❌ {liveness_res.message if liveness_res else 'Chưa có chiều sâu 3D'}!")
            return

        # 2. Bắt buộc đầu / khuôn mặt phải nằm trong khung Oval
        inside_oval = bool(
            face_res.detected
            and face_res.face_count == 1
            and not face_res.is_upside_down
            and face_res.oval_dist <= C.OVAL_INSIDE_TOL
            and abs(face_res.yaw or 0.0) <= 15.0
            and abs(face_res.pitch or 0.0) <= 18.0
        )
        if not inside_oval:
            self.consecutive_match_frames = max(0, self.consecutive_match_frames - 1)
            self.match_progress.setValue(self.consecutive_match_frames)
            if face_res.oval_dist > C.OVAL_INSIDE_TOL:
                self.lbl_status_text.setText("⚠️ Vui lòng đưa đầu và khuôn mặt vào giữa khung Oval để xác thực!")
            elif abs(face_res.yaw or 0.0) > 15.0:
                self.lbl_status_text.setText("⚠️ Vui lòng nhìn thẳng vào camera, không quay đầu!")
            else:
                self.lbl_status_text.setText("⚠️ Căn đầu vào giữa khung Oval để xác thực.")
            return

        # 3. Trích xuất ArcFace 512D và so khớp độ tương đồng
        emb = self.extractor.extract_embedding(
            frame,
            landmarks_5pts=face_res.arcface_kps,
            bbox=face_res.bbox,
        )
        if emb is None or self.target_embedding is None:
            return

        sim = float(self.extractor.compute_similarity(emb, self.target_embedding))
        sim_pct = int(sim * 100)
        self.card_similarity.set_value(f"{sim_pct} %")

        # Ngưỡng khớp khuôn mặt ArcFace: >= 55%
        if sim >= 0.55:
            self.card_similarity.set_status("Trùng khớp", True)
            self.consecutive_match_frames += 1
            self.match_progress.setValue(self.consecutive_match_frames)
            self.lbl_status_text.setText(
                f"Đang xác thực ({sim_pct}%)... Giữ yên ({self.consecutive_match_frames}/{self.match_threshold_frames})"
            )

            if self.consecutive_match_frames >= self.match_threshold_frames:
                self.verified_success = True
                self.is_verifying = False
                self.btn_toggle_verify.setText("✅ ĐÃ XÁC THỰC THÀNH CÔNG")
                self.btn_toggle_verify.setEnabled(False)
                self.btn_enter_exam.setEnabled(True)
                self.lbl_status_text.setText(
                    f"✅ ĐÃ XÁC THỰC THÀNH CÔNG THÍ SINH: {self.selected_user_info['full_name']}!\n"
                    f"Người thật 3D (Độ sâu +{liveness_res.nose_relief_mm}mm) | Độ khớp: {sim_pct}%."
                )
                play_chime_success()
        else:
            self.card_similarity.set_status("Không khớp", False)
            self.consecutive_match_frames = max(0, self.consecutive_match_frames - 1)
            self.match_progress.setValue(self.consecutive_match_frames)
            self.lbl_status_text.setText(f"⚠️ Khuôn mặt không khớp hồ sơ ({sim_pct}% < 55%). Vui lòng kiểm tra lại.")

    def _on_enter_exam_clicked(self):
        """Chuyển thí sinh vào quy trình Hiệu chuẩn Eye Tracking và Phòng thi."""
        if self.selected_user_info:
            self.enter_exam_requested.emit(self.selected_user_info)
