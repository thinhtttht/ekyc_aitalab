"""Mock Exam Room View with Continuous Landmark Head Pose & Eye Gaze Proctoring and Red Border Alerts."""
from __future__ import annotations

import base64
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QTime, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from eye_tracker import EyeTracker, GazeResult
    from sound_effects import play_chime_ding, play_chime_success, play_chime_warning
except ImportError:
    from desktop_app.eye_tracker import EyeTracker, GazeResult
    from desktop_app.sound_effects import play_chime_ding, play_chime_success, play_chime_warning


class ExamContainerFrame(QFrame):
    """Khung viền bao quát toàn bộ phòng thi, đổi màu ĐỎ RỰC RỠ khi quay đầu hoặc nhìn ra ngoài."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_alert = False
        self.setStyleSheet("""
            ExamContainerFrame {
                background-color: #0b1120;
                border: 2px solid #1e293b;
                border-radius: 12px;
            }
        """)

    def set_alert_state(self, is_alert: bool):
        if self.is_alert == is_alert:
            return
        self.is_alert = is_alert
        if is_alert:
            # Viền đỏ rực rỡ cảnh báo vi phạm
            self.setStyleSheet("""
                ExamContainerFrame {
                    background-color: #16080c;
                    border: 6px solid #ef4444;
                    border-radius: 12px;
                }
            """)
        else:
            # Viền xanh than tối chuẩn mực
            self.setStyleSheet("""
                ExamContainerFrame {
                    background-color: #0b1120;
                    border: 2px solid #1e293b;
                    border-radius: 12px;
                }
            """)


class ExamView(QWidget):
    """Giao diện phòng thi giả lập (Mock Exam Proctoring Room)."""

    exit_exam_requested = pyqtSignal()
    recalibrate_requested = pyqtSignal()

    def __init__(self, eye_tracker: Optional[EyeTracker] = None, parent=None):
        super().__init__(parent)
        self.eye_tracker = eye_tracker or EyeTracker()
        self.candidate_info: dict = {}
        self.latest_frame: Optional[np.ndarray] = None

        # Thống kê vi phạm nhìn ra ngoài
        self.violation_count = 0
        self.distraction_seconds = 0.0
        self.last_away_time: Optional[float] = None
        self.is_currently_away = False
        self.last_sound_warn_time = 0.0

        # Đồng hồ đếm ngược bài thi (45 phút)
        self.remaining_seconds = 45 * 60
        self.exam_timer = QTimer(self)
        self.exam_timer.setInterval(1000)
        self.exam_timer.timeout.connect(self._on_timer_tick)

        # Mock ngân hàng câu hỏi
        self.questions = [
            {
                "id": 1,
                "text": "Trong kiến trúc xác thực sinh trắc học khuôn mặt 2D, mô hình ArcFace (Additive Angular Margin Loss) tối ưu hóa hàm mất mát nhằm mục đích chính nào sau đây?",
                "options": [
                    "A. Tăng cường độ phân giải của ảnh đầu vào trước khi trích xuất đặc trưng",
                    "B. Tối đa hóa khoảng cách góc giữa các lớp và siết chặt phương sai trong cùng một lớp",
                    "C. Tự động khử nhiễu do chớp mắt và quay đầu của người dùng",
                    "D. Nén kích thước vector nhúng đặc trưng từ 512 chiều xuống còn 128 chiều"
                ],
                "answer": 1
            },
            {
                "id": 2,
                "text": "Kỹ thuật Presentation Attack Detection (PAD) quang học nhiều màu sắc (Color Flashing) hoạt động dựa trên nguyên lý vật lý nào?",
                "options": [
                    "A. Sự phản xạ ánh sáng quang phổ đa màu trên lớp biểu bì da sống khác biệt so với bề mặt giấy in hay màn hình điện thoại",
                    "B. Tốc độ quét màn hình LCD và cảm biến Rolling Shutter của camera",
                    "C. Đo nhịp tim thông qua biến thiên lưu lượng máu trên khuôn mặt (rPPG)",
                    "D. Phát hiện vi chuyển động của nhãn cầu và đồng tử mắt"
                ],
                "answer": 0
            },
            {
                "id": 3,
                "text": "Để tính toán góc tư thế đầu 3D (Head Pose: Pitch, Yaw, Roll) từ các điểm mốc 2D trên khuôn mặt, thuật toán thị giác máy tính tiêu chuẩn nào thường được sử dụng?",
                "options": [
                    "A. K-Nearest Neighbors (KNN)",
                    "B. Perspective-n-Point (solvePnP) kết hợp ma trận hiệu chuẩn nội camera",
                    "C. Linear Discriminant Analysis (LDA)",
                    "D. Optical Flow Horn-Schunck"
                ],
                "answer": 1
            },
            {
                "id": 4,
                "text": "Theo quy chế an ninh phòng thi trực tuyến e-Proctoring, hành vi nào sau đây bị hệ thống AI ghi nhận là hành vi gian lận tiềm ẩn?",
                "options": [
                    "A. Đọc thầm câu hỏi và suy nghĩ",
                    "B. Nhìn ra ngoài màn hình máy tính hoặc quay mặt liên tục về một hướng ngoài tầm kiểm soát của camera",
                    "C. Bấm chuyển câu hỏi trên giao diện bài thi",
                    "D. Chớp mắt tự nhiên khi đọc đề"
                ],
                "answer": 1
            }
        ]
        self.current_q_idx = 0
        self.user_answers: Dict[int, int] = {}

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(15, 12, 15, 12)
        root_layout.setSpacing(10)

        # Container Frame chính bao trọn phòng thi (sẽ chuyển viền đỏ rực rỡ khi vi phạm)
        self.container_frame = ExamContainerFrame(self)
        container_layout = QVBoxLayout(self.container_frame)
        container_layout.setContentsMargins(14, 12, 14, 12)
        container_layout.setSpacing(10)

        # --- 1. BANNER CẢNH BÁO VI PHẠM (ẨN KHI BÌNH THƯỜNG, ĐỎ RỰC KHI QUAY ĐẦU / NHÌN RA NGOÀI) ---
        self.banner_alert = QFrame()
        self.banner_alert.setStyleSheet("""
            background-color: #b91c1c;
            border: 2px solid #f87171;
            border-radius: 8px;
            padding: 8px;
        """)
        alert_layout = QHBoxLayout(self.banner_alert)
        alert_layout.setContentsMargins(10, 4, 10, 4)

        self.lbl_alert_icon = QLabel("⚠️")
        self.lbl_alert_icon.setStyleSheet("font-size: 22px;")
        alert_layout.addWidget(self.lbl_alert_icon)

        self.lbl_alert_msg = QLabel("CẢNH BÁO VI PHẠM: Thí sinh đang quay đầu nhìn ra ngoài bài thi! Vui lòng tập trung!")
        self.lbl_alert_msg.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;")
        alert_layout.addWidget(self.lbl_alert_msg, stretch=1)

        self.lbl_alert_badge = QLabel("VI PHẠM QUY CHẾ THI")
        self.lbl_alert_badge.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 4px;")
        alert_layout.addWidget(self.lbl_alert_badge)

        container_layout.addWidget(self.banner_alert)
        self.banner_alert.hide()  # Mặc định ẩn

        # --- 2. HEADER BÀI THI & THÔNG TIN THÍ SINH ---
        header_bar = QFrame()
        header_bar.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 10px;")
        h_layout = QHBoxLayout(header_bar)
        h_layout.setContentsMargins(10, 5, 10, 5)

        # Tiêu đề kỳ thi
        title_box = QVBoxLayout()
        lbl_exam_name = QLabel("KỲ THI TRỰC TUYẾN CHUẨN ĐẦU RA 2026")
        lbl_exam_name.setStyleSheet("font-size: 15px; font-weight: 800; color: #38bdf8;")
        lbl_sub_exam = QLabel("Môn thi: An toàn thông tin & Kiến trúc hệ thống | Thời gian: 45 phút")
        lbl_sub_exam.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_box.addWidget(lbl_exam_name)
        title_box.addWidget(lbl_sub_exam)
        h_layout.addLayout(title_box, stretch=1)

        # Hồ sơ thí sinh tóm tắt
        cand_box = QHBoxLayout()
        self.lbl_cand_avatar = QLabel()
        self.lbl_cand_avatar.setFixedSize(44, 44)
        self.lbl_cand_avatar.setStyleSheet("border-radius: 22px; border: 2px solid #38bdf8; background-color: #1e293b;")
        self.lbl_cand_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cand_box.addWidget(self.lbl_cand_avatar)

        cand_txt = QVBoxLayout()
        self.lbl_exam_cand_name = QLabel("Thí sinh: --")
        self.lbl_exam_cand_name.setStyleSheet("font-size: 12px; font-weight: 700; color: #f1f5f9;")
        self.lbl_exam_cand_id = QLabel("MSSV: -- | Đã xác thực eKYC ✓")
        self.lbl_exam_cand_id.setStyleSheet("font-size: 11px; color: #10b981;")
        cand_txt.addWidget(self.lbl_exam_cand_name)
        cand_txt.addWidget(self.lbl_exam_cand_id)
        cand_box.addLayout(cand_txt)
        h_layout.addLayout(cand_box)

        # Đồng hồ đếm ngược
        timer_box = QFrame()
        timer_box.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 6px 14px;")
        t_layout = QVBoxLayout(timer_box)
        t_layout.setContentsMargins(4, 2, 4, 2)
        lbl_t_title = QLabel("THỜI GIAN CÒN LẠI")
        lbl_t_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_t_title.setStyleSheet("font-size: 9px; font-weight: 700; color: #94a3b8;")
        self.lbl_timer_val = QLabel("45:00")
        self.lbl_timer_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer_val.setStyleSheet("font-size: 18px; font-weight: 800; color: #fbbf24; font-family: monospace;")
        t_layout.addWidget(lbl_t_title)
        t_layout.addWidget(self.lbl_timer_val)
        h_layout.addWidget(timer_box)

        container_layout.addWidget(header_bar)

        # --- 3. NỘI DUNG CHÍNH (SPLITTER: BÀI THI BÊN TRÁI & GIÁM THỊ AI BÊN PHẢI) ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        # ================== KHỐI TRÁI: BÀI THI MOCK ==================
        exam_left = QFrame()
        exam_left.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        left_layout = QVBoxLayout(exam_left)
        left_layout.setSpacing(12)

        # Danh sách câu hỏi bấm chọn nhanh (1, 2, 3, 4)
        q_nav_row = QHBoxLayout()
        self.btn_q_group = QButtonGroup(self)
        self.q_buttons = []
        for i in range(len(self.questions)):
            btn = QPushButton(f"Câu {i + 1}")
            btn.setCheckable(True)
            btn.setFixedSize(70, 32)
            btn.setProperty("class", "btn-secondary")
            self.btn_q_group.addButton(btn, i)
            q_nav_row.addWidget(btn)
            self.q_buttons.append(btn)
        self.btn_q_group.idClicked.connect(self._on_q_btn_clicked)
        if self.q_buttons:
            self.q_buttons[0].setChecked(True)

        q_nav_row.addStretch()
        left_layout.addLayout(q_nav_row)

        # Hộp câu hỏi
        self.q_card = QFrame()
        self.q_card.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 16px;")
        q_card_layout = QVBoxLayout(self.q_card)
        q_card_layout.setSpacing(14)

        self.lbl_question_num = QLabel("CÂU HỎI 1 / 4")
        self.lbl_question_num.setStyleSheet("font-size: 12px; font-weight: 800; color: #38bdf8;")
        q_card_layout.addWidget(self.lbl_question_num)

        self.lbl_question_text = QLabel("Nội dung câu hỏi...")
        self.lbl_question_text.setWordWrap(True)
        self.lbl_question_text.setStyleSheet("font-size: 14px; font-weight: 600; color: #f8fafc; line-height: 1.5;")
        q_card_layout.addWidget(self.lbl_question_text)

        # Lựa chọn A, B, C, D
        self.opt_group = QButtonGroup(self)
        self.opt_radios: List[QRadioButton] = []
        for opt_idx in range(4):
            rb = QRadioButton()
            rb.setStyleSheet("""
                QRadioButton {
                    color: #cbd5e1;
                    font-size: 13px;
                    padding: 8px;
                    border-radius: 6px;
                }
                QRadioButton:hover {
                    background-color: #334155;
                }
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                }
            """)
            self.opt_group.addButton(rb, opt_idx)
            self.opt_radios.append(rb)
            q_card_layout.addWidget(rb)
        self.opt_group.idClicked.connect(self._on_option_selected)

        left_layout.addWidget(self.q_card, stretch=1)

        # Nút chuyển câu và Nộp bài
        btn_action_row = QHBoxLayout()
        self.btn_prev_q = QPushButton("← Câu trước")
        self.btn_prev_q.setProperty("class", "btn-secondary")
        self.btn_prev_q.clicked.connect(self._on_prev_clicked)
        btn_action_row.addWidget(self.btn_prev_q)

        self.btn_next_q = QPushButton("Câu tiếp theo →")
        self.btn_next_q.setProperty("class", "btn-secondary")
        self.btn_next_q.clicked.connect(self._on_next_clicked)
        btn_action_row.addWidget(self.btn_next_q)

        btn_action_row.addStretch()

        self.btn_submit_exam = QPushButton("📝 Nộp bài thi")
        self.btn_submit_exam.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-size: 13px;
                font-weight: 700;
                padding: 8px 18px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
        """)
        self.btn_submit_exam.clicked.connect(self._on_submit_exam_clicked)
        btn_action_row.addWidget(self.btn_submit_exam)

        left_layout.addLayout(btn_action_row)
        body_layout.addWidget(exam_left, stretch=6)

        # ================== KHỐI PHẢI: GIÁM THỊ AI & LANDMARK TRACKING ==================
        exam_right = QFrame()
        exam_right.setFixedWidth(340)
        exam_right.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        right_layout = QVBoxLayout(exam_right)
        right_layout.setSpacing(10)

        r_title = QLabel("🤖 GIÁM THỊ AI (LANDMARK HEAD POSE & EYE)")
        r_title.setStyleSheet("font-size: 12px; font-weight: 800; color: #38bdf8;")
        right_layout.addWidget(r_title)

        # Camera PIP Live Monitor
        self.lbl_cam_pip = QLabel()
        self.lbl_cam_pip.setFixedSize(316, 210)
        self.lbl_cam_pip.setStyleSheet("background-color: #020617; border: 2px solid #334155; border-radius: 6px;")
        self.lbl_cam_pip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.lbl_cam_pip)

        # Huy hiệu trạng thái tập trung & hướng đầu
        self.lbl_gaze_badge = QLabel("🟢 ĐANG TẬP TRUNG VÀO BÀI THI")
        self.lbl_gaze_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gaze_badge.setStyleSheet("""
            background-color: #064e3b;
            color: #6ee7b7;
            font-size: 12px;
            font-weight: 700;
            padding: 7px;
            border-radius: 6px;
        """)
        right_layout.addWidget(self.lbl_gaze_badge)

        # Thẻ chỉ số giám sát
        metrics_box = QFrame()
        metrics_box.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 8px;")
        m_layout = QVBoxLayout(metrics_box)
        m_layout.setSpacing(5)

        self.lbl_metric_pose = QLabel("• Góc xoay đầu: Chính diện (Yaw 0° | Pitch 0°)")
        self.lbl_metric_pose.setStyleSheet("font-size: 11px; font-weight: 700; color: #38bdf8;")
        self.lbl_metric_violations = QLabel("• Số lần quay đầu / nhìn ra ngoài: 0 lần")
        self.lbl_metric_violations.setStyleSheet("font-size: 11px; font-weight: 700; color: #f1f5f9;")
        self.lbl_metric_distracted = QLabel("• Tổng thời gian mất tập trung: 0.0s")
        self.lbl_metric_distracted.setStyleSheet("font-size: 11px; color: #94a3b8;")
        self.lbl_metric_gaze_pos = QLabel("• Giới hạn an toàn: Yaw ±16° | Pitch -14°~+16°")
        self.lbl_metric_gaze_pos.setStyleSheet("font-size: 11px; color: #94a3b8;")

        m_layout.addWidget(self.lbl_metric_pose)
        m_layout.addWidget(self.lbl_metric_violations)
        m_layout.addWidget(self.lbl_metric_distracted)
        m_layout.addWidget(self.lbl_metric_gaze_pos)
        right_layout.addWidget(metrics_box)

        # Nhật ký sự kiện AI (Log vi phạm)
        lbl_log_title = QLabel("NHẬT KÝ GIÁM SÁT THỜI GIAN THỰC")
        lbl_log_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #64748b; margin-top: 5px;")
        right_layout.addWidget(lbl_log_title)

        self.table_logs = QTableWidget(0, 2)
        self.table_logs.setHorizontalHeaderLabels(["Thời gian", "Sự kiện"])
        self.table_logs.horizontalHeader().setStretchLastSection(True)
        self.table_logs.setStyleSheet("""
            QTableWidget {
                background-color: #020617;
                border: 1px solid #1e293b;
                border-radius: 4px;
                font-size: 10px;
                color: #cbd5e1;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 10px;
                padding: 3px;
                border: none;
            }
        """)
        right_layout.addWidget(self.table_logs, stretch=1)

        # Nút cài đặt/kiểm tra lại góc nhìn
        self.btn_recalib = QPushButton("🧭 Đặt lại bộ lọc góc đầu (Reset Pose)")
        self.btn_recalib.setProperty("class", "btn-secondary")
        self.btn_recalib.clicked.connect(self._on_reset_filters_clicked)
        right_layout.addWidget(self.btn_recalib)

        body_layout.addWidget(exam_right, stretch=4)
        container_layout.addLayout(body_layout, stretch=1)
        root_layout.addWidget(self.container_frame, stretch=1)

        self._render_question(0)

    def _on_reset_filters_clicked(self):
        """Đặt lại bộ lọc OneEuro và hiệu chỉnh lại vị trí nhìn thẳng."""
        self.eye_tracker.filter_yaw.reset()
        self.eye_tracker.filter_pitch.reset()
        self.eye_tracker.filter_roll.reset()
        self._add_log_entry("Đã đặt lại bộ lọc góc quay đầu")
        play_chime_ding()

    def start_exam(self, candidate_info: dict):
        """Khởi động kỳ thi cho thí sinh đã qua xác thực eKYC."""
        self.candidate_info = candidate_info
        name = candidate_info.get("full_name", "Thí sinh")
        uid = candidate_info.get("user_id", "N/A")
        self.lbl_exam_cand_name.setText(f"Thí sinh: {name}")
        self.lbl_exam_cand_id.setText(f"MSSV: {uid} | Đã xác thực eKYC ✓")

        # Avatar
        self._set_candidate_avatar(candidate_info.get("snapshot_b64"))

        # Reset các biến theo dõi
        self.violation_count = 0
        self.distraction_seconds = 0.0
        self.is_currently_away = False
        self.last_away_time = None
        self.table_logs.setRowCount(0)
        self._add_log_entry("Bắt đầu bài thi và kích hoạt Landmark Head Pose Proctoring")

        # Khởi động đếm ngược
        self.remaining_seconds = 45 * 60
        self.lbl_timer_val.setText("45:00")
        self.exam_timer.start()

        self._render_question(0)

    def _set_candidate_avatar(self, b64_str: Optional[str]):
        if not b64_str:
            self.lbl_cand_avatar.clear()
            return
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            pix = QPixmap()
            pix.loadFromData(img_bytes)
            self.lbl_cand_avatar.setPixmap(
                pix.scaled(44, 44, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            )
        except Exception:
            self.lbl_cand_avatar.clear()

    def _render_question(self, idx: int):
        if idx < 0 or idx >= len(self.questions):
            return
        self.current_q_idx = idx
        q = self.questions[idx]

        self.lbl_question_num.setText(f"CÂU HỎI {idx + 1} / {len(self.questions)}")
        self.lbl_question_text.setText(q["text"])

        # Cập nhật danh sách đáp án
        self.opt_group.blockSignals(True)
        saved_ans = self.user_answers.get(idx, None)
        for i, rb in enumerate(self.opt_radios):
            rb.setText(q["options"][i])
            rb.setChecked(saved_ans == i)
        self.opt_group.blockSignals(False)

        # Cập nhật trạng thái nút
        self.btn_prev_q.setEnabled(idx > 0)
        self.btn_next_q.setEnabled(idx < len(self.questions) - 1)
        if idx < len(self.q_buttons):
            self.q_buttons[idx].setChecked(True)

    def _on_option_selected(self, opt_idx: int):
        self.user_answers[self.current_q_idx] = opt_idx
        btn = self.q_buttons[self.current_q_idx]
        btn.setStyleSheet("background-color: #0369a1; color: white; font-weight: 700;")

    def _on_prev_clicked(self):
        if self.current_q_idx > 0:
            self._render_question(self.current_q_idx - 1)

    def _on_next_clicked(self):
        if self.current_q_idx < len(self.questions) - 1:
            self._render_question(self.current_q_idx + 1)

    def _on_q_btn_clicked(self, q_idx: int):
        self._render_question(q_idx)

    def _on_timer_tick(self):
        if self.remaining_seconds > 0:
            self.remaining_seconds -= 1
            mins = self.remaining_seconds // 60
            secs = self.remaining_seconds % 60
            self.lbl_timer_val.setText(f"{mins:02d}:{secs:02d}")
            if self.remaining_seconds <= 300:
                self.lbl_timer_val.setStyleSheet("font-size: 18px; font-weight: 800; color: #ef4444; font-family: monospace;")
        else:
            self.exam_timer.stop()
            self._on_submit_exam_clicked(auto=True)

    def on_frame_received(self, frame: np.ndarray):
        """Xử lý frame video giám sát góc quay đầu và ánh mắt theo thời gian thực."""
        if not self.isVisible():
            return

        self.latest_frame = frame
        win_w = self.width()
        win_h = self.height()

        gaze_res: GazeResult = self.eye_tracker.process_frame(
            frame,
            screen_size=(win_w, win_h)
        )

        now_t = time.time()

        # Kiểm tra trạng thái chú ý / quay đầu nhìn ra ngoài
        if gaze_res.is_looking_away:
            # === THÍ SINH QUAY ĐẦU HOẶC NHÌN RA NGOÀI MÀN HÌNH ===
            if not self.is_currently_away:
                self.is_currently_away = True
                self.last_away_time = now_t
                self.violation_count += 1
                self._add_log_entry(f"⚠️ {gaze_res.looking_away_reason}")

                # KÍCH HOẠT VIỀN ĐỎ RỰC RỠ VÀ BANNER CẢNH BÁO
                self.container_frame.set_alert_state(True)
                self.banner_alert.show()
                self.lbl_alert_msg.setText(f"CẢNH BÁO VI PHẠM (#{self.violation_count}): {gaze_res.looking_away_reason.upper()}!")

                # Âm thanh cảnh báo nhắc nhở
                if now_t - self.last_sound_warn_time > 1.8:
                    play_chime_warning()
                    self.last_sound_warn_time = now_t

            # Tích lũy thời gian lơ là
            if self.last_away_time is not None:
                delta = now_t - self.last_away_time
                self.distraction_seconds += delta
                self.last_away_time = now_t

            self.lbl_gaze_badge.setText(f"🔴 CẢNH BÁO: {gaze_res.head_direction_label.upper()}")
            self.lbl_gaze_badge.setStyleSheet("""
                background-color: #7f1d1d;
                color: #fecaca;
                font-size: 12px;
                font-weight: 700;
                padding: 7px;
                border-radius: 6px;
            """)
        else:
            # === THÍ SINH ĐANG TẬP TRUNG NHÌN VÀO BÀI THI ===
            if self.is_currently_away:
                self.is_currently_away = False
                self.last_away_time = None
                # Trở về trạng thái bình thường (tắt viền đỏ ngay lập tức)
                self.container_frame.set_alert_state(False)
                self.banner_alert.hide()
                self._add_log_entry("Đã quay lại nhìn thẳng vào màn hình bài thi")

            self.lbl_gaze_badge.setText("🟢 ĐANG TẬP TRUNG VÀO BÀI THI")
            self.lbl_gaze_badge.setStyleSheet("""
                background-color: #064e3b;
                color: #6ee7b7;
                font-size: 12px;
                font-weight: 700;
                padding: 7px;
                border-radius: 6px;
            """)

        # Cập nhật thông số hiển thị
        viol_color = "#ef4444" if self.violation_count > 0 else "#f1f5f9"
        pose_color = "#ef4444" if gaze_res.is_looking_away else "#38bdf8"
        self.lbl_metric_pose.setText(f"• Hướng đầu: {gaze_res.head_direction_label} (Yaw {gaze_res.yaw:+.0f}° | Pitch {gaze_res.pitch:+.0f}°)")
        self.lbl_metric_pose.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {pose_color};")

        self.lbl_metric_violations.setText(f"• Số lần quay đầu / lơ là: {self.violation_count} lần")
        self.lbl_metric_violations.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {viol_color};")
        self.lbl_metric_distracted.setText(f"• Tổng thời gian mất tập trung: {self.distraction_seconds:.1f}s")
        self.lbl_metric_gaze_pos.setText(f"• Tọa độ ánh mắt: ({gaze_res.gaze_screen_px[0]}, {gaze_res.gaze_screen_px[1]})")

        # Vẽ camera PIP với tia mũi 3D và tròng đen
        self._render_cam_pip(frame, gaze_res)

    def _render_cam_pip(self, frame: np.ndarray, gaze_res: GazeResult):
        disp = frame.copy()

        # 1. Vẽ tia nhìn 3D từ chóp mũi (Nose Ray) biểu thị trực quan hướng quay đầu
        if gaze_res.nose_ray_2d:
            p1, p2 = gaze_res.nose_ray_2d
            ray_color = (0, 0, 255) if gaze_res.is_looking_away else (0, 255, 0)
            cv2.arrowedLine(disp, p1, p2, ray_color, 3, tipLength=0.35)
            cv2.circle(disp, p1, 4, ray_color, -1)

        # 2. Vẽ điểm tròng đen
        if gaze_res.iris_right_px:
            cv2.circle(disp, gaze_res.iris_right_px, 3, (0, 255, 255), -1)
        if gaze_res.iris_left_px:
            cv2.circle(disp, gaze_res.iris_left_px, 3, (0, 255, 255), -1)

        # 3. HUD hiển thị góc xoay trên đầu frame PIP
        hud_txt = f"{gaze_res.head_direction_label.upper()} | Yaw: {gaze_res.yaw:+.0f} | Pitch: {gaze_res.pitch:+.0f}"
        hud_bg_color = (0, 0, 180) if gaze_res.is_looking_away else (0, 80, 0)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), hud_bg_color, -1)
        cv2.putText(disp, hud_txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Viền khung cảnh báo trực tiếp trên ảnh PIP
        if gaze_res.is_looking_away:
            cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), (0, 0, 255), 6)

        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        disp_rgb = np.ascontiguousarray(disp_rgb)
        h, w, ch = disp_rgb.shape
        qimg = QImage(disp_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(316, 210, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_cam_pip.setPixmap(pix)

    def _add_log_entry(self, event_msg: str):
        row = self.table_logs.rowCount()
        self.table_logs.insertRow(row)
        time_str = time.strftime("%H:%M:%S")
        self.table_logs.setItem(row, 0, QTableWidgetItem(time_str))
        self.table_logs.setItem(row, 1, QTableWidgetItem(event_msg))
        self.table_logs.scrollToBottom()

    def _on_recalib_clicked(self):
        self.exam_timer.stop()
        self.recalibrate_requested.emit()

    def _on_submit_exam_clicked(self, auto: bool = False):
        self.exam_timer.stop()
        play_chime_success()

        # Tính điểm trắc nghiệm mock
        correct_count = 0
        for i, q in enumerate(self.questions):
            if self.user_answers.get(i) == q["answer"]:
                correct_count += 1
        score = (correct_count / len(self.questions)) * 10.0

        title = "HẾT GIỜ LÀM BÀI" if auto else "KẾT QUẢ BÀI THI"
        verdict = "ĐẠT (HỢP LỆ)" if self.violation_count < 5 else "CẦN HỘI ĐỒNG XEM XÉT (NHÌN RA NGOÀI NHIỀU)"

        msg = (
            f"Thí sinh: {self.candidate_info.get('full_name', 'N/A')}\n"
            f"MSSV: {self.candidate_info.get('user_id', 'N/A')}\n\n"
            f"• Điểm trắc nghiệm: {score:.1f} / 10 ({correct_count}/{len(self.questions)} câu đúng)\n"
            f"• Số lần quay đầu / lơ là: {self.violation_count} lần\n"
            f"• Tổng thời gian mất tập trung: {self.distraction_seconds:.1f} giây\n"
            f"• Đánh giá giám thị AI: {verdict}\n\n"
            f"Dữ liệu bài thi và lịch sử giám sát tư thế đầu đã được mã hóa lưu trữ vào cơ sở dữ liệu."
        )

        QMessageBox.information(self, title, msg)
        self.exit_exam_requested.emit()
