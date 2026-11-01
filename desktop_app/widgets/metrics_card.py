"""Reusable metrics card for displaying FQA, Anti-Spoofing values, or single telemetry metric tiles."""
from __future__ import annotations

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class MetricsCard(QFrame):
    """Card widget showing technical metrics of face analysis (multi-row) or a single telemetry tile."""

    def __init__(
        self,
        title: str = "",
        default_val: str = "",
        subtitle: str = "",
        parent=None,
    ):
        # Hỗ trợ trường hợp truyền parent làm tham số đầu tiên
        if isinstance(title, QWidget) and parent is None:
            parent = title
            title = ""

        super().__init__(parent)
        self.setProperty("class", "card")
        self.setObjectName("metricsCard")

        self._is_tile_mode = bool(title)
        if self._is_tile_mode:
            self._init_tile_ui(title, default_val, subtitle)
        else:
            self._init_multi_row_ui()

    def _init_tile_ui(self, title: str, default_val: str, subtitle: str):
        """Khởi tạo thẻ chỉ số đơn (Tile) dùng cho Telemetry 3D RealSense."""
        self.setStyleSheet(
            "MetricsCard { background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase;")
        layout.addWidget(self.lbl_title)

        self.lbl_value = QLabel(default_val)
        self.lbl_value.setStyleSheet("font-size: 15px; font-weight: 800; color: #38bdf8;")
        layout.addWidget(self.lbl_value)

        self.lbl_status = QLabel(subtitle)
        self.lbl_status.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 500;")
        layout.addWidget(self.lbl_status)

    def set_value(self, val: str):
        """Cập nhật giá trị hiển thị trên thẻ tile."""
        if hasattr(self, "lbl_value"):
            self.lbl_value.setText(val)

    def set_status(self, text: str, is_ok: bool = True):
        """Cập nhật trạng thái và màu sắc (xanh nếu đạt, đỏ nếu cảnh báo)."""
        if hasattr(self, "lbl_status"):
            color = "#34d399" if is_ok else "#f87171"
            self.lbl_status.setText(text)
            self.lbl_status.setStyleSheet(f"font-size: 10px; color: {color}; font-weight: 600;")

    def _init_multi_row_ui(self):
        """Khởi tạo thẻ đa chỉ số FQA và MiniFASNet."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("CHỈ SỐ SINH TRẮC HỌC (FQA & LIVENESS)")
        title.setStyleSheet("font-size: 11px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        # Labels
        self.lbl_fps = self._create_row(grid, 0, "FPS thực tế:", "-- FPS")
        self.lbl_brightness = self._create_row(grid, 1, "Độ sáng:", "--")
        self.lbl_sharpness = self._create_row(grid, 2, "Độ nét:", "--")
        self.lbl_pose = self._create_row(grid, 3, "Góc quay (Yaw):", "--°")
        self.lbl_pad = self._create_row(grid, 4, "MiniFASNet PAD:", "--%")

        layout.addLayout(grid)

    def _create_row(self, grid: QGridLayout, row: int, label_text: str, default_val: str) -> QLabel:
        title_lbl = QLabel(label_text)
        title_lbl.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 500;")
        grid.addWidget(title_lbl, row, 0)

        val_lbl = QLabel(default_val)
        val_lbl.setStyleSheet("color: #f8fafc; font-size: 12px; font-weight: 600;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(val_lbl, row, 1)
        return val_lbl

    def update_metrics(
        self,
        fps: float = 0.0,
        brightness: float = 0.0,
        sharpness: float = 0.0,
        yaw: float = 0.0,
        pad_prob: float = 0.0,
        is_real: bool = True,
    ):
        """Cập nhật các chỉ số FQA thời gian thực."""
        if hasattr(self, "lbl_fps"):
            self.lbl_fps.setText(f"{fps:.1f} FPS")

        if hasattr(self, "lbl_brightness"):
            b_color = "#10b981" if 60 <= brightness <= 180 else "#f59e0b"
            self.lbl_brightness.setText(f"{brightness:.1f}")
            self.lbl_brightness.setStyleSheet(f"color: {b_color}; font-size: 12px; font-weight: 600;")

        if hasattr(self, "lbl_sharpness"):
            s_color = "#10b981" if sharpness >= 15.0 else "#f59e0b"
            self.lbl_sharpness.setText(f"{sharpness:.1f}")
            self.lbl_sharpness.setStyleSheet(f"color: {s_color}; font-size: 12px; font-weight: 600;")

        if hasattr(self, "lbl_pose"):
            self.lbl_pose.setText(f"{yaw:+.1f}°")

        if hasattr(self, "lbl_pad"):
            pad_pct = pad_prob * 100.0
            p_color = "#10b981" if is_real else "#ef4444"
            verdict = "Real" if is_real else "Spoof"
            self.lbl_pad.setText(f"{verdict} ({pad_pct:.1f}%)")
            self.lbl_pad.setStyleSheet(f"color: {p_color}; font-size: 12px; font-weight: 600;")


# Alias
MetricTile = MetricsCard
