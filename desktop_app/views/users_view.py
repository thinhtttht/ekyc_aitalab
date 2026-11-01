"""Users Management View - Browse, inspect 512D embeddings, and manage students."""
from __future__ import annotations

import base64
import os
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import get_user_repository


class VectorDetailDialog(QDialog):
    """Hộp thoại xem chi tiết vector nhúng ArcFace 512 chiều."""

    def __init__(self, user: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Chi tiết Vector Sinh trắc học - {user.get('user_id')}")
        self.resize(600, 480)
        self.setStyleSheet("background-color: #0f172a; color: #f8fafc;")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_lbl = QLabel(f"Thí sinh: <b>{user.get('full_name')}</b> | MSSV: <b>{user.get('user_id')}</b>")
        info_lbl.setStyleSheet("font-size: 14px; color: #38bdf8;")
        layout.addWidget(info_lbl)

        dim_lbl = QLabel("Kích thước vector: 512 chiều (Chuẩn ArcFace ResNet50 L2-normalized)")
        dim_lbl.setStyleSheet("font-size: 12px; color: #94a3b8;")
        layout.addWidget(dim_lbl)

        # Trình diễn vector
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #334155; border-radius: 6px;"
        )

        emb = user.get("embedding")
        if emb is not None:
            formatted_vals = []
            for i, val in enumerate(emb):
                formatted_vals.append(f"[{i:03d}]: {val:+.6f}")
            text_edit.setText("\n".join(formatted_vals))
        else:
            text_edit.setText("Không có dữ liệu vector.")

        layout.addWidget(text_edit)

        btn_close = QPushButton("Đóng")
        btn_close.setProperty("class", "btn-secondary")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class UsersView(QWidget):
    """Giao diện quản lý danh sách hồ sơ sinh viên trong SQLite ekyc.db."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_repo = get_user_repository()
        self.all_users = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Header bar & Controls
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        title = QLabel("DANH SÁCH HỒ SƠ THÍ SINH (CSDL ekyc.db)")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #38bdf8;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Ô tìm kiếm
        self.input_search = QLineEdit()
        self.input_search.setPlaceholderText("🔍 Tìm kiếm theo MSSV hoặc Họ tên...")
        self.input_search.setFixedWidth(280)
        self.input_search.textChanged.connect(self._filter_users)
        header_layout.addWidget(self.input_search)

        self.btn_refresh = QPushButton("🔄 Tải lại")
        self.btn_refresh.setProperty("class", "btn-secondary")
        self.btn_refresh.clicked.connect(self.load_users)
        header_layout.addWidget(self.btn_refresh)

        self.btn_delete_all = QPushButton("🗑️ Xoá toàn bộ CSDL")
        self.btn_delete_all.setProperty("class", "btn-danger")
        self.btn_delete_all.clicked.connect(self._on_delete_all_clicked)
        header_layout.addWidget(self.btn_delete_all)

        layout.addLayout(header_layout)

        # Thống kê nhanh
        self.stats_bar = QLabel("Tổng số thí sinh: 0 hồ sơ | Chuẩn vector: ArcFace 512D float32")
        self.stats_bar.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 500;")
        layout.addWidget(self.stats_bar)

        # Bảng danh sách thí sinh
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Ảnh đại diện",
            "Mã sinh viên (MSSV)",
            "Họ và tên",
            "Ngày đăng ký",
            "Vector ArcFace",
            "Thao tác",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(65)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        self.load_users()

    def load_users(self):
        """Đọc toàn bộ hồ sơ từ SQLite database."""
        self.all_users = self.user_repo.list_users()
        self.stats_bar.setText(
            f"Tổng số thí sinh: {len(self.all_users)} hồ sơ đã lưu | Chuẩn vector: ArcFace 512D float32 | Database: backend/data/ekyc.db"
        )
        self._populate_table(self.all_users)

    def _filter_users(self, keyword: str):
        kw = keyword.strip().lower()
        if not kw:
            self._populate_table(self.all_users)
            return

        filtered = [
            u for u in self.all_users
            if kw in u.get("user_id", "").lower() or kw in u.get("full_name", "").lower()
        ]
        self._populate_table(filtered)

    def _populate_table(self, users: list):
        self.table.setRowCount(len(users))

        for row, u in enumerate(users):
            # 0. Thumbnail ảnh
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b64_str = u.get("snapshot_b64", "")
            if b64_str:
                try:
                    if "," in b64_str:
                        b64_str = b64_str.split(",")[1]
                    raw_bytes = base64.b64decode(b64_str)
                    pix = QPixmap()
                    pix.loadFromData(raw_bytes)
                    img_lbl.setPixmap(
                        pix.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )
                except Exception:
                    img_lbl.setText("Ảnh lỗi")
            else:
                img_lbl.setText("Không ảnh")
            self.table.setCellWidget(row, 0, img_lbl)

            # 1. MSSV
            id_item = QTableWidgetItem(u.get("user_id", ""))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setForeground(Qt.GlobalColor.cyan)
            self.table.setItem(row, 1, id_item)

            # 2. Họ và tên kèm tag Nhánh 2D/3D
            branch = u.get("metadata", {}).get("branch", "2d_webcam") if isinstance(u.get("metadata"), dict) else "2d"
            tag = " [🌐 3D]" if "3d" in branch else " [📷 2D]"
            name_item = QTableWidgetItem(f"{u.get('full_name', '')}{tag}")
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table.setItem(row, 2, name_item)

            # 3. Ngày đăng ký
            date_item = QTableWidgetItem(u.get("created_at", ""))
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, date_item)

            # 4. Preview Vector
            prev = u.get("embedding_preview", [])
            prev_str = ", ".join([f"{v:.3f}" for v in prev[:3]]) + "..."
            vec_item = QTableWidgetItem(f"512D [{prev_str}]")
            vec_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, vec_item)

            # 5. Cột Thao tác (Xem vector, Xoá)
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 4, 4, 4)
            action_layout.setSpacing(6)

            btn_view = QPushButton("Chi tiết")
            btn_view.setProperty("class", "btn-secondary")
            btn_view.clicked.connect(lambda _, uid=u["user_id"]: self._view_vector(uid))
            action_layout.addWidget(btn_view)

            btn_del = QPushButton("Xoá")
            btn_del.setProperty("class", "btn-danger")
            btn_del.clicked.connect(lambda _, uid=u["user_id"], name=u["full_name"]: self._delete_user(uid, name))
            action_layout.addWidget(btn_del)

            self.table.setCellWidget(row, 5, action_widget)

    def _view_vector(self, user_id: str):
        full_u = self.user_repo.get_user(user_id)
        if not full_u:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy hồ sơ thí sinh!")
            return
        dlg = VectorDetailDialog(full_u, self)
        dlg.exec()

    def _delete_user(self, user_id: str, full_name: str):
        confirm = QMessageBox.question(
            self,
            "Xác nhận xoá",
            f"Bạn có chắc chắn muốn xoá hồ sơ thí sinh '{full_name}' (MSSV: {user_id}) khỏi CSDL không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            ok = self.user_repo.delete_user(user_id)
            if ok:
                QMessageBox.information(self, "Đã xoá", f"Đã xoá thành công thí sinh {user_id}.")
                self.load_users()
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể xoá thí sinh khỏi SQLite!")

    def _on_delete_all_clicked(self):
        confirm = QMessageBox.warning(
            self,
            "CẢNH BÁO NGUY HIỂM",
            "Hành động này sẽ XOÁ TOÀN BỘ hồ sơ thí sinh và vector sinh trắc học trong CSDL ekyc.db!\nBạn có chắc chắn muốn tiếp tục không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            count = self.user_repo.delete_all_users()
            QMessageBox.information(self, "Đã xoá CSDL", f"Đã xoá toàn bộ {count} hồ sơ thí sinh.")
            self.load_users()
