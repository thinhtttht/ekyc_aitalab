"""RealSense 3D & 2D Camera Test View for eKYC System.
Provides dual live streams:
  - 2D RGB Color Stream (Auto-exposure & White-balance enabled for bright clarity)
  - 3D Depth Map Stream (Real-time aligned depth colormap with selectable palettes)
  - Millimeter-accurate distance measurement at center crosshair and interactive mouse clicks
  - Hardware device telemetry (USB 3.x vs 2.x, Firmware, Serial, FPS)
  - Snapshots capturing 2D RGB, 3D Depth, and combined side-by-side images
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QPoint, QRectF, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Thử import pyrealsense2
try:
    import pyrealsense2 as rs
    HAS_REALSENSE_LIB = True
except ImportError:
    rs = None
    HAS_REALSENSE_LIB = False

# Sound effects
try:
    from sound_effects import play_chime_ding, play_chime_step, play_chime_success, play_chime_warning
except ImportError:
    def play_chime_ding(): pass
    def play_chime_step(): pass
    def play_chime_success(): pass
    def play_chime_warning(): pass


# Bảng màu Colorizer chuẩn của Intel RealSense
DEPTH_COLOR_SCHEMES = [
    ("Jet (Xanh gần -> Đỏ xa)", 0),
    ("Classic (Cầu vồng chuẩn)", 1),
    ("White to Black", 2),
    ("Black to White", 3),
    ("Bio (Y sinh)", 4),
    ("Cold (Màu lạnh)", 5),
    ("Warm (Màu ấm)", 6),
]


class RealSenseThread(QThread):
    """Worker thread quản lý vòng lặp thu thập và đồng bộ luồng 2D RGB & 3D Depth."""

    frames_ready = pyqtSignal(np.ndarray, np.ndarray, float, float)  # color_bgr, depth_colormap, dist_center, dist_target
    device_info_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str, bool)  # message, is_ok
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps

        self._running = False
        self._mutex = QMutex()

        self.color_scheme_idx = 0
        self.enable_filters = True
        self.target_point: Optional[Tuple[int, int]] = None  # (x, y) trên ảnh 640x480
        self.demo_mode = False

    def set_target_point(self, pt: Optional[Tuple[int, int]]):
        """Đặt tọa độ điểm cần đo khoảng cách (theo tọa độ 640x480)."""
        with QMutexLocker(self._mutex):
            self.target_point = pt

    def set_color_scheme(self, idx: int):
        """Đổi bảng màu depth map."""
        with QMutexLocker(self._mutex):
            self.color_scheme_idx = idx

    def set_filters_enabled(self, enabled: bool):
        """Bật/tắt bộ lọc làm mịn độ sâu."""
        with QMutexLocker(self._mutex):
            self.enable_filters = enabled

    def set_demo_mode(self, enabled: bool):
        """Bật chế độ mô phỏng khi không có phần cứng."""
        with QMutexLocker(self._mutex):
            self.demo_mode = enabled

    def stop(self):
        """Dừng luồng an toàn."""
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2500)

    def run(self):
        with QMutexLocker(self._mutex):
            self._running = True
            is_demo = self.demo_mode

        if is_demo:
            self._run_demo_loop()
            return

        if not HAS_REALSENSE_LIB:
            self.status_changed.emit("Thư viện pyrealsense2 chưa được cài đặt", False)
            self.error_occurred.emit("Không tìm thấy pyrealsense2. Vui lòng chạy pip install pyrealsense2.")
            return

        # Khởi tạo kết nối phần cứng RealSense
        ctx = rs.context()
        devices = ctx.devices
        if len(devices) == 0:
            self.status_changed.emit("Không tìm thấy camera Intel RealSense", False)
            self.error_occurred.emit(
                "Chưa phát hiện thiết bị Intel RealSense.\n"
                "Vui lòng cắm cáp USB 3.0 (chân màu xanh) vào máy tính và bấm 'Quét lại thiết bị'."
            )
            return

        device = devices[0]
        dev_info = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
        }
        self.device_info_ready.emit(dev_info)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        try:
            profile = pipeline.start(config)
        except Exception as e:
            self.status_changed.emit(f"Không thể khởi động camera: {e}", False)
            self.error_occurred.emit(
                f"Lỗi mở luồng RealSense: {e}\n"
                "Gợi ý: Nếu phần mềm 'RealSense Viewer' đang mở, hãy tắt đi vì camera chỉ cho phép 1 ứng dụng truy cập."
            )
            return

        # Bật Auto-Exposure và Auto-White-Balance cho cảm biến màu RGB
        try:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception as err:
            print("Auto-Exposure exception:", err)

        # Căn chỉnh Depth khớp 1:1 với Color
        align = rs.align(rs.stream.color)

        # Bộ lọc làm mịn
        spatial_filter = rs.spatial_filter()
        temporal_filter = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        colorizer = rs.colorizer()

        self.status_changed.emit("Đang phát luồng 2D RGB & 3D Depth (RealSense Active)", True)

        frame_count = 0
        last_t = time.time()

        try:
            while True:
                with QMutexLocker(self._mutex):
                    if not self._running:
                        break
                    cur_scheme = self.color_scheme_idx
                    use_filters = self.enable_filters
                    target_pt = self.target_point

                frames = pipeline.wait_for_frames(2500)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Áp dụng bộ lọc nếu bật
                if use_filters:
                    depth_processed = spatial_filter.process(depth_frame)
                    depth_processed = temporal_filter.process(depth_processed)
                    depth_processed = hole_filling.process(depth_processed)
                else:
                    depth_processed = depth_frame

                colorizer.set_option(rs.option.color_scheme, float(cur_scheme))
                depth_colormap = np.asanyarray(colorizer.colorize(depth_processed).get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Đo khoảng cách
                cx, cy = self.width // 2, self.height // 2
                dist_center = depth_frame.get_distance(cx, cy)

                dist_target = 0.0
                if target_pt is not None:
                    tx = max(0, min(self.width - 1, target_pt[0]))
                    ty = max(0, min(self.height - 1, target_pt[1]))
                    dist_target = depth_frame.get_distance(tx, ty)

                self.frames_ready.emit(color_image, depth_colormap, dist_center, dist_target)

                # Đo FPS
                frame_count += 1
                now_t = time.time()
                if now_t - last_t >= 1.0:
                    fps_val = frame_count / (now_t - last_t)
                    self.fps_updated.emit(round(fps_val, 1))
                    frame_count = 0
                    last_t = now_t

        except Exception as e:
            self.error_occurred.emit(f"Lỗi luồng RealSense: {e}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.status_changed.emit("Camera RealSense đã ngắt kết nối", False)

    def _run_demo_loop(self):
        """Tạo dữ liệu mô phỏng 2D + 3D giả lập trực quan sinh động."""
        self.device_info_ready.emit({
            "name": "Intel RealSense D435 (MÔ PHỎNG DEMO)",
            "serial": "DEMO-94827103",
            "firmware": "5.13.0.50 (Virtual)",
            "usb_type": "3.2 (SuperSpeed Virtual)",
        })
        self.status_changed.emit("Đang chạy luồng mô phỏng 3D & 2D (Demo Mode)", True)

        phase = 0.0
        frame_count = 0
        last_t = time.time()

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                target_pt = self.target_point
                cur_scheme = self.color_scheme_idx

            w, h = self.width, self.height
            phase += 0.06

            # Tạo ảnh 2D RGB mô phỏng
            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            # Nền gradient nhẹ
            for y in range(h):
                color_img[y, :, 0] = int(24 + 16 * (y / h))
                color_img[y, :, 1] = int(32 + 20 * (y / h))
                color_img[y, :, 2] = int(48 + 30 * (y / h))

            # Vẽ mặt người mô phỏng
            face_cx = int(w // 2 + 30 * np.sin(phase * 0.5))
            face_cy = int(h // 2 + 15 * np.cos(phase * 0.7))
            cv2.ellipse(color_img, (face_cx, face_cy), (110, 150), 0, 0, 360, (180, 210, 240), -1)
            cv2.ellipse(color_img, (face_cx, face_cy), (110, 150), 0, 0, 360, (255, 255, 255), 2)
            # Mắt, mũi, miệng
            cv2.circle(color_img, (face_cx - 40, face_cy - 30), 12, (70, 80, 100), -1)
            cv2.circle(color_img, (face_cx + 40, face_cy - 30), 12, (70, 80, 100), -1)
            cv2.circle(color_img, (face_cx, face_cy + 15), 14, (140, 160, 190), -1)
            cv2.ellipse(color_img, (face_cx, face_cy + 70), (45, 18), 0, 0, 180, (80, 90, 180), 3)

            # Tạo bản đồ độ sâu 3D (Heatmap gradient)
            Y, X = np.ogrid[:h, :w]
            dist_from_face = np.sqrt((X - face_cx) ** 2 + (Y - face_cy) ** 2)
            depth_raw = np.clip(500 + dist_from_face * 3.5 + 40 * np.sin(phase + X / 40.0), 400, 2500).astype(np.float32)

            # Chuẩn hóa sang 0-255 và áp bảng màu
            depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colormap_type = cv2.COLORMAP_JET if cur_scheme == 0 else (cv2.COLORMAP_RAINBOW if cur_scheme == 1 else cv2.COLORMAP_OCEAN)
            depth_colormap = cv2.applyColorMap(depth_norm, colormap_type)

            cx, cy = w // 2, h // 2
            dist_center = float(depth_raw[cy, cx] / 1000.0)

            dist_target = 0.0
            if target_pt is not None:
                tx = max(0, min(w - 1, target_pt[0]))
                ty = max(0, min(h - 1, target_pt[1]))
                dist_target = float(depth_raw[ty, tx] / 1000.0)

            self.frames_ready.emit(color_img, depth_colormap, dist_center, dist_target)

            frame_count += 1
            now_t = time.time()
            if now_t - last_t >= 1.0:
                self.fps_updated.emit(round(frame_count / (now_t - last_t), 1))
                frame_count = 0
                last_t = now_t

            time.sleep(0.033)


class InteractiveStreamView(QWidget):
    """Khung hiển thị video có tương tác chuột để đo khoảng cách tại bất kỳ tọa độ nào."""

    point_hovered = pyqtSignal(int, int)  # x, y trên ảnh gốc 640x480
    point_clicked = pyqtSignal(int, int)
    point_cleared = pyqtSignal()

    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)

        self._pixmap: Optional[QPixmap] = None
        self.img_w = 640
        self.img_h = 480
        self.crosshair_pt = (self.img_w // 2, self.img_h // 2)
        self.selected_pt: Optional[Tuple[int, int]] = None
        self.hover_pt: Optional[Tuple[int, int]] = None
        self.target_dist_txt = ""

    def update_image(self, bgr_img: np.ndarray, target_dist_txt: str = ""):
        """Cập nhật frame BGR mới lên viewport."""
        if bgr_img is None or bgr_img.size == 0:
            return

        self.img_h, self.img_w = bgr_img.shape[:2]
        self.target_dist_txt = target_dist_txt

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_selected_point(self, pt: Optional[Tuple[int, int]]):
        self.selected_pt = pt
        self.update()

    def _map_widget_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        """Chuyển đổi tọa độ widget sang tọa độ pixel ảnh gốc."""
        rect = self._get_image_rect()
        if not rect.contains(pos.x(), pos.y()):
            return None

        norm_x = (pos.x() - rect.x()) / max(1.0, rect.width())
        norm_y = (pos.y() - rect.y()) / max(1.0, rect.height())
        img_x = int(norm_x * self.img_w)
        img_y = int(norm_y * self.img_h)
        return (max(0, min(self.img_w - 1, img_x)), max(0, min(self.img_h - 1, img_y)))

    def _map_image_to_widget(self, ix: int, iy: int) -> QPoint:
        rect = self._get_image_rect()
        wx = rect.x() + (ix / self.img_w) * rect.width()
        wy = rect.y() + (iy / self.img_h) * rect.height()
        return QPoint(int(wx), int(wy))

    def _get_image_rect(self) -> QRectF:
        """Tính toán khung chữ nhật ảnh theo tỷ lệ 4:3 bên trong widget."""
        avail_w = self.width() - 16
        avail_h = self.height() - 48  # Trừ header
        target_aspect = self.img_w / max(1, self.img_h)

        if avail_w / max(1, avail_h) > target_aspect:
            dest_h = avail_h
            dest_w = int(dest_h * target_aspect)
        else:
            dest_w = avail_w
            dest_h = int(dest_w / target_aspect)

        dx = (self.width() - dest_w) / 2
        dy = 40 + (avail_h - dest_h) / 2
        return QRectF(dx, dy, dest_w, dest_h)

    def mouseMoveEvent(self, event):
        pt = self._map_widget_to_image(event.position().toPoint())
        if pt is not None:
            self.hover_pt = pt
            self.point_hovered.emit(pt[0], pt[1])
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self._map_widget_to_image(event.position().toPoint())
            if pt is not None:
                self.selected_pt = pt
                self.point_clicked.emit(pt[0], pt[1])
                play_chime_tick()
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.selected_pt = None
            self.point_cleared.emit()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Nền card tối chuyên nghiệp
        painter.fillRect(self.rect(), QColor(15, 23, 42))

        # Tiêu đề Header
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(16, 26, self.title)

        if self.subtitle:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(self.width() - 280, 26, self.subtitle)

        # Viền viewport
        img_rect = self._get_image_rect()
        painter.setPen(QPen(QColor(51, 65, 85), 2))
        painter.setBrush(QColor(2, 6, 23))
        painter.drawRoundedRect(img_rect, 12, 12)

        if self._pixmap is None:
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(img_rect, Qt.AlignmentFlag.AlignCenter, "Đang kết nối luồng camera...")
            return

        # Vẽ ảnh nội dung
        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(img_rect, 12, 12)
        painter.setClipPath(clip_path)

        scaled = self._pixmap.scaled(
            int(img_rect.width()),
            int(img_rect.height()),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(int(img_rect.x()), int(img_rect.y()), scaled)

        # 1. Vẽ dấu tâm chữ thập (Center Crosshair)
        center_w = self._map_image_to_widget(self.crosshair_pt[0], self.crosshair_pt[1])
        painter.setPen(QPen(QColor(255, 230, 0, 220), 2))
        painter.drawLine(center_w.x() - 14, center_w.y(), center_w.x() + 14, center_w.y())
        painter.drawLine(center_w.x(), center_w.y() - 14, center_w.x(), center_w.y() + 14)
        painter.drawEllipse(center_w, 4, 4)

        # 2. Vẽ điểm tương tác chuột (Selected / Hover Point)
        active_pt = self.selected_pt or self.hover_pt
        if active_pt is not None:
            wpt = self._map_image_to_widget(active_pt[0], active_pt[1])
            is_locked = (self.selected_pt is not None)

            dot_color = QColor(16, 185, 129) if is_locked else QColor(56, 189, 248)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(dot_color)
            painter.drawEllipse(wpt, 6, 6)

            # Vòng tròn radar xung quanh điểm
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(dot_color, 2, Qt.PenStyle.DashLine))
            painter.drawEllipse(wpt, 14, 14)

            # Tooltip hiển thị khoảng cách
            if self.target_dist_txt:
                badge_w = 140
                badge_h = 32
                bx = min(int(img_rect.right() - badge_w - 10), max(int(img_rect.left() + 10), wpt.x() + 12))
                by = max(int(img_rect.top() + 10), wpt.y() - 36)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(15, 23, 42, 235))
                painter.drawRoundedRect(QRectF(bx, by, badge_w, badge_h), 6, 6)

                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.drawText(QRectF(bx, by, badge_w, badge_h), Qt.AlignmentFlag.AlignCenter, self.target_dist_txt)

        painter.restore()


class RealSenseView(QWidget):
    """Giao diện kiểm tra toàn diện camera 3D & 2D Intel RealSense trong ứng dụng chính."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: Optional[RealSenseThread] = None
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.center_dist_m: float = 0.0
        self.target_dist_m: float = 0.0

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # --- 1. HEADER SECTION ---
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "background-color: #1e293b; border-radius: 12px; padding: 12px 18px; border: 1px solid #334155;"
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_box = QVBoxLayout()
        lbl_title = QLabel("📷 KIỂM TRA CAMERA 3D & 2D INTEL REALSENSE")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;")
        lbl_desc = QLabel("Đồng bộ luồng 2D RGB (Auto-Exposure sáng nét) & Bản đồ 3D Depth Map đo cự ly milimét")
        lbl_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_desc)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Badge thông tin thiết bị
        self.badge_device = QLabel("Đang quét thiết bị...")
        self.badge_device.setStyleSheet(
            "background-color: #0f172a; color: #38bdf8; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 8px; border: 1px solid #1e293b;"
        )
        header_layout.addWidget(self.badge_device)

        self.btn_reconnect = QPushButton("🔄 Quét lại")
        self.btn_reconnect.setStyleSheet(
            "background-color: #38bdf8; color: #0f172a; font-weight: 700; padding: 7px 16px; border-radius: 8px; border: none;"
        )
        self.btn_reconnect.clicked.connect(self.restart_stream)
        header_layout.addWidget(self.btn_reconnect)

        self.btn_demo_mode = QPushButton("▶️ Bật Demo 3D")
        self.btn_demo_mode.setStyleSheet(
            "background-color: #6366f1; color: #ffffff; font-weight: 700; padding: 7px 16px; border-radius: 8px; border: none;"
        )
        self.btn_demo_mode.clicked.connect(self.toggle_demo_mode)
        header_layout.addWidget(self.btn_demo_mode)

        main_layout.addWidget(header_frame)

        # --- 2. DUAL VIEWPORT (2D RGB & 3D DEPTH MAP) ---
        view_splitter = QSplitter(Qt.Orientation.Horizontal)
        view_splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; width: 4px; border-radius: 2px; }")

        # Cột trái: 2D RGB Stream
        self.view_2d = InteractiveStreamView(
            title="1. KHUNG HÌNH 2D RGB (COLOR)",
            subtitle="Auto-Exposure sáng rõ | Nhấp chuột để đo cự ly",
        )
        self.view_2d.point_hovered.connect(self._on_mouse_point_selected)
        self.view_2d.point_clicked.connect(self._on_mouse_point_selected)
        self.view_2d.point_cleared.connect(self._on_point_cleared)
        view_splitter.addWidget(self.view_2d)

        # Cột phải: 3D Depth Map
        self.view_3d = InteractiveStreamView(
            title="2. BẢN ĐỒ ĐỘ SÂU 3D (DEPTH MAP)",
            subtitle="Độ sâu 1:1 Color Aligned | Xanh = Gần, Đỏ = Xa",
        )
        self.view_3d.point_hovered.connect(self._on_mouse_point_selected)
        self.view_3d.point_clicked.connect(self._on_mouse_point_selected)
        self.view_3d.point_cleared.connect(self._on_point_cleared)
        view_splitter.addWidget(self.view_3d)

        main_layout.addWidget(view_splitter, stretch=1)

        # --- 3. TELEMETRY CARDS & CONTROLS ---
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet(
            "background-color: #1e293b; border-radius: 12px; padding: 12px 16px; border: 1px solid #334155;"
        )
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(14)

        # Card 1: Khoảng cách tâm
        self.card_center_dist = self._create_metric_card("KHOẢNG CÁCH TÂM", "-- cm", "#f59e0b")
        bottom_layout.addWidget(self.card_center_dist)

        # Card 2: Điểm chọn chuột
        self.card_target_dist = self._create_metric_card("ĐIỂM CHỌN CHUỘT", "-- cm", "#10b981")
        bottom_layout.addWidget(self.card_target_dist)

        # Card 3: Cổng kết nối USB
        self.card_usb = self._create_metric_card("CỔNG USB", "--", "#38bdf8")
        bottom_layout.addWidget(self.card_usb)

        # Card 4: Tốc độ FPS
        self.card_fps = self._create_metric_card("FPS REALTIME", "-- fps", "#a855f7")
        bottom_layout.addWidget(self.card_fps)

        bottom_layout.addStretch()

        # Điều khiển bộ lọc & Bảng màu 3D
        ctrl_box = QHBoxLayout()
        ctrl_box.setSpacing(10)

        lbl_pal = QLabel("Bảng màu 3D:")
        lbl_pal.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
        ctrl_box.addWidget(lbl_pal)

        self.combo_palettes = QComboBox()
        self.combo_palettes.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; font-size: 11px;"
        )
        for name, idx in DEPTH_COLOR_SCHEMES:
            self.combo_palettes.addItem(name, idx)
        self.combo_palettes.currentIndexChanged.connect(self._on_palette_changed)
        ctrl_box.addWidget(self.combo_palettes)

        self.chk_filters = QCheckBox("Lọc nhiễu độ sâu")
        self.chk_filters.setChecked(True)
        self.chk_filters.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
        self.chk_filters.toggled.connect(self._on_filter_toggled)
        ctrl_box.addWidget(self.chk_filters)

        self.btn_snapshot = QPushButton("📸 Chụp ảnh 2D & 3D")
        self.btn_snapshot.setStyleSheet(
            "background-color: #10b981; color: #042f2e; font-weight: 800; padding: 8px 18px; border-radius: 8px; border: none; font-size: 12px;"
        )
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        ctrl_box.addWidget(self.btn_snapshot)

        bottom_layout.addLayout(ctrl_box)
        main_layout.addWidget(bottom_bar)

    def _create_metric_card(self, title: str, init_val: str, color_hex: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 8px 14px; min-width: 130px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 10px; font-weight: 700; color: #64748b; letter-spacing: 0.5px;")
        lbl_v = QLabel(init_val)
        lbl_v.setStyleSheet(f"font-size: 14px; font-weight: 800; color: {color_hex};")
        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)

        card.val_label = lbl_v  # type: ignore
        return card

    def start_stream(self, demo_mode: bool = False):
        """Khởi động luồng camera RealSense."""
        self.stop_stream()

        self.thread = RealSenseThread(width=640, height=480, fps=30)
        self.thread.set_demo_mode(demo_mode)
        self.thread.set_color_scheme(self.combo_palettes.currentData())
        self.thread.set_filters_enabled(self.chk_filters.isChecked())

        self.thread.frames_ready.connect(self._on_frames_received)
        self.thread.device_info_ready.connect(self._on_device_info_received)
        self.thread.status_changed.connect(self._on_status_changed)
        self.thread.fps_updated.connect(self._on_fps_updated)
        self.thread.error_occurred.connect(self._on_error_occurred)

        self.thread.start()

    def stop_stream(self):
        """Dừng luồng camera giải phóng tài nguyên."""
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

    def restart_stream(self):
        """Quét và kết nối lại thiết bị phần cứng."""
        play_chime_step()
        self.btn_demo_mode.setText("▶️ Bật Demo 3D")
        self.btn_demo_mode.setStyleSheet(
            "background-color: #6366f1; color: #ffffff; font-weight: 700; padding: 7px 16px; border-radius: 8px; border: none;"
        )
        self.start_stream(demo_mode=False)

    def toggle_demo_mode(self):
        """Bật/tắt chế độ mô phỏng khi không có phần cứng."""
        if self.thread and self.thread.demo_mode:
            self.btn_demo_mode.setText("▶️ Bật Demo 3D")
            self.btn_demo_mode.setStyleSheet(
                "background-color: #6366f1; color: #ffffff; font-weight: 700; padding: 7px 16px; border-radius: 8px; border: none;"
            )
            self.start_stream(demo_mode=False)
        else:
            self.btn_demo_mode.setText("⏹️ Tắt Demo 3D")
            self.btn_demo_mode.setStyleSheet(
                "background-color: #ef4444; color: #ffffff; font-weight: 700; padding: 7px 16px; border-radius: 8px; border: none;"
            )
            self.start_stream(demo_mode=True)

    @pyqtSlot(np.ndarray, np.ndarray, float, float)
    def _on_frames_received(self, color_bgr: np.ndarray, depth_colormap: np.ndarray, dist_center: float, dist_target: float):
        self.latest_color = color_bgr.copy()
        self.latest_depth = depth_colormap.copy()
        self.center_dist_m = dist_center
        self.target_dist_m = dist_target

        center_txt = f"{dist_center * 100:.1f} cm ({dist_center:.2f} m)" if dist_center > 0 else "Ngoài phạm vi"
        self.card_center_dist.val_label.setText(center_txt)  # type: ignore

        target_txt = f"{dist_target * 100:.1f} cm ({dist_target:.2f} m)" if dist_target > 0 else "-- cm"
        self.card_target_dist.val_label.setText(target_txt)  # type: ignore

        # Cập nhật hiển thị lên 2 màn hình
        self.view_2d.update_image(color_bgr, target_dist_txt=target_txt if dist_target > 0 else "")
        self.view_3d.update_image(depth_colormap, target_dist_txt=target_txt if dist_target > 0 else "")

    @pyqtSlot(dict)
    def _on_device_info_received(self, info: dict):
        name = info.get("name", "Unknown Device")
        usb = info.get("usb_type", "Unknown")
        fw = info.get("firmware", "N/A")

        self.badge_device.setText(f"{name} | USB {usb} | FW: {fw}")
        self.card_usb.val_label.setText(f"USB {usb}")  # type: ignore
        if "2." in usb:
            self.card_usb.val_label.setStyleSheet("font-size: 14px; font-weight: 800; color: #ef4444;")  # type: ignore
        else:
            self.card_usb.val_label.setStyleSheet("font-size: 14px; font-weight: 800; color: #10b981;")  # type: ignore

    @pyqtSlot(str, bool)
    def _on_status_changed(self, msg: str, is_ok: bool):
        color = "#10b981" if is_ok else "#ef4444"
        self.badge_device.setStyleSheet(
            f"background-color: #0f172a; color: {color}; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 8px; border: 1px solid {color};"
        )

    @pyqtSlot(float)
    def _on_fps_updated(self, fps: float):
        self.card_fps.val_label.setText(f"{fps:.1f} fps")  # type: ignore

    @pyqtSlot(str)
    def _on_error_occurred(self, err_msg: str):
        self.badge_device.setText("Chưa kết nối phần cứng")
        self.badge_device.setStyleSheet(
            "background-color: #0f172a; color: #ef4444; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 8px; border: 1px solid #ef4444;"
        )

    def _on_mouse_point_selected(self, x: int, y: int):
        """Đồng bộ điểm chọn chuột giữa 2 khung nhìn 2D và 3D."""
        if self.thread is not None:
            self.thread.set_target_point((x, y))
        self.view_2d.set_selected_point((x, y))
        self.view_3d.set_selected_point((x, y))

    def _on_point_cleared(self):
        if self.thread is not None:
            self.thread.set_target_point(None)
        self.view_2d.set_selected_point(None)
        self.view_3d.set_selected_point(None)

    def _on_palette_changed(self, idx: int):
        scheme_idx = self.combo_palettes.currentData()
        if self.thread is not None:
            self.thread.set_color_scheme(scheme_idx)

    def _on_filter_toggled(self, checked: bool):
        if self.thread is not None:
            self.thread.set_filters_enabled(checked)

    def take_snapshot(self):
        """Chụp và lưu ảnh 2D RGB, ảnh 3D Depth Map và ảnh ghép sang thư mục realsense_snapshots/."""
        if self.latest_color is None or self.latest_depth is None:
            QMessageBox.warning(self, "Chưa có hình ảnh", "Camera chưa cung cấp khung hình hợp lệ để lưu!")
            return

        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "realsense_snapshots"))
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        f_color = os.path.join(out_dir, f"snapshot_2d_{timestamp}.png")
        f_depth = os.path.join(out_dir, f"snapshot_3d_{timestamp}.png")
        f_comb = os.path.join(out_dir, f"snapshot_combined_{timestamp}.png")

        combined = np.hstack((self.latest_color, self.latest_depth))

        cv2.imwrite(f_color, self.latest_color)
        cv2.imwrite(f_depth, self.latest_depth)
        cv2.imwrite(f_comb, combined)

        play_chime_success()
        QMessageBox.information(
            self,
            "Đã lưu ảnh RealSense thành công",
            f"Đã lưu ảnh 2D và 3D vào thư mục:\n{out_dir}\n\n"
            f"• 2D Color: snapshot_2d_{timestamp}.png\n"
            f"• 3D Depth: snapshot_3d_{timestamp}.png\n"
            f"• Ghép đôi: snapshot_combined_{timestamp}.png",
        )
