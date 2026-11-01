"""RealSense 3D Depth & 2D Color Camera Thread for eKYC System.
Provides synchronized Color (BGR) and Depth (Millimeters + Colormap) frames.
Supports hardware RealSense pipelines and Virtual 3D Simulation fallback.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False


class RealSenseCaptureThread(QThread):
    """Worker thread quản lý vòng lặp thu thập và đồng bộ luồng 2D RGB & 3D Depth (Z16)."""

    # color_bgr (480, 640, 3), raw_depth_mm (480, 640 uint16), depth_colormap (480, 640, 3)
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
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
        self.demo_mode = False

    def set_color_scheme(self, idx: int):
        with QMutexLocker(self._mutex):
            self.color_scheme_idx = idx

    def set_filters_enabled(self, enabled: bool):
        with QMutexLocker(self._mutex):
            self.enable_filters = enabled

    def set_demo_mode(self, enabled: bool):
        with QMutexLocker(self._mutex):
            self.demo_mode = enabled

    def stop(self):
        """Dừng luồng và đợi pipeline giải phóng an toàn."""
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2500)

    def run(self):
        with QMutexLocker(self._mutex):
            self._running = True
            is_demo = self.demo_mode

        if is_demo or not HAS_REALSENSE:
            self._run_demo_loop()
            return

        # Kiểm tra kết nối thiết bị phần cứng
        try:
            ctx = rs.context()
            devices = ctx.devices
            if len(devices) == 0:
                self.status_changed.emit("Không tìm thấy camera Intel RealSense (Tự động chuyển Demo)", False)
                self._run_demo_loop()
                return

            dev = devices[0]
            dev_name = dev.get_info(rs.camera_info.name)
            dev_serial = dev.get_info(rs.camera_info.serial_number)
            dev_fw = dev.get_info(rs.camera_info.firmware_version)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor) if hasattr(rs.camera_info, "usb_type_descriptor") else "3.x"

            self.device_info_ready.emit({
                "name": dev_name,
                "serial": dev_serial,
                "firmware": dev_fw,
                "usb_type": usb_type,
            })
            self.status_changed.emit(f"Đã kết nối {dev_name} (USB {usb_type})", True)
        except Exception as e:
            self.error_occurred.emit(f"Không thể đọc thông tin thiết bị: {e}")
            self._run_demo_loop()
            return

        # Khởi tạo Pipeline RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        try:
            pipeline_profile = pipeline.start(config)

            # Cấu hình cảm biến RGB: Auto Exposure & Auto White Balance
            try:
                color_sensor = pipeline_profile.get_device().first_color_sensor()
                if color_sensor:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            except Exception:
                pass

            # Đối tượng căn chỉnh tọa độ 1:1 Color và Depth
            align = rs.align(rs.stream.color)

            # Bộ lọc làm mịn dữ liệu chiều sâu
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)

            temporal = rs.temporal_filter()
            hole_filling = rs.hole_filling_filter(1)

            # Colorizer để tạo ảnh màu độ sâu
            colorizer = rs.colorizer()

            frame_count = 0
            last_t = time.time()

            while True:
                with QMutexLocker(self._mutex):
                    if not self._running:
                        break
                    color_scheme = self.color_scheme_idx
                    use_filters = self.enable_filters

                colorizer.set_option(rs.option.color_scheme, float(color_scheme))

                frames = pipeline.wait_for_frames(2500)
                if not frames:
                    continue

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_processed = depth_frame
                if use_filters:
                    depth_processed = spatial.process(depth_processed)
                    depth_processed = temporal.process(depth_processed)
                    depth_processed = hole_filling.process(depth_processed)

                color_image = np.asanyarray(color_frame.get_data())
                raw_depth_mm = np.asanyarray(depth_processed.get_data())
                depth_colormap = np.asanyarray(colorizer.colorize(depth_processed).get_data())

                self.frames_ready.emit(color_image, raw_depth_mm, depth_colormap)

                frame_count += 1
                now_t = time.time()
                if now_t - last_t >= 1.0:
                    fps_val = frame_count / (now_t - last_t)
                    self.fps_updated.emit(round(fps_val, 1))
                    frame_count = 0
                    last_t = now_t

        except Exception as e:
            self.error_occurred.emit(f"Lỗi phần cứng RealSense: {e}")
            self.status_changed.emit("Lỗi luồng RealSense. Chuyển sang chế độ Demo.", False)
            self._run_demo_loop()
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.status_changed.emit("Camera RealSense đã ngắt kết nối", False)

    def _run_demo_loop(self):
        """Mô phỏng dữ liệu 3D Depth và RGB thực tế khi không có camera cắm trực tiếp."""
        self.device_info_ready.emit({
            "name": "Intel RealSense D435 (MÔ PHỎNG 3D VIRTUAL)",
            "serial": "DEMO-3D-998822",
            "firmware": "5.13.0 (Demo)",
            "usb_type": "3.2 (Virtual)",
        })
        self.status_changed.emit("Đang chạy luồng mô phỏng 3D RealSense", True)

        phase = 0.0
        frame_count = 0
        last_t = time.time()

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            # Tạo khung hình 2D RGB mô phỏng mặt người ngồi trước máy
            color_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Nền phòng sáng
            color_img[:, :] = (35, 45, 55)

            # Vẽ mặt người 3D dạng elip
            cx = int(self.width / 2 + math.sin(phase * 0.8) * 15)
            cy = int(self.height / 2 + math.cos(phase * 0.6) * 10)
            rx, ry = 110, 140

            # Màu da người
            cv2.ellipse(color_img, (cx, cy), (rx, ry), 0, 0, 360, (180, 205, 235), -1)
            # Mắt, lông mày, miệng
            cv2.circle(color_img, (cx - 40, cy - 20), 8, (60, 40, 30), -1)
            cv2.circle(color_img, (cx + 40, cy - 20), 8, (60, 40, 30), -1)
            # Sống mũi
            cv2.line(color_img, (cx, cy - 15), (cx, cy + 25), (140, 160, 190), 4)
            # Miệng cười
            cv2.ellipse(color_img, (cx, cy + 65), (35, 18), 0, 0, 180, (80, 70, 180), 3)

            # Tạo bản đồ chiều sâu raw_depth_mm (mm)
            # Nền phía sau: 1400mm (1.4m)
            depth_map = np.full((self.height, self.width), 1400, dtype=np.uint16)

            # Người ngồi cự ly khoảng 580mm, sống mũi nhô ra 32mm -> 548mm
            y_indices, x_indices = np.ogrid[:self.height, :self.width]
            dist_sq = ((x_indices - cx) / float(rx)) ** 2 + ((y_indices - cy) / float(ry)) ** 2
            face_mask = dist_sq < 1.0

            relief = 34.0 * np.sqrt(np.maximum(0.0, 1.0 - dist_sq[face_mask]))
            base_dist = 580.0 + math.sin(phase) * 20.0
            depth_map[face_mask] = np.clip(base_dist - relief, 200, 3000).astype(np.uint16)

            # Thêm nhiễu lượng tử nhẹ cho giống cảm biến thực
            noise = np.random.randint(-2, 3, (self.height, self.width), dtype=np.int16)
            depth_map = np.clip(depth_map.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

            # Tạo ảnh màu Colormap
            norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)

            self.frames_ready.emit(color_img, depth_map, depth_colormap)

            phase += 0.05
            frame_count += 1
            now_t = time.time()
            if now_t - last_t >= 1.0:
                fps_val = frame_count / (now_t - last_t)
                self.fps_updated.emit(round(fps_val, 1))
                frame_count = 0
                last_t = now_t

            time.sleep(0.033)
