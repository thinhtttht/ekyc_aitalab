"""Camera capture thread using OpenCV and PyQt6 QThread.
Provides auto-brightness enhancement and 30-60 FPS smooth video streaming.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

# Add backend to path if needed
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from camera_quality import auto_enhance_brightness
except ImportError:
    # Fallback if standalone
    def auto_enhance_brightness(f): return f


class CameraThread(QThread):
    """Worker thread that captures webcam frames continuously."""

    frame_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str, bool)  # message, is_ok
    fps_updated = pyqtSignal(float)

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._running = False
        self._mutex = QMutex()
        self._cap: Optional[cv2.VideoCapture] = None
        self.auto_enhance = True

    def run(self):
        with QMutexLocker(self._mutex):
            self._running = True

        # Khởi tạo camera OpenCV qua DirectShow trên Windows để khởi động nhanh
        backend_flag = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend_flag)
        if not self._cap.isOpened():
            # Thử lại với backend mặc định
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            self.status_changed.emit(f"Không thể mở Webcam (index {self.camera_index})", False)
            with QMutexLocker(self._mutex):
                self._running = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.status_changed.emit(f"Webcam {self.camera_index} đã sẵn sàng", True)

        frame_count = 0
        fps_start_time = time.time()

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Lật ngang frame (Mirror effect) cho camera tự nhiên
            frame = cv2.flip(frame, 1)

            # Cắt khung hình chuẩn tỉ lệ dọc 4:5 (480x600 chuẩn eKYC) để Oval khuôn mặt đứng thẳng
            h, w = frame.shape[:2]
            target_ratio = 4.0 / 5.0
            if (w / h) > target_ratio:
                new_w = int(h * target_ratio)
                offset_x = (w - new_w) // 2
                frame = np.ascontiguousarray(frame[:, offset_x : offset_x + new_w])
            elif (w / h) < target_ratio:
                new_h = int(w / target_ratio)
                offset_y = (h - new_h) // 2
                frame = np.ascontiguousarray(frame[offset_y : offset_y + new_h, :])
            else:
                frame = np.ascontiguousarray(frame)

            # Áp dụng tự động nâng sáng thích ứng nếu bật
            if self.auto_enhance:
                try:
                    frame = auto_enhance_brightness(frame)
                except Exception:
                    pass

            self.frame_ready.emit(frame)

            # Tính FPS thực tế
            frame_count += 1
            now = time.time()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(round(fps, 1))
                frame_count = 0
                fps_start_time = now

            # Hạn chế CPU quá tải (khoảng 30-35 FPS)
            time.sleep(0.01)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status_changed.emit("Webcam đã dừng", False)

    def stop(self):
        """Dừng luồng camera an toàn."""
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)
