"""Unit tests cho module kiểm tra chất lượng Camera (REQ-BE-01)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
from camera_quality import CameraMonitor, ClientStats, estimate_noise, global_sharpness


def test_black_frame_detection():
    monitor = CameraMonitor()
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)

    res = monitor.evaluate(black_img, stats)
    assert not res.checks["not_black"]
    assert res.severity == "error"
    assert "bị che" in res.message or "đen" in res.message


def test_frozen_frame_detection():
    monitor = CameraMonitor()
    static_img = np.full((480, 640, 3), 120, dtype=np.uint8)
    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)

    # Gửi liên tiếp các frame giống hệt nhau
    res = None
    for _ in range(C.FROZEN_FRAMES + 2):
        res = monitor.evaluate(static_img, stats)

    assert not res.checks["not_frozen"]
    assert res.severity == "error"
    assert "đóng băng" in res.message or "treo" in res.message


def test_browser_reported_frozen():
    monitor = CameraMonitor()
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=False)

    res = monitor.evaluate(img, stats)
    assert not res.checks["not_frozen"]
    assert res.severity == "error"


def test_resolution_thresholds():
    monitor = CameraMonitor()
    img = np.full((480, 640, 3), 130, dtype=np.uint8)

    # Resolution thấp < 640x480
    stats_low_res = ClientStats(fps=30.0, width=320, height=240, frame_advancing=True)
    res2 = monitor.evaluate(img, stats_low_res)
    assert not res2.checks["resolution_ok"]


def test_noise_estimation():
    # Ảnh sạch phẳng -> noise = 0
    clean = np.full((300, 300), 128, dtype=np.uint8)
    assert estimate_noise(clean) == 0.0

    # Thêm nhiễu Gauss sigma = 6.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 6.0, (300, 300))
    noisy = np.clip(clean.astype(float) + noise, 0, 255).astype(np.uint8)
    estimated = estimate_noise(noisy)

    # Ước lượng sai số trong khoảng ±1.5
    assert abs(estimated - 6.0) < 1.5


def test_camera_all_ok():
    monitor = CameraMonitor()
    # Frame bình thường có chi tiết nhẹ
    rng = np.random.default_rng(123)
    frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    # Thêm biến thiên nhỏ để tránh diff = 0
    frame += rng.integers(-5, 5, frame.shape, dtype=np.int8).astype(np.uint8)

    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)
    res = monitor.evaluate(frame, stats)

    assert res.all_ok
    assert res.severity == "ok"
