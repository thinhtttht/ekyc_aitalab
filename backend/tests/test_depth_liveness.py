import os
import sys

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np
import pytest
from depth_liveness import DepthLivenessDetector, DepthLivenessResult


def test_depth_liveness_empty_or_none():
    detector = DepthLivenessDetector()
    res = detector.analyze(None, (0.2, 0.2, 0.8, 0.8))
    assert not res.is_real_3d
    assert res.spoof_type == "no_depth_signal"

    empty = np.zeros((0, 0), dtype=np.uint16)
    res2 = detector.analyze(empty, (0.2, 0.2, 0.8, 0.8))
    assert not res2.is_real_3d


def test_depth_liveness_planar_2d_attack():
    """Mô phỏng mặt phẳng màn hình iPad / ảnh in 2D đặt vuông góc ở cự ly 600mm."""
    h, w = 480, 640
    # Mặt phẳng hoàn hảo 600mm kèm nhiễu đo lường nhỏ (+- 1mm)
    depth_map = np.full((h, w), 600, dtype=np.uint16)
    noise = np.random.randint(-1, 2, (h, w), dtype=np.int16)
    depth_map = np.clip(depth_map.astype(np.int32) + noise, 0, 5000).astype(np.uint16)

    detector = DepthLivenessDetector()
    res = detector.analyze(depth_map, (0.3, 0.2, 0.7, 0.8))

    assert not res.is_real_3d
    assert res.spoof_type == "planar_2d_spoof"
    assert res.nose_relief_mm < 5.0
    assert res.curvature_residual_mm < 4.0
    assert res.liveness_score < 0.40


def test_depth_liveness_tilted_planar_attack():
    """Mô phỏng điện thoại bị nghiêng một góc (z = a*x + b*y + c). Vẫn là mặt phẳng 2D!"""
    h, w = 480, 640
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    # Mặt phẳng nghiêng: Z = 600 + 0.15*x - 0.1*y
    plane_z = 600.0 + 0.15 * x_coords - 0.10 * y_coords
    depth_map = plane_z.astype(np.uint16)

    detector = DepthLivenessDetector()
    res = detector.analyze(depth_map, (0.3, 0.2, 0.7, 0.8))

    assert not res.is_real_3d
    # Sai số phi phẳng của mặt phẳng nghiêng vẫn gần như bằng 0
    assert res.curvature_residual_mm < 3.0
    assert res.liveness_score < 0.65


def test_depth_liveness_real_3d_face_simulation():
    """Mô phỏng hình cầu/chóp nổi 3D của khuôn mặt người thật (mũi nhô cao hơn má 35mm)."""
    h, w = 480, 640
    depth_map = np.full((h, w), 600.0, dtype=np.float32)

    # Vùng mặt từ y: 100..380, x: 200..440
    # Chóp mũi tại (320, 240) có độ sâu 565mm (nhô 35mm về phía camera)
    cy, cx = 240, 320
    for y in range(100, 380):
        for x in range(200, 440):
            dist_sq = ((x - cx) / 100.0) ** 2 + ((y - cy) / 120.0) ** 2
            if dist_sq < 1.0:
                # Bán elip nhô lên phía trước (giảm Z)
                relief = 35.0 * np.sqrt(1.0 - dist_sq)
                depth_map[y, x] = 600.0 - relief

    detector = DepthLivenessDetector()
    res = detector.analyze(depth_map.astype(np.uint16), (200, 100, 440, 380))

    assert res.is_real_3d
    assert res.spoof_type == "real_human"
    assert res.nose_relief_mm >= 7.0
    assert res.curvature_residual_mm >= 8.0
    assert res.liveness_score >= 0.50


def test_depth_liveness_out_of_range():
    """Kiểm tra ngoài tầm cự ly (quá gần < 35cm hoặc quá xa > 100cm)."""
    h, w = 480, 640
    too_close = np.full((h, w), 250, dtype=np.uint16)
    detector = DepthLivenessDetector()
    res_close = detector.analyze(too_close, (0.3, 0.2, 0.7, 0.8))
    assert not res_close.is_real_3d
    assert res_close.spoof_type == "out_of_range_too_close"

    too_far = np.full((h, w), 1400, dtype=np.uint16)
    res_far = detector.analyze(too_far, (0.3, 0.2, 0.7, 0.8))
    assert not res_far.is_real_3d
    assert res_far.spoof_type == "out_of_range_too_far"
