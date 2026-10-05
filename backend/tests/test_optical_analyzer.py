"""Kiểm thử đơn vị cho Module backend/optical_analyzer.py (Active Optical Liveness)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optical_analyzer import (
    OpticalAnalyzer,
    calculate_pearson_correlation,
    extract_polygon_mean_rgb,
    extract_background_mean_rgb,
)


def test_pearson_correlation_identical_and_opposite():
    vec_a = np.array([10.0, 20.0, 30.0, -10.0, -20.0, 0.0])
    # Tỷ lệ tuyến tính dương
    vec_b = vec_a * 2.5 + 3.0
    r_pos = calculate_pearson_correlation(vec_a, vec_b)
    assert pytest.approx(r_pos, 0.001) == 1.0

    # Tỷ lệ tuyến tính âm
    vec_c = -vec_a * 1.5
    r_neg = calculate_pearson_correlation(vec_a, vec_c)
    assert pytest.approx(r_neg, 0.001) == -1.0


def test_extract_polygon_mean_rgb():
    # Khung hình BGR 200x200
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Vẽ đa giác màu RGB = (200, 100, 50) -> BGR = (50, 100, 200)
    poly = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)
    img[50:151, 50:151] = (50, 100, 200)

    mean_rgb = extract_polygon_mean_rgb(img, poly)
    assert np.allclose(mean_rgb, [200.0, 100.0, 50.0], atol=1.0)


def _create_mock_landmarks(h=480, w=640) -> np.ndarray:
    """Tạo mảng mốc giả lập 478 điểm mốc với các vùng trán và gò má hợp lệ."""
    pts = np.zeros((478, 2), dtype=np.int32)
    # Trán: khoảng x: 260..380, y: 100..150
    for idx in [10, 67, 109, 108, 151, 337, 297, 284]:
        pts[idx] = [320, 120]
    # Gò má trái: x: 240, y: 240
    for idx in [116, 123, 147, 213, 192, 214]:
        pts[idx] = [240, 240]
    # Gò má phải: x: 400, y: 240
    for idx in [345, 352, 376, 433, 416, 434]:
        pts[idx] = [400, 240]
    return pts


def test_optical_analyzer_live_face_passes():
    """Mô phỏng da người thật: Phản xạ màu da tăng/giảm đồng pha với nguồn phát."""
    analyzer = OpticalAnalyzer(min_pearson=0.65, min_amplitude=4.0)

    # Chuỗi 3 màu phát từ màn hình: GREEN -> RED -> BLUE
    screen_colors = [
        (0, 230, 118),   # Green
        (255, 61, 0),    # Red
        (41, 121, 255),  # Blue
    ]

    mock_lm = _create_mock_landmarks()
    frames = []

    # Màu da ban đầu + tỷ lệ phản xạ quang học 15% từ màn hình
    base_skin_bgr = np.array([120, 140, 180], dtype=np.float64)  # BGR

    for step_rgb in screen_colors:
        f = np.full((480, 640, 3), 40, dtype=np.uint8)  # Nền tối
        # Phản xạ da mặt: base_skin + 0.15 * screen_color
        step_bgr = np.array([step_rgb[2], step_rgb[1], step_rgb[0]], dtype=np.float64)
        skin_color = np.clip(base_skin_bgr + 0.15 * step_bgr, 0, 255).astype(np.uint8)

        # Gán màu da vào vùng trán và gò má
        f[100:150, 260:380] = skin_color
        f[220:260, 220:260] = skin_color
        f[220:260, 380:420] = skin_color
        frames.append(f)

    res = analyzer.analyze_sequence(
        frames_bgr=frames,
        expected_colors_rgb=screen_colors,
        custom_landmarks_list=[mock_lm, mock_lm, mock_lm],
    )

    assert res.passed is True
    assert res.verdict == "LIVENESS_PASS"
    assert res.correlation_score >= 0.85
    assert res.amplitude >= 4.0


def test_optical_analyzer_static_print_attack_fails():
    """Mô phỏng tấn công ảnh in tĩnh: Màu sắc mặt không đổi qua 3 khung hình."""
    analyzer = OpticalAnalyzer(min_pearson=0.65, min_amplitude=4.0)

    screen_colors = [(0, 230, 118), (255, 61, 0), (41, 121, 255)]
    mock_lm = _create_mock_landmarks()

    # Cả 3 khung hình ảnh in đều giữ màu tĩnh không đổi
    frames = []
    static_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    for _ in range(3):
        frames.append(static_frame.copy())

    res = analyzer.analyze_sequence(
        frames_bgr=frames,
        expected_colors_rgb=screen_colors,
        custom_landmarks_list=[mock_lm, mock_lm, mock_lm],
    )

    assert res.passed is False
    assert res.verdict == "STATIC_SPOOF_REPLAY"
    assert "ảnh in hoặc màn hình tĩnh" in res.message


def test_optical_analyzer_video_replay_mismatch_fails():
    """Mô phỏng tấn công video replay: Chuỗi biến thiên không khớp với màu phát."""
    analyzer = OpticalAnalyzer(min_pearson=0.65, min_amplitude=4.0)

    screen_colors = [(0, 230, 118), (255, 61, 0), (41, 121, 255)]
    mock_lm = _create_mock_landmarks()

    frames = []
    # Biến thiên nghịch pha hoặc ngẫu nhiên
    for i in range(3):
        f = np.full((480, 640, 3), 40, dtype=np.uint8)
        # Tăng dần kênh BGR theo hướng hoàn toàn khác
        spoof_color = np.array([50 + i * 30, 200 - i * 40, 30 + i * 10], dtype=np.uint8)
        f[100:150, 260:380] = spoof_color
        f[220:260, 220:260] = spoof_color
        f[220:260, 380:420] = spoof_color
        frames.append(f)

    res = analyzer.analyze_sequence(
        frames_bgr=frames,
        expected_colors_rgb=screen_colors,
        custom_landmarks_list=[mock_lm, mock_lm, mock_lm],
    )

    assert res.passed is False
    assert res.verdict == "OPTICAL_MISMATCH_SPOOF"


def test_optical_analyzer_face_lost():
    """Mô phỏng trường hợp mất dấu khuôn mặt ở khung hình thứ 2."""
    analyzer = OpticalAnalyzer(min_pearson=0.65, min_amplitude=4.0)
    screen_colors = [(0, 230, 118), (255, 61, 0), (41, 121, 255)]
    mock_lm = _create_mock_landmarks()

    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    # Khung 0 có mốc, Khung 1 mất mốc (None), Khung 2 có mốc
    res = analyzer.analyze_sequence(
        frames_bgr=frames,
        expected_colors_rgb=screen_colors,
        custom_landmarks_list=[mock_lm, None, mock_lm],
    )

    assert res.passed is False
    assert res.verdict == "FACE_LOST"
    assert res.details.get("failed_frame_index") == 1
