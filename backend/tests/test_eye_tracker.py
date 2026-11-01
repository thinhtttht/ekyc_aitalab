"""Unit tests for EyeTracker and OneEuroFilter."""
import sys
import os
import pytest
import numpy as np

# Thêm đường dẫn desktop_app vào sys.path
desktop_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "desktop_app"))
if desktop_dir not in sys.path:
    sys.path.insert(0, desktop_dir)

from eye_tracker import EyeTracker, OneEuroFilter, GazeResult


def test_one_euro_filter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    # Lần đầu tiên
    v1 = f(0.0, 10.0)
    assert v1 == 10.0

    # Lần thứ hai với t tăng dần
    v2 = f(0.033, 10.5)
    assert 10.0 <= v2 <= 10.5

    # Reset
    f.reset()
    assert f.x_prev is None


def test_eye_tracker_init():
    tracker = EyeTracker()
    assert tracker.is_calibrating is False
    assert tracker.is_calibrated is False
    assert len(tracker.calib_samples) == 9


def test_eye_tracker_no_face_frame():
    tracker = EyeTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res: GazeResult = tracker.process_frame(frame, screen_size=(1280, 720))

    assert res.has_face is False
    assert res.is_looking_away is True
    assert "Không phát hiện khuôn mặt" in res.looking_away_reason


def test_eye_tracker_calibration_lifecycle():
    tracker = EyeTracker()
    tracker.start_calibration()
    assert tracker.is_calibrating is True
    assert tracker.calib_step == 0

    # Giả lập mẫu 11 chiều cho 9 điểm
    targets = [
        (0.15, 0.18), (0.50, 0.18), (0.85, 0.18),
        (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
        (0.15, 0.82), (0.50, 0.82), (0.85, 0.82)
    ]
    for step in range(9):
        for _ in range(10):
            dummy_feat = np.random.randn(11)
            dummy_feat[0] = 1.0
            tracker.calib_samples[step].append(dummy_feat)

    trained = tracker.train_regression(targets)
    assert trained is True
    assert tracker.is_calibrated is True
    assert tracker.regression_weights_x is not None
    assert tracker.regression_weights_y is not None
