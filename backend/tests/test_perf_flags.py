"""Cờ hiệu năng: tắt MediaPipe Hands, rút gọn face_metrics khi không bật EKYC_DEBUG."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
from anti_spoofing import AntiSpoofResult
from enrollment import _face_metrics
from face_analyzer import FaceAnalyzer, FaceResult

UI_FACE_METRICS = {
    "face_count", "yaw", "pitch", "roll", "is_upside_down", "scale_ratio",
    "oval_dist", "brightness_mean", "brightness_std", "anti_spoof_real_prob",
}


def test_hand_detection_can_be_disabled(monkeypatch):
    monkeypatch.setattr(C, "HAND_DETECTION_ENABLED", False)
    analyzer = FaceAnalyzer(stream=True)
    try:
        assert analyzer._hands is None
        res = analyzer.analyze(np.zeros((240, 320, 3), dtype=np.uint8), C.OVAL_NORMAL)
        assert res.face_count == 0
    finally:
        analyzer.close()


def _face():
    return FaceResult(face_count=1, bbox=(0.3, 0.3, 0.7, 0.7), yaw=1.0, pitch=2.0, roll=0.5,
                      anti_spoof=AntiSpoofResult(real_prob=0.9, model_ran=True))


def test_face_metrics_trimmed_without_debug(monkeypatch):
    monkeypatch.setattr(C, "DEBUG_METRICS", False)
    m = _face_metrics(_face())
    assert UI_FACE_METRICS <= set(m)
    assert "moire_ratio" not in m and "perspective_ratio" not in m


def test_face_metrics_full_with_debug(monkeypatch):
    monkeypatch.setattr(C, "DEBUG_METRICS", True)
    m = _face_metrics(_face())
    assert UI_FACE_METRICS <= set(m)
    assert {"moire_ratio", "perspective_ratio", "anti_spoof_print_prob"} <= set(m)
