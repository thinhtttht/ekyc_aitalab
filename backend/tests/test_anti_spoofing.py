"""Unit tests cho Module Chống Giả Mạo Ảnh & Video (Passive Anti-Spoofing / PAD)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
from anti_spoofing import AntiSpoofDetector, AntiSpoofResult, crop_face_margin
from face_analyzer import FaceResult, quality_checks, check_continuous_face_quality


def test_crop_face_margin():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.6)
    crop = crop_face_margin(img, bbox, scale=2.7)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_3d_depth_liveness():
    detector = AntiSpoofDetector()

    # 1. Khuôn mặt 3D thật: mũi nhô ra trước mắt (z mũi âm hơn z mắt)
    lm_real = np.zeros((468, 3), dtype=np.float32)
    lm_real[1] = [0.5, 0.5, -0.06]    # mũi nhô cao
    lm_real[4] = [0.5, 0.52, -0.05]
    lm_real[33] = [0.4, 0.45, 0.0]     # mắt trái
    lm_real[263] = [0.6, 0.45, 0.0]    # mắt phải
    lm_real[133] = [0.45, 0.45, -0.01]
    lm_real[362] = [0.55, 0.45, -0.01]

    is_3d, delta = detector.check_3d_depth(lm_real)
    assert is_3d is True
    assert delta >= C.DEPTH_3D_MIN_DELTA

    # 2. Ảnh in 2D phẳng: z của mũi và mắt gần như bằng nhau trên cùng một mặt phẳng
    lm_flat = np.zeros((468, 3), dtype=np.float32)
    lm_flat[:, 2] = 0.002  # phẳng lỳ

    is_3d_flat, delta_flat = detector.check_3d_depth(lm_flat)
    assert is_3d_flat is False
    assert delta_flat < C.DEPTH_3D_MIN_DELTA


def test_missing_model_is_rejected(monkeypatch):
    detector = AntiSpoofDetector(model_path="khong-ton-tai.onnx")
    assert detector.session is None
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    res = detector.evaluate(frame, (0.3, 0.3, 0.7, 0.7))
    assert res.is_real is False
    assert res.spoof_type == "model_unavailable"
    assert res.severity == "error"

    monkeypatch.setattr(C, "ANTISPOOF_REQUIRE_MODEL", False)
    res = detector.evaluate(frame, (0.3, 0.3, 0.7, 0.7))
    assert res.spoof_type != "model_unavailable"


def test_anti_spoof_detection_print_attack():
    detector = AntiSpoofDetector()
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.8, 0.8)

    # Giả lập kết quả phát hiện ảnh in
    res = AntiSpoofResult(
        is_real=False,
        real_prob=0.08,
        print_prob=0.88,
        replay_prob=0.04,
        spoof_type="print_attack",
        depth_3d_ok=False,
        severity="error",
        message="⚠️ PHÁT HIỆN ẢNH IN 2D – Vui lòng sử dụng khuôn mặt thật trực tiếp!",
    )

    face = FaceResult(
        face_count=1,
        bbox=bbox,
        center=(0.5, 0.5),
        face_h=0.35,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        fill=0.75,
        oval_dist=0.6,
        offset=(0.0, 0.0),
        brightness=120.0,
        brightness_mean=120.0,
        brightness_std=30.0,
        sharpness=80.0,
        scale_ratio=0.60,
        corners_inside=True,
        anti_spoof=res,
    )

    # 1. Kiểm tra quality_checks
    checks, msg, sev = quality_checks(face)
    assert checks["anti_spoof_ok"] is False
    assert checks["no_print_attack"] is False
    assert sev == "error"
    assert "ảnh in 2d" in msg.lower()

    # 2. Kiểm tra continuous quality
    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face)
    assert cont_ok is False
    assert cont_sev == "error"
    assert "ảnh in 2d" in cont_msg.lower()


def test_anti_spoof_detection_replay_attack():
    res = AntiSpoofResult(
        is_real=False,
        real_prob=0.05,
        print_prob=0.05,
        replay_prob=0.90,
        spoof_type="replay_attack",
        screen_detected=True,
        bezel_detected=True,
        severity="error",
        message="⚠️ PHÁT HIỆN MÀN HÌNH / VIDEO PHÁT LẠI – Vui lòng không giơ điện thoại trước camera!",
    )

    face = FaceResult(
        face_count=1,
        bbox=(0.2, 0.2, 0.8, 0.8),
        center=(0.5, 0.5),
        face_h=0.35,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        fill=0.75,
        oval_dist=0.6,
        offset=(0.0, 0.0),
        brightness=120.0,
        brightness_mean=120.0,
        brightness_std=30.0,
        sharpness=80.0,
        scale_ratio=0.60,
        corners_inside=True,
        anti_spoof=res,
    )

    checks, msg, sev = quality_checks(face)
    assert checks["anti_spoof_ok"] is False
    assert checks["no_screen_attack"] is False
    assert sev == "error"
    assert "màn hình" in msg.lower() or "video" in msg.lower()

    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face)
    assert cont_ok is False
    assert cont_sev == "error"


def test_anti_spoof_real_face_pass():
    res = AntiSpoofResult(
        is_real=True,
        real_prob=0.96,
        print_prob=0.02,
        replay_prob=0.02,
        spoof_type="real",
        depth_3d_ok=True,
        depth_3d_delta=0.048,
        severity="ok",
        message="Khuôn mặt người thật hợp lệ",
    )

    face = FaceResult(
        face_count=1,
        bbox=(0.2, 0.2, 0.8, 0.8),
        center=(0.5, 0.5),
        face_h=0.35,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        fill=0.75,
        oval_dist=0.6,
        offset=(0.0, 0.0),
        brightness=120.0,
        brightness_mean=120.0,
        brightness_std=30.0,
        sharpness=80.0,
        scale_ratio=0.60,
        corners_inside=True,
        anti_spoof=res,
        parts_status={
            "left_eye": True, "right_eye": True, "nose": True,
            "mouth": True, "left_eyebrow": True, "right_eyebrow": True,
        },
    )

    checks, msg, sev = quality_checks(face)
    assert checks["anti_spoof_ok"] is True
    assert checks["no_print_attack"] is True
    assert checks["no_screen_attack"] is True
    assert checks["depth_3d_ok"] is True
    assert sev == "ok"


def test_early_spoof_blocking_in_camera_check():
    """Kiểm tra: Khi ở Stage.CAMERA_CHECK, nếu có ảnh in/replay xuất hiện thì bị chặn ngay lập tức."""
    from enrollment import EnrollmentSession, Stage
    from camera_quality import ClientStats

    sess = EnrollmentSession("test-early-spoof-sess")
    assert sess.stage == Stage.CAMERA_CHECK

    # Khung hình giả lập có print attack
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    stats = ClientStats(width=1280, height=720, frame_advancing=True)

    spoof_res = AntiSpoofResult(
        is_real=False,
        print_prob=0.85,
        spoof_type="print_attack",
        severity="error",
        message="⚠️ PHÁT HIỆN ẢNH IN 2D – Vui lòng sử dụng khuôn mặt thật trực tiếp!",
    )

    fake_face = FaceResult(
        face_count=1,
        bbox=(0.2, 0.2, 0.8, 0.8),
        anti_spoof=spoof_res,
    )

    # Mock analyzer trả về fake_face
    class MockAnalyzer:
        def analyze(self, f, oval):
            return fake_face

    sess.face_analyzer = MockAnalyzer()

    # Gọi process_frame
    resp = sess.process_frame(frame, stats)

    # Phải bị chặn ngay lập tức, không được phép chuyển stage hoặc tăng progress
    assert resp["color"] == "red"
    assert "ảnh in 2d" in resp["message"].lower()
    assert sess.stage == Stage.CAMERA_CHECK
    assert resp["progress"] == 0.0

