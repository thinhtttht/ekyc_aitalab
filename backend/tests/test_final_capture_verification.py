"""Unit tests cho Cơ chế kiểm định an ninh bức ảnh chân dung cuối cùng (verify_final_capture)."""
import os
import sys
import base64
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enrollment import EnrollmentSession, Stage
from face_analyzer import FaceResult
from anti_spoofing import AntiSpoofResult


def make_valid_face_result() -> FaceResult:
    """Tạo FaceResult hợp lệ, người thật, đầy đủ ngũ quan và nhìn thẳng."""
    return FaceResult(
        face_count=1,
        bbox=(0.2, 0.2, 0.8, 0.8),
        center=(0.5, 0.5),
        face_h=0.45,
        yaw=2.0,
        pitch=1.0,
        roll=0.5,
        fill=0.75,
        scale_ratio=0.65,
        corners_inside=True,
        is_upside_down=False,
        hand_occlusion=False,
        mask_detected=False,
        sunglasses_detected=False,
        glare_detected=False,
        brightness_mean=120.0,
        brightness_std=35.0,
        sharpness=85.0,
        parts_status={
            "left_eye": True,
            "right_eye": True,
            "nose": True,
            "mouth": True,
            "left_eyebrow": True,
            "right_eyebrow": True,
        },
        occluded_part_name=None,
        anti_spoof=AntiSpoofResult(
            is_real=True,
            real_prob=0.98,
            print_prob=0.01,
            replay_prob=0.01,
            spoof_type="real_face",
            depth_3d_ok=True,
            depth_3d_delta=0.045,
            message="Khuôn mặt người thật hợp lệ",
        ),
    )


def test_verify_final_capture_success():
    """Kiểm tra trường hợp ảnh chụp chân dung hợp lệ đạt chuẩn toàn bộ."""
    sess = EnrollmentSession("session-test-pass")
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = make_valid_face_result()
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is True
    assert "thành công" in res["message"].lower()
    assert sess.stage == Stage.CAPTURE
    assert sess.enrolled_image_bgr is not None
    assert "final_capture" in sess.history
    assert sess.history["final_capture"]["resolution"] == "640x480"


def test_verify_final_capture_no_face_rejected():
    """Từ chối khi không phát hiện khuôn mặt khi chụp."""
    sess = EnrollmentSession("session-test-noface")
    mock_analyzer = MagicMock()
    no_face = FaceResult(face_count=0, bbox=None)
    mock_analyzer.analyze.return_value = no_face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "no_face"
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_multiple_faces_rejected():
    """Từ chối khi phát hiện nhiều khuôn mặt khi chụp."""
    sess = EnrollmentSession("session-test-multi")
    mock_analyzer = MagicMock()
    multi_face = make_valid_face_result()
    multi_face.face_count = 2
    mock_analyzer.analyze.return_value = multi_face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "multiple_faces"
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_upside_down_rejected():
    """Từ chối khi khuôn mặt bị lật ngược (mắt ở dưới, miệng ở trên)."""
    sess = EnrollmentSession("session-test-inverted")
    mock_analyzer = MagicMock()
    face = make_valid_face_result()
    face.is_upside_down = True
    mock_analyzer.analyze.return_value = face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "upside_down"
    assert "lật ngược" in res["message"].lower()
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_hand_occlusion_rejected():
    """Từ chối khi phát hiện bàn tay che mặt khi chụp."""
    sess = EnrollmentSession("session-test-hand")
    mock_analyzer = MagicMock()
    face = make_valid_face_result()
    face.hand_occlusion = True
    mock_analyzer.analyze.return_value = face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "hand_occlusion"
    assert "bàn tay" in res["message"].lower()
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_mask_or_sunglasses_rejected():
    """Từ chối khi người dùng đeo khẩu trang hoặc kính râm khi chụp."""
    sess = EnrollmentSession("session-test-mask")
    mock_analyzer = MagicMock()
    face_mask = make_valid_face_result()
    face_mask.mask_detected = True
    mock_analyzer.analyze.return_value = face_mask
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)
    assert res["passed"] is False
    assert res["error_type"] == "mask_detected"

    face_sunglass = make_valid_face_result()
    face_sunglass.sunglasses_detected = True
    mock_analyzer.analyze.return_value = face_sunglass
    res2 = sess.verify_final_capture(dummy_frame)
    assert res2["passed"] is False
    assert res2["error_type"] == "sunglasses_detected"


def test_verify_final_capture_print_attack_rejected():
    """Từ chối ngay lập tức khi phát hiện tấn công giả mạo ảnh in 2D."""
    sess = EnrollmentSession("session-test-print-spoof")
    mock_analyzer = MagicMock()
    face = make_valid_face_result()
    face.anti_spoof = AntiSpoofResult(
        is_real=False,
        real_prob=0.12,
        print_prob=0.88,
        replay_prob=0.00,
        spoof_type="print_attack",
        depth_3d_ok=False,
        message="Phát hiện ảnh in 2D giả mạo",
    )
    mock_analyzer.analyze.return_value = face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "spoof_detected"
    assert "ảnh in 2d" in res["message"].lower()
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_screen_replay_rejected():
    """Từ chối ngay lập tức khi phát hiện tấn công màn hình/video phát lại."""
    sess = EnrollmentSession("session-test-replay-spoof")
    mock_analyzer = MagicMock()
    face = make_valid_face_result()
    face.anti_spoof = AntiSpoofResult(
        is_real=False,
        real_prob=0.08,
        print_prob=0.02,
        replay_prob=0.90,
        spoof_type="replay_attack",
        depth_3d_ok=False,
        message="Phát hiện màn hình/video phát lại",
    )
    mock_analyzer.analyze.return_value = face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "spoof_detected"
    assert "màn hình" in res["message"].lower()
    assert sess.stage == Stage.FAILED


def test_verify_final_capture_bad_pose_rejected():
    """Từ chối khi người dùng quay mặt lệch hoặc cúi/nghiêng quá nhiều khi chụp."""
    sess = EnrollmentSession("session-test-pose")
    mock_analyzer = MagicMock()
    face = make_valid_face_result()
    face.yaw = 25.0  # Quay mặt sang một bên
    mock_analyzer.analyze.return_value = face
    sess.face_analyzer = mock_analyzer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sess.verify_final_capture(dummy_frame)

    assert res["passed"] is False
    assert res["error_type"] == "bad_pose"
    assert sess.stage == Stage.FAILED
