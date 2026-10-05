"""Unit tests cho luồng máy trạng thái đăng ký EnrollmentSession."""
import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
from camera_quality import ClientStats
from enrollment import EnrollmentSession, Stage
from face_analyzer import FaceResult


class MockFaceAnalyzer:
    """Mock FaceAnalyzer để mô phỏng chính xác các tình huống kiểm thử."""

    def __init__(self, mode="perfect"):
        self.mode = mode
        self.yaw = 0.0
        self.face_h = 0.35
        self.fill = 0.75

    def analyze(self, frame_bgr, oval):
        if self.mode == "no_face":
            return FaceResult(face_count=0)

        if self.mode == "mask":
            return FaceResult(face_count=1, bbox=(0.2, 0.2, 0.8, 0.8), center=(0.5, 0.5), face_h=0.35, fill=0.75, mask_detected=True)

        # Chế độ dynamic
        return FaceResult(
            face_count=1,
            bbox=(0.2, 0.2, 0.8, 0.8),
            center=(0.5, 0.5),
            face_h=self.face_h,
            yaw=self.yaw,
            pitch=0.0,
            roll=0.0,
            fill=self.fill,
            oval_dist=0.6,
            offset=(0.0, 0.0),
            brightness=120.0,
            brightness_mean=120.0,
            brightness_std=30.0,
            side_ratio=1.0,
            sharpness=85.0,
            scale_ratio=0.60,
            corners_inside=True,
            glare_detected=False,
            hand_occlusion=False,
            mask_detected=False,
            sunglasses_detected=False,
            parts_status={"left_eye": True, "right_eye": True, "nose": True, "mouth": True, "left_eyebrow": True, "right_eyebrow": True},
            occluded_part_name=None,
            perspective_ratio=0.6,
        )

    def close(self):
        pass


def test_full_enrollment_pipeline(monkeypatch):
    sess = EnrollmentSession("test-full-session")
    mock_analyzer = MockFaceAnalyzer(mode="dynamic")
    sess.face_analyzer = mock_analyzer

    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)
    frame_step = [0]

    def get_live_frame():
        # Mô phỏng camera thật có biến thiên ánh sáng giữa các frame
        frame_step[0] += 1
        val = 130 + (frame_step[0] % 10)
        return np.full((600, 480, 3), val, dtype=np.uint8)

    # 1. Bắt đầu ở CAMERA_CHECK
    assert sess.stage == Stage.CAMERA_CHECK
    res = sess.process_frame(get_live_frame(), stats)
    assert res["stage"] == Stage.CAMERA_CHECK

    # Giả lập trôi qua 1.6s để hoàn thành CAMERA_CHECK
    sess.stable_start_time = time.time() - 1.6
    res = sess.process_frame(get_live_frame(), stats)
    assert sess.stage == Stage.FACE_QUALITY
    assert res["stage"] == Stage.FACE_QUALITY

    # 2. FACE_QUALITY -> Giữ yên 15 frames liên tiếp (FQA_CONSECUTIVE_FRAMES)
    for _ in range(C.FQA_CONSECUTIVE_FRAMES):
        res = sess.process_frame(get_live_frame(), stats)

    # Đã chốt baseline và chuyển sang thử thách quay đầu 1
    assert sess.baseline_face_h is not None
    first_challenge = sess.challenge_sequence[0]
    assert sess.stage == first_challenge

    # 3. Thực hiện thử thách quay đầu 1
    mock_analyzer.yaw = 28.0 if first_challenge == Stage.TURN_LEFT else -28.0
    for _ in range(C.TURN_HOLD_FRAMES):
        res = sess.process_frame(get_live_frame(), stats)
    assert sess.stage == Stage.RECENTER

    # Nhìn thẳng lại để hoàn thành RECENTER
    mock_analyzer.yaw = 0.0
    res = sess.process_frame(get_live_frame(), stats)
    second_challenge = sess.challenge_sequence[1]
    assert sess.stage == second_challenge

    # 4. Thực hiện thử thách quay đầu 2
    mock_analyzer.yaw = 28.0 if second_challenge == Stage.TURN_LEFT else -28.0
    for _ in range(C.TURN_HOLD_FRAMES):
        res = sess.process_frame(get_live_frame(), stats)
    assert sess.stage == Stage.RECENTER

    # Nhìn thẳng lại -> chuyển sang ZOOM_IN
    mock_analyzer.yaw = 0.0
    res = sess.process_frame(get_live_frame(), stats)
    assert sess.stage == Stage.ZOOM_IN
    assert res["oval"] == C.OVAL_ZOOM

    # 5. ZOOM_IN: Chưa tăng kích thước (growth < 1.25) -> nhắc tiến gần
    res = sess.process_frame(get_live_frame(), stats)
    assert "tiến lại gần" in res["message"] or "125%" in res["message"]

    # Người dùng tiến gần: face_h tăng lên 1.35x
    mock_analyzer.face_h = sess.baseline_face_h * 1.35
    sess.stable_start_time = time.time() - 1.1
    res = sess.process_frame(get_live_frame(), stats)

    # Đạt FLASHING (Chuẩn bị quét ánh sáng màu quang học)
    assert sess.stage == Stage.FLASHING
    assert res["stage"] == Stage.FLASHING
    assert "quét ánh sáng" in res["message"]

    # Áp dụng kết quả xác thực quang học PASS -> chuyển sang CAPTURE
    sess.apply_optical_result(passed=True, correlation=0.89, amplitude=14.2, verdict="LIVENESS_PASS")
    assert sess.stage == Stage.CAPTURE
    assert sess.history["optical_liveness"]["passed"] is True
    assert sess.history["zoom"]["growth_ratio"] >= 1.25


def test_hand_occlusion_rejection():
    sess = EnrollmentSession("test-hand-session")
    sess.stage = Stage.FACE_QUALITY

    class HandOccludedAnalyzer:
        def analyze(self, frame_bgr, oval):
            return FaceResult(
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
                side_ratio=1.0,
                sharpness=40.0,
                hand_occlusion=True,
                mask_detected=False,
                sunglasses_detected=False,
            )
        def close(self): pass

    sess.face_analyzer = HandOccludedAnalyzer()
    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)
    frame = np.full((600, 480, 3), 130, dtype=np.uint8)

    res = sess.process_frame(frame, stats)
    assert res["face_checks"]["no_hand_occlusion"] is False
    assert res["color"] == "red"
    assert "bỏ tay" in res["message"].lower()
    assert res["progress"] == 0.0


def test_part_occlusion_rejection():
    sess = EnrollmentSession("test-parts-session")
    sess.stage = Stage.FACE_QUALITY

    class MouthOccludedAnalyzer:
        def analyze(self, frame_bgr, oval):
            return FaceResult(
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
                side_ratio=1.0,
                sharpness=40.0,
                hand_occlusion=False,
                mask_detected=False,
                sunglasses_detected=False,
                parts_status={
                    "left_eye": True, "right_eye": True,
                    "left_eyebrow": True, "right_eyebrow": True,
                    "nose": True, "mouth": False,
                },
                occluded_part_name="miệng/môi",
                keypoints={
                    "mouth": [[0.4, 0.6], [0.6, 0.6]],
                    "nose": [[0.5, 0.4], [0.5, 0.5]],
                },
            )
        def close(self): pass

    sess.face_analyzer = MouthOccludedAnalyzer()
    stats = ClientStats(fps=30.0, width=1280, height=720, frame_advancing=True)
    frame = np.full((600, 480, 3), 130, dtype=np.uint8)

    res = sess.process_frame(frame, stats)
    assert res["face_checks"]["has_mouth"] is False
    assert res["face_checks"]["has_eyes"] is True
    assert res["color"] == "red"
    assert "che khuất miệng/môi" in res["message"].lower()
    assert res["occluded_part_name"] == "miệng/môi"
    assert res["parts_status"]["mouth"] is False
    assert "mouth" in res["keypoints"]
    assert res["progress"] == 0.0

