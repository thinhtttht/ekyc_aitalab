"""Unit tests cho 4 Tiêu chí Cốt lõi FQA và Cơ chế Kiểm soát Liên tục."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
from face_analyzer import FaceResult, quality_checks, check_continuous_face_quality


def create_valid_face():
    """Tạo đối tượng FaceResult đạt chuẩn toàn bộ tiêu chí FQA."""
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
        brightness_mean=120.0,
        brightness_std=30.0,
        sharpness=80.0,
        scale_ratio=0.60,
        corners_inside=True,
        glare_detected=False,
        hand_occlusion=False,
        mask_detected=False,
        sunglasses_detected=False,
        parts_status={
            "left_eye": True,
            "right_eye": True,
            "nose": True,
            "mouth": True,
            "left_eyebrow": True,
            "right_eyebrow": True,
        },
        occluded_part_name=None,
        perspective_ratio=0.6,
    )


def test_fqa_valid_face():
    face = create_valid_face()
    checks, msg, sev = quality_checks(face)
    assert all(checks.values())
    assert sev == "ok"
    assert "hợp lệ" in msg.lower()

    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face, is_turning=False)
    assert cont_ok is True
    assert cont_sev == "ok"


def test_pillar_1_framing_scale():
    # Trường hợp 1: Quá xa (scale < 0.40)
    face_far = create_valid_face()
    face_far.scale_ratio = 0.35
    checks, msg, sev = quality_checks(face_far)
    assert checks["scale_ok"] is False
    assert sev == "warn"
    assert "lại gần" in msg.lower() or "khung oval" in msg.lower()

    # Trường hợp 2: Quá gần (scale > 0.85)
    face_close = create_valid_face()
    face_close.scale_ratio = 0.90
    checks, msg, sev = quality_checks(face_close)
    assert checks["scale_ok"] is False
    assert sev == "warn"
    assert "ra xa" in msg.lower() or "khung oval" in msg.lower()

    # Trường hợp 3: Viền mặt tràn ra ngoài oval
    face_outside = create_valid_face()
    face_outside.oval_dist = 1.35
    checks, msg, sev = quality_checks(face_outside)
    assert checks["inside_oval"] is False
    assert sev == "warn"
    assert "vào giữa" in msg.lower() or "khung oval" in msg.lower()

    # Trường hợp 4 (regression): mặt căn giữa, scale 0.73 -> góc bbox nhô ra elip
    # nhưng viền mặt thật nằm gọn trong oval -> PHẢI đạt.
    face_fit = create_valid_face()
    face_fit.scale_ratio = 0.73
    face_fit.fill = 0.70
    face_fit.oval_dist = 0.53
    face_fit.corners_inside = False
    checks, msg, sev = quality_checks(face_fit)
    assert checks["inside_oval"] is True


def test_pillar_2_illumination():
    # Quá tối (Mean < 40) -> Báo Đỏ
    face_dark = create_valid_face()
    face_dark.brightness_mean = 32.0
    checks, msg, sev = quality_checks(face_dark)
    assert checks["illumination_ok"] is False
    assert sev == "error"
    assert "tối" in msg.lower()

    # Quá sáng (Mean > 210) -> Báo Đỏ
    face_bright = create_valid_face()
    face_bright.brightness_mean = 230.0
    checks, msg, sev = quality_checks(face_bright)
    assert checks["illumination_ok"] is False
    assert sev == "error"
    assert "sáng" in msg.lower()

    # Ngược sáng / bệt màu (Std < 10) -> Báo Vàng
    face_backlight = create_valid_face()
    face_backlight.brightness_std = 8.0
    checks, msg, sev = quality_checks(face_backlight)
    assert checks["no_backlight"] is False
    assert sev == "warn"
    assert "ngược sáng" in msg.lower() or "chi tiết" in msg.lower()


def test_pillar_3_sharpness():
    # Độ sắc nét đã được bỏ chặn, luôn đạt chuẩn khi có khuôn mặt hợp lệ
    face = create_valid_face()
    face.sharpness = 20.0
    checks, msg, sev = quality_checks(face)
    assert checks["sharpness_ok"] is True


def test_pillar_4_occlusion():
    # Khẩu trang -> Báo Đỏ
    face_mask = create_valid_face()
    face_mask.mask_detected = True
    checks, msg, sev = quality_checks(face_mask)
    assert checks["no_mask"] is False
    assert sev == "error"
    assert "khẩu trang" in msg.lower()

    # Kính râm đen -> Báo Đỏ
    face_sunglasses = create_valid_face()
    face_sunglasses.sunglasses_detected = True
    checks, msg, sev = quality_checks(face_sunglasses)
    assert checks["no_sunglasses"] is False
    assert sev == "error"
    assert "kính râm" in msg.lower() or "kính đen" in msg.lower()

    # Kính lóa phản quang -> Báo Vàng
    face_glare = create_valid_face()
    face_glare.glare_detected = True
    checks, msg, sev = quality_checks(face_glare)
    assert checks["no_glare"] is False
    assert sev == "warn"
    assert "lóa kính" in msg.lower()

    # Tay che mặt -> Báo Đỏ
    face_hand = create_valid_face()
    face_hand.hand_occlusion = True
    checks, msg, sev = quality_checks(face_hand)
    assert checks["no_hand_occlusion"] is False
    assert sev == "error"
    assert "bỏ tay" in msg.lower()

    # Che mắt -> Báo Đỏ
    face_eyes = create_valid_face()
    face_eyes.parts_status["left_eye"] = False
    face_eyes.occluded_part_name = "mắt trái"
    checks, msg, sev = quality_checks(face_eyes)
    assert checks["has_eyes"] is False
    assert sev == "error"
    assert "che khuất" in msg.lower()


def test_continuous_quality_during_turning():
    # Khi xoay đầu: cho phép độ nét thấp hơn (35 thay vì 60) và bỏ qua backlight
    face_turn = create_valid_face()
    face_turn.sharpness = 45.0
    face_turn.brightness_std = 12.0
    face_turn.yaw = 28.0

    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_turn, is_turning=True)
    assert cont_ok is True

    # Nhưng nếu đeo khẩu trang hoặc lấy tay che mặt khi xoay đầu -> Vẫn bị chặn ngay lập tức
    face_turn.hand_occlusion = True
    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_turn, is_turning=True)
    assert cont_ok is False
    assert cont_sev == "error"
    assert "bỏ tay" in cont_msg.lower()


def test_upside_down_detection():
    # Trường hợp lật ngược đầu (mắt ở dưới, miệng ở trên)
    face_inverted = create_valid_face()
    face_inverted.is_upside_down = True

    checks, msg, sev = quality_checks(face_inverted)
    assert checks["not_upside_down"] is False
    assert checks["head_straight"] is False
    assert sev == "error"
    assert "lật ngược" in msg.lower()

    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_inverted)
    assert cont_ok is False
    assert cont_sev == "error"
    assert "lật ngược" in cont_msg.lower()


def test_head_tilt_and_direction_guidance():
    # 1. Nghiêng đầu sang trái (Roll > 10 độ)
    face_roll_left = create_valid_face()
    face_roll_left.roll = 16.0
    checks, msg, sev = quality_checks(face_roll_left)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "nghiêng đầu sang trái" in msg.lower()

    # 2. Nghiêng đầu sang phải (Roll < -10 độ)
    face_roll_right = create_valid_face()
    face_roll_right.roll = -18.0
    checks, msg, sev = quality_checks(face_roll_right)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "nghiêng đầu sang phải" in msg.lower()

    # 3. Nghiêng đầu quá nhiều (Roll > 25 độ)
    face_roll_heavy = create_valid_face()
    face_roll_heavy.roll = 35.0
    checks, msg, sev = quality_checks(face_roll_heavy)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "nghiêng đầu quá nhiều" in msg.lower()

    # 4. Ngẩng đầu quá cao (Pitch > 15 độ)
    face_pitch_up = create_valid_face()
    face_pitch_up.pitch = 22.0
    checks, msg, sev = quality_checks(face_pitch_up)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "ngẩng đầu" in msg.lower()

    # 5. Cúi đầu quá thấp (Pitch < -15 độ)
    face_pitch_down = create_valid_face()
    face_pitch_down.pitch = -20.0
    checks, msg, sev = quality_checks(face_pitch_down)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "cúi đầu" in msg.lower()

    # 6. Quay mặt sang trái (Yaw > 12 độ)
    face_yaw_left = create_valid_face()
    face_yaw_left.yaw = 18.0
    checks, msg, sev = quality_checks(face_yaw_left)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "quay mặt sang trái" in msg.lower()

    # 7. Quay mặt sang phải (Yaw < -12 độ)
    face_yaw_right = create_valid_face()
    face_yaw_right.yaw = -19.0
    checks, msg, sev = quality_checks(face_yaw_right)
    assert checks["head_straight"] is False
    assert sev == "warn"
    assert "quay mặt sang phải" in msg.lower()


def test_distance_guidance_overflow_and_small():
    # Bị tràn khung khi mặt to -> Hướng dẫn lùi ra xa
    face_overflow = create_valid_face()
    face_overflow.oval_dist = 1.30
    face_overflow.scale_ratio = 0.78
    face_overflow.fill = 0.86
    checks, msg, sev = quality_checks(face_overflow)
    assert checks["inside_oval"] is False
    assert sev == "warn"
    assert "lùi ra xa" in msg.lower()

    # Bị lệch ngoài khung khi mặt nhỏ -> Hướng dẫn tiến lại gần
    face_small_outside = create_valid_face()
    face_small_outside.oval_dist = 1.30
    face_small_outside.scale_ratio = 0.44
    face_small_outside.fill = 0.50
    checks, msg, sev = quality_checks(face_small_outside)
    assert checks["inside_oval"] is False
    assert sev == "warn"
    assert "lại gần" in msg.lower()


def test_mask_false_positive_with_clear_mouth_and_nose():
    """Người dùng không đeo khẩu trang (mũi và miệng rõ ràng) dù cằm có bóng đổ cũng không bị báo giả."""
    face = create_valid_face()
    face.parts_status["mouth"] = True
    face.parts_status["nose"] = True
    face.mask_detected = False

    checks, msg, sev = quality_checks(face)
    assert checks["no_mask"] is True
    assert sev != "error"


def test_multiple_faces_filtering_concept():
    """Kiểm tra logic lọc khuôn mặt: artifact nhỏ ở background không biến thành nhiều người."""
    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class DummyFace:
        def __init__(self, x0, y0, x1, y1):
            self.landmark = [DummyLandmark(x0, y0), DummyLandmark(x1, y1)]

    # Mặt chính: 0.20 -> 0.80 (rộng 0.6, cao 0.7 -> diện tích 0.42)
    main_f = DummyFace(0.2, 0.1, 0.8, 0.8)
    # Nhiễu ở nền: 0.01 -> 0.05 (diện tích 0.0016 < 0.025 và < 20% mặt chính)
    noise_f = DummyFace(0.01, 0.01, 0.05, 0.05)

    all_faces = [main_f, noise_f]
    def face_area(f):
        xs = [p.x for p in f.landmark]
        ys = [p.y for p in f.landmark]
        return float((max(xs) - min(xs)) * (max(ys) - min(ys)))

    sorted_faces = sorted(all_faces, key=face_area, reverse=True)
    primary_area = face_area(sorted_faces[0])
    valid_faces = [
        f for f in sorted_faces
        if face_area(f) >= max(0.025, primary_area * 0.20)
    ]
    assert len(valid_faces) == 1  # Chỉ còn lại 1 mặt chính duy nhất!

