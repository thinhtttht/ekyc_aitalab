"""Unit tests cho module Head Pose estimation."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from head_pose import estimate_head_pose, EYE_L, EYE_R, FOREHEAD, CHIN


def make_canonical_face_pts(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Tạo mảng landmark (500, 3) giả lập hình học khuôn mặt."""
    pts = np.zeros((500, 3), dtype=np.float64)

    # Đặt các điểm mốc cơ bản (nhìn thẳng, pitch ~ 0 sau bias)
    pts[EYE_L] = np.array([200.0, 240.0, 0.0])
    pts[EYE_R] = np.array([280.0, 240.0, 0.0])
    # Trán gần hơn cằm ~20 độ
    pts[FOREHEAD] = np.array([240.0, 160.0, -29.0])
    pts[CHIN] = np.array([240.0, 340.0, 36.0])

    # Xoay quanh trục Y (Y hướng xuống): x' = x*c - z*s, z' = x*s + z*c
    rad = np.radians(yaw_deg)
    c, s = np.cos(rad), np.sin(rad)
    R_yaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

    for idx in [EYE_L, EYE_R, FOREHEAD, CHIN]:
        # Quay quanh tâm (240, 240, 0)
        p = pts[idx] - np.array([240.0, 240.0, 0.0])
        p_rot = R_yaw @ p
        pts[idx] = p_rot + np.array([240.0, 240.0, 0.0])

    return pts


def test_straight_face():
    pts = make_canonical_face_pts(yaw_deg=0.0)
    yaw, pitch, roll = estimate_head_pose(pts)
    assert abs(yaw) < 3.0
    assert abs(pitch) < 5.0
    assert abs(roll) < 3.0


def test_turn_left():
    # Quay sang trái người dùng 28 độ
    pts = make_canonical_face_pts(yaw_deg=28.0)
    yaw, pitch, roll = estimate_head_pose(pts)
    assert yaw > 22.0
    assert abs(yaw - 28.0) < 4.0


def test_turn_right():
    # Quay sang phải người dùng 28 độ
    pts = make_canonical_face_pts(yaw_deg=-28.0)
    yaw, pitch, roll = estimate_head_pose(pts)
    assert yaw < -22.0
    assert abs(yaw - (-28.0)) < 4.0
