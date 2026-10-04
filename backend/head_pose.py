"""Ước lượng tư thế đầu (yaw / pitch / roll) từ landmark 3D của MediaPipe FaceMesh.

Dựng hệ trục gắn với khuôn mặt trong hệ toạ độ camera (x sang phải ảnh, y hướng xuống,
z hướng ra xa camera):
  - trục X: khoé mắt ngoài trái-ảnh (33) -> phải-ảnh (263)
  - trục Y: giữa trán (10) -> cằm (152), trực giao hoá theo X
  - trục Z = X × Y
Ma trận R = [X Y Z] ≈ I khi nhìn thẳng. Cách này không có nghiệm lật/cực tiểu cục bộ
như solvePnP với mô hình 3D chung.

Quy ước dấu (ảnh GỐC, chưa lật gương):
  yaw   > 0 : người dùng quay đầu sang TRÁI của họ   (mũi lệch về phía phải ảnh gốc)
  pitch > 0 : người dùng ngẩng đầu lên
  roll  > 0 : người dùng nghiêng đầu về vai TRÁI của họ
"""
from __future__ import annotations

import math

import numpy as np

EYE_L, EYE_R, FOREHEAD, CHIN = 33, 263, 10, 152

# Độ nghiêng tự nhiên của trục trán-cằm so với mặt phẳng ảnh khi nhìn thẳng
# (trán gần camera hơn cằm trên mesh chuẩn của MediaPipe ~20°). Hiệu chỉnh để pitch ≈ 0 khi nhìn thẳng.
PITCH_BIAS_DEG = -20.0


def rotation_to_angles(R: np.ndarray) -> tuple[float, float, float]:
    """Trả về (rx, ry, rz) độ với R = Rz·Ry·Rx."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # pragma: no cover - suy biến (quay 90°)
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def face_rotation(pts3d: np.ndarray) -> np.ndarray:
    """pts3d: mảng (N, 3) landmark đã đổi sang pixel (x*w, y*h, z*w)."""
    x = pts3d[EYE_R] - pts3d[EYE_L]
    x /= np.linalg.norm(x) + 1e-9
    y = pts3d[CHIN] - pts3d[FOREHEAD]
    y = y - np.dot(y, x) * x
    y /= np.linalg.norm(y) + 1e-9
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def estimate_head_pose(pts3d: np.ndarray) -> tuple[float, float, float]:
    """Trả về (yaw, pitch, roll) độ theo quy ước người dùng (xem docstring)."""
    rx, ry, rz = rotation_to_angles(face_rotation(np.asarray(pts3d, dtype=np.float64)))
    yaw = -ry
    pitch = -rx - PITCH_BIAS_DEG
    roll = rz
    return yaw, pitch, roll
