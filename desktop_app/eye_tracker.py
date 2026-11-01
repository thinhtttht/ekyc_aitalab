"""Eye Tracker and Head Pose Attention Analyzer using MediaPipe Face Mesh.
Adapted with high-precision 3D Head Pose Landmark Tracking, Iris Gaze, and 9-Point Fullscreen Calibration.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Ensure backend directory is in sys.path for head_pose
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from head_pose import estimate_head_pose
except ImportError:
    estimate_head_pose = None


class OneEuroFilter:
    """Bộ lọc thích ứng One Euro Filter: Triệt tiêu rung giật vi thể, giữ zero-lag."""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.04, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, t: float, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return x

        t_e = t - self.t_prev
        if t_e <= 1e-5:
            return self.x_prev

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


@dataclass
class GazeResult:
    """Kết quả phân tích ánh mắt và góc xoay đầu từ landmark khuôn mặt."""
    has_face: bool = False
    is_looking_away: bool = False
    looking_away_reason: str = ""
    head_direction_label: str = "Chính diện"  # Chính diện | Quay Trái | Quay Phải | Cúi Đầu | Ngửa Đầu | Nghiêng Đầu
    gaze_screen_norm: Tuple[float, float] = (0.5, 0.5)  # (x, y) [0.0, 1.0]
    gaze_screen_px: Tuple[int, int] = (0, 0)
    iris_left_px: Optional[Tuple[int, int]] = None
    iris_right_px: Optional[Tuple[int, int]] = None
    nose_ray_2d: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # (nose_start, ray_end)
    ear: float = 0.3
    is_blinking: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    distance_cm: float = 50.0
    is_calibrated: bool = False


class EyeTracker:
    """Mô-đun nhận diện ánh mắt và giám sát tư thế quay đầu (Head Pose Landmark Tracking)."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        # Landmark MediaPipe định danh mắt và tròng đen
        self.RIGHT_IRIS = 468
        self.RIGHT_EYE_CORNER_L = 33
        self.RIGHT_EYE_CORNER_R = 133
        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145
        self.RIGHT_EAR_PTS = [33, 160, 158, 133, 153, 144]

        self.LEFT_IRIS = 473
        self.LEFT_EYE_CORNER_L = 362
        self.LEFT_EYE_CORNER_R = 263
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374
        self.LEFT_EAR_PTS = [362, 385, 387, 263, 373, 380]

        # 6 điểm 3D metric (mm) dùng để giải PnP tính tia nhìn 3D của mũi
        self.HEAD_POSE_LMS = [1, 199, 33, 263, 61, 291]
        self.model_3d_metric = np.array([
            (0.0, 0.0, 0.0),        # Đỉnh mũi
            (0.0, -65.0, -25.0),    # Cằm
            (-32.0, 32.0, -30.0),   # Khóe mắt phải
            (32.0, 32.0, -30.0),    # Khóe mắt trái
            (-25.0, -35.0, -20.0),  # Khóe miệng phải
            (25.0, -35.0, -20.0),   # Khóe miệng trái
        ], dtype=np.float64)

        self.HUMAN_IPD_MM = 63.0  # Khoảng cách 2 đồng tử trung bình (63 mm)
        self.EAR_BLINK_THRESHOLD = 0.18

        # Bộ lọc khử rung mượt mà
        self.filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.04)
        self.filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.04)
        self.filter_yaw = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self.filter_pitch = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self.filter_roll = OneEuroFilter(min_cutoff=1.0, beta=0.05)

        # Trạng thái hiệu chuẩn 9 điểm
        self.is_calibrating = False
        self.calib_step = 0  # 0 đến 8
        self.calib_samples: List[List[np.ndarray]] = [[] for _ in range(9)]
        self.calib_step_start_time = 0.0
        self.calib_duration_per_point = 1.3  # giây mỗi điểm
        self.is_calibrated = False
        self.regression_weights_x: Optional[np.ndarray] = None
        self.regression_weights_y: Optional[np.ndarray] = None

        # Trạng thái giám sát vi phạm nhìn ra ngoài dựa trên góc xoay đầu (Head Pose)
        self.yaw_threshold = 16.0          # Ngưỡng góc quay sang Trái / Phải (độ)
        self.pitch_down_threshold = 14.0   # Ngưỡng góc cúi đầu xuống (độ)
        self.pitch_up_threshold = 16.0     # Ngưỡng góc ngửa đầu lên (độ)
        self.roll_threshold = 18.0         # Ngưỡng góc nghiêng đầu (độ)

        self.away_debounce_timer = 0.0
        self.away_debounce_sec = 0.12  # Nhạy bén ~120ms khi quay đầu
        self.blink_start_time = 0.0

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_ear(self, landmarks, ear_indices: List[int], img_w: int, img_h: int) -> float:
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in ear_indices]
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h = self._dist(pts[0], pts[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def calculate_distance_cm(self, landmarks, img_w: int, img_h: int, focal_length: float) -> float:
        p_r = (landmarks[self.RIGHT_IRIS].x * img_w, landmarks[self.RIGHT_IRIS].y * img_h)
        p_l = (landmarks[self.LEFT_IRIS].x * img_w, landmarks[self.LEFT_IRIS].y * img_h)
        d_px = self._dist(p_r, p_l)
        if d_px < 5.0:
            return 50.0
        z_mm = (focal_length * self.HUMAN_IPD_MM) / d_px
        return float(np.clip(z_mm / 10.0, 30.0, 95.0))

    def solve_head_pose_and_ray(self, landmarks, img_w: int, img_h: int):
        """Tính góc đầu 3D kết hợp chiếu tia nhìn 3D từ mũi (Nose Ray)."""
        image_points = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h) for i in self.HEAD_POSE_LMS
        ], dtype=np.float64)

        focal_length = img_w * 0.85
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_3d_metric, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        nose_ray = None
        if success:
            # Chiếu tia 3D dài 90mm từ chóp mũi ra không gian
            nose_end_point3D = np.array([(0.0, 0.0, -90.0)], dtype=np.float64)
            proj_pts, _ = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)
            nose_pt = (int(landmarks[1].x * img_w), int(landmarks[1].y * img_h))
            ray_end = (int(proj_pts[0][0][0]), int(proj_pts[0][0][1]))
            nose_ray = (nose_pt, ray_end)

        # Sử dụng giải thuật estimate_head_pose trực giao 3D chuẩn xác từ facial landmarks
        pts3d = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks], dtype=np.float64)
        if estimate_head_pose is not None:
            # Quy ước: người dùng quay đầu sang TRÁI của họ -> Yaw > 0; PHẢI -> Yaw < 0
            y, p, r = estimate_head_pose(pts3d)
            yaw = float(y)
            pitch = float(p)
            roll = float(r)
        else:
            # Fallback solvePnP Rodrigues
            if not success:
                return 0.0, 0.0, 0.0, focal_length, None
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            if sy > 1e-6:
                pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
                yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
            else:
                pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
                yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                roll = 0.0

        return pitch, yaw, roll, focal_length, nose_ray

    def extract_features(self, landmarks, img_w: int, img_h: int):
        """Trích xuất vector 11 chiều và tọa độ gaze chuẩn hóa [0.0, 1.0]."""
        pitch, yaw, roll, focal_length, nose_ray = self.solve_head_pose_and_ray(landmarks, img_w, img_h)
        dist_cm = self.calculate_distance_cm(landmarks, img_w, img_h, focal_length)

        # Đo liếc mắt
        r_iris = (landmarks[self.RIGHT_IRIS].x * img_w, landmarks[self.RIGHT_IRIS].y * img_h)
        r_cL = (landmarks[self.RIGHT_EYE_CORNER_L].x * img_w, landmarks[self.RIGHT_EYE_CORNER_L].y * img_h)
        r_cR = (landmarks[self.RIGHT_EYE_CORNER_R].x * img_w, landmarks[self.RIGHT_EYE_CORNER_R].y * img_h)
        r_T = (landmarks[self.RIGHT_EYE_TOP].x * img_w, landmarks[self.RIGHT_EYE_TOP].y * img_h)
        r_B = (landmarks[self.RIGHT_EYE_BOTTOM].x * img_w, landmarks[self.RIGHT_EYE_BOTTOM].y * img_h)

        l_iris = (landmarks[self.LEFT_IRIS].x * img_w, landmarks[self.LEFT_IRIS].y * img_h)
        l_cL = (landmarks[self.LEFT_EYE_CORNER_L].x * img_w, landmarks[self.LEFT_EYE_CORNER_L].y * img_h)
        l_cR = (landmarks[self.LEFT_EYE_CORNER_R].x * img_w, landmarks[self.LEFT_EYE_CORNER_R].y * img_h)
        l_T = (landmarks[self.LEFT_EYE_TOP].x * img_w, landmarks[self.LEFT_EYE_TOP].y * img_h)
        l_B = (landmarks[self.LEFT_EYE_BOTTOM].x * img_w, landmarks[self.LEFT_EYE_BOTTOM].y * img_h)

        def ratio_calc(iris, cL, cR, top, bot):
            min_x, max_x = min(cL[0], cR[0]), max(cL[0], cR[0])
            min_y, max_y = min(top[1], bot[1]), max(top[1], bot[1])
            rx = (iris[0] - min_x) / (max_x - min_x + 1e-6) - 0.5
            ry = (iris[1] - min_y) / (max_y - min_y + 1e-6) - 0.5
            return rx, ry

        rx1, ry1 = ratio_calc(r_iris, r_cL, r_cR, r_T, r_B)
        rx2, ry2 = ratio_calc(l_iris, l_cL, l_cR, l_T, l_B)
        eye_off_x = (rx1 + rx2) / 2.0
        eye_off_y = (ry1 + ry2) / 2.0

        # Khử góc nghiêng đầu (Roll)
        roll_rad = math.radians(roll)
        derot_eye_x = eye_off_x * math.cos(-roll_rad) - eye_off_y * math.sin(-roll_rad)
        derot_eye_y = eye_off_x * math.sin(-roll_rad) + eye_off_y * math.cos(-roll_rad)

        # Chuẩn hóa sang tỉ lệ [0.0, 1.0]
        fused_gx = derot_eye_x * 1.8 + (yaw / 25.0) * 0.40 + 0.5
        fused_gy = derot_eye_y * 1.8 + (pitch / 25.0) * 0.40 + 0.5
        norm_x = float(np.clip(fused_gx, 0.0, 1.0))
        norm_y = float(np.clip(fused_gy, 0.0, 1.0))

        feat = np.array([
            1.0,
            norm_x,
            norm_y,
            derot_eye_x,
            derot_eye_y,
            yaw / 30.0,
            pitch / 30.0,
            dist_cm / 50.0,
            norm_x ** 2,
            norm_y ** 2,
            norm_x * norm_y
        ], dtype=np.float64)

        return (feat, norm_x, norm_y, pitch, yaw, roll, dist_cm,
                (int(r_iris[0]), int(r_iris[1])), (int(l_iris[0]), int(l_iris[1])), nose_ray)

    def start_calibration(self):
        """Khởi động quy trình cân chỉnh 9 điểm."""
        self.is_calibrating = True
        self.calib_step = 0
        self.calib_samples = [[] for _ in range(9)]
        self.calib_step_start_time = time.time()
        self.is_calibrated = False
        self.filter_x.reset()
        self.filter_y.reset()

    def train_regression(self, target_points_norm: List[Tuple[float, float]]) -> bool:
        """Huấn luyện mô hình hồi quy Ridge L2 từ 9 điểm mốc."""
        X_list, Y_list_x, Y_list_y = [], [], []

        for pt_idx in range(9):
            target_pt = target_points_norm[pt_idx]
            samples = self.calib_samples[pt_idx]
            if len(samples) < 5:
                continue
            for feat in samples:
                X_list.append(feat)
                Y_list_x.append(target_pt[0])
                Y_list_y.append(target_pt[1])

        if len(X_list) < 25:
            self.is_calibrating = False
            return False

        X = np.array(X_list)
        Y_x = np.array(Y_list_x)
        Y_y = np.array(Y_list_y)

        l2_reg = 0.05
        reg_matrix = l2_reg * np.eye(X.shape[1])
        reg_matrix[0, 0] = 0.0

        inv_term = np.linalg.pinv(X.T @ X + reg_matrix)
        self.regression_weights_x = inv_term @ X.T @ Y_x
        self.regression_weights_y = inv_term @ X.T @ Y_y
        self.is_calibrated = True
        self.is_calibrating = False
        return True

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        screen_size: Tuple[int, int],
        calibration_targets: Optional[List[Tuple[float, float]]] = None
    ) -> GazeResult:
        """Xử lý frame camera và phân tích sự chú ý dựa trên Landmark Head Pose & Iris Tracking."""
        now_t = time.time()
        h, w = frame_bgr.shape[:2]
        scr_w, scr_h = screen_size

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            # Không phát hiện khuôn mặt thí sinh
            return GazeResult(
                has_face=False,
                is_looking_away=True,
                head_direction_label="Mất khuôn mặt",
                looking_away_reason="Không phát hiện khuôn mặt thí sinh trước camera!",
                is_calibrated=self.is_calibrated,
            )

        landmarks = results.multi_face_landmarks[0].landmark

        # Tính EAR đo chớp mắt
        ear_r = self.calculate_ear(landmarks, self.RIGHT_EAR_PTS, w, h)
        ear_l = self.calculate_ear(landmarks, self.LEFT_EAR_PTS, w, h)
        ear = (ear_r + ear_l) / 2.0
        is_blinking = (ear < self.EAR_BLINK_THRESHOLD)

        (feat, norm_x, norm_y, pitch, yaw, roll,
         dist_cm, r_iris_px, l_iris_px, nose_ray) = self.extract_features(landmarks, w, h)

        # Lọc chuyển động mượt mà cho góc đầu
        smooth_yaw = self.filter_yaw(now_t, yaw)
        smooth_pitch = self.filter_pitch(now_t, pitch)
        smooth_roll = self.filter_roll(now_t, roll)

        # Xử lý quy trình hiệu chuẩn nếu đang trong tiến trình
        if self.is_calibrating:
            if not is_blinking:
                self.calib_samples[self.calib_step].append(feat)

            elapsed = now_t - self.calib_step_start_time
            if elapsed >= self.calib_duration_per_point:
                self.calib_step += 1
                self.calib_step_start_time = now_t
                if self.calib_step >= 9:
                    if calibration_targets is not None:
                        self.train_regression(calibration_targets)
                    else:
                        # Mặc định lưới 9 điểm Fullscreen
                        targets = [
                            (0.06, 0.08), (0.50, 0.08), (0.94, 0.08),
                            (0.06, 0.50), (0.50, 0.50), (0.94, 0.50),
                            (0.06, 0.92), (0.50, 0.92), (0.94, 0.92),
                        ]
                        self.train_regression(targets)

        # Tính toán tọa độ ánh mắt trên màn hình
        if self.is_calibrated and self.regression_weights_x is not None:
            pred_norm_x = float(feat @ self.regression_weights_x)
            pred_norm_y = float(feat @ self.regression_weights_y)
        else:
            pred_norm_x = norm_x
            pred_norm_y = norm_y

        # Lọc chuyển động mượt mà
        filtered_norm_x = self.filter_x(now_t, pred_norm_x)
        filtered_norm_y = self.filter_y(now_t, pred_norm_y)

        # Chuyển đổi thành Pixel màn hình
        screen_px_x = int(filtered_norm_x * scr_w)
        screen_px_y = int(filtered_norm_y * scr_h)

        # =========================================================================
        # TIÊU CHUẨN XÁC ĐỊNH NHÌN RA NGOÀI DỰA TRÊN LANDMARK HEAD POSE 3D (CỐT LÕI)
        # Góc xoay đầu từ landmark là chỉ số ổn định và chính xác tuyệt đối:
        # - Quay đầu sang TRÁI của người dùng: Yaw >= +13.0°
        # - Quay đầu sang PHẢI của người dùng: Yaw <= -13.0°
        # - Cúi gằm mặt (nhìn tài liệu / điện thoại): Pitch <= -12.0°
        # - Ngửa mặt lên trời: Pitch >= +14.0°
        # - Nghiêng đầu lệch: |Roll| >= 16.0°
        # =========================================================================
        raw_looking_away = False
        reason = ""
        head_label = "Chính diện"

        if smooth_yaw >= self.yaw_threshold:
            raw_looking_away = True
            head_label = "Quay Trái"
            reason = f"Thí sinh đang QUAY ĐẦU SANG TRÁI ({smooth_yaw:.1f}° > {self.yaw_threshold:.1f}°)"
        elif smooth_yaw <= -self.yaw_threshold:
            raw_looking_away = True
            head_label = "Quay Phải"
            reason = f"Thí sinh đang QUAY ĐẦU SANG PHẢI ({abs(smooth_yaw):.1f}° > {self.yaw_threshold:.1f}°)"
        elif smooth_pitch <= -self.pitch_down_threshold:
            raw_looking_away = True
            head_label = "Cúi Đầu"
            reason = f"Thí sinh đang CÚI ĐẦU XUỐNG DƯỚI ({smooth_pitch:.1f}°)"
        elif smooth_pitch >= self.pitch_up_threshold:
            raw_looking_away = True
            head_label = "Ngửa Đầu"
            reason = f"Thí sinh đang NGỬA ĐẦU LÊN TRÊN ({smooth_pitch:.1f}°)"
        elif abs(smooth_roll) >= self.roll_threshold:
            raw_looking_away = True
            head_label = "Nghiêng Đầu"
            side = "trái" if smooth_roll > 0 else "phải"
            reason = f"Thí sinh đang NGHIÊNG ĐẦU sang {side} ({abs(smooth_roll):.1f}°)"
        elif is_blinking:
            if self.blink_start_time == 0.0:
                self.blink_start_time = now_t
            elif (now_t - self.blink_start_time) > 1.5:
                raw_looking_away = True
                head_label = "Nhắm Mắt"
                reason = "Thí sinh nhắm mắt quá lâu"
        else:
            self.blink_start_time = 0.0

        # Phản ứng siêu nhạy ~120ms khi quay đầu
        if raw_looking_away:
            self.away_debounce_timer += 0.045
            is_looking_away = self.away_debounce_timer >= self.away_debounce_sec
        else:
            self.away_debounce_timer = max(0.0, self.away_debounce_timer - 0.06)
            is_looking_away = False
            reason = "Đang tập trung nhìn vào bài thi"
            head_label = "Chính diện"

        return GazeResult(
            has_face=True,
            is_looking_away=is_looking_away,
            head_direction_label=head_label,
            looking_away_reason=reason,
            gaze_screen_norm=(filtered_norm_x, filtered_norm_y),
            gaze_screen_px=(screen_px_x, screen_px_y),
            iris_right_px=r_iris_px,
            iris_left_px=l_iris_px,
            nose_ray_2d=nose_ray,
            ear=ear,
            is_blinking=is_blinking,
            pitch=smooth_pitch,
            yaw=smooth_yaw,
            roll=smooth_roll,
            distance_cm=dist_cm,
            is_calibrated=self.is_calibrated,
        )
