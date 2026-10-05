"""Module phân tích phản xạ quang học mô da người sống (Active Optical Liveness Detection).

Tuân thủ chuẩn ASD-STE100, W3C WCAG 2.1, và ISO/IEC 30107-3.
Tham chiếu học thuật: Tang et al. (NDSS 2018).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

# Mốc MediaPipe chuẩn cho 3 vùng da mặt
FOREHEAD_LANDMARKS = [10, 67, 109, 108, 151, 337, 297, 284]
LEFT_CHEEK_LANDMARKS = [116, 123, 147, 213, 192, 214]
RIGHT_CHEEK_LANDMARKS = [345, 352, 376, 433, 416, 434]

# Hệ số bù trừ Auto White Balance (AWB) thực nghiệm
DEFAULT_AWB_GAMMA = 0.45
MIN_OPTICAL_PEARSON = 0.65
MIN_OPTICAL_AMPLITUDE = 4.0


@dataclass
class OpticalResult:
    """Kết quả đánh giá phản xạ quang phổ thời gian thực."""
    passed: bool
    correlation_score: float
    amplitude: float
    verdict: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def calculate_pearson_correlation(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính hệ số tương quan Pearson giữa hai véc-tơ đa chiều.

    Args:
        vec_a: Véc-tơ thứ nhất (biến thiên màu da mặt).
        vec_b: Véc-tơ thứ hai (biến thiên màu nguồn phát màn hình).

    Returns:
        Hệ số tương quan r nằm trong khoảng [-1.0, 1.0].
    """
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0

    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    diff_a = a - mean_a
    diff_b = b - mean_b

    var_a = np.sum(diff_a ** 2)
    var_b = np.sum(diff_b ** 2)

    denominator = np.sqrt(var_a * var_b)
    if denominator < 1e-8:
        return 0.0

    r = float(np.sum(diff_a * diff_b) / denominator)
    return max(-1.0, min(1.0, r))


def extract_polygon_mean_rgb(frame_bgr: np.ndarray, polygon_pts: np.ndarray) -> np.ndarray:
    """Tính giá trị trung bình (R, G, B) bên trong đa giác mốc hình học.

    Args:
        frame_bgr: Khung hình BGR từ camera.
        polygon_pts: Mảng toạ độ pixel (N, 2) kiểu int32.

    Returns:
        Mảng numpy 1D 3 phần tử [R, G, B].
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)

    mean_val = cv2.mean(frame_bgr, mask=mask)[:3]  # (B, G, R)
    # Đổi sang thứ tự RGB
    return np.array([mean_val[2], mean_val[1], mean_val[0]], dtype=np.float64)


def extract_background_mean_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Trích xuất vùng nền đối chứng ở góc khung hình (15% trên cùng bên trái).

    Args:
        frame_bgr: Khung hình BGR từ camera.

    Returns:
        Mảng numpy 1D 3 phần tử [R, G, B].
    """
    h, w = frame_bgr.shape[:2]
    bg_h = max(10, int(0.15 * h))
    bg_w = max(10, int(0.15 * w))
    bg_patch = frame_bgr[0:bg_h, 0:bg_w]

    mean_bgr = np.mean(bg_patch, axis=(0, 1))
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


class OpticalAnalyzer:
    """Bộ phân tích phản xạ quang phổ trên mô da khuôn mặt."""

    def __init__(self, min_pearson: float = MIN_OPTICAL_PEARSON, min_amplitude: float = MIN_OPTICAL_AMPLITUDE) -> None:
        self.min_pearson = min_pearson
        self.min_amplitude = min_amplitude
        self._mesh = None
        if mp is not None:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def close(self) -> None:
        """Giải phóng tài nguyên MediaPipe."""
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def extract_skin_and_background_rgb(
        self, frame_bgr: np.ndarray, custom_landmarks: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Trích xuất véc-tơ màu RGB của da mặt và nền trên một khung hình.

        Args:
            frame_bgr: Khung hình camera định dạng BGR.
            custom_landmarks: Toạ độ mốc tuỳ chọn (N, 2 hoặc N, 3) dùng cho unit tests.

        Returns:
            Tuple (skin_rgb, bg_rgb). Nếu không tìm thấy mặt, skin_rgb là None.
        """
        h, w = frame_bgr.shape[:2]
        bg_rgb = extract_background_mean_rgb(frame_bgr)

        # 1. Lấy toạ độ landmarks
        landmarks_pixel: Optional[np.ndarray] = None
        if custom_landmarks is not None:
            if custom_landmarks.max() <= 1.0:
                landmarks_pixel = (custom_landmarks[:, :2] * np.array([w, h])).astype(np.int32)
            else:
                landmarks_pixel = custom_landmarks[:, :2].astype(np.int32)
        elif self._mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._mesh.process(rgb)
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in face.landmark], dtype=np.int32)
                landmarks_pixel = pts

        if landmarks_pixel is None:
            return None, bg_rgb

        # 2. Trích xuất màu trên 3 vùng da mặt
        skin_rgb_samples: List[np.ndarray] = []
        for landmark_indices in (FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS):
            if max(landmark_indices) < len(landmarks_pixel):
                roi_poly = landmarks_pixel[landmark_indices]
                roi_rgb = extract_polygon_mean_rgb(frame_bgr, roi_poly)
                skin_rgb_samples.append(roi_rgb)

        if not skin_rgb_samples:
            return None, bg_rgb

        mean_skin_rgb = np.mean(skin_rgb_samples, axis=0)
        return mean_skin_rgb, bg_rgb

    def analyze_sequence(
        self,
        frames_bgr: List[np.ndarray],
        expected_colors_rgb: List[Tuple[int, int, int]],
        custom_landmarks_list: Optional[List[Optional[np.ndarray]]] = None,
        awb_gamma: float = DEFAULT_AWB_GAMMA,
    ) -> OpticalResult:
        """Phân tích chuỗi phản xạ quang học qua các khung hình nháy màu.

        Args:
            frames_bgr: Danh sách các khung hình BGR (thường là 3 khung hình).
            expected_colors_rgb: Danh sách các màu RGB nguồn phát từ màn hình.
            custom_landmarks_list: Danh sách mốc landmarks kiểm thử (nếu có).
            awb_gamma: Hệ số bù trừ độ lệch cân bằng trắng AWB.

        Returns:
            OpticalResult với verdict, correlation_score và thông báo.
        """
        n_frames = len(frames_bgr)
        if n_frames < 2 or len(expected_colors_rgb) != n_frames:
            return OpticalResult(
                passed=False,
                correlation_score=0.0,
                amplitude=0.0,
                verdict="INVALID_SEQUENCE_LENGTH",
                message="Số lượng khung hình không khớp với chuỗi thách thức màu.",
            )

        skin_rgbs: List[np.ndarray] = []
        bg_rgbs: List[np.ndarray] = []

        for i, frame in enumerate(frames_bgr):
            cl = custom_landmarks_list[i] if custom_landmarks_list and i < len(custom_landmarks_list) else None
            skin_rgb, bg_rgb = self.extract_skin_and_background_rgb(frame, custom_landmarks=cl)
            if skin_rgb is None:
                return OpticalResult(
                    passed=False,
                    correlation_score=0.0,
                    amplitude=0.0,
                    verdict="FACE_LOST",
                    message="Không tìm thấy khuôn mặt trong suốt quá trình quét ánh sáng.",
                    details={"failed_frame_index": i},
                )
            skin_rgbs.append(skin_rgb)
            bg_rgbs.append(bg_rgb)

        # 3. Tính toán véc-tơ biến thiên vi sai
        v_skin_parts: List[float] = []
        v_screen_parts: List[float] = []

        for k in range(1, n_frames):
            delta_skin = skin_rgbs[k] - skin_rgbs[k - 1]
            delta_bg = bg_rgbs[k] - bg_rgbs[k - 1]
            delta_screen = np.array(expected_colors_rgb[k], dtype=np.float64) - np.array(
                expected_colors_rgb[k - 1], dtype=np.float64
            )

            # Bù trừ hiện tượng AWB
            delta_skin_compensated = delta_skin - awb_gamma * delta_bg

            v_skin_parts.extend(delta_skin_compensated.tolist())
            v_screen_parts.extend(delta_screen.tolist())

        v_skin = np.array(v_skin_parts, dtype=np.float64)
        v_screen = np.array(v_screen_parts, dtype=np.float64)

        amplitude = float(np.linalg.norm(v_skin))
        correlation = calculate_pearson_correlation(v_skin, v_screen)

        details = {
            "v_skin": [round(x, 2) for x in v_skin.tolist()],
            "v_screen": [round(x, 2) for x in v_screen.tolist()],
            "amplitude": round(amplitude, 2),
            "correlation": round(correlation, 3),
            "awb_gamma": awb_gamma,
        }

        # 4. Kiểm tra điều kiện biên độ phản xạ (Chống tấn công phát lại ảnh in tĩnh)
        if amplitude < self.min_amplitude:
            return OpticalResult(
                passed=False,
                correlation_score=correlation,
                amplitude=amplitude,
                verdict="STATIC_SPOOF_REPLAY",
                message="Xác thực thất bại. Bề mặt không có phản xạ ánh sáng (ảnh in hoặc màn hình tĩnh).",
                details=details,
            )

        # 5. Kiểm tra tương quan Pearson (Chống tấn công phát lại video khác pha)
        if correlation < self.min_pearson:
            return OpticalResult(
                passed=False,
                correlation_score=correlation,
                amplitude=amplitude,
                verdict="OPTICAL_MISMATCH_SPOOF",
                message="Xác thực thất bại. Phản xạ quang phổ không khớp với chuỗi màu thách thức.",
                details=details,
            )

        return OpticalResult(
            passed=True,
            correlation_score=correlation,
            amplitude=amplitude,
            verdict="LIVENESS_PASS",
            message="Xác thực phản xạ quang học thành công.",
            details=details,
        )
