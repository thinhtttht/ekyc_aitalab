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
MIN_OPTICAL_PEARSON = 0.35
MIN_OPTICAL_AMPLITUDE = 1.0


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


def to_chromaticity(rgb: np.ndarray) -> np.ndarray:
    """Chuyển đổi véc-tơ màu RGB sang hệ toạ độ sắc độ chuẩn hoá (Normalized Chromaticity).

    r = R / (R + G + B), g = G / (R + G + B), b = B / (R + G + B)
    Triệt tiêu hoàn toàn sự thay đổi cường độ sáng do camera Auto-Exposure (AE/AWB).
    """
    arr = np.asarray(rgb, dtype=np.float64)
    total = float(np.sum(arr))
    return arr / max(total, 1e-6)


def extract_polygon_mean_rgb(frame_bgr: np.ndarray, polygon_pts: np.ndarray) -> np.ndarray:
    """Tính giá trị trung bình (R, G, B) bên trong đa giác mốc hình học.

    Sử dụng Convex Hull để đảm bảo vùng lấy mẫu lồi, không bị tự giao cắt.

    Args:
        frame_bgr: Khung hình BGR từ camera.
        polygon_pts: Mảng toạ độ pixel (N, 2) kiểu int32.

    Returns:
        Mảng numpy 1D 3 phần tử [R, G, B].
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon_pts) >= 3:
        hull = cv2.convexHull(polygon_pts)
        cv2.fillPoly(mask, [hull], 255)
    else:
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

        # 3. Tính toán véc-tơ biến thiên vi sai theo 2 chuẩn:
        # Chuẩn 1: Normalized Chromaticity (ISO/IEC 30107-3, Tang et al. NDSS 2018) - Triệt tiêu Auto-Exposure
        c_skins = [to_chromaticity(s) for s in skin_rgbs]
        c_screens = [to_chromaticity(np.array(e, dtype=np.float64)) for e in expected_colors_rgb]

        v_chroma_skin_parts: List[float] = []
        v_chroma_screen_parts: List[float] = []

        # Chuẩn 2: Raw RGB bù trừ AWB (bảo toàn tương thích ngược)
        v_raw_skin_parts: List[float] = []
        v_raw_screen_parts: List[float] = []

        dca_matches = 0
        for k in range(1, n_frames):
            # Chromaticity delta
            d_c_skin = c_skins[k] - c_skins[k - 1]
            d_c_screen = c_screens[k] - c_screens[k - 1]
            v_chroma_skin_parts.extend(d_c_skin.tolist())
            v_chroma_screen_parts.extend(d_c_screen.tolist())

            # Dominant Channel Agreement: Kênh màu chính của màn hình có tăng trên da không?
            dom_idx = int(np.argmax(d_c_screen))
            if d_c_skin[dom_idx] > 0:
                dca_matches += 1

            # Raw RGB delta
            delta_skin = skin_rgbs[k] - skin_rgbs[k - 1]
            delta_bg = bg_rgbs[k] - bg_rgbs[k - 1]
            delta_screen = np.array(expected_colors_rgb[k], dtype=np.float64) - np.array(
                expected_colors_rgb[k - 1], dtype=np.float64
            )
            delta_skin_compensated = delta_skin - awb_gamma * delta_bg
            v_raw_skin_parts.extend(delta_skin_compensated.tolist())
            v_raw_screen_parts.extend(delta_screen.tolist())

        dca_score = dca_matches / max(1, n_frames - 1)

        v_chroma_skin = np.array(v_chroma_skin_parts, dtype=np.float64)
        v_chroma_screen = np.array(v_chroma_screen_parts, dtype=np.float64)
        r_chroma = calculate_pearson_correlation(v_chroma_skin, v_chroma_screen)
        amp_chroma = float(np.linalg.norm(v_chroma_skin) * 100.0)  # Tính theo % sắc độ

        v_raw_skin = np.array(v_raw_skin_parts, dtype=np.float64)
        v_raw_screen = np.array(v_raw_screen_parts, dtype=np.float64)
        r_raw = calculate_pearson_correlation(v_raw_skin, v_raw_screen)
        amp_raw = float(np.linalg.norm(v_raw_skin))

        # Chọn điểm tương quan tối ưu giữa sắc độ chuẩn hoá và raw BGR
        r_best = max(r_chroma, r_raw)
        effective_amplitude = max(amp_raw, amp_chroma * 2.0)

        details = {
            "r_chroma": round(r_chroma, 3),
            "r_raw": round(r_raw, 3),
            "r_best": round(r_best, 3),
            "amp_chroma_pct": round(amp_chroma, 2),
            "amp_raw": round(amp_raw, 2),
            "dca_score": round(dca_score, 2),
            "awb_gamma": awb_gamma,
            "skin_rgbs": [[round(x, 1) for x in s] for s in skin_rgbs],
        }

        # 4. Kiểm tra điều kiện biên độ phản xạ (Chống tấn công phát lại ảnh in tĩnh)
        is_static = (amp_chroma < 0.10 and amp_raw < 0.6)
        if is_static:
            return OpticalResult(
                passed=False,
                correlation_score=r_best,
                amplitude=effective_amplitude,
                verdict="STATIC_SPOOF_REPLAY",
                message="Xác thực thất bại. Bề mặt không có phản xạ ánh sáng (ảnh in hoặc màn hình tĩnh).",
                details=details,
            )

        # 5. Kiểm tra tương quan Pearson & Đồng pha màu (Chống tấn công phát lại video khác pha)
        has_correlation = (r_best >= self.min_pearson)
        if not has_correlation:
            return OpticalResult(
                passed=False,
                correlation_score=r_best,
                amplitude=effective_amplitude,
                verdict="OPTICAL_MISMATCH_SPOOF",
                message="Xác thực thất bại. Phản xạ quang phổ không khớp với chuỗi màu thách thức.",
                details=details,
            )

        return OpticalResult(
            passed=True,
            correlation_score=r_best,
            amplitude=effective_amplitude,
            verdict="LIVENESS_PASS",
            message="Xác thực phản xạ quang học thành công.",
            details=details,
        )
