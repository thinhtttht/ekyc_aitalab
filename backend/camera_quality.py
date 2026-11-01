"""(a) Kiểm tra chất lượng Camera - REQ-BE-01.

- Độ phân giải & FPS: do trình duyệt đo trên luồng gốc và gửi kèm (client_stats).
- Khung đen / bị che: độ sáng trung bình quá thấp.
- Đóng băng: nhiều frame liên tiếp giống hệt nhau, hoặc trình duyệt báo không còn frame mới.
- Độ sáng tổng thể & độ nhiễu (ước lượng Immerkær 1996).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config as C

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise(gray: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu Gauss của ảnh xám (J. Immerkær, 1996)."""
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REFLECT)
    total = np.abs(conv[1:-1, 1:-1]).sum()
    return float(total * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))


def global_sharpness(gray: np.ndarray) -> float:
    """Phương sai Laplacian - càng lớn ảnh càng nét."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def auto_enhance_brightness(
    frame_bgr: np.ndarray,
    target_mean: Optional[float] = None,
    low_threshold: Optional[float] = None,
    black_threshold: Optional[float] = None,
) -> np.ndarray:
    """Tự động nâng sáng và cân bằng tương phản thích ứng (Adaptive CLAHE + Gamma) cho ảnh 2D RGB.

    Khi người dùng ở trong môi trường thiếu sáng hoặc camera bị tối:
      1. Kiểm tra độ sáng trung bình của khung hình.
      2. Nếu [black_threshold <= mean < low_threshold], tự động chuyển sang không gian màu LAB.
      3. Áp dụng CLAHE lên kênh L (Lightness) để tăng chi tiết cục bộ không gây cháy sáng.
      4. Tính toán Gamma thích ứng để đưa độ sáng trung bình lên mức lý tưởng target_mean (~115.0).
      5. Giữ nguyên 100% màu sắc và sắc tố da tự nhiên (kênh A và B).
    """
    if not getattr(C, "AUTO_ENHANCE_BRIGHTNESS", True):
        return frame_bgr
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    tgt = target_mean if target_mean is not None else getattr(C, "TARGET_IDEAL_BRIGHTNESS", 115.0)
    low_th = low_threshold if low_threshold is not None else getattr(C, "LOW_BRIGHTNESS_THRESHOLD", 80.0)
    blk_th = black_threshold if black_threshold is not None else getattr(C, "BLACK_FRAME_MEAN", 18.0)

    # Tính độ sáng trung bình trên ảnh xám
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    current_mean = float(gray.mean())

    # Nếu camera bị che / đen hoàn toàn (< 18) hoặc đã đủ sáng (>= 80), giữ nguyên
    if current_mean < blk_th or current_mean >= low_th:
        return frame_bgr

    # Tách kênh LAB
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. CLAHE trên kênh L
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # 2. Adaptive Gamma Curve
    clahe_mean = max(1.0, float(l_clahe.mean()))
    if clahe_mean < tgt:
        # gamma < 1.0 làm sáng vùng tối
        gamma = float(np.clip(np.log(tgt / 255.0) / np.log(clahe_mean / 255.0), 0.35, 1.0))
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        l_enhanced = cv2.LUT(l_clahe, lut)
    else:
        l_enhanced = l_clahe

    # Ghép lại kênh LAB và chuyển về BGR
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


@dataclass
class ClientStats:
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_advancing: bool = True


@dataclass
class CameraResult:
    checks: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    message: str = ""
    severity: str = "ok"          # ok | warn | error

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def signal_ok(self) -> bool:
        """Lỗi nghiêm trọng làm hỏng mọi bước sau (đen/đóng băng)."""
        return self.checks.get("not_black", True) and self.checks.get("not_frozen", True)


class CameraMonitor:
    """Theo dõi chất lượng camera theo thời gian (mỗi phiên một instance)."""

    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None
        self._static_count = 0

    def reset(self) -> None:
        self._prev = None
        self._static_count = 0

    def evaluate(self, frame_bgr: np.ndarray, stats: ClientStats) -> CameraResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        noise = estimate_noise(gray)

        g = gray.astype(np.float32)
        if self._prev is not None and self._prev.shape == g.shape:
            diff = float(np.abs(g - self._prev).mean())
            self._static_count = self._static_count + 1 if diff < C.FROZEN_DIFF else 0
        else:
            diff = None
        self._prev = g

        res_known = bool(stats.width) and bool(stats.height)
        checks = {
            "resolution_ok": res_known and stats.width >= C.MIN_WIDTH and stats.height >= C.MIN_HEIGHT,
            "not_black": brightness >= C.BLACK_FRAME_MEAN,
            "not_frozen": stats.frame_advancing and self._static_count < C.FROZEN_FRAMES,
            "brightness_ok": C.FRAME_BRIGHTNESS[0] <= brightness <= C.FRAME_BRIGHTNESS[1],
            "noise_ok": noise <= C.MAX_NOISE_SIGMA,
        }
        metrics = {
            "width": stats.width,
            "height": stats.height,
            "fps": round(stats.fps, 1) if stats.fps else None,
            "fps_low": bool(stats.fps) and stats.fps < C.MIN_FPS,
            "frame_brightness": round(brightness, 1),
            "noise_sigma": round(noise, 2),
            "frame_diff": round(diff, 3) if diff is not None else None,
        }

        result = CameraResult(checks=checks, metrics=metrics)
        # Thông báo theo thứ tự ưu tiên
        if not checks["not_black"]:
            result.message, result.severity = "Camera bị che hoặc khung hình đen – hãy kiểm tra ống kính", "error"
        elif not checks["not_frozen"]:
            result.message, result.severity = "Camera bị treo / đóng băng – hãy chọn lại camera", "error"
        elif not res_known:
            result.message, result.severity = "Đang đọc độ phân giải camera...", "warn"
        elif not checks["resolution_ok"]:
            result.message = f"Độ phân giải {stats.width}×{stats.height} thấp hơn {C.MIN_WIDTH}×{C.MIN_HEIGHT} – hãy chọn camera khác"
            result.severity = "error"
        elif brightness < C.FRAME_BRIGHTNESS[0]:
            result.message, result.severity = "Khung hình quá tối – hãy bật thêm đèn", "warn"
        elif brightness > C.FRAME_BRIGHTNESS[1]:
            result.message, result.severity = "Khung hình quá chói – tránh nguồn sáng chiếu thẳng vào camera", "warn"
        elif not checks["noise_ok"]:
            result.message, result.severity = "Hình ảnh bị nhiễu hạt – hãy tăng ánh sáng phòng", "warn"
        elif stats.fps and stats.fps < C.MIN_FPS:
            result.message = f"Camera đạt chuẩn – FPS thấp ({stats.fps:.0f}), nên đóng bớt ứng dụng đang dùng camera/CPU"
        else:
            result.message, result.severity = "Camera đạt chuẩn – đang xác nhận...", "ok"
        return result
