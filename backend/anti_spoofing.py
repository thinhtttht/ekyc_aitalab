"""Chống giả mạo thụ động (Passive Anti-Spoofing / Presentation Attack Detection).

Quyết định chính dựa trên MiniFASNetV2 (Silent-Face-Anti-Spoofing, ONNX Runtime).
Ba heuristic thị giác máy tính dưới đây chưa được kiểm chứng trên dữ liệu thật nên mặc định TẮT
(bật bằng cờ trong config):
  - Độ sâu 3D từ trục Z của MediaPipe (ANTISPOOF_USE_DEPTH). Lưu ý Z là giá trị model tự suy ra
    từ mesh chuẩn, nên ảnh in phẳng vẫn có thể cho mũi "nhô lên".
  - Vân Moiré qua phổ Fourier 2D (ANTISPOOF_USE_MOIRE).
  - Viền điện thoại / mép giấy qua Hough Lines (ANTISPOOF_USE_BEZEL).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import cv2
import numpy as np

import config as C

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@dataclass
class AntiSpoofResult:
    is_real: bool = True
    real_prob: float = 0.0
    print_prob: float = 0.0
    replay_prob: float = 0.0
    spoof_type: str = "real"  # 'real' | 'print_attack' | 'replay_attack' | 'planar_2d' | 'screen_moire' | 'suspect' | 'model_unavailable'
    model_ran: bool = False
    depth_3d_delta: float = 0.0
    depth_3d_ok: bool = True
    moire_ratio: float = 0.0
    screen_detected: bool = False
    bezel_detected: bool = False
    severity: str = "ok"      # 'ok' | 'warn' | 'error'
    message: str = ""


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _bbox_to_px(bbox: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    """bbox chuẩn hoá (~[0, 1], có thể tràn nhẹ khi mặt sát mép) -> pixel; bbox đã là pixel thì giữ nguyên."""
    x0, y0, x1, y1 = bbox
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 2.0:
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    return int(x0), int(y0), int(x1), int(y1)


def crop_face_margin(image: np.ndarray, bbox: Tuple[float, float, float, float], scale: float = 2.7, out_w: int = 80, out_h: int = 80) -> np.ndarray:
    """Cắt vùng khuôn mặt mở rộng theo tỷ lệ scale chuẩn của Silent-Face-Anti-Spoofing (2.7x bbox)."""
    src_h, src_w = image.shape[:2]
    x0, y0, x1, y1 = _bbox_to_px(bbox, src_w, src_h)
    box_w = max(10.0, float(x1 - x0))
    box_h = max(10.0, float(y1 - y0))
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)

    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x0 + box_w / 2.0, y0 + box_h / 2.0
    left, top = cx - new_w / 2.0, cy - new_h / 2.0
    right, bottom = cx + new_w / 2.0, cy + new_h / 2.0

    # Dịch khung vào trong ảnh thay vì cắt bớt (giữ nguyên tỉ lệ như upstream)
    if left < 0:
        right, left = right - left, 0
    if top < 0:
        bottom, top = bottom - top, 0
    if right > src_w - 1:
        left, right = left - (right - src_w + 1), src_w - 1
    if bottom > src_h - 1:
        top, bottom = top - (bottom - src_h + 1), src_h - 1

    cropped = image[max(0, int(top)):min(src_h, int(bottom) + 1), max(0, int(left)):min(src_w, int(right) + 1)]
    if cropped.size == 0:
        return cv2.resize(image, (out_w, out_h))
    return cv2.resize(cropped, (out_w, out_h))


class AntiSpoofDetector:
    """MiniFASNet + các heuristic tuỳ chọn. Dùng chung toàn tiến trình (không giữ trạng thái theo phiên)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or C.MINIFASNET_MODEL_PATH
        self.session = None
        self._init_model()

    def _init_model(self) -> None:
        if not HAS_ORT:
            print("[AntiSpoof] onnxruntime không có sẵn.")
            return
        if not os.path.exists(self.model_path):
            print(f"[AntiSpoof] Không tìm thấy file model tại {self.model_path}")
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            print(f"[AntiSpoof] Đã khởi tạo MiniFASNet ONNX: {self.model_path}")
        except Exception as e:
            print(f"[AntiSpoof] Lỗi khởi tạo MiniFASNet: {e}")
            self.session = None

    # --- Heuristic tuỳ chọn ------------------------------------------------

    def check_3d_depth(self, landmarks_3d: Optional[np.ndarray]) -> Tuple[bool, float]:
        """Chóp mũi (lm 1, 4) phải gần camera hơn mặt phẳng mắt (33, 263, 133, 362) ít nhất DEPTH_3D_MIN_DELTA."""
        if landmarks_3d is None or len(landmarks_3d) < 468:
            return True, 0.0
        z_nose = float(landmarks_3d[1, 2] + landmarks_3d[4, 2]) / 2.0
        z_eyes = float(landmarks_3d[[33, 263, 133, 362], 2].mean())
        delta_z = z_eyes - z_nose
        return delta_z >= C.DEPTH_3D_MIN_DELTA, round(delta_z, 4)

    def check_moire_pattern(self, face_bgr: np.ndarray) -> Tuple[bool, float]:
        """Tỷ lệ năng lượng dải tần cao (bán kính 25-55 px trên phổ 128x128) so với toàn phổ."""
        if face_bgr.size == 0 or face_bgr.shape[0] < 30 or face_bgr.shape[1] < 30:
            return False, 0.0
        gray = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), (128, 128)).astype(np.float32)
        spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1.0)
        y, x = np.ogrid[:128, :128]
        dist = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
        high = float(np.mean(spectrum[(dist >= 25) & (dist <= 55)]))
        ratio = high / (float(np.mean(spectrum)) + 1e-6)
        return ratio > C.MOIRE_ENERGY_RATIO_THRESH, round(ratio, 3)

    def check_screen_bezel(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """Có >= 4 đường thẳng gần ngang/đứng quanh mặt (viền điện thoại, máy tính bảng, mép giấy)."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = _bbox_to_px(bbox, w, h)
        bw, bh = x1 - x0, y1 - y0
        pad_x, pad_y = int(bw * 0.45), int(bh * 0.45)
        rx0, ry0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        rx1, ry1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
        roi = frame_bgr[ry0:ry1, rx0:rx1]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return False

        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 160)
        edges[max(0, y0 - ry0):y1 - ry0, max(0, x0 - rx0):x1 - rx0] = 0  # chỉ xét vùng bên ngoài mặt
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=int(bw * 0.40), maxLineGap=12)
        if lines is None:
            return False
        angles = [abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) for l in lines]
        return sum(1 for a in angles if a <= 12 or a >= 168 or abs(a - 90) <= 12) >= 4

    # --- Đo & quyết định ---------------------------------------------------

    def _run_model(self, frame_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Xác suất [print, real, replay] của MiniFASNet hoặc None nếu không chạy được."""
        if self.session is None:
            return None
        try:
            # Upstream nhận ảnh BGR 80x80 giá trị thô [0, 255] (KHÔNG chia 255), NCHW
            crop = crop_face_margin(frame_bgr, bbox, scale=2.7, out_w=80, out_h=80).astype(np.float32)
            logits = self.session.run(None, {"input": np.transpose(crop, (2, 0, 1))[np.newaxis, ...]})[0][0]
            return softmax(logits)
        except Exception as e:
            print(f"[AntiSpoof] Inference error: {e}")
            return None

    def measure(self, frame_bgr: np.ndarray, bbox, landmarks_3d: Optional[np.ndarray] = None) -> AntiSpoofResult:
        """Chỉ đo xác suất và các heuristic đang bật, chưa đưa ra kết luận."""
        res = AntiSpoofResult()
        probs = self._run_model(frame_bgr, bbox)
        if probs is not None:
            # Index 0: ảnh in 2D, 1: người thật, 2: màn hình / video phát lại
            res.print_prob, res.real_prob, res.replay_prob = (round(float(p), 3) for p in probs[:3])
            res.model_ran = True

        if C.ANTISPOOF_USE_DEPTH:
            res.depth_3d_ok, res.depth_3d_delta = self.check_3d_depth(landmarks_3d)
        if C.ANTISPOOF_USE_MOIRE:
            h, w = frame_bgr.shape[:2]
            x0, y0, x1, y1 = _bbox_to_px(bbox, w, h)
            face_roi = frame_bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            res.screen_detected, res.moire_ratio = self.check_moire_pattern(face_roi)
        if C.ANTISPOOF_USE_BEZEL:
            res.bezel_detected = self.check_screen_bezel(frame_bgr, bbox)
        return res

    @staticmethod
    def decide(res: AntiSpoofResult) -> AntiSpoofResult:
        """Áp ngưỡng lên kết quả đo (có thể đã được làm mượt) và điền is_real/spoof_type/message."""
        def verdict(is_real: bool, spoof_type: str, severity: str, message: str) -> AntiSpoofResult:
            return replace(res, is_real=is_real, spoof_type=spoof_type, severity=severity, message=message)

        if not res.model_ran and C.ANTISPOOF_REQUIRE_MODEL:
            return verdict(False, "model_unavailable", "error",
                           "Không chạy được mô hình chống giả mạo MiniFASNet – Kiểm tra file backend/models/minifasnet_v2.onnx")
        if res.print_prob >= C.ANTISPOOF_PRINT_THRESH:
            return verdict(False, "print_attack", "error", "⚠️ PHÁT HIỆN ẢNH IN 2D – Vui lòng sử dụng khuôn mặt thật trực tiếp!")
        if res.replay_prob >= C.ANTISPOOF_REPLAY_THRESH:
            return verdict(False, "replay_attack", "error",
                           "⚠️ PHÁT HIỆN MÀN HÌNH / VIDEO PHÁT LẠI – Vui lòng không giơ điện thoại trước camera!")
        bezel = C.ANTISPOOF_USE_BEZEL and res.bezel_detected
        if C.ANTISPOOF_USE_MOIRE and res.screen_detected and (bezel or res.replay_prob >= 0.35):
            return verdict(False, "screen_moire", "error", "⚠️ PHÁT HIỆN MÀN HÌNH THIẾT BỊ – Nghi ngờ phát lại video!")
        if C.ANTISPOOF_USE_DEPTH and not res.depth_3d_ok and (res.print_prob >= 0.30 or res.real_prob < 0.70):
            return verdict(False, "planar_2d", "error",
                           "⚠️ PHÁT HIỆN HÌNH ẢNH PHẲNG 2D – Yêu cầu khuôn mặt người thật có chiều sâu 3D!")
        if res.real_prob >= C.ANTISPOOF_REAL_THRESH or not res.model_ran:
            return verdict(True, "real", "ok", "Khuôn mặt người thật hợp lệ")
        return verdict(False, "suspect", "warn", "Đang kiểm tra bảo mật chống giả mạo sinh trắc...")

    def evaluate(self, frame_bgr: np.ndarray, bbox, landmarks_3d: Optional[np.ndarray] = None) -> AntiSpoofResult:
        """Đo và quyết định trên một frame duy nhất (không làm mượt)."""
        return self.decide(self.measure(frame_bgr, bbox, landmarks_3d))


class AntiSpoofSampler:
    """Chạy PAD mỗi N frame và lấy trung bình xác suất trên K lần đo gần nhất. Mỗi phiên một instance."""

    def __init__(self, detector: AntiSpoofDetector, every_n: int = 1, window: int = 1):
        self.detector = detector
        self.every_n = max(1, every_n)
        self._frame_idx = 0
        self._samples: deque[AntiSpoofResult] = deque(maxlen=max(1, window))
        self._last: Optional[AntiSpoofResult] = None

    def evaluate(self, frame_bgr: np.ndarray, bbox, landmarks_3d: Optional[np.ndarray] = None) -> AntiSpoofResult:
        due = self._last is None or self._frame_idx % self.every_n == 0
        self._frame_idx += 1
        if not due:
            return self._last

        latest = self.detector.measure(frame_bgr, bbox, landmarks_3d)
        self._samples.append(latest)
        ran = [s for s in self._samples if s.model_ran]
        if latest.model_ran and ran:
            latest = replace(
                latest,
                real_prob=round(float(np.mean([s.real_prob for s in ran])), 3),
                print_prob=round(float(np.mean([s.print_prob for s in ran])), 3),
                replay_prob=round(float(np.mean([s.replay_prob for s in ran])), 3),
            )
        self._last = self.detector.decide(latest)
        return self._last
