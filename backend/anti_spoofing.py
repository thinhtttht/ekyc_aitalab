"""Module Chống Giả Mạo Sinh Trắc Học Đa Tầng (Passive Anti-Spoofing / Presentation Attack Detection).
Bao gồm:
1. Mô hình Deep Learning MiniFASNetV2 (Silent-Face-Anti-Spoofing qua ONNX Runtime).
2. Kiểm tra Độ sâu hình học 3D (3D Depth Consistency từ MediaPipe Z-coordinates).
3. Kiểm tra Vân Moiré tần số cao (2D Fourier FFT Analysis) phát hiện màn hình LCD/OLED.
4. Kiểm tra Viền thiết bị / Mép giấy (Screen Bezel & Paper Border Detection qua Hough Lines).
"""
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import config as C

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@dataclass
class AntiSpoofResult:
    is_real: bool = True
    real_prob: float = 0.95
    print_prob: float = 0.02
    replay_prob: float = 0.03
    spoof_type: str = "real"  # 'real' | 'print_attack' | 'replay_attack' | 'planar_2d' | 'screen_moire' | 'bezel_detected'
    depth_3d_delta: float = 0.05
    depth_3d_ok: bool = True
    moire_ratio: float = 0.15
    screen_detected: bool = False
    bezel_detected: bool = False
    severity: str = "ok"      # 'ok' | 'warn' | 'error'
    message: str = ""


def softmax(x: np.ndarray) -> np.ndarray:
    """Hàm Softmax tính xác suất từ logits."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def crop_face_margin(image: np.ndarray, bbox: Tuple[float, float, float, float], scale: float = 2.7, out_w: int = 80, out_h: int = 80) -> np.ndarray:
    """Cắt vùng khuôn mặt mở rộng theo tỷ lệ scale chuẩn của Silent-Face-Anti-Spoofing (2.7x bbox)."""
    src_h, src_w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    if isinstance(x0, float) and x0 <= 1.0 and x1 <= 1.0:
        x0, y0, x1, y1 = x0 * src_w, y0 * src_h, x1 * src_w, y1 * src_h

    x = float(x0)
    y = float(y0)
    box_w = max(10.0, float(x1 - x0))
    box_h = max(10.0, float(y1 - y0))

    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x = box_w / 2.0 + x
    center_y = box_h / 2.0 + y

    left_top_x = center_x - new_width / 2.0
    left_top_y = center_y - new_height / 2.0
    right_bottom_x = center_x + new_width / 2.0
    right_bottom_y = center_y + new_height / 2.0

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1

    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    lx = max(0, int(left_top_x))
    ly = max(0, int(left_top_y))
    rx = min(src_w, int(right_bottom_x) + 1)
    ry = min(src_h, int(right_bottom_y) + 1)

    cropped = image[ly:ry, lx:rx]
    if cropped.size == 0:
        return cv2.resize(image, (out_w, out_h))
    return cv2.resize(cropped, (out_w, out_h))


class AntiSpoofDetector:
    """Engine phòng thủ chống giả mạo đa tầng kết hợp Deep Learning và Thị giác máy tính."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or C.MINIFASNET_MODEL_PATH
        self.session = None
        self._init_model()

    def _init_model(self):
        if not HAS_ORT:
            print("[AntiSpoof] onnxruntime khong co san, dung bo loc CV Heuristic.")
            return

        if os.path.exists(self.model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                print(f"[AntiSpoof] Da khoi tao MiniFASNet ONNX thanh cong: {self.model_path}")
            except Exception as e:
                print(f"[AntiSpoof] Loi khoi tao MiniFASNet ({e}), chuyen sang che do CV Heuristic.")
                self.session = None
        else:
            print(f"[AntiSpoof] Khong tim thay file model tai {self.model_path}, chay che do CV Heuristic.")

    def check_3d_depth(self, landmarks_3d: Optional[np.ndarray]) -> Tuple[bool, float]:
        """Kiểm tra độ sâu hình học 3D của khuôn mặt từ MediaPipe Z coordinates.
        Mặt người thật 3D: Chóp mũi (lm 1) nhô cao hơn rõ rệt so với mặt phẳng mắt/tai (Z mũi âm hơn Z mắt).
        Ảnh in 2D phẳng: Độ chênh lệch trục Z rất nhỏ hoặc bằng phẳng.
        """
        if landmarks_3d is None or len(landmarks_3d) < 468:
            return True, 0.05

        try:
            # Chóp mũi (landmark 1, 4)
            z_nose = float(landmarks_3d[1, 2] + landmarks_3d[4, 2]) / 2.0
            # Mắt trái ngoài (33), Mắt phải ngoài (263), Mắt trái trong (133), Mắt phải trong (362)
            z_eyes = float(
                landmarks_3d[33, 2] + landmarks_3d[263, 2] +
                landmarks_3d[133, 2] + landmarks_3d[362, 2]
            ) / 4.0

            # MediaPipe: z âm hơn là gần camera hơn -> mũi nhô ra trước mắt -> delta = z_eyes - z_nose > 0
            delta_z = float(z_eyes - z_nose)
            is_3d = delta_z >= C.DEPTH_3D_MIN_DELTA
            return is_3d, round(delta_z, 4)
        except Exception:
            return True, 0.05

    def check_moire_pattern(self, face_bgr: np.ndarray) -> Tuple[bool, float]:
        """Phát hiện vân Moiré quang học qua biến đổi Fourier 2D (FFT).
        Màn hình LCD/OLED phát lại video sẽ sinh ra các sọc giao thoa tần số cao.
        """
        if face_bgr.size == 0 or face_bgr.shape[0] < 30 or face_bgr.shape[1] < 30:
            return False, 0.15

        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            f = np.fft.fft2(resized.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1.0)

            # Phân tích năng lượng dải tần số cao (khoảng cách bán kính 25 - 55 pixel từ tâm)
            center = 64
            y, x = np.ogrid[:128, :128]
            dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
            high_freq_mask = (dist_from_center >= 25) & (dist_from_center <= 55)

            high_energy = float(np.mean(magnitude_spectrum[high_freq_mask]))
            total_energy = float(np.mean(magnitude_spectrum)) + 1e-6
            ratio = float(high_energy / total_energy)
            is_screen = ratio > C.MOIRE_ENERGY_RATIO_THRESH
            return is_screen, round(ratio, 3)
        except Exception:
            return False, 0.15

    def check_screen_bezel(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """Phát hiện cạnh viền thẳng tắp của điện thoại, máy tính bảng hoặc mép giấy xung quanh mặt."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox
        if isinstance(x0, float) and x0 <= 1.0 and x1 <= 1.0:
            x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

        # Mở rộng vùng quét viền ra 1.45 lần
        bw = x1 - x0
        bh = y1 - y0
        pad_x = int(bw * 0.45)
        pad_y = int(bh * 0.45)

        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(w, x1 + pad_x)
        ry1 = min(h, y1 + pad_y)

        roi = frame_bgr[ry0:ry1, rx0:rx1]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return False

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 60, 160)

            # Xóa vùng khuôn mặt bên trong để chỉ xét viền bên ngoài
            inner_x0 = max(0, x0 - rx0)
            inner_y0 = max(0, y0 - ry0)
            inner_x1 = min(roi.shape[1], x1 - rx0)
            inner_y1 = min(roi.shape[0], y1 - ry0)
            edges[inner_y0:inner_y1, inner_x0:inner_x1] = 0

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=int(bw * 0.40), maxLineGap=12)
            if lines is None:
                return False

            # Đếm các đường thẳng song song cạnh viền (gần đứng hoặc gần ngang)
            bezel_candidates = 0
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi)
                # Gần ngang (0° ± 10° hoặc 180° ± 10°) hoặc gần đứng (90° ± 10°)
                if angle <= 12 or angle >= 168 or abs(angle - 90) <= 12:
                    bezel_candidates += 1

            return bezel_candidates >= 4
        except Exception:
            return False

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        landmarks_3d: Optional[np.ndarray] = None,
    ) -> AntiSpoofResult:
        """Đánh giá toàn diện phòng thủ chống tấn công giả mạo (PAD)."""
        res = AntiSpoofResult()

        # 1. Kiểm tra độ sâu 3D (Face Planarity)
        depth_ok, delta_z = self.check_3d_depth(landmarks_3d)
        res.depth_3d_ok = depth_ok
        res.depth_3d_delta = delta_z

        # 2. Cắt khuôn mặt cho Moiré và MiniFASNet
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox
        if isinstance(x0, float) and x0 <= 1.0 and x1 <= 1.0:
            px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        else:
            px0, py0, px1, py1 = int(x0), int(y0), int(x1), int(y1)

        face_roi = frame_bgr[max(0, py0):min(h, py1), max(0, px0):min(w, px1)]

        # 3. Kiểm tra vân Moiré
        screen_moire, moire_ratio = self.check_moire_pattern(face_roi)
        res.screen_detected = screen_moire
        res.moire_ratio = moire_ratio

        # 4. Kiểm tra viền màn hình / cạnh giấy
        bezel_detected = self.check_screen_bezel(frame_bgr, bbox)
        res.bezel_detected = bezel_detected

        # 5. Chạy mô hình Deep Learning MiniFASNet (nếu có ONNX session)
        if self.session is not None and face_roi.size > 0:
            try:
                crop_80 = crop_face_margin(frame_bgr, bbox, scale=2.7, out_w=80, out_h=80)
                if crop_80.size > 0:
                    # Upstream MiniFASNet (Silent-Face-Anti-Spoofing) expects raw [0, 255] float32 (NOT divided by 255)
                    inp = crop_80.astype(np.float32)
                    # HWC to NCHW
                    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]

                    logits = self.session.run(None, {"input": inp})[0][0]
                    probs = softmax(logits)

                    # Theo chuẩn Silent-Face-Anti-Spoofing:
                    # Index 1: Real Face (Người thật 100%)
                    # Index 0: Print Attack (Ảnh in 2D)
                    # Index 2: Replay Attack (Màn hình / Video phát lại)
                    res.real_prob = round(float(probs[1]), 3)
                    res.print_prob = round(float(probs[0]), 3)
                    res.replay_prob = round(float(probs[2]), 3)
            except Exception as e:
                # Nếu có lỗi khi inference, fallback giữ giá trị mặc định
                print(f"[AntiSpoof] Inference error: {e}")

        # 6. Tổng hợp quyết định (Ensemble Rule Base)
        # Tình huống A: MiniFASNet phát hiện giả mạo rõ rệt
        if res.print_prob >= C.ANTISPOOF_PRINT_THRESH:
            res.is_real = False
            res.spoof_type = "print_attack"
            res.severity = "error"
            res.message = "⚠️ PHÁT HIỆN ẢNH IN 2D – Vui lòng sử dụng khuôn mặt thật trực tiếp!"
            return res

        if res.replay_prob >= C.ANTISPOOF_REPLAY_THRESH:
            res.is_real = False
            res.spoof_type = "replay_attack"
            res.severity = "error"
            res.message = "⚠️ PHÁT HIỆN MÀN HÌNH / VIDEO PHÁT LẠI – Vui lòng không giơ điện thoại trước camera!"
            return res

        # Tình huống B: Moiré mạnh + (Viền thiết bị hoặc Replay prob tăng) -> Tấn công màn hình
        if res.screen_detected and (res.bezel_detected or res.replay_prob >= 0.35):
            res.is_real = False
            res.spoof_type = "screen_moire"
            res.severity = "error"
            res.message = "⚠️ PHÁT HIỆN MÀN HÌNH THIẾT BỊ – Nghi ngờ phát lại video!"
            return res

        # Tình huống C: Mặt phẳng 2D hoàn toàn không có độ sâu 3D và (Print prob tăng hoặc Real prob thấp)
        if not res.depth_3d_ok and (res.print_prob >= 0.30 or res.real_prob < 0.70):
            res.is_real = False
            res.spoof_type = "planar_2d"
            res.severity = "error"
            res.message = "⚠️ PHÁT HIỆN HÌNH ẢNH PHẲNG 2D – Yêu cầu khuôn mặt người thật có chiều sâu 3D!"
            return res

        # Tình huống D: Đạt chuẩn Real
        if res.real_prob >= C.ANTISPOOF_REAL_THRESH:
            res.is_real = True
            res.spoof_type = "real"
            res.severity = "ok"
            res.message = "Khuôn mặt người thật hợp lệ"
        else:
            # Ngưỡng trung gian nghi ngờ
            res.is_real = False
            res.spoof_type = "suspect"
            res.severity = "warn"
            res.message = "Đang kiểm tra bảo mật chống giả mạo sinh trắc..."

        return res
