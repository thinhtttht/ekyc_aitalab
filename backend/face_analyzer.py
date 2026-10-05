"""(b) Đánh giá chất lượng khuôn mặt (FQA) bằng MediaPipe FaceMesh + OpenCV.

Trả về FaceResult chứa số liệu thô; hàm `quality_checks` áp ngưỡng để ra checks/thông báo.
Lưu ý: toạ độ trong FaceResult là toạ độ ảnh GỐC (chưa lật gương); các gợi ý hướng di chuyển
được quy đổi sang hướng trên màn hình (đã lật gương) để người dùng làm theo tự nhiên.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config as C
from head_pose import estimate_head_pose
from anti_spoofing import AntiSpoofDetector, AntiSpoofResult

try:  # mediapipe chỉ cần khi chạy thật; unit test có thể chạy không cần
    import mediapipe as mp

    _FACE_OVAL_IDX = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
except Exception:  # pragma: no cover
    mp = None
    _FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                      377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Landmark dùng cho heuristic
_UPPER_SKIN = [9, 151, 168, 108, 337]   # giữa 2 chân mày, trán, sống mũi trên (khẩu trang không che)
_LOWER_FACE = [152, 175, 200, 420, 214]  # cằm, má dưới (bị khẩu trang che, không dính môi)
_IRIS_CENTERS = [468, 473]              # tâm mống mắt (refine_landmarks)
_EYE_FALLBACK = [159, 386]              # mí trên nếu không có iris

PARTS_CONFIG = {
    "left_eye": {
        "indices": [33, 160, 158, 133, 153, 144],
        "name": "Mắt phải",
        "min_edge": 4.0,
        "min_contrast": 25.0,
    },
    "right_eye": {
        "indices": [362, 385, 387, 263, 373, 380],
        "name": "Mắt trái",
        "min_edge": 4.0,
        "min_contrast": 25.0,
    },
    "left_eyebrow": {
        "indices": [70, 63, 105, 66, 107],
        "name": "Chân mày phải",
        "min_edge": 1.8,
        "min_contrast": 18.0,
    },
    "right_eyebrow": {
        "indices": [336, 296, 334, 293, 300],
        "name": "Chân mày trái",
        "min_edge": 1.8,
        "min_contrast": 18.0,
    },
    "nose": {
        "indices": [1, 2, 98, 327, 168],
        "name": "Vùng Mũi",
        "min_edge": 2.2,
        "min_contrast": 20.0,
    },
    "mouth": {
        "indices": [61, 291, 0, 17, 13, 14, 78, 308],
        "name": "Vùng Miệng",
        "min_edge": 2.5,
        "min_contrast": 22.0,
    },
}


@dataclass
class FaceResult:
    face_count: int = 0
    bbox: Optional[tuple] = None                 # (x0, y0, x1, y1) chuẩn hoá
    center: Optional[tuple] = None               # (x, y) chuẩn hoá
    face_h: float = 0.0                          # chiều cao mặt / chiều cao khung
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    oval_dist: float = 99.0                      # max ((x-cx)/rx)^2 + ((y-cy)/ry)^2 trên viền mặt
    fill: float = 0.0                            # face_h / (2*ry)
    scale_ratio: float = 0.0                     # Width_face / Width_oval (Tỷ lệ vàng 0.40 - 0.85)
    corners_inside: bool = True                  # 4 góc bounding box có lọt elip không
    offset: tuple = (0.0, 0.0)                   # lệch tâm (chuẩn hoá theo bán trục), toạ độ ảnh gốc
    brightness: float = 0.0                      # độ sáng trung bình vùng mặt
    brightness_mean: float = 0.0                 # Mean kênh độ sáng
    brightness_std: float = 0.0                  # Độ lệch chuẩn (kiểm tra ngược sáng)
    side_ratio: float = 1.0
    sharpness: float = 0.0                       # Phương sai Laplacian
    mask_detected: bool = False
    sunglasses_detected: bool = False
    glare_detected: bool = False                 # Lóa sáng kính phản quang (> 30% mắt)
    hand_occlusion: bool = False
    parts_status: dict = field(default_factory=dict)     # {"left_eye": True, "mouth": False, ...}
    occluded_part_name: Optional[str] = None             # Tên bộ phận bị che nếu có
    keypoints: dict = field(default_factory=dict)        # Toạ độ các điểm mốc vẽ lưới biometric
    perspective_ratio: Optional[float] = None            # độ dài sống mũi / khoảng cách 2 mắt
    anti_spoof: Optional[AntiSpoofResult] = None         # Kết quả kiểm tra giả mạo ảnh/video
    is_upside_down: bool = False                         # Khuôn mặt bị lật ngược (mắt ở dưới, miệng ở trên)
    arcface_kps: Optional[np.ndarray] = None             # 5 mốc chuẩn ArcFace [left_eye, right_eye, nose, mouth_l, mouth_r]
    debug: dict = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return self.face_count >= 1 and self.bbox is not None


_anti_spoof_detector: Optional[AntiSpoofDetector] = None

def get_anti_spoof_detector() -> AntiSpoofDetector:
    global _anti_spoof_detector
    if _anti_spoof_detector is None:
        _anti_spoof_detector = AntiSpoofDetector()
    return _anti_spoof_detector


def _patch(img: np.ndarray, x: float, y: float, r: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1) + img.shape[2:], dtype=img.dtype)
    return img[y0:y1, x0:x1]


def _is_skin(cr: float, cb: float) -> bool:
    return 133 <= cr <= 180 and 77 <= cb <= 135


class FaceAnalyzer:
    """Bọc MediaPipe FaceMesh (chế độ video - có tracking), mỗi phiên một instance."""

    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe chưa được cài đặt")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()
        self._hands.close()

    def analyze(self, frame_bgr: np.ndarray, oval: dict) -> FaceResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        hands_res = self._hands.process(rgb)
        faces = res.multi_face_landmarks or []
        hands = hands_res.multi_hand_landmarks or []
        out = FaceResult(face_count=len(faces))
        if not faces:
            return out
        # Chọn mặt lớn nhất
        def area(f):
            xs = [p.x for p in f.landmark]; ys = [p.y for p in f.landmark]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
        face = max(faces, key=area)
        lm = np.array([[p.x, p.y, p.z] for p in face.landmark], dtype=np.float64)   # chuẩn hoá
        out = analyze_landmarks(frame_bgr, lm, oval, out)

        # Phát hiện bàn tay che mặt hoặc đè vào vùng Oval
        if hands and out.bbox:
            cx, cy, rx, ry = oval["cx"], oval["cy"], oval["rx"], oval["ry"]
            hand_pts_in_face = 0
            for hand in hands:
                for pt in hand.landmark:
                    hx, hy = pt.x, pt.y
                    in_oval = ((hx - cx) / rx) ** 2 + ((hy - cy) / ry) ** 2 <= 1.0
                    in_bbox = out.bbox[0] <= hx <= out.bbox[2] and out.bbox[1] <= hy <= out.bbox[3]
                    if in_oval or in_bbox:
                        hand_pts_in_face += 1
            if hand_pts_in_face >= 2:
                out.hand_occlusion = True
                out.debug["hand_pts_in_face"] = hand_pts_in_face
        return out


def analyze_landmarks(frame_bgr: np.ndarray, lm3: np.ndarray, oval: dict, out: FaceResult | None = None) -> FaceResult:
    """Tính toàn bộ chỉ số từ mảng landmark chuẩn hoá (N, 3) [x, y, z]. Tách riêng để dễ test."""
    out = out or FaceResult(face_count=1)
    h, w = frame_bgr.shape[:2]
    lm = lm3[:, :2]
    px = lm * np.array([w, h])

    contour = lm[_FACE_OVAL_IDX]
    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    out.bbox = (float(x0), float(y0), float(x1), float(y1))
    out.center = (float((x0 + x1) / 2), float((y0 + y1) / 2))
    out.face_h = float(y1 - y0)

    # --- Lọt khung Oval & cự ly (Framing & Scale Check) ---
    cx, cy, rx, ry = oval["cx"], oval["cy"], oval["rx"], oval["ry"]
    e = ((contour[:, 0] - cx) / rx) ** 2 + ((contour[:, 1] - cy) / ry) ** 2
    out.oval_dist = float(e.max())
    out.fill = out.face_h / (2 * ry)
    out.offset = ((out.center[0] - cx) / rx, (out.center[1] - cy) / ry)

    # 4 góc Bounding Box kiểm tra có lọt lòng elip không
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    corner_e = ((corners[:, 0] - cx) / rx) ** 2 + ((corners[:, 1] - cy) / ry) ** 2
    out.corners_inside = bool(np.all(corner_e <= 1.0))
    face_w = float(x1 - x0)
    oval_w = float(2 * rx)
    out.scale_ratio = float(face_w / max(0.01, oval_w))

    # --- Tư thế đầu (landmark 3D, z cùng thang đo với x) ---
    if lm3.shape[1] >= 3:
        pts3d = lm3 * np.array([w, h, w])
        y, p, r = estimate_head_pose(pts3d)
        # Hiệu chuẩn dấu cho luồng video selfie eKYC (màn hình lật gương):
        # Quay sang TRÁI của người dùng (theo hướng mũi tên bên trái) -> Yaw > 0
        # Quay sang PHẢI của người dùng (theo hướng mũi tên bên phải) -> Yaw < 0
        out.yaw = round(-y, 1)
        out.pitch = round(p, 1)
        out.roll = round(-r, 1)

    # --- Kiểm tra lật ngược đầu (Inverted / Upside-Down: Mắt ở dưới, Miệng ở trên) ---
    # Toạ độ ảnh: y = 0 ở đỉnh ảnh, y = 1 ở đáy ảnh.
    # Chuẩn: Trán (10) < Mắt (33, 263) < Mũi (1) < Miệng (13, 14) < Cằm (152)
    if lm.shape[0] > 263:
        eye_y = float((lm[33, 1] + lm[263, 1]) / 2.0)
        mouth_y = float((lm[13, 1] + lm[14, 1]) / 2.0) if lm.shape[0] > 14 else float(lm[0, 1])
        forehead_y = float(lm[10, 1]) if lm.shape[0] > 10 else 0.0
        chin_y = float(lm[152, 1]) if lm.shape[0] > 152 else 1.0

        is_inverted = bool(
            (eye_y >= mouth_y)
            or (chin_y <= forehead_y)
            or (out.roll is not None and abs(out.roll) >= 85.0)
        )
        out.is_upside_down = is_inverted

    # --- Ánh sáng khuôn mặt (Illumination: Histogram Grayscale / Y) ---
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hull = cv2.convexHull(px[_FACE_OVAL_IDX].astype(np.int32))
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, hull, 255)
    if mask.any():
        pixels = gray[mask > 0]
        out.brightness = float(pixels.mean())
        out.brightness_mean = float(pixels.mean())
        out.brightness_std = float(pixels.std())
        nose_x = int(px[1, 0])
        left = mask.copy(); left[:, nose_x:] = 0
        right = mask.copy(); right[:, :nose_x] = 0
        if left.any() and right.any():
            bl, br = float(gray[left > 0].mean()), float(gray[right > 0].mean())
            out.side_ratio = max(bl, br) / max(1.0, min(bl, br))

    # --- Độ nét: Laplacian trên ROI mặt chuẩn hoá về rộng 200px ---
    bx0, by0 = int(max(0, x0 * w)), int(max(0, y0 * h))
    bx1, by1 = int(min(w, x1 * w)), int(min(h, y1 * h))
    if bx1 - bx0 > 10 and by1 - by0 > 10:
        roi = gray[by0:by1, bx0:bx1]
        scale = 200.0 / roi.shape[1]
        roi = cv2.resize(roi, (200, max(10, int(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        out.sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    # --- Heuristic khẩu trang & Kính râm / Kính lóa (Occlusion) ---
    face_w_px = max(1.0, (x1 - x0) * w)
    r = max(3, int(face_w_px * 0.05))
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)

    def stats(indices, rad):
        ps = [_patch(ycc, px[i, 0], px[i, 1], rad).reshape(-1, 3) for i in indices]
        allp = np.concatenate(ps, axis=0)
        return allp.mean(axis=0), allp[:, 0].std()

    (yu, cru, cbu), _ = stats(_UPPER_SKIN, r)
    (yl, crl, cbl), _ = stats(_LOWER_FACE, r)
    chroma = float(np.hypot(crl - cru, cbl - cbu))
    y_ratio = float(yl / max(1.0, yu))
    upper_skin, lower_skin = bool(_is_skin(cru, cbu)), bool(_is_skin(crl, cbl))
    if upper_skin and lower_skin:
        out.mask_detected = chroma > 36.0 or y_ratio < C.MASK_DARK_RATIO
    elif upper_skin and not lower_skin:
        out.mask_detected = True
    else:
        out.mask_detected = chroma > C.MASK_CHROMA_DIST * 1.5 or y_ratio < C.MASK_DARK_RATIO * 0.85

    # Vùng mắt: kiểm tra Kính râm đen (Black Sunglasses) và Kính bị lóa (Glare)
    eye_rois = []
    for eye_indices in ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]):
        sub = px[eye_indices]
        ex0, ey0 = sub.min(axis=0)
        ex1, ey1 = sub.max(axis=0)
        pad = max(2, int(face_w_px * 0.02))
        e_roi = gray[max(0, int(ey0 - pad)):min(h, int(ey1 + pad)), max(0, int(ex0 - pad)):min(w, int(ex1 + pad))]
        if e_roi.size > 0:
            eye_rois.append(e_roi)

    if eye_rois:
        all_eye_pixels = np.concatenate([r.ravel() for r in eye_rois])
        eye_mean = float(all_eye_pixels.mean())
        eye_std = float(all_eye_pixels.std())

        # Kính râm đen: Mean < 30 VÀ Std < 10 (mắt tối đồng nhất)
        black_sunglasses = bool(eye_mean < C.SUNGLASSES_EYE_MEAN_MAX and eye_std < C.SUNGLASSES_EYE_STD_MAX)

        # Kính lóa phản quang: Điểm ảnh > 240 chiếm > 30% diện tích mắt
        glare_pixels = int(np.count_nonzero(all_eye_pixels >= C.GLARE_PIXEL_THRESH))
        glare_ratio = float(glare_pixels / max(1, len(all_eye_pixels)))
        out.glare_detected = bool(glare_ratio >= C.GLARE_AREA_RATIO_MAX)
        out.sunglasses_detected = black_sunglasses

    # --- Tỉ lệ phối cảnh ---
    iod = float(np.linalg.norm(px[33] - px[263]))
    if iod > 1:
        out.perspective_ratio = round(float(np.linalg.norm(px[168] - px[1])) / iod, 4)

    # --- Kiểm tra độ rõ nét & sự hiện diện của từng bộ phận ngũ quan (Occlusion Detection) ---
    parts_status = {}
    occluded_name = None
    for p_key, p_cfg in PARTS_CONFIG.items():
        sub_pts = px[p_cfg["indices"]]
        sx0, sy0 = sub_pts.min(axis=0)
        sx1, sy1 = sub_pts.max(axis=0)
        pad = max(3, int(face_w_px * 0.025))
        roi = gray[max(0, int(sy0 - pad)):min(h, int(sy1 + pad)), max(0, int(sx0 - pad)):min(w, int(sx1 + pad))]
        if roi.size == 0:
            parts_status[p_key] = False
            if occluded_name is None:
                occluded_name = p_cfg["name"]
            continue

        sobel = cv2.Sobel(roi, cv2.CV_64F, 1, 1, ksize=3)
        edge_energy = float(np.mean(np.abs(sobel)))
        contrast = float(roi.max() - roi.min())
        is_ok = bool(edge_energy >= p_cfg["min_edge"] and contrast >= p_cfg["min_contrast"])
        parts_status[p_key] = is_ok
        if not is_ok and occluded_name is None:
            occluded_name = p_cfg["name"]

    out.parts_status = parts_status
    out.occluded_part_name = occluded_name

    # --- Trích xuất toạ độ các đường mốc (Keypoints) để vẽ lưới Biometric trên frontend ---
    def round_pts(indices):
        return [[round(float(lm[i, 0]), 3), round(float(lm[i, 1]), 3)] for i in indices]

    out.keypoints = {
        "left_eye": round_pts([33, 160, 158, 133, 153, 144, 33]),
        "right_eye": round_pts([362, 385, 387, 263, 373, 380, 362]),
        "left_eyebrow": round_pts([70, 63, 105, 66, 107]),
        "right_eyebrow": round_pts([336, 296, 334, 293, 300]),
        "nose": round_pts([168, 6, 195, 5, 4, 1, 2, 98, 327]),
        "mouth": round_pts([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]),
        "jaw": round_pts(_FACE_OVAL_IDX),
    }

    out.debug = {
        "chroma_dist": round(chroma, 1), "y_ratio": round(y_ratio, 2), "upper_skin": upper_skin, "lower_skin": lower_skin,
        "brightness_std": round(out.brightness_std, 1),
    }

    # --- 5 mốc chuẩn ArcFace Alignment (Mắt trái, Mắt phải, Chóp mũi, Khóe miệng trái, Khóe miệng phải) ---
    p_eye_1 = px[468] if len(px) > 468 else (px[33] + px[133]) / 2.0
    p_eye_2 = px[473] if len(px) > 473 else (px[362] + px[263]) / 2.0
    p_left_eye = p_eye_1 if p_eye_1[0] < p_eye_2[0] else p_eye_2
    p_right_eye = p_eye_2 if p_eye_1[0] < p_eye_2[0] else p_eye_1
    p_nose = px[1]
    p_m1 = px[61]
    p_m2 = px[291]
    p_mouth_left = p_m1 if p_m1[0] < p_m2[0] else p_m2
    p_mouth_right = p_m2 if p_m1[0] < p_m2[0] else p_m1
    out.arcface_kps = np.array([p_left_eye, p_right_eye, p_nose, p_mouth_left, p_mouth_right], dtype=np.float32)

    # --- Chống Giả Mạo Sinh Trắc Học (Passive Anti-Spoofing / PAD) ---
    if out.bbox is not None:
        out.anti_spoof = get_anti_spoof_detector().evaluate(frame_bgr, out.bbox, landmarks_3d=lm3)

    return out


def check_continuous_face_quality(face: Optional[FaceResult], is_turning: bool = False) -> tuple[bool, str, str]:
    """Kiểm tra các tiêu chuẩn chất lượng khuôn mặt liên tục trong suốt mọi giai đoạn của quy trình.
    Trả về: (is_ok, error_message, severity: 'ok' | 'warn' | 'error')
    """
    if face is None or not face.detected:
        return False, "Vui lòng đưa mặt vào khung hình", "warn"

    if face.face_count > 1:
        return False, "Phát hiện nhiều khuôn mặt – chỉ một người duy nhất trong khung hình", "error"

    # KIỂM TRA CHỐNG GIẢ MẠO ẢNH & VIDEO (PASSIVE ANTI-SPOOFING / PAD)
    if face.anti_spoof and not face.anti_spoof.is_real:
        return False, face.anti_spoof.message, face.anti_spoof.severity

    # KIỂM TRA LẬT NGƯỢC ĐẦU (MẮT Ở DƯỚI, MIỆNG Ở TRÊN)
    if face.is_upside_down:
        return False, "Khuôn mặt bị lật ngược – Vui lòng giữ thẳng đầu (mắt ở trên, miệng ở dưới)", "error"

    if face.hand_occlusion:
        return False, "Vui lòng bỏ tay ra khỏi khuôn mặt", "error"

    if face.occluded_part_name:
        return False, f"Phát hiện che khuất {face.occluded_part_name} – Vui lòng để lộ toàn bộ khuôn mặt", "error"

    if face.mask_detected:
        return False, "Vui lòng tháo khẩu trang để tiếp tục", "error"

    if face.sunglasses_detected:
        return False, "Vui lòng tháo kính râm / kính đen để tiếp tục", "error"

    if face.glare_detected:
        return False, "Nghiêng mặt nhẹ để tránh lóa kính", "warn"

    # Kiểm tra ánh sáng khuôn mặt (Illumination)
    if face.brightness_mean < C.FACE_BRIGHTNESS_MIN:
        return False, "Không gian quá tối – hãy tăng thêm ánh sáng", "error"
    if face.brightness_mean > C.FACE_BRIGHTNESS_MAX:
        return False, "Ánh sáng quá mạnh – tránh đèn chiếu thẳng vào mặt", "error"
    if face.brightness_std < C.FACE_BRIGHTNESS_STD_MIN and not is_turning:
        return False, "Khuôn mặt bị ngược sáng / thiếu chi tiết", "warn"

    return True, "", "ok"


def quality_checks(face: FaceResult, require_oval: bool = True, is_turning: bool = False) -> tuple[dict, str, str]:
    """Áp ngưỡng FQA theo tài liệu thiết kế chi tiết 4 tiêu chí.
    Trả về (checks, message, severity: ok|warn|error).
    """
    has_eyes = bool(face.parts_status.get("left_eye", True) and face.parts_status.get("right_eye", True))
    has_nose = bool(face.parts_status.get("nose", True))
    has_mouth = bool(face.parts_status.get("mouth", True))
    has_eyebrows = bool(face.parts_status.get("left_eyebrow", True) and face.parts_status.get("right_eyebrow", True))
    all_features_detected = bool(face.detected and not face.occluded_part_name and not face.hand_occlusion)

    head_straight = (
        face.detected and not face.is_upside_down and face.yaw is not None
        and abs(face.yaw) <= C.MAX_STRAIGHT["yaw"]
        and abs(face.pitch) <= C.MAX_STRAIGHT["pitch"]
        and abs(face.roll) <= C.MAX_STRAIGHT["roll"]
    ) if not is_turning else True

    sharpness_ok = bool(face.detected)

    scale_ok = face.detected and (C.FACE_SCALE_RANGE[0] <= face.scale_ratio <= C.FACE_SCALE_RANGE[1])
    illumination_ok = face.detected and (C.FACE_BRIGHTNESS_MIN <= face.brightness_mean <= C.FACE_BRIGHTNESS_MAX)
    no_backlight = face.detected and (face.brightness_std >= C.FACE_BRIGHTNESS_STD_MIN)

    # Chống giả mạo ảnh & video
    anti_spoof_ok = bool(face.detected and (not face.anti_spoof or face.anti_spoof.is_real))
    no_print_attack = bool(face.detected and (not face.anti_spoof or (face.anti_spoof.spoof_type != "print_attack" and face.anti_spoof.spoof_type != "planar_2d")))
    no_screen_attack = bool(face.detected and (not face.anti_spoof or (face.anti_spoof.spoof_type != "replay_attack" and face.anti_spoof.spoof_type != "screen_moire")))
    depth_3d_ok = bool(face.detected and (not face.anti_spoof or face.anti_spoof.depth_3d_ok))

    checks = {
        "face_detected": face.detected,
        "single_face": face.face_count == 1,
        "not_upside_down": bool(face.detected and not face.is_upside_down),
        "anti_spoof_ok": anti_spoof_ok,
        "no_print_attack": no_print_attack,
        "no_screen_attack": no_screen_attack,
        "depth_3d_ok": depth_3d_ok,
        "all_features_detected": all_features_detected,
        "has_eyes": has_eyes,
        "has_nose": has_nose,
        "has_mouth": has_mouth,
        "has_eyebrows": has_eyebrows,
        "no_mask": face.detected and not face.mask_detected,
        "no_sunglasses": face.detected and not face.sunglasses_detected,
        "no_glare": face.detected and not face.glare_detected,
        "no_hand_occlusion": face.detected and not face.hand_occlusion,
        "scale_ok": scale_ok,
        "inside_oval": face.detected and (face.oval_dist <= C.OVAL_INSIDE_TOL and face.corners_inside),
        "centered_ok": face.detected and abs(face.offset[0]) <= C.MAX_CENTER_OFFSET and abs(face.offset[1]) <= C.MAX_CENTER_OFFSET,
        "head_straight": head_straight,
        "illumination_ok": illumination_ok,
        "no_backlight": no_backlight,
        "sharpness_ok": sharpness_ok,
        "distance_ok": scale_ok,  # Alias cho tương thích
    }
    if not require_oval:
        checks["inside_oval"] = checks["centered_ok"] = checks["scale_ok"] = checks["distance_ok"] = face.detected

    # 1. Kiểm tra tiêu chuẩn chất lượng liên tục trước (Vật cản, Khẩu trang, Kính, Ánh sáng, Độ nét, Lật ngược đầu)
    cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face, is_turning=is_turning)
    if not cont_ok:
        return checks, cont_msg, cont_sev

    # 2. Kiểm tra Framing & Khung Oval (Geometry & Scale - Hướng dẫn cự ly tiến lại gần / lùi ra xa)
    if require_oval:
        if not checks["scale_ok"]:
            if face.scale_ratio < C.FACE_SCALE_RANGE[0]:
                return checks, "Hãy tiến lại gần camera hơn", "warn"
            return checks, "Hãy lùi ra xa camera một chút", "warn"
        if not checks["inside_oval"]:
            if face.scale_ratio > 0.72 or face.fill > 0.82:
                return checks, "Khuôn mặt tràn khung – Hãy lùi ra xa camera một chút và căn vào giữa khung Oval", "warn"
            if face.scale_ratio < 0.48 or face.fill < 0.52:
                return checks, "Khuôn mặt quá nhỏ – Hãy tiến lại gần camera hơn và căn vào giữa khung Oval", "warn"
            dx_screen, dy = -face.offset[0], face.offset[1]
            if abs(dx_screen) > 0.20:
                return checks, ("Dịch mặt sang trái" if dx_screen > 0 else "Dịch mặt sang phải") + " vào giữa khung Oval", "warn"
            if abs(dy) > 0.20:
                return checks, ("Hạ mặt xuống một chút" if dy < 0 else "Nâng mặt lên một chút") + " vào giữa khung Oval", "warn"
            return checks, "Đưa toàn bộ khuôn mặt vào giữa khung Oval", "warn"
        if not checks["centered_ok"]:
            dx_screen, dy = -face.offset[0], face.offset[1]
            if abs(dx_screen) >= abs(dy):
                return checks, ("Dịch mặt sang trái" if dx_screen > 0 else "Dịch mặt sang phải") + " vào giữa khung Oval", "warn"
            return checks, ("Hạ mặt xuống một chút" if dy < 0 else "Nâng mặt lên một chút") + " vào giữa khung Oval", "warn"

    # 3. Kiểm tra nhìn thẳng & độ nghiêng đầu (Roll / Pitch / Yaw)
    if not checks["head_straight"]:
        if face.is_upside_down:
            return checks, "Khuôn mặt bị lật ngược – Vui lòng giữ thẳng đầu (mắt ở trên, miệng ở dưới)", "error"
        if face.roll is not None and abs(face.roll) > C.MAX_STRAIGHT["roll"]:
            if abs(face.roll) > 25.0:
                return checks, "Đang nghiêng đầu quá nhiều – Vui lòng giữ thẳng đầu", "warn"
            if face.roll > C.MAX_STRAIGHT["roll"]:
                return checks, "Đang nghiêng đầu sang trái – Vui lòng giữ thẳng đầu", "warn"
            return checks, "Đang nghiêng đầu sang phải – Vui lòng giữ thẳng đầu", "warn"
        if face.pitch is not None and abs(face.pitch) > C.MAX_STRAIGHT["pitch"]:
            if face.pitch > C.MAX_STRAIGHT["pitch"]:
                return checks, "Đang ngẩng đầu quá cao – Hãy hạ cằm xuống và nhìn thẳng", "warn"
            return checks, "Đang cúi đầu quá thấp – Hãy nâng cằm lên và nhìn thẳng", "warn"
        if face.yaw is not None and abs(face.yaw) > C.MAX_STRAIGHT["yaw"]:
            if face.yaw > C.MAX_STRAIGHT["yaw"]:
                return checks, "Đang quay mặt sang trái – Hãy nhìn thẳng vào camera", "warn"
            return checks, "Đang quay mặt sang phải – Hãy nhìn thẳng vào camera", "warn"
        return checks, "Hãy nhìn thẳng vào camera", "warn"

    return checks, "Khuôn mặt hợp lệ! Giữ yên...", "ok"
