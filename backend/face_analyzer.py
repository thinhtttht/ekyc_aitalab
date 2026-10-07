"""(b) Đánh giá chất lượng khuôn mặt (FQA) bằng MediaPipe FaceMesh + OpenCV.

Trả về FaceResult chứa số liệu thô; hàm `evaluate_face` áp ngưỡng để ra checks/thông báo.
Lưu ý: frontend lật gương ảnh TRƯỚC khi gửi lên, nên trục x của ảnh trùng với màn hình người
dùng đang nhìn: x nhỏ = bên trái màn hình = bên trái của người dùng (như soi gương).
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
_LOWER_FACE = [50, 280, 205, 425, 118, 347, 187, 411]  # má dưới hai bên (tránh chóp cằm 152 hay bị bóng đổ trần)
_IRIS_CENTERS = [468, 473]              # tâm mống mắt (refine_landmarks)
_EYE_FALLBACK = [159, 386]              # mí trên nếu không có iris

PARTS_CONFIG = {
    "left_eye": {
        "indices": [33, 160, 158, 133, 153, 144],
        "name": "Mắt trái",
        "min_edge": 4.0,
        "min_contrast": 25.0,
    },
    "right_eye": {
        "indices": [362, 385, 387, 263, 373, 380],
        "name": "Mắt phải",
        "min_edge": 4.0,
        "min_contrast": 25.0,
    },
    "left_eyebrow": {
        "indices": [70, 63, 105, 66, 107],
        "name": "Chân mày trái",
        "min_edge": 1.8,
        "min_contrast": 18.0,
    },
    "right_eyebrow": {
        "indices": [336, 296, 334, 293, 300],
        "name": "Chân mày phải",
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
    offset: tuple = (0.0, 0.0)                   # lệch tâm (chuẩn hoá theo bán trục); x > 0 = lệch sang phải màn hình
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
        if not faces:
            return FaceResult(face_count=0)

        # Chọn mặt lớn nhất và tính diện tích
        def face_area(f):
            xs = [p.x for p in f.landmark]
            ys = [p.y for p in f.landmark]
            return float((max(xs) - min(xs)) * (max(ys) - min(ys)))

        sorted_faces = sorted(faces, key=face_area, reverse=True)
        primary_face = sorted_faces[0]
        primary_area = face_area(primary_face)

        # Chỉ tính là "nhiều khuôn mặt" khi khuôn mặt thứ 2 có kích thước đáng kể
        # (diện tích >= 20% mặt chính VÀ diện tích >= 0.025 tổng khung hình).
        # Lọc bỏ hoàn toàn các vết nhiễu / tranh ảnh nhỏ xíu ở hậu cảnh.
        valid_faces = [
            f for f in sorted_faces
            if face_area(f) >= max(0.025, primary_area * 0.20)
        ]
        out = FaceResult(face_count=len(valid_faces))
        face = primary_face
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

    # 4 góc Bounding Box (CHỈ MANG TÍNH THAM KHẢO / DEBUG).
    # KHÔNG dùng để quyết định "lọt khung": khuôn mặt là hình elip, 4 góc hình chữ nhật
    # bao quanh luôn nhô ra ngoài elip. Với mặt căn giữa hoàn hảo, góc bbox sẽ vượt elip
    # khi scale² + fill² > 1 (≈ scale ≥ 0.70) -> mâu thuẫn với dải scale hợp lệ 0.40–0.85.
    # Việc kiểm tra lọt khung dùng `oval_dist` trên 36 điểm viền mặt thật (FACEMESH_FACE_OVAL).
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

    face_w_px = max(1.0, (x1 - x0) * w)

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

        # Kiểm tra hình học & chất liệu sinh học chuyên sâu cho Vùng Miệng
        # (Chống bị đánh lừa bởi chữ in / hoa văn trên bao bì khăn giấy, sách báo, điện thoại, khẩu trang)
        if is_ok and p_key == "mouth":
            w_m = float(np.linalg.norm(px[61] - px[291]))
            h_m = float(np.linalg.norm(px[0] - px[17]))
            m_aspect = w_m / max(1.0, h_m)
            # Ngũ quan miệng người bình thường có tỷ lệ w/h <= 3.5 và chiều cao h_m >= 3.0px
            if m_aspect > 3.5 or h_m < 3.0:
                is_ok = False

            # Phân tích chất liệu trong ROI miệng
            roi_bgr = frame_bgr[max(0, int(sy0 - pad)):min(h, int(sy1 + pad)), max(0, int(sx0 - pad)):min(w, int(sx1 + pad))]
            if is_ok and roi_bgr.size > 0:
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                # Giấy trắng tẩy trắng / bao bì nilon phản quang (V > 180 và S < 32)
                white_paper_mask = (roi_hsv[:, :, 2] > 180) & (roi_hsv[:, :, 1] < 32)
                paper_ratio = float(np.count_nonzero(white_paper_mask)) / max(1, roi_bgr.shape[0] * roi_bgr.shape[1])
                # Màu sắc phi sinh học (xanh lá, xanh cyan, tím vải khẩu trang)
                unnatural_mask = (roi_hsv[:, :, 0] >= 40) & (roi_hsv[:, :, 0] <= 155) & (roi_hsv[:, :, 1] >= 40)
                unnatural_ratio = float(np.count_nonzero(unnatural_mask)) / max(1, roi_bgr.shape[0] * roi_bgr.shape[1])

                # Mốc 17 (đáy bờ môi dưới): kiểm tra xem có bị giấy/nilon che không
                p17_y, p17_x = int(px[17, 1]), int(px[17, 0])
                p17_patch = frame_bgr[max(0, p17_y - 2):min(h, p17_y + 3), max(0, p17_x - 2):min(w, p17_x + 3)]
                p17_is_paper = False
                if p17_patch.size > 0:
                    p17_hsv = cv2.cvtColor(p17_patch, cv2.COLOR_BGR2HSV)
                    p17_is_paper = bool(np.mean(p17_hsv[:, :, 1]) < 32 and np.mean(p17_hsv[:, :, 2]) > 180)

                if paper_ratio >= 0.15 or unnatural_ratio >= 0.25 or p17_is_paper:
                    is_ok = False

        parts_status[p_key] = is_ok
        if not is_ok and occluded_name is None:
            occluded_name = p_cfg["name"]

    # --- Kiểm tra vật cản che nửa dưới mặt / cằm & quai hàm ---
    # Phát hiện các trường hợp đưa khăn giấy, tờ giấy, sách báo, điện thoại hoặc khẩu trang che cằm
    chin_jaw_indices = [152, 175, 199, 148, 176, 377, 400]
    chin_paper_count = 0
    chin_plastic_count = 0
    for idx in chin_jaw_indices:
        pt = px[idx]
        patch = frame_bgr[max(0, int(pt[1] - 2)):min(h, int(pt[1] + 3)), max(0, int(pt[0] - 2)):min(w, int(pt[0] + 3))]
        if patch.size == 0:
            continue
        p_ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        p_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        cr_val = float(np.mean(p_ycc[:, :, 1]))
        cb_val = float(np.mean(p_ycc[:, :, 2]))
        s_val = float(np.mean(p_hsv[:, :, 1]))
        v_val = float(np.mean(p_hsv[:, :, 2]))
        if s_val < 32 and v_val > 180:
            chin_paper_count += 1
        if (cb_val >= cr_val + 3) and (cr_val < 120):
            chin_plastic_count += 1

    lower_face_occluded = bool(chin_paper_count >= 2 or chin_plastic_count >= 3)
    if lower_face_occluded:
        parts_status["mouth"] = False
        if occluded_name is None:
            occluded_name = "nửa dưới khuôn mặt (vật cản / giấy che mặt)"

    out.parts_status = parts_status
    out.occluded_part_name = occluded_name

    # --- Heuristic khẩu trang & vật cản thông minh (Robust Mask & Occlusion Detection) ---
    # Khẩu trang thực tế (y tế hoặc vải) che kín MŨI VÀ MIỆNG.
    # Nếu miệng và mũi đều rõ nét (có bờ môi, viền môi, chóp mũi) -> chắc chắn KHÔNG đeo khẩu trang!
    has_mouth = bool(parts_status.get("mouth", True))
    has_nose = bool(parts_status.get("nose", True))

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

    if has_mouth and has_nose:
        # Miệng và mũi nhìn thấy rõ ràng -> không phải khẩu trang!
        # Chỉ báo nếu vùng mặt dưới bị phủ màu nhân tạo hoàn toàn (vải xanh y tế: Cb > 145 & Cr < 120, hoặc đen kịt Y < 25)
        artificial_mask_color = bool((cbl > 145 and crl < 120) or yl < 25 or lower_face_occluded)
        out.mask_detected = artificial_mask_color
    else:
        # Vùng miệng hoặc mũi bị che khuất -> kiểm tra xem có phải do khẩu trang / vật cản che mặt không
        if upper_skin and not lower_skin:
            out.mask_detected = True
        elif not upper_skin and not lower_skin:
            out.mask_detected = True
        else:
            out.mask_detected = bool(
                chroma > C.MASK_CHROMA_DIST or y_ratio < C.MASK_DARK_RATIO
                or (cbl > 135 and crl < 125) or lower_face_occluded or not has_mouth
            )

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


# ---------------------------------------------------------------------------
# ĐÁNH GIÁ CHẤT LƯỢNG (FQA) - chạy một lần cho mỗi frame
# ---------------------------------------------------------------------------

Issue = Optional[tuple]  # (message, severity) hoặc None nếu không có lỗi


@dataclass
class FaceVerdict:
    checks: dict
    message: str
    severity: str        # ok | warn | error
    continuous_ok: bool  # đạt các tiêu chí phải giữ suốt quy trình (giả mạo, che khuất, ánh sáng)

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def evaluate_face(face: Optional[FaceResult], require_oval: bool = True, is_turning: bool = False) -> FaceVerdict:
    """Áp toàn bộ ngưỡng FQA, trả về checks hiển thị và một thông báo có độ ưu tiên cao nhất."""
    if face is None:
        return FaceVerdict({}, "Vui lòng đưa mặt vào khung hình", "warn", False)

    checks = _build_checks(face, require_oval, is_turning)
    continuous = _continuous_issue(face, is_turning)
    issue = (
        continuous
        or (require_oval and _framing_issue(face, checks))
        or (not checks["head_straight"] and _pose_issue(face))
        or (not checks["sharpness_ok"] and ("Hình ảnh bị mờ – Giữ yên đầu và camera", "warn"))
    )
    message, severity = issue or ("Khuôn mặt hợp lệ! Giữ yên...", "ok")
    return FaceVerdict(checks, message, severity, continuous_ok=continuous is None)


def quality_checks(face: FaceResult, require_oval: bool = True, is_turning: bool = False) -> tuple[dict, str, str]:
    """Trả về (checks, message, severity) – dạng tuple của `evaluate_face`."""
    v = evaluate_face(face, require_oval, is_turning)
    return v.checks, v.message, v.severity


def check_continuous_face_quality(face: Optional[FaceResult], is_turning: bool = False) -> tuple[bool, str, str]:
    """Trả về (is_ok, message, severity) cho các tiêu chí phải giữ suốt quy trình."""
    issue = _continuous_issue(face, is_turning)
    return (True, "", "ok") if issue is None else (False, *issue)


def _build_checks(face: FaceResult, require_oval: bool, is_turning: bool) -> dict:
    detected = face.detected
    parts = face.parts_status
    spoof = face.anti_spoof
    spoof_type = spoof.spoof_type if spoof else "real"

    head_straight = True if is_turning else bool(
        detected and not face.is_upside_down and face.yaw is not None
        and abs(face.yaw) <= C.MAX_STRAIGHT["yaw"]
        and abs(face.pitch) <= C.MAX_STRAIGHT["pitch"]
        and abs(face.roll) <= C.MAX_STRAIGHT["roll"]
    )
    min_sharpness = C.MIN_FACE_SHARPNESS_TURNING if is_turning else C.MIN_FACE_SHARPNESS

    checks = {
        "face_detected": detected,
        "single_face": face.face_count == 1,
        "not_upside_down": bool(detected and not face.is_upside_down),
        "anti_spoof_ok": bool(detected and (not spoof or spoof.is_real)),
        "no_print_attack": bool(detected and spoof_type not in ("print_attack", "planar_2d")),
        "no_screen_attack": bool(detected and spoof_type not in ("replay_attack", "screen_moire")),
        "depth_3d_ok": bool(detected and (not spoof or spoof.depth_3d_ok)),
        "all_features_detected": bool(detected and not face.occluded_part_name and not face.hand_occlusion),
        "has_eyes": bool(parts.get("left_eye", True) and parts.get("right_eye", True)),
        "has_nose": bool(parts.get("nose", True)),
        "has_mouth": bool(parts.get("mouth", True)),
        "has_eyebrows": bool(parts.get("left_eyebrow", True) and parts.get("right_eyebrow", True)),
        "no_mask": detected and not face.mask_detected,
        "no_sunglasses": detected and not face.sunglasses_detected,
        "no_glare": detected and not face.glare_detected,
        "no_hand_occlusion": detected and not face.hand_occlusion,
        "scale_ok": detected and C.FACE_SCALE_RANGE[0] <= face.scale_ratio <= C.FACE_SCALE_RANGE[1],
        "inside_oval": detected and face.oval_dist <= C.OVAL_INSIDE_TOL,
        "centered_ok": detected and abs(face.offset[0]) <= C.MAX_CENTER_OFFSET and abs(face.offset[1]) <= C.MAX_CENTER_OFFSET,
        "head_straight": head_straight,
        "illumination_ok": detected and C.FACE_BRIGHTNESS_MIN <= face.brightness_mean <= C.FACE_BRIGHTNESS_MAX,
        "no_backlight": detected and face.brightness_std >= C.FACE_BRIGHTNESS_STD_MIN,
        "sharpness_ok": bool(detected and (not C.SHARPNESS_CHECK_ENABLED or face.sharpness >= min_sharpness)),
    }
    if not require_oval:
        checks["inside_oval"] = checks["centered_ok"] = checks["scale_ok"] = detected
    return checks


def _continuous_issue(face: Optional[FaceResult], is_turning: bool) -> Issue:
    """Giả mạo, che khuất, ánh sáng: phải đạt ở MỌI giai đoạn của quy trình."""
    if face is None or not face.detected:
        return "Vui lòng đưa mặt vào khung hình", "warn"
    if face.face_count > 1:
        return "Phát hiện nhiều khuôn mặt – chỉ một người duy nhất trong khung hình", "error"
    if face.anti_spoof and not face.anti_spoof.is_real:
        return face.anti_spoof.message, face.anti_spoof.severity
    if face.is_upside_down:
        return "Khuôn mặt bị lật ngược – Vui lòng giữ thẳng đầu (mắt ở trên, miệng ở dưới)", "error"
    if face.hand_occlusion:
        return "Vui lòng bỏ tay ra khỏi khuôn mặt", "error"
    if face.occluded_part_name:
        return f"Phát hiện che khuất {face.occluded_part_name} – Vui lòng để lộ toàn bộ khuôn mặt", "error"
    if face.mask_detected:
        return "Vui lòng tháo khẩu trang để tiếp tục", "error"
    if face.sunglasses_detected:
        return "Vui lòng tháo kính râm / kính đen để tiếp tục", "error"
    if face.glare_detected:
        return "Nghiêng mặt nhẹ để tránh lóa kính", "warn"
    if face.brightness_mean < C.FACE_BRIGHTNESS_MIN:
        return "Không gian quá tối – hãy tăng thêm ánh sáng", "error"
    if face.brightness_mean > C.FACE_BRIGHTNESS_MAX:
        return "Ánh sáng quá mạnh – tránh đèn chiếu thẳng vào mặt", "error"
    if face.brightness_std < C.FACE_BRIGHTNESS_STD_MIN and not is_turning:
        return "Khuôn mặt bị ngược sáng / thiếu chi tiết", "warn"
    return None


def _framing_issue(face: FaceResult, checks: dict) -> Issue:
    """Cự ly và vị trí mặt so với khung Oval."""
    if not checks["scale_ok"]:
        if face.scale_ratio < C.FACE_SCALE_RANGE[0]:
            return "Hãy tiến lại gần camera hơn", "warn"
        return "Hãy lùi ra xa camera một chút", "warn"
    if not checks["inside_oval"]:
        dx, dy = face.offset
        off_center = abs(dx) > 0.20 or abs(dy) > 0.20
        # Chỉ báo "lùi ra xa" khi mặt thực sự quá lớn và KHÔNG phải do lệch tâm gây ra.
        if face.fill > C.FACE_FILL[1] or (not off_center and face.scale_ratio > 0.72):
            return "Khuôn mặt tràn khung – Hãy lùi ra xa camera một chút và căn vào giữa khung Oval", "warn"
        if face.scale_ratio < 0.48 or face.fill < 0.52:
            return "Khuôn mặt quá nhỏ – Hãy tiến lại gần camera hơn và căn vào giữa khung Oval", "warn"
        if off_center:
            return _shift_hint(dx, dy), "warn"
        return "Đưa toàn bộ khuôn mặt vào giữa khung Oval", "warn"
    if not checks["centered_ok"]:
        return _shift_hint(*face.offset), "warn"
    return None


def _pose_issue(face: FaceResult) -> tuple:
    """Nhìn thẳng & độ nghiêng đầu (Roll / Pitch / Yaw)."""
    lim = C.MAX_STRAIGHT
    if face.is_upside_down:
        return "Khuôn mặt bị lật ngược – Vui lòng giữ thẳng đầu (mắt ở trên, miệng ở dưới)", "error"
    if face.roll is not None and abs(face.roll) > lim["roll"]:
        if abs(face.roll) > 25.0:
            return "Đang nghiêng đầu quá nhiều – Vui lòng giữ thẳng đầu", "warn"
        side = "trái" if face.roll > 0 else "phải"
        return f"Đang nghiêng đầu sang {side} – Vui lòng giữ thẳng đầu", "warn"
    if face.pitch is not None and abs(face.pitch) > lim["pitch"]:
        if face.pitch > 0:
            return "Đang ngẩng đầu quá cao – Hãy hạ cằm xuống và nhìn thẳng", "warn"
        return "Đang cúi đầu quá thấp – Hãy nâng cằm lên và nhìn thẳng", "warn"
    if face.yaw is not None and abs(face.yaw) > lim["yaw"]:
        side = "trái" if face.yaw > 0 else "phải"
        return f"Đang quay mặt sang {side} – Hãy nhìn thẳng vào camera", "warn"
    return "Hãy nhìn thẳng vào camera", "warn"


def _shift_hint(dx: float, dy: float) -> str:
    """Hướng dẫn dịch mặt về tâm oval. Ảnh đã lật gương nên trái/phải trùng với màn hình."""
    if abs(dx) >= abs(dy):
        direction = "Dịch mặt sang trái" if dx > 0 else "Dịch mặt sang phải"
    else:
        direction = "Hạ mặt xuống một chút" if dy < 0 else "Nâng mặt lên một chút"
    return direction + " vào giữa khung Oval"

