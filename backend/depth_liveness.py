"""3D Depth Liveness Detection & Anti-Spoofing using RealSense Depth Data.

Detects Presentation Attacks (printed photos, phone/tablet screens, 2D masks)
by analyzing physical 3D facial geometry:
1. Nose-to-cheek relief (nose tip protrudes forward by 20-65mm on real faces, near 0mm on flat screens/photos).
2. Non-planar curvature residuals (plane-fitting RMS error: flat screens < 4mm, real faces >= 12mm).
3. Depth standard deviation across facial topography.
4. Physical distance range validation (35cm - 100cm for valid eKYC sitting distance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class DepthLivenessResult:
    """Kết quả phân tích chiều sâu sinh trắc học 3D chống giả mạo."""
    is_real_3d: bool = False
    liveness_score: float = 0.0          # Điểm độ thật 3D (0.0 -> 1.0)
    spoof_type: str = "unknown"          # 'real_human', 'planar_2d_spoof', 'out_of_range_too_close', 'out_of_range_too_far', 'no_depth_signal'
    distance_mm: float = 0.0             # Khoảng cách từ camera đến mặt (mm)
    nose_relief_mm: float = 0.0          # Độ lồi sống mũi so với 2 gò má (mm)
    curvature_residual_mm: float = 0.0   # Độ cong sai số phi phẳng (Plane fit RMS)
    depth_std_mm: float = 0.0            # Độ lệch chuẩn chiều sâu vùng mặt (mm)
    valid_depth_ratio: float = 0.0       # Tỷ lệ pixel có tín hiệu chiều sâu hợp lệ (>0)
    message: str = ""                    # Thông điệp trạng thái cho người dùng
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def distance_cm(self) -> float:
        return self.distance_mm / 10.0


class DepthLivenessDetector:
    """Bộ kiểm tra người thật thời gian thực dựa trên luồng chiều sâu 3D (RealSense Depth)."""

    def __init__(
        self,
        min_distance_mm: float = 350.0,   # Cự ly tối thiểu (35cm)
        max_distance_mm: float = 1000.0,  # Cự ly tối đa (100cm)
        min_nose_relief_mm: float = 10.0, # Độ nhô sống mũi tối thiểu của mặt người thật (10mm)
        min_curvature_rms: float = 6.0,   # Sai số độ cong phi phẳng tối thiểu (6mm)
        min_depth_std_mm: float = 8.0,    # Độ lệch chuẩn chiều sâu tối thiểu (8mm)
        min_liveness_score: float = 0.50, # Điểm ngưỡng kết luận người thật
    ):
        self.min_dist = min_distance_mm
        self.max_dist = max_distance_mm
        self.min_relief = min_nose_relief_mm
        self.min_rms = min_curvature_rms
        self.min_std = min_depth_std_mm
        self.threshold = min_liveness_score

    def analyze(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[float, float, float, float],
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> DepthLivenessResult:
        """Phân tích ma trận chiều sâu tại vùng khuôn mặt.

        Args:
            depth_map: Mảng 2D (H, W) chứa giá trị chiều sâu theo mm (uint16 hoặc float).
            bbox: (x0, y0, x1, y1) toạ độ chuẩn hoá [0, 1] hoặc toạ độ pixel của khuôn mặt.
            landmarks: Dict các điểm mốc (nếu có: 'nose', 'left_cheek', 'right_cheek', ...).
            keypoints: Mảng các điểm mốc (ví dụ MediaPipe 468 mốc x, y).

        Returns:
            DepthLivenessResult chứa thông số và kết luận.
        """
        if depth_map is None or depth_map.size == 0:
            return DepthLivenessResult(
                is_real_3d=False,
                spoof_type="no_depth_signal",
                message="Không có tín hiệu chiều sâu từ camera 3D",
            )

        h, w = depth_map.shape[:2]

        # Chuẩn hoá toạ độ bbox sang pixel
        bx0, by0, bx1, by1 = bbox
        if max(abs(bx0), abs(by0), abs(bx1), abs(by1)) <= 2.0:
            px0, py0 = int(bx0 * w), int(by0 * h)
            px1, py1 = int(bx1 * w), int(by1 * h)
        else:
            px0, py0, px1, py1 = int(bx0), int(by0), int(bx1), int(by1)

        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w - 1, px1), min(h - 1, py1)

        face_w = px1 - px0
        face_h = py1 - py0
        if face_w < 30 or face_h < 30:
            return DepthLivenessResult(
                is_real_3d=False,
                spoof_type="face_too_small",
                message="Khuôn mặt quá nhỏ hoặc ở quá xa để phân tích chiều sâu",
            )

        # Trích xuất vùng chiều sâu khuôn mặt
        face_depth = depth_map[py0:py1, px0:px1].astype(np.float32)

        # Lọc các điểm hợp lệ (> 0 mm và < 3000 mm)
        valid_mask = (face_depth > 150.0) & (face_depth < 3000.0)
        valid_count = np.count_nonzero(valid_mask)
        total_pixels = face_depth.size
        valid_ratio = valid_count / max(1, total_pixels)

        if valid_ratio < 0.20:
            return DepthLivenessResult(
                is_real_3d=False,
                valid_depth_ratio=valid_ratio,
                spoof_type="no_depth_signal",
                message="Tín hiệu chiều sâu hồng ngoại vùng mặt bị mất hoặc chói sáng",
            )

        valid_depths = face_depth[valid_mask]
        dist_mm = float(np.median(valid_depths))

        # 1. Kiểm tra Cự ly hợp lệ (Distance Gate)
        if dist_mm < self.min_dist:
            return DepthLivenessResult(
                is_real_3d=False,
                distance_mm=dist_mm,
                valid_depth_ratio=valid_ratio,
                spoof_type="out_of_range_too_close",
                message=f"Quá gần camera ({dist_mm/10:.1f} cm). Vui lòng lùi xa ra (~50-70 cm)",
            )
        if dist_mm > self.max_dist:
            return DepthLivenessResult(
                is_real_3d=False,
                distance_mm=dist_mm,
                valid_depth_ratio=valid_ratio,
                spoof_type="out_of_range_too_far",
                message=f"Quá xa camera ({dist_mm/10:.1f} cm). Vui lòng ngồi lại gần (~50-70 cm)",
            )

        # 2. Đo độ nhô sống mũi (Nose-to-Cheek Relief)
        # Sống mũi nằm ở trung tâm mặt (y: 35-55%, x: 40-60%)
        # Gò má trái (x: 15-32%, y: 40-65%), gò má phải (x: 68-85%, y: 40-65%)
        fh, fw = face_depth.shape[:2]

        nose_crop = face_depth[int(fh * 0.35) : int(fh * 0.58), int(fw * 0.40) : int(fw * 0.60)]
        l_cheek_crop = face_depth[int(fh * 0.40) : int(fh * 0.65), int(fw * 0.15) : int(fw * 0.32)]
        r_cheek_crop = face_depth[int(fh * 0.40) : int(fh * 0.65), int(fw * 0.68) : int(fw * 0.85)]

        def safe_median(arr):
            v = arr[(arr > 150.0) & (arr < 3000.0)]
            return float(np.median(v)) if len(v) >= 5 else None

        z_nose = safe_median(nose_crop)
        z_lcheek = safe_median(l_cheek_crop)
        z_rcheek = safe_median(r_cheek_crop)

        if z_nose is not None and (z_lcheek is not None or z_rcheek is not None):
            cheeks = [c for c in (z_lcheek, z_rcheek) if c is not None]
            avg_cheeks = float(np.mean(cheeks))
            # Ở mặt người thật: Mũi nhô ra gần camera hơn má -> z_nose < avg_cheeks -> relief > 0
            nose_relief = avg_cheeks - z_nose
        else:
            nose_relief = 0.0

        # 3. Phân tích độ cong phi phẳng (Plane Fitting Residual)
        # Lấy toạ độ lưới (x, y, z) của các điểm hợp lệ trong vùng mặt
        # Ở ảnh phẳng hoặc màn hình điện thoại: Các điểm nằm trên 1 mặt phẳng z = Ax + By + C
        # Sai số residual sẽ cực nhỏ (< 4mm). Ở mặt thật, mũi, hốc mắt, cằm làm tăng residual >= 12mm.
        ys, xs = np.where(valid_mask)
        zs = face_depth[valid_mask]

        # Lấy mẫu tối đa ~250 điểm theo bước nhảy đều (cực nhanh < 0.1ms, không phân mảnh RAM)
        if len(zs) > 250:
            step = len(zs) // 250
            xs_sub = xs[::step].astype(np.float64)
            ys_sub = ys[::step].astype(np.float64)
            zs_sub = zs[::step].astype(np.float64)
        else:
            xs_sub = xs.astype(np.float64)
            ys_sub = ys.astype(np.float64)
            zs_sub = zs.astype(np.float64)

        if len(zs_sub) >= 30:
            # Khớp mặt phẳng bậc 1: Z = a*X + b*Y + c
            A = np.column_stack([xs_sub, ys_sub, np.ones_like(xs_sub)])
            try:
                plane_coeffs, residuals, rank, s = np.linalg.lstsq(A, zs_sub, rcond=None)
                z_pred = A @ plane_coeffs
                diff = np.abs(zs_sub - z_pred)
                # Loại bỏ 5% outlier ở viền tóc/cổ
                inlier_diff = diff[diff < np.percentile(diff, 95)]
                curvature_rms = float(np.sqrt(np.mean(inlier_diff ** 2)))
            except Exception:
                curvature_rms = 0.0
        else:
            curvature_rms = 0.0

        # 4. Độ lệch chuẩn chiều sâu (Depth Standard Deviation)
        depth_std = float(np.std(zs_sub)) if len(zs_sub) >= 20 else 0.0

        # 5. Tính điểm tổng hợp 3D Liveness Score
        # Chuẩn hoá các thành phần về dải [0, 1]
        score_relief = float(np.clip((nose_relief - 3.0) / 18.0, 0.0, 1.0))
        score_rms = float(np.clip((curvature_rms - 3.0) / 10.0, 0.0, 1.0))
        score_std = float(np.clip((depth_std - 5.0) / 14.0, 0.0, 1.0))

        # Trọng số: Độ lồi sống mũi (40%) + Độ cong phi phẳng (35%) + Độ phân tán (25%)
        liveness_score = 0.40 * score_relief + 0.35 * score_rms + 0.25 * score_std

        # Kết luận người thật vs Giả mạo màn hình / ảnh phẳng
        is_real = (
            liveness_score >= self.threshold
            and nose_relief >= self.min_relief * 0.70
            and curvature_rms >= self.min_rms * 0.70
        )

        if is_real:
            spoof_type = "real_human"
            msg = f"✅ Người thật 3D ({liveness_score*100:.0f}% | Mũi: +{nose_relief:.1f}mm | Cự ly: {dist_mm/10:.1f}cm)"
        else:
            spoof_type = "planar_2d_spoof"
            if nose_relief < 8.0 and curvature_rms < 6.0:
                msg = f"❌ CẢNH BÁO: Phát hiện ảnh/màn hình phẳng 2D! (Độ lồi mũi: {nose_relief:.1f}mm)"
            else:
                msg = f"⚠️ Độ sâu 3D chưa đạt ({liveness_score*100:.0f}% < {self.threshold*100:.0f}%). Vui lòng nhìn thẳng camera"

        return DepthLivenessResult(
            is_real_3d=is_real,
            liveness_score=round(liveness_score, 3),
            spoof_type=spoof_type,
            distance_mm=round(dist_mm, 1),
            nose_relief_mm=round(nose_relief, 1),
            curvature_residual_mm=round(curvature_rms, 1),
            depth_std_mm=round(depth_std, 1),
            valid_depth_ratio=round(valid_ratio, 2),
            message=msg,
            details={
                "z_nose": z_nose,
                "z_lcheek": z_lcheek,
                "z_rcheek": z_rcheek,
                "score_relief": round(score_relief, 3),
                "score_rms": round(score_rms, 3),
                "score_std": round(score_std, 3),
            },
        )
