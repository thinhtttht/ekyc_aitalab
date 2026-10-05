"""ArcFace Feature Extractor sử dụng ONNX Runtime & InsightFace WebFace600K ResNet-50.

Trích xuất vector đặc trưng khuôn mặt 512 chiều đã chuẩn hóa L2 (Unit Hypersphere)
phục vụ cho định danh 1:1 (Verification) và 1:N (Identification).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import insightface.utils.face_align as face_align

try:
    import config as C
except ImportError:
    from backend import config as C


class ArcFaceExtractor:
    """Trích xuất vector nhúng đặc trưng 512 chiều từ ảnh khuôn mặt."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # 1. Ưu tiên trong backend/models/w600k_r50.onnx
            local_path = os.path.join(os.path.dirname(__file__), "models", "w600k_r50.onnx")
            # 2. Dự phòng trong cache insightface của hệ thống
            system_cache = os.path.expanduser(r"~/.insightface/models/buffalo_l/w600k_r50.onnx")

            if os.path.exists(local_path):
                model_path = local_path
            elif os.path.exists(system_cache):
                model_path = system_cache
            else:
                model_path = local_path

        self.model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._init_session()

    def _init_session(self):
        if not os.path.exists(self.model_path):
            print(f"[ArcFace] Cảnh báo: Chưa tìm thấy model tại {self.model_path}")
            return

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        print(f"[ArcFace] Model loaded successfully from {self.model_path}")

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def align_face(self, image_bgr: np.ndarray, landmarks_5pts: np.ndarray, image_size: int = 112) -> np.ndarray:
        """Căn chỉnh khuôn mặt theo 5 mốc chuẩn ArcFace (2 mắt, chóp mũi, 2 khóe miệng)."""
        kps = np.array(landmarks_5pts, dtype=np.float32)
        if kps.shape != (5, 2):
            raise ValueError(f"Landmarks phải có shape (5, 2), hiện tại là {kps.shape}")

        # Chuẩn hoá thứ tự: đảm bảo mắt trái và khóe miệng trái có x nhỏ hơn bên phải
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        warped = face_align.norm_crop(image_bgr, kps, image_size=image_size)
        return warped

    def extract_from_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Trích xuất vector 512D từ ảnh khuôn mặt 112x112 đã căn chỉnh."""
        if not self.is_ready:
            raise RuntimeError("ArcFace model chưa được khởi tạo")

        if aligned_bgr.shape[:2] != (112, 112):
            aligned_bgr = cv2.resize(aligned_bgr, (112, 112), interpolation=cv2.INTER_AREA)

        # Preprocessing: BGR -> RGB, (img - 127.5) / 127.5, shape (1, 3, 112, 112)
        blob = cv2.dnn.blobFromImages(
            [aligned_bgr],
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )

        raw_feat = self._session.run([self._output_name], {self._input_name: blob})[0][0]

        # Chuẩn hóa L2 về mặt cầu đơn vị (Unit Hypersphere)
        norm = np.linalg.norm(raw_feat)
        if norm > 1e-6:
            emb = raw_feat / norm
        else:
            emb = raw_feat

        return emb.astype(np.float32)

    def extract_embedding(
        self,
        image_bgr: np.ndarray,
        landmarks_5pts: Optional[np.ndarray] = None,
        bbox: Optional[list] = None,
    ) -> np.ndarray:
        """Trích xuất embedding từ ảnh chụp gốc (có hỗ trợ tự động căn chỉnh hoặc crop)."""
        h, w = image_bgr.shape[:2]

        if landmarks_5pts is not None and len(landmarks_5pts) == 5:
            aligned = self.align_face(image_bgr, landmarks_5pts, 112)
        elif bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Thêm lề an toàn 15%
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = image_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                crop = image_bgr
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        else:
            # Fallback: Cắt vùng trung tâm
            crop_size = min(h, w)
            sy = (h - crop_size) // 2
            sx = (w - crop_size) // 2
            crop = image_bgr[sy : sy + crop_size, sx : sx + crop_size]
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

        return self.extract_from_aligned(aligned)

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Tính Cosine Similarity giữa 2 vector đã chuẩn hoá L2. Kết quả trong [-1.0, 1.0]."""
        dot = float(np.dot(emb1.flatten(), emb2.flatten()))
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def compute_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Khoảng cách Euclid giữa 2 vector đặc trưng."""
        return float(np.linalg.norm(emb1.flatten() - emb2.flatten()))

    def verify(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.45,
    ) -> dict:
        """So khớp 1:1 giữa 2 vector đặc trưng."""
        sim = self.compute_similarity(emb1, emb2)
        dist = self.compute_distance(emb1, emb2)
        is_match = sim >= threshold

        # Tính % độ tự tin (Confidence score) chuẩn hóa
        if is_match:
            # Từ threshold -> 1.0 map về 60% -> 100%
            conf = 60.0 + ((sim - threshold) / max(0.01, (1.0 - threshold))) * 40.0
        else:
            # Dưới threshold map về 0% -> 59.9%
            conf = max(0.0, (sim / max(0.01, threshold))) * 59.9

        return {
            "is_match": bool(is_match),
            "similarity": round(sim, 4),
            "distance": round(dist, 4),
            "threshold": threshold,
            "confidence_pct": round(min(100.0, max(0.0, conf)), 1),
        }


# Singleton instance
_extractor_instance: Optional[ArcFaceExtractor] = None


def get_arcface_extractor() -> ArcFaceExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ArcFaceExtractor()
    return _extractor_instance
