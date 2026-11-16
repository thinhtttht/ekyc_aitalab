"""Core Face Processing Engine for eKYC REST API.
Handles MediaPipe FaceMesh & Hands, MiniFASNet PAD, ArcFace 512D, and FQA Policies.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from typing import Optional, Dict, Any, Tuple, List

# Standard 5 landmarks on 112x112 aligned face
ARCFACE_SRC_STD = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Face oval canonical indices
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


class FaceEngine:
    """Singleton Engine that holds ONNX sessions and MediaPipe instances."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

        # 1. MediaPipe FaceMesh (478 landmarks with iris)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # 2. MediaPipe Hands (Hand occlusion detection)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )

        # 3. ArcFace ResNet50 ONNX Session
        arcface_path = os.path.join(models_dir, "w600k_r50.onnx")
        if not os.path.exists(arcface_path):
            raise FileNotFoundError(f"Model ArcFace khong ton tai tai: {arcface_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.arcface_sess = ort.InferenceSession(
            arcface_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.arcface_in = self.arcface_sess.get_inputs()[0].name
        self.arcface_out = self.arcface_sess.get_outputs()[0].name

        # 4. MiniFASNetV2 Anti-Spoofing ONNX Session
        minifas_path = os.path.join(models_dir, "minifasnet_v2.onnx")
        if not os.path.exists(minifas_path):
            raise FileNotFoundError(f"Model MiniFASNet khong ton tai tai: {minifas_path}")

        self.minifas_sess = ort.InferenceSession(
            minifas_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.minifas_in = self.minifas_sess.get_inputs()[0].name
        self.minifas_out = self.minifas_sess.get_outputs()[0].name

        print("[FaceEngine] All models initialized successfully on CPU!")

    def detect_face_and_hands(self, frame_bgr: np.ndarray) -> Tuple[List[Any], List[Any]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mesh_res = self.face_mesh.process(rgb)
        hands_res = self.hands_detector.process(rgb)
        faces = mesh_res.multi_face_landmarks or []
        hands = hands_res.multi_hand_landmarks or []
        return faces, hands

    def evaluate_fqa_and_policy(
        self,
        frame_bgr: np.ndarray,
        faces: list,
        hands: list,
        enforce_oval: bool = False,
    ) -> Dict[str, Any]:
        """Kiểm tra toàn bộ quy chuẩn chất lượng khuôn mặt (Policy & FQA)."""
        h, w = frame_bgr.shape[:2]

        if len(faces) == 0:
            return {
                "is_valid": False,
                "error_code": "NO_FACE_DETECTED",
                "message": "Không tìm thấy khuôn mặt nào trong ảnh.",
                "policy_checks": {
                    "face_detected": False,
                    "single_face": False,
                    "no_mask": False,
                    "no_sunglasses": False,
                    "no_hand_occlusion": True,
                    "inside_oval": False,
                    "illumination_ok": False,
                    "sharpness_ok": False,
                },
                "warnings": [],
            }

        if len(faces) > 1:
            return {
                "is_valid": False,
                "error_code": "MULTIPLE_FACES",
                "message": "Phát hiện nhiều hơn một khuôn mặt trong khung hình.",
                "policy_checks": {
                    "face_detected": True,
                    "single_face": False,
                    "no_mask": True,
                    "no_sunglasses": True,
                    "no_hand_occlusion": True,
                    "inside_oval": False,
                    "illumination_ok": True,
                    "sharpness_ok": True,
                },
                "warnings": [],
            }

        face = faces[0]
        lm = np.array([[p.x, p.y, p.z] for p in face.landmark], dtype=np.float64)
        px = np.column_stack([lm[:, 0] * w, lm[:, 1] * h])

        x_min, y_min = np.min(px, axis=0)
        x_max, y_max = np.max(px, axis=0)
        bbox = (int(max(0, x_min)), int(max(0, y_min)), int(min(w, x_max)), int(min(h, y_max)))

        # 1. Độ sáng & Độ nét
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        jaw_pts = px[FACE_OVAL_INDICES].astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(jaw_pts), 255)
        mean_brightness = float(gray[mask > 0].mean()) if mask.any() else float(gray.mean())

        bx0, by0, bx1, by1 = bbox
        face_roi = gray[by0:by1, bx0:bx1]
        sharpness = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) if face_roi.size > 0 else 0.0

        illumination_ok = 35.0 <= mean_brightness <= 225.0
        sharpness_ok = sharpness >= 10.0

        # 2. Khung Oval chuẩn (cx=0.5, cy=0.5, rx=0.25, ry=0.35)
        cx, cy, rx, ry = 0.5, 0.5, 0.25, 0.35
        norm_cx = (x_min + x_max) / (2.0 * w)
        norm_cy = (y_min + y_max) / (2.0 * h)
        oval_dist = ((norm_cx - cx) / rx) ** 2 + ((norm_cy - cy) / ry) ** 2
        inside_oval = bool(oval_dist <= 1.0)

        # 3. Phát hiện bàn tay che mặt
        hand_occlusion = False
        if hands:
            for hand in hands:
                for pt in hand.landmark:
                    hx, hy = pt.x * w, pt.y * h
                    if bx0 <= hx <= bx1 and by0 <= hy <= by1:
                        hand_occlusion = True
                        break
                if hand_occlusion:
                    break

        # 4. Phát hiện khẩu trang (Mask)
        # So sánh sắc độ da trán (Landmark 10, 67, 297) với vùng cằm/miệng (152, 0, 17)
        ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        forehead_pt = px[10].astype(int)
        chin_pt = px[152].astype(int)
        
        def sample_ycc(pt):
            y_val, x_val = np.clip(pt[1], 0, h - 1), np.clip(pt[0], 0, w - 1)
            patch = ycc[max(0, y_val - 5):min(h, y_val + 5), max(0, x_val - 5):min(w, x_val + 5)]
            return patch.mean(axis=(0, 1)) if patch.size > 0 else np.array([0, 0, 0])

        forehead_color = sample_ycc(forehead_pt)
        chin_color = sample_ycc(chin_pt)
        color_diff = np.linalg.norm(forehead_color[1:] - chin_color[1:]) # Cr, Cb diff
        mask_detected = bool(color_diff > 35.0 or chin_color[0] < 25.0)

        # 5. Phát hiện kính râm đen (Sunglasses)
        # Lấy vùng 2 mắt
        eye_pts = px[[33, 160, 158, 133, 362, 385, 387, 263]].astype(int)
        ex0, ey0 = np.min(eye_pts, axis=0)
        ex1, ey1 = np.max(eye_pts, axis=0)
        eye_roi = gray[max(0, ey0 - 5):min(h, ey1 + 5), max(0, ex0 - 5):min(w, ex1 + 5)]
        sunglasses_detected = False
        if eye_roi.size > 0:
            eye_mean = float(eye_roi.mean())
            eye_std = float(eye_roi.std())
            if eye_mean < 25.0 and eye_std < 9.0 and (eye_mean / max(1.0, mean_brightness)) < 0.25:
                sunglasses_detected = True

        # Quyết định tổng thể
        policy_checks = {
            "face_detected": True,
            "single_face": True,
            "no_mask": not mask_detected,
            "no_sunglasses": not sunglasses_detected,
            "no_hand_occlusion": not hand_occlusion,
            "inside_oval": inside_oval,
            "illumination_ok": illumination_ok,
            "sharpness_ok": sharpness_ok,
        }

        warnings = []
        is_valid = True
        error_msg = ""
        error_code = ""

        if mask_detected:
            is_valid = False
            error_code = "MASK_DETECTED"
            error_msg = "Vui lòng tháo khẩu trang để tiếp tục."
        elif sunglasses_detected:
            is_valid = False
            error_code = "SUNGLASSES_DETECTED"
            error_msg = "Vui lòng tháo kính râm/kính đen để tiếp tục."
        elif hand_occlusion:
            is_valid = False
            error_code = "HAND_OCCLUSION"
            error_msg = "Vui lòng bỏ tay ra khỏi khuôn mặt."
        elif not illumination_ok:
            is_valid = False
            error_code = "BAD_ILLUMINATION"
            error_msg = "Ánh sáng không đạt yêu cầu (quá tối hoặc quá chói)."
        elif not sharpness_ok:
            is_valid = False
            error_code = "BLURRY_IMAGE"
            error_msg = "Ảnh bị mờ nhòe, vui lòng giữ yên camera."
        elif not inside_oval:
            if enforce_oval:
                is_valid = False
                error_code = "OUT_OF_OVAL"
                error_msg = "Khuôn mặt nằm ngoài khung Oval chuẩn."
            else:
                warnings.append("Khuôn mặt nằm ngoài vùng Oval chuẩn. Độ chính xác nhận diện có thể bị giảm sút.")

        return {
            "is_valid": is_valid,
            "error_code": error_code,
            "message": error_msg or "Khuôn mặt đạt chuẩn chất lượng.",
            "policy_checks": policy_checks,
            "warnings": warnings,
            "bbox": bbox,
            "landmarks": lm,
            "metrics": {
                "brightness": round(mean_brightness, 1),
                "sharpness": round(sharpness, 1),
                "oval_distance": round(float(oval_dist), 2),
            },
        }

    def check_liveness(self, frame_bgr: np.ndarray, bbox: tuple) -> Dict[str, Any]:
        """Kiểm tra chống giả mạo ảnh in/màn hình bằng MiniFASNetV2."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bw, bh = (x1 - x0) * 1.5, (y1 - y0) * 1.5

        nx0 = max(0, int(cx - bw / 2.0))
        ny0 = max(0, int(cy - bh / 2.0))
        nx1 = min(w, int(cx + bw / 2.0))
        ny1 = min(h, int(cy + bh / 2.0))

        crop = frame_bgr[ny0:ny1, nx0:nx1]
        if crop.size == 0:
            return {"is_real": False, "real_prob": 0.0, "message": "Không tìm thấy khuôn mặt."}

        resized = cv2.resize(crop, (80, 80))
        blob = np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32), axis=0)

        out = self.minifas_sess.run([self.minifas_out], {self.minifas_in: blob})[0][0]
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / np.sum(exp_out)
        real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])

        is_real = real_prob >= 0.70
        return {
            "is_real": is_real,
            "real_prob": round(real_prob, 4),
            "message": "Xác nhận người thật." if is_real else "Nghi vấn giả mạo (Ảnh in hoặc màn hình phát lại).",
        }

    def extract_embedding(self, frame_bgr: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
        """Căn chỉnh 5 mốc chuẩn và trích xuất vector 512D ArcFace chuẩn hóa L2."""
        h, w = frame_bgr.shape[:2]
        kps_idx = [33, 263, 1, 61, 291]
        kps = np.array([[landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h] for i in kps_idx], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_SRC_STD, method=cv2.LMEDS)
        if M is not None:
            aligned = cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)
        else:
            aligned = cv2.resize(frame_bgr, (112, 112))

        blob = cv2.dnn.blobFromImages(
            [aligned], scalefactor=1.0 / 127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True
        )

        feat = self.arcface_sess.run([self.arcface_out], {self.arcface_in: blob})[0][0]
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            emb = feat / norm
        else:
            emb = feat
        return emb.astype(np.float32)

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity giữa 2 vector chuẩn hóa L2."""
        dot = float(np.dot(emb1.flatten(), emb2.flatten()))
        return max(-1.0, min(1.0, dot))
