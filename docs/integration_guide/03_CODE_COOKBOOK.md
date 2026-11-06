# 💻 SỔ TAY LẬP TRÌNH THỰC CHIẾN (CODE COOKBOOK)
## 03_CODE_COOKBOOK.md

Tài liệu này cung cấp các đoạn mã nguồn Python hoàn chỉnh, được viết theo chuẩn module hóa sạch sẽ, có chú thích chi tiết để nhóm phát triển có thể **sao chép trực tiếp vào dự án backend** của mình.

---

## Mục Lục Các Đoạn Mã (Recipes)

1. [Khởi tạo các mô hình AI](#1-khởi-tạo-các-mô-hình-ai)
2. [Trích xuất 468 mốc tọa độ & Kiểm tra quy chuẩn chất lượng (FQA)](#2-kiểm-tra-quy-chuẩn-khuôn-mặt-fqa)
3. [Phát hiện giả mạo ảnh in / màn hình (MiniFASNet Liveness)](#3-phát-hiện-giả-mạo-ảnh-in--màn-hình-liveness)
4. [Trích xuất vector 512D ArcFace & So khớp nhận diện 1:1](#4-trích-xuất-đặc-trưng-arcface--so-khớp-11)
5. [Tính toán góc xoay đầu 3D (Head Pose: Yaw, Pitch, Roll)](#5-tính-toán-góc-xoay-đầu-head-pose)
6. [Theo dõi hướng nhìn của mắt (Eye Tracking & Gaze)](#6-theo-dõi-hướng-nhìn-của-mắt-eye-tracking)
7. [Vòng lặp giám thị phòng thi hoàn chỉnh (Full Proctoring Engine)](#7-vòng-lặp-giám-thị-phòng-thi-hoàn-chỉnh)

---

## 1. Khởi Tạo Các Mô Hình AI

```python
import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

class AIModelManager:
    """Quản lý nạp và vòng đời các mô hình AI."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        
        # 1. Khởi tạo MediaPipe FaceMesh (468 landmarks 3D)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,              # Cho phép bắt tối đa 2 mặt để phát hiện gian lận
            refine_landmarks=True,         # Bật chế độ 478 điểm gồm cả đồng tử mắt (Iris)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. Khởi tạo ArcFace ONNX Session
        arcface_path = os.path.join(models_dir, "w600k_r50.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.arcface_session = ort.InferenceSession(
            arcface_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.arcface_in_name = self.arcface_session.get_inputs()[0].name
        self.arcface_out_name = self.arcface_session.get_outputs()[0].name
        
        # 3. Khởi tạo MiniFASNet ONNX Session (Anti-Spoofing)
        minifas_path = os.path.join(models_dir, "minifasnet_v2.onnx")
        self.minifas_session = ort.InferenceSession(
            minifas_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.minifas_in_name = self.minifas_session.get_inputs()[0].name
        self.minifas_out_name = self.minifas_session.get_outputs()[0].name
        
        print("✅ Đã khởi tạo thành công toàn bộ các mô hình AI!")
```

---

## 2. Kiểm Tra Quy Chuẩn Khuôn Mặt (FQA)

Hàm kiểm tra xem khuôn mặt thí sinh có đủ điều kiện để đăng ký / xác thực hay không (không khẩu trang, không kính râm, đủ sáng, lọt khung Oval):

```python
def check_face_quality(frame_bgr: np.ndarray, landmarks_norm: np.ndarray) -> dict:
    """
    Kiểm tra chất lượng khuôn mặt theo quy chuẩn FQA.
    landmarks_norm: mảng shape (468, 3) với tọa độ chuẩn hóa [0.0, 1.0].
    """
    h, w = frame_bgr.shape[:2]
    px = np.column_stack([landmarks_norm[:, 0] * w, landmarks_norm[:, 1] * h])
    
    # 1. Bounding box khuôn mặt
    x_min, y_min = np.min(px, axis=0)
    x_max, y_max = np.max(px, axis=0)
    face_w = max(1.0, x_max - x_min)
    face_h = max(1.0, y_max - y_min)
    
    # 2. Kiểm tra độ sáng (Brightness mean)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    jaw_pts = px[[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]].astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(jaw_pts), 255)
    mean_brightness = float(gray[mask > 0].mean()) if mask.any() else 0.0
    
    # 3. Kiểm tra độ nét (Laplacian Sharpness)
    bx0, by0 = max(0, int(x_min)), max(0, int(y_min))
    bx1, by1 = min(w, int(x_max)), min(h, int(y_max))
    face_roi = gray[by0:by1, bx0:bx1]
    sharpness = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) if face_roi.size > 0 else 0.0
    
    # 4. Kiểm tra vị trí so với khung Oval chuẩn ở giữa màn hình
    # Khung oval chuẩn: cx=0.5, cy=0.5, rx=0.22, ry=0.32
    cx, cy, rx, ry = 0.5, 0.5, 0.22, 0.32
    norm_x = (x_min + x_max) / (2.0 * w)
    norm_y = (y_min + y_max) / (2.0 * h)
    oval_dist = ((norm_x - cx) / rx) ** 2 + ((norm_y - cy) / ry) ** 2
    inside_oval = bool(oval_dist <= 1.0)
    
    # 5. Đánh giá tổng hợp
    is_valid = True
    reasons = []
    
    if not inside_oval:
        is_valid = False
        reasons.append("Căn đầu vào giữa khung hình Oval")
    if mean_brightness < 40.0:
        is_valid = False
        reasons.append("Ánh sáng quá tối - Hãy bật đèn")
    elif mean_brightness > 215.0:
        is_valid = False
        reasons.append("Ánh sáng quá chói - Tránh đèn chiếu thẳng vào mặt")
    if sharpness < 15.0:
        is_valid = False
        reasons.append("Hình ảnh bị mờ nhòe - Giữ yên camera")
        
    return {
        "is_valid": is_valid,
        "reasons": reasons,
        "brightness": mean_brightness,
        "sharpness": sharpness,
        "inside_oval": inside_oval,
        "bbox": (int(bx0), int(by0), int(bx1), int(by1))
    }
```

---

## 3. Phát Hiện Giả Mạo Ảnh In / Màn Hình (Liveness)

```python
def check_anti_spoofing(model_manager: AIModelManager, frame_bgr: np.ndarray, bbox: tuple) -> dict:
    """
    Kiểm tra người thật bằng MiniFASNetV2.
    bbox: (x0, y0, x1, y1) tọa độ pixel khuôn mặt.
    """
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    
    # Cắt ROI mở rộng 1.5 lần theo chuẩn huấn luyện của MiniFASNet
    scale = 1.5
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    new_w, new_h = bw * scale, bh * scale
    
    nx0 = max(0, int(cx - new_w / 2.0))
    ny0 = max(0, int(cy - new_h / 2.0))
    nx1 = min(w, int(cx + new_w / 2.0))
    ny1 = min(h, int(cy + new_h / 2.0))
    
    crop = frame_bgr[ny0:ny1, nx0:nx1]
    if crop.size == 0:
        return {"is_real": False, "real_prob": 0.0, "message": "Không tìm thấy khuôn mặt"}
    
    # Preprocessing: resize về (80, 80), shape (1, 3, 80, 80), kiểu float32
    resized = cv2.resize(crop, (80, 80))
    blob = resized.transpose(2, 0, 1).astype(np.float32)
    blob = np.expand_dims(blob, axis=0)
    
    # Inference ONNX
    outputs = model_manager.minifas_session.run(
        [model_manager.minifas_out_name],
        {model_manager.minifas_in_name: blob}
    )[0]
    
    # Softmax probabilities
    exp_out = np.exp(outputs[0] - np.max(outputs[0]))
    probs = exp_out / np.sum(exp_out)
    
    # MiniFASNet class 1 đại diện cho người thật (Real face)
    real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
    is_real = real_prob >= 0.70
    
    return {
        "is_real": is_real,
        "real_prob": round(real_prob, 3),
        "message": "Người thật" if is_real else "Nghi vấn giả mạo (Ảnh in hoặc màn hình)"
    }
```

---

## 4. Trích Xuất Đặc Trưng ArcFace & So Khớp 1:1

```python
def extract_arcface_embedding(model_manager: AIModelManager, frame_bgr: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Căn chỉnh khuôn mặt theo 5 mốc chuẩn và trích xuất vector 512D chuẩn hóa L2.
    """
    h, w = frame_bgr.shape[:2]
    
    # 5 mốc chuẩn: [Mắt trái, Mắt phải, Chóp mũi, Khóe miệng trái, Khóe miệng phải]
    kps_idx = [33, 263, 1, 61, 291]
    kps = np.array([[landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h] for i in kps_idx], dtype=np.float32)
    
    # 5 mốc mục tiêu trên ảnh 112x112 theo chuẩn InsightFace/ArcFace
    src_std = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    # Ma trận biến đổi Affine Similarity Transform
    M, _ = cv2.estimateAffinePartial2D(kps, src_std, method=cv2.LMEDS)
    if M is None:
        aligned = cv2.resize(frame_bgr, (112, 112))
    else:
        aligned = cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)
    
    # Preprocessing: BGR -> RGB, chuẩn hóa (img - 127.5) / 127.5
    blob = cv2.dnn.blobFromImages(
        [aligned], scalefactor=1.0 / 127.5, size=(112, 112),
        mean=(127.5, 127.5, 127.5), swapRB=True
    )
    
    # Inference ONNX
    feat = model_manager.arcface_session.run(
        [model_manager.arcface_out_name],
        {model_manager.arcface_in_name: blob}
    )[0][0]
    
    # Chuẩn hóa L2 Norm về mặt cầu đơn vị
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        embedding = feat / norm
    else:
        embedding = feat
        
    return embedding.astype(np.float32)


def compute_match_score(emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.55) -> dict:
    """
    Tính Cosine Similarity giữa 2 vector đặc trưng 512D.
    """
    sim = float(np.dot(emb1.flatten(), emb2.flatten()))
    is_match = sim >= threshold
    pct = int(max(0.0, min(1.0, sim)) * 100)
    
    return {
        "is_match": is_match,
        "similarity": round(sim, 4),
        "percentage": pct,
        "message": f"Khớp ({pct}%)" if is_match else f"Không khớp ({pct}% < {int(threshold*100)}%)"
    }
```

---

## 5. Tính Toán Góc Xoay Đầu (Head Pose)

Sử dụng thuật toán phối cảnh **Perspective-n-Point (`solvePnP`)** để lấy chính xác 3 góc: **Yaw (quay trái/phải)**, **Pitch (cúi/ngửa)**, và **Roll (nghiêng)**:

```python
# Tọa độ 3D thế giới thực chuẩn của 6 mốc khuôn mặt người (đơn vị mm)
STANDARD_FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],          # Chóp mũi (Landmark 1)
    [0.0, -330.0, -65.0],     # Cằm (Landmark 199)
    [-225.0, 170.0, -135.0],  # Khóe mắt trái (Landmark 33)
    [225.0, 170.0, -135.0],   # Khóe mắt phải (Landmark 263)
    [-150.0, -150.0, -125.0], # Khóe miệng trái (Landmark 61)
    [150.0, -150.0, -125.0]   # Khóe miệng phải (Landmark 291)
], dtype=np.float64)

KEYPOINT_INDICES = [1, 199, 33, 263, 61, 291]

def estimate_head_pose(landmarks_norm: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Ước lượng góc xoay đầu (Yaw, Pitch, Roll) theo đơn vị độ (Degrees).
    """
    # Lấy tọa độ 2D tương ứng trên ảnh
    image_points = np.array([
        [landmarks_norm[i, 0] * img_w, landmarks_norm[i, 1] * img_h]
        for i in KEYPOINT_INDICES
    ], dtype=np.float64)
    
    # Giả định ma trận tham số camera nội (Camera Matrix)
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1)) # Giả định không méo thấu kính
    
    # Giải bài toán PnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        STANDARD_FACE_MODEL_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "status": "UNKNOWN"}
    
    # Chuyển đổi Rotation Vector sang Ma trận xoay Euler
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_matrix = np.hstack((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])
    
    # Phân loại trạng thái
    status = "STRAIGHT" # Nhìn thẳng
    if yaw < -20.0:
        status = "TURN_LEFT"   # Quay sang trái
    elif yaw > 20.0:
        status = "TURN_RIGHT"  # Quay sang phải
    elif pitch < -18.0:
        status = "HEAD_DOWN"   # Cúi gầm mặt xuống
    elif pitch > 18.0:
        status = "HEAD_UP"     # Ngửa mặt lên trần
        
    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "roll": round(roll, 1),
        "status": status,
        "is_looking_away": bool(abs(yaw) > 20.0 or abs(pitch) > 18.0)
    }
```

---

## 6. Theo Dõi Hướng Nhìn Của Mắt (Eye Tracking)

Dựa trên tọa độ đồng tử mắt (Iris landmarks 468 - 477 do MediaPipe cung cấp khi bật `refine_landmarks=True`):

```python
def check_eye_gaze(landmarks_norm: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Theo dõi tỷ lệ đồng tử trong hốc mắt để phát hiện liếc mắt ra ngoài màn hình.
    """
    # Mắt trái: Khóe trong 133, Khóe ngoài 33, Tâm đồng tử 468
    lx_inner = landmarks_norm[133, 0] * img_w
    lx_outer = landmarks_norm[33, 0] * img_w
    l_iris_x = landmarks_norm[468, 0] * img_w
    
    # Mắt phải: Khóe trong 362, Khóe ngoài 263, Tâm đồng tử 473
    rx_inner = landmarks_norm[362, 0] * img_w
    rx_outer = landmarks_norm[263, 0] * img_w
    r_iris_x = landmarks_norm[473, 0] * img_w
    
    # Tỷ lệ vị trí đồng tử từ 0.0 (sát góc trong) đến 1.0 (sát góc ngoài)
    l_ratio = (l_iris_x - lx_inner) / max(1.0, lx_outer - lx_inner)
    r_ratio = (r_iris_x - rx_inner) / max(1.0, rx_outer - rx_inner)
    avg_ratio = (l_ratio + r_ratio) / 2.0
    
    # Bình thường khi nhìn màn hình: ratio dao động quanh 0.35 - 0.65
    is_looking_left = avg_ratio > 0.70
    is_looking_right = avg_ratio < 0.30
    
    return {
        "gaze_ratio": round(avg_ratio, 2),
        "is_looking_off_screen": bool(is_looking_left or is_looking_right),
        "direction": "LEFT" if is_looking_left else ("RIGHT" if is_looking_right else "CENTER")
    }
```

---

## 7. Vòng Lặp Giám Thị Phòng Thi Hoàn Chỉnh

Đoạn code tích hợp đầy đủ các module trên để chạy nền trong suốt buổi thi vấn đáp:

```python
import time

class ExamProctorEngine:
    def __init__(self, model_manager: AIModelManager, registered_embedding: np.ndarray):
        self.models = model_manager
        self.target_emb = registered_embedding
        self.violation_counter = 0
        self.last_identity_check_time = 0.0
        
    def process_exam_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Gọi hàm này định kỳ mỗi 100ms (10 FPS) với frame video từ webcam thí sinh.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.models.face_mesh.process(rgb)
        
        # 1. Phát hiện vắng mặt hoặc có nhiều người
        faces = res.multi_face_landmarks or []
        if len(faces) == 0:
            return {"alert_level": "RED", "code": "NO_FACE", "message": "🚨 Cảnh báo: Thí sinh không có mặt trước camera!"}
        if len(faces) > 1:
            return {"alert_level": "RED", "code": "MULTIPLE_FACES", "message": "🚨 Cảnh báo: Phát hiện nhiều người trong phòng thi!"}
            
        landmarks = np.array([[p.x, p.y, p.z] for p in faces[0].landmark], dtype=np.float64)
        
        # 2. Kiểm tra góc xoay đầu (Head Pose)
        pose = estimate_head_pose(landmarks, w, h)
        if pose["is_looking_away"]:
            return {
                "alert_level": "YELLOW",
                "code": "HEAD_TURN",
                "message": f"⚠️ Vui lòng nhìn thẳng vào màn hình! (Góc quay: {pose['status']})"
            }
            
        # 3. Kiểm tra hướng nhìn của mắt (Eye Gaze)
        gaze = check_eye_gaze(landmarks, w, h)
        if gaze["is_looking_off_screen"]:
            return {
                "alert_level": "YELLOW",
                "code": "GAZE_AWAY",
                "message": f"⚠️ Phát hiện hướng nhìn bất thường ({gaze['direction']})!"
            }
            
        # 4. Kiểm tra tráo người thi định kỳ mỗi 4 giây (Identity Swap Check)
        now = time.time()
        if now - self.last_identity_check_time >= 4.0:
            self.last_identity_check_time = now
            curr_emb = extract_arcface_embedding(self.models, frame_bgr, landmarks)
            match = compute_match_score(curr_emb, self.target_emb, threshold=0.50)
            if not match["is_match"]:
                return {
                    "alert_level": "CRITICAL",
                    "code": "IDENTITY_SWAP",
                    "message": "🚨 BÁO ĐỘNG ĐỎ: Phát hiện tráo đổi người thi hộ!"
                }
                
        return {"alert_level": "GREEN", "code": "NORMAL", "message": "Trạng thái thi hợp lệ"}
```
