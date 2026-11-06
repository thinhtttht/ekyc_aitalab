"""QUICK START DEMO - EKYC & AI PROCTORING INTEGRATION KIT
Script minh họa trọn vẹn luồng AI eKYC & Giám Thị Phòng Thi:
1. Vẽ khung Oval hướng dẫn sinh trắc học
2. Tính toán góc xoay đầu 3D (Head Pose: Yaw, Pitch, Roll) & vẽ trục tọa độ 3D
3. Theo dõi hướng nhìn của mắt (Eye Tracking & Gaze direction)
4. Đánh giá chất lượng khuôn mặt (FQA: Ánh sáng, Độ nét, Cự ly Oval)
5. Chống giả mạo ảnh in/màn hình (MiniFASNet Passive Anti-Spoofing)
6. Bấm [SPACE] để Lưu mẫu thí sinh -> So khớp ArcFace 512D thời gian thực!

Thao tác:
- Bấm [SPACE]: Đăng ký khuôn mặt hiện tại làm thí sinh mẫu
- Bấm [R]: Đặt lại (Reset) để đăng ký lại
- Bấm [Q]: Thoát chương trình
"""
import os
import sys
import time

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# Tự động tìm thư mục models
POSSIBLE_MODEL_DIRS = [
    os.path.join(os.path.dirname(__file__), "models"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend", "models")),
]
MODELS_DIR = next((d for d in POSSIBLE_MODEL_DIRS if os.path.exists(d)), "models")

try:
    import mediapipe as mp
    import onnxruntime as ort
except ImportError:
    print("❌ Lỗi: Thiếu thư viện mediapipe hoặc onnxruntime.")
    print("👉 Hãy chạy lệnh: pip install opencv-python mediapipe==0.10.14 onnxruntime numpy")
    sys.exit(1)


# ==========================================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI
# ==========================================================
print("🔄 Đang khởi tạo MediaPipe FaceMesh & Iris...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

arcface_path = os.path.join(MODELS_DIR, "w600k_r50.onnx")
minifas_path = os.path.join(MODELS_DIR, "minifasnet_v2.onnx")

arcface_session = None
minifas_session = None

if os.path.exists(arcface_path):
    print(f"🔄 Đang tải ArcFace ResNet50 từ {arcface_path}...")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    arcface_session = ort.InferenceSession(arcface_path, sess_options=opts, providers=["CPUExecutionProvider"])
    arcface_in = arcface_session.get_inputs()[0].name
    arcface_out = arcface_session.get_outputs()[0].name
    print("✅ Đã nạp ArcFace 512D!")
else:
    print(f"⚠️ Cảnh báo: Chưa tìm thấy {arcface_path}. Tính năng so khớp ArcFace sẽ tạm tắt.")

if os.path.exists(minifas_path):
    print(f"🔄 Đang tải MiniFASNetV2 Anti-Spoofing từ {minifas_path}...")
    minifas_session = ort.InferenceSession(minifas_path, providers=["CPUExecutionProvider"])
    minifas_in = minifas_session.get_inputs()[0].name
    minifas_out = minifas_session.get_outputs()[0].name
    print("✅ Đã nạp MiniFASNet!")
else:
    print(f"⚠️ Cảnh báo: Chưa tìm thấy {minifas_path}.")


# ==========================================================
# 2. HÀM TÍNH TOÁN HEAD POSE (SOLVE PNP)
# ==========================================================
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Mũi (1)
    [0.0, -330.0, -65.0],     # Cằm (199)
    [-225.0, 170.0, -135.0],  # Khóe mắt trái (33)
    [225.0, 170.0, -135.0],   # Khóe mắt phải (263)
    [-150.0, -150.0, -125.0], # Khóe miệng trái (61)
    [150.0, -150.0, -125.0],  # Khóe miệng phải (291)
], dtype=np.float64)

KEYPOINTS_6 = [1, 199, 33, 263, 61, 291]

def get_head_pose(landmarks_norm, img_w, img_h):
    image_points = np.array([
        [landmarks_norm[i, 0] * img_w, landmarks_norm[i, 1] * img_h]
        for i in KEYPOINTS_6
    ], dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2.0],
        [0, focal_length, img_h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0, rot_vec, trans_vec, cam_matrix

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_matrix = np.hstack((rot_mat, trans_vec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = float(euler[0, 0]), float(euler[1, 0]), float(euler[2, 0])
    return yaw, pitch, roll, rot_vec, trans_vec, cam_matrix


# ==========================================================
# 3. HÀM TRÍCH XUẤT ARCFACE 512D
# ==========================================================
def extract_embedding(frame_bgr, landmarks_norm):
    if arcface_session is None:
        return None
    h, w = frame_bgr.shape[:2]
    kps_idx = [33, 263, 1, 61, 291]
    kps = np.array([[landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h] for i in kps_idx], dtype=np.float32)

    src_std = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014],
        [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(kps, src_std, method=cv2.LMEDS)
    if M is not None:
        aligned = cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)
    else:
        aligned = cv2.resize(frame_bgr, (112, 112))

    blob = cv2.dnn.blobFromImages(
        [aligned], scalefactor=1.0 / 127.5, size=(112, 112),
        mean=(127.5, 127.5, 127.5), swapRB=True
    )
    feat = arcface_session.run([arcface_out], {arcface_in: blob})[0][0]
    norm = np.linalg.norm(feat)
    return (feat / norm).astype(np.float32) if norm > 1e-6 else feat.astype(np.float32)


# ==========================================================
# 4. HÀM CHỐNG GIẢ MẠO MINIFASNET & NÂNG SÁNG TỰ ĐỘNG
# ==========================================================
def predict_liveness(frame_bgr, bbox):
    if minifas_session is None:
        return True, 0.99
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = (x1 - x0) * 1.5, (y1 - y0) * 1.5
    nx0, ny0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    nx1, ny1 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
    crop = frame_bgr[ny0:ny1, nx0:nx1]
    if crop.size == 0:
        return False, 0.0

    resized = cv2.resize(crop, (80, 80))
    blob = np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32), axis=0)
    out = minifas_session.run([minifas_out], {minifas_in: blob})[0][0]
    exp_out = np.exp(out - np.max(out))
    probs = exp_out / np.sum(exp_out)
    real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
    return real_prob >= 0.70, real_prob


def auto_enhance_brightness(frame_bgr, target_mean=115.0):
    """Tự động nâng sáng thông minh cho phòng tối bằng LAB + CLAHE + Adaptive Gamma."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    curr_mean = float(gray.mean())
    if curr_mean < 18.0 or curr_mean >= 85.0:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    clahe_mean = max(1.0, float(l_clahe.mean()))
    if clahe_mean < target_mean:
        gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(clahe_mean / 255.0), 0.35, 1.0))
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        l_out = cv2.LUT(l_clahe, lut)
    else:
        l_out = l_clahe
    return cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2BGR)


# ==========================================================
# 5. VÒNG LẶP CHÍNH (WEBCAM INTERFACE)
# ==========================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam. Vui lòng kiểm tra kết nối camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("🚀 EKYC & AI PROCTORING DEMO ĐANG CHẠY")
    print("👉 Nhìn thẳng vào khung Oval giữa màn hình")
    print("👉 Bấm [SPACE] để đăng ký khuôn mặt thí sinh mẫu")
    print("👉 Bấm [E] để Bật/Tắt Tự Động Nâng Sáng (Auto-Enhance)")
    print("👉 Bấm [R] để xóa mẫu đăng ký")
    print("👉 Bấm [Q] để thoát")
    print("=" * 60 + "\n")

    registered_embedding = None
    last_pad_time = 0.0
    cached_pad_real = True
    cached_pad_prob = 0.99
    auto_enhance = False
    prev_gray = None
    frozen_count = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(raw_frame, 1) # Lật gương cho tự nhiên
        h, w = frame.shape[:2]

        # 0. KIỂM TRA PHẦN CỨNG CAMERA & ÁNH SÁNG
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        # Kiểm tra đóng băng (Frozen camera)
        g_float = gray.astype(np.float32)
        if prev_gray is not None and prev_gray.shape == g_float.shape:
            diff = float(np.abs(g_float - prev_gray).mean())
            frozen_count = frozen_count + 1 if diff < 0.25 else 0
        prev_gray = g_float

        is_frozen = frozen_count >= 15
        is_black = brightness < 18.0

        # Áp dụng tự động nâng sáng nếu bật
        if auto_enhance:
            frame = auto_enhance_brightness(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # Vẽ khung Oval chuẩn eKYC
        cx_px, cy_px = int(w * 0.5), int(h * 0.5)
        rx_px, ry_px = int(w * 0.22), int(h * 0.32)
        oval_color = (100, 100, 100) # Mặc định xám
        status_text = "Dua khuon mat vao giua khung Oval"

        if is_black:
            status_text = "CANH BAO: Camera bi che hoac qua toi!"
            oval_color = (0, 0, 255)
        elif is_frozen:
            status_text = "CANH BAO: Camera bi dong bang / treo hinh!"
            oval_color = (0, 0, 255)
        else:
            faces = res.multi_face_landmarks or []
            face_count = len(faces)

            if face_count == 0:
                status_text = "Khong tim thay khuon mat!"
                oval_color = (0, 0, 255)
            elif face_count > 1:
                status_text = "CANH BAO: Co nhieu nguoi trong khung hinh!"
                oval_color = (0, 0, 255)
            else:
                lm = np.array([[p.x, p.y, p.z] for p in faces[0].landmark], dtype=np.float64)
                px = np.column_stack([lm[:, 0] * w, lm[:, 1] * h])
                x0, y0 = int(np.min(px[:, 0])), int(np.min(px[:, 1]))
                x1, y1 = int(np.max(px[:, 0])), int(np.max(px[:, 1]))

                # 1. Tính khoảng cách Oval
                mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                oval_dist = ((mid_x - cx_px) / rx_px) ** 2 + ((mid_y - cy_px) / ry_px) ** 2
                inside_oval = oval_dist <= 1.0

                # 2. Tính Head Pose
                yaw, pitch, roll, rot_vec, trans_vec, cam_mat = get_head_pose(lm, w, h)

                # 3. Tính Gaze (mắt)
                lx_in, lx_out, l_iris = lm[133, 0] * w, lm[33, 0] * w, lm[468, 0] * w
                rx_in, rx_out, r_iris = lm[362, 0] * w, lm[263, 0] * w, lm[473, 0] * w
                gaze_ratio = ((l_iris - lx_in) / max(1.0, lx_out - lx_in) + (r_iris - rx_in) / max(1.0, rx_out - rx_in)) / 2.0
                looking_away = gaze_ratio < 0.28 or gaze_ratio > 0.72

                # 4. Anti-Spoofing mỗi 150ms
                now = time.time()
                if now - last_pad_time >= 0.15:
                    last_pad_time = now
                    cached_pad_real, cached_pad_prob = predict_liveness(frame, (x0, y0, x1, y1))

                # Đánh giá trạng thái
                if brightness < 40.0:
                    oval_color = (0, 0, 255)
                    status_text = "Anh sang phong qua toi! Hay bat them den"
                elif brightness > 220.0:
                    oval_color = (0, 0, 255)
                    status_text = "Anh sang qua choi! Tranh den chieu vao cam"
                elif not cached_pad_real:
                    oval_color = (0, 0, 255)
                    status_text = f"CANH BAO GIA MAO: Anh in/Man hinh ({int(cached_pad_prob*100)}%)"
                elif not inside_oval:
                    oval_color = (0, 200, 255)
                    status_text = "Can chinh dau lot vao giua khung Oval"
                elif abs(yaw) > 20.0 or abs(pitch) > 18.0:
                    oval_color = (0, 165, 255)
                    turn_dir = "Trai" if yaw < -20 else ("Phai" if yaw > 20 else ("Cui" if pitch < -18 else "Ngua"))
                    status_text = f"Canh bao: Quay dau ({turn_dir} {abs(int(yaw))} deg)"
                elif looking_away:
                    oval_color = (0, 165, 255)
                    status_text = "Canh bao: Huong nhin mat lech ra ngoai!"
                else:
                    oval_color = (0, 255, 0)
                    status_text = "Chuan! Nhin thang va giu yen"

                # Vẽ trục tọa độ 3D Head Pose
                axis_len = 80.0
                axis_3d = np.array([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]], dtype=np.float64)
                proj_pts, _ = cv2.projectPoints(axis_3d, rot_vec, trans_vec, cam_mat, np.zeros((4, 1)))
                nose_pt = (int(px[1, 0]), int(px[1, 1]))
                cv2.line(frame, nose_pt, (int(proj_pts[0, 0, 0]), int(proj_pts[0, 0, 1])), (0, 0, 255), 2)  # X: Đỏ
                cv2.line(frame, nose_pt, (int(proj_pts[1, 0, 0]), int(proj_pts[1, 0, 1])), (0, 255, 0), 2)  # Y: Xanh lá
                cv2.line(frame, nose_pt, (int(proj_pts[2, 0, 0]), int(proj_pts[2, 0, 1])), (255, 0, 0), 2)  # Z: Xanh dương

                # Vẽ mắt đồng tử
                cv2.circle(frame, (int(l_iris), int(lm[468, 1] * h)), 3, (0, 255, 255), -1)
                cv2.circle(frame, (int(r_iris), int(lm[473, 1] * h)), 3, (0, 255, 255), -1)

                # So khớp ArcFace nếu đã đăng ký
                if registered_embedding is not None and arcface_session is not None:
                    curr_emb = extract_embedding(frame, lm)
                    if curr_emb is not None:
                        sim = float(np.dot(curr_emb, registered_embedding))
                        sim_pct = int(max(0.0, sim) * 100)
                        match_col = (0, 255, 0) if sim >= 0.55 else (0, 0, 255)
                        cv2.putText(frame, f"Match: {sim_pct}% ({'TRUNG KHOP' if sim>=0.55 else 'KHONG KHOP'})",
                                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_col, 2)

                # Thông số góc
                cv2.putText(frame, f"Yaw: {int(yaw)} | Pitch: {int(pitch)} | Roll: {int(roll)}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Hiển thị thước đo ánh sáng (Brightness) & Auto-enhance
        b_color = (0, 255, 0) if 45 <= brightness <= 215 else (0, 0, 255)
        cv2.putText(frame, f"Do sang: {int(brightness)}/255 | Auto-Light [E]: {'BAT' if auto_enhance else 'TAT'}",
                    (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1)

        # Vẽ elip Oval
        cv2.ellipse(frame, (cx_px, cy_px), (rx_px, ry_px), 0, 0, 360, oval_color, 2)

        # Banner trạng thái
        cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, oval_color, 2)

        # Hướng dẫn phím bấm phía dưới
        footer = "[SPACE]: Chup dang ky | [E]: Bat/Tat sang | [R]: Reset | [Q]: Thoat"
        if registered_embedding is not None:
            footer += " | DA LUU MAU!"
        cv2.putText(frame, footer, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("eKYC & AI Proctoring Integration Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if 'face_count' in locals() and face_count == 1:
                registered_embedding = extract_embedding(frame, lm)
                print(f"🎉 ĐÃ ĐĂNG KÝ THÀNH CÔNG KHUÔN MẶT MẪU! (512D Vector shape: {registered_embedding.shape})")
        elif key == ord('e'):
            auto_enhance = not auto_enhance
            print(f"💡 Tự động nâng sáng thích ứng: {'BẬT' if auto_enhance else 'TẮT'}")
        elif key == ord('r'):
            registered_embedding = None
            print("🔄 Đã reset mẫu đăng ký.")

        cv2.imshow("eKYC & AI Proctoring Integration Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if face_count == 1:
                registered_embedding = extract_embedding(frame, lm)
                print(f"🎉 ĐÃ ĐĂNG KÝ THÀNH CÔNG KHUÔN MẶT MẪU! (512D Vector shape: {registered_embedding.shape})")
        elif key == ord('r'):
            registered_embedding = None
            print("🔄 Đã reset mẫu đăng ký.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
