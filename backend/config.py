"""Ngưỡng cấu hình tập trung cho luồng Đăng ký khuôn mặt (Enrollment).

Mọi toạ độ Oval được chuẩn hoá theo khung phân tích tỉ lệ 4:5 (mặc định 480x600):
x, rx tính theo chiều rộng; y, ry tính theo chiều cao.
"""
import os

# Kích thước khung phân tích mà frontend gửi lên (cắt giữa theo đúng phần đang hiển thị)
ANALYSIS_W, ANALYSIS_H = 480, 600

# ---------------------------------------------------------------------------
# (a) KIỂM TRA CHẤT LƯỢNG CAMERA
# ---------------------------------------------------------------------------
MIN_WIDTH, MIN_HEIGHT = 640, 480       # độ phân giải gốc tối thiểu của webcam
MIN_FPS = 25.0                         # FPS thực tế đo trên trình duyệt
BLACK_FRAME_MEAN = 18.0                # độ sáng trung bình < 18 -> camera bị che / khung đen
FROZEN_DIFF = 0.15                     # chênh lệch trung bình giữa 2 frame liên tiếp coi như "giống hệt"
FROZEN_FRAMES = 12                     # số lần liên tiếp giống hệt -> camera bị treo
FRAME_BRIGHTNESS = (45.0, 215.0)       # độ sáng tổng thể khung hình
MAX_NOISE_SIGMA = 10.0                 # độ lệch chuẩn nhiễu (Immerkær) tối đa
CAMERA_STABLE_SEC = 1.5                # camera phải đạt chuẩn liên tục trong 1.5s

# ---------------------------------------------------------------------------
# (b) KIỂM TRA CHẤT LƯỢNG KHUÔN MẶT (FQA - FACE QUALITY ASSESSMENT)
# ---------------------------------------------------------------------------
OVAL_NORMAL = dict(cx=0.50, cy=0.47, rx=0.33, ry=0.36)
OVAL_ZOOM = dict(cx=0.50, cy=0.50, rx=0.43, ry=0.46)
OVAL_INSIDE_TOL = 1.0                  # max ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1.0 cho 4 góc và viền

# 1. Framing & Scale Check (Tỷ lệ vàng Width_face / Width_oval)
FACE_SCALE_RANGE = (0.40, 0.85)        # 0.40 <= scale <= 0.85; < 0.40 -> Vàng (Lại gần); > 0.85 -> Vàng (Ra xa)
FACE_FILL = (0.50, 0.92)               # Chiều cao mặt / chiều cao oval
MAX_CENTER_OFFSET = 0.28               # Lệch tâm tối đa theo bán trục oval
MAX_STRAIGHT = dict(yaw=12.0, pitch=15.0, roll=10.0)

# 2. Illumination Check (Độ sáng Histogram Y/Grayscale)
FACE_BRIGHTNESS_MIN = 40.0             # Mean < 40 -> Báo Đỏ ("Không gian quá tối")
FACE_BRIGHTNESS_MAX = 210.0            # Mean > 210 -> Báo Đỏ ("Ánh sáng quá mạnh")
FACE_BRIGHTNESS_STD_MIN = 10.0         # Std Dev < 10 kèm Mean trung bình -> Báo Vàng ("Ngược sáng / thiếu chi tiết")
MAX_SIDE_LIGHT_RATIO = 1.6             # Tỉ lệ sáng nửa mặt sáng / nửa mặt tối

# 3. Sharpness Check (Laplacian Variance trên ROI mặt chuẩn hoá 200px)
MIN_FACE_SHARPNESS = 25.0              # laplacian_var < 25 -> Báo Vàng ("Giữ yên camera / Giữ chắc máy")
MIN_FACE_SHARPNESS_TURNING = 18.0      # Dung sai độ nét khi quay đầu

# 4. Occlusion Check (Kính râm đen, Kính lóa phản quang, Khẩu trang, Bàn tay)
SUNGLASSES_EYE_MEAN_MAX = 30.0         # Eye ROI Mean < 30 VÀ Std < 10 -> Kính râm đen -> Báo Đỏ
SUNGLASSES_EYE_STD_MAX = 10.0
GLARE_PIXEL_THRESH = 240               # Điểm ảnh > 240 trong vùng mắt
GLARE_AREA_RATIO_MAX = 0.30            # Glare > 30% diện tích mắt -> Báo Vàng ("Nghiêng mặt nhẹ để tránh lóa kính")
MASK_CHROMA_DIST = 18.0                # Chênh lệch sắc độ cằm so với trán
MASK_DARK_RATIO = 0.45                 # Độ sáng nửa dưới / trán
SUNGLASS_EYE_RATIO = 0.42
SUNGLASS_EYE_STD = 22.0

# 5. Consecutive Frames Smoothing (Bộ lọc ổn định)
FQA_CONSECUTIVE_FRAMES = 6             # 6 frames liên tiếp (~0.25s) đạt chuẩn để chuyển sang Active Liveness (nhạy bén, không delay)
HOLD_SEC = 0.25                        # Giữ yên đạt chuẩn trong 0.25s (~6 frames)

# ---------------------------------------------------------------------------
# (c) LIVENESS QUAY ĐẦU
# ---------------------------------------------------------------------------
TURN_YAW_DEG = 20.0                     # Góc quay đầu chuẩn eKYC (20 độ tự nhiên, không mất góc nhìn)
TURN_HOLD_FRAMES = 3
RECENTER_YAW_DEG = 10.0
CHALLENGE_TIMEOUT_SEC = 10.0
MAX_LOST_FRAMES = 6                    # cho phép mất mặt ngắn khi quay góc lớn
MAX_CENTER_JUMP = 0.18                 # tâm mặt nhảy > 18% khung giữa 2 frame -> nghi đổi mặt
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# (d) PHÓNG TO OVAL / TIẾN GẦN
# ---------------------------------------------------------------------------
ZOOM_MIN_GROWTH = 1.25                 # chiều cao mặt phải tăng >= 25% so với mốc ở bước (b)
ZOOM_TIMEOUT_SEC = 12.0

# ---------------------------------------------------------------------------
# (e) CHỐNG GIẢ MẠO ẢNH & VIDEO (PASSIVE ANTI-SPOOFING / PAD)
# ---------------------------------------------------------------------------
MINIFASNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "minifasnet_v2.onnx")
ANTISPOOF_REAL_THRESH = 0.65           # Ngưỡng tin cậy Real face của MiniFASNet (chuẩn production)
ANTISPOOF_PRINT_THRESH = 0.50          # Ngưỡng phát hiện ảnh in 2D (nhạy bén chặn ảnh in)
ANTISPOOF_REPLAY_THRESH = 0.50         # Ngưỡng phát hiện video/màn hình phát lại (chặn Replay)
MOIRE_ENERGY_RATIO_THRESH = 0.42       # Tỷ lệ năng lượng Fourier tần số cao (vân Moiré)
DEPTH_3D_MIN_DELTA = 0.025             # Độ nhô tối thiểu trục Z của chóp mũi so với 2 mắt (MediaPipe)

# ---------------------------------------------------------------------------
# (f) CHỐNG GIẢ MẠO QUANG HỌC CHỦ ĐỘNG (ACTIVE OPTICAL COLOR FLASHING PAD)
# ---------------------------------------------------------------------------
FLASH_MIN_PEARSON = 0.65                # Ngưỡng tương quan Pearson giữa phản xạ da và nguồn phát
FLASH_MIN_AMPLITUDE = 4.0               # Biên độ phản xạ tối thiểu (chống ảnh in tĩnh/video bất động)
FLASH_STEP_DURATION_MS = 330            # Thời lượng chiếu mỗi bước màu (ms)
FLASH_AWB_GAMMA = 0.45                  # Hệ số bù trừ độ lệch cân bằng trắng (AWB)
FLASH_CHALLENGE_TIMEOUT_SEC = 15.0      # Thời gian sống của Token thách thức (giây)

SESSION_TTL_SEC = 600
