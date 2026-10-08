"""Ngưỡng cấu hình tập trung cho luồng Đăng ký khuôn mặt (Enrollment).

Mọi toạ độ Oval được chuẩn hoá theo khung phân tích tỉ lệ 4:5 mà frontend gửi lên (480x600):
x, rx tính theo chiều rộng; y, ry tính theo chiều cao.
"""
import os

# ---------------------------------------------------------------------------
# (a) KIỂM TRA CHẤT LƯỢNG CAMERA
# ---------------------------------------------------------------------------
MIN_WIDTH, MIN_HEIGHT = 640, 480       # độ phân giải gốc tối thiểu của webcam
MIN_FPS = 25.0                         # FPS thực tế đo trên trình duyệt (dưới mức này chỉ cảnh báo, không chặn)
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
# 3. Sharpness Check (Laplacian Variance trên ROI mặt chuẩn hoá 200px)
# Webcam laptop thường cho 30-150. Nếu bị chặn oan, hạ ngưỡng thay vì tắt hẳn.
SHARPNESS_CHECK_ENABLED = True
MIN_FACE_SHARPNESS = 15.0              # laplacian_var < 15 -> Báo Vàng ("Giữ yên đầu và camera")
MIN_FACE_SHARPNESS_TURNING = 10.0      # Dung sai độ nét khi quay đầu (nhoè chuyển động)

# 4. Occlusion Check (Kính râm đen, Kính lóa phản quang, Khẩu trang, Bàn tay)
SUNGLASSES_EYE_MEAN_MAX = 30.0         # Eye ROI Mean < 30 VÀ Std < 10 -> Kính râm đen -> Báo Đỏ
SUNGLASSES_EYE_STD_MAX = 10.0
GLARE_PIXEL_THRESH = 240               # Điểm ảnh > 240 trong vùng mắt
GLARE_AREA_RATIO_MAX = 0.30            # Glare > 30% diện tích mắt -> Báo Vàng ("Nghiêng mặt nhẹ để tránh lóa kính")
MASK_CHROMA_DIST = 32.0                # Chênh lệch sắc độ má dưới so với trán (khi mũi/miệng bị che)
MASK_DARK_RATIO = 0.40                 # Độ sáng má dưới / trán
HAND_DETECTION_ENABLED = True          # MediaPipe Hands tốn gần bằng FaceMesh; tắt nếu máy yếu
HAND_DETECTION_EVERY_N_FRAMES = 2      # Chạy Hands mỗi N frame, các frame giữa dùng lại kết quả gần nhất

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
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# (d) PHÓNG TO OVAL / TIẾN GẦN
# ---------------------------------------------------------------------------
ZOOM_MIN_GROWTH = 1.25                 # chiều cao mặt phải tăng >= 25% so với mốc ở bước (b)
ZOOM_TIMEOUT_SEC = 12.0

# Ảnh chân dung cuối cùng (verify_final_capture) cho phép lệch tư thế nhiều hơn bước FQA một chút
FINAL_MAX_POSE = dict(yaw=16.0, pitch=16.0, roll=12.0)

# ---------------------------------------------------------------------------
# (e) CHỐNG GIẢ MẠO ẢNH & VIDEO (PASSIVE ANTI-SPOOFING / PAD)
# ---------------------------------------------------------------------------
MINIFASNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "minifasnet_v2.onnx")
ANTISPOOF_REQUIRE_MODEL = True         # Thiếu/lỗi model -> từ chối thay vì mặc định coi là người thật
ANTISPOOF_REAL_THRESH = 0.65           # Ngưỡng tin cậy Real face của MiniFASNet (chuẩn production)
ANTISPOOF_PRINT_THRESH = 0.50          # Ngưỡng phát hiện ảnh in 2D (nhạy bén chặn ảnh in)
ANTISPOOF_REPLAY_THRESH = 0.50         # Ngưỡng phát hiện video/màn hình phát lại (chặn Replay)
# Ba heuristic dưới đây chưa được kiểm chứng trên dữ liệu thật -> mặc định tắt, chỉ MiniFASNet quyết định
ANTISPOOF_USE_DEPTH = False            # Độ sâu 3D từ trục Z MediaPipe
ANTISPOOF_USE_MOIRE = False            # Vân Moiré (FFT)
ANTISPOOF_USE_BEZEL = False            # Viền thiết bị / mép giấy (Hough Lines)
ANTISPOOF_EVERY_N_FRAMES = 3           # Trong phiên đăng ký: chạy MiniFASNet mỗi N frame, các frame giữa dùng lại kết quả
ANTISPOOF_SMOOTH_WINDOW = 5            # Lấy trung bình xác suất trên K lần chạy gần nhất
MOIRE_ENERGY_RATIO_THRESH = 0.42       # Tỷ lệ năng lượng Fourier tần số cao (vân Moiré)
DEPTH_3D_MIN_DELTA = 0.025             # Độ nhô tối thiểu trục Z của chóp mũi so với 2 mắt (MediaPipe)

# ---------------------------------------------------------------------------
# (f) CHỐNG GIẢ MẠO QUANG HỌC CHỦ ĐỘNG (ACTIVE OPTICAL COLOR FLASHING PAD)
# ---------------------------------------------------------------------------
FLASH_MIN_PEARSON = 0.10                # Ngưỡng tương quan Pearson thích ứng với độ trễ camera thực tế
FLASH_MIN_AMPLITUDE = 1.0               # Biên độ phản xạ tối thiểu trên da mặt người thật
FLASH_STEP_DURATION_MS = 330            # Thời lượng chiếu mỗi bước màu (ms)
FLASH_AWB_GAMMA = 0.45                  # Hệ số bù trừ độ lệch cân bằng trắng (AWB)
FLASH_CHALLENGE_TIMEOUT_SEC = 15.0      # Thời gian sống của Token thách thức (giây)

SESSION_TTL_SEC = 600
# EKYC_DEBUG=1: trả thêm toàn bộ chỉ số thô (xác suất PAD chi tiết, heuristic, fill, perspective...) mỗi frame
DEBUG_METRICS = os.environ.get("EKYC_DEBUG", "0") == "1"
CORS_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]
