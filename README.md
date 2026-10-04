# Smart-eKYC: Hệ Thống Định Danh Sinh Trắc Học Đa Tầng (Biometric Authentication PoC)

Hệ thống eKYC sinh trắc học khuôn mặt chuẩn ngân hàng số, kết hợp kiểm định chất lượng hình ảnh (FQA - Face Quality Assessment), thử thách chuyển động chủ động (Active Liveness) và phòng thủ chống tấn công giả mạo đa tầng (Passive Presentation Attack Detection - PAD theo chuẩn ISO/IEC 30107-3).

---

## 🌟 Tính Năng Nổi Bật (Features)

1. **Kiểm Tra Thiết Bị & Camera (Camera Quality)**:
   - Tự động nhận diện độ phân giải video ($\ge 640\times 480$).
   - Phát hiện mất tín hiệu, khung hình đen hoặc camera bị đóng băng/treo.
   - Cơ chế Early Gatekeeper: Khóa chặn giả mạo ngay từ Frame 0.

2. **Căn Chỉnh Khuôn Mặt Thông Minh (Smart Oval FQA)**:
   - Khung Oval SVG động phản hồi thời gian thực theo tỷ lệ chuẩn.
   - Kiểm tra định vị khuôn mặt: Cự ly, lọt khung, lệch tâm, xoay thẳng.
   - Kiểm tra điều kiện ánh sáng (độ sáng trung bình, chống ngược sáng).
   - Kiểm tra che khuất: Tự động phát hiện khẩu trang, kính râm đen, kính lóa phản quang, tay che mặt.

3. **Thử Thách Chuyển Động Chủ Động (Active Liveness 3D)**:
   - Thử thách ngẫu nhiên quay đầu Trái / Phải ($20^\circ$) theo chuẩn Selfie phản chiếu gương.
   - Thử thách tiến gần (Zoom in $\ge 125\%$) kiểm tra hiệu ứng biến thiên phối cảnh thực tế.

4. **Phòng Thủ Chống Giả Mạo Đa Tầng (Multi-Layer PAD / ISO/IEC 30107-3)**:
   - **MiniFASNetV2 (Silent-Face-Anti-Spoofing)**: Chạy ONNX Runtime phân loại 3 lớp (Người thật, Ảnh in 2D, Màn hình phát lại video).
   - **Độ sâu hình học 3D**: Trích xuất độ nhô sống mũi từ 468 điểm MediaPipe FaceMesh ($\Delta Z = Z_{eyes} - Z_{nose} \ge 0.025$).
   - **Phổ tần số cao 2D Fourier (FFT)**: Bóc tách năng lượng vân Moiré quang học của màn hình điện thoại/laptop.
   - **Nhận diện viền thiết bị**: Canny Edge + Hough Lines phát hiện cạnh viền điện thoại hoặc mép giấy in.

5. **Chụp Chân Dung HD Tự Động (HD Snapshot & Report)**:
   - Tự động chụp ảnh chân dung độ nét cao khi hoàn thành quy trình.
   - Bảng tổng kết số liệu kỹ thuật chi tiết.

---

## 🚀 Khởi Động Nhanh (Quick Start)

### Cách 1: Chạy 1-Click (Khuyến nghị trên Windows)
Nhấp đúp chuột vào file `run.bat` tại thư mục dự án:
```powershell
.\run.bat
```
File script sẽ tự động kích hoạt môi trường ảo Python `.venv`, khởi động FastAPI server và tự động mở trình duyệt web tại `http://127.0.0.1:8000`.

### Cách 2: Khởi động thủ công qua Terminal
```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 2. Cài đặt thư viện (nếu cài mới)
pip install -r backend/requirements.txt

# 3. Khởi chạy FastAPI Server (Backend + Giao diện Web)
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
Sau đó truy cập: [http://127.0.0.1:8000](http://127.0.0.1:8000) trên trình duyệt (Chrome, Edge) và cho phép quyền truy cập Camera.

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```
EKYC/
├── run.bat                     # Script khởi động tự động 1-click
├── README.md                   # Tài liệu hướng dẫn dự án
├── RFC_BIOMETRIC_PAD_SYSTEM.md # Tài liệu Thiết kế Kỹ thuật Chi tiết (RFC-004)
├── PIPELINE.md                 # Sơ đồ kiến trúc 3 tầng
├── SRS_EKYC_System.md          # Bản đặc tả yêu cầu hệ thống
│
├── web/                        # Giao diện người dùng Web (Vanilla JS / CSS)
│   ├── index.html              # Màn hình camera, khung Oval, checklist thời gian thực
│   ├── styles.css              # Giao diện Dark Mode ngân hàng số hiện đại
│   └── app.js                  # Điều khiển camera, vẽ lưới sinh trắc học, gọi API
│
├── backend/                    # Bộ não AI Engine (FastAPI)
│   ├── app.py                  # API endpoints & Static server
│   ├── config.py               # Cấu hình ngưỡng kỹ thuật trung tâm
│   ├── camera_quality.py       # Kiểm định chất lượng thiết bị camera
│   ├── head_pose.py            # Ước lượng tư thế đầu 3D (Yaw, Pitch, Roll)
│   ├── face_analyzer.py        # Phân tích FQA 4 tiêu chí cốt lõi
│   ├── anti_spoofing.py        # Phòng thủ chống giả mạo đa tầng
│   ├── enrollment.py           # Quản lý phiên và điều phối quy trình 5 bước
│   ├── requirements.txt        # Danh mục thư viện Python
│   │
│   ├── models/                 # Thư mục chứa mô hình AI
│   │   └── minifasnet_v2.onnx  # Model MiniFASNetV2 ONNX (~1.7 MB)
│   │
│   └── tests/                  # Bộ kiểm thử tự động (28 unit tests)
│       ├── test_anti_spoofing.py
│       ├── test_api.py
│       ├── test_camera_quality.py
│       ├── test_enrollment_flow.py
│       ├── test_fqa_pillars.py
│       └── test_head_pose.py
│
└── frontend/                   # Ứng dụng React / TypeScript dự phòng (Vite)
```

---

## 🧪 Chạy Kiểm Thử Tự Động (Run Tests)

Dự án đi kèm bộ kiểm thử tự động toàn diện:
```powershell
pytest backend/tests
```
Kết quả: **28/28 bài tests pass 100%**.

---

## 📜 Tài Liệu Thiết Kế (Documentation)

* [RFC-004: Biometric Presentation Attack Detection Architecture](RFC_BIOMETRIC_PAD_SYSTEM.md)
* [PIPELINE: Sơ đồ kiến trúc phòng thủ 3 tầng](PIPELINE.md)
* [SRS: Đặc tả yêu cầu kỹ thuật hệ thống](SRS_EKYC_System.md)
