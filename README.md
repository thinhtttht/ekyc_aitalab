# QAT eKYC – Đăng ký và xác thực khuôn mặt

Hệ thống eKYC khuôn mặt chạy trên trình duyệt: kiểm tra camera và chất lượng ảnh mặt (FQA), kiểm tra liveness chủ động (quay đầu, tiến gần, nháy màu quang học), chống giả mạo thụ động bằng MiniFASNet (PAD), trích xuất vector ArcFace 512 chiều, mã hóa ảnh chân dung và so khớp 1:1 / 1:N tốc độ cao.

> Những hạng mục mở rộng tiếp theo được liệt kê trong [ROADMAP.md](ROADMAP.md). Các tài liệu đặc tả gốc nằm trong [`docs/specs/`](docs/specs/).

---

## Tính năng hiện có

| Bước | Nội dung | Code |
| --- | --- | --- |
| Camera | Độ phân giải tối thiểu, mất tín hiệu / đóng băng, độ sáng, nhiễu. FPS thấp chỉ cảnh báo | `camera_quality.py` |
| Chất lượng mặt (FQA) | Một mặt duy nhất, cự ly, lọt Oval, đầu thẳng, ánh sáng, ngược sáng, độ nét, khẩu trang / kính râm / lóa kính / tay che / ngũ quan bị che | `face_analyzer.py` (`evaluate_face`) |
| Quay đầu | Trái / phải theo thứ tự ngẫu nhiên, mỗi bên ≥ 20°, rồi quay lại chính diện | `enrollment.py` |
| Tiến gần | Mặt phải to thêm ≥ 25% trong Oval phóng to | `enrollment.py` |
| Nháy màu | Màn hình chiếu chuỗi màu ngẫu nhiên, đo phản xạ trên da (tương quan Pearson) | `color_challenge.py`, `optical_analyzer.py` |
| Chống giả mạo thụ động | MiniFASNetV2 (ONNX) phân loại người thật / ảnh in / màn hình. Trong phiên chạy mỗi 3 frame, lấy trung bình 5 lần gần nhất | `anti_spoofing.py` |
| Ảnh chân dung cuối | Kiểm tra lại toàn bộ trên ảnh HD (PAD chạy mới, không dùng bộ đệm) rồi trích xuất ArcFace | `enrollment.py` (`verify_final_capture`) |
| Nhận diện | ArcFace w600k_r50 (ONNX), căn mặt 112×112 bằng OpenCV, cosine similarity vector hóa (mặc định ngưỡng 0.55) | `feature_extractor.py`, `database.py` |
| Lưu trữ & Mã hóa PII | SQLite `backend/data/ekyc.db` lưu vector float32; ảnh chân dung được nén & mã hóa đối xứng có xác thực (Encrypt-then-MAC) lưu tại `backend/data/portraits/*.enc` | `database.py` |
| Bảo mật API | Ràng buộc phiên đã kiểm định khi đăng ký user, hỗ trợ Admin API Key (`X-API-Key` / `Bearer`), Rate Limiting và giới hạn kích thước ảnh tải lên | `security.py`, `routers/` |

Ảnh được lật gương ở trình duyệt trước khi gửi lên, nên "trái / phải" trong backend khớp với những gì người dùng thấy trên màn hình.

---

## Chạy dự án

### Cách 1: Chạy trực tiếp trên máy (Python 3.11)

Yêu cầu Python 3.11 (mediapipe 0.10.14). Hai file model nằm trong `backend/models/`.
`w600k_r50.onnx` được lưu bằng Git LFS, nên cần chạy `git lfs pull` sau khi clone.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt          # chạy app
pip install -r backend/requirements-dev.txt      # thêm pytest + httpx để chạy test

.\run.bat                                         # hoặc:
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

### Cách 2: Chạy bằng Docker Compose

```powershell
docker compose up --build
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000) bằng trình duyệt và cho phép dùng camera. Nếu báo
"Camera đang bị chiếm giữ", hãy đóng các ứng dụng khác đang mở webcam (Zoom, Teams, tab trình duyệt khác).

Chạy test:

```powershell
python -m pytest backend/tests -q
```

---

## Cấu hình chính (`backend/config.py` & `.env.example`)

| Cờ / Biến môi trường | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `EKYC_ENROLL_REQUIRE_SESSION` | `1` (`True`) | Bắt buộc `/api/users/enroll` gắn với phiên đã qua `verify_final_capture` |
| `EKYC_ADMIN_API_KEY` | `""` | Nếu đặt, yêu cầu `X-API-Key` hoặc `Bearer` token cho API quản lý người dùng |
| `EKYC_RATE_LIMIT_PER_MIN` | `3600` | Giới hạn số request/phút trên mỗi IP (~60 FPS luồng video) |
| `EKYC_MAX_IMAGE_BYTES` | `10485760` (10MB) | Giới hạn dung lượng ảnh Base64 tải lên |
| `EKYC_FLASH_DEMO_BYPASS` | `1` (`True`) | Đặt `0` nếu muốn bắt buộc kiểm định tương quan nháy màu nghiêm ngặt |
| `ANTISPOOF_REQUIRE_MODEL` | `True` | Thiếu hoặc lỗi model MiniFASNet thì từ chối, không mặc định coi là người thật |
| `ANTISPOOF_EVERY_N_FRAMES` / `ANTISPOOF_SMOOTH_WINDOW` | `3` / `5` | Tần suất chạy và cửa sổ làm mượt MiniFASNet trong phiên đăng ký |
| `HAND_DETECTION_ENABLED` / `HAND_DETECTION_EVERY_N_FRAMES` | `True` / `2` | MediaPipe Hands (phát hiện tay che mặt) |
| `SHARPNESS_CHECK_ENABLED`, `MIN_FACE_SHARPNESS` | `True`, `15` | Chặn ảnh mờ (khi quay đầu dùng ngưỡng `MIN_FACE_SHARPNESS_TURNING`) |
| `EKYC_CORS_ORIGINS` | `127.0.0.1:8000,localhost:8000` | Danh sách Origin được phép gọi API |

Đặt biến môi trường `EKYC_DEBUG=1` để mỗi frame trả thêm chỉ số thô (xác suất PAD chi tiết, kết quả heuristic, fill, perspective...) phục vụ tinh chỉnh ngưỡng.

---

## API & WebSocket

| Method | Đường dẫn | Mô tả |
| --- | --- | --- |
| GET | `/api/health` | Kiểm tra trạng thái server và số phiên đang hoạt động |
| POST | `/api/enroll/start` · `/frame` · `/reset` | Tạo phiên, gửi từng frame HTTP, làm lại |
| WS | `/ws/enroll/{session_id}` | Truyền luồng frame nhị phân (JPEG Blob) thời gian thực độ trễ thấp |
| POST | `/api/enroll/color_challenge` · `/color_verify` | Nhận chuỗi màu và gửi kết quả nháy màu |
| POST | `/api/enroll/verify_capture` | Kiểm định ảnh chân dung HD cuối và trích xuất ArcFace |
| POST | `/api/users/enroll` | Lưu hồ sơ (yêu cầu `session_id` hợp lệ đã qua `verify_final_capture`) |
| GET / DELETE | `/api/users`, `/api/users/{user_id}` | Danh sách, chi tiết, xoá hồ sơ (hỗ trợ bảo vệ bằng `EKYC_ADMIN_API_KEY`) |
| POST | `/api/verify/face` | So khớp 1:1 (`target_user_id`) hoặc 1:N |

---

## Cấu trúc thư mục

```
ekyc_aitalab/
├── run.bat · Dockerfile · docker-compose.yml · .env.example
├── README.md · ROADMAP.md · pytest.ini
├── .github/workflows/ci.yml   # GitHub Actions tự động chạy pytest
├── docs/
│   ├── papers/                # Các bài báo khoa học (ArcFace, Silent-Face)
│   ├── specs/                 # Tài liệu đặc tả SRS, RFC, PRD/FSD/TDD/RTM, Blueprint, Pipeline
│   └── diagrams/              # Sơ đồ kiến trúc Draw.io (.xml) & trình xem trực quan
├── web/                       # Giao diện Web (được FastAPI phục vụ ở "/")
│   ├── index.html · styles.css · app.js
│   └── assets/                # Ảnh hướng dẫn khung chân dung
└── backend/
    ├── app.py                 # Khởi tạo FastAPI, Lifespan cleaner, Middleware & Static mount
    ├── config.py              # Toàn bộ ngưỡng và biến môi trường
    ├── state.py               # Quản lý singleton, khoá đồng bộ luồng (Lock) và Rotating Logger
    ├── security.py            # Xác thực API Key, Sliding Window Rate Limiter, kiểm tra kích thước ảnh
    ├── routers/               # Các API Router tách biệt: health.py, enroll.py, users.py, verify.py
    ├── camera_quality.py      # Kiểm tra chất lượng tín hiệu camera
    ├── face_analyzer.py       # MediaPipe FaceMesh/Hands + FQA (FaceVerdict)
    ├── head_pose.py           # Yaw / pitch / roll
    ├── anti_spoofing.py       # MiniFASNet + heuristic tuỳ chọn + bộ lấy mẫu theo phiên
    ├── enrollment.py          # Máy trạng thái đăng ký (an toàn đa luồng)
    ├── color_challenge.py · optical_analyzer.py   # Nháy màu quang học
    ├── feature_extractor.py   # ArcFace + căn mặt 5 điểm
    ├── database.py            # SQLite + mã hóa ảnh chân dung + tìm kiếm 1:N vector hóa
    ├── models/                # minifasnet_v2.onnx, w600k_r50.onnx (LFS)
    ├── data/                  # ekyc.db & portraits/*.enc (không commit)
    └── tests/                 # Bộ kiểm thử tự động (74 tests)
```
