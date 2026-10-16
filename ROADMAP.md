# ROADMAP – Lộ trình phát triển tiếp theo

Những gì đã hoàn thiện được mô tả chi tiết trong [README.md](README.md). Các tài liệu thiết kế gốc trong [`docs/specs/`](docs/specs/) mô tả hệ thống mục tiêu dài hạn.

## Đã hoàn thành trong bản nâng cấp kiến trúc hiện tại

- ✅ **Khóa bảo mật `/api/users/enroll`**: Bắt buộc gắn với phiên đăng ký (`session_id`) đã ở trạng thái `CAPTURE` và đã qua `verify_final_capture` (`EKYC_ENROLL_REQUIRE_SESSION=1`).
- ✅ **Tắt mặc định Demo Bypass**: `FLASH_DEMO_BYPASS` mặc định là `False` (chỉ bật khi cấu hình rõ biến môi trường `EKYC_FLASH_DEMO_BYPASS=1`).
- ✅ **Bảo vệ API Quản trị & Chống DoS**: Hỗ trợ xác thực `X-API-Key` / `Bearer` token qua `EKYC_ADMIN_API_KEY`, tích hợp `SlidingWindowRateLimiter` và giới hạn kích thước ảnh tải lên (`MAX_IMAGE_PAYLOAD_BYTES`).
- ✅ **Mã hóa & Tách lưu trữ ảnh chân dung (PII)**: Nén và mã hóa có xác thực (Encrypt-then-MAC với BLAKE2b + HMAC-SHA256) ảnh chân dung lưu ra `backend/data/portraits/*.enc` thay vì lưu chuỗi Base64 thô trong SQLite.
- ✅ **Tối ưu đối sánh 1:N & Đồng bộ đa luồng**: Loại bỏ N+1 SQLite queries trong `match_user_1_to_n` bằng phép nhân ma trận vector hóa (`np.dot`), bổ sung `threading.Lock` / `RLock` cho `verifier_analyzer` và `EnrollmentSession`, thêm background task định kỳ dọn phiên quá hạn.
- ✅ **Tách module Backend & Frontend + WebSocket**: Chia `app.py` thành các router trong `backend/routers/`; tách `web/app.js` thành các ES Modules trong `web/js/` kèm luồng truyền frame nhị phân qua WebSocket (`/ws/enroll/{session_id}`).
- ✅ **Đóng gói & CI/CD**: Bổ sung `Dockerfile`, `docker-compose.yml`, `.env.example`, `pytest.ini` và GitHub Actions `.github/workflows/ci.yml`.

---

## Các hạng mục nghiên cứu & mở rộng tiếp theo

### 1. Độ chính xác & Hiệu chỉnh sinh trắc học
- **Hiệu chỉnh ngưỡng trên tập dữ liệu thực tế**: Đo lường APCER / BPCER theo chuẩn ISO/IEC 30107-3 trên tập dữ liệu thực nghiệm (người thật, ảnh in chất lượng cao, màn hình OLED/IPS) để tinh chỉnh `ANTISPOOF_*_THRESH`, ngưỡng ArcFace và `FLASH_MIN_PEARSON`.
- **Kết hợp đa mô hình Silent-Face**: Kết hợp thêm nhánh `MiniFASNetV1SE` (scale 4.0) song song với `MiniFASNetV2` (scale 2.7) để tăng độ ổn định phát hiện biên thiết bị/mép giấy.
- **Liveness chủ động ở luồng Xác thực (`/api/verify/face`)**: Bổ sung tùy chọn yêu cầu thử thách nhanh (nháy màu hoặc quay đầu) trước khi đối sánh ở môi trường rủi ro cao.

### 2. Hạ tầng mở rộng quy mô lớn (High-Scale Production)
- **Distributed Session Store**: Chuyển bộ lưu phiên `sessions` từ RAM sang Redis khi triển khai cụm nhiều worker / nhiều container.
- **Dedicated Vector Database**: Tích hợp `sqlite-vec`, Faiss hoặc Qdrant khi số lượng hồ sơ vượt ngưỡng 100.000+ người dùng.
- **Nâng cấp MediaPipe Tasks / ONNX Face Detector**: Chuyển từ `mp.solutions.face_mesh` sang MediaPipe Tasks API hoặc SCRFD ONNX để tháo bỏ phụ thuộc vào Python 3.11 và hỗ trợ Python 3.12+.
