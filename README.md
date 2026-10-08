# Smart-eKYC – Đăng ký và xác thực khuôn mặt

Bản thử nghiệm eKYC khuôn mặt chạy trên trình duyệt: kiểm tra camera và chất lượng ảnh mặt, kiểm tra liveness, 
chống giả mạo thụ động bằng MiniFASNet, trích xuất vector ArcFace 512 chiều, lưu SQLite và so khớp 1:1 / 1:N.

> Đây là bản chưa đạt chuẩn production (chưa có xác thực API, ngưỡng chưa hiệu chỉnh
> trên dữ liệu thật). Những gì còn thiếu được liệt kê trong [ROADMAP.md](ROADMAP.md).

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
| Nhận diện | ArcFace w600k_r50 (ONNX), căn mặt 112×112 bằng OpenCV, cosine similarity (mặc định ngưỡng 0.45) | `feature_extractor.py` |
| Lưu trữ | SQLite `backend/data/ekyc.db`: vector float32 + ảnh chân dung base64 | `database.py` |

Ảnh được lật gương ở trình duyệt trước khi gửi lên, nên "trái / phải" trong backend khớp với những gì người dùng thấy trên màn hình.

---

## Chạy dự án

Yêu cầu Python 3.11 (mediapipe 0.10.14 chưa hỗ trợ 3.12+). Hai file model nằm trong `backend/models/`.
`w600k_r50.onnx` được lưu bằng Git LFS, nên cần chạy `git lfs pull` sau khi clone.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt          # chạy app
pip install -r backend/requirements-dev.txt      # thêm pytest + httpx để chạy test

.\run.bat                                         # hoặc:
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000) bằng trình duyệt và cho phép dùng camera. Nếu báo
"Camera đang bị chiếm giữ", hãy đóng các ứng dụng khác đang mở webcam (Zoom, Teams, tab trình duyệt khác).

Chạy test:

```powershell
python -m pytest backend/tests -q
```

---

## Cấu hình chính (`backend/config.py`)

| Cờ | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ANTISPOOF_REQUIRE_MODEL` | `True` | Thiếu hoặc lỗi model MiniFASNet thì từ chối, không mặc định coi là người thật |
| `ANTISPOOF_USE_DEPTH` / `_MOIRE` / `_BEZEL` | `False` | Heuristic độ sâu Z, vân Moiré, viền thiết bị. Chưa kiểm chứng nên tắt, chỉ MiniFASNet quyết định |
| `ANTISPOOF_EVERY_N_FRAMES` / `ANTISPOOF_SMOOTH_WINDOW` | `3` / `5` | Tần suất chạy và cửa sổ làm mượt MiniFASNet trong phiên đăng ký |
| `HAND_DETECTION_ENABLED` / `HAND_DETECTION_EVERY_N_FRAMES` | `True` / `2` | MediaPipe Hands (phát hiện tay che mặt) |
| `SHARPNESS_CHECK_ENABLED`, `MIN_FACE_SHARPNESS` | `True`, `15` | Chặn ảnh mờ (khi quay đầu dùng ngưỡng `MIN_FACE_SHARPNESS_TURNING`) |
| `MIN_FPS` | `25` | Dưới mức này chỉ cảnh báo |
| `FINAL_MAX_POSE` | yaw/pitch 16°, roll 12° | Giới hạn tư thế của ảnh chân dung cuối |
| `CORS_ORIGINS` | `127.0.0.1:8000`, `localhost:8000` | Origin được phép gọi API |

Đặt biến môi trường `EKYC_DEBUG=1` để mỗi frame trả thêm chỉ số thô (xác suất PAD chi tiết, kết quả heuristic,
fill, perspective...) phục vụ tinh chỉnh ngưỡng.

---

## API

| Method | Đường dẫn | Mô tả |
| --- | --- | --- |
| GET | `/api/health` | Kiểm tra server |
| POST | `/api/enroll/start` · `/frame` · `/reset` | Tạo phiên, gửi từng frame, làm lại |
| POST | `/api/enroll/color_challenge` · `/color_verify` | Nhận chuỗi màu và gửi kết quả nháy màu |
| POST | `/api/enroll/verify_capture` | Kiểm định ảnh chân dung HD cuối và trích xuất ArcFace |
| POST | `/api/users/enroll` | Lưu hồ sơ (tên + vector + ảnh) |
| GET / DELETE | `/api/users`, `/api/users/{user_id}` | Danh sách, chi tiết, xoá hồ sơ |
| POST | `/api/verify/face` | So khớp 1:1 (`target_user_id`) hoặc 1:N |

---

## Cấu trúc thư mục

```
ekyc_aitalab/
├── run.bat                    # Khởi động 1-click trên Windows
├── README.md · ROADMAP.md · PIPELINE.md
├── *.md (SRS, RFC, PRD/FSD/TDD/RTM Color Flashing, Blueprint)   # Tài liệu thiết kế gốc (xem ghi chú đầu mỗi file)
├── web/                       # Giao diện Vanilla JS (được FastAPI phục vụ ở "/")
│   ├── index.html · styles.css · app.js
└── backend/
    ├── app.py                 # FastAPI endpoints + static web
    ├── config.py              # Toàn bộ ngưỡng và cờ
    ├── camera_quality.py      # Kiểm tra camera
    ├── face_analyzer.py       # MediaPipe FaceMesh/Hands + FQA (FaceVerdict)
    ├── head_pose.py           # Yaw / pitch / roll
    ├── anti_spoofing.py       # MiniFASNet + heuristic tuỳ chọn + bộ lấy mẫu theo phiên
    ├── enrollment.py          # Máy trạng thái đăng ký, mỗi bước một hàm
    ├── color_challenge.py · optical_analyzer.py   # Nháy màu
    ├── feature_extractor.py   # ArcFace + căn mặt
    ├── database.py            # SQLite
    ├── models/                # minifasnet_v2.onnx, w600k_r50.onnx (LFS)
    ├── data/                  # ekyc.db (không commit)
    ├── requirements.txt · requirements-dev.txt
    └── tests/
```
