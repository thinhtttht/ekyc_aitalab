# Pipeline eKYC (hiện trạng branch `quoc`)

Hai luồng: **đăng ký** (nhiều bước, chạy trên luồng video) và **xác thực** (một ảnh). Các ngưỡng nằm trong
`backend/config.py`. Những gì chưa làm xem ở [ROADMAP.md](ROADMAP.md).

## 1. Đăng ký

Trình duyệt gửi từng frame (đã lật gương) lên `/api/enroll/frame`. Mỗi frame, backend chạy kiểm tra camera,
MediaPipe FaceMesh (+ Hands mỗi 2 frame), MiniFASNet (mỗi 3 frame, trung bình 5 lần gần nhất) và
`evaluate_face` một lần, rồi máy trạng thái trong `enrollment.py` quyết định bước tiếp theo.

```mermaid
flowchart TD
    START(["/api/enroll/start"]) --> CAM["CAMERA_CHECK<br/>độ phân giải, tín hiệu, sáng, nhiễu"]
    CAM --> FQA["FACE_QUALITY<br/>1 mặt, cự ly, trong Oval, thẳng, sáng, nét, không che, PAD"]
    FQA --> TURN["TURN_LEFT / TURN_RIGHT (thứ tự ngẫu nhiên)<br/>mỗi bên ≥ 20°"]
    TURN --> RC["RECENTER<br/>quay lại chính diện"]
    RC --> ZOOM["ZOOM_IN<br/>mặt to thêm ≥ 25%"]
    ZOOM --> FLASH["FLASHING<br/>/color_challenge → chiếu màu → /color_verify"]
    FLASH --> CAP["CAPTURE<br/>chụp ảnh HD → /verify_capture"]
    CAP --> FINAL{"Ảnh HD đạt?<br/>(PAD chạy mới, tư thế, che khuất, sáng)"}
    FINAL -- "Đạt" --> EMB["ArcFace 512D (căn mặt 112×112)"]
    EMB --> SAVE[("/api/users/enroll → SQLite backend/data/ekyc.db")]
    FINAL -- "Không đạt" --> FAILED["FAILED → làm lại từ đầu"]

    TURN -. "hết giờ / mất mặt" .-> FQA
    ZOOM -. "hết giờ" .-> FQA
    FQA -. "quá MAX_ATTEMPTS = 3 lần" .-> FAILED
```

Giả mạo (MiniFASNet báo ảnh in hoặc màn hình) bị chặn ở mọi bước, kể cả CAMERA_CHECK.

## 2. Xác thực (`/api/verify/face`)

```mermaid
flowchart LR
    IMG["Một ảnh"] --> DET{"Đúng 1 mặt?"}
    DET -- "Không" --> NOFACE["NO_FACE / MULTIPLE_FACES"]
    DET -- "Có" --> PAD{"MiniFASNet: người thật?"}
    PAD -- "Không" --> SPOOF["SPOOF_REJECTED"]
    PAD -- "Có" --> ARC["ArcFace 512D"]
    ARC --> MODE{"target_user_id?"}
    MODE -- "Có" --> ONE["1:1 cosine ≥ ngưỡng (mặc định 0.45)"]
    MODE -- "Không" --> MANY["1:N, trả top-5 ứng viên"]
```

Luồng xác thực hiện chưa có thử thách chủ động (nháy màu / quay đầu) như luồng đăng ký.

## 3. Các tầng phòng thủ

| Tầng | Công nghệ | Trạng thái |
| --- | --- | --- |
| Thử thách chủ động | Quay đầu ngẫu nhiên, tiến gần, nháy màu | Chỉ có ở luồng đăng ký |
| Chống giả mạo thụ động | MiniFASNetV2 (ONNX) | Bật. Thiếu model thì từ chối |
| Heuristic bổ sung | Độ sâu Z, vân Moiré, viền thiết bị | Có code, mặc định tắt (`ANTISPOOF_USE_*`) |
| Nhận diện | ArcFace w600k_r50 (ONNX) + cosine | Bật. Ngưỡng chưa hiệu chỉnh trên dữ liệu thật |
