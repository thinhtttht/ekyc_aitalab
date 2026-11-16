---
title: Ekyc Face Recognition Api
emoji: 👤
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 👤 eKYC & Biometric Face Verification REST API
## Hệ Thống API Đăng Ký, Chống Giả Mạo & Nhận Diện Khuôn Mặt (FastAPI + Docker)

API sinh trắc học khuôn mặt hiệu năng cao, tối ưu hóa cho CPU, sẵn sàng tích hợp vào các ứng dụng Web / Mobile / Thi Vấn Đáp / Điểm Danh / Chấm Công.

---

## 🌟 Tính Năng Cốt Lõi

1. **Đăng ký khuôn mặt (`POST /api/v1/register`):**
   * Tự động kiểm tra chất lượng FQA (đủ sáng, không mờ nhòe).
   * Kiểm duyệt Policy: Bắt buộc không đeo khẩu trang, không kính râm, không lấy tay che mặt, chỉ duy nhất 1 người.
   * Kiểm tra người thật bằng MiniFASNetV2 (`real_prob >= 0.70`).
   * Trích xuất vector đặc trưng ArcFace 512D và lưu vào CSDL.
2. **Xác thực danh tính 1:1 (`POST /api/v1/verify`):**
   * Xác thực ảnh selfie hiện tại với hồ sơ thí sinh theo mã `user_id`.
   * So khớp Cosine Similarity với vector gốc trong CSDL (ngưỡng mặc định `0.55`).
3. **So khớp trực tiếp 2 ảnh (`POST /api/v1/compare`):**
   * So khớp 1:1 giữa `image_source` (ảnh thẻ/CCCD) và `image_live` (ảnh selfie).
   * Hoàn toàn Stateless, không cần lưu trữ CSDL.
4. **Quản lý dữ liệu & Sao lưu (`GET /api/v1/database/backup`):**
   * Tải file sao lưu SQLite về máy hoặc khôi phục dữ liệu tức thì.

---

## 🚀 Hướng Dẫn Sử Dụng API (Code Mẫu Cho Khách Hàng)

Tài liệu Swagger UI tương tác trực tiếp tại: **`https://<your-space-name>.hf.space/docs`**

### 1. Đăng Ký Khuôn Mặt (`register`) - Python

```python
import requests

url = "https://<your-space-name>.hf.space/api/v1/register"

data = {
    "user_id": "SV001",
    "full_name": "Nguyen Van A",
    "enforce_oval": "false",       # true nếu bắt buộc đầu phải trong khung Oval
    "require_liveness": "true"     # Bắt buộc người thật
}

files = {
    "image": open("portrait.jpg", "rb")
}

response = requests.post(url, data=data, files=files)
print(response.json())
```

### 2. Xác Thực Điểm Danh (`verify`) - Python

```python
import requests

url = "https://<your-space-name>.hf.space/api/v1/verify"

data = {
    "user_id": "SV001",
    "enforce_oval": "false",
    "threshold": "0.55"           # Ngưỡng khớp 55%
}

files = {
    "image": open("webcam_selfie.jpg", "rb")
}

response = requests.post(url, data=data, files=files)
res = response.json()

if res.get("verified"):
    print(f"✅ Điểm danh thành công! Độ khớp: {res['confidence_pct']}%")
else:
    print(f"❌ Từ chối: {res.get('message')}")
```

### 3. Gọi Bằng cURL (Terminal / Postman)

```bash
# Đăng ký
curl -X POST "https://<your-space-name>.hf.space/api/v1/register" \
  -F "user_id=SV001" \
  -F "full_name=Nguyen Van A" \
  -F "image=@portrait.jpg"

# Xác thực
curl -X POST "https://<your-space-name>.hf.space/api/v1/verify" \
  -F "user_id=SV001" \
  -F "image=@selfie.jpg"
```

---

## 🧠 Các Mô Hình AI Sử Dụng

* **ArcFace ResNet50 (`w600k_r50.onnx` - 174 MB):** Trích xuất vector đặc trưng 512D chuẩn hóa L2. Độ chính xác nhận diện > 99.7%.
* **MiniFASNetV2 (`minifasnet_v2.onnx` - 1.7 MB):** Chống giả mạo ảnh in (Print attack), video phát lại trên màn hình iPad/điện thoại (Replay attack).
* **MediaPipe FaceMesh & Hands:** Cung cấp 478 mốc tọa độ 3D và 21 khớp tay để phân tích khẩu trang, kính râm, che tay.
