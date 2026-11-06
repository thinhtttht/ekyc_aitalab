# 📦 CHI TIẾT CÁC MODEL AI & THƯ VIỆN KÈM THEO
## 01_MODELS_AND_DEPENDENCIES.md

Tài liệu này cung cấp danh sách đầy đủ các mô hình trí tuệ nhân tạo (AI Models), thư viện phụ thuộc, kích thước tệp và hướng dẫn đóng gói file ZIP để bàn giao cho các bên tích hợp.

---

## 1. Bảng Tổng Hợp Các Model AI Sử Dụng

Hệ thống được thiết kế theo tiêu chí **Nhẹ - Nhanh - Hoạt động mượt mà trên CPU thông thường**, không bắt buộc máy tính của thí sinh phải có card đồ họa rời (GPU).

| Tên Model | Tên File Vật Lý | Kích Thước | Định Dạng | Input Tensor | Output Tensor | Độ Trễ (CPU i5 gen 10) | Mục Đích Sử Dụng |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ArcFace ResNet50** | `w600k_r50.onnx` | ~174 MB | ONNX | `(1, 3, 112, 112)` RGB chuẩn hóa `[-1, 1]` | `(1, 512)` float32 vector | ~35 - 50 ms | Trích xuất vector đặc trưng khuôn mặt 512 chiều, so khớp nhận diện 1:1 và 1:N với độ chính xác >99.7%. |
| **MiniFASNetV2** | `minifasnet_v2.onnx` | ~1.7 MB | ONNX | `(1, 3, 80, 80)` BGR chuẩn hóa `[0, 255]` | `(1, 3)` Softmax probabilities | ~8 - 12 ms | Chống giả mạo ảnh in (Print attack), video phát lại trên điện thoại/iPad/màn hình (Replay attack). |
| **MediaPipe FaceMesh** | Tích hợp sẵn trong thư viện `mediapipe` | ~8 MB (tự động load) | TFLite | Frame hình RGB bất kỳ kích thước | 468 hoặc 478 tọa độ 3D $(x, y, z)$ | ~12 - 18 ms | Xác định điểm mốc khuôn mặt, tính toán góc xoay đầu (Head Pose), phát hiện chớp mắt (EAR), điểm mốc đồng tử (Iris Gaze). |
| **MediaPipe Hands** | Tích hợp sẵn trong thư viện `mediapipe` | ~5 MB (tự động load) | TFLite | Frame hình RGB | 21 tọa độ 3D khớp tay $(x, y, z)$ | ~15 - 20 ms | Bắt cử động bàn tay, phát hiện thí sinh cố tình lấy tay che miệng, che mũi hoặc che khuất khuôn mặt. |
| **RealSense Depth Topography** *(Tùy chọn)* | Thuật toán toán học ma trận Numpy | 0 MB (Code thuần) | Python / Numpy | Ma trận Depth uint16 $(H, W)$ đơn vị mm | Điểm số độ cong 3D & sống mũi | < 1 ms | Xác thực người thật tuyệt đối nếu máy trạm có cắm camera 3D Intel RealSense (D415/D435/D455). |

---

## 2. Chi Tiết Kỹ Thuật Từng Model

### 2.1. Model Nhận Diện Khuôn Mặt: ArcFace ResNet50 (`w600k_r50.onnx`)
* **Kiến trúc:** Deep Residual Network (ResNet-50) kết hợp hàm mất mát Additive Angular Margin Loss (ArcFace).
* **Tập dữ liệu huấn luyện:** WebFace600K / Glint360k (hơn 600,000 danh tính, hàng chục triệu ảnh).
* **Đặc tính đầu ra:** Vector 512 chiều kiểu số thực `float32`. Sau khi trích xuất, vector được chuẩn hóa **L2 Norm** về mặt cầu đơn vị ($||\mathbf{v}||_2 = 1.0$).
* **Độ đo tương đồng (Cosine Similarity):**
  $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i \cdot v_i$$
* **Ngưỡng quyết định (Thresholds):**
  * `Similarity >= 0.55`: **Cùng một người** (Độ tin cậy > 99.8%, False Acceptance Rate FAR < 0.001%).
  * `Similarity < 0.45`: **Chắc chắn là hai người khác nhau**.
  * `0.45 <= Similarity < 0.55`: Vùng nghi vấn (yêu cầu điều chỉnh góc mặt hoặc chụp lại).

### 2.2. Model Chống Giả Mạo: MiniFASNetV2 (`minifasnet_v2.onnx`)
* **Kiến trúc:** MobileNet nhẹ chuyên dụng cho Presentation Attack Detection (PAD).
* **Đầu vào:** Vùng mặt cắt mở rộng theo tỷ lệ $1.5\times$ bounding box, resize về $80 \times 80$.
* **Đầu ra:** Xác suất `real_prob` (người thật) và `spoof_prob` (giả mạo).
* **Ngưỡng quyết định:**
  * `real_prob >= 0.70`: Được xem là người thật hợp lệ.
  * `real_prob < 0.50`: Phát hiện giả mạo (ảnh in giấy, màn hình điện thoại, máy tính bảng).

### 2.3. MediaPipe FaceMesh & 3D Head Pose
* **Đặc tính:** Nhận diện 468 mốc tọa độ 3D trên gương mặt thí sinh trong không gian chuẩn hóa `[0.0, 1.0]`.
* **Tính toán Head Pose (Góc xoay đầu):**
  Sử dụng thuật toán phối cảnh **PnP (Perspective-n-Point)** của OpenCV (`cv2.solvePnP`) kết hợp mô hình 3D chuẩn khuôn mặt (Standard 3D Face Model) và các điểm mốc:
  * Chóp mũi (Nose tip: Landmark 1)
  * Cằm (Chin: Landmark 199)
  * Khóe mắt trái (Left eye corner: Landmark 33)
  * Khóe mắt phải (Right eye corner: Landmark 263)
  * Khóe miệng trái (Left mouth corner: Landmark 61)
  * Khóe miệng phải (Right mouth corner: Landmark 291)
* **Kết quả trả về:** 3 góc Euler:
  * **Yaw (Trục Y - Xoay Trái/Phải):** Bình thường trong khoảng $[-15^\circ, +15^\circ]$. Nếu $|Yaw| > 25^\circ \rightarrow$ Thí sinh đang quay đầu nhìn ra ngoài.
  * **Pitch (Trục X - Cúi/Ngửa Đầu):** Bình thường trong khoảng $[-15^\circ, +12^\circ]$. Nếu $Pitch < -20^\circ \rightarrow$ Thí sinh đang cúi đầu nhìn điện thoại/tài liệu bên dưới.
  * **Roll (Trục Z - Nghiêng Đầu Sang Vai):** Bình thường trong khoảng $[-12^\circ, +12^\circ]$.

---

## 3. Danh Sách Thư Viện Cần Cài Đặt (`requirements.txt`)

Tạo tệp `requirements.txt` trong dự án backend của bạn với nội dung chuẩn hóa sau:

```text
# Thị giác máy tính & Xử lý ảnh
opencv-python>=4.9.0
numpy>=1.24.0,<2.0.0

# Suy luận mạng nơ-ron sâu (ONNX Runtime CPU)
onnxruntime>=1.16.0

# Trích xuất 468 mốc tọa độ khuôn mặt & cử động tay
mediapipe==0.10.14
protobuf>=4.25.3,<5.0.0

# Tùy chọn (Nếu dùng camera 3D Intel RealSense)
# pyrealsense2>=2.54.0
```

> [!IMPORTANT]
> **Lưu ý về phiên bản:**
> 1. `mediapipe==0.10.14` yêu cầu `protobuf < 5.0.0` và `numpy < 2.0.0`. Không cài đặt NumPy 2.x vì sẽ xung đột C-API với MediaPipe.
> 2. `onnxruntime` phiên bản CPU chạy rất tối ưu bằng tập lệnh AVX2/AVX-512, không cần cài đặt `onnxruntime-gpu` trừ khi bạn có server GPU riêng và muốn xử lý đồng thời hàng chục luồng camera.

---

## 4. Hướng Dẫn Đóng Gói File ZIP Bàn Giao (Delivery Checklist)

Khi đóng gói file ZIP để gửi cho nhóm khác, cấu trúc thư mục chuẩn nên bao gồm:

```text
ekyc_proctoring_kit/
├── models/
│   ├── w600k_r50.onnx           # Model ArcFace 512D (~174 MB)
│   └── minifasnet_v2.onnx       # Model MiniFASNet PAD (~1.7 MB)
├── core/
│   ├── __init__.py
│   ├── face_analyzer.py         # Trích xuất 468 điểm, FQA kiểm tra quy chuẩn
│   ├── feature_extractor.py     # ArcFace ONNX, trích xuất vector, tính tương đồng
│   ├── eye_tracker.py           # Theo dõi hướng nhìn và góc xoay đầu
│   ├── depth_liveness.py        # Module 3D Depth (tùy chọn camera 3D)
│   └── config.py                # Cấu hình ngưỡng an toàn & quy chuẩn
├── docs/                        # Thư mục tài liệu hướng dẫn
├── requirements.txt             # File danh mục thư viện pip
└── quick_start_demo.py          # Script chạy thử nghiệm độc lập
```
