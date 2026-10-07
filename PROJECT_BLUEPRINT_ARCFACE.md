# PROJECT BLUEPRINT: MODULE ĐĂNG KÝ SINH TRẮC HỌC ARCFACE & CƠ SỞ DỮ LIỆU (CỔNG 3 - BƯỚC 5)

> **Ghi chú trạng thái (branch `quoc`):** Đây là tài liệu thiết kế mục tiêu, có thể không khớp với code hiện tại. Hiện trạng xem [README.md](README.md) và [PIPELINE.md](PIPELINE.md); phần chưa làm xem [ROADMAP.md](ROADMAP.md).

> **Mã tài liệu:** `BLUEPRINT-ARCFACE-01`  
> **Phiên bản:** `1.0.0`  
> **Tiêu chuẩn áp dụng:** DO-178C / ASPICE Level 2 / ASD-STE100 / ISO/IEC 19794-5  
> **Vai trò ban hành:** Ban Kỹ Thuật (Mini Engineering Board - PM, System Architect, Senior AI Engineer, Physical UX Designer)  
> **Căn cứ nghiệp vụ:** [SRS_EKYC_System.md](file:///d:/EKYC/SRS_EKYC_System.md) ([REQ-BE-04], [REQ-BE-05]) & [PIPELINE.md](file:///d:/EKYC/PIPELINE.md) (Tầng 3: ArcFace)

---

## 1. TỔNG QUAN BÀI TOÁN (PROBLEM STATEMENT)

### 1.1. Hiện trạng hệ thống (As-Is)
- Hệ thống eKYC hiện tại đã hoàn tất 4 bước đầu tiên:
  1. **Bước 1 (Camera Check):** Kiểm tra tín hiệu, độ phân giải, độ sáng, chống đóng băng frame.
  2. **Bước 2 (FQA):** Căn chỉnh khuôn mặt vào khung Oval, kiểm tra ngũ quan, chặn khẩu trang, kính râm, tay che mặt.
  3. **Bước 3 (Active Liveness):** Thử thách quay đầu ngẫu nhiên Trái/Phải với góc $\ge 20^\circ$.
  4. **Bước 4 (Zoom & Optical Flashing):** Tiến gần $\ge 125\%$ kích thước ban đầu, phát chuỗi 3 màu ngẫu nhiên khoét rỗng khung oval, kiểm tra phản xạ quang phổ mô da (AWB-compensated Pearson correlation $r \ge 0.65$).
- **Thiếu sót tại Bước 5:** Sau khi quét quang học thành công, client chỉ chụp một ảnh snapshot toàn cảnh thô hiển thị trên modal. Hệ thống **chưa** thực hiện trích xuất vector sinh trắc học ArcFace 512 chiều, **chưa** nắn thẳng mặt $112 \times 112$, **chưa** cho người dùng nhập danh tính (Họ tên, CCCD/User ID), và **chưa** lưu trữ vào cơ sở dữ liệu `database.json`.

### 1.2. Mục tiêu tương lai (To-Be)
- Xây dựng hoàn chỉnh **Bước 5: Hoàn tất Đăng ký (Cổng 3 - Biometric Face Enrollment)**:
  - Tự động chụp frame HD chuẩn ngay khi vượt qua Color Flashing.
  - Sử dụng 5 điểm mốc chuẩn (2 mắt, đỉnh mũi, 2 khóe miệng) để biến đổi Affine Transform, tạo ảnh chân dung nắn thẳng chuẩn kích thước $112 \times 112$.
  - Đưa ảnh $112 \times 112$ qua mạng **ArcFace ResNet-50** (`w600k_r50.onnx`), trích xuất vector nhúng đặc trưng 512 số thực với chuẩn hóa $L_2$ ($\|\mathbf{v}\|_2 = 1.0$).
  - Giao diện Bước 5 hiển thị ảnh chân dung chuẩn hóa $112 \times 112$, cho phép nhập Họ và Tên, CCCD / Mã người dùng.
  - Lưu trữ thông tin và vector nhúng an toàn vào `data/database.json`.
  - Chuẩn bị sẵn cấu trúc dữ liệu để phục vụ module So khớp Sinh trắc học (Verification Phase) trong tương lai.

---

## 2. MỤC TIÊU & PHI MỤC TIÊU (GOALS & NON-GOALS)

### 2.1. Mục tiêu cốt lõi (Core Goals)
- **[GOAL-01] Face Alignment $112 \times 112$:** Thuật toán chuẩn hóa hình học 5 điểm mốc (Standard ArcFace 5-point template) dùng phép biến đổi Affine `cv2.estimateAffinePartial2D` hoặc `similarity transform`.
- **[GOAL-02] ArcFace 512-d Feature Extraction:** Khởi chạy mô hình ONNX `w600k_r50.onnx` trên CPU/GPU thông qua ONNXRuntime, trích xuất vector $\mathbf{v} \in \mathbb{R}^{512}$ và chuẩn hóa $L_2$.
- **[GOAL-03] Biometric Database Persistence:** Quản lý cơ sở dữ liệu JSON chuẩn ISO/IEC 19794-5 (`data/database.json`), hỗ trợ thêm mới, đọc danh sách, chống trùng lặp User ID.
- **[GOAL-04] User Enrollment Physical UX:** Form giao diện Bước 5 mượt mà, trực quan, có preview ảnh $112 \times 112$, visualizer vector đặc trưng (dải màu/thanh sóng), trường nhập Họ tên, nút "Xác nhận Lưu Hồ Sơ".
- **[GOAL-05] API Endpoints:** 
  - `POST /api/enroll/extract_biometrics`: Nhận frame HD, trả về ảnh crop $112 \times 112$ (base64) và vector đặc trưng.
  - `POST /api/enroll/save_user`: Nhận User ID, Họ tên, vector đặc trưng, lưu vào database và trả về kết quả đăng ký thành công.
- **[GOAL-06] Comprehensive Testing:** Đạt độ bao phủ kiểm thử Unit & Integration Test $100\%$ cho bộ mã mới.

### 2.2. Phi mục tiêu (Non-Goals)
- **[NON-GOAL-01]:** Chưa xây dựng màn hình Xác thực chuyển tiền (Verification screen). Bài toán đó thuộc Phase tiếp theo (sau khi đã có dữ liệu đăng ký).
- **[NON-GOAL-02]:** Không lưu ảnh thô dung lượng lớn vào database; chỉ lưu vector 512 số và ảnh chân dung đã align kích thước chuẩn $112 \times 112$ vào thư mục lưu trữ cục bộ.
- **[NON-GOAL-03]:** Không đào tạo lại (train from scratch) mô hình ArcFace; kế thừa mô hình tiền huấn luyện uy tín `w600k_r50.onnx` của InsightFace có sẵn trong máy.

---

## 3. KẾ HOẠCH TỔNG THỂ (MASTER PLAN & MILESTONES)

| Giai đoạn (Phase) | Tên tài liệu / Hành vi | Nội dung trọng tâm | Trạng thái |
| :---: | :--- | :--- | :---: |
| **Phase 0** | `PROJECT_BLUEPRINT_ARCFACE.md` | Xác lập Mục tiêu, Phi mục tiêu, Cấu trúc dự án, Tiêu chuẩn DO-178C. | **ĐANG THỰC HIỆN** |
| **Phase 1** | `PRD_ARCFACE_ENROLLMENT.md` | Đặc tả Sản phẩm & Trải nghiệm Người dùng Bước 5 (State Machine, Error Prevention). | Chờ duyệt Phase 0 |
| **Phase 2** | `FSD_ARCFACE_ENROLLMENT.md` | Đặc tả Chức năng, Hợp đồng API, Cấu trúc Database schema, Edge cases. | Chờ duyệt Phase 1 |
| **Phase 3** | `TDD_ARCFACE_ENROLLMENT.md` | Thiết kế Kỹ thuật chi tiết, Toán học Affine Transformation, Cấu hình ONNXRuntime. | Chờ duyệt Phase 2 |
| **Phase 4** | `RTM_ARCFACE_ENROLLMENT.md` | Ma trận truy vết yêu cầu (Requirement Traceability Matrix). | Chờ duyệt Phase 3 |
| **Phase 5** | `Micro-Implementation` | Triển khai code từng module nhỏ kèm Unit test kiểm thử độc lập. | Chờ duyệt Phase 4 |

---

## 4. ĐỀ XUẤT CẤU TRÚC THƯ MỤC MỞ RỘNG

```text
d:\EKYC\
├── backend\
│   ├── models\
│   │   ├── minifasnet_v2.onnx        # Model chống giả mạo tĩnh (Tầng 2)
│   │   └── arcface_w600k_r50.onnx    # Model ArcFace ResNet-50 (Tầng 3, liên kết từ cache)
│   ├── face_encoder.py               # Module căn nắn 112x112 & trích xuất vector ArcFace 512-d
│   ├── user_db.py                    # Module quản lý cơ sở dữ liệu database.json
│   ├── app.py                        # Bổ sung 2 endpoints: /extract_biometrics và /save_user
│   └── tests\
│       ├── test_face_encoder.py      # Unit test căn nắn Affine & vector 512 số
│       └── test_user_db.py           # Unit test CRUD database.json
├── data\
│   ├── database.json                 # Cơ sở dữ liệu danh tính người dùng eKYC
│   └── enrollments\                  # Thư mục lưu ảnh chân dung 112x112 đã align
└── web\
    ├── index.html                    # Nâng cấp Modal Bước 5: Preview 112x112, Input Họ tên, Biometric Card
    ├── styles.css                    # CSS hiện đại phong cách FinTech cho Bước 5
    └── app.js                        # Tích hợp luồng gọi API trích xuất và lưu hồ sơ
```

---

## 5. CĂN CỨ KHOA HỌC & TÀI NGUYÊN HỆ THỐNG CÓ SẴN (AUTO-RAG & ZERO HALLUCINATION)

1. **Mô hình ArcFace ResNet-50:**
   - Tập tin có sẵn tại: `C:\Users\Admin\.insightface\models\buffalo_l\w600k_r50.onnx`
   - Kích thước tập tin: $174.38\text{ MB}$
   - Cấu trúc mạng: Input `['input.1'] (1, 3, 112, 112)`, Output `['683'] (1, 512)`
   - Độ chính xác: Đạt $99.83\%$ trên tập dữ liệu chuẩn LFW (Labeled Faces in the Wild).
2. **Khuôn mẫu 5 điểm mốc chuẩn ArcFace ($112 \times 112$ reference landmarks):**
   ```python
   ARCFACE_SRC_5PTS = np.array([
       [38.2946, 51.6963],  # Mắt trái
       [73.5318, 51.5014],  # Mắt phải
       [56.0252, 71.7366],  # Đỉnh mũi
       [41.5493, 92.3655],  # Khóe miệng trái
       [70.7299, 92.2041],  # Khóe miệng phải
   ], dtype=np.float32)
   ```
3. **Thư viện sẵn có trong venv:**
   - `insightface 2.0`, `onnxruntime 1.23.2`, `mediapipe 0.10.14`, `opencv-python 5.0.0.93`, `numpy 2.2.6`.

---

## 6. PHÊ DUYỆT TỪ BAN GIÁM ĐỐC (CHECKPOINT)

Theo quy chế **Document-Driven Development** của Ban Kỹ Thuật:
> Tuyệt đối KHÔNG viết mã lệnh khi chưa có sự phê duyệt tài liệu thiết kế.

Anh/Chị có đồng ý với Kế hoạch Tổng thể và Mục tiêu của Blueprint này không?  
**Vui lòng gõ: `Duyệt Blueprint` để tôi chuyển sang Phase 1: PRD (Product Requirements Document).**
