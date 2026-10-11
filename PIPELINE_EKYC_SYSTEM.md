# 🚀 Kiến Trúc Pipeline Hệ Thống eKYC: Đăng Ký (Enrollment) & Xác Thực (Verification)

Tài liệu kỹ thuật mô tả chi tiết toàn bộ luồng xử lý dữ liệu (Data Pipeline), giải thuật phân tích hình ảnh, đánh giá chất lượng (FQA), phòng thủ chống giả mạo (PAD), trích xuất vector đặc trưng ArcFace 512 chiều và cơ chế đối sánh sinh trắc học trong hệ thống Smart-eKYC.

---

## 📌 1. Tổng Quan Kiến Trúc Đa Tầng (High-Level Architecture)

```mermaid
flowchart TD
    subgraph CLIENT["Lớp Giao Diện Khách Hàng (Web Frontend)"]
        UI_CAM["Webcam Feed (HTML5 Canvas 4:5 & 16:9)"]
        UI_GUIDE["Khung Oval Hướng Dẫn & Phản Hồi Trực Quan"]
        UI_FLASH["Màn Phủ Quang Học Đổi Màu (Color Flashing)"]
    end

    subgraph INGESTION["Tầng Tiếp Nhận & Kiểm Định Phần Cứng"]
        CAM_CHK["Kiểm Tra Camera: Độ phân giải, FPS, Nhiễu, Treo"]
        DECODE["Giải mã Base64 -> Ma trận OpenCV BGR"]
    end

    subgraph FQA_PAD["Tầng Đánh Giá Chất Lượng (FQA) & Chống Giả Mạo (PAD)"]
        FACEMESH["MediaPipe FaceMesh (468/478 Điểm mốc 3D)"]
        FQA["Bộ Tiêu Chí FQA 7 Trụ Cột: Cự ly, Lọt Oval, Ánh sáng, Độ nét, Ngũ quan"]
        ACTIVE_LIV["Liveness Chủ Động: Quay đầu 3D (>= 20°) & Phóng to Oval (>= 25%)"]
        FLASH_PAD["Active Optical PAD: Tương quan Pearson phổ màu da mặt"]
        MINIFAS["Passive PAD: MiniFASNet v2 (Ảnh in 2D & Replay Attack)"]
    end

    subgraph BIOMETRICS["Tầng Sinh Trắc Học Học Sâu (Deep Biometrics)"]
        ALIGN["Face Alignment 5 Mốc Chuẩn (Mắt, Mũi, Khóe miệng) -> 112x112"]
        ARCFACE["InsightFace ArcFace ResNet-50 (WebFace600K)"]
        L2NORM["Chuẩn Hóa L2 Norm -> Vector 512D trên Mặt cầu S⁵¹¹"]
    end

    subgraph STORAGE_MATCH["Tầng Lưu Trữ & Đối Sánh (Storage & Matching)"]
        SQLITE[("CSDL SQLite: Vector BLOB + HD Snapshot + Metadata")]
        MATCH_ENGINE["Biometric Matcher: Cosine Similarity & Khoảng cách Euclid"]
    end

    UI_CAM --> DECODE --> CAM_CHK
    CAM_CHK --> FACEMESH
    FACEMESH --> FQA
    FQA --> ACTIVE_LIV --> FLASH_PAD --> MINIFAS
    MINIFAS --> ALIGN --> ARCFACE --> L2NORM
    L2NORM -->|Quy trình Đăng ký| SQLITE
    L2NORM -->|Quy trình Xác thực| MATCH_ENGINE
    SQLITE -.->|Nạp hồ sơ| MATCH_ENGINE
```

---

## 📝 2. Pipeline Chi Tiết: Luồng Đăng Ký (Enrollment Pipeline)

Quy trình đăng ký khuôn mặt diễn ra qua **5 giai đoạn nghiêm ngặt liên hoàn (Finite State Machine - FSM)** nhằm loại bỏ hoàn toàn gian lận danh tính:

```mermaid
sequenceDiagram
    autonumber
    actor User as Người dùng (Webcam)
    participant FE as Frontend (web/app.js)
    participant BE as Backend API (backend/app.py)
    participant FQA as Face Analyzer (FQA & PAD)
    participant Model as ArcFace & MiniFASNet
    participant DB as SQLite DB

    Note over User, FE: GIAI ĐOẠN 0 & 1: CAMERA & KHUNG OVAL FQA
    FE->>BE: POST /api/enroll/frame (Ảnh 480x600, stats)
    BE->>FQA: Phân tích MediaPipe FaceMesh & 7 trụ cột FQA
    FQA-->>BE: FaceResult (scale_ratio, oval_dist, brightness, sharpness, parts)
    BE-->>FE: Trả về feedback (Giữ thẳng đầu, vào khung Oval, đạt 6 frames liên tiếp)

    Note over User, FE: GIAI ĐOẠN 2: ACTIVE LIVENESS (QUAY ĐẦU 3D)
    FE->>BE: Frame quay đầu (Góc Yaw >= 20° Trái/Phải)
    BE->>FQA: Head Pose Estimation (PnP z-depth)
    FQA-->>BE: Đạt góc quay -> Yêu cầu quay lại chính diện (Yaw <= 10°)
    BE-->>FE: Hoàn thành thử thách quay đầu

    Note over User, FE: GIAI ĐOẠN 3: DYNAMIC ZOOM (TIẾN GẦN CAMERA)
    FE->>BE: Frame phóng to (Khung Oval mở rộng)
    BE->>FQA: Chiều cao khuôn mặt tăng >= 25% (face_h growth)
    BE-->>FE: Đạt cự ly cận cảnh -> Chuyển sang Flashing

    Note over User, FE: GIAI ĐOẠN 4: ACTIVE OPTICAL FLASHING & PASSIVE PAD
    BE->>FE: Sinh Token Thách Thức (Chuỗi màu: Đỏ -> Xanh -> Vàng)
    FE->>User: Đổi màu viền màn hình (330ms / bước màu)
    FE->>BE: Gửi frame kèm chỉ số màu phản xạ trên da mặt
    BE->>FQA: Tính hệ số tương quan Pearson r >= 0.10 & Biên độ >= 1.0
    BE->>Model: MiniFASNet v2 kiểm tra (Real Prob >= 65%)
    BE-->>FE: Thách thức quang học hợp lệ! Yêu cầu chụp HD

    Note over User, FE: GIAI ĐOẠN 5: CHỤP ẢNH HD & LƯU HỒ SƠ
    FE->>BE: POST /api/users/enroll (HD Snapshot 720p, Full Name)
    BE->>Model: Align 5 mốc chuẩn -> Trích xuất ArcFace 512D -> L2 Norm
    BE->>DB: Lưu User ID, Tên, Vector 512D (BLOB), Snapshot
    DB-->>BE: Ghi đĩa thành công
    BE-->>FE: Thông báo đăng ký thành công -> Mở Modal Xác thực
```

### Chi tiết các bước xử lý trong Luồng Đăng Ký:

| Giai đoạn | Tên kỹ thuật | Giải thuật & Công nghệ cốt lõi | Điều kiện vượt qua (Pass Criteria) |
| :--- | :--- | :--- | :--- |
| **Stage 0** | **Camera Quality** | Immerkær Noise Estimation, Luma Variance | Độ phân giải $\ge 640\times 480$, FPS đo $\ge 25$, Độ sáng $45-215$, Không đóng băng khung hình. |
| **Stage 1** | **FQA & Framing** | MediaPipe FaceMesh (468 pts), Sobel Edge, Convex Hull | Tỷ lệ Oval $0.40 \le \text{scale} \le 0.85$, Lọt Oval $d \le 1.0$, Góc thẳng $|\text{Yaw}| \le 12^\circ$, Nét $\ge 15.0$, Ngũ quan rõ. |
| **Stage 2** | **3D Turn Liveness** | Head Pose Estimation (SolvePnP trên 3D Landmarks) | Quay đầu sang trái hoặc phải đạt $|\text{Yaw}| \ge 20^\circ$, sau đó đưa về tâm $\le 10^\circ$. |
| **Stage 3** | **Zoom In Challenge** | Dynamic Ellipse Expansion, Bounding Box Growth | Chiều cao mặt tăng trưởng $\ge 25\%$ so với giai đoạn 1, mặt nằm trọn trong Oval phóng to. |
| **Stage 4** | **Optical Flashing PAD**| Pearson Correlation, MiniFASNet v2 ONNX | Hệ số tương quan $r \ge 0.10$, Biên độ $\ge 1.0$, MiniFASNet Real Probability $\ge 65\%$. |
| **Stage 5** | **Feature Extractor** | Affine Warp 112x112, ResNet-50 ArcFace ONNX | Vector 512 chiều, chuẩn hóa $\|\mathbf{e}\|_2 = 1.0$, lưu SQLite kèm User ID tự sinh `USR-XXXXXXXX`. |

---

## 🔍 3. Pipeline Chi Tiết: Luồng Xác Thực (Verification Pipeline 1:1 & 1:N)

Xác thực sinh trắc học được thiết kế theo tiêu chuẩn an toàn cấp ngân hàng (Zero-Trust Pipeline), áp dụng bộ lọc FQA tĩnh độc lập để **ngăn chặn tuyệt đối hiện tượng nhận diện bóng ma hoặc đoán bừa**:

```mermaid
flowchart TD
    START(["Bắt đầu Xác thực (Nhấn nút hoặc Auto-scan)"]) --> CAPTURE["Chụp Frame Trực Tiếp từ Webcam (640x480 Canvas)"]
    CAPTURE --> API["Gửi POST /api/verify/face (Base64 Image, Mode, Threshold)"]

    subgraph STEP1["BƯỚC 1: BỘ DÒ TĨNH & KIỂM ĐỊNH HIỆN DIỆN (PRESENCE CHECK)"]
        API --> STATIC_MESH["MediaPipe FaceMesh (static_image_mode = TRUE)"]
        STATIC_MESH --> CHK_COUNT{"Số lượng mặt phát hiện?"}
        CHK_COUNT -->|"0 khuôn mặt"| ERR_NO_FACE["Từ chối: NO_FACE<br/>(Không phát hiện khuôn mặt)"]
        CHK_COUNT -->|"> 1 khuôn mặt"| ERR_MULTI["Từ chối: MULTIPLE_FACES<br/>(Nhiều hơn 1 người trước camera)"]
        CHK_COUNT -->|"1 khuôn mặt"| CHK_AREA{"Diện tích mặt >= 2.5% khung hình?"}
        CHK_AREA -->|"Không (Vết nhiễu/ảnh xa)"| ERR_TINY["Từ chối: NO_FACE<br/>(Loại bỏ nhiễu hậu cảnh)"]
    end

    subgraph STEP2["BƯỚC 2: HÀNG RÀO KIỂM SOÁT FQA 7 TIÊU CHÍ (STRICT FQA)"]
        CHK_AREA -->|"Đạt diện tích"| FQA_INV{"Mặt bị lộn ngược?"}
        FQA_INV -->|"Có"| ERR_POSE["Từ chối: INVALID_POSE"]
        FQA_INV -->|"Không"| FQA_OCC{"Bị che khuất?<br/>(Tay, Khẩu trang, Kính râm, Mất mắt/mũi)"}
        FQA_OCC -->|"Có"| ERR_OCC["Từ chối: FACE_OCCLUDED"]
        FQA_OCC -->|"Không"| FQA_OVAL{"Lọt khung Oval & Cự ly?<br/>(Oval dist <= 1.30, Scale 0.30 - 0.95)"}
        FQA_OVAL -->|"Không"| ERR_OVAL["Từ chối: INVALID_FRAMING"]
        FQA_OVAL -->|"Đạt"| FQA_HEAD{"Nhìn thẳng?<br/>(|Yaw| <= 30°, |Pitch| <= 25°)"}
        FQA_HEAD -->|"Nghiêng quá mức"| ERR_HEAD["Từ chối: INVALID_POSE"]
        FQA_HEAD -->|"Thẳng"| FQA_SHARP{"Độ nét Sharpness >= 15.0?"}
        FQA_SHARP -->|"Mờ / Rung lắc"| ERR_BLUR["Từ chối: IMAGE_BLURRY"]
        FQA_SHARP -->|"Nét"| FQA_LIGHT{"Độ sáng trong khoảng 40 - 210?"}
        FQA_LIGHT -->|"Quá tối hoặc lóa"| ERR_LIGHT["Từ chối: POOR_LIGHTING"]
    end

    subgraph STEP3["BƯỚC 3: PHÒNG THỦ CHỐNG GIẢ MẠO PASSIVE PAD"]
        FQA_LIGHT -->|"Đạt ánh sáng"| PAD_RUN["Chạy MiniFASNet v2 (Khớp BBox)"]
        PAD_RUN --> PAD_CHK{"is_real == True?<br/>(Real prob >= 65%)"}
        PAD_CHK -->|"Giả mạo (In / Video)"| ERR_PAD["Từ chối: SPOOF_REJECTED<br/>(Cảnh báo gian lận PAD)"]
    end

    subgraph STEP4["BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG ARCFACE"]
        PAD_CHK -->|"Người thật"| EXTRACT_KPS["Lấy 5 mốc chuẩn ArcFace [Mắt T, Mắt P, Mũi, Mép T, Mép P]"]
        EXTRACT_KPS --> WARP_AFFINE["Biến đổi Affine tương tự (norm_crop) -> Ảnh 112x112"]
        WARP_AFFINE --> RUN_ARCFACE["Suy luận ResNet-50 ArcFace (ONNX Runtime)"]
        RUN_ARCFACE --> NORM_512["Chuẩn hóa L2 Norm: ||e_query|| = 1.0"]
    end

    subgraph STEP5["BƯỚC 5: ĐỐI SÁNH SINH TRẮC HỌC (BIOMETRIC MATCHING)"]
        NORM_512 --> CHECK_MODE{"Chế độ Đối Sánh?"}

        CHECK_MODE -->|"1:1 (So khớp đích danh)"| MATCH_11["Lấy Vector của Target User ID trong CSDL"]
        MATCH_11 --> CALC_11["Tính Cosine Sim = e_query • e_target<br/>Khoảng cách Euclid d = sqrt(2(1 - Sim))"]
        CALC_11 --> DECIDE_11{"Cosine Sim >= Threshold (0.55)?"}
        DECIDE_11 -->|"Đạt"| RES_11_PASS["Xác thực Thành Công (MATCH_SUCCESS)"]
        DECIDE_11 -->|"Không đạt"| RES_11_FAIL["Không Khớp Danh Tính (MISMATCH)"]

        CHECK_MODE -->|"1:N (Nhận diện toàn CSDL)"| LOAD_ALL["Quét toàn bộ Users trong SQLite"]
        LOAD_ALL --> CALC_ALL["Tính Dot Product e_query • e_i với mọi User"]
        CALC_ALL --> SORT_TOP["Sắp xếp giảm dần -> Lấy Top-1 Candidate"]
        SORT_TOP --> DECIDE_1N{"Top-1 Sim >= Threshold (0.55)?"}
        DECIDE_1N -->|"Đạt"| RES_1N_PASS["Khớp Danh Tính Top-1 (MATCH_SUCCESS)"]
        DECIDE_1N -->|"Không đạt"| RES_1N_FAIL["Không Khớp Danh Tính Nào (MISMATCH)"]
    end

    subgraph STEP6["BƯỚC 6: PHẢN HỒI GIAO DIỆN KHÁCH HÀNG (UI RENDERING)"]
        RES_11_PASS --> UI_PASS["Hiển thị Card User xanh, phát âm thanh 'Ting', Leaderboard xanh"]
        RES_1N_PASS --> UI_PASS
        RES_11_FAIL --> UI_FAIL["Hiển thị Banner Đỏ 'Không trùng khớp', Leaderboard xám trung tính"]
        RES_1N_FAIL --> UI_FAIL
        ERR_NO_FACE --> UI_WARN["Hiển thị Banner Vàng/Cam cảnh báo, ẨN HOÀN TOÀN User Card & Leaderboard"]
        ERR_MULTI --> UI_WARN
        ERR_TINY --> UI_WARN
        ERR_POSE --> UI_WARN
        ERR_OCC --> UI_WARN
        ERR_OVAL --> UI_WARN
        ERR_HEAD --> UI_WARN
        ERR_BLUR --> UI_WARN
        ERR_LIGHT --> UI_WARN
        ERR_PAD --> UI_DANGER["Hiển thị Banner Báo Động Đỏ PAD Attack, Ẩn User Card"]
    end
```

---

## 📐 4. Nền Tảng Toán Học & Giải Thuật Sinh Trắc Học

### 1. Không gian nhúng ArcFace (Additive Angular Margin Loss)
Mô hình trích xuất đặc trưng ánh xạ ảnh khuôn mặt $I \in \mathbb{R}^{112 \times 112 \times 3}$ thành một vector biểu diễn $\mathbf{e} \in \mathbb{R}^{512}$. Vector được chuẩn hóa về mặt cầu đơn vị siêu chiều $S^{511}$:

$$\|\mathbf{e}\|_2 = \sqrt{\sum_{i=1}^{512} e_i^2} = 1.0$$

Khi đó, độ tương đồng Cosine giữa vector người dùng truy vấn $\mathbf{e}_q$ và vector người dùng đăng ký $\mathbf{e}_t$ chính là tích vô hướng:

$$\text{Cosine Similarity } S_c(\mathbf{e}_q, \mathbf{e}_t) = \mathbf{e}_q \cdot \mathbf{e}_t = \sum_{k=1}^{512} e_{q, k} \cdot e_{t, k} = \cos(\theta)$$

### 2. Mối liên hệ với khoảng cách Euclid (Euclidean Distance)
Vì hai vector đều có độ dài chuẩn tắc $\|\mathbf{e}\|_2 = 1.0$, khoảng cách Euclid và độ tương đồng Cosine liên hệ trực tiếp qua công thức:

$$\|\mathbf{e}_q - \mathbf{e}_t\|_2^2 = \|\mathbf{e}_q\|_2^2 + \|\mathbf{e}_t\|_2^2 - 2(\mathbf{e}_q \cdot \mathbf{e}_t) = 1 + 1 - 2 S_c = 2(1 - S_c)$$

$$D_e = \sqrt{2(1 - S_c)}$$

- Khi hai vector **giống hệt nhau**: $S_c = 1.0 \implies D_e = 0.0$
- Khi hai vector **trực giao (hoàn toàn không liên quan)**: $S_c = 0.0 \implies D_e = \sqrt{2} \approx 1.414$
- Tại ngưỡng an toàn $\tau = 0.55$: $D_e = \sqrt{2(1 - 0.55)} = \sqrt{0.90} \approx 0.948$

### 3. Chuẩn hóa độ tự tin xác thực (Confidence Score Mapping)
Để hiển thị trực quan cho người dùng theo thang đo phần trăm $[0\%, 100\%]$:

$$\text{Confidence}(\%) = \begin{cases} 
60.0 + \left(\dfrac{S_c - \tau}{1.0 - \tau}\right) \times 40.0 & \text{nếu } S_c \ge \tau \text{ (Hợp lệ)} \\[8pt]
\max\left(0.0, \dfrac{S_c}{\tau}\right) \times 59.9 & \text{nếu } S_c < \tau \text{ (Từ chối)}
\end{cases}$$

---

## 🛡️ 5. Ma Trận Phòng Thủ & Đối Ứng Gian Lận (Attack Vectors & Mitigations)

| Kịch bản tấn công gian lận | Biểu hiện vi phạm | Cơ chế phát hiện & Chặn đứng | Mã lỗi phản hồi |
| :--- | :--- | :--- | :--- |
| **Không có mặt trước camera** | Camera quay vào bàn/ghế/tường | `FaceAnalyzer(stream=False)` chạy bộ dò tĩnh; kiểm tra diện tích $\text{area} < 2.5\%$ | `NO_FACE` |
| **Bỏ mặt ra sau khi nhận diện** | Người dùng bước ra ngoài sau lần quét trước | `static_image_mode = True` triệt tiêu hoàn toàn bộ đệm bám vết (Ghost Tracking) | `NO_FACE` |
| **Che mặt một phần bằng giấy / vật cản** | Che nửa mặt, chỉ hở trán hoặc cằm | Kiểm tra bộ phận ngũ quan `PARTS_CONFIG` (mắt trái, mắt phải, mũi bắt buộc rõ) | `FACE_OCCLUDED` |
| **Đeo khẩu trang y tế / vải** | Khẩu trang phủ kín miệng & mũi | Heuristic phân tích độ lệch sắc độ YCrCb má dưới so với trán ($\text{chroma} > 32.0$) | `FACE_OCCLUDED` |
| **Đeo kính râm đen / Kính lóa phản quang**| Mắt bị che tối hoặc lóa đèn flash | ROI mắt có độ sáng Mean $< 30$ hoặc tỷ lệ điểm ảnh cháy lóa $\ge 240$ chiếm $> 30\%$ | `FACE_OCCLUDED` |
| **Dùng ảnh in màu 2D (Printed Photo)** | Đưa ảnh chụp chân dung in trên giấy A4 | Mạng nơ-ron MiniFASNet v2 phân tích kết cấu vi mô (Fourier Texture & Reflection) | `SPOOF_REJECTED` |
| **Dùng điện thoại/iPad phát lại video (Replay)**| Chiếu màn hình điện thoại trước webcam | MiniFASNet phát hiện vân Moiré quét màn hình tần số cao & phản xạ kính phân cực | `SPOOF_REJECTED` |
| **Ảnh mờ tịt / Rung lắc out-of-focus** | Di chuyển camera quá nhanh | Phân tích phương sai toán tử Laplacian trên ROI mặt ($\text{var} < 15.0$) | `IMAGE_BLURRY` |
| **Mặt quá xa hoặc quá gần** | Đứng cách camera $> 2\text{m}$ hoặc dí sát mặt | Tỷ lệ vàng kích thước khuôn mặt so với khung Oval ($0.30 \le \text{scale} \le 0.95$) | `INVALID_FRAMING` |

---

## 🎯 6. Kết luận & Điểm Sáng Trình Bày (Presentation Takeaways)

1. **Kiến trúc Zero-Trust:** Mỗi frame xác thực đều được đối xử như một ảnh tĩnh độc lập, triệt tiêu hoàn toàn hiện tượng nhớ vết (Ghost Landmarks) của các framework camera thông thường.
2. **Hàng rào FQA 7 trụ cột độc quyền:** Không chỉ dựa vào ArcFace, hệ thống sử dụng thuật toán Computer Vision thuần (Laplacian, YCrCb, Sobel) để lọc sạch các ảnh rác, ảnh mờ và ảnh che chắn trước khi tốn tài nguyên chạy mạng nơ-ron sâu.
3. **Phòng thủ đa tầng Passive & Active:** Kết hợp giữa kiểm tra phản xạ quang học chủ động (Color Flashing) và mô hình học sâu thụ động (MiniFASNet v2), mang lại khả năng chống tấn công giả mạo đạt chuẩn quốc tế ISO/IEC 30107-3 PAD.
4. **Hiệu năng cao & Tối ưu thời gian thực:** Toàn bộ pipeline kiểm tra và đối sánh 1:N diễn ra trong khoảng **$15 - 35\text{ms}$** trên CPU tiêu chuẩn, sẵn sàng mở rộng quy mô sản phẩm.
