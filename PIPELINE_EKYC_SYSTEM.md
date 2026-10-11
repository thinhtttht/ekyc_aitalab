# 🚀 Kiến Trúc Pipeline Hệ Thống eKYC: Đăng Ký (Enrollment) & Xác Thực (Verification)

Tài liệu kỹ thuật mô tả chi tiết toàn bộ luồng xử lý dữ liệu (Data Pipeline), giải thuật phân tích hình ảnh, đánh giá chất lượng (FQA), phòng thủ chống giả mạo (PAD), trích xuất vector đặc trưng ArcFace 512 chiều và cơ chế đối sánh sinh trắc học trong hệ thống Smart-eKYC.

---

## 📌 1. Tổng Quan Kiến Trúc Đa Tầng (High-Level Architecture)

```mermaid
flowchart TD
    classDef clientStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef ingestStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef fqaStyle fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6;
    classDef bioStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#15803d;
    classDef storageStyle fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#c2410c;

    subgraph CLIENT["📱 Lớp Giao Diện Khách Hàng (Web Frontend)"]
        UI_CAM["Webcam Feed<br/>(HTML5 Canvas 4:5 & 16:9)"]:::clientStyle
        UI_GUIDE["Khung Oval Hướng Dẫn<br/>& Phản Hồi Trực Quan"]:::clientStyle
        UI_FLASH["Màn Phủ Quang Học Đổi Màu<br/>(Color Flashing Layer)"]:::clientStyle
    end

    subgraph INGESTION["⚡ Tầng Tiếp Nhận & Kiểm Định Phần Cứng"]
        CAM_CHK["Kiểm Tra Camera:<br/>Độ phân giải, FPS, Nhiễu, Treo"]:::ingestStyle
        DECODE["Giải mã Base64 -> Ma trận OpenCV BGR"]:::ingestStyle
    end

    subgraph FQA_PAD["🛡️ Tầng Đánh Giá Chất Lượng (FQA) & Chống Giả Mạo (PAD)"]
        FACEMESH["MediaPipe FaceMesh<br/>(468/478 Điểm mốc 3D)"]:::fqaStyle
        FQA["Bộ Tiêu Chí FQA 7 Trụ Cột:<br/>Cự ly, Lọt Oval, Ánh sáng, Độ nét, Ngũ quan"]:::fqaStyle
        ACTIVE_LIV["Liveness Chủ Động:<br/>Quay đầu 3D (>= 20°) & Phóng to Oval (>= 25%)"]:::fqaStyle
        FLASH_PAD["Active Optical PAD:<br/>Tương quan Pearson phổ màu da mặt"]:::fqaStyle
        MINIFAS["Passive PAD: MiniFASNet v2<br/>(Chống ảnh in 2D & Replay Attack)"]:::fqaStyle
    end

    subgraph BIOMETRICS["🧠 Tầng Sinh Trắc Học Học Sâu (Deep Biometrics)"]
        ALIGN["Face Alignment 5 Mốc Chuẩn<br/>(Mắt, Mũi, Khóe miệng) -> 112x112"]:::bioStyle
        ARCFACE["InsightFace ArcFace ResNet-50<br/>(WebFace600K Backbone)"]:::bioStyle
        L2NORM["Chuẩn Hóa L2 Norm -> Vector 512D<br/>trên Mặt cầu S⁵¹¹"]:::bioStyle
    end

    subgraph STORAGE_MATCH["🗄️ Tầng Lưu Trữ & Đối Sánh (Storage & Matching)"]
        SQLITE[("CSDL SQLite:<br/>Vector BLOB + HD Snapshot + Metadata")]:::storageStyle
        MATCH_ENGINE["Biometric Matcher:<br/>Cosine Similarity & Khoảng cách Euclid"]:::storageStyle
    end

    UI_CAM --> DECODE --> CAM_CHK
    CAM_CHK --> FACEMESH
    FACEMESH --> FQA
    FQA --> ACTIVE_LIV --> FLASH_PAD --> MINIFAS
    MINIFAS --> ALIGN --> ARCFACE --> L2NORM
    L2NORM -->|Quy trình Đăng ký| SQLITE
    L2NORM -->|Quy trình Xác thực| MATCH_ENGINE
    SQLITE -.->|Nạp hồ sơ| MATCH_ENGINE

    style CLIENT fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,stroke-dasharray: 5 5;
    style INGESTION fill:#fffbeb,stroke:#d97706,stroke-width:2px,stroke-dasharray: 5 5;
    style FQA_PAD fill:#faf5ff,stroke:#7c3aed,stroke-width:2px,stroke-dasharray: 5 5;
    style BIOMETRICS fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,stroke-dasharray: 5 5;
    style STORAGE_MATCH fill:#fff7ed,stroke:#ea580c,stroke-width:2px,stroke-dasharray: 5 5;
```

---

## 📝 2. Pipeline Chi Tiết: Luồng Đăng Ký (Enrollment Pipeline)

Quy trình đăng ký khuôn mặt diễn ra qua **5 giai đoạn nghiêm ngặt liên hoàn (Finite State Machine - FSM)** nhằm loại bỏ hoàn toàn gian lận danh tính và đảm bảo chỉ có dữ liệu sinh trắc học đạt chuẩn tối đa mới được ghi vào hệ thống:

### 🌈 Lưu Đồ Trạng Thái FSM Luồng Đăng Ký (Enrollment State Machine Flowchart)

```mermaid
flowchart TD
    classDef startStyle fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef s0Style fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e,font-weight:bold;
    classDef s1Style fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1,font-weight:bold;
    classDef s2Style fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-weight:bold;
    classDef s3Style fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e40af,font-weight:bold;
    classDef s4Style fill:#ffe4e6,stroke:#e11d48,stroke-width:2px,color:#9f1239,font-weight:bold;
    classDef s5Style fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#065f46,font-weight:bold;
    classDef decisionStyle fill:#ffffff,stroke:#6366f1,stroke-width:2px,color:#312e81,font-weight:bold;
    classDef rejectStyle fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b,font-weight:bold;
    classDef successStyle fill:#16a34a,stroke:#15803d,stroke-width:2.5px,color:#ffffff,font-weight:bold;

    START(["🚀 Khởi Động Đăng Ký Khuôn Mặt (Enrollment Start)"]):::startStyle --> S0_INIT

    subgraph S0["🟡 GIAI ĐOẠN 0: KIỂM ĐỊNH PHẦN CỨNG CAMERA"]
        S0_INIT["📷 Khởi tạo Camera Feed (480x600 Canvas)"]:::s0Style
        S0_CHECK{"Phần cứng & đường truyền đạt chuẩn?<br/>• FPS >= 25<br/>• Noise < 18.0 (Immerkær)<br/>• Độ sáng Luma: 45 - 215"}:::decisionStyle
        S0_INIT --> S0_CHECK
        S0_CHECK -->|"❌ Không đạt"| S0_WARN["⚠️ Cảnh báo UI: Webcam mờ/nhiễu hoặc ngược sáng"]:::rejectStyle
        S0_WARN --> S0_INIT
    end

    S0_CHECK -->|"✅ Đạt chuẩn phần cứng"| S1_INIT

    subgraph S1["🔵 GIAI ĐOẠN 1: CANH KHUNG OVAL & 7 TRỤ CỘT FQA"]
        S1_INIT["🎯 Hiển thị Khung Oval Hướng dẫn (web/app.js)"]:::s1Style
        S1_AI["🛡️ Phân tích MediaPipe FaceMesh (468 3D Points)"]:::s1Style
        S1_FQA{"Kiểm tra 7 tiêu chí FQA nghiêm ngặt:<br/>1. Lọt trọn Oval: d <= 1.0<br/>2. Tỷ lệ diện tích: 40% <= Scale <= 85%<br/>3. Góc chính diện: |Yaw|, |Pitch| <= 12°<br/>4. Độ nét Sobel Edge >= 15.0<br/>5. Đủ 5 vùng ngũ quan chính"}:::decisionStyle
        S1_STABLE{"Tích lũy liên tiếp<br/>đủ 6 frames chuẩn FQA?"}:::decisionStyle

        S1_INIT --> S1_AI --> S1_FQA
        S1_FQA -->|"❌ Sai tư thế / Lệch oval"| S1_HINT["💡 Phản hồi UI tức thời:<br/>'Vui lòng nhìn thẳng' / 'Tiến gần hơn'"]:::rejectStyle
        S1_HINT --> S1_INIT
        S1_FQA -->|"✅ Khung hình đạt"| S1_STABLE
        S1_STABLE -->|"Chưa đủ 6 frames"| S1_AI
    end

    S1_STABLE -->|"✅ Đạt 6/6 frames chuẩn"| S2_INIT

    subgraph S2["🟣 GIAI ĐOẠN 2: THỬ THÁCH 3D HEAD TURN (ACTIVE LIVENESS)"]
        S2_INIT["🔄 Phát sinh Thử thách Ngẫu nhiên: Quay đầu 3D<br/>(Random Challenge: Trái hoặc Phải)"]:::s2Style
        S2_CHECK{"Góc quay đầu Yaw >= 20°?<br/>(Head Pose SolvePnP z-depth)"}:::decisionStyle
        S2_RETURN["🔙 Nhận diện thành công góc quay!<br/>Yêu cầu đưa khuôn mặt về chính diện (|Yaw| <= 10°)"]:::s2Style
        S2_CENTER{"Đã quay về chính diện?"}:::decisionStyle

        S2_INIT --> S2_CHECK
        S2_CHECK -->|"Hết thời gian (Timeout)"| S2_TIMEOUT["⏳ Timeout: Reset về bắt đầu Stage 2"]:::rejectStyle
        S2_TIMEOUT --> S2_INIT
        S2_CHECK -->|"✅ Đạt góc quay"| S2_RETURN --> S2_CENTER
        S2_CENTER -->|"Chưa về giữa"| S2_RETURN
    end

    S2_CENTER -->|"✅ Hoàn thành quay đầu 3D"| S3_INIT

    subgraph S3["🔷 GIAI ĐOẠN 3: DYNAMIC ZOOM (THỬ THÁCH TIẾN GẦN)"]
        S3_INIT["🔍 Kích hoạt Dynamic Oval Zoom (Mở rộng Oval 1.25x)"]:::s3Style
        S3_CHECK{"Khuôn mặt tiến sát camera?<br/>(Chiều cao mặt tăng trưởng >= 25%)"}:::decisionStyle
        S3_INIT --> S3_CHECK
        S3_CHECK -->|"Đứng yên / Không chuyển động"| S3_HINT["💡 Gợi ý UI: 'Hãy tiến lại gần camera hơn'"]:::rejectStyle
        S3_HINT --> S3_INIT
    end

    S3_CHECK -->|"✅ Đạt cự ly cận cảnh"| S4_INIT

    subgraph S4["🔴 GIAI ĐOẠN 4: ACTIVE OPTICAL FLASHING & PASSIVE PAD"]
        S4_INIT["🌈 Kích hoạt Chuỗi Đổi Màu Viền Màn Hình<br/>(330ms/màu: Đỏ 🔴 ➔ Xanh Lá 🟢 ➔ Vàng 🟡)"]:::s4Style
        S4_PEARSON{"Kiểm định Phản quang Sinh học:<br/>• Tương quan Pearson r >= 0.10<br/>• Biên độ phản xạ da mặt >= 1.0"}:::decisionStyle
        S4_MINIFAS{"MiniFASNet v2 Passive PAD:<br/>Xác suất Real >= 65%?"}:::decisionStyle

        S4_INIT --> S4_PEARSON
        S4_PEARSON -->|"❌ Ánh sáng đồng nhất / Màn hình điện thoại"| S4_REJECT["🚫 TỪ CHỐI GIAN LẬN:<br/>Phát hiện màn hình hoặc ảnh in giả mạo!"]:::rejectStyle
        S4_PEARSON -->|"✅ Da người thật hấp thụ quang phổ"| S4_MINIFAS
        S4_MINIFAS -->|"❌ Giả mạo 2D"| S4_REJECT
    end

    S4_MINIFAS -->|"✅ Vượt qua tầng phòng thủ PAD quang học"| S5_INIT

    subgraph S5["🟢 GIAI ĐOẠN 5: CHỤP ẢNH HD & TRÍCH XUẤT ARCFACE 512D"]
        S5_INIT["📸 Kích hoạt Chụp Snapshot HD (Độ phân giải >= 720p)"]:::s5Style
        S5_ALIGN["📐 Chuẩn hóa hình học 5 mốc chuẩn (Affine Warp 112x112)"]:::s5Style
        S5_ARCFACE["🧠 InsightFace ArcFace ResNet-50:<br/>Trích xuất 512D Biometric Embeddings"]:::s5Style
        S5_NORM["⚙️ Chuẩn hóa L2 Norm: ||e||₂ = 1.0 trên mặt cầu S⁵¹¹"]:::s5Style
        S5_DB[("🗄️ Lưu Trữ SQLite Database:<br/>ID: USR-XXXXXXXX + Tên + 512D BLOB + Ảnh HD")]:::s5Style

        S5_INIT --> S5_ALIGN --> S5_ARCFACE --> S5_NORM --> S5_DB
    end

    S5_DB --> SUCCESS(["🎉 ĐĂNG KÝ HỒ SƠ THÀNH CÔNG!<br/>Hệ thống sẵn sàng cho Xác Thực Sinh Trắc"]):::successStyle

    style S0 fill:#fffbeb,stroke:#d97706,stroke-width:2px,stroke-dasharray: 4 4;
    style S1 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,stroke-dasharray: 4 4;
    style S2 fill:#faf5ff,stroke:#7c3aed,stroke-width:2px,stroke-dasharray: 4 4;
    style S3 fill:#eff6ff,stroke:#2563eb,stroke-width:2px,stroke-dasharray: 4 4;
    style S4 fill:#fff1f2,stroke:#e11d48,stroke-width:2px,stroke-dasharray: 4 4;
    style S5 fill:#f0fdf4,stroke:#059669,stroke-width:2px,stroke-dasharray: 4 4;
```

---

### 📡 Sơ Đồ Tuần Tự Trao Đổi Dữ Liệu (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 Người dùng (Webcam)
    participant FE as 💻 Frontend (web/app.js)
    participant BE as ⚙️ Backend API (backend/app.py)
    participant FQA as 🛡️ Face Analyzer (FQA & PAD)
    participant Model as 🧠 ArcFace & MiniFASNet
    participant DB as 🗄️ SQLite Database

    rect rgb(254, 243, 199)
    Note over User, FE: 🟡 GIAI ĐOẠN 0: KIỂM ĐỊNH CAMERA & MÔI TRƯỜNG (Vàng Nhạt)
    FE->>BE: Gửi chỉ số phần cứng (FPS Canvas, Độ sáng trung bình, Noise)
    BE-->>FE: Camera ổn định -> Cấp phép mở khung Oval
    end

    rect rgb(224, 242, 254)
    Note over User, FE: 🔵 GIAI ĐOẠN 1: KHUNG OVAL & 7 TIÊU CHÍ FQA (Xanh Biển)
    FE->>BE: POST /api/enroll/frame (Ảnh 480x600, Frame ID)
    BE->>FQA: Phân tích MediaPipe FaceMesh & 7 trụ cột FQA
    FQA-->>BE: FaceResult (scale_ratio, oval_dist, brightness, sharpness, parts)
    BE-->>FE: Phản hồi hướng dẫn (Đạt 6 frames liên tiếp -> Mở Stage 2)
    end

    rect rgb(237, 233, 254)
    Note over User, FE: 🟣 GIAI ĐOẠN 2: ACTIVE 3D HEAD TURN LIVENESS (Tím Nhạt)
    FE->>BE: Frame quay đầu (Góc Yaw >= 20° Trái/Phải)
    BE->>FQA: Head Pose Estimation (PnP z-depth)
    FQA-->>BE: Đạt góc quay -> Yêu cầu quay lại chính diện (Yaw <= 10°)
    BE-->>FE: Hoàn thành thử thách quay đầu -> Mở Stage 3
    end

    rect rgb(219, 234, 254)
    Note over User, FE: 🔷 GIAI ĐOẠN 3: DYNAMIC ZOOM (TIẾN GẦN CAMERA) (Xanh Chàm)
    FE->>BE: Frame phóng to (Khung Oval tự mở rộng)
    BE->>FQA: Chiều cao khuôn mặt tăng >= 25% (face_h growth)
    BE-->>FE: Đạt cự ly cận cảnh -> Mở Stage 4
    end

    rect rgb(255, 228, 230)
    Note over User, FE: 🔴 GIAI ĐOẠN 4: ACTIVE OPTICAL FLASHING & PASSIVE PAD (Hồng Cam)
    BE->>FE: Sinh Token Thách Thức (Chuỗi màu: Đỏ -> Xanh Lá -> Vàng)
    FE->>User: Đổi màu viền màn hình (330ms / bước màu)
    FE->>BE: Gửi frame kèm chỉ số màu phản xạ trên da mặt
    BE->>FQA: Tính hệ số tương quan Pearson r >= 0.10 & Biên độ >= 1.0
    BE->>Model: MiniFASNet v2 kiểm tra (Real Prob >= 65%)
    BE-->>FE: Thách thức quang học hợp lệ! Yêu cầu chụp HD
    end

    rect rgb(209, 250, 229)
    Note over User, FE: 🟢 GIAI ĐOẠN 5: CHỤP ẢNH HD & TRÍCH XUẤT ARCFACE 512D (Xanh Ngọc)
    FE->>BE: POST /api/users/enroll (HD Snapshot 720p, Full Name)
    BE->>Model: Align 5 mốc chuẩn -> Trích xuất ArcFace 512D -> L2 Norm
    BE->>DB: Lưu User ID (USR-XXXXXXXX), Tên, Vector 512D (BLOB), Snapshot
    DB-->>BE: Ghi đĩa thành công
    BE-->>FE: Thông báo đăng ký thành công -> Mở Modal Xác thực
    end
```

---

### 📊 Bảng Tiêu Chuẩn Kỹ Thuật 5 Giai Đoạn Đăng Ký

| Giai đoạn | Tên kỹ thuật | Giải thuật & Công nghệ cốt lõi | Điều kiện vượt qua (Pass Criteria) |
| :---: | :--- | :--- | :--- |
| <span style="background-color:#fef3c7;color:#92400e;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🟡 Stage 0</span> | **Camera Hardware Quality** | Immerkær Noise Estimation, Luma Variance | Độ phân giải $\ge 640\times 480$, FPS đo $\ge 25$, Độ sáng $45-215$, Không đóng băng khung hình. |
| <span style="background-color:#e0f2fe;color:#0369a1;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🔵 Stage 1</span> | **Oval Framing & 7 FQA Pillars** | MediaPipe FaceMesh (468 pts), Sobel Edge, Convex Hull | Tỷ lệ Oval $0.40 \le \text{scale} \le 0.85$, Lọt Oval $d \le 1.0$, Góc thẳng $\|\text{Yaw}\| \le 12^\circ$, Nét $\ge 15.0$, Đạt liên tiếp 6 frames. |
| <span style="background-color:#ede9fe;color:#5b21b6;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🟣 Stage 2</span> | **3D Active Turn Liveness** | Head Pose Estimation (SolvePnP trên 3D Landmarks) | Quay đầu sang trái/phải đạt $\|\text{Yaw}\| \ge 20^\circ$, sau đó đưa về chính diện $\le 10^\circ$. |
| <span style="background-color:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🔷 Stage 3</span> | **Dynamic Oval Zoom Challenge** | Dynamic Ellipse Expansion, Bounding Box Growth | Chiều cao khuôn mặt tăng trưởng $\ge 25\%$ so với giai đoạn 1, mặt nằm trọn trong Oval phóng to. |
| <span style="background-color:#ffe4e6;color:#9f1239;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🔴 Stage 4</span> | **Optical Color Flashing PAD** | Pearson Correlation Analysis, MiniFASNet v2 ONNX | Hệ số tương quan $r \ge 0.10$, Biên độ $\ge 1.0$, MiniFASNet Real Probability $\ge 65\%$. |
| <span style="background-color:#d1fae5;color:#065f46;padding:3px 10px;border-radius:6px;font-weight:700;display:inline-block;">🟢 Stage 5</span> | **HD Biometric Feature Extractor** | Affine Warp 112x112, ResNet-50 ArcFace ONNX | Vector 512 chiều, chuẩn hóa $\|\mathbf{e}\|_2 = 1.0$, lưu SQLite kèm User ID tự sinh `USR-XXXXXXXX`. |

> [!IMPORTANT]
> **Cơ Chế Khóa Trạng Thái Chống Bỏ Bước (FSM State Guarding)**: Bất kỳ giai đoạn nào vi phạm (lệch khung oval, chuyển động đột ngột, ánh sáng không phản quang sinh học hoặc giả mạo màn hình) hệ thống sẽ giữ nguyên trạng thái hoặc kích hoạt thử thách lại. Tuyệt đối không cho phép chụp ảnh Snapshot hay trích xuất vector ArcFace nếu chưa hoàn tất toàn bộ chuỗi bảo mật 5 giai đoạn.

---

## 🔍 3. Pipeline Chi Tiết: Luồng Xác Thực (Verification Pipeline 1:1 & 1:N)

Xác thực sinh trắc học được thiết kế theo tiêu chuẩn an toàn cấp ngân hàng (Zero-Trust Pipeline), áp dụng bộ lọc FQA tĩnh độc lập để **ngăn chặn tuyệt đối hiện tượng nhận diện bóng ma hoặc đoán bừa**:

```mermaid
flowchart TD
    classDef startStyle fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef decisionStyle fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e40af,font-weight:bold;
    classDef fqaRejectStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309,font-weight:bold;
    classDef padRejectStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,font-weight:bold;
    classDef aiStyle fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,color:#5b21b6,font-weight:bold;
    classDef successStyle fill:#d1fae5,stroke:#059669,stroke-width:2.5px,color:#065f46,font-weight:bold;
    classDef mismatchStyle fill:#ffe4e6,stroke:#e11d48,stroke-width:2px,color:#9f1239,font-weight:bold;

    START(["🚀 Bắt đầu Xác thực (Nhấn nút hoặc Auto-scan)"]):::startStyle --> CAPTURE["📷 Chụp Frame Trực Tiếp từ Webcam (640x480 Canvas)"]
    CAPTURE --> API["🌐 Gửi POST /api/verify/face (Base64, Mode, Threshold)"]

    subgraph STEP1["BƯỚC 1: BỘ DÒ TĨNH & KIỂM ĐỊNH HIỆN DIỆN (PRESENCE CHECK)"]
        API --> STATIC_MESH["MediaPipe FaceMesh<br/>(static_image_mode = TRUE)"]:::aiStyle
        STATIC_MESH --> CHK_COUNT{"Số lượng mặt phát hiện?"}:::decisionStyle
        CHK_COUNT -->|"0 khuôn mặt"| ERR_NO_FACE["❌ Từ chối: NO_FACE<br/>(Không phát hiện khuôn mặt)"]:::fqaRejectStyle
        CHK_COUNT -->|"> 1 khuôn mặt"| ERR_MULTI["❌ Từ chối: MULTIPLE_FACES<br/>(Nhiều người trước camera)"]:::fqaRejectStyle
        CHK_COUNT -->|"Đúng 1 mặt"| CHK_AREA{"Diện tích mặt >= 2.5% khung hình?"}:::decisionStyle
        CHK_AREA -->|"Không (Vết nhiễu/ảnh xa)"| ERR_TINY["❌ Từ chối: NO_FACE<br/>(Loại bỏ nhiễu hậu cảnh)"]:::fqaRejectStyle
    end

    subgraph STEP2["BƯỚC 2: HÀNG RÀO KIỂM SOÁT FQA 7 TIÊU CHÍ (STRICT FQA)"]
        CHK_AREA -->|"Đạt diện tích"| FQA_INV{"Mặt bị lộn ngược?"}:::decisionStyle
        FQA_INV -->|"Có"| ERR_POSE["⚠️ Từ chối: INVALID_POSE"]:::fqaRejectStyle
        FQA_INV -->|"Không"| FQA_OCC{"Bị che khuất?<br/>(Khẩu trang, Kính râm, Tay, Mất mắt/mũi)"}:::decisionStyle
        FQA_OCC -->|"Có"| ERR_OCC["⚠️ Từ chối: FACE_OCCLUDED"]:::fqaRejectStyle
        FQA_OCC -->|"Không"| FQA_OVAL{"Lọt khung Oval & Cự ly?<br/>(Oval dist <= 1.30, Scale 0.30 - 0.95)"}:::decisionStyle
        FQA_OVAL -->|"Lệch khung"| ERR_OVAL["⚠️ Từ chối: INVALID_FRAMING"]:::fqaRejectStyle
        FQA_OVAL -->|"Đạt"| FQA_HEAD{"Nhìn thẳng?<br/>(|Yaw| <= 30°, |Pitch| <= 25°)"}:::decisionStyle
        FQA_HEAD -->|"Nghiêng quá mức"| ERR_HEAD["⚠️ Từ chối: INVALID_POSE"]:::fqaRejectStyle
        FQA_HEAD -->|"Thẳng"| FQA_SHARP{"Độ nét Sharpness >= 15.0?"}:::decisionStyle
        FQA_SHARP -->|"Mờ / Rung lắc"| ERR_BLUR["⚠️ Từ chối: IMAGE_BLURRY"]:::fqaRejectStyle
        FQA_SHARP -->|"Nét"| FQA_LIGHT{"Độ sáng trong khoảng 40 - 210?"}:::decisionStyle
        FQA_LIGHT -->|"Quá tối hoặc chói"| ERR_LIGHT["⚠️ Từ chối: POOR_LIGHTING"]:::fqaRejectStyle
    end

    subgraph STEP3["BƯỚC 3: PHÒNG THỦ CHỐNG GIẢ MẠO PASSIVE PAD"]
        FQA_LIGHT -->|"Đạt ánh sáng"| PAD_RUN["Chạy MiniFASNet v2 (Khớp BBox)"]:::aiStyle
        PAD_RUN --> PAD_CHK{"is_real == True?<br/>(Real prob >= 65%)"}:::decisionStyle
        PAD_CHK -->|"Giả mạo (In / Video)"| ERR_PAD["🚨 Từ chối: SPOOF_REJECTED<br/>(Cảnh báo gian lận PAD)"]:::padRejectStyle
    end

    subgraph STEP4["BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG ARCFACE"]
        PAD_CHK -->|"Người thật"| EXTRACT_KPS["Lấy 5 mốc chuẩn ArcFace<br/>[Mắt T, Mắt P, Mũi, Mép T, Mép P]"]:::aiStyle
        EXTRACT_KPS --> WARP_AFFINE["Biến đổi Affine tương tự (norm_crop) -> 112x112"]:::aiStyle
        WARP_AFFINE --> RUN_ARCFACE["Suy luận ResNet-50 ArcFace (ONNX Runtime)"]:::aiStyle
        RUN_ARCFACE --> NORM_512["Chuẩn hóa L2 Norm: ||e_query|| = 1.0"]:::aiStyle
    end

    subgraph STEP5["BƯỚC 5: ĐỐI SÁNH SINH TRẮC HỌC (BIOMETRIC MATCHING)"]
        NORM_512 --> CHECK_MODE{"Chế độ Đối Sánh?"}:::decisionStyle

        CHECK_MODE -->|"1:1 (So khớp đích danh)"| MATCH_11["Lấy Vector của Target User ID trong CSDL"]
        MATCH_11 --> CALC_11["Tính Cosine Sim = e_query • e_target<br/>Khoảng cách Euclid d = sqrt(2(1 - Sim))"]
        CALC_11 --> DECIDE_11{"Cosine Sim >= Threshold (0.55)?"}:::decisionStyle
        DECIDE_11 -->|"Đạt"| RES_11_PASS["✅ Xác thực Thành Công<br/>(MATCH_SUCCESS)"]:::successStyle
        DECIDE_11 -->|"Không đạt"| RES_11_FAIL["❌ Không Khớp Danh Tính<br/>(MISMATCH)"]:::mismatchStyle

        CHECK_MODE -->|"1:N (Nhận diện toàn CSDL)"| LOAD_ALL["Quét toàn bộ Users trong SQLite"]
        LOAD_ALL --> CALC_ALL["Tính Dot Product e_query • e_i với mọi User"]
        CALC_ALL --> SORT_TOP["Sắp xếp giảm dần -> Lấy Top-1 Candidate"]
        SORT_TOP --> DECIDE_1N{"Top-1 Sim >= Threshold (0.55)?"}:::decisionStyle
        DECIDE_1N -->|"Đạt"| RES_1N_PASS["✅ Khớp Danh Tính Top-1<br/>(MATCH_SUCCESS)"]:::successStyle
        DECIDE_1N -->|"Không đạt"| RES_1N_FAIL["❌ Không Khớp Danh Tính Nào<br/>(MISMATCH)"]:::mismatchStyle
    end

    subgraph STEP6["BƯỚC 6: PHẢN HỒI GIAO DIỆN KHÁCH HÀNG (UI RENDERING)"]
        RES_11_PASS --> UI_PASS["🟢 Hiển thị Card User xanh, phát âm thanh 'Ting', Leaderboard xanh"]:::successStyle
        RES_1N_PASS --> UI_PASS
        RES_11_FAIL --> UI_FAIL["🔴 Hiển thị Banner Đỏ 'Không trùng khớp', Leaderboard xám trung tính"]:::mismatchStyle
        RES_1N_FAIL --> UI_FAIL
        ERR_NO_FACE --> UI_WARN["🟡 Hiển thị Banner Cảnh Báo, ẨN HOÀN TOÀN User Card & Leaderboard"]:::fqaRejectStyle
        ERR_MULTI --> UI_WARN
        ERR_TINY --> UI_WARN
        ERR_POSE --> UI_WARN
        ERR_OCC --> UI_WARN
        ERR_OVAL --> UI_WARN
        ERR_HEAD --> UI_WARN
        ERR_BLUR --> UI_WARN
        ERR_LIGHT --> UI_WARN
        ERR_PAD --> UI_DANGER["🚨 Hiển thị Banner Báo Động Đỏ PAD Attack, Ẩn User Card"]:::padRejectStyle
    end

    style STEP1 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
    style STEP2 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
    style STEP3 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
    style STEP4 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
    style STEP5 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
    style STEP6 fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px;
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
