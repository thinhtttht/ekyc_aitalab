# HỆ THỐNG PIPELINE eKYC CHUẨN 3 TẦNG BẢO MẬT (DEFENSE IN DEPTH)
> **Kiến trúc hoàn chỉnh: Tầng 1 (Tương tác & Color Flashing) -> Tầng 2 (MiniFASNet) -> Tầng 3 (ArcFace / InsightFace)**

---

## 1. SƠ ĐỒ GIAI ĐOẠN 1: ĐĂNG KÝ KHUÔN MẶT (ENROLLMENT)
> **Mục tiêu:** Kiểm tra kỹ lưỡng 3 tầng một lần duy nhất để lưu lại "vector 512 số sạch nhất" của người dùng vào `database.json`.

```mermaid
flowchart TD
    START(["Bắt đầu: Đăng ký khuôn mặt"]) --> OPEN_CAM["Khởi động Webcam"]

    subgraph TANG1["TẦNG 1: TƯƠNG TÁC VẬT LÝ & QUANG HỌC"]
        OPEN_CAM --> OVAL_CHECK["1. Khung Oval UI: Căn mặt vào giữa & đúng cự ly"]
        OVAL_CHECK --> HEAD_POSE["2. Thử thách 3D: Yêu cầu hơi quay đầu (Trái/Phải)"]
        HEAD_POSE --> FLASH["3. Color Flashing: Nhìn thẳng giữ yên 1s<br/>(Màn hình nháy 2 màu Random kiểm tra mao mạch da)"]
        FLASH --> PASS_TANG1{"Vượt qua Tầng 1?"}
        PASS_TANG1 -- "Không khớp phản xạ" --> REJECT1["TỪ CHỐI: Phát hiện giả mạo!"]
    end

    subgraph TANG2["TẦNG 2: THỊ GIÁC SÂU (PASSIVE ANTI-SPOOFING)"]
        PASS_TANG1 -- "Đạt chuẩn" --> MINIFAS["Đưa ảnh qua MiniFASNet<br/>(Soi vân Moiré màn hình & chất liệu ảnh in)"]
        MINIFAS --> PASS_TANG2{"MiniFASNet: Real hay Spoof?"}
        PASS_TANG2 -- "Spoof (Giả mạo)" --> REJECT2["TỪ CHỐI: Phát hiện ảnh/màn hình giả!"]
    end

    subgraph TANG3["TẦNG 3: TRÍCH XUẤT ĐẶC TRƯNG (INSIGHTFACE)"]
        PASS_TANG2 -- "Real (Người thật 100%)" --> ALIGN["Tự động căn chỉnh & Nắn thẳng mặt 112x112"]
        ALIGN --> ARCFACE["ArcFace trích xuất Vector đặc trưng (512 số)"]
        ARCFACE --> NORM["Chuẩn hóa Vector (L2 Normalization)"]
        NORM --> SAVE_DB[("Lưu vào CSDL: database.json<br/>{ 'user_01': { 'name': '...', 'vector': [512 số] } }")]
        SAVE_DB --> SUCCESS_ENROLL["Tắt Webcam & Báo Đăng ký thành công!"]
    end

    style TANG1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    style TANG2 fill:#fff8e1,stroke:#ffa000,stroke-width:2px;
    style TANG3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    style SUCCESS_ENROLL fill:#a5d6a7,stroke:#2e7d32,stroke-width:3px;
    style REJECT1 fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    style REJECT2 fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
```

---

## 2. SƠ ĐỒ GIAI ĐOẠN 2: XÁC THỰC CHUYỂN TIỀN (VERIFICATION)
> **Mục tiêu:** Xác thực cực nhanh (dưới 2 giây), người dùng chỉ cần giữ yên mặt trong khung Oval, hệ thống tự nháy màu và quét ngầm 3 tầng để duyệt lệnh.

```mermaid
flowchart TD
    START_TX(["Người dùng bấm: Xác thực chuyển tiền"]) --> OPEN_CAM2["Bật Webcam"]

    subgraph V_TANG1["TẦNG 1: KHUNG OVAL & COLOR FLASHING (Siêu tốc)"]
        OPEN_CAM2 --> SHOW_OVAL["Hiển thị Khung Oval trên màn hình"]
        SHOW_OVAL --> FIT_OVAL{"Mặt lọt khít khung Oval<br/>(Viền Xanh Lá)?"}
        FIT_OVAL -- "Chưa chuẩn" --> WAIT["Nhắc người dùng giữ yên mặt vào form"]
        WAIT --> SHOW_OVAL

        FIT_OVAL -- "Đã khớp form" --> AUTO_FLASH["Tự động Color Flashing nháy 2 màu Random (0.6s)<br/>(Kiểm tra biến thiên sắc độ quang học trên da)"]
        AUTO_FLASH --> CHECK_SKIN{"Sắc độ da phản xạ đúng chuỗi màu?"}
        CHECK_SKIN -- "Không phản xạ" --> V_REJECT1["CẢNH BÁO GIẢ MẠO: Hủy bỏ giao dịch!"]
    end

    subgraph V_TANG2["TẦNG 2: QUÉT NGẦM MINIFASNET (15 mili-giây)"]
        CHECK_SKIN -- "Đúng người thật" --> V_MINI["MiniFASNet quét vi vân ảnh / màn hình"]
        V_MINI --> V_CHECK_SPOOF{"Kết quả MiniFASNet?"}
        V_CHECK_SPOOF -- "Phát hiện màn hình/ảnh in" --> V_REJECT2["CẢNH BÁO: Tấn công giả mạo (Spoof)!"]
    end

    subgraph V_TANG3["TẦNG 3: NHẬN DIỆN ARCFACE & DUYỆT LỆNH"]
        V_CHECK_SPOOF -- "Real (An toàn tuyệt đối)" --> V_ALIGN["InsightFace nắn thẳng mặt 112x112"]
        V_ALIGN --> V_EMBED["ArcFace trích xuất Vector_HienTai (512 số)"]
        V_EMBED --> LOAD_DB[("Đọc Vector_Goc từ database.json")]
        LOAD_DB --> COSINE["Tính độ tương đồng Cosine Similarity"]
        COSINE --> CHECK_THRESHOLD{"Cosine Score >= 0.65?"}

        CHECK_THRESHOLD -- "ĐÚNG (Ví dụ: Score = 0.84)" --> V_SUCCESS["XÁC THỰC THÀNH CÔNG!<br/>(Duyệt lệnh chuyển 50.000.000 VNĐ)"]
        CHECK_THRESHOLD -- "SAI (Ví dụ: Score = 0.31)" --> V_REJECT3["TỪ CHỐI GIAO DỊCH<br/>(Khuôn mặt không khớp chủ tài khoản!)"]
    end

    style V_TANG1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    style V_TANG2 fill:#fff8e1,stroke:#ffa000,stroke-width:2px;
    style V_TANG3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    style V_SUCCESS fill:#a5d6a7,stroke:#2e7d32,stroke-width:3px;
    style V_REJECT1 fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    style V_REJECT2 fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    style V_REJECT3 fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
```

---

## 3. BẢNG PHÂN CÔNG 3 TẦNG BẢO MẬT (DÙNG ĐỂ THUYẾT TRÌNH)

| Tầng | Tên Tầng Bảo Mật | Công Nghệ Sử Dụng | Nhiệm Vụ Cụ Thể | Loại Tấn Công Bị Chặn |
| :---: | :--- | :--- | :--- | :--- |
| **TẦNG 1** | **Tương tác & Quang học** | Khung Oval UI + Random Color Flashing | Kiểm tra góc quay mặt 3D và sự thay đổi màu sắc quang học mao mạch da dưới ánh sáng ngẫu nhiên | Chặn ảnh chụp 2D bất động, ảnh giấy không phản xạ da |
| **TẦNG 2** | **Thị giác sâu chống giả mạo** | Mô hình **MiniFASNet** (Silent-Face-Anti-Spoofing) | Soi các chi tiết vi mô: vân Moiré của màn hình điện thoại/iPad, độ chói bóng của mặt kính, kết cấu bề mặt | Chặn video quay lén phát lại trên điện thoại/máy tính bảng, ảnh in chất lượng cao |
| **TẦNG 3** | **Định danh sinh trắc học** | Mô hình **ArcFace** (thư viện **InsightFace**) | Nắn thẳng mặt $112 \times 112$, trích xuất 512 số đặc trưng và so sánh Cosine Similarity với ngưỡng $\ge 0.65$ | Chặn người lạ, chỉ cho phép đúng chủ tài khoản thực hiện giao dịch |
