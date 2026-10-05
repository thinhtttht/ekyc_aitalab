# PROJECT BLUEPRINT: ACTIVE OPTICAL COLOR FLASHING LIVENESS DETECTION

> **Tài liệu Kế hoạch & Mục tiêu Dự án (Plan & Goal)**  
> **Phiên bản:** 1.0.0  
> **Trạng thái:** Chờ phê duyệt (Pending Review)  
> **Chuẩn tuân thủ:** ASD-STE100, W3C WCAG 2.1 (Photosensitive Safety), ISO/IEC 30107-3 (Biometric Presentation Attack Detection)  
> **Tác giả:** Ban Kỹ Thuật (Mini Engineering Board)

---

## 1. TỔNG QUAN DỰ ÁN & VẤN ĐỀ NGHIỆP VỤ (EXECUTIVE SUMMARY)

### 1.1. Bối cảnh bài toán
Hệ thống eKYC hiện tại đã có hai lớp bảo vệ:
1. **Lớp vật lý/tương tác:** Hướng dẫn căn mặt vào khung Oval và kiểm tra góc quay đầu 3D (Head Pose Yaw).
2. **Lớp thụ động (Passive PAD):** Mô hình MiniFASNetV2 kiểm tra vi vân màn hình (Moiré pattern), độ sâu 3D (MediaPipe Z-depth) và viền thiết bị (Bezel detection).

Tuy nhiên, các phương thức tấn công tinh vi vẫn tiềm ẩn rủi ro:
- **Tấn công phát lại qua màn hình độ phân giải siêu cao (8K OLED Replay Attack):** Kẻ tấn công phát video người thật đã quay sẵn.
- **Tấn công mặt nạ 3D Silicon/Latex siêu thực:** Kẻ tấn công đeo mặt nạ có hình khối 3D chính xác.
- **Tấn công tiêm luồng video ảo (Virtual Camera / Video Injection):** Kẻ tấn công giả lập webcam và nạp luồng video giả mạo trực tiếp vào trình duyệt.

### 1.2. Giải pháp: Color Flashing (Active Optical Challenge-Response)
Color Flashing là cơ chế xác thực quang học chủ động:
- **Nguyên lý sinh học quang phổ (Photometric & Chrominance Skin Reflection):** Mô da người sống có tính chất phản xạ và tán xạ dưới bề mặt (subsurface scattering). Khi màn hình phát ra ánh sáng màu (ví dụ: Đỏ, Xanh lá, Xanh dương), các mao mạch máu và mô da phản xạ lại quang phổ tương ứng.
- **Nguyên lý cơ chế Thách thức - Phản hồi (Challenge-Response):** Máy chủ sinh ngẫu nhiên chuỗi màu sắc tại thời gian thực kèm mã xác thực (Session Token). Trình duyệt phát chuỗi màu này lên mặt người dùng. Hệ thống đo đạc sự biến thiên màu sắc trên da mặt đồng bộ theo thời gian. Màn hình giả mạo, ảnh in hoặc video ghi sẵn không thể đoán trước chuỗi màu ngẫu nhiên này.

---

## 2. MỤC TIÊU CỐT LÕI (GOALS) & PHI MỤC TIÊU (NON-GOALS)

### 2.1. Mục tiêu cốt lõi (Goals)
1. **Tạo cơ chế Challenge-Response ngẫu nhiên:**
   - Máy chủ sinh chuỗi 3 màu ngẫu nhiên từ tập màu cơ bản $\{ \text{Red}, \text{Green}, \text{Blue}, \text{Yellow}, \text{Cyan} \}$.
   - Chuỗi màu gắn liền với phiên làm việc (Session ID) và có thời gian sống (TTL = 10 giây).
2. **Đồng bộ hóa phát quang và thu nhận khung hình:**
   - Trình duyệt điều khiển lớp phủ màu (Flashing Overlay) hiển thị tuần tự từng màu.
   - Thời gian hiển thị mỗi màu: $350\text{ ms} - 400\text{ ms}$.
   - Tổng thời gian thực hiện kiểm tra quang học: $\le 1.2\text{ giây}$.
   - Trình duyệt đóng gói khung hình kèm siêu dữ liệu (Timestamp, Color Step Index) và gửi về máy chủ.
3. **Thuật toán phân tích phản xạ quang học đa vùng da (Facial Tissue Photometric Analysis):**
   - Trích xuất 3 vùng ROI (Region of Interest) da mặt: Trán (Forehead), Gò má trái (Left Cheek), Gò má phải (Right Cheek).
   - Đo lường độ dịch chuyển kênh màu ($\Delta R, \Delta G, \Delta B$) trong không gian màu RGB và Cr/Cb trong không gian màu YCrCb.
   - Tính toán hệ số tương quan quang học (Pearson Correlation Coefficient $\ge 0.70$) giữa màu phát ra và màu phản xạ trên da.
4. **Phát hiện màn hình và bề mặt nhân tạo:**
   - Phân biệt độ phản xạ của mô sống với màn hình phát sáng thứ cấp (LCD/OLED) và bề mặt giấy in tĩnh.
5. **Đảm bảo an toàn thị giác tuyệt đối:**
   - Tuân thủ tiêu chuẩn W3C WCAG 2.1 (Success Criterion 2.3.1) và ISO 9241-391.
   - Tần số chuyển màu $< 3\text{ Hz}$ để loại bỏ nguy cơ co giật do nhạy cảm ánh sáng (Photosensitive Seizure).
   - Sử dụng hiệu ứng chuyển màu mềm (Soft Gradient Overlay) với bán kính hở tại vùng mắt để người dùng không bị chói mắt.

### 2.2. Phi mục tiêu (Non-Goals)
1. **Không yêu cầu phần cứng chuyên dụng:**
   - Không sử dụng Camera hồng ngoại (IR / NIR).
   - Không sử dụng cảm biến độ sâu Structured Light hay LiDAR/ToF.
   - Hệ thống chỉ sử dụng Webcam RGB tiêu chuẩn (720p/1080p).
2. **Không thay thế các tầng kiểm tra hiện tại:**
   - Color Flashing không thay thế MediaPipe (FQA, Head Pose) hay MiniFASNetV2.
   - Color Flashing đóng vai trò là Lớp kiểm thử chủ động (Active Challenge Layer) bổ trợ cho Lớp kiểm thử thụ động (Passive Layer).
3. **Không xử lý nhận diện danh tính tại module này:**
   - Module chỉ trả về kết quả Liveness (PASS / FAIL).
   - Quá trình so khớp danh tính (ArcFace 512-d feature matching) được thực hiện ở tầng tiếp theo.

---

## 3. PHẢN BIỆN KỸ THUẬT & QUẢN TRỊ RỦI RO (ENGINEERING PUSH-BACK)

Ban Kỹ Thuật đưa ra 3 cảnh báo kỹ thuật bắt buộc phải giải quyết trong thiết kế:

### 3.1. Rủi ro 1: Hiện tượng Cân bằng trắng tự động của Webcam (Auto White Balance - AWB)
- **Vấn đề:** Khi màn hình phát ánh sáng màu mạnh, cảm biến webcam tự động điều chỉnh thuật toán cân bằng trắng (AWB) và phơi sáng tự động (Auto Exposure - AE) để bù trừ màu. Việc này có thể làm triệt tiêu tín hiệu phản xạ màu trên da mặt.
- **Giải pháp kỹ thuật:**
  1. Phân tích vi sai tương đối (Relative Differential Analysis): Đo lường tỷ số giữa biến thiên của vùng da mặt ($\Delta \text{Skin}$) và biến thiên của vùng nền tĩnh không đổi ($\Delta \text{Background}$).
  2. Thời gian chiếu mỗi màu khống chế ở mức $350\text{ ms}$: Thuật toán AWB của phần cứng webcam thông thường cần từ $500\text{ ms} - 1000\text{ ms}$ để hội tụ. Chúng tôi thu nhận khung hình ngay trong $150\text{ ms} - 250\text{ ms}$ đầu tiên trước khi AWB kịp triệt tiêu màu.

### 3.2. Rủi ro 2: Độ trễ mạng và Lệch pha khung hình (Frame-Color Desynchronization)
- **Vấn đề:** Nếu gửi từng frame qua mạng Internet, độ trễ mạng (Network Jitter) làm sai lệch thứ tự khung hình với bước nháy màu tương ứng.
- **Giải pháp kỹ thuật:**
  - Client-driven Synchronization: Trình duyệt đóng dấu thời gian (Microsecond Timestamp) và gắn nhãn bước màu (`color_index: 0, 1, 2`) trực tiếp vào gói dữ liệu trước khi truyền về backend.
  - Backend đối soát nhãn khung hình với chuỗi Token đã cấp cho phiên làm việc đó.

### 3.3. Rủi ro 3: Trải nghiệm người dùng và An toàn thị giác (Physical UX Safety)
- **Vấn đề:** Ánh sáng nhấp nháy toàn màn hình ở cự ly gần gây khó chịu cho mắt và có thể kích phát cơn động kinh ở người có tiền sử nhạy cảm thị giác.
- **Giải pháp kỹ thuật:**
  - Thiết kế vòng cung phát sáng xung quanh khung Oval (Radial Gradient Glow Border) thay vì chớp toàn bộ màn hình màu trắng chói.
  - Vùng trung tâm khuôn mặt và mắt giữ độ mờ nhẹ (alpha channel = 0.55 - 0.70) đủ để hắt sáng lên da nhưng không gây chói trực diện vào con ngươi.

---

## 4. KẾ HOẠCH TỔNG THỂ (MASTER PLAN & MILESTONES)

Dự án triển khai theo quy trình nghiêm ngặt 6 Phase:

```
[Phase 0: Project Blueprint] (Hiện tại)
        │
        ▼ (Duyệt Blueprint)
[Phase 1: PRD - Product Requirements & Physical UX]
        │
        ▼ (Duyệt PRD)
[Phase 2: FSD - Functional Specification & Protocol]
        │
        ▼ (Duyệt FSD)
[Phase 3: TDD/RFC - Technical Design & Mathematics]
        │
        ▼ (Duyệt TDD)
[Phase 4: RTM - Requirements Traceability Matrix]
        │
        ▼ (Duyệt RTM)
[Phase 5: Micro-Implementation & Test Từng Bước]
        ├── Module 5.1: Session Challenge Protocol (Backend)
        ├── Module 5.2: Flashing Light Engine (Frontend UI)
        ├── Module 5.3: Optical Reflection Analyzer (Backend)
        └── Module 5.4: Pipeline Integration & End-to-End Tests
```

### Chi tiết các Milestones:
- **Milestone 0 (Phase 0):** Hoàn thành `PROJECT_BLUEPRINT.md` (Mục tiêu, Phi mục tiêu, Kiến trúc tổng thể, Rủi ro kỹ thuật).
- **Milestone 1 (Phase 1):** Soạn thảo `PRD_COLOR_FLASHING.md` (Chân dung người dùng, State Machine giao diện, Thông báo ASD-STE100, Tiêu chuẩn an toàn thị giác WCAG).
- **Milestone 2 (Phase 2):** Soạn thảo `FSD_COLOR_FLASHING.md` (Đặc tả chi tiết hàm, Cấu trúc dữ liệu Request/Response, Điều kiện chuyển trạng thái, Xử lý ngoại lệ).
- **Milestone 3 (Phase 3):** Soạn thảo `TDD_COLOR_FLASHING.md` (Công thức toán học Pearson Correlation, Giải thuật bù trừ AWB, Phân tích ROI MediaPipe, Thiết kế API FastAPI).
- **Milestone 4 (Phase 4):** Xây dựng bảng ma trận truy vết `RTM_COLOR_FLASHING.md` đảm bảo 100% yêu cầu được ánh xạ tới code và test case.
- **Milestone 5 (Phase 5):** Triển khai mã nguồn chia nhỏ từng module kèm Unit Test:
  - *Bước 5.1:* Xây dựng Backend Challenge Generator (`backend/color_challenge.py`).
  - *Bước 5.2:* Xây dựng Frontend Flashing Controller & Safe Overlay UI (`web/app.js`, `web/styles.css`).
  - *Bước 5.3:* Xây dựng Backend Optical Reflection Analyzer (`backend/optical_analyzer.py`).
  - *Bước 5.4:* Tích hợp vào Enrollment State Machine (`backend/enrollment.py`) và thực thi bộ kiểm thử tự động.

---

## 5. ĐỀ XUẤT CẤU TRÚC THƯ MỤC VÀ TÀI LIỆU DỰ ÁN

```
d:\EKYC\
├── backend/
│   ├── app.py                      # FastAPI App (Cập nhật endpoint challenge/verify)
│   ├── config.py                   # Bổ sung cấu hình ngưỡng Color Flashing
│   ├── color_challenge.py          # Module sinh chuỗi màu ngẫu nhiên & Token
│   ├── optical_analyzer.py         # Module giải thuật phân tích phản xạ quang phổ
│   ├── face_analyzer.py            # Trích xuất ROI trán và 2 má từ MediaPipe
│   ├── anti_spoofing.py            # MiniFASNet + 3D Depth + Moiré
│   ├── enrollment.py               # State Machine bổ sung bước FLASHING_CHECK
│   └── tests/
│       ├── test_color_challenge.py # Test module sinh màu & Token
│       ├── test_optical_analyzer.py# Test thuật toán phản xạ quang phổ
│       └── ...
├── web/
│   ├── index.html                  # Bổ sung phần tử Canvas/DOM Flashing Overlay
│   ├── styles.css                  # Bổ sung CSS Animation & Safe Glow Filter
│   └── app.js                      # Điều khiển đồng bộ nháy màu & gửi frame
├── docs/                           # Thư mục tài liệu kỹ thuật chuẩn mực
│   ├── PROJECT_BLUEPRINT.md        # Tài liệu Phase 0
│   ├── PRD_COLOR_FLASHING.md       # Tài liệu Phase 1
│   ├── FSD_COLOR_FLASHING.md       # Tài liệu Phase 2
│   ├── TDD_COLOR_FLASHING.md       # Tài liệu Phase 3
│   └── RTM_COLOR_FLASHING.md       # Tài liệu Phase 4
├── PIPELINE.md                     # Tài liệu tổng thể hệ thống 3 tầng
├── RFC_BIOMETRIC_PAD_SYSTEM.md     # RFC hệ thống PAD hiện tại
└── run.bat                         # Kịch bản khởi động 1-click
```

---

## 6. TIÊU CHÍ NGHIỆM THU PHASE 0 (ACCEPTANCE CRITERIA)
- [x] Xác định rõ ràng bài toán, giá trị bảo mật và cơ chế chống tấn công Replay/Mask của Color Flashing.
- [x] Định nghĩa ranh giới tường minh giữa Goals (Mục tiêu) và Non-Goals (Phi mục tiêu).
- [x] Phân tích và đưa ra giải pháp cho 3 thách thức kỹ thuật cốt lõi (AWB Drift, Frame Sync, Visual Safety).
- [x] Thiết lập lộ trình 6 Phase theo chuẩn Document-Driven Engineering.
- [x] Đề xuất cấu trúc file và module không làm gián đoạn mã nguồn hiện có.

---
*Tài liệu kết thúc tại đây. Ban Kỹ Thuật chờ ý kiến chỉ đạo từ Ban Giám Đốc.*
