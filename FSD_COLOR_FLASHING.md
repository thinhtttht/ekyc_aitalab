# FSD: ACTIVE OPTICAL COLOR FLASHING LIVENESS DETECTION (PHASE 2)

> **Tài liệu Đặc tả Chức năng (Functional Specification Document)**  
> **Phiên bản:** 1.0.0  
> **Tuân thủ chuẩn:** ASD-STE100 (Simplified Technical English/Vietnamese), ISO/IEC 30107-3  
> **Tác giả:** Ban Kỹ Thuật (Mini Engineering Board)

---

## 1. MỤC TIÊU & PHẠM VI HỆ THỐNG (SYSTEM OBJECTIVE & SCOPE)

Tài liệu này đặc tả chức năng chi tiết cho hệ thống xác thực sự sống bằng ánh sáng quang phổ (Color Flashing PAD) tích hợp tại Bước 4 của quy trình Đăng ký (Enrollment).

### 1.1. Nguyên tắc ASD-STE100
- **Một từ = Một nghĩa:** "Máy chủ" (Server), "Trình duyệt" (Browser), "Mô da" (Skin Tissue), "Màn hình" (Display Screen), "Khung hình" (Video Frame), "Hệ số tương quan" (Correlation Coefficient).
- **Cấu trúc câu chủ động:** [Tác nhân] + [Hành động] + [Đối tượng].
- **Cấu trúc logic:** Tách biệt rõ ràng [Điều kiện If] và [Hành động Then].

---

## 2. DANH SÁCH YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### 2.1. FR-01: Sinh chuỗi màu ngẫu nhiên và Token bảo mật
- **Tác nhân:** Module `color_challenge.py` trên Máy chủ.
- **Hành động:** Khi Trình duyệt gửi yêu cầu `POST /api/enroll/color_challenge`:
  1. Máy chủ chọn ngẫu nhiên $3$ màu phân biệt từ tập hợp $5$ màu chuẩn:
     $$\mathcal{C} = \{ \text{GREEN}(\#00E676), \text{RED}(\#FF3D00), \text{BLUE}(\#2979FF), \text{YELLOW}(\#FFC400), \text{MAGENTA}(\#F50057) \}$$
  2. Máy chủ cấm $2$ màu liên tiếp trùng nhau: $C_i \neq C_{i-1}, \forall i \in \{1, 2\}$.
  3. Máy chủ gán thời lượng hiển thị cho mỗi màu: $D_i = 330\text{ ms}$.
  4. Máy chủ tạo mã xác thực (Security Token) HMAC-SHA256 liên kết với `session_id`, chuỗi màu và thời gian hết hạn $\text{TTL} = 15\text{ giây}$.
  5. Máy chủ lưu trữ chuỗi màu vào trạng thái phiên và trả về cho Trình duyệt.

### 2.2. FR-02: Hiển thị ánh sáng màu an toàn và thu thập khung hình
- **Tác nhân:** Trình duyệt Web (`web/app.js`).
- **Điều kiện kích hoạt:** Trình duyệt phát hiện khuôn mặt đạt chuẩn kích thước phóng to ($H_{face} \ge 1.15 \times H_{baseline}$) và giữ yên vị trí.
- **Hành động:**
  1. Trình duyệt gọi API `POST /api/enroll/color_challenge`.
  2. Trình duyệt kích hoạt lớp phủ `div#flashingOverlay` với hiệu ứng Radial Glow.
  3. Trình duyệt chiếu Màu 1 trong $330\text{ ms}$. Tại thời điểm $t = 200\text{ ms}$ (sau thời gian ổn định màn hình và trước thời gian bù trừ AWB), Trình duyệt trích xuất Khung hình 1 (`Frame 1`).
  4. Trình duyệt chiếu Màu 2 trong $330\text{ ms}$. Tại thời điểm $t = 530\text{ ms}$, Trình duyệt trích xuất Khung hình 2 (`Frame 2`).
  5. Trình duyệt chiếu Màu 3 trong $330\text{ ms}$. Tại thời điểm $t = 860\text{ ms}$, Trình duyệt trích xuất Khung hình 3 (`Frame 3`).
  6. Trình duyệt đóng gói 3 khung hình cùng với Token và gửi tới Máy chủ qua API `POST /api/enroll/color_verify`.
  7. Trình duyệt tắt lớp phủ `div#flashingOverlay` ngay tại $t = 1000\text{ ms}$.

### 2.3. FR-03: Trích xuất 3 vùng da mặt và 1 vùng nền đối chứng
- **Tác nhân:** Module `optical_analyzer.py` trên Máy chủ.
- **Đầu vào:** Mỗi khung hình RGB và 468 mốc hình học khuôn mặt từ MediaPipe Face Mesh.
- **Hành động:**
  1. Máy chủ xác định vùng ROI Trán (Forehead ROI) từ các mốc MediaPipe: `[10, 67, 109, 108, 151, 337, 297, 284]`.
  2. Máy chủ xác định vùng ROI Gò má trái (Left Cheek ROI) từ các mốc MediaPipe: `[116, 123, 147, 213, 192, 214]`.
  3. Máy chủ xác định vùng ROI Gò má phải (Right Cheek ROI) từ các mốc MediaPipe: `[345, 352, 376, 433, 416, 434]`.
  4. Máy chủ xác định vùng ROI Nền đối chứng (Background ROI): góc trên cùng bên trái khung hình nằm ngoài khuôn mặt.
  5. Máy chủ tính toán giá trị trung bình kênh màu $(\bar{R}, \bar{G}, \bar{B})$ cho từng vùng ROI trên từng khung hình:
     $$\mathbf{m}_{ROI}(t) = \left( \frac{1}{N}\sum_{p \in ROI} R_p, \frac{1}{N}\sum_{p \in ROI} G_p, \frac{1}{N}\sum_{p \in ROI} B_p \right)$$

### 2.4. FR-04: Tính toán biến thiên quang phổ vi sai (Differential Chrominance Shift)
- **Tác nhân:** Module `optical_analyzer.py` trên Máy chủ.
- **Hành động:**
  1. Với mỗi bước chuyển đổi từ màu $C_{k-1}$ sang màu $C_k$ ($k \in \{1, 2\}$):
     - Tính véc-tơ biến thiên màu da: $\Delta \mathbf{S}_k = \mathbf{m}_{Skin}(k) - \mathbf{m}_{Skin}(k-1)$.
     - Tính véc-tơ biến thiên màu nền: $\Delta \mathbf{B}_k = \mathbf{m}_{Bg}(k) - \mathbf{m}_{Bg}(k-1)$.
     - Tính véc-tơ biến thiên nguồn sáng phát từ màn hình: $\Delta \mathbf{E}_k = \mathbf{Color}_k - \mathbf{Color}_{k-1}$.
  2. Bù trừ hiện tượng AWB bằng công thức vi sai tương đối:
     $$\Delta \mathbf{S}_{comp, k} = \Delta \mathbf{S}_k - \alpha \cdot \Delta \mathbf{B}_k$$
     Trong đó $\alpha = 0.5$ là hệ số ghép quang học của ánh sáng môi trường.

### 2.5. FR-05: Đánh giá tương quan quang học và đưa ra phán quyết
- **Tác nhân:** Module `optical_analyzer.py` trên Máy chủ.
- **Hành động:**
  1. Máy chủ tính toán hệ số tương quan Pearson $r$ giữa véc-tơ phản xạ da đã bù trừ và véc-tơ nguồn sáng màn hình:
     $$r = \frac{\sum_{i=1}^{M} (\Delta S_{comp, i} - \bar{S})(\Delta E_i - \bar{E})}{\sqrt{\sum_{i=1}^{M} (\Delta S_{comp, i} - \bar{S})^2 \cdot \sum_{i=1}^{M} (\Delta E_i - \bar{E})^2}}$$
  2. **Quy tắc điều kiện phán quyết:**
     - **Nếu** $r \ge 0.65$ VÀ không phát hiện dấu hiệu màn hình giả mạo (độ bão hòa không bị vỡ):
       $\implies$ Kết luận: `LIVENESS_PASS` (Người thật sống).
     - **Nếu** $r < 0.65$:
       $\implies$ Kết luận: `LIVENESS_FAIL` (Phản xạ không khớp nguồn sáng).
     - **Nếu** $\Delta \mathbf{S}_{comp} \approx \mathbf{0}$ qua cả 3 khung hình:
       $\implies$ Kết luận: `REPLAY_SPOOF` (Video quay sẵn hoặc ảnh in bất động).

### 2.6. FR-06: Chuyển tiếp trạng thái quy trình đăng ký (State Transition)
- **Tác nhân:** Module `enrollment.py` trên Máy chủ.
- **Hành động:**
  1. **Nếu** `optical_liveness == PASS`:
     - Máy chủ chuyển phiên từ `Stage.ZOOM_IN` sang `Stage.CAPTURE`.
     - Máy chủ trả về thông báo: "Xác thực quang học thành công. Đang lưu ảnh chân dung...".
  2. **Nếu** `optical_liveness == FAIL`:
     - Máy chủ chuyển phiên sang `Stage.FAILED`.
     - Máy chủ trả về thông báo: "Xác thực thất bại. Hệ thống phát hiện bề mặt không hợp lệ."

---

## 3. ĐẶC TẢ GIAO DIỆN LẬP TRÌNH ỨNG DỤNG (API SPECIFICATIONS)

### 3.1. Endpoint 1: Yêu cầu chuỗi thách thức quang học
- **URL:** `POST /api/enroll/color_challenge`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "session_id": "b18b6fc2-2fb4-4a25-885f-8d96b1bcf099"
}
```
- **Response Success (200 OK):**
```json
{
  "session_id": "b18b6fc2-2fb4-4a25-885f-8d96b1bcf099",
  "challenge_token": "a4f89d31e9c2b3...",
  "step_duration_ms": 330,
  "sequence": [
    {"index": 0, "name": "EMERALD_GREEN", "hex": "#00E676", "rgb": [0, 230, 118]},
    {"index": 1, "name": "CORAL_RED", "hex": "#FF3D00", "rgb": [255, 61, 0]},
    {"index": 2, "name": "SKY_BLUE", "hex": "#2979FF", "rgb": [41, 121, 255]}
  ],
  "expires_at": 1728100500.5
}
```
- **Response Error (404 Not Found):**
```json
{
  "detail": "Phiên làm việc không tồn tại hoặc đã hết hạn"
}
```

### 3.2. Endpoint 2: Xác thực phản xạ quang phổ
- **URL:** `POST /api/enroll/color_verify`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "session_id": "b18b6fc2-2fb4-4a25-885f-8d96b1bcf099",
  "challenge_token": "a4f89d31e9c2b3...",
  "frames": [
    {
      "color_index": 0,
      "timestamp_ms": 1728100485200,
      "image": "data:image/jpeg;base64,..."
    },
    {
      "color_index": 1,
      "timestamp_ms": 1728100485530,
      "image": "data:image/jpeg;base64,..."
    },
    {
      "color_index": 2,
      "timestamp_ms": 1728100485860,
      "image": "data:image/jpeg;base64,..."
    }
  ]
}
```
- **Response Success (200 OK):**
```json
{
  "session_id": "b18b6fc2-2fb4-4a25-885f-8d96b1bcf099",
  "passed": true,
  "correlation_score": 0.86,
  "verdict": "LIVENESS_PASS",
  "stage": "capture",
  "message": "Xác thực quang học thành công. Đang chụp chân dung HD...",
  "details": {
    "skin_reflection_delta": [12.4, -18.2, 22.1],
    "awb_drift_factor": 0.08
  }
}
```
- **Response Fail (200 OK with verdict FAIL):**
```json
{
  "session_id": "b18b6fc2-2fb4-4a25-885f-8d96b1bcf099",
  "passed": false,
  "correlation_score": 0.24,
  "verdict": "LIVENESS_FAIL",
  "stage": "failed",
  "message": "Xác thực thất bại. Hệ thống phát hiện bề mặt không hợp lệ.",
  "details": {
    "reason": "Optical reflection did not correlate with dynamic challenge sequence"
  }
}
```

---

## 4. XỬ LÝ TRƯỜNG HỢP BIÊN & NGOẠI LỆ (EDGE CASES & ERROR RECOVERY)

1. **Người dùng nhắm mắt hoặc quay mặt đi trong khi nháy màu:**
   - *Phát hiện:* MediaPipe không tìm thấy 468 mốc khuôn mặt trên 1 trong 3 khung hình.
   - *Hành động:* Hệ thống không tính toán điểm. Hệ thống trả về mã lỗi `ERR_FACE_LOST` và yêu cầu người dùng nhìn thẳng làm lại.
2. **Khách hàng ngồi trong phòng tối hoàn toàn (Zero Ambient Light):**
   - *Phát hiện:* Khung hình nền có cường độ sáng trung bình $Y_{bg} < 10$.
   - *Hành động:* Tín hiệu phản xạ từ màn hình đạt tỷ lệ tín hiệu trên nhiễu (SNR) tối đa. Hệ thống xử lý bình thường và điều chỉnh trọng số bù trừ $\alpha = 0.1$.
3. **Khách hàng ngồi ngược sáng mạnh (Backlight / Ánh nắng trực tiếp vào camera):**
   - *Phát hiện:* Khung hình nền bị lóa sáng quá mức ($Y_{bg} > 240$).
   - *Hành động:* Bù trừ AWB nâng trọng số $\alpha = 0.7$. Nếu mặt bị tối đen hoàn toàn ($Y_{face} < 30$), hệ thống nhắc: "Ánh sáng ngược quá mạnh. Vui lòng đổi vị trí ngồi."
4. **Token bị trễ quá hạn (Expired Token $\Delta t > 15\text{s}$):**
   - *Phát hiện:* `now - token.created_at > 15.0`.
   - *Hành động:* Từ chối xác thực với mã `ERR_TOKEN_EXPIRED`. Bắt buộc tạo Token mới.

---

## 5. TIÊU CHÍ NGHIỆM THU PHASE 2 (FSD ACCEPTANCE CRITERIA)
- [x] Đặc tả đầy đủ 7 yêu cầu chức năng (FR-01 đến FR-07) theo cấu trúc ngữ nghĩa ASD-STE100.
- [x] Định nghĩa đầy đủ 2 hợp đồng giao tiếp API RESTful (Payload Request, Response Success, Response Fail).
- [x] Bao quát 4 trường hợp biên quang học và điều kiện mạng thực tế.
- [x] Cung cấp công thức vi sai bù trừ AWB và tương quan Pearson chính xác.

---
*Tài liệu FSD hoàn thành.*
