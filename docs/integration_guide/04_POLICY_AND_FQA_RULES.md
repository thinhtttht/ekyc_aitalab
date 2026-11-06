# 📋 BỘ QUY CHUẨN CHẤT LƯỢNG KHUÔN MẶT (FQA POLICY)
## 04_POLICY_AND_FQA_RULES.md

Tài liệu này cung cấp các chính sách (Policy), quy chuẩn kỹ thuật và bộ thông điệp gợi ý giao diện (UI/UX Messages) để nhóm phát triển tích hợp hướng dẫn thí sinh đạt chất lượng ảnh đầu vào tối ưu nhất.

---

## 1. Bảng Quy Chuẩn Kiểm Duyệt Đầu Vào (Checklist Tiêu Chuẩn)

| Hạng Mục Kiểm Tra | Tiêu Chuẩn Cho Phép (Pass) | Trường Hợp Bị Từ Chối (Fail) | Cơ Chế AI Phát Hiện |
| :--- | :--- | :--- | :--- |
| **1. Khẩu trang & Vải che mặt** | Để lộ toàn bộ ngũ quan (Trán, 2 mắt, mũi, miệng, cằm). | Đeo khẩu trang y tế, khẩu trang vải, khăn trùm kín đầu che cằm/miệng. | Thuật toán phân tích sắc độ da vùng má dưới so với trán ($YCrCb$) và độ nổi mốc sống mũi. |
| **2. Kính mắt** | $\bullet$ Kính cận, kính trắng trong suốt.<br>$\bullet$ Nhìn rõ 2 con ngươi mắt. | $\bullet$ Kính râm đen sì, kính đổi màu sẫm.<br>$\bullet$ Kính bị đèn/màn hình phản chiếu lóa trắng $>30\%$ diện tích mắt. | Phân tích độ sáng trung bình vùng mắt (Sunglasses mean) và điểm ảnh lóa $>240$ (Glare detection). |
| **3. Bàn tay che mặt** | Hai tay đặt trên bàn hoặc để tự nhiên, không chạm vào mặt. | Lấy tay chống cằm che miệng, gãi mũi, lấy tay che mặt khi suy nghĩ. | MediaPipe Hands: Đếm số lượng điểm mốc khớp tay đè vào vùng bounding box khuôn mặt. |
| **4. Cự ly & Khung hình (Framing)** | Ngồi cách camera **50 - 70 cm**. Khuôn mặt chiếm **45% - 85%** chiều cao khung hình. | $\bullet$ Ngồi quá xa ($<40\%$): Khuôn mặt bé tí, mất nét chi tiết.<br>$\bullet$ Dí sát mặt ($>85\%$): Bị tràn khung hình, méo phối cảnh. | Tỷ lệ `scale_ratio` giữa chiều rộng khuôn mặt và chiều rộng khung Oval. |
| **5. Góc quay đầu (Pose)** | Nhìn thẳng vào webcam ($\|Yaw\| \le 12^\circ, \|Pitch\| \le 15^\circ$). | Quay đầu sang trái/phải, ngửa mặt lên trần hoặc cúi gầm mặt xuống bàn. | Giải thuật 3D Pose `solvePnP` tính 3 góc Euler. |
| **6. Điều kiện ánh sáng (Illumination)** | Ánh sáng tản đều, độ sáng trung bình $40 \le \text{Mean} \le 210$. | $\bullet$ Phòng tối thui ($\text{Mean} < 40$).<br>$\bullet$ Đèn pha thẳng vào mặt ($\text{Mean} > 215$).<br>$\bullet$ Ngược sáng (Ngồi trước cửa sổ/đèn trần làm mặt bị đen kịt). | Histogram độ sáng trên kênh Grayscale / kênh Y ($YCrCb$) kết hợp độ lệch chuẩn `std_dev`. |

---

## 2. Bảng Ngưỡng Thông Số Kỹ Thuật Chi Tiết (Technical Thresholds)

Các ngưỡng này được đúc kết từ hàng nghìn lần thử nghiệm thực tế trên nhiều dòng webcam máy tính xách tay:

```python
# CẤU HÌNH NGƯỠNG CHẤT LƯỢNG KHUÔN MẶT (FQA THRESHOLDS)
CONFIG_FQA = {
    # 1. Khung Oval hướng dẫn (Tọa độ chuẩn hóa [0.0 - 1.0])
    "OVAL_CX": 0.50,            # Tâm X (chính giữa)
    "OVAL_CY": 0.50,            # Tâm Y (chính giữa)
    "OVAL_RX": 0.22,            # Bán trục ngang (chiều rộng)
    "OVAL_RY": 0.32,            # Bán trục dọc (chiều cao)
    "OVAL_TOLERANCE": 1.00,     # Dung sai lọt elip ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1.0
    
    # 2. Cự ly khuôn mặt (Scale Ratio)
    "FACE_SCALE_MIN": 0.40,     # Chiều rộng mặt tối thiểu = 40% chiều rộng Oval
    "FACE_SCALE_MAX": 0.85,     # Chiều rộng mặt tối đa = 85% chiều rộng Oval
    
    # 3. Góc xoay đầu cho phép khi Đăng ký / Điểm danh
    "MAX_YAW": 12.0,            # Quay trái/phải tối đa 12 độ
    "MAX_PITCH": 15.0,          # Cúi/ngửa đầu tối đa 15 độ
    "MAX_ROLL": 10.0,           # Nghiêng đầu sang vai tối đa 10 độ
    
    # 4. Góc xoay đầu cảnh báo trong lúc Thi Vấn Đáp (Proctoring)
    "EXAM_WARN_YAW": 22.0,      # Quay đầu > 22 độ sẽ nhắc nhở
    "EXAM_VIOLATION_YAW": 28.0, # Quay đầu > 28 độ tính vi phạm
    "EXAM_WARN_PITCH_DOWN": -18.0, # Cúi đầu < -18 độ (nhìn điện thoại/tài liệu)
    
    # 5. Ánh sáng & Độ nét
    "BRIGHTNESS_MIN": 40.0,     # Dưới 40 là quá tối
    "BRIGHTNESS_MAX": 215.0,    # Trên 215 là quá chói
    "BACKLIGHT_STD_MIN": 10.0,  # Độ lệch chuẩn ánh sáng tối thiểu chống ngược sáng
    "SHARPNESS_MIN": 15.0,      # Phương sai Laplacian tối thiểu chống mờ nhòe
    
    # 6. Ngưỡng chống giả mạo & Nhận diện
    "LIVENESS_PROB_MIN": 0.70,  # MiniFASNet xác suất người thật >= 70%
    "ARCFACE_MATCH_THRESH": 0.55 # Cosine similarity >= 55% coi là cùng một người
}
```

---

## 3. Gợi Ý Thông Điệp Giao Diện Thân Thiện (UI/UX Prompt Messages)

Để trải nghiệm người dùng không bị ức chế, hệ thống UI nên hiển thị các thông báo ngắn gọn, rõ ràng theo ngữ cảnh kèm màu sắc quy ước:
* **Xanh lục (`#10b981`):** Trạng thái hoàn hảo, đang giữ yên để chụp/xác thực.
* **Vàng cam (`#f59e0b`):** Cần điều chỉnh nhẹ (cự ly, ánh sáng, góc nhìn).
* **Đỏ (`#ef4444`):** Vi phạm quy chế thi hoặc nghi vấn gian lận.

### 3.1. Thông điệp ở Giai đoạn Đăng ký & Điểm danh

| Trường Hợp | Mã Code | Thông Điệp Hiển Thị Trên UI | Màu Sắc |
| :--- | :--- | :--- | :--- |
| Chưa thấy mặt | `NO_FACE` | "👤 Vui lòng đưa khuôn mặt vào giữa khung hình" | Xám |
| Lệch ngoài Oval | `OUT_OF_OVAL` | "🎯 Vui lòng căn đầu lọt gọn vào giữa hình Oval" | Vàng |
| Ngồi quá xa | `TOO_FAR` | "🔍 Hãy tiến lại gần camera hơn một chút" | Vàng |
| Ngồi quá gần | `TOO_CLOSE` | "↔️ Hãy lùi ra xa camera một chút" | Vàng |
| Đang quay đầu | `HEAD_TURNED` | "👀 Hãy nhìn thẳng vào camera, không quay đầu" | Vàng |
| Đeo khẩu trang | `MASK_DETECTED` | "😷 Vui lòng tháo khẩu trang để tiếp tục" | Đỏ |
| Đeo kính đen | `SUNGLASSES` | "🕶️ Vui lòng tháo kính râm / kính đen" | Đỏ |
| Kính bị lóa | `GLASS_GLARE` | "💡 Nghiêng nhẹ đầu hoặc chỉnh góc màn hình để tránh lóa kính" | Vàng |
| Lấy tay che mặt | `HAND_OCCLUSION` | "✋ Vui lòng bỏ tay ra khỏi khuôn mặt" | Đỏ |
| Phòng quá tối | `TOO_DARK` | "🌙 Không gian quá tối – Hãy bật thêm đèn" | Đỏ |
| Bị ngược sáng | `BACKLIGHT` | "☀️ Tránh ngồi quay lưng về phía cửa sổ hoặc nguồn sáng chói" | Vàng |
| Bị nhòe chuyển động | `BLURRY` | "📸 Giữ yên khuôn mặt và camera trong giây lát..." | Vàng |
| Đạt chuẩn | `HOLDING` | "✅ Rất tốt! Giữ nguyên tư thế trong 1 giây... (8/10)" | Xanh lục |

### 3.2. Thông điệp ở Giai đoạn Phòng Thi Vấn Đáp (Giám Thị AI)

| Trường Hợp | Thông Điệp Cho Thí Sinh | Thông Điệp Báo Về Cho Giám Khảo |
| :--- | :--- | :--- |
| **Quay đầu sang bên** | Viền cam màn hình: *"⚠️ Chú ý: Vui lòng nhìn thẳng vào màn hình thi"* | Bảng danh sách: *"Thí sinh [A] quay đầu sang trái (12 giây)"* |
| **Liếc mắt liên tục** | Viền cam: *"⚠️ Cảnh báo: Phát hiện hướng nhìn ra ngoài màn hình"* | Log: *"Nghi vấn nhìn màn hình phụ bên cạnh"* |
| **Nhiều người trong phòng** | Viền đỏ: *"🚨 Cảnh báo: Phát hiện nhiều người trong phòng thi!"* | Popup đỏ: *"🚨 PHÁT HIỆN 2 NGƯỜI TRONG KHUNG HÌNH THÍ SINH [A]"* kèm ảnh chụp |
| **Thí sinh vắng mặt** | Viền đỏ: *"🚨 Thí sinh rời khỏi vị trí thi!"* | Icon cảnh báo: *"Thí sinh vắng mặt khỏi camera > 5s"* |
| **Tráo người thi** | Viền đỏ: *"🚨 DANH TÍNH KHÔNG TRÙNG KHỚP HỒ SƠ"* | Chuông báo động: *"🚨 PHÁT HIỆN TRÁO ĐỔI NGƯỜI THI HỘ!"* |

---

## 4. Xử Lý Các Trường Hợp Đặc Biệt (Edge Cases Handling)

1. **Người đeo kính cận tròng dày:**
   * Hệ thống **chấp nhận 100% người đeo kính cận**.
   * Chỉ kích hoạt cảnh báo kính khi điểm ảnh phản xạ lóa sáng (Reflection Glare) vượt quá 30% diện tích mắt hoặc tròng kính đen sì (kính đổi màu ra nắng).
2. **Thí sinh chớp mắt tự nhiên:**
   * Cử động chớp mắt sinh lý (Eye Blink) chỉ kéo dài 100 - 250 ms. Thuật toán theo dõi mắt và FQA luôn có cửa sổ lọc mượt (Temporal Smoothing Window) 5 - 10 frames để không bao giờ bắt lỗi oan khi thí sinh chớp mắt.
3. **Màu da và điều kiện bóng đổ:**
   * Phép kiểm tra độ nét và bộ phận ngũ quan kết hợp đa không gian màu ($RGB$, $Grayscale$ và $YCrCb$). Không gian $YCrCb$ tách biệt hoàn toàn độ sáng ($Y$) và sắc độ màu ($Cr, Cb$), giúp thuật toán hoạt động công bằng, chính xác trên mọi sắc tộc và màu da.
