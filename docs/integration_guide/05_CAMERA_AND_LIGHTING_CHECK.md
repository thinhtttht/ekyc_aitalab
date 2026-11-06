# 📷 KIỂM TRA PHẦN CỨNG CAMERA & ĐIỀU KIỆN ÁNH SÁNG
## 05_CAMERA_AND_LIGHTING_CHECK.md

Tài liệu này hướng dẫn chi tiết cách kiểm tra phần cứng thiết bị camera (REQ-BE-01) và điều kiện ánh sáng môi trường phòng thi, kèm thuật toán **Tự động tăng sáng thích ứng (Adaptive Brightness Enhancement)** để cứu các trường hợp thí sinh ngồi trong phòng thiếu sáng.

---

## 1. Tại Sao Phải Kiểm Tra Camera & Ánh Sáng Trước Khi Vào Thi?

Trong các kỳ thi vấn đáp trực tuyến, thí sinh thường gặp các vấn đề phần cứng:
1. **Camera bị che hoặc nắp camera chưa mở:** Màn hình đen thui (`Black Frame`).
2. **Webcam ảo / Video đóng băng (`Frozen Video`):** Dùng phần mềm giả lập webcam (OBS Virtual Cam, ManyCam) phát video lặp lại hoặc camera bị treo.
3. **Độ phân giải quá thấp hoặc FPS quá giật:** Camera điện thoại cắm qua app lậu có độ phân giải $320 \times 240$ hoặc 5 FPS khiến AI không nhận diện được.
4. **Phòng quá tối hoặc ngược sáng:** Ngồi quay lưng vào cửa sổ/đèn trần làm mặt bị đen kịt, hoặc phòng không bật đèn khiến ảnh bị nhiễu hạt (Noise).

---

## 2. Tiêu Chuẩn Kỹ Thuật Kiểm Tra Camera (Camera Quality Criteria)

| Tiêu chí | Ngưỡng đạt chuẩn (Pass) | Cảnh báo / Từ chối (Fail) | Thuật toán kiểm tra |
| :--- | :--- | :--- | :--- |
| **1. Khung hình đen / Bị che ống kính (`not_black`)** | $\text{Mean Brightness} \ge 18.0$ | $\text{Mean} < 18.0$ $\rightarrow$ Báo lỗi: *"Camera bị che hoặc khung hình đen – hãy kiểm tra nắp che ống kính"*. | Trung bình mức xám toàn khung hình (`gray.mean()`). |
| **2. Camera bị treo / Đóng băng (`not_frozen`)** | Các frame có sự dịch chuyển tự nhiên. | 15 frame liên tiếp độ sai khác $\Delta < 0.2$ pixel $\rightarrow$ Báo lỗi: *"Camera bị treo hoặc đóng băng hình ảnh"*. | Trừ ảnh tuyệt đối giữa 2 frame liên tiếp: $\|I_t - I_{t-1}\|$. |
| **3. Độ phân giải tối thiểu (`resolution_ok`)** | $\ge 640 \times 480$ (Khuyên dùng $1280 \times 720$). | Chiều rộng $< 640$ hoặc Chiều cao $< 480$ $\rightarrow$ Báo lỗi: *"Độ phân giải camera quá thấp"*. | `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`. |
| **4. Tốc độ khung hình (`fps_ok`)** | $\ge 15.0$ FPS (Lý tưởng 24 - 30 FPS). | $< 12.0$ FPS $\rightarrow$ Nhắc nhở: *"FPS thấp, nên đóng bớt các ứng dụng nặng"*. | Đếm số frame nhận được trên 1 giây. |
| **5. Nhiễu hạt cảm biến (`noise_ok`)** | $\sigma_{\text{noise}} \le 12.0$ | $\sigma_{\text{noise}} > 12.0$ $\rightarrow$ Báo vàng: *"Ảnh bị nhiễu hạt nặng, hãy bật thêm đèn phòng"*. | Giải thuật ước lượng nhiễu Gauss **Immerkær (1996)** bằng ma trận tích chập Laplace. |
| **6. Độ sáng tổng thể (`frame_brightness`)** | $45.0 \le \text{Brightness} \le 220.0$ | $\bullet < 45.0$: Phòng quá tối.<br>$\bullet > 220.0$: Đèn chiếu chói lóa camera. | Histogram Grayscale toàn khung hình. |
| **7. Chống ngược sáng khuôn mặt (`no_backlight`)** | $\text{Std\_Dev} \ge 10.0$ & Tỉ lệ 2 má $\le 1.8$. | Mặt đen kịt trong khi hậu cảnh sáng rực. | Độ lệch chuẩn ánh sáng trên vùng khuôn mặt. |

---

## 3. Mã Nguồn Kiểm Tra Camera Độc Lập (Ready-to-use Code)

Dưới đây là module Python hoàn chỉnh để nhóm đối tác nhúng vào trước bước Đăng ký hoặc Điểm danh:

```python
import math
import cv2
import numpy as np

# Kernel tích chập đo nhiễu Gauss theo nghiên cứu Immerkær (1996)
_IMMERKAER_KERNEL = np.array([
    [ 1, -2,  1],
    [-2,  4, -2],
    [ 1, -2,  1]
], dtype=np.float64)

class CameraHardwareChecker:
    """Module kiểm tra sức khỏe phần cứng camera và ánh sáng phòng thi."""
    
    def __init__(self, min_w=640, min_h=480, min_fps=15):
        self.min_w = min_w
        self.min_h = min_h
        self.min_fps = min_fps
        self.prev_gray = None
        self.frozen_counter = 0

    def estimate_noise(self, gray: np.ndarray) -> float:
        """Ước lượng độ lệch chuẩn nhiễu hạt cảm biến (Sensor Noise Sigma)."""
        h, w = gray.shape[:2]
        if h < 3 or w < 3:
            return 0.0
        conv = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REFLECT)
        total = np.abs(conv[1:-1, 1:-1]).sum()
        sigma = float(total * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))
        return round(sigma, 2)

    def evaluate_frame(self, frame_bgr: np.ndarray, fps: float = 30.0) -> dict:
        """
        Đánh giá khung hình camera:
        Trả về: {is_healthy: bool, severity: 'ok'|'warn'|'error', message: str, metrics: dict}
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        noise = self.estimate_noise(gray)

        # 1. Kiểm tra đóng băng (Frozen Camera Detection)
        g_float = gray.astype(np.float32)
        if self.prev_gray is not None and self.prev_gray.shape == g_float.shape:
            diff = float(np.abs(g_float - self.prev_gray).mean())
            if diff < 0.25: # Frame gần như giống hệt 100% frame trước
                self.frozen_counter += 1
            else:
                self.frozen_counter = 0
        else:
            diff = None
        self.prev_gray = g_float

        is_frozen = self.frozen_counter >= 15 # Đóng băng trên 15 frame liên tiếp (~0.5s)
        is_black = brightness < 18.0          # Khung hình đen / che ống kính
        res_ok = (w >= self.min_w and h >= self.min_h)
        bright_ok = (45.0 <= brightness <= 220.0)
        noise_ok = (noise <= 12.0)

        # Đánh giá thông điệp ưu tiên
        if is_black:
            return {
                "is_healthy": False, "severity": "error",
                "message": "🚨 Camera bị che hoặc khung hình đen – Hãy mở nắp ống kính!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }
        if is_frozen:
            return {
                "is_healthy": False, "severity": "error",
                "message": "🚨 Camera bị đóng băng / treo hình – Hãy chọn lại thiết bị!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }
        if not res_ok:
            return {
                "is_healthy": False, "severity": "error",
                "message": f"❌ Độ phân giải {w}x{h} thấp hơn chuẩn {self.min_w}x{self.min_h}!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }
        if brightness < 45.0:
            return {
                "is_healthy": False, "severity": "warn",
                "message": "⚠️ Phòng thi quá tối – Hãy bật thêm đèn phòng!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }
        if brightness > 220.0:
            return {
                "is_healthy": False, "severity": "warn",
                "message": "⚠️ Quá chói sáng – Tránh bóng đèn rọi thẳng vào camera!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }
        if not noise_ok:
            return {
                "is_healthy": False, "severity": "warn",
                "message": "⚠️ Hình ảnh bị nhiễu hạt nặng – Hãy tăng thêm ánh sáng!",
                "metrics": {"brightness": brightness, "noise": noise, "w": w, "h": h}
            }

        return {
            "is_healthy": True, "severity": "ok",
            "message": "✅ Camera & Ánh sáng đạt chuẩn phòng thi!",
            "metrics": {
                "brightness": round(brightness, 1),
                "noise": noise,
                "fps": round(fps, 1),
                "resolution": f"{w}x{h}"
            }
        }
```

---

## 4. Thuật Toán Tự Động Nâng Sáng Cứu Ảnh Tối (Adaptive CLAHE + Gamma)

Khi thí sinh ở phòng hơi tối ($25 \le \text{Brightness} < 75$), hệ thống có thể **tự động tăng sáng và cân bằng tương phản thông minh** mà không làm cháy da mặt hay đổi màu sắc tố da tự nhiên:

```python
def auto_enhance_brightness(frame_bgr: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    """
    Tự động nâng sáng thích ứng cho ảnh thiếu sáng bằng LAB + CLAHE + Adaptive Gamma.
    Không đổi màu da, chỉ tăng chi tiết vùng tối.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    curr_mean = float(gray.mean())

    # Nếu camera đã đủ sáng (>= 80) hoặc bị đen hoàn toàn (< 20), giữ nguyên
    if curr_mean < 20.0 or curr_mean >= 80.0:
        return frame_bgr

    # Tách không gian màu LAB (L: Độ sáng, A-B: Sắc tố màu)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. Áp dụng CLAHE cân bằng tương phản cục bộ trên kênh sáng L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # 2. Tính toán đường cong Gamma Curve thích ứng
    clahe_mean = max(1.0, float(l_clahe.mean()))
    if clahe_mean < target_mean:
        gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(clahe_mean / 255.0), 0.40, 1.0))
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        l_enhanced = cv2.LUT(l_clahe, lut)
    else:
        l_enhanced = l_clahe

    # Gộp lại kênh LAB và chuyển về BGR
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
```

---

## 5. Tích Hợp Vào Giao Diện Trước Khi Bắt Đầu Thi (Pre-Exam Checklist Modal)

Team làm UI nên hiển thị một màn hình **"Kiểm Tra Thiết Bị" (Hardware Check Modal)** trước khi cho thí sinh vào phòng thi với danh sách tích xanh trực quan:

```
┌────────────────────────────────────────────────────────┐
│  KIỂM TRA THIẾT BỊ TRƯỚC KHI VÀO PHÒNG THI             │
├────────────────────────────────────────────────────────┤
│  [✅] Thiết bị camera: Đã kết nối                      │
│  [✅] Độ phân giải: 1280 x 720 (Đạt chuẩn HD)           │
│  [✅] Tốc độ khung hình: 30 FPS (Mượt mà)              │
│  [✅] Độ sáng phòng thi: 112 / 255 (Ánh sáng tốt)      │
│  [✅] Độ nhiễu cảm biến: 3.2 (Ảnh sắc nét)             │
│  [✅] Kiểm tra chống đóng băng: Hoạt động bình thường │
│                                                        │
│  👉 Trạng thái: SẴN SÀNG ĐỂ ĐIỂM DANH VÀO THI         │
│  [ NÚT: BẮT ĐẦU ĐIỂM DANH SINH TRẮC HỌC ]             │
└────────────────────────────────────────────────────────┘
```
