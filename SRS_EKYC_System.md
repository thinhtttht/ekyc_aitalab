# BẢN ĐẶC TẢ YÊU CẦU PHẦN MỀM (SRS)
# SOFTWARE REQUIREMENTS SPECIFICATION: HỆ THỐNG eKYC & SINH TRẮC HỌC KHUÔN MẶT
> **Đề tài:** Hệ thống eKYC Đa người dùng trên Laptop Webcam RGB với Cơ chế Phòng thủ Đa tầng  
> **Phiên bản:** 1.0  
> **Chuẩn tài liệu:** Tham chiếu IEEE Std 830-1998  

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)
* **Tên hệ thống:** Smart-eKYC Biometric Authentication System.
* **Mục tiêu cốt lõi:** Chứng minh năng lực luồng thị giác máy tính (PoC) trong việc xác thực danh tính sinh trắc học và phòng chống tấn công giả mạo (Anti-Spoofing) đa tầng.
* **Môi trường hoạt động:** 
  * Chạy cục bộ (Localhost) trên máy tính cá nhân/laptop.
  * Tận dụng Camera RGB phổ thông (Webcam 720p/480p).
  * Tự động tận dụng Card đồ họa rời (GPU / CUDA) nếu có để tăng tốc độ tính toán, hoặc tối ưu hóa chạy mượt mà trên CPU nếu không có GPU.
  * Hỗ trợ lưu trữ và quản lý đa người dùng (Multi-user Database).

---

## 2. KIẾN TRÚC PHÒNG THỦ ĐA TẦNG (3-TIER DEFENSE PIPELINE)

Hệ thống bắt buộc tuân theo nguyên lý **Phòng thủ theo chiều sâu (Defense in Depth)** qua 3 cổng kiểm soát:

```
[Webcam RGB] ──► [Tiền xử lý & FQA] ──► [CỔNG 1: Active & Flash] ──► [CỔNG 2: MiniFASNet] ──► [CỔNG 3: ArcFace] ──► [Duyệt / Từ chối]
```

* **Cổng Tiền xử lý (Face Quality Assessment - FQA):** Kiểm tra độ sáng, độ sắc nét, kiểm tra lọt khung Oval, phát hiện và cảnh báo tháo khẩu trang / kính râm đen.
* **Cổng 1 (Active Liveness & Optical Flash):** 
  * Bắt chuyển động hình học: Nghiêng mặt Trái/Phải (Yaw), Lên/Xuống (Pitch), Tiến gần/Lùi xa (Zoom).
  * Ứng dụng kiểm tra tính nhất quán chuyển động phông nền (Motion Parallax).
  * Random Color Flashing: Chiếu chuỗi màu ngẫu nhiên kiểm tra mao mạch da người.
* **Cổng 2 (Passive Deep Anti-Spoofing):**
  * Mô hình **MiniFASNet** (Silent-Face-Anti-Spoofing) soi vân sọc Moiré màn hình iPad/điện thoại và chất liệu phản xạ của ảnh in.
* **Cổng 3 (Biometric Face Matching):**
  * Mô hình **ArcFace** (InsightFace): Cắt nắn chuẩn $112 \times 112$, trích xuất vector 512 chiều, so khớp Cosine Similarity ($\text{Threshold} \ge 0.65$).

---

## 3. YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### 3.1. Các Module Backend (Phía Xử Lý Thuật Toán)
* **[REQ-BE-01] Module Kiểm tra Thiết bị & Chất lượng (Camera & FQA):**
  * Đọc luồng video từ webcam ở độ phân giải tối thiểu $640 \times 480$ với $\text{FPS} \ge 25$.
  * Tự động phát hiện khuôn mặt và kiểm tra xem mặt đã lọt vừa khít khung Oval chưa.
  * Phát hiện vật cản che mặt (khẩu trang, kính râm đen) và trả về cảnh báo yêu cầu tháo bỏ.
* **[REQ-BE-02] Module Active Liveness & Optical Physics (Cổng 1):**
  * Đo đạc góc quay đầu dựa trên 5 điểm mốc (MediaPipe).
  * Sinh mã màu ngẫu nhiên (Random Hex) và điều khiển viền UI/màn hình nhấp nháy trong $0.5 - 0.8$ giây.
  * Tính toán mức độ biến thiên sắc độ màu quang học trên da mặt (vùng trán/gò má).
  * Thuật toán phân tích chuyển động phông nền (Motion Parallax) để loại trừ ảnh in phẳng.
* **[REQ-BE-03] Module Passive Anti-Spoofing (Cổng 2):**
  * Chạy mô hình mạng nơ-ron tích chập nhẹ **MiniFASNet**.
  * Nhận diện dấu vết tấn công màn hình (Replay Attack) và ảnh in (Print Attack). Ngưỡng tin cậy $P(\text{Real}) \ge 0.85$.
* **[REQ-BE-04] Module Nhận diện & So khớp Sinh trắc học (Cổng 3):**
  * Sử dụng thư viện **InsightFace / ArcFace**.
  * Tự động biến đổi Affine Transform đưa 2 mắt về trục ngang $0^\circ$, kích thước $112 \times 112$.
  * Rút trích vector nhúng 512 số thực (Face Embedding) và chuẩn hóa chuẩn L2.
  * Tính toán độ tương đồng Cosine Similarity giữa vector hiện tại và vector lưu trong cơ sở dữ liệu.
* **[REQ-BE-05] Module Cơ sở dữ liệu & Quản lý Người dùng (Storage):**
  * Lưu trữ thông tin người dùng và vector đặc trưng vào `database.json` (hỗ trợ mở rộng đa người dùng).
  * Không lưu trữ ảnh mặt thô để bảo mật quyền riêng tư sinh trắc học.
* **[REQ-BE-06] Module Mở rộng (Bonus/Future Scope - Cổng 4: Eye-Gaze Challenge Liveness):**
  * Tích hợp công nghệ **Eye-Tracking / Gaze Estimation** (tham chiếu ETH-XGaze / L2CS-Net / MediaPipe Iris) để triển khai cơ chế **Thử thách - Phản hồi bằng ánh mắt (Eye Gaze Challenge-Response)**.
  * *Cơ chế hoạt động:* Hệ thống hiển thị một điểm sáng/mục tiêu di chuyển ngẫu nhiên trên màn hình và yêu cầu người dùng dùng mắt dõi theo. Thuật toán kiểm tra sự đồng bộ giữa hướng nhìn của con ngươi và quỹ đạo di chuyển của mục tiêu.
  * *Ý nghĩa bảo mật:* Chặn đứng hoàn toàn các hình thức tấn công video quay lén hoặc Deepfake nâng cao, vì video phát lại không thể dự đoán và liếc mắt theo quỹ đạo ngẫu nhiên của hệ thống theo thời gian thực.

---

### 3.2. Các Module Frontend & User Flows (Giao Diện Người Dùng)
* **[REQ-FE-01] Màn hình Đăng ký (Enrollment Flow):**
  * Form nhập thông tin: Mã sinh viên (MSSV) + Mật khẩu.
  * Khung quét Camera tròn/Oval có hiển thị viền trạng thái động (Vàng: căn chỉnh, Đỏ: lỗi/đeo khẩu trang, Xanh lá: hợp lệ).
  * Hướng dẫn người dùng quay mặt nhẹ và giữ yên để hệ thống tự nháy màu và chụp ảnh tự động.
* **[REQ-FE-02] Màn hình Đăng nhập / Xác thực chuyển tiền (Verification Flow):**
  * Form xác nhận giao dịch (Số tiền, Người nhận).
  * Bật cửa sổ Modal Camera: Người dùng đưa mặt vào khung Oval giữ yên trong 1 giây $\rightarrow$ Hệ thống tự quét ngầm 3 cổng $\rightarrow$ Trả kết quả duyệt/từ chối ngay trên màn hình.
* **[REQ-FE-03] Màn hình Bảng điều khiển (Main Dashboard):**
  * Hiển thị thông tin người dùng đang đăng nhập, số dư giả lập và nút Đăng xuất (Logout).
  * **Chế độ Quản trị (Admin View):** Hiển thị danh sách các tài khoản/MSSV đã đăng ký trong hệ thống để phục vụ việc kiểm tra và chấm điểm của giảng viên.

---

## 4. YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)

| Tiêu chí | Yêu cầu kỹ thuật chi tiết |
| :--- | :--- |
| **Tốc độ khung hình (FPS)** | Đạt ổn định từ **$25 - 30\text{ FPS}$** khi chạy trên CPU laptop phổ thông (và $\ge 60\text{ FPS}$ nếu có GPU rời). |
| **Thời gian xác thực (Latency)** | Tổng thời gian xác thực chuyển tiền toàn trình (từ lúc mặt vào form đến khi duyệt) **$\le 1.5 - 2.0\text{ giây}$**. |
| **Độ chính xác (Accuracy)** | Độ chính xác nhận diện sinh trắc học ArcFace $\ge 99\%$ trên ảnh chuẩn; Tỷ lệ chặn giả mạo (Anti-Spoofing) $\ge 98\%$. |
| **Khả năng chịu tải (Scalability)** | Tìm kiếm vector 1:N trong cơ sở dữ liệu giả lập $1.000$ người dùng dưới $0.05\text{ giây}$. |
| **Bảo mật dữ liệu** | Dữ liệu khuôn mặt chỉ được lưu dưới dạng vector mã hóa 512 chiều, không lưu trữ ảnh mặt thô. |

---

## 5. LỘ TRÌNH TRIỂN KHAI THEO CÁC GIAI ĐOẠN (PROJECT PHASES)

* **Phase 1 (Nền tảng & Tiền xử lý):** Thiết lập môi trường Python, kiểm tra webcam, FQA (khung Oval, độ nét, phát hiện khẩu trang/kính).
* **Phase 2 (Giao diện Khung Oval & Color Flash):** Xây dựng UI chuẩn ngân hàng, hoàn thiện cơ chế nhấp nháy màu ngẫu nhiên.
* **Phase 3 (Active Liveness):** Hoàn thiện thuật toán nhận biết quay đầu (Head Pose), cự ly (Zoom) và thị sai chuyển động phông nền (Motion Parallax).
* **Phase 4 (Passive Liveness):** Tích hợp mô hình MiniFASNet chống video và ảnh in.
* **Phase 5 (Biometric Matching):** Tích hợp InsightFace / ArcFace, tính Cosine Similarity, hoàn thiện file `database.json` đa người dùng.
* **Phase 6 (Mở rộng & Nâng cao):** Bổ sung module Eye-Tracking / Gaze Assistance, viết script đo stress test và hoàn thiện báo cáo khoa học.
