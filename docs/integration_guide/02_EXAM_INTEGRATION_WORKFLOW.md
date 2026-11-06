# 🎓 KIẾN TRÚC TÍCH HỢP VÀO PHẦN MỀM THI VẤN ĐÁP
## 02_EXAM_INTEGRATION_WORKFLOW.md

Tài liệu này hướng dẫn cách thiết kế và kết nối các module eKYC vào luồng nghiệp vụ của **Phần Mềm Thi Vấn Đáp Trực Tuyến** qua 3 giai đoạn: **Đăng ký**, **Điểm danh vào thi**, và **Giám sát thông minh trong phòng thi**.

---

## 1. Tổng Quan 3 Giai Đoạn Vận Hành

```
[ GIAI ĐOẠN 1: ĐĂNG KÝ HỒ SƠ ]
           │
           ▼
[ GIAI ĐOẠN 2: ĐIỂM DANH VÀO PHÒNG THI ]
           │
           ▼
[ GIAI ĐOẠN 3: GIÁM THỊ AI SUỐT BUỔI THI VẤN ĐÁP ]
```

---

## 2. Giai Đoạn 1: Đăng Ký Hồ Sơ Thí Sinh (Candidate Enrollment)

Giai đoạn này diễn ra **trước ngày thi** (khi thí sinh khởi tạo tài khoản hoặc cập nhật hồ sơ dự thi).

### 2.1. Mục tiêu
Thu thập ảnh chân dung chất lượng cao nhất của thí sinh, tạo ra vector 512D ArcFace chuẩn và lưu trữ an toàn vào cơ sở dữ liệu.

### 2.2. Luồng xử lý kỹ thuật (State Machine)
1. **Hiển thị giao diện:** Mở camera, hiển thị một khung Oval hướng dẫn ở trung tâm màn hình.
2. **Kiểm tra chuẩn chất lượng (FQA Loop):**
   * Khuôn mặt lọt trọn vẹn trong khung Oval (`oval_dist <= 1.0`).
   * Cự ly chuẩn: Chiều cao mặt chiếm 45% - 85% chiều cao khung hình (khoảng cách 50 - 70 cm).
   * Góc mặt nhìn thẳng: $|\text{Yaw}| \le 12^\circ$, $|\text{Pitch}| \le 15^\circ$, $|\text{Roll}| \le 10^\circ$.
   * Không vi phạm: Không khẩu trang, không kính râm đen, không bị lóa kính quá 30%, không có bàn tay che mặt.
   * Ánh sáng đạt chuẩn: Không gian không quá tối, không quá chói, không bị ngược sáng.
3. **Kiểm tra người thật (Anti-Spoofing):**
   * Model MiniFASNet kiểm tra liên tục: `real_prob >= 0.70` (hoặc camera 3D Depth xác nhận độ cong 3D).
   * Chặn đứng việc thí sinh giơ ảnh in của người khác hoặc màn hình điện thoại trước camera.
4. **Trích xuất & Lưu trữ:**
   * Sau khi thí sinh giữ yên đúng tư thế trong **10 khung hình liên tiếp** (~0.5 - 1.0 giây), hệ thống tự động chụp khung hình nét nhất.
   * Chạy ArcFace trích xuất vector đặc trưng `embedding` (512 số thực `float32`).
   * Lưu vào Database: `(candidate_id, full_name, embedding_bytes, snapshot_jpg)`.

---

## 3. Giai Đoạn 2: Điểm Danh & Xác Thực Vào Phòng Thi (Check-in Verification)

Giai đoạn này diễn ra **ngay trước khi thí sinh bắt đầu phiên vấn đáp với giám khảo** (thường trong phòng chờ hoặc 5 phút trước giờ thi).

### 3.1. Mục tiêu
Chứng minh người đang ngồi trước máy tính **chính là thí sinh đã đăng ký** và **là người thật bằng xương bằng thịt**, không phải ảnh chụp hay video phát lại.

### 3.2. Luồng xử lý kỹ thuật
1. Thí sinh đăng nhập hoặc chọn mã số phòng thi / MSSV của mình. Hệ thống truy xuất vector gốc `target_embedding` đã đăng ký từ Database.
2. Mở camera, yêu cầu thí sinh nhìn thẳng vào khung Oval:
   * Kiểm tra người thật: `liveness.is_real == True`.
   * Kiểm tra vị trí: Đầu nằm trong khung Oval, nhìn thẳng camera.
3. **So khớp ArcFace 1:1:**
   * Trích xuất vector `current_embedding` từ khung hình hiện tại.
   * Tính Cosine Similarity:
     $$\text{Similarity} = \sum_{i=1}^{512} \text{curr}_i \times \text{target}_i$$
   * Chuyển đổi thành phần trăm tương đồng: `similarity_pct = int(Similarity * 100)`.
4. **Quyết định kết quả:**
   * Nếu `Similarity >= 0.55` (tương đương $\ge 55\%$ khớp) liên tục trong **5 frame** (~0.5 giây):
     * ✅ **Xác thực thành công!** 
     * Hệ thống tự động mở nút "VÀO PHÒNG THI" hoặc thông báo cho Giám khảo: *"Thí sinh [Nguyễn Văn A] đã được định danh chính chủ"*.
   * Nếu `Similarity < 0.50`:
     * ❌ **Từ chối:** *"Khuôn mặt không khớp với hồ sơ đăng ký. Vui lòng liên hệ cán bộ giám thị."*

---

## 4. Giai Đoạn 3: Giám Thị AI Suốt Buổi Thi Vấn Đáp (In-Exam AI Proctoring)

Trong suốt thời gian thi vấn đáp (thí sinh trả lời câu hỏi của hội đồng giám khảo, kéo dài 10 - 30 phút), luồng camera của thí sinh được AI theo dõi liên tục ở chế độ nền (Background Worker Thread).

### 4.1. Tần số xử lý tối ưu
* **Không cần chạy 30 FPS:** Việc chạy AI ở 30 FPS sẽ làm nóng máy thí sinh và gây giật lag đường truyền video thi vấn đáp.
* **Tần số khuyến nghị:** **8 - 12 FPS** (mỗi 80 - 120 ms phân tích một khung hình). Tần số này hoàn toàn đủ để bắt trọn mọi cử động gian lận mà tải CPU chỉ chiếm < 15%.

### 4.2. Bốn Hành Vi Gian Lận Được Hệ Thống Tự Động Bắt

| Loại Vi Phạm | Dấu Hiệu Kỹ Thuật | Ý Nghĩa Thực Tế | Hành Động Xử Lý |
| :--- | :--- | :--- | :--- |
| **1. Quay đầu ra ngoài (Head Turn)** | $\|Yaw\| > 25^\circ$ hoặc $Pitch < -20^\circ$ kéo dài $> 2.0$ giây. | Thí sinh quay sang bên cạnh nhìn tài liệu hoặc có người thứ 3 đứng ngoài camera nhắc bài; hoặc cúi gầm mặt nhìn điện thoại dưới bàn. | Cảnh báo viền vàng trên màn hình: *"Vui lòng nhìn thẳng vào màn hình thi"*. Ghi nhận log thời gian vi phạm. |
| **2. Liếc mắt ra ngoài (Gaze Violation)** | Đầu nhìn thẳng nhưng đồng tử mắt liếc lệch hẳn sang mép màn hình $> 3.0$ giây. | Thí sinh đang đọc tài liệu mở trên cửa sổ phụ hoặc nhìn màn hình phụ bên cạnh. | Nhắc nhở: *"Phát hiện hướng nhìn bất thường"*. |
| **3. Bất thường số lượng người (Face Count)** | $\bullet$ `face_count == 0` kéo dài $> 3$ giây.<br>$\bullet$ `face_count > 1`. | $\bullet$ Thí sinh rời khỏi vị trí thi.<br>$\bullet$ Có người lạ xuất hiện trong khung hình để trợ giúp. | Cảnh báo đỏ ngay lập tức trên màn hình Giám khảo: *"Phát hiện nhiều người trong phòng thi / Thí sinh vắng mặt"*. Chụp ảnh bằng chứng. |
| **4. Tráo đổi người thi hộ (Identity Swap)** | Cosine similarity giữa frame hiện tại và vector đăng ký tụt xuống $< 0.45$. | Có người khác nhảy vào ngồi thế chỗ để trả lời thay câu hỏi của giám khảo! | **BÁO ĐỘNG ĐỎ CAO NHẤT:** Tự động chụp ảnh bằng chứng gửi ngay đến màn hình Giám khảo kèm còi cảnh báo. |

---

## 5. Bảng Dữ Liệu Khuyến Nghị Lưu Trữ (Database Schema)

Để nhóm bạn dễ dàng tích hợp vào hệ thống cơ sở dữ liệu (PostgreSQL, MySQL hoặc SQLite), dưới đây là cấu trúc bảng mẫu:

```sql
-- 1. Bảng lưu trữ hồ sơ đăng ký thí sinh
CREATE TABLE candidates (
    id VARCHAR(64) PRIMARY KEY,          -- Mã sinh viên / Mã định danh
    full_name VARCHAR(255) NOT NULL,     -- Họ và tên
    class_name VARCHAR(100),             -- Lớp / Khóa học
    arcface_embedding BLOB NOT NULL,     -- 512 số float32 (kích thước đúng 2,048 bytes)
    snapshot_path VARCHAR(500),          -- Đường dẫn lưu ảnh chân dung gốc
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng ghi nhận nhật ký giám sát trong phòng thi (Proctoring Logs)
CREATE TABLE proctoring_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id VARCHAR(64) NOT NULL,
    exam_session_id VARCHAR(64) NOT NULL,
    violation_type VARCHAR(50) NOT NULL, -- 'HEAD_TURN_LEFT', 'HEAD_TURN_RIGHT', 'HEAD_DOWN', 'GAZE_AWAY', 'MULTIPLE_FACES', 'NO_FACE', 'IDENTITY_SWAP'
    duration_seconds FLOAT NOT NULL,     -- Thời gian vi phạm kéo dài (giây)
    evidence_image_path VARCHAR(500),    -- Đường dẫn lưu ảnh chụp vi phạm làm bằng chứng
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
);
```
