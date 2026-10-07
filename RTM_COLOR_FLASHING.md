# RTM: REQUIREMENTS TRACEABILITY MATRIX (PHASE 4)

> **Ghi chú trạng thái (branch `quoc`):** Đây là tài liệu thiết kế mục tiêu, có thể không khớp với code hiện tại. Hiện trạng xem [README.md](README.md) và [PIPELINE.md](PIPELINE.md); phần chưa làm xem [ROADMAP.md](ROADMAP.md).

> **Ma trận Truy vết Yêu cầu Phần mềm (Requirements Traceability Matrix)**  
> **Dự án:** Active Optical Color Flashing Liveness Detection  
> **Tiêu chuẩn tuân thủ:** DO-178C / ASPICE Traceability Level 3  
> **Phiên bản:** 1.0.0  
> **Tác giả:** Ban Kỹ Thuật (Mini Engineering Board)

---

## 1. BẢNG MA TRẬN TRUY VẾT TOÀN DIỆN (FULL TRACEABILITY MATRIX)

| ID Yêu cầu (FSD) | Nội dung Yêu cầu Chức năng | ID Thiết kế (TDD) | Thành phần Mã nguồn (Source Code) | Hàm / Lớp Cụ thể (Function/Class) | Test Case Kiểm thử (Pytest) | Trạng thái Nghiệm thu |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **FR-01** | Sinh chuỗi 3 màu ngẫu nhiên và Token bảo mật HMAC | **TDD-01** | `backend/color_challenge.py` | `ChallengeManager.create_challenge()`<br>`ChallengeManager.verify_token()` | `tests/test_color_challenge.py::test_create_challenge_valid`<br>`tests/test_color_challenge.py::test_token_verification` | [Sẵn sàng triển khai] |
| **FR-02** | Hiển thị ánh sáng an toàn và thu thập 3 khung hình | **TDD-02** | `web/app.js`<br>`web/styles.css`<br>`web/index.html` | `triggerColorFlashingSequence()`<br>`captureFlashingFrame()`<br>`#flashingOverlay` | Kiểm thử giao diện tự động & Kiểm thử tích hợp End-to-End | [Sẵn sàng triển khai] |
| **FR-03** | Trích xuất 3 vùng ROI da (trán, má trái, má phải) & 1 vùng nền | **TDD-03** | `backend/optical_analyzer.py` | `extract_facial_skin_rois()`<br>`extract_background_roi()` | `tests/test_optical_analyzer.py::test_extract_rois_valid_face` | [Sẵn sàng triển khai] |
| **FR-04** | Tính toán biến thiên quang phổ vi sai và bù trừ AWB | **TDD-04** | `backend/optical_analyzer.py` | `compute_differential_chrominance()`<br>`compensate_awb_drift()` | `tests/test_optical_analyzer.py::test_differential_computation` | [Sẵn sàng triển khai] |
| **FR-05** | Đánh giá hệ số tương quan Pearson đa chiều ($r \ge 0.65$) | **TDD-05** | `backend/optical_analyzer.py` | `evaluate_optical_liveness()`<br>`pearson_correlation()` | `tests/test_optical_analyzer.py::test_pearson_real_face_passes`<br>`tests/test_optical_analyzer.py::test_pearson_spoof_fails` | [Sẵn sàng triển khai] |
| **FR-06** | Chuyển tiếp trạng thái quy trình từ `ZOOM_IN` sang `CAPTURE` | **TDD-06** | `backend/enrollment.py`<br>`backend/app.py` | `EnrollmentSession.apply_optical_liveness()`<br>`POST /api/enroll/color_verify` | `tests/test_enrollment_flashing.py::test_enrollment_flashing_transition` | [Sẵn sàng triển khai] |
| **FR-07** | Xử lý ngoại lệ: Token hết hạn, mất khuôn mặt, ngược sáng | **TDD-07** | `backend/color_challenge.py`<br>`backend/optical_analyzer.py` | `OpticalAnalyzer.verify()` | `tests/test_optical_analyzer.py::test_face_lost_exception`<br>`tests/test_color_challenge.py::test_expired_token` | [Sẵn sàng triển khai] |

---

## 2. KẾ HOẠCH BẢO ĐẢM ĐỘ BAO PHỦ KIỂM THỬ (TEST COVERAGE PLAN)
- $100\%$ các hàm nghiệp vụ trong `backend/color_challenge.py` và `backend/optical_analyzer.py` phải có Unit Test độc lập.
- Kiểm thử dữ liệu giả lập (Synthetic Test Cases):
  1. Trường hợp người thật: Tín hiệu da biến thiên cùng pha với chuỗi màu ($\implies r > 0.80$).
  2. Trường hợp ảnh in: Tín hiệu da không đổi qua cả 3 khung hình ($\implies \|\Delta \mathbf{S}\| < 2.0$, báo lỗi phát lại).
  3. Trường hợp màn hình phát lại: Tín hiệu biến thiên ngược pha hoặc ngẫu nhiên ($\implies r < 0.30$, báo lỗi giả mạo).
  4. Trường hợp token bị giả mạo hoặc quá hạn $15\text{ giây}$ ($\implies 400 Bad Request$).

---
*Ma trận RTM hoàn thành. Sẵn sàng bước vào Phase 5: Micro-Implementation & Testing.*
