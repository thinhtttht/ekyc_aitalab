# ROADMAP – Những việc chưa làm

Trạng thái tính đến branch `quoc`. Những gì đã có được mô tả trong [README.md](README.md). Các tài liệu
thiết kế gốc (SRS, RFC, PRD/FSD/TDD/RTM, Blueprint) mô tả hệ thống mục tiêu, không phải hiện trạng.

## Bảo mật (cần trước khi đưa ra ngoài môi trường lab)

- **`/api/users/enroll` không ràng buộc với phiên đăng ký.** Khi không có `session_id` hợp lệ, API tự trích xuất
  vector từ bất kỳ ảnh nào được gửi lên, nên bỏ qua toàn bộ bước kiểm tra người thật. Cần yêu cầu một phiên đã ở
  trạng thái `CAPTURE` và đã qua `verify_final_capture`, rồi chỉ dùng `final_embedding` của phiên đó.
- **`/api/verify/face` chỉ kiểm tra người thật bằng MiniFASNet trên một ảnh.** Chưa có thử thách chủ động
  (nháy màu / quay đầu) cho luồng xác thực như PIPELINE mô tả.
- Chưa có xác thực / phân quyền cho các API quản lý người dùng (liệt kê, xem, xoá).
- Ảnh chân dung được lưu nguyên dạng base64 trong SQLite, chưa mã hoá, chưa có chính sách thời gian lưu trữ.
- Chưa giới hạn tần suất gọi API và kích thước ảnh tải lên.

## Độ chính xác

- **Chưa hiệu chỉnh ngưỡng trên dữ liệu thật**: `ANTISPOOF_*_THRESH`, ngưỡng ArcFace 0.45, `FLASH_MIN_PEARSON`,
  độ nét. Cần một bộ dữ liệu nhỏ (người thật, ảnh in, phát lại trên màn hình) và đo APCER / BPCER theo ISO/IEC 30107-3.
- Ba heuristic chống giả mạo (độ sâu Z, Moiré, viền thiết bị) đang tắt. Chỉ bật lại sau khi đo được là chúng giảm
  APCER mà không làm tăng BPCER. Riêng độ sâu Z của MediaPipe là giá trị model suy ra từ mesh chuẩn, nên ảnh phẳng
  vẫn có thể cho mũi "nhô lên".
- MiniFASNet đang dùng một model (V2, scale 2.7). Bản gốc Silent-Face kết hợp V1SE (scale 4.0) và V2 để ổn định hơn.

## Hệ thống

- `verifier_analyzer` trong `app.py` là một instance FaceMesh dùng chung cho mọi request, ở chế độ video
  (`static_image_mode=False`). Với các ảnh đơn lẻ không liên quan nhau, nên dùng chế độ ảnh tĩnh và có khoá hoặc pool khi chạy nhiều luồng.
- Phiên đăng ký được giữ trong bộ nhớ một tiến trình, nên không chạy được nhiều worker hoặc nhiều máy.
- Chưa có CI chạy `pytest` tự động.
- `web/app.js` khoảng 1.900 dòng trong một file; nên tách module (camera, enrollment, verify, UI checklist).
