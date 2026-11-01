"""ỨNG DỤNG TEST CAMERA 3D & 2D INTEL REALSENSE (PYREALSENSE2 + OPENCV)
Hiển thị đồng thời:
  - Khung hình 2D RGB Màu sắc chuẩn (Auto-Exposure sáng rõ)
  - Khung hình 3D Bản đồ độ sâu (Colorized Depth Map)
  - Đo khoảng cách chính xác từng milimét tại tâm hoặc tại vị trí di chuột
=============================================================================
Phím tắt điều khiển:
  [q] hoặc [ESC] : Thoát chương trình
  [s]           : Chụp và lưu ảnh 2D, 3D ra file PNG
  [c]           : Đổi bảng màu bản đồ độ sâu 3D (Jet, Classic, Cold, Warm...)
  [Click chuột] : Đo khoảng cách vật lý tại bất kỳ điểm nào trên khung hình
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    import winsound
    def play_beep(freq=1200, duration=100):
        winsound.Beep(freq, duration)
except Exception:
    def play_beep(freq=1200, duration=100):
        pass


class RealSenseTester:
    """Bộ kiểm tra trực quan luồng 2D RGB và 3D Depth của camera Intel RealSense."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        # Tọa độ chuột dùng để đo khoảng cách tương tác
        self.mouse_pos = (width // 2, height // 2)
        self.selected_point = None

        # Bảng màu Colorizer
        self.color_scheme_idx = 0
        self.color_schemes = [
            ("Jet (Xanh gần -> Đỏ xa)", 0),
            ("Classic (Cầu vồng chuẩn)", 1),
            ("White to Black", 2),
            ("Black to White", 3),
            ("Bio (Y sinh)", 4),
            ("Cold (Màu lạnh)", 5),
            ("Warm (Màu ấm)", 6),
        ]

        # Bộ lọc làm mịn độ sâu (Spatial & Temporal Filters)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # Biến đếm FPS và thông báo chụp ảnh
        self.fps_val = 0.0
        self.last_time = time.time()
        self.frame_count = 0
        self.notification_text = ""
        self.notification_time = 0.0

    def on_mouse_event(self, event, x, y, flags, param):
        """Bắt sự kiện di chuột hoặc nhấp chuột để đo khoảng cách theo tọa độ."""
        if event == cv2.EVENT_MOUSEMOVE:
            # Nếu di chuột trên nửa bên trái (RGB) hoặc nửa bên phải (Depth)
            norm_x = x % self.width
            norm_y = min(self.height - 1, max(0, y))
            self.mouse_pos = (norm_x, norm_y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            norm_x = x % self.width
            norm_y = min(self.height - 1, max(0, y))
            self.selected_point = (norm_x, norm_y)
            play_beep(1400, 80)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_point = None

    def run(self):
        print("=" * 72)
        print("  KHỞI TẠO KẾT NỐI CAMERA INTEL REALSENSE (NATIVE PYREALSENSE2)")
        print("=" * 72)

        pipeline = rs.pipeline()
        config = rs.config()

        # Kiểm tra thiết bị có đang kết nối không
        ctx = rs.context()
        devices = ctx.devices
        if len(devices) == 0:
            print("\n[LỖI] Không tìm thấy camera Intel RealSense nào được cắm vào máy tính!")
            print("Vui lòng kiểm tra cáp USB và cắm vào cổng USB 3.0 (màu xanh dương).")
            return

        device = devices[0]
        dev_name = device.get_info(rs.camera_info.name)
        dev_serial = device.get_info(rs.camera_info.serial_number)
        dev_fw = device.get_info(rs.camera_info.firmware_version)
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)

        print(f"  • Tên thiết bị: {dev_name}")
        print(f"  • Số Serial : {dev_serial}")
        print(f"  • Firmware   : {dev_fw}")
        print(f"  • Cổng USB   : USB {usb_type}")

        if "2." in usb_type:
            print("  ⚠️ [CẢNH BÁO]: Bạn đang cắm camera vào cổng USB 2.0!")
            print("      Vui lòng chuyển sang cổng USB 3.0 (chân màu xanh hoặc có chữ SS)")
            print("      để đạt độ sáng tối đa và băng thông 3D đầy đủ.")
        else:
            print("  ✓ Cổng kết nối đạt chuẩn SuperSpeed USB 3.x!")

        # Cấu hình luồng Color 2D (BGR8) và luồng Depth 3D (Z16)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"\n[LỖI KHỞI ĐỘNG]: {e}")
            print("Gợi ý: Nếu phần mềm 'RealSense Viewer' đang mở, hãy tắt nó đi vì camera chỉ cho phép 1 ứng dụng truy cập tại 1 thời điểm.")
            return

        # Lấy depth scale (thường là 0.001m = 1mm đối với D435)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # BẬT TỰ ĐỘNG PHƠI SÁNG (AUTO-EXPOSURE) ĐỂ KHÔNG BỊ TỐI THUI
        try:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print("  ✓ Đã kích hoạt Auto-Exposure phần cứng cho cảm biến RGB màu!")
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception as err:
            print("  Lưu ý khi bật Auto-Exposure RGB:", err)

        # Căn chỉnh (Align) luồng Depth 3D trùng khít 100% với luồng Color 2D
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Bộ tô màu độ sâu (Colorizer)
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, float(self.color_scheme_idx))

        window_name = f"Test Intel RealSense 3D & 2D - [{dev_name}]"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse_event)

        print("\n" + "=" * 72)
        print("  ĐANG STREAM DỮ LIỆU... NHẤN [Q] ĐỂ THOÁT.")
        print("=" * 72 + "\n")

        play_beep(1000, 150)

        output_dir = os.path.join(os.path.dirname(__file__), "realsense_snapshots")
        os.makedirs(output_dir, exist_ok=True)

        try:
            while True:
                # 1. Đợi nhận cặp frame Color và Depth đã đồng bộ
                frames = pipeline.wait_for_frames(5000)

                # Đồng bộ tọa độ Depth khớp 1:1 với Color
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Lọc làm mịn màng độ sâu
                depth_filtered = self.spatial_filter.process(depth_frame)
                depth_filtered = self.temporal_filter.process(depth_filtered)
                depth_filtered = self.hole_filling.process(depth_filtered)

                # Chuyển đổi sang mảng Numpy
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = np.asanyarray(colorizer.colorize(depth_filtered).get_data())

                # Tính toán FPS
                self.frame_count += 1
                now_t = time.time()
                if now_t - self.last_time >= 1.0:
                    self.fps_val = self.frame_count / (now_t - self.last_time)
                    self.frame_count = 0
                    self.last_time = now_t

                # 2. Đo khoảng cách tại các điểm mấu chốt
                # Điểm tâm màn hình (Center Point)
                cx, cy = self.width // 2, self.height // 2
                dist_center_m = depth_frame.get_distance(cx, cy)

                # Điểm đang di chuột hoặc đã nhấp chọn (Interactive Point)
                target_pt = self.selected_point if self.selected_point else self.mouse_pos
                dist_target_m = depth_frame.get_distance(target_pt[0], target_pt[1])

                # 3. Vẽ thông tin đo đạc lên khung 2D RGB (Bên trái)
                # Dấu chữ thập tâm
                cv2.drawMarker(color_image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # Vẽ điểm tương tác chuột nếu có
                if self.selected_point or (self.mouse_pos != (cx, cy)):
                    px, py = target_pt
                    pt_color = (0, 140, 255) if self.selected_point else (0, 255, 0)
                    cv2.circle(color_image, (px, py), 6, pt_color, -1)
                    cv2.circle(color_image, (px, py), 10, (255, 255, 255), 2)
                    cv2.circle(depth_colormap, (px, py), 6, pt_color, -1)
                    cv2.circle(depth_colormap, (px, py), 10, (255, 255, 255), 2)

                    dist_txt = f"{dist_target_m * 100:.1f} cm ({dist_target_m:.2f} m)" if dist_target_m > 0 else "Ngoai pham vi"
                    cv2.putText(color_image, dist_txt, (px + 12, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(color_image, dist_txt, (px + 12, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, pt_color, 2, cv2.LINE_AA)

                # Tiêu đề khung 2D RGB
                cv2.rectangle(color_image, (0, 0), (self.width, 36), (18, 18, 24), -1)
                cv2.putText(color_image, "1. KHUNG HINH 2D RGB (COLOR)", (12, 24),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

                # Tiêu đề khung 3D Depth
                cv2.rectangle(depth_colormap, (0, 0), (self.width, 36), (18, 18, 24), -1)
                cv2.putText(depth_colormap, f"2. BAN DO DO SAU 3D ({self.color_schemes[self.color_scheme_idx][0]})",
                            (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

                # 4. Ghép 2 khung hình ngang nhau (Side-by-Side)
                combined_canvas = np.hstack((color_image, depth_colormap))

                # HUD thông số tổng quát bên dưới đáy màn hình
                canvas_h, canvas_w = combined_canvas.shape[:2]
                hud_h = 75
                hud_panel = np.zeros((hud_h, canvas_w, 3), dtype=np.uint8)
                hud_panel[:] = (12, 14, 20)

                # Dòng thông tin 1: Thiết bị, cổng kết nối và khoảng cách
                center_txt = f"Tam man hinh: {dist_center_m * 100:.1f} cm ({dist_center_m:.2f} m)" if dist_center_m > 0 else "Tam: Ngoai pham vi"
                hud_str1 = f"Camera: {dev_name} | Cong: USB {usb_type} | FPS: {self.fps_val:.1f} | {center_txt}"
                cv2.putText(hud_panel, hud_str1, (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                # Dòng hướng dẫn phím tắt
                hud_str2 = "[Click]: Do khoang cach tai diem bat ky | [S]: Chup anh | [C]: Doi mau 3D | [ESC/Q]: Thoat"
                cv2.putText(hud_panel, hud_str2, (16, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

                final_view = np.vstack((combined_canvas, hud_panel))

                # Hiển thị thông báo chụp ảnh nếu có
                if now_t - self.notification_time < 2.0:
                    cv2.rectangle(final_view, (canvas_w // 2 - 250, 45), (canvas_w // 2 + 250, 95), (0, 180, 50), -1)
                    cv2.rectangle(final_view, (canvas_w // 2 - 250, 45), (canvas_w // 2 + 250, 95), (255, 255, 255), 2)
                    cv2.putText(final_view, self.notification_text, (canvas_w // 2 - 220, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(window_name, final_view)

                # Bắt phím điều khiển
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # ESC hoặc Q
                    break

                elif key in (ord('c'), ord('C')):  # Đổi bảng màu 3D
                    self.color_scheme_idx = (self.color_scheme_idx + 1) % len(self.color_schemes)
                    colorizer.set_option(rs.option.color_scheme, float(self.color_schemes[self.color_scheme_idx][1]))
                    play_beep(1100, 80)

                elif key in (ord('s'), ord('S')):  # Chụp ảnh 2D & 3D lưu file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn_color = os.path.join(output_dir, f"realsense_color_{timestamp}.png")
                    fn_depth = os.path.join(output_dir, f"realsense_depth_{timestamp}.png")
                    fn_combo = os.path.join(output_dir, f"realsense_combined_{timestamp}.png")

                    cv2.imwrite(fn_color, color_image)
                    cv2.imwrite(fn_depth, depth_colormap)
                    cv2.imwrite(fn_combo, final_view)

                    play_beep(1600, 150)
                    self.notification_text = f"Da luu anh vao /realsense_snapshots!"
                    self.notification_time = time.time()
                    print(f"\n[DA LUU ANH]: {fn_combo}")

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("\n[THÔNG BÁO] Đã đóng camera Intel RealSense an toàn.")


if __name__ == "__main__":
    app = RealSenseTester(width=640, height=480, fps=30)
    app.run()
