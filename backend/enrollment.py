"""Máy trạng thái quy trình Đăng ký khuôn mặt eKYC (Enrollment State Machine).

Quản lý chuyển đổi qua 4 giai đoạn chuẩn PoC:
  (a) CAMERA_CHECK: Kiểm tra tín hiệu, độ phân giải, FPS, độ sáng/nhiễu
  (b) FACE_QUALITY (FQA): Căn mặt vào Oval chuẩn, loại bỏ vật cản, chốt mốc kích thước
  (c) LIVENESS (Turn Left & Turn Right ngẫu nhiên): Bắt chuyển động quay đầu thật
  (d) ZOOM_IN: Phóng to oval, yêu cầu người dùng tiến gần, chụp ảnh HD + tổng kết
"""
from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import config as C
from camera_quality import CameraMonitor, CameraResult, ClientStats
from face_analyzer import FaceAnalyzer, FaceResult, quality_checks, check_continuous_face_quality


class Stage(str, Enum):
    CAMERA_CHECK = "camera_check"
    FACE_QUALITY = "face_quality"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    RECENTER = "recenter"
    ZOOM_IN = "zoom_in"
    FLASHING = "flashing"
    CAPTURE = "capture"
    FAILED = "failed"


class EnrollmentSession:
    """Phiên làm việc đăng ký eKYC của một người dùng."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.created_at = time.time()
        self.stage = Stage.CAMERA_CHECK
        self.previous_stage = Stage.CAMERA_CHECK

        # Bộ thử thách quay đầu ngẫu nhiên: [TURN_LEFT, TURN_RIGHT] hoặc [TURN_RIGHT, TURN_LEFT]
        turns = [Stage.TURN_LEFT, Stage.TURN_RIGHT]
        random.shuffle(turns)
        self.challenge_sequence = turns
        self.current_turn_index = 0

        # Mốc và bộ đếm
        self.camera_monitor = CameraMonitor()
        self.face_analyzer: Optional[FaceAnalyzer] = None
        self.stage_start_time = time.time()
        self.stable_start_time: Optional[float] = None
        self.fqa_consecutive_passes = 0
        self.turn_hold_counter = 0
        self.lost_face_counter = 0
        self.attempts = 0

        # Dữ liệu ghi nhận
        self.baseline_face_h: Optional[float] = None
        self.baseline_persp: Optional[float] = None
        self.max_left_yaw = 0.0
        self.max_right_yaw = 0.0
        self.max_zoom_growth = 1.0
        self.history: Dict[str, Any] = {
            "session_id": session_id,
            "challenges": [t.value for t in self.challenge_sequence],
            "camera": {},
            "fqa": {},
            "liveness": {},
            "zoom": {},
            "timings": {},
        }

    def _get_analyzer(self) -> FaceAnalyzer:
        if self.face_analyzer is None:
            self.face_analyzer = FaceAnalyzer()
        return self.face_analyzer

    def close(self) -> None:
        if self.face_analyzer is not None:
            self.face_analyzer.close()
            self.face_analyzer = None

    def current_oval(self) -> dict:
        """Kích thước và toạ độ Oval tương ứng với từng giai đoạn."""
        if self.stage == Stage.ZOOM_IN:
            return C.OVAL_ZOOM
        return C.OVAL_NORMAL

    def process_frame(self, frame_bgr: Any, client_stats: ClientStats) -> Dict[str, Any]:
        """Xử lý từng frame từ client gửi lên và trả về lệnh điều khiển UI."""
        now = time.time()
        cam_res: CameraResult = self.camera_monitor.evaluate(frame_bgr, client_stats)

        # Luôn phân tích khuôn mặt để kiểm tra chất lượng liên tục trong mọi frame
        analyzer = self._get_analyzer()
        oval = self.current_oval()
        face_res = analyzer.analyze(frame_bgr, oval)

        # Lỗi camera nghiêm trọng (đen/đóng băng) -> luôn giữ hoặc lùi về CAMERA_CHECK
        if not cam_res.signal_ok:
            if self.stage != Stage.CAMERA_CHECK and self.stage != Stage.FAILED:
                self.stage = Stage.CAMERA_CHECK
                self.stable_start_time = None
                self.fqa_consecutive_passes = 0
            return self._build_response(
                message=cam_res.message,
                color="red",
                cam_res=cam_res,
                face_res=face_res,
                progress=0.0,
            )

        # --- GIAI ĐOẠN (a): CAMERA_CHECK ---
        if self.stage == Stage.CAMERA_CHECK:
            # Chặn giả mạo ngay từ đầu nếu phát hiện ảnh in hoặc màn hình phát lại
            if face_res is not None and face_res.detected and face_res.anti_spoof and not face_res.anti_spoof.is_real:
                self.stable_start_time = None
                return self._build_response(
                    message=face_res.anti_spoof.message,
                    color="red" if face_res.anti_spoof.severity == "error" else "yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

            if cam_res.all_ok:
                if self.stable_start_time is None:
                    self.stable_start_time = now
                elapsed = now - self.stable_start_time
                progress = min(1.0, elapsed / C.CAMERA_STABLE_SEC)
                if elapsed >= C.CAMERA_STABLE_SEC:
                    self.history["camera"] = cam_res.metrics
                    self.history["timings"]["camera_check"] = round(now - self.stage_start_time, 2)
                    self.stage = Stage.FACE_QUALITY
                    self.stage_start_time = now
                    self.stable_start_time = None
                    self.fqa_consecutive_passes = 0
                    return self._build_response(
                        message="Camera đạt chuẩn! Đưa khuôn mặt vào khung Oval",
                        color="cyan",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=1.0,
                    )
                return self._build_response(
                    message=f"Đang kiểm tra chất lượng Camera ({int(progress * 100)}%)...",
                    color="green",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=progress,
                )
            else:
                self.stable_start_time = None
                color = "red" if cam_res.severity == "error" else "yellow"
                return self._build_response(
                    message=cam_res.message,
                    color=color,
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

        # --- GIAI ĐOẠN (b): FACE_QUALITY (FQA VỚI 15 FRAMES SMOOTHING) ---
        if self.stage == Stage.FACE_QUALITY:
            checks, msg, sev = quality_checks(face_res, require_oval=True, is_turning=False)
            all_fqa_ok = bool(checks) and all(checks.values())

            if all_fqa_ok:
                self.fqa_consecutive_passes += 1
                progress = min(1.0, self.fqa_consecutive_passes / C.FQA_CONSECUTIVE_FRAMES)
                if self.fqa_consecutive_passes >= C.FQA_CONSECUTIVE_FRAMES:
                    # Chốt mốc ban đầu
                    self.baseline_face_h = face_res.face_h
                    self.baseline_persp = face_res.perspective_ratio
                    self.history["fqa"] = {
                        "brightness": face_res.brightness,
                        "brightness_mean": face_res.brightness_mean,
                        "brightness_std": face_res.brightness_std,
                        "sharpness": face_res.sharpness,
                        "scale_ratio": face_res.scale_ratio,
                        "baseline_face_h": self.baseline_face_h,
                        "baseline_persp": self.baseline_persp,
                    }
                    self.history["timings"]["face_quality"] = round(now - self.stage_start_time, 2)
                    self.stage = self.challenge_sequence[0]
                    self.current_turn_index = 0
                    self.stage_start_time = now
                    self.turn_hold_counter = 0
                    self.stable_start_time = None
                    return self._build_response(
                        message=self._turn_instruction(self.stage),
                        color="yellow",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=0.0,
                        fqa_passed_sound=True,
                    )
                return self._build_response(
                    message=f"Khuôn mặt hợp lệ! Giữ yên ({int(progress * 100)}%)...",
                    color="green",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=progress,
                )
            else:
                self.fqa_consecutive_passes = max(0, self.fqa_consecutive_passes - 1)
                curr_prog = min(1.0, self.fqa_consecutive_passes / C.FQA_CONSECUTIVE_FRAMES)
                color = "red" if sev == "error" else ("gray" if not face_res.detected else "yellow")
                return self._build_response(
                    message=msg,
                    color=color,
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=curr_prog,
                )

        # --- GIAI ĐOẠN (c): LIVENESS QUAY ĐẦU (TURN_LEFT / TURN_RIGHT / RECENTER) ---
        if self.stage in (Stage.TURN_LEFT, Stage.TURN_RIGHT):
            # Kiểm tra timeout
            if now - self.stage_start_time > C.CHALLENGE_TIMEOUT_SEC:
                return self._handle_challenge_timeout(cam_res, face_res)

            # KIỂM TRA CHẤT LƯỢNG KHUÔN MẶT TRONG SUỐT QUÁ TRÌNH QUAY ĐẦU
            cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_res, is_turning=True)
            if not cont_ok:
                self.turn_hold_counter = 0
                return self._build_response(
                    message=cont_msg,
                    color="red" if cont_sev == "error" else "yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

            if not face_res.detected or face_res.yaw is None:
                self.lost_face_counter += 1
                if self.lost_face_counter > C.MAX_LOST_FRAMES:
                    return self._handle_challenge_timeout(cam_res, face_res, "Mất dấu khuôn mặt khi quay")
                return self._build_response(
                    message=self._turn_instruction(self.stage),
                    color="yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )
            self.lost_face_counter = 0

            # Cập nhật số đo góc quay lớn nhất đạt được
            yaw = face_res.yaw
            if yaw > 0:
                self.max_left_yaw = max(self.max_left_yaw, yaw)
            else:
                self.max_right_yaw = max(self.max_right_yaw, abs(yaw))

            # Kiểm tra góc quay đúng hướng
            target_reached = (
                (self.stage == Stage.TURN_LEFT and yaw >= C.TURN_YAW_DEG)
                or (self.stage == Stage.TURN_RIGHT and yaw <= -C.TURN_YAW_DEG)
            )

            if target_reached:
                self.turn_hold_counter += 1
                progress = min(1.0, self.turn_hold_counter / C.TURN_HOLD_FRAMES)
                if self.turn_hold_counter >= C.TURN_HOLD_FRAMES:
                    # Ghi nhận kết quả thử thách này
                    self.history["liveness"][self.stage.value] = {
                        "achieved_yaw": round(yaw, 1),
                        "duration": round(now - self.stage_start_time, 2),
                    }
                    self.stage = Stage.RECENTER
                    self.stage_start_time = now
                    self.turn_hold_counter = 0
                    return self._build_response(
                        message="Đã nhận diện! Hãy quay đầu nhìn thẳng lại vào camera",
                        color="green",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=1.0,
                    )
                return self._build_response(
                    message="Giữ nguyên góc quay...",
                    color="green",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=progress,
                )
            else:
                self.turn_hold_counter = 0
                return self._build_response(
                    message=self._turn_instruction(self.stage),
                    color="yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

        # Trạng thái phụ: RECENTER (Yêu cầu nhìn thẳng lại giữa 2 lần quay hoặc trước khi zoom)
        if self.stage == Stage.RECENTER:
            # KIỂM TRA CHẤT LƯỢNG KHUÔN MẶT TRONG SUỐT QUÁ TRÌNH NHÌN THẲNG LẠI
            cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_res, is_turning=False)
            if not cont_ok:
                return self._build_response(
                    message=cont_msg,
                    color="red" if cont_sev == "error" else "yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

            if not face_res.detected or face_res.yaw is None:
                return self._build_response(
                    message="Hãy quay đầu nhìn thẳng vào camera",
                    color="yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )
            if abs(face_res.yaw) <= C.RECENTER_YAW_DEG:
                # Kiểm tra tiếp theo là lượt quay thứ 2 hay chuyển sang ZOOM
                if self.current_turn_index == 0:
                    self.current_turn_index = 1
                    self.stage = self.challenge_sequence[1]
                    self.stage_start_time = now
                    self.turn_hold_counter = 0
                    return self._build_response(
                        message=self._turn_instruction(self.stage),
                        color="yellow",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=0.0,
                    )
                else:
                    # Đã hoàn tất cả 2 hướng quay -> chuyển sang ZOOM_IN
                    self.history["timings"]["liveness_total"] = round(now - self.created_at, 2)
                    self.stage = Stage.ZOOM_IN
                    self.stage_start_time = now
                    self.stable_start_time = None
                    return self._build_response(
                        message="Khung Oval đang phóng to. Hãy TIẾN GẦN camera hơn để mặt lọt vừa Oval lớn!",
                        color="yellow",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=0.0,
                    )
            return self._build_response(
                message="Hãy nhìn thẳng vào camera để tiếp tục...",
                color="yellow",
                cam_res=cam_res,
                face_res=face_res,
                progress=0.0,
            )

        # --- GIAI ĐOẠN (d): ZOOM_IN (Phóng to Oval, yêu cầu tiến gần, chụp ảnh HD) ---
        if self.stage == Stage.ZOOM_IN:
            if now - self.stage_start_time > C.ZOOM_TIMEOUT_SEC:
                return self._handle_challenge_timeout(cam_res, face_res, "Quá thời gian tiến gần camera")

            # KIỂM TRA CHẤT LƯỢNG KHUÔN MẶT TRONG SUỐT QUÁ TRÌNH TIẾN GẦN
            cont_ok, cont_msg, cont_sev = check_continuous_face_quality(face_res, is_turning=False)
            if not cont_ok:
                self.stable_start_time = None
                return self._build_response(
                    message=cont_msg,
                    color="red" if cont_sev == "error" else "yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

            checks, msg, sev = quality_checks(face_res, require_oval=True)

            # Tính mức tăng trưởng chiều cao khuôn mặt so với mốc ban đầu
            base_h = self.baseline_face_h or 0.3
            growth = face_res.face_h / max(0.05, base_h)
            self.max_zoom_growth = max(self.max_zoom_growth, growth)
            growth_ok = growth >= C.ZOOM_MIN_GROWTH

            # Điều kiện hoàn thành Zoom:
            # 1) Mặt lớn hơn ít nhất 25% so với ban đầu
            # 2) Mặt nằm gọn trong khung Oval phóng to (checks inside_oval, centered, straight, sharp, etc.)
            ready_to_capture = checks.get("face_detected", False) and checks.get("inside_oval", False) and checks.get("sharpness_ok", False) and checks.get("head_straight", False) and growth_ok

            if ready_to_capture:
                if self.stable_start_time is None:
                    self.stable_start_time = now
                elapsed = now - self.stable_start_time
                progress = min(1.0, elapsed / C.HOLD_SEC)

                if elapsed >= C.HOLD_SEC:
                    self.stage = Stage.FLASHING
                    self.history["zoom"] = {
                        "baseline_face_h": base_h,
                        "final_face_h": face_res.face_h,
                        "growth_ratio": round(growth, 2),
                        "baseline_persp": self.baseline_persp,
                        "final_persp": face_res.perspective_ratio,
                    }
                    self.history["timings"]["zoom"] = round(now - self.stage_start_time, 2)
                    self.stage_start_time = now

                    return self._build_response(
                        message="Giữ yên khuôn mặt! Chuẩn bị quét ánh sáng màu...",
                        color="green",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=1.0,
                    )
                return self._build_response(
                    message=f"Tuyệt vời! Giữ yên khuôn mặt ({int(progress * 100)}%)...",
                    color="green",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=progress,
                )
            else:
                self.stable_start_time = None
                if not growth_ok:
                    hint = f"Hãy tiến lại gần camera hơn nữa (hiện đạt {int(growth * 100)}% / 125%)"
                    return self._build_response(
                        message=hint,
                        color="yellow",
                        cam_res=cam_res,
                        face_res=face_res,
                        progress=0.0,
                    )
                return self._build_response(
                    message=msg,
                    color="red" if sev == "error" else "yellow",
                    cam_res=cam_res,
                    face_res=face_res,
                    progress=0.0,
                )

        # --- GIAI ĐOẠN (e): FLASHING (Quét ánh sáng màu quang học) ---
        if self.stage == Stage.FLASHING:
            return self._build_response(
                message="Đang quét ánh sáng quang học... Hãy nhìn thẳng vào màn hình",
                color="green",
                cam_res=cam_res,
                face_res=face_res,
                progress=1.0,
            )

        # Giai đoạn CAPTURE hoặc FAILED
        if self.stage == Stage.FAILED:
            failed_msg = (
                self.history.get("optical_liveness", {}).get("message")
                or "Chưa đạt chuẩn phản xạ quang học trên da. Vui lòng thử lại."
            )
            return self._build_response(
                message=failed_msg,
                color="red",
                cam_res=cam_res,
                face_res=face_res,
                progress=0.0,
            )

        return self._build_response(
            message="Xác thực sinh trắc học thành công! Đang chuyển sang Bước 5...",
            color="green",
            cam_res=cam_res,
            face_res=face_res,
            progress=1.0,
        )

    def apply_optical_result(
        self,
        passed: bool,
        correlation: float,
        amplitude: float,
        verdict: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cập nhật kết quả xác thực quang học từ endpoint verify."""
        now = time.time()
        self.history["optical_liveness"] = {
            "passed": passed,
            "correlation_score": round(correlation, 3),
            "amplitude": round(amplitude, 2),
            "verdict": verdict,
            "details": details or {},
        }
        if passed:
            self.stage = Stage.CAPTURE
            self.history["timings"]["optical"] = round(now - self.stage_start_time, 2)
            self.history["timings"]["total"] = round(now - self.created_at, 2)
        else:
            self.stage = Stage.FAILED

    def _turn_instruction(self, stage: Stage) -> str:
        if stage == Stage.TURN_LEFT:
            return "Hơi quay đầu sang TRÁI (theo hướng bạn nhìn) khoảng 20°"
        if stage == Stage.TURN_RIGHT:
            return "Hơi quay đầu sang PHẢI (theo hướng bạn nhìn) khoảng 20°"
        return "Hãy làm theo hướng dẫn"

    def _handle_challenge_timeout(self, cam_res: CameraResult, face_res: Optional[FaceResult], extra: str = "") -> Dict[str, Any]:
        self.attempts += 1
        now = time.time()
        if self.attempts >= C.MAX_ATTEMPTS:
            self.stage = Stage.FAILED
            msg = f"Đăng ký không thành công ({extra or 'Quá thời gian thử thách'}). Vui lòng bấm Thử lại từ đầu."
            return self._build_response(msg, color="red", cam_res=cam_res, face_res=face_res, progress=0.0)
        else:
            # Quay về bước FQA để căn chỉnh lại
            self.stage = Stage.FACE_QUALITY
            self.stage_start_time = now
            self.stable_start_time = None
            self.fqa_consecutive_passes = 0
            self.turn_hold_counter = 0
            self.lost_face_counter = 0
            msg = f"{extra or 'Chưa hoàn thành thử thách'}. Đưa khuôn mặt vào khung Oval để thực hiện lại (lần {self.attempts}/{C.MAX_ATTEMPTS})"
            return self._build_response(msg, color="yellow", cam_res=cam_res, face_res=face_res, progress=0.0)

    def _build_response(
        self,
        message: str,
        color: str,
        cam_res: CameraResult,
        face_res: Optional[FaceResult],
        progress: float,
        fqa_passed_sound: bool = False,
    ) -> Dict[str, Any]:
        """Tạo gói JSON trả về cho frontend."""
        f_checks = {}
        f_metrics = {}
        if face_res is not None:
            is_turning = self.stage in (Stage.TURN_LEFT, Stage.TURN_RIGHT)
            chk, _, _ = quality_checks(
                face_res,
                require_oval=(self.stage in (Stage.FACE_QUALITY, Stage.ZOOM_IN)),
                is_turning=is_turning,
            )
            f_checks = chk
            f_metrics = {
                "face_count": face_res.face_count,
                "yaw": face_res.yaw,
                "pitch": face_res.pitch,
                "roll": face_res.roll,
                "fill": round(face_res.fill, 2),
                "scale_ratio": round(face_res.scale_ratio, 2),
                "corners_inside": face_res.corners_inside,
                "brightness": round(face_res.brightness, 1),
                "brightness_mean": round(face_res.brightness_mean, 1),
                "brightness_std": round(face_res.brightness_std, 1),
                "sharpness": round(face_res.sharpness, 1),
                "oval_dist": round(face_res.oval_dist, 2),
                "mask_detected": face_res.mask_detected,
                "sunglasses_detected": face_res.sunglasses_detected,
                "glare_detected": face_res.glare_detected,
                "hand_occlusion": face_res.hand_occlusion,
                "perspective_ratio": face_res.perspective_ratio,
            }
            if face_res.anti_spoof:
                f_metrics["anti_spoof_real_prob"] = round(face_res.anti_spoof.real_prob, 3)
                f_metrics["anti_spoof_print_prob"] = round(face_res.anti_spoof.print_prob, 3)
                f_metrics["anti_spoof_replay_prob"] = round(face_res.anti_spoof.replay_prob, 3)
                f_metrics["anti_spoof_type"] = face_res.anti_spoof.spoof_type
                f_metrics["depth_3d_delta"] = round(face_res.anti_spoof.depth_3d_delta, 4)
                f_metrics["depth_3d_ok"] = face_res.anti_spoof.depth_3d_ok
                f_metrics["moire_ratio"] = round(face_res.anti_spoof.moire_ratio, 3)
                f_metrics["screen_detected"] = face_res.anti_spoof.screen_detected
                f_metrics["bezel_detected"] = face_res.anti_spoof.bezel_detected

        turn_arrow: Optional[str] = None
        if self.stage == Stage.TURN_LEFT:
            turn_arrow = "left"
        elif self.stage == Stage.TURN_RIGHT:
            turn_arrow = "right"

        resp: Dict[str, Any] = {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "color": color,
            "message": message,
            "progress": round(progress, 2),
            "oval": self.current_oval(),
            "turn_arrow": turn_arrow,
            "fqa_passed_sound": fqa_passed_sound,
            "camera_checks": cam_res.checks,
            "camera_metrics": cam_res.metrics,
            "face_checks": f_checks,
            "face_metrics": f_metrics,
            "anti_spoof": {
                "is_real": face_res.anti_spoof.is_real,
                "real_prob": round(face_res.anti_spoof.real_prob, 3),
                "print_prob": round(face_res.anti_spoof.print_prob, 3),
                "replay_prob": round(face_res.anti_spoof.replay_prob, 3),
                "spoof_type": face_res.anti_spoof.spoof_type,
                "depth_3d_ok": face_res.anti_spoof.depth_3d_ok,
                "message": face_res.anti_spoof.message,
                "severity": face_res.anti_spoof.severity,
            } if (face_res and face_res.anti_spoof) else None,
            "parts_status": face_res.parts_status if face_res else {},
            "occluded_part_name": face_res.occluded_part_name if face_res else None,
            "keypoints": face_res.keypoints if face_res else {},
        }

        # Nếu hoàn thành hoặc thất bại, gửi kèm summary
        if self.stage == Stage.CAPTURE:
            resp["summary"] = self.history
        return to_native_types(resp)


def to_native_types(val: Any) -> Any:
    """Đệ quy chuyển đổi kiểu numpy sang kiểu Python chuẩn để JSON serializer không lỗi."""
    import numpy as np

    if isinstance(val, dict):
        return {k: to_native_types(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_native_types(v) for v in val]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val
