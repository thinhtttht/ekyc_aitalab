"""Máy trạng thái quy trình Đăng ký khuôn mặt eKYC (Enrollment State Machine).

  (a) CAMERA_CHECK: Kiểm tra tín hiệu, độ phân giải, FPS, độ sáng/nhiễu
  (b) FACE_QUALITY (FQA): Căn mặt vào Oval chuẩn, loại bỏ vật cản, chốt mốc kích thước
  (c) TURN_LEFT / TURN_RIGHT (thứ tự ngẫu nhiên) + RECENTER: Bắt chuyển động quay đầu thật
  (d) ZOOM_IN: Phóng to oval, yêu cầu người dùng tiến gần
  (e) FLASHING: Frontend chiếu chuỗi màu, kết quả được đưa vào qua `apply_optical_result`
  (f) CAPTURE: Kiểm định ảnh chân dung cuối (`verify_final_capture`) và trích xuất ArcFace
"""
from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

import config as C
from camera_quality import CameraMonitor, CameraResult, ClientStats
from face_analyzer import FaceAnalyzer, FaceResult, FaceVerdict, evaluate_face
from feature_extractor import get_arcface_extractor


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


TURN_STAGES = (Stage.TURN_LEFT, Stage.TURN_RIGHT)


def _color_for(severity: str) -> str:
    return "red" if severity == "error" else "yellow"


class EnrollmentSession:
    """Phiên làm việc đăng ký eKYC của một người dùng."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.created_at = time.time()
        self.stage = Stage.CAMERA_CHECK

        turns = [Stage.TURN_LEFT, Stage.TURN_RIGHT]
        random.shuffle(turns)
        self.challenge_sequence = turns
        self.current_turn_index = 0

        self.camera_monitor = CameraMonitor()
        self.face_analyzer: Optional[FaceAnalyzer] = None
        self.stage_start_time = time.time()
        self.stable_start_time: Optional[float] = None
        self.fqa_consecutive_passes = 0
        self.turn_hold_counter = 0
        self.lost_face_counter = 0
        self.attempts = 0

        self.baseline_face_h: Optional[float] = None
        self.baseline_persp: Optional[float] = None
        self.enrolled_image_bgr: Optional[np.ndarray] = None
        self.enrolled_face: Optional[FaceResult] = None
        self.final_embedding: Optional[np.ndarray] = None
        self.history: Dict[str, Any] = {
            "session_id": session_id,
            "challenges": [t.value for t in self.challenge_sequence],
            "camera": {},
            "fqa": {},
            "liveness": {},
            "zoom": {},
            "timings": {},
        }

        # Kết quả phân tích của frame đang xử lý (gán trong process_frame)
        self._now = self.created_at
        self._cam: Optional[CameraResult] = None
        self._face: Optional[FaceResult] = None
        self._verdict: Optional[FaceVerdict] = None

    def _get_analyzer(self) -> FaceAnalyzer:
        if self.face_analyzer is None:
            self.face_analyzer = FaceAnalyzer()
        return self.face_analyzer

    def close(self) -> None:
        if self.face_analyzer is not None:
            self.face_analyzer.close()
            self.face_analyzer = None

    def current_oval(self) -> dict:
        return C.OVAL_ZOOM if self.stage == Stage.ZOOM_IN else C.OVAL_NORMAL

    # ------------------------------------------------------------------
    # Vòng xử lý từng frame
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: Any, client_stats: ClientStats) -> Dict[str, Any]:
        """Xử lý một frame từ client và trả về lệnh điều khiển UI."""
        self._now = time.time()
        self._cam = self.camera_monitor.evaluate(frame_bgr, client_stats)
        self._face = self._get_analyzer().analyze(frame_bgr, self.current_oval())
        self._verdict = evaluate_face(
            self._face,
            require_oval=self.stage in (Stage.FACE_QUALITY, Stage.ZOOM_IN),
            is_turning=self.stage in TURN_STAGES,
        )

        # Lỗi camera nghiêm trọng (đen/đóng băng) -> lùi về CAMERA_CHECK
        if not self._cam.signal_ok:
            if self.stage not in (Stage.CAMERA_CHECK, Stage.FAILED):
                self.stage = Stage.CAMERA_CHECK
                self.stable_start_time = None
                self.fqa_consecutive_passes = 0
            return self._reply(self._cam.message, "red")

        handler = {
            Stage.CAMERA_CHECK: self._on_camera_check,
            Stage.FACE_QUALITY: self._on_face_quality,
            Stage.TURN_LEFT: self._on_turn,
            Stage.TURN_RIGHT: self._on_turn,
            Stage.RECENTER: self._on_recenter,
            Stage.ZOOM_IN: self._on_zoom,
            Stage.FLASHING: self._on_flashing,
        }.get(self.stage, self._on_done)
        return handler()

    def _on_camera_check(self) -> Dict[str, Any]:
        cam, face = self._cam, self._face
        spoof = face.anti_spoof
        # Chặn giả mạo ngay từ đầu nếu phát hiện ảnh in hoặc màn hình phát lại
        if face.detected and spoof and not spoof.is_real:
            self.stable_start_time = None
            return self._reply(spoof.message, _color_for(spoof.severity))

        if not cam.all_ok:
            self.stable_start_time = None
            return self._reply(cam.message, _color_for(cam.severity))

        elapsed = self._hold_elapsed()
        if elapsed >= C.CAMERA_STABLE_SEC:
            self.history["camera"] = cam.metrics
            self._record_timing("camera_check")
            self._enter(Stage.FACE_QUALITY)
            return self._reply("Camera đạt chuẩn! Đưa khuôn mặt vào khung Oval", "cyan", 1.0)

        progress = elapsed / C.CAMERA_STABLE_SEC
        fps_hint = " – FPS thấp, nên đóng bớt ứng dụng nền" if cam.metrics.get("fps_low") else ""
        return self._reply(f"Đang kiểm tra chất lượng Camera ({int(progress * 100)}%)...{fps_hint}", "green", progress)

    def _on_face_quality(self) -> Dict[str, Any]:
        v, face = self._verdict, self._face
        if not v.all_ok:
            self.fqa_consecutive_passes = max(0, self.fqa_consecutive_passes - 1)
            color = "red" if v.severity == "error" else ("gray" if not face.detected else "yellow")
            return self._reply(v.message, color, self.fqa_consecutive_passes / C.FQA_CONSECUTIVE_FRAMES)

        self.fqa_consecutive_passes += 1
        if self.fqa_consecutive_passes >= C.FQA_CONSECUTIVE_FRAMES:
            self.baseline_face_h = face.face_h
            self.baseline_persp = face.perspective_ratio
            self.history["fqa"] = {
                "brightness": face.brightness,
                "brightness_mean": face.brightness_mean,
                "brightness_std": face.brightness_std,
                "sharpness": face.sharpness,
                "scale_ratio": face.scale_ratio,
                "baseline_face_h": self.baseline_face_h,
                "baseline_persp": self.baseline_persp,
            }
            self._record_timing("face_quality")
            self.current_turn_index = 0
            self._enter(self.challenge_sequence[0])
            return self._reply(self._turn_instruction(), "yellow", fqa_passed_sound=True)

        progress = self.fqa_consecutive_passes / C.FQA_CONSECUTIVE_FRAMES
        light_hint = " – Ánh sáng đang lệch một bên mặt" if face.side_ratio > C.MAX_SIDE_LIGHT_RATIO else ""
        return self._reply(f"Khuôn mặt hợp lệ! Giữ yên ({int(progress * 100)}%)...{light_hint}", "green", progress)

    def _on_turn(self) -> Dict[str, Any]:
        if self._now - self.stage_start_time > C.CHALLENGE_TIMEOUT_SEC:
            return self._handle_challenge_timeout()

        v, face = self._verdict, self._face
        if not v.continuous_ok:
            self.turn_hold_counter = 0
            return self._reply(v.message, _color_for(v.severity))

        if face.yaw is None:
            self.lost_face_counter += 1
            if self.lost_face_counter > C.MAX_LOST_FRAMES:
                return self._handle_challenge_timeout("Mất dấu khuôn mặt khi quay")
            return self._reply(self._turn_instruction(), "yellow")
        self.lost_face_counter = 0

        yaw = face.yaw
        target_reached = (
            (self.stage == Stage.TURN_LEFT and yaw >= C.TURN_YAW_DEG)
            or (self.stage == Stage.TURN_RIGHT and yaw <= -C.TURN_YAW_DEG)
        )
        if not target_reached:
            self.turn_hold_counter = 0
            return self._reply(self._turn_instruction(), "yellow")

        self.turn_hold_counter += 1
        if self.turn_hold_counter >= C.TURN_HOLD_FRAMES:
            self.history["liveness"][self.stage.value] = {
                "achieved_yaw": round(yaw, 1),
                "duration": round(self._now - self.stage_start_time, 2),
            }
            self._enter(Stage.RECENTER)
            return self._reply("Đã nhận diện! Vui lòng nhìn thẳng lại vào camera", "green", 1.0)
        return self._reply("Giữ nguyên góc quay đầu...", "green", self.turn_hold_counter / C.TURN_HOLD_FRAMES)

    def _on_recenter(self) -> Dict[str, Any]:
        v, face = self._verdict, self._face
        if not v.continuous_ok:
            return self._reply(v.message, _color_for(v.severity))
        if face.yaw is None:
            return self._reply("Vui lòng quay đầu nhìn thẳng vào camera", "yellow")
        if abs(face.yaw) > C.RECENTER_YAW_DEG:
            return self._reply("Vui lòng nhìn thẳng vào camera để tiếp tục...", "yellow")

        if self.current_turn_index == 0:
            self.current_turn_index = 1
            self._enter(self.challenge_sequence[1])
            return self._reply(self._turn_instruction(), "yellow")

        self.history["timings"]["liveness_total"] = round(self._now - self.created_at, 2)
        self._enter(Stage.ZOOM_IN)
        return self._reply("Khung Oval đã mở rộng. Vui lòng tiến lại gần camera hơn để vừa khung", "yellow")

    def _on_zoom(self) -> Dict[str, Any]:
        if self._now - self.stage_start_time > C.ZOOM_TIMEOUT_SEC:
            return self._handle_challenge_timeout("Quá thời gian tiến gần camera")

        v, face = self._verdict, self._face
        if not v.continuous_ok:
            self.stable_start_time = None
            return self._reply(v.message, _color_for(v.severity))

        base_h = self.baseline_face_h or 0.3
        growth = face.face_h / max(0.05, base_h)
        growth_ok = growth >= C.ZOOM_MIN_GROWTH
        # Mặt lớn hơn mốc ban đầu VÀ nằm gọn, nét, nhìn thẳng trong oval phóng to
        framed_ok = all(v.checks.get(k, False) for k in ("face_detected", "inside_oval", "sharpness_ok", "head_straight"))

        if not (growth_ok and framed_ok):
            self.stable_start_time = None
            if not growth_ok:
                hint = f"Hãy tiến lại gần camera hơn nữa (hiện đạt {int(growth * 100)}% / {int(C.ZOOM_MIN_GROWTH * 100)}%)"
                return self._reply(hint, "yellow")
            return self._reply(v.message, _color_for(v.severity))

        elapsed = self._hold_elapsed()
        if elapsed >= C.HOLD_SEC:
            self.history["zoom"] = {
                "baseline_face_h": base_h,
                "final_face_h": face.face_h,
                "growth_ratio": round(growth, 2),
                "baseline_persp": self.baseline_persp,
                "final_persp": face.perspective_ratio,
            }
            self._record_timing("zoom")
            self._enter(Stage.FLASHING)
            return self._reply("Giữ yên khuôn mặt! Chuẩn bị quét ánh sáng màu...", "green", 1.0)
        progress = elapsed / C.HOLD_SEC
        return self._reply(f"Tuyệt vời! Giữ yên khuôn mặt ({int(progress * 100)}%)...", "green", progress)

    def _on_flashing(self) -> Dict[str, Any]:
        return self._reply("Đang quét ánh sáng quang học... Hãy nhìn thẳng vào màn hình", "green", 1.0)

    def _on_done(self) -> Dict[str, Any]:
        if self.stage == Stage.FAILED:
            failed_msg = (
                self.history.get("optical_liveness", {}).get("message")
                or "Chưa đạt chuẩn phản xạ quang học trên da. Vui lòng thử lại."
            )
            return self._reply(failed_msg, "red")
        return self._reply("Xác thực sinh trắc học thành công! Đang chuyển sang Bước 5...", "green", 1.0)

    # ------------------------------------------------------------------
    # Trợ giúp chuyển trạng thái
    # ------------------------------------------------------------------

    def _enter(self, stage: Stage) -> None:
        self.stage = stage
        self.stage_start_time = self._now
        self.stable_start_time = None
        self.fqa_consecutive_passes = 0
        self.turn_hold_counter = 0
        self.lost_face_counter = 0

    def _hold_elapsed(self) -> float:
        """Số giây điều kiện đã được giữ liên tục (bắt đầu đếm ở lần gọi đầu tiên)."""
        if self.stable_start_time is None:
            self.stable_start_time = self._now
        return self._now - self.stable_start_time

    def _record_timing(self, key: str) -> None:
        self.history["timings"][key] = round(self._now - self.stage_start_time, 2)

    def _turn_instruction(self) -> str:
        if self.stage == Stage.TURN_LEFT:
            return "Từ từ quay đầu sang Trái"
        if self.stage == Stage.TURN_RIGHT:
            return "Từ từ quay đầu sang Phải"
        return "Làm theo chỉ dẫn trên màn hình"

    def _handle_challenge_timeout(self, extra: str = "") -> Dict[str, Any]:
        self.attempts += 1
        if self.attempts >= C.MAX_ATTEMPTS:
            self.stage = Stage.FAILED
            msg = f"Đăng ký không thành công ({extra or 'Quá thời gian thử thách'}). Vui lòng bấm Thử lại từ đầu."
            return self._reply(msg, "red")
        self._enter(Stage.FACE_QUALITY)
        msg = f"{extra or 'Chưa hoàn thành thử thách'}. Đưa khuôn mặt vào khung Oval để thực hiện lại (lần {self.attempts}/{C.MAX_ATTEMPTS})"
        return self._reply(msg, "yellow")

    # ------------------------------------------------------------------
    # Kết quả từ các endpoint khác
    # ------------------------------------------------------------------

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

    def verify_final_capture(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Kiểm định ảnh chân dung cuối cùng; không đạt -> FAILED, bắt buộc đăng ký lại từ đầu."""
        face = self._get_analyzer().analyze(frame_bgr, C.OVAL_NORMAL)

        issue = _final_capture_issue(face)
        if issue is not None:
            self.stage = Stage.FAILED
            error_type, message, details = issue
            resp: Dict[str, Any] = {"passed": False, "error_type": error_type, "message": message}
            if details:
                resp["details"] = details
            return resp

        self.enrolled_image_bgr = frame_bgr
        self.enrolled_face = face
        self.stage = Stage.CAPTURE
        h, w = frame_bgr.shape[:2]

        emb_preview: list[float] = []
        try:
            extractor = get_arcface_extractor()
            if extractor.is_ready:
                self.final_embedding = extractor.extract_embedding(
                    frame_bgr, landmarks_5pts=face.arcface_kps, bbox=face.bbox
                )
                emb_preview = [round(float(v), 4) for v in self.final_embedding[:5]]
        except Exception as emb_err:
            print("[ArcFace] Extraction warning:", emb_err)
            self.final_embedding = None

        real_prob = round(face.anti_spoof.real_prob, 3) if face.anti_spoof else None
        embedding_dim = 512 if self.final_embedding is not None else 0
        metrics = {
            "resolution": f"{w}x{h}",
            "sharpness": round(face.sharpness, 1),
            "brightness": round(face.brightness_mean, 1),
            "anti_spoof_prob": real_prob,
            "embedding_dim": embedding_dim,
            "embedding_preview": emb_preview,
        }
        self.history["final_capture"] = {
            "passed": True,
            "resolution": metrics["resolution"],
            "sharpness": metrics["sharpness"],
            "brightness": metrics["brightness"],
            "yaw": face.yaw,
            "pitch": face.pitch,
            "roll": face.roll,
            "anti_spoof_real_prob": real_prob,
            "embedding_dim": embedding_dim,
            "embedding_preview": emb_preview,
            "verified_at": round(time.time(), 2),
        }
        return {
            "passed": True,
            "message": "Kiểm định khuôn mặt toàn vẹn & chống giả mạo thành công!",
            "summary": self.history,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Gói JSON trả về frontend
    # ------------------------------------------------------------------

    def _reply(self, message: str, color: str, progress: float = 0.0, fqa_passed_sound: bool = False) -> Dict[str, Any]:
        face, spoof = self._face, self._face.anti_spoof if self._face else None
        turn_arrow = {Stage.TURN_LEFT: "left", Stage.TURN_RIGHT: "right"}.get(self.stage)

        resp: Dict[str, Any] = {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "color": color,
            "message": message,
            "progress": round(min(1.0, max(0.0, progress)), 2),
            "oval": self.current_oval(),
            "turn_arrow": turn_arrow,
            "fqa_passed_sound": fqa_passed_sound,
            "camera_checks": self._cam.checks,
            "camera_metrics": self._cam.metrics,
            "face_checks": self._verdict.checks if self._verdict else {},
            "face_metrics": _face_metrics(face) if face else {},
            "anti_spoof": {
                "is_real": spoof.is_real,
                "real_prob": round(spoof.real_prob, 3),
                "print_prob": round(spoof.print_prob, 3),
                "replay_prob": round(spoof.replay_prob, 3),
                "spoof_type": spoof.spoof_type,
                "depth_3d_ok": spoof.depth_3d_ok,
                "message": spoof.message,
                "severity": spoof.severity,
            } if spoof else None,
            "parts_status": face.parts_status if face else {},
            "occluded_part_name": face.occluded_part_name if face else None,
            "keypoints": face.keypoints if face else {},
        }
        if self.stage == Stage.CAPTURE:
            resp["summary"] = self.history
        return to_native_types(resp)


def _face_metrics(face: FaceResult) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "face_count": face.face_count,
        "yaw": face.yaw,
        "pitch": face.pitch,
        "roll": face.roll,
        "is_upside_down": face.is_upside_down,
        "fill": round(face.fill, 2),
        "scale_ratio": round(face.scale_ratio, 2),
        "corners_inside": face.corners_inside,
        "brightness": round(face.brightness, 1),
        "brightness_mean": round(face.brightness_mean, 1),
        "brightness_std": round(face.brightness_std, 1),
        "sharpness": round(face.sharpness, 1),
        "oval_dist": round(face.oval_dist, 2),
        "mask_detected": face.mask_detected,
        "sunglasses_detected": face.sunglasses_detected,
        "glare_detected": face.glare_detected,
        "hand_occlusion": face.hand_occlusion,
        "perspective_ratio": face.perspective_ratio,
    }
    s = face.anti_spoof
    if s:
        m.update({
            "anti_spoof_real_prob": round(s.real_prob, 3),
            "anti_spoof_print_prob": round(s.print_prob, 3),
            "anti_spoof_replay_prob": round(s.replay_prob, 3),
            "anti_spoof_type": s.spoof_type,
            "depth_3d_delta": round(s.depth_3d_delta, 4),
            "depth_3d_ok": s.depth_3d_ok,
            "moire_ratio": round(s.moire_ratio, 3),
            "screen_detected": s.screen_detected,
            "bezel_detected": s.bezel_detected,
        })
    return m


def _final_capture_issue(face: FaceResult) -> Optional[tuple]:
    """Lỗi đầu tiên khiến ảnh chân dung bị từ chối: (error_type, message, details) hoặc None."""
    if not face.detected:
        return "no_face", "Không phát hiện khuôn mặt khi chụp ảnh chân dung. Vui lòng thử lại từ đầu.", None
    if face.face_count > 1:
        return "multiple_faces", "Phát hiện nhiều khuôn mặt trong ảnh chân dung. Chỉ một người duy nhất được phép đăng ký.", None
    if face.is_upside_down:
        return "upside_down", "Khuôn mặt bị lật ngược khi chụp chân dung. Vui lòng giữ thẳng đầu và thử lại từ đầu.", None
    if face.hand_occlusion:
        return "hand_occlusion", "Phát hiện bàn tay che mặt khi chụp ảnh. Vui lòng không chạm tay vào mặt và thử lại.", None
    if face.mask_detected:
        return "mask_detected", "Phát hiện khẩu trang khi chụp ảnh chân dung. Vui lòng tháo khẩu trang và thử lại từ đầu.", None
    if face.sunglasses_detected:
        return "sunglasses_detected", "Phát hiện kính râm khi chụp ảnh chân dung. Vui lòng tháo kính râm và thử lại từ đầu.", None
    if face.occluded_part_name:
        return "feature_occluded", f"Khuôn mặt không toàn vẹn (bị che khuất {face.occluded_part_name}). Vui lòng để lộ toàn bộ khuôn mặt.", None

    parts = face.parts_status
    if not all(parts.get(k, True) for k in ("left_eye", "right_eye", "nose", "mouth")):
        return "features_incomplete", "Ngũ quan khuôn mặt không đầy đủ hoặc bị che cản. Vui lòng thử lại từ đầu.", None

    lim = C.FINAL_MAX_POSE
    if face.yaw is not None and abs(face.yaw) > lim["yaw"]:
        return "bad_pose", "Khuôn mặt quay lệch khi chụp ảnh chân dung. Vui lòng nhìn thẳng vào camera và thử lại.", None
    if face.pitch is not None and abs(face.pitch) > lim["pitch"]:
        return "bad_pose", "Khuôn mặt ngẩng quá cao hoặc cúi quá thấp khi chụp. Vui lòng giữ thẳng đầu và thử lại.", None
    if face.roll is not None and abs(face.roll) > lim["roll"]:
        return "bad_pose", "Đang nghiêng đầu khi chụp ảnh chân dung. Vui lòng giữ thẳng đầu và thử lại.", None

    s = face.anti_spoof
    if s and not s.is_real:
        if s.spoof_type in ("print_attack", "planar_2d"):
            msg = "Phát hiện ảnh in 2D giả mạo khi chụp chân dung. Yêu cầu người thật trước camera."
        elif s.spoof_type in ("replay_attack", "screen_moire"):
            msg = "Phát hiện màn hình/video phát lại khi chụp chân dung. Yêu cầu người thật trước camera."
        else:
            msg = s.message
        return "spoof_detected", msg, {"spoof_type": s.spoof_type, "real_prob": round(s.real_prob, 3)}

    if face.brightness_mean < C.FACE_BRIGHTNESS_MIN:
        return "too_dark", "Ảnh chụp chân dung quá tối. Vui lòng tăng sáng và thử lại.", None
    if face.brightness_mean > C.FACE_BRIGHTNESS_MAX:
        return "too_bright", "Ảnh chụp chân dung bị chói sáng. Vui lòng điều chỉnh ánh sáng và thử lại.", None
    return None


def to_native_types(val: Any) -> Any:
    """Đệ quy chuyển đổi kiểu numpy sang kiểu Python chuẩn để JSON serializer không lỗi."""
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
