"""API Router xác thực khuôn mặt sinh trắc học thời gian thực (1:1 & 1:N)."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config as C
from security import decode_or_400
import state

router = APIRouter(tags=["Verification"])


class FaceVerifyPayload(BaseModel):
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    target_user_id: Optional[str] = None  # None -> 1:N; có giá trị -> 1:1
    threshold: float = Field(default=C.DEFAULT_VERIFY_THRESHOLD, ge=0.1, le=0.9)


@router.post("/api/verify/face")
def verify_face_biometrics(payload: FaceVerifyPayload):
    """Xác thực khuôn mặt thời gian thực với FQA & Anti-Spoofing nghiêm ngặt (1:1 & 1:N)."""
    frame_bgr = decode_or_400(payload.image)

    start_t = time.time()
    with state.verifier_lock:
        face_res = state.verifier_analyzer.analyze(frame_bgr, oval=C.OVAL_NORMAL, fresh=True)

    # 1. Kiểm tra sự hiện diện khuôn mặt
    if face_res.face_count == 0 or not face_res.detected:
        return {
            "passed": False,
            "verdict": "NO_FACE",
            "message": "Không phát hiện khuôn mặt trong khung hình. Vui lòng nhìn thẳng vào camera.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    if face_res.face_count > 1:
        return {
            "passed": False,
            "verdict": "MULTIPLE_FACES",
            "message": "Phát hiện nhiều hơn 1 khuôn mặt trước ống kính. Chỉ một người duy nhất trong khung hình.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 2. Kiểm tra lật ngược đầu
    if face_res.is_upside_down:
        return {
            "passed": False,
            "verdict": "INVALID_POSE",
            "message": "Khuôn mặt bị lật ngược. Vui lòng giữ thẳng đầu (mắt ở trên, miệng ở dưới).",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 3. Kiểm tra che khuất ngũ quan & khẩu trang & kính râm & tay che mặt
    if face_res.hand_occlusion:
        return {
            "passed": False,
            "verdict": "FACE_OCCLUDED",
            "message": "Vui lòng bỏ tay ra khỏi khuôn mặt trước khi xác thực.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    if face_res.mask_detected:
        return {
            "passed": False,
            "verdict": "FACE_OCCLUDED",
            "message": "Vui lòng tháo khẩu trang để hệ thống xác thực khuôn mặt.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    if face_res.sunglasses_detected:
        return {
            "passed": False,
            "verdict": "FACE_OCCLUDED",
            "message": "Vui lòng tháo kính râm/kính đen để nhận diện rõ vùng mắt.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    if face_res.occluded_part_name:
        return {
            "passed": False,
            "verdict": "FACE_OCCLUDED",
            "message": f"Phát hiện che khuất {face_res.occluded_part_name}. Vui lòng để lộ toàn bộ khuôn mặt.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    parts = face_res.parts_status or {}
    if not (parts.get("left_eye", True) and parts.get("right_eye", True) and parts.get("nose", True)):
        return {
            "passed": False,
            "verdict": "FACE_OCCLUDED",
            "message": "Ngũ quan khuôn mặt (mắt, mũi) bị che khuất hoặc không rõ ràng.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 4. Kiểm tra cự ly & lọt khung Oval (Framing & Scale)
    if face_res.scale_ratio < 0.30:
        return {
            "passed": False,
            "verdict": "INVALID_FRAMING",
            "message": "Khuôn mặt quá xa. Vui lòng tiến lại gần camera hơn.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }
    if face_res.scale_ratio > 0.95:
        return {
            "passed": False,
            "verdict": "INVALID_FRAMING",
            "message": "Khuôn mặt quá gần. Vui lòng lùi ra xa camera một chút.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }
    if face_res.oval_dist > 1.30:
        return {
            "passed": False,
            "verdict": "INVALID_FRAMING",
            "message": "Vui lòng căn chỉnh khuôn mặt vào giữa khung Oval hướng dẫn.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 5. Kiểm tra tư thế nhìn thẳng (Pose)
    if face_res.yaw is not None and abs(face_res.yaw) > 30.0:
        return {
            "passed": False,
            "verdict": "INVALID_POSE",
            "message": "Vui lòng nhìn thẳng vào camera (không quay đầu quá mức).",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }
    if face_res.pitch is not None and abs(face_res.pitch) > 25.0:
        return {
            "passed": False,
            "verdict": "INVALID_POSE",
            "message": "Vui lòng giữ thẳng đầu (không cúi gằm hoặc ngửa mặt quá cao).",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 6. Kiểm tra độ sắc nét (Sharpness)
    if C.SHARPNESS_CHECK_ENABLED and face_res.sharpness < C.MIN_FACE_SHARPNESS:
        return {
            "passed": False,
            "verdict": "IMAGE_BLURRY",
            "message": "Hình ảnh bị mờ hoặc rung lắc. Vui lòng giữ yên camera và nhìn rõ.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 7. Kiểm tra ánh sáng (Brightness)
    if face_res.brightness_mean < C.FACE_BRIGHTNESS_MIN:
        return {
            "passed": False,
            "verdict": "POOR_LIGHTING",
            "message": "Không gian quá tối. Vui lòng tăng thêm ánh sáng.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }
    if face_res.brightness_mean > C.FACE_BRIGHTNESS_MAX:
        return {
            "passed": False,
            "verdict": "POOR_LIGHTING",
            "message": "Ánh sáng quá chói hoặc lóa. Vui lòng điều chỉnh lại hướng đèn.",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 8. Phòng thủ chống giả mạo Passive Anti-Spoofing (MiniFASNet)
    if face_res.anti_spoof and face_res.anti_spoof.is_real is False:
        return {
            "passed": False,
            "verdict": "SPOOF_REJECTED",
            "message": f"Từ chối giả mạo: {face_res.anti_spoof.message}",
            "anti_spoof": {
                "is_real": False,
                "real_prob": round(face_res.anti_spoof.real_prob, 3),
                "spoof_type": face_res.anti_spoof.spoof_type,
            },
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # 9. Trích xuất vector đặc trưng ArcFace 512 chiều khi đã vượt qua FQA
    try:
        query_emb = state.arcface_extractor.extract_embedding(
            frame_bgr,
            landmarks_5pts=face_res.arcface_kps,
            bbox=face_res.bbox,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trích xuất ArcFace: {e}")

    # 10. Đối sánh 1:1 đích danh hoặc 1:N nhận diện trong CSDL
    if payload.target_user_id and payload.target_user_id.strip():
        match_data = state.user_repo.match_user_1_to_1(
            query_emb,
            payload.target_user_id.strip(),
            threshold=payload.threshold,
        )
    else:
        match_data = state.user_repo.match_user_1_to_n(
            query_emb,
            threshold=payload.threshold,
        )

    latency_ms = round((time.time() - start_t) * 1000, 1)
    is_match = bool(match_data.get("is_match", False))
    matched_name = match_data.get("matched_user", {}).get("full_name") if match_data.get("matched_user") else ""

    return {
        "passed": is_match,
        "verdict": "MATCH_SUCCESS" if is_match else "MISMATCH",
        "message": f"Xác thực danh tính thành công cho '{matched_name}'!" if is_match else "Không trùng khớp với hồ sơ khuôn mặt nào.",
        "anti_spoof": {
            "is_real": True,
            "real_prob": round(face_res.anti_spoof.real_prob, 3) if face_res.anti_spoof else 1.0,
            "spoof_type": "real",
        },
        "match_result": match_data,
        "latency_ms": latency_ms,
    }
