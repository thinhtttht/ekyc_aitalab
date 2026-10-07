"""FastAPI application cho hệ thống eKYC Biometric Authentication.

- Phục vụ API kiểm tra chất lượng Camera & FQA & Liveness quay đầu & Zoom.
- Phục vụ tĩnh thư mục giao diện HTML/CSS/JS thuần `web/`.
"""
from __future__ import annotations

import base64
import os
import time
import uuid
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config as C
from camera_quality import ClientStats
from enrollment import EnrollmentSession, Stage
from color_challenge import challenge_manager
from optical_analyzer import OpticalAnalyzer
from database import get_user_repository
from feature_extractor import get_arcface_extractor
from face_analyzer import FaceAnalyzer

optical_analyzer = OpticalAnalyzer(
    min_pearson=C.FLASH_MIN_PEARSON, min_amplitude=C.FLASH_MIN_AMPLITUDE
)
user_repo = get_user_repository()
arcface_extractor = get_arcface_extractor()
verifier_analyzer = FaceAnalyzer()

app = FastAPI(
    title="Smart-eKYC Biometric PoC",
    description="Hệ thống eKYC Đa tầng - Khung Oval, FQA, Liveness Quay đầu & Zoom",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=C.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    """Vô hiệu hoá browser cache đối với frontend static files trong môi trường phát triển."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Quản lý phiên trong RAM
sessions: Dict[str, EnrollmentSession] = {}


def cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, s in sessions.items() if now - s.created_at > C.SESSION_TTL_SEC
    ]
    for sid in expired:
        try:
            sessions[sid].close()
        except Exception:
            pass
        sessions.pop(sid, None)


def get_session(session_id: str) -> EnrollmentSession:
    cleanup_expired_sessions()
    sess = sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã hết hạn")
    return sess


def decode_base64_image(raw_b64: str) -> np.ndarray:
    """Giải mã ảnh base64 (hỗ trợ cả Data URL và chuỗi thuần)."""
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(raw_b64)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không thể decode ảnh JPEG")
    return frame


def decode_or_400(raw_b64: str) -> np.ndarray:
    try:
        return decode_base64_image(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu ảnh không hợp lệ: {e}")


class StartResponse(BaseModel):
    session_id: str
    stage: str
    oval: dict
    challenges: list[str]


class ClientStatsPayload(BaseModel):
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_advancing: bool = True


class FramePayload(BaseModel):
    session_id: str
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    client_stats: ClientStatsPayload = Field(default_factory=ClientStatsPayload)


@app.get("/api/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/api/enroll/start", response_model=StartResponse)
def enroll_start():
    cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    sess = EnrollmentSession(session_id)
    sessions[session_id] = sess
    return StartResponse(
        session_id=session_id,
        stage=sess.stage.value,
        oval=sess.current_oval(),
        challenges=[t.value for t in sess.challenge_sequence],
    )


@app.post("/api/enroll/frame")
def enroll_frame(payload: FramePayload):
    sess = get_session(payload.session_id)
    frame_bgr = decode_or_400(payload.image)
    stats = ClientStats(
        fps=payload.client_stats.fps,
        width=payload.client_stats.width,
        height=payload.client_stats.height,
        frame_advancing=payload.client_stats.frame_advancing,
    )

    result = sess.process_frame(frame_bgr, stats)
    return result


@app.post("/api/enroll/reset")
def enroll_reset(payload: dict):
    session_id = payload.get("session_id")
    if session_id and session_id in sessions:
        try:
            sessions[session_id].close()
        except Exception:
            pass
        sessions.pop(session_id, None)
    return enroll_start()


class ColorChallengeRequest(BaseModel):
    session_id: str


class FrameColorItem(BaseModel):
    color_index: int
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    timestamp_ms: Optional[int] = None


class ColorVerifyRequest(BaseModel):
    session_id: str
    challenge_token: str
    frames: list[FrameColorItem]


@app.post("/api/enroll/color_challenge")
def enroll_color_challenge(payload: ColorChallengeRequest):
    get_session(payload.session_id)
    challenge = challenge_manager.create_challenge(payload.session_id, length=3)
    return {
        "session_id": payload.session_id,
        "challenge_token": challenge.token,
        "step_duration_ms": C.FLASH_STEP_DURATION_MS,
        "sequence": [
            {
                "index": step.index,
                "name": step.name,
                "hex": step.hex,
                "rgb": list(step.rgb),
                "duration_ms": step.duration_ms,
            }
            for step in challenge.sequence
        ],
        "expires_at": challenge.expires_at,
    }


@app.post("/api/enroll/color_verify")
def enroll_color_verify(payload: ColorVerifyRequest):
    sess = get_session(payload.session_id)
    is_valid, challenge_data, reason = challenge_manager.verify_token(
        payload.session_id, payload.challenge_token
    )
    if not is_valid or challenge_data is None:
        raise HTTPException(status_code=400, detail=reason)

    if len(payload.frames) != len(challenge_data.sequence):
        raise HTTPException(
            status_code=400,
            detail=f"Số lượng khung hình ({len(payload.frames)}) không khớp với chuỗi thách thức ({len(challenge_data.sequence)})",
        )

    frames_bgr = [decode_or_400(item.image) for item in payload.frames]

    expected_colors = [tuple(step.rgb) for step in challenge_data.sequence]
    res = optical_analyzer.analyze_sequence(
        frames_bgr=frames_bgr,
        expected_colors_rgb=expected_colors,
        awb_gamma=C.FLASH_AWB_GAMMA,
    )

    print(
        f"[OPTICAL VERIFY] passed={res.passed} | Pearson r={res.correlation_score:.3f} (min={C.FLASH_MIN_PEARSON}) | "
        f"Amp={res.amplitude:.2f} (min={C.FLASH_MIN_AMPLITUDE}) | verdict={res.verdict} | msg={res.message}"
    )

    try:
        log_path = os.path.join(os.path.dirname(__file__), "optical_verify.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] session={payload.session_id} | "
                f"passed={res.passed} | r_best={res.correlation_score:.3f} | amp={res.amplitude:.2f} | "
                f"verdict={res.verdict} | msg={res.message} | details={res.details}\n"
            )
    except Exception as log_err:
        print("Log error:", log_err)

    # Đánh dấu tiêu thụ token chống replay
    challenge_manager.consume_challenge(payload.session_id)

    # Cập nhật kết quả vào session
    sess.apply_optical_result(
        passed=res.passed,
        correlation=res.correlation_score,
        amplitude=res.amplitude,
        verdict=res.verdict,
        details=res.details,
    )

    return {
        "session_id": payload.session_id,
        "passed": res.passed,
        "correlation_score": res.correlation_score,
        "amplitude": res.amplitude,
        "verdict": res.verdict,
        "stage": sess.stage.value,
        "message": res.message,
        "details": res.details,
        "summary": sess.history if sess.stage == Stage.CAPTURE else None,
    }


class FinalCaptureRequest(BaseModel):
    session_id: str
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64 của ảnh chụp chân dung HD")


@app.post("/api/enroll/verify_capture")
def enroll_verify_capture(payload: FinalCaptureRequest):
    sess = get_session(payload.session_id)
    return sess.verify_final_capture(decode_or_400(payload.image))


# ---------------------------------------------------------------------------
# QUẢN LÝ NGƯỜI DÙNG & XÁC THỰC SINH TRẮC HỌC (ARCFACE RECOGNITION & DB)
# ---------------------------------------------------------------------------


class UserEnrollPayload(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    full_name: str = Field(min_length=1, max_length=120)
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    metadata: Optional[dict] = None


class FaceVerifyPayload(BaseModel):
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    target_user_id: Optional[str] = None  # None -> 1:N; có giá trị -> 1:1
    threshold: float = Field(default=0.45, ge=0.1, le=0.9)


@app.post("/api/users/enroll")
def enroll_user(payload: UserEnrollPayload):
    """Lưu hồ sơ người dùng kèm vector ArcFace 512D vào cơ sở dữ liệu."""
    frame_bgr = decode_or_400(payload.image)

    # 1. Nếu có session_id và đã có final_embedding thì ưu tiên sử dụng
    sess = sessions.get(payload.session_id) if payload.session_id else None
    if sess and sess.final_embedding is not None:
        emb = sess.final_embedding
    else:
        # Tự trích xuất từ ảnh
        face_res = verifier_analyzer.analyze(frame_bgr, oval=C.OVAL_NORMAL)
        kps = face_res.arcface_kps if face_res.detected else None
        bbox = face_res.bbox if face_res.detected else None
        try:
            emb = arcface_extractor.extract_embedding(
                frame_bgr,
                landmarks_5pts=kps,
                bbox=bbox,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi trích xuất ArcFace: {e}")

    # Đảm bảo format Data URL
    snapshot_b64 = payload.image
    if not snapshot_b64.startswith("data:image/"):
        snapshot_b64 = f"data:image/jpeg;base64,{snapshot_b64}"

    user = user_repo.save_user(
        full_name=payload.full_name,
        embedding=emb,
        snapshot_b64=snapshot_b64,
        user_id=payload.user_id,
        metadata=payload.metadata,
    )
    return {
        "status": "ok",
        "message": f"Đã lưu hồ sơ sinh trắc học thành công cho '{user['full_name']}'!",
        "user": user,
    }


@app.get("/api/users")
def get_users_list():
    """Lấy danh sách tất cả người dùng trong CSDL phục vụ UI quản lý."""
    users = user_repo.list_users()
    return {"status": "ok", "total": len(users), "users": users}


@app.get("/api/users/{user_id}")
def get_user_detail(user_id: str):
    """Lấy chi tiết hồ sơ người dùng theo user_id."""
    u = user_repo.get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    return {
        "user_id": u["user_id"],
        "full_name": u["full_name"],
        "embedding": [round(float(v), 6) for v in u["embedding"]],
        "embedding_dim": u["embedding_dim"],
        "embedding_preview": u["embedding_preview"],
        "created_at": u["created_at"],
        "snapshot_b64": u["snapshot_b64"],
        "metadata": u["metadata"],
    }


@app.delete("/api/users/{user_id}")
def delete_user_record(user_id: str):
    """Xoá một hồ sơ người dùng khỏi CSDL."""
    deleted = user_repo.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng để xoá")
    return {"status": "ok", "message": f"Đã xoá người dùng {user_id}"}


@app.post("/api/verify/face")
def verify_face_biometrics(payload: FaceVerifyPayload):
    """Xác thực khuôn mặt thời gian thực (hỗ trợ 1:1 đích danh và 1:N nhận diện)."""
    frame_bgr = decode_or_400(payload.image)

    start_t = time.time()
    face_res = verifier_analyzer.analyze(frame_bgr, oval=C.OVAL_NORMAL)

    if face_res.face_count == 0:
        return {
            "passed": False,
            "verdict": "NO_FACE",
            "message": "Không phát hiện khuôn mặt trong khung hình",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    if face_res.face_count > 1:
        return {
            "passed": False,
            "verdict": "MULTIPLE_FACES",
            "message": "Phát hiện nhiều hơn 1 khuôn mặt trước ống kính",
            "match_result": None,
            "latency_ms": round((time.time() - start_t) * 1000, 1),
        }

    # Kiểm tra phòng thủ chống giả mạo Passive Anti-Spoofing
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

    # Trích xuất vector ArcFace 512 chiều
    try:
        query_emb = arcface_extractor.extract_embedding(
            frame_bgr,
            landmarks_5pts=face_res.arcface_kps,
            bbox=face_res.bbox,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trích xuất ArcFace: {e}")

    # 1:1 hay 1:N
    if payload.target_user_id and payload.target_user_id.strip():
        match_data = user_repo.match_user_1_to_1(
            query_emb,
            payload.target_user_id.strip(),
            threshold=payload.threshold,
        )
    else:
        match_data = user_repo.match_user_1_to_n(
            query_emb,
            threshold=payload.threshold,
        )

    latency_ms = round((time.time() - start_t) * 1000, 1)
    is_match = bool(match_data.get("is_match", False))

    return {
        "passed": is_match,
        "verdict": "MATCH_SUCCESS" if is_match else "MISMATCH",
        "message": "Xác thực danh tính thành công!" if is_match else "Không trùng khớp với hồ sơ khuôn mặt nào.",
        "anti_spoof": {
            "is_real": True,
            "real_prob": round(face_res.anti_spoof.real_prob, 3) if face_res.anti_spoof else 1.0,
        },
        "match_result": match_data,
        "latency_ms": latency_ms,
    }


WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
