"""FastAPI application cho hệ thống eKYC Biometric Authentication.

- Phục vụ API kiểm tra chất lượng Camera & FQA & Liveness quay đầu & Zoom.
- Phục vụ tĩnh thư mục giao diện HTML/CSS/JS thuần `web/`.
"""
from __future__ import annotations

import base64
import os
import sys
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

optical_analyzer = OpticalAnalyzer(
    min_pearson=C.FLASH_MIN_PEARSON, min_amplitude=C.FLASH_MIN_AMPLITUDE
)

app = FastAPI(
    title="Smart-eKYC Biometric PoC",
    description="Hệ thống eKYC Đa tầng - Khung Oval, FQA, Liveness Quay đầu & Zoom",
    version="1.0.0",
)

# Cho phép CORS linh hoạt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    sess = sessions.get(payload.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã hết hạn")

    # Giải mã ảnh base64
    raw_b64 = payload.image
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Không thể decode ảnh JPEG")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu ảnh không hợp lệ: {e}")

    # Chuyển đổi payload client stats
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
    sess = sessions.get(payload.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã hết hạn")

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
    sess = sessions.get(payload.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã hết hạn")

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

    frames_bgr = []
    for item in payload.frames:
        raw_b64 = item.image
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            f_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if f_bgr is None:
                raise ValueError(f"Không thể decode ảnh frame {item.color_index}")
            frames_bgr.append(f_bgr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Dữ liệu ảnh không hợp lệ: {e}")

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


# Gắn frontend tĩnh tại `d:\EKYC\web`
WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if not os.path.exists(WEB_DIR):
    os.makedirs(WEB_DIR, exist_ok=True)

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
