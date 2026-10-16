"""API Router cho quy trình Đăng ký khuôn mặt eKYC (Enrollment & Active Liveness)."""
from __future__ import annotations

import json
from typing import Optional
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import config as C
from camera_quality import ClientStats
from color_challenge import challenge_manager
from enrollment import EnrollmentSession, Stage
from security import decode_or_400
import state

router = APIRouter(tags=["Enrollment"])


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


class FinalCaptureRequest(BaseModel):
    session_id: str
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64 của ảnh chụp chân dung HD")


@router.post("/api/enroll/start", response_model=StartResponse)
def enroll_start():
    state.cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    sess = EnrollmentSession(session_id)
    with state.sessions_lock:
        state.sessions[session_id] = sess
    return StartResponse(
        session_id=session_id,
        stage=sess.stage.value,
        oval=sess.current_oval(),
        challenges=[t.value for t in sess.challenge_sequence],
    )


@router.post("/api/enroll/frame")
def enroll_frame(payload: FramePayload):
    sess = state.get_session(payload.session_id)
    frame_bgr = decode_or_400(payload.image)
    stats = ClientStats(
        fps=payload.client_stats.fps,
        width=payload.client_stats.width,
        height=payload.client_stats.height,
        frame_advancing=payload.client_stats.frame_advancing,
    )
    return sess.process_frame(frame_bgr, stats)


@router.websocket("/ws/enroll/{session_id}")
async def enroll_ws(websocket: WebSocket, session_id: str):
    """Luồng WebSocket thời gian thực để truyền frame nhị phân hoặc JSON với độ trễ thấp."""
    await websocket.accept()
    with state.sessions_lock:
        sess = state.sessions.get(session_id)
    if sess is None:
        await websocket.send_json({"error": "Phiên làm việc không tồn tại hoặc đã hết hạn", "status_code": 404})
        await websocket.close(code=1008)
        return

    last_stats = ClientStats(fps=30.0, width=640, height=480, frame_advancing=True)
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                raw_bytes = msg["bytes"]
                frame_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    await websocket.send_json({"error": "Không thể decode ảnh JPEG binary"})
                    continue
                res = sess.process_frame(frame_bgr, last_stats)
                await websocket.send_json(res)
            elif "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "stats":
                    cs = data.get("client_stats") or {}
                    last_stats = ClientStats(
                        fps=cs.get("fps"),
                        width=cs.get("width"),
                        height=cs.get("height"),
                        frame_advancing=cs.get("frame_advancing", True),
                    )
                    continue

                if "client_stats" in data and isinstance(data["client_stats"], dict):
                    cs = data["client_stats"]
                    last_stats = ClientStats(
                        fps=cs.get("fps"),
                        width=cs.get("width"),
                        height=cs.get("height"),
                        frame_advancing=cs.get("frame_advancing", True),
                    )
                if "image" in data:
                    frame_bgr = decode_or_400(data["image"])
                    res = sess.process_frame(frame_bgr, last_stats)
                    await websocket.send_json(res)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


@router.post("/api/enroll/reset")
def enroll_reset(payload: dict):
    session_id = payload.get("session_id")
    if session_id:
        with state.sessions_lock:
            old = state.sessions.pop(session_id, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
    return enroll_start()


@router.post("/api/enroll/color_challenge")
def enroll_color_challenge(payload: ColorChallengeRequest):
    state.get_session(payload.session_id)
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


@router.post("/api/enroll/color_verify")
def enroll_color_verify(payload: ColorVerifyRequest):
    sess = state.get_session(payload.session_id)
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
    res = state.optical_analyzer.analyze_sequence(
        frames_bgr=frames_bgr,
        expected_colors_rgb=expected_colors,
        awb_gamma=C.FLASH_AWB_GAMMA,
    )

    # Ép đạt chỉ khi bật rõ cờ EKYC_FLASH_DEMO_BYPASS=1
    if getattr(C, "FLASH_DEMO_BYPASS", False) and not res.passed:
        res.passed = True
        res.verdict = "LIVENESS_PASS"
        res.message = "Xác thực phản xạ quang học thành công (Demo Bypass)."
        if res.correlation_score < C.FLASH_MIN_PEARSON:
            res.correlation_score = float(C.FLASH_MIN_PEARSON)

    state.optical_logger.info(
        "session=%s | passed=%s | r_best=%.3f | amp=%.2f | verdict=%s | msg=%s | details=%s",
        payload.session_id,
        res.passed,
        res.correlation_score,
        res.amplitude,
        res.verdict,
        res.message,
        res.details,
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


@router.post("/api/enroll/verify_capture")
def enroll_verify_capture(payload: FinalCaptureRequest):
    sess = state.get_session(payload.session_id)
    return sess.verify_final_capture(decode_or_400(payload.image))
