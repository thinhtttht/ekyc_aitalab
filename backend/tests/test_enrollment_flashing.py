"""Kiểm thử tích hợp các endpoint Color Flashing Challenge và Verification."""
import base64
import os
import sys
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from optical_analyzer import OpticalResult


@pytest.fixture
def client():
    return TestClient(app)


def make_b64_image(color_bgr=(100, 150, 200), h=480, w=640) -> str:
    img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def test_color_challenge_flow(client):
    # 1. Bắt đầu session
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    # 2. Gọi lấy challenge
    res = client.post("/api/enroll/color_challenge", json={"session_id": sid})
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == sid
    assert "challenge_token" in data
    assert len(data["sequence"]) == 3
    assert data["step_duration_ms"] == 330
    assert data["expires_at"] > 0


def test_color_verify_invalid_token(client):
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    # Challenge hợp lệ
    client.post("/api/enroll/color_challenge", json={"session_id": sid}).json()

    # Verify với token giả mạo
    fake_payload = {
        "session_id": sid,
        "challenge_token": "wrong_token_123456",
        "frames": [
            {"color_index": 0, "image": make_b64_image()},
            {"color_index": 1, "image": make_b64_image()},
            {"color_index": 2, "image": make_b64_image()},
        ],
    }
    res = client.post("/api/enroll/color_verify", json=fake_payload)
    assert res.status_code == 400
    assert "Token thách thức không hợp lệ" in res.json()["detail"]


def test_color_verify_frame_count_mismatch(client):
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    ch_data = client.post("/api/enroll/color_challenge", json={"session_id": sid}).json()
    token = ch_data["challenge_token"]

    # Gửi chỉ 2 frame thay vì 3
    mismatch_payload = {
        "session_id": sid,
        "challenge_token": token,
        "frames": [
            {"color_index": 0, "image": make_b64_image()},
            {"color_index": 1, "image": make_b64_image()},
        ],
    }
    res = client.post("/api/enroll/color_verify", json=mismatch_payload)
    assert res.status_code == 400
    assert "không khớp với chuỗi thách thức" in res.json()["detail"]


def test_color_verify_success_mocked(client, monkeypatch):
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    ch_data = client.post("/api/enroll/color_challenge", json={"session_id": sid}).json()
    token = ch_data["challenge_token"]

    # Mock optical_analyzer.analyze_sequence để trả về kết quả đạt
    from app import optical_analyzer

    def mock_analyze(*args, **kwargs):
        return OpticalResult(
            passed=True,
            correlation_score=0.92,
            amplitude=16.4,
            verdict="LIVENESS_PASS",
            message="Xác thực phản xạ quang học thành công.",
            details={"mocked": True},
        )

    monkeypatch.setattr(optical_analyzer, "analyze_sequence", mock_analyze)

    payload = {
        "session_id": sid,
        "challenge_token": token,
        "frames": [
            {"color_index": 0, "image": make_b64_image()},
            {"color_index": 1, "image": make_b64_image()},
            {"color_index": 2, "image": make_b64_image()},
        ],
    }
    res = client.post("/api/enroll/color_verify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["passed"] is True
    assert data["stage"] == "capture"
    assert data["correlation_score"] == 0.92
    assert "summary" in data and data["summary"] is not None
    assert data["summary"]["optical_liveness"]["passed"] is True
