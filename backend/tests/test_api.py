"""Unit tests cho các endpoint REST API FastAPI."""
import base64
import os
import sys
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def make_dummy_b64():
    img = np.full((600, 480, 3), 130, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_enroll_start(client):
    res = client.post("/api/enroll/start")
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert data["stage"] == "camera_check"
    assert "oval" in data
    assert len(data["challenges"]) == 2
    assert set(data["challenges"]) == {"turn_left", "turn_right"}


def test_enroll_frame(client):
    # Khởi tạo session
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    # Gửi 1 frame
    b64 = make_dummy_b64()
    payload = {
        "session_id": sid,
        "image": b64,
        "client_stats": {
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "frame_advancing": True,
        },
    }
    res = client.post("/api/enroll/frame", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == sid
    assert "camera_checks" in data
    assert "stage" in data


def test_web_static_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "eKYC" in res.text
    assert "webcamVideo" in res.text

    for path in (
        "/app.js",
        "/styles.css",
        "/assets/guide_portrait_top.png",
        "/assets/guide_portrait_left.png",
        "/assets/guide_portrait_right.png",
    ):
        r = client.get(path)
        assert r.status_code == 200, f"Không tải được static file {path}"



def test_websocket_enroll_stream(client):
    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    img = np.full((600, 480, 3), 130, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with client.websocket_connect(f"/ws/enroll/{sid}") as ws:
        # Gửi thông số client_stats trước, sau đó gửi binary JPEG
        ws.send_text('{"type": "stats", "client_stats": {"fps": 30.0, "width": 1280, "height": 720, "frame_advancing": true}}')
        ws.send_bytes(buf.tobytes())
        data = ws.receive_json()
        assert data["session_id"] == sid
        assert "camera_checks" in data
        assert "stage" in data


def test_admin_api_key_protection(client, monkeypatch):
    import config as C

    monkeypatch.setattr(C, "ADMIN_API_KEY", "secret-admin-key-999")

    # Không có key -> 401
    res_no_key = client.get("/api/users")
    assert res_no_key.status_code == 401

    # Sai key -> 401
    res_bad_key = client.get("/api/users", headers={"X-API-Key": "wrong-key"})
    assert res_bad_key.status_code == 401

    # Đúng X-API-Key -> 200
    res_ok = client.get("/api/users", headers={"X-API-Key": "secret-admin-key-999"})
    assert res_ok.status_code == 200

    # Đúng Bearer Token -> 200
    res_bearer = client.get("/api/users", headers={"Authorization": "Bearer secret-admin-key-999"})
    assert res_bearer.status_code == 200


def test_payload_size_limit_and_rate_limiter(client, monkeypatch):
    import config as C
    from security import rate_limiter

    start_res = client.post("/api/enroll/start").json()
    sid = start_res["session_id"]

    # 1. Kiểm tra chặn ảnh vượt quá kích thước tối đa (413)
    monkeypatch.setattr(C, "MAX_IMAGE_PAYLOAD_BYTES", 100)
    res_large = client.post(
        "/api/enroll/frame",
        json={"session_id": sid, "image": make_dummy_b64()},
    )
    assert res_large.status_code == 413

    # 2. Kiểm tra Rate Limiter (429)
    monkeypatch.setattr(C, "MAX_IMAGE_PAYLOAD_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(C, "RATE_LIMIT_PER_MINUTE", 2)
    rate_limiter.reset()
    try:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health").status_code == 429
    finally:
        rate_limiter.reset()

