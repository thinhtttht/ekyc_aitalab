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
    assert "Smart-eKYC" in res.text
    assert "webcamVideo" in res.text
