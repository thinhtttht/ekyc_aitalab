"""Kiểm thử đơn vị cho Module backend/color_challenge.py."""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from color_challenge import ChallengeManager, ColorStep


def test_create_challenge_valid():
    manager = ChallengeManager(ttl_sec=10.0)
    data = manager.create_challenge("session-123", length=3)

    assert data.session_id == "session-123"
    assert len(data.sequence) == 3
    assert data.token is not None and len(data.token) == 64
    assert data.expires_at > time.time()
    assert not data.consumed

    # Kiểm tra 2 màu liên tiếp không trùng nhau
    for i in range(1, len(data.sequence)):
        assert data.sequence[i].name != data.sequence[i - 1].name

    # Kiểm tra định dạng RGB
    for step in data.sequence:
        assert isinstance(step, ColorStep)
        assert len(step.rgb) == 3
        for val in step.rgb:
            assert 0 <= val <= 255
        assert step.hex.startswith("#")
        assert step.duration_ms == 330


def test_token_verification_success():
    manager = ChallengeManager(ttl_sec=10.0)
    data = manager.create_challenge("session-abc", length=3)

    is_valid, ret_data, msg = manager.verify_token("session-abc", data.token)
    assert is_valid is True
    assert ret_data is not None
    assert ret_data.session_id == "session-abc"
    assert "hợp lệ" in msg


def test_token_verification_tampered():
    manager = ChallengeManager(ttl_sec=10.0)
    data = manager.create_challenge("session-abc", length=3)

    # Thử truyền token bị can thiệp
    fake_token = "0" * 64
    is_valid, ret_data, msg = manager.verify_token("session-abc", fake_token)
    assert is_valid is False
    assert ret_data is None
    assert "không hợp lệ" in msg


def test_token_verification_expired():
    # TTL siêu ngắn 0.05s
    manager = ChallengeManager(ttl_sec=0.05)
    data = manager.create_challenge("session-exp", length=3)

    time.sleep(0.08)
    is_valid, ret_data, msg = manager.verify_token("session-exp", data.token)
    assert is_valid is False
    assert ret_data is None
    assert "quá hạn" in msg or "không tồn tại" in msg


def test_token_consumption_anti_replay():
    manager = ChallengeManager(ttl_sec=10.0)
    data = manager.create_challenge("session-replay", length=3)

    # Lần 1 verify hợp lệ
    is_valid, _, _ = manager.verify_token("session-replay", data.token)
    assert is_valid is True

    # Tiêu thụ token
    manager.consume_challenge("session-replay")

    # Lần 2 verify phải bị từ chối
    is_valid2, ret_data2, msg2 = manager.verify_token("session-replay", data.token)
    assert is_valid2 is False
    assert ret_data2 is None
    assert "đã được sử dụng" in msg2
