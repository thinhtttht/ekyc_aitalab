import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proctoring_service import ProctoringService, get_proctoring_service


@pytest.fixture
def proctor_service():
    return ProctoringService()


def test_progressive_penalty_escalation(proctor_service):
    test_user_id = f"TEST-CANDIDATE-{int(time.time()*1000)}"
    full_name = "Thí Sinh Kiểm Thử"

    # 1. Trạng thái ban đầu: active, count = 0
    st0 = proctor_service.get_penalty_status(test_user_id)
    assert st0["violation_count"] == 0
    assert st0["is_locked"] is False
    assert st0["is_banned"] is False

    # 2. Vi phạm LẦN 1: Cảnh báo, không khóa
    r1 = proctor_service.record_violation(test_user_id, full_name, "PRINT_ATTACK: Phát hiện ảnh in")
    assert r1["violation_count"] == 1
    assert r1["status"] == "warning"
    assert r1["is_locked"] is False
    assert r1["is_banned"] is False

    # 3. Vi phạm LẦN 2: Khóa 1 phút (60 giây)
    r2 = proctor_service.record_violation(test_user_id, full_name, "REPLAY_ATTACK: Phát hiện điện thoại")
    assert r2["violation_count"] == 2
    assert r2["status"] == "locked_1m"
    assert r2["is_locked"] is True
    assert 55 <= r2["remaining_seconds"] <= 60

    st_locked = proctor_service.get_penalty_status(test_user_id)
    assert st_locked["is_locked"] is True
    assert st_locked["status"] == "locked_1m"

    # 4. Vi phạm LẦN 3: Khóa 5 phút (300 giây)
    r3 = proctor_service.record_violation(test_user_id, full_name, "MULTIPLE_FACES: Nhiều người trong phòng")
    assert r3["violation_count"] == 3
    assert r3["status"] == "locked_5m"
    assert r3["is_locked"] is True
    assert 295 <= r3["remaining_seconds"] <= 300

    # 5. Vi phạm LẦN 4: ĐÌNH CHỈ THI (BANNED)
    r4 = proctor_service.record_violation(test_user_id, full_name, "PRINT_ATTACK: Tái phạm lần 4")
    assert r4["violation_count"] == 4
    assert r4["status"] == "banned"
    assert r4["is_locked"] is True
    assert r4["is_banned"] is True
    assert r4["remaining_seconds"] == -1

    st_banned = proctor_service.get_penalty_status(test_user_id)
    assert st_banned["is_banned"] is True
    assert st_banned["status"] == "banned"

    # 6. GIÁM THỊ / ADMIN MỞ KHÓA
    unlock_res = proctor_service.admin_unlock(test_user_id)
    assert unlock_res["success"] is True

    st_unlocked = proctor_service.get_penalty_status(test_user_id)
    assert st_unlocked["violation_count"] == 0
    assert st_unlocked["is_locked"] is False
    assert st_unlocked["is_banned"] is False
    assert st_unlocked["status"] == "active"
