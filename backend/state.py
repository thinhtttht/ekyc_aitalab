"""Quản lý trạng thái dùng chung toàn tiến trình, khoá đồng bộ luồng và bộ ghi log."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import time
from typing import Dict

from fastapi import HTTPException

import config as C
from database import get_user_repository
from enrollment import EnrollmentSession
from face_analyzer import FaceAnalyzer
from feature_extractor import get_arcface_extractor
from optical_analyzer import OpticalAnalyzer

# Khởi tạo logger chuẩn ghi vào thư mục backend/logs/
os.makedirs(C.LOG_DIR, exist_ok=True)
optical_logger = logging.getLogger("ekyc.optical")
if not optical_logger.handlers:
    optical_logger.setLevel(logging.INFO)
    _handler = RotatingFileHandler(
        os.path.join(C.LOG_DIR, "optical_verify.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    optical_logger.addHandler(_handler)
    optical_logger.propagate = False

# Các singleton dịch vụ dùng chung
optical_analyzer = OpticalAnalyzer(
    min_pearson=C.FLASH_MIN_PEARSON,
    min_amplitude=C.FLASH_MIN_AMPLITUDE,
    static_amp_chroma=getattr(C, "FLASH_STATIC_AMP_CHROMA", 0.10),
    static_amp_raw=getattr(C, "FLASH_STATIC_AMP_RAW", 0.6),
)
user_repo = get_user_repository()
arcface_extractor = get_arcface_extractor()
verifier_analyzer = FaceAnalyzer()
verifier_lock = threading.Lock()

# Quản lý phiên trong RAM kèm khoá đồng bộ
sessions: Dict[str, EnrollmentSession] = {}
sessions_lock = threading.RLock()


def cleanup_expired_sessions() -> int:
    """Dọn dẹp các phiên đã quá hạn SESSION_TTL_SEC và giải phóng tài nguyên MediaPipe."""
    now = time.time()
    removed = 0
    with sessions_lock:
        expired = [
            sid
            for sid, s in list(sessions.items())
            if now - getattr(s, "last_active", s.created_at) > C.SESSION_TTL_SEC
        ]
        for sid in expired:
            sess = sessions.pop(sid, None)
            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass
                removed += 1

        # Giới hạn số phiên đồng thời tối đa để bảo vệ bộ nhớ RAM
        max_sess = getattr(C, "MAX_ACTIVE_SESSIONS", 50)
        if max_sess > 0 and len(sessions) > max_sess:
            sorted_sids = sorted(
                sessions.keys(),
                key=lambda k: getattr(sessions[k], "last_active", sessions[k].created_at),
            )
            while len(sessions) > max_sess and sorted_sids:
                oldest_sid = sorted_sids.pop(0)
                old_sess = sessions.pop(oldest_sid, None)
                if old_sess is not None:
                    try:
                        old_sess.close()
                    except Exception:
                        pass
                    removed += 1
    return removed


def get_session(session_id: str) -> EnrollmentSession:
    cleanup_expired_sessions()
    with sessions_lock:
        sess = sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã hết hạn")
    return sess
