"""Endpoint kiểm tra sức khỏe hệ thống (Healthcheck)."""
from __future__ import annotations

from fastapi import APIRouter

import state

router = APIRouter(tags=["Health"])


@router.get("/api/health")
def health():
    with state.sessions_lock:
        active_count = len(state.sessions)
    return {"status": "ok", "active_sessions": active_count}
