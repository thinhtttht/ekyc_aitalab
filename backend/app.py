"""FastAPI application cho hệ thống eKYC Biometric Authentication.

- Khởi tạo ứng dụng, CORS, Rate Limiting và vòng đời dọn dẹp phiên tự động (Background Lifespan).
- Kết nối các module API Router (`routers/health.py`, `routers/enroll.py`, `routers/users.py`, `routers/verify.py`).
- Phục vụ tĩnh thư mục giao diện HTML/CSS/JS ES Modules `web/`.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
import sys

# Đặt stdout/stderr về UTF-8 để log tiếng Việt không làm crash endpoint
# trên console Windows mặc định cp1252 (tránh UnicodeEncodeError -> HTTP 500).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config as C
from routers import enroll as enroll_router
from routers import health as health_router
from routers import users as users_router
from routers import verify as verify_router
from routers import idcard as idcard_router
from security import decode_base64_image, decode_or_400, rate_limiter
import state

# Re-export các thành phần trạng thái dùng chung để tương thích ngược với bộ kiểm thử
optical_analyzer = state.optical_analyzer
user_repo = state.user_repo
arcface_extractor = state.arcface_extractor
verifier_analyzer = state.verifier_analyzer
sessions = state.sessions
cleanup_expired_sessions = state.cleanup_expired_sessions
get_session = state.get_session


async def _periodic_session_cleaner() -> None:
    """Tác vụ nền định kỳ giải phóng các phiên quá hạn ngay cả khi không có request mới."""
    while True:
        interval = max(5, getattr(C, "SESSION_CLEANUP_INTERVAL_SEC", 30))
        await asyncio.sleep(interval)
        try:
            state.cleanup_expired_sessions()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    cleaner_task = asyncio.create_task(_periodic_session_cleaner())
    # Nạp nền mô hình OCR CCCD để lần quét đầu tiên không phải chờ
    if os.environ.get("EKYC_IDCARD_OCR_PRELOAD", "1") == "1" and "pytest" not in sys.modules:
        idcard_router.ocr_engine.warmup_async()
    try:
        yield
    finally:
        cleaner_task.cancel()
        try:
            await cleaner_task
        except asyncio.CancelledError:
            pass
        with state.sessions_lock:
            for s in list(state.sessions.values()):
                try:
                    s.close()
                except Exception:
                    pass
            state.sessions.clear()


app = FastAPI(
    title="Smart-eKYC Biometric PoC",
    description="Hệ thống eKYC Đa tầng - Khung Oval, FQA, Liveness Quay đầu & Zoom",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=C.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    """Kiểm soát tần suất gọi API (Rate Limiting) và vô hiệu hoá cache cho static files."""
    path = request.url.path
    if path.startswith("/api/"):
        client_ip = request.client.host if request.client else "127.0.0.1"
        try:
            rate_limiter.check(client_ip)
        except Exception as exc:
            status_code = getattr(exc, "status_code", 429)
            detail = getattr(exc, "detail", "Quá nhiều yêu cầu.")
            return JSONResponse(status_code=status_code, content={"detail": detail})

    response = await call_next(request)
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Đăng ký các API Routers
app.include_router(health_router.router)
app.include_router(enroll_router.router)
app.include_router(users_router.router)
app.include_router(verify_router.router)
app.include_router(idcard_router.router)

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
