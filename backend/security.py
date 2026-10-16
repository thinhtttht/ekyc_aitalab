"""Cơ chế bảo mật cho hệ thống eKYC: Xác thực API Key, Rate Limiting và Kiểm tra kích thước ảnh."""
from __future__ import annotations

import base64
from collections import defaultdict, deque
import hmac
import threading
import time
from typing import Deque, Dict, Optional

import cv2
import numpy as np
from fastapi import Header, HTTPException, Request

import config as C


class SlidingWindowRateLimiter:
    """Bộ giới hạn tần suất gọi API theo cửa sổ trượt 60 giây trên mỗi địa chỉ IP."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_ip: str) -> None:
        limit = getattr(C, "RATE_LIMIT_PER_MINUTE", 600)
        if limit <= 0:
            return
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            q = self._hits[client_ip]
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail="Quá nhiều yêu cầu trong thời gian ngắn. Vui lòng thử lại sau ít giây.",
                )
            q.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = SlidingWindowRateLimiter()


def verify_admin_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    """Dependency kiểm tra quyền quản trị nếu EKYC_ADMIN_API_KEY được cấu hình."""
    expected_key = getattr(C, "ADMIN_API_KEY", "").strip()
    if not expected_key:
        return

    provided = x_api_key or ""
    if not provided and authorization:
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        else:
            provided = authorization.strip()

    if not provided or not hmac.compare_digest(provided, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Yêu cầu xác thực quản trị viên (Thiếu hoặc sai X-API-Key / Bearer Token).",
        )


def decode_base64_image(raw_b64: str) -> np.ndarray:
    """Giải mã ảnh base64 (hỗ trợ cả Data URL và chuỗi thuần) kèm kiểm tra dung lượng."""
    max_bytes = getattr(C, "MAX_IMAGE_PAYLOAD_BYTES", 10 * 1024 * 1024)
    if len(raw_b64) > max_bytes:
        raise OverflowError(f"Kích thước dữ liệu ảnh vượt quá giới hạn ({max_bytes} bytes)")

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(raw_b64)
    if len(img_bytes) > max_bytes:
        raise OverflowError(f"Kích thước ảnh giải mã vượt quá giới hạn ({max_bytes} bytes)")

    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không thể decode ảnh JPEG/PNG")
    return frame


def decode_or_400(raw_b64: str) -> np.ndarray:
    try:
        return decode_base64_image(raw_b64)
    except OverflowError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu ảnh không hợp lệ: {e}")
