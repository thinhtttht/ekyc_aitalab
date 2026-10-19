"""API Router cho bước quét & OCR thẻ Căn cước công dân (CCCD) trước khi đăng ký khuôn mặt."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from id_card_ocr import engine as ocr_engine
from security import decode_or_400
import state

router = APIRouter(tags=["ID Card"])


class IdCardScanPayload(BaseModel):
    image: str = Field(description="Base64 JPEG (data URL hoặc raw) vùng khung chữ nhật chứa mặt trước CCCD")
    known_fields: Optional[dict] = Field(
        default=None,
        description="Các trường đã trích xuất được ở các frame trước (để khóa Tên & Số CCCD và chỉ quét phần dưới)",
    )


class IdCardCheckPayload(BaseModel):
    id_number: Optional[str] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None


@router.get("/api/idcard/status")
def idcard_status():
    """Trạng thái mô hình OCR (đã nạp / đang nạp / lỗi). Tự kích hoạt nạp nền nếu chưa có."""
    ocr_engine.warmup_async()
    return ocr_engine.status()


@router.post("/api/idcard/scan")
def idcard_scan(payload: IdCardScanPayload):
    """Nhận diện thẻ trong khung, kiểm tra chất lượng và OCR trích xuất thông tin."""
    return ocr_engine.scan(decode_or_400(payload.image), known_fields=payload.known_fields)


@router.post("/api/idcard/check")
def idcard_check(payload: IdCardCheckPayload):
    """Kiểm tra CCCD đã được đăng ký trong CSDL hay chưa."""
    existing = state.user_repo.find_by_identity(
        id_number=payload.id_number,
        full_name=payload.full_name,
        dob=payload.dob,
        gender=payload.gender,
    )
    return {"registered": existing is not None, "user": existing}
