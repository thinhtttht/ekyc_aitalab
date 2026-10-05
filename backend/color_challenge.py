"""Module sinh chuỗi màu ngẫu nhiên và Token xác thực cho Color Flashing.

Tuân thủ chuẩn ASD-STE100 và ISO/IEC 30107-3 Presentation Attack Detection.
"""
from __future__ import annotations

import hashlib
import hmac
import random
import time
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Bảng 5 màu quang học tiêu chuẩn an toàn thị giác W3C WCAG 2.1
COLOR_PALETTE: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("EMERALD_GREEN", "#00E676", (0, 230, 118)),
    ("CORAL_RED", "#FF3D00", (255, 61, 0)),
    ("SKY_BLUE", "#2979FF", (41, 121, 255)),
    ("AMBER_YELLOW", "#FFC400", (255, 196, 0)),
    ("MAGENTA_PINK", "#F50057", (245, 0, 87)),
]

CHALLENGE_TTL_SEC: float = 15.0
CHALLENGE_SECRET: bytes = b"smart_ekyc_optical_flashing_secret_key_2026"


class ColorStep(BaseModel):
    """Mô hình dữ liệu cho từng bước nháy màu quang học."""
    index: int = Field(..., description="Chỉ số bước nháy, bắt đầu từ 0")
    name: str = Field(..., description="Tên định danh màu sắc")
    hex: str = Field(..., description="Mã màu HEX cho hiển thị CSS")
    rgb: Tuple[int, int, int] = Field(..., description="Giá trị (R, G, B) từ 0 đến 255")
    duration_ms: int = Field(330, description="Thời lượng hiển thị tính bằng mili-giây")


class ChallengeSessionData(BaseModel):
    """Dữ liệu lưu trữ phiên thách thức quang học."""
    session_id: str
    token: str
    sequence: List[ColorStep]
    created_at: float
    expires_at: float
    consumed: bool = False


class ChallengeManager:
    """Quản lý việc tạo và kiểm tra chuỗi màu quang học cho từng phiên eKYC."""

    def __init__(self, ttl_sec: float = CHALLENGE_TTL_SEC) -> None:
        self.ttl_sec = ttl_sec
        self._challenges: Dict[str, ChallengeSessionData] = {}

    def cleanup(self) -> None:
        """Dọn dẹp các token đã quá hạn."""
        now = time.time()
        expired_keys = [
            sid for sid, data in self._challenges.items() if now > data.expires_at
        ]
        for sid in expired_keys:
            self._challenges.pop(sid, None)

    def create_challenge(self, session_id: str, length: int = 3) -> ChallengeSessionData:
        """Sinh chuỗi màu ngẫu nhiên không trùng màu kế tiếp và tạo Token HMAC.

        Args:
            session_id: Mã định danh phiên eKYC.
            length: Số lượng bước nháy (mặc định 3 màu).

        Returns:
            ChallengeSessionData chứa token và danh sách các bước màu.
        """
        self.cleanup()
        length = max(2, min(length, 5))

        # Chọn ngẫu nhiên chuỗi màu đảm bảo 2 màu liên tiếp không trùng nhau
        sequence_steps: List[ColorStep] = []
        last_idx = -1
        available_indices = list(range(len(COLOR_PALETTE)))

        for i in range(length):
            candidates = [idx for idx in available_indices if idx != last_idx]
            chosen_idx = random.choice(candidates)
            name, hex_code, rgb = COLOR_PALETTE[chosen_idx]
            sequence_steps.append(
                ColorStep(
                    index=i,
                    name=name,
                    hex=hex_code,
                    rgb=rgb,
                    duration_ms=330,
                )
            )
            last_idx = chosen_idx

        now = time.time()
        expires_at = now + self.ttl_sec

        # Tạo mã Token HMAC-SHA256
        seq_str = ",".join(s.name for s in sequence_steps)
        token_payload = f"{session_id}:{seq_str}:{now}".encode("utf-8")
        token = hmac.new(CHALLENGE_SECRET, token_payload, hashlib.sha256).hexdigest()

        challenge_data = ChallengeSessionData(
            session_id=session_id,
            token=token,
            sequence=sequence_steps,
            created_at=now,
            expires_at=expires_at,
            consumed=False,
        )
        self._challenges[session_id] = challenge_data
        return challenge_data

    def verify_token(
        self, session_id: str, token: str
    ) -> Tuple[bool, Optional[ChallengeSessionData], str]:
        """Xác thực tính hợp lệ của Token thách thức.

        Args:
            session_id: Mã phiên.
            token: Token do client gửi lên.

        Returns:
            Tuple (is_valid, challenge_data, reason).
        """
        self.cleanup()
        data = self._challenges.get(session_id)
        if data is None:
            return False, None, "Phiên thách thức không tồn tại hoặc đã hết hạn"

        if data.consumed:
            return False, None, "Token thách thức đã được sử dụng trước đó"

        now = time.time()
        if now > data.expires_at:
            self._challenges.pop(session_id, None)
            return False, None, "Token thách thức đã quá hạn thời gian hiệu lực"

        if not hmac.compare_digest(data.token, token):
            return False, None, "Token thách thức không hợp lệ hoặc bị can thiệp"

        return True, data, "Token hợp lệ"

    def consume_challenge(self, session_id: str) -> None:
        """Đánh dấu thách thức đã sử dụng để chống tấn công phát lại."""
        if session_id in self._challenges:
            self._challenges[session_id].consumed = True


# Khởi tạo thể hiện Singleton mặc định
challenge_manager = ChallengeManager()
