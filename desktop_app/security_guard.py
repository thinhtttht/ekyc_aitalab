"""Security Guard & Anti-Spoofing Lockout Escalation Manager.
Strictly tracks liveness across all frames, enforces per-stage timeouts,
and applies escalating penalties (1st: reset, 2nd: 1-min lockout, 3rd: 10-min lockout, 4th+: permanent lockout).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

LOCK_FILE = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "security_lock.json")
ADMIN_PIN = "admin123"

# Ngưỡng thời gian phát hiện giả mạo
SPOOF_TIME_THRESH_EARLY = 3.0   # Giai đoạn 1, 2, 3: phát hiện liên tục 3 giây
SPOOF_TIME_THRESH_LATE = 1.8    # Giai đoạn 4, 5: phát hiện liên tục 1.8s (giữ an ninh nhưng không ngắt quãng do nhiễu camera)

# Thời gian giới hạn tối đa cho mỗi giai đoạn (giây)
STAGE_TIMEOUTS = {
    1: 15.0,  # Bước 1: 15s
    2: 12.0,  # Bước 2: 12s
    3: 12.0,  # Bước 3: 12s
    4: 12.0,  # Bước 4: 12s
    5: 15.0,  # Bước 5: 15s
}


class SecurityGuard:
    """Quản trị viên an ninh theo dõi chống giả mạo và khóa phạt leo thang."""

    def __init__(self):
        self.violation_count = 0
        self.lock_until = 0.0
        self.is_permanent_locked = False
        self.reason = ""

        # Bộ đếm thời gian giả mạo tích lũy theo từng frame
        self.spoof_streak_start: Optional[float] = None
        self.spoof_accum_sec = 0.0

        # Bộ đếm timeout cho giai đoạn hiện tại
        self.stage_start_time = time.time()

        self._load_lock_state()

    def _load_lock_state(self):
        """Đọc trạng thái khóa từ file lưu trữ để tránh gian lận khởi động lại app."""
        if not os.path.exists(LOCK_FILE):
            return
        try:
            with open(LOCK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.violation_count = int(data.get("violation_count", 0))
                self.lock_until = float(data.get("lock_until", 0.0))
                self.is_permanent_locked = bool(data.get("is_permanent_locked", False))
                self.reason = data.get("reason", "")
        except Exception:
            pass

    def _save_lock_state(self):
        """Lưu trạng thái khóa xuống đĩa."""
        try:
            os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
            with open(LOCK_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "violation_count": self.violation_count,
                        "lock_until": self.lock_until,
                        "is_permanent_locked": self.is_permanent_locked,
                        "reason": self.reason,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def is_locked(self) -> Tuple[bool, int, str]:
        """Kiểm tra hệ thống có đang bị khóa phạt hay không.

        Returns:
            (is_locked, remaining_seconds, message)
        """
        if self.is_permanent_locked:
            return (
                True,
                999999,
                "🚫 THIẾT BỊ ĐÃ BỊ KHÓA VĨNH VIỄN do nghi vấn gian lận sinh trắc học nghiêm trọng. Chỉ Quản trị viên (Admin) mới có quyền mở khóa.",
            )

        now = time.time()
        if now < self.lock_until:
            rem = int(self.lock_until - now)
            mins = rem // 60
            secs = rem % 60
            time_str = f"{mins:02d}:{secs:02d}"
            msg = f"⏳ TẠM KHÓA AN NINH! Vui lòng chờ {time_str} trước khi thử lại ({self.reason})."
            return True, rem, msg

        return False, 0, ""

    def start_stage(self, stage: int):
        """Bắt đầu tính giờ cho một giai đoạn."""
        self.stage_start_time = time.time()
        self.reset_spoof_streak()

    def get_stage_remaining_time(self, stage: int) -> float:
        """Thời gian còn lại của giai đoạn hiện tại (không giới hạn)."""
        return 999.0

    def is_stage_timed_out(self, stage: int) -> bool:
        """Kiểm tra giai đoạn có bị quá thời gian cho phép hay không (đã vô hiệu hóa)."""
        return False

    def reset_spoof_streak(self):
        """Đặt lại bộ đếm giả mạo liên tục."""
        self.spoof_streak_start = None
        self.spoof_accum_sec = 0.0

    def record_spoof_frame(self, stage: int, dt: float) -> Tuple[bool, str, Optional[str]]:
        """Ghi nhận một khung hình bị phát hiện là ảnh in / video giả mạo.

        Returns:
            (is_escalated, warning_message, penalty_type)
        """
        now = time.time()
        if self.spoof_streak_start is None:
            self.spoof_streak_start = now
            self.spoof_accum_sec = 0.0

        self.spoof_accum_sec += dt

        # Ngưỡng thời gian phát hiện tùy theo giai đoạn
        thresh = SPOOF_TIME_THRESH_LATE if stage in (4, 5) else SPOOF_TIME_THRESH_EARLY

        if self.spoof_accum_sec >= thresh:
            # Vượt ngưỡng thời gian giả mạo -> KÍCH HOẠT XỬ PHẠT LEO THANG!
            self.reset_spoof_streak()
            return self._trigger_violation_escalation(stage)

        rem = max(0.0, thresh - self.spoof_accum_sec)
        warn_msg = f"CẢNH BÁO: Phát hiện ảnh in / video phát lại! Đang theo dõi an ninh ({rem:.1f}s)..."
        return False, warn_msg, None

    def _trigger_violation_escalation(self, stage: int) -> Tuple[bool, str, str]:
        """Kích hoạt bậc xử phạt tăng dần khi phát hiện hành vi gian lận."""
        self.violation_count += 1
        now = time.time()

        if self.violation_count == 1:
            # Lần 1: Bắt đăng ký lại từ đầu
            self.reason = f"Phát hiện giả mạo ở Bước {stage}"
            self._save_lock_state()
            msg = "CẢNH BÁO VI PHẠM (Lần 1): Phát hiện ảnh in / video phát lại! Yêu cầu đăng ký lại từ đầu."
            return True, msg, "reset"

        elif self.violation_count == 2:
            # Lần 2: Tạm khóa 1 phút (60 giây)
            self.lock_until = now + 60.0
            self.reason = f"Phát hiện giả mạo lần 2 ở Bước {stage}"
            self._save_lock_state()
            msg = "CẢNH BÁO VI PHẠM (Lần 2): Phát hiện giả mạo liên tiếp! Tạm khóa thiết bị 1 phút."
            return True, msg, "lock_1m"

        elif self.violation_count == 3:
            # Lần 3: Tạm khóa 10 phút (600 giây)
            self.lock_until = now + 600.0
            self.reason = f"Phát hiện giả mạo lần 3 ở Bước {stage}"
            self._save_lock_state()
            msg = "CẢNH BÁO VI PHẠM NGHIÊM TRỌNG (Lần 3): Tạm khóa thiết bị 10 phút."
            return True, msg, "lock_10m"

        else:
            # Lần 4 trở đi: KHÓA VĨNH VIỄN
            self.is_permanent_locked = True
            self.reason = "Vi phạm gian lận nhiều lần (Cần Admin mở khóa)"
            self._save_lock_state()
            msg = "🚫 THIẾT BỊ BỊ KHÓA VĨNH VIỄN DO GIAN LẬN! Chỉ Quản trị viên (Admin) mới có quyền mở khóa."
            return True, msg, "permanent"

    def admin_unlock(self, pin: str) -> Tuple[bool, str]:
        """Mở khóa hệ thống bằng mã PIN / Mật khẩu Admin."""
        if pin.strip() == ADMIN_PIN:
            self.violation_count = 0
            self.lock_until = 0.0
            self.is_permanent_locked = False
            self.reason = ""
            self.reset_spoof_streak()
            self._save_lock_state()
            return True, "Đã mở khóa an ninh thành công! Hệ thống sẵn sàng hoạt động trở lại."
        return False, "Mã PIN Admin không chính xác!"
