"""Dịch vụ Giám sát Thi Vấn Đáp Trực Tuyến Chống Gian Lận (AI Oral Exam Proctoring).

Quản lý cơ chế xử phạt sinh trắc học leo thang:
- Lần 1: Cảnh báo nhắc nhở & bắt đầu lại lượt kiểm tra
- Lần 2: Trì hoãn 1 phút (Cooldown 60s)
- Lần 3: Trì hoãn 5 phút (Cooldown 300s)
- Lần 4: Đình chỉ thi vĩnh viễn (Banned) -> Đợi Giám thị/Admin mở lại
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from backend.database import _get_connection, _db_lock, get_user_repository
    from backend.face_analyzer import FaceAnalyzer
    from backend.feature_extractor import get_arcface_extractor
    import backend.config as C
except ImportError:
    from database import _get_connection, _db_lock, get_user_repository
    from face_analyzer import FaceAnalyzer
    from feature_extractor import get_arcface_extractor
    import config as C


def init_proctoring_db():
    """Khởi tạo bảng quản lý kỷ luật và xử phạt phòng thi."""
    with _db_lock:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS proctoring_penalties (
                    user_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    violation_count INTEGER DEFAULT 0,
                    lock_until REAL,
                    is_banned INTEGER DEFAULT 0,
                    last_reason TEXT,
                    last_violation_at TEXT,
                    history_json TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


class ProctoringService:
    """Quản lý logic giám sát thi vấn đáp và xử phạt leo thang."""

    def __init__(self):
        init_proctoring_db()
        self.user_repo = get_user_repository()
        self.analyzer = FaceAnalyzer(stream=False)
        self.extractor = get_arcface_extractor()

    def get_penalty_status(self, user_id: str) -> Dict[str, Any]:
        """Lấy trạng thái xử phạt hiện tại của thí sinh."""
        now = time.time()
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT user_id, full_name, violation_count, lock_until, is_banned, last_reason, last_violation_at FROM proctoring_penalties WHERE user_id = ?",
                    (user_id,)
                )
                row = cur.fetchone()
                if not row:
                    return {
                        "user_id": user_id,
                        "full_name": "",
                        "violation_count": 0,
                        "status": "active",  # active | warning | locked_1m | locked_5m | banned
                        "is_locked": False,
                        "is_banned": False,
                        "remaining_seconds": 0,
                        "last_reason": None,
                        "last_violation_at": None,
                    }

                v_count = row["violation_count"]
                lock_until = row["lock_until"]
                is_banned = bool(row["is_banned"])
                last_reason = row["last_reason"]
                last_at = row["last_violation_at"]
                full_name = row["full_name"]

                if is_banned:
                    return {
                        "user_id": user_id,
                        "full_name": full_name,
                        "violation_count": v_count,
                        "status": "banned",
                        "is_locked": True,
                        "is_banned": True,
                        "remaining_seconds": -1,
                        "message": "ĐÌNH CHỈ THI: Bạn đã vi phạm quy chế thi gian lận 4 lần. Tài khoản bị khóa, vui lòng đợi Giám thị mở lại.",
                        "last_reason": last_reason,
                        "last_violation_at": last_at,
                    }

                if lock_until and lock_until > now:
                    remaining = int(np.ceil(lock_until - now))
                    status_name = "locked_1m" if v_count == 2 else "locked_5m"
                    return {
                        "user_id": user_id,
                        "full_name": full_name,
                        "violation_count": v_count,
                        "status": status_name,
                        "is_locked": True,
                        "is_banned": False,
                        "remaining_seconds": remaining,
                        "message": f"Phòng thi bị tạm khóa! Thí sinh vi phạm lần {v_count}. Vui lòng chờ {remaining} giây.",
                        "last_reason": last_reason,
                        "last_violation_at": last_at,
                    }

                return {
                    "user_id": user_id,
                    "full_name": full_name,
                    "violation_count": v_count,
                    "status": "active" if v_count == 0 else "warning",
                    "is_locked": False,
                    "is_banned": False,
                    "remaining_seconds": 0,
                    "last_reason": last_reason,
                    "last_violation_at": last_at,
                }
            finally:
                conn.close()

    def record_violation(
        self, user_id: str, full_name: str, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ghi nhận một lần vi phạm gian lận và tính toán mức xử phạt leo thang."""
        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT violation_count, history_json FROM proctoring_penalties WHERE user_id = ?", (user_id,))
                row = cur.fetchone()

                history = []
                if row:
                    v_count = row["violation_count"] + 1
                    try:
                        history = json.loads(row["history_json"] or "[]")
                    except Exception:
                        history = []
                else:
                    v_count = 1

                # Áp dụng thang phạt
                lock_until = None
                is_banned = 0
                penalty_status = "warning"
                remaining_seconds = 0
                msg = ""

                if v_count == 1:
                    # Lần 1: Cảnh báo & bắt đầu lại
                    penalty_status = "warning"
                    msg = "Cảnh báo vi phạm lần 1: Phát hiện dấu hiệu bất thường! Vui lòng nhìn thẳng camera để xác thực lại."
                elif v_count == 2:
                    # Lần 2: Khóa 1 phút (60s)
                    lock_until = now + 60.0
                    penalty_status = "locked_1m"
                    remaining_seconds = 60
                    msg = "Phát hiện gian lận lần 2! Phòng thi bị tạm khóa 1 phút (Cooldown: 60s)."
                elif v_count == 3:
                    # Lần 3: Khóa 5 phút (300s)
                    lock_until = now + 300.0
                    penalty_status = "locked_5m"
                    remaining_seconds = 300
                    msg = "Phát hiện gian lận lần 3! Phòng thi bị tạm khóa 5 phút (Cooldown: 300s)."
                else:
                    # Lần 4+: Đình chỉ thi vĩnh viễn
                    is_banned = 1
                    penalty_status = "banned"
                    remaining_seconds = -1
                    msg = "ĐÌNH CHỈ THI: Bạn đã vi phạm quy chế thi gian lận quá số lần cho phép. Tài khoản đã bị khóa vĩnh viễn, vui lòng liên hệ Giám thị."

                history.append({
                    "count": v_count,
                    "reason": reason,
                    "penalty": penalty_status,
                    "timestamp": now_str,
                    "details": details or {}
                })
                history_str = json.dumps(history, ensure_ascii=False)

                cur.execute(
                    """
                    INSERT INTO proctoring_penalties (user_id, full_name, violation_count, lock_until, is_banned, last_reason, last_violation_at, history_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        full_name = excluded.full_name,
                        violation_count = excluded.violation_count,
                        lock_until = excluded.lock_until,
                        is_banned = excluded.is_banned,
                        last_reason = excluded.last_reason,
                        last_violation_at = excluded.last_violation_at,
                        history_json = excluded.history_json
                    """,
                    (user_id, full_name, v_count, lock_until, is_banned, reason, now_str, history_str)
                )
                conn.commit()

                return {
                    "user_id": user_id,
                    "full_name": full_name,
                    "violation_count": v_count,
                    "status": penalty_status,
                    "is_locked": lock_until is not None or is_banned == 1,
                    "is_banned": bool(is_banned),
                    "remaining_seconds": remaining_seconds,
                    "message": msg,
                    "last_reason": reason,
                    "last_violation_at": now_str,
                }
            finally:
                conn.close()

    def admin_unlock(self, user_id: str) -> Dict[str, Any]:
        """Giám thị mở khóa và khôi phục quyền thi cho thí sinh."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT history_json, full_name FROM proctoring_penalties WHERE user_id = ?", (user_id,))
                row = cur.fetchone()
                if not row:
                    return {"success": False, "message": f"Không tìm thấy thí sinh {user_id}"}

                full_name = row["full_name"]
                try:
                    history = json.loads(row["history_json"] or "[]")
                except Exception:
                    history = []

                history.append({
                    "action": "ADMIN_UNLOCKED",
                    "timestamp": now_str,
                    "message": "Giám thị phòng thi đã mở khóa và reset kỷ luật."
                })

                cur.execute(
                    """
                    UPDATE proctoring_penalties
                    SET violation_count = 0,
                        lock_until = NULL,
                        is_banned = 0,
                        last_reason = 'Đã được Giám thị mở khóa',
                        last_violation_at = ?,
                        history_json = ?
                    WHERE user_id = ?
                    """,
                    (now_str, json.dumps(history, ensure_ascii=False), user_id)
                )
                conn.commit()
                return {
                    "success": True,
                    "user_id": user_id,
                    "full_name": full_name,
                    "message": f"Đã mở khóa thành công cho thí sinh {full_name} ({user_id})! Thí sinh có thể tiếp tục làm bài thi."
                }
            finally:
                conn.close()

    def admin_list_all_penalties(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả thí sinh đang có tiền án vi phạm hoặc bị khóa."""
        now = time.time()
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM proctoring_penalties ORDER BY violation_count DESC, last_violation_at DESC")
                rows = cur.fetchall()
                res = []
                for r in rows:
                    lock_until = r["lock_until"]
                    is_banned = bool(r["is_banned"])
                    remaining = 0
                    if is_banned:
                        status = "banned"
                    elif lock_until and lock_until > now:
                        remaining = int(np.ceil(lock_until - now))
                        status = "locked_1m" if r["violation_count"] == 2 else "locked_5m"
                    elif r["violation_count"] > 0:
                        status = "warning"
                    else:
                        status = "active"

                    res.append({
                        "user_id": r["user_id"],
                        "full_name": r["full_name"],
                        "violation_count": r["violation_count"],
                        "status": status,
                        "is_locked": status in ("locked_1m", "locked_5m", "banned"),
                        "is_banned": is_banned,
                        "remaining_seconds": remaining,
                        "last_reason": r["last_reason"],
                        "last_violation_at": r["last_violation_at"],
                    })
                return res
            finally:
                conn.close()

    def verify_frame_for_exam(self, frame_bgr: np.ndarray, user_id: str) -> Dict[str, Any]:
        """Kiểm tra frame trong cửa sổ giám sát thi:

        - Kiểm tra xem thí sinh có đang bị khóa hay không
        - Kiểm tra 1 người duy nhất, không quay lệch quá lớn
        - Kiểm tra chống giả mạo MiniFASNet PAD (ảnh in, màn hình)
        - So khớp khuôn mặt 1:1 với ArcFace vector đã lưu của user_id
        """
        penalty_status = self.get_penalty_status(user_id)
        if penalty_status["is_locked"]:
            return {
                "is_valid": False,
                "penalty": penalty_status,
                "error_code": "LOCKED",
                "message": penalty_status.get("message", "Phòng thi đang bị khóa"),
            }

        user_data = self.user_repo.get_user_by_id(user_id)
        if not user_data:
            return {
                "is_valid": False,
                "error_code": "USER_NOT_FOUND",
                "message": f"Không tìm thấy dữ liệu hồ sơ của thí sinh {user_id}",
            }

        face = self.analyzer.analyze(frame_bgr, C.OVAL_NORMAL, fresh=True)

        if not face.detected or face.face_count == 0:
            return {
                "is_valid": False,
                "error_code": "ABSENT_NO_FACE",
                "message": "Không phát hiện khuôn mặt thí sinh trước camera!",
                "face_count": 0,
            }

        if face.face_count > 1:
            return {
                "is_valid": False,
                "error_code": "MULTIPLE_FACES",
                "message": f"Phát hiện {face.face_count} người trong phòng thi! Quy chế chỉ cho phép 1 thí sinh duy nhất.",
                "face_count": face.face_count,
            }

        if face.is_upside_down:
            return {
                "is_valid": False,
                "error_code": "BAD_POSE",
                "message": "Góc đầu bị lật ngược hoặc sai lệch nghiêm trọng.",
            }

        # Kiểm tra chống giả mạo (MiniFASNet v2)
        spoof = face.anti_spoof
        if spoof and not spoof.is_real:
            spoof_code = "PRINT_ATTACK" if spoof.spoof_type in ("print_attack", "planar_2d") else "REPLAY_ATTACK"
            return {
                "is_valid": False,
                "error_code": spoof_code,
                "message": f"PHÁT HIỆN GIAN LẬN: {spoof.message}",
                "spoof_details": {"real_prob": spoof.real_prob, "spoof_type": spoof.spoof_type},
            }

        # Trích xuất ArcFace 512D và so khớp 1:1 với hồ sơ thí sinh
        if not self.extractor.is_ready:
            return {
                "is_valid": True,
                "message": "Đang chạy chế độ mô phỏng AI (ArcFace model chưa nạp)",
                "similarity": 0.95,
            }

        emb = self.extractor.extract_embedding(frame_bgr, landmarks_5pts=face.arcface_kps, bbox=face.bbox)
        if emb is None:
            return {
                "is_valid": False,
                "error_code": "EXTRACTION_FAILED",
                "message": "Không thể trích xuất đặc trưng sinh trắc học từ khuôn mặt",
            }

        # So khớp 1:1 với user_data["embedding"]
        target_emb = np.frombuffer(user_data["embedding"], dtype=np.float32)
        target_norm = np.linalg.norm(target_emb)
        if target_norm > 1e-6:
            target_emb = target_emb / target_norm

        sim = float(np.dot(emb, target_emb))
        distance = float(np.linalg.norm(emb - target_emb))

        match_ok = sim >= C.MATCH_THRESHOLD
        if not match_ok:
            return {
                "is_valid": False,
                "error_code": "WRONG_PERSON",
                "similarity": round(sim, 3),
                "distance": round(distance, 3),
                "message": f"Khuôn mặt trước camera không khớp với thí sinh {user_data['full_name']} (Độ tương đồng: {round(sim*100, 1)}% < {int(C.MATCH_THRESHOLD*100)}%)!",
            }

        return {
            "is_valid": True,
            "user_id": user_id,
            "full_name": user_data["full_name"],
            "similarity": round(sim, 3),
            "distance": round(distance, 3),
            "message": f"Xác thực hợp lệ: Đúng thí sinh {user_data['full_name']} (Độ tin cậy: {round(sim*100, 1)}%)",
        }


# Singleton
_proctoring_service_instance: Optional[ProctoringService] = None


def get_proctoring_service() -> ProctoringService:
    global _proctoring_service_instance
    if _proctoring_service_instance is None:
        _proctoring_service_instance = ProctoringService()
    return _proctoring_service_instance
