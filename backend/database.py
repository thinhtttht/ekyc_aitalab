"""SQLite Database lưu trữ & quản lý hồ sơ người dùng sinh trắc học eKYC.

Lưu trữ vector nhúng ArcFace 512 chiều (dạng BLOB nhị phân float32) cùng ảnh chân dung HD.
Hỗ trợ tìm kiếm 1:N và xác thực 1:1 tốc độ cao.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from backend.feature_extractor import get_arcface_extractor
except ImportError:
    from feature_extractor import get_arcface_extractor

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "ekyc.db")

_db_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo cấu trúc bảng SQLite nếu chưa tồn tại."""
    with _db_lock:
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    snapshot_b64 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)")
            conn.commit()
        finally:
            conn.close()


class UserRepository:
    """Quản lý CRUD và đối sánh 1:1, 1:N cho hồ sơ sinh trắc học."""

    def __init__(self):
        init_db()
        self.extractor = get_arcface_extractor()

    def save_user(
        self,
        full_name: str,
        embedding: np.ndarray,
        snapshot_b64: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Tạo hoặc cập nhật hồ sơ người dùng với vector đặc trưng 512D."""
        if not user_id or not user_id.strip():
            user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
        else:
            user_id = user_id.strip()

        full_name = full_name.strip() if full_name else "Người dùng Chưa đặt tên"
        emb_arr = np.array(embedding, dtype=np.float32).flatten()
        if emb_arr.shape != (512,):
            raise ValueError(f"Vector ArcFace phải có 512 chiều, hiện tại: {emb_arr.shape}")

        # Chuẩn hoá L2 trước khi lưu
        norm = np.linalg.norm(emb_arr)
        if norm > 1e-6:
            emb_arr = emb_arr / norm

        emb_bytes = emb_arr.tobytes()
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)

        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO users (user_id, full_name, embedding, snapshot_b64, created_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        full_name = excluded.full_name,
                        embedding = excluded.embedding,
                        snapshot_b64 = excluded.snapshot_b64,
                        created_at = excluded.created_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (user_id, full_name, emb_bytes, snapshot_b64, created_at, meta_str),
                )
                conn.commit()
            finally:
                conn.close()

        preview = [round(float(v), 4) for v in emb_arr[:5]]
        return {
            "user_id": user_id,
            "full_name": full_name,
            "embedding_dim": 512,
            "embedding_preview": preview,
            "created_at": created_at,
            "snapshot_b64": snapshot_b64,
            "metadata": metadata or {},
        }

    def list_users(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả người dùng kèm vector preview và ảnh đại diện."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT user_id, full_name, embedding, snapshot_b64, created_at, metadata_json FROM users ORDER BY id DESC")
                rows = cur.fetchall()
            finally:
                conn.close()

        results = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            meta = {}
            if r["metadata_json"]:
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    pass

            results.append({
                "user_id": r["user_id"],
                "full_name": r["full_name"],
                "embedding_dim": len(emb),
                "embedding_preview": [round(float(v), 4) for v in emb[:5]],
                "created_at": r["created_at"],
                "snapshot_b64": r["snapshot_b64"],
                "metadata": meta,
            })
        return results

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết một người dùng kèm toàn bộ vector numpy 512D."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT user_id, full_name, embedding, snapshot_b64, created_at, metadata_json FROM users WHERE user_id = ?", (user_id,))
                r = cur.fetchone()
            finally:
                conn.close()

        if r is None:
            return None

        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        meta = {}
        if r["metadata_json"]:
            try:
                meta = json.loads(r["metadata_json"])
            except Exception:
                pass

        return {
            "user_id": r["user_id"],
            "full_name": r["full_name"],
            "embedding": emb,
            "embedding_dim": len(emb),
            "embedding_preview": [round(float(v), 4) for v in emb[:5]],
            "created_at": r["created_at"],
            "snapshot_b64": r["snapshot_b64"],
            "metadata": meta,
        }

    def delete_user(self, user_id: str) -> bool:
        """Xoá người dùng theo user_id."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                affected = cur.rowcount
                conn.commit()
            finally:
                conn.close()
        return affected > 0

    def match_user_1_to_1(
        self,
        query_embedding: np.ndarray,
        target_user_id: str,
        threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """So khớp đích danh 1:1 khuôn mặt người dùng với hồ sơ trong cơ sở dữ liệu."""
        target = self.get_user(target_user_id)
        if target is None:
            return {
                "is_match": False,
                "error": f"Không tìm thấy người dùng có ID {target_user_id}",
                "similarity": 0.0,
                "threshold": threshold,
            }

        verification = self.extractor.verify(query_embedding, target["embedding"], threshold=threshold)
        return {
            **verification,
            "target_user": {
                "user_id": target["user_id"],
                "full_name": target["full_name"],
                "snapshot_b64": target["snapshot_b64"],
                "created_at": target["created_at"],
            },
        }

    def match_user_1_to_n(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.45,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Định danh 1:N tìm kiếm người dùng tương đồng nhất trong toàn bộ cơ sở dữ liệu."""
        all_users = self.list_users()
        if not all_users:
            return {
                "is_match": False,
                "message": "Cơ sở dữ liệu người dùng đang trống. Chưa có hồ sơ đăng ký nào.",
                "similarity": 0.0,
                "top_match": None,
                "candidates": [],
            }

        candidates = []
        q_emb = np.array(query_embedding, dtype=np.float32).flatten()

        for u in all_users:
            full_u = self.get_user(u["user_id"])
            if not full_u:
                continue
            v_res = self.extractor.verify(q_emb, full_u["embedding"], threshold=threshold)
            candidates.append({
                "user_id": u["user_id"],
                "full_name": u["full_name"],
                "similarity": v_res["similarity"],
                "distance": v_res["distance"],
                "is_match": v_res["is_match"],
                "confidence_pct": v_res["confidence_pct"],
                "snapshot_b64": u["snapshot_b64"],
                "created_at": u["created_at"],
            })

        # Sắp xếp độ tương đồng giảm dần
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_match = candidates[0] if candidates else None
        is_match = bool(top_match and top_match["is_match"])

        return {
            "is_match": is_match,
            "threshold": threshold,
            "top_match": top_match if is_match else None,
            "candidates": candidates[:top_k],
        }


# Singleton instance
_user_repo_instance: Optional[UserRepository] = None


def get_user_repository() -> UserRepository:
    global _user_repo_instance
    if _user_repo_instance is None:
        _user_repo_instance = UserRepository()
    return _user_repo_instance
