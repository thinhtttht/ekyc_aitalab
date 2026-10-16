"""SQLite Database lưu trữ & quản lý hồ sơ người dùng sinh trắc học eKYC.

Lưu trữ vector nhúng ArcFace 512 chiều (dạng BLOB nhị phân float32) trong SQLite,
đồng thời tách và mã hóa ảnh chân dung (PII) lưu riêng trên đĩa (`data/portraits/*.enc`).
Hỗ trợ tìm kiếm 1:N vector hóa (BLAS Matrix-Vector) và xác thực 1:1 tốc độ cao.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from feature_extractor import get_arcface_extractor
    import config as C
except ImportError:
    from backend.feature_extractor import get_arcface_extractor
    import backend.config as C

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "ekyc.db")
PORTRAIT_DIR = os.path.join(DB_DIR, "portraits")
ENC_PREFIX = "enc_file://"

_db_lock = threading.Lock()


def _safe_filename(user_id: str) -> str:
    """Chuyển user_id thành tên file an toàn chống path traversal."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
    return f"{cleaned}.enc"


def _derive_keystream(key: bytes, nonce: bytes, length: int) -> np.ndarray:
    """Sinh chuỗi khóa giả ngẫu nhiên từ key + nonce bằng BLAKE2b ở chế độ counter."""
    blocks = (length + 63) // 64
    out = bytearray()
    for counter in range(blocks):
        h = hashlib.blake2b(
            nonce + counter.to_bytes(4, "little"),
            key=key[:64],
            digest_size=64,
        )
        out.extend(h.digest())
    return np.frombuffer(bytes(out[:length]), dtype=np.uint8)


def _encrypt_snapshot(raw_b64: str, secret: str) -> bytes:
    """Nén zlib và mã hóa có xác thực (Encrypt-then-MAC) chuỗi ảnh Base64."""
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    nonce = os.urandom(16)
    compressed = zlib.compress(raw_b64.encode("utf-8"), level=6)
    plain_arr = np.frombuffer(compressed, dtype=np.uint8)
    ks = _derive_keystream(key, nonce, len(plain_arr))
    cipher_bytes = np.bitwise_xor(plain_arr, ks).tobytes()
    mac = hmac.new(key, nonce + cipher_bytes, hashlib.sha256).digest()
    return nonce + mac + cipher_bytes


def _decrypt_snapshot(blob: bytes, secret: str) -> str:
    """Kiểm tra tính toàn vẹn HMAC và giải mã chuỗi ảnh Base64."""
    if len(blob) < 48:
        raise ValueError("Dữ liệu ảnh mã hóa bị hỏng")
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    nonce = blob[:16]
    mac_stored = blob[16:48]
    cipher_bytes = blob[48:]
    mac_calc = hmac.new(key, nonce + cipher_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(mac_stored, mac_calc):
        raise ValueError("Xác thực HMAC thất bại cho ảnh chân dung")
    cipher_arr = np.frombuffer(cipher_bytes, dtype=np.uint8)
    ks = _derive_keystream(key, nonce, len(cipher_arr))
    compressed = np.bitwise_xor(cipher_arr, ks).tobytes()
    return zlib.decompress(compressed).decode("utf-8")


def _save_encrypted_portrait(user_id: str, snapshot_b64: str) -> str:
    """Lưu ảnh chân dung đã mã hóa xuống đĩa và trả về URI tham chiếu."""
    os.makedirs(PORTRAIT_DIR, exist_ok=True)
    fname = _safe_filename(user_id)
    fpath = os.path.join(PORTRAIT_DIR, fname)
    secret = getattr(C, "DATA_ENCRYPTION_KEY", "qat-ekyc-biometric-secret-key-2026")
    enc_blob = _encrypt_snapshot(snapshot_b64, secret)
    with open(fpath, "wb") as f:
        f.write(enc_blob)
    return f"{ENC_PREFIX}{fname}"


def _resolve_portrait(stored_val: str) -> str:
    """Đọc và giải mã ảnh chân dung từ URI; tương thích ngược với dữ liệu Base64 cũ."""
    if not stored_val or not stored_val.startswith(ENC_PREFIX):
        return stored_val
    fname = os.path.basename(stored_val[len(ENC_PREFIX):])
    fpath = os.path.join(PORTRAIT_DIR, fname)
    if not os.path.exists(fpath):
        return ""
    try:
        secret = getattr(C, "DATA_ENCRYPTION_KEY", "qat-ekyc-biometric-secret-key-2026")
        with open(fpath, "rb") as f:
            return _decrypt_snapshot(f.read(), secret)
    except Exception:
        return ""


def _delete_portrait_file(user_id: str) -> None:
    fname = _safe_filename(user_id)
    fpath = os.path.join(PORTRAIT_DIR, fname)
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
        except OSError:
            pass


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
        """Tạo hoặc cập nhật hồ sơ người dùng với vector đặc trưng 512D và ảnh mã hóa."""
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
            stored_ref = _save_encrypted_portrait(user_id, snapshot_b64)
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
                    (user_id, full_name, emb_bytes, stored_ref, created_at, meta_str),
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
                "snapshot_b64": _resolve_portrait(r["snapshot_b64"]),
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
            "snapshot_b64": _resolve_portrait(r["snapshot_b64"]),
            "metadata": meta,
        }

    def delete_user(self, user_id: str) -> bool:
        """Xoá người dùng theo user_id kèm file ảnh mã hóa trên đĩa."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                affected = cur.rowcount
                conn.commit()
            finally:
                conn.close()
            if affected > 0:
                _delete_portrait_file(user_id)
        return affected > 0

    def delete_all_users(self) -> int:
        """Xoá toàn bộ hồ sơ người dùng khỏi CSDL SQLite và thư mục ảnh mã hóa."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM users")
                affected = cur.rowcount
                conn.commit()
            finally:
                conn.close()
            if os.path.exists(PORTRAIT_DIR):
                for fn in os.listdir(PORTRAIT_DIR):
                    if fn.endswith(".enc"):
                        try:
                            os.remove(os.path.join(PORTRAIT_DIR, fn))
                        except OSError:
                            pass
        return affected

    def match_user_1_to_1(
        self,
        query_embedding: np.ndarray,
        target_user_id: str,
        threshold: float = C.DEFAULT_VERIFY_THRESHOLD,
    ) -> Dict[str, Any]:
        """So khớp đích danh 1:1 khuôn mặt người dùng với hồ sơ trong cơ sở dữ liệu."""
        target = self.get_user(target_user_id)
        if target is None:
            return {
                "is_match": False,
                "error": f"Không tìm thấy người dùng có ID {target_user_id}",
                "similarity": 0.0,
                "distance": 2.0,
                "confidence_pct": 0.0,
                "threshold": threshold,
                "matched_user": None,
                "target_user": None,
            }

        verification = self.extractor.verify(query_embedding, target["embedding"], threshold=threshold)
        is_match = bool(verification["is_match"])
        user_info = {
            "user_id": target["user_id"],
            "full_name": target["full_name"],
            "snapshot_b64": target["snapshot_b64"],
            "created_at": target["created_at"],
        }
        return {
            **verification,
            "matched_user": user_info if is_match else None,
            "target_user": user_info,
        }

    def match_user_1_to_n(
        self,
        query_embedding: np.ndarray,
        threshold: float = C.DEFAULT_VERIFY_THRESHOLD,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Định danh 1:N vector hóa (Matrix-Vector Dot Product) trên toàn bộ cơ sở dữ liệu."""
        with _db_lock:
            conn = _get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT user_id, full_name, embedding, snapshot_b64, created_at FROM users")
                rows = cur.fetchall()
            finally:
                conn.close()

        if not rows:
            return {
                "is_match": False,
                "message": "Cơ sở dữ liệu người dùng đang trống. Chưa có hồ sơ đăng ký nào.",
                "similarity": 0.0,
                "distance": 2.0,
                "confidence_pct": 0.0,
                "threshold": threshold,
                "matched_user": None,
                "top_match": None,
                "candidates": [],
            }

        q_emb = np.array(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-6:
            q_emb = q_emb / q_norm

        valid_rows = []
        emb_list = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            if vec.shape == (512,):
                v_norm = np.linalg.norm(vec)
                if v_norm > 1e-6:
                    vec = vec / v_norm
                valid_rows.append(r)
                emb_list.append(vec)

        if not emb_list:
            return {
                "is_match": False,
                "message": "Không có vector hợp lệ trong CSDL.",
                "similarity": 0.0,
                "distance": 2.0,
                "confidence_pct": 0.0,
                "threshold": threshold,
                "matched_user": None,
                "top_match": None,
                "candidates": [],
            }

        # Tính đồng loạt Cosine Similarity bằng phép nhân ma trận (N x 512) @ (512,)
        emb_matrix = np.vstack(emb_list)
        sims = np.dot(emb_matrix, q_emb)
        sorted_indices = np.argsort(sims)[::-1][:top_k]

        candidates = []
        for idx in sorted_indices:
            r = valid_rows[int(idx)]
            v_res = self.extractor.verify(q_emb, emb_list[int(idx)], threshold=threshold)
            candidates.append({
                "user_id": r["user_id"],
                "full_name": r["full_name"],
                "similarity": v_res["similarity"],
                "distance": v_res["distance"],
                "is_match": v_res["is_match"],
                "confidence_pct": v_res["confidence_pct"],
                "snapshot_b64": _resolve_portrait(r["snapshot_b64"]),
                "created_at": r["created_at"],
            })

        top_match = candidates[0] if candidates else None
        is_match = bool(top_match and top_match["is_match"])

        matched_user_info = None
        if is_match and top_match:
            matched_user_info = {
                "user_id": top_match["user_id"],
                "full_name": top_match["full_name"],
                "snapshot_b64": top_match["snapshot_b64"],
                "created_at": top_match["created_at"],
            }

        return {
            "is_match": is_match,
            "threshold": threshold,
            "similarity": top_match["similarity"] if top_match else 0.0,
            "distance": top_match["distance"] if top_match else 2.0,
            "confidence_pct": top_match["confidence_pct"] if top_match else 0.0,
            "matched_user": matched_user_info,
            "top_match": top_match if is_match else None,
            "candidates": candidates,
        }


# Singleton instance
_user_repo_instance: Optional[UserRepository] = None


def get_user_repository() -> UserRepository:
    global _user_repo_instance
    if _user_repo_instance is None:
        _user_repo_instance = UserRepository()
    return _user_repo_instance
