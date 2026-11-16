"""Database Module for eKYC API.
Supports lightweight local SQLite by default, or PostgreSQL (Supabase) via DATABASE_URL.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional, List, Dict, Any
import numpy as np

DB_FILE = os.environ.get("SQLITE_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ekyc.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_connection():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_updated ON users(updated_at);")
    conn.close()


def save_user(user_id: str, full_name: str, embedding: np.ndarray) -> bool:
    """Lưu hoặc cập nhật người dùng với vector ArcFace 512D."""
    emb_bytes = embedding.astype(np.float32).tobytes()
    now = time.time()
    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO users (user_id, full_name, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                full_name=excluded.full_name,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at;
        """, (user_id, full_name, emb_bytes, now, now))
    conn.close()
    return True


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Truy vấn thông tin người dùng và vector 512D."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, full_name, embedding, created_at, updated_at FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    emb = np.frombuffer(row["embedding"], dtype=np.float32)
    return {
        "user_id": row["user_id"],
        "full_name": row["full_name"],
        "embedding": emb,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_users(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, full_name, created_at, updated_at FROM users ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "user_id": r["user_id"],
            "full_name": r["full_name"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def delete_user(user_id: str) -> bool:
    conn = get_connection()
    with conn:
        cur = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        count = cur.rowcount
    conn.close()
    return count > 0


def count_users() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_db_dump() -> bytes:
    """Đọc toàn bộ file SQLite ra mảng bytes để sao lưu."""
    if not os.path.exists(DB_FILE):
        return b""
    with open(DB_FILE, "rb") as f:
        return f.read()


def restore_db_dump(raw_bytes: bytes) -> bool:
    """Ghi đè file SQLite từ bản sao lưu."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with open(DB_FILE, "wb") as f:
        f.write(raw_bytes)
    return True
