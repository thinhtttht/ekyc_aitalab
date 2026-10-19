"""API Router quản lý hồ sơ người dùng sinh trắc học (CRUD & Enrollment)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import config as C
from enrollment import Stage
from security import decode_or_400, verify_admin_auth
import state

router = APIRouter(tags=["Users"])


class UserEnrollPayload(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    full_name: str = Field(min_length=1, max_length=120)
    image: str = Field(description="Base64 encoded JPEG data URL hoặc raw base64")
    metadata: Optional[dict] = None


@router.post("/api/users/enroll")
def enroll_user(payload: UserEnrollPayload):
    """Lưu hồ sơ người dùng kèm vector ArcFace 512D vào cơ sở dữ liệu."""
    frame_bgr = decode_or_400(payload.image)
    require_session = getattr(C, "ENROLL_REQUIRE_SESSION", True)

    if payload.session_id:
        with state.sessions_lock:
            sess = state.sessions.get(payload.session_id)
        if sess is None:
            raise HTTPException(
                status_code=403,
                detail="Phiên đăng ký không tồn tại hoặc đã hết hạn. Vui lòng thực hiện lại quy trình eKYC.",
            )
        if sess.stage != Stage.CAPTURE or sess.final_embedding is None:
            raise HTTPException(
                status_code=403,
                detail="Phiên đăng ký chưa hoàn tất bước kiểm định ảnh chân dung cuối (verify_final_capture).",
            )
        emb = sess.final_embedding
    else:
        if require_session:
            raise HTTPException(
                status_code=403,
                detail="Bắt buộc phải có session_id hợp lệ đã vượt qua quy trình kiểm tra người thật (Liveness & PAD).",
            )
        with state.verifier_lock:
            face_res = state.verifier_analyzer.analyze(frame_bgr, oval=C.OVAL_NORMAL, fresh=True)
        kps = face_res.arcface_kps if face_res.detected else None
        bbox = face_res.bbox if face_res.detected else None
        try:
            emb = state.arcface_extractor.extract_embedding(
                frame_bgr,
                landmarks_5pts=kps,
                bbox=bbox,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi trích xuất ArcFace: {e}")

    # Chặn đăng ký trùng khi hồ sơ đến từ bước OCR CCCD
    meta = payload.metadata or {}
    if meta.get("id_number"):
        existing = state.user_repo.find_by_identity(
            id_number=meta.get("id_number"),
            full_name=payload.full_name,
            dob=meta.get("dob"),
            gender=meta.get("gender"),
        )
        if existing and existing["user_id"] != (payload.user_id or "").strip():
            raise HTTPException(
                status_code=409,
                detail=f"CCCD này đã được đăng ký (User ID: {existing['user_id']} - {existing['full_name']}).",
            )

    # Đảm bảo format Data URL
    snapshot_b64 = payload.image
    if not snapshot_b64.startswith("data:image/"):
        snapshot_b64 = f"data:image/jpeg;base64,{snapshot_b64}"

    user = state.user_repo.save_user(
        full_name=payload.full_name,
        embedding=emb,
        snapshot_b64=snapshot_b64,
        user_id=payload.user_id,
        metadata=payload.metadata,
    )
    return {
        "status": "ok",
        "message": f"Đã lưu hồ sơ sinh trắc học thành công cho '{user['full_name']}'!",
        "user": user,
    }


@router.get("/api/users", dependencies=[Depends(verify_admin_auth)])
def get_users_list():
    """Lấy danh sách tất cả người dùng trong CSDL phục vụ UI quản lý."""
    users = state.user_repo.list_users()
    return {"status": "ok", "total": len(users), "users": users}


@router.get("/api/users/{user_id}", dependencies=[Depends(verify_admin_auth)])
def get_user_detail(user_id: str):
    """Lấy chi tiết hồ sơ người dùng theo user_id."""
    u = state.user_repo.get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    return {
        "user_id": u["user_id"],
        "full_name": u["full_name"],
        "embedding": [round(float(v), 6) for v in u["embedding"]],
        "embedding_dim": u["embedding_dim"],
        "embedding_preview": u["embedding_preview"],
        "created_at": u["created_at"],
        "snapshot_b64": u["snapshot_b64"],
        "metadata": u["metadata"],
    }


@router.delete("/api/users/{user_id}", dependencies=[Depends(verify_admin_auth)])
def delete_user_record(user_id: str):
    """Xoá một hồ sơ người dùng khỏi CSDL."""
    deleted = state.user_repo.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng để xoá")
    return {"status": "ok", "message": f"Đã xoá người dùng {user_id}"}


@router.delete("/api/users", dependencies=[Depends(verify_admin_auth)])
def delete_all_users_records():
    """Xoá toàn bộ hồ sơ người dùng khỏi CSDL SQLite."""
    count = state.user_repo.delete_all_users()
    return {
        "status": "ok",
        "message": f"Đã xoá toàn bộ {count} hồ sơ người dùng thành công",
        "deleted_count": count,
    }
