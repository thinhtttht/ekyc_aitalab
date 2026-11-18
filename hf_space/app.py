"""FastAPI REST API for eKYC & Biometric Face Verification.
Designed for high performance, zero-latency caching, and deployment on Hugging Face Spaces.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.face_engine import FaceEngine
from core.database import init_db, save_user, get_user, list_users, delete_user, count_users, get_all_embeddings, get_db_dump, restore_db_dump

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Global Engine Instance
engine: Optional[FaceEngine] = None

STARTUP_BANNER = r"""
================================================================================
  ███████╗██████╗ ████████╗    ███████╗██╗  ██╗██╗   ██╗ ██████╗
  ██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝██║ ██╔╝╚██╗ ██╔╝██╔════╝
  █████╗  ██████╔╝   ██║       █████╗  █████╔╝  ╚████╔╝ ██║     
  ██╔══╝  ██╔═══╝    ██║       ██╔══╝  ██╔═██╗   ╚██╔╝  ██║     
  ██║     ██║        ██║       ██║     ██║  ██╗   ██║   ╚██████╗
  ╚═╝     ╚═╝        ╚═╝       ╚═╝     ╚═╝  ╚═╝   ╚═╝    ╚═════╝
             FPT UNIVERSITY - BIOMETRIC eKYC LAB SERVER
    Core AI: ArcFace 512D | MiniFASNet v2 Anti-Spoofing | MediaPipe
    Dedicated Exam Attendance & Face Verification System
================================================================================
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print(STARTUP_BANNER)
    print("[Startup] Initializing Database...")
    init_db()
    print(f"[Startup] Loading AI models from: {MODELS_DIR}...")
    engine = FaceEngine(models_dir=MODELS_DIR)
    print("[Startup] FPT eKYC Lab Server is online & ready to serve requests!")
    yield
    print("[Shutdown] Cleaning up resources...")


app = FastAPI(
    title="FPT University - eKYC & Biometric Face Verification API",
    description="Dedicated Biometric Authentication & Liveness Detection Server for FPT University Exam System.",
    version="2.1.0",
    lifespan=lifespan,
)

# Enable CORS for third-party web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Đọc ảnh từ file upload sang ma trận OpenCV BGR (không ghi file tạm vào đĩa)."""
    try:
        contents = upload_file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Không thể giải mã file ảnh.")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi đọc file ảnh {upload_file.filename}: {str(e)}",
        )


@app.get("/", tags=["General"])
async def root():
    """Trang chủ và kiểm tra trạng thái sức khỏe API."""
    return {
        "status": "online",
        "service": "FPT University eKYC & Biometric Face API",
        "institution": "FPT University",
        "version": "2.1.0",
        "registered_users": count_users(),
        "models": {
            "arcface": "w600k_r50.onnx (512D)",
            "anti_spoofing": "minifasnet_v2.onnx",
            "landmarks": "mediapipe_facemesh_478",
        },
        "docs_url": "/docs",
        "demo_client_url": "/demo",
    }


@app.get("/demo", response_class=HTMLResponse, tags=["General"])
async def demo_client():
    """Giao diện Web Client trực quan để test Webcam và gọi các API Đăng ký / Xác thực."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_test_app.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "client_test_app.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h3>client_test_app.html not found</h3>", status_code=404)


@app.post("/api/v1/register", tags=["eKYC Core"])
async def register_face(
    user_id: str = Form(..., description="Mã định danh duy nhất (MSSV, CCCD, Staff ID)"),
    full_name: str = Form(..., description="Họ và tên thí sinh / khách hàng"),
    image: UploadFile = File(..., description="File ảnh chân dung chuẩn (JPEG/PNG)"),
    enforce_oval: bool = Form(False, description="Bắt buộc đầu phải lọt trọn vẹn trong khung Oval"),
    require_liveness: bool = Form(True, description="Bắt buộc phải là người thật (Chống ảnh in / màn hình)"),
):
    """Đăng ký khuôn mặt thí sinh vào hệ thống.
    Thực hiện đầy đủ: FQA Policy Check -> MiniFASNet Liveness -> ArcFace 512D -> Lưu Database.
    """
    assert engine is not None, "FaceEngine chưa sẵn sàng"
    user_id = user_id.strip()
    full_name = full_name.strip()
    if not user_id or not full_name:
        raise HTTPException(status_code=400, detail="user_id và full_name không được để trống.")

    frame = read_image_from_upload(image)

    # 1. Phát hiện mốc khuôn mặt và bàn tay
    faces, hands = engine.detect_face_and_hands(frame)

    # 2. Đánh giá chất lượng & Policy FQA
    fqa = engine.evaluate_fqa_and_policy(frame, faces, hands, enforce_oval=enforce_oval)
    if not fqa["is_valid"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_code": fqa["error_code"],
                "message": fqa["message"],
                "policy_checks": fqa["policy_checks"],
                "metrics": fqa.get("metrics", {}),
            },
        )

    bbox = fqa["bbox"]
    landmarks = fqa["landmarks"]

    # 3. Kiểm tra người thật (Anti-Spoofing PAD)
    liveness = engine.check_liveness(frame, bbox)
    if require_liveness and not liveness["is_real"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error_code": "SPOOFING_DETECTED",
                "message": liveness["message"],
                "liveness_score": liveness["real_prob"],
                "policy_checks": fqa["policy_checks"],
            },
        )

    # 4. Trích xuất ArcFace 512D
    embedding = engine.extract_embedding(frame, landmarks)

    # 5. Lưu vào Database
    save_user(user_id=user_id, full_name=full_name, embedding=embedding)

    return {
        "success": True,
        "message": f"Đăng ký thành công thí sinh: {full_name} ({user_id})",
        "user_id": user_id,
        "full_name": full_name,
        "embedding_dim": len(embedding),
        "liveness_score": liveness["real_prob"],
        "policy_checks": fqa["policy_checks"],
        "warnings": fqa["warnings"],
    }


@app.post("/api/v1/verify", tags=["eKYC Core"])
async def verify_face(
    user_id: str = Form(..., description="Mã định danh cần xác thực điểm danh"),
    image: UploadFile = File(..., description="File ảnh webcam chụp trực tiếp lúc vào thi"),
    enforce_oval: bool = Form(False, description="Bắt buộc đầu phải lọt trọn vẹn trong khung Oval"),
    require_liveness: bool = Form(True, description="Bắt buộc kiểm tra người thật 3D/2D"),
    threshold: float = Form(0.55, description="Ngưỡng trùng khớp ArcFace Cosine (mặc định 0.55 ~ 55%)"),
):
    """Xác thực danh tính 1:1 giữa ảnh selfie hiện tại và vector đã đăng ký trong Database."""
    assert engine is not None, "FaceEngine chưa sẵn sàng"
    user_id = user_id.strip()

    # Truy vấn thí sinh trong DB
    user_record = get_user(user_id)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy hồ sơ người dùng với mã '{user_id}'. Vui lòng đăng ký trước.",
        )

    target_embedding = user_record["embedding"]
    frame = read_image_from_upload(image)

    # 1. Phát hiện mốc khuôn mặt và bàn tay
    faces, hands = engine.detect_face_and_hands(frame)

    # 2. Đánh giá chất lượng & Policy FQA
    fqa = engine.evaluate_fqa_and_policy(frame, faces, hands, enforce_oval=enforce_oval)
    if not fqa["is_valid"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "verified": False,
                "error_code": fqa["error_code"],
                "message": fqa["message"],
                "policy_checks": fqa["policy_checks"],
                "metrics": fqa.get("metrics", {}),
            },
        )

    bbox = fqa["bbox"]
    landmarks = fqa["landmarks"]

    # 3. Kiểm tra người thật (Anti-Spoofing PAD)
    liveness = engine.check_liveness(frame, bbox)
    if require_liveness and not liveness["is_real"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "verified": False,
                "error_code": "SPOOFING_DETECTED",
                "message": liveness["message"],
                "liveness_score": liveness["real_prob"],
                "policy_checks": fqa["policy_checks"],
            },
        )

    # 4. Trích xuất ArcFace 512D
    current_embedding = engine.extract_embedding(frame, landmarks)

    # 5. So khớp Cosine Similarity
    similarity = engine.compute_similarity(current_embedding, target_embedding)
    is_match = similarity >= threshold

    # Độ tự tin (%)
    if is_match:
        conf_pct = 60.0 + ((similarity - threshold) / max(0.01, (1.0 - threshold))) * 40.0
    else:
        conf_pct = max(0.0, (similarity / max(0.01, threshold))) * 59.9

    return {
        "verified": bool(is_match),
        "similarity": round(float(similarity), 4),
        "confidence_pct": round(min(100.0, max(0.0, conf_pct)), 1),
        "threshold": threshold,
        "is_real_person": bool(liveness["is_real"]),
        "liveness_score": liveness["real_prob"],
        "user_info": {
            "user_id": user_record["user_id"],
            "full_name": user_record["full_name"],
        },
        "message": "Xác thực danh tính thành công." if is_match else f"Khuôn mặt không khớp với hồ sơ ({int(similarity*100)}% < {int(threshold*100)}%).",
        "policy_checks": fqa["policy_checks"],
        "warnings": fqa["warnings"],
    }


@app.post("/api/v1/identify", tags=["eKYC Core"])
async def identify_face(
    image: UploadFile = File(..., description="File ảnh khuôn mặt cần tìm kiếm (1:N)"),
    threshold: float = Form(0.55, description="Ngưỡng trùng khớp ArcFace Cosine (mặc định 0.55)"),
    top_k: int = Form(3, description="Số lượng ứng viên phù hợp nhất muốn lấy"),
    enforce_oval: bool = Form(False, description="Bắt buộc đầu phải lọt trọn vẹn trong khung Oval"),
    require_liveness: bool = Form(True, description="Bắt buộc kiểm tra người thật 3D/2D"),
):
    """Tìm kiếm khuôn mặt 1:N trong toàn bộ Cơ sở dữ liệu (Không cần truyền user_id).
    Hệ thống sẽ đối chiếu vector với toàn bộ thí sinh và trả về người trùng khớp nhất.
    """
    assert engine is not None, "FaceEngine chưa sẵn sàng"

    all_users = get_all_embeddings()
    if not all_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cơ sở dữ liệu hiện chưa có thí sinh nào. Vui lòng đăng ký trước.",
        )

    frame = read_image_from_upload(image)

    # 1. Phát hiện khuôn mặt và bàn tay
    faces, hands = engine.detect_face_and_hands(frame)

    # 2. Đánh giá chất lượng & Policy FQA
    fqa = engine.evaluate_fqa_and_policy(frame, faces, hands, enforce_oval=enforce_oval)
    if not fqa["is_valid"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "found": False,
                "error_code": fqa["error_code"],
                "message": fqa["message"],
                "policy_checks": fqa["policy_checks"],
                "metrics": fqa.get("metrics", {}),
            },
        )

    bbox = fqa["bbox"]
    landmarks = fqa["landmarks"]

    # 3. Kiểm tra người thật (Anti-Spoofing PAD)
    liveness = engine.check_liveness(frame, bbox)
    if require_liveness and not liveness["is_real"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "found": False,
                "error_code": "SPOOFING_DETECTED",
                "message": liveness["message"],
                "liveness_score": liveness["real_prob"],
                "policy_checks": fqa["policy_checks"],
            },
        )

    # 4. Trích xuất ArcFace 512D
    query_emb = engine.extract_embedding(frame, landmarks)

    # 5. So sánh 1:N bằng tích ma trận (Vectorized Cosine Similarity)
    embeddings_matrix = np.vstack([u["embedding"] for u in all_users])
    sim_scores = np.dot(embeddings_matrix, query_emb)

    # Sắp xếp giảm dần theo độ tương đồng
    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        score = float(sim_scores[idx])
        u = all_users[idx]
        candidates.append({
            "user_id": u["user_id"],
            "full_name": u["full_name"],
            "similarity": round(score, 4),
            "is_match": bool(score >= threshold),
        })

    best_match = candidates[0] if candidates else None
    is_identified = best_match is not None and best_match["is_match"]

    return {
        "found": bool(is_identified),
        "total_database_users": len(all_users),
        "threshold": threshold,
        "is_real_person": bool(liveness["is_real"]),
        "liveness_score": liveness["real_prob"],
        "best_match": best_match if is_identified else None,
        "top_candidates": candidates,
        "message": f"Đã tìm thấy thí sinh: {best_match['full_name']} ({best_match['user_id']}) với độ khớp {int(best_match['similarity']*100)}%"
                   if is_identified else "Không tìm thấy thí sinh nào trùng khớp trong hệ thống (Người lạ).",
        "policy_checks": fqa["policy_checks"],
        "warnings": fqa["warnings"],
    }


@app.post("/api/v1/compare", tags=["eKYC Core"])
async def compare_faces(
    image_source: UploadFile = File(..., description="Ảnh gốc / Ảnh thẻ CCCD"),
    image_live: UploadFile = File(..., description="Ảnh chụp trực tiếp"),
    threshold: float = Form(0.55, description="Ngưỡng trùng khớp ArcFace Cosine"),
    enforce_oval: bool = Form(False, description="Bắt buộc ảnh live phải lọt Oval"),
):
    """So khớp trực tiếp 2 tấm ảnh 1:1 không cần lưu Database.
    Thích hợp cho bên thứ 3 tự quản lý database riêng của họ.
    """
    assert engine is not None, "FaceEngine chưa sẵn sàng"

    frame_src = read_image_from_upload(image_source)
    frame_live = read_image_from_upload(image_live)

    # Trích xuất ảnh gốc
    faces_src, _ = engine.detect_face_and_hands(frame_src)
    if len(faces_src) == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh nguồn (image_source).")
    lm_src = np.array([[p.x, p.y, p.z] for p in faces_src[0].landmark], dtype=np.float64)
    emb_src = engine.extract_embedding(frame_src, lm_src)

    # Trích xuất ảnh live + Check Policy
    faces_live, hands_live = engine.detect_face_and_hands(frame_live)
    fqa_live = engine.evaluate_fqa_and_policy(frame_live, faces_live, hands_live, enforce_oval=enforce_oval)
    if not fqa_live["is_valid"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "verified": False,
                "error_code": fqa_live["error_code"],
                "message": fqa_live["message"],
                "policy_checks": fqa_live["policy_checks"],
            },
        )

    liveness_live = engine.check_liveness(frame_live, fqa_live["bbox"])
    emb_live = engine.extract_embedding(frame_live, fqa_live["landmarks"])

    # Tính độ tương đồng
    similarity = engine.compute_similarity(emb_src, emb_live)
    is_match = similarity >= threshold

    return {
        "verified": bool(is_match),
        "similarity": round(float(similarity), 4),
        "threshold": threshold,
        "is_real_person": bool(liveness_live["is_real"]),
        "liveness_score": liveness_live["real_prob"],
        "message": "Trùng khớp khuôn mặt." if is_match else "Hai khuôn mặt không trùng khớp.",
        "policy_checks": fqa_live["policy_checks"],
    }


@app.get("/api/v1/users", tags=["User Management"])
async def get_users_list(limit: int = 100, offset: int = 0):
    """Lấy danh sách các thí sinh đã đăng ký."""
    return {
        "total": count_users(),
        "limit": limit,
        "offset": offset,
        "users": list_users(limit=limit, offset=offset),
    }


@app.delete("/api/v1/users/{user_id}", tags=["User Management"])
async def delete_registered_user(user_id: str):
    """Xóa hồ sơ thí sinh khỏi Database."""
    success = delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy người dùng '{user_id}'.")
    return {"success": True, "message": f"Đã xóa thành công người dùng '{user_id}'."}


@app.get("/api/v1/database/backup", tags=["Database Backup"])
async def download_database_backup():
    """Tải toàn bộ file SQLite về máy tính dự phòng (tránh mất dữ liệu khi HF Space sleep)."""
    raw_data = get_db_dump()
    return Response(
        content=raw_data,
        media_type="application/x-sqlite3",
        headers={"Content-Disposition": "attachment; filename=ekyc_backup.db"},
    )


@app.post("/api/v1/database/restore", tags=["Database Backup"])
async def upload_database_restore(file: UploadFile = File(...)):
    """Khôi phục Database từ file SQLite sao lưu."""
    content = file.file.read()
    restore_db_dump(content)
    return {"success": True, "message": "Đã khôi phục Database thành công!", "users_count": count_users()}
