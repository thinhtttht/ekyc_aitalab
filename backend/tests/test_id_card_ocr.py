"""Unit & integration tests cho module nhận diện, OCR CCCD và kiểm tra trùng lặp danh tính."""
from __future__ import annotations

import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import app
from database import UserRepository
from id_card_ocr import (
    assess_quality,
    engine as ocr_engine,
    group_lines,
    is_id_card_text,
    locate_card,
    parse_cccd_lines,
    warp_card,
)


def _encode_jpg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def test_parse_cccd_2021_chip_format():
    lines = [
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc",
        "CĂN CƯỚC CÔNG DÂN",
        "Citizen Identity Card",
        "Số / No.: 079203001234",
        "Họ và tên / Full name:",
        "NGUYỄN VĂN AN",
        "Ngày sinh / Date of birth: 15/08/2003",
        "Giới tính / Sex: Nam   Quốc tịch / Nationality: Việt Nam",
        "Quê quán / Place of origin: Quận 1, TP. Hồ Chí Minh",
        "Có giá trị đến / Date of expiry: 15/08/2028",
    ]
    res = parse_cccd_lines(lines)
    assert res == {
        "id_number": "079203001234",
        "full_name": "NGUYỄN VĂN AN",
        "dob": "15/08/2003",
        "gender": "Nam",
    }
    assert is_id_card_text(lines)


def test_parse_can_cuoc_2024_female_and_ocr_noise():
    lines = [
        "CĂN CƯỚC / IDENTITY CARD",
        "Số định danh cá nhân / Personal identification number",
        "0011 9800 567O",  # chữ O cuối bị nhầm thay vì 0, có khoảng trắng giữa các cụm số
        "Họ, chữ đệm và tên khai sinh / Full name TRẦN THỊ BÍCH NGỌC",  # mất dấu ':'
        "Ngày, tháng, năm sinh / Date of birth: 02-11-1998",
        "Giới tính / Sex: Nữ   Quốc tịch / Nationality: Việt Nam",
    ]
    res = parse_cccd_lines(lines)
    assert res == {
        "id_number": "001198005670",
        "full_name": "TRẦN THỊ BÍCH NGỌC",
        "dob": "02/11/1998",
        "gender": "Nữ",
    }


def test_locate_and_warp_card_on_synthetic_scene():
    scene = np.full((720, 1280, 3), 25, dtype=np.uint8)
    # Vẽ thẻ trắng tỉ lệ ~1.586 ở giữa khung
    card_w, card_h = 640, 404
    x0, y0 = (1280 - card_w) // 2, (720 - card_h) // 2
    cv2.rectangle(scene, (x0, y0), (x0 + card_w, y0 + card_h), (235, 235, 235), -1)
    cv2.putText(scene, "CAN CUOC CONG DAN", (x0 + 80, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(scene, "079203001234", (x0 + 80, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)

    quad = locate_card(scene)
    assert quad is not None
    assert quad.shape == (4, 2)

    warped = warp_card(scene, quad)
    assert warped.shape[1] == 1200
    q = assess_quality(warped)
    assert q["ok"] is True


def test_group_lines_reading_order():
    raw = [
        ([[200, 10], [320, 10], [320, 40], [200, 40]], "NGUYỄN", 0.92),
        ([[330, 12], [420, 12], [420, 40], [330, 40]], "VĂN A", 0.90),
        ([[200, 70], [450, 70], [450, 100], [200, 100]], "15/08/2003", 0.95),
    ]
    assert group_lines(raw) == ["NGUYỄN VĂN A", "15/08/2003"]


def test_find_by_identity_and_api_check_and_duplicate_block(monkeypatch):
    repo = UserRepository()
    repo.delete_all_users()

    emb = np.ones(512, dtype=np.float32)
    saved = repo.save_user(
        full_name="NGUYỄN VĂN AN",
        embedding=emb,
        snapshot_b64="data:image/jpeg;base64,AA==",
        metadata={
            "id_number": "079203001234",
            "dob": "15/08/2003",
            "gender": "Nam",
            "identity_source": "cccd_ocr",
        },
    )

    # 1. Khớp theo Số CCCD
    by_id = repo.find_by_identity(id_number="079203001234")
    assert by_id is not None
    assert by_id["user_id"] == saved["user_id"]
    assert by_id["matched_by"] == "id_number"

    # 2. Khớp theo tổ hợp (Họ tên + Ngày sinh + Giới tính)
    by_person = repo.find_by_identity(
        id_number="999999999999",
        full_name="nguyễn văn an",
        dob="15/08/2003",
        gender="Nam",
    )
    assert by_person is not None
    assert by_person["user_id"] == saved["user_id"]
    assert by_person["matched_by"] == "name_dob_gender"

    # 3. Không khớp khi khác thông tin
    assert repo.find_by_identity(id_number="000000000000", full_name="LÊ THỊ B", dob="01/01/2000", gender="Nữ") is None

    # 4. Kiểm tra qua API /api/idcard/check và /api/idcard/scan (với mock read_lines)
    client = TestClient(app)
    r_chk = client.post("/api/idcard/check", json={"id_number": "079203001234"})
    assert r_chk.status_code == 200
    assert r_chk.json()["registered"] is True

    monkeypatch.setattr(ocr_engine, "_reader", object())
    monkeypatch.setattr(
        ocr_engine,
        "read_lines",
        lambda _img: [
            "CĂN CƯỚC CÔNG DÂN",
            "Số / No.: 079203001234",
            "Họ và tên / Full name: NGUYỄN VĂN AN",
            "Ngày sinh / Date of birth: 15/08/2003",
            "Giới tính / Sex: Nam   Quốc tịch: Việt Nam",
        ],
    )

    card_img = np.full((600, 960, 3), 210, dtype=np.uint8)
    for y in range(80, 520, 60):
        cv2.putText(card_img, "CAN CUOC 079203001234", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)

    r_scan = client.post("/api/idcard/scan", json={"image": _encode_jpg_b64(card_img)})
    assert r_scan.status_code == 200
    scan_data = r_scan.json()
    assert scan_data["complete"] is True
    assert scan_data["fields"] == {
        "id_number": "079203001234",
        "full_name": "NGUYỄN VĂN AN",
        "dob": "15/08/2003",
        "gender": "Nam",
    }

    repo.delete_all_users()


def test_phase2_locked_name_id_and_noisy_dob_gender(monkeypatch):
    # Khi đã khóa Tên & Số CCCD: giữ nguyên không đổi, kể cả khi ảnh zoom chỉ có phần dưới
    # và dấu '/' trong ngày sinh bị OCR đọc nhầm thành '1' (1510812003) hoặc mất nhãn Giới tính
    known = {
        "id_number": "079203001234",
        "full_name": "NGUYỄN HOÀI NAM",
        "dob": None,
        "gender": None,
    }
    zoomed_lines = [
        "NGUYỄN HOÀI NAM",  # dòng tên (chứa chữ NAM trong tên, không được nhầm sang giới tính)
        "Ngav sinh / Date of birth 1510812003",  # dấu '/' bị đọc nhầm thành '1'
        "G10i tinh / Sex:Nữ Quốc tịch Việt Nam",  # nhãn bị nhiễu, Nữ dính sát dấu ':'
        "Quê quán: Nam Định",
    ]
    parsed = parse_cccd_lines(zoomed_lines, known_fields=known)
    assert parsed == {
        "id_number": "079203001234",
        "full_name": "NGUYỄN HOÀI NAM",
        "dob": "15/08/2003",
        "gender": "Nữ",
    }

    # Kiểm tra qua API /api/idcard/scan ở Pha 2 khi zoom sát (không thấy viền thẻ, không có tiêu đề CCCD)
    client = TestClient(app)
    monkeypatch.setattr(ocr_engine, "_reader", object())
    monkeypatch.setattr(
        ocr_engine,
        "read_lines",
        lambda _img, sensitive=False: [
            "15 08 2003",
            "Narn   Quốc tịch: Việt Nam",
        ],
    )

    zoom_img = np.full((400, 640, 3), 200, dtype=np.uint8)
    cv2.putText(zoom_img, "15/08/2003 Nam", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    r_scan = client.post(
        "/api/idcard/scan",
        json={"image": _encode_jpg_b64(zoom_img), "known_fields": known},
    )
    assert r_scan.status_code == 200
    data = r_scan.json()
    assert data["focus_lower"] is True
    assert data["complete"] is True
    assert data["fields"] == {
        "id_number": "079203001234",
        "full_name": "NGUYỄN HOÀI NAM",
        "dob": "15/08/2003",
        "gender": "Nam",
    }


def test_reject_can_cuoc_cong_dan_as_name_and_require_name_below_id():
    # Trường hợp OCR bị nhiễu chữ "Căn Cuoc cong dan" / "CĂN CU0C C0NG OAN" ở phía trên số CCCD,
    # và chưa đọc rõ dòng tên bên dưới -> tuyệt đối không lấy tiêu đề thẻ làm Họ và tên
    noisy_header_only = [
        "C0NG HOA XA HOI CHU NGHIA VIET NAM",
        "Căn Cuoc cong dan",
        "CĂN CU0C C0NG OAN",
        "Số / No.: 079203001234",
        "Họ và tên / Full name:",
    ]
    res_partial = parse_cccd_lines(noisy_header_only)
    assert res_partial["id_number"] == "079203001234"
    assert res_partial["full_name"] is None

    # Khi có dòng Họ và tên hợp lệ nằm BÊN DƯỚI dòng số CCCD -> lấy đúng dòng bên dưới
    with_real_name_below = noisy_header_only + [
        "PHẠM MINH Kiệt",
        "Ngày sinh: 115/08/2003",  # dấu ':' trước ngày sinh bị đọc nhầm thành '1'
        "Giới tính: Nam",
    ]
    res_full = parse_cccd_lines(with_real_name_below)
    assert res_full == {
        "id_number": "079203001234",
        "full_name": "PHẠM MINH KIỆT",
        "dob": "15/08/2003",
        "gender": "Nam",
    }

