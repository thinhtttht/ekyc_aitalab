"""Nhận diện & OCR mặt trước thẻ Căn cước công dân (CCCD) Việt Nam bằng EasyOCR.

Pipeline 2 pha (tối ưu cho webcam thực tế):
- Pha 1 (Chưa có đủ Số CCCD & Họ tên):
  Đặt mặt trước CCCD vào khung chữ nhật. Trích xuất Số CCCD (12 chữ số) trước,
  sau đó chỉ tìm Họ và tên ở các dòng BÊN DƯỚI dòng Số CCCD (tuyệt đối không nhầm
  tiêu đề "CĂN CƯỚC CÔNG DÂN" nằm phía trên Số CCCD thành Họ tên).
- Pha 2 (Đã khóa Số CCCD & Họ tên, chỉ còn thiếu Ngày sinh / Giới tính):
  Giữ nguyên Số CCCD & Họ tên không kiểm tra lại; KHÔNG bắt buộc đặt trọn thẻ vào khung
  (người dùng có thể đưa thẻ zoom sát vào vùng bên dưới Họ tên). Tự động cắt/phóng đại
  vùng bên dưới tên và hạ ngưỡng phát hiện chữ nhỏ của EasyOCR để bắt nhạy Ngày sinh & Giới tính.
"""
from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CARD_RATIO = 85.6 / 53.98          # Tỉ lệ chuẩn ISO/IEC 7810 ID-1
WARP_W = 1200
WARP_H = int(round(WARP_W / CARD_RATIO))

MIN_SHARPNESS = 22.0               # Phương sai Laplacian tối thiểu ở Pha 1 (nới nhạy cho webcam)
MIN_SHARPNESS_ZOOM = 10.0          # Ngưỡng nét nới lỏng tối đa ở Pha 2 khi zoom gần
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 245.0
MAX_GLARE_RATIO = 0.12             # Tỉ lệ pixel cháy sáng tối đa

FIELDS = ("id_number", "full_name", "dob", "gender")


# -----------------------------------------------------------------------------
# Tiện ích chuỗi tiếng Việt & sửa lỗi ký tự số của OCR
# -----------------------------------------------------------------------------

def fold(text: str) -> str:
    """Bỏ dấu tiếng Việt + chữ thường để so khớp nhãn bền vững với lỗi OCR."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn").lower()


_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "l": "1", "|": "1", "i": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "B": "8",
})


def _fix_digit_tokens(line: str) -> str:
    """Sửa lỗi OCR nhầm chữ/số (O->0, I->1, S->5...) trong các cụm có chứa chữ số."""
    out = []
    for tok in line.split(" "):
        core = re.sub(r"[^0-9A-Za-z|!]", "", tok)
        digits = sum(c.isdigit() for c in core)
        if (len(core) >= 2 and digits / len(core) >= 0.4) or digits >= 2:
            tok = tok.translate(_DIGIT_FIX)
        out.append(tok)
    return " ".join(out)


# Nhãn & cụm tiêu đề cố định trên thẻ (dạng đã bỏ dấu) - tuyệt đối loại khỏi ứng viên họ tên
_LABEL_WORDS = (
    "cong hoa", "xa hoi", "chu nghia", "viet nam", "doc lap", "tu do", "hanh phuc", "can cuoc",
    "cong dan", "citizen", "identity", "card", "socialist", "republic", "independence", "freedom",
    "happiness", "ho va ten", "full name", "ngay sinh", "date of birth", "gioi tinh", "sex",
    "quoc tich", "nationality", "que quan", "place of", "noi thuong tru", "residence", "co gia tri",
    "date of expiry", "so dinh danh", "personal", "identification", "number", "chu dem", "surname",
    "given name", "middle name", "khai sinh", "noi cu tru", "noi dang ky",
)

# Regex bắt mọi biến thể OCR đọc nhầm tiêu đề "CĂN CƯỚC CÔNG DÂN" / "CỘNG HÒA..." / nhãn tiếng Anh
_NON_NAME_FUZZY_RE = re.compile(
    r"("
    r"c[aăâe]n\s*c[uưo0ơu][o0ơu]?[ce]|\bc[uưo0ơ]{1,2}c\s*c[o0ô]ng\b|c[o0ô]ng\s*[dđo0]a?n|"
    r"c[o0ộ]ng\s*h[o0ò]a|x[aã]\s*h[o0ộ]i|ch[uủ]\s*ngh[iĩ]a|[dđ][o0ộ]c\s*l[aăậ]p|h[aạ]nh\s*ph[uú]c|"
    r"ng[aà]y\s*sinh|gi[o0ớ]i\s*t[ií]nh|qu[o0ố]c\s*t[iị]ch|qu[eê]\s*qu[aá]n|th[uư][oơờ]ng\s*tr[uú]|"
    r"gi[aá]\s*tr[iị]|k[h]ai\s*sinh|ch[uữ]\s*[dđ][eệ]m|[dđ][iị]nh\s*danh|"
    r"\b(?:citizen|identity|card|socialist|republic|independence|freedom|happiness|"
    r"full|name|surname|given|middle|date|birth|sex|nationality|place|origin|residence|expiry|"
    r"personal|identification|number)\b"
    r")",
    re.IGNORECASE,
)

_NAME_LABEL_RE = re.compile(r"(ho\s*va\s*ten|full\s*name|chu\s*dem|surname|given\s*name|middle\s*name|ten\s*khai\s*sinh)")
_INLINE_NAME_PREFIX_RE = re.compile(
    r"^.*(?:full\s*name|given\s*name|middle\s*name|surname|khai\s*sinh|ho\s*va\s*ten)\s*[:/.\-]*\s*",
    re.IGNORECASE,
)
_DOB_LABEL_RE = re.compile(
    r"(ng[aà]y\s*,?\s*th[aá]ng|ng[aàv]y\s*s[i1l]nh|n[aă]m\s*s[i1l]nh|date\s*[o0]f\s*b[i1l]rth|\bb[i1l]rth\b|\bd[o0]b\b)"
)
_EXPIRY_LINE_RE = re.compile(r"(c[o0]\s*g[i1l]a\s*tr[i1l]|g[i1l]a\s*tr[i1l]\s*d[eê]n|date\s*[o0]f\s*exp|exp[i1l]ry)")
_SEX_LABEL_RE = re.compile(r"(g[i1l][o0d]i\s*t[i1l]nh|gi[o0]i\s*tinh|\bs[e3]x\b|\bgi[o0]i\b)")
_STOP_RE = re.compile(
    r"(qu[o0ôơ]c\s*t[i1l]ch|nat[i1l][o0]n|qu[eê]\s*qu[aá]n|place\s*[o0]f|n[o0ơ]i\s*th|n[o0ơ]i\s*cu|n[o0ơ]i\s*dang|res[i1l]d|c[o0]\s*g[i1l]a\s*tr)"
)
_ADDRESS_START_RE = re.compile(
    r"(qu[eê]\s*qu[aá]n|place\s*[o0]f\s*[o0]r[i1l]g|n[o0ơ]i\s*th[uư][o0ơ]ng|place\s*[o0]f\s*res|n[o0ơ]i\s*cu\s*tru|n[o0ơ]i\s*dang\s*ky)"
)

# Bắt ngày sinh: cho phép cả trường hợp dấu ':' trước ngày bị đọc nhầm thành số '1',
# dấu '/' bị đọc nhầm thành '1'/'7'/'|' hoặc khoảng trắng, hoặc bị nuốt mất ('15082003')
_DATE_SEP_RE = re.compile(r"(\d{1,2})\s*[/\-.,:]\s*(\d{1,2})\s*[/\-.,:\s]\s*((?:19|20)\d{2})(?!\d)")
_DATE_SPACE_RE = re.compile(r"(?<!\d)(\d{1,2})\s+(\d{1,2})\s+((?:19|20)\d{2})(?!\d)")
_DATE_FLEX_RE = re.compile(
    r"(\d{2})(?:[/\-.,:\s]|[17|lI])?(\d{2})(?:[/\-.,:\s]|[17|lI])?((?:19|20)\d{2})(?!\d)"
)
_ID_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
_CARD_KEYWORDS = ("can cuoc", "citizen", "identity card", "cong hoa", "socialist", "ho va ten",
                  "full name", "date of birth", "ngay sinh", "gioi tinh", "quoc tich", "nationality")


def _looks_like_name(text: str) -> bool:
    cleaned = _clean_name(text)
    letters = [c for c in cleaned if c.isalpha()]
    if len(letters) < 4 or any(c.isdigit() for c in text):
        return False
    words = [w for w in cleaned.split() if w]
    if not (2 <= len(words) <= 7):
        return False
    orig_letters = [c for c in text if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in orig_letters) / max(1, len(orig_letters))
    if upper_ratio < 0.65:
        return False
    f = fold(cleaned)
    if any(lbl in f for lbl in _LABEL_WORDS):
        return False
    if _NON_NAME_FUZZY_RE.search(f):
        return False
    return True


def _clean_name(text: str) -> str:
    text = fold(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def _strip_name_label(line: str) -> str:
    """Tách phần tên nằm cùng dòng với nhãn kể cả khi OCR bỏ sót dấu hai chấm ':'."""
    if ":" in line:
        after_colon = line.split(":")[-1].strip()
        if after_colon:
            return after_colon
    f = fold(line)
    m = _INLINE_NAME_PREFIX_RE.match(f)
    if m and m.end() < len(line):
        return line[m.end():].strip(" :.-/")
    return ""


def _valid_date(d: int, m: int, y: int) -> Optional[str]:
    try:
        dt = date(y, m, d)
    except ValueError:
        return None
    if not (1920 <= y <= date.today().year):
        return None
    return dt.strftime("%d/%m/%Y")


def _extract_dates_in_line(line: str) -> List[Tuple[int, str]]:
    """Tìm mọi ngày hợp lệ (năm, 'dd/mm/yyyy') trong 1 dòng, kể cả khi dấu '/' hoặc ':' bị nhiễu."""
    # Bỏ qua nếu dòng chứa số CCCD 12 số trùng khớp hoàn toàn mà không có ngày khác
    norm_line = _fix_digit_tokens(line)
    # Xóa tạm chuỗi 12 số CCCD (nếu có trên cùng dòng) để không cắt nhầm 8 số cuối của CCCD thành ngày
    norm_no_id = _ID_RE.sub(" ", norm_line)

    found: List[Tuple[int, str]] = []
    for regex in (_DATE_SEP_RE, _DATE_SPACE_RE, _DATE_FLEX_RE):
        for m in regex.finditer(norm_no_id):
            d_str, m_str, y_str = m.group(1), m.group(2), m.group(3)
            val = _valid_date(int(d_str), int(m_str), int(y_str))
            if val and all(v != val for _, v in found):
                found.append((int(y_str), val))
        if found:
            break
    return found


def _find_sex_in_text(text: str, locked_name_folded: str = "") -> Optional[str]:
    """Nhận diện Nam / Nữ từ chuỗi đã fold, chống nhầm 'Việt Nam' hoặc tên người dùng."""
    text = re.sub(r"v[i1l][eê]?t\s*nam", " ", text)
    if locked_name_folded:
        text = text.replace(locked_name_folded, " ")
    text = _STOP_RE.split(text)[0]
    if re.search(r"(?:^|[\s:.;,/\-]|sex|tinh)(nu|nv|nii|nir|nư|female)(?=$|[\s,.;:/\-]|qu|na)", text):
        return "Nữ"
    if re.search(r"(?:^|[\s:.;,/\-]|sex|tinh)(nam|narn|nan|male)(?=$|[\s,.;:/\-]|qu|na)", text):
        return "Nam"
    return None


# -----------------------------------------------------------------------------
# Trích xuất trường thông tin từ các dòng văn bản (hỗ trợ khóa trường Pha 2)
# -----------------------------------------------------------------------------

def parse_cccd_lines(
    lines: List[str],
    known_fields: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Optional[str]]:
    """Trích xuất Số CCCD, Họ tên, Ngày sinh, Giới tính theo đúng thứ tự bố cục thẻ CCCD:

    Bố cục từ trên xuống dưới trên mặt trước thẻ CCCD:
      Tiêu đề ("CĂN CƯỚC CÔNG DÂN") -> Số CCCD (12 số) -> Họ và tên -> Ngày sinh -> Giới tính.
    Do đó Họ và tên LUÔN chỉ được tìm ở các dòng BÊN DƯỚI dòng Số CCCD.
    """
    known = known_fields or {}
    locked_id = (known.get("id_number") or "").strip() or None
    raw_locked_name = (known.get("full_name") or "").strip()
    locked_name = _clean_name(raw_locked_name) if raw_locked_name else None
    focus_lower = bool(locked_id and locked_name)

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln and ln.strip()]
    fixed = [_fix_digit_tokens(ln) for ln in lines]
    folded = [fold(ln) for ln in lines]

    result: Dict[str, Optional[str]] = {
        "id_number": locked_id,
        "full_name": locked_name,
        "dob": (known.get("dob") or "").strip() or None,
        "gender": (known.get("gender") or "").strip() or None,
    }

    # 1. Tìm dòng chứa Số CCCD 12 chữ số (dùng làm mốc: Họ tên bắt buộc nằm BÊN DƯỚI dòng này)
    id_line_idx: Optional[int] = None
    for idx, ln in enumerate(fixed):
        m = _ID_RE.search(re.sub(r"(?<=\d)[\s.\-](?=\d)", "", ln))
        if m:
            id_line_idx = idx
            if not result["id_number"]:
                result["id_number"] = m.group(1)
            break

    # 2. Tìm Họ và tên (nếu chưa khóa): CHỈ xét các dòng bên dưới Số CCCD (idx > id_line_idx)
    name_line_idx: Optional[int] = None
    name_start_idx = (id_line_idx + 1) if id_line_idx is not None else 0

    if not result["full_name"]:
        for i in range(name_start_idx, len(folded)):
            if not _NAME_LABEL_RE.search(folded[i]):
                continue
            tail = _strip_name_label(lines[i])
            cands = [(i, tail)] if tail else []
            for offset in (1, 2):
                if i + offset < len(lines):
                    cands.append((i + offset, lines[i + offset]))
            for idx_cand, c in cands:
                c = c.strip()
                if c and _looks_like_name(c):
                    result["full_name"] = _clean_name(c)
                    name_line_idx = idx_cand
                    break
            if result["full_name"]:
                break

    # Nếu đã có Họ tên (từ Pha 1 hoặc vừa tìm thấy), xác định vị trí dòng Họ tên để chỉ quét bên dưới
    locked_name_folded = fold(result["full_name"]) if result["full_name"] else ""
    if name_line_idx is None and locked_name_folded:
        for i, f in enumerate(folded):
            if locked_name_folded in f or _NAME_LABEL_RE.search(f):
                name_line_idx = i

    # Phạm vi dòng bắt đầu quét Ngày sinh & Giới tính (bên dưới Họ tên / Số CCCD)
    anchor_idx = name_line_idx if name_line_idx is not None else id_line_idx
    start_below_name = (anchor_idx + 1) if (anchor_idx is not None and anchor_idx + 1 < len(lines)) else 0

    # 3. Ngày sinh: ưu tiên dòng có nhãn Ngày sinh, sau đó quét mọi dòng bên dưới Họ tên (bỏ qua dòng hạn thẻ)
    dob_idx: Optional[int] = None
    if not result["dob"]:
        for i in range(start_below_name, len(folded)):
            if _DOB_LABEL_RE.search(folded[i]):
                for j in (i, i + 1):
                    if j < len(fixed) and not _EXPIRY_LINE_RE.search(folded[j]):
                        dates = _extract_dates_in_line(fixed[j])
                        if dates:
                            result["dob"], dob_idx = dates[0][1], j
                            break
                if result["dob"]:
                    break

    if not result["dob"]:
        candidates: List[Tuple[int, str, int]] = []
        search_range = range(start_below_name, len(fixed)) if (focus_lower or anchor_idx is not None) else range(len(fixed))
        for i in search_range:
            if _EXPIRY_LINE_RE.search(folded[i]):
                continue
            for yr, val in _extract_dates_in_line(fixed[i]):
                candidates.append((yr, val, i))
        if candidates:
            _, result["dob"], dob_idx = min(candidates)

    # Fallback tìm Họ tên: bắt buộc phải nằm DƯỚI dòng Số CCCD (id_line_idx < i < dob_idx)
    if not result["full_name"] and id_line_idx is not None:
        stop = dob_idx if dob_idx is not None else len(lines)
        for i in range(id_line_idx + 1, stop):
            if _looks_like_name(lines[i]):
                result["full_name"] = _clean_name(lines[i])
                locked_name_folded = fold(result["full_name"])
                name_line_idx = i
                break

    # 4. Giới tính: tìm theo nhãn -> tìm quanh dòng Ngày sinh -> tìm mọi dòng bên dưới tên (trước Quê quán)
    if not result["gender"]:
        for i in range(start_below_name, len(folded)):
            m = _SEX_LABEL_RE.search(folded[i])
            if m:
                sex = _find_sex_in_text(folded[i][m.end():], locked_name_folded)
                if not sex and i + 1 < len(folded):
                    sex = _find_sex_in_text(folded[i + 1], locked_name_folded)
                if sex:
                    result["gender"] = sex
                    break

    if not result["gender"] and dob_idx is not None:
        for j in range(dob_idx, min(dob_idx + 3, len(folded))):
            if _ADDRESS_START_RE.search(folded[j]):
                break
            sex = _find_sex_in_text(folded[j], locked_name_folded)
            if sex:
                result["gender"] = sex
                break

    if not result["gender"]:
        for i in range(start_below_name, len(folded)):
            if name_line_idx is not None and i <= name_line_idx:
                continue
            if id_line_idx is not None and i <= id_line_idx:
                continue
            if _ADDRESS_START_RE.search(folded[i]):
                break
            if _NON_NAME_FUZZY_RE.search(folded[i]) and not _SEX_LABEL_RE.search(folded[i]):
                continue
            sex = _find_sex_in_text(folded[i], locked_name_folded)
            if sex:
                result["gender"] = sex
                break

    return result


def is_id_card_text(lines: List[str]) -> bool:
    joined = fold(" ".join(lines))
    return any(k in joined for k in _CARD_KEYWORDS)


# -----------------------------------------------------------------------------
# Thị giác máy tính: định vị thẻ, tiền xử lý & đánh giá chất lượng
# -----------------------------------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def locate_card(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Tìm tứ giác thẻ trong ảnh. Trả về 4 đỉnh (TL, TR, BR, BL) theo toạ độ ảnh gốc hoặc None."""
    h, w = img_bgr.shape[:2]
    scale = 640.0 / max(w, h)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 110)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.22 * area_img:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = _order_quad(approx)
        else:
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw * rh <= 0 or area / (rw * rh) < 0.85:
                continue
            quad = _order_quad(cv2.boxPoints(rect))
        wtop = np.linalg.norm(quad[1] - quad[0])
        hleft = np.linalg.norm(quad[3] - quad[0])
        if hleft < 1:
            continue
        ratio = max(wtop, hleft) / min(wtop, hleft)
        if 1.30 <= ratio <= 1.90:
            return quad / scale
    return None


def warp_card(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    wtop = np.linalg.norm(quad[1] - quad[0])
    hleft = np.linalg.norm(quad[3] - quad[0])
    if hleft > wtop:  # Thẻ đang dựng đứng -> xoay đỉnh để luôn ra ảnh nằm ngang
        quad = np.roll(quad, -1, axis=0)
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (WARP_W, WARP_H), flags=cv2.INTER_CUBIC)


def assess_quality(img_bgr: np.ndarray, relaxed: bool = False) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    scale = 640.0 / max(w, h)
    gray = cv2.cvtColor(cv2.resize(img_bgr, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((gray >= 250).mean())
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())

    min_sharp = MIN_SHARPNESS_ZOOM if relaxed else MIN_SHARPNESS
    max_glare = 0.18 if relaxed else MAX_GLARE_RATIO

    issue = None
    if brightness < MIN_BRIGHTNESS:
        issue = "Ảnh quá tối, vui lòng tăng ánh sáng"
    elif brightness > MAX_BRIGHTNESS:
        issue = "Ảnh quá sáng, vui lòng giảm ánh sáng"
    elif glare > max_glare:
        issue = "Thẻ bị lóa sáng, hãy nghiêng nhẹ thẻ để tránh phản quang"
    elif sharpness < min_sharp:
        issue = "Ảnh hơi mờ, giữ yên hoặc đưa gần vùng Ngày sinh & Giới tính"
    return {
        "ok": issue is None,
        "issue": issue,
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "glare_ratio": round(glare, 4),
        "edge_density": round(edge_density, 4),
    }


def _preprocess_for_ocr(img_bgr: np.ndarray, sharpen: bool = False) -> np.ndarray:
    """Tăng tương phản nhẹ CLAHE (tránh làm nổi hoa văn chìm của thẻ) và làm nét chữ nhỏ."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8 if sharpen else 1.4, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    if sharpen:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        enhanced = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def group_lines(results: List[Tuple[Any, str, float]], min_conf: float = 0.04) -> List[str]:
    """Gom các box chữ của EasyOCR thành dòng theo thứ tự đọc (trên->dưới, trái->phải)."""
    items = []
    for box, text, conf in results:
        if conf < min_conf or not text.strip():
            continue
        pts = np.array(box, dtype=np.float32)
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        items.append({"yc": (y0 + y1) / 2, "h": max(1.0, y1 - y0), "x": pts[:, 0].min(), "text": text.strip()})
    if not items:
        return []
    items.sort(key=lambda it: it["yc"])
    med_h = float(np.median([it["h"] for it in items]))
    lines: List[List[dict]] = []
    for it in items:
        if lines and abs(it["yc"] - np.mean([x["yc"] for x in lines[-1]])) < 0.55 * med_h:
            lines[-1].append(it)
        else:
            lines.append([it])
    return [" ".join(x["text"] for x in sorted(ln, key=lambda x: x["x"])) for ln in lines]


# -----------------------------------------------------------------------------
# Động cơ OCR (singleton, nạp lười trong luồng nền)
# -----------------------------------------------------------------------------

class IdCardOCREngine:
    def __init__(self) -> None:
        self._reader = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loading = False
        self._error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def status(self) -> Dict[str, Any]:
        return {"ready": self.ready, "loading": self._loading, "error": self._error}

    def _load(self) -> None:
        with self._load_lock:
            if self._reader is not None:
                return
            self._loading = True
            self._error = None
            try:
                import torch  # noqa: WPS433
                torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
                import easyocr  # noqa: WPS433
                self._reader = easyocr.Reader(["vi"], gpu=False, verbose=False)
            except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
                self._error = f"Không thể nạp EasyOCR: {exc}"
            finally:
                self._loading = False

    def warmup_async(self) -> None:
        if self._reader is None and not self._loading:
            threading.Thread(target=self._load, name="easyocr-warmup", daemon=True).start()

    def read_lines(self, img_bgr: np.ndarray, sensitive: bool = False) -> List[str]:
        rgb = _preprocess_for_ocr(img_bgr, sharpen=sensitive)
        with self._infer_lock:
            results = self._reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                decoder="greedy",
                batch_size=8,
                canvas_size=1280,
                mag_ratio=1.55 if sensitive else 1.25,
                text_threshold=0.35 if sensitive else 0.45,
                low_text=0.22 if sensitive else 0.30,
                width_ths=0.5,
            )
        return group_lines(results, min_conf=0.03 if sensitive else 0.06)

    def _read_lines_compat(self, img_bgr: np.ndarray, sensitive: bool = False) -> List[str]:
        """Gọi read_lines tương thích ngược với monkeypatch trong unit test (1 tham số)."""
        try:
            return self.read_lines(img_bgr, sensitive=sensitive)
        except TypeError:
            return self.read_lines(img_bgr)

    def scan(
        self,
        img_bgr: np.ndarray,
        known_fields: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        known = known_fields or {}
        locked_id = (known.get("id_number") or "").strip() or None
        locked_name = (known.get("full_name") or "").strip() or None
        # Nếu truyền vào tên giả (ví dụ từ frame cũ bị nhầm tiêu đề), tự động lọc sạch
        if locked_name and not _looks_like_name(locked_name):
            locked_name = None
        elif locked_name:
            locked_name = _clean_name(locked_name)

        # Pha 2: đã có Tên & Số CCCD hợp lệ -> khóa lại, không bắt buộc đặt trọn thẻ vào khung chữ nhật
        focus_lower = bool(locked_id and locked_name)

        resp: Dict[str, Any] = {
            "ocr_ready": self.ready,
            "focus_lower": focus_lower,
            "card_detected": False,
            "quad": None,
            "quality": None,
            "ocr_ran": False,
            "is_id_card": focus_lower,
            "fields": {
                "id_number": locked_id,
                "full_name": locked_name,
                "dob": (known.get("dob") or "").strip() or None,
                "gender": (known.get("gender") or "").strip() or None,
            },
            "complete": False,
            "guidance": "",
            "color": "cyan",
        }

        quad = locate_card(img_bgr)
        # Chỉ dùng warp_card khi tứ giác chiếm diện tích đủ lớn (>= 45% khung), tránh bắt nhầm khung ảnh chân dung nhỏ bên trong thẻ
        if quad is not None:
            quad_area = float(cv2.contourArea(quad.astype(np.float32)))
            if quad_area >= 0.45 * (w * h):
                resp["card_detected"] = True
                resp["quad"] = [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in quad]
                card = warp_card(img_bgr, quad)
            else:
                quad = None
                card = img_bgr
        else:
            card = img_bgr

        quality = assess_quality(card, relaxed=focus_lower)
        resp["quality"] = quality

        if not focus_lower and quad is None and quality["edge_density"] < 0.022:
            resp.update(guidance="Chưa phát hiện thẻ. Đặt mặt trước CCCD vào khung chữ nhật", color="cyan")
        elif not quality["ok"]:
            resp.update(guidance=quality["issue"], color="yellow")
        elif not self.ready:
            self.warmup_async()
            msg = self._error or "Đang tải mô hình OCR, vui lòng giữ thẻ trong khung..."
            resp.update(guidance=msg, color="red" if self._error else "yellow")
        else:
            if focus_lower:
                # Pha 2: chỉ quét vùng bên dưới Họ tên để lấy Ngày sinh & Giới tính nhạy nhất
                target_w = 1200
                base_img = cv2.resize(card, (target_w, int(target_w * card.shape[0] / card.shape[1])), interpolation=cv2.INTER_CUBIC)
                lines = self._read_lines_compat(base_img, sensitive=True)
                fields = parse_cccd_lines(lines, known_fields=resp["fields"])

                # Nếu vẫn thiếu Ngày sinh hoặc Giới tính, cắt riêng dải giữa-dưới (y: 32%..88%) phóng đại 1.6x để đọc chữ nhỏ
                if not (fields["dob"] and fields["gender"]):
                    bh, bw = base_img.shape[:2]
                    roi = base_img[int(bh * 0.32):int(bh * 0.88), int(bw * 0.18):int(bw * 0.98)]
                    roi_up = cv2.resize(roi, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
                    sub_lines = self._read_lines_compat(roi_up, sensitive=True)
                    fields = parse_cccd_lines(sub_lines, known_fields=fields)

                resp["ocr_ran"] = True
                resp["is_id_card"] = True
            else:
                if card is img_bgr:
                    card = cv2.resize(img_bgr, (WARP_W, int(WARP_W * h / w)), interpolation=cv2.INTER_CUBIC)
                lines = self._read_lines_compat(card, sensitive=False)
                resp["ocr_ran"] = True
                fields = parse_cccd_lines(lines, known_fields=resp["fields"])
                resp["is_id_card"] = is_id_card_text(lines) or bool(fields["id_number"])

                # Nếu ngay ở Pha 1 đã đọc được Tên & Số CCCD mà còn thiếu Ngày sinh/Giới tính -> cắt luôn vùng dưới tên để thử lấy ngay
                if fields["id_number"] and fields["full_name"] and not (fields["dob"] and fields["gender"]):
                    ch, cw = card.shape[:2]
                    roi = card[int(ch * 0.35):int(ch * 0.88), int(cw * 0.20):int(cw * 0.98)]
                    roi_up = cv2.resize(roi, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    sub_lines = self._read_lines_compat(roi_up, sensitive=True)
                    fields = parse_cccd_lines(sub_lines, known_fields=fields)

            resp["fields"] = fields
            resp["focus_lower"] = bool(fields["id_number"] and fields["full_name"])
            resp["complete"] = resp["is_id_card"] and all(fields.values())

            if resp["complete"]:
                resp.update(guidance="Đã trích xuất đầy đủ thông tin CCCD!", color="green")
            elif not resp["is_id_card"]:
                resp.update(guidance="Không nhận ra CCCD. Hãy đưa MẶT TRƯỚC thẻ vào khung chữ nhật", color="yellow")
            elif resp["focus_lower"]:
                missing = [lbl for k, lbl in (("dob", "Ngày sinh"), ("gender", "Giới tính")) if not fields[k]]
                resp.update(
                    guidance=f"Đã khóa Số CCCD & Họ tên! Đưa gần phần {', '.join(missing)} vào khung để đọc rõ hơn",
                    color="yellow",
                )
            else:
                missing = [lbl for k, lbl in zip(FIELDS, ("Số CCCD", "Họ tên", "Ngày sinh", "Giới tính")) if not fields[k]]
                resp.update(guidance=f"Đang đọc... chưa rõ: {', '.join(missing)}. Giữ thẻ trong khung", color="yellow")

        resp["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return resp


engine = IdCardOCREngine()
