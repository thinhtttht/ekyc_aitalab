/**
 * Bộ điều khiển bước Quét & OCR thẻ CCCD (trước bước Đăng ký khuôn mặt)
 * - Hiển thị khung chữ nhật chuẩn tỉ lệ thẻ ID-1 (85.6 × 53.98 mm)
 * - Tự động cắt vùng thẻ, gửi lên EasyOCR backend và tích lũy 4 trường thông tin
 * - Hiển thị popup xác nhận thông tin & kiểm tra trùng lặp CSDL trước khi sang bước Khuôn mặt
 */

const ID_CARD_RATIO = 85.6 / 53.98;
const ID_FIELDS = ['id_number', 'full_name', 'dob', 'gender'];

// DOM Elements cho View Quét CCCD
const idcardViewportBox = document.getElementById('idcardViewportBox');
const idcardVideo = document.getElementById('idcardVideo');
const idcardSvg = document.getElementById('idcardSvg');
const idcardMaskHole = document.getElementById('idcardMaskHole');
const idcardFrame = document.getElementById('idcardFrame');
const idcardQuad = document.getElementById('idcardQuad');
const idcardPill = document.getElementById('idcardPill');
const idcardProgressLabel = document.getElementById('idcardProgressLabel');
const idcardProgressBar = document.getElementById('idcardProgressBar');
const btnIdcardRestart = document.getElementById('btnIdcardRestart');

// DOM Elements cho Popup Xác nhận CCCD
const idCardModal = document.getElementById('idCardModal');
const idModalIcon = document.getElementById('idModalIcon');
const idModalTitle = document.getElementById('idModalTitle');
const idModalSubtitle = document.getElementById('idModalSubtitle');
const idModalSnapshot = document.getElementById('idModalSnapshot');
const idModalNumber = document.getElementById('idModalNumber');
const idModalName = document.getElementById('idModalName');
const idModalDob = document.getElementById('idModalDob');
const idModalGender = document.getElementById('idModalGender');
const idModalRegisteredNote = document.getElementById('idModalRegisteredNote');
const btnIdModalRescan = document.getElementById('btnIdModalRescan');
const btnIdModalConfirm = document.getElementById('btnIdModalConfirm');
const btnIdModalHome = document.getElementById('btnIdModalHome');

// Canvas cắt khung thẻ CCCD gửi lên server (1120 × 706 - chuẩn tỉ lệ ID-1 độ nét cao)
const idcardCropCanvas = document.createElement('canvas');
idcardCropCanvas.width = 1120;
idcardCropCanvas.height = Math.round(1120 / ID_CARD_RATIO);
const idcardCropCtx = idcardCropCanvas.getContext('2d');

// Trạng thái phiên quét CCCD
let isIdCardLoopActive = false;
let isIdCardSending = false;
let idCardLoopTimer = null;
let frameRectPx = { x: 20, y: 60, w: 680, h: 429, boxW: 720, boxH: 560 };
let accumFields = { id_number: null, full_name: null, dob: null, gender: null };
let bestCardSnapshot = '';

// -----------------------------------------------------------------------------
// 1. CĂN CHỈNH KHUNG CHỮ NHẬT SVG THEO KÍCH THƯỚC VIEWPORT (MỞ RỘNG TO RÕ)
// -----------------------------------------------------------------------------

function layoutIdCardFrame() {
  if (!idcardViewportBox || !idcardSvg || !idcardMaskHole || !idcardFrame) return;
  const boxW = idcardViewportBox.clientWidth || 680;
  const boxH = idcardViewportBox.clientHeight || 540;

  idcardSvg.setAttribute('viewBox', `0 0 ${boxW} ${boxH}`);

  const fw = Math.round(Math.min(boxW * 0.95, boxH * 0.86 * ID_CARD_RATIO));
  const fh = Math.round(fw / ID_CARD_RATIO);
  const fx = Math.round((boxW - fw) / 2);
  const fy = Math.round((boxH - fh) / 2 - boxH * 0.02);

  frameRectPx = { x: fx, y: fy, w: fw, h: fh, boxW, boxH };

  [idcardMaskHole, idcardFrame].forEach((el) => {
    el.setAttribute('x', fx);
    el.setAttribute('y', fy);
    el.setAttribute('width', fw);
    el.setAttribute('height', fh);
  });
}

window.addEventListener('resize', () => {
  if (currentTab === 'viewIdCard') layoutIdCardFrame();
});

// -----------------------------------------------------------------------------
// 2. CẮT KHUNG HÌNH VIDEO THEO KHUNG CHỮ NHẬT (KHÔNG LẬT GƯƠNG ĐỂ ĐỌC CHỮ)
// -----------------------------------------------------------------------------

function captureIdCardCrop() {
  const vw = idcardVideo.videoWidth || 1280;
  const vh = idcardVideo.videoHeight || 720;
  const { x, y, w, h, boxW, boxH } = frameRectPx;

  // Quy đổi toạ độ khung trên viewport (object-fit: cover) sang toạ độ pixel gốc của video
  const scale = Math.max(boxW / vw, boxH / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const offX = (dispW - boxW) / 2;
  const offY = (dispH - boxH) / 2;

  // Pha 1: mở rộng biên 5% để bắt trọn viền thẻ; Pha 2 (zoom gần): lấy đúng vùng trong khung
  const isZoomPhase = Boolean(accumFields.id_number && accumFields.full_name);
  const padRatio = isZoomPhase ? 0.0 : 0.05;
  const padX = w * padRatio;
  const padY = h * padRatio;
  const sx = Math.max(0, Math.round((x - padX + offX) / scale));
  const sy = Math.max(0, Math.round((y - padY + offY) / scale));
  const sw = Math.min(vw - sx, Math.round((w + padX * 2) / scale));
  const sh = Math.min(vh - sy, Math.round((h + padY * 2) / scale));

  idcardCropCtx.drawImage(
    idcardVideo,
    sx,
    sy,
    Math.max(1, sw),
    Math.max(1, sh),
    0,
    0,
    idcardCropCanvas.width,
    idcardCropCanvas.height
  );

  return idcardCropCanvas.toDataURL('image/jpeg', 0.92);
}

// -----------------------------------------------------------------------------
// 3. CẬP NHẬT GIAO DIỆN STEPPER, CHECKLIST & KHUNG VIỀN
// -----------------------------------------------------------------------------

function updateIdCardStepper(stepName) {
  const order = ['place', 'detect', 'ocr', 'confirm'];
  const activeIdx = order.indexOf(stepName);
  order.forEach((name, idx) => {
    const el = document.getElementById(`idstep-${name}`);
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (idx < activeIdx || (stepName === 'confirm' && idx === activeIdx)) {
      el.classList.add('completed');
    } else if (idx === activeIdx) {
      el.classList.add('active');
    }
  });
}

function resetIdCardUI() {
  accumFields = { id_number: null, full_name: null, dob: null, gender: null };
  bestCardSnapshot = '';

  updateIdCardStepper('place');
  if (idcardFrame) idcardFrame.setAttribute('class', 'oval-stroke cyan');
  if (idcardQuad) idcardQuad.classList.add('hidden');
  if (idcardPill) {
    idcardPill.textContent = 'Đưa trọn mặt trước CCCD vào khung chữ nhật';
    idcardPill.className = 'bottom-guidance-pill cyan';
  }
  if (idcardProgressLabel) idcardProgressLabel.textContent = 'Đã đọc 0/4 trường thông tin';
  if (idcardProgressBar) idcardProgressBar.style.width = '0%';

  setRow('idchk-card', 'idval-card', null, '--');
  setRow('idchk-quality', 'idval-quality', null, '--');
  setRow('idchk-ocr', 'idval-ocr', null, '--');
  ID_FIELDS.forEach((f) => setRow(`idchk-${f}`, `idval-${f}`, null, '--'));
}

function renderIdCardScanFeedback(data, snapshotB64) {
  const q = data.quality || {};
  const hadLockedPair = Boolean(accumFields.id_number && accumFields.full_name);

  // Tích lũy các trường đọc được: trường nào đã có sẽ GIỮ NGUYÊN (không ghi đè lại)
  const incoming = data.fields || {};
  if (data.is_id_card || incoming.id_number || incoming.full_name || hadLockedPair) {
    ID_FIELDS.forEach((k) => {
      if (!accumFields[k] && incoming[k]) {
        accumFields[k] = k === 'full_name' && typeof removeVietnameseTones === 'function'
          ? removeVietnameseTones(incoming[k]).toUpperCase()
          : incoming[k];
      }
    });
    // Lưu ảnh chụp toàn thẻ ở Pha 1 làm ảnh đại diện trên popup
    if (snapshotB64 && (!bestCardSnapshot || !hadLockedPair)) {
      bestCardSnapshot = snapshotB64;
    }
  }

  const nowLockedPair = Boolean(accumFields.id_number && accumFields.full_name);
  if (!hadLockedPair && nowLockedPair && !(accumFields.dob && accumFields.gender)) {
    // Báo hiệu âm thanh nhẹ khi đã khóa xong Tên & Số CCCD để người dùng có thể zoom gần
    if (typeof playTingSound === 'function') playTingSound();
  }

  const hasCard = Boolean(
    nowLockedPair || data.card_detected || data.is_id_card || (q.edge_density && q.edge_density >= 0.030)
  );

  setRow(
    'idchk-card',
    'idval-card',
    hasCard,
    nowLockedPair
      ? 'Đã khóa Tên & Số (Zoom tự do)'
      : data.card_detected
      ? 'Đã bắt viền thẻ'
      : hasCard
      ? 'Trong khung'
      : 'Chưa thấy thẻ'
  );
  setRow(
    'idchk-quality',
    'idval-quality',
    q.ok,
    q.ok ? `Nét (${Math.round(q.sharpness || 0)})` : (q.issue || '--')
  );
  setRow(
    'idchk-ocr',
    'idval-ocr',
    data.ocr_ready ? true : null,
    data.ocr_ready ? `Sẵn sàng (${data.latency_ms || '--'} ms)` : 'Đang nạp model...'
  );

  // Vẽ viền tứ giác phát hiện được (chỉ khi chưa zoom gần)
  if (idcardQuad) {
    if (!nowLockedPair && Array.isArray(data.quad) && data.quad.length === 4) {
      const { x, y, w, h } = frameRectPx;
      const pts = data.quad
        .map(([nx, ny]) => `${Math.round(x + nx * w)},${Math.round(y + ny * h)}`)
        .join(' ');
      idcardQuad.setAttribute('points', pts);
      idcardQuad.classList.remove('hidden');
    } else {
      idcardQuad.classList.add('hidden');
    }
  }

  // Cập nhật từng dòng thông tin bên phải
  let filledCount = 0;
  ID_FIELDS.forEach((k) => {
    const val = accumFields[k];
    if (val) {
      filledCount++;
      setRow(`idchk-${k}`, `idval-${k}`, true, val);
    } else if (data.ocr_ran) {
      setRow(`idchk-${k}`, `idval-${k}`, null, 'Đang đọc...');
    }
  });

  // Cập nhật thanh tiến độ X/4 trường
  const pct = Math.round((filledCount / ID_FIELDS.length) * 100);
  if (idcardProgressLabel) {
    idcardProgressLabel.textContent = nowLockedPair && filledCount < 4
      ? `Đã khóa Tên & Số CCCD (${filledCount}/4) • Có thể đưa gần thẻ để đọc Ngày sinh & Giới tính`
      : `Đã đọc ${filledCount}/4 trường thông tin (${pct}%)`;
  }
  if (idcardProgressBar) {
    idcardProgressBar.style.width = `${pct}%`;
  }

  // Cập nhật Stepper & màu khung
  const allDone = filledCount === ID_FIELDS.length;
  if (allDone) {
    updateIdCardStepper('confirm');
    if (idcardFrame) idcardFrame.setAttribute('class', 'oval-stroke green');
    if (idcardPill) {
      idcardPill.textContent = 'Đã trích xuất đầy đủ thông tin CCCD!';
      idcardPill.className = 'bottom-guidance-pill green';
    }
  } else if (nowLockedPair) {
    updateIdCardStepper('ocr');
    if (idcardFrame) idcardFrame.setAttribute('class', 'oval-stroke yellow');
    if (idcardPill) {
      const missing = [];
      if (!accumFields.dob) missing.push('Ngày sinh');
      if (!accumFields.gender) missing.push('Giới tính');
      idcardPill.textContent = `Đã khóa Tên & Số CCCD! Đưa gần phần ${missing.join(' & ')} vào khung để đọc rõ`;
      idcardPill.className = 'bottom-guidance-pill yellow';
    }
  } else if (filledCount > 0 || data.ocr_ran) {
    updateIdCardStepper('ocr');
    if (idcardFrame) idcardFrame.setAttribute('class', `oval-stroke ${data.color || 'yellow'}`);
    if (idcardPill && data.guidance) {
      idcardPill.textContent = data.guidance;
      idcardPill.className = `bottom-guidance-pill ${data.color || 'yellow'}`;
    }
  } else if (hasCard) {
    updateIdCardStepper('detect');
    if (idcardFrame) idcardFrame.setAttribute('class', `oval-stroke ${data.color || 'yellow'}`);
    if (idcardPill && data.guidance) {
      idcardPill.textContent = data.guidance;
      idcardPill.className = `bottom-guidance-pill ${data.color || 'yellow'}`;
    }
  } else {
    updateIdCardStepper('place');
    if (idcardFrame) idcardFrame.setAttribute('class', 'oval-stroke cyan');
    if (idcardPill && data.guidance) {
      idcardPill.textContent = data.guidance;
      idcardPill.className = 'bottom-guidance-pill cyan';
    }
  }

  return allDone;
}

// -----------------------------------------------------------------------------
// 4. VÒNG LẶP QUÉT CCCD & KIỂM TRA TRÙNG LẶP HỒ SƠ
// -----------------------------------------------------------------------------

async function idCardScanStep() {
  if (!isIdCardLoopActive || currentTab !== 'viewIdCard') return;

  // Đảm bảo luồng video camera luôn được gắn vào idcardVideo
  if (currentStream && idcardVideo && idcardVideo.srcObject !== currentStream) {
    idcardVideo.srcObject = currentStream;
    idcardVideo.play().catch(() => {});
  }

  if (isIdCardSending || !idcardVideo || idcardVideo.readyState < 2) {
    idCardLoopTimer = setTimeout(idCardScanStep, 250);
    return;
  }

  layoutIdCardFrame();
  const cropB64 = captureIdCardCrop();
  isIdCardSending = true;

  try {
    const res = await fetch(`${API_BASE}/api/idcard/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: cropB64,
        known_fields: accumFields,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateBackendStatus(true);

    if (!isIdCardLoopActive || currentTab !== 'viewIdCard') return;
    const isComplete = renderIdCardScanFeedback(data, cropB64);

    if (isComplete) {
      isIdCardLoopActive = false;
      playTingSound();
      await checkAndOpenIdCardModal({ ...accumFields }, bestCardSnapshot || cropB64);
      return;
    }
  } catch (err) {
    console.warn('Lỗi quét CCCD:', err);
  } finally {
    isIdCardSending = false;
  }

  if (isIdCardLoopActive && currentTab === 'viewIdCard') {
    idCardLoopTimer = setTimeout(idCardScanStep, 260);
  }
}

async function checkIdentityDuplicate(fields) {
  try {
    const res = await fetch(`${API_BASE}/api/idcard/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(fields),
    });
    if (!res.ok) return { registered: false, user: null };
    return await res.json();
  } catch (e) {
    return { registered: false, user: null };
  }
}

function setModalRegisteredState(existingUser) {
  if (idModalIcon) {
    idModalIcon.textContent = '⚠️';
    idModalIcon.className = 'modal-icon-warning';
  }
  if (idModalTitle) idModalTitle.textContent = 'Người dùng đã đăng ký!';
  if (idModalSubtitle) {
    idModalSubtitle.textContent = 'Thẻ CCCD này đã tồn tại trong cơ sở dữ liệu sinh trắc học.';
  }
  if (idModalRegisteredNote) {
    const rawName = existingUser?.full_name || accumFields.full_name || '--';
    const uName = typeof removeVietnameseTones === 'function' ? removeVietnameseTones(rawName) : rawName;
    const uId = existingUser?.user_id || '--';
    const uDate = existingUser?.created_at ? formatDateTime(existingUser.created_at) : '--';
    idModalRegisteredNote.innerHTML = `
      <strong>⚠️ Đã đăng ký trước đó:</strong> Hồ sơ <b>${escapeHtml(uName)}</b> (Mã: <code>${escapeHtml(uId)}</code> • Ngày ĐK: ${escapeHtml(uDate)}).
      Vui lòng sử dụng thẻ khác hoặc chuyển sang mục Xác thực.
    `;
    idModalRegisteredNote.classList.remove('hidden');
  }
  if (btnIdModalConfirm) btnIdModalConfirm.classList.add('hidden');
  if (btnIdModalHome) btnIdModalHome.classList.remove('hidden');
  if (typeof showToast === 'function') {
    showToast('Người dùng mang CCCD này đã đăng ký trong hệ thống!', 'error');
  }
}

function setModalConfirmableState() {
  if (idModalIcon) {
    idModalIcon.textContent = '✓';
    idModalIcon.className = 'modal-icon-success';
  }
  if (idModalTitle) idModalTitle.textContent = 'Xác nhận thông tin CCCD';
  if (idModalSubtitle) {
    idModalSubtitle.textContent = 'Vui lòng kiểm tra thông tin được trích xuất tự động từ thẻ.';
  }
  if (idModalRegisteredNote) {
    idModalRegisteredNote.classList.add('hidden');
    idModalRegisteredNote.innerHTML = '';
  }
  if (btnIdModalConfirm) {
    btnIdModalConfirm.disabled = false;
    btnIdModalConfirm.classList.remove('hidden');
  }
  if (btnIdModalHome) btnIdModalHome.classList.add('hidden');
}

async function checkAndOpenIdCardModal(fields, snapshotB64) {
  if (!idCardModal) return;

  const cleanName = fields.full_name
    ? (typeof removeVietnameseTones === 'function' ? removeVietnameseTones(fields.full_name) : fields.full_name)
    : '--';
  if (idModalSnapshot) idModalSnapshot.src = snapshotB64 || '';
  if (idModalNumber) idModalNumber.textContent = fields.id_number || '--';
  if (idModalName) idModalName.textContent = cleanName;
  if (idModalDob) idModalDob.textContent = fields.dob || '--';
  if (idModalGender) idModalGender.textContent = fields.gender || '--';

  const checkData = await checkIdentityDuplicate(fields);
  if (checkData && checkData.registered) {
    setModalRegisteredState(checkData.user);
  } else {
    setModalConfirmableState();
  }

  idCardModal.classList.remove('hidden');
}

// -----------------------------------------------------------------------------
// 5. SỰ KIỆN NÚT ĐIỀU KHIỂN & ĐIỀU HƯỚNG LUỒNG ĐĂNG KÝ
// -----------------------------------------------------------------------------

function restartIdCardScan() {
  if (idCardLoopTimer) {
    clearTimeout(idCardLoopTimer);
    idCardLoopTimer = null;
  }
  if (idCardModal) idCardModal.classList.add('hidden');
  resetIdCardUI();
  layoutIdCardFrame();
  isIdCardLoopActive = true;
  idCardScanStep();
}

window.startIdCardFlow = function () {
  idCardInfo = null;
  if (summaryModal) summaryModal.classList.add('hidden');
  if (idCardModal) idCardModal.classList.add('hidden');
  switchTab('viewIdCard');
  restartIdCardScan();
};

window.idcardOnTabChange = function (targetTabId) {
  if (targetTabId === 'viewIdCard') {
    if (idcardVideo && currentStream && idcardVideo.srcObject !== currentStream) {
      idcardVideo.srcObject = currentStream;
      idcardVideo.play().catch(() => {});
    }
    fetch(`${API_BASE}/api/idcard/status`).catch(() => {});
    requestAnimationFrame(() => layoutIdCardFrame());
  } else {
    isIdCardLoopActive = false;
    if (idCardLoopTimer) {
      clearTimeout(idCardLoopTimer);
      idCardLoopTimer = null;
    }
  }
};

if (btnIdcardRestart) {
  btnIdcardRestart.addEventListener('click', restartIdCardScan);
}

if (btnIdModalRescan) {
  btnIdModalRescan.addEventListener('click', restartIdCardScan);
}

if (btnIdModalHome) {
  btnIdModalHome.addEventListener('click', () => {
    if (idCardModal) idCardModal.classList.add('hidden');
    switchTab('viewLanding');
  });
}

if (btnIdModalConfirm) {
  btnIdModalConfirm.addEventListener('click', async () => {
    btnIdModalConfirm.disabled = true;
    const checkData = await checkIdentityDuplicate(accumFields);
    btnIdModalConfirm.disabled = false;

    if (checkData && checkData.registered) {
      setModalRegisteredState(checkData.user);
      return;
    }

    // Lưu thông tin CCCD đã xác nhận và chuyển sang bước Đăng ký khuôn mặt
    idCardInfo = { ...accumFields };
    if (idCardModal) idCardModal.classList.add('hidden');
    switchTab('viewEnroll');
    restartSession();
  });
}
