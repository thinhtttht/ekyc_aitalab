/**
 * Smart-eKYC Frontend Controller (Vanilla JavaScript)
 * - Tự động đo FPS & kiểm tra tín hiệu camera
 * - Điều khiển SVG Oval mượt mà (chuyển đổi kích thước, màu viền, vòng tiến độ)
 * - Hướng dẫn quay đầu và tự động chụp chân dung HD khi hoàn tất Zoom
 */

const API_BASE = window.location.origin; // Cùng origin với FastAPI server

// DOM Elements
const video = document.getElementById('webcamVideo');
const cameraSelect = document.getElementById('cameraSelect');
const backendStatus = document.getElementById('backendStatus');
const viewportBox = document.getElementById('viewportBox');
const meshCanvas = document.getElementById('meshCanvas');
const meshCtx = meshCanvas ? meshCanvas.getContext('2d') : null;

// SVG Elements
const svgMaskHole = document.getElementById('svgMaskHole');
const svgOvalBorder = document.getElementById('svgOvalBorder');
const svgProgressBorder = document.getElementById('svgProgressBorder');
const turnArrowLeft = document.getElementById('turnArrowLeft');
const turnArrowRight = document.getElementById('turnArrowRight');

// Badges & Pills
const badgeFpsRes = document.getElementById('badgeFpsRes');
const badgeHeadPose = document.getElementById('badgeHeadPose');
const bottomPill = document.getElementById('bottomPill');
const holdLabel = document.getElementById('holdLabel');
const holdProgressBar = document.getElementById('holdProgressBar');

// Error & Modal
const cameraErrorOverlay = document.getElementById('cameraErrorOverlay');
const cameraErrorTitle = document.getElementById('cameraErrorTitle');
const cameraErrorDesc = document.getElementById('cameraErrorDesc');
const btnRetryCamera = document.getElementById('btnRetryCamera');
const btnRestartSession = document.getElementById('btnRestartSession');
const summaryModal = document.getElementById('summaryModal');
const snapshotImg = document.getElementById('snapshotImg');
const btnModalRetake = document.getElementById('btnModalRetake');

// Tab Navigation
const tabBtnEnroll = document.getElementById('tabBtnEnroll');
const tabBtnVerify = document.getElementById('tabBtnVerify');
const tabBtnUsers = document.getElementById('tabBtnUsers');
const viewEnroll = document.getElementById('viewEnroll');
const viewVerify = document.getElementById('viewVerify');
const viewUsers = document.getElementById('viewUsers');
const userCountBadge = document.getElementById('userCountBadge');

// Enrollment Save Form in Summary Modal
const enrollFormBox = document.getElementById('enrollFormBox');
const enrollSavedBox = document.getElementById('enrollSavedBox');
const inputUserName = document.getElementById('inputUserName');
const inputUserId = document.getElementById('inputUserId');
const btnSaveUser = document.getElementById('btnSaveUser');
const btnGoToVerify = document.getElementById('btnGoToVerify');
const btnGoToUsers = document.getElementById('btnGoToUsers');

// Verify Tab Elements
const verifyVideo = document.getElementById('verifyVideo');
const btnMode1N = document.getElementById('btnMode1N');
const btnMode11 = document.getElementById('btnMode11');
const verifyUserSelectBox = document.getElementById('verifyUserSelectBox');
const selectVerifyUser = document.getElementById('selectVerifyUser');
const verifyThreshold = document.getElementById('verifyThreshold');
const valVerifyThreshold = document.getElementById('valVerifyThreshold');
const btnTriggerVerify = document.getElementById('btnTriggerVerify');
const chkAutoVerify = document.getElementById('chkAutoVerify');
const verifyOverlayBanner = document.getElementById('verifyOverlayBanner');
const verifyOverlayText = document.getElementById('verifyOverlayText');

// Verify Result Elements
const verifyVerdictBox = document.getElementById('verifyVerdictBox');
const verdictIcon = document.getElementById('verdictIcon');
const verdictTitle = document.getElementById('verdictTitle');
const verdictSubtitle = document.getElementById('verdictSubtitle');
const matchedUserCard = document.getElementById('matchedUserCard');
const matchedAvatar = document.getElementById('matchedAvatar');
const matchedName = document.getElementById('matchedName');
const matchedId = document.getElementById('matchedId');
const matchedTime = document.getElementById('matchedTime');
const metSim = document.getElementById('metSim');
const metConf = document.getElementById('metConf');
const metDist = document.getElementById('metDist');
const metLatency = document.getElementById('metLatency');
const metPad = document.getElementById('metPad');
const candidatesLeaderboardBox = document.getElementById('candidatesLeaderboardBox');
const candidatesList = document.getElementById('candidatesList');

// Users Dashboard Elements
const statTotalUsers = document.getElementById('statTotalUsers');
const inputSearchUsers = document.getElementById('inputSearchUsers');
const btnRefreshUsers = document.getElementById('btnRefreshUsers');
const btnNewEnroll = document.getElementById('btnNewEnroll');
const btnEmptyEnroll = document.getElementById('btnEmptyEnroll');
const usersTableBody = document.getElementById('usersTableBody');
const usersEmptyState = document.getElementById('usersEmptyState');

// Vector Modal
const vectorModal = document.getElementById('vectorModal');
const vectorModalTitle = document.getElementById('vectorModalTitle');
const vectorModalSubtitle = document.getElementById('vectorModalSubtitle');
const vectorJsonContent = document.getElementById('vectorJsonContent');
const btnCopyVector = document.getElementById('btnCopyVector');
const btnCloseVectorModal = document.getElementById('btnCloseVectorModal');

// State Variables
let currentStream = null;
let currentSessionId = null;
let isLoopRunning = false;
let isSendingFrame = false;
let isFlashingActive = false;
let activeDeviceId = localStorage.getItem('ekyc_preferred_camera') || '';
let currentTab = 'viewEnroll';
let lastCapturedDataUrl = null;
let lastSavedUserId = null;
let currentVerifyMode = '1_to_n'; // '1_to_n' hoặc '1_to_1'
let autoVerifyIntervalId = null;
let isVerifyingFace = false;
let cachedUsersList = [];
let currentVectorJsonRaw = '';

// FPS & Signal Meter
let measuredFps = 0;
let frameCountInWindow = 0;
let lastFpsCalcTime = performance.now();
let lastFrameTimestamp = performance.now();
let isFrameAdvancing = true;

// Hidden Canvas cho cắt khung 4:5
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 480;
offscreenCanvas.height = 600;
const offCtx = offscreenCanvas.getContext('2d', { willReadFrequently: false });

// -----------------------------------------------------------------------------
// 1. QUẢN LÝ CAMERA (WEBRTC & ENUMERATE DEVICES)
// -----------------------------------------------------------------------------

async function initCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    showCameraError('Trình duyệt không hỗ trợ', 'Trình duyệt này không hỗ trợ WebRTC Camera API.');
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');

    cameraSelect.innerHTML = '';
    if (videoDevices.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Không tìm thấy camera';
      cameraSelect.appendChild(opt);
      return;
    }

    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Camera ${idx + 1}`;
      if (activeDeviceId && dev.deviceId === activeDeviceId) {
        opt.selected = true;
      }
      cameraSelect.appendChild(opt);
    });

    if (!activeDeviceId && videoDevices.length > 0) {
      activeDeviceId = videoDevices[0].deviceId;
    }
  } catch (err) {
    console.warn('Lỗi liệt kê camera:', err);
  }
}

cameraSelect.addEventListener('change', async (e) => {
  activeDeviceId = e.target.value;
  localStorage.setItem('ekyc_preferred_camera', activeDeviceId);
  await startCamera(activeDeviceId);
  await restartSession();
});

async function startCamera(deviceId = '') {
  hideCameraError();
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
    currentStream = null;
  }

  const constraints = {
    audio: false,
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1280, min: 640 },
      height: { ideal: 720, min: 480 },
      frameRate: { ideal: 30, min: 20 },
      facingMode: 'user',
    },
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play().then(resolve).catch(resolve);
      };
    });

    // Cập nhật lại nhãn camera nếu trước đó chưa cấp quyền
    await initCameraDevices();
    startFpsWatcher();
  } catch (err) {
    console.error('Không thể mở camera:', err);
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showCameraError('Yêu cầu quyền truy cập Camera', 'Vui lòng cho phép quyền Camera trên thanh địa chỉ của trình duyệt.');
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      showCameraError('Không tìm thấy Camera', 'Hãy kiểm tra lại cáp nối hoặc thiết bị webcam.');
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      showCameraError('Camera đang bị chiếm giữ', 'Camera có thể đang được mở bởi ứng dụng khác (Zoom, Teams, Zalo,...).');
    } else {
      showCameraError('Lỗi Camera', err.message || 'Không thể kết nối với camera.');
    }
  }
}

function showCameraError(title, desc) {
  cameraErrorTitle.textContent = title;
  cameraErrorDesc.textContent = desc;
  cameraErrorOverlay.classList.remove('hidden');
}

function hideCameraError() {
  cameraErrorOverlay.classList.add('hidden');
}

btnRetryCamera.addEventListener('click', () => {
  startCamera(activeDeviceId);
});

// -----------------------------------------------------------------------------
// 2. BỘ ĐO FPS & THEO DÕI TÍN HIỆU ĐÓNG BĂNG
// -----------------------------------------------------------------------------

function startFpsWatcher() {
  frameCountInWindow = 0;
  lastFpsCalcTime = performance.now();
  lastFrameTimestamp = performance.now();
  isFrameAdvancing = true;

  function onFrameAvailable() {
    frameCountInWindow++;
    const now = performance.now();
    lastFrameTimestamp = now;
    isFrameAdvancing = true;

    // Tính FPS trung bình mỗi 1.0 giây
    const elapsed = now - lastFpsCalcTime;
    if (elapsed >= 1000) {
      measuredFps = Math.round((frameCountInWindow * 1000) / elapsed);
      frameCountInWindow = 0;
      lastFpsCalcTime = now;
    }

    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback(onFrameAvailable);
    }
  }

  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback(onFrameAvailable);
  } else {
    // Fallback nếu trình duyệt cũ không có requestVideoFrameCallback
    setInterval(() => {
      if (video.readyState >= 2) {
        frameCountInWindow++;
      }
    }, 33);
  }

  // Bộ kiểm tra xem luồng video có đang tiếp tục chạy hay bị đóng băng
  setInterval(() => {
    const timeSinceLastFrame = performance.now() - lastFrameTimestamp;
    isFrameAdvancing = timeSinceLastFrame < 1200;
  }, 500);
}

// -----------------------------------------------------------------------------
// 3. QUẢN LÝ PHIÊN (SESSION) & VÒNG LẶP GỬI FRAME
// -----------------------------------------------------------------------------

async function startSession() {
  try {
    const res = await fetch(`${API_BASE}/api/enroll/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentSessionId = data.session_id;
    updateBackendStatus(true);
    updateOvalSvg(data.oval, 'cyan', 0);
    return data;
  } catch (err) {
    console.error('Không thể khởi tạo phiên backend:', err);
    updateBackendStatus(false);
    return null;
  }
}

async function restartSession() {
  summaryModal.classList.add('hidden');
  resetChecklistUI();
  resetStepperUI();
  updateOvalSvg({ cx: 0.5, cy: 0.47, rx: 0.33, ry: 0.36 }, 'cyan', 0);
  isFlashingActive = false;
  const fo = document.getElementById('flashingOverlay');
  if (fo) fo.classList.remove('active', 'pulse');

  if (currentSessionId) {
    try {
      await fetch(`${API_BASE}/api/enroll/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId }),
      });
    } catch (e) {}
  }
  if (meshCtx) {
    meshCtx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  }
  await startSession();
  isLoopRunning = true;
}

btnRestartSession.addEventListener('click', restartSession);
if (btnModalRetake) {
  btnModalRetake.addEventListener('click', () => {
    summaryModal.classList.add('hidden');
    restartSession();
  });
}

const btnRejectionRetry = document.getElementById('btnRejectionRetry');
if (btnRejectionRetry) {
  btnRejectionRetry.addEventListener('click', () => {
    const rejModal = document.getElementById('rejectionModal');
    if (rejModal) rejModal.classList.add('hidden');
    restartSession();
  });
}

// Vòng lặp gửi frame tự điều tốc
async function frameLoop() {
  if (!isLoopRunning) {
    requestAnimationFrame(frameLoop);
    return;
  }

  if (isSendingFrame || !currentSessionId || video.readyState < 2) {
    requestAnimationFrame(frameLoop);
    return;
  }

  // 1. Trích xuất frame ở tỉ lệ 4:5 từ webcam
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  // Tính vùng crop ở giữa theo tỉ lệ 4:5
  const targetRatio = 4 / 5;
  let cropW, cropH;
  if (vw / vh > targetRatio) {
    // Video rộng hơn 4:5 -> cắt 2 bên
    cropH = vh;
    cropW = vh * targetRatio;
  } else {
    // Video cao hơn 4:5 -> cắt trên dưới
    cropW = vw;
    cropH = vw / targetRatio;
  }
  const sx = (vw - cropW) / 2;
  const sy = (vh - cropH) / 2;

  // Vẽ vào canvas 480x600 (lật gương như gương soi)
  offCtx.save();
  offCtx.translate(480, 0);
  offCtx.scale(-1, 1);
  offCtx.drawImage(video, sx, sy, cropW, cropH, 0, 0, 480, 600);
  offCtx.restore();

  const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.82);

  isSendingFrame = true;
  try {
    const payload = {
      session_id: currentSessionId,
      image: base64Data,
      client_stats: {
        fps: measuredFps,
        width: vw,
        height: vh,
        frame_advancing: isFrameAdvancing,
      },
    };

    const res = await fetch(`${API_BASE}/api/enroll/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateBackendStatus(true);
    renderFeedback(data);

    // Nếu đạt giai đoạn FLASHING -> kích hoạt chuỗi nháy màu quang học an toàn
    if (data.stage === 'flashing' && !isFlashingActive) {
      triggerOpticalFlashing();
    }

    // Nếu đạt giai đoạn CAPTURE -> tự động chụp snapshot HD và dừng vòng lặp
    if (data.stage === 'capture') {
      isLoopRunning = false;
      captureHdAndVerify(data.summary || {});
    }
  } catch (err) {
    console.warn('Lỗi gửi frame:', err);
    updateBackendStatus(false);
  } finally {
    isSendingFrame = false;
  }

  requestAnimationFrame(frameLoop);
}

// -----------------------------------------------------------------------------
// 3.5. XÁC THỰC QUANG HỌC CHỦ ĐỘNG (ACTIVE OPTICAL COLOR FLASHING)
// -----------------------------------------------------------------------------

function applyFlashingColor(hexColor, stepName, stepIndex, totalSteps) {
  const overlay = document.getElementById('flashingOverlay');
  const svgFlashRect = document.getElementById('svgFlashRect');
  const colorTextEl = document.getElementById('flashingColorText');
  const ovalBorder = document.getElementById('svgOvalBorder');

  // 1. Chiếu sáng vùng ngoài khung Oval trong viewport camera (trừ bên trong oval)
  if (svgFlashRect) {
    svgFlashRect.style.transition = 'none';
    svgFlashRect.setAttribute('fill', hexColor);
    svgFlashRect.setAttribute('opacity', '1.0');
  }

  // 2. Viền Oval phát sáng rực rỡ theo màu hiện tại
  if (ovalBorder) {
    ovalBorder.style.stroke = hexColor;
    ovalBorder.style.filter = `drop-shadow(0 0 36px ${hexColor}) drop-shadow(0 0 12px ${hexColor})`;
  }

  // 3. Chiếu sáng toàn màn hình xung quanh (khoét rỗng bên trong khung oval)
  if (overlay && ovalBorder) {
    overlay.style.transition = 'none';
    const rect = ovalBorder.getBoundingClientRect();
    const cx = Math.round(rect.left + rect.width / 2);
    const cy = Math.round(rect.top + rect.height / 2);
    const rx = Math.round(rect.width / 2);
    const ry = Math.round(rect.height / 2);

    // CSS Mask khoét rỗng bên trong khung oval (transparent), toàn bộ bên ngoài màn hình phủ màu
    const maskVal = `radial-gradient(ellipse ${rx}px ${ry}px at ${cx}px ${cy}px, transparent 96%, black 100%)`;
    overlay.style.webkitMaskImage = maskVal;
    overlay.style.maskImage = maskVal;
    overlay.style.backgroundColor = hexColor;
    overlay.classList.add('active');
  }

  if (colorTextEl) {
    colorTextEl.textContent = `Đang quét quang phổ [Màu ${stepIndex + 1}/${totalSteps}: ${stepName}]`;
  }
}

function clearFlashingColor() {
  const overlay = document.getElementById('flashingOverlay');
  const svgFlashRect = document.getElementById('svgFlashRect');
  const ovalBorder = document.getElementById('svgOvalBorder');

  if (svgFlashRect) {
    svgFlashRect.style.transition = '';
    svgFlashRect.setAttribute('fill', 'transparent');
    svgFlashRect.setAttribute('opacity', '0');
  }
  if (ovalBorder) {
    ovalBorder.style.stroke = '';
    ovalBorder.style.filter = '';
  }
  if (overlay) {
    overlay.style.transition = '';
    overlay.classList.remove('active');
    overlay.style.backgroundColor = 'transparent';
    overlay.style.webkitMaskImage = '';
    overlay.style.maskImage = '';
  }
}

async function triggerOpticalFlashing() {
  if (isFlashingActive) return;
  isFlashingActive = true;
  // Tạm dừng vòng lặp gửi frame thông thường để tránh xung đột khung hình và đè trạng thái
  isLoopRunning = false;

  try {
    // 1. Gửi yêu cầu lấy chuỗi thách thức từ máy chủ
    const chRes = await fetch(`${API_BASE}/api/enroll/color_challenge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: currentSessionId }),
    });

    if (!chRes.ok) throw new Error(`Lỗi khởi tạo challenge: ${chRes.status}`);
    const chData = await chRes.json();
    const sequence = chData.sequence || [];
    const token = chData.challenge_token;

    bottomPill.className = 'bottom-guidance-pill green';
    bottomPill.textContent = 'Giữ yên khuôn mặt! Đang quét phản xạ ánh sáng...';

    const collectedFrames = [];

    // 2. Chiếu từng màu theo chuỗi thời gian thực (Trừ bên trong khung oval)
    for (let i = 0; i < sequence.length; i++) {
      const step = sequence[i];

      applyFlashingColor(step.hex, step.name, i, sequence.length);

      holdLabel.textContent = `Quét quang phổ ánh sáng ${i + 1}/${sequence.length}: ${step.name}`;
      const pct = Math.round(((i + 1) / sequence.length) * 100);
      holdProgressBar.style.width = `${pct}%`;

      // Chờ 260ms để ánh sáng màn hình phủ đều lên da mặt và webcam buffer cập nhật khung hình mới
      await new Promise((r) => setTimeout(r, 260));

      // Trích xuất khung hình từ webcam
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      const targetRatio = 4 / 5;
      let cropW, cropH;
      if (vw / vh > targetRatio) {
        cropH = vh;
        cropW = vh * targetRatio;
      } else {
        cropW = vw;
        cropH = vw / targetRatio;
      }
      const sx = (vw - cropW) / 2;
      const sy = (vh - cropH) / 2;

      offCtx.save();
      offCtx.translate(480, 0);
      offCtx.scale(-1, 1);
      offCtx.drawImage(video, sx, sy, cropW, cropH, 0, 0, 480, 600);
      offCtx.restore();

      const frameB64 = offscreenCanvas.toDataURL('image/jpeg', 0.85);
      collectedFrames.push({
        color_index: step.index,
        image: frameB64,
        timestamp_ms: Date.now(),
      });

      // Chờ hết thời lượng của màu này (mặc định 400ms)
      const remainingMs = Math.max(50, (step.duration_ms || 400) - 260);
      await new Promise((r) => setTimeout(r, remainingMs));
    }

    // 3. Tắt lớp phủ sau khi chiếu xong
    clearFlashingColor();

    bottomPill.className = 'bottom-guidance-pill cyan';
    bottomPill.textContent = 'Đang phân tích phản xạ quang phổ mô da...';

    // 4. Gửi các khung hình lên máy chủ để xác thực quang học
    const verifyRes = await fetch(`${API_BASE}/api/enroll/color_verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        challenge_token: token,
        frames: collectedFrames,
      }),
    });

    if (!verifyRes.ok) {
      const errData = await verifyRes.json().catch(() => ({}));
      throw new Error(errData.detail || 'Xác thực quang học thất bại');
    }

    const verifyData = await verifyRes.json();
    if (verifyData.passed) {
      playTingSound();
      bottomPill.className = 'bottom-guidance-pill green';
      bottomPill.textContent = 'Xác thực sinh trắc học hoàn tất! Đang chụp chân dung HD...';
      updateStepper('capture');
      setRow('chk-optical-liveness', 'val-optical-liveness', true, `Đạt (r = ${verifyData.correlation_score})`);
      isLoopRunning = false;
      captureHdAndVerify(verifyData.summary || {});
    } else {
      bottomPill.className = 'bottom-guidance-pill red';
      bottomPill.textContent = verifyData.message || 'Chưa đủ độ phản xạ quang học trên da. Vui lòng tăng sáng màn hình và thử lại.';
      setRow('chk-optical-liveness', 'val-optical-liveness', false, 'Không đạt');
      setTimeout(() => {
        isFlashingActive = false;
        isLoopRunning = true;
        requestAnimationFrame(frameLoop);
      }, 3000);
    }
  } catch (err) {
    console.error('Lỗi quy trình Color Flashing:', err);
    clearFlashingColor();
    bottomPill.className = 'bottom-guidance-pill red';
    bottomPill.textContent = err.message || 'Lỗi quét quang học. Vui lòng thử lại.';
    setTimeout(() => {
      isFlashingActive = false;
      isLoopRunning = true;
      requestAnimationFrame(frameLoop);
    }, 3000);
  }
}

// -----------------------------------------------------------------------------
// 4. CẬP NHẬT GIAO DIỆN & LƯỚI SINH TRẮC HỌC (BIOMETRIC MESH - APPLE FACE ID / STRIPE IDENTITY AESTHETIC)
// -----------------------------------------------------------------------------

function drawBiometricMesh(keypoints, partsStatus = {}, occludedPartName = null, hasHandOcclusion = false, overallColor = 'cyan', antiSpoof = null, isUpsideDown = false) {
  if (!meshCtx) return;
  meshCtx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);

  if (!keypoints || Object.keys(keypoints).length === 0) return;

  const w = meshCanvas.width;
  const h = meshCanvas.height;
  const toPx = (pt) => [pt[0] * w, pt[1] * h];

  const isGreen = overallColor === 'green';
  const defaultStroke = isGreen ? 'rgba(34, 197, 94, 0.52)' : 'rgba(56, 189, 248, 0.52)';
  const defaultFill = isGreen ? '#a7f3d0' : '#bae6fd';
  const defaultShadow = isGreen ? 'rgba(34, 197, 94, 0.4)' : 'rgba(56, 189, 248, 0.4)';

  const occludedStroke = 'rgba(244, 63, 94, 0.85)';
  const occludedFill = '#fecdd3';
  const occludedShadow = 'rgba(244, 63, 94, 0.6)';

  // 1. Đường vi liên kết sinh trắc học tinh gọn (Apple TrueDepth / LiDAR Hairline Grid)
  meshCtx.save();
  meshCtx.lineWidth = 0.55;
  meshCtx.strokeStyle = isGreen ? 'rgba(34, 197, 94, 0.16)' : 'rgba(56, 189, 248, 0.16)';
  meshCtx.setLineDash([2, 3]);

  // Nối lông mày tới sống mũi
  if (keypoints.left_eyebrow && keypoints.nose && keypoints.right_eyebrow) {
    const noseRoot = toPx(keypoints.nose[0]);
    const lbInner = toPx(keypoints.left_eyebrow[keypoints.left_eyebrow.length - 1]);
    const rbInner = toPx(keypoints.right_eyebrow[0]);
    meshCtx.beginPath();
    meshCtx.moveTo(...lbInner);
    meshCtx.lineTo(...noseRoot);
    meshCtx.lineTo(...rbInner);
    meshCtx.stroke();
  }

  // Nối chóp mũi tới 2 khóe miệng
  if (keypoints.nose && keypoints.mouth) {
    const noseTip = toPx(keypoints.nose[Math.min(6, keypoints.nose.length - 1)]);
    const mouthLeft = toPx(keypoints.mouth[0]);
    const mouthRight = toPx(keypoints.mouth[Math.min(10, keypoints.mouth.length - 1)]);
    meshCtx.beginPath();
    meshCtx.moveTo(...mouthLeft);
    meshCtx.lineTo(...noseTip);
    meshCtx.lineTo(...mouthRight);
    meshCtx.stroke();
  }

  // Nối đáy môi dưới tới đỉnh cằm
  if (keypoints.mouth && keypoints.jaw) {
    const mouthBottom = toPx(keypoints.mouth[Math.min(15, keypoints.mouth.length - 1)]);
    const chin = toPx(keypoints.jaw[Math.floor(keypoints.jaw.length / 2)]);
    meshCtx.beginPath();
    meshCtx.moveTo(...mouthBottom);
    meshCtx.lineTo(...chin);
    meshCtx.stroke();
  }
  meshCtx.restore();

  // 2. Vẽ đường bao ngũ quan tinh tế và các micro-landmarks
  const partsList = [
    { key: 'jaw', isClosed: false, isOk: true, name: 'Khung hàm' },
    { key: 'left_eyebrow', isClosed: false, isOk: partsStatus.left_eyebrow !== false, name: 'Chân mày trái' },
    { key: 'right_eyebrow', isClosed: false, isOk: partsStatus.right_eyebrow !== false, name: 'Chân mày phải' },
    { key: 'left_eye', isClosed: true, isOk: partsStatus.left_eye !== false, name: 'Mắt trái' },
    { key: 'right_eye', isClosed: true, isOk: partsStatus.right_eye !== false, name: 'Mắt phải' },
    { key: 'nose', isClosed: false, isOk: partsStatus.nose !== false, name: 'Sống mũi' },
    { key: 'mouth', isClosed: true, isOk: partsStatus.mouth !== false, name: 'Miệng/Môi' },
  ];

  partsList.forEach((part) => {
    const pts = keypoints[part.key];
    if (!pts || pts.length === 0) return;

    meshCtx.save();
    const isOk = part.isOk;
    meshCtx.strokeStyle = isOk ? defaultStroke : occludedStroke;
    meshCtx.lineWidth = isOk ? 0.95 : 1.6;
    meshCtx.shadowColor = isOk ? defaultShadow : occludedShadow;
    meshCtx.shadowBlur = isOk ? 3 : 8;

    // Vẽ đường viền ngũ quan mượt mà
    meshCtx.beginPath();
    const [startPx, startPy] = toPx(pts[0]);
    meshCtx.moveTo(startPx, startPy);
    for (let i = 1; i < pts.length; i++) {
      const [px, py] = toPx(pts[i]);
      meshCtx.lineTo(px, py);
    }
    if (part.isClosed) {
      meshCtx.closePath();
    }
    meshCtx.stroke();

    // Micro-landmarks (tinh tế, không che lấp khuôn mặt người dùng)
    meshCtx.fillStyle = isOk ? defaultFill : occludedFill;
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = toPx(pts[i]);
      meshCtx.beginPath();
      meshCtx.arc(px, py, isOk ? 1.25 : 1.9, 0, Math.PI * 2);
      meshCtx.fill();
    }

    meshCtx.restore();
  });

  // 3. Floating Security Alert Capsule (Apple Glassmorphism Design)
  let alertText = null;
  if (antiSpoof && antiSpoof.is_real === false) {
    if (antiSpoof.spoof_type === 'print_attack' || antiSpoof.spoof_type === 'planar_2d') {
      alertText = 'Phát hiện ảnh in 2D • Yêu cầu người thật';
    } else if (antiSpoof.spoof_type === 'replay_attack' || antiSpoof.spoof_type === 'screen_moire') {
      alertText = 'Phát hiện màn hình / video • Yêu cầu người thật';
    } else {
      alertText = 'Phát hiện dấu hiệu giả mạo sinh trắc';
    }
  } else if (isUpsideDown) {
    alertText = 'Vui lòng giữ thẳng khuôn mặt';
  } else if (hasHandOcclusion) {
    alertText = 'Vui lòng không để tay che khuôn mặt';
  } else if (occludedPartName) {
    alertText = `Khuôn mặt bị che: ${occludedPartName}`;
  }

  if (alertText) {
    meshCtx.save();
    meshCtx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    const textMetrics = meshCtx.measureText(alertText);
    const boxW = Math.min(w - 32, textMetrics.width + 42);
    const boxH = 34;
    const boxX = (w - boxW) / 2;
    const boxY = 56;

    // Nền Frosted Glass Capsule sang trọng
    meshCtx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    meshCtx.shadowColor = 'rgba(0, 0, 0, 0.35)';
    meshCtx.shadowBlur = 16;
    meshCtx.beginPath();
    if (meshCtx.roundRect) {
      meshCtx.roundRect(boxX, boxY, boxW, boxH, 17);
    } else {
      meshCtx.rect(boxX, boxY, boxW, boxH);
    }
    meshCtx.fill();

    // Viền hairline cảnh báo
    meshCtx.strokeStyle = 'rgba(239, 68, 68, 0.55)';
    meshCtx.lineWidth = 1;
    meshCtx.stroke();

    // Chấm đỏ breathing status indicator
    meshCtx.fillStyle = '#ef4444';
    meshCtx.shadowColor = '#ef4444';
    meshCtx.shadowBlur = 6;
    meshCtx.beginPath();
    meshCtx.arc(boxX + 18, boxY + boxH / 2, 4, 0, Math.PI * 2);
    meshCtx.fill();

    // Chữ thông báo hiện đại
    meshCtx.fillStyle = '#f8fafc';
    meshCtx.textAlign = 'left';
    meshCtx.textBaseline = 'middle';
    meshCtx.shadowBlur = 0;
    meshCtx.fillText(alertText, boxX + 30, boxY + boxH / 2);
    meshCtx.restore();
  }
}

function playTingSound() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.35, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.6);
  } catch (e) {
    console.debug('Không thể phát âm thanh chime:', e);
  }
}

function renderFeedback(data) {
  // Phát âm thanh ting khi hoàn thành FQA chuẩn bị sang liveness
  if (data.fqa_passed_sound) {
    playTingSound();
  }

  // Cập nhật thông điệp và màu sắc viền Oval
  bottomPill.textContent = data.message;
  bottomPill.className = `bottom-guidance-pill ${data.color || 'cyan'}`;

  // Cập nhật khung Oval SVG (vị trí, kích thước, tiến độ)
  updateOvalSvg(data.oval, data.color, data.progress);

  // Vẽ lưới Biometric Mesh & ngũ quan công nghệ cao lên meshCanvas
  drawBiometricMesh(
    data.keypoints,
    data.parts_status || {},
    data.occluded_part_name,
    data.face_checks?.no_hand_occlusion === false,
    data.color || 'cyan',
    data.anti_spoof,
    Boolean(data.face_metrics?.is_upside_down || data.face_checks?.not_upside_down === false)
  );

  // Cập nhật mũi tên xoay đầu
  if (data.turn_arrow === 'left') {
    turnArrowLeft.classList.remove('hidden');
    turnArrowRight.classList.add('hidden');
  } else if (data.turn_arrow === 'right') {
    turnArrowRight.classList.remove('hidden');
    turnArrowLeft.classList.add('hidden');
  } else {
    turnArrowLeft.classList.add('hidden');
    turnArrowRight.classList.add('hidden');
  }

  // Cập nhật thanh tiến độ giữ yên
  const pct = Math.round((data.progress || 0) * 100);
  holdProgressBar.style.width = `${pct}%`;
  if (data.stage === 'flashing') {
    holdLabel.textContent = `Đang quét quang phổ ánh sáng • ${pct}%`;
  } else if (data.stage === 'zoom_in') {
    holdLabel.textContent = `Đưa khuôn mặt lại gần hơn • ${pct}%`;
  } else if (data.stage.startsWith('turn_')) {
    holdLabel.textContent = `Giữ nguyên góc quay đầu • ${pct}%`;
  } else if (data.stage === 'face_quality') {
    holdLabel.textContent = `Căn chỉnh & giữ yên khuôn mặt • ${pct}%`;
  } else {
    holdLabel.textContent = `Tiến độ xác thực • ${pct}%`;
  }

  // Cập nhật badge thông số trên video
  const cMet = data.camera_metrics || {};
  const fMet = data.face_metrics || {};
  badgeFpsRes.innerHTML = `<span>${cMet.width ?? '--'}×${cMet.height ?? '--'}</span>`;

  if (fMet.yaw !== undefined && fMet.yaw !== null) {
    const rollStr = (fMet.roll !== undefined && fMet.roll !== null) ? `<span class="divider">•</span><span>Roll: ${fMet.roll > 0 ? '+' : ''}${fMet.roll}°</span>` : '';
    badgeHeadPose.innerHTML = `<span>Yaw: ${fMet.yaw > 0 ? '+' : ''}${fMet.yaw}°</span><span class="divider">•</span><span>Pitch: ${fMet.pitch > 0 ? '+' : ''}${fMet.pitch}°</span>${rollStr}`;
  } else {
    badgeHeadPose.innerHTML = `<span>Chưa có mặt</span>`;
  }

  // Cập nhật Stepper
  updateStepper(data.stage);

  // Cập nhật Realtime Checklist
  updateChecklist(data.camera_checks, cMet, data.face_checks, fMet, data.stage);
}

// Cập nhật hình dạng Oval SVG
function updateOvalSvg(oval, color, progress) {
  if (!oval) return;
  // Quy đổi toạ độ chuẩn hoá sang hệ toạ độ SVG 480x600
  const cx = Math.round(oval.cx * 480);
  const cy = Math.round(oval.cy * 600);
  const rx = Math.round(oval.rx * 480);
  const ry = Math.round(oval.ry * 600);

  // Cập nhật lỗ khoét mặt nạ
  svgMaskHole.setAttribute('cx', cx);
  svgMaskHole.setAttribute('cy', cy);
  svgMaskHole.setAttribute('rx', rx);
  svgMaskHole.setAttribute('ry', ry);

  // Cập nhật viền oval
  svgOvalBorder.setAttribute('cx', cx);
  svgOvalBorder.setAttribute('cy', cy);
  svgOvalBorder.setAttribute('rx', rx);
  svgOvalBorder.setAttribute('ry', ry);
  svgOvalBorder.className = `oval-stroke ${color || 'cyan'}`;

  // Cập nhật viền tiến độ (chu vi elip Ramanujan: P ≈ π * [3(a+b) - sqrt((3a+b)(a+3b))])
  const a = rx;
  const b = ry;
  const perimeter = Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));

  svgProgressBorder.setAttribute('cx', cx);
  svgProgressBorder.setAttribute('cy', cy);
  svgProgressBorder.setAttribute('rx', rx);
  svgProgressBorder.setAttribute('ry', ry);
  svgProgressBorder.style.strokeDasharray = `${perimeter}`;

  const clampedProgress = Math.max(0, Math.min(1, progress || 0));
  const offset = perimeter * (1 - clampedProgress);
  svgProgressBorder.style.strokeDashoffset = `${offset}`;
}

// Cập nhật trạng thái Stepper
function updateStepper(stage) {
  const steps = ['camera_check', 'face_quality', 'liveness', 'zoom_in', 'capture'];
  const stageMap = {
    camera_check: 0,
    face_quality: 1,
    turn_left: 2,
    turn_right: 2,
    recenter: 2,
    zoom_in: 3,
    flashing: 3,
    capture: 4,
    failed: 3,
  };

  const activeIdx = stageMap[stage] ?? 0;
  steps.forEach((sName, idx) => {
    const el = document.getElementById(`step-${sName}`);
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (idx < activeIdx || (stage === 'capture' && idx === activeIdx)) {
      el.classList.add('completed');
    } else if (idx === activeIdx) {
      el.classList.add('active');
    }
  });
}

function resetStepperUI() {
  updateStepper('camera_check');
}

// Cập nhật Realtime Checklist
function updateChecklist(camChk = {}, camMet = {}, faceChk = {}, faceMet = {}, stage = '') {
  // Nhóm 1: Camera
  setRow('chk-resolution', 'val-resolution', camChk.resolution_ok, `${camMet.width || '--'}×${camMet.height || '--'}`);
  setRow('chk-signal', 'val-signal', camChk.not_black && camChk.not_frozen, camChk.not_black ? (camChk.not_frozen ? 'Bình thường' : 'Đóng băng') : 'Mất tín hiệu');
  setRow('chk-cam-bright', 'val-cam-bright', camChk.brightness_ok, camMet.frame_brightness !== undefined ? `${camMet.frame_brightness}` : '--');
  setRow('chk-cam-noise', 'val-cam-noise', camChk.noise_ok, camMet.noise_sigma !== undefined ? `σ = ${camMet.noise_sigma}` : '--');

  // Nhóm 2: Chống Giả Mạo Ảnh & Video (PAD MiniFASNet)
  setRow('chk-anti-spoof-real', 'val-anti-spoof-real', faceChk.anti_spoof_ok, (faceMet.anti_spoof_real_prob !== undefined && faceMet.anti_spoof_real_prob !== null) ? `${Math.round(faceMet.anti_spoof_real_prob * 100)}%` : '--');
  setRow('chk-anti-print', 'val-anti-print', faceChk.no_print_attack, faceChk.no_print_attack ? 'Không có' : 'Phát hiện ảnh in');
  setRow('chk-anti-replay', 'val-anti-replay', faceChk.no_screen_attack, faceChk.no_screen_attack ? 'Không có' : 'Phát hiện màn hình');
  setRow('chk-3d-depth', 'val-3d-depth', faceChk.depth_3d_ok, faceChk.depth_3d_ok ? 'Đạt chuẩn 3D' : 'Mặt phẳng 2D');

  // Nhóm 3: Khuôn mặt (FQA - 4 Tiêu chí Cốt lõi)
  setRow('chk-face-count', 'val-face-count', faceChk.single_face, faceMet.face_count !== undefined ? `${faceMet.face_count} người` : 'Chưa có');
  let scaleText = '--';
  if (faceMet.scale_ratio !== undefined && faceMet.scale_ratio !== null) {
    const pct = Math.round(faceMet.scale_ratio * 100);
    if (faceMet.scale_ratio < 0.40) {
      scaleText = `Quá xa (${pct}%)`;
    } else if (faceMet.scale_ratio > 0.85) {
      scaleText = `Quá gần (${pct}%)`;
    } else {
      scaleText = `Vừa vặn (${pct}%)`;
    }
  }
  setRow('chk-scale', 'val-scale', faceChk.scale_ok, scaleText);
  setRow('chk-inside-oval', 'val-inside-oval', faceChk.inside_oval, faceChk.inside_oval ? 'Trọn trong Oval' : 'Tràn ngoài');

  let straightText = '--';
  if (faceMet.is_upside_down || faceChk.not_upside_down === false) {
    straightText = 'Lật ngược đầu!';
  } else if (faceMet.roll !== undefined && faceMet.roll !== null && Math.abs(faceMet.roll) > 10.0) {
    straightText = `Nghiêng (${faceMet.roll > 0 ? '+' : ''}${faceMet.roll}°)`;
  } else if (faceMet.pitch !== undefined && faceMet.pitch !== null && Math.abs(faceMet.pitch) > 15.0) {
    straightText = `Ngẩng/cúi (${faceMet.pitch > 0 ? '+' : ''}${faceMet.pitch}°)`;
  } else if (faceMet.yaw !== undefined && faceMet.yaw !== null && Math.abs(faceMet.yaw) > 12.0) {
    straightText = `Quay (${faceMet.yaw > 0 ? '+' : ''}${faceMet.yaw}°)`;
  } else if (faceMet.yaw !== undefined && faceMet.yaw !== null) {
    straightText = `Thẳng (${faceMet.yaw > 0 ? '+' : ''}${faceMet.yaw}°)`;
  }
  setRow('chk-straight', 'val-straight', faceChk.head_straight, straightText);
  setRow('chk-face-bright', 'val-face-bright', faceChk.illumination_ok, (faceMet.brightness_mean !== undefined && faceMet.brightness_mean !== null) ? `${Math.round(faceMet.brightness_mean)}` : '--');
  setRow('chk-backlight', 'val-backlight', faceChk.no_backlight, (faceMet.brightness_std !== undefined && faceMet.brightness_std !== null) ? `σ = ${Math.round(faceMet.brightness_std)}` : '--');
  setRow('chk-mask', 'val-mask', faceChk.no_mask, faceChk.no_mask ? 'Không có' : 'Phát hiện khẩu trang');
  setRow('chk-sunglasses', 'val-sunglasses', faceChk.no_sunglasses, faceChk.no_sunglasses ? 'Không có' : 'Phát hiện kính râm');
  setRow('chk-glare', 'val-glare', faceChk.no_glare, faceChk.no_glare ? 'Không có' : 'Bị lóa kính');
  setRow('chk-hand', 'val-hand', faceChk.no_hand_occlusion, faceChk.no_hand_occlusion ? 'Không có' : 'Phát hiện tay che');
  setRow('chk-eyes', 'val-eyes', faceChk.has_eyes, faceChk.has_eyes ? 'Rõ nét' : 'Mắt bị che');
  setRow('chk-nose', 'val-nose', faceChk.has_nose, faceChk.has_nose ? 'Rõ nét' : 'Mũi bị che');
  setRow('chk-mouth', 'val-mouth', faceChk.has_mouth, faceChk.has_mouth ? 'Rõ nét' : 'Miệng bị che');
  setRow('chk-eyebrows', 'val-eyebrows', faceChk.has_eyebrows, faceChk.has_eyebrows ? 'Rõ nét' : 'Chân mày bị che');

  // Nhóm 4: Thử thách Sinh trắc
  const isLivenessActive = stage.startsWith('turn_') || stage === 'recenter';
  const isZoomActive = stage === 'zoom_in';
  const isFlashingActiveStage = stage === 'flashing';
  const isPastLiveness = stage === 'zoom_in' || stage === 'flashing' || stage === 'capture';
  const isPastZoom = stage === 'flashing' || stage === 'capture';
  const isPastFlashing = stage === 'capture';

  setRow('chk-liveness-turn', 'val-liveness-turn', isPastLiveness ? true : (isLivenessActive ? null : null), isPastLiveness ? 'Đạt' : (isLivenessActive ? 'Đang thực hiện' : 'Chờ'));
  setRow('chk-zoom', 'val-zoom', isPastZoom ? true : (isZoomActive ? null : null), isPastZoom ? 'Đạt' : (isZoomActive ? 'Đang thực hiện' : 'Chờ'));
  setRow('chk-optical-liveness', 'val-optical-liveness', isPastFlashing ? true : (isFlashingActiveStage ? null : null), isPastFlashing ? 'Đạt' : (isFlashingActiveStage ? 'Đang quét màu' : 'Chờ'));
}

function setRow(rowId, valId, isPass, textVal) {
  const row = document.getElementById(rowId);
  const val = document.getElementById(valId);
  if (!row || !val) return;

  val.textContent = textVal;
  row.classList.remove('pass', 'fail');

  const icon = row.querySelector('.chk-icon');
  if (isPass === true) {
    row.classList.add('pass');
    if (icon) icon.textContent = '✓';
  } else if (isPass === false) {
    row.classList.add('fail');
    if (icon) icon.textContent = '✗';
  } else {
    if (icon) icon.textContent = '⚪';
  }
}

function resetChecklistUI() {
  const rows = document.querySelectorAll('.check-row');
  rows.forEach((r) => {
    r.classList.remove('pass', 'fail');
    const ic = r.querySelector('.chk-icon');
    if (ic) ic.textContent = '⚪';
    const vl = r.querySelector('.chk-val');
    if (vl) vl.textContent = '--';
  });
}

function updateBackendStatus(isOnline) {
  if (isOnline) {
    backendStatus.className = 'status-chip online';
    backendStatus.querySelector('.status-label').textContent = 'AI Backend Sẵn sàng';
  } else {
    backendStatus.className = 'status-chip offline';
    backendStatus.querySelector('.status-label').textContent = 'Mất kết nối Backend (8000)';
  }
}

// -----------------------------------------------------------------------------
// 5. CHỤP CHÂN DUNG HD VÀ HIỂN THỊ MODAL TỔNG KẾT
// -----------------------------------------------------------------------------

function showRejectionModal(reason) {
  const modal = document.getElementById('rejectionModal');
  const reasonEl = document.getElementById('rejectionReason');
  if (modal && reasonEl) {
    reasonEl.textContent = reason || 'Khuôn mặt không đạt tiêu chuẩn an toàn sinh trắc học.';
    modal.classList.remove('hidden');
  } else {
    restartSession();
  }
}

async function captureHdAndVerify(initialSummary = {}) {
  // 1. Tạo canvas độ phân giải gốc của camera để chụp ảnh chân dung HD
  const hdCanvas = document.createElement('canvas');
  hdCanvas.width = video.videoWidth || 1280;
  hdCanvas.height = video.videoHeight || 720;
  const hdCtx = hdCanvas.getContext('2d');

  // Lật gương chuẩn như ảnh người dùng nhìn thấy
  hdCtx.save();
  hdCtx.translate(hdCanvas.width, 0);
  hdCtx.scale(-1, 1);
  hdCtx.drawImage(video, 0, 0, hdCanvas.width, hdCanvas.height);
  hdCtx.restore();

  const snapshotDataUrl = hdCanvas.toDataURL('image/jpeg', 0.95);

  // 2. Hiển thị trạng thái đang kiểm định an ninh backend
  bottomPill.className = 'bottom-guidance-pill yellow';
  bottomPill.textContent = 'Đang kiểm định toàn vẹn khuôn mặt & chống giả mạo chân dung...';

  try {
    const res = await fetch(`${API_BASE}/api/enroll/verify_capture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        image: snapshotDataUrl,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Lỗi kiểm định chân dung backend');
    }

    const data = await res.json();
    if (data.passed) {
      playTingSound();
      bottomPill.className = 'bottom-guidance-pill green';
      bottomPill.textContent = 'Đăng ký thành công! Khuôn mặt toàn vẹn & đạt chuẩn sinh trắc học.';
      snapshotImg.src = snapshotDataUrl;
      lastCapturedDataUrl = snapshotDataUrl;

      // Điền số liệu tổng kết
      const summary = data.summary || initialSummary || {};
      const cam = summary.camera || {};
      const fqa = summary.fqa || {};
      const zoom = summary.zoom || {};
      const liv = summary.liveness || {};
      const tim = summary.timings || {};

      document.getElementById('sumRes').textContent = `${cam.width || video.videoWidth}×${cam.height || video.videoHeight}`;
      document.getElementById('sumBright').textContent = fqa.brightness ? `${Math.round(fqa.brightness)}` : 'Đạt chuẩn';

      const leftYaw = liv.turn_left?.achieved_yaw || '--';
      const rightYaw = liv.turn_right?.achieved_yaw || '--';
      document.getElementById('sumTurnLeft').textContent = `${leftYaw}°`;
      document.getElementById('sumTurnRight').textContent = `${rightYaw}°`;

      const growthPct = zoom.growth_ratio ? `${Math.round(zoom.growth_ratio * 100)}%` : '≥ 125%';
      document.getElementById('sumZoom').textContent = growthPct;

      const totalTime = tim.total || '--';
      document.getElementById('sumTime').textContent = `${totalTime}s`;

      // Chuẩn bị form đăng ký người dùng
      if (enrollFormBox && enrollSavedBox) {
        enrollFormBox.classList.remove('hidden');
        enrollSavedBox.classList.add('hidden');
      }
      if (inputUserName) {
        inputUserName.value = '';
        inputUserName.classList.remove('error');
      }
      if (inputUserId) {
        inputUserId.value = '';
      }

      // Hiển thị modal hoàn tất
      summaryModal.classList.remove('hidden');
    } else {
      // Từ chối đăng ký và mở Modal cảnh báo hiện đại
      bottomPill.className = 'bottom-guidance-pill red';
      bottomPill.textContent = `Không đạt: ${data.message}`;

      setTimeout(() => {
        showRejectionModal(data.message);
      }, 300);
    }
  } catch (err) {
    console.error('Lỗi kiểm định capture:', err);
    bottomPill.className = 'bottom-guidance-pill red';
    bottomPill.textContent = `Lỗi kiểm tra bảo mật: ${err.message}`;
    setTimeout(() => {
      showRejectionModal(err.message);
    }, 300);
  }
}

// -----------------------------------------------------------------------------
// 6. TIỆN ÍCH HELPER (FORMAT, ESCAPE, CLIPBOARD)
// -----------------------------------------------------------------------------

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatDateTime(isoStr) {
  if (!isoStr) return '--';
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch (e) {
    return isoStr;
  }
}

async function copyToClipboard(text, triggerEl = null) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    if (triggerEl) {
      const origText = triggerEl.textContent;
      triggerEl.textContent = '✓ Đã sao chép';
      triggerEl.classList.add('copied');
      setTimeout(() => {
        triggerEl.textContent = origText;
        triggerEl.classList.remove('copied');
      }, 1500);
    }
  } catch (err) {
    console.warn('Lỗi copy clipboard:', err);
  }
}

// -----------------------------------------------------------------------------
// 7. ĐIỀU HƯỚNG TABS & QUẢN LÝ VIEW (TABS CONTROLLER)
// -----------------------------------------------------------------------------

function switchTab(targetTabId) {
  currentTab = targetTabId;

  // 1. Cập nhật trạng thái các nút Tabs
  [tabBtnEnroll, tabBtnVerify, tabBtnUsers].forEach((btn) => {
    if (!btn) return;
    if (btn.getAttribute('data-tab') === targetTabId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // 2. Chuyển đổi hiển thị view panels
  const panels = [
    { el: viewEnroll, id: 'viewEnroll' },
    { el: viewVerify, id: 'viewVerify' },
    { el: viewUsers, id: 'viewUsers' },
  ];

  panels.forEach(({ el, id }) => {
    if (!el) return;
    if (id === targetTabId) {
      el.classList.remove('hidden');
      el.classList.add('active');
    } else {
      el.classList.add('hidden');
      el.classList.remove('active');
    }
  });

  // 3. Xử lý logic đặc thù cho từng view
  if (targetTabId === 'viewVerify') {
    stopAutoVerify();
    // Đảm bảo video verify nhận stream webcam hiện tại
    if (verifyVideo && currentStream) {
      verifyVideo.srcObject = currentStream;
      verifyVideo.play().catch(() => {});
    }
    loadUsersListForSelect();
    resetVerifyVerdictUI();
  } else if (targetTabId === 'viewUsers') {
    stopAutoVerify();
    loadUsersTable();
  } else if (targetTabId === 'viewEnroll') {
    stopAutoVerify();
    if (video && currentStream && video.srcObject !== currentStream) {
      video.srcObject = currentStream;
      video.play().catch(() => {});
    }
  }
}

// Gắn sự kiện chuyển tab
if (tabBtnEnroll) tabBtnEnroll.addEventListener('click', () => switchTab('viewEnroll'));
if (tabBtnVerify) tabBtnVerify.addEventListener('click', () => switchTab('viewVerify'));
if (tabBtnUsers) tabBtnUsers.addEventListener('click', () => switchTab('viewUsers'));

// -----------------------------------------------------------------------------
// 8. LƯU HỒ SƠ NGƯỜI DÙNG VÀO CƠ SỞ DỮ LIỆU (SAVE TO USERS DB)
// -----------------------------------------------------------------------------

async function handleSaveUser() {
  if (!inputUserName) return;
  const fullName = inputUserName.value.trim();
  const userId = inputUserId ? inputUserId.value.trim() : '';

  if (!fullName) {
    inputUserName.classList.add('error');
    inputUserName.focus();
    return;
  }
  inputUserName.classList.remove('error');

  const origBtnHtml = btnSaveUser.innerHTML;
  btnSaveUser.disabled = true;
  btnSaveUser.innerHTML = `
    <span class="spinner-inline"></span>
    <span>Đang trích xuất & lưu ArcFace...</span>
  `;

  try {
    const payload = {
      session_id: currentSessionId,
      full_name: fullName,
      user_id: userId || undefined,
      image: lastCapturedDataUrl || snapshotImg.src,
      metadata: {
        enrolled_via: 'web_ekyc',
        timestamp: new Date().toISOString(),
      },
    };

    const res = await fetch(`${API_BASE}/api/users/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Lỗi lưu hồ sơ vào máy chủ');
    }

    const data = await res.json();
    lastSavedUserId = data.user?.user_id;

    // Chuyển sang box thông báo thành công
    if (enrollFormBox && enrollSavedBox) {
      enrollFormBox.classList.add('hidden');
      enrollSavedBox.classList.remove('hidden');
    }

    // Cập nhật số lượng user
    loadUsersCount();
  } catch (err) {
    console.error('Lỗi lưu user:', err);
    alert(`Không thể lưu hồ sơ: ${err.message}`);
  } finally {
    btnSaveUser.disabled = false;
    btnSaveUser.innerHTML = origBtnHtml;
  }
}

if (btnSaveUser) btnSaveUser.addEventListener('click', handleSaveUser);

if (btnGoToVerify) {
  btnGoToVerify.addEventListener('click', () => {
    summaryModal.classList.add('hidden');
    switchTab('viewVerify');
    if (lastSavedUserId) {
      setVerifyMode('1_to_1');
      if (selectVerifyUser) {
        selectVerifyUser.value = lastSavedUserId;
      }
    }
  });
}

if (btnGoToUsers) {
  btnGoToUsers.addEventListener('click', () => {
    summaryModal.classList.add('hidden');
    switchTab('viewUsers');
  });
}

// -----------------------------------------------------------------------------
// 9. QUẢN LÝ NGƯỜI DÙNG PHONG CÁCH KỸ THUẬT (USERS DASHBOARD)
// -----------------------------------------------------------------------------

async function loadUsersCount() {
  try {
    const res = await fetch(`${API_BASE}/api/users`);
    if (!res.ok) return;
    const data = await res.json();
    const total = data.total || 0;
    if (userCountBadge) userCountBadge.textContent = total;
    if (statTotalUsers) statTotalUsers.textContent = total;
  } catch (e) {}
}

async function loadUsersTable() {
  if (!usersTableBody) return;

  usersTableBody.innerHTML = `
    <tr>
      <td colspan="7" style="text-align: center; padding: 32px; color: var(--text-dim);">
        <span class="spinner-inline" style="margin-right: 8px;"></span>
        Đang tải cơ sở dữ liệu khuôn mặt...
      </td>
    </tr>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/users`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    cachedUsersList = data.users || [];

    if (userCountBadge) userCountBadge.textContent = cachedUsersList.length;
    if (statTotalUsers) statTotalUsers.textContent = cachedUsersList.length;

    renderUsersTable(cachedUsersList);
  } catch (err) {
    console.error('Lỗi nạp danh sách users:', err);
    usersTableBody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 24px; color: var(--danger-color);">
          Lỗi kết nối cơ sở dữ liệu: ${err.message}
        </td>
      </tr>
    `;
  }
}

function renderUsersTable(users) {
  if (!usersTableBody) return;

  const searchTerm = inputSearchUsers ? inputSearchUsers.value.trim().toLowerCase() : '';
  const filtered = users.filter((u) => {
    if (!searchTerm) return true;
    return (
      (u.full_name && u.full_name.toLowerCase().includes(searchTerm)) ||
      (u.user_id && u.user_id.toLowerCase().includes(searchTerm))
    );
  });

  if (users.length === 0) {
    usersTableBody.innerHTML = '';
    if (usersEmptyState) usersEmptyState.classList.remove('hidden');
    return;
  }

  if (usersEmptyState) usersEmptyState.classList.add('hidden');

  if (filtered.length === 0) {
    usersTableBody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 32px; color: var(--text-dim);">
          Không tìm thấy hồ sơ nào khớp với từ khóa "<strong>${escapeHtml(searchTerm)}</strong>"
        </td>
      </tr>
    `;
    return;
  }

  usersTableBody.innerHTML = filtered
    .map((u, idx) => {
      const previewStr = (u.embedding_preview || []).map((v) => Number(v).toFixed(3)).join(', ');
      const avatarSrc = u.snapshot_b64 || '';
      return `
        <tr>
          <td class="col-center" style="color: var(--text-dim);">${idx + 1}</td>
          <td>
            <div class="user-avatar-cell">
              ${avatarSrc ? `<img src="${avatarSrc}" alt="Avatar" class="tbl-avatar" />` : '<div class="tbl-avatar-placeholder">👤</div>'}
            </div>
          </td>
          <td>
            <span class="user-id-code" title="Nhấn để sao chép User ID" onclick="copyToClipboard('${u.user_id}', this)">
              ${u.user_id}
            </span>
          </td>
          <td class="user-name-cell">
            <strong>${escapeHtml(u.full_name)}</strong>
          </td>
          <td>
            <div class="vector-preview-cell">
              <span class="vec-chip" title="5 số đầu của vector ArcFace">[${previewStr} ...]</span>
              <button class="btn-tbl-code" onclick="openVectorModal('${u.user_id}', '${escapeHtml(u.full_name)}')" title="Xem chi tiết toàn bộ 512 số float">
                { } 512D
              </button>
            </div>
          </td>
          <td class="col-date">${formatDateTime(u.created_at)}</td>
          <td class="col-actions">
            <button class="btn-tbl-verify" onclick="quickVerifyUser('${u.user_id}')" title="Chuyển sang xác thực thử với người này">
              ⚡ Xác thực
            </button>
            <button class="btn-tbl-del" onclick="deleteUserRecord('${u.user_id}', '${escapeHtml(u.full_name)}')" title="Xóa hồ sơ khỏi CSDL">
              🗑️
            </button>
          </td>
        </tr>
      `;
    })
    .join('');
}

// Tìm kiếm hồ sơ người dùng
if (inputSearchUsers) {
  inputSearchUsers.addEventListener('input', () => {
    renderUsersTable(cachedUsersList);
  });
}

// Nút làm mới danh sách
if (btnRefreshUsers) {
  btnRefreshUsers.addEventListener('click', loadUsersTable);
}

// Nút đăng ký mới từ toolbar hoặc empty state
if (btnNewEnroll) {
  btnNewEnroll.addEventListener('click', () => {
    switchTab('viewEnroll');
    restartSession();
  });
}
if (btnEmptyEnroll) {
  btnEmptyEnroll.addEventListener('click', () => {
    switchTab('viewEnroll');
    restartSession();
  });
}

// Chuyển nhanh sang xác thực đích danh 1:1
window.quickVerifyUser = function (userId) {
  switchTab('viewVerify');
  setVerifyMode('1_to_1');
  if (selectVerifyUser) {
    selectVerifyUser.value = userId;
  }
};

// Xóa hồ sơ người dùng
window.deleteUserRecord = async function (userId, userName) {
  const confirmed = confirm(`Bạn có chắc chắn muốn xóa hồ sơ sinh trắc học của "${userName}" (${userId})?`);
  if (!confirmed) return;

  try {
    const res = await fetch(`${API_BASE}/api/users/${userId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadUsersTable();
    loadUsersListForSelect();
  } catch (err) {
    alert(`Lỗi khi xóa người dùng: ${err.message}`);
  }
};

// -----------------------------------------------------------------------------
// 10. MODAL XEM VECTOR JSON 512D (DÀNH CHO KỸ THUẬT VIÊN)
// -----------------------------------------------------------------------------

window.openVectorModal = async function (userId, userName) {
  if (!vectorModal) return;
  vectorModal.classList.remove('hidden');
  if (vectorModalTitle) vectorModalTitle.textContent = `Vector Đặc Trưng ArcFace 512D`;
  if (vectorModalSubtitle) vectorModalSubtitle.textContent = `Hồ sơ: ${userName} • ID: ${userId}`;
  if (vectorJsonContent) vectorJsonContent.textContent = '// Đang tải toàn bộ 512 số float từ CSDL SQLite...';

  try {
    const res = await fetch(`${API_BASE}/api/users/${userId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentVectorJsonRaw = JSON.stringify(data.embedding || data.embedding_preview || [], null, 2);
    if (vectorJsonContent) {
      vectorJsonContent.textContent = currentVectorJsonRaw;
    }
  } catch (err) {
    if (vectorJsonContent) {
      vectorJsonContent.textContent = `// Lỗi tải vector: ${err.message}`;
    }
  }
};

if (btnCopyVector) {
  btnCopyVector.addEventListener('click', () => {
    if (currentVectorJsonRaw) {
      copyToClipboard(currentVectorJsonRaw, btnCopyVector);
    }
  });
}

if (btnCloseVectorModal) {
  btnCloseVectorModal.addEventListener('click', () => {
    if (vectorModal) vectorModal.classList.add('hidden');
  });
}

// -----------------------------------------------------------------------------
// 11. TRẠM XÁC THỰC SINH TRẮC HỌC THỜI GIAN THỰC (VERIFICATION 1:1 & 1:N)
// -----------------------------------------------------------------------------

function setVerifyMode(mode) {
  currentVerifyMode = mode;
  if (mode === '1_to_1') {
    if (btnMode11) btnMode11.classList.add('active');
    if (btnMode1N) btnMode1N.classList.remove('active');
    if (verifyUserSelectBox) verifyUserSelectBox.classList.remove('hidden');
  } else {
    currentVerifyMode = '1_to_n';
    if (btnMode1N) btnMode1N.classList.add('active');
    if (btnMode11) btnMode11.classList.remove('active');
    if (verifyUserSelectBox) verifyUserSelectBox.classList.add('hidden');
  }
}

if (btnMode1N) btnMode1N.addEventListener('click', () => setVerifyMode('1_to_n'));
if (btnMode11) btnMode11.addEventListener('click', () => setVerifyMode('1_to_1'));

// Slider thay đổi ngưỡng Cosine Similarity
if (verifyThreshold && valVerifyThreshold) {
  verifyThreshold.addEventListener('input', (e) => {
    valVerifyThreshold.textContent = Number(e.target.value).toFixed(2);
  });
}

async function loadUsersListForSelect() {
  if (!selectVerifyUser) return;
  try {
    const res = await fetch(`${API_BASE}/api/users`);
    if (!res.ok) return;
    const data = await res.json();
    const users = data.users || [];

    selectVerifyUser.innerHTML = '';
    if (users.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '-- CSDL chưa có người dùng --';
      selectVerifyUser.appendChild(opt);
      return;
    }

    users.forEach((u) => {
      const opt = document.createElement('option');
      opt.value = u.user_id;
      opt.textContent = `${u.full_name} (${u.user_id})`;
      selectVerifyUser.appendChild(opt);
    });
  } catch (e) {
    console.warn('Lỗi tải danh sách cho select:', e);
  }
}

function resetVerifyVerdictUI() {
  if (verifyVerdictBox) {
    verifyVerdictBox.className = 'verify-verdict-box idle';
  }
  if (verdictIcon) verdictIcon.textContent = '👁️';
  if (verdictTitle) verdictTitle.textContent = 'Sẵn sàng đối sánh';
  if (verdictSubtitle) {
    verdictSubtitle.textContent =
      currentVerifyMode === '1_to_1'
        ? 'Chọn người dùng cần so khớp và nhấn Chụp & Xác thực ngay'
        : 'Nhìn thẳng vào ống kính và nhấn Chụp & Xác thực ngay';
  }
  if (matchedUserCard) matchedUserCard.classList.add('hidden');
  if (candidatesLeaderboardBox) candidatesLeaderboardBox.classList.add('hidden');
}

async function triggerFaceVerification() {
  if (isVerifyingFace) return;
  if (!verifyVideo || verifyVideo.readyState < 2) {
    console.warn('Camera verify chưa sẵn sàng');
    return;
  }

  isVerifyingFace = true;
  if (verifyOverlayBanner) {
    verifyOverlayBanner.classList.remove('hidden');
    if (verifyOverlayText) verifyOverlayText.textContent = 'Đang trích xuất ArcFace 512D & đối sánh...';
  }

  try {
    // 1. Chụp frame hiện tại từ video
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = 640;
    snapCanvas.height = 480;
    const sCtx = snapCanvas.getContext('2d');
    sCtx.save();
    sCtx.translate(640, 0);
    sCtx.scale(-1, 1);
    sCtx.drawImage(verifyVideo, 0, 0, 640, 480);
    sCtx.restore();

    const frameBase64 = snapCanvas.toDataURL('image/jpeg', 0.9);
    const thresholdVal = verifyThreshold ? parseFloat(verifyThreshold.value) : 0.45;
    const targetUserId = currentVerifyMode === '1_to_1' && selectVerifyUser ? selectVerifyUser.value : null;

    if (currentVerifyMode === '1_to_1' && !targetUserId) {
      alert('Vui lòng chọn hồ sơ người dùng để so khớp 1:1');
      return;
    }

    const res = await fetch(`${API_BASE}/api/verify/face`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: frameBase64,
        target_user_id: targetUserId,
        threshold: thresholdVal,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Lỗi đối sánh máy chủ');
    }

    const data = await res.json();
    renderVerifyResult(data);
  } catch (err) {
    console.error('Lỗi xác thực:', err);
    if (verifyVerdictBox) verifyVerdictBox.className = 'verify-verdict-box danger';
    if (verdictIcon) verdictIcon.textContent = '⚠️';
    if (verdictTitle) verdictTitle.textContent = 'LỖI HỆ THỐNG XÁC THỰC';
    if (verdictSubtitle) verdictSubtitle.textContent = err.message;
  } finally {
    isVerifyingFace = false;
    if (verifyOverlayBanner) verifyOverlayBanner.classList.add('hidden');
  }
}

function renderVerifyResult(data) {
  const match = data.match_result || {};
  const isMatch = Boolean(match.is_match);
  const latency = data.latency_ms != null ? `${data.latency_ms} ms` : '-- ms';
  const padOk = Boolean(data.anti_spoof?.is_real);
  const sim = match.similarity != null ? Number(match.similarity).toFixed(4) : '--';
  const conf = match.confidence != null ? `${(match.confidence * 100).toFixed(1)}%` : '--%';
  const dist = match.distance != null ? Number(match.distance).toFixed(4) : '--';

  // Cập nhật thẻ thông số kỹ thuật
  if (metSim) metSim.textContent = sim;
  if (metConf) metConf.textContent = conf;
  if (metDist) metDist.textContent = dist;
  if (metLatency) metLatency.textContent = latency;
  if (metPad) {
    metPad.textContent = padOk
      ? `✓ Người thật (${Math.round((data.anti_spoof.real_prob || 1) * 100)}%)`
      : `✗ Giả mạo (${data.anti_spoof?.spoof_type || 'Attack'})`;
  }

  // Xử lý Verdict Banner & User Card
  if (data.verdict === 'MATCH_SUCCESS') {
    playTingSound();
    if (verifyVerdictBox) verifyVerdictBox.className = 'verify-verdict-box success';
    if (verdictIcon) verdictIcon.textContent = '✅';
    if (verdictTitle) verdictTitle.textContent = 'XÁC THỰC THÀNH CÔNG';
    if (verdictSubtitle) {
      verdictSubtitle.textContent = `Khớp danh tính: ${match.matched_user?.full_name || ''} (Sim: ${sim})`;
    }

    if (matchedUserCard && match.matched_user) {
      matchedUserCard.classList.remove('hidden');
      if (matchedAvatar) matchedAvatar.src = match.matched_user.snapshot_b64 || '';
      if (matchedName) matchedName.textContent = match.matched_user.full_name;
      if (matchedId) matchedId.textContent = match.matched_user.user_id;
      if (matchedTime) matchedTime.textContent = new Date().toLocaleTimeString('vi-VN');
    }
  } else if (data.verdict === 'MISMATCH') {
    if (verifyVerdictBox) verifyVerdictBox.className = 'verify-verdict-box mismatch';
    if (verdictIcon) verdictIcon.textContent = '❌';
    if (verdictTitle) verdictTitle.textContent = 'KHÔNG KHỚP DANH TÍNH';
    if (verdictSubtitle) {
      verdictSubtitle.textContent =
        currentVerifyMode === '1_to_1'
          ? `Khuôn mặt không trùng với hồ sơ đã chọn (Sim: ${sim} < Ngưỡng)`
          : `Không tìm thấy hồ sơ tương đồng trong CSDL (Cao nhất: ${sim})`;
    }
    if (matchedUserCard) matchedUserCard.classList.add('hidden');
  } else if (data.verdict === 'SPOOF_REJECTED') {
    if (verifyVerdictBox) verifyVerdictBox.className = 'verify-verdict-box danger';
    if (verdictIcon) verdictIcon.textContent = '🚨';
    if (verdictTitle) verdictTitle.textContent = 'CẢNH BÁO GIẢ MẠO (PAD ATTACK)';
    if (verdictSubtitle) verdictSubtitle.textContent = data.message || 'Phát hiện ảnh in hoặc màn hình!';
    if (matchedUserCard) matchedUserCard.classList.add('hidden');
  } else {
    // NO_FACE, MULTIPLE_FACES
    if (verifyVerdictBox) verifyVerdictBox.className = 'verify-verdict-box warning';
    if (verdictIcon) verdictIcon.textContent = '⚠️';
    if (verdictTitle) verdictTitle.textContent = 'CHẤT LƯỢNG HÌNH ẢNH CHƯA ĐẠT';
    if (verdictSubtitle) verdictSubtitle.textContent = data.message;
    if (matchedUserCard) matchedUserCard.classList.add('hidden');
  }

  // Render Top Candidates Leaderboard (1:N)
  if (candidatesLeaderboardBox && candidatesList) {
    if (currentVerifyMode === '1_to_n' && match.candidates && match.candidates.length > 0) {
      candidatesLeaderboardBox.classList.remove('hidden');
      candidatesList.innerHTML = match.candidates
        .map((c, i) => {
          const cSim = Number(c.similarity).toFixed(4);
          const cPct = Math.max(0, Math.min(100, Math.round(c.similarity * 100)));
          const isTop = i === 0 && isMatch;
          return `
            <div class="candidate-row ${isTop ? 'matched' : ''}">
              <span class="c-rank">#${i + 1}</span>
              <img src="${c.snapshot_b64 || ''}" alt="Avatar" class="c-avatar" />
              <div class="c-info">
                <div class="c-name">${escapeHtml(c.full_name)}</div>
                <div class="c-id">${c.user_id}</div>
              </div>
              <div class="c-meter">
                <div class="c-bar-track">
                  <div class="c-bar-fill ${isTop ? 'fill-match' : ''}" style="width: ${cPct}%;"></div>
                </div>
                <span class="c-score">${cSim}</span>
              </div>
            </div>
          `;
        })
        .join('');
    } else {
      candidatesLeaderboardBox.classList.add('hidden');
    }
  }
}

if (btnTriggerVerify) {
  btnTriggerVerify.addEventListener('click', triggerFaceVerification);
}

// Auto Verify Loop
function startAutoVerify() {
  if (autoVerifyIntervalId) clearInterval(autoVerifyIntervalId);
  autoVerifyIntervalId = setInterval(() => {
    if (currentTab === 'viewVerify' && !isVerifyingFace) {
      triggerFaceVerification();
    }
  }, 1500);
}

function stopAutoVerify() {
  if (autoVerifyIntervalId) {
    clearInterval(autoVerifyIntervalId);
    autoVerifyIntervalId = null;
  }
  if (chkAutoVerify) chkAutoVerify.checked = false;
}

if (chkAutoVerify) {
  chkAutoVerify.addEventListener('change', (e) => {
    if (e.target.checked) {
      startAutoVerify();
    } else {
      stopAutoVerify();
    }
  });
}

// -----------------------------------------------------------------------------
// 12. KHỞI CHẠY HỆ THỐNG KHI LOAD TRANG
// -----------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', async () => {
  await initCameraDevices();
  await startCamera(activeDeviceId);
  await startSession();
  await loadUsersCount();
  isLoopRunning = true;
  requestAnimationFrame(frameLoop);
});
