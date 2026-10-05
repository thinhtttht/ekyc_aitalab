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
const btnModalNext = document.getElementById('btnModalNext');

// State Variables
let currentStream = null;
let currentSessionId = null;
let isLoopRunning = false;
let isSendingFrame = false;
let isFlashingActive = false;
let activeDeviceId = localStorage.getItem('ekyc_preferred_camera') || '';

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
btnModalRetake.addEventListener('click', restartSession);
btnModalNext.addEventListener('click', () => {
  alert('Giai đoạn Đăng ký khuôn mặt với Khung Oval & Active Liveness hoàn tất xuất sắc!\nSẵn sàng tích hợp sang Cổng Color Flashing và MiniFASNet tiếp theo.');
  summaryModal.classList.add('hidden');
});

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
      captureHdAndShowSummary(data.summary || {});
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
async function triggerOpticalFlashing() {
  if (isFlashingActive) return;
  isFlashingActive = true;

  const flashingOverlay = document.getElementById('flashingOverlay');
  if (!flashingOverlay) return;

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

    // 2. Chiếu từng màu theo chuỗi thời gian thực (Toàn màn hình phát sáng)
    const colorTextEl = document.getElementById('flashingColorText');
    for (let i = 0; i < sequence.length; i++) {
      const step = sequence[i];

      // Đổi màu nền toàn màn hình
      flashingOverlay.style.backgroundColor = step.hex;
      flashingOverlay.classList.add('active');

      if (colorTextEl) {
        colorTextEl.textContent = `Đang quét quang phổ [Màu ${i + 1}/${sequence.length}: ${step.name}]`;
      }

      holdLabel.textContent = `🌈 Quét quang phổ ${i + 1}/${sequence.length}: ${step.name}`;
      const pct = Math.round(((i + 1) / sequence.length) * 100);
      holdProgressBar.style.width = `${pct}%`;

      // Chờ 180ms để ánh sáng màn hình phủ đều lên da mặt trước khi AWB triệt tiêu
      await new Promise((r) => setTimeout(r, 180));

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

      // Chờ hết thời lượng của màu này (330ms)
      const remainingMs = Math.max(50, (step.duration_ms || 330) - 180);
      await new Promise((r) => setTimeout(r, remainingMs));
    }

    // 3. Tắt lớp phủ sau khi chiếu xong
    flashingOverlay.classList.remove('active');
    flashingOverlay.style.backgroundColor = 'transparent';

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
      captureHdAndShowSummary(verifyData.summary || {});
    } else {
      bottomPill.className = 'bottom-guidance-pill red';
      bottomPill.textContent = verifyData.message || 'Xác thực thất bại. Hệ thống phát hiện bề mặt không hợp lệ.';
      setRow('chk-optical-liveness', 'val-optical-liveness', false, 'Không đạt');
      setTimeout(() => {
        isFlashingActive = false;
      }, 2500);
    }
  } catch (err) {
    console.error('Lỗi quy trình Color Flashing:', err);
    flashingOverlay.classList.remove('active', 'pulse');
    bottomPill.className = 'bottom-guidance-pill red';
    bottomPill.textContent = err.message || 'Lỗi quét quang học. Vui lòng thử lại.';
    setTimeout(() => {
      isFlashingActive = false;
    }, 2000);
  }
}

// -----------------------------------------------------------------------------
// 4. CẬP NHẬT GIAO DIỆN & LƯỚI SINH TRẮC HỌC (BIOMETRIC MESH)
// -----------------------------------------------------------------------------

function drawBiometricMesh(keypoints, partsStatus = {}, occludedPartName = null, hasHandOcclusion = false, overallColor = 'cyan', antiSpoof = null) {
  if (!meshCtx) return;
  meshCtx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);

  if (!keypoints || Object.keys(keypoints).length === 0) return;

  const w = meshCanvas.width;
  const h = meshCanvas.height;
  const toPx = (pt) => [pt[0] * w, pt[1] * h];

  const isGreen = overallColor === 'green';
  const defaultStroke = isGreen ? 'rgba(34, 197, 94, 0.78)' : 'rgba(56, 189, 248, 0.78)';
  const defaultFill = isGreen ? '#86efac' : '#bae6fd';
  const defaultShadow = isGreen ? '#22c55e' : '#38bdf8';

  const occludedStroke = 'rgba(239, 68, 68, 0.95)';
  const occludedFill = '#fca5a5';
  const occludedShadow = '#ef4444';

  // 1. Đường lưới tam giác / vi liên kết sinh trắc (Triangulation HUD)
  meshCtx.save();
  meshCtx.lineWidth = 0.9;
  meshCtx.strokeStyle = isGreen ? 'rgba(34, 197, 94, 0.28)' : 'rgba(56, 189, 248, 0.28)';
  meshCtx.setLineDash([3, 3]);

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

  // Nối chóp mũi tới 2 góc khóe miệng
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

  // 2. Vẽ từng bộ phận ngũ quan và viền hàm
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
    meshCtx.lineWidth = isOk ? 1.6 : 2.6;
    meshCtx.shadowColor = isOk ? defaultShadow : occludedShadow;
    meshCtx.shadowBlur = isOk ? 6 : 14;

    // Vẽ đường bao
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

    // Vẽ các điểm mốc (Landmarks)
    meshCtx.fillStyle = isOk ? defaultFill : occludedFill;
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = toPx(pts[i]);
      meshCtx.beginPath();
      meshCtx.arc(px, py, isOk ? 2.2 : 3.4, 0, Math.PI * 2);
      meshCtx.fill();
    }

    // Nếu bộ phận bị che khuất -> vẽ khung cảnh báo tech brackets
    if (!isOk) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      pts.forEach((pt) => {
        const [px, py] = toPx(pt);
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      });
      const pad = 10;
      minX -= pad; minY -= pad; maxX += pad; maxY += pad;

      meshCtx.lineWidth = 2.0;
      meshCtx.strokeStyle = '#ef4444';
      meshCtx.shadowColor = '#ef4444';
      meshCtx.shadowBlur = 10;
      const bLen = 8;

      // Góc trên trái
      meshCtx.beginPath();
      meshCtx.moveTo(minX, minY + bLen); meshCtx.lineTo(minX, minY); meshCtx.lineTo(minX + bLen, minY);
      // Góc trên phải
      meshCtx.moveTo(maxX - bLen, minY); meshCtx.lineTo(maxX, minY); meshCtx.lineTo(maxX, minY + bLen);
      // Góc dưới trái
      meshCtx.moveTo(minX, maxY - bLen); meshCtx.lineTo(minX, maxY); meshCtx.lineTo(minX + bLen, maxY);
      // Góc dưới phải
      meshCtx.moveTo(maxX - bLen, maxY); meshCtx.lineTo(maxX, maxY); meshCtx.lineTo(maxX, maxY - bLen);
      meshCtx.stroke();

      meshCtx.font = 'bold 11px system-ui, sans-serif';
      meshCtx.fillStyle = '#fee2e2';
      meshCtx.shadowBlur = 4;
      meshCtx.fillText(`[ ! ] ${part.name.toUpperCase()} BỊ CHE`, minX, Math.max(14, minY - 4));
    }

    meshCtx.restore();
  });

  // 3. Hiển thị HUD Alert Banner nếu phát hiện tấn công giả mạo, tay che hoặc ngũ quan bị cản trở
  let alertText = null;
  if (antiSpoof && antiSpoof.is_real === false) {
    if (antiSpoof.spoof_type === 'print_attack' || antiSpoof.spoof_type === 'planar_2d') {
      alertText = '⚠️ CẢNH BÁO: PHÁT HIỆN ẢNH IN 2D';
    } else if (antiSpoof.spoof_type === 'replay_attack' || antiSpoof.spoof_type === 'screen_moire') {
      alertText = '⚠️ CẢNH BÁO: PHÁT HIỆN MÀN HÌNH / VIDEO';
    } else {
      alertText = '⚠️ CẢNH BÁO: TẤN CÔNG GIẢ MẠO';
    }
  } else if (hasHandOcclusion) {
    alertText = '⚠️ PHÁT HIỆN TAY TRÊN KHUÔN MẶT';
  } else if (occludedPartName) {
    alertText = `⚠️ BỊ CHE KHUẤT: ${occludedPartName.toUpperCase()}`;
  }

  if (alertText) {
    meshCtx.save();
    meshCtx.font = 'bold 12px system-ui, sans-serif';
    const textMetrics = meshCtx.measureText(alertText);
    const boxW = textMetrics.width + 28;
    const boxH = 30;
    const boxX = (w - boxW) / 2;
    const boxY = 48;

    meshCtx.fillStyle = 'rgba(185, 28, 28, 0.92)';
    meshCtx.shadowColor = '#ef4444';
    meshCtx.shadowBlur = 16;
    meshCtx.beginPath();
    if (meshCtx.roundRect) {
      meshCtx.roundRect(boxX, boxY, boxW, boxH, 8);
    } else {
      meshCtx.rect(boxX, boxY, boxW, boxH);
    }
    meshCtx.fill();

    meshCtx.strokeStyle = '#fecaca';
    meshCtx.lineWidth = 1.5;
    meshCtx.stroke();

    meshCtx.fillStyle = '#ffffff';
    meshCtx.textAlign = 'center';
    meshCtx.textBaseline = 'middle';
    meshCtx.shadowBlur = 4;
    meshCtx.fillText(alertText, w / 2, boxY + boxH / 2);
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
    data.anti_spoof
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
    holdLabel.textContent = `🌈 Đang quét quang phổ: ${pct}%`;
  } else if (data.stage === 'zoom_in') {
    holdLabel.textContent = `🔍 Tiến gần Oval lớn: ${pct}%`;
  } else if (data.stage.startsWith('turn_')) {
    holdLabel.textContent = `🔄 Giữ góc quay đầu: ${pct}%`;
  } else if (data.stage === 'face_quality') {
    holdLabel.textContent = `🎯 Giữ yên khuôn mặt: ${pct}%`;
  } else {
    holdLabel.textContent = `🎯 Tiến độ: ${pct}%`;
  }

  // Cập nhật badge thông số trên video
  const cMet = data.camera_metrics || {};
  const fMet = data.face_metrics || {};
  badgeFpsRes.innerHTML = `<span>${cMet.width ?? '--'}×${cMet.height ?? '--'}</span>`;

  if (fMet.yaw !== undefined && fMet.yaw !== null) {
    badgeHeadPose.innerHTML = `<span>Yaw: ${fMet.yaw > 0 ? '+' : ''}${fMet.yaw}°</span><span class="divider">•</span><span>Pitch: ${fMet.pitch}°</span>`;
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
    failed: 1,
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
  setRow('chk-scale', 'val-scale', faceChk.scale_ok, (faceMet.scale_ratio !== undefined && faceMet.scale_ratio !== null) ? `${Math.round(faceMet.scale_ratio * 100)}%` : '--');
  setRow('chk-inside-oval', 'val-inside-oval', faceChk.inside_oval, faceChk.inside_oval ? 'Trọn trong Oval' : 'Tràn ngoài');
  setRow('chk-straight', 'val-straight', faceChk.head_straight, (faceMet.yaw !== undefined && faceMet.yaw !== null) ? `Yaw ${faceMet.yaw > 0 ? '+' : ''}${faceMet.yaw}°` : '--');
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

function captureHdAndShowSummary(summary = {}) {
  // Tạo canvas độ phân giải gốc của camera
  const hdCanvas = document.createElement('canvas');
  hdCanvas.width = video.videoWidth || 1280;
  hdCanvas.height = video.videoHeight || 720;
  const hdCtx = hdCanvas.getContext('2d');

  // Lật gương
  hdCtx.save();
  hdCtx.translate(hdCanvas.width, 0);
  hdCtx.scale(-1, 1);
  hdCtx.drawImage(video, 0, 0, hdCanvas.width, hdCanvas.height);
  hdCtx.restore();

  const snapshotDataUrl = hdCanvas.toDataURL('image/jpeg', 0.95);
  snapshotImg.src = snapshotDataUrl;

  // Điền số liệu tổng kết
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

  // Hiển thị modal
  summaryModal.classList.remove('hidden');
}

// -----------------------------------------------------------------------------
// 6. KHỞI CHẠY HỆ THỐNG KHI LOAD TRANG
// -----------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', async () => {
  await initCameraDevices();
  await startCamera(activeDeviceId);
  await startSession();
  isLoopRunning = true;
  requestAnimationFrame(frameLoop);
});
