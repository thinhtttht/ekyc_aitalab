import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera,
  Globe,
  HelpCircle,
  Sun,
  User,
  EyeOff,
  Move,
  ChevronDown,
  CheckCircle,
  AlertTriangle,
  XCircle,
  RefreshCw,
  ShieldCheck,
  Check,
  X
} from 'lucide-react';

interface FQAChecks {
  camera_ready?: boolean;
  lighting_ok?: boolean;
  sharpness_ok?: boolean;
  face_detected?: boolean;
  single_face?: boolean;
  centered_ok?: boolean;
  scale_ok?: boolean;
  head_straight?: boolean;
  has_eyebrows?: boolean;
  has_eyes?: boolean;
  has_nose?: boolean;
  has_mouth?: boolean;
  all_features_detected?: boolean;
  inside_oval?: boolean;
  no_hat?: boolean;
  no_mask?: boolean;
  no_sunglasses?: boolean;
  no_hand_occlusion?: boolean;
}

interface FQAMetrics {
  brightness?: number;
  blur?: number;
  face_count?: number;
  scale?: number;
  dx?: number;
  dy?: number;
  tilt_angle?: number;
  [key: string]: any;
}

interface FQAResponse {
  is_valid: boolean;
  status: string;
  message: string;
  color: 'cyan' | 'green' | 'yellow' | 'red';
  checks: FQAChecks;
  metrics: FQAMetrics;
  bbox?: number[] | null;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isEvaluatingRef = useRef<boolean>(false);

  // Trạng thái Camera
  const [cameraState, setCameraState] = useState<'initializing' | 'active' | 'permission_denied' | 'error'>('initializing');
  const [cameraErrorMsg, setCameraErrorMsg] = useState<string>('');

  // Trạng thái AI Backend
  const [backendOnline, setBackendOnline] = useState<boolean>(true);

  // Trạng thái FQA
  const [fqaData, setFqaData] = useState<FQAResponse>({
    is_valid: false,
    status: 'initializing',
    message: 'Đang khởi động camera và hệ thống kiểm tra...',
    color: 'cyan',
    checks: {},
    metrics: {}
  });

  const [currentStep, setCurrentStep] = useState<number>(1);
  const [capturedSnapshot, setCapturedSnapshot] = useState<string | null>(null);
  const [showSuccessModal, setShowSuccessModal] = useState<boolean>(false);

  // Bộ đếm khung hình ổn định liên tiếp (Stability Hold)
  const STABLE_TARGET = 8; // Cần 8 frame liên tiếp hợp lệ (~2.2 giây)
  const [stableCount, setStableCount] = useState<number>(0);
  const stableCountRef = useRef<number>(0);

  // 1. KHỞI TẠO WEBCAM
  const startCamera = useCallback(async () => {
    setCameraState('initializing');
    setCameraErrorMsg('');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Trình duyệt của bạn không hỗ trợ truy cập Camera (WebRTC).');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => console.error('Lỗi phát video:', e));
          setCameraState('active');
        };
      }
    } catch (err: any) {
      console.error('Không thể mở camera:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraState('permission_denied');
        setCameraErrorMsg('Bạn chưa cấp quyền truy cập Camera. Vui lòng cho phép quyền trên trình duyệt.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraState('error');
        setCameraErrorMsg('Không tìm thấy thiết bị Camera nào được kết nối với máy tính.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setCameraState('error');
        setCameraErrorMsg('Camera đang bị một ứng dụng khác chiếm giữ (Zoom, Teams, Zalo,...).');
      } else {
        setCameraState('error');
        setCameraErrorMsg(err.message || 'Lỗi không xác định khi kết nối camera.');
      }
    }
  }, []);

  useEffect(() => {
    startCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [startCamera]);

  // 2. VÒNG LẶP ĐÁNH GIÁ CHẤT LƯỢNG KHUÔN MẶT (FQA SAMPLING LOOP)
  useEffect(() => {
    if (cameraState !== 'active') return;

    const intervalId = setInterval(async () => {
      if (isEvaluatingRef.current) return;
      if (!videoRef.current || videoRef.current.readyState < 2) return;
      if (!canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Giảm độ phân giải xuống 640x480 để gửi nhanh, mượt mà ~30-50ms qua localhost
      canvas.width = 640;
      canvas.height = 480;

      // Lật ngược gương cho canvas để đồng bộ với video flip
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      const base64Image = canvas.toDataURL('image/jpeg', 0.8);

      isEvaluatingRef.current = true;
      try {
        const response = await fetch('http://localhost:8000/api/fqa_check', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image })
        });

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }

        const data: FQAResponse = await response.json();
        setFqaData(data);
        setBackendOnline(true);

        // Kiểm tra tính ổn định liên tục (Hold Still Requirement)
        if (data.is_valid && !capturedSnapshot) {
          const nextCount = stableCountRef.current + 1;
          stableCountRef.current = nextCount;
          setStableCount(nextCount);

          if (nextCount >= STABLE_TARGET) {
            // Tự động chụp frame HD khi đã giữ yên ổn định đủ số frame yêu cầu
            triggerCapture();
          }
        } else {
          stableCountRef.current = 0;
          setStableCount(0);
        }
      } catch (error) {
        console.warn('Backend FQA offline / retry:', error);
        setBackendOnline(false);
        stableCountRef.current = 0;
        setStableCount(0);
      } finally {
        isEvaluatingRef.current = false;
      }
    }, 280);

    return () => clearInterval(intervalId);
  }, [cameraState, capturedSnapshot]);

  // 3. XỬ LÝ CHỤP ẢNH TỰ ĐỘNG HOẶC THỦ CÔNG
  const triggerCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Chụp frame ở độ phân giải gốc của camera (HD 1280x720)
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    const snapshot = canvas.toDataURL('image/jpeg', 0.95);
    setCapturedSnapshot(snapshot);
    setShowSuccessModal(true);
  };

  const handleRetake = () => {
    setCapturedSnapshot(null);
    setShowSuccessModal(false);
    stableCountRef.current = 0;
    setStableCount(0);
  };

  // Trích xuất checks
  const checks = fqaData.checks;
  const isFaceInside = Boolean(checks.face_detected && checks.inside_oval && checks.centered_ok && checks.scale_ok);
  const isFeaturesComplete = Boolean(checks.has_eyebrows && checks.has_eyes && checks.has_nose && checks.has_mouth);
  const isLightingOk = Boolean(checks.lighting_ok);
  const isNoObstruction = Boolean(checks.no_hat && checks.no_mask && checks.no_sunglasses && checks.no_hand_occlusion);
  const isSharpnessOk = Boolean(checks.sharpness_ok);

  return (
    <div>
      {/* Hidden canvas dùng để render frame phân tích */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* 1. TOP NAVBAR */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon"></div>
          <span className="brand-name">Oval eKYC</span>
          <span className="academic-badge">Academic PoC</span>
        </div>
        <div className="nav-actions">
          <div className={`backend-indicator ${backendOnline ? 'online' : 'offline'}`}>
            <span className="dot"></span>
            <span>{backendOnline ? 'AI Model Sẵn sàng' : 'Mất kết nối Backend (8000)'}</span>
          </div>
          <button className="lang-btn">
            <Globe size={16} />
            <span>Tiếng Việt</span>
            <ChevronDown size={14} />
          </button>
          <button className="help-btn" title="Trợ giúp">
            <HelpCircle size={18} />
          </button>
        </div>
      </nav>

      {/* 2. MAIN 3-COLUMN LAYOUT */}
      <div className="main-container">
        {/* CỘT TRÁI: STEPPER */}
        <div className="stepper">
          {/* BƯỚC 1 */}
          <div className={`step-item ${currentStep === 1 ? 'active' : currentStep > 1 ? 'completed' : ''}`}>
            <div className="step-left">
              <div className="step-circle">
                {currentStep > 1 ? <CheckCircle size={18} /> : '1'}
              </div>
              <div className="step-line"></div>
            </div>
            <div className="step-content">
              <div className="step-title">Xác thực khuôn mặt</div>
              <div className="step-desc">Khung Oval & Chuẩn FQA</div>
            </div>
          </div>

          {/* BƯỚC 2 */}
          <div className={`step-item ${currentStep === 2 ? 'active' : currentStep > 2 ? 'completed' : ''}`}>
            <div className="step-left">
              <div className="step-circle">2</div>
              <div className="step-line"></div>
            </div>
            <div className="step-content">
              <div className="step-title">Thử thách Liveness</div>
              <div className="step-desc">Quay đầu & Chớp màu Flash</div>
            </div>
          </div>

          {/* BƯỚC 3 */}
          <div className={`step-item ${currentStep === 3 ? 'active' : ''}`}>
            <div className="step-left">
              <div className="step-circle">3</div>
            </div>
            <div className="step-content">
              <div className="step-title">Hoàn tất đăng ký</div>
              <div className="step-desc">Trích xuất ArcFace 512-d</div>
            </div>
          </div>
        </div>

        {/* CỘT GIỮA: CAMERA VIEWPORT & KHUNG OVAL */}
        <div className="camera-card">
          <div className="camera-box">
            {/* Tag trạng thái camera ở góc trên bên phải */}
            <div className="camera-badge">
              <Camera size={14} />
              <span>
                {cameraState === 'active'
                  ? 'Camera 720p HD'
                  : cameraState === 'initializing'
                  ? 'Đang mở camera...'
                  : 'Lỗi camera'}
              </span>
              <div className={`live-dot ${cameraState === 'active' ? 'active' : 'offline'}`}></div>
            </div>

            {/* Chỉ số đo đạc thời gian thực góc trên bên trái */}
            {cameraState === 'active' && fqaData.metrics && (
              <div className="metrics-badge">
                <span>Sáng: {fqaData.metrics.brightness ?? '--'}</span>
                <span className="divider">•</span>
                <span>Nét: {fqaData.metrics.blur ?? '--'}</span>
              </div>
            )}

            {/* 4 Corner Brackets màu xanh (L-shaped) */}
            <div className="corner-bracket corner-tl"></div>
            <div className="corner-bracket corner-tr"></div>
            <div className="corner-bracket corner-bl"></div>
            <div className="corner-bracket corner-br"></div>

            {/* Video Webcam */}
            <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />

            {/* KHUNG OVAL CHUẨN 1/3 NGANG, 2/4 DỌC */}
            {cameraState === 'active' && (
              <div className={`oval-guide ${fqaData.color || 'cyan'}`}></div>
            )}

            {/* THÔNG BÁO LỖI CAMERA NẾU CÓ */}
            {cameraState !== 'active' && (
              <div className="camera-error-overlay">
                <div className="error-icon">
                  {cameraState === 'permission_denied' ? <AlertTriangle size={36} /> : <XCircle size={36} />}
                </div>
                <h3>{cameraState === 'permission_denied' ? 'Yêu cầu quyền truy cập Camera' : 'Không thể mở Camera'}</h3>
                <p>{cameraErrorMsg || 'Đang kiểm tra kết nối thiết bị...'}</p>
                <button className="btn-retry" onClick={startCamera}>
                  <RefreshCw size={16} />
                  <span>Thử lại</span>
                </button>
              </div>
            )}

            {/* Pill trạng thái hướng dẫn nằm ở đáy video */}
            {cameraState === 'active' && (
              <div className={`bottom-pill ${fqaData.color || 'cyan'}`}>
                {fqaData.message}
              </div>
            )}
          </div>

          {/* THANH TIẾN ĐỘ GIỮ YÊN (STABILITY PROGRESS BAR) */}
          {cameraState === 'active' && fqaData.is_valid && (
            <div style={{ width: '100%', maxWidth: '520px', marginTop: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, color: '#10b981', marginBottom: '6px' }}>
                <span>🎯 Giữ yên đầu: {stableCount}/{STABLE_TARGET}</span>
                <span>{Math.round((stableCount / STABLE_TARGET) * 100)}%</span>
              </div>
              <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.min(100, (stableCount / STABLE_TARGET) * 100)}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #10b981, #059669)',
                  transition: 'width 0.25s ease-out',
                  borderRadius: '4px'
                }}></div>
              </div>
            </div>
          )}

          {/* Nút bấm chụp: Sáng lên khi đạt chuẩn FQA */}
          <button
            className={`btn-capture ${fqaData.is_valid ? 'ready' : ''}`}
            disabled={!fqaData.is_valid}
            onClick={triggerCapture}
          >
            <Camera size={18} />
            <span>
              {fqaData.is_valid
                ? stableCount >= STABLE_TARGET
                  ? 'Đang chụp ảnh HD...'
                  : `Giữ yên (${stableCount}/${STABLE_TARGET}) hoặc Bấm để chụp ngay`
                : 'Vui lòng căn chỉnh theo hướng dẫn'}
            </span>
          </button>
        </div>

        {/* CỘT PHẢI: THẺ HƯỚNG DẪN & KIỂM TRA ĐIỀU KIỆN (REAL-TIME CHECKLIST) */}
        <div className="instructions-card">
          <h2>Hướng dẫn chuẩn eKYC</h2>
          <p className="card-desc">
            Vui lòng tuân thủ các quy tắc dưới đây để hệ thống AI xác thực chất lượng sinh trắc học đạt chuẩn.
          </p>

          <div className="instruction-list">
            {/* MỤC 1: TOÀN BỘ ĐẦU & NGŨ QUAN TRONG KHUNG OVAL */}
            <div className={`instruction-item ${isFaceInside ? 'passed' : checks.face_detected ? 'warning' : ''}`}>
              <div className="inst-icon-box icon-blue">
                <User size={22} />
              </div>
              <div className="inst-text">
                <div className="inst-header">
                  <h4>Đầu & mặt lọt trọn trong Oval</h4>
                  {isFaceInside ? (
                    <span className="badge-status badge-pass"><Check size={12} /> Trọn trong Oval</span>
                  ) : checks.inside_oval === false ? (
                    <span className="badge-status badge-warn">Tràn ngoài viền</span>
                  ) : checks.face_detected ? (
                    <span className="badge-status badge-warn">Căn chỉnh</span>
                  ) : (
                    <span className="badge-status badge-idle">Chờ mặt</span>
                  )}
                </div>
                <p>Đầu, chân mày, 2 mắt, mũi, miệng đều nằm gọn trong khung Oval</p>
              </div>
            </div>

            {/* MỤC 2: ĐỦ ÁNH SÁNG & KHÔNG NGƯỢC SÁNG */}
            <div className={`instruction-item ${isLightingOk ? 'passed' : 'warning'}`}>
              <div className="inst-icon-box icon-cyan">
                <Sun size={22} />
              </div>
              <div className="inst-text">
                <div className="inst-header">
                  <h4>Đủ ánh sáng</h4>
                  {isLightingOk ? (
                    <span className="badge-status badge-pass"><Check size={12} /> Đạt</span>
                  ) : (
                    <span className="badge-status badge-warn">Chưa đạt</span>
                  )}
                </div>
                <p>Không quá tối, không bị chói sáng hoặc ngược sáng</p>
              </div>
            </div>

            {/* MỤC 3: ĐỦ CHÂN MÀY, MẮT, MŨI, MIỆNG & KHÔNG VẬT CẢN */}
            <div className={`instruction-item ${isNoObstruction && isFeaturesComplete ? 'passed' : (checks.no_hat === false || checks.no_mask === false || checks.no_sunglasses === false || checks.no_hand_occlusion === false || checks.has_eyebrows === false) ? 'error' : ''}`}>
              <div className="inst-icon-box icon-red">
                <EyeOff size={22} />
              </div>
              <div className="inst-text">
                <div className="inst-header">
                  <h4>Đủ chân mày, mắt, mũi, miệng</h4>
                  {isNoObstruction && isFeaturesComplete ? (
                    <span className="badge-status badge-pass"><Check size={12} /> Đủ ngũ quan</span>
                  ) : checks.no_hand_occlusion === false ? (
                    <span className="badge-status badge-fail"><X size={12} /> Bị che mặt</span>
                  ) : checks.has_eyebrows === false ? (
                    <span className="badge-status badge-fail"><X size={12} /> Che chân mày</span>
                  ) : checks.no_mask === false ? (
                    <span className="badge-status badge-fail"><X size={12} /> Có khẩu trang</span>
                  ) : checks.no_hat === false ? (
                    <span className="badge-status badge-fail"><X size={12} /> Có đội mũ</span>
                  ) : checks.no_sunglasses === false ? (
                    <span className="badge-status badge-fail"><X size={12} /> Có kính râm</span>
                  ) : (
                    <span className="badge-status badge-idle">Kiểm tra</span>
                  )}
                </div>
                <p>Lộ rõ 2 chân mày, 2 mắt, mũi, miệng, không lấy tay che mặt</p>
              </div>
            </div>

            {/* MỤC 4: GIỮ YÊN TRONG QUÁ TRÌNH QUÉT */}
            <div className={`instruction-item ${isSharpnessOk ? 'passed' : 'warning'}`}>
              <div className="inst-icon-box icon-purple">
                <Move size={22} />
              </div>
              <div className="inst-text">
                <div className="inst-header">
                  <h4>Ảnh sắc nét & Giữ yên</h4>
                  {isSharpnessOk ? (
                    <span className="badge-status badge-pass"><Check size={12} /> Rõ nét</span>
                  ) : (
                    <span className="badge-status badge-warn">Mờ / Rung</span>
                  )}
                </div>
                <p>Giữ yên thiết bị, tránh cử động rung lắc mạnh</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* POPUP XÁC NHẬN CHỤP ẢNH FQA THÀNH CÔNG */}
      {showSuccessModal && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <div className="modal-header">
              <ShieldCheck size={32} color="#10b981" />
              <h3>Chụp ảnh FQA Giai đoạn 1 Thành công!</h3>
            </div>
            {capturedSnapshot && (
              <div className="snapshot-preview">
                <img src={capturedSnapshot} alt="Chân dung đạt chuẩn FQA" />
              </div>
            )}
            <div className="snapshot-info">
              <p><strong>Trạng thái:</strong> Ảnh đạt chuẩn chất lượng Face Quality Assessment.</p>
              <p><strong>Độ nét:</strong> {fqaData.metrics?.blur} | <strong>Độ sáng:</strong> {fqaData.metrics?.brightness}</p>
              <p><strong>Cự ly:</strong> Chuẩn khớp Oval | <strong>Vật cản:</strong> Không phát hiện mũ, khẩu trang, kính râm.</p>
            </div>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={handleRetake}>
                Chụp lại
              </button>
              <button
                className="btn-primary"
                onClick={() => {
                  alert('Giai đoạn 1 (Camera & FQA) đã hoàn tất xuất sắc!\nSẵn sàng tích hợp sang Cổng 1 (Active Liveness & Random Color Flash) theo Pipeline.');
                  setShowSuccessModal(false);
                }}
              >
                Tiếp tục sang Cổng Liveness
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
