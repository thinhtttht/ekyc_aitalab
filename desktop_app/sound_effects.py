"""Sound effects generator and player for eKYC Desktop Application.
Uses native Windows multimedia sound system with zero third-party dependencies.
"""
from __future__ import annotations

import math
import os
import struct
import wave
import winsound

SOUND_DIR = os.path.join(os.path.dirname(__file__), "assets", "sounds")
os.makedirs(SOUND_DIR, exist_ok=True)

DING_PATH = os.path.join(SOUND_DIR, "ding.wav")
SUCCESS_PATH = os.path.join(SOUND_DIR, "success.wav")
WARNING_PATH = os.path.join(SOUND_DIR, "warning.wav")
TICK_PATH = os.path.join(SOUND_DIR, "tick.wav")


def _generate_wav(filepath: str, duration_sec: float, sample_rate: int = 44100, note_freqs: list = None):
    """Tạo file WAV sóng sin 16-bit PCM với phong cách chuông chime êm ái."""
    if os.path.exists(filepath):
        return

    note_freqs = note_freqs or [880.0]
    n_samples = int(duration_sec * sample_rate)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            # Phân đoạn nốt nếu có nhiều nốt (hợp âm arpeggio)
            seg_len = duration_sec / len(note_freqs)
            note_idx = min(len(note_freqs) - 1, int(t / seg_len))
            freq = note_freqs[note_idx]
            seg_t = t - (note_idx * seg_len)

            # Decay phong cách chuông thủy tinh
            decay = math.exp(-12.0 * seg_t)
            # Sóng sin mượt với hài hòa nốt bậc hai nhẹ
            v1 = math.sin(2.0 * math.pi * freq * t)
            v2 = 0.25 * math.sin(2.0 * math.pi * (freq * 2) * t)
            val = int(decay * 18000.0 * (v1 + v2))
            samples.append(struct.pack("<h", max(-32767, min(32767, val))))

        wf.writeframes(b"".join(samples))


# Khởi tạo các file âm thanh mẫu chuẩn
try:
    # 1. Âm Ting hoàn thành bước: 880Hz (A5) -> 1320Hz (E6)
    _generate_wav(DING_PATH, duration_sec=0.22, note_freqs=[880.0, 1320.0])
    # 2. Âm Hoàn tất xuất sắc: Hợp âm Đô - Mi - Sol - Đô cao (C5, E5, G5, C6)
    _generate_wav(SUCCESS_PATH, duration_sec=0.45, note_freqs=[523.25, 659.25, 783.99, 1046.50])
    # 3. Âm Cảnh báo nhẹ nhàng: 440Hz
    _generate_wav(WARNING_PATH, duration_sec=0.18, note_freqs=[440.0])
    # 4. Âm xung quét quang học: 1600Hz cực ngắn (40ms)
    _generate_wav(TICK_PATH, duration_sec=0.045, note_freqs=[1600.0])
except Exception as e:
    print("[SoundEffects] Không thể khởi tạo file âm thanh:", e)


def play_chime_step():
    """Phát âm thanh 'Ting!' nhẹ nhàng khi hoàn thành từng bước quy trình."""
    try:
        if os.path.exists(DING_PATH):
            winsound.PlaySound(DING_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


play_chime_ding = play_chime_step


def play_chime_success():
    """Phát âm thanh chuông chúc mừng khi đăng ký hoặc xác thực thành công."""
    try:
        if os.path.exists(SUCCESS_PATH):
            winsound.PlaySound(SUCCESS_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


def play_chime_warning():
    """Phát âm thanh nhắc nhở nhẹ nhàng."""
    try:
        if os.path.exists(WARNING_PATH):
            winsound.PlaySound(WARNING_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


def play_chime_tick():
    """Phát âm thanh xung quét quang học cực ngắn theo từng nhịp chớp màu."""
    try:
        if os.path.exists(TICK_PATH):
            winsound.PlaySound(TICK_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass

