"""
声学前端：AEC（参考 NLMS）+ NS（频谱维纳/减法）+ 双麦空间掩蔽（BSS 近似）+ WebRTC VAD 句切分。

说明（重要）：
- 单路麦克风无法做经典「盲源分离」；双声道交错 PCM（左/右麦）时启用 STFT 相干掩蔽，相当于轻量空间滤波/BSS 近似。
- AEC 需要播放参考：在下发 TTS WAV 时调用 feed_playback_wav / feed_playback_pcm，与 mic 流均为 16 kHz、s16、单声道片段对齐。
- 播放端到麦克风的延迟可用 ACOUSTIC_AEC_DELAY_SAMPLES 微调。
"""

from __future__ import annotations

import io
import threading
import wave
from collections import deque

import numpy as np
import webrtcvad

__all__ = [
    "AcousticVoicePipeline",
    "wav_bytes_to_pcm_mono_s16",
]


def wav_bytes_to_pcm_mono_s16(wav_bytes: bytes, target_sr: int = 16000) -> bytes:
    """从 WAV 容器解析出 s16 单声道 PCM（必要时简单线性重采样到 target_sr）。"""
    if not wav_bytes or wav_bytes[:4] != b"RIFF":
        return b""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except Exception:
        return b""
    if sw != 2:
        return b""
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    elif ch != 1:
        return b""
    if sr <= 0:
        return b""
    if sr != target_sr and len(x) > 1:
        t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        n_new = max(2, int(round(len(x) * float(target_sr) / float(sr))))
        t_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        x = np.interp(t_new, t_old, x)
    y = np.clip(np.round(x), -32767.0, 32767.0).astype(np.int16)
    return y.tobytes()


class _NLMSReferenceAEC:
    """频域对齐的 NLMS 回声抵消（单声道 mic + 单声道参考）。"""

    def __init__(
        self,
        filter_len: int,
        mu: float,
        delay_samples: int,
        ref_energy_min: float = 5e4,
        ref_ring_max: int = 48000 * 3,
    ) -> None:
        self.L = max(16, int(filter_len))
        self.mu = float(mu)
        self.delay = max(0, int(delay_samples))
        self.ref_energy_min = float(ref_energy_min)
        self.w = np.zeros(self.L, dtype=np.float64)
        self._ref_ring = np.zeros(max(self.L + self.delay + 4, ref_ring_max), dtype=np.int16)
        self._ref_mask = len(self._ref_ring) - 1
        self._abs_ref_pos = 0
        self._mic_abs = 0

    def reset(self) -> None:
        self.w.fill(0.0)
        self._ref_ring.fill(0)
        self._abs_ref_pos = 0
        self._mic_abs = 0

    def feed_reference_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        x = np.frombuffer(pcm, dtype=np.int16)
        for i in range(len(x)):
            self._ref_ring[self._abs_ref_pos & self._ref_mask] = x[i]
            self._abs_ref_pos += 1

    def _ref_vec_end_exclusive(self, abs_idx_end_exc: int) -> np.ndarray | None:
        """取参考向量 [abs_idx_end_exc - L, abs_idx_end_exc)。"""
        if abs_idx_end_exc < self.L:
            return None
        start = abs_idx_end_exc - self.L
        out = np.empty(self.L, dtype=np.float64)
        for k in range(self.L):
            out[k] = float(self._ref_ring[(start + k) & self._ref_mask])
        return out

    def process_frame(self, mic_s16: np.ndarray) -> np.ndarray:
        """mic_s16: shape (frame_samples,) int16"""
        mic = mic_s16.astype(np.float64)
        out = np.empty_like(mic, dtype=np.float64)
        for i in range(len(mic)):
            abs_mic = self._mic_abs + i
            ref_end = abs_mic - self.delay
            rv = self._ref_vec_end_exclusive(ref_end)
            if rv is None:
                out[i] = mic[i]
                continue
            ref_e = float(np.dot(rv, rv))
            if ref_e < self.ref_energy_min:
                out[i] = mic[i]
                continue
            y_hat = float(np.dot(self.w, rv))
            e = mic[i] - y_hat
            denom = ref_e + 1e-6
            self.w += (self.mu * e / denom) * rv
            out[i] = e
        self._mic_abs += len(mic)
        return np.clip(out, -32767.0, 32767.0).astype(np.int16)


class _SpectralNoiseSuppressor:
    """短时傅里叶域噪声估计 + 维纳型增益（重叠相加）。"""

    def __init__(
        self,
        frame_samples: int,
        n_fft: int = 512,
        noise_decay: float = 0.98,
        speech_decay: float = 0.92,
        gain_floor: float = 0.08,
        noise_update_thresh_db: float = -55.0,
        oversubtraction: float = 1.0,
    ) -> None:
        self.frame_samples = frame_samples
        self.n_fft = int(n_fft)
        self.noise_decay = float(noise_decay)
        self.speech_decay = float(speech_decay)
        self.gain_floor = float(gain_floor)
        self.noise_update_thresh_db = float(noise_update_thresh_db)
        self.oversubtraction = float(oversubtraction)
        self._win = np.hanning(self.n_fft).astype(np.float64)
        self._ola = np.zeros(self.n_fft, dtype=np.float64)
        self._buf = np.zeros(self.n_fft, dtype=np.float64)
        self._noise_mag = np.ones(self.n_fft // 2 + 1, dtype=np.float64) * 1e-3

    def reset(self) -> None:
        self._ola.fill(0.0)
        self._buf.fill(0.0)
        self._noise_mag.fill(1e-3)

    def process_frame(self, mic_s16: np.ndarray) -> np.ndarray:
        x = mic_s16.astype(np.float64)
        self._buf[: self.n_fft - self.frame_samples] = self._buf[self.frame_samples :]
        self._buf[self.n_fft - self.frame_samples :] = x

        xf = self._buf * self._win
        X = np.fft.rfft(xf)
        mag = np.abs(X) + 1e-12
        phase = X / mag

        frame_rms = float(np.sqrt(np.mean(x * x)))
        frame_db = 20.0 * np.log10(max(frame_rms, 1.0) / 32768.0)
        if frame_db < self.noise_update_thresh_db:
            self._noise_mag = self.noise_decay * self._noise_mag + (1.0 - self.noise_decay) * mag
        else:
            self._noise_mag = self.speech_decay * self._noise_mag + (1.0 - self.speech_decay) * np.minimum(
                self._noise_mag, mag
            )

        ratio = (self._noise_mag**2) / (mag**2 + 1e-12)
        wiener = 1.0 - self.oversubtraction * ratio
        wiener = np.clip(wiener, self.gain_floor, 1.0)
        Y = wiener * mag * phase
        yf = np.fft.irfft(Y, n=self.n_fft)

        self._ola += yf * self._win
        out_chunk = self._ola[: self.frame_samples].copy()
        self._ola[:-self.frame_samples] = self._ola[self.frame_samples :]
        self._ola[-self.frame_samples :] = 0.0

        return np.clip(out_chunk, -32767.0, 32767.0).astype(np.int16)


class _StereoSpatialMask:
    """双麦相干掩蔽：增强目标方向相干分量（BSS/波束 forming 的轻量近似）。"""

    def __init__(self, frame_samples: int, n_fft: int = 512, coherence_blend: float = 0.35) -> None:
        self.frame_samples = frame_samples
        self.n_fft = int(n_fft)
        self.coherence_blend = float(coherence_blend)
        self._win = np.hanning(self.n_fft).astype(np.float64)
        self._ola = np.zeros(self.n_fft, dtype=np.complex128)
        self._buf0 = np.zeros(self.n_fft, dtype=np.float64)
        self._buf1 = np.zeros(self.n_fft, dtype=np.float64)

    def reset(self) -> None:
        self._ola.fill(0j)
        self._buf0.fill(0.0)
        self._buf1.fill(0.0)

    def process_frame(self, ch0: np.ndarray, ch1: np.ndarray) -> np.ndarray:
        self._buf0[: self.n_fft - self.frame_samples] = self._buf0[self.frame_samples :]
        self._buf1[: self.n_fft - self.frame_samples] = self._buf1[self.frame_samples :]
        self._buf0[self.n_fft - self.frame_samples :] = ch0.astype(np.float64)
        self._buf1[self.n_fft - self.frame_samples :] = ch1.astype(np.float64)

        f0 = self._buf0 * self._win
        f1 = self._buf1 * self._win
        X0 = np.fft.rfft(f0)
        X1 = np.fft.rfft(f1)
        mag0 = np.abs(X0) + 1e-12
        mag1 = np.abs(X1) + 1e-12
        cross = X0 * np.conj(X1)
        coh = np.abs(cross) / (mag0 * mag1 + 1e-12)
        coh = np.clip(coh, 0.0, 1.0)
        mask = self.coherence_blend + (1.0 - self.coherence_blend) * coh

        Xmix = 0.5 * (X0 + X1)
        Y = mask * Xmix
        y = np.fft.irfft(Y, n=self.n_fft)

        self._ola[: self.n_fft] += y * self._win
        out_chunk = self._ola[: self.frame_samples].real.copy()
        self._ola[:-self.frame_samples] = self._ola[self.frame_samples :]
        self._ola[-self.frame_samples :] = 0j

        return np.clip(out_chunk, -32767.0, 32767.0).astype(np.int16)


class AcousticVoicePipeline:
    """
    处理 MQTT 下发的 PCM 块：可选 AEC →（双麦）空间掩蔽 → NS → WebRTC VAD 句切分。
    返回本块内完成的语句 PCM 列表（与原先 handle_voice 语义一致）。
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self.vad = webrtcvad.Vad(cfg.VAD_AGGRESSIVENESS)

        self._aec: _NLMSReferenceAEC | None = None
        if cfg.ACOUSTIC_PIPELINE_ENABLED and cfg.ACOUSTIC_AEC_ENABLED:
            self._aec = _NLMSReferenceAEC(
                filter_len=cfg.ACOUSTIC_AEC_FILTER_LEN,
                mu=cfg.ACOUSTIC_AEC_MU,
                delay_samples=cfg.ACOUSTIC_AEC_DELAY_SAMPLES,
                ref_energy_min=cfg.ACOUSTIC_AEC_REF_ENERGY_MIN,
            )

        self._ns: _SpectralNoiseSuppressor | None = None
        if cfg.ACOUSTIC_PIPELINE_ENABLED and cfg.ACOUSTIC_NS_ENABLED:
            self._ns = _SpectralNoiseSuppressor(
                frame_samples=cfg.FRAME_SAMPLES,
                n_fft=cfg.ACOUSTIC_NS_N_FFT,
                noise_decay=cfg.ACOUSTIC_NS_NOISE_DECAY,
                speech_decay=cfg.ACOUSTIC_NS_SPEECH_DECAY,
                gain_floor=cfg.ACOUSTIC_NS_GAIN_FLOOR,
                noise_update_thresh_db=cfg.ACOUSTIC_NS_NOISE_UPDATE_DB,
                oversubtraction=cfg.ACOUSTIC_NS_OVERSUBTRACTION,
            )

        self._spatial: _StereoSpatialMask | None = None
        if cfg.ACOUSTIC_PIPELINE_ENABLED and cfg.ACOUSTIC_BSS_ENABLED and cfg.ACOUSTIC_INPUT_CHANNELS == 2:
            self._spatial = _StereoSpatialMask(
                frame_samples=cfg.FRAME_SAMPLES,
                n_fft=cfg.ACOUSTIC_BSS_N_FFT,
                coherence_blend=cfg.ACOUSTIC_BSS_COHERENCE_BLEND,
            )

        self._audio_buffer = b""
        self._speech_buffer = b""
        self._silence_count = 0
        self._speech_streak = 0
        self._in_speech = False
        self._preroll_frames: deque[bytes] = deque(maxlen=max(0, cfg.SPEECH_PREROLL_FRAMES))

    def reset_state(self) -> None:
        with self._lock:
            self._audio_buffer = b""
            self._speech_buffer = b""
            self._silence_count = 0
            self._speech_streak = 0
            self._in_speech = False
            self._preroll_frames.clear()
            if self._aec is not None:
                self._aec.reset()
            if self._ns is not None:
                self._ns.reset()
            if self._spatial is not None:
                self._spatial.reset()

    def feed_playback_wav(self, wav_bytes: bytes) -> None:
        pcm = wav_bytes_to_pcm_mono_s16(wav_bytes, target_sr=self._cfg.SAMPLE_RATE)
        self.feed_playback_pcm(pcm)

    def feed_playback_pcm(self, pcm_mono_s16: bytes) -> None:
        if not pcm_mono_s16 or self._aec is None:
            return
        with self._lock:
            self._aec.feed_reference_pcm(pcm_mono_s16)

    def _preprocess_mono_frame(self, frame_mono: bytes) -> bytes:
        """单帧 10ms mono PCM → 处理后 mono PCM。"""
        fs = self._cfg.FRAME_SAMPLES
        x = np.frombuffer(frame_mono, dtype=np.int16)
        if x.size != fs:
            return frame_mono

        if self._spatial is not None:
            # 双麦模式下不应走到此处；兜底直通
            pass

        y = x
        if self._aec is not None:
            y = self._aec.process_frame(y)
        if self._ns is not None:
            y = self._ns.process_frame(y)
        return y.astype(np.int16).tobytes()

    def _preprocess_stereo_frame(self, frame_stereo: bytes) -> bytes:
        fs = self._cfg.FRAME_SAMPLES
        inter = np.frombuffer(frame_stereo, dtype=np.int16)
        if inter.size != fs * 2:
            return frame_stereo[: fs * 2] if len(frame_stereo) >= fs * 4 else frame_stereo
        ch0 = inter[0::2].copy()
        ch1 = inter[1::2].copy()

        if self._spatial is not None:
            mono = self._spatial.process_frame(ch0, ch1)
        else:
            mono = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)

        if self._aec is not None:
            mono = self._aec.process_frame(mono)
        if self._ns is not None:
            mono = self._ns.process_frame(mono)
        return mono.astype(np.int16).tobytes()

    def process_capture_chunk(self, pcm_bytes: bytes) -> list[bytes]:
        cfg = self._cfg
        pending: list[bytes] = []

        with self._lock:
            self._audio_buffer += pcm_bytes
            use_dsp = cfg.ACOUSTIC_PIPELINE_ENABLED
            frame_bytes = cfg.FRAME_SIZE * cfg.ACOUSTIC_INPUT_CHANNELS if use_dsp else cfg.FRAME_SIZE

            while len(self._audio_buffer) >= frame_bytes:
                raw_frame = self._audio_buffer[:frame_bytes]
                self._audio_buffer = self._audio_buffer[frame_bytes:]

                if use_dsp and cfg.ACOUSTIC_INPUT_CHANNELS == 2:
                    proc = self._preprocess_stereo_frame(raw_frame)
                elif use_dsp:
                    proc = self._preprocess_mono_frame(raw_frame)
                else:
                    proc = raw_frame

                pending.extend(self._vad_accumulate_frame(proc))

        return pending

    def _vad_accumulate_frame(self, mono_frame: bytes) -> list[bytes]:
        cfg = self._cfg
        pending: list[bytes] = []
        self._preroll_frames.append(mono_frame)
        try:
            is_speech = self.vad.is_speech(mono_frame, cfg.SAMPLE_RATE)
        except Exception as e:
            print("VAD 处理异常:", e)
            return pending

        if is_speech:
            self._speech_streak += 1
            self._silence_count = 0
            if not self._in_speech:
                if self._speech_streak >= cfg.VAD_START_TRIGGER_FRAMES:
                    self._in_speech = True
                    self._speech_buffer = b"".join(self._preroll_frames)
            else:
                self._speech_buffer += mono_frame
            if len(self._speech_buffer) >= cfg.MAX_UTTERANCE_BYTES:
                pending.append(self._speech_buffer[: cfg.MAX_UTTERANCE_BYTES])
                self._speech_buffer = self._speech_buffer[cfg.MAX_UTTERANCE_BYTES :]
        else:
            self._speech_streak = 0
            if self._in_speech:
                self._silence_count += 1
                if self._silence_count > cfg.SILENCE_FRAMES_END:
                    if self._speech_buffer:
                        pending.append(self._speech_buffer)
                    self._speech_buffer = b""
                    self._silence_count = 0
                    self._in_speech = False

        return pending
