"""ASR 前音频处理：提升小音量/远讲场景的识别率。"""

import numpy as np


def preemphasis_int16(pcm: bytes, coef: float = 0.97) -> bytes:
    """一阶预加重 y[n]=x[n]-coef*x[n-1]，强化高频辅音，减轻低频车流掩蔽。"""
    if len(pcm) < 6:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    c = float(coef)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - c * x[:-1]
    return np.clip(y, -32767.0, 32767.0).astype(np.int16).tobytes()


def normalize_int16_peak(pcm: bytes, max_gain: float = 4.0, target_peak: float = 26000.0) -> bytes:
    """
    将 int16 PCM 峰值拉向 target_peak（避免长期过小导致 ASR 信噪比差）。
    max_gain 限制最大放大倍数，防止底噪被过度放大。
    """
    if len(pcm) < 4:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if peak < 200.0:
        return pcm
    gain = min(target_peak / peak, max_gain)
    y = np.clip(x * gain, -32767.0, 32767.0).astype(np.int16)
    return y.tobytes()


def trim_int16_by_frame_rms(
    pcm: bytes,
    sample_rate: int = 16000,
    frame_ms: int = 20,
    threshold_db: float = -50.0,
) -> bytes:
    """
    依据逐帧 RMS dBFS 裁剪句首/句尾低能片段，减少噪声对 ASR 的影响。
    仅做头尾裁剪，不改中间内容；若找不到有效语音则返回原始 pcm。
    """
    if len(pcm) < 4:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16)
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    total = len(x)
    if total < frame_samples:
        return pcm

    def frame_db(start: int) -> float:
        seg = x[start : start + frame_samples].astype(np.float64)
        if seg.size == 0:
            return -100.0
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms < 1.0:
            return -100.0
        return 20.0 * np.log10(rms / 32768.0)

    # 从前向后找首个高于阈值的帧
    start_idx = 0
    found_start = False
    for i in range(0, total - frame_samples + 1, frame_samples):
        if frame_db(i) >= threshold_db:
            start_idx = i
            found_start = True
            break
    if not found_start:
        return pcm

    # 从后向前找末个高于阈值的帧
    end_idx = total
    for i in range(total - frame_samples, -1, -frame_samples):
        if frame_db(i) >= threshold_db:
            end_idx = min(total, i + frame_samples)
            break

    if end_idx <= start_idx:
        return pcm
    return x[start_idx:end_idx].astype(np.int16).tobytes()


def denoise_int16_spectral_min_stats(
    pcm: bytes,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop=None,
    noise_quantile: float = 0.18,
    oversubtraction: float = 1.65,
    mag_floor: float = 0.06,
    traffic_low_cut_hz=None,
    traffic_low_bin_gain: float = 1.0,
) -> bytes:
    """
    整句谱减：在各频率上取 magnitude 沿时间的低分位数作为噪声估计，再过减并保留原相位。
    适合稳态/絮状环境噪声，过强会引入音乐噪声；mag_floor 抑制空洞。
    traffic_low_cut_hz：对低于该频率的 bin 在谱减后再乘 traffic_low_bin_gain，抑制车流低频轰鸣。
    """
    if len(pcm) < n_fft * 2:
        return pcm
    hop = int(hop if hop is not None else n_fft // 4)
    hop = max(1, hop)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n_orig = len(x)
    win = np.hanning(n_fft).astype(np.float64)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    if traffic_low_cut_hz is not None and traffic_low_bin_gain < 1.0:
        low_mask = freqs < float(traffic_low_cut_hz)
    else:
        low_mask = None
    num_frames = 1 + max(0, (n_orig - n_fft) // hop)
    if num_frames < 4:
        return pcm
    total_len = n_fft + (num_frames - 1) * hop
    if total_len > n_orig:
        x = np.pad(x, (0, total_len - n_orig))

    n_bins = n_fft // 2 + 1
    mag_stack = np.empty((n_bins, num_frames), dtype=np.float64)
    phase_stack = np.empty((n_bins, num_frames), dtype=np.float64)
    for t in range(num_frames):
        sl = x[t * hop : t * hop + n_fft]
        frame = sl * win
        X = np.fft.rfft(frame)
        mag_stack[:, t] = np.abs(X) + 1e-12
        phase_stack[:, t] = np.angle(X)

    q = float(np.clip(noise_quantile, 0.02, 0.45))
    noise_mag = np.quantile(mag_stack, q, axis=1)

    out = np.zeros(total_len, dtype=np.float64)
    win_sum = np.zeros(total_len, dtype=np.float64)
    for t in range(num_frames):
        mag = mag_stack[:, t]
        cleaned = mag - oversubtraction * noise_mag
        cleaned = np.maximum(cleaned, mag_floor * mag)
        if low_mask is not None:
            cleaned = cleaned.copy()
            cleaned[low_mask] *= float(traffic_low_bin_gain)
        Xc = cleaned * np.exp(1j * phase_stack[:, t])
        frm = np.fft.irfft(Xc, n=n_fft) * win
        start = t * hop
        out[start : start + n_fft] += frm
        win_sum[start : start + n_fft] += win * win

    nz = win_sum > 1e-12
    out[nz] /= win_sum[nz]
    out = np.clip(out[:n_orig], -32767.0, 32767.0).astype(np.int16)
    return out.tobytes()
