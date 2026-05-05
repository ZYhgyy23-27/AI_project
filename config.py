"""全局配置：MQTT、模型路径、VAD/音频参数。"""

import os

# 延迟主要来源：①句尾静音 ②Paraformer ③通义 Agent
MQTT_BROKER = "47.97.49.60"
MQTT_PORT = 1883
MQTT_TOPIC_CAM = "esp32/camera"
MQTT_TOPIC_VOICE = "esp32/voice"
MQTT_TOPIC_RESULT = "esp32/result"
MQTT_TOPIC_ASR_TEXT = "esp32/asr_text"
MQTT_TOPIC_AGENT_REPLY = "agent/reply"
MQTT_TOPIC_NAV = "esp32/navigation"
MQTT_TOPIC_SESSION = "esp32/session"
MQTT_TOPIC_TTS_AUDIO = "esp32/tts_audio"
MQTT_TOPIC_TTS_META = "esp32/tts_meta"
# ESP32 本机 Mahony 姿态（JSON：t_us, seq, w, x, y, z）
MQTT_TOPIC_ATTITUDE = "esp32/attitude"

MODEL_PATH = "/root/best_all.pt"

# ASR：默认同 FunASR Hub；可改为更大模型换准确率（更慢），如：
# iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
PARAFORMER_MODEL = os.environ.get("FUNASR_MODEL", "paraformer-zh")

# FunASR 设备：cuda 通常比 cpu 快很多；无 GPU 时设 FUNASR_DEVICE=cpu
FUNASR_DEVICE = os.environ.get("FUNASR_DEVICE", "cuda")

# 可选：与 ASR 同 pipeline 的 VAD/标点（首次会多下载模型；标点利于分句，略增耗时）
# 例：FUNASR_VAD_MODEL=fsmn-vad  FUNASR_PUNC_MODEL=ct-punc-c
FUNASR_VAD_MODEL = os.environ.get("FUNASR_VAD_MODEL") or None
FUNASR_PUNC_MODEL = os.environ.get("FUNASR_PUNC_MODEL") or None

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 10
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_SIZE = FRAME_SAMPLES * 2

# VAD：webrtc 模式 0~3，数值越大越严格（抗噪更强，但可能漏轻声）
# 默认改为 1，提升轻声/远讲命中率。
# 马路/嘈杂人声旁白多时可试 VAD_AGGRESSIVENESS=2，减少「路人说话触发一句」的假起点。
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "1"))
# 进入语音段所需连续语音帧数（10ms/帧），用于抑制瞬时噪声误触发
# 默认改为 2，加快起句触发，减少轻短句漏检。
VAD_START_TRIGGER_FRAMES = int(os.environ.get("VAD_START_TRIGGER_FRAMES", "2"))
# 触发后补回前置帧，避免截断句首
SPEECH_PREROLL_FRAMES = int(os.environ.get("SPEECH_PREROLL_FRAMES", "8"))

# 句尾静音帧数（10ms/帧）：越小越快结束一句，但易切碎；越大延迟高
SILENCE_FRAMES_END = int(os.environ.get("SILENCE_FRAMES_END", "55"))

MIN_SPEECH_FRAMES = int(os.environ.get("MIN_SPEECH_FRAMES", "20"))
MIN_UTTERANCE_BYTES = FRAME_SIZE * MIN_SPEECH_FRAMES
MAX_UTTERANCE_BYTES = FRAME_SIZE * 800

# 能量门限（dBFS）：过低会丢轻声；略放宽利于识别，可能多噪声句
# 默认放宽到 -52，降低“说了但未识别”的概率。
MIN_UTTERANCE_RMS_DB = float(os.environ.get("MIN_UTTERANCE_RMS_DB", "-52"))

# ASR 前是否做峰值归一化（推荐开启，ESP32/远讲场景常偏小）
ASR_NORMALIZE_PCM = os.environ.get("ASR_NORMALIZE_PCM", "1") not in ("0", "false", "False")
# ASR 前是否按帧能量裁掉句首/句尾静音和低能噪声
ASR_TRIM_SILENCE = os.environ.get("ASR_TRIM_SILENCE", "1") not in ("0", "false", "False")
# 裁剪能量阈值（dBFS），数值越高裁剪越激进
# 默认降低到 -55，避免把轻声句首/句尾过度裁掉。
ASR_TRIM_RMS_DB = float(os.environ.get("ASR_TRIM_RMS_DB", "-55"))

# ASR 前整句谱减（车流等宽带噪声有用；旁人说话与语音频谱重叠，过减会伤识别——默认偏保守）
ASR_UTTERANCE_DENOISE_ENABLED = os.environ.get("ASR_UTTERANCE_DENOISE_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
ASR_UTTERANCE_DENOISE_N_FFT = int(os.environ.get("ASR_UTTERANCE_DENOISE_N_FFT", "512"))
ASR_UTTERANCE_DENOISE_OVERSUB = float(os.environ.get("ASR_UTTERANCE_DENOISE_OVERSUB", "1.42"))
ASR_UTTERANCE_DENOISE_Q = float(os.environ.get("ASR_UTTERANCE_DENOISE_Q", "0.24"))
ASR_UTTERANCE_DENOISE_MAG_FLOOR = float(os.environ.get("ASR_UTTERANCE_DENOISE_MAG_FLOOR", "0.07"))
# 谱减后对低于该频率的能量再乘一档增益，专门压马路轰鸣（不影响高频辨音）
ASR_UTTERANCE_TRAFFIC_LOW_CUT_HZ = float(os.environ.get("ASR_UTTERANCE_TRAFFIC_LOW_CUT_HZ", "320"))
ASR_UTTERANCE_TRAFFIC_LOW_GAIN = float(os.environ.get("ASR_UTTERANCE_TRAFFIC_LOW_GAIN", "0.52"))

# ASR 预加重：削弱低频掩蔽、抬高辅音（与车流场景相配）
ASR_PREEMPHASIS_ENABLED = os.environ.get("ASR_PREEMPHASIS_ENABLED", "1") not in ("0", "false", "False")
ASR_PREEMPHASIS_COEF = float(os.environ.get("ASR_PREEMPHASIS_COEF", "0.97"))

# TTS（用于把闲聊/导航回复转换成语音给 ESP 播放）
TTS_ENABLED = os.environ.get("TTS_ENABLED", "1") not in ("0", "false", "False")
TTS_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
# 向 ESP 下发时 agent 一律转为 16 kHz/mono/s16 的 RIFF WAV（不再走 MP3）
TTS_AUDIO_FORMAT = os.environ.get("TTS_AUDIO_FORMAT", "wav").lower()
# 1=将整句 TTS WAV 切成多段「独立有效 WAV」按序多次发 tts_meta+tts_audio（ESP 需按 chunk_seq 排队播放）
TTS_MQTT_STREAM = os.environ.get("TTS_MQTT_STREAM", "0") in ("1", "true", "True")
TTS_STREAM_CHUNK_MS = int(os.environ.get("TTS_STREAM_CHUNK_MS", "400"))
# 分片之间的间隔（毫秒），减轻 ESP/MQTT 瞬时压力；0 表示不等待
TTS_STREAM_GAP_MS = int(os.environ.get("TTS_STREAM_GAP_MS", "0"))

# ---------- 声学前端（AEC / NS / BSS 近似 + WebRTC VAD 句切，见 acoustic_module）----------
# 总开关：0 时仅保留原有 mono PCM + VAD（与未接入模块行为一致）
ACOUSTIC_PIPELINE_ENABLED = os.environ.get("ACOUSTIC_PIPELINE_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
# 采集声道：1=单麦（默认，整段 PCM 即 mono）；2=仅双麦硬件时使用（左/右交错各 160 点/10ms）
ACOUSTIC_INPUT_CHANNELS = int(os.environ.get("ACOUSTIC_INPUT_CHANNELS", "1"))

ACOUSTIC_AEC_ENABLED = os.environ.get("ACOUSTIC_AEC_ENABLED", "1") not in ("0", "false", "False")
ACOUSTIC_AEC_FILTER_LEN = int(os.environ.get("ACOUSTIC_AEC_FILTER_LEN", "256"))
ACOUSTIC_AEC_MU = float(os.environ.get("ACOUSTIC_AEC_MU", "0.35"))
# 播放参考相对麦克风的样点延迟（取决于 ESP 播放链路；回声滞后时可增大）
ACOUSTIC_AEC_DELAY_SAMPLES = int(os.environ.get("ACOUSTIC_AEC_DELAY_SAMPLES", "48"))
# 参考能量低于该阈值时不自适应（避免静音段滤波器发散）
ACOUSTIC_AEC_REF_ENERGY_MIN = float(os.environ.get("ACOUSTIC_AEC_REF_ENERGY_MIN", "5e4"))

ACOUSTIC_NS_ENABLED = os.environ.get("ACOUSTIC_NS_ENABLED", "1") not in ("0", "false", "False")
ACOUSTIC_NS_N_FFT = int(os.environ.get("ACOUSTIC_NS_N_FFT", "512"))
ACOUSTIC_NS_NOISE_DECAY = float(os.environ.get("ACOUSTIC_NS_NOISE_DECAY", "0.97"))
ACOUSTIC_NS_SPEECH_DECAY = float(os.environ.get("ACOUSTIC_NS_SPEECH_DECAY", "0.90"))
ACOUSTIC_NS_GAIN_FLOOR = float(os.environ.get("ACOUSTIC_NS_GAIN_FLOOR", "0.035"))
ACOUSTIC_NS_NOISE_UPDATE_DB = float(os.environ.get("ACOUSTIC_NS_NOISE_UPDATE_DB", "-52"))
# >1 时维纳抑制更强（可能略增音乐噪声）
# 实时 NS 略收敛：旁瓣人声多时过重会与整句谱减叠加以致发糊
ACOUSTIC_NS_OVERSUBTRACTION = float(os.environ.get("ACOUSTIC_NS_OVERSUBTRACTION", "1.22"))

# 双麦 BSS 近似（相干掩蔽）。单麦请保持 0（无第二路信号，开了也无分离效果）
ACOUSTIC_BSS_ENABLED = os.environ.get("ACOUSTIC_BSS_ENABLED", "0") not in ("0", "false", "False")
ACOUSTIC_BSS_N_FFT = int(os.environ.get("ACOUSTIC_BSS_N_FFT", "512"))
ACOUSTIC_BSS_COHERENCE_BLEND = float(os.environ.get("ACOUSTIC_BSS_COHERENCE_BLEND", "0.35"))

# ---------- DashScope：Paraformer 云 ASR + Qwen-Omni 多模态 ----------
# ASR：funasr=本地；dashscope=百炼 Paraformer（当前为 VAD 切段后的整句 WAV；需 DASHSCOPE_API_KEY）
ASR_BACKEND = os.environ.get("ASR_BACKEND", "funasr").strip().lower()
DASHSCOPE_ASR_MODEL = os.environ.get("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2")

# LLM：tongyi=LangChain+工具链；omni=Qwen-Omni（OpenAI 兼容，图像+文本；导盲时由代码预注入 YOLO 文本）
LLM_BACKEND = os.environ.get("LLM_BACKEND", "tongyi").strip().lower()
DASHSCOPE_OMNI_MODEL = os.environ.get("DASHSCOPE_OMNI_MODEL", "qwen-omni-turbo-latest")
# 北京：https://dashscope.aliyuncs.com/compatible-mode/v1  新加坡国际：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
DASHSCOPE_COMPAT_BASE_URL = os.environ.get(
    "DASHSCOPE_COMPAT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
OMNI_ATTACH_CAMERA_IMAGE = os.environ.get("OMNI_ATTACH_CAMERA_IMAGE", "1") not in ("0", "false", "False")
OMNI_TTS_VOICE = os.environ.get("OMNI_TTS_VOICE", "Tina")
# 1=用模型流式语音下发 ESP（常为 24k，服务端 ffmpeg 转 16k）；0=仍用 edge-tts 播报文本（默认，兼容 ESP）
OMNI_STREAM_MODEL_AUDIO_TO_ESP = os.environ.get("OMNI_STREAM_MODEL_AUDIO_TO_ESP", "0") in (
    "1",
    "true",
    "True",
)
