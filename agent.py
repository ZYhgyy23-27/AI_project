"""
语音 Agent 工作流：VAD/ASR/MQTT/Flask。
状态机见 navigation_master；工具见 agent_tool；配置见 config；共享状态见 state。
"""
import io
import os
import queue
import tempfile
import threading
import time
import wave
import json
import asyncio
import subprocess
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

import agent_tool
import attitude_viz
import acoustic_module
import asr_audio
import config
import dashscope_asr
import intent_router
import navigation_master
import omni_client
import state

_funasr_import_error = None
try:
    from funasr import AutoModel
except Exception as e:
    AutoModel = None
    _funasr_import_error = e

_edge_tts_import_error = None
try:
    import edge_tts
except Exception as e:
    edge_tts = None
    _edge_tts_import_error = e

voice_queue: queue.Queue[bytes] = queue.Queue(maxsize=512)

voice_pipeline = acoustic_module.AcousticVoicePipeline(config)

_asr_model = None
_asr_lock = threading.Lock()

_agent = None
_agent_lock = threading.Lock()

app = Flask(__name__)
attitude_viz.register(app)


def _mqtt_topic_str(msg) -> str:
    """paho-mqtt 2.x 下 topic 可能为 bytes，与 config 里 str 比较会永远不相等。"""
    t = getattr(msg, "topic", None)
    if t is None:
        return ""
    if isinstance(t, bytes):
        return t.decode("utf-8")
    return str(t)


def pcm_int16_rms_db(pcm: bytes) -> float:
    if len(pcm) < 4:
        return -100.0
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1.0:
        return -100.0
    return 20.0 * np.log10(rms / 32768.0)


def pcm_to_wav(pcm_bytes: bytes, filename: str) -> None:
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def get_asr_model():
    global AutoModel, _funasr_import_error
    if AutoModel is None:
        try:
            from funasr import AutoModel as _AutoModel

            AutoModel = _AutoModel
            _funasr_import_error = None
        except Exception as e:
            _funasr_import_error = e
        return None
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            kw: dict = {
                "model": config.PARAFORMER_MODEL,
                "device": config.FUNASR_DEVICE,
                "disable_pbar": True,
                "disable_log": True,
            }
            if config.FUNASR_VAD_MODEL:
                kw["vad_model"] = config.FUNASR_VAD_MODEL
            if config.FUNASR_PUNC_MODEL:
                kw["punc_model"] = config.FUNASR_PUNC_MODEL
            _asr_model = AutoModel(**kw)
        return _asr_model


def transcribe_wav(wav_path: str) -> str:
    """ASR：可选 DashScope Paraformer（整句 WAV）或本地 FunASR。"""
    if config.ASR_BACKEND == "dashscope":
        t = dashscope_asr.transcribe_wav_file(
            wav_path, model=config.DASHSCOPE_ASR_MODEL, sample_rate=config.SAMPLE_RATE
        )
        if t:
            return t
        print("DashScope ASR 无有效结果，回退本地 FunASR")
    return transcribe_with_paraformer(wav_path)


def transcribe_with_paraformer(wav_path: str) -> str:
    model = get_asr_model()
    if model is None:
        if _funasr_import_error is not None:
            print(f"funasr 导入失败: {_funasr_import_error!r}")
        else:
            print("未安装 funasr，无法执行 Paraformer ASR")
        return ""
    try:
        result = model.generate(input=wav_path, disable_pbar=True, batch_size=1)
        if not result:
            return ""
        first = result[0] if isinstance(result, list) else result
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
        return str(first).strip()
    except Exception as e:
        print("Paraformer 识别异常:", e)
        return ""


def get_agent():
    global _agent
    with _agent_lock:
        if _agent is not None:
            return _agent
        try:
            _llm_kw: dict = {}
            if os.environ.get("TONGYI_MODEL"):
                _llm_kw["model"] = os.environ["TONGYI_MODEL"]
            if os.environ.get("TONGYI_MAX_TOKENS"):
                _llm_kw.setdefault("model_kwargs", {})["max_tokens"] = int(
                    os.environ["TONGYI_MAX_TOKENS"]
                )
            llm = ChatTongyi(**_llm_kw)
            _agent = create_agent(
                model=llm,
                tools=agent_tool.AGENT_TOOLS,
                system_prompt=(
                    "你是一个智能助盲眼镜语音助手。用户输入来自ASR。"
                    "系统有三种会话状态：空闲（默认）、闲聊（天气/时间）、导航/导盲（视觉避障）。"
                    "导航须调用 yolo_detect_current_frame；闲聊中时间用 get_server_time，天气用 get_weather。"
                    "在导航模式下用户每句话通常都要再次调用 YOLO。"
                    "用户说退出导盲/导航或退出闲聊时由系统处理状态切换，你无需再调用工具。"
                    "用户询问视觉场景时也应调用 YOLO。"
                    "请给出简洁中文回复。"
                ),
            )
        except Exception as e:
            print("Agent 初始化失败:", e)
            _agent = None
        return _agent


def run_llm_unified(prompt: str) -> str:
    """对话：Tongyi+LangChain 工具 或 Qwen-Omni 多模态（图文）。"""
    if config.LLM_BACKEND == "omni":
        txt, wav = omni_client.omni_chat(
            prompt,
            attach_camera=config.OMNI_ATTACH_CAMERA_IMAGE,
            stream=True,
            want_model_audio=config.OMNI_STREAM_MODEL_AUDIO_TO_ESP,
        )
        if wav:
            omni_client.set_pending_model_wav(wav)
        return (txt or "抱歉，我没有理解。").strip()
    return run_agent_with_text(prompt)


def run_agent_with_text(text: str) -> str:
    agent = get_agent()
    if agent is None:
        return "Agent 未初始化成功"
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            content = getattr(messages[-1], "content", "")
            if isinstance(content, list):
                content = "".join(str(x) for x in content)
            return str(content).strip()
        return str(result)
    except Exception as e:
        print("Agent 执行异常:", e)
        return "Agent 执行失败"


async def _edge_tts_synthesize_mp3(text: str) -> bytes:
    try:
        # 新版 edge-tts 支持指定 RIFF PCM，ESP 端更容易直接播放。
        communicate = edge_tts.Communicate(
            text=text,
            voice=config.TTS_VOICE,
            output_format="riff-16khz-16bit-mono-pcm",
        )
    except TypeError:
        communicate = edge_tts.Communicate(text=text, voice=config.TTS_VOICE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data")
            if isinstance(data, (bytes, bytearray)):
                chunks.append(bytes(data))
    return b"".join(chunks)


def _split_wav_s16_mono_into_chunk_wavs(wav_bytes: bytes, chunk_ms: int) -> list[bytes]:
    """
    将标准 mono s16 WAV 按时长切成多段，每段均为合法 RIFF WAV（便于 ESP 逐段解码播放）。
    解析失败时返回原整段。
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return [wav_bytes]
    try:
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                return [wav_bytes]
            sr = wf.getframerate()
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)
    except Exception:
        return [wav_bytes]
    chunk_frames = max(1, int(sr * max(50, chunk_ms) / 1000))
    bytes_per_chunk = chunk_frames * 2
    out: list[bytes] = []
    for i in range(0, len(frames), bytes_per_chunk):
        slab = frames[i : i + bytes_per_chunk]
        if not slab:
            continue
        bo = io.BytesIO()
        with wave.open(bo, "wb") as wo:
            wo.setnchannels(1)
            wo.setsampwidth(2)
            wo.setframerate(sr)
            wo.writeframes(slab)
        out.append(bo.getvalue())
    return out if out else [wav_bytes]


def _mqtt_publish_tts_meta_and_audio(meta: dict, audio: bytes) -> None:
    if config.ACOUSTIC_PIPELINE_ENABLED and config.ACOUSTIC_AEC_ENABLED and audio[:4] == b"RIFF":
        voice_pipeline.feed_playback_wav(audio)
    state.publish(config.MQTT_TOPIC_TTS_META, json.dumps(meta, ensure_ascii=False))
    state.publish(config.MQTT_TOPIC_TTS_AUDIO, audio)


def _ffmpeg_bytes_to_wav_s16_mono16k(raw: bytes, input_suffix: str = ".bin") -> bytes:
    """
    将任意 ffmpeg 可识别的编码转为 ESP 可播的 WAV：16 kHz、mono、s16（RIFF WAVE PCM）。
    使用不带格式暗示的后缀，便于 ffmpeg 自动探测（应对 edge-tts 非 WAV 输出）。
    """
    if not raw:
        return b""
    in_fd, in_path = tempfile.mkstemp(suffix=input_suffix, prefix="tts_in_")
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="tts_out_")
    os.close(in_fd)
    os.close(out_fd)
    try:
        with open(in_path, "wb") as f:
            f.write(raw)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("ffmpeg 转 WAV 失败:", proc.stderr.strip())
            return b""
        with open(out_path, "rb") as f:
            out = f.read()
        if len(out) >= 12 and out[:4] == b"RIFF" and out[8:12] == b"WAVE":
            return out
        print("ffmpeg 输出不是有效 WAV，丢弃")
        return b""
    except Exception as e:
        print("音频转 WAV 异常:", e)
        return b""
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def synthesize_tts_audio(text: str) -> tuple[bytes, str]:
    """ESP 仅支持 WAV（内为 PCM S16）；本函数只返回 wav 或空。"""
    if edge_tts is None:
        if _edge_tts_import_error is not None:
            print(f"edge-tts 导入失败: {_edge_tts_import_error!r}")
        else:
            print("未安装 edge-tts，无法执行 TTS。")
        return b"", ""
    try:
        audio = asyncio.run(_edge_tts_synthesize_mp3(text))
        if not audio:
            return b"", ""
        # 若已是 RIFF/WAV，仍规范到 16k/mono/s16，避免格式字段与设备解码不一致
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            normalized = _ffmpeg_bytes_to_wav_s16_mono16k(audio, input_suffix=".wav")
            if normalized:
                return normalized, "wav"
            print("TTS WAV 无法规范到 16k/mono/s16（检查 ffmpeg），跳过下发")
            return b"", ""
        wav = _ffmpeg_bytes_to_wav_s16_mono16k(audio, input_suffix=".bin")
        if wav:
            return wav, "wav"
        print("TTS 无法转为 ESP 可用 WAV，跳过下发")
        return b"", ""
    except Exception as e:
        print("TTS 合成异常:", e)
        return b"", ""


def publish_raw_wav_bytes_for_esp(wav_bytes: bytes, summary_text: str = "") -> None:
    """将已是 WAV（或可被 ffmpeg 识别的）音频下发 ESP，不走 edge-tts。"""
    if not config.TTS_ENABLED or not wav_bytes:
        return
    wav = wav_bytes
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        wav = _ffmpeg_bytes_to_wav_s16_mono16k(wav_bytes, ".bin")
    if not wav or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        print("Omni 语音：无法得到有效 WAV，跳过下发")
        return
    base = {
        "format": "wav",
        "voice": config.DASHSCOPE_OMNI_MODEL,
        "text": summary_text or "",
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 16,
    }
    if config.TTS_MQTT_STREAM:
        chunks = _split_wav_s16_mono_into_chunk_wavs(wav, config.TTS_STREAM_CHUNK_MS)
        n = len(chunks)
        for i, ch in enumerate(chunks):
            meta = {
                **base,
                "bytes": len(ch),
                "stream": True,
                "chunk_seq": i,
                "chunk_total": n,
                "eof": i == n - 1,
                "text": summary_text or "" if i == 0 else "",
            }
            _mqtt_publish_tts_meta_and_audio(meta, ch)
            if config.TTS_STREAM_GAP_MS > 0 and i < n - 1:
                time.sleep(config.TTS_STREAM_GAP_MS / 1000.0)
        return
    meta = {**base, "bytes": len(wav), "stream": False}
    _mqtt_publish_tts_meta_and_audio(meta, wav)


def publish_tts_for_esp(text: str) -> None:
    if not config.TTS_ENABLED:
        return
    audio, audio_format = synthesize_tts_audio(text)
    if not audio or audio_format != "wav":
        return
    if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        print("TTS 载荷不是 WAV 头，不向 ESP 发送")
        return
    base = {
        "format": "wav",
        "voice": config.TTS_VOICE,
        "text": text,
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 16,
    }
    if config.TTS_MQTT_STREAM:
        chunks = _split_wav_s16_mono_into_chunk_wavs(audio, config.TTS_STREAM_CHUNK_MS)
        n = len(chunks)
        for i, ch in enumerate(chunks):
            meta = {
                **base,
                "bytes": len(ch),
                "stream": True,
                "chunk_seq": i,
                "chunk_total": n,
                "eof": i == n - 1,
                "text": text if i == 0 else "",
            }
            _mqtt_publish_tts_meta_and_audio(meta, ch)
            if config.TTS_STREAM_GAP_MS > 0 and i < n - 1:
                time.sleep(config.TTS_STREAM_GAP_MS / 1000.0)
        return
    meta = {**base, "bytes": len(audio), "stream": False}
    _mqtt_publish_tts_meta_and_audio(meta, audio)


def process_sentence(pcm_utterance: bytes) -> None:
    if len(pcm_utterance) < config.MIN_UTTERANCE_BYTES:
        return
    rms_db = pcm_int16_rms_db(pcm_utterance)
    if rms_db < config.MIN_UTTERANCE_RMS_DB:
        return

    pcm_for_asr = pcm_utterance
    if config.ASR_TRIM_SILENCE:
        pcm_for_asr = asr_audio.trim_int16_by_frame_rms(
            pcm_for_asr,
            sample_rate=config.SAMPLE_RATE,
            threshold_db=config.ASR_TRIM_RMS_DB,
        )
    if config.ASR_UTTERANCE_DENOISE_ENABLED:
        pcm_for_asr = asr_audio.denoise_int16_spectral_min_stats(
            pcm_for_asr,
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.ASR_UTTERANCE_DENOISE_N_FFT,
            noise_quantile=config.ASR_UTTERANCE_DENOISE_Q,
            oversubtraction=config.ASR_UTTERANCE_DENOISE_OVERSUB,
            mag_floor=config.ASR_UTTERANCE_DENOISE_MAG_FLOOR,
            traffic_low_cut_hz=config.ASR_UTTERANCE_TRAFFIC_LOW_CUT_HZ,
            traffic_low_bin_gain=config.ASR_UTTERANCE_TRAFFIC_LOW_GAIN,
        )
    if config.ASR_PREEMPHASIS_ENABLED:
        pcm_for_asr = asr_audio.preemphasis_int16(pcm_for_asr, coef=config.ASR_PREEMPHASIS_COEF)
    if config.ASR_NORMALIZE_PCM:
        pcm_for_asr = asr_audio.normalize_int16_peak(pcm_for_asr)

    t0 = time.perf_counter()
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    try:
        pcm_to_wav(pcm_for_asr, wav_path)
        text = transcribe_wav(wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    t_after_asr = time.perf_counter()

    if not text:
        return
    text = navigation_master.normalize_asr_text(text)
    text = navigation_master.apply_asr_domain_fixes(text)
    if not text:
        return
    if navigation_master.is_low_value_asr(text):
        return

    print("ASR:", text)
    state.publish(config.MQTT_TOPIC_ASR_TEXT, text)

    prev_state = navigation_master.session_state
    omni_client.clear_pending_model_wav()
    reply = intent_router.dispatch_asr_text(text, run_llm_unified)
    if not reply:
        omni_client.clear_pending_model_wav()

    if reply:
        if os.environ.get("VOICE_DEBUG_TIMING"):
            t1 = time.perf_counter()
            print(
                f"[timing] asr={t_after_asr - t0:.2f}s llm={t1 - t_after_asr:.2f}s total={t1 - t0:.2f}s"
            )
        print("Agent:", reply)
        state.publish(config.MQTT_TOPIC_AGENT_REPLY, reply)
        curr_state = navigation_master.session_state
        want_tts = curr_state in (
            navigation_master.SessionState.CHAT,
            navigation_master.SessionState.NAVIGATION,
        ) or prev_state in (
            navigation_master.SessionState.CHAT,
            navigation_master.SessionState.NAVIGATION,
        )
        if want_tts:
            omni_wav = omni_client.pop_pending_model_wav()
            if omni_wav and config.OMNI_STREAM_MODEL_AUDIO_TO_ESP:
                publish_raw_wav_bytes_for_esp(omni_wav, summary_text=reply)
            else:
                publish_tts_for_esp(reply)
        else:
            omni_client.clear_pending_model_wav()


def handle_voice(pcm_bytes: bytes) -> None:
    pending = voice_pipeline.process_capture_chunk(pcm_bytes)
    for chunk in pending:
        process_sentence(chunk)


def voice_worker() -> None:
    while True:
        try:
            chunk = voice_queue.get()
            handle_voice(chunk)
        except Exception as e:
            print("语音处理线程异常:", e)


def on_connect(client, userdata, flags, reason_code, properties=None):
    if hasattr(reason_code, "is_failure") and reason_code.is_failure:
        print("MQTT 连接失败:", reason_code)
        return
    print("MQTT 连接成功")
    client.subscribe(config.MQTT_TOPIC_CAM)
    client.subscribe(config.MQTT_TOPIC_VOICE)
    client.subscribe(attitude_viz.mqtt_subscribe_topic())

    def _deferred_sync_state() -> None:
        time.sleep(0.05)
        try:
            state.publish(
                config.MQTT_TOPIC_NAV,
                "1"
                if navigation_master.session_state
                == navigation_master.SessionState.NAVIGATION
                else "0",
            )
            navigation_master.publish_session_state()
        except Exception as e:
            print("同步会话状态失败:", e)

    threading.Thread(target=_deferred_sync_state, daemon=True).start()


def on_message(client, userdata, msg):
    topic = _mqtt_topic_str(msg)
    if topic == config.MQTT_TOPIC_CAM:
        state.handle_camera(msg.payload)
    elif topic == config.MQTT_TOPIC_VOICE:
        try:
            voice_queue.put_nowait(msg.payload)
        except queue.Full:
            print("语音队列已满，丢弃一包 PCM")
    elif topic == attitude_viz.mqtt_subscribe_topic():
        attitude_viz.handle_mqtt_payload(msg.payload)
    else:
        print("未知 topic:", topic)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    print("MQTT 断开:", reason_code)


def mqtt_thread():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    state.set_mqtt_client(client)
    threading.Thread(target=voice_worker, daemon=True, name="voice-worker").start()
    client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
    client.loop_forever()


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>视频流</title></head>
<body>
  <h1>视频流</h1>
  <p><a href="/attitude">实时姿态（IMU 四元数 3D）</a></p>
  <img src="/video" alt="stream">
</body>
</html>
"""


@app.route("/video")
def video_feed():
    def generate():
        while True:
            frame = state.copy_latest_frame()
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _warmup_asr() -> None:
    if config.ASR_BACKEND == "dashscope":
        print("ASR 使用 DashScope Paraformer，跳过本地 FunASR 预热。")
        return
    try:
        get_asr_model()
        print("ASR 模型已预热，首次语音识别延迟会更低。")
    except Exception as e:
        print("ASR 预热跳过:", e)


if __name__ == "__main__":
    threading.Thread(target=_warmup_asr, daemon=True).start()
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    print("打开浏览器：http://47.97.49.60:5000")
    print("姿态 3D：http://47.97.49.60:5000/attitude（订阅 MQTT esp32/attitude）")
    print("说明：主线程阻塞在 Flask 属正常；MQTT/语音在后台线程。")
    print(
        "调延迟/识别：SILENCE_FRAMES_END=55 VAD_AGGRESSIVENESS=1 "
        "FUNASR_DEVICE=cuda TONGYI_MODEL=qwen-turbo VOICE_DEBUG_TIMING=1"
    )
    print(
        "提抗噪：VAD_START_TRIGGER_FRAMES=2 ASR_TRIM_SILENCE=1 ASR_TRIM_RMS_DB=-55；"
        "轻声：MIN_UTTERANCE_RMS_DB=-52 ASR_NORMALIZE_PCM=1"
    )
    print(
        "声学模块：ACOUSTIC_PIPELINE_ENABLED=1 ACOUSTIC_AEC_ENABLED=1 ACOUSTIC_NS_ENABLED=1 "
        "ACOUSTIC_AEC_DELAY_SAMPLES=48 ACOUSTIC_INPUT_CHANNELS=1；双麦：INPUT_CHANNELS=2 "
        "ACOUSTIC_BSS_ENABLED=1"
    )
    print(
        "百炼能力：ASR_BACKEND=dashscope|funasr  DASHSCOPE_API_KEY  DASHSCOPE_ASR_MODEL=paraformer-realtime-v2；"
        "LLM_BACKEND=omni|tongyi  DASHSCOPE_OMNI_MODEL=qwen-omni-turbo-latest  "
        "DASHSCOPE_COMPAT_BASE_URL=…  OMNI_STREAM_MODEL_AUDIO_TO_ESP=0|1"
    )
    print(
        "TTS 分片下发：TTS_MQTT_STREAM=1 TTS_STREAM_CHUNK_MS=400 TTS_STREAM_GAP_MS=0（整句合成后再切 WAV 多包）"
    )
    app.run(host="0.0.0.0", port=5000, threaded=True)
