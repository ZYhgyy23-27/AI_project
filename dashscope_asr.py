"""
阿里云 DashScope Paraformer 语音识别（非流式：整段 WAV 一次提交）。

说明：与「麦克风边采边出字」的 WebSocket 流式不同，当前工程仍是 VAD 切段后整句 WAV；
      本模块对应 Paraformer 实时模型的「文件识别」接口，云端仍为 Paraformer 链路。
      真流式请后续把 ESP PCM 直接送入 Recognition.start + send_audio_frame。
"""

from __future__ import annotations

import os
from http import HTTPStatus


def transcribe_wav_file(wav_path: str, model: str, sample_rate: int = 16000) -> str:
    """
    返回识别文本；失败返回空串。
    需环境变量 DASHSCOPE_API_KEY；中国内地模型见 DashScope 文档。
    """
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("DashScope ASR：未设置 DASHSCOPE_API_KEY")
        return ""
    try:
        from dashscope.audio.asr import Recognition
    except ImportError:
        print("DashScope ASR：请 pip install dashscope")
        return ""

    recognition = Recognition(
        model=model,
        format="wav",
        sample_rate=sample_rate,
        language_hints=["zh", "en"],
        callback=None,
    )
    try:
        result = recognition.call(wav_path)
    except Exception as e:
        print("DashScope ASR 调用异常:", e)
        return ""

    if result.status_code != HTTPStatus.OK:
        print("DashScope ASR 错误:", getattr(result, "message", result))
        return ""

    return _flatten_sentence(result.get_sentence())


def _flatten_sentence(s) -> str:
    if s is None:
        return ""
    if isinstance(s, str):
        return s.strip()
    if isinstance(s, dict):
        return str(s.get("text", "") or "").strip()
    if isinstance(s, list):
        parts: list[str] = []
        for item in s:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(s).strip()
