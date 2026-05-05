"""
Qwen-Omni（DashScope OpenAI 兼容模式）：图像 + 文本输入，可选流式文本 + 模型语音输出。

依赖：pip install 'openai>=1.52.0'
环境：DASHSCOPE_API_KEY、DASHSCOPE_COMPAT_BASE_URL（地域与密钥需一致）
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import agent_tool
import config
import state

_pending_model_wav: Optional[bytes] = None


def clear_pending_model_wav() -> None:
    global _pending_model_wav
    _pending_model_wav = None


def set_pending_model_wav(data: Optional[bytes]) -> None:
    global _pending_model_wav
    _pending_model_wav = data


def pop_pending_model_wav() -> Optional[bytes]:
    global _pending_model_wav
    t = _pending_model_wav
    _pending_model_wav = None
    return t


def _maybe_prepend_yolo_for_navigation_prompt(prompt: str) -> str:
    """导航/导盲类提示中附带当前 YOLO 结果，弥补 Omni 路径未走 LangChain 工具。"""
    keys = ("导航模式", "导盲", "YOLO", "yolo_detect", "前方视野", "避障")
    if not any(k in prompt for k in keys):
        return prompt
    try:
        tool = agent_tool.yolo_detect_current_frame
        ytxt = tool.invoke({})
    except Exception as e:
        ytxt = f"视觉工具调用失败：{e}"
    return f"[系统视觉参考]\n{ytxt}\n\n{prompt}"


def _jpeg_data_url(frame: np.ndarray, quality: int = 85) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or buf is None:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def omni_chat(
    user_text: str,
    *,
    attach_camera: bool = True,
    stream: bool = True,
    want_model_audio: bool = False,
) -> Tuple[str, Optional[bytes]]:
    """
    调用 Qwen-Omni（默认 qwen-omni-turbo-latest，可改 DASHSCOPE_OMNI_MODEL）。
    返回 (文本, 可选整段 WAV 字节)。模型语音常为 24kHz，下发 ESP 前需 ffmpeg 转 16k。
    """
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Qwen-Omni：未设置 DASHSCOPE_API_KEY")
        return "", None
    try:
        from openai import OpenAI
    except ImportError:
        print("Qwen-Omni：请 pip install 'openai>=1.52.0'")
        return "", None

    api_key = os.environ["DASHSCOPE_API_KEY"]
    base_url = config.DASHSCOPE_COMPAT_BASE_URL
    model = config.DASHSCOPE_OMNI_MODEL

    prompt = _maybe_prepend_yolo_for_navigation_prompt(user_text)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if attach_camera:
        frame = state.copy_latest_frame()
        if frame is not None:
            url = _jpeg_data_url(frame)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_msg = (
        "你是智能助盲眼镜的中文语音助手。用户输入来自语音识别，可能附带一张用户前方摄像头画面。"
        "回答简短、口语化；不要编造画面里明显不存在的东西。"
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
        ],
        "stream": stream,
    }
    if want_model_audio:
        kwargs["modalities"] = ["text", "audio"]
        kwargs["audio"] = {"voice": config.OMNI_TTS_VOICE, "format": "wav"}

    text_parts: list[str] = []
    audio_b64_parts: list[str] = []

    try:
        if stream:
            stream_resp = client.chat.completions.create(**kwargs)
            for chunk in stream_resp:
                if not chunk.choices:
                    continue
                ch0 = chunk.choices[0]
                delta = ch0.delta
                if getattr(delta, "content", None):
                    text_parts.append(str(delta.content))
                if want_model_audio and getattr(delta, "audio", None):
                    ad = delta.audio
                    if isinstance(ad, dict) and ad.get("data"):
                        audio_b64_parts.append(str(ad["data"]))
        else:
            resp = client.chat.completions.create(**{**kwargs, "stream": False})
            if resp.choices:
                msg = resp.choices[0].message
                if getattr(msg, "content", None):
                    text_parts.append(str(msg.content))
    except Exception as e:
        print("Qwen-Omni 调用失败:", e)
        return "", None

    full_text = "".join(text_parts).strip()
    raw_audio: Optional[bytes] = None
    if audio_b64_parts:
        try:
            raw_audio = base64.b64decode("".join(audio_b64_parts))
        except Exception:
            raw_audio = None

    return full_text, raw_audio
