"""会话状态机：空闲 / 闲聊 / 导航，及 ASR 文本路由。"""

import re
from datetime import datetime
from enum import Enum
from typing import Callable

import config
import state


class SessionState(str, Enum):
    """语音会话状态机：空闲 → 闲聊 / 导航；可经关键词退回空闲。"""

    IDLE = "idle"
    CHAT = "chat"
    NAVIGATION = "navigation"


session_state: SessionState = SessionState.IDLE

_NAV_ON_KEYWORDS = (
    "开启导航",
    "导航模式",
    "开始导航",
    "打开导航",
    "进入导航",
    "启动导航",
    "开启导盲",
    "导盲模式",
    "开始导盲",
    "打开导盲",
    "进入导盲",
    "启动导盲",
)
_NAV_OFF_KEYWORDS = (
    "关闭导航",
    "退出导航",
    "结束导航",
    "停止导航",
    "关闭导盲",
    "退出导盲",
    "结束导盲",
    "停止导盲",
)

_CHAT_INTENT_KEYWORDS = (
    "天气",
    "气温",
    "下雨",
    "晴天",
    "多云",
    "阴天",
    "刮风",
    "雾霾",
    "时间",
    "几点",
    "日期",
    "星期",
    "几号",
    "现在几点",
    "礼拜",
)
_CHAT_OFF_KEYWORDS = (
    "退出闲聊",
    "结束闲聊",
    "关闭闲聊",
    "不聊了",
    "返回",
    "回到空闲",
    "退出聊天",
)

_QUICK_ACK_PHRASES = frozenset(
    {
        "好了",
        "好的",
        "好",
        "嗯好",
        "好吧",
        "行",
        "行的",
        "可以",
        "可以了",
        "谢谢",
        "谢谢你",
        "多谢",
        "再见",
        "没事了",
        "没事",
    }
)

_CJK_INTER_SPACE = re.compile(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])")
_FILLER_CHARS = frozenset("嗯啊哦呃诶喂唔呀哈吧呢嘛哼噢哟哒咯呐")


def normalize_asr_text(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    prev = None
    while prev != t:
        prev = t
        t = _CJK_INTER_SPACE.sub(r"\1\2", t)
    return t


# 常见同音/近音误识（可按实际设备日志继续加）
_ASR_DOMAIN_FIXES = (
    ("到忙", "导盲"),
    ("倒盲", "导盲"),
    ("岛盲", "导盲"),
    ("岛忙", "导盲"),
    ("到盲", "导盲"),
    ("档盲", "导盲"),
    ("导航摸式", "导航模式"),
    ("导航莫式", "导航模式"),
)


def apply_asr_domain_fixes(text: str) -> str:
    """在合并空格后做领域词纠错，减轻 Paraformer 专有词误识。"""
    t = text
    for wrong, right in _ASR_DOMAIN_FIXES:
        if wrong in t:
            t = t.replace(wrong, right)
    return t


def is_low_value_asr(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    compact = t.replace(" ", "").replace("　", "")
    if len(compact) < 2:
        return True
    if len(compact) <= 8 and all(ch in _FILLER_CHARS for ch in compact):
        return True
    return False


def publish_session_state() -> None:
    state.publish(config.MQTT_TOPIC_SESSION, session_state.value)


def _asr_requests_navigation_off(text: str) -> bool:
    return any(k in text for k in _NAV_OFF_KEYWORDS)


def _asr_requests_navigation_on(text: str) -> bool:
    return any(k in text for k in _NAV_ON_KEYWORDS)


def _asr_chat_intent(text: str) -> bool:
    return any(k in text for k in _CHAT_INTENT_KEYWORDS)


def _asr_chat_off(text: str) -> bool:
    return any(k in text for k in _CHAT_OFF_KEYWORDS)


def _chat_has_weather_intent(text: str) -> bool:
    return any(
        m in text
        for m in (
            "天气",
            "气温",
            "下雨",
            "下雪",
            "刮风",
            "多云",
            "阴天",
            "晴天",
            "雾霾",
            "冷不冷",
            "热不热",
        )
    )


def _chat_time_only_intent(text: str) -> bool:
    if _chat_has_weather_intent(text):
        return False
    return any(
        m in text
        for m in ("几点", "时间", "日期", "星期", "几号", "礼拜", "哪天", "几月", "多少号")
    )


def _is_quick_ack(text: str) -> bool:
    t = text.strip().replace(" ", "").replace("　", "")
    if len(t) > 8:
        return False
    return t in _QUICK_ACK_PHRASES


def _fast_time_reply() -> str:
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。"


def _build_chat_entry_prompt(user_asr_text: str) -> str:
    return (
        f"用户说：{user_asr_text}\n\n"
        "【闲聊模式】若问当前时间或日期，请调用 get_server_time。"
        "若问天气，可调用 get_weather(city)；无城市时可先询问用户所在城市。"
        "回复简短自然。"
    )


def _build_chat_continue_prompt(user_asr_text: str) -> str:
    return (
        f"用户说：{user_asr_text}\n\n"
        "【仍在闲聊模式】时间用 get_server_time；天气用 get_weather。"
        "回复简短自然。"
    )


def _build_navigation_mode_prompt(user_asr_text: str) -> str:
    return (
        f"用户刚才说：{user_asr_text}\n\n"
        "【导航模式已开启】你必须先调用工具 yolo_detect_current_frame，"
        "根据返回的物体名称，用一两句简洁中文描述用户前方视野，便于避障与行走。"
        "不要编造画面中不存在的物体。"
    )


def _build_navigation_continue_prompt(user_asr_text: str) -> str:
    return (
        f"用户说：{user_asr_text}\n\n"
        "【当前处于导盲/导航模式】请先调用工具 yolo_detect_current_frame，"
        "再根据识别结果用简洁中文描述前方环境与避障提示。"
        "不要编造画面中不存在的物体。"
    )


def dispatch_asr_text(
    text: str,
    invoke_agent: Callable[[str], str],
) -> str:
    """
    根据当前会话状态处理一句 ASR 文本，返回要播报的字符串。
    会原地更新 session_state 并发布 MQTT。
    """
    global session_state

    if _asr_requests_navigation_off(text):
        if session_state == SessionState.NAVIGATION:
            session_state = SessionState.IDLE
            state.publish(config.MQTT_TOPIC_NAV, "0")
            publish_session_state()
            return "好的，已退出导盲模式。有需要再叫我。"
        return "当前未在导盲模式。需要的话可以说开启导盲。"

    if _asr_chat_off(text):
        if session_state == SessionState.CHAT:
            session_state = SessionState.IDLE
            publish_session_state()
            return "好的，已退出闲聊模式。"
        if session_state == SessionState.NAVIGATION:
            return "当前是导盲模式，未在闲聊。要结束导盲请说退出导航。"
        return "当前不在闲聊模式。"

    if _asr_requests_navigation_on(text):
        entering_nav = session_state != SessionState.NAVIGATION
        session_state = SessionState.NAVIGATION
        if entering_nav:
            state.publish(config.MQTT_TOPIC_NAV, "1")
            publish_session_state()
            return invoke_agent(_build_navigation_mode_prompt(text))
        return invoke_agent(_build_navigation_continue_prompt(text))

    if session_state != SessionState.NAVIGATION and _asr_chat_intent(text):
        was_chat = session_state == SessionState.CHAT
        session_state = SessionState.CHAT
        publish_session_state()
        if _chat_time_only_intent(text):
            return _fast_time_reply()
        if was_chat:
            return invoke_agent(_build_chat_continue_prompt(text))
        return invoke_agent(_build_chat_entry_prompt(text))

    if session_state == SessionState.NAVIGATION:
        return invoke_agent(_build_navigation_continue_prompt(text))

    if session_state == SessionState.CHAT:
        if _is_quick_ack(text):
            return "不客气，还需要帮忙可以说。"
        if _chat_time_only_intent(text):
            return _fast_time_reply()
        return invoke_agent(_build_chat_continue_prompt(text))

    if _is_quick_ack(text):
        return "嗯，我在。可以说开启导盲、问时间，或问天气。"

    return invoke_agent(text)
