"""
智能指令路由与上下文过滤（在 navigation_master 之前/包裹 LLM 调用）。

- 导盲模式下拦截与前方视觉无关的闲聊/天气类话，提示先退出导航。
- 空闲态下对「查找/搜索」类说法给 LLM 加前缀，便于模型走检索式回答。
"""

from __future__ import annotations

from typing import Callable, Optional

import navigation_master


def _is_visual_or_nav_followup(text: str) -> bool:
    """与避障、路况、朝向相关的短句，导盲模式下放行。"""
    keys = (
        "前面",
        "眼前",
        "看见",
        "看到",
        "障碍",
        "台阶",
        "楼梯",
        "盲道",
        "红绿灯",
        "怎么走",
        "避让",
        "路况",
        "有车",
        "行人",
        "转弯",
        "停",
        "往哪",
        "左边",
        "右边",
        "离",
        "多远",
        "注意",
        "避开",
        "小心",
    )
    return any(k in text for k in keys)


def _is_search_intent(text: str) -> bool:
    keys = ("查找", "搜索", "搜一下", "附近有", "在哪买", "在哪吃", "哪家", "推荐", "帮我找")
    return any(k in text for k in keys)


def _maybe_block_irrelevant_in_navigation(text: str) -> Optional[str]:
    """在导航态下过滤与导盲无关的闲聊意图；返回固定话术或 None 表示放行。"""
    if navigation_master.session_state != navigation_master.SessionState.NAVIGATION:
        return None
    if navigation_master._asr_requests_navigation_off(text):
        return None
    if navigation_master._asr_chat_intent(text) and not _is_visual_or_nav_followup(text):
        return "当前是导盲模式。想聊天气或别的话题，请先说一句：退出导航。"
    return None


def dispatch_asr_text(text: str, invoke_agent: Callable[[str], str]) -> str:
    """
    先做上下文过滤与指令类型标注，再交给 navigation_master.dispatch_asr_text。
    invoke_agent 通常为 Tongyi Agent 或 Qwen-Omni 统一入口。
    """
    blocked = _maybe_block_irrelevant_in_navigation(text)
    if blocked is not None:
        return blocked

    def invoke_wrapped(prompt: str) -> str:
        if navigation_master.session_state in (
            navigation_master.SessionState.IDLE,
            navigation_master.SessionState.CHAT,
        ) and _is_search_intent(text):
            prompt = (
                "【用户意图：查找/检索/周边信息】请简洁中文回答；缺条件时可反问一句。\n" + prompt
            )
        return invoke_agent(prompt)

    return navigation_master.dispatch_asr_text(text, invoke_wrapped)
