"""LangChain 工具：YOLO、时间、天气、MQTT 文本发布、网络检索。"""

from datetime import datetime

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

import config
import navigation_master
import state


def _chat_only_network_denied() -> str | None:
    """若非闲聊态，返回拒绝说明（供网络检索类工具使用）。"""
    if navigation_master.session_state == navigation_master.SessionState.CHAT:
        return None
    return (
        "当前不在闲聊模式，不能使用网络检索。"
        "请先说到天气、时间等进入闲聊，或先说退出导航/退出过马路后再查资料。"
    )


@tool
def yolo_detect_current_frame() -> str:
    """识别当前最新相机画面中的目标，返回逗号分隔标签。"""
    frame = state.copy_latest_frame()
    if frame is None:
        return "当前没有可用相机画面"

    model = state.get_yolo_model()
    results = model.predict(frame, save=False, verbose=False)

    objects = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            objects.append(model.names[cls_id])
    objects = list(dict.fromkeys(objects))
    result_str = ",".join(objects) if objects else "无目标"
    state.publish(config.MQTT_TOPIC_RESULT, result_str if result_str != "无目标" else "")
    return f"YOLO识别结果: {result_str}"


@tool
def get_server_time() -> str:
    """获取当前服务器时间（年-月-日 时:分:秒）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """
    调用 wttr.in 实时天气 API，返回温度及天气状况。
    city：城市名称，如 杭州 / hangzhou。
    """
    try:
        url = f"https://wttr.in/{city}"
        params = {"format": "j1"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        current = (data.get("current_condition") or [{}])[0]
        temp_c = current.get("temp_C")
        desc = ((current.get("weatherDesc") or [{}])[0]).get("value")
        if temp_c is None and not desc:
            return f"{city}：天气接口暂无有效数据，请稍后重试。"
        return f"{city}：{temp_c}°C，{desc}"
    except Exception as e:
        return (
            f"{city}：天气查询失败（{e}）。请稍后重试，"
            "或直接告诉我你想去的景点，我可先按常规天气为你规划。"
        )


@tool
def publish_text_result(text: str) -> str:
    """将文本发布到 MQTT 主题 agent/reply（与语音播报链路一致）。参数为要发布的完整回复文本。"""
    state.publish(config.MQTT_TOPIC_AGENT_REPLY, text)
    return f"已发布: {text}"


def _search_duckduckgo_impl(query: str) -> str:
    try:
        search = DuckDuckGoSearchRun(timeout=10)
        result = search.run(query)
        return f"DuckDuckGo搜索结果：\n{result}"
    except Exception:
        pass

    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("a", class_="result__snippet")
        if results:
            snippets = [r.get_text(strip=True) for r in results[:5]]
            return "DuckDuckGo搜索结果：\n" + "\n".join(snippets)
        return f"DuckDuckGo未找到关于'{query}'的结果"
    except Exception as e:
        return f"DuckDuckGo搜索失败（可能是网络原因）: {str(e)}"


@tool
def search_duckduckgo(query: str) -> str:
    """仅在闲聊模式下可用。使用 DuckDuckGo 搜索网页，返回若干条结果摘要。"""
    denied = _chat_only_network_denied()
    if denied:
        return denied
    return _search_duckduckgo_impl(query)


def _search_wikipedia_impl(query: str) -> str:
    try:
        wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=1,
                doc_content_chars_max=2000,
                language="zh",
            )
        )
        result = wikipedia.run(query)
        if result:
            return f"维基百科结果：\n{result}"
    except Exception:
        pass

    try:
        wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=1,
                doc_content_chars_max=2000,
                language="en",
            )
        )
        result = wikipedia.run(query)
        if result:
            return f"维基百科结果（英文）：\n{result}"
    except Exception as e:
        return f"维基百科搜索失败: {str(e)}"

    return f"维基百科中未找到关于'{query}'的信息"


@tool
def search_wikipedia(query: str) -> str:
    """仅在闲聊模式下可用。在维基百科中搜索信息，优先中文条目，失败则尝试英文。"""
    denied = _chat_only_network_denied()
    if denied:
        return denied
    return _search_wikipedia_impl(query)


def _load_webpage_impl(url: str, max_length: int = 3000) -> str:
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs:
            content = docs[0].page_content[:max_length]
            return f"网页内容（来源: {url}）：\n{content}..."
        return f"未能从 {url} 加载到内容"
    except Exception:
        pass

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return f"网页内容（来源: {url}）：\n{text[:max_length]}..."
    except Exception as e:
        return f"网页加载失败: {str(e)}"


@tool
def load_webpage(url: str, max_length: int = 3000) -> str:
    """仅在闲聊模式下可用。加载指定网页 URL 的正文内容（去掉脚本/导航等）。"""
    denied = _chat_only_network_denied()
    if denied:
        return denied
    return _load_webpage_impl(url, max_length=max_length)


# 供 agent.py 中 create_agent 注册
AGENT_TOOLS = [
    yolo_detect_current_frame,
    get_server_time,
    get_weather,
    publish_text_result,
    search_duckduckgo,
    search_wikipedia,
    load_webpage,
]
