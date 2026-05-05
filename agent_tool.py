"""LangChain 工具：YOLO、时间、天气、MQTT 文本发布。"""

from datetime import datetime

import requests
from langchain_core.tools import tool

import config
import state


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
    """获取当前服务器时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(description="获取城市天气")
def get_weather(city: str) -> str:
    """
    调用 wttr.in 实时天气API，返回温度及天气状况
    参数:
    city：城市名称 如(杭州 / hangzhou)
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
        return f"{city}：天气查询失败（{e}）。请稍后重试，或直接告诉我你想去的景点，我可先按常规天气为你规划。"


@tool
def publish_text_result(text: str) -> str:
    """将文本发布到 agent/reply topic。参数是要发布的文本。"""
    state.publish(config.MQTT_TOPIC_AGENT_REPLY, text)
    return f"已发布: {text}"


# 供 create_agent 注册
AGENT_TOOLS = [
    yolo_detect_current_frame,
    get_server_time,
    get_weather,
    publish_text_result,
]
