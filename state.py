"""运行时状态：相机帧、MQTT、YOLO 单例。"""

import threading
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config

latest_frame = None
_frame_lock = threading.Lock()

# 设备上行最新姿态 JSON 字符串（与 esp32/attitude 负载一致），供 WebSocket 新连接先发一帧
_latest_attitude_json: Optional[str] = None
_attitude_lock = threading.Lock()

_mqtt_client = None
_mqtt_lock = threading.Lock()

_model = None
_model_lock = threading.Lock()


def get_mqtt_client():
    with _mqtt_lock:
        return _mqtt_client


def set_mqtt_client(client) -> None:
    global _mqtt_client
    with _mqtt_lock:
        _mqtt_client = client


def publish(topic: str, payload) -> None:
    client = get_mqtt_client()
    if client is None:
        return
    client.publish(topic, payload)


def get_yolo_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = YOLO(config.MODEL_PATH)
        return _model


def copy_latest_frame():
    with _frame_lock:
        return None if latest_frame is None else latest_frame.copy()


def set_latest_attitude_json(text: str) -> None:
    global _latest_attitude_json
    with _attitude_lock:
        _latest_attitude_json = text


def get_latest_attitude_json() -> Optional[str]:
    with _attitude_lock:
        return _latest_attitude_json


def handle_camera(payload: bytes) -> None:
    global latest_frame
    try:
        img_np = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            print("无法解码图片")
            return
        with _frame_lock:
            latest_frame = img.copy()
    except Exception as e:
        print("相机帧处理异常:", e)
