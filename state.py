"""运行时状态：相机帧、MQTT、YOLO 单例。"""

import threading

import cv2
import numpy as np
from ultralytics import YOLO

import config

latest_frame = None
_frame_lock = threading.Lock()

_mqtt_client = None
_mqtt_lock = threading.Lock()

_model = None
_model_lock = threading.Lock()

_traffic_model = None
_traffic_model_lock = threading.Lock()

_blind_path_model = None
_blind_path_model_lock = threading.Lock()


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


def get_yolo_traffic_model():
    global _traffic_model
    with _traffic_model_lock:
        if _traffic_model is None:
            _traffic_model = YOLO(config.TRAFFIC_MODEL_PATH)
        return _traffic_model


def get_blind_path_model():
    global _blind_path_model
    with _blind_path_model_lock:
        if _blind_path_model is None:
            _blind_path_model = YOLO(config.BLIND_PATH_MODEL_PATH)
        return _blind_path_model


def copy_latest_frame():
    with _frame_lock:
        return None if latest_frame is None else latest_frame.copy()


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
