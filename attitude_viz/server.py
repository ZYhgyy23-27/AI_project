"""
设备上行四元数（MQTT esp32/attitude）的 Web 可视化：Flask 页面 + WebSocket 广播。

解算在 ESP32 本机完成，此处仅订阅、缓存最后一帧、推送给浏览器。

注意：MQTT on_message 跑在 paho 网络线程里，不可在该线程直接 ws.send（易无声失败）。
通过每连接 queue，仅在 WebSocket 处理线程里 send。
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, Response
from flask_sock import Sock

import config

_PACKAGE_DIR = Path(__file__).resolve().parent
_dashboard_html: Optional[str] = None

_latest_json: Optional[str] = None
_json_lock = threading.Lock()

_client_queues: list[queue.Queue[str]] = []
_client_q_lock = threading.Lock()


def _load_dashboard_html() -> str:
    global _dashboard_html
    if _dashboard_html is None:
        _dashboard_html = (_PACKAGE_DIR / "dashboard.html").read_text(encoding="utf-8")
    return _dashboard_html


def _fanout_to_websocket_threads(text: str) -> None:
    """仅 put 队列；由各浏览器连接的 WS 线程负责 ws.send。"""
    with _client_q_lock:
        targets = list(_client_queues)
    for q in targets:
        try:
            q.put_nowait(text)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(text)
            except queue.Full:
                pass


def handle_mqtt_payload(payload: bytes) -> None:
    """MQTT 收到 esp32/attitude 时由主程序调用。"""
    global _latest_json
    try:
        text = payload.decode("utf-8")
    except Exception:
        text = ""
    if not text:
        return
    with _json_lock:
        _latest_json = text
    _fanout_to_websocket_threads(text)


def register(app: Flask) -> None:
    """挂载 /attitude 与 /ws/attitude，并创建本模块内的 Sock 实例。"""
    sock = Sock(app)

    @app.route("/attitude")
    def attitude_page():
        return Response(_load_dashboard_html(), mimetype="text/html; charset=utf-8")

    @sock.route("/ws/attitude")
    def ws_attitude(ws):
        out_q: queue.Queue[str] = queue.Queue(64)
        with _client_q_lock:
            _client_queues.append(out_q)
        try:
            with _json_lock:
                last = _latest_json
            if last:
                try:
                    ws.send(last)
                except Exception:
                    return
            while True:
                try:
                    text = out_q.get(timeout=1.0)
                except queue.Empty:
                    try:
                        ws.receive(timeout=0.05)
                    except TimeoutError:
                        continue
                    except Exception:
                        break
                    continue
                try:
                    ws.send(text)
                except Exception:
                    break
        finally:
            with _client_q_lock:
                try:
                    _client_queues.remove(out_q)
                except ValueError:
                    pass


def mqtt_subscribe_topic() -> str:
    """与 config 一致，供主程序 on_connect 订阅。"""
    return config.MQTT_TOPIC_ATTITUDE
