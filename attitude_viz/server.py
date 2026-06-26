"""
设备上行四元数（MQTT esp32/attitude）的 Web 可视化：Flask 页面 + WebSocket 广播。

解算在 ESP32 本机完成，此处仅订阅、缓存最后一帧、推送给浏览器。

同源提供 /attitude/vendor/three.min.js（便于微信小程序 web-view 仅白名单单一域名）。

可选：环境变量 WECHAT_MP_VERIFY_FILE_PATH 指向 MP_verify_*.txt，在站点根路径提供该校验文件。

注意：MQTT on_message 跑在 paho 网络线程里，不可在该线程直接 ws.send（易无声失败）。
通过每连接 queue，仅在 WebSocket 处理线程里 send。
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, Response, send_from_directory
from flask_sock import Sock

import config

_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PACKAGE_DIR / "static" / "vendor"
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


def _register_wechat_mp_verify(app: Flask) -> None:
    """微信公众平台业务域名校验：根路径返回 MP_verify_*.txt 纯文本。"""
    raw = getattr(config, "WECHAT_MP_VERIFY_FILE_PATH", None)
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_file():
        return
    name = path.name
    if not (name.startswith("MP_verify_") and name.endswith(".txt")):
        return
    try:
        body = path.read_text(encoding="utf-8")
    except OSError:
        return

    @app.route(f"/{name}", methods=["GET"])
    def wechat_mp_domain_verify() -> Response:
        return Response(body, mimetype="text/plain; charset=utf-8")


def register(app: Flask) -> None:
    """挂载 /attitude、同源 three.min.js、/ws/attitude；可选微信业务域名校验文件。"""
    sock = Sock(app)
    _register_wechat_mp_verify(app)

    @app.route("/attitude/vendor/three.min.js")
    def attitude_three_vendor():
        return send_from_directory(
            _VENDOR_DIR,
            "three.min.js",
            mimetype="application/javascript; charset=utf-8",
        )

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
