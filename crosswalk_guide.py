"""
过马路辅助：斑马线（主分割模型或 traffic 模型）+ 红绿灯（trafficlight.pt）+ 对齐语音 + MQTT 状态。

预置 WAV 放在 CROSSWALK_AUDIO_DIR（默认与导视共用 /root/guide_audio），文件名：
  crosswalk_align_left / crosswalk_align_right / crosswalk_aligned
  red_light_wait / yellow_light_wait / green_light_go
  no_crosswalk（可选）
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

import cv2
import numpy as np

import config
import state

CueId = str

CUE_ALIGN_LEFT = "crosswalk_align_left"
CUE_ALIGN_RIGHT = "crosswalk_align_right"
CUE_ALIGNED = "crosswalk_aligned"
CUE_RED = "red_light_wait"
CUE_YELLOW = "yellow_light_wait"
CUE_GREEN = "green_light_go"
CUE_NO_CROSSWALK = "no_crosswalk"


@dataclass
class CrosswalkConfig:
    lateral_dead_zone: float = 0.09
    lateral_soft_zone: float = 0.20
    align_edge_cooldown_sec: float = 0.5
    align_repeat_sec: float = 2.5
    light_stable_frames: int = 4
    light_vote_window: int = 7
    red_cooldown_sec: float = 4.0
    yellow_cooldown_sec: float = 4.0
    green_cooldown_sec: float = 5.0
    no_crosswalk_cooldown_sec: float = 5.0
    traffic_roi_max_y_frac: float = 0.78  # 灯通常在画面上方，忽略更靠下的框


@dataclass
class _Runtime:
    last_cue_time: dict[str, float] = field(default_factory=dict)
    last_align_key: Optional[str] = None
    light_votes: Deque[str] = field(default_factory=lambda: deque(maxlen=16))
    last_light_announced: Optional[str] = None


_rt_lock = threading.Lock()
_rt = _Runtime()


def reset_state() -> None:
    with _rt_lock:
        _rt.last_cue_time.clear()
        _rt.last_align_key = None
        _rt.light_votes.clear()
        _rt.last_light_announced = None


def _now() -> float:
    return time.monotonic()


def _parse_id_list(s: str | None) -> Optional[list[int]]:
    if not s or not str(s).strip():
        return None
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or None


def _norm_names(names) -> dict[int, str]:
    if not names:
        return {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _match_ids(names: dict[int, str], substrings: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for cid, raw in names.items():
        low = raw.lower()
        for s in substrings:
            if s.lower() in low:
                out.append(int(cid))
                break
    return sorted(set(out))


def _resolve_ids(
    names: dict[int, str],
    env_ids: str | None,
    subs: tuple[str, ...],
) -> list[int]:
    parsed = _parse_id_list(env_ids)
    if parsed is not None:
        return parsed
    return _match_ids(names, subs)


def _merge_class_masks(r, class_ids: list[int], H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    if not class_ids or r.boxes is None or len(r.boxes) == 0:
        return out
    cset = set(class_ids)
    if r.masks is None:
        return out
    if hasattr(r.masks, "xy") and r.masks.xy is not None:
        for poly, cls_t in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            if cls_t not in cset:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if pts.shape[0] < 3:
                continue
            cv2.fillPoly(out, [pts], 255)
        if out.any():
            return out
    if hasattr(r.masks, "data") and r.masks.data is not None:
        data = r.masks.data.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(data.shape[0]):
            if int(cls_arr[i]) not in cset:
                continue
            m = data[i]
            m_full = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            out = np.maximum(out, (m_full > 0.5).astype(np.uint8) * 255)
    return out


def _zebra_proxy_cx_from_boxes(r, class_ids: list[int], H: int, W: int) -> Optional[float]:
    if r.boxes is None or len(r.boxes) == 0:
        return None
    cset = set(class_ids)
    best_a = 0.0
    best_cx: Optional[float] = None
    for box in r.boxes:
        if int(box.cls[0]) not in cset:
            continue
        xyxy = box.xyxy[0].detach().cpu().numpy()
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        a = max(0.0, (x2 - x1) * (y2 - y1))
        if a > best_a:
            best_a = a
            best_cx = 0.5 * (x1 + x2)
    return best_cx


def _mask_orientation_deg(mask: np.ndarray) -> Optional[float]:
    ys, xs = np.where(mask > 127)
    if ys.size < 120:
        return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean = pts.mean(axis=0)
    c = pts - mean
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    vx, vy = float(vt[0, 0]), float(vt[0, 1])
    return math.degrees(math.atan2(vy, vx))


def _bucket_light_label(names: dict[int, str], cls_id: int) -> Optional[str]:
    label = str(names.get(cls_id, "")).lower()
    if any(k in label for k in ("红", "red", "stop")):
        return "red"
    if any(k in label for k in ("黄", "yellow", "amber", "橙")):
        return "yellow"
    if any(k in label for k in ("绿", "green", "go", "行", "walk")):
        return "green"
    return None


def _pick_traffic_light(
    r,
    names: dict[int, str],
    red_ids: list[int],
    yellow_ids: list[int],
    green_ids: list[int],
    H: int,
    W: int,
    roi_max_y: float,
) -> Optional[str]:
    all_ids = set(red_ids + yellow_ids + green_ids)
    if not all_ids or r.boxes is None or len(r.boxes) == 0:
        return None
    best_score = -1.0
    best_bucket: Optional[str] = None
    y_cut = H * roi_max_y
    for box in r.boxes:
        cid = int(box.cls[0])
        if cid not in all_ids:
            continue
        conf = float(box.conf[0]) if box.conf is not None else 0.5
        xyxy = box.xyxy[0].detach().cpu().numpy()
        y1, y2 = float(xyxy[1]), float(xyxy[3])
        cy = 0.5 * (y1 + y2)
        if cy > y_cut:
            continue
        bucket = _bucket_light_label(names, cid)
        if bucket is None:
            if cid in red_ids:
                bucket = "red"
            elif cid in yellow_ids:
                bucket = "yellow"
            elif cid in green_ids:
                bucket = "green"
        if bucket is None:
            continue
        score = conf * (1.15 - cy / max(float(H), 1.0))
        if score > best_score:
            best_score = score
            best_bucket = bucket
    return best_bucket


def _cooldown(key: str, sec: float) -> bool:
    t = _now()
    with _rt_lock:
        last = _rt.last_cue_time.get(key, 0.0)
        if t - last >= sec:
            _rt.last_cue_time[key] = t
            return True
    return False


def _compute_stable_light(current: str, cfg: CrosswalkConfig) -> Optional[str]:
    with _rt_lock:
        _rt.light_votes.append(current)
        recent = list(_rt.light_votes)[-cfg.light_vote_window :]
    if len(recent) < cfg.light_stable_frames:
        return None
    c = Counter(recent)
    for cand in ("red", "yellow", "green"):
        if c[cand] >= cfg.light_stable_frames:
            return cand
    return None


def _publish_status(
    zebra_visible: bool,
    light: str,
    align_nx: Optional[float],
    orient_deg: Optional[float],
) -> None:
    payload = {
        "zebra_visible": zebra_visible,
        "light": light,
        "align_nx": None if align_nx is None else round(float(align_nx), 4),
        "orient_deg": None if orient_deg is None else round(float(orient_deg), 2),
        "t_ms": int(time.time() * 1000),
    }
    try:
        state.publish(config.MQTT_TOPIC_CROSSWALK_STATUS, json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def tick_crosswalk_frame(
    frame_bgr: np.ndarray,
    seg_model,
    traffic_model,
    cfg: CrosswalkConfig,
    play_cue: Callable[[str], None],
) -> None:
    if frame_bgr is None or frame_bgr.size == 0:
        return
    H, W = frame_bgr.shape[:2]

    use_traffic_zebra = config.CROSSWALK_ZEBRA_USE_TRAFFIC_MODEL

    zebra_names = _norm_names(getattr(traffic_model if use_traffic_zebra else seg_model, "names", {}) or {})
    traffic_names = _norm_names(getattr(traffic_model, "names", {}) or {})

    zebra_ids = _resolve_ids(
        zebra_names,
        config.CROSSWALK_ZEBRA_CLASS_IDS,
        tuple(s.strip() for s in config.CROSSWALK_ZEBRA_NAME_SUBSTRINGS.split(",") if s.strip()),
    )
    red_ids = _resolve_ids(
        traffic_names,
        config.CROSSWALK_RED_CLASS_IDS,
        ("红", "red", "stop"),
    )
    yellow_ids = _resolve_ids(
        traffic_names,
        config.CROSSWALK_YELLOW_CLASS_IDS,
        ("黄", "yellow", "amber", "橙"),
    )
    green_ids = _resolve_ids(
        traffic_names,
        config.CROSSWALK_GREEN_CLASS_IDS,
        ("绿", "green", "go", "walk", "行"),
    )

    r_zebra = None
    if use_traffic_zebra:
        r_tr = traffic_model.predict(frame_bgr, save=False, verbose=False, imgsz=640)
        r_zebra = r_tr[0] if r_tr else None
        r_light = r_zebra
    else:
        r_seg = seg_model.predict(frame_bgr, save=False, verbose=False, imgsz=640)
        r_zebra = r_seg[0] if r_seg else None
        r_li = traffic_model.predict(frame_bgr, save=False, verbose=False, imgsz=640)
        r_light = r_li[0] if r_li else None

    mask = np.zeros((H, W), dtype=np.uint8)
    if r_zebra is not None:
        mask = _merge_class_masks(r_zebra, zebra_ids, H, W)
    zebra_visible = bool(mask.any())
    cx: Optional[float] = None
    align_nx: Optional[float] = None
    orient_deg: Optional[float] = None

    if zebra_visible:
        M = cv2.moments(mask)
        if M["m00"] > 200:
            cx = float(M["m10"] / M["m00"])
        orient_deg = _mask_orientation_deg(mask)
    else:
        if r_zebra is not None:
            cx = _zebra_proxy_cx_from_boxes(r_zebra, zebra_ids, H, W)
        if cx is None:
            if _cooldown("no_crosswalk", cfg.no_crosswalk_cooldown_sec):
                play_cue(CUE_NO_CROSSWALK)

    if cx is not None:
        align_nx = (cx - 0.5 * W) / max(1.0, 0.5 * W)
        dead = cfg.lateral_dead_zone
        steer: Optional[str] = None
        if abs(align_nx) <= dead:
            steer = CUE_ALIGNED
        elif align_nx < -dead:
            steer = CUE_ALIGN_LEFT
        else:
            steer = CUE_ALIGN_RIGHT
        # 强弱偏移共用左右对齐语音；可按需增加 slight 类 WAV
        if steer == CUE_ALIGN_LEFT or steer == CUE_ALIGN_RIGHT:
            with _rt_lock:
                prev = _rt.last_align_key
            changed = prev != steer
            if changed:
                if _cooldown("align_edge", cfg.align_edge_cooldown_sec):
                    play_cue(steer)
                    with _rt_lock:
                        _rt.last_align_key = steer
            elif _cooldown("align_repeat", cfg.align_repeat_sec):
                play_cue(steer)
        else:
            with _rt_lock:
                prev = _rt.last_align_key
            if prev != CUE_ALIGNED:
                if _cooldown("align_ok_edge", 0.45):
                    play_cue(CUE_ALIGNED)
                    with _rt_lock:
                        _rt.last_align_key = CUE_ALIGNED
            elif _cooldown("align_ok_repeat", cfg.align_repeat_sec * 1.4):
                play_cue(CUE_ALIGNED)

    light = "unknown"
    if r_light is not None:
        picked = _pick_traffic_light(
            r_light,
            traffic_names,
            red_ids,
            yellow_ids,
            green_ids,
            H,
            W,
            cfg.traffic_roi_max_y_frac,
        )
        if picked is not None:
            light = picked

    _publish_status(zebra_visible, light, align_nx, orient_deg)

    stable = _compute_stable_light(light, cfg)
    if stable is None:
        return

    cue_map = {"red": CUE_RED, "yellow": CUE_YELLOW, "green": CUE_GREEN}
    cue = cue_map.get(stable)
    if not cue:
        return

    with _rt_lock:
        last_ann = _rt.last_light_announced
    cd = {
        "red": cfg.red_cooldown_sec,
        "yellow": cfg.yellow_cooldown_sec,
        "green": cfg.green_cooldown_sec,
    }[stable]

    if last_ann != stable:
        play_cue(cue)
        with _rt_lock:
            _rt.last_light_announced = stable
        with _rt_lock:
            _rt.last_cue_time[f"light_{stable}"] = _now()
        return

    if _cooldown(f"light_remind_{stable}", cd):
        play_cue(cue)
