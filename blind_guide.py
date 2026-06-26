"""导盲辅助：盲道分割、LK 稳定、方向/转弯/障碍物 cue。"""
from __future__ import annotations

import json, math, threading, time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import config, state

CUE_GO_STRAIGHT = "blind_go_straight"
CUE_ADJUST_LEFT = "blind_adjust_left"
CUE_ADJUST_RIGHT = "blind_adjust_right"
CUE_PATH_LOST = "blind_path_lost"
CUE_TURN_LEFT = "blind_turn_left"
CUE_TURN_RIGHT = "blind_turn_right"
CUE_OBSTACLE_STOP = "blind_obstacle_stop"
CUE_OBSTACLE_LEFT = "blind_obstacle_left"
CUE_OBSTACLE_RIGHT = "blind_obstacle_right"


@dataclass
class BlindGuideConfig:
    path_conf: float = 0.25
    path_roi_top_frac: float = 0.42
    min_mask_area_frac: float = 0.006
    center_deadzone: float = 0.12
    turn_dx_threshold: float = 0.22
    turn_angle_threshold_deg: float = 28.0
    go_straight_repeat_sec: float = 8.0
    guidance_repeat_sec: float = 2.2
    guidance_edge_cooldown_sec: float = 0.55
    path_lost_cooldown_sec: float = 3.5
    turn_cooldown_sec: float = 4.0
    obstacle_conf: float = 0.35
    obstacle_cooldown_sec: float = 2.5
    lk_enabled: bool = True
    lk_blend_prev: float = 0.28


@dataclass
class _Runtime:
    last_cue_time: dict[str, float] = field(default_factory=dict)
    last_guidance_key: Optional[str] = None
    prev_gray: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None


_lock = threading.Lock()
_rt = _Runtime()


def reset_state() -> None:
    with _lock:
        _rt.last_cue_time.clear(); _rt.last_guidance_key = None; _rt.prev_gray = None; _rt.prev_mask = None


def _cooldown(key: str, sec: float) -> bool:
    now = time.monotonic()
    with _lock:
        last = _rt.last_cue_time.get(key, 0.0)
        if now - last >= sec:
            _rt.last_cue_time[key] = now
            return True
    return False


def _norm_names(names) -> dict[int, str]:
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in (names or {}).items()}


def _parse_ids(s: str | None) -> Optional[list[int]]:
    if not s or not str(s).strip():
        return None
    out = []
    for p in str(s).split(","):
        try: out.append(int(p.strip()))
        except ValueError: pass
    return out or None


def _path_ids(names: dict[int, str]) -> list[int]:
    ids = _parse_ids(config.NAVIGATION_BLIND_PATH_CLASS_IDS)
    if ids is not None:
        return ids
    subs = [s.strip().lower() for s in config.NAVIGATION_BLIND_PATH_NAME_SUBSTRINGS.split(",") if s.strip()]
    return [cid for cid, n in names.items() if any(s in n.lower() for s in subs)]


def _merge_masks(r, ids: list[int], h: int, w: int, conf_min: float) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    if r is None or not ids or r.boxes is None or len(r.boxes) == 0 or r.masks is None:
        return out
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones_like(cls, dtype=float)
    keep = set(ids)
    if getattr(r.masks, "xy", None) is not None:
        for poly, c, cf in zip(r.masks.xy, cls.tolist(), conf.tolist()):
            if c in keep and cf >= conf_min:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) >= 3: cv2.fillPoly(out, [pts], 255)
        if out.any(): return out
    data = getattr(r.masks, "data", None)
    if data is None: return out
    for i, m in enumerate(data.cpu().numpy()):
        if int(cls[i]) in keep and float(conf[i]) >= conf_min:
            full = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            out = np.maximum(out, (full > 0.5).astype(np.uint8) * 255)
    return out


def _clean(mask: np.ndarray, cfg: BlindGuideConfig) -> np.ndarray:
    h, w = mask.shape[:2]
    roi = np.zeros_like(mask); roi[int(cfg.path_roi_top_frac * h):, :] = mask[int(cfg.path_roi_top_frac * h):, :]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi = cv2.morphologyEx(cv2.morphologyEx(roi, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 127).astype(np.uint8), 8)
    if n <= 1: return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[best, cv2.CC_STAT_AREA]) < int(h * w * cfg.min_mask_area_frac): return np.zeros_like(mask)
    return (labels == best).astype(np.uint8) * 255


def _stabilize(gray: np.ndarray, mask: np.ndarray, cfg: BlindGuideConfig) -> np.ndarray:
    if not cfg.lk_enabled: return mask
    with _lock:
        pg = None if _rt.prev_gray is None else _rt.prev_gray.copy(); pm = None if _rt.prev_mask is None else _rt.prev_mask.copy()
    if pg is None or pm is None or not pm.any(): return mask
    try:
        p0 = cv2.goodFeaturesToTrack(pg, 120, 0.01, 8, mask=pm)
        if p0 is None or len(p0) < 8: return mask
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, gray, p0, None)
        if p1 is None or st is None: return mask
        g0 = p0[st.reshape(-1) == 1].reshape(-1, 2); g1 = p1[st.reshape(-1) == 1].reshape(-1, 2)
        if len(g0) < 8: return mask
        mat, _ = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if mat is None: return mask
        warped = cv2.warpAffine(pm, mat, (gray.shape[1], gray.shape[0]))
        a = max(0.0, min(0.8, cfg.lk_blend_prev))
        return (cv2.addWeighted(mask, 1.0 - a, warped, a, 0) > 127).astype(np.uint8) * 255
    except Exception:
        return mask


def _row_center(mask: np.ndarray, y1: int, y2: int) -> Optional[float]:
    h, w = mask.shape[:2]; _, xs = np.where(mask[max(0, y1):min(h, y2), :] > 127)
    return None if xs.size < 30 else float(np.median(xs)) / max(1.0, float(w))


def _angle(mask: np.ndarray) -> Optional[float]:
    ys, xs = np.where(mask > 127)
    if ys.size < 120: return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    deg = math.degrees(math.atan2(float(vt[0, 1]), float(vt[0, 0])))
    return deg + 180 if deg < -90 else deg - 180 if deg > 90 else deg


def _analyze(mask: np.ndarray) -> dict:
    h, _ = mask.shape[:2]
    b = _row_center(mask, int(h * 0.78), int(h * 0.96)); m = _row_center(mask, int(h * 0.58), int(h * 0.74)); t = _row_center(mask, int(h * 0.42), int(h * 0.58))
    c = b if b is not None else m
    return {"center": c, "offset": None if c is None else c - 0.5, "turn_dx": None if t is None or b is None else t - b, "angle_deg": _angle(mask), "coverage": float(np.count_nonzero(mask)) / max(1.0, float(mask.size))}


def _is_obstacle(name: str) -> bool:
    low = name.lower(); keys = [s.strip().lower() for s in config.NAVIGATION_OBSTACLE_NAME_SUBSTRINGS.split(",") if s.strip()]
    return any(k in low for k in keys)


def _detect_obstacle(r, names: dict[int, str], h: int, w: int, cfg: BlindGuideConfig) -> Optional[dict]:
    if r is None or r.boxes is None or len(r.boxes) == 0: return None
    danger = (0.32 * w, 0.50 * h, 0.68 * w, float(h)); da = max(1.0, (danger[2] - danger[0]) * (danger[3] - danger[1]))
    best, score = None, 0.0
    for box in r.boxes:
        cf = float(box.conf[0]) if box.conf is not None else 0.5
        if cf < cfg.obstacle_conf: continue
        cid = int(box.cls[0]); name = names.get(cid, str(cid))
        if not _is_obstacle(name): continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().numpy()]
        ix = max(0.0, min(x2, danger[2]) - max(x1, danger[0])); iy = max(0.0, min(y2, danger[3]) - max(y1, danger[1]))
        overlap = ix * iy / da; area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(h * w))
        if overlap < 0.08 and area < 0.08: continue
        cx = 0.5 * (x1 + x2) / max(1.0, float(w)); action = "stop" if area >= 0.08 else ("avoid_right" if cx < 0.5 else "avoid_left")
        sc = cf + overlap * 3 + area
        if sc > score:
            score = sc; best = {"class": name, "conf": round(cf, 3), "cx": round(cx, 3), "overlap": round(overlap, 3), "area_frac": round(area, 3), "action": action}
    return best


def _path_cue(path: dict, cfg: BlindGuideConfig) -> tuple[str, str]:
    off, dx, ang = path.get("offset"), path.get("turn_dx"), path.get("angle_deg")
    if off is None: return CUE_PATH_LOST, "lost"
    if dx is not None and abs(float(dx)) >= cfg.turn_dx_threshold: return (CUE_TURN_RIGHT if float(dx) > 0 else CUE_TURN_LEFT), "turn"
    if ang is not None and abs(float(ang)) >= cfg.turn_angle_threshold_deg: return (CUE_TURN_RIGHT if float(ang) > 0 else CUE_TURN_LEFT), "turn"
    if abs(float(off)) <= cfg.center_deadzone: return CUE_GO_STRAIGHT, "straight"
    return (CUE_ADJUST_RIGHT if float(off) > 0 else CUE_ADJUST_LEFT), "adjust"


def _play(cue: str, kind: str, cfg: BlindGuideConfig, play_cue: Callable[[str], None]) -> None:
    if kind == "obstacle":
        if _cooldown(f"obs_{cue}", cfg.obstacle_cooldown_sec): play_cue(cue)
        return
    if kind == "lost":
        if _cooldown("lost", cfg.path_lost_cooldown_sec): play_cue(cue)
        return
    if kind == "turn":
        if _cooldown(f"turn_{cue}", cfg.turn_cooldown_sec): play_cue(cue)
        return
    with _lock: prev = _rt.last_guidance_key
    if prev != cue:
        if _cooldown("guide_edge", cfg.guidance_edge_cooldown_sec):
            play_cue(cue)
            with _lock: _rt.last_guidance_key = cue
        return
    repeat = cfg.go_straight_repeat_sec if cue == CUE_GO_STRAIGHT else cfg.guidance_repeat_sec
    if _cooldown(f"guide_{cue}", repeat): play_cue(cue)


def _status(payload: dict) -> None:
    try: state.publish(config.MQTT_TOPIC_NAVIGATION_STATUS, json.dumps({**payload, "t_ms": int(time.time() * 1000)}, ensure_ascii=False))
    except Exception: pass


def tick_blind_frame(frame_bgr: np.ndarray, path_model, obstacle_model, cfg: BlindGuideConfig, play_cue: Callable[[str], None]) -> None:
    if frame_bgr is None or frame_bgr.size == 0: return
    h, w = frame_bgr.shape[:2]; gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    names = _norm_names(getattr(path_model, "names", {}) or {}); ids = _path_ids(names)
    r = path_model.predict(frame_bgr, save=False, verbose=False, imgsz=640, conf=cfg.path_conf)
    mask = _clean(_merge_masks(r[0] if r else None, ids, h, w, cfg.path_conf), cfg)
    mask = _clean(_stabilize(gray, mask, cfg), cfg)
    visible = bool(mask.any()); path = _analyze(mask) if visible else {}
    obstacle = None
    if obstacle_model is not None:
        onames = _norm_names(getattr(obstacle_model, "names", {}) or {})
        ro = obstacle_model.predict(frame_bgr, save=False, verbose=False, imgsz=640, conf=cfg.obstacle_conf)
        obstacle = _detect_obstacle(ro[0] if ro else None, onames, h, w, cfg)
    cue, kind = (CUE_PATH_LOST, "lost") if not visible else _path_cue(path, cfg)
    if obstacle is not None:
        a = obstacle.get("action"); cue = CUE_OBSTACLE_STOP if a == "stop" else (CUE_OBSTACLE_LEFT if a == "avoid_right" else CUE_OBSTACLE_RIGHT); kind = "obstacle"
    _status({"path_visible": visible, "path_class_ids": ids, "path_center": None if path.get("center") is None else round(float(path["center"]), 4), "path_offset": None if path.get("offset") is None else round(float(path["offset"]), 4), "turn_dx": None if path.get("turn_dx") is None else round(float(path["turn_dx"]), 4), "angle_deg": None if path.get("angle_deg") is None else round(float(path["angle_deg"]), 2), "coverage": None if path.get("coverage") is None else round(float(path["coverage"]), 4), "obstacle": obstacle, "cue": cue})
    _play(cue, kind, cfg, play_cue)
    with _lock:
        _rt.prev_gray = gray.copy(); _rt.prev_mask = mask.copy()
