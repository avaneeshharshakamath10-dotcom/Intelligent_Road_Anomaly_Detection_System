#!/usr/bin/env python3
"""
Virtual Sensor Fusion ADAS Dashboard
=====================================
Late-fusion perception stack combining a YOLOv8 semantic detector
("road_small.pt") with a MiDaS_small monocular "virtual LiDAR" depth
estimator to produce metric range estimates and a sci-fi cockpit HUD
for road anomaly detection (potholes and speed humps).

Designed to run in real time on a CPU-only laptop (e.g. Intel i5) by
interleaving heavy inference (YOLO + MiDaS) every Nth frame and reusing
cached results for the frames in between.

Requirements
------------
pip install opencv-python ultralytics torch torchvision timm numpy

Inputs (place next to this script)
-----------------------------------
- road_test.mp4      : input driving video
- road_small.pt       : YOLOv8s weights trained on {pothole, hump}

Controls
--------
q / ESC  : quit
"""

import os
import time
import collections
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "ultralytics is required for this script. Install with:\n"
        "    pip install ultralytics"
    ) from exc


# ======================================================================
# CONFIGURATION
# ======================================================================

@dataclass
class Config:
    video_path: str = "road_test.mp4"
    yolo_weights: str = "road_small.pt"
    midas_model_name: str = "MiDaS_small"

    canvas_w: int = 1280
    canvas_h: int = 720
    panel_w: int = 480
    main_w: int = field(init=False)
    main_h: int = field(init=False)

    infer_every_n_frames: int = 5
    yolo_conf_thresh: float = 0.35
    yolo_iou_thresh: float = 0.45

    critical_dist_m: float = 3.0
    caution_dist_m: float = 7.0
    max_dist_m: float = 15.0
    min_dist_m: float = 2.0

    camera_height_m: float = 1.20
    focal_length_px: float = 700.0
    horizon_ratio: float = 0.42

    lane_center_tolerance_px: int = 40

    def __post_init__(self):
        self.main_w = self.canvas_w - self.panel_w
        self.main_h = self.canvas_h


CFG = Config()

# ======================================================================
# COLOR PALETTE (BGR) -- dark-mode / sci-fi cockpit theme
# ======================================================================
COL_BG          = (18, 14, 12)
COL_PANEL_BG    = (28, 22, 18)
COL_PANEL_CARD  = (46, 36, 28)
COL_CYAN        = (255, 226, 60)
COL_CYAN_DIM    = (150, 120, 40)
COL_GREEN       = (94, 214, 68)
COL_AMBER       = (0, 170, 255)
COL_RED         = (50, 50, 235)
COL_WHITE       = (240, 240, 240)
COL_GREY        = (150, 150, 150)
COL_TEXT_DIM    = (120, 120, 130)


# ======================================================================
# CPU RUNTIME OPTIMIZATION
# ======================================================================

def configure_cpu_runtime() -> int:
    """Tune torch / OpenCV threading for a laptop-class CPU (e.g. i5)."""
    cpu_count = os.cpu_count() or 4
    threads = max(1, cpu_count - 1)
    torch.set_num_threads(threads)
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"[Runtime] OpenCL acceleration enabled. Torch threads={threads}")
        else:
            print(f"[Runtime] OpenCL unavailable. Torch threads={threads}")
    except Exception:
        print(f"[Runtime] OpenCL check failed. Torch threads={threads}")
    return threads


# ======================================================================
# DISTANCE CALIBRATION
# ======================================================================

class DistanceCalibrator:
    """
    Maps an 8-bit median MiDaS inverse-depth value (0-255, higher value
    means closer object) to a metric distance estimate in meters using
    an empirically fitted 3rd-order polynomial calibration curve.

    If the depth network is unavailable, `geometric_fallback` provides a
    robust flat-ground-plane monocular distance estimate derived from the
    bounding box's contact point with the road surface.
    """

    def __init__(self):
        # Empirical calibration control points: (median_depth_8u, distance_m)
        depth_vals = np.array([255, 220, 190, 160, 130, 100, 70, 40, 15, 0], dtype=np.float64)
        dist_vals = np.array([2.0, 2.8, 3.6, 4.6, 6.0, 7.8, 9.6, 11.6, 13.4, 15.0], dtype=np.float64)
        self._coeffs = np.polyfit(depth_vals, dist_vals, deg=3)
        self._poly = np.poly1d(self._coeffs)

    def depth_to_distance(self, median_depth_8u: float) -> float:
        distance = float(self._poly(median_depth_8u))
        return float(np.clip(distance, CFG.min_dist_m, CFG.max_dist_m))

    @staticmethod
    def geometric_fallback(bbox_bottom_y: int, frame_h: int) -> float:
        """Pinhole flat-ground-plane distance estimate using the object's
        ground-contact pixel row relative to the assumed horizon line."""
        horizon_y = frame_h * CFG.horizon_ratio
        denom = bbox_bottom_y - horizon_y
        if denom <= 1.0:
            return CFG.max_dist_m
        distance = (CFG.camera_height_m * CFG.focal_length_px) / denom
        return float(np.clip(distance, CFG.min_dist_m, CFG.max_dist_m))


# ======================================================================
# MODEL 2: VIRTUAL LIDAR (MiDaS DEPTH)
# ======================================================================

class DepthEstimator:
    """Wraps Intel MiDaS_small loaded via torch.hub. Degrades gracefully
    to a `available=False` state if the hub download is blocked by the
    network, in which case callers should use the geometric fallback."""

    def __init__(self, model_name: str = CFG.midas_model_name):
        self.available = False
        self.model = None
        self.transform = None
        self.device = torch.device("cpu")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self.available = True
            print("[DepthEstimator] MiDaS_small loaded successfully - virtual LiDAR ONLINE.")
        except Exception as exc:
            print(f"[DepthEstimator] WARNING: MiDaS unavailable ({exc}).")
            print("[DepthEstimator] Falling back to geometric ground-plane distance model.")
            self.available = False

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns an 8-bit normalized inverse-depth map (H, W), or None
        if the depth network is unavailable / inference fails."""
        if not self.available:
            return None
        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()
            d_min, d_max = float(depth_map.min()), float(depth_map.max())
            if d_max - d_min < 1e-6:
                return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            depth_norm = (depth_map - d_min) / (d_max - d_min)
            depth_8u = (depth_norm * 255.0).astype(np.uint8)
            return depth_8u
        except Exception as exc:
            print(f"[DepthEstimator] inference error, disabling depth engine: {exc}")
            self.available = False
            return None


# ======================================================================
# MODEL 1: SEMANTIC ANOMALY DETECTOR (YOLOv8s)
# ======================================================================

class AnomalyDetector:
    """Wraps a YOLOv8s model trained to detect {pothole, hump}."""

    def __init__(self, weights_path: str):
        self.available = False
        self.model = None
        if not os.path.isfile(weights_path):
            print(f"[AnomalyDetector] WARNING: weights '{weights_path}' not found. "
                  f"Detector disabled until the file is present next to this script.")
            return
        try:
            self.model = YOLO(weights_path)
            self.available = True
            print("[AnomalyDetector] YOLOv8s road-anomaly model loaded.")
        except Exception as exc:
            print(f"[AnomalyDetector] ERROR loading weights: {exc}")
            self.available = False

    def infer(self, frame_bgr: np.ndarray) -> List[dict]:
        if not self.available:
            return []
        results = self.model.predict(
            frame_bgr,
            conf=CFG.yolo_conf_thresh,
            iou=CFG.yolo_iou_thresh,
            device="cpu",
            verbose=False,
        )
        detections: List[dict] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = xyxy
            detections.append({
                "label": label,
                "conf": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })
        return detections


# ======================================================================
# FUSION LAYER (LATE FEATURE FUSION)
# ======================================================================

@dataclass
class FusedTrack:
    label: str
    conf: float
    bbox: Tuple[int, int, int, int]
    distance_m: float
    risk: str
    lane_side: str
    timestamp: float


def classify_risk(distance_m: float) -> str:
    if distance_m < CFG.critical_dist_m:
        return "CRITICAL"
    elif distance_m < CFG.caution_dist_m:
        return "CAUTION"
    return "SAFE"


def determine_lane_side(bbox_center_x: int, frame_w: int) -> str:
    center = frame_w / 2.0
    if bbox_center_x < center - CFG.lane_center_tolerance_px:
        return "LEFT"
    elif bbox_center_x > center + CFG.lane_center_tolerance_px:
        return "RIGHT"
    return "CENTER"


def fuse_detections(detections: List[dict], depth_map: Optional[np.ndarray],
                     calibrator: DistanceCalibrator, frame_shape) -> List[FusedTrack]:
    """For every YOLO box, crop the matching ROI from the MiDaS depth map,
    take the MEDIAN depth (robust to edge noise) and convert it to a
    metric distance. Falls back to the geometric estimator when depth is
    unavailable or the ROI is degenerate."""
    frame_h, frame_w = frame_shape[:2]
    tracks: List[FusedTrack] = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame_w - 1, x2), min(frame_h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        if depth_map is not None:
            roi = depth_map[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                distance = calibrator.geometric_fallback(y2c, frame_h)
            else:
                median_depth = float(np.median(roi))
                distance = calibrator.depth_to_distance(median_depth)
        else:
            distance = calibrator.geometric_fallback(y2c, frame_h)

        cx = (x1c + x2c) // 2
        risk = classify_risk(distance)
        lane_side = determine_lane_side(cx, frame_w)

        tracks.append(FusedTrack(
            label=det["label"],
            conf=det["conf"],
            bbox=(x1c, y1c, x2c, y2c),
            distance_m=distance,
            risk=risk,
            lane_side=lane_side,
            timestamp=time.time(),
        ))

    tracks.sort(key=lambda t: t.distance_m)
    return tracks


# ======================================================================
# HUD PRIMITIVES
# ======================================================================

def _risk_color(risk: str) -> Tuple[int, int, int]:
    return {"CRITICAL": COL_RED, "CAUTION": COL_AMBER, "SAFE": COL_GREEN}.get(risk, COL_GREEN)


def draw_translucent_rect(img: np.ndarray, pt1, pt2, color, alpha: float = 0.18) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_corner_bracket(img: np.ndarray, x1, y1, x2, y2, color, thickness: int = 2,
                         ratio: float = 0.22) -> None:
    """Sleek sci-fi tracking bracket (open corners instead of a full box)."""
    w, h = x2 - x1, y2 - y1
    lx = max(6, int(w * ratio))
    ly = max(6, int(h * ratio))
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness, cv2.LINE_AA)


def draw_text(img: np.ndarray, text: str, org, scale: float = 0.55, color=COL_WHITE,
              thickness: int = 1, font=cv2.FONT_HERSHEY_SIMPLEX, shadow: bool = True) -> None:
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_glass_card(panel: np.ndarray, x: int, y: int, w: int, h: int,
                     alpha: float = 0.55, border_color=COL_CYAN_DIM) -> None:
    overlay = panel.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COL_PANEL_CARD, -1)
    cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, dst=panel)
    cv2.rectangle(panel, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)


# ======================================================================
# MAIN DRIVING VIEW (LEFT SIDE)
# ======================================================================

def render_main_view(frame_bgr: np.ndarray, tracks: List[FusedTrack], frame_count: int) -> np.ndarray:
    view = frame_bgr.copy()
    h, w = view.shape[:2]

    for trk in tracks:
        x1, y1, x2, y2 = trk.bbox
        color = _risk_color(trk.risk)
        draw_translucent_rect(view, (x1, y1), (x2, y2), color, alpha=0.16)
        draw_corner_bracket(view, x1, y1, x2, y2, color, thickness=2)

        label_txt = f"{trk.label.upper()} {trk.distance_m:4.1f}m"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y = max(th + 8, y1 - 6)
        cv2.rectangle(view, (x1, tag_y - th - 8), (x1 + tw + 12, tag_y), color, -1)
        draw_text(view, label_txt, (x1 + 6, tag_y - 5), scale=0.5,
                  color=(15, 15, 15), thickness=1, shadow=False)

        if trk.risk == "CAUTION":
            draw_text(view, "CAUTION: HAZARD CLOSING", (x1, min(h - 6, y2 + 18)),
                      scale=0.48, color=COL_AMBER, thickness=1)

    critical_tracks = [t for t in tracks if t.risk == "CRITICAL"]
    if critical_tracks:
        flash_on = (frame_count // 4) % 2 == 0
        if flash_on:
            bar_h = 46
            overlay = view.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), COL_RED, -1)
            cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)
            msg = "CRITICAL THREAT: IMPACT IMMINENT"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            draw_text(view, msg, ((w - tw) // 2, bar_h // 2 + th // 2), scale=0.85,
                      color=COL_WHITE, thickness=2, font=cv2.FONT_HERSHEY_DUPLEX)

    cv2.line(view, (w // 2, h - 60), (w // 2, h), COL_CYAN_DIM, 1, cv2.LINE_AA)
    draw_text(view, "MAIN CAMERA FEED :: LANE-FORWARD", (14, 22), scale=0.5,
              color=COL_CYAN, thickness=1)

    resized = cv2.resize(view, (CFG.main_w, CFG.main_h), interpolation=cv2.INTER_NEAREST)
    return resized


# ======================================================================
# TELEMETRY PANEL (RIGHT SIDE)
# ======================================================================

def compute_evasion_guidance(tracks: List[FusedTrack]) -> Tuple[str, Tuple[int, int, int]]:
    urgent = [t for t in tracks if t.risk in ("CRITICAL", "CAUTION")]
    if not urgent:
        return "KEEP LINE", COL_GREEN
    nearest = min(urgent, key=lambda t: t.distance_m)
    urgency_color = COL_RED if nearest.risk == "CRITICAL" else COL_AMBER
    if nearest.lane_side == "LEFT":
        return "MANEUVER: SWERVE RIGHT", urgency_color
    elif nearest.lane_side == "RIGHT":
        return "MANEUVER: SWERVE LEFT", urgency_color
    return "MANEUVER: BRAKE - HAZARD AHEAD", urgency_color


def update_virtual_speed(prev_speed: float, tracks: List[FusedTrack], dt: float) -> float:
    target = 58.0
    if any(t.risk == "CRITICAL" for t in tracks):
        target = 8.0
    elif any(t.risk == "CAUTION" for t in tracks):
        target = 28.0
    rate = 18.0  # km/h per second
    delta = float(np.clip(target - prev_speed, -rate * dt, rate * dt))
    new_speed = prev_speed + delta
    return float(np.clip(new_speed, 0.0, 80.0))


def render_telemetry_panel(tracks: List[FusedTrack], hazard_log: collections.deque,
                            speed_kmh: float, frame_count: int, midas_online: bool,
                            yolo_online: bool) -> np.ndarray:
    pw, ph = CFG.panel_w, CFG.canvas_h
    panel = np.full((ph, pw, 3), COL_PANEL_BG, dtype=np.uint8)

    draw_text(panel, "ADAS COCKPIT :: SENSOR FUSION HUD", (18, 30), scale=0.58,
              color=COL_CYAN, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX)
    cv2.line(panel, (18, 40), (pw - 18, 40), COL_CYAN_DIM, 1, cv2.LINE_AA)

    # --- Status card ---
    draw_glass_card(panel, 18, 56, pw - 36, 84)
    yolo_txt = "SCANNING" if yolo_online else "OFFLINE"
    midas_txt = "LOCKED (MiDaS)" if midas_online else "GEO-FALLBACK"
    yolo_col = COL_GREEN if yolo_online else COL_RED
    midas_col = COL_GREEN if midas_online else COL_AMBER
    draw_text(panel, "SYSTEM STATUS", (32, 78), scale=0.42, color=COL_TEXT_DIM)
    draw_text(panel, f"OBJECT ENGINE:  {yolo_txt}", (32, 100), scale=0.48, color=yolo_col)
    draw_text(panel, f"DEPTH ENGINE:   {midas_txt}", (32, 122), scale=0.48, color=midas_col)

    # --- Speed card ---
    draw_glass_card(panel, 18, 152, pw - 36, 78)
    draw_text(panel, "VEHICLE SPEED", (32, 174), scale=0.42, color=COL_TEXT_DIM)
    draw_text(panel, f"{speed_kmh:5.1f} km/h", (32, 210), scale=0.9, color=COL_WHITE,
              thickness=2, font=cv2.FONT_HERSHEY_DUPLEX)

    # --- Nearest hazard distance card ---
    draw_glass_card(panel, 18, 242, pw - 36, 118)
    draw_text(panel, "FUSED RANGE TO NEAREST HAZARD", (32, 264), scale=0.42, color=COL_TEXT_DIM)
    if tracks:
        nearest = tracks[0]
        col = _risk_color(nearest.risk)
        draw_text(panel, f"{nearest.distance_m:4.1f} m", (32, 320), scale=1.35,
                  color=col, thickness=3, font=cv2.FONT_HERSHEY_DUPLEX)
        draw_text(panel, f"{nearest.label.upper()}  |  {nearest.risk}", (32, 344),
                  scale=0.46, color=col, thickness=1)
    else:
        draw_text(panel, "-- m", (32, 320), scale=1.35, color=COL_GREY,
                  thickness=3, font=cv2.FONT_HERSHEY_DUPLEX)
        draw_text(panel, "ROAD CLEAR", (32, 344), scale=0.46, color=COL_GREEN)

    # --- Evasion guidance card ---
    draw_glass_card(panel, 18, 372, pw - 36, 92)
    draw_text(panel, "EVASION GUIDANCE", (32, 394), scale=0.42, color=COL_TEXT_DIM)
    guidance, gcol = compute_evasion_guidance(tracks)
    draw_text(panel, guidance, (32, 432), scale=0.6, color=gcol, thickness=2,
              font=cv2.FONT_HERSHEY_DUPLEX)

    # --- Hazard tracking log card ---
    log_y = 476
    draw_glass_card(panel, 18, log_y, pw - 36, 200)
    draw_text(panel, "ACTIVE HAZARD TRACKING LOG", (32, log_y + 22), scale=0.42, color=COL_TEXT_DIM)
    row_y = log_y + 48
    recent = list(hazard_log)[-3:][::-1]
    if not recent:
        draw_text(panel, "No hazards logged.", (32, row_y), scale=0.46, color=COL_GREY)
    else:
        for i, entry in enumerate(recent):
            col = _risk_color(entry["risk"])
            line1 = f"{i + 1}. {entry['label'].upper()}"
            line2 = f"   {entry['distance']:4.1f} m | {entry['risk']} | {entry['lane']}"
            draw_text(panel, line1, (32, row_y), scale=0.48, color=col, thickness=1)
            draw_text(panel, line2, (32, row_y + 20), scale=0.42, color=COL_TEXT_DIM, thickness=1)
            row_y += 52

    draw_text(panel, f"FRAME {frame_count:06d}", (18, ph - 16), scale=0.4, color=COL_TEXT_DIM)
    return panel


# ======================================================================
# MAIN LOOP
# ======================================================================

def main() -> None:
    configure_cpu_runtime()

    if not os.path.isfile(CFG.video_path):
        raise SystemExit(
            f"Video source '{CFG.video_path}' not found. Place the file next to this script."
        )

    cap = cv2.VideoCapture(CFG.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source '{CFG.video_path}'.")

    detector = AnomalyDetector(CFG.yolo_weights)
    depth_estimator = DepthEstimator(CFG.midas_model_name)
    calibrator = DistanceCalibrator()

    hazard_log: collections.deque = collections.deque(maxlen=50)

    cached_tracks: List[FusedTrack] = []

    speed_kmh = 55.0
    frame_count = 0
    prev_time = time.time()

    window_name = "ADAS Virtual Sensor Fusion Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CFG.canvas_w, CFG.canvas_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        now = time.time()
        dt = max(1e-3, now - prev_time)
        prev_time = now

        run_inference = (frame_count % CFG.infer_every_n_frames == 0)

        if run_inference:
            detections = detector.infer(frame)
            depth_map = depth_estimator.infer(frame)
            cached_tracks = fuse_detections(detections, depth_map, calibrator, frame.shape)

            for trk in cached_tracks:
                hazard_log.append({
                    "label": trk.label,
                    "distance": trk.distance_m,
                    "risk": trk.risk,
                    "lane": trk.lane_side,
                })
        # else: reuse cached_tracks from the last inference frame to save CPU

        speed_kmh = update_virtual_speed(speed_kmh, cached_tracks, dt)

        main_view = render_main_view(frame, cached_tracks, frame_count)
        panel = render_telemetry_panel(
            cached_tracks, hazard_log, speed_kmh, frame_count,
            depth_estimator.available, detector.available,
        )

        canvas = np.full((CFG.canvas_h, CFG.canvas_w, 3), COL_BG, dtype=np.uint8)
        canvas[0:CFG.main_h, 0:CFG.main_w] = main_view
        canvas[0:CFG.canvas_h, CFG.main_w:CFG.canvas_w] = panel
        cv2.line(canvas, (CFG.main_w, 0), (CFG.main_w, CFG.canvas_h), COL_CYAN_DIM, 2, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()