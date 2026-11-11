import os
from dataclasses import dataclass, field

@dataclass
class Config:
    video_path: str = "Untitled design.mp4"
    yolo_weights: str = "road_small.pt"
    midas_model_name: str = "MiDaS_small"

    # Screen Layout Geometry
    canvas_w: int = 1280
    canvas_h: int = 720
    panel_w: int = 480
    main_w: int = field(init=False)
    main_h: int = field(init=False)

    # Core Thresholds
    infer_every_n_frames: int = 2
    uart_resend_interval_s: float = 1.5  # Min seconds between hazard command resends to the STM32
    yolo_conf_thresh: float = 0.70
    yolo_iou_thresh: float = 0.45
    temporal_confirm_frames: int = 1

    # Safety Proximity Boundaries (Meters)
    critical_dist_m: float = 3.0
    caution_dist_m: float = 7.0
    max_dist_m: float = 15.0
    min_dist_m: float = 1.5

    # Camera Calibration Constraints (Geometric Fallback Engine)
    camera_height_m: float = 1.20
    focal_length_px: float = 700.0
    horizon_ratio: float = 0.42
    lane_center_tolerance_px: int = 50

    def __post_init__(self):
        self.main_w = self.canvas_w - self.panel_w
        self.main_h = self.canvas_h

CFG = Config()

# BGR Color Matrix Cockpit Palette
COL_BG          = (12, 14, 18)
COL_PANEL_BG    = (18, 22, 28)
COL_PANEL_CARD  = (28, 36, 46)
COL_CYAN        = (240, 226, 60)
COL_CYAN_DIM    = (120, 110, 40)
COL_GREEN       = (68, 214, 94)
COL_AMBER       = (0, 140, 255)
COL_RED         = (55, 55, 240)
COL_WHITE       = (245, 245, 245)
COL_TEXT_DIM    = (140, 140, 150)