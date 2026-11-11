import torch
import cv2
import numpy as np
from typing import Optional
from config import CFG

class VirtualLidarEngine:
    """Perception Branch 2: Structural Range Matrix Generator."""
    def __init__(self, model_name: str):
        self.available = False
        self.model = None
        self.transform = None
        self.device = torch.device("cpu")
        
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self.available = True
            print("[DepthEngine] MiDaS Virtual LiDAR Online (Deep Learning Mode).")
        except Exception as e:
            print(f"[DepthEngine] Network Timeout / Load Error ({e}). Falling back to geometric model.")
            self.available = False

    @torch.no_grad()
    def calculate_depth_field(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
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
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-5:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        return ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)

    def generate_proximity_verdict(self, distance_m: float) -> str:
        if distance_m <= CFG.critical_dist_m:
            return "PROXIMITY_CRITICAL"
        elif distance_m <= CFG.caution_dist_m:
            return "PROXIMITY_CAUTION"
        return "PROXIMITY_SAFE"

    @staticmethod
    def calculate_geometric_fallback(bbox_bottom_y: int, frame_h: int) -> float:
        """Pinhole flat-ground-plane perspective projection model."""
        horizon_y = frame_h * CFG.horizon_ratio
        denom = bbox_bottom_y - horizon_y
        if denom <= 1.0:
            return CFG.max_dist_m
        distance = (CFG.camera_height_m * CFG.focal_length_px) / denom
        return float(np.clip(distance, CFG.min_dist_m, CFG.max_dist_m))