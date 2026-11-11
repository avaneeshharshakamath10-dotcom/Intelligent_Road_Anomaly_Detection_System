import os
from ultralytics import YOLO

class SemanticDetector:
    """Perception Branch 1: Appearance Domain Engine."""
    def __init__(self, weights_path: str, conf_thresh: float, iou_thresh: float):
        self.available = False
        self.model = None
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        if not os.path.isfile(weights_path):
            print(f"[Detector] WARNING: Model weights '{weights_path}' missing. Operating in offline bypass mode.")
            return
        try:
            self.model = YOLO(weights_path)
            self.available = True
            print("[Detector] Custom YOLOv8s Road-Anomaly model loaded successfully.")
        except Exception as e:
            print(f"[Detector] Initialization Failure: {e}")
            self.available = False

    def generate_semantic_verdict(self, confidence: float) -> str:
        """Enforces a discrete classification verdict baseline."""
        return "HAZARD" if confidence >= 0.70 else "UNCERTAIN"