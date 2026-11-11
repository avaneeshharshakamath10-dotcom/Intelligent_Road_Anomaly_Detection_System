import numpy as np
from typing import List, Dict

class ObjectTracker:
    """Performs persistent multi-object tracking using YOLO + ByteTrack."""

    def __init__(self, detector_instance):
        self.detector = detector_instance

    def process_tracking_stream(self, frame: np.ndarray) -> List[Dict]:

        if not self.detector.available or self.detector.model is None:
            return []

        # Run YOLO with ByteTrack
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.detector.conf_thresh,
            iou=self.detector.iou_thresh,
            device="cpu",
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            # Skip detections that have no tracking ID
            if box.id is None:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            track_id = int(box.id.item())

            label = self.detector.model.names.get(cls_id, f"Object_{cls_id}")

            # Keep only potholes and humps
            if "pothole" in label.lower() or "hump" in label.lower():

                detections.append({
                    "id": track_id,
                    "label": label,
                    "conf": conf,
                    "bbox": (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3])
                    )
                })

        return detections