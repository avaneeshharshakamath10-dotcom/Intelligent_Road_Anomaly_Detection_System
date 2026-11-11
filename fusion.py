import numpy as np
from typing import List, Dict

class FusedAnomalyTrack:
    def __init__(self, track_id, label, distance, action, confidence, agreement, lane_side, bbox, yolo_decision, proximity_decision):
        self.id = track_id
        self.label = label
        self.distance_m = distance
        self.master_action = action
        self.fusion_confidence = confidence
        self.agreement = agreement
        self.lane_side = lane_side
        self.bbox = bbox
        self.yolo_decision = yolo_decision
        self.proximity_decision = proximity_decision

class DecisionLevelFusionEngine:
    """Fuses camera bounding metrics and depth arrays with persistence threat latching."""
    
    def __init__(self):
        # Keeps track of historical threat states across frames to prevent box-collapse glitches
        self.historical_threat_registry = {}

    def process_fusion_layer(self, detections: List[Dict], depth_map: np.ndarray, detector_mod, lidar_mod, frame_shape) -> List[FusedAnomalyTrack]:
        fused_results = []
        if depth_map is None or len(depth_map.shape) < 2:
            return fused_results

        h, w = frame_shape[0], frame_shape[1]
        
        # Strict Ego-Lane boundaries (middle ~40% of camera sweep)
        lane_left_boundary = int(w * 0.30)
        lane_right_boundary = int(w * 0.70)
        
        current_frame_ids = set()

        for det in detections:
            track_id = det["id"]
            current_frame_ids.add(track_id)
            
            bbox = det["bbox"]  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox
            box_center_x = (x1 + x2) // 2
            
            # Identify lane orientation assignment
            if box_center_x < lane_left_boundary:
                lane_side = "LEFT"
            elif box_center_x > lane_right_boundary:
                lane_side = "RIGHT"
            else:
                lane_side = "CENTER"

            # --- DYNAMIC DISTANCE CALIBRATION MATRIX WITH OVERFLOW GATE ---
            if y2 >= int(h * 0.85):
                final_distance = 1.0  
            else:
                screen_ratio = max(0.01, min(1.0, y2 / h))
                calculated_distance = 25.0 * (1.0 - screen_ratio)
                
                sample_y = max(0, min(h - 1, int(y2 - (y2 - y1) * 0.15)))
                sample_x = max(0, min(w - 1, box_center_x))
                raw_depth_val = depth_map[sample_y, sample_x]
                
                if raw_depth_val > 0:
                    tuned_distance = 150.0 / (raw_depth_val + 1e-5)
                    final_distance = float(np.clip((calculated_distance * 0.6) + (tuned_distance * 0.4), 1.0, 30.0))
                else:
                    final_distance = float(np.clip(calculated_distance, 1.0, 30.0))

            # --- RIGID THREAT ESCALATION MATRIX ---
            yolo_verdict = detector_mod.generate_semantic_verdict(det["conf"])
            
            # Base logic assignment
            if lane_side == "CENTER":
                if final_distance <= 4.5 or y2 >= int(h * 0.80):
                    master_action = "EMERGENCY_BRAKE"
                    proximity_verdict = "CRITICAL"
                elif final_distance <= 9.5:
                    master_action = "SLOW_DOWN"
                    proximity_verdict = "WARNING"
                else:
                    master_action = "MONITOR"
                    proximity_verdict = "SAFE"
            else:
                master_action = "MONITOR"
                proximity_verdict = "SAFE"

            # --- HYSTERESIS STATE THREAT LATCH ---
            # If this specific pothole ID was already flagged as highly critical in a previous frame,
            # and it is within the immediate approach window, freeze the alarm state.
            if track_id in self.historical_threat_registry:
                previous_action = self.historical_threat_registry[track_id]
                if previous_action in ["EMERGENCY_BRAKE", "SLOW_DOWN"] and final_distance <= 6.0:
                    # Enforce state latching logic: do not allow threat state markdown
                    if previous_action == "EMERGENCY_BRAKE":
                        master_action = "EMERGENCY_BRAKE"
                        proximity_verdict = "CRITICAL"
                        final_distance = min(final_distance, 2.5) # Force numerical indicator to stay low
                    elif previous_action == "SLOW_DOWN" and master_action == "MONITOR":
                        master_action = "SLOW_DOWN"
                        proximity_verdict = "WARNING"

            # Save the current state into history for the next frame calculation
            self.historical_threat_registry[track_id] = master_action

            fused_results.append(FusedAnomalyTrack(
                track_id=track_id,
                label=det["label"],
                distance=final_distance,
                action=master_action,
                confidence=det["conf"],
                agreement="YES" if yolo_verdict == "HAZARD" and proximity_verdict != "SAFE" else "NO",
                lane_side=lane_side,
                bbox=bbox,
                yolo_decision=yolo_verdict,
                proximity_decision=proximity_verdict
            ))

        # Clean memory cache of old tracking identifiers no longer on the camera sensor view
        obsolete_history_ids = [k for k in list(self.historical_threat_registry.keys()) if k not in current_frame_ids]
        for obs_id in obsolete_history_ids:
            self.historical_threat_registry.pop(obs_id, None)

        # Sort array index 0 priorities
        fused_results.sort(key=lambda t: (0 if t.lane_side == "CENTER" else 1, t.distance_m))
        return fused_results
    