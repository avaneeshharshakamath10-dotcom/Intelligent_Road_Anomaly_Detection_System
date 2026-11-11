import time
from typing import Dict
from fusion import FusedAnomalyTrack

class HazardTelemetryPacketGenerator:
    """Serializes structured telemetry frames ready for downstream UART/Embedded channels."""
    @staticmethod
    def construct_telemetry_packet(track: FusedAnomalyTrack, vehicle_speed: float) -> Dict:
        return {
            "vehicle_id": 1,
            "timestamp": int(time.time()),
            "track_id": int(track.id),
            "hazard_class": str(track.label),
            "metric_distance_m": float(round(track.distance_m, 2)),
            "yolo_semantic_verdict": str(track.yolo_decision),
            "lidar_proximity_verdict": str(track.proximity_decision),
            "fused_master_action": str(track.master_action),
            "fused_confidence_pct": float(round(track.fusion_confidence * 100.0, 1)),
            "sensor_agreement_flag": str(track.agreement),
            "lane_offset_side": str(track.lane_side),
            "current_vehicle_speed_kmh": float(round(vehicle_speed, 1))
        }