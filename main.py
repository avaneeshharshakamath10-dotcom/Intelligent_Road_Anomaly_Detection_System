import cv2
import torch
import os
import time
import collections
import numpy as np
import serial
import serial.tools.list_ports
from config import CFG
from detector import SemanticDetector
from tracker import ObjectTracker
from depth_estimator import VirtualLidarEngine
from fusion import DecisionLevelFusionEngine
from hud import CockpitHUDHUD
from packet_generator import HazardTelemetryPacketGenerator

def optimize_cpu_threading_infrastructure():
    """Sets CPU thread bindings to eliminate laptop hardware stalls."""
    allocated_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(allocated_threads)
    try:
        cv2.setNumThreads(allocated_threads)
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"[CoreInit] OpenCL Acceleration enabled. Allocated Threads: {allocated_threads}")
    except Exception as e:
        print(f"[CoreInit] Performance Thread Assignment Warning: {e}")

# Static structural class definition with all required attributes to perfectly match hud.py expectations
class MockTrack:
    def __init__(self, action, label="pothole (cached)", distance=1.2, conf=0.85, agreement="YES"):
        self.master_action = action
        self.label = label
        self.distance_m = distance
        self.fusion_confidence = conf
        self.agreement = agreement
        self.bbox = (0, 0, 0, 0)  # Safe empty coordinates so hud.py doesn't crash
        self.id = 99              # Safe numeric identifier for the HUD print stream
        self.lane_side = "CENTER" # Safe lane orientation property to stop HUD crashes

def main():
    optimize_cpu_threading_infrastructure()

    if not os.path.isfile(CFG.video_path):
        raise SystemExit(f"[CoreCritical] Target input file asset missing from path boundaries: '{CFG.video_path}'")

    cap = cv2.VideoCapture(CFG.video_path)
    if not cap.isOpened():
        raise SystemExit(f"[CoreCritical] Device interface stream failure targeting source: '{CFG.video_path}'")

    # --- MICROCONTROLLER NODE SERIAL INITIALIZATION ---
    # Hardcoded to COM3 (STM32 ST-Link VCP). The ESP32 sits on a separate COM port, so
    # auto-scanning by description was risky and could grab the wrong port.
    STM32_PORT = "COM3"
    ser = None
    try:
        ser = serial.Serial(port=STM32_PORT, baudrate=115200, timeout=0.1)
        print(f"[HardwareLink] Successfully connected to STM32 Node on port: {STM32_PORT}")

        # --- FORCE STARTUP HARDWARE LIGHT TEST ---
        print("[HardwareLink] Running 2-second serial stabilization window...")
        time.sleep(2)
        ser.write(b'A\r\n')  # Terminated so the STM32 flushes it as an unknown command, not a buffer prefix
        ser.flush()
        print("[HardwareLink] Force-sent 'A' trigger command down the link! Verify if green LD3 turned on.")
        # ------------------------------------------

    except Exception as e:
        print(f"[HardwareLink] Could not open {STM32_PORT} ({e}). Running in simulation mode.")

    # Module Pipeline Instantiations
    detector_module = SemanticDetector(CFG.yolo_weights, CFG.yolo_conf_thresh, CFG.yolo_iou_thresh)
    tracker_module = ObjectTracker(detector_module)
    lidar_module = VirtualLidarEngine(CFG.midas_model_name)
    fusion_engine = DecisionLevelFusionEngine()
    hud_dashboard = CockpitHUDHUD()
    packet_engine = HazardTelemetryPacketGenerator()

    # Runtime Memory Management State Pools
    active_historical_log = collections.deque(maxlen=40)
    cached_fused_tracks = []
    track_persistence_counters = {}  # Tracks temporal confirmations per discrete tracking ID

    # Session-wide hazard counters (each unique track ID counted exactly once, on first confirmation)
    hazard_counts = {"pothole": 0, "hump": 0}
    counted_hazard_ids = set()

    simulated_vehicle_speed = 60.0
    global_frame_count = 0
    time_stamp_marker = time.time()
    last_uart_send_time = 0.0  # Throttles hazard command resends to the STM32

    display_window_name = "COCKPIT ADAS ENGINE MONITOR"
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window_name, CFG.canvas_w, CFG.canvas_h)

    # Initialize targeted demonstration window for Virtual LiDAR
    lidar_window_name = "TARGETED VIRTUAL LIDAR SCANNER (DEBUG)"
    cv2.namedWindow(lidar_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lidar_window_name, CFG.main_w, CFG.main_h)

    # Performance Fallback Cache States
    last_known_tracks = []
    blind_spot_memory_cache = None
    dense_depth_array = None  # Persistent frame placeholder for skipped inference steps
    
    # Calculate exact frame delay from video properties to normalize playback speed
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ideal_frame_delay_ms = max(1, int(1000 / video_fps))

    print("\n[Perception System Executing Pipeline Operations... Press 'q' to terminate cleanly]\n")

    while True:
        success, frame = cap.read()
        if not success:
            # Check if we are actually at the end of the file, or if it's just a temporary disk lag
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if current_frame >= (total_frames - 2):
                # Genuinely at the end of the video, loop it back smoothly
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                # Minor storage reading hiccup; pause briefly and retry reading
                cv2.waitKey(5)
                continue

        current_time = time.time()
        time_delta = max(1e-3, current_time - time_stamp_marker)
        time_stamp_marker = current_time

        execute_inference_step = (global_frame_count % CFG.infer_every_n_frames == 0)

        if execute_inference_step:
            # Run heavy models only on designated step intervals
            raw_detections = tracker_module.process_tracking_stream(frame)
            dense_depth_array = lidar_module.calculate_depth_field(frame)
            synchronized_fused_set = fusion_engine.process_fusion_layer(
                raw_detections, dense_depth_array, detector_module, lidar_module, frame.shape
            )

            validated_fused_tracks = []
            current_frame_detected_ids = set()

            for track in synchronized_fused_set:
                t_id = track.id
                current_frame_detected_ids.add(t_id)
                track_persistence_counters[t_id] = track_persistence_counters.get(t_id, 0) + 1

                if track_persistence_counters[t_id] >= CFG.temporal_confirm_frames:
                    validated_fused_tracks.append(track)
                    active_historical_log.append({
                        "id": track.id, "label": track.label, "dist": track.distance_m,
                        "action": track.master_action, "conf": track.fusion_confidence, "agree": track.agreement
                    })
                    packet_engine.construct_telemetry_packet(track, simulated_vehicle_speed)

                    if t_id not in counted_hazard_ids:
                        counted_hazard_ids.add(t_id)
                        if "hump" in track.label.lower():
                            hazard_counts["hump"] += 1
                        else:
                            hazard_counts["pothole"] += 1

            obsolete_ids = [k for k in track_persistence_counters if k not in current_frame_detected_ids]
            for o_id in obsolete_ids:
                track_persistence_counters.pop(o_id, None)

            cached_fused_tracks = validated_fused_tracks
            last_known_tracks = cached_fused_tracks  # Update processing cache window
        else:
            # On skipped frames, reuse previous tracks so bounding boxes do not flicker
            cached_fused_tracks = last_known_tracks

        # --- SMART BLIND-SPOT MEMORY ACCUMULATOR ---
        if cached_fused_tracks:
            closest_anomaly = cached_fused_tracks[0]
            if hasattr(closest_anomaly, 'master_action') and closest_anomaly.master_action in ["EMERGENCY_BRAKE", "SLOW_DOWN"] and getattr(closest_anomaly, 'distance_m', 99.0) <= 3.5:
                blind_spot_memory_cache = {"action": closest_anomaly.master_action, "hold_frames": 15}
            else:
                blind_spot_memory_cache = None
        else:
            if blind_spot_memory_cache is not None and blind_spot_memory_cache["hold_frames"] > 0:
                blind_spot_memory_cache["hold_frames"] -= 1
                cached_fused_tracks = [MockTrack(action=blind_spot_memory_cache["action"])]
            else:
                blind_spot_memory_cache = None

        # --- CHASSIS TELEMETRY PHYSICS & STM32 UART TRANSMISSION LOGIC ---
        hazard_command = None  # None = clear road; otherwise "POTHOLE" or "HUMP"

        if cached_fused_tracks:
            closest_anomaly = cached_fused_tracks[0]
            if hasattr(closest_anomaly, 'master_action'):
                if closest_anomaly.master_action == "EMERGENCY_BRAKE":
                    simulated_vehicle_speed = max(0.0, simulated_vehicle_speed - 32.0 * time_delta)
                    hazard_command = "HUMP" if "hump" in closest_anomaly.label.lower() else "POTHOLE"
                elif closest_anomaly.master_action == "SLOW_DOWN":
                    simulated_vehicle_speed = max(20.0, simulated_vehicle_speed - 14.0 * time_delta)
                    hazard_command = "HUMP" if "hump" in closest_anomaly.label.lower() else "POTHOLE"
        else:
            simulated_vehicle_speed = min(65.0, simulated_vehicle_speed + 5.0 * time_delta)

        # STM32's UART ISR (HAL_UART_RxCpltCallback) only fires on a complete line ending in \r or \n,
        # and only recognizes the exact strings "POTHOLE" / "HUMP" -- so we must send those, terminated.
        # Throttled so we're not pushing a full nRF24 transmission every single video frame.
        if ser and ser.is_open and hazard_command:
            if (current_time - last_uart_send_time) >= CFG.uart_resend_interval_s:
                try:
                    ser.write(f"{hazard_command}\r\n".encode('utf-8'))
                    ser.flush()  # Forces system cache clearing so the bytes instantly exit via USB link
                    last_uart_send_time = current_time
                except Exception as e:
                    print(f"[HardwareLink] UART Transmission Dropped: {e}")

        # Draw Complete Stitched Telemetry Canvas HUD
        visual_dashboard_canvas = hud_dashboard.render_hud_matrix(
            frame, cached_fused_tracks, global_frame_count,
            detector_module.available, lidar_module.available, active_historical_log, hazard_counts
        )

        cv2.imshow(display_window_name, visual_dashboard_canvas)

        # --- RENDER TARGETED RADAR SCAN DEPTH MAP ---
        if dense_depth_array is not None:
            # Start with an isolated black environment matrix
            depth_canvas = np.zeros_like(frame)
            
            # Normalize floating point values to unsigned 8-bit integers
            depth_uint8 = cv2.normalize(dense_depth_array, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            
            # Extract depth data exclusively within detected pothole boundaries
            for trk in cached_fused_tracks:
                if hasattr(trk, 'bbox'):
                    x1, y1, x2, y2 = trk.bbox
                    if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                        # Bound clamping safeguards to completely stop matrix clipping crashes
                        img_h, img_w, _ = frame.shape
                        x1, y1 = max(0, min(img_w - 1, x1)), max(0, min(img_h - 1, y1))
                        x2, y2 = max(0, min(img_w - 1, x2)), max(0, min(img_h - 1, y2))
                        
                        if (x2 > x1) and (y2 > y1):
                            # Copy cross-sections inside the black viewport array layout
                            depth_canvas[y1:y2, x1:x2] = depth_heatmap[y1:y2, x1:x2]
                            
                            # Draw clean target identifiers inside Window 2
                            cv2.rectangle(depth_canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            label_txt = f"SCAN ID {trk.id}: {trk.distance_m:.1f}m"
                            cv2.putText(depth_canvas, label_txt, (x1, max(y1 - 8, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Resize window layout parameters seamlessly
            depth_heatmap_resized = cv2.resize(depth_canvas, (CFG.main_w, CFG.main_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(lidar_window_name, depth_heatmap_resized)

        global_frame_count += 1

        # Synchronize waitKey delay directly to match target input asset frame rates
        termination_interrupt = cv2.waitKey(ideal_frame_delay_ms) & 0xFF
        if termination_interrupt == ord('q') or termination_interrupt == 27:
            break

    cap.release()
    if ser and ser.is_open:
        ser.close()
        print("[HardwareLink] Serial communication link closed safely.")
    cv2.destroyAllWindows()
    print("\n[Perception Dashboard Operations Suspended Cleanly.]")

if __name__ == "__main__":
    main()