import cv2
import numpy as np
from typing import List, Tuple
from config import CFG, COL_BG, COL_PANEL_BG, COL_PANEL_CARD, COL_CYAN, COL_CYAN_DIM, COL_GREEN, COL_AMBER, COL_RED, COL_WHITE, COL_TEXT_DIM
from fusion import FusedAnomalyTrack

class CockpitHUDHUD:
    """Stitches dual views into a comprehensive graphical telemetry dashboard."""
    def __init__(self):
        pass

    def draw_glass_panel(self, target_img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.4):
        overlay = target_img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COL_PANEL_CARD, -1)
        cv2.addWeighted(overlay, alpha, target_img, 1 - alpha, 0, dst=target_img)
        cv2.rectangle(target_img, (x, y), (x + w, y + h), COL_CYAN_DIM, 1, cv2.LINE_AA)

    def draw_corner_brackets(self, img: np.ndarray, bbox: Tuple[int,int,int,int], color: Tuple[int,int,int]):
        x1, y1, x2, y2 = bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            return
        w, h = x2 - x1, y2 - y1
        length = max(8, int(min(w, h) * 0.25))
        t = 3  # Slightly thicker lines to balance with larger text indicators
        
        # Top-Left Bracket
        cv2.line(img, (x1, y1), (x1 + length, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, t, cv2.LINE_AA)
        # Top-Right Bracket
        cv2.line(img, (x2, y1), (x2 - length, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, t, cv2.LINE_AA)
        # Bottom-Left Bracket
        cv2.line(img, (x1, y2), (x1 + length, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, t, cv2.LINE_AA)
        # Bottom-Right Bracket
        cv2.line(img, (x2, y2), (x2 - length, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, t, cv2.LINE_AA)

    def select_action_color(self, action: str) -> Tuple[int,int,int]:
        if action in ["EMERGENCY_BRAKE", "VERIFY_HAZARD"]: return COL_RED
        if action == "SLOW_DOWN": return COL_AMBER
        return COL_GREEN

    def render_hud_matrix(self, frame: np.ndarray, tracks: List[FusedAnomalyTrack], 
                          global_frame_count: int,
                          yolo_on: bool, midas_on: bool, history_queue, hazard_counts: dict) -> np.ndarray:
        # Create full canvas bounds
        canvas = np.full((CFG.canvas_h, CFG.canvas_w, 3), COL_BG, dtype=np.uint8)
        main_view = frame.copy()
        
        # --- LEFT MAIN feed rendering logic ---
        is_emergency = False
        for trk in tracks:
            color = self.select_action_color(trk.master_action)
            self.draw_corner_brackets(main_view, trk.bbox, color)
            
            # Draw semi-opaque background fill over bounding footprints
            overlay = main_view.copy()
            cv2.rectangle(overlay, (trk.bbox[0], trk.bbox[1]), (trk.bbox[2], trk.bbox[3]), color, -1)
            cv2.addWeighted(overlay, 0.12, main_view, 0.88, 0, dst=main_view)

            # --- ENLARGED TELEMETRY LABELS OVER BOUNDING BOXES ---
            label_str = f"ID: {trk.id} {trk.label.upper()} | {trk.distance_m:.1f}m"
            action_str = f"STATE: {trk.master_action}"
            
            # Font parameters scaled up to 0.65 scale and 2 thickness for superior contrast
            cv2.putText(main_view, label_str, (trk.bbox[0], max(trk.bbox[1] - 26, 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            cv2.putText(main_view, action_str, (trk.bbox[0], max(trk.bbox[1] - 6, 42)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            
            if trk.master_action == "EMERGENCY_BRAKE":
                is_emergency = True

        # Render Flashing Overhead Bar if Emergency Condition flags match
        if is_emergency and (global_frame_count // 3) % 2 == 0:
            overlay_bar = main_view.copy()
            cv2.rectangle(overlay_bar, (0, 0), (main_view.shape[1], 45), COL_RED, -1)
            cv2.addWeighted(overlay_bar, 0.6, main_view, 0.4, 0, dst=main_view)
            cv2.putText(main_view, "CRITICAL THREAT: IMPACT IMMINENT", (main_view.shape[1]//2 - 180, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)

        cv2.putText(main_view, "MAIN CAMERA FEED :: LANE-FORWARD SCANNER", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CYAN, 1, cv2.LINE_AA)
        main_resized = cv2.resize(main_view, (CFG.main_w, CFG.main_h), interpolation=cv2.INTER_NEAREST)
        canvas[0:CFG.main_h, 0:CFG.main_w] = main_resized

        # --- RIGHT TELEMETRY CONTROL DISPLAY PANEL ---
        panel = np.full((CFG.canvas_h, CFG.panel_w, 3), COL_PANEL_BG, dtype=np.uint8)
        
        cv2.putText(panel, "ADAS COCKPIT SYSTEM ENGINE", (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.55, COL_CYAN, 1, cv2.LINE_AA)
        cv2.line(panel, (20, 48), (CFG.panel_w - 20, 48), COL_CYAN_DIM, 1, cv2.LINE_AA)

        # Status block card layout
        self.draw_glass_panel(panel, 20, 60, CFG.panel_w - 40, 75)
        cv2.putText(panel, "PERCEPTION INTERFACE STATUS", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, "STRICT DECISION-LEVEL FUSION : ACTIVE", (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GREEN, 1, cv2.LINE_AA)
        mode_str = "LOCKED (Neural MiDaS Engine)" if midas_on else "PROJECTION FALLBACK MODEL"
        cv2.putText(panel, f"LiDAR MODE: {mode_str}", (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_CYAN if midas_on else COL_AMBER, 1, cv2.LINE_AA)

        # Session hazard counter instrument card
        self.draw_glass_panel(panel, 20, 148, CFG.panel_w - 40, 70)
        cv2.putText(panel, "SESSION HAZARD COUNT", (30, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
        pothole_ct = hazard_counts.get("pothole", 0)
        hump_ct = hazard_counts.get("hump", 0)
        cv2.putText(panel, f"POTHOLES: {pothole_ct}", (30, 193), cv2.FONT_HERSHEY_TRIPLEX, 0.55, COL_WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, f"HUMPS: {hump_ct}", (30, 213), cv2.FONT_HERSHEY_TRIPLEX, 0.55, COL_WHITE, 1, cv2.LINE_AA)

        # Unified Proximity Range evaluation block maps
        self.draw_glass_panel(panel, 20, 230, CFG.panel_w - 40, 115)
        cv2.putText(panel, "FUSED RANGE VALUE PROTOCOLS TO TARGET", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
        if tracks:
            nr = tracks[0]
            act_col = self.select_action_color(nr.master_action)
            cv2.putText(panel, f"{nr.distance_m:.1f} m", (30, 298), cv2.FONT_HERSHEY_TRIPLEX, 1.2, act_col, 3, cv2.LINE_AA)
            cv2.putText(panel, f"OBJ: {nr.label.upper()} | MATCH AGREEMENT: {nr.agreement} | CONF: {nr.fusion_confidence*100:.1f}%", (30, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.38, act_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "--- M", (30, 298), cv2.FONT_HERSHEY_TRIPLEX, 1.2, COL_TEXT_DIM, 2, cv2.LINE_AA)
            cv2.putText(panel, "SYSTEM MONITOR TRACKS: CLEAR SURFACE PATHS", (30, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1, cv2.LINE_AA)

        # Dynamic Driver Action Directive
        self.draw_glass_panel(panel, 20, 358, CFG.panel_w - 40, 85)
        cv2.putText(panel, "RECOMMENDED DRIVER ACTION", (30, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)

        action_str = "CLEAR ROAD"
        ev_color = COL_GREEN
        if tracks:
            nr = tracks[0]
            if nr.master_action == "EMERGENCY_BRAKE":
                if nr.lane_side == "LEFT":
                    action_str, ev_color = "TURN RIGHT", COL_AMBER
                elif nr.lane_side == "RIGHT":
                    action_str, ev_color = "TURN LEFT", COL_AMBER
                else:
                    action_str, ev_color = "BRAKE", COL_RED
            elif nr.master_action == "SLOW_DOWN":
                action_str, ev_color = "SLOW DOWN", COL_AMBER
        cv2.putText(panel, action_str, (30, 420), cv2.FONT_HERSHEY_TRIPLEX, 0.9, ev_color, 2, cv2.LINE_AA)

        # Rolling tracking history queues cards
        log_start_y = 455
        self.draw_glass_panel(panel, 20, log_start_y, CFG.panel_w - 40, 220)
        cv2.putText(panel, "ACTIVE SYSTEM TARGET METRIC PROFILE RECORDS LOG", (30, log_start_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
        
        recent_log = list(history_queue)[-3:][::-1]
        step_offset_y = log_start_y + 55
        if not recent_log:
            cv2.putText(panel, "System tracking queues are empty.", (30, step_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_TEXT_DIM, 1, cv2.LINE_AA)
        else:
            for idx, entry in enumerate(recent_log):
                log_col = self.select_action_color(entry["action"])
                meta_l1 = f"[{idx+1}] ID:{entry['id']} {entry['label'].upper()} -> {entry['action']}"
                meta_l2 = f"    METERS: {entry['dist']:.1f}m | CONF: {entry['conf']*100:.0f}% | AGREE: {entry['agree']}"
                cv2.putText(panel, meta_l1, (30, step_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, log_col, 1, cv2.LINE_AA)
                cv2.putText(panel, meta_l2, (30, step_offset_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
                step_offset_y += 50

        cv2.putText(panel, f"SYS_FRAME_IDX: {global_frame_count:05d}", (20, CFG.canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_TEXT_DIM, 1, cv2.LINE_AA)
        canvas[0:CFG.canvas_h, 0:CFG.main_w] = main_resized
        canvas[0:CFG.canvas_h, CFG.main_w:CFG.canvas_w] = panel
        
        # Draw explicit demarcation panel interface splitting boundary
        cv2.line(canvas, (CFG.main_w, 0), (CFG.main_w, CFG.canvas_h), COL_CYAN_DIM, 2, cv2.LINE_AA)
        return canvas
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    