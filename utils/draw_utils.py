"""
Advanced Holographic UI & HUD Visualization Engine.
Features:
- Glassmorphic dark cyber HUD with real-time telemetry
- High-contrast glowing corner brackets & centroid radar lock
- Velocity motion vector arrows
- Multi-point fading gradient trajectory trails
- High-visibility ID and confidence badge rendering
"""

import cv2
import numpy as np


def get_id_color(track_id: int):
    """
    Generate vibrant, high-contrast, visually distinct colors.
    Uses golden ratio hue spacing with maximum saturation.
    """
    golden_ratio = 0.618033988749895
    h = int(((int(track_id) * golden_ratio) % 1.0) * 180)
    s = 230
    v = 245
    hsv_pixel = np.uint8([[[h, s, v]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr_pixel)


def draw_hud(
    frame,
    fps: float,
    active_tracks: list,
    total_count: int,
    tracker_name: str,
    latency_ms: float = 0.0,
    notification: str = None,
    paused: bool = False
):
    """
    Draw a sleek holographic HUD telemetry dashboard in the top-left corner.
    """
    h, w = frame.shape[:2]

    # Category frequency histogram
    class_counts = {}
    for trk in active_tracks:
        cname = trk.get("cls_name", "object")
        class_counts[cname] = class_counts.get(cname, 0) + 1

    panel_w = 400
    panel_h = 145 if len(class_counts) == 0 else min(190, 145 + 26)

    # 1. Glassmorphism dark backing
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (14 + panel_w, 14 + panel_h), (16, 16, 20), -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Tech borders and corner glowing brackets
    cv2.rectangle(frame, (14, 14), (14 + panel_w, 14 + panel_h), (50, 50, 65), 1)
    # Neon cyan corner highlights
    cv2.line(frame, (14, 14), (14 + 60, 14), (0, 240, 255), 2)
    cv2.line(frame, (14, 14), (14, 14 + 60), (0, 240, 255), 2)
    cv2.line(frame, (14 + panel_w - 30, 14), (14 + panel_w, 14), (0, 240, 255), 2)
    cv2.line(frame, (14 + panel_w, 14), (14 + panel_w, 14 + 30), (0, 240, 255), 2)

    # 3. Header title & active radar pulse dot
    display_title = "AI OBJECT DETECTION & TRACKING"
    cv2.putText(frame, display_title, (28, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)

    status_color = (0, 165, 255) if paused else (0, 255, 120)
    cv2.circle(frame, (380, 36), 5, status_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (380, 36), 8, status_color, 1, cv2.LINE_AA)

    # 4. Row 1: Tracker Algorithm & Real-time FPS + Latency
    fps_color = (50, 255, 120) if fps >= 25 else (0, 220, 255) if fps >= 15 else (0, 100, 255)
    cv2.putText(frame, f"Algorithm: {tracker_name}", (28, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (230, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"({latency_ms:.0f}ms)", (315, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 175), 1, cv2.LINE_AA)

    # 5. Row 2: Target Counters
    cv2.putText(frame, f"Active Targets: {len(active_tracks)}", (28, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total Seen: {total_count}", (230, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 220, 130), 1, cv2.LINE_AA)

    # 6. Row 3: Category breakdown badges
    if class_counts:
        summary_str = " | ".join([f"{k}: {v}" for k, v in list(class_counts.items())[:4]])
        cv2.putText(frame, summary_str, (28, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 255, 200), 1, cv2.LINE_AA)
        ctrl_y = 148
    else:
        ctrl_y = 122

    # 7. Hotkey Hint Bar
    cv2.putText(frame, "[Q] Quit  [P] Pause  [F] Mirror  [T] Trails  [L] Line  [M] Tracker", (28, ctrl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (145, 145, 155), 1, cv2.LINE_AA)

    # 8. Notification Banner
    if notification:
        notif_w = min(w - 40, len(notification) * 11 + 45)
        notif_x = (w - notif_w) // 2
        overlay_notif = frame.copy()
        cv2.rectangle(overlay_notif, (notif_x, h - 65), (notif_x + notif_w, h - 25), (15, 15, 20), -1)
        cv2.addWeighted(overlay_notif, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (notif_x, h - 65), (notif_x + notif_w, h - 25), (0, 240, 255), 1)
        cv2.putText(frame, notification, (notif_x + 18, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 9. Pause Watermark
    if paused:
        overlay_p = frame.copy()
        cv2.rectangle(overlay_p, (0, 0), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay_p, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "-- STREAM PAUSED --", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'P' or SPACE to resume", (w // 2 - 130, h // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_trajectories(frame, trajectories: dict, max_points: int = 35):
    """
    Draw motion trajectory trails that dynamically glow and taper off.
    """
    for track_id, points in trajectories.items():
        if len(points) < 2:
            continue
        color = get_id_color(track_id)
        pts = list(points)
        num_pts = len(pts)

        for i in range(1, num_pts):
            progress = i / num_pts
            thickness = max(1, int(1 + progress * 3))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


def draw_bounding_box(frame, bbox, track_id: int, label: str, conf: float = None, velocity = None):
    """
    Draw high-tech bounding box with:
    - Glowing corner brackets
    - Radar centroid tracking dot
    - Directional velocity vector arrow
    - Sleek high-contrast ID + Confidence badge
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = get_id_color(track_id)

    # 1. Main bounding box outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 2. Glowing corner brackets
    corner_len = min(18, max(6, (x2 - x1) // 5), max(6, (y2 - y1) // 5))
    accent_color = (255, 255, 255)
    t = 2

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), accent_color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), accent_color, t, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), accent_color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), accent_color, t, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), accent_color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), accent_color, t, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), accent_color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), accent_color, t, cv2.LINE_AA)

    # 3. Radar Centroid Dot
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, accent_color, 1, cv2.LINE_AA)

    # 4. Directional Velocity Vector Arrow
    if velocity is not None:
        vx, vy = velocity
        speed = np.hypot(vx, vy)
        if speed > 1.2:
            scale = min(28.0 / speed, 3.5)
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.35)

    # 5. High-contrast Badge
    if conf is not None and conf > 0:
        badge_text = f"#{track_id} {label.upper()} {conf*100:.0f}%"
    else:
        badge_text = f"#{track_id} {label.upper()}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    thickness = 1
    (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

    bx1 = max(0, x1)
    by1 = max(0, y1 - th - 8)
    bx2 = bx1 + tw + 12
    by2 = y1

    # Badge background
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    
    # Badge text
    text_color = (15, 15, 20) if sum(color) > 370 else (255, 255, 255)
    cv2.putText(frame, badge_text, (bx1 + 6, by2 - 5),
                font, font_scale, text_color, thickness, cv2.LINE_AA)
