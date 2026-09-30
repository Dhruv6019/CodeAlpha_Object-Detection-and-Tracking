"""
High-Performance Real-Time Object Detection and Multi-Object Tracking Pipeline (Task 4)
Supports live webcam stream, IP camera/RTSP, or local video files.
Integrated with:
- Ultralytics YOLOv8 / YOLO11 Deep Learning Detection
- Deep SORT with Deep CNN Re-ID embeddings & Cosine Distance Matching
- Enhanced Class-Aware SORT
- ByteTrack & BoT-SORT Multi-Object Trackers
- Low-latency Threaded Camera reader (zero-lag MJPG)
- Virtual Tripwire Line-Crossing Counter (IN / OUT)
- Spatial Density Motion Heatmap
- Telemetry & Analytics Logger (CSV / JSON export)
"""

import os
import sys
import time
import warnings
import argparse
from datetime import datetime
from collections import deque

warnings.filterwarnings("ignore")

import cv2
import numpy as np

from detector.yolo_detector import YoloDetector
from tracker.yolo_tracker import YoloTracker
from analytics.line_counter import LineCounter
from analytics.heatmap import MotionHeatmap
from analytics.data_logger import AnalyticsLogger
from utils.draw_utils import draw_hud, draw_bounding_box, draw_trajectories
from utils.camera import ThreadedCamera


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Performance Real-Time Object Detection & Deep SORT Tracking"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index ('0', '1') or path to video file (default: '0')"
    )
    parser.add_argument(
        "--tracker", type=str, default="deepsort", choices=["deepsort", "sort", "bytetrack", "botsort"],
        help="Tracking algorithm: 'deepsort' (default), 'sort', 'bytetrack', or 'botsort'"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLO model weights (e.g. 'yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt')"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image resolution (default: 640)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Filter detection classes by name (e.g., --classes person car dog)"
    )
    parser.add_argument(
        "--save-output", type=str, default=None,
        help="Path to save annotated output video (e.g. 'output.mp4')"
    )
    parser.add_argument(
        "--tripwire", action="store_true", default=True,
        help="Enable virtual tripwire line-crossing counter (toggleable with 'L')"
    )
    parser.add_argument(
        "--heatmap", action="store_true", default=False,
        help="Enable spatial density motion heatmap (toggleable with 'H')"
    )
    parser.add_argument(
        "--no-trails", action="store_true",
        help="Disable trajectory trail rendering"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable automatic session analytics CSV logging"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Headless execution mode without GUI display window"
    )
    parser.add_argument(
        "--mirror", action="store_true", default=None,
        help="Mirror video stream horizontally (default: True for webcams)"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Disable camera mirroring"
    )
    return parser.parse_args()


def _is_black_frame(frame, threshold=5):
    """Check if frame is entirely black (camera blocked/driver issue)."""
    if frame is None or frame.size == 0:
        return True
    return float(np.mean(frame)) < threshold


def open_video_source(source_str: str):
    """
    Open video source:
    - Uses ThreadedCamera for webcams (0, 1) for zero-latency, high FPS capture
    - Uses standard VideoCapture for file playback
    """
    if source_str.isdigit():
        cam_idx = int(source_str)
        threaded_cam = ThreadedCamera(src=cam_idx, width=None, height=None, fps=None)
        if threaded_cam.isOpened():
            return threaded_cam, True
        # Fallback: try standard capture
        print(f"[Warning] ThreadedCamera failed for index {cam_idx}, falling back to cv2.VideoCapture.")
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, False
        print("[Error] Could not open webcam. Make sure a camera is connected.")
        return None, True

    # Video file path
    if not os.path.exists(source_str):
        print(f"\n[Error] Video file not found: '{source_str}'")
        print("  -> To use your live webcam, run: python main.py --source 0")
        print("  -> Or provide a valid path to an existing video file (.mp4, .avi, etc.).\n")
        sys.exit(1)

    return cv2.VideoCapture(source_str), False


def draw_camera_error(frame):
    """Draw a clear camera error overlay when webcam returns black frames."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Warning icon box
    bx, by, bw, bh = w//2 - 300, h//2 - 120, 600, 240
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 180), -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)

    cv2.putText(frame, "CAMERA DRIVER ERROR", (bx+60, by+50),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Camera returns black frames.", (bx+40, by+90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "FIX: Right-click FIX_CAMERA_RUN_AS_ADMIN.bat", (bx+20, by+120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "     -> Run as Administrator", (bx+20, by+148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "OR: Device Manager -> Cameras -> Disable -> Enable", (bx+20, by+176),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (bx+200, by+210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    print("=" * 65)
    print("   AI OBJECT DETECTION & DEEP SORT TRACKING PIPELINE")
    print("=" * 65)
    print(f" Source:           {'Webcam ' + args.source if args.source.isdigit() else args.source}")
    print(f" Model:            {args.model} (imgsz={args.imgsz})")
    print(f" Active Tracker:   {args.tracker.upper()}")
    print(f" Conf Threshold:   {args.conf}")
    print(f" Filtered Classes: {args.classes if args.classes else 'All COCO Classes (80 categories)'}")
    print(f" Trajectory Trails: {'Disabled' if args.no_trails else 'Enabled'}")
    print(f" Analytics Logger: {'Enabled' if not args.no_log else 'Disabled'}")
    print("=" * 65)

    # 1. Initialize Video Stream
    cap, is_threaded = open_video_source(args.source)
    if cap is None or not cap.isOpened():
        print(f"[Error] Failed to access video source: {args.source}")
        print("  - If using a webcam, check that your camera is connected and permitted.")
        print("  - If using a video file, verify the file path.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_in = fps_in if (fps_in and fps_in > 0 and fps_in <= 120) else 30.0
    mode_tag = "Threaded Zero-Lag" if is_threaded else "File Stream"
    print(f"[Info] Video stream active ({mode_tag}): {width}x{height} @ {fps_in:.1f} FPS")

    # 2. Output Video Recording
    writer = None
    if args.save_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_output)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_output, fourcc, fps_in, (width, height))
        print(f"[Info] Recording output stream to: {args.save_output}")

    # 3. Initialize YOLO Detector
    detector = YoloDetector(
        model_name=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        target_classes=args.classes,
        imgsz=args.imgsz
    )

    # 4. Initialize Multi-Object Tracker (Deep SORT by default)
    tracker_type = args.tracker
    tracker = YoloTracker(tracker_type=tracker_type, yolo_detector=detector)

    # 5. Initialize Analytics Modules
    line_counter = LineCounter() if args.tripwire else None
    heatmap = MotionHeatmap() if args.heatmap else None
    logger = AnalyticsLogger() if not args.no_log else None

    # Pre-init line counter with frame dimensions so update() works correctly
    if line_counter is not None:
        line_counter.auto_init(width, height)

    # State toggles
    trajectories = {}
    show_trails = not args.no_trails
    show_tripwire = args.tripwire
    show_heatmap = args.heatmap

    # Mirroring: enabled by default for webcam streams (index 0, 1...)
    if args.no_mirror:
        show_mirror = False
    elif args.mirror is not None and args.mirror:
        show_mirror = True
    else:
        show_mirror = args.source.isdigit()

    paused = False
    tracks = []  # Initialize to empty list to avoid NameError on first paused render

    # Telemetry
    notification_msg = f"Started: {tracker_type.upper()}"
    notification_expiry = time.time() + 2.5
    fps = 0.0
    prev_time = time.perf_counter()
    fps_history = deque(maxlen=20)
    frame_count = 0
    annotated = np.zeros((height, width, 3), dtype=np.uint8)  # Safety fallback frame

    window_name = f"Object Tracking Studio [{tracker_type.upper()}]"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    print("\n[Controls] [Q]: Exit | [P]: Pause | [F]: Mirror | [T]: Trails | [L]: Tripwire | [H]: Heatmap | [S]: Snapshot | [M]: Tracker\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if not is_threaded:
                        print("[Info] End of video stream reached.")
                        break
                    time.sleep(0.005)
                    continue

                # Mirror frame horizontally for natural selfie webcam view
                if show_mirror:
                    frame = cv2.flip(frame, 1)

                frame_count += 1
                curr_time = time.perf_counter()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_history.append(1.0 / dt)
                    fps = sum(fps_history) / len(fps_history)

                # Detect black camera frame (driver issue) — show error overlay
                if is_threaded and _is_black_frame(frame) and frame_count <= 60:
                    annotated = frame.copy()
                    draw_camera_error(annotated)
                    if not args.no_show:
                        cv2.imshow(window_name, annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                    continue

                # Run Detection + Tracking
                try:
                    tracks = tracker.track(frame)
                except Exception as e:
                    print(f"[Warning] Tracking error on frame {frame_count}: {e}")
                    tracks = []

                active_ids = set()

                # Update motion trajectory centers
                for trk in tracks:
                    tid = trk["id"]
                    active_ids.add(tid)
                    box = trk["bbox"]
                    cx = int((box[0] + box[2]) / 2.0)
                    cy = int((box[1] + box[3]) / 2.0)

                    if tid not in trajectories:
                        trajectories[tid] = deque(maxlen=35)
                    trajectories[tid].append((cx, cy))

                # Periodically purge dead trajectory buffers
                if frame_count % 30 == 0:
                    dead_ids = [oid for oid in list(trajectories.keys()) if oid not in active_ids]
                    for oid in dead_ids:
                        if len(trajectories[oid]) > 0:
                            trajectories[oid].popleft()
                        else:
                            del trajectories[oid]

                # Update Analytics
                if show_tripwire and line_counter is not None:
                    line_counter.update(tracks)

                if show_heatmap and heatmap is not None:
                    heatmap.update(frame.shape, tracks)

                if logger is not None:
                    logger.log_frame(tracks)

                # --- Render Visual Overlays ---
                annotated = frame.copy()

                # 1. Heatmap layer (bottom-most overlay)
                if show_heatmap and heatmap is not None:
                    annotated = heatmap.apply(annotated)

                # 2. Virtual tripwire line
                if show_tripwire and line_counter is not None:
                    line_counter.draw(annotated)

                # 3. Trajectory trails
                if show_trails:
                    draw_trajectories(annotated, trajectories)

                # 4. Bounding boxes & ID badges
                for trk in tracks:
                    draw_bounding_box(
                        annotated,
                        trk["bbox"],
                        trk["id"],
                        trk["cls_name"],
                        trk.get("conf"),
                        trk.get("velocity")
                    )

                # 5. Holographic HUD Telemetry
                current_time_sec = time.time()
                active_notif = notification_msg if current_time_sec < notification_expiry else None

                draw_hud(
                    annotated,
                    fps=fps,
                    active_tracks=tracks,
                    total_count=tracker.get_total_unique_count(),
                    tracker_name=tracker_type.upper(),
                    latency_ms=detector.last_inference_ms + tracker.last_track_ms,
                    notification=active_notif,
                    paused=False
                )

                if writer is not None:
                    writer.write(annotated)

            else:
                # Paused: re-draw last frame with HUD overlay
                paused_frame = annotated.copy()
                draw_hud(
                    paused_frame,
                    fps=fps,
                    active_tracks=tracks,
                    total_count=tracker.get_total_unique_count(),
                    tracker_name=tracker_type.upper(),
                    paused=True
                )
                annotated = paused_frame

            # --- Interactive GUI Keystrokes ---
            if not args.no_show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1 if not paused else 50) & 0xFF

                if key == ord('q') or key == 27:
                    print("[Info] Shutdown requested by user.")
                    break
                elif key == ord('p') or key == 32:
                    paused = not paused
                    notification_msg = "STREAM PAUSED" if paused else "STREAM RESUMED"
                    notification_expiry = time.time() + 2.0
                elif key == ord('f'):
                    show_mirror = not show_mirror
                    notification_msg = f"Mirror View: {'ENABLED' if show_mirror else 'DISABLED'}"
                    notification_expiry = time.time() + 1.8
                    print(f"[Info] Camera Mirroring: {'ENABLED' if show_mirror else 'DISABLED'}")
                elif key == ord('t'):
                    show_trails = not show_trails
                    notification_msg = f"Trails: {'ENABLED' if show_trails else 'DISABLED'}"
                    notification_expiry = time.time() + 1.8
                elif key == ord('l'):
                    show_tripwire = not show_tripwire
                    if show_tripwire and line_counter is None:
                        line_counter = LineCounter()
                        line_counter.auto_init(width, height)
                    notification_msg = f"Tripwire: {'ENABLED' if show_tripwire else 'DISABLED'}"
                    notification_expiry = time.time() + 1.8
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    if show_heatmap and heatmap is None:
                        heatmap = MotionHeatmap()
                    notification_msg = f"Heatmap: {'ENABLED' if show_heatmap else 'DISABLED'}"
                    notification_expiry = time.time() + 1.8
                elif key == ord('s'):
                    snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    notification_msg = f"Saved: {snap_name}"
                    notification_expiry = time.time() + 2.5
                    print(f"[Info] Captured snapshot: {snap_name}")
                elif key == ord('m'):
                    cycle = {
                        "deepsort": "bytetrack",
                        "bytetrack": "botsort",
                        "botsort": "sort",
                        "sort": "deepsort"
                    }
                    tracker_type = cycle.get(tracker_type, "deepsort")
                    tracker = YoloTracker(tracker_type=tracker_type, yolo_detector=detector)
                    cv2.setWindowTitle(window_name, f"Object Tracking Studio [{tracker_type.upper()}]")
                    notification_msg = f"Tracker: {tracker_type.upper()}"
                    notification_expiry = time.time() + 2.0
                    print(f"[Info] Switched tracker to: {tracker_type.upper()}")
                elif key == ord('r'):
                    # Reset trajectories
                    trajectories.clear()
                    notification_msg = "Trajectories Reset"
                    notification_expiry = time.time() + 1.8
            else:
                if frame_count % 30 == 0:
                    print(
                        f"[Frame {frame_count:04d}] FPS: {fps:4.1f} | "
                        f"Active: {len(tracks)} | Total: {tracker.get_total_unique_count()} | "
                        f"Latency: {detector.last_inference_ms + tracker.last_track_ms:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Info] Recorded video saved: {args.save_output}")
        if logger is not None:
            in_c = line_counter.in_count if line_counter else 0
            out_c = line_counter.out_count if line_counter else 0
            try:
                json_file, csv_file = logger.save_summary(
                    tracker_name=tracker_type.upper(), line_in=in_c, line_out=out_c
                )
                print(f"[Info] Session log CSV exported:  {csv_file}")
                print(f"[Info] Session report JSON saved: {json_file}")
            except Exception as e:
                print(f"[Warning] Could not save analytics: {e}")
        if not args.no_show:
            cv2.destroyAllWindows()
        print("[Info] Pipeline closed cleanly.")


if __name__ == "__main__":
    main()
