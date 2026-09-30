"""
Web Dashboard for Object Detection & Deep SORT Tracking (Task 4)
Run with: streamlit run app.py
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector.yolo_detector import YoloDetector
from tracker.yolo_tracker import YoloTracker
from analytics.line_counter import LineCounter
from analytics.heatmap import MotionHeatmap
from utils.draw_utils import draw_bounding_box, draw_trajectories, draw_hud

# Page configuration
st.set_page_config(
    page_title="AI Object Detection & Tracking Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .stMetric label { color: #58a6ff !important; font-weight: 600; }
    .stMetric div[data-testid="stMetricValue"] { color: #00f0ff !important; font-size: 1.8rem; }
    .css-1544g2n { padding: 1.5rem; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🎯 AI Object Detection & Multi-Object Tracking Studio")
    st.caption("Powered by Ultralytics YOLOv8/11 & Deep SORT Re-ID Tracking Engine")

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")
    
    source_type = st.sidebar.radio("Input Source", ["Live Webcam", "Upload Video File"])
    
    tracker_choice = st.sidebar.selectbox(
        "Tracking Algorithm",
        ["deepsort", "sort", "bytetrack", "botsort"],
        index=0,
        help="Deep SORT uses deep appearance Re-ID embeddings. SORT uses spatial Kalman + IoU."
    )
    
    model_choice = st.sidebar.selectbox(
        "YOLO Model Architecture",
        ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"],
        index=0
    )
    
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.15, 0.90, 0.35, 0.05)
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)
    
    target_classes = st.sidebar.multiselect(
        "Filter Specific Categories",
        ["person", "car", "motorcycle", "bus", "truck", "traffic light", "dog", "cat", "backpack", "bottle", "chair", "laptop", "cell phone", "book"],
        default=[]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analytics & Visuals")
    mirror_camera = st.sidebar.checkbox("Mirror Camera (Selfie View)", value=(source_type == "Live Webcam"))
    enable_trails = st.sidebar.checkbox("Motion Trajectory Trails", value=True)
    enable_tripwire = st.sidebar.checkbox("Virtual Tripwire Counter", value=True)
    enable_heatmap = st.sidebar.checkbox("Spatial Motion Heatmap", value=False)
    
    # Initialize components
    detector = YoloDetector(
        model_name=model_choice,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        target_classes=target_classes if target_classes else None
    )
    tracker = YoloTracker(tracker_type=tracker_choice, yolo_detector=detector)
    line_counter = LineCounter() if enable_tripwire else None
    heatmap = MotionHeatmap() if enable_heatmap else None

    # Video Source selection
    video_source = None
    if source_type == "Live Webcam":
        cam_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        video_source = cam_index
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_source = tfile.name

    # Metric placeholders
    col1, col2, col3, col4 = st.columns(4)
    metric_fps = col1.empty()
    metric_active = col2.empty()
    metric_total = col3.empty()
    metric_crossed = col4.empty()

    # Video display placeholder
    video_placeholder = st.empty()
    chart_placeholder = st.sidebar.empty()

    start_btn = st.sidebar.button("▶️ Start Tracking", use_container_width=True)
    stop_btn = st.sidebar.button("⏹️ Stop Stream", use_container_width=True)

    if start_btn and video_source is not None:
        backend = cv2.CAP_DSHOW if (sys.platform.startswith("win") and isinstance(video_source, int)) else cv2.CAP_ANY
        cap = cv2.VideoCapture(video_source, backend)
        
        if not cap.isOpened():
            st.error(f"Could not open video source: {video_source}")
            return

        trajectories = {}
        fps_history = []
        frame_count = 0
        prev_time = time.perf_counter()
        
        # Session log data
        log_rows = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if mirror_camera:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(fps)
            avg_fps = sum(fps_history[-15:]) / len(fps_history[-15:])

            # Tracking update
            tracks = tracker.track(frame)
            active_ids = set()

            # Update trajectories
            for trk in tracks:
                tid = trk["id"]
                active_ids.add(tid)
                box = trk["bbox"]
                cx = int((box[0] + box[2]) / 2.0)
                cy = int((box[1] + box[3]) / 2.0)
                if tid not in trajectories:
                    trajectories[tid] = []
                trajectories[tid].append((cx, cy))
                if len(trajectories[tid]) > 35:
                    trajectories[tid].pop(0)

                # Log
                log_rows.append({
                    "Frame": frame_count,
                    "Track_ID": tid,
                    "Class": trk.get("cls_name", "object"),
                    "Confidence": trk.get("conf") or 0.0,
                    "CX": cx,
                    "CY": cy
                })

            # Tripwire
            if enable_tripwire and line_counter:
                line_counter.update(tracks)

            # Heatmap
            if enable_heatmap and heatmap:
                heatmap.update(frame.shape, tracks)

            # Render
            annotated = frame.copy()
            if enable_heatmap and heatmap:
                annotated = heatmap.apply(annotated)

            if enable_trails:
                draw_trajectories(annotated, trajectories)

            if enable_tripwire and line_counter:
                line_counter.draw(annotated)

            for trk in tracks:
                draw_bounding_box(
                    annotated,
                    trk["bbox"],
                    trk["id"],
                    trk["cls_name"],
                    trk.get("conf"),
                    trk.get("velocity")
                )

            draw_hud(
                annotated,
                fps=avg_fps,
                active_tracks=tracks,
                total_count=tracker.get_total_unique_count(),
                tracker_name=tracker_choice.upper(),
                latency_ms=detector.last_inference_ms + tracker.last_track_ms
            )

            # Convert BGR to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update live metrics
            metric_fps.metric("⚡ Real-time FPS", f"{avg_fps:.1f}")
            metric_active.metric("🎯 Active Objects", len(tracks))
            metric_total.metric("📈 Total Seen", tracker.get_total_unique_count())
            in_c = line_counter.in_count if line_counter else 0
            out_c = line_counter.out_count if line_counter else 0
            metric_crossed.metric("🚪 Crossed (IN / OUT)", f"{in_c} / {out_c}")

            # Update class chart in sidebar every 15 frames
            if frame_count % 15 == 0 and tracks:
                counts = {}
                for t in tracks:
                    cname = t.get("cls_name", "object")
                    counts[cname] = counts.get(cname, 0) + 1
                df_chart = pd.DataFrame(list(counts.items()), columns=["Class", "Count"]).set_index("Class")
                chart_placeholder.bar_chart(df_chart)

        cap.release()
        st.success("Session completed!")

        # Download CSV
        if log_rows:
            df_log = pd.DataFrame(log_rows)
            csv_data = df_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session Analytics CSV",
                data=csv_data,
                file_name="tracking_session_data.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
