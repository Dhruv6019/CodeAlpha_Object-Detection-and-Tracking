# AI Object Detection & Multi-Object Tracking Studio (Task 4)

A complete, production-grade computer vision system featuring **YOLOv8/YOLO11**, **Deep SORT** (Deep Visual Re-ID + Kalman Filter Motion Tracking), **Class-Aware SORT**, **ByteTrack**, and **BoT-SORT**. Equipped with real-time video analytics (virtual tripwire line counting, motion heatmaps, and telemetry data logging) and accessible via both a **High-Tech CLI Pipeline** and an **Interactive Streamlit Web Dashboard**.

---

## 🌟 Core Features & Modules

- **Deep Learning Object Detection**:
  - Pretrained **YOLOv8** / **YOLO11** (COCO 80 categories).
  - Hardware-accelerated (CUDA FP16 / Multi-threaded CPU with `torch.inference_mode()`).
  - Tuned confidence (`0.35`) and NMS IoU (`0.45`) thresholds.
- **Multi-Object Tracking (MOT) Suite**:
  - **Deep SORT (Default)**: Deep CNN visual appearance embeddings (576-dim L2-normalized vectors), nearest-neighbor cosine distance gallery, and Hungarian assignment.
  - **Class-Aware SORT**: High-speed Kalman filter tracking with strict category-gated Hungarian matching.
  - **ByteTrack**: Two-stage association recovering low-confidence occluded objects.
  - **BoT-SORT**: Camera motion compensation and Re-ID.
  - *Switchable dynamically on the fly with the `M` key.*
- **Zero-Latency Threaded Video Streaming**:
  - Dedicated background thread (`ThreadedCamera`) reading at high FPS without USB buffer lag.
- **Advanced Video Analytics**:
  - **Virtual Tripwire Line Counter**: Bidirectional entry/exit counting (`IN` / `OUT`) with class breakdown.
  - **Spatial Density Heatmap**: Live motion accumulation visualization of dwell areas and traffic corridors.
  - **Automated Telemetry Data Logging**: Automatically records every frame event to `analytics_output/tracking_log_*.csv` and exports `session_summary_*.json`.
- **Dual User Interface**:
  - **Holographic CLI Interface**: Glowing corner brackets, radar centroid dots, velocity vectors, and glassmorphic HUD.
  - **Interactive Web Studio**: Browser-based Streamlit dashboard (`app.py`) with real-time controls, analytics charts, and CSV download.

---

## 📁 Project Structure

```
Task-4/
├── main.py                 # Core CLI application loop with real-time visual HUD
├── app.py                  # Streamlit Web Dashboard for interactive browser control
├── requirements.txt        # Package dependencies
├── PROJECT_REPORT.md       # In-depth academic & technical research report with citations
│
├── detector/
│   ├── __init__.py
│   └── yolo_detector.py    # YOLOv8/11 detector wrapper with hardware acceleration
│
├── tracker/
│   ├── __init__.py
│   ├── deep_sort.py        # Deep SORT (Deep CNN Re-ID + Cosine Gallery + Kalman)
│   ├── sort.py             # Class-Aware SORT (Hungarian + EMA smoothing)
│   └── yolo_tracker.py     # Unified multi-tracker coordinator
│
├── analytics/
│   ├── __init__.py
│   ├── line_counter.py     # Virtual tripwire line-crossing counter
│   ├── heatmap.py          # Spatial motion density heatmap generator
│   └── data_logger.py      # Telemetry CSV & JSON logger
│
└── utils/
    ├── __init__.py
    ├── camera.py           # Threaded asynchronous camera capture
    └── draw_utils.py       # Holographic HUD, velocity vectors & glowing brackets
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Launch the Interactive Web Studio
```bash
streamlit run app.py
```
*Opens an interactive browser interface with real-time sliders, category filters, analytics charts, and instant CSV downloads.*

---

### Option B: Run the High-Performance CLI Pipeline

```bash
# 1. Default run on live webcam (Deep SORT enabled):
python main.py

# 2. Toggle specific analytics on launch:
python main.py --source 0 --tripwire --heatmap

# 3. Filter specific classes:
python main.py --classes person "cell phone" laptop

# 4. Run on a video file and save output:
python main.py --source "path/to/video.mp4" --save-output "annotated_output.mp4"

# 5. Switch tracking engine directly:
python main.py --tracker deepsort    # Deep SORT (Deep CNN Re-ID)
python main.py --tracker bytetrack   # ByteTrack
python main.py --tracker botsort     # BoT-SORT
python main.py --tracker sort        # Classic Class-Aware SORT
```

---

## 🎮 Interactive Keyboard Shortcuts (CLI Mode)

| Key | Action |
|:---:|:---|
| `Q` / `ESC` | Quit cleanly and export session telemetry |
| `P` / `SPACE` | Pause / Resume video playback |
| `F` | Toggle camera horizontal mirror view (Selfie Mode) ON / OFF |
| `T` | Toggle motion trajectory trails ON / OFF |
| `L` | Toggle virtual tripwire line counter ON / OFF |
| `H` | Toggle spatial motion density heatmap ON / OFF |
| `S` | Capture instant high-resolution screenshot snapshot |
| `M` | Cycle active tracker live on screen (**Deep SORT** ➔ **ByteTrack** ➔ **BoT-SORT** ➔ **SORT**) |

---

## 📊 Analytics Outputs

When you run `python main.py`, analytics are automatically saved in `analytics_output/`:
- **`tracking_log_<timestamp>.csv`**: Granular per-frame telemetry records including timestamp, track ID, category, confidence, $(x, y)$ coordinates, bounding box dimensions, and calculated speed.
- **`session_summary_<timestamp>.json`**: High-level executive summary including session duration, average FPS, total unique objects tracked, tripwire counts (`IN` / `OUT`), and class distributions.

---

## 📚 Technical Report & Citations

For complete mathematical formulations (Kalman filter discrete state equations, Hungarian bipartite matching, deep appearance cosine metrics), algorithmic derivations, and benchmark comparison matrices across trackers, see [`PROJECT_REPORT.md`](file:///c:/xampp/htdocs/Task-4/PROJECT_REPORT.md).
