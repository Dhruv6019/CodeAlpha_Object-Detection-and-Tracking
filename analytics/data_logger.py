"""
Telemetry & Data Analytics Logger.
Logs tracking metadata, class occurrences, average speeds, and session statistics to CSV and JSON.
"""

import os
import csv
import json
import time
from datetime import datetime
import numpy as np


class AnalyticsLogger:
    def __init__(self, log_dir="analytics_output"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"tracking_log_{timestamp}.csv")
        self.json_path = os.path.join(self.log_dir, f"session_summary_{timestamp}.json")
        
        self.records = {}
        self.start_time = time.time()
        self.total_frames = 0
        
        # Initialize CSV file with headers
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Track_ID", "Class", "Confidence",
                "Center_X", "Center_Y", "BBox_W", "BBox_H", "Speed_px_per_frame"
            ])

    def log_frame(self, tracks):
        """Log per-frame active tracks."""
        self.total_frames += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        rows = []
        for trk in tracks:
            tid = trk["id"]
            box = trk["bbox"]
            w = max(1.0, box[2] - box[0])
            h = max(1.0, box[3] - box[1])
            cx = box[0] + w / 2.0
            cy = box[1] + h / 2.0
            vx, vy = trk.get("velocity", (0.0, 0.0))
            speed = float(np.hypot(vx, vy))
            conf = float(trk.get("conf") or 0.0)
            cname = trk.get("cls_name", "object")

            rows.append([
                now_str, tid, cname, f"{conf:.2f}",
                f"{cx:.1f}", f"{cy:.1f}", f"{w:.1f}", f"{h:.1f}", f"{speed:.2f}"
            ])

            # Maintain aggregate statistics per track ID
            if tid not in self.records:
                self.records[tid] = {
                    "class": cname,
                    "first_seen": now_str,
                    "last_seen": now_str,
                    "frames_tracked": 0,
                    "speeds": [],
                    "confs": []
                }
            self.records[tid]["last_seen"] = now_str
            self.records[tid]["frames_tracked"] += 1
            self.records[tid]["speeds"].append(speed)
            if conf > 0:
                self.records[tid]["confs"].append(conf)

        if rows:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    def save_summary(self, tracker_name="DEEPSORT", line_in=0, line_out=0):
        """Export session summary JSON report."""
        elapsed = max(1.0, time.time() - self.start_time)
        avg_fps = self.total_frames / elapsed

        class_distribution = {}
        for trk_data in self.records.values():
            c = trk_data["class"]
            class_distribution[c] = class_distribution.get(c, 0) + 1

        summary = {
            "session_start": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_sec": round(elapsed, 2),
            "total_frames_processed": self.total_frames,
            "average_fps": round(avg_fps, 2),
            "tracker_algorithm": tracker_name,
            "total_unique_objects": len(self.records),
            "class_distribution": class_distribution,
            "tripwire_line_crossing": {
                "in_count": line_in,
                "out_count": line_out,
                "total_crossed": line_in + line_out
            },
            "csv_log_file": self.csv_path
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return self.json_path, self.csv_path
