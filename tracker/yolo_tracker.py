"""
Unified Multi-Object Tracking Coordinator.
Supports:
1. 'deepsort': Deep SORT (Deep CNN Re-ID embeddings + Cosine Distance Matching + Kalman Filter)
2. 'sort': Enhanced Class-Aware SORT (Kalman Filter + Hungarian Matching + EMA smoothing)
3. 'bytetrack': ByteTrack (Two-stage association for occluded objects)
4. 'botsort': BoT-SORT (Camera motion compensation + ReID)
"""

import time
import numpy as np
from .sort import Sort
from .deep_sort import DeepSort


class YoloTracker:
    def __init__(
        self,
        tracker_type: str = "deepsort",
        yolo_detector = None,
        max_age: int = 30,
        min_hits: int = 1,
        iou_thresh: float = 0.3,
        max_cosine_dist: float = 0.35
    ):
        self.tracker_type = tracker_type.lower()
        self.detector = yolo_detector
        self.total_unique_ids = set()
        self.last_track_ms = 0.0

        if self.tracker_type == "deepsort":
            self.sort_tracker = None
            self.deep_sort_tracker = DeepSort(
                max_cosine_dist=max_cosine_dist,
                max_age=max_age,
                min_hits=min_hits,
                iou_thresh=iou_thresh
            )
        elif self.tracker_type == "sort":
            self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
            self.deep_sort_tracker = None
        elif self.tracker_type in ["bytetrack", "botsort"]:
            self.sort_tracker = None
            self.deep_sort_tracker = None
        else:
            raise ValueError(f"Unsupported tracker '{tracker_type}'. Choose 'deepsort', 'sort', 'bytetrack', or 'botsort'.")

    def track(self, frame):
        """
        Process a single frame and return tracked objects.
        Returns:
            tracks: list of dicts:
                [{
                    "bbox": [x1, y1, x2, y2],
                    "id": int,
                    "cls_name": str,
                    "conf": float,
                    "velocity": (vx, vy)
                }, ...]
        """
        t0 = time.perf_counter()

        if self.tracker_type == "deepsort":
            # 1. Run YOLO detector
            boxes, confs, cls_ids, cls_names = self.detector.detect(frame)
            if len(boxes) == 0:
                dets = np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.column_stack([boxes, confs])

            # 2. Deep SORT update with deep feature re-identification
            tracks = self.deep_sort_tracker.update(frame, dets, class_names=cls_names)
            for t in tracks:
                self.total_unique_ids.add(t["id"])
            self.last_track_ms = (time.perf_counter() - t0) * 1000.0
            return tracks

        elif self.tracker_type == "sort":
            # 1. Run YOLO detector
            boxes, confs, cls_ids, cls_names = self.detector.detect(frame)
            if len(boxes) == 0:
                dets = np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.column_stack([boxes, confs])

            # 2. Update Class-Aware SORT
            raw_tracks = self.sort_tracker.update(dets, class_ids=cls_ids, class_names=cls_names)
            tracks = []
            for t in raw_tracks:
                track_id = int(t["id"])
                self.total_unique_ids.add(track_id)
                tracks.append({
                    "bbox": t["bbox"],
                    "id": track_id,
                    "cls_name": t.get("cls_name", "object"),
                    "conf": t.get("conf"),
                    "velocity": t.get("velocity", (0.0, 0.0))
                })
            self.last_track_ms = (time.perf_counter() - t0) * 1000.0
            return tracks

        else:
            # Native YOLO tracker (ByteTrack or BoT-SORT)
            tracker_cfg = "bytetrack.yaml" if self.tracker_type == "bytetrack" else "botsort.yaml"
            results = self.detector.model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=self.detector.conf_thresh,
                iou=self.detector.iou_thresh,
                imgsz=self.detector.imgsz,
                device=self.detector.device,
                verbose=False
            )[0]

            tracks = []
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
                    if self.detector.target_class_ids is not None and cid not in self.detector.target_class_ids:
                        continue
                    track_id = int(tid)
                    self.total_unique_ids.add(track_id)
                    tracks.append({
                        "bbox": box.tolist(),
                        "id": track_id,
                        "cls_name": self.detector.names.get(cid, "object"),
                        "conf": float(conf),
                        "velocity": (0.0, 0.0)
                    })
            self.last_track_ms = (time.perf_counter() - t0) * 1000.0
            return tracks

    def get_total_unique_count(self) -> int:
        if self.tracker_type == "deepsort":
            return self.deep_sort_tracker.get_total_unique_count()
        return len(self.total_unique_ids)
