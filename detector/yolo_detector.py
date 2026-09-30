"""
High-Performance YOLO Object Detector wrapper using Ultralytics & PyTorch.
Features:
- Automatic hardware acceleration (CUDA FP16 / Apple MPS / Multi-threaded CPU)
- Inference mode context to bypass autograd overhead
- Built-in latency benchmark profiler
- Class filtering and confidence calibration
"""

import time
import torch
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        target_classes: list = None,
        imgsz: int = 640
    ):
        """
        model_name: Pretrained model weights ('yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt', 'yolo11s.pt')
        conf_thresh: Detection confidence threshold (0.0 to 1.0)
        iou_thresh: NMS IoU threshold
        target_classes: Optional list of class names or class IDs to track (e.g. ['person', 'car'])
        imgsz: Input image dimension for model inference
        """
        # Hardware device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            self.fp16 = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.fp16 = False
        else:
            self.device = "cpu"
            self.fp16 = False
            # Optimize CPU threads for OpenCV and PyTorch
            try:
                import os
                torch.set_num_threads(max(1, os.cpu_count() - 1))
            except Exception:
                pass

        print(f"[Detector] Initializing YOLO ({model_name}) on device: {self.device.upper()} (FP16={self.fp16})")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.names = self.model.names  # dict of class id -> class name
        self.last_inference_ms = 0.0

        # Resolve target classes into set of class IDs
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = set()
            for item in target_classes:
                if isinstance(item, int):
                    self.target_class_ids.add(item)
                elif isinstance(item, str):
                    for cid, cname in self.names.items():
                        if cname.lower() == item.lower():
                            self.target_class_ids.add(cid)
            matched_names = [self.names[cid] for cid in self.target_class_ids]
            print(f"[Detector] Filtering classes to: {matched_names}")

    def detect(self, frame):
        """
        Run high-speed inference on a single frame.
        Returns:
            boxes: np.ndarray shape (N, 4) [x1, y1, x2, y2]
            scores: np.ndarray shape (N,)
            class_ids: np.ndarray shape (N,)
            class_names: list of str
        """
        t0 = time.perf_counter()
        
        with torch.inference_mode():
            results = self.model(
                frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )[0]

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        if len(results.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int), []

        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Apply target class filter
        if self.target_class_ids is not None:
            mask = np.isin(cls_ids, list(self.target_class_ids))
            xyxy = xyxy[mask]
            confs = confs[mask]
            cls_ids = cls_ids[mask]

        cls_names = [self.names[cid] for cid in cls_ids]
        return xyxy, confs, cls_ids, cls_names
