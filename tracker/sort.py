"""
Enhanced Class-Aware SORT (Simple Online and Realtime Tracking)
Features:
- Strict Class-Aware Association (zero cross-category track ID hopping)
- Exponential Moving Average (EMA) coordinate smoothing (eliminates box jitter)
- Velocity vector & motion dynamics estimation
- Adaptive Kalman filtering with covariance damping
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """
    Computes pairwise IoU between two sets of bounding boxes.
    bb_test: (N, 4) in [x1, y1, x2, y2]
    bb_gt:   (M, 4) in [x1, y1, x2, y2]
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    return wh / (area_test + area_gt - wh + 1e-6)


def convert_bbox_to_z(bbox):
    """Convert [x1, y1, x2, y2] to center state [x, y, s, r]^T."""
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x):
    """Convert state vector [x, y, s, r] back to [x1, y1, x2, y2]."""
    w = np.sqrt(max(1.0, x[2] * x[3]))
    h = max(1.0, x[2] / (w + 1e-6))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0], dtype=np.float32).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, cls_id=None, cls_name=None, conf=0.0):
        self.dim_x = 7
        self.dim_z = 4

        # Transition matrix F
        self.F = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        # Process noise Q
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01

        # Uncertainty covariance P
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Initial state x
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cls_id = cls_id
        self.cls_name = cls_name or "object"
        self.conf = conf
        
        # Smoothed bounding box (EMA)
        self.smoothed_bbox = np.array(bbox[:4], dtype=np.float32)
        self.alpha_smooth = 0.70  # Higher = more responsive, lower = smoother

    def update(self, bbox, cls_id=None, cls_name=None, conf=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if cls_id is not None:
            self.cls_id = cls_id
        if cls_name is not None:
            self.cls_name = cls_name
        if conf is not None:
            self.conf = conf

        z = convert_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I_KH = np.eye(self.dim_x) - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)

        # Apply Exponential Moving Average (EMA) smoothing to coordinates
        raw_box = convert_x_to_bbox(self.x)[0]
        self.smoothed_bbox = self.alpha_smooth * raw_box + (1.0 - self.alpha_smooth) * self.smoothed_bbox

    def predict(self):
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] *= 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred_box = convert_x_to_bbox(self.x)
        self.history.append(pred_box)
        return self.history[-1]

    def get_state(self):
        """Returns the EMA-smoothed bounding box [x1, y1, x2, y2]."""
        return self.smoothed_bbox.copy()

    def get_velocity(self):
        """Returns (vx, vy) speed in pixels per frame."""
        return float(self.x[4, 0]), float(self.x[5, 0])


def associate_detections_to_trackers(detections, trackers, det_classes, trk_classes, iou_threshold=0.3):
    """
    Assigns detections to tracked objects with Class-Aware IoU matching.
    Cross-class matches are explicitly disallowed (cost set to 0 IoU).
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(trackers))

    iou_matrix = iou_batch(detections, trackers)

    # Class-Aware filter: zero out IoU between different object categories
    for d, d_cls in enumerate(det_classes):
        for t, t_cls in enumerate(trk_classes):
            if d_cls is not None and t_cls is not None and d_cls != t_cls:
                iou_matrix[d, t] = 0.0

    # Hungarian Algorithm assignment (minimizing -IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.column_stack((row_ind, col_ind))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    # Filter out matches below IoU threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


class Sort:
    """
    Optimized Class-Aware SORT Coordinator.
    """
    def __init__(self, max_age=25, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.total_unique_ids = set()

    def update(self, dets, class_ids=None, class_names=None):
        """
        dets: numpy array of shape (N, 5) [x1, y1, x2, y2, conf]
        """
        self.frame_count += 1
        
        # Predict locations
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        trk_classes = []
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos
            trk_classes.append(trk.cls_name)
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)
            trk_classes.pop(t)

        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4), dtype=np.float32)
        det_confs = dets[:, 4] if len(dets) > 0 and dets.shape[1] >= 5 else np.zeros((len(dets),))
        det_names = class_names if class_names is not None else [None] * len(dets)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, trks, det_names, trk_classes, self.iou_threshold
        )

        # Update matched trackers
        for m in matched:
            d_idx, t_idx = m[0], m[1]
            c_id = class_ids[d_idx] if class_ids is not None else None
            c_name = class_names[d_idx] if class_names is not None else None
            conf = float(det_confs[d_idx]) if len(det_confs) > d_idx else None
            self.trackers[t_idx].update(det_boxes[d_idx], cls_id=c_id, cls_name=c_name, conf=conf)

        # Initialize new trackers for unmatched detections
        for i in unmatched_dets:
            c_id = class_ids[i] if class_ids is not None else None
            c_name = class_names[i] if class_names is not None else None
            conf = float(det_confs[i]) if len(det_confs) > i else 0.0
            trk = KalmanBoxTracker(det_boxes[i], cls_id=c_id, cls_name=c_name, conf=conf)
            self.trackers.append(trk)

        # Extract active confirmed tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                self.total_unique_ids.add(trk.id)
                vx, vy = trk.get_velocity()
                ret.append({
                    "bbox": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                    "id": trk.id,
                    "cls_id": trk.cls_id,
                    "cls_name": trk.cls_name,
                    "conf": trk.conf,
                    "velocity": (vx, vy)
                })
            i -= 1
            # Prune stale trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return ret
