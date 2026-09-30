"""
Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric
Implements:
1. Deep CNN Re-ID feature extractor (MobileNetV3-Small backbone, 576-dim L2-normalized embeddings)
2. Kalman Filter motion state estimation
3. Nearest-Neighbor Cosine Distance Gallery
4. Matching Cascade (Deep Appearance Association + Spatial IoU fallback)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment


class Extractor(nn.Module):
    """
    Lightweight Deep Feature Extractor for Re-Identification.
    Uses pretrained MobileNetV3-Small features projected to a normalized embedding vector.
    """
    def __init__(self, device="cpu"):
        super(Extractor, self).__init__()
        self.device = device
        # Load backbone
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(self.device)
        self.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def extract(self, crops):
        """
        Extract L2-normalized deep visual features from an array of image patches.
        crops: list of BGR/RGB numpy arrays
        Returns: np.ndarray shape (N, 576)
        """
        if len(crops) == 0:
            return np.empty((0, 576), dtype=np.float32)

        tensors = []
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                # Fallback for degenerate crop
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                crop = transforms.functional.resize(
                    transforms.functional.to_pil_image(crop),
                    [128, 64]
                )
            tensors.append(self.transform(crop))

        batch = torch.stack(tensors).to(self.device)
        feat = self.features(batch)
        feat = self.pool(feat).flatten(1)
        # L2-normalize embeddings for cosine distance
        norm = feat.norm(p=2, dim=1, keepdim=True) + 1e-12
        feat = feat.div(norm)
        return feat.cpu().numpy()


def iou_batch(bb_test, bb_gt):
    """Pairwise IoU matrix between bounding boxes."""
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
    """Convert [x1, y1, x2, y2] to center state [cx, cy, aspect_ratio, height]."""
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    aspect = w / float(h)
    return np.array([cx, cy, aspect, h], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x):
    """Convert state [cx, cy, aspect_ratio, height] to [x1, y1, x2, y2]."""
    cx, cy, aspect, h = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
    w = aspect * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32).reshape((1, 4))


class DeepTrack:
    """Individual object track in Deep SORT."""
    count = 0

    def __init__(self, bbox, feature, cls_name="object", conf=0.0):
        self.id = DeepTrack.count
        DeepTrack.count += 1
        
        # Kalman filter state: [cx, cy, aspect, h, vx, vy, va, vh]
        self.dim_x = 8
        self.dim_z = 4
        
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        self.R = np.eye(4, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        # Appearance gallery (storing historical Re-ID features for occlusion recovery)
        self.features = [feature] if feature is not None else []
        self.max_features = 40
        
        self.cls_name = cls_name
        self.conf = conf
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = True  # Confirmed immediately for instant detection

        # Coordinate smoothing (EMA)
        self.smoothed_bbox = np.array(bbox[:4], dtype=np.float32)
        self.alpha = 0.72

    def update(self, bbox, feature, cls_name=None, conf=None):
        self.time_since_update = 0
        self.hits += 1
        if cls_name is not None:
            self.cls_name = cls_name
        if conf is not None:
            self.conf = conf
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.max_features:
                self.features.pop(0)

        z = convert_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I_KH = np.eye(self.dim_x) - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)

        raw_box = convert_x_to_bbox(self.x)[0]
        self.smoothed_bbox = self.alpha * raw_box + (1.0 - self.alpha) * self.smoothed_bbox

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.age += 1
        self.time_since_update += 1
        return convert_x_to_bbox(self.x)[0]

    def get_state(self):
        return self.smoothed_bbox.copy()

    def get_velocity(self):
        return float(self.x[4, 0]), float(self.x[5, 0])


class DeepSort:
    """
    Deep SORT Multi-Object Tracker with Deep CNN Embeddings & Hungarian Association.
    """
    def __init__(self, max_cosine_dist=0.35, nn_budget=40, max_age=30, min_hits=1, iou_thresh=0.3):
        self.max_cosine_dist = max_cosine_dist
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        
        # Initialize ReID feature extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = Extractor(device=device)
        self.tracks = []
        self.total_unique_ids = set()

    def _cosine_distance(self, track, det_features):
        """
        Compute minimum cosine distance between all stored features in track gallery
        and each candidate detection feature.
        Cosine distance = 1 - (A . B)
        """
        if len(track.features) == 0 or len(det_features) == 0:
            return np.ones((len(det_features),), dtype=np.float32)

        track_feats = np.array(track.features)  # shape (K, D)
        # Cosine similarity matrix: (K, N)
        similarity = np.dot(track_feats, det_features.T)
        # Min distance = 1 - Max similarity
        dist = 1.0 - np.max(similarity, axis=0)
        return np.maximum(0.0, dist)

    def update(self, frame, dets, class_names=None):
        """
        Frame-level Deep SORT update:
        dets: np.ndarray shape (N, 5) [x1, y1, x2, y2, conf]
        """
        h, w = frame.shape[:2]

        # 1. Predict next state for all existing tracks
        for trk in self.tracks:
            trk.predict()

        if len(dets) == 0:
            # No detections, prune dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._format_results()

        # 2. Extract Deep Appearance Features for detected bounding boxes
        crops = []
        for box in dets[:, :4]:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crops.append(frame[y1:y2, x1:x2])

        det_features = self.extractor.extract(crops)
        det_boxes = dets[:, :4]
        det_confs = dets[:, 4]
        det_names = class_names if class_names is not None else ["object"] * len(dets)

        # 3. Stage 1: Deep Appearance Matching via Cosine Distance
        num_tracks = len(self.tracks)
        num_dets = len(dets)

        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                cost_matrix[t_idx] = self._cosine_distance(trk, det_features)
                # Enforce class consistency
                for d_idx, d_name in enumerate(det_names):
                    if trk.cls_name != d_name:
                        cost_matrix[t_idx, d_idx] = 1e5

            # Hungarian matching on cosine distance
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_stage1 = []
            unmatched_tracks = set(range(num_tracks))
            unmatched_dets = set(range(num_dets))

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.max_cosine_dist:
                    matched_stage1.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

            # 4. Stage 2: Spatial IoU Matching Fallback for unmatched items
            unmatched_track_list = list(unmatched_tracks)
            unmatched_det_list = list(unmatched_dets)

            matched_stage2 = []
            if len(unmatched_track_list) > 0 and len(unmatched_det_list) > 0:
                t_boxes = np.array([self.tracks[i].get_state() for i in unmatched_track_list])
                d_boxes = det_boxes[unmatched_det_list]
                iou_mat = iou_batch(d_boxes, t_boxes)

                # Penalize mismatched classes
                for di, orig_d in enumerate(unmatched_det_list):
                    for ti, orig_t in enumerate(unmatched_track_list):
                        if det_names[orig_d] != self.tracks[orig_t].cls_name:
                            iou_mat[di, ti] = 0.0

                r_ind, c_ind = linear_sum_assignment(-iou_mat)
                for di, ti in zip(r_ind, c_ind):
                    if iou_mat[di, ti] >= self.iou_thresh:
                        matched_stage2.append((unmatched_track_list[ti], unmatched_det_list[di]))
                        unmatched_tracks.discard(unmatched_track_list[ti])
                        unmatched_dets.discard(unmatched_det_list[di])

            # Apply matched updates
            for t_idx, d_idx in matched_stage1 + matched_stage2:
                feat = det_features[d_idx] if len(det_features) > d_idx else None
                self.tracks[t_idx].update(
                    det_boxes[d_idx],
                    feature=feat,
                    cls_name=det_names[d_idx],
                    conf=float(det_confs[d_idx])
                )
        else:
            unmatched_dets = set(range(num_dets))

        # 5. Initialize new tracks for unmatched detections
        for d_idx in unmatched_dets:
            feat = det_features[d_idx] if len(det_features) > d_idx else None
            new_track = DeepTrack(
                det_boxes[d_idx],
                feature=feat,
                cls_name=det_names[d_idx],
                conf=float(det_confs[d_idx])
            )
            self.tracks.append(new_track)

        # 6. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._format_results()

    def _format_results(self):
        ret = []
        for trk in self.tracks:
            if trk.time_since_update < 1 and trk.hits >= self.min_hits:
                self.total_unique_ids.add(trk.id)
                box = trk.get_state()
                vx, vy = trk.get_velocity()
                ret.append({
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "id": trk.id,
                    "cls_name": trk.cls_name,
                    "conf": trk.conf,
                    "velocity": (vx, vy)
                })
        return ret

    def get_total_unique_count(self):
        return len(self.total_unique_ids)
