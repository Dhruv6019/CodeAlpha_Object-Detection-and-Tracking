"""
Spatial Motion Density Heatmap generator.
Accumulates spatial presence of tracked objects to visualize traffic corridors and dwell hotspots.
"""

import cv2
import numpy as np


class MotionHeatmap:
    def __init__(self, decay=0.992):
        self.accum_map = None
        self.decay = decay

    def update(self, frame_shape, tracks):
        """
        Accumulate density at tracked object centroids.
        tracks: list of track dicts
        """
        h, w = frame_shape[:2]
        if self.accum_map is None or self.accum_map.shape != (h, w):
            self.accum_map = np.zeros((h, w), dtype=np.float32)

        # Decay previous heat slowly
        self.accum_map *= self.decay

        # Add Gaussian heat splat for each active track
        for trk in tracks:
            box = trk["bbox"]
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)
            
            radius = max(10, int(min(box[2] - box[0], box[3] - box[1]) / 2.5))
            
            # Draw heat spot
            cv2.circle(self.accum_map, (cx, cy), radius, 1.0, -1)

    def apply(self, frame, alpha=0.45):
        """
        Blend the accumulated heatmap onto the frame.
        """
        if self.accum_map is None or np.max(self.accum_map) == 0:
            return frame

        # Normalize 0 - 255
        norm_map = cv2.normalize(self.accum_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_map = np.uint8(norm_map)

        # Colorize using JET colormap
        colored_heat = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Mask low heat
        mask = norm_map > 15
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(frame[mask], 1 - alpha, colored_heat[mask], alpha, 0)
        return blended
