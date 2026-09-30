"""
Virtual Tripwire & Line-Crossing Counter module.
Detects when tracked object centroids cross a user-defined virtual boundary,
maintaining bidirectional directional counts (IN/OUT or UP/DOWN).
"""

import cv2
import numpy as np


def ccw(A, B, C):
    """Checks if three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """Returns True if line segment AB intersects with line segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class LineCounter:
    def __init__(self, start_point=None, end_point=None, line_position_ratio=0.55):
        """
        start_point: (x1, y1)
        end_point:   (x2, y2)
        If None, automatically initialized across frame width at line_position_ratio.
        """
        self.p1 = start_point
        self.p2 = end_point
        self.line_ratio = line_position_ratio
        
        self.in_count = 0
        self.out_count = 0
        self.counted_ids = set()
        self.prev_positions = {}
        self.class_counts = {}

    def auto_init(self, frame_w, frame_h):
        """Initialize line if not explicitly set."""
        if self.p1 is None or self.p2 is None:
            y = int(frame_h * self.line_ratio)
            self.p1 = (30, y)
            self.p2 = (frame_w - 30, y)

    def update(self, tracks):
        """
        Process current frame tracks and detect line crossing events.
        tracks: list of dicts with 'id', 'bbox', 'cls_name'
        """
        if self.p1 is None or self.p2 is None:
            return

        for trk in tracks:
            tid = trk["id"]
            box = trk["bbox"]
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)
            curr_pos = (cx, cy)

            if tid in self.prev_positions:
                prev_pos = self.prev_positions[tid]
                
                # Check for line intersection
                if tid not in self.counted_ids:
                    if intersect(prev_pos, curr_pos, self.p1, self.p2):
                        # Determine direction
                        # Vector of line: (p2[0] - p1[0], p2[1] - p1[1])
                        # Normal vector: (-(p2[1] - p1[1]), p2[0] - p1[0])
                        # Movement vector: (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                        dy = curr_pos[1] - prev_pos[1]
                        cls_name = trk.get("cls_name", "object")

                        if dy > 0:
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        self.counted_ids.add(tid)
                        self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1

            self.prev_positions[tid] = curr_pos

    def draw(self, frame):
        """Draw the glowing virtual tripwire and count badges on the frame."""
        if self.p1 is None or self.p2 is None:
            self.auto_init(frame.shape[1], frame.shape[0])

        # Draw glowing line
        # Underlay glow
        cv2.line(frame, self.p1, self.p2, (0, 140, 255), 4, cv2.LINE_AA)
        # Foreground sharp line
        cv2.line(frame, self.p1, self.p2, (0, 255, 255), 2, cv2.LINE_AA)

        # End markers
        cv2.circle(frame, self.p1, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, self.p2, 6, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw Tripwire Badge near center of line
        mid_x = (self.p1[0] + self.p2[0]) // 2
        mid_y = (self.p1[1] + self.p2[1]) // 2

        badge_txt = f"TRIPWIRE | IN: {self.in_count}  OUT: {self.out_count}"
        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)

        # Badge backdrop
        bx1, by1 = mid_x - tw // 2 - 8, mid_y - 14
        bx2, by2 = mid_x + tw // 2 + 8, mid_y + 10
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 25), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
        cv2.putText(frame, badge_txt, (bx1 + 8, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
