"""
Threaded Camera Capture module.
Reads frames in a dedicated daemon thread to eliminate webcam buffering lag.

Root fix: Never force resolution or format on Windows webcams.
Many webcams (especially built-in laptop cameras) return black frames when
OpenCV tries to set a resolution they don't natively support via DirectShow.
We let the driver negotiate everything automatically.
"""

import sys
import time
import threading
import cv2
import numpy as np


class ThreadedCamera:
    """
    Asynchronous Camera Reader.
    - Uses the camera's NATIVE resolution (no forced width/height)
    - No forced FOURCC codec (was causing black frames on many Windows webcams)
    - Warmup: skips first few frames while sensor stabilizes
    """
    def __init__(self, src=0, width=None, height=None, fps=None):
        self.src = src
        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)

        if not self.cap.isOpened():
            # Fallback to index 1
            if isinstance(src, int) and src == 0:
                print("[Camera] Index 0 failed, probing index 1...")
                self.cap = cv2.VideoCapture(1, backend)
            if not self.cap.isOpened():
                print("[Camera] ERROR: Could not open any camera.")
                return

        # Reduce internal buffer to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Camera warmup: discard first N frames (sensor needs time to initialize)
        print("[Camera] Warming up sensor (discarding first 10 frames)...")
        for _ in range(10):
            self.cap.read()

        # Read the first real frame
        self.ret, self.frame = self.cap.read()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] Ready: {actual_w}x{actual_h} @ {actual_fps:.0f}fps  "
              f"(first frame mean brightness: {np.mean(self.frame):.1f})")

        # Start background capture thread
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.005)
            else:
                break

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened() if self.cap else False

    def get(self, propId):
        return self.cap.get(propId) if self.cap else 0

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
