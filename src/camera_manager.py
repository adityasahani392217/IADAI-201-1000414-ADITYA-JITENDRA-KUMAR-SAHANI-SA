"""
ParkVision AI - Camera Stream & Live Ingestion Manager
Supports Local USB Webcam, RTSP/IP Camera Streams, and Demo Video Feeds
with thread-safe non-blocking frame retrieval, reconnection logic, and FPS measurement.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class CameraStreamManager:
    """
    Manages continuous video stream acquisition from Webcams, RTSP URLs, or Video Files.
    Runs background reader thread to always yield the freshest frame with zero latency buffer bloat.
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.stream_source: Optional[Any] = None
        self.source_type: str = "disconnected"  # 'webcam', 'rtsp', 'video_file', 'disconnected'
        
        self.is_running: bool = False
        self.is_paused: bool = False
        self._lock = threading.Lock()
        
        self._current_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0

        self.worker_thread: Optional[threading.Thread] = None
        self.last_error_message: Optional[str] = None

    def connect_webcam(self, device_index: int = 0) -> bool:
        """Connect to local webcam hardware."""
        self.stop()
        self.last_error_message = None
        try:
            # On Windows, cv2.CAP_DSHOW provides fast non-blocking initialization
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(device_index)

            if not cap.isOpened():
                self.last_error_message = f"Webcam index {device_index} could not be opened. Ensure no other application is using it."
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Read first frame to verify stream health
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self.last_error_message = "Webcam opened, but failed to retrieve visual frame."
                return False

            self.cap = cap
            self.stream_source = device_index
            self.source_type = "webcam"
            self._start_worker()
            return True
        except Exception as e:
            self.last_error_message = f"Webcam connection exception: {str(e)}"
            return False

    def connect_rtsp(self, rtsp_url: str, timeout_seconds: int = 5) -> bool:
        """Connect to an IP Camera via RTSP / HTTP URL."""
        self.stop()
        self.last_error_message = None

        if not rtsp_url or not isinstance(rtsp_url, str):
            self.last_error_message = "Invalid RTSP address."
            return False

        # Basic safety: ensure no plaintext passwords logged in error strings
        safe_url = rtsp_url.split("@")[-1] if "@" in rtsp_url else rtsp_url

        try:
            cap = cv2.VideoCapture(rtsp_url)
            start_check = time.time()
            opened = False
            while time.time() - start_check < timeout_seconds:
                if cap.isOpened():
                    opened = True
                    break
                time.sleep(0.2)

            if not opened:
                cap.release()
                self.last_error_message = f"Failed to connect to IP camera at {safe_url} within {timeout_seconds}s timeout."
                return False

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self.last_error_message = f"Connected to {safe_url}, but stream returned empty frames."
                return False

            self.cap = cap
            self.stream_source = rtsp_url
            self.source_type = "rtsp"
            self._start_worker()
            return True
        except Exception as e:
            self.last_error_message = f"IP Camera connection error: {str(e)}"
            return False

    def connect_video_file(self, video_path: str, loop: bool = True) -> bool:
        """Connect to a local video file (for Upload or Demo mode)."""
        self.stop()
        self.last_error_message = None

        p = Path(video_path)
        if not p.exists():
            self.last_error_message = f"Video file not found at: {video_path}"
            return False

        try:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                self.last_error_message = f"OpenCV failed to decode video container at {p.name}"
                return False

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self.last_error_message = f"Video file contains no decodable visual frames."
                return False

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap = cap
            self.stream_source = str(p)
            self.source_type = "video_file"
            self._start_worker(loop=loop)
            return True
        except Exception as e:
            self.last_error_message = f"Video playback exception: {str(e)}"
            return False

    def _start_worker(self, loop: bool = False) -> None:
        """Start background frame capture loop."""
        self.is_running = True
        self.is_paused = False
        self._fps_counter = 0
        self._last_fps_calc_time = time.time()
        
        self.worker_thread = threading.Thread(target=self._capture_loop, args=(loop,), daemon=True)
        self.worker_thread.start()

    def _capture_loop(self, loop: bool) -> None:
        """Continuous background frame reading loop."""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if loop and self.source_type == "video_file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Stream ended or disconnected
                    break

            with self._lock:
                self._current_frame = frame
                self._frame_count += 1
                self._fps_counter += 1

            # Update FPS calculation once per second
            now = time.time()
            elapsed = now - self._last_fps_calc_time
            if elapsed >= 1.0:
                self._fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._last_fps_calc_time = now

            # Rest slightly to yield CPU cycle to inference thread
            time.sleep(0.005)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieve the latest captured frame.
        Returns: (success, frame_bgr, measured_fps)
        """
        with self._lock:
            if self._current_frame is not None:
                return True, self._current_frame.copy(), self._fps
            return False, None, 0.0

    def pause(self) -> None:
        """Pause stream capture."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume stream capture."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop worker and release camera capture hardware."""
        self.is_running = False
        self.is_paused = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        with self._lock:
            self._current_frame = None

        self.source_type = "disconnected"
        self._fps = 0.0
