"""
Intel RealSense D435i depth camera interface for Go2 deployment.
Captures depth frames and preprocesses them for the vision policy.

Camera is mounted UPRIGHT (no rotation needed, unlike parkour code).
"""
import numpy as np
import threading
import time
from typing import Optional, Tuple

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("WARNING: pyrealsense2 not installed. Using dummy depth frames.")

from config import (
    DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS,
    CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT,
    DEPTH_OUTPUT_WIDTH, DEPTH_OUTPUT_HEIGHT,
    DEPTH_NEAR, DEPTH_FAR
)


class DepthCamera:
    """
    Intel RealSense D435i depth camera wrapper.

    Captures depth frames in a background thread and provides
    preprocessed frames for the vision policy.
    """

    def __init__(self, enable_filters: bool = True):
        """
        Initialize the depth camera.

        Args:
            enable_filters: Whether to apply RealSense post-processing filters
        """
        self.enable_filters = enable_filters
        self.pipeline = None
        self.config = None
        self.running = False
        self.thread = None

        # Latest frame storage (thread-safe)
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_raw_frame: Optional[np.ndarray] = None  # Raw frame before preprocessing
        self._frame_timestamp: float = 0.0

        # Initialize filters if enabled
        if enable_filters and HAS_REALSENSE:
            self._init_filters()

    def _init_filters(self):
        """Initialize RealSense post-processing filters."""
        # Hole filling filter
        self.hole_filter = rs.hole_filling_filter()

        # Spatial filter (edge-preserving smoothing)
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.spatial_filter.set_option(rs.option.holes_fill, 4)

        # Temporal filter (reduces noise over time)
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 1)

        self.filters = [self.hole_filter, self.spatial_filter, self.temporal_filter]

    def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            True if started successfully, False otherwise
        """
        if not HAS_REALSENSE:
            print("RealSense not available, using dummy frames")
            self.running = True
            return True

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure depth stream
            self.config.enable_stream(
                rs.stream.depth,
                DEPTH_WIDTH,
                DEPTH_HEIGHT,
                rs.format.z16,
                DEPTH_FPS
            )

            # Start pipeline
            self.pipeline.start(self.config)

            # Wait for auto-exposure to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.running = True

            # Start background capture thread
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            print(f"Depth camera started: {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS}fps")
            return True

        except Exception as e:
            print(f"Failed to start depth camera: {e}")
            return False

    def stop(self):
        """Stop the camera capture."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except:
                pass
        print("Depth camera stopped")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue

                # Apply filters
                if self.enable_filters:
                    for f in self.filters:
                        depth_frame = f.process(depth_frame)

                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())

                # Preprocess for policy
                processed = self._preprocess(depth_image)

                # Store with thread safety
                with self._frame_lock:
                    self._latest_frame = processed
                    self._latest_raw_frame = depth_image.copy()  # Store raw frame (in mm)
                    self._frame_timestamp = time.time()

            except Exception as e:
                if self.running:
                    print(f"Depth capture error: {e}")
                time.sleep(0.01)

    def _preprocess(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Preprocess depth image for the vision policy.

        Args:
            depth_image: Raw depth image from camera (H, W) in millimeters

        Returns:
            Normalized depth image (DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH)
        """
        # Camera is UPRIGHT - no rotation needed (unlike parkour which rotates 180)

        # Crop edges
        h, w = depth_image.shape
        crop_h_end = h - CROP_BOTTOM if CROP_BOTTOM > 0 else h
        crop_w_end = w - CROP_RIGHT if CROP_RIGHT > 0 else w

        cropped = depth_image[CROP_TOP:crop_h_end, CROP_LEFT:crop_w_end]

        # Convert to meters (from mm)
        depth_m = cropped.astype(np.float32) / 1000.0

        # Clip to valid range
        depth_m = np.clip(depth_m, DEPTH_NEAR, DEPTH_FAR)

        # Resize to policy input size
        resized = self._resize(depth_m, (DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH))

        # Normalize to [-0.5, 0.5] (matching training go2_student_config.py)
        # Training: (depth - near_clip) / (far_clip - near_clip) - 0.5
        normalized = (resized - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR) - 0.5

        return normalized.astype(np.float32)

    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image using area averaging (similar to BICUBIC downsampling).

        Args:
            image: Input image (H, W)
            size: Target size (H, W)

        Returns:
            Resized image
        """
        try:
            import cv2
            return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback to simple nearest-neighbor if cv2 not available
            h, w = image.shape
            target_h, target_w = size

            row_indices = (np.arange(target_h) * h / target_h).astype(int)
            col_indices = (np.arange(target_w) * w / target_w).astype(int)

            return image[row_indices[:, None], col_indices]

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest preprocessed depth frame.

        Returns:
            Preprocessed depth image (H, W) normalized to [0, 1], or None if no frame
        """
        if not HAS_REALSENSE:
            # Return dummy frame for testing without camera
            return self._get_dummy_frame()

        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest raw depth frame (before preprocessing).

        Returns:
            Raw depth image (H, W) in millimeters, or None if no frame
        """
        if not HAS_REALSENSE:
            return None

        with self._frame_lock:
            if self._latest_raw_frame is None:
                return None
            return self._latest_raw_frame.copy()

    def get_frame_age(self) -> float:
        """
        Get the age of the latest frame in seconds.

        Returns:
            Time since last frame was captured
        """
        with self._frame_lock:
            if self._frame_timestamp == 0:
                return float('inf')
            return time.time() - self._frame_timestamp

    def get_timestamp(self) -> float:
        """
        Get the timestamp of the latest frame.

        Returns:
            Timestamp of the latest frame
        """
        with self._frame_lock:
            return self._frame_timestamp

    def _get_dummy_frame(self) -> np.ndarray:
        """Generate a dummy depth frame for testing without camera."""
        # Create a simple gradient pattern
        frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)

        # Add some structure - horizontal gradient
        for i in range(DEPTH_OUTPUT_HEIGHT):
            frame[i, :] = 0.3 + 0.4 * (i / DEPTH_OUTPUT_HEIGHT)

        return frame


def test_camera():
    """Test the depth camera."""
    print("Testing depth camera...")

    camera = DepthCamera(enable_filters=True)

    if not camera.start():
        print("Failed to start camera")
        return

    print("Camera started. Capturing 10 frames...")

    for i in range(10):
        time.sleep(0.1)
        frame = camera.get_frame()
        if frame is not None:
            print(f"Frame {i+1}: shape={frame.shape}, min={frame.min():.3f}, max={frame.max():.3f}")
        else:
            print(f"Frame {i+1}: No frame available")

    camera.stop()
    print("Test complete")


if __name__ == "__main__":
    test_camera()
