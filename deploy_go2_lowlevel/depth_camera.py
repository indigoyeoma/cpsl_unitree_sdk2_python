"""
Intel RealSense D435i Depth Camera Interface for Go2 Vision Policy Deployment
"""

import numpy as np
import cv2
import threading
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Using dummy depth camera.")


class D435iCamera:
    """
    Interface for Intel RealSense D435i depth camera.

    Captures depth images and processes them for policy input.
    D435i specs: 86° x 57° FOV, up to 1280x720 @30fps depth
    """

    def __init__(
        self,
        width: int = 424,
        height: int = 240,
        fps: int = 30,
        target_width: int = 87,
        target_height: int = 58,
        near_clip: float = 0.0,
        far_clip: float = 2.0,
    ):
        """
        Initialize D435i camera.

        Args:
            width: Capture width (native resolution)
            height: Capture height (native resolution)
            fps: Capture framerate
            target_width: Output width for policy (after resize)
            target_height: Output height for policy (after resize)
            near_clip: Minimum depth in meters
            far_clip: Maximum depth in meters
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.target_width = target_width
        self.target_height = target_height
        self.near_clip = near_clip
        self.far_clip = far_clip

        self.pipeline = None
        self.config = None
        self.running = False
        self.latest_depth = None
        self.latest_depth_lock = threading.Lock()
        self.capture_thread = None

        if not REALSENSE_AVAILABLE:
            print("RealSense not available - using dummy depth images")

    def start(self):
        """Start the depth camera capture."""
        if not REALSENSE_AVAILABLE:
            self.running = True
            return

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure depth stream
        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps
        )

        # Start streaming
        profile = self.pipeline.start(self.config)

        # Get depth scale for converting to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Set high accuracy preset for better depth quality
        try:
            depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        except Exception as e:
            print(f"Could not set visual preset: {e}")

        self.running = True

        # Start background capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"D435i camera started: {self.width}x{self.height}@{self.fps}fps")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                depth_frame = frames.get_depth_frame()

                if depth_frame:
                    depth_image = self._process_depth_frame(depth_frame)
                    with self.latest_depth_lock:
                        self.latest_depth = depth_image
            except Exception as e:
                if self.running:
                    print(f"Depth capture error: {e}")
                time.sleep(0.01)

    def _process_depth_frame(self, depth_frame) -> np.ndarray:
        """
        Process raw depth frame to policy input format.

        IMPORTANT: This MUST match training preprocessing exactly!
        Training: crop → clip → resize → normalize to [-0.5, 0.5]

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Processed depth image (target_height x target_width), normalized [-0.5, 0.5]
        """
        # Convert to numpy array (in millimeters for z16)
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # Convert to meters (negative as in Isaac Gym depth)
        depth_image = -depth_image * self.depth_scale

        # STEP 1: CROP edges (matching training: depth_image[:-2, 4:-4])
        # Remove 4 pixels from left/right, 2 pixels from bottom
        # From 424x240 → 416x238
        depth_image = depth_image[:-2, 4:-4]

        # STEP 2: Clip to valid range (negative values!)
        depth_image = np.clip(depth_image, -self.far_clip, -self.near_clip)

        # STEP 3: Resize to target dimensions
        # From 416x238 → 87x58
        depth_image = cv2.resize(
            depth_image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )

        # STEP 4: Normalize (matching training exactly)
        # depth_image = depth_image * -1  (already negative)
        # depth_image = (depth_image - near_clip) / (far_clip - near_clip) - 0.5
        depth_image = depth_image * -1  # Make positive
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        # Result: range [-0.5, 0.5]
        #   near (0m) → 0.5
        #   mid (1m) → 0.0
        #   far (2m) → -0.5

        return depth_image

    def get_depth(self) -> np.ndarray:
        """
        Get latest depth image.

        Returns:
            Depth image (target_height x target_width), normalized 0-1
            Returns zeros if no frame available
        """
        if not REALSENSE_AVAILABLE:
            # Return dummy depth image for testing
            return np.zeros((self.target_height, self.target_width), dtype=np.float32)

        with self.latest_depth_lock:
            if self.latest_depth is not None:
                return self.latest_depth.copy()
            else:
                return np.zeros((self.target_height, self.target_width), dtype=np.float32)

    def stop(self):
        """Stop the camera capture."""
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.stop()
        print("D435i camera stopped")


class DummyCamera:
    """Dummy camera for testing without hardware."""

    def __init__(self, target_width=87, target_height=58):
        self.target_width = target_width
        self.target_height = target_height
        self.running = False

    def start(self):
        self.running = True
        print("Dummy camera started")

    def get_depth(self) -> np.ndarray:
        # Return flat ground at 1.0m
        return np.ones((self.target_height, self.target_width), dtype=np.float32) * 0.5

    def stop(self):
        self.running = False
        print("Dummy camera stopped")


def create_camera(use_real: bool = True, **kwargs):
    """
    Factory function to create camera instance.

    Args:
        use_real: If True, try to use real D435i camera
        **kwargs: Additional arguments for camera initialization

    Returns:
        Camera instance (D435iCamera or DummyCamera)
    """
    if use_real and REALSENSE_AVAILABLE:
        return D435iCamera(**kwargs)
    else:
        return DummyCamera(
            target_width=kwargs.get('target_width', 87),
            target_height=kwargs.get('target_height', 58)
        )
