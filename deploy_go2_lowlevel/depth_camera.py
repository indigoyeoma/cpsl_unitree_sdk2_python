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
        width: int = 640,       # Match parkour resolution exactly
        height: int = 480,      # Match parkour resolution exactly
        fps: int = 30,
        target_width: int = 87,
        target_height: int = 58,
        near_clip: float = 0.3,
        far_clip: float = 3.0,
        rotate_180: bool = False,  # Set True if camera is mounted inverted
        # Cropping settings - MORE aggressive left crop to remove D435i edge artifact
        # Parkour uses crop_left=28 but our D435i has artifact extending to cols 0-3 after resize
        # Increase left crop to remove ~5% more (640 * 0.09 ≈ 58)
        crop_left: int = 58,    # Increased from 28 to remove left edge artifact
        crop_right: int = 36,   # Exact parkour value
        crop_top: int = 48,     # Exact parkour value
        crop_bottom: int = 0,   # Exact parkour value
    ):
        """
        Initialize D435i camera.

        Args:
            width: Capture width (640 to match parkour exactly)
            height: Capture height (480 to match parkour exactly)
            fps: Capture framerate
            target_width: Output width for policy (after resize)
            target_height: Output height for policy (after resize)
            near_clip: Minimum depth in meters
            far_clip: Maximum depth in meters
            rotate_180: Rotate image 180 degrees (for inverted camera mounting)
            crop_left/right/top/bottom: Pixels to crop from each edge (parkour defaults)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.target_width = target_width
        self.target_height = target_height
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.rotate_180 = rotate_180
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

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

        # Set medium density preset for balanced performance
        try:
            depth_sensor.set_option(rs.option.visual_preset, 5)  # Medium Density
            print(f"✓ Medium Density mode enabled (balanced quality & speed)")
        except Exception as e:
            print(f"Could not set visual preset: {e}")

        # Build RealSense filters (from parkour go2_visual.py)
        # These improve depth quality significantly
        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)

        # Filter pipeline order (from parkour)
        self.rs_filters = [
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]
        print(f"✓ RealSense filters enabled (hole filling, spatial, temporal)")

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
                    # Apply RealSense filters (from parkour)
                    for rs_filter in self.rs_filters:
                        depth_frame = rs_filter.process(depth_frame)

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
        Processing order (matching CPSL training):
        1. Rotate 180° (if camera inverted)
        2. Crop edges (training uses [:-2, 4:-4])
        3. Clip depth range
        4. Resize to target (87x58)
        5. Normalize to [-0.5, 0.5]
        6. Fix edge artifacts

        Args:
            depth_frame: RealSense depth frame

        Returns:
            Processed depth image (target_height x target_width), normalized [-0.5, 0.5]
        """
        # Convert to numpy array (in millimeters for z16)
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # STEP 1: Rotate 180 degrees if camera is mounted inverted
        if self.rotate_180:
            depth_image = np.rot90(depth_image, k=2)  # k=2 for 180 degree rotation

        # Convert to meters (negative as in Isaac Gym depth)
        depth_image = -depth_image * self.depth_scale

        # STEP 2: CROP edges (matching CPSL training)
        # Training uses [:-2, 4:-4] on 106x60 → we use similar ratio on 424x240
        top = self.crop_top if self.crop_top > 0 else None
        bottom = -self.crop_bottom if self.crop_bottom > 0 else None
        left = self.crop_left if self.crop_left > 0 else None
        right = -self.crop_right if self.crop_right > 0 else None

        # Build slice - handle None cases
        row_start = top if top else 0
        row_end = bottom  # None means to end
        col_start = left if left else 0
        col_end = right  # None means to end

        depth_image = depth_image[row_start:row_end, col_start:col_end]

        # STEP 3: Clip to valid range (negative values!)
        depth_image = np.clip(depth_image, -self.far_clip, -self.near_clip)

        # STEP 4: Resize to target dimensions (87x58)
        depth_image = cv2.resize(
            depth_image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )

        # STEP 5: Normalize (matching training exactly)
        # Training formula: (depth - near) / (far - near) - 0.5
        # This gives: close (0.3m) → -0.5, far (3.0m) → +0.5
        depth_image = depth_image * -1  # Make positive (0.3m to 3.0m range)

        # Match training normalization exactly:
        # (depth - near_clip) / (far_clip - near_clip) - 0.5
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

        # STEP 6: Fix left-edge artifacts (stuck pixels at -0.5 cause right-turning)
        # Copy column 1 to column 0 to fix camera edge artifacts
        depth_image[:, 0] = depth_image[:, 1]

        # Result: range [-0.5, 0.5]
        #   near (0.3m) → -0.5  (CLOSE = LOW)
        #   far (3.0m) → +0.5   (FAR = HIGH)

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
        # Return realistic depth: floor at bottom, far at top
        # Simulates looking forward and down at flat ground
        depth = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        for row in range(self.target_height):
            # Top rows = far (+0.3), bottom rows = close (-0.4 = floor at ~0.5m)
            t = row / (self.target_height - 1)  # 0 at top, 1 at bottom
            depth[row, :] = 0.3 - 0.7 * t  # +0.3 at top, -0.4 at bottom
        return depth

    def stop(self):
        self.running = False
        print("Dummy camera stopped")


def create_camera(use_real: bool = True, **kwargs):
    """
    Factory function to create camera instance.

    Args:
        use_real: If True, try to use real D435i camera
        **kwargs: Additional arguments for camera initialization
            - rotate_180: Set True if camera is mounted inverted (like parkour Go2 setup)

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
