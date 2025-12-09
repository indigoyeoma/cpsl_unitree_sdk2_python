#!/usr/bin/env python3
"""
Depth Camera Verification Tool for Go2 Vision Policy

Verifies that D435i depth camera orientation and processing matches training simulation.

CRITICAL CHECKS:
1. Image orientation (not flipped/rotated)
2. Depth value representation (near=close, far=distant)
3. Camera mounting angle (pitch down ~5-15°)
4. Processing pipeline (crop, clip, resize, normalize)

Usage:
    python verify_depth_camera.py

Controls:
    q - Quit
    s - Save snapshot
    r - Toggle raw/processed view
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("ERROR: pyrealsense2 not installed!")
    print("Install with: pip install pyrealsense2")
    sys.exit(1)

from config import DeployConfig


class DepthVerifier:
    """Verify D435i depth camera setup matches training."""

    def __init__(self):
        self.config = DeployConfig()

        # Camera parameters
        self.width = 424
        self.height = 240
        self.fps = 30
        self.target_width = self.config.depth_width
        self.target_height = self.config.depth_height
        self.near_clip = self.config.depth_near
        self.far_clip = self.config.depth_far

        # Visualization state
        self.show_raw = True
        self.paused = False

        print("="*70)
        print("D435i Depth Camera Verification Tool")
        print("="*70)
        print(f"Target resolution: {self.target_width}x{self.target_height}")
        print(f"Depth range: {self.near_clip}m - {self.far_clip}m")
        print(f"Expected mounting: ~28cm forward, ~15cm up, ~5-15° pitch down")
        print("="*70)

    def start_camera(self):
        """Initialize D435i camera."""
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()

        # Configure depth stream
        self.config_rs.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps
        )

        # Start streaming
        print("\nStarting camera...")
        profile = self.pipeline.start(self.config_rs)

        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"✓ Camera started")
        print(f"  Depth scale: {self.depth_scale}")

        # Set high accuracy preset
        try:
            depth_sensor.set_option(rs.option.visual_preset, 3)
            print(f"✓ High accuracy mode enabled")
        except Exception as e:
            print(f"⚠ Could not set visual preset: {e}")

    def process_depth(self, depth_frame):
        """Process depth frame matching training pipeline."""
        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # Convert to meters (negative as in Isaac Gym)
        depth_image = -depth_image * self.depth_scale

        # Crop edges (matching training)
        depth_cropped = depth_image[:-2, 4:-4]

        # Clip to range
        depth_clipped = np.clip(depth_cropped, -self.far_clip, -self.near_clip)

        # Resize
        depth_resized = cv2.resize(
            depth_clipped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )

        # Normalize to [-0.5, 0.5]
        depth_normalized = depth_resized * -1  # Make positive
        depth_normalized = (depth_normalized - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

        return depth_image, depth_cropped, depth_clipped, depth_resized, depth_normalized

    def visualize(self, depth_raw, depth_cropped, depth_clipped, depth_resized, depth_normalized):
        """Create visualization with verification guides."""

        # RAW VIEW (top half)
        raw_vis = self.depth_to_color(depth_raw, -self.far_clip, -self.near_clip)
        raw_vis = cv2.resize(raw_vis, (640, 360))

        # Add orientation guides
        h, w = raw_vis.shape[:2]
        cv2.line(raw_vis, (w//2, 0), (w//2, h), (0, 255, 0), 1)  # Vertical center
        cv2.line(raw_vis, (0, h//2), (w, h//2), (0, 255, 0), 1)  # Horizontal center
        cv2.putText(raw_vis, "RAW DEPTH (424x240)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(raw_vis, "Ground should be: BOTTOM HALF", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(raw_vis, "Close objects: BRIGHT", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(raw_vis, "Far objects: DARK", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # PROCESSED VIEW (87x58 - scaled up for visibility)
        proc_vis = self.depth_to_color(depth_normalized, -0.5, 0.5)
        proc_vis = cv2.resize(proc_vis, (640, 428), interpolation=cv2.INTER_NEAREST)

        # Add grid to show 87x58 structure
        for i in range(0, 640, 640//87):
            cv2.line(proc_vis, (i, 0), (i, 428), (100, 100, 100), 1)
        for j in range(0, 428, 428//58):
            cv2.line(proc_vis, (0, j), (640, j), (100, 100, 100), 1)

        cv2.putText(proc_vis, "PROCESSED DEPTH (87x58 scaled)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(proc_vis, "This is what policy sees!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Combine views
        combined = np.vstack([raw_vis, proc_vis])

        # Add verification checklist
        checklist_y = 800
        cv2.putText(combined, "VERIFICATION CHECKLIST:", (10, checklist_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        checks = [
            "[ ] Ground appears in BOTTOM half of image",
            "[ ] Close objects (hand) are BRIGHTER than far",
            "[ ] Image is NOT upside down or rotated",
            "[ ] Camera tilted DOWN (sees ground ahead)",
            "[ ] Processed view shows clear terrain features"
        ]

        for i, check in enumerate(checks):
            cv2.putText(combined, check, (10, checklist_y + 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add stats
        stats_y = 10
        stats = [
            f"Min depth: {-depth_raw.max():.2f}m",
            f"Max depth: {-depth_raw.min():.2f}m",
            f"Mean depth: {-depth_raw.mean():.2f}m",
        ]
        for i, stat in enumerate(stats):
            cv2.putText(combined, stat, (450, stats_y + 20 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return combined

    def depth_to_color(self, depth, vmin, vmax):
        """Convert depth to color visualization."""
        # Normalize to 0-255
        depth_norm = ((depth - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        depth_norm = np.clip(depth_norm, 0, 255)

        # Apply colormap
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        return depth_color

    def run(self):
        """Main verification loop."""
        self.start_camera()

        print("\n" + "="*70)
        print("VERIFICATION INSTRUCTIONS:")
        print("="*70)
        print("1. Point camera at ground ahead (~1-2m away)")
        print("2. Check ground appears in BOTTOM half (camera pitched down)")
        print("3. Wave hand in front - should appear BRIGHT (close)")
        print("4. Far ground should be DARK")
        print("5. Image should NOT be upside down or mirrored")
        print("\nPress 's' to save snapshot, 'q' to quit")
        print("="*70 + "\n")

        cv2.namedWindow("Depth Verification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Verification", 640, 950)

        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue

                # Process
                depth_raw, depth_cropped, depth_clipped, depth_resized, depth_normalized = \
                    self.process_depth(depth_frame)

                # Visualize
                vis = self.visualize(depth_raw, depth_cropped, depth_clipped,
                                   depth_resized, depth_normalized)

                cv2.imshow("Depth Verification", vis)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting...")
                    break
                elif key == ord('s'):
                    filename = f"depth_verification_{int(time.time())}.png"
                    cv2.imwrite(filename, vis)
                    print(f"✓ Saved: {filename}")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Camera stopped")


if __name__ == "__main__":
    verifier = DepthVerifier()
    verifier.run()
