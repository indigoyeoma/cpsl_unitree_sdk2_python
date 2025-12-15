#!/usr/bin/env python3
"""
Test script to visualize depth camera input and verify preprocessing.

Run this to check if the camera is working correctly before deployment.

Usage:
    python test_camera.py              # Test with real camera
    python test_camera.py --dummy      # Test with dummy camera
"""

import numpy as np
import cv2
import time
import argparse

from depth_camera import create_camera
from config import DeployConfig


def test_camera(use_dummy: bool = False):
    """Test camera and visualize depth output."""

    print("=" * 60)
    print("Depth Camera Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Target size: {DeployConfig.depth_width} x {DeployConfig.depth_height}")
    print(f"  Near clip: {DeployConfig.depth_near} m")
    print(f"  Far clip: {DeployConfig.depth_far} m")
    print(f"  Expected range: [-0.5, +0.5]")
    print(f"    Close objects (0.3m) → -0.5 (BLACK)")
    print(f"    Far objects (3.0m) → +0.5 (WHITE)")
    print("=" * 60)

    # Create camera
    camera = create_camera(
        use_real=not use_dummy,
        target_width=DeployConfig.depth_width,
        target_height=DeployConfig.depth_height,
        near_clip=DeployConfig.depth_near,
        far_clip=DeployConfig.depth_far,
    )

    print("\nStarting camera...")
    camera.start()
    time.sleep(2.0)  # Warmup

    print("\nPress 'q' to quit, 's' to save current frame")
    print("Displaying depth image (close=dark, far=bright)...\n")

    cv2.namedWindow("Depth Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Test", 640, 480)

    frame_count = 0
    try:
        while True:
            # Get depth image
            depth = camera.get_depth()

            # Print stats every 50 frames
            if frame_count % 50 == 0:
                print(f"Frame {frame_count}:")
                print(f"  Shape: {depth.shape}")
                print(f"  Range: [{depth.min():.3f}, {depth.max():.3f}]")
                print(f"  Mean: {depth.mean():.3f}")

                # Check if values are in expected range
                if depth.min() < -0.6 or depth.max() > 0.6:
                    print("  WARNING: Values outside expected [-0.5, +0.5] range!")
                else:
                    print("  OK: Values in expected range")

            # Convert to displayable image
            # depth is in [-0.5, +0.5], convert to [0, 255]
            # -0.5 (close) → 0 (black)
            # +0.5 (far) → 255 (white)
            display = ((depth + 0.5) * 255).astype(np.uint8)

            # Apply colormap for better visualization
            display_color = cv2.applyColorMap(display, cv2.COLORMAP_JET)

            # Add text overlay
            cv2.putText(display_color, f"Min: {depth.min():.2f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_color, f"Max: {depth.max():.2f}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_color, f"Mean: {depth.mean():.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show
            cv2.imshow("Depth Test", display_color)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"depth_frame_{frame_count}.png"
                cv2.imwrite(filename, display_color)
                np.save(f"depth_raw_{frame_count}.npy", depth)
                print(f"Saved: {filename} and depth_raw_{frame_count}.npy")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\nCamera test complete.")


def test_known_distances():
    """
    Test with objects at known distances to verify normalization.

    Instructions:
    1. Place an object at exactly 0.3m (near clip) - should show ~-0.5 (dark blue)
    2. Place an object at exactly 1.5m (middle) - should show ~0.0 (green)
    3. Place an object at exactly 3.0m (far clip) - should show ~+0.5 (red)
    """
    print("\n" + "=" * 60)
    print("Distance Calibration Test")
    print("=" * 60)
    print("Place objects at known distances to verify:")
    print("  0.3m → should be DARK BLUE (value ~ -0.5)")
    print("  1.5m → should be GREEN (value ~ 0.0)")
    print("  3.0m → should be RED (value ~ +0.5)")
    print("=" * 60)

    camera = create_camera(
        use_real=True,
        target_width=DeployConfig.depth_width,
        target_height=DeployConfig.depth_height,
        near_clip=DeployConfig.depth_near,
        far_clip=DeployConfig.depth_far,
    )

    camera.start()
    time.sleep(2.0)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 640, 480)

    print("\nClick on the image to see depth value at that point")
    print("Press 'q' to quit\n")

    current_depth = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_depth
        if event == cv2.EVENT_LBUTTONDOWN and current_depth is not None:
            # Scale click position to depth image size
            h, w = current_depth.shape
            img_h, img_w = 480, 640
            dx = int(x * w / img_w)
            dy = int(y * h / img_h)
            if 0 <= dx < w and 0 <= dy < h:
                val = current_depth[dy, dx]
                # Convert back to meters
                meters = val * (DeployConfig.depth_far - DeployConfig.depth_near) + DeployConfig.depth_near + 0.5 * (DeployConfig.depth_far - DeployConfig.depth_near)
                # Actually: val = (depth - near) / (far - near) - 0.5
                # So: depth = (val + 0.5) * (far - near) + near
                meters = (val + 0.5) * (DeployConfig.depth_far - DeployConfig.depth_near) + DeployConfig.depth_near
                print(f"Click at ({dx}, {dy}): value={val:.3f}, ~{meters:.2f}m")

    cv2.setMouseCallback("Calibration", mouse_callback)

    try:
        while True:
            depth = camera.get_depth()
            current_depth = depth

            display = ((depth + 0.5) * 255).astype(np.uint8)
            display_color = cv2.applyColorMap(display, cv2.COLORMAP_JET)

            # Add colorbar legend
            cv2.putText(display_color, "CLOSE (0.3m)", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(display_color, "FAR (3.0m)", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow("Calibration", display_color)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test depth camera")
    parser.add_argument("--dummy", action="store_true", help="Use dummy camera")
    parser.add_argument("--calibrate", action="store_true",
                       help="Run distance calibration test")
    args = parser.parse_args()

    if args.calibrate:
        test_known_distances()
    else:
        test_camera(use_dummy=args.dummy)
