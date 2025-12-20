#!/usr/bin/env python3
"""
Test D435i depth camera for Go2 deployment - with obstacle detection diagnostics.

This script verifies:
1. D435i camera is connected and working
2. Depth images match deployment preprocessing
3. Obstacles are visible and properly detected (key for sim-to-real debugging)

Usage:
    python test_camera.py                           # Basic test
    python test_camera.py --save_images             # Save images
    python test_camera.py --obstacle_distance 0.5   # Test with known obstacle at 0.5m
    python test_camera.py --continuous              # Continuous mode for debugging
"""

import sys
import os
import argparse
import numpy as np
import cv2
import time
from pathlib import Path

# Add deploy_go2_lowlevel to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_camera import DepthCamera
from config import (
    DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS,
    DEPTH_OUTPUT_WIDTH, DEPTH_OUTPUT_HEIGHT,
    DEPTH_NEAR, DEPTH_FAR,
    CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT
)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


def test_camera_basic():
    """Test if RealSense camera is available."""
    print("=" * 70)
    print("RealSense Camera Basic Test")
    print("=" * 70)

    if not REALSENSE_AVAILABLE:
        print("ERROR: pyrealsense2 not installed!")
        print("\nInstall with:")
        print("  pip install pyrealsense2")
        return False

    print("pyrealsense2 is installed")

    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            print("ERROR: No RealSense devices found!")
            print("\nTroubleshooting:")
            print("  1. Check USB connection (USB 3.0 recommended)")
            print("  2. Run: rs-enumerate-devices")
            print("  3. Check: lsusb | grep Intel")
            return False

        print(f"Found {len(devices)} RealSense device(s)")

        for i, dev in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Name: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
            try:
                usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                print(f"  USB Type: {usb_type}")
                if "2." in usb_type:
                    print("  WARNING: USB 2.0 detected - expect higher latency!")
            except:
                pass

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def analyze_obstacle_visibility(depth_frame: np.ndarray, expected_distance: float = None):
    """
    Analyze depth frame to check obstacle visibility.

    Args:
        depth_frame: Normalized depth frame [-0.5, 0.5]
        expected_distance: Expected obstacle distance in meters (optional)

    Returns:
        dict with analysis results
    """
    # Convert back to meters for analysis
    depth_m = (depth_frame + 0.5) * (DEPTH_FAR - DEPTH_NEAR) + DEPTH_NEAR

    h, w = depth_frame.shape
    results = {
        'valid_pixels': np.sum(depth_m > DEPTH_NEAR),
        'valid_percent': 100 * np.sum(depth_m > DEPTH_NEAR) / depth_frame.size,
        'min_depth_m': np.min(depth_m[depth_m > DEPTH_NEAR]) if np.any(depth_m > DEPTH_NEAR) else 0,
        'max_depth_m': np.max(depth_m),
        'mean_depth_m': np.mean(depth_m[depth_m > DEPTH_NEAR]) if np.any(depth_m > DEPTH_NEAR) else 0,
    }

    # Analyze regions (bottom=close obstacles, top=far)
    bottom_third = depth_m[2*h//3:, :]
    middle_third = depth_m[h//3:2*h//3, :]
    top_third = depth_m[:h//3, :]

    results['bottom_mean_m'] = np.mean(bottom_third[bottom_third > DEPTH_NEAR]) if np.any(bottom_third > DEPTH_NEAR) else 0
    results['middle_mean_m'] = np.mean(middle_third[middle_third > DEPTH_NEAR]) if np.any(middle_third > DEPTH_NEAR) else 0
    results['top_mean_m'] = np.mean(top_third[top_third > DEPTH_NEAR]) if np.any(top_third > DEPTH_NEAR) else 0

    # Check for close obstacles (< 0.8m)
    close_mask = (depth_m > DEPTH_NEAR) & (depth_m < 0.8)
    results['close_obstacle_pixels'] = np.sum(close_mask)
    results['close_obstacle_percent'] = 100 * np.sum(close_mask) / depth_frame.size

    # Check center region specifically (where robot is heading)
    center_col_start = w // 3
    center_col_end = 2 * w // 3
    center_region = depth_m[:, center_col_start:center_col_end]
    center_valid = center_region[center_region > DEPTH_NEAR]
    results['center_min_m'] = np.min(center_valid) if len(center_valid) > 0 else 0
    results['center_mean_m'] = np.mean(center_valid) if len(center_valid) > 0 else 0

    # If expected distance provided, check if obstacle is visible
    if expected_distance is not None:
        tolerance = 0.15  # 15cm tolerance
        obstacle_mask = (depth_m > expected_distance - tolerance) & (depth_m < expected_distance + tolerance)
        results['expected_distance_m'] = expected_distance
        results['obstacle_detected'] = np.sum(obstacle_mask) > (depth_frame.size * 0.01)  # >1% of pixels
        results['obstacle_pixels'] = np.sum(obstacle_mask)

    return results


def create_debug_visualization(depth_frame: np.ndarray, analysis: dict):
    """Create visualization with obstacle analysis overlay."""
    # Convert to display range [0, 255]
    display = ((depth_frame + 0.5) * 255).astype(np.uint8)
    display_color = cv2.applyColorMap(display, cv2.COLORMAP_VIRIDIS)

    # Resize for better visibility (4x)
    scale = 4
    h, w = display_color.shape[:2]
    display_large = cv2.resize(display_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Add text overlay
    y_offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)

    texts = [
        f"Valid: {analysis['valid_percent']:.1f}%",
        f"Min: {analysis['min_depth_m']:.2f}m  Max: {analysis['max_depth_m']:.2f}m",
        f"Center min: {analysis['center_min_m']:.2f}m",
        f"Close obstacles (<0.8m): {analysis['close_obstacle_percent']:.1f}%",
        f"Bottom: {analysis['bottom_mean_m']:.2f}m  Top: {analysis['top_mean_m']:.2f}m",
    ]

    if 'obstacle_detected' in analysis:
        status = "VISIBLE" if analysis['obstacle_detected'] else "NOT VISIBLE"
        texts.append(f"Expected {analysis['expected_distance_m']:.2f}m obstacle: {status}")

    for i, text in enumerate(texts):
        cv2.putText(display_large, text, (10, y_offset + i * 18),
                    font, font_scale, color, 1)

    # Draw center region box
    cx1 = (w // 3) * scale
    cx2 = (2 * w // 3) * scale
    cv2.rectangle(display_large, (cx1, 0), (cx2, h * scale), (0, 255, 255), 1)

    return display_large


def test_camera_capture(args):
    """Test camera capture with obstacle detection analysis."""
    print("\n" + "=" * 70)
    print("Depth Camera Capture Test")
    print("=" * 70)

    # Print configuration
    print(f"\nConfiguration (matching deployment):")
    print(f"  Capture: {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS}fps")
    print(f"  Output:  {DEPTH_OUTPUT_WIDTH}x{DEPTH_OUTPUT_HEIGHT}")
    print(f"  Depth range: {DEPTH_NEAR}m - {DEPTH_FAR}m")
    print(f"  Crop: top={CROP_TOP}, bottom={CROP_BOTTOM}, left={CROP_LEFT}, right={CROP_RIGHT}")

    if args.obstacle_distance:
        print(f"\n  Expected obstacle at: {args.obstacle_distance}m")
        print("  Place a clear obstacle at this distance for testing")

    # Create output directory if saving
    output_dir = None
    if args.save_images:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"\n  Output directory: {output_dir}")

    # Initialize camera
    print("\nInitializing camera...")
    camera = DepthCamera(enable_filters=True)

    if not camera.start():
        print("ERROR: Failed to start camera")
        return False

    print("Camera started successfully")

    # Warmup
    print(f"\nWarming up camera...")
    warmup_frames = 30
    for i in range(warmup_frames):
        camera.get_frame()
        time.sleep(0.05)
    print("Camera ready")

    try:
        if args.continuous:
            print("\nContinuous mode - press 'q' to quit, 's' to save frame")
            frame_count = 0
            while True:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                analysis = analyze_obstacle_visibility(frame, args.obstacle_distance)

                # Create visualization
                vis = create_debug_visualization(frame, analysis)
                cv2.imshow("Depth Debug", vis)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and output_dir:
                    path = output_dir / f"depth_{frame_count:04d}.png"
                    cv2.imwrite(str(path), vis)
                    print(f"  Saved: {path}")

                # Print analysis periodically
                if frame_count % 30 == 0:
                    print(f"\n[Frame {frame_count}]")
                    print(f"  Valid: {analysis['valid_percent']:.1f}%")
                    print(f"  Depth range: {analysis['min_depth_m']:.2f}m - {analysis['max_depth_m']:.2f}m")
                    print(f"  Center min: {analysis['center_min_m']:.2f}m")
                    print(f"  Close obstacles: {analysis['close_obstacle_percent']:.1f}%")
                    if 'obstacle_detected' in analysis:
                        status = "VISIBLE" if analysis['obstacle_detected'] else "NOT VISIBLE"
                        print(f"  Expected obstacle: {status} ({analysis['obstacle_pixels']} pixels)")

        else:
            # Single capture mode
            print(f"\nCapturing {args.num_frames} frames...")
            all_analysis = []

            for i in range(args.num_frames):
                frame = camera.get_frame()
                if frame is None:
                    print(f"  Frame {i+1}: No frame")
                    continue

                analysis = analyze_obstacle_visibility(frame, args.obstacle_distance)
                all_analysis.append(analysis)

                print(f"\n[Frame {i+1}/{args.num_frames}]")
                print(f"  Valid pixels: {analysis['valid_percent']:.1f}%")
                print(f"  Depth: min={analysis['min_depth_m']:.2f}m, max={analysis['max_depth_m']:.2f}m, mean={analysis['mean_depth_m']:.2f}m")
                print(f"  Center region min: {analysis['center_min_m']:.2f}m")
                print(f"  Close obstacles (<0.8m): {analysis['close_obstacle_percent']:.1f}%")
                print(f"  By region: bottom={analysis['bottom_mean_m']:.2f}m, middle={analysis['middle_mean_m']:.2f}m, top={analysis['top_mean_m']:.2f}m")

                if 'obstacle_detected' in analysis:
                    status = "VISIBLE" if analysis['obstacle_detected'] else "NOT VISIBLE"
                    print(f"  Expected obstacle at {analysis['expected_distance_m']}m: {status} ({analysis['obstacle_pixels']} pixels)")

                if args.save_images and output_dir:
                    vis = create_debug_visualization(frame, analysis)
                    path = output_dir / f"depth_{i:03d}_debug.png"
                    cv2.imwrite(str(path), vis)

                    # Also save raw normalized depth
                    raw_path = output_dir / f"depth_{i:03d}_raw.npy"
                    np.save(str(raw_path), frame)

                if args.display:
                    vis = create_debug_visualization(frame, analysis)
                    cv2.imshow("Depth Debug", vis)
                    key = cv2.waitKey(500) & 0xFF
                    if key == ord('q'):
                        break

                time.sleep(args.delay)

            # Print summary
            if all_analysis:
                print("\n" + "=" * 70)
                print("Summary")
                print("=" * 70)
                avg_valid = np.mean([a['valid_percent'] for a in all_analysis])
                avg_close = np.mean([a['close_obstacle_percent'] for a in all_analysis])
                avg_center_min = np.mean([a['center_min_m'] for a in all_analysis])

                print(f"Average valid pixels: {avg_valid:.1f}%")
                print(f"Average close obstacles: {avg_close:.1f}%")
                print(f"Average center minimum depth: {avg_center_min:.2f}m")

                if avg_valid < 50:
                    print("\nWARNING: Low valid pixel percentage!")
                    print("  - Check camera is not obstructed")
                    print("  - Ensure adequate lighting")
                    print("  - Avoid pointing at reflective surfaces")

                if args.obstacle_distance:
                    detections = [a.get('obstacle_detected', False) for a in all_analysis]
                    detection_rate = 100 * sum(detections) / len(detections)
                    print(f"\nObstacle detection rate: {detection_rate:.0f}%")

                    if detection_rate < 50:
                        print("\nWARNING: Obstacle not reliably detected!")
                        print("Possible causes:")
                        print("  1. Obstacle too close (< 0.3m, below camera min range)")
                        print("  2. Obstacle below camera FOV (check mounting angle)")
                        print("  3. Obstacle too thin or at edge of FOV")
                        print("  4. Reflective or transparent surface")

        print("\nCamera test complete")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Test D435i camera for Go2 deployment with obstacle detection diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic test:           python test_camera.py
  Test with obstacle:   python test_camera.py --obstacle_distance 0.5
  Continuous debug:     python test_camera.py --continuous --display
  Save for analysis:    python test_camera.py --save_images --num_frames 30
        """
    )

    # Test options
    parser.add_argument('--skip_basic_test', action='store_true',
                        help='Skip basic device enumeration test')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to capture (default: 10)')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay between captures in seconds (default: 0.2)')
    parser.add_argument('--continuous', action='store_true',
                        help='Continuous capture mode for real-time debugging')

    # Obstacle testing
    parser.add_argument('--obstacle_distance', type=float, default=None,
                        help='Expected obstacle distance in meters (e.g., 0.5 for 50cm)')

    # Output options
    parser.add_argument('--save_images', action='store_true',
                        help='Save captured depth images and analysis')
    parser.add_argument('--output_dir', type=str, default='camera_debug',
                        help='Output directory for saved images (default: camera_debug)')
    parser.add_argument('--display', action='store_true',
                        help='Display depth images with analysis overlay')

    args = parser.parse_args()

    print("=" * 70)
    print("D435i Camera Test for Go2 Vision Policy Deployment")
    print("=" * 70)
    print("\nThis script verifies camera functionality and obstacle detection")
    print("Use --obstacle_distance to test detection of known obstacles\n")

    # Print expected config from deployment
    print("Deployment Configuration:")
    print(f"  Depth range: {DEPTH_NEAR}m - {DEPTH_FAR}m")
    print(f"  Output resolution: {DEPTH_OUTPUT_WIDTH}x{DEPTH_OUTPUT_HEIGHT}")
    print()

    # Basic test
    if not args.skip_basic_test:
        if not test_camera_basic():
            print("\nBasic camera test failed. Fix hardware issues before proceeding.")
            return 1

    # Capture test
    if not test_camera_capture(args):
        print("\nCapture test failed.")
        return 1

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

    print("\nNext steps for debugging 25cm obstacle issue:")
    print("  1. Run: python test_camera.py --obstacle_distance 0.25 --display")
    print("     Place a 25cm obstacle and verify it's visible")
    print("  2. Check if obstacle appears in center region of depth image")
    print("  3. Verify depth encoder yaw prediction is enabled in depth_encoder_process.py")
    print("  4. Run deployment with --show-depth to see live depth buffer")
    print()
    print("If obstacle is visible but robot doesn't react:")
    print("  - Check USE_DEPTH_ENCODER_YAW = True in depth_encoder_process.py")
    print("  - Verify depth encoder model is trained on similar obstacles")

    return 0


if __name__ == '__main__':
    sys.exit(main())
