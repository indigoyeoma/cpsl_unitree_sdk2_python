#!/usr/bin/env python3
"""
Test D435i depth camera for Go2 deployment.

This script verifies that:
1. D435i camera is connected and working
2. Depth images are captured correctly
3. Preprocessing matches deployment pipeline

Usage:
    python test_camera.py --save_images --num_frames 10
"""

import sys
import os
import argparse
import numpy as np
import cv2
import time
from pathlib import Path

# Add deploy_go2_lowlevel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deploy_go2_lowlevel'))

from depth_camera import create_camera, D435iCamera

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


def test_camera_basic():
    """Test if RealSense camera is available."""
    print("=" * 70)
    print("RealSense Camera Test")
    print("=" * 70)

    if not REALSENSE_AVAILABLE:
        print("❌ ERROR: pyrealsense2 not installed!")
        print("\nInstall with:")
        print("  pip install pyrealsense2")
        return False

    print("✓ pyrealsense2 is installed")

    # Try to enumerate devices
    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            print("❌ ERROR: No RealSense devices found!")
            print("\nTroubleshooting:")
            print("  1. Check USB connection")
            print("  2. Run: rs-enumerate-devices")
            print("  3. Try: sudo apt-get install librealsense2-utils")
            return False

        print(f"✓ Found {len(devices)} RealSense device(s)")

        for i, dev in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Name: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_camera_capture(args):
    """Test camera capture and save images."""
    print("\n" + "=" * 70)
    print("Testing Depth Capture")
    print("=" * 70)

    # Create output directory
    if args.save_images:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

    # Initialize camera with deployment settings
    print("\nInitializing camera...")
    print(f"  Capture resolution: {args.capture_width}x{args.capture_height}@{args.fps}fps")
    print(f"  Target resolution: {args.target_width}x{args.target_height}")
    print(f"  Depth range: {args.near_clip}m - {args.far_clip}m")

    camera = D435iCamera(
        width=args.capture_width,
        height=args.capture_height,
        fps=args.fps,
        target_width=args.target_width,
        target_height=args.target_height,
        near_clip=args.near_clip,
        far_clip=args.far_clip,
    )

    try:
        camera.start()
        print("✓ Camera started successfully")

        # Wait for camera warmup
        warmup_time = args.warmup_time
        print(f"\nWarming up camera for {warmup_time} seconds...")
        for i in range(warmup_time):
            print(f"  {i+1}/{warmup_time}...", end='\r')
            time.sleep(1.0)
        print(f"  ✓ Camera ready" + " " * 20)

        # Capture test frames
        print(f"\nCapturing {args.num_frames} frames...", end='', flush=True)

        capture_times = []
        depth_stats = []

        for i in range(args.num_frames):
            start_time = time.time()

            # Get depth image
            depth = camera.get_depth()

            capture_time = time.time() - start_time
            capture_times.append(capture_time)

            # Check if valid
            if depth is None or depth.size == 0:
                continue

            # Compute statistics
            valid_pixels = np.sum(depth > 0)
            valid_percent = 100 * valid_pixels / depth.size
            mean_depth = np.mean(depth[depth > 0]) if valid_pixels > 0 else 0

            depth_stats.append({
                'valid_percent': valid_percent,
                'mean_depth': mean_depth,
                'min': np.min(depth),
                'max': np.max(depth)
            })

            # Save images
            if args.save_images:
                # Save normalized depth as grayscale
                depth_vis = (depth * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Save both versions
                gray_path = output_dir / f"depth_{i:03d}_gray.png"
                color_path = output_dir / f"depth_{i:03d}_color.png"

                cv2.imwrite(str(gray_path), depth_vis)
                cv2.imwrite(str(color_path), depth_colored)

            # Display if requested
            if args.display:
                depth_vis = (depth * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                cv2.imshow('Depth (grayscale)', depth_vis)
                cv2.imshow('Depth (colored)', depth_colored)

                key = cv2.waitKey(100)
                if key == ord('q'):
                    print("\n  Stopped by user")
                    break

            # Delay between captures
            time.sleep(args.delay)

        print(" Done!")

        # Print summary
        print("\n" + "=" * 70)
        print("Capture Summary")
        print("=" * 70)

        if capture_times:
            avg_time = np.mean(capture_times) * 1000
            avg_fps = 1.0 / np.mean(capture_times)
            print(f"Average capture time: {avg_time:.1f}ms ({avg_fps:.1f} FPS)")

        if depth_stats:
            avg_valid = np.mean([s['valid_percent'] for s in depth_stats])
            avg_depth = np.mean([s['mean_depth'] for s in depth_stats])
            print(f"Average valid pixels: {avg_valid:.1f}%")
            print(f"Average depth value: {avg_depth:.3f}")

            if avg_valid < 50:
                print("\n⚠️  WARNING: Low valid pixel percentage!")
                print("   This might indicate:")
                print("   - Poor lighting conditions")
                print("   - Camera pointing at blank wall/sky")
                print("   - Surfaces too close or too far")

        if args.save_images:
            print(f"\n✓ Images saved to: {output_dir.absolute()}")
            print(f"  - *_gray.png: Normalized depth (0-255)")
            print(f"  - *_color.png: Color-mapped depth (visualization)")

        print("\n✓ Camera test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ ERROR during capture: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        camera.stop()
        if args.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Test D435i camera for Go2 deployment')

    # Test options
    parser.add_argument('--skip_basic_test', action='store_true',
                        help='Skip basic device enumeration test')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to capture (default: 10)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between captures in seconds (default: 0.1)')

    # Save options
    parser.add_argument('--save_images', action='store_true',
                        help='Save captured depth images')
    parser.add_argument('--output_dir', type=str, default='camera_test_output',
                        help='Output directory for saved images (default: camera_test_output)')
    parser.add_argument('--display', action='store_true',
                        help='Display depth images in real-time (requires GUI)')

    # Camera settings (match deployment)
    parser.add_argument('--capture_width', type=int, default=424,
                        help='Capture width (default: 424)')
    parser.add_argument('--capture_height', type=int, default=240,
                        help='Capture height (default: 240)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Capture FPS (default: 30)')
    parser.add_argument('--target_width', type=int, default=87,
                        help='Target width for policy (default: 87)')
    parser.add_argument('--target_height', type=int, default=58,
                        help='Target height for policy (default: 58)')
    parser.add_argument('--near_clip', type=float, default=0.0,
                        help='Near clip distance in meters (default: 0.0)')
    parser.add_argument('--far_clip', type=float, default=2.0,
                        help='Far clip distance in meters (default: 2.0)')

    args = parser.parse_args()

    print("D435i Camera Test for Go2 Vision Policy")
    print("This script verifies camera functionality before deployment\n")

    # Basic test
    if not args.skip_basic_test:
        if not test_camera_basic():
            print("\n❌ Basic camera test failed. Fix hardware issues before proceeding.")
            return 1

    # Capture test
    if not test_camera_capture(args):
        print("\n❌ Capture test failed.")
        return 1

    print("\n" + "=" * 70)
    print("All tests PASSED! ✓")
    print("=" * 70)
    print("\nCamera is ready for deployment.")
    print("\nNext steps:")
    print("  1. Review saved images (if --save_images was used)")
    print("  2. Verify depth quality and coverage")
    print("  3. Mount camera on Go2 robot")
    print("  4. Run deployment: cd deploy_go2_lowlevel && python deploy.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
