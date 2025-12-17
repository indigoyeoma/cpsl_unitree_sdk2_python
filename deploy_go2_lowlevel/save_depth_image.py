"""
Save depth camera images for debugging.

Usage:
    python save_depth_image.py              # Save single frame
    python save_depth_image.py --raw        # Also save raw (unprocessed) frame
    python save_depth_image.py --continuous # Save frames continuously
"""
import numpy as np
import time
import argparse
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: cv2 not available, will save as .npy files")

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("WARNING: pyrealsense2 not installed")

from config import (
    DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS,
    CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT,
    DEPTH_OUTPUT_WIDTH, DEPTH_OUTPUT_HEIGHT,
    DEPTH_NEAR, DEPTH_FAR
)


def capture_raw_frame():
    """Capture depth and RGB frames from the RealSense camera."""
    if not HAS_REALSENSE:
        print("RealSense not available!")
        return None, None, None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, DEPTH_FPS)

    try:
        pipeline.start(config)

        # Wait for auto-exposure to stabilize
        print("Waiting for camera to stabilize...")
        for _ in range(30):
            pipeline.wait_for_frames()

        # Capture frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame:
            print("Failed to get depth frame")
            return None, None, None

        # Get raw depth data (in mm)
        raw_depth = np.asanyarray(depth_frame.get_data())

        # Get RGB image
        rgb_image = None
        if color_frame:
            rgb_image = np.asanyarray(color_frame.get_data())
        else:
            print("Warning: No RGB frame available")

        # Apply filters for processed version
        hole_filter = rs.hole_filling_filter()
        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 5)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        spatial_filter.set_option(rs.option.holes_fill, 4)

        filtered_frame = hole_filter.process(depth_frame)
        filtered_frame = spatial_filter.process(filtered_frame)
        filtered_depth = np.asanyarray(filtered_frame.get_data())

        pipeline.stop()

        return raw_depth, filtered_depth, rgb_image

    except Exception as e:
        print(f"Error capturing frame: {e}")
        try:
            pipeline.stop()
        except:
            pass
        return None, None, None


def preprocess_depth(depth_mm):
    """Preprocess depth image same as in deployment."""
    # Crop edges
    h, w = depth_mm.shape
    crop_h_end = h - CROP_BOTTOM if CROP_BOTTOM > 0 else h
    crop_w_end = w - CROP_RIGHT if CROP_RIGHT > 0 else w

    cropped = depth_mm[CROP_TOP:crop_h_end, CROP_LEFT:crop_w_end]

    # Convert to meters
    depth_m = cropped.astype(np.float32) / 1000.0

    # Clip to valid range
    depth_m = np.clip(depth_m, DEPTH_NEAR, DEPTH_FAR)

    # Resize
    if HAS_CV2:
        resized = cv2.resize(depth_m, (DEPTH_OUTPUT_WIDTH, DEPTH_OUTPUT_HEIGHT),
                            interpolation=cv2.INTER_AREA)
    else:
        # Simple nearest neighbor
        target_h, target_w = DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH
        row_indices = (np.arange(target_h) * cropped.shape[0] / target_h).astype(int)
        col_indices = (np.arange(target_w) * cropped.shape[1] / target_w).astype(int)
        resized = depth_m[row_indices[:, None], col_indices]

    # Normalize to [0, 1]
    normalized = (resized - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)

    return normalized


def save_images(raw_depth, filtered_depth, rgb_image, output_dir=".", prefix="depth"):
    """Save depth and RGB images in multiple formats."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # 1. Save RGB image first (most useful for debugging)
    if rgb_image is not None and HAS_CV2:
        rgb_path = os.path.join(output_dir, f"{prefix}_RGB_{timestamp}.png")
        cv2.imwrite(rgb_path, rgb_image)
        saved_files.append(rgb_path)
        print(f"Saved RGB image: {rgb_path}")
        print(f"  Shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

    # 2. Save raw depth (mm) as numpy
    raw_path = os.path.join(output_dir, f"{prefix}_raw_{timestamp}.npy")
    np.save(raw_path, raw_depth)
    saved_files.append(raw_path)
    print(f"\nSaved raw depth: {raw_path}")
    print(f"  Shape: {raw_depth.shape}, dtype: {raw_depth.dtype}")
    print(f"  Min: {raw_depth.min()} mm, Max: {raw_depth.max()} mm")
    print(f"  Mean: {raw_depth.mean():.1f} mm, Nonzero: {np.count_nonzero(raw_depth)}/{raw_depth.size}")

    # 3. Save filtered depth (mm) as numpy
    filtered_path = os.path.join(output_dir, f"{prefix}_filtered_{timestamp}.npy")
    np.save(filtered_path, filtered_depth)
    saved_files.append(filtered_path)
    print(f"\nSaved filtered depth: {filtered_path}")
    print(f"  Shape: {filtered_depth.shape}, dtype: {filtered_depth.dtype}")
    print(f"  Min: {filtered_depth.min()} mm, Max: {filtered_depth.max()} mm")

    # 4. Save preprocessed (what policy sees) as numpy
    processed = preprocess_depth(filtered_depth)
    processed_path = os.path.join(output_dir, f"{prefix}_processed_{timestamp}.npy")
    np.save(processed_path, processed)
    saved_files.append(processed_path)
    print(f"\nSaved processed depth (policy input): {processed_path}")
    print(f"  Shape: {processed.shape}, dtype: {processed.dtype}")
    print(f"  Min: {processed.min():.4f}, Max: {processed.max():.4f}")
    print(f"  Mean: {processed.mean():.4f}")

    if HAS_CV2:
        # 5. Save raw as grayscale visualization (scaled to 0-255)
        raw_vis = (raw_depth.astype(np.float32) / raw_depth.max() * 255).astype(np.uint8)
        raw_vis_path = os.path.join(output_dir, f"{prefix}_raw_grayscale_{timestamp}.png")
        cv2.imwrite(raw_vis_path, raw_vis)
        saved_files.append(raw_vis_path)
        print(f"\nSaved raw grayscale: {raw_vis_path}")

        # 6. Save raw as COLOR visualization (JET colormap)
        raw_color = cv2.applyColorMap(raw_vis, cv2.COLORMAP_JET)
        raw_color_path = os.path.join(output_dir, f"{prefix}_raw_COLOR_{timestamp}.png")
        cv2.imwrite(raw_color_path, raw_color)
        saved_files.append(raw_color_path)
        print(f"Saved raw COLOR: {raw_color_path}")

        # 7. Save processed as grayscale visualization
        processed_vis = (processed * 255).astype(np.uint8)
        processed_vis_path = os.path.join(output_dir, f"{prefix}_processed_grayscale_{timestamp}.png")
        cv2.imwrite(processed_vis_path, processed_vis)
        saved_files.append(processed_vis_path)
        print(f"Saved processed grayscale: {processed_vis_path}")

        # 8. Save processed as COLOR visualization (JET colormap) - THIS IS WHAT POLICY SEES
        processed_color = cv2.applyColorMap(processed_vis, cv2.COLORMAP_JET)
        processed_color_path = os.path.join(output_dir, f"{prefix}_processed_COLOR_{timestamp}.png")
        cv2.imwrite(processed_color_path, processed_color)
        saved_files.append(processed_color_path)
        print(f"Saved processed COLOR (policy input): {processed_color_path}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Save depth camera images for debugging")
    parser.add_argument("--output", "-o", default="camera_debug", help="Output directory")
    parser.add_argument("--continuous", "-c", action="store_true", help="Continuous capture mode")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between captures (seconds)")
    args = parser.parse_args()

    print("=" * 50)
    print("Depth Camera Image Saver")
    print("=" * 50)
    print(f"\nCamera settings:")
    print(f"  Resolution: {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS}fps")
    print(f"  Crop: top={CROP_TOP}, bottom={CROP_BOTTOM}, left={CROP_LEFT}, right={CROP_RIGHT}")
    print(f"  Output size: {DEPTH_OUTPUT_WIDTH}x{DEPTH_OUTPUT_HEIGHT}")
    print(f"  Depth range: {DEPTH_NEAR}m - {DEPTH_FAR}m")
    print()

    if args.continuous:
        print("Continuous mode - Press Ctrl+C to stop")
        count = 0
        try:
            while True:
                raw, filtered, rgb = capture_raw_frame()
                if raw is not None:
                    save_images(raw, filtered, rgb, args.output, f"depth_{count:04d}")
                    count += 1
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nStopped. Captured {count} frames.")
    else:
        print("Capturing single frame...")
        raw, filtered, rgb = capture_raw_frame()
        if raw is not None:
            files = save_images(raw, filtered, rgb, args.output)
            print(f"\n{'=' * 50}")
            print("Done! Files saved:")
            for f in files:
                print(f"  {f}")
        else:
            print("Failed to capture frame")


if __name__ == "__main__":
    main()
