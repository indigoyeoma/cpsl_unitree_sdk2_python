#!/usr/bin/env python3
"""
Depth Encoder Process - Runs separately from policy.

Captures depth frames, runs depth encoder, writes embedding to shared memory.
Runs at ~10Hz (or as fast as the encoder allows).
"""
import time
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Array, Value
from ctypes import c_float, c_bool

from config import (
    DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH, N_PROPRIO,
    VISION_WEIGHT_PATH
)


class DepthOnlyFCBackbone48x64(nn.Module):
    """CNN backbone for 48x64 depth images (matches training)."""

    def __init__(self, num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()

        self.image_compression = nn.Sequential(
            # Input: [1, 48, 64]
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            # [32, 44, 60]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 22, 30]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 20, 28]
            activation,
            nn.Flatten(),
            nn.Linear(64 * 20 * 28, 128),
            activation,
            nn.Linear(128, 32)
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        compressed = self.image_compression(images)
        return self.output_activation(compressed)


class SimpleDepthEncoder(nn.Module):
    """Feedforward depth encoder (no GRU)."""

    def __init__(self, n_proprio: int = N_PROPRIO):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.base_backbone = DepthOnlyFCBackbone48x64(num_frames=1)

        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(32, 32 + 2),
            last_activation
        )

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        depth_latent = self.base_backbone(depth_image)
        combined = torch.cat([depth_latent, proprioception], dim=-1)
        latent = self.combination_mlp(combined)
        output = self.output_mlp(latent)
        return output


def depth_encoder_loop(
    shared_embedding: Array,
    shared_proprio: Array,
    embedding_ready: Value,
    proprio_ready: Value,
    should_stop: Value,
    use_camera: bool = True,
    show_gui: bool = False,
    save_images: Value = None
):
    """
    Main loop for depth encoder process.

    Args:
        shared_embedding: Shared array for 32-dim embedding output
        shared_proprio: Shared array for N_PROPRIO proprio input from policy
        embedding_ready: Flag indicating new embedding is available
        proprio_ready: Flag indicating new proprio is available
        should_stop: Flag to stop the process
        use_camera: Whether to use real camera or dummy frames
        show_gui: Whether to display depth visualization window
        save_images: Flag to save depth images (set by main process)
    """
    print("[DepthEncoder] Starting depth encoder process...")

    # Import OpenCV if GUI is requested
    cv2 = None
    if show_gui:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
            print("[DepthEncoder] GUI enabled - will show depth visualization")
        except ImportError:
            print("[DepthEncoder] WARNING: OpenCV not available, GUI disabled")
            show_gui = False

    # Use CPU - often faster for small models on Jetson due to less overhead
    device = torch.device("cpu")
    print(f"[DepthEncoder] Using device: {device}")

    # Load depth encoder
    print(f"[DepthEncoder] Loading weights from: {VISION_WEIGHT_PATH}")
    state_dict = torch.load(VISION_WEIGHT_PATH, map_location=device)

    depth_encoder = SimpleDepthEncoder(n_proprio=N_PROPRIO)
    depth_encoder.load_state_dict(state_dict['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    print("[DepthEncoder] Model loaded")

    # Initialize camera
    camera = None
    if use_camera:
        from depth_camera import DepthCamera
        camera = DepthCamera(enable_filters=True)
        if not camera.start():
            print("[DepthEncoder] WARNING: Failed to start camera, using dummy frames")
            use_camera = False

    # Warmup
    print("[DepthEncoder] Running warmup...")
    dummy_depth = torch.zeros(1, DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH, dtype=torch.float32, device=device)
    dummy_proprio = torch.zeros(1, N_PROPRIO, dtype=torch.float32, device=device)

    for i in range(5):
        t0 = time.time()
        with torch.no_grad():
            _ = depth_encoder(dummy_depth, dummy_proprio)
        t1 = time.time()
        print(f"[DepthEncoder] Warmup {i+1}/5: {(t1-t0)*1000:.1f}ms")

    print("[DepthEncoder] Ready! Starting main loop...")

    # Image saving state
    images_to_save = 0
    # Save to current working directory (works on robot as any user)
    import os
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_captures")

    loop_count = 0
    while not should_stop.value:
        t_start = time.time()

        # Check if we should start saving images
        if save_images is not None and save_images.value and images_to_save == 0:
            images_to_save = 10  # Save 10 images
            save_images.value = False  # Reset flag
            # Create save directory
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_subdir = os.path.join(save_dir, f"capture_{timestamp}")
            os.makedirs(save_subdir, exist_ok=True)
            print(f"[DepthEncoder] Saving 10 depth images to {save_subdir}")
            save_count = 0

        # Get depth frame
        if use_camera and camera is not None:
            depth_frame = camera.get_frame()
            if depth_frame is None:
                depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)
        else:
            depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)

        # Save depth image if requested
        if images_to_save > 0:
            # Save processed frame
            save_path = os.path.join(save_subdir, f"depth_{save_count:03d}_processed.npy")
            np.save(save_path, depth_frame)

            # Save raw frame if available
            if camera is not None:
                raw_frame = camera.get_raw_frame()
                if raw_frame is not None:
                    raw_path = os.path.join(save_subdir, f"depth_{save_count:03d}_raw.npy")
                    np.save(raw_path, raw_frame)
                    print(f"[DepthEncoder] Saved {save_count+1}/10: raw={raw_frame.shape} processed={depth_frame.shape}")
                else:
                    print(f"[DepthEncoder] Saved {save_count+1}/10: processed only (no raw)")
            else:
                print(f"[DepthEncoder] Saved {save_count+1}/10: processed={depth_frame.shape}")

            save_count += 1
            images_to_save -= 1
            if images_to_save == 0:
                print(f"[DepthEncoder] Done! Images saved to: {save_subdir}")
            time.sleep(0.1)  # Small delay between saves

        # Get proprio from shared memory (written by policy process)
        proprio_np = np.array(shared_proprio[:], dtype=np.float32)

        # Convert to tensors
        depth_tensor = torch.from_numpy(depth_frame).float().unsqueeze(0).to(device)
        proprio_tensor = torch.from_numpy(proprio_np).float().unsqueeze(0).to(device)

        # Run encoder
        with torch.no_grad():
            output = depth_encoder(depth_tensor, proprio_tensor)
            embedding = output[0, :32].numpy()  # First 32 dims (not yaw prediction)

        # Write to shared memory (32 depth latent + 2 yaw prediction = 34 total)
        for i in range(32):
            shared_embedding[i] = embedding[i]

        # =====================================================================
        # Yaw Prediction Options
        # =====================================================================
        # Option A: Use depth encoder prediction (for terrain with visual features)
        # Option B: Use fixed goal direction (for flat ground testing)
        # =====================================================================

        USE_DEPTH_ENCODER_YAW = False  # Disabled for debugging - robot walks straight

        # Fixed goal direction (used when USE_DEPTH_ENCODER_YAW = False)
        # GOAL_X: meters ahead (always positive)
        # GOAL_Y: meters to the side (positive = left, negative = right)
        GOAL_X = 10.0   # 10 meters ahead
        GOAL_Y = 0.0    # 0 = straight, +2 = 2m left, -2 = 2m right

        # Hardware drift correction (add to delta_yaw to counteract drift)
        DRIFT_CORRECTION = 0.25  # positive = turn left, tune if needed

        import math

        if USE_DEPTH_ENCODER_YAW:
            # Use depth encoder's yaw prediction (for navigating terrain)
            # Output is [delta_yaw, delta_next_yaw] in RADIANS (not sin/cos!)
            # Model output is Tanh [-1,1], scale by 1.5 to get [-1.5, 1.5] radians
            yaw_pred = output[0, 32:34].numpy() * 1.5

            # =====================================================================
            # CALIBRATION: Bias correction for straight-line walking
            # Set to 0 initially, then tune based on observed drift:
            #   - Robot drifts LEFT: increase this value (e.g., 0.05)
            #   - Robot drifts RIGHT: decrease this value (e.g., -0.05)
            # =====================================================================
            YAW_BIAS_CORRECTION = 0.0  # Start with 0, tune if needed

            shared_embedding[32] = yaw_pred[0] - YAW_BIAS_CORRECTION  # delta_yaw (radians)
            shared_embedding[33] = yaw_pred[1]   # delta_next_yaw (radians)
        else:
            # Use fixed goal direction
            delta_yaw = math.atan2(GOAL_Y, GOAL_X) + DRIFT_CORRECTION
            shared_embedding[32] = delta_yaw  # delta_yaw in radians
            shared_embedding[33] = 0.0

        embedding_ready.value = True

        t_end = time.time()
        loop_count += 1

        # Print timing periodically
        if loop_count % 50 == 0:
            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
            print(f"[DepthEncoder] Loop {loop_count}: {(t_end-t_start)*1000:.1f}ms ({fps:.1f} Hz)")

        # Show depth visualization if GUI enabled
        if show_gui and cv2 is not None:
            # Convert normalized depth [-0.5, 0.5] to display range [0, 255]
            display_frame = ((depth_frame + 0.5) * 255).astype(np.uint8)
            # Apply colormap for better visualization
            display_color = cv2.applyColorMap(display_frame, cv2.COLORMAP_VIRIDIS)
            # Resize for better visibility (4x)
            display_large = cv2.resize(display_color, (DEPTH_OUTPUT_WIDTH * 4, DEPTH_OUTPUT_HEIGHT * 4),
                                       interpolation=cv2.INTER_NEAREST)
            # Add text overlay
            cv2.putText(display_large, f"Depth Buffer (Loop {loop_count})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Depth Buffer", display_large)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[DepthEncoder] 'q' pressed, stopping...")
                should_stop.value = True

        # Small sleep to prevent busy-waiting
        time.sleep(0.001)

    # Cleanup
    if camera is not None:
        camera.stop()
    if show_gui and cv2 is not None:
        cv2.destroyAllWindows()
    print("[DepthEncoder] Process stopped")


if __name__ == "__main__":
    # Test the depth encoder standalone
    from multiprocessing import Array, Value

    # 32 depth latent + 2 yaw prediction = 34 total
    shared_embedding = Array(c_float, 34)
    shared_proprio = Array(c_float, N_PROPRIO)
    embedding_ready = Value(c_bool, False)
    proprio_ready = Value(c_bool, False)
    should_stop = Value(c_bool, False)

    # Initialize proprio with zeros
    for i in range(N_PROPRIO):
        shared_proprio[i] = 0.0

    import signal
    def signal_handler(sig, frame):
        print("\n[DepthEncoder] Stopping...")
        should_stop.value = True

    signal.signal(signal.SIGINT, signal_handler)

    depth_encoder_loop(
        shared_embedding,
        shared_proprio,
        embedding_ready,
        proprio_ready,
        should_stop,
        use_camera=True
    )
