#!/usr/bin/env python3
"""
Depth Encoder Process - Runs separately from policy.

Captures depth frames, runs depth encoder, writes embedding to shared memory.
Runs at ~10Hz (or as fast as the encoder allows).

Architecture matches training (RecurrentDepthBackbone + DepthOnlyFCBackbone58x87):
  - Input: [1, 58, 87] depth image (height=58, width=87)
  - Backbone output: 32-dim
  - combination_mlp: Linear(32+53, 128) -> Linear(128, 32)
  - GRU: input=32, hidden=512
  - output_mlp: Linear(512, 34) -> 32 depth latent + 2 yaw
"""
import time
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Array, Value
from ctypes import c_float, c_bool

from config import (
    DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH, N_PROPRIO,
    VISION_WEIGHT_PATH, DEPTH_LATENT_DIM
)


class DepthOnlyFCBackbone58x87(nn.Module):
    """CNN backbone for 58x87 depth images (matches training DepthOnlyFCBackbone58x87)."""

    def __init__(self, scandots_output_dim: int = 32, num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()

        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 25, 39]
            activation,
            nn.Flatten(),
            # [64 * 25 * 39 = 62400]
            nn.Linear(62400, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.image_compression(images.unsqueeze(1)))


class RecurrentDepthBackbone(nn.Module):
    """
    Recurrent depth encoder matching training RecurrentDepthBackbone.

    Flow: depth[58,87] -> backbone -> [32] -> cat proprio[53] -> [85]
          -> combination_mlp -> [32] -> GRU(512) -> output_mlp -> [34]
    Output: [batch, 34] = 32 depth latent + 2 yaw estimate
    """

    def __init__(self, n_proprio: int = N_PROPRIO, latent_dim: int = 32):
        super().__init__()
        activation = nn.ELU()
        self.latent_dim = latent_dim

        self.base_backbone = DepthOnlyFCBackbone58x87(scandots_output_dim=latent_dim)

        self.combination_mlp = nn.Sequential(
            nn.Linear(latent_dim + n_proprio, 128),
            activation,
            nn.Linear(128, latent_dim)
        )

        self.rnn = nn.GRU(input_size=latent_dim, hidden_size=512, batch_first=True)

        self.output_mlp = nn.Sequential(
            nn.Linear(512, latent_dim + 2),
            nn.Tanh()
        )

        self.hidden_states = None

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        depth_feat = self.base_backbone(depth_image)
        combined = torch.cat([depth_feat, proprioception], dim=-1)
        latent = self.combination_mlp(combined)
        latent_seq, self.hidden_states = self.rnn(latent.unsqueeze(1), self.hidden_states)
        output = self.output_mlp(latent_seq.squeeze(1))
        return output

    def reset_hidden(self):
        self.hidden_states = None

    def detach_hidden(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()


def depth_encoder_loop(
    shared_embedding: Array,
    shared_proprio: Array,
    embedding_ready: Value,
    proprio_ready: Value,
    should_stop: Value,
    use_camera: bool = True,
    show_gui: bool = False,
    save_images: Value = None,
):
    """
    Main loop for depth encoder process.

    Args:
        shared_embedding: Shared array for (DEPTH_LATENT_DIM + 2) floats
                          [0:DEPTH_LATENT_DIM] = depth latent,
                          [DEPTH_LATENT_DIM:] = (delta_yaw, delta_next_yaw)
        shared_proprio: Shared array for N_PROPRIO proprio input (written by policy process)
        embedding_ready: Flag set True each time a new embedding is written
        proprio_ready: Flag set True when proprio is valid (unused currently)
        should_stop: Set True to terminate the loop
        use_camera: Whether to use real RealSense camera or dummy zero frames
        show_gui: Whether to show a live depth visualization window (requires cv2)
        save_images: Optional flag; set True from main process to trigger saving 10 frames
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
    ckpt = torch.load(VISION_WEIGHT_PATH, map_location=device)
    state_dict = ckpt['depth_encoder_state_dict']

    depth_encoder = RecurrentDepthBackbone(n_proprio=N_PROPRIO, latent_dim=DEPTH_LATENT_DIM)
    depth_encoder.load_state_dict(state_dict)
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

    depth_encoder.reset_hidden()
    print("[DepthEncoder] Ready! Starting main loop...")

    # Image saving state
    images_to_save = 0
    import os
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_captures")

    loop_count = 0
    last_frame_ts = 0.0

    while not should_stop.value:
        t_start = time.time()

        # Check if we should start saving images
        if save_images is not None and save_images.value and images_to_save == 0:
            images_to_save = 10
            save_images.value = False
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_subdir = os.path.join(save_dir, f"capture_{timestamp}")
            os.makedirs(save_subdir, exist_ok=True)
            print(f"[DepthEncoder] Saving 10 depth images to {save_subdir}")
            save_count = 0

        # Get depth frame
        if use_camera and camera is not None:
            try:
                current_ts = camera.get_timestamp()
            except AttributeError:
                current_ts = time.time()

            if current_ts <= last_frame_ts and last_frame_ts > 0:
                time.sleep(0.001)
                continue

            last_frame_ts = current_ts
            depth_frame = camera.get_frame()

            if depth_frame is None:
                depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)
        else:
            depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)
            time.sleep(0.03)  # Simulate ~30Hz in dummy mode

        # Save depth image if requested
        if images_to_save > 0:
            save_path = os.path.join(save_subdir, f"depth_{save_count:03d}_processed.npy")
            np.save(save_path, depth_frame)

            try:
                import cv2
                proc_normalized = ((depth_frame + 0.5) * 255).astype(np.uint8)
                proc_colored = cv2.applyColorMap(proc_normalized, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(os.path.join(save_subdir, f"depth_{save_count:03d}_processed.png"), proc_colored)
            except ImportError:
                pass

            if camera is not None:
                raw_frame = camera.get_raw_frame()
                if raw_frame is not None:
                    np.save(os.path.join(save_subdir, f"depth_{save_count:03d}_raw.npy"), raw_frame)
                    try:
                        import cv2
                        raw_clipped = np.clip(raw_frame, 300, 2000)  # 0.3m to 2m in mm
                        raw_normalized = ((raw_clipped - 300) / 1700 * 255).astype(np.uint8)
                        raw_colored = cv2.applyColorMap(raw_normalized, cv2.COLORMAP_VIRIDIS)
                        cv2.imwrite(os.path.join(save_subdir, f"depth_{save_count:03d}_raw.png"), raw_colored)
                    except ImportError:
                        pass
                    print(f"[DepthEncoder] Saved {save_count+1}/10: raw={raw_frame.shape} processed={depth_frame.shape}")
                else:
                    print(f"[DepthEncoder] Saved {save_count+1}/10: processed only (no raw)")

            save_count += 1
            images_to_save -= 1
            if images_to_save == 0:
                print(f"[DepthEncoder] Done! Images saved to: {save_subdir}")
            time.sleep(0.1)

        # Get proprio from shared memory (written by policy process)
        proprio_np = np.array(shared_proprio[:], dtype=np.float32)

        # Convert to tensors
        depth_tensor = torch.from_numpy(depth_frame).float().unsqueeze(0).to(device)
        proprio_tensor = torch.from_numpy(proprio_np).float().unsqueeze(0).to(device)

        # Run encoder (with GRU hidden state maintained across frames)
        t_inf_start = time.time()
        with torch.no_grad():
            output = depth_encoder(depth_tensor, proprio_tensor)
            embedding = output[0, :DEPTH_LATENT_DIM].numpy()  # First 32 dims
        depth_encoder.detach_hidden()
        t_inf_end = time.time()

        # Write depth latent to shared memory (indices 0..31)
        for i in range(DEPTH_LATENT_DIM):
            shared_embedding[i] = float(embedding[i])

        # =====================================================================
        # Yaw Prediction (indices 32 and 33)
        # =====================================================================
        # Option A: Use depth encoder prediction (for terrain with visual features)
        # Option B: Use fixed goal direction (for flat ground testing)
        # =====================================================================

        USE_DEPTH_ENCODER_YAW = False  # Set True to use model's yaw prediction

        # Fixed goal direction (used when USE_DEPTH_ENCODER_YAW = False)
        GOAL_X = 10.0   # meters ahead (always positive)
        GOAL_Y = 0.0    # meters to side (positive=left, negative=right)
        DRIFT_CORRECTION = 0.25  # positive=turn left; tune if robot drifts

        import math

        if USE_DEPTH_ENCODER_YAW:
            # Model output is Tanh [-1,1], scale to [-1.5, 1.5] radians
            yaw_pred = output[0, DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2].numpy() * 1.5
            YAW_BIAS_CORRECTION = 0.0
            shared_embedding[DEPTH_LATENT_DIM] = float(yaw_pred[0] - YAW_BIAS_CORRECTION)
            shared_embedding[DEPTH_LATENT_DIM + 1] = float(yaw_pred[1])
        else:
            delta_yaw = math.atan2(GOAL_Y, GOAL_X) + DRIFT_CORRECTION
            shared_embedding[DEPTH_LATENT_DIM] = float(delta_yaw)
            shared_embedding[DEPTH_LATENT_DIM + 1] = 0.0

        embedding_ready.value = True

        t_end = time.time()
        loop_count += 1

        # Print timing every 50 loops (~5 seconds at 10Hz)
        if loop_count % 50 == 0:
            total_ms = (t_end - t_start) * 1000
            inf_ms = (t_inf_end - t_inf_start) * 1000
            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
            print(f"[Encoder] Loop {loop_count}: total={total_ms:.1f}ms (inf={inf_ms:.1f}ms), rate={fps:.1f} Hz")

        # Show depth visualization if GUI enabled
        if show_gui and cv2 is not None:
            display_frame = ((depth_frame + 0.5) * 255).astype(np.uint8)
            display_color = cv2.applyColorMap(display_frame, cv2.COLORMAP_VIRIDIS)
            display_large = cv2.resize(display_color, (DEPTH_OUTPUT_WIDTH * 4, DEPTH_OUTPUT_HEIGHT * 4),
                                       interpolation=cv2.INTER_NEAREST)
            cv2.putText(display_large, f"Depth Buffer (Loop {loop_count})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Depth Buffer", display_large)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[DepthEncoder] 'q' pressed, stopping...")
                should_stop.value = True

        time.sleep(0.001)

    # Cleanup
    if camera is not None:
        camera.stop()
    if show_gui and cv2 is not None:
        cv2.destroyAllWindows()
    print("[DepthEncoder] Process stopped")


if __name__ == "__main__":
    from multiprocessing import Array, Value

    # 32 depth latent + 2 yaw prediction = 34 total
    shared_embedding = Array(c_float, 34)
    shared_proprio = Array(c_float, N_PROPRIO)
    embedding_ready = Value(c_bool, False)
    proprio_ready = Value(c_bool, False)
    should_stop = Value(c_bool, False)

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
