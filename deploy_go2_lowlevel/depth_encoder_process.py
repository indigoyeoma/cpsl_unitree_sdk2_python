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


class DepthOnlyFCBackbone58x87(nn.Module):
    """CNN backbone for 58x87 depth images."""

    def __init__(self, num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()

        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
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

        self.base_backbone = DepthOnlyFCBackbone58x87(num_frames=1)

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
    use_camera: bool = True
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
    """
    print("[DepthEncoder] Starting depth encoder process...")

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

    loop_count = 0
    while not should_stop.value:
        t_start = time.time()

        # Get depth frame
        if use_camera and camera is not None:
            depth_frame = camera.get_frame()
            if depth_frame is None:
                depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)
        else:
            depth_frame = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)

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

        # Also write yaw prediction (last 2 dims of output, scaled by 1.5 like parkour)
        yaw_pred = output[0, 32:34].numpy() * 1.5
        shared_embedding[32] = yaw_pred[0]  # delta_yaw_sin
        shared_embedding[33] = yaw_pred[1]  # delta_yaw_cos

        embedding_ready.value = True

        t_end = time.time()
        loop_count += 1

        # Print timing periodically
        if loop_count % 50 == 0:
            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
            print(f"[DepthEncoder] Loop {loop_count}: {(t_end-t_start)*1000:.1f}ms ({fps:.1f} Hz)")

        # Small sleep to prevent busy-waiting
        time.sleep(0.001)

    # Cleanup
    if camera is not None:
        camera.stop()
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
