"""
Policy runner for Go2 vision-based locomotion.

Loads the JIT-traced policy and depth encoder, provides inference methods.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from config import (
    BASE_JIT_PATH, VISION_WEIGHT_PATH,
    N_PROPRIO, N_SCAN, N_PRIV_EXPLICIT, N_PRIV_LATENT, HISTORY_LEN, N_OBS,
    DEPTH_LATENT_DIM, DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH,
    SDK_TO_TRAIN_JOINTS, TRAIN_TO_SDK_JOINTS, SDK_TO_TRAIN_FEET,
    DEFAULT_STAND_ANGLES_TRAIN, ACTION_SCALE,
    ObsScales
)


class DepthOnlyFCBackbone58x87(nn.Module):
    """
    CNN backbone for 58x87 depth images.
    Matches training architecture exactly.
    """

    def __init__(self, num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()

        self.image_compression = nn.Sequential(
            # Input: [1, 58, 87]
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            # Output: [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # Output: [64, 25, 39]
            activation,
            nn.Flatten(),
            # Output: [64 * 25 * 39] = [62400]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, 32)
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Depth images [batch, H, W] or [batch, 1, H, W]

        Returns:
            Latent representation [batch, 32]
        """
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension
        compressed = self.image_compression(images)
        return self.output_activation(compressed)


class SimpleDepthEncoder(nn.Module):
    """
    Feedforward depth encoder (no GRU).
    Combines depth backbone output with proprioception.
    """

    def __init__(self, n_proprio: int = N_PROPRIO):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.base_backbone = DepthOnlyFCBackbone58x87(num_frames=1)

        # Combine depth latent with proprioception
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        # Output: 32 depth latent + 2 yaw prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(32, 32 + 2),
            last_activation
        )

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_image: Depth frame [batch, H, W]
            proprioception: Proprio state [batch, n_proprio]

        Returns:
            Depth latent + yaw prediction [batch, 34]
        """
        depth_latent = self.base_backbone(depth_image)
        combined = torch.cat([depth_latent, proprioception], dim=-1)
        latent = self.combination_mlp(combined)
        output = self.output_mlp(latent)
        return output


class PolicyRunner:
    """
    Runs the vision-based locomotion policy.

    Handles:
    - Loading JIT-traced base policy
    - Loading depth encoder weights
    - Joint reordering between SDK and training conventions
    - Building observations from robot state
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the policy runner.

        Args:
            device: Torch device ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"PolicyRunner using device: {self.device}")

        self.base_policy = None
        self.depth_encoder = None

        # History buffer for observations
        self.obs_history = None
        self.last_actions = np.zeros(12, dtype=np.float32)

    def load_models(self) -> bool:
        """
        Load the policy and depth encoder models.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load JIT-traced base policy
            print(f"Loading base policy from: {BASE_JIT_PATH}")
            self.base_policy = torch.jit.load(BASE_JIT_PATH, map_location=self.device)
            self.base_policy.eval()
            print("Base policy loaded successfully")

            # Load depth encoder weights
            print(f"Loading depth encoder from: {VISION_WEIGHT_PATH}")
            state_dict = torch.load(VISION_WEIGHT_PATH, map_location=self.device)

            self.depth_encoder = SimpleDepthEncoder(n_proprio=N_PROPRIO)
            self.depth_encoder.load_state_dict(state_dict['depth_encoder_state_dict'])
            self.depth_encoder.to(self.device)
            self.depth_encoder.eval()
            print("Depth encoder loaded successfully")

            # Initialize history buffer
            self.reset()

            return True

        except Exception as e:
            print(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reset(self):
        """Reset the policy state (history buffers, last actions)."""
        self.obs_history = None
        self.last_actions = np.zeros(12, dtype=np.float32)

    def reindex_joints_to_train(self, joints_sdk: np.ndarray) -> np.ndarray:
        """Reorder joints from SDK order to training order."""
        return joints_sdk[SDK_TO_TRAIN_JOINTS]

    def reindex_joints_to_sdk(self, joints_train: np.ndarray) -> np.ndarray:
        """Reorder joints from training order to SDK order."""
        return joints_train[TRAIN_TO_SDK_JOINTS]

    def reindex_feet_to_train(self, feet_sdk: np.ndarray) -> np.ndarray:
        """Reorder feet contacts from SDK order to training order."""
        return feet_sdk[SDK_TO_TRAIN_FEET]

    def build_proprio_obs(
        self,
        ang_vel: np.ndarray,       # [3] gyroscope in rad/s
        roll: float,               # roll angle in rad
        pitch: float,              # pitch angle in rad
        dof_pos: np.ndarray,       # [12] joint positions in SDK order
        dof_vel: np.ndarray,       # [12] joint velocities in SDK order
        foot_contacts: np.ndarray, # [4] contact states in SDK order [FR, FL, RR, RL]
        cmd_vel_x: float = 0.5,    # forward velocity command
    ) -> np.ndarray:
        """
        Build the proprioceptive observation vector.

        Returns:
            Proprio observation [N_PROPRIO] in training convention
        """
        # Reindex to training order
        dof_pos_train = self.reindex_joints_to_train(dof_pos)
        dof_vel_train = self.reindex_joints_to_train(dof_vel)
        last_actions_train = self.reindex_joints_to_train(self.last_actions)
        contacts_train = self.reindex_feet_to_train(foot_contacts)

        # Apply observation scales
        ang_vel_scaled = ang_vel * ObsScales.ang_vel
        dof_pos_normalized = (dof_pos_train - DEFAULT_STAND_ANGLES_TRAIN) * ObsScales.dof_pos
        dof_vel_scaled = dof_vel_train * ObsScales.dof_vel

        # Build observation vector (matching training order)
        proprio = np.concatenate([
            ang_vel_scaled,                      # [3] base angular velocity
            np.array([roll, pitch]),             # [2] orientation
            np.array([0.0]),                     # [1] delta_yaw (masked)
            np.array([0.0]),                     # [1] delta_yaw (actual - using 0)
            np.array([0.0]),                     # [1] delta_next_yaw
            np.array([0.0, 0.0]),                # [2] commands (masked)
            np.array([cmd_vel_x]),               # [1] forward velocity command
            np.array([1.0]),                     # [1] env_class != 17
            np.array([0.0]),                     # [1] env_class == 17
            dof_pos_normalized,                  # [12] joint positions
            dof_vel_scaled,                      # [12] joint velocities
            last_actions_train,                  # [12] last actions
            contacts_train - 0.5,                # [4] contact states (centered)
        ]).astype(np.float32)

        return proprio

    def build_full_obs(self, proprio: np.ndarray) -> np.ndarray:
        """
        Build the full observation vector including history.

        Args:
            proprio: Current proprioceptive observation [N_PROPRIO]

        Returns:
            Full observation [N_OBS]
        """
        # Initialize history if needed
        if self.obs_history is None:
            self.obs_history = np.tile(proprio, (HISTORY_LEN, 1))

        # Mask yaw in history (indices 6:8)
        proprio_for_history = proprio.copy()
        proprio_for_history[6:8] = 0.0

        # Update history buffer (FIFO)
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = proprio_for_history

        # Build full observation
        scan = np.zeros(N_SCAN, dtype=np.float32)  # Placeholder (replaced by depth latent)
        priv_explicit = np.zeros(N_PRIV_EXPLICIT, dtype=np.float32)  # Estimated by network
        priv_latent = np.zeros(N_PRIV_LATENT, dtype=np.float32)  # Placeholder

        obs = np.concatenate([
            proprio,                           # [53]
            scan,                              # [132]
            priv_explicit,                     # [9]
            priv_latent,                       # [29]
            self.obs_history.flatten(),        # [530]
        ]).astype(np.float32)

        return obs

    @torch.no_grad()
    def get_depth_latent(
        self,
        depth_image: np.ndarray,
        proprio: np.ndarray
    ) -> np.ndarray:
        """
        Get depth latent from the depth encoder.

        Args:
            depth_image: Preprocessed depth frame [H, W]
            proprio: Proprioceptive observation [N_PROPRIO]

        Returns:
            Depth latent [32]
        """
        # Convert to tensors
        depth_tensor = torch.from_numpy(depth_image).float().unsqueeze(0).to(self.device)
        proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)

        # Run depth encoder
        output = self.depth_encoder(depth_tensor, proprio_tensor)

        # Return only the depth latent (first 32 dims), not yaw prediction
        return output[0, :32].cpu().numpy()

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        depth_latent: np.ndarray
    ) -> np.ndarray:
        """
        Get action from the base policy.

        Args:
            obs: Full observation [N_OBS]
            depth_latent: Depth encoder output [32]

        Returns:
            Joint position targets in SDK order [12]
        """
        # Convert to tensors
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_latent).float().unsqueeze(0).to(self.device)

        # Run policy
        actions_train = self.base_policy(obs_tensor, depth_tensor)
        actions_train = actions_train[0].cpu().numpy()

        # Store for next step
        self.last_actions = actions_train.copy()

        # Convert to joint targets
        targets_train = actions_train * ACTION_SCALE + DEFAULT_STAND_ANGLES_TRAIN

        # Reindex to SDK order
        targets_sdk = self.reindex_joints_to_sdk(targets_train)

        return targets_sdk

    def run_inference(
        self,
        depth_image: np.ndarray,
        ang_vel: np.ndarray,
        roll: float,
        pitch: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        foot_contacts: np.ndarray,
        cmd_vel_x: float = 0.5,
    ) -> np.ndarray:
        """
        Run full inference pipeline.

        Args:
            depth_image: Preprocessed depth frame [H, W]
            ang_vel: Angular velocity from IMU [3]
            roll, pitch: Orientation angles
            dof_pos, dof_vel: Joint states in SDK order [12]
            foot_contacts: Contact states in SDK order [4]
            cmd_vel_x: Forward velocity command

        Returns:
            Joint position targets in SDK order [12]
        """
        # Build proprioceptive observation
        proprio = self.build_proprio_obs(
            ang_vel, roll, pitch, dof_pos, dof_vel, foot_contacts, cmd_vel_x
        )

        # Build full observation
        obs = self.build_full_obs(proprio)

        # Get depth latent
        depth_latent = self.get_depth_latent(depth_image, proprio)

        # Get action
        targets = self.get_action(obs, depth_latent)

        return targets


def test_policy():
    """Test the policy runner with dummy inputs."""
    print("Testing policy runner...")

    runner = PolicyRunner(device="cuda")

    if not runner.load_models():
        print("Failed to load models")
        return

    print("\nRunning inference with dummy inputs...")

    # Create dummy inputs
    depth_image = np.random.rand(DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH).astype(np.float32)
    ang_vel = np.zeros(3, dtype=np.float32)
    roll, pitch = 0.0, 0.0
    dof_pos = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5], dtype=np.float32)
    dof_vel = np.zeros(12, dtype=np.float32)
    foot_contacts = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Run inference
    for i in range(5):
        targets = runner.run_inference(
            depth_image, ang_vel, roll, pitch, dof_pos, dof_vel, foot_contacts
        )
        print(f"Step {i+1}: targets = {targets}")

    print("\nTest complete")


if __name__ == "__main__":
    test_policy()
