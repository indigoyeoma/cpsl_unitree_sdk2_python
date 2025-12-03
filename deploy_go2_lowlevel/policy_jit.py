"""
JIT Policy Loader for Go2 Vision Deployment (STANDALONE - No Training Dependencies)

Loads JIT traced models (vision_weight.pt + base_jit.pt) for deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import os


# ============================================================================
# Network Architectures (Standalone - no rsl_rl dependency)
# ============================================================================

class DepthOnlyFCBackbone58x87(nn.Module):
    """
    CNN for processing 58x87 depth images.
    Standalone version - no external dependencies.
    """
    def __init__(self, scandots_output_dim=32, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()

        self.image_compression = nn.Sequential(
            # Input: [batch, 1, 58, 87]
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            # [batch, 32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [batch, 32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [batch, 64, 25, 39]
            activation,
            nn.Flatten(),
            # [batch, 64 * 25 * 39] = [batch, 62400]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [batch, H, W] depth images
        Returns:
            [batch, scandots_output_dim] features
        """
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dim
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)
        return latent


class RecurrentDepthBackbone(nn.Module):
    """
    Combines depth CNN output with proprioception.
    Standalone version - no recurrence (buffer_len=1).
    """
    def __init__(self, base_backbone, n_proprio=53):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.base_backbone = base_backbone

        # Combination MLP: depth_latent (32) + proprio (53) -> 32
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        # Output MLP: 32 -> 34 (32 latent + 2 yaw)
        self.output_mlp = nn.Sequential(
            nn.Linear(32, 64),
            activation,
            nn.Linear(64, 34),  # 32 latent + 2 yaw
            last_activation
        )

    def forward(self, depth_image, proprioception):
        """
        Args:
            depth_image: [batch, H, W] depth image
            proprioception: [batch, n_proprio] robot state
        Returns:
            [batch, 34] = [32 latent + 2 yaw]
        """
        # CNN: depth -> 32-dim
        depth_latent = self.base_backbone(depth_image)

        # Combine with proprio
        combined = torch.cat([depth_latent, proprioception], dim=-1)
        depth_latent = self.combination_mlp(combined)

        # Output: latent + yaw prediction
        output = self.output_mlp(depth_latent)
        return output


# ============================================================================
# JIT Policy Runner
# ============================================================================

class JITPolicyRunner:
    """Runs inference using JIT traced vision policy."""

    def __init__(self, vision_weight_path: str, base_jit_path: str, device: str = "cuda"):
        """
        Load JIT traced models for deployment.

        Args:
            vision_weight_path: Path to vision_weight.pt (depth encoder weights)
            base_jit_path: Path to base_jit.pt (traced policy)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dimensions from training
        self.n_proprio = 53
        self.depth_output_dim = 32

        # Load depth encoder
        print(f"Loading depth encoder from: {os.path.basename(vision_weight_path)}")
        self.depth_encoder = self._load_depth_encoder(vision_weight_path)
        self.depth_encoder.to(self.device)
        self.depth_encoder.eval()

        # Load JIT traced policy
        print(f"Loading JIT policy from: {os.path.basename(base_jit_path)}")
        self.policy_jit = torch.jit.load(base_jit_path, map_location=self.device)
        self.policy_jit.eval()

        print("✓ Policy loaded successfully")

    def _load_depth_encoder(self, vision_weight_path):
        """
        Load depth encoder from vision_weight.pt.

        The depth encoder consists of:
        - DepthOnlyFCBackbone58x87: CNN (58x87 depth -> 32-dim)
        - RecurrentDepthBackbone: MLP (32 + 53 proprio -> 34 output)
        """
        # Load state dict
        checkpoint = torch.load(vision_weight_path, map_location=self.device)
        depth_encoder_state = checkpoint['depth_encoder_state_dict']

        # Build depth encoder architecture
        # CNN backbone
        base_backbone = DepthOnlyFCBackbone58x87(
            scandots_output_dim=32,
            num_frames=1
        )

        # Full depth encoder with MLP
        depth_encoder = RecurrentDepthBackbone(
            base_backbone=base_backbone,
            n_proprio=self.n_proprio
        )

        # Load weights
        depth_encoder.load_state_dict(depth_encoder_state, strict=True)

        return depth_encoder

    @torch.no_grad()
    def get_action(self, depth_image: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Run policy inference.

        Args:
            depth_image: Depth image (H x W), normalized [-0.5, 0.5]
            obs: Full observation vector (proprio + scan + priv + history)

        Returns:
            Action (12 joint position targets as deltas from default)
        """
        # Convert to tensors
        depth_tensor = torch.from_numpy(depth_image).float().to(self.device).unsqueeze(0)
        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)

        # Get proprioception for depth encoder (first n_proprio elements)
        proprio = obs_tensor[:, :self.n_proprio].clone()
        proprio[:, 6:8] = 0  # Mask yaw as in training

        # Run depth encoder to get depth latent + yaw
        depth_latent_and_yaw = self.depth_encoder(depth_tensor, proprio)
        depth_latent = depth_latent_and_yaw[:, :-2]
        yaw_pred = depth_latent_and_yaw[:, -2:]

        # Update observation with predicted yaw
        obs_tensor[:, 6:8] = 1.5 * yaw_pred

        # Run JIT policy (estimator + actor)
        # The JIT policy expects (obs, depth_latent)
        actions = self.policy_jit(obs_tensor, depth_latent)

        return actions.cpu().numpy().flatten()


def load_jit_policy(traced_dir: str, exptid: str, checkpoint: str, device: str = "cuda"):
    """
    Convenience function to load JIT policy from traced directory.

    Args:
        traced_dir: Path to traced directory (e.g., logs/parkour_go2/student_depth/traced)
        exptid: Experiment ID (e.g., "student_depth")
        checkpoint: Checkpoint number (e.g., "5000")
        device: Device to run on

    Returns:
        JITPolicyRunner instance
    """
    vision_weight_path = os.path.join(traced_dir, f"{exptid}-{checkpoint}-vision_weight.pt")
    base_jit_path = os.path.join(traced_dir, f"{exptid}-{checkpoint}-base_jit.pt")

    if not os.path.exists(vision_weight_path):
        raise FileNotFoundError(f"Vision weights not found: {vision_weight_path}")
    if not os.path.exists(base_jit_path):
        raise FileNotFoundError(f"JIT policy not found: {base_jit_path}")

    return JITPolicyRunner(vision_weight_path, base_jit_path, device)
