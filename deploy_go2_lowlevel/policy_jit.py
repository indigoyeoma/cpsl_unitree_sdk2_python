"""
JIT Policy Loader for Go2 Vision Deployment

Loads JIT traced models (vision_weight.pt + base_jit.pt) for deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add rsl_rl to path for depth encoder architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../cpsl_go2_rl_repo/rsl_rl'))

from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from rsl_rl.modules.actor_critic import get_activation


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
        print(f"Loading depth encoder from: {vision_weight_path}")
        self.depth_encoder = self._load_depth_encoder(vision_weight_path)
        self.depth_encoder.to(self.device)
        self.depth_encoder.eval()

        # Load JIT traced policy
        print(f"Loading JIT policy from: {base_jit_path}")
        self.policy_jit = torch.jit.load(base_jit_path, map_location=self.device)
        self.policy_jit.eval()

        print("Policy loaded successfully!")

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
        activation = get_activation('elu')

        # CNN backbone
        base_backbone = DepthOnlyFCBackbone58x87(
            prop_dim=self.n_proprio,
            scandots_output_dim=32,
            hidden_state_dim=512,
            output_activation=None,
            num_frames=1
        )

        # Full depth encoder with MLP
        class DummyConfig:
            class env:
                n_proprio = 53

        env_cfg = DummyConfig()
        depth_encoder = RecurrentDepthBackbone(base_backbone, env_cfg)

        # Load weights
        depth_encoder.load_state_dict(depth_encoder_state, strict=True)

        return depth_encoder

    @torch.no_grad()
    def get_action(self, depth_image: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Run policy inference.

        Args:
            depth_image: Depth image (H x W), normalized 0-1
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
