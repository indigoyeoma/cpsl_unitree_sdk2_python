"""
Policy runner for Go2 vision-based locomotion.

Two-process architecture:
- This module runs the BASE POLICY only (fast, ~10ms)
- Depth encoder runs in separate process (depth_encoder_process.py)
- Communication via shared memory buffers
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from multiprocessing import Array, Value
from ctypes import c_float, c_bool

from config import (
    BASE_JIT_PATH,
    N_PROPRIO, N_SCAN, N_PRIV_EXPLICIT, N_PRIV_LATENT, HISTORY_LEN, N_OBS,
    DEPTH_LATENT_DIM,
    DEFAULT_STAND_ANGLES_SDK, ACTION_SCALE, CLIP_ACTIONS,
    ObsScales
)


class PolicyRunner:
    """
    Runs the vision-based locomotion policy.

    Two-process architecture:
    - Depth encoder runs in separate process, writes to shared_embedding
    - This class reads embedding from shared memory, runs base policy
    - Writes proprio to shared memory for depth encoder
    """

    def __init__(
        self,
        shared_embedding: Array,
        shared_proprio: Array,
        device: str = "cuda"
    ):
        """
        Initialize the policy runner.

        Args:
            shared_embedding: Shared array for 32-dim depth embedding (read from depth encoder)
            shared_proprio: Shared array for N_PROPRIO proprio (write for depth encoder)
            device: Torch device ("cuda" or "cpu")
        """
        self.shared_embedding = shared_embedding
        self.shared_proprio = shared_proprio

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"PolicyRunner using device: {self.device}")

        # Enable cudnn optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        self.base_policy = None

        # History buffer for observations
        self.obs_history = None
        self.last_actions = np.zeros(12, dtype=np.float32)

        # Cache for depth latent
        self._cached_depth_latent = np.zeros(DEPTH_LATENT_DIM, dtype=np.float32)

        # Cache for yaw prediction from depth encoder (sin, cos)
        self._cached_yaw_pred = np.zeros(2, dtype=np.float32)

    def load_models(self) -> bool:
        """
        Load the base policy model.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load JIT-traced base policy
            print(f"Loading base policy from: {BASE_JIT_PATH}")
            self.base_policy = torch.jit.load(BASE_JIT_PATH, map_location=self.device)
            self.base_policy.eval()
            print("Base policy loaded successfully")

            # Initialize history buffer
            self.reset()

            # GPU warmup for base policy
            print("Running GPU warmup for base policy...")
            print(f"  Device: {self.device}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

            dummy_obs = torch.zeros(1, N_OBS, dtype=torch.float32, device=self.device)
            dummy_latent = torch.zeros(1, DEPTH_LATENT_DIM, dtype=torch.float32, device=self.device)

            import time
            for i in range(5):
                t0 = time.time()
                with torch.no_grad():
                    _ = self.base_policy(dummy_obs, dummy_latent)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                print(f"  Warmup {i+1}/5: {(t1-t0)*1000:.1f}ms")

            self.reset()
            print("GPU warmup complete")

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

    def get_depth_latent_from_shared(self) -> np.ndarray:
        """Read depth latent from shared memory (written by depth encoder process)."""
        for i in range(DEPTH_LATENT_DIM):
            self._cached_depth_latent[i] = self.shared_embedding[i]
        return self._cached_depth_latent.copy()

    def get_yaw_pred_from_shared(self) -> np.ndarray:
        """Read yaw prediction from shared memory (indices 32-33, written by depth encoder)."""
        self._cached_yaw_pred[0] = self.shared_embedding[32]  # delta_yaw_sin
        self._cached_yaw_pred[1] = self.shared_embedding[33]  # delta_yaw_cos
        return self._cached_yaw_pred.copy()

    def write_proprio_to_shared(self, proprio: np.ndarray):
        """Write proprio to shared memory for depth encoder process."""
        for i in range(min(len(proprio), N_PROPRIO)):
            self.shared_proprio[i] = proprio[i]

    def build_proprio_obs(
        self,
        ang_vel: np.ndarray,       # [3] gyroscope in rad/s
        roll: float,               # roll angle in rad
        pitch: float,              # pitch angle in rad
        dof_pos: np.ndarray,       # [12] joint positions in SDK order
        dof_vel: np.ndarray,       # [12] joint velocities in SDK order
        foot_contacts: np.ndarray, # [4] contact states in SDK order [FR, FL, RR, RL]
        cmd_vel_x: float = 0.5,    # forward velocity command
        yaw_pred: Optional[np.ndarray] = None,  # [2] yaw prediction from depth encoder
    ) -> np.ndarray:
        """
        Build the proprioceptive observation vector.

        Args:
            yaw_pred: Yaw prediction from depth encoder [delta_yaw_sin, delta_yaw_cos]
                      If None, uses zeros.

        Returns:
            Proprio observation [N_PROPRIO] in SDK order
        """
        # Apply observation scales
        ang_vel_scaled = ang_vel * ObsScales.ang_vel
        dof_pos_normalized = (dof_pos - DEFAULT_STAND_ANGLES_SDK) * ObsScales.dof_pos
        dof_vel_scaled = dof_vel * ObsScales.dof_vel
        last_actions_sdk = self.last_actions
        contacts_sdk = foot_contacts

        # Use yaw prediction from depth encoder if available
        # These are delta_yaw and delta_next_yaw in RADIANS (not sin/cos!)
        delta_yaw = yaw_pred[0] if yaw_pred is not None else 0.0
        delta_next_yaw = yaw_pred[1] if yaw_pred is not None else 0.0

        # Build observation vector
        proprio = np.concatenate([
            ang_vel_scaled,                      # [3] base angular velocity
            np.array([roll, pitch]),             # [2] orientation
            np.array([0.0]),                     # [1] delta_yaw (masked with 0*)
            np.array([delta_yaw]),               # [1] delta_yaw (radians, from depth encoder)
            np.array([delta_next_yaw]),          # [1] delta_next_yaw (radians, from depth encoder)
            np.array([0.0, 0.0]),                # [2] commands (masked with 0*)
            np.array([cmd_vel_x]),               # [1] forward velocity command
            np.array([1.0]),                     # [1] env_class != 17
            np.array([0.0]),                     # [1] env_class == 17
            dof_pos_normalized,                  # [12] joint positions (SDK order)
            dof_vel_scaled,                      # [12] joint velocities (SDK order)
            last_actions_sdk,                    # [12] last actions (SDK order)
            contacts_sdk - 0.5,                  # [4] contact states (SDK order)
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
            proprio_masked = proprio.copy()
            proprio_masked[6:8] = 0.0
            self.obs_history = np.tile(proprio_masked, (HISTORY_LEN, 1))

        # Build full observation using history from previous steps
        scan = np.zeros(N_SCAN, dtype=np.float32)
        priv_explicit = np.zeros(N_PRIV_EXPLICIT, dtype=np.float32)
        priv_latent = np.zeros(N_PRIV_LATENT, dtype=np.float32)

        obs = np.concatenate([
            proprio,                           # [53] current proprio
            scan,                              # [132]
            priv_explicit,                     # [9]
            priv_latent,                       # [29]
            self.obs_history.flatten(),        # [530] history from PREVIOUS steps
        ]).astype(np.float32)

        # Update history buffer (FIFO) for next step
        proprio_for_history = proprio.copy()
        proprio_for_history[6:8] = 0.0

        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = proprio_for_history

        return obs

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        depth_latent: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from the base policy.

        Args:
            obs: Full observation [N_OBS]
            depth_latent: Depth encoder output [32]

        Returns:
            Tuple of (joint position targets in SDK order [12], raw actions for logging [12])
        """
        # Convert to tensors
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_latent).float().unsqueeze(0).to(self.device)

        # Run policy
        actions_raw = self.base_policy(obs_tensor, depth_tensor)
        actions_raw = actions_raw[0].cpu().numpy()

        # Store RAW actions for next step
        self.last_actions = actions_raw.copy()

        # Clip actions before scaling
        effective_clip = CLIP_ACTIONS / ACTION_SCALE  # = 4.8
        actions_clipped = np.clip(actions_raw, -effective_clip, effective_clip)

        # Convert to joint targets
        targets_sdk = actions_clipped * ACTION_SCALE + DEFAULT_STAND_ANGLES_SDK

        return targets_sdk, actions_raw

    def run_inference(
        self,
        ang_vel: np.ndarray,
        roll: float,
        pitch: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        foot_contacts: np.ndarray,
        cmd_vel_x: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full inference pipeline.

        Reads depth embedding and yaw prediction from shared memory (from depth encoder process).
        Writes proprio to shared memory for depth encoder.

        Args:
            ang_vel: Angular velocity from IMU [3]
            roll, pitch: Orientation angles
            dof_pos, dof_vel: Joint states in SDK order [12]
            foot_contacts: Contact states in SDK order [4]
            cmd_vel_x: Forward velocity command

        Returns:
            Tuple of (joint position targets in SDK order [12], raw actions for logging [12])
        """
        # Read yaw prediction from shared memory (from depth encoder process)
        yaw_pred = self.get_yaw_pred_from_shared()

        # Debug: print yaw prediction periodically
        self._inference_count = getattr(self, '_inference_count', 0) + 1
        if self._inference_count % 50 == 1:
            print(f"  [YawPred] sin={yaw_pred[0]:.4f}, cos={yaw_pred[1]:.4f}")

        # Build proprioceptive observation with yaw prediction
        proprio = self.build_proprio_obs(
            ang_vel, roll, pitch, dof_pos, dof_vel, foot_contacts, cmd_vel_x,
            yaw_pred=yaw_pred
        )

        # Write proprio to shared memory for depth encoder process
        self.write_proprio_to_shared(proprio)

        # Build full observation
        obs = self.build_full_obs(proprio)

        # Read depth latent from shared memory (from depth encoder process)
        depth_latent = self.get_depth_latent_from_shared()

        # Get action
        targets, raw_actions = self.get_action(obs, depth_latent)

        return targets, raw_actions
