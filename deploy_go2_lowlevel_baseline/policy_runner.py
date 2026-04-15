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
    ObsScales,
)

# Shared embedding layout: [0:32] depth latent, [32:34] yaw prediction
SHARED_EMBEDDING_LEN = DEPTH_LATENT_DIM + 2


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

        # Zero-copy numpy views over the shared ctypes arrays. One memcpy per
        # transfer instead of 32+53 python attribute accesses per policy tick.
        self._emb_view = np.frombuffer(
            shared_embedding.get_obj(), dtype=np.float32
        )  # shape (34,)
        self._prop_view = np.frombuffer(
            shared_proprio.get_obj(), dtype=np.float32
        )  # shape (N_PROPRIO,)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"PolicyRunner using device: {self.device}")

        # Enable cudnn optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        self.base_policy = None

        # History buffer for observations (circular, avoids per-tick np.roll alloc)
        self.obs_history = np.zeros((HISTORY_LEN, N_PROPRIO), dtype=np.float32)
        self._hist_idx = 0
        self._hist_initialized = False
        self.last_actions = np.zeros(12, dtype=np.float32)

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

            # Preallocate inference tensors (reused every tick; no per-tick allocator churn).
            pin = self.device.type == "cuda"
            self._obs_cpu = torch.zeros(1, N_OBS, dtype=torch.float32, pin_memory=pin)
            self._depth_cpu = torch.zeros(1, DEPTH_LATENT_DIM, dtype=torch.float32, pin_memory=pin)
            self._obs_gpu = torch.zeros(1, N_OBS, dtype=torch.float32, device=self.device)
            self._depth_gpu = torch.zeros(1, DEPTH_LATENT_DIM, dtype=torch.float32, device=self.device)
            # numpy views so we can write into the pinned buffer without extra copy
            self._obs_cpu_np = self._obs_cpu.numpy()[0]
            self._depth_cpu_np = self._depth_cpu.numpy()[0]

            # GPU warmup for base policy
            print("Running GPU warmup for base policy...")
            print(f"  Device: {self.device}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

            import time
            for i in range(5):
                t0 = time.time()
                with torch.inference_mode():
                    _ = self.base_policy(self._obs_gpu, self._depth_gpu)
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
        self.obs_history.fill(0.0)
        self._hist_idx = 0
        self._hist_initialized = False
        self.last_actions[:] = 0.0

    def get_depth_latent_from_shared(self) -> np.ndarray:
        """Read depth latent from shared memory (written by depth encoder process)."""
        return self._emb_view[:DEPTH_LATENT_DIM].copy()

    def get_yaw_pred_from_shared(self) -> np.ndarray:
        """Read yaw prediction from shared memory (written by depth encoder after depth latent)."""
        return self._emb_view[DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + 2].copy()

    def write_proprio_to_shared(self, proprio: np.ndarray):
        """Write proprio to shared memory for depth encoder process."""
        self._prop_view[:] = proprio

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
            yaw_pred: Yaw prediction from depth encoder [delta_yaw, delta_next_yaw] in radians.
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
        # Masked proprio for history (training zeros delta_yaw/delta_next_yaw in history)
        proprio_masked = proprio.copy()
        proprio_masked[6:8] = 0.0

        # On first tick, warm-fill the circular buffer with the current masked proprio
        # so the history section is self-consistent rather than zeros.
        if not self._hist_initialized:
            self.obs_history[:] = proprio_masked
            self._hist_initialized = True

        # Write directly into the preallocated obs buffer. Layout matches
        # training: [proprio, scan, priv_explicit, priv_latent, history_flat].
        # scan / priv_explicit / priv_latent stay zero — the JIT overwrites
        # priv_explicit via its internal estimator, scan is replaced by
        # depth_latent in the actor, and priv_latent is unused at inference.
        obs_flat = self._obs_cpu_np
        obs_flat[:N_PROPRIO] = proprio
        # [N_PROPRIO : N_PROPRIO + N_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT] stays 0
        hist_start = N_PROPRIO + N_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT
        # Unroll the circular buffer into the history slot in correct (oldest→newest) order
        head = self._hist_idx
        first_rows = HISTORY_LEN - head
        hist_slice = obs_flat[hist_start:].reshape(HISTORY_LEN, N_PROPRIO)
        hist_slice[:first_rows] = self.obs_history[head:]
        if head > 0:
            hist_slice[first_rows:] = self.obs_history[:head]

        # Append current masked proprio as the newest entry (overwrites oldest slot).
        self.obs_history[self._hist_idx] = proprio_masked
        self._hist_idx = (self._hist_idx + 1) % HISTORY_LEN

        return obs_flat

    @torch.inference_mode()
    def get_action(
        self,
        depth_latent: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from the base policy.

        Reads the observation from the preallocated CPU buffer populated by
        build_full_obs(). Copies it to GPU, runs inference, returns results.

        Args:
            depth_latent: Depth encoder output [32]

        Returns:
            Tuple of (joint position targets in SDK order [12], raw actions for logging [12])
        """
        # Copy depth latent into pinned buffer and then to GPU
        self._depth_cpu_np[:] = depth_latent
        if self.device.type == "cuda":
            self._obs_gpu.copy_(self._obs_cpu, non_blocking=True)
            self._depth_gpu.copy_(self._depth_cpu, non_blocking=True)
        else:
            self._obs_gpu.copy_(self._obs_cpu)
            self._depth_gpu.copy_(self._depth_cpu)

        actions_raw_t = self.base_policy(self._obs_gpu, self._depth_gpu)
        actions_raw = actions_raw_t[0].detach().cpu().numpy().astype(np.float32).copy()

        # Store RAW actions for next step's proprio[37:49]
        self.last_actions[:] = actions_raw

        # Clip raw actions then scale to joint targets (matches training pipeline)
        actions_clipped = np.clip(actions_raw, -CLIP_ACTIONS, CLIP_ACTIONS)

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
            print(f"  [YawPred] delta_yaw={yaw_pred[0]:+.4f} rad, delta_next_yaw={yaw_pred[1]:+.4f} rad")

        # Build proprioceptive observation with yaw prediction
        proprio = self.build_proprio_obs(
            ang_vel, roll, pitch, dof_pos, dof_vel, foot_contacts, cmd_vel_x,
            yaw_pred=yaw_pred
        )

        # Write proprio to shared memory for depth encoder process
        self.write_proprio_to_shared(proprio)

        # Populate the preallocated obs buffer with current proprio + history
        self.build_full_obs(proprio)

        # Read depth latent from shared memory (from depth encoder process)
        depth_latent = self.get_depth_latent_from_shared()

        # Get action
        targets, raw_actions = self.get_action(depth_latent)

        return targets, raw_actions
