"""
Configuration for Go2 vision policy deployment.
Matches training parameters from cpsl_go2_rl_repo.
"""
import numpy as np

# =============================================================================
# Control Timing
# =============================================================================
CONTROL_DT = 0.02          # 50Hz policy (decimation=4 from 200Hz sim)
MOTOR_DT = 0.002           # 500Hz motor control loop
POLICY_DECIMATION = 10     # Run policy every 10 motor ticks (500Hz / 10 = 50Hz)

# =============================================================================
# PD Gains (matching training)
# =============================================================================
KP_WALK = 40.0             # Walking stiffness [N*m/rad] - matches training go2_student_config.py stiffness={joint:40}
KD_WALK = 1.0              # Walking damping [N*m*s/rad] - matches training damping={joint:1.0}
KP_STAND = 60.0            # Standing stiffness (matches Unitree example)
KD_STAND = 5.0             # Standing damping (matches Unitree example)

# =============================================================================
# Action Scaling and Clipping (matching training)
# =============================================================================
ACTION_SCALE = 0.25        # Policy output scale (target = default + scale * action)
CLIP_ACTIONS = 1.2         # Clip raw actions before scaling (matches training cfg.normalization.clip_actions)

# =============================================================================
# Observation Dimensions
# =============================================================================
N_PROPRIO = 53             # Proprioceptive observation size
N_SCAN = 132               # Height scan (placeholder, filled with zeros for vision)
N_PRIV_EXPLICIT = 9        # Privileged explicit info (estimated by network)
N_PRIV_LATENT = 29         # Privileged latent info (placeholder)
HISTORY_LEN = 10           # History length (10 frames * 53 proprio = 530)

# Total observation size
N_OBS = N_PROPRIO + N_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO  # 753

# Depth encoder output
DEPTH_LATENT_DIM = 32      # Depth encoder latent dimension (scan_encoder_dims[-1]=32)

# =============================================================================
# Observation Scales (matching training)
# =============================================================================
class ObsScales:
    lin_vel = 2.0          # Linear velocity scale
    ang_vel = 0.25         # Angular velocity scale
    dof_pos = 1.0          # Joint position scale
    dof_vel = 0.05         # Joint velocity scale

# =============================================================================
# Joint Ordering
# SDK order:      [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#                  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
# Training order: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#                  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
# =============================================================================

# SDK order  -> Training order: [FR,FL,RR,RL] -> [FL,FR,RL,RR]
# Note: this permutation is self-inverse (applying it twice returns the original order).
SDK_TO_TRAIN_JOINTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
TRAIN_TO_SDK_JOINTS = SDK_TO_TRAIN_JOINTS  # same permutation is its own inverse

# Feet ordering: SDK [FR, FL, RR, RL] -> Training [FL, FR, RL, RR]
SDK_TO_TRAIN_FEET = [1, 0, 3, 2]

# =============================================================================
# Default Standing Pose (SDK order)
# =============================================================================
DEFAULT_STAND_ANGLES_SDK = np.array([
    -0.1, 0.8, -1.5,    # FR: hip, thigh, calf
     0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 1.0, -1.5,    # RR: hip, thigh, calf
     0.1, 1.0, -1.5,    # RL: hip, thigh, calf
], dtype=np.float32)

# Default pose in training order (for policy offset)
DEFAULT_STAND_ANGLES_TRAIN = DEFAULT_STAND_ANGLES_SDK[SDK_TO_TRAIN_JOINTS]

# =============================================================================
# Joint Limits (SDK order) - for safety clipping
# =============================================================================
JOINT_POS_LIMITS = {
    'hip_min': -1.047,     # -60 degrees
    'hip_max': 1.047,      # +60 degrees
    'thigh_min': -1.5,     # ~-86 degrees
    'thigh_max': 3.4,      # ~195 degrees
    'calf_min': -2.7,      # ~-155 degrees
    'calf_max': -0.83,     # ~-48 degrees
}

# Per-joint limits in SDK order
JOINT_POS_MIN = np.array([
    JOINT_POS_LIMITS['hip_min'], JOINT_POS_LIMITS['thigh_min'], JOINT_POS_LIMITS['calf_min'],  # FR
    JOINT_POS_LIMITS['hip_min'], JOINT_POS_LIMITS['thigh_min'], JOINT_POS_LIMITS['calf_min'],  # FL
    JOINT_POS_LIMITS['hip_min'], JOINT_POS_LIMITS['thigh_min'], JOINT_POS_LIMITS['calf_min'],  # RR
    JOINT_POS_LIMITS['hip_min'], JOINT_POS_LIMITS['thigh_min'], JOINT_POS_LIMITS['calf_min'],  # RL
], dtype=np.float32)

JOINT_POS_MAX = np.array([
    JOINT_POS_LIMITS['hip_max'], JOINT_POS_LIMITS['thigh_max'], JOINT_POS_LIMITS['calf_max'],  # FR
    JOINT_POS_LIMITS['hip_max'], JOINT_POS_LIMITS['thigh_max'], JOINT_POS_LIMITS['calf_max'],  # FL
    JOINT_POS_LIMITS['hip_max'], JOINT_POS_LIMITS['thigh_max'], JOINT_POS_LIMITS['calf_max'],  # RR
    JOINT_POS_LIMITS['hip_max'], JOINT_POS_LIMITS['thigh_max'], JOINT_POS_LIMITS['calf_max'],  # RL
], dtype=np.float32)

# Torque limits per joint type
TORQUE_LIMITS = np.array([
    25.0, 40.0, 40.0,   # FR: hip, thigh, calf
    25.0, 40.0, 40.0,   # FL
    25.0, 40.0, 40.0,   # RR
    25.0, 40.0, 40.0,   # RL
], dtype=np.float32)

# =============================================================================
# Depth Camera Settings (D435i)
# =============================================================================
DEPTH_WIDTH = 640          # Native capture width
DEPTH_HEIGHT = 480         # Native capture height
DEPTH_FPS = 30             # Frame rate

# Cropping
# Training uses 160x120 intermediate with: crop_top=12, crop_bottom=0, crop_left=7, crop_right=9
# Training aspect ratio after crop: 144/108 = 1.3333
#
# Deployment: 640x480 - crop to center region to remove edge artifacts
# (edges have spurious "close" readings from camera/IR interference)
# Crop 80px from each side, then top to match training aspect ratio
CROP_LEFT = 28             # Training crop_left=7 * 4 scale factor
CROP_RIGHT = 36            # Training crop_right=9 * 4 scale factor
CROP_TOP = 48              # Training crop_top=12 * 4 scale factor
CROP_BOTTOM = 0            # Training crop_bottom=0
# Result: Matches training FOV exactly (scale factor = 640/160 = 4)

# Output resolution (matching training go2_student_config.py: resized = (87, 58))
DEPTH_OUTPUT_WIDTH = 87
DEPTH_OUTPUT_HEIGHT = 58

# Depth range (matching training go2_student_config.py: near_clip=0, far_clip=2)
DEPTH_NEAR = 0.0           # meters (near_clip in training)
DEPTH_FAR = 2.0            # meters (far_clip in training)

# =============================================================================
# Fixed Velocity Command
# =============================================================================
FIXED_VEL_X = 0.5          # Forward velocity [m/s]
FIXED_VEL_Y = 0.0          # Lateral velocity [m/s]
FIXED_VEL_YAW = 0.0        # Yaw rate [rad/s]

# =============================================================================
# Depth camera filters
# Training never applied RealSense spatial/temporal filters; the encoder was
# trained against raw+noise depth. Filters also add frame latency. Default OFF.
# =============================================================================
ENABLE_DEPTH_FILTERS = False

# =============================================================================
# Yaw source for proprioceptive observation [6:8]
# True  : use depth encoder's learned prediction (matches training pipeline)
# False : inject a constant direction (debug/straight-walk only)
# =============================================================================
USE_DEPTH_ENCODER_YAW = True
GOAL_X = 10.0              # (fallback mode) meters ahead
GOAL_Y = 0.0               # (fallback mode) meters to side
DRIFT_CORRECTION = 0.25    # (fallback mode) empirical yaw bias

# =============================================================================
# Foot contact detection — 20/15 N hysteresis
# Training uses sim contact-force magnitude > 2 N; deploy uses Go2 piezo scalar.
# Different sensors, so values are empirical; hysteresis mirrors training's
# contact_filt = contact OR last_contact and removes chatter during lift-off.
# =============================================================================
FOOT_CONTACT_ON_N = 20.0
FOOT_CONTACT_OFF_N = 15.0

# =============================================================================
# Depth encoder heartbeat / watchdog
# If the depth encoder process doesn't advance its heartbeat within this many
# policy ticks, the main thread trips to EMERGENCY. Policy runs at 50 Hz, depth
# at ~10 Hz, so we allow 20 ticks (≈400 ms) of stale data before reacting.
# =============================================================================
DEPTH_STALE_TICKS = 20

# =============================================================================
# Standing Sequence Timing (matches Unitree example: 500 ticks @ 500Hz = 1.0s)
# =============================================================================
STAND_UP_DURATION = 1.0    # seconds to interpolate to stand
SIT_DOWN_DURATION = 1.0    # seconds to interpolate to sit

# =============================================================================
# Model Paths (relative to this file's directory)
# =============================================================================
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "policy")
BASE_JIT_PATH = os.path.join(MODEL_DIR, "test_student_v2-10000-base_jit.pt")
VISION_WEIGHT_PATH = os.path.join(MODEL_DIR, "test_student_v2-10000-vision_weight.pt")
