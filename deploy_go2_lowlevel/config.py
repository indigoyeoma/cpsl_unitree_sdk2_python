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
KP_WALK = 25.0             # Walking stiffness [N*m/rad]
KD_WALK = 0.6              # Walking damping [N*m*s/rad]
KP_STAND = 60.0            # Standing stiffness (matches Unitree example)
KD_STAND = 5.0             # Standing damping (matches Unitree example)

# =============================================================================
# Action Scaling and Clipping (matching training)
# =============================================================================
ACTION_SCALE = 0.25        # Policy output scale
CLIP_ACTIONS = 1.2         # Raw action clipping value from training
# Effective clip range: CLIP_ACTIONS / ACTION_SCALE = 1.2 / 0.25 = 4.8

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
DEPTH_LATENT_DIM = 32      # Depth encoder latent dimension

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

# From SDK order to Training order (for building observations)
SDK_TO_TRAIN_JOINTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# From Training order to SDK order (for applying actions)
TRAIN_TO_SDK_JOINTS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # Same mapping

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

# Cropping (matching training go2_student_config.py)
# Training: 106x60 with crop_top=6, crop_bottom=0, crop_left=5, crop_right=6
# Deployment: 640x480 scaled proportionally
CROP_TOP = 48              # 6/60 * 480 = 48
CROP_BOTTOM = 0            # 0 (matches training)
CROP_LEFT = 30             # 5/106 * 640 ≈ 30
CROP_RIGHT = 36            # 6/106 * 640 ≈ 36

# Output resolution (matching training)
DEPTH_OUTPUT_WIDTH = 87
DEPTH_OUTPUT_HEIGHT = 58

# Depth range (matching training go2_student_config.py)
DEPTH_NEAR = 0.3           # meters (near_clip in training)
DEPTH_FAR = 3.0            # meters (far_clip in training)

# =============================================================================
# Fixed Velocity Command
# =============================================================================
FIXED_VEL_X = 0.5          # Forward velocity [m/s]
FIXED_VEL_Y = 0.0          # Lateral velocity [m/s]
FIXED_VEL_YAW = 0.0        # Yaw rate [rad/s]

# =============================================================================
# Standing Sequence Timing
# =============================================================================
STAND_UP_DURATION = 1.5    # seconds to interpolate to stand
SIT_DOWN_DURATION = 1.0    # seconds to interpolate to sit

# =============================================================================
# Model Paths (relative to this file's directory)
# =============================================================================
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "policy")
BASE_JIT_PATH = os.path.join(MODEL_DIR, "go2_student-15000-base_jit.pt")
VISION_WEIGHT_PATH = os.path.join(MODEL_DIR, "go2_student-15000-vision_weight.pt")
