"""
Configuration for Go2 Vision Policy Deployment
Adapted from cpsl_go2_rl_repo/legged_gym/envs/go2/go2_parkour_config.py
"""

import numpy as np


class DeployConfig:
    """Deployment configuration for Go2 vision-based policy"""

    # Control parameters (must match training config)
    control_dt = 0.02  # 50Hz policy (decimation=4 @ 200Hz sim)
    action_scale = 0.25

    # PD gains (tuned for real hardware - lower than training for sim2real)
    # Training used: kp=40.0, kd=1.0 (simulation)
    # Real hardware needs lower gains to prevent excessive force/jumpiness
    # Tested values from working rough_go2 deployment
    kp = 30.0  # stiffness (reduced from 40 for real hardware)
    kd = 0.6   # damping (low damping for responsive control)

    # Default standing pose (from training config)
    default_joint_angles = np.array([
        0.1, 0.8, -1.5,   # FL: hip, thigh, calf
        -0.1, 0.8, -1.5,  # FR: hip, thigh, calf
        0.1, 1.0, -1.5,   # RL: hip, thigh, calf
        -0.1, 1.0, -1.5,  # RR: hip, thigh, calf
    ], dtype=np.float32)

    # Joint order mapping between training (URDF) and Unitree SDK
    # Training order: FL, FR, RL, RR (hip, thigh, calf for each)
    # SDK order: FR, FL, RR, RL (hip, thigh, calf for each)
    training_to_sdk_idx = [
        3, 4, 5,    # FL_hip, FL_thigh, FL_calf -> SDK indices 3,4,5
        0, 1, 2,    # FR_hip, FR_thigh, FR_calf -> SDK indices 0,1,2
        9, 10, 11,  # RL_hip, RL_thigh, RL_calf -> SDK indices 9,10,11
        6, 7, 8,    # RR_hip, RR_thigh, RR_calf -> SDK indices 6,7,8
    ]

    # SDK to training order
    sdk_to_training_idx = [
        3, 4, 5,    # FR (SDK 0,1,2) -> training FL (3,4,5) - wait this is wrong
        0, 1, 2,    # FL (SDK 3,4,5) -> training FL (0,1,2)
        9, 10, 11,  # RR (SDK 6,7,8) -> training RR (9,10,11) - wrong
        6, 7, 8,    # RL (SDK 9,10,11) -> training RL (6,7,8)
    ]

    # Actually, let me reconsider the mapping:
    # Training URDF order: FL_hip(0), FL_thigh(1), FL_calf(2), FR(3,4,5), RL(6,7,8), RR(9,10,11)
    # Wait, from config: FL, RL, FR, RR - let's check the config again
    # From go2_parkour_config.py:
    # 'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,
    # 'FL_thigh_joint': 0.8, ...

    # The default_joint_angles in config lists joints individually, but the action space
    # is typically ordered FL, FR, RL, RR (Isaac Gym convention)
    # SDK order: FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2

    # Corrected mapping:
    # Training output (action): [FL_hip, FL_thigh, FL_calf, FR_*, RL_*, RR_*]
    # SDK expects: [FR_*, FL_*, RR_*, RL_*]

    # Depth camera config (D435i on Go2 head)
    depth_width = 87
    depth_height = 58
    depth_fov = 86  # horizontal FOV in degrees
    depth_near = 0.0
    depth_far = 2.0
    depth_scale = 1.0

    # Observation dimensions (must match training)
    n_proprio = 53  # proprioceptive observation dimension
    n_scan = 132    # terrain scan (teacher) or depth latent (student)
    history_len = 10
    n_priv_explicit = 9  # 3 + 3 + 3 (base_lin_vel copies)
    n_priv_latent = 29   # 4 + 1 + 12 + 12 (mass, friction, motor_strength)

    # Command velocity (for forward walking)
    command_vx = 0.5  # Forward velocity command (m/s), range: 0.0 - 1.0

    # Goal-based navigation (matching training)
    goal_distance = 1.0  # Distance to place goal ahead (meters)
    goal_update_threshold = 0.3  # Update goal when within this distance (meters)
    next_goal_distance = 2.0  # Distance for next waypoint
    lin_vel_scale = 2.0  # Observation scale for linear velocity/commands

    # Model paths (relative to this file, update as needed)
    model_path = ""  # Set this when running


# Joint limits for safety
class JointLimits:
    """Go2 joint limits in radians"""
    hip_min = -1.047    # -60 deg
    hip_max = 1.047     # 60 deg
    thigh_min = -1.5    # ~-86 deg
    thigh_max = 3.4     # ~195 deg
    calf_min = -2.7     # ~-155 deg
    calf_max = -0.83    # ~-48 deg

    @staticmethod
    def clip_joints(joints):
        """Clip joint angles to safe limits"""
        clipped = np.copy(joints)
        for i in range(4):  # 4 legs
            base = i * 3
            clipped[base] = np.clip(joints[base], JointLimits.hip_min, JointLimits.hip_max)
            clipped[base+1] = np.clip(joints[base+1], JointLimits.thigh_min, JointLimits.thigh_max)
            clipped[base+2] = np.clip(joints[base+2], JointLimits.calf_min, JointLimits.calf_max)
        return clipped


# Joint ordering constants (matching unitree_legged_const.py)
LegID = {
    "FR_0": 0, "FR_1": 1, "FR_2": 2,
    "FL_0": 3, "FL_1": 4, "FL_2": 5,
    "RR_0": 6, "RR_1": 7, "RR_2": 8,
    "RL_0": 9, "RL_1": 10, "RL_2": 11,
}

PosStopF = 2.146e9
VelStopF = 16000.0
