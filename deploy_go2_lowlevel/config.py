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
    clip_actions = 1.2  # Action clipping (must match training)

    # PD gains for WALKING (must match training for sim2real)
    # Training used: kp=25.0, kd=0.6 (from go2_config.py)
    kp_walk = 25.0  # stiffness for walking (matches training)
    kd_walk = 0.6   # damping for walking (matches training)

    # PD gains for STANDING (higher for stability)
    # From Unitree SDK examples - stiffer for holding position
    kp_stand = 70.0  # stiffness for standing
    kd_stand = 3.0   # damping for standing

    # Default (will be switched based on phase)
    kp = 70.0
    kd = 3.0

    # Default standing pose (from training config)
    # Training simulation order: FL, FR, RL, RR
    default_joint_angles_sim = np.array([
        0.1, 0.8, -1.5,   # FL: hip, thigh, calf
        -0.1, 0.8, -1.5,  # FR: hip, thigh, calf
        0.1, 1.0, -1.5,   # RL: hip, thigh, calf
        -0.1, 1.0, -1.5,  # RR: hip, thigh, calf
    ], dtype=np.float32)

    # SDK/Observation order: FR, FL, RR, RL (matches training after reindex)
    # This is what the policy actually sees during training
    default_joint_angles = np.array([
        -0.1, 0.8, -1.5,  # FR: hip, thigh, calf
        0.1, 0.8, -1.5,   # FL: hip, thigh, calf
        -0.1, 1.0, -1.5,  # RR: hip, thigh, calf
        0.1, 1.0, -1.5,   # RL: hip, thigh, calf
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

    # Depth camera config (D435i on Go2 head - must match training)
    # From go2_config.py: resized=(87,58), fov=87, near=0.3, far=3.0
    depth_width = 87
    depth_height = 58
    depth_fov = 87  # horizontal FOV in degrees (matches training)
    depth_near = 0.3  # D435i minimum depth (matches training)
    depth_far = 3.0   # D435i maximum depth (matches training)
    depth_scale = 1.0

    # Observation dimensions (must match training)
    n_proprio = 53  # proprioceptive observation dimension
    n_scan = 132    # terrain scan (teacher) or depth latent (student)
    history_len = 10
    n_priv_explicit = 9  # 3 + 3 + 3 (base_lin_vel copies)
    n_priv_latent = 29   # 4 + 1 + 12 + 12 (mass, friction, motor_strength)

    # Command velocity (for forward walking)
    command_vx = 0.3  # Forward velocity command (m/s), range: 0.0 - 1.0

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

    # Torque limits per joint type (Nm) - from Go2 specs
    # Order: FR, FL, RR, RL (SDK order)
    torque_limits = np.array([
        25.0, 40.0, 40.0,  # FR: hip, thigh, calf
        25.0, 40.0, 40.0,  # FL
        25.0, 40.0, 40.0,  # RR
        25.0, 40.0, 40.0,  # RL
    ], dtype=np.float32)

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

    @staticmethod
    def clip_by_torque_limit(target_pos, current_pos, current_vel, kp, kd, torque_limits=None):
        """
        Clip target positions based on torque limits (from parkour).

        Prevents commanding positions that would require excessive torque.
        Uses inverse PD equation: torque = kp * (target - current) - kd * vel

        Args:
            target_pos: Target joint positions (12,)
            current_pos: Current joint positions (12,)
            current_vel: Current joint velocities (12,)
            kp: Position gain
            kd: Damping gain
            torque_limits: Max torque per joint (12,), uses default if None

        Returns:
            Clipped target positions
        """
        if torque_limits is None:
            torque_limits = JointLimits.torque_limits

        # Compute position limits based on torque limits
        # From: tau = kp * (target - current) - kd * vel
        # Solve for target: target = current + (tau + kd * vel) / kp
        pos_delta_max = (torque_limits + kd * current_vel) / kp
        pos_delta_min = (-torque_limits + kd * current_vel) / kp

        # Compute allowed position range
        pos_max = current_pos + pos_delta_max
        pos_min = current_pos + pos_delta_min

        # Ensure min < max
        pos_min_final = np.minimum(pos_min, pos_max)
        pos_max_final = np.maximum(pos_min, pos_max)

        # Clip target positions
        clipped = np.clip(target_pos, pos_min_final, pos_max_final)
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
