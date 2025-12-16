# Go2 Vision Policy Deployment

Deploy trained vision-based parkour policies to the Unitree Go2 robot using low-level control.

## Overview

This deploys a student (vision) policy trained with the CPSL teacher-student framework. The system uses:
- **Intel RealSense D435i** depth camera for vision
- **Real robot sensors** (IMU, joint encoders, foot forces)
- **Low-level motor control** at 500Hz with 50Hz policy inference

The vision policy replaces terrain height scans with depth images, enabling real-world deployment.

## Architecture

```
Training (Simulation)                    Deployment (Real Robot)
─────────────────────                    ───────────────────────
Teacher Policy                           Student Policy
  - Uses height scans (132 pts)            - Uses D435i depth camera
  - Privileged terrain info                - No privileged info

       ↓ Distillation                     D435i Camera (424x240)
                                                ↓
Student Policy                           Crop (parkour ratios)
  - Uses simulated depth                       ↓
  - Learns from teacher                  Resize to 87x58
                                               ↓
                                         Normalize [-0.5, +0.5]
                                               ↓
                                         Depth Encoder → 32-dim latent
                                               ↓
                                         Policy: obs + latent → actions
```

## Files

```
deploy_go2_lowlevel/
├── deploy.py              # Main deployment script (state machine)
├── config.py              # Configuration (PD gains, joint mapping, etc.)
├── depth_camera.py        # D435i camera interface with parkour preprocessing
├── policy_jit.py          # JIT policy loader (depth encoder + base policy)
├── test_camera.py         # Test camera independently
├── CAMERA_SETUP_GUIDE.md  # D435i setup instructions
├── D435I_MODES_GUIDE.md   # Camera mode documentation
└── policy/                # Trained models directory
    ├── *-base_jit.pt      # JIT traced policy
    └── *-vision_weight.pt # Depth encoder weights
```

## Camera Settings (Matching Parkour)

**Critical:** Camera preprocessing must match training exactly!

| Setting | Parkour Reference (640x480) | Deployment (424x240) | Training Sim (106x60) |
|---------|----------------------------|----------------------|----------------------|
| crop_top | 48 (10%) | 24 (10%) | 6 (10%) |
| crop_bottom | 0 (0%) | 0 (0%) | 0 (0%) |
| crop_left | 28 (4.4%) | 19 (4.5%) | 5 (4.7%) |
| crop_right | 36 (5.6%) | 24 (5.7%) | 6 (5.7%) |
| near_clip | - | 0.3m | 0.3m |
| far_clip | - | 3.0m | 3.0m |
| output | 87x58 | 87x58 | 87x58 |
| normalization | - | [-0.5, +0.5] | [-0.5, +0.5] |

**Normalization formula:**
```python
depth_normalized = (depth_meters - near_clip) / (far_clip - near_clip) - 0.5
# Result: -0.5 = close (0.3m), +0.5 = far (3.0m)
```

**Edge artifact fix:** Left column copied from column 1 to handle D435i edge artifacts.

## Control Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Policy frequency | 50 Hz | Matches training decimation=4 |
| Motor command rate | 500 Hz | Smooth control |
| Kp (walking) | 25.0 | Matches training stiffness |
| Kd (walking) | 0.6 | Matches training damping |
| Kp (standing) | 70.0 | Stiffer for stability |
| Kd (standing) | 3.0 | Higher damping |
| action_scale | 0.25 | Matches training |
| clip_actions | 1.2 | Observation clipping (parkour style) |

## Joint Order Mapping

**Training (URDF) order:** FL, FR, RL, RR (hip, thigh, calf each)
**SDK order:** FR, FL, RR, RL (hip, thigh, calf each)

The deployment code automatically converts between these orders.

## Button Controls

| Button | Action |
|--------|--------|
| **L1** | Start walking (STANDING → WALKING) |
| **Y** | Stop walking (WALKING → STANDING) |
| **B** | Print depth + IMU debug info (saves to depth_imu_log.txt) |
| **Ctrl+C** | Emergency stop → damping mode |

## State Machine

```
                    ┌─────────┐
                    │  IDLE   │ (damping mode, waiting)
                    └────┬────┘
                         │ Y button
                         ▼
                    ┌─────────┐
                    │STANDING │ (interpolate to stand pose)
                    └────┬────┘
                         │ L1 button
                         ▼
                    ┌─────────┐
                    │ WALKING │ (vision policy active)
                    └────┬────┘
                         │ Y button
                         ▼
                    ┌─────────┐
                    │STANDING │
                    └─────────┘
```

## Quick Start

### 1. Train Student Policy

```bash
cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym

# Train student with depth camera (distills from teacher)
python train.py --task go2_student \
    --exptid go2_student \
    --load_run ../../logs/go2_teacher/go2_teacher \
    --use_camera \
    --headless \
    --no_wandb \
    --max_iterations 15005
```

**Training uses:**
- Single depth frame per timestep (buffer_len=1)
- Parkour-style crop ratios
- near_clip=0.3m, far_clip=3.0m
- 192 parallel environments (camera_num_envs)

### 2. Export JIT Models

```bash
cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym

# Export to JIT format for deployment
python legged_gym/scripts/save_jit.py --exptid go2_student --checkpoint -1
```

Creates in `logs/go2_student/traced/`:
- `*-base_jit.pt` - JIT traced policy
- `*-vision_weight.pt` - Depth encoder weights

### 3. Copy Models

```bash
cp logs/go2_student/traced/*.pt \
   /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel/policy/
```

### 4. Deploy to Robot

```bash
cd /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel

# Run deployment
python deploy.py
```

**Deployment sequence:**
1. Press **Y** to enter STANDING mode
2. Robot interpolates to standing pose
3. Press **L1** to start WALKING
4. Robot walks forward using vision policy
5. Press **Y** to stop, **Ctrl+C** for emergency stop

## Command Line Options

```bash
python deploy.py [options]

Options:
  --command_vx FLOAT       Forward velocity goal (default: 0.3 m/s)
  --device DEVICE          cuda or cpu (default: cuda)
  --use_dummy_camera       Use dummy depth for testing without camera
  --network_interface IF   Network interface for DDS
```

## Debugging

### B Button Debug Output

Press **B** during operation to log:
- IMU data (roll, pitch, yaw, angular velocity)
- Depth image statistics and samples
- Current observations
- Saves to `depth_imu_log.txt`

### Walk Log

During walking, comprehensive logs saved to `walk_log.txt` every 0.2 seconds:
- Joint positions/velocities
- Actions (raw and clipped)
- Depth image analysis
- Navigation state

### Common Issues

**Robot turns right constantly:**
- Left-edge camera artifact causing phantom obstacle detection
- Fixed by cropping and edge copy (`depth[:, 0] = depth[:, 1]`)

**Jerky movements:**
- PD gains mismatch (should be Kp=25, Kd=0.6 for walking)
- Action scale mismatch (should be 0.25)
- Check depth normalization range

**Robot doesn't respond to obstacles:**
- Verify camera is connected: `rs-enumerate-devices`
- Check depth values with B button
- Ensure near_clip/far_clip match training

**Policy loading fails:**
- Ensure both `.pt` files are in `policy/` directory
- Check PyTorch version compatibility

## Observation Structure

```
Total observation: 753 dimensions

proprio (53):
  [0:3]   angular_velocity * 0.25
  [3:5]   roll, pitch
  [5]     delta_yaw_mask (0 = use delta_yaw)
  [6]     delta_yaw (to goal)
  [7]     delta_next_yaw
  [8:10]  commands (masked, zeros)
  [10]    command_vx (forward velocity goal)
  [11:13] env_class one-hot
  [13:25] dof_pos - default_pos (12 joints)
  [25:37] dof_vel * 0.05 (12 joints)
  [37:49] last_action (12 joints)
  [49:53] foot_contacts (±0.5)

scan (132): ZEROS (replaced by depth encoder)

priv_explicit (9): ZEROS

priv_latent (29): ZEROS

history (530): Last 10 proprio observations (53 * 10)

Depth processing (separate):
  D435i (424x240) → crop → resize (87x58) → normalize → encoder → latent (32)
```

## Safety Features

- **Smooth standup**: 2-second interpolation to standing pose
- **Walking ramp**: Gradual action scaling during first 10 seconds
- **Joint limits**: Hardware limits enforced
- **Torque clipping**: Prevents excessive forces
- **Emergency stop**: Ctrl+C → immediate damping mode
- **State confirmation**: Button presses required for state transitions

## Technical Notes

### Why Parkour Crop Ratios?

The parkour repository (extreme-parkour) uses specific crop ratios to:
1. Remove top 10% (sky/ceiling artifacts)
2. Asymmetric left/right crops (4.4% vs 5.6%) for camera alignment
3. No bottom crop (floor is important for locomotion)

We match these ratios proportionally in both training and deployment.

### Depth Normalization

Training uses Isaac Gym depth convention:
- Negative values in simulation (closer = more negative)
- We convert to positive, then normalize to [-0.5, +0.5]
- -0.5 = close (0.3m), +0.5 = far (3.0m)

### Joint Order

Isaac Gym URDF loads joints in FL, FR, RL, RR order.
Unitree SDK uses FR, FL, RR, RL order.
`config.py` contains the mapping indices.

## References

- CPSL Go2 RL Repo: `/home/nvidiasims/ws_go2/cpsl_go2_rl_repo/`
- Parkour Reference: `/home/nvidiasims/ws_go2/parkour/`
- Training Config: `cpsl_go2_rl_repo/legged_gym/legged_gym/envs/go2/go2_student_config.py`
