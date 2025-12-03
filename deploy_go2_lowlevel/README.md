# Go2 Vision Policy Deployment

Deploy trained vision-based parkour policies to the Unitree Go2 robot using low-level control.

## Overview

This deploys a student (vision) policy trained with the teacher-student framework. The policy uses:
- **Intel RealSense D435i** depth camera for vision
- **Real robot sensors** (IMU, joint encoders, foot forces)
- **Low-level motor control** at 500Hz (50Hz policy)

The vision policy replaces terrain scans, enabling real-world deployment without privileged information.

## Files

```
deploy_go2_lowlevel/
├── deploy.py          # Main deployment script
├── config.py          # Configuration (PD gains, command velocity, etc.)
├── depth_camera.py    # D435i camera interface
├── policy_jit.py      # JIT policy loader
├── README.md          # This file
└── policy/            # Your trained models go here
    ├── student_depth-xxxxx-base_jit.pt
    └── student_depth-xxxxx-vision_weight.pt
```

## Prerequisites

### Software
```bash
# Install pyrealsense2 for D435i camera
pip install pyrealsense2

# Install PyTorch (if not already installed)
pip install torch

# Unitree SDK is already in parent directory
```

### Hardware
- Unitree Go2 robot
- Intel RealSense D435i depth camera mounted on robot head
- Network connection to robot (Ethernet or WiFi)

## Quick Start

### 1. Train Student Policy

```bash
cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym

# Train teacher (uses terrain scans)
python legged_gym/scripts/train.py --task go2 --proj_name parkour_go2 --exptid teacher_step

# Train student (uses depth camera, distills from teacher)
python legged_gym/scripts/train.py --task go2 --proj_name parkour_go2 --exptid student_depth \
       --use_camera --resume --resumeid teacher_step

# Verify in simulation
python legged_gym/scripts/play.py --task go2 --proj_name parkour_go2 --exptid student_depth --use_camera
```

### 2. Export JIT Models

```bash
cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym

# Export to JIT format
python legged_gym/scripts/save_jit.py --exptid student_depth --checkpoint -1
```

This creates in `logs/parkour_go2/student_depth/traced/`:
- `student_depth-<ckpt>-vision_weight.pt` - Depth encoder weights
- `student_depth-<ckpt>-base_jit.pt` - JIT traced policy

### 3. Copy Models to Deployment

```bash
# Copy JIT models to deployment directory
cp logs/parkour_go2/student_depth/traced/*.pt \
   /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel/policy/
```

### 4. Deploy to Robot

**Test with dummy camera first (safe!):**
```bash
cd /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel

python deploy.py --command_vx 0.5 --use_dummy_camera
```

**Deploy to real robot:**
```bash
cd /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel

python deploy.py --command_vx 0.5
```

## Command Line Options

```bash
python deploy.py [options]

Options:
  --command_vx FLOAT       Forward velocity goal in m/s (default: 0.5)
  --device DEVICE          cuda or cpu (default: cuda)
  --use_dummy_camera       Use dummy depth images for testing
  --network_interface IF   Network interface for DDS (e.g., eth0)
  --policy_dir DIR         Directory with JIT models (default: ./policy/)
  --vision_weight PATH     Direct path to vision_weight.pt (optional)
  --base_jit PATH          Direct path to base_jit.pt (optional)
```

## Command Velocity Guide

The `--command_vx` parameter sets the **walking speed goal**:

| Value | Behavior | Recommended Use |
|-------|----------|-----------------|
| 0.0   | Stand still | Testing stance |
| 0.3   | Slow walk | Initial testing |
| 0.5   | Medium walk | **Start here** |
| 0.8   | Fast walk | After confidence |
| 1.0   | Very fast | Use with caution |

**Important:** This is a **goal**, not actual velocity. The policy tries to achieve this speed.

## Deployment Flow

1. **Initialization** (5 seconds)
   - Connects to robot via DDS
   - Releases built-in motion controller
   - Loads JIT models
   - Starts D435i camera

2. **Standup** (2 seconds)
   - Smoothly interpolates to standing pose

3. **Vision Control Loop**
   - **500Hz**: Motor commands (PD control)
   - **50Hz**: Policy inference
     1. Capture depth image from D435i
     2. Read sensors (IMU, joints, contacts)
     3. depth_encoder: depth image → 32-dim latent
     4. policy: obs + depth_latent → 12 joint actions
     5. Apply with smooth 10-second startup ramp

4. **Shutdown**
   - Press Ctrl+C anytime
   - Motors switch to damping mode

## How Vision Works (No Height Scans!)

**Training:** Policy learned with terrain height scans (132 points)

**Deployment:** D435i depth camera replaces height scans

```
Observation Vector (753 dims total):

  proprio (53):
    - IMU: angular velocity (3), roll/pitch (2)
    - Commands: yaw (3), velocity GOAL (1)  ← YOU SET THIS
    - Env flags (2)
    - Joint positions relative to default (12)
    - Joint velocities (12)
    - Last action (12)
    - Foot contacts (4)

  scan (132): ALL ZEROS - no height scans in real world!

  priv_explicit (9): ZEROS - estimator predicts internally

  priv_latent (29): ZEROS - not used by student

  history (530): Last 10 proprio observations

Vision Processing (separate from observation):
  depth_image (58x87) → depth_encoder → depth_latent (32)

Policy receives: obs (753) + depth_latent (32) → actions (12)
```

**Key insight:** Vision replaces scans. The depth encoder processes D435i images into a 32-dim feature that the policy uses instead of terrain heights.

## Safety Features

- ✅ **Smooth startup**: 10-second ramp avoids sudden movements
- ✅ **Joint limits**: Commands clipped to safe ranges
- ✅ **Action scaling**: 0.25× scaling limits speed
- ✅ **Emergency stop**: Ctrl+C anytime → damping mode
- ✅ **Safety confirmation**: Requires user confirmation before starting

## Camera Mounting

Mount D435i on Go2 head:
- **Position**: ~30cm forward, 7cm up from base_link
- **Orientation**: Forward-facing, ~5° down angle
- **FOV**: 86° horizontal (D435i spec)
- **Must match**: Training config in `go2_parkour_config.py`

Verify mounting matches training, or robot may behave erratically!

## Troubleshooting

### Camera not found
```bash
# Check if D435i is connected
rs-enumerate-devices

# Test camera
realsense-viewer
```

### Robot not responding
- Check power and network
- Try `--network_interface eth0`
- Verify robot is not in error state

### Robot falls or jerky movement
- **Start slower**: Use `--command_vx 0.3`
- Check camera is firmly mounted
- Verify training worked well in simulation
- Check floor lighting (affects depth camera)

### Model loading errors
- Verify files exist in `policy/` directory
- Check PyTorch version matches training
- Ensure both `.pt` files are present

### ImportError or module issues
```bash
# Make sure you're in the right directory
cd /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel

# Install SDK if needed
cd .. && pip install -e .
```

## Technical Details

- **Control frequency**: 500Hz motor commands, 50Hz policy (matches training decimation)
- **PD gains**: Kp=40, Kd=1 (from training config)
- **Action scale**: 0.25 (from training config)
- **Depth processing**: 424x240@30fps → resize to 87x58 → normalize [0,1]
- **Joint order**: Converts between training (FL,FR,RL,RR) and SDK (FR,FL,RR,RL)

## Tips for Success

1. **Start conservative**: Begin with `--command_vx 0.3`, increase gradually
2. **Test in simulation first**: Verify policy works with `play.py --use_camera`
3. **Clear space**: Remove obstacles, test on flat ground first
4. **Good lighting**: Depth camera needs decent lighting
5. **Secure mounting**: Camera must not move during operation
6. **Monitor startup**: Watch the 10-second ramp, stop if issues

## What to Expect

- **Seconds 0-10**: Gradual ramp from standing to walking
- **After 10s**: Policy runs at full strength
- **Walking**: Should achieve roughly the commanded velocity
- **Vision**: Robot should react to visible obstacles/terrain

If robot walks but ignores obstacles, check camera mounting and focus.

## Files Explained

- **`deploy.py`**: Main script, handles everything
- **`policy_jit.py`**: Loads depth_encoder + JIT policy
- **`depth_camera.py`**: D435i capture and preprocessing
- **`config.py`**: Robot parameters (PD gains, limits, dimensions)

## Contact

For issues, see main repository or check logs in deployment directory.
