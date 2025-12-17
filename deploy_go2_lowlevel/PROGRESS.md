# Joint Ordering Notes for Go2 Deployment
**Date: 2025-12-16**

## Summary

CPSL training code uses `reindex()` in observations, which means:
- **Policy sees SDK order** (after reindex from SIM)
- **Policy outputs SDK order**
- **Deployment needs NO reindexing** (sensors already give SDK order)

---

## Joint Order Definitions

### SIM Order (Isaac Gym / URDF order)
```
Index 0-2:   FL (Front Left)  - hip, thigh, calf
Index 3-5:   FR (Front Right) - hip, thigh, calf
Index 6-8:   RL (Rear Left)   - hip, thigh, calf
Index 9-11:  RR (Rear Right)  - hip, thigh, calf
```

### SDK Order (Real Robot Hardware order)
```
Index 0-2:   FR (Front Right) - hip, thigh, calf
Index 3-5:   FL (Front Left)  - hip, thigh, calf
Index 6-8:   RR (Rear Right)  - hip, thigh, calf
Index 9-11:  RL (Rear Left)   - hip, thigh, calf
```

---

## The reindex() Function

```python
def reindex(self, vec):
    return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
```

This mapping:
- Swaps FL (indices 0-2) with FR (indices 3-5)
- Swaps RL (indices 6-8) with RR (indices 9-11)
- Converts SIM order ↔ SDK order (self-inverse)

---

## CPSL Training Code Flow

Location: `cpsl_go2_rl_repo/legged_gym/legged_gym/envs/base/legged_robot.py`

### Observations (compute_observations):
```python
# Line 431-434
self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
self.reindex(self.dof_vel * self.obs_scales.dof_vel),
self.reindex(self.action_history_buf[:, -1]),
self.reindex_feet(self.contact_filt.float()-0.5),
```
- Simulator state is SIM order
- `reindex()` converts to SDK order
- **Policy sees SDK order observations**

### Actions (step):
```python
# Line 120
actions = self.reindex(actions)
```
- Policy outputs SDK order
- `reindex()` converts to SIM order for physics

---

## Deployment Logic (policy_runner.py)

### Observations:
```
Real sensors (SDK order) → Policy (expects SDK order)
NO CONVERSION NEEDED
```

### Actions:
```
Policy output (SDK order) → Motors (expect SDK order)
NO CONVERSION NEEDED
```

---

## Comparison: Parkour vs CPSL

| Aspect | Parkour | CPSL/extreme-parkour |
|--------|---------|---------------------|
| Training observations | SIM order (no reindex) | SDK order (has reindex) |
| Policy output | SIM order | SDK order |
| Deployment conversion | SDK↔SIM needed | No conversion needed |

### Parkour training (NO reindex):
```python
# parkour/legged_gym/legged_gym/envs/base/legged_robot.py
def _get_dof_pos_obs(self, privileged=False):
    return (self.dof_pos - self.default_dof_pos)  # NO reindex
```

### CPSL training (HAS reindex):
```python
# cpsl_go2_rl_repo/legged_gym/legged_gym/envs/base/legged_robot.py
self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos)  # HAS reindex
```

---

## Key Files Reference

- Training code: `/home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym/legged_gym/envs/base/legged_robot.py`
- Deployment code: `/home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel/policy_runner.py`
- Parkour deployment: `/home/nvidiasims/ws_go2/parkour/onboard_codes/go2/unitree_ros2_real.py`

---

## Default Standing Angles

### SDK Order (used in deployment):
```python
DEFAULT_STAND_ANGLES_SDK = [
    -0.1, 0.8, -1.5,    # FR: hip, thigh, calf
     0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 1.0, -1.5,    # RR: hip, thigh, calf
     0.1, 1.0, -1.5,    # RL: hip, thigh, calf
]
```

### SIM Order (used in training simulation):
```python
DEFAULT_STAND_ANGLES_SIM = [
     0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 0.8, -1.5,    # FR: hip, thigh, calf
     0.1, 1.0, -1.5,    # RL: hip, thigh, calf
    -0.1, 1.0, -1.5,    # RR: hip, thigh, calf
]
```

Note: Hip signs are opposite for left vs right legs.

---

## Two-Process Architecture (Added 2025-12-17)

### Problem: Slow Inference

Initial single-process deployment had ~420ms per policy step:
- Depth encoder: ~320ms (CNN on GPU)
- Base policy: ~100ms
- Result: Policy running at ~2Hz instead of 50Hz

### Solution: Decoupled Processes (like Parkour)

**Process 1: `deploy.py`** (Main process)
- Motor control at 500Hz
- Policy at 50Hz (every 10 ticks)
- Reads depth embedding from shared memory
- Writes proprio to shared memory

**Process 2: `depth_encoder_process.py`** (Spawned)
- Captures depth from camera
- Runs depth encoder on **CPU** (like parkour)
- Writes embedding to shared memory
- Runs at ~10Hz (or as fast as encoder allows)

### Architecture Diagram

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│  deploy.py (50Hz policy)         │      │  depth_encoder_process.py (~10Hz)│
│                                  │      │                                  │
│  1. Read robot state             │      │  1. Capture depth from camera    │
│  2. Write proprio → shared mem ──┼──────┼─▶ 2. Read proprio from shared    │
│  3. Read embedding ← shared mem ◀┼──────┼── 3. Run depth encoder (CPU)     │
│  4. Run base policy (GPU, fast)  │      │  4. Write embedding → shared     │
│  5. Send motor commands          │      │                                  │
└──────────────────────────────────┘      └──────────────────────────────────┘
```

### Shared Memory Buffers

```python
shared_embedding = Array(c_float, 32)      # Depth latent [32] from encoder
shared_proprio = Array(c_float, N_PROPRIO) # Proprio [53] for encoder
```

### Comparison: Parkour vs Our Implementation

| Aspect | Parkour | Our Implementation |
|--------|---------|-------------------|
| IPC | ROS2 topics | Python shared memory |
| Launch | Two separate scripts | Single script (spawns process) |
| Dependency | Requires ROS2 | No ROS2 needed |
| Depth encoder device | CPU | CPU |

### Why CPU for Depth Encoder?

Parkour runs depth encoder on CPU (`device = "cpu"` in go2_visual.py line 278):
1. Avoids GPU memory contention with policy
2. More predictable timing (no CUDA sync overhead)
3. CPU inference is often faster for small models on Jetson

### Key Files

- Main deployment: `deploy.py`
- Depth encoder process: `depth_encoder_process.py`
- Policy runner (base policy only): `policy_runner.py`

### Expected Timing After Fix

- Depth encoder: ~50-100ms on CPU (runs independently)
- Base policy: ~10-20ms on GPU
- Policy loop: 50Hz (not blocked by depth encoder)

---

## Yaw Prediction Fix (Added 2025-12-17)

### Problem: Robot Turning Right

Robot was walking but continuously turning right (yaw drifting to -32 degrees).

### Root Cause

The depth encoder outputs 34 values:
- Indices 0-31: Depth latent (32 dims)
- Indices 32-33: Yaw prediction (delta_yaw_sin, delta_yaw_cos)

The yaw prediction tells the policy how to correct heading based on visual input,
but it wasn't being used in the observation.

### Solution

1. **depth_encoder_process.py**: Write yaw prediction to shared_embedding[32:34]
   ```python
   yaw_pred = output[0, 32:34].numpy() * 1.5  # Scale like parkour
   shared_embedding[32] = yaw_pred[0]  # delta_yaw_sin
   shared_embedding[33] = yaw_pred[1]  # delta_yaw_cos
   ```

2. **deploy.py**: Increase shared memory size from 32 to 34
   ```python
   self.shared_embedding = Array(c_float, DEPTH_LATENT_DIM + 2)  # 34 total
   ```

3. **policy_runner.py**: Read yaw prediction and use in observation
   ```python
   def get_yaw_pred_from_shared(self):
       yaw_pred[0] = self.shared_embedding[32]  # delta_yaw_sin
       yaw_pred[1] = self.shared_embedding[33]  # delta_yaw_cos
   ```
   Then pass to `build_proprio_obs()` which puts values at observation indices 6-7.

### Observation Structure (first 13 elements)

```
Index 0-2:   ang_vel_scaled [3]
Index 3-4:   roll, pitch [2]
Index 5:     delta_yaw_masked (always 0)
Index 6:     delta_yaw_sin (from depth encoder)
Index 7:     delta_yaw_cos (from depth encoder)
Index 8-9:   commands_masked (always 0)
Index 10:    cmd_vel_x (forward velocity command)
Index 11:    env_class_flag_1
Index 12:    env_class_flag_2
```

### How Yaw Correction Works

1. Robot walks forward with `cmd_vel_x = 0.5 m/s`
2. Depth encoder sees terrain and predicts yaw correction needed
3. If robot turns right, encoder should predict left correction (positive delta_yaw)
4. Policy uses yaw prediction to adjust leg movements for steering

---

## Goal-Based Navigation Fix (Added 2025-12-17)

### Problem: Depth Encoder Outputs Wrong Yaw on Flat Ground

Even after implementing yaw prediction, robot kept drifting right.

**Root Cause**: The depth encoder was trained with explicit goal waypoints:
```python
# Training code computes:
delta_yaw = target_yaw - current_yaw  # direction to goal waypoint
```

On flat ground with no visual features, the encoder has no "goal" to navigate toward,
so it outputs its learned bias (which happened to be negative = turn right).

### Solution: Fixed Goal Direction

Instead of using depth encoder's yaw prediction, set a fixed goal direction:

```python
# In depth_encoder_process.py:
USE_DEPTH_ENCODER_YAW = False  # Disable encoder yaw prediction

# Fixed goal (meters relative to robot)
GOAL_X = 10.0   # ahead
GOAL_Y = 0.0    # side (+ left, - right)

# Compute delta_yaw to goal
delta_yaw = math.atan2(GOAL_Y, GOAL_X) + DRIFT_CORRECTION
```

### Hardware Drift Correction

Even with fixed goal, robot had inherent rightward drift (hardware/policy bias).
Added constant correction:

```python
DRIFT_CORRECTION = 0.23  # Tuned value (positive = turn left)
```

**Tuning process:**
| Value | Result |
|-------|--------|
| 0.0   | Drifted right ~52° |
| 0.15  | Drifted right ~54° |
| 0.35  | Drifted left ~48° |
| 0.23  | Approximately straight |

### Configuration Options

```python
# depth_encoder_process.py settings:

USE_DEPTH_ENCODER_YAW = False  # True for terrain with obstacles
GOAL_X = 10.0                  # meters ahead
GOAL_Y = 0.0                   # meters to side (+ left, - right)
DRIFT_CORRECTION = 0.23        # hardware drift compensation
```

### Goal Examples

| Scenario | GOAL_X | GOAL_Y | delta_yaw |
|----------|--------|--------|-----------|
| Straight ahead | 10.0 | 0.0 | 0° |
| Slight left | 10.0 | 2.0 | ~11° |
| Hard left | 10.0 | 10.0 | ~45° |
| Slight right | 10.0 | -2.0 | ~-11° |

### When to Use Each Mode

| Mode | Setting | Use Case |
|------|---------|----------|
| Fixed goal | `USE_DEPTH_ENCODER_YAW = False` | Flat ground, walk straight |
| Depth encoder | `USE_DEPTH_ENCODER_YAW = True` | Terrain with obstacles/features |

### Key Insight

The depth encoder predicts yaw corrections based on **visual features**.
Without features (flat ground), it outputs meaningless values.
For flat ground testing, use fixed goal direction instead.

---

## Vibration Fix (Added 2025-12-17)

### Problem: Robot Vibrating During Walking

### Solution: Increased Damping

```python
# config.py
KD_WALK = 1.0  # Increased from 0.6
```

Original ratio KP/KD = 25/0.6 = 41.7 (too high, causes oscillation)
New ratio KP/KD = 25/1.0 = 25 (more stable)
