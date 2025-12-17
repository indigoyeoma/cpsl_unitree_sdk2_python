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
