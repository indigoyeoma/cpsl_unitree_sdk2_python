# D435i Camera Setup & Verification Guide

## 🎯 Critical: Camera Orientation Must Match Training!

If the camera orientation doesn't match simulation, **the policy will fail completely**.

---

## 📐 Camera Mounting Specifications

### **Position (from training config)**
```
Relative to robot base_link:
- Forward: 0.28m (28cm in front of base)
- Lateral: 0.0m (centered)
- Vertical: 0.15m (15cm above base)
```

### **Orientation**
```
- Pitch: 5-15° down (camera looks at ground ahead)
- Roll: 0° (camera level, not tilted sideways)
- Yaw: 0° (camera faces forward)
```

### **Visual Reference**
```
Side view of Go2:

    /====\  ← D435i camera (15cm up, 28cm forward)
   /      \    ↓ (pitched 5-15° down)
  |  HEAD  |
  |________|
  |        |
  |  BODY  |  ← base_link
  |________|
 /|        |\
/ |        | \
  ^^      ^^   ← Legs
```

---

## ✅ Verification Checklist

### **Step 1: Run Verification Tool**
```bash
cd /home/nvidiasims/ws_go2/cpsl_unitree_sdk2_python/deploy_go2_lowlevel
conda activate go2gym
python verify_depth_camera.py
```

### **Step 2: Check Camera View**

**✓ CORRECT Setup:**
```
┌─────────────────────┐
│      SKY/CEILING    │ ← Top of image
│                     │
├─────────────────────┤ ← Horizon line (middle-ish)
│                     │
│   GROUND/FLOOR      │ ← Bottom half
│   (visible ahead)   │
└─────────────────────┘
```

**✗ WRONG Setups:**

1. **Upside Down:**
```
┌─────────────────────┐
│   GROUND/FLOOR      │ ← Ground at TOP (BAD!)
│                     │
├─────────────────────┤
│      SKY/CEILING    │ ← Sky at BOTTOM (BAD!)
└─────────────────────┘
Fix: Flip camera 180°
```

2. **Pitched Up (looking at sky):**
```
┌─────────────────────┐
│      SKY/CEILING    │ ← Mostly sky (BAD!)
│      SKY/CEILING    │
├─────────────────────┤
│   tiny bit ground   │
└─────────────────────┘
Fix: Tilt camera down 10-15°
```

3. **Pitched Too Far Down:**
```
┌─────────────────────┐
│   GROUND (too close)│ ← Only robot feet (BAD!)
│   ROBOT FEET        │
│   ROBOT BODY        │
└─────────────────────┘
Fix: Tilt camera up slightly
```

4. **Rotated 90°:**
```
┌─────┐
│GROU │
│ND   │ ← Image is sideways (BAD!)
│     │
│SKY  │
└─────┘
Fix: Rotate camera to landscape orientation
```

### **Step 3: Interactive Tests**

While running `verify_depth_camera.py`:

1. **Hand Test:**
   - Wave hand 30cm in front of camera
   - Should appear **BRIGHT/HOT** (close objects)
   - If appears dark/cold → depth values inverted

2. **Ground Test:**
   - Point at ground 1-2m ahead
   - Ground should be visible in **BOTTOM HALF**
   - If ground in top half → camera upside down

3. **Distance Test:**
   - Point at wall 0.5m away → BRIGHT
   - Point at wall 2.5m away → DARK
   - If reversed → depth processing wrong

4. **Step Test:**
   - Place small box/book 1m ahead
   - Should see clear depth discontinuity
   - Obstacle should be brighter than ground behind it

---

## 🔍 Common Issues & Fixes

### **Issue 1: Image Upside Down**
**Symptom:** Ground appears at top of image
**Cause:** Camera mounted upside down
**Fix:** Rotate camera 180° around its forward axis

### **Issue 2: Depth Values Inverted**
**Symptom:** Close objects dark, far objects bright
**Cause:** Missing negative sign or wrong invert flag
**Fix:** Check `config.py` has correct `depth_scale` and processing

### **Issue 3: Image Rotated 90°**
**Symptom:** Image is portrait instead of landscape
**Cause:** Camera rotated sideways
**Fix:** Rotate camera to landscape orientation (87° horizontal FOV)

### **Issue 4: Ground Not Visible**
**Symptom:** Only see ceiling or robot body
**Cause:** Camera pitch angle wrong
**Fix:** Adjust pitch to 5-15° down

### **Issue 5: Mirrored Horizontally**
**Symptom:** Obstacles on left appear on right in policy behavior
**Cause:** Camera or image processing flipped
**Fix:** Check camera mounting, may need cv2.flip in processing

---

## 📊 Expected Depth Values

### **Training Normalization**
```python
depth_normalized = (depth - 0.3) / (3.0 - 0.3) - 0.5
# Range: [-0.5, 0.5]
#   0.3m (near) → 0.5
#   1.65m (mid) → 0.0
#   3.0m (far) → -0.5
```

### **Verification Values**
Place objects at known distances and verify depth readings:

| Distance | Expected Normalized Value | Visual Color |
|----------|-------------------------|--------------|
| 0.3m | +0.5 | Bright red/yellow |
| 1.0m | ~0.0 | Green |
| 2.0m | ~-0.3 | Blue |
| 3.0m | -0.5 | Dark blue |

---

## 🎬 Pre-Deployment Verification

Before deploying to real robot, verify ALL of these:

- [ ] Camera mounted at correct position (28cm forward, 15cm up)
- [ ] Camera pitched down 5-15°
- [ ] Ground visible in bottom half of image
- [ ] Close objects appear brighter than far objects
- [ ] Image is landscape orientation (not rotated)
- [ ] Image is right-side up (not flipped)
- [ ] Processed image (87x58) shows clear terrain features
- [ ] Hand test passes (close = bright)
- [ ] Distance test passes (far = dark)
- [ ] Step/obstacle test passes (clear depth edges)

---

## 🚨 Failure Modes

### **If Orientation is Wrong:**
- Policy will see **incorrect terrain**
- Robot may step at wrong times
- May try to step where there's no obstacle
- May ignore real obstacles
- **Will likely fall or crash!**

### **If Depth Values are Wrong:**
- Close obstacles may appear far (robot steps too late)
- Far terrain may appear close (robot steps too early)
- Robot may lose balance or trip

---

## 💡 Quick Validation Test

```bash
# 1. Run verification
python verify_depth_camera.py

# 2. Point camera at ground 1m ahead
#    → Ground should be in BOTTOM HALF ✓
#    → Should see clear depth gradient ✓

# 3. Wave hand 30cm in front
#    → Hand should be BRIGHT (hot color) ✓
#    → Background should be DARK (cool color) ✓

# 4. Place small box 1m ahead
#    → Box should be BRIGHTER than ground ✓
#    → Should see clear edge of box ✓

# 5. Save snapshot (press 's')
#    → Review saved image carefully

# If ALL tests pass → Ready to deploy!
```

---

## 📝 Mounting Tips

1. **Use rigid mount** - Camera must not vibrate or move during walking
2. **Secure cables** - Ensure USB cable doesn't interfere with robot movement
3. **Check FOV** - 87° horizontal should give ~1.2m width at 1m distance
4. **Test mounting** - Walk robot manually, verify camera stays stable
5. **Protect lens** - Keep lens clean, no fingerprints or dust

---

## 🔧 Troubleshooting Commands

```bash
# List RealSense devices
rs-enumerate-devices

# Test camera with RealSense viewer
realsense-viewer

# Check camera firmware
rs-fw-update -l

# Reset camera if frozen
rs-enumerate-devices | grep Serial  # Get serial number
rs-enumerate-devices -r <serial>    # Reset device
```

---

## ✅ Final Check

Before deploying policy:

```bash
# 1. Verify camera orientation
python verify_depth_camera.py

# 2. Test with dummy camera first
python deploy.py --use_dummy_camera

# 3. Test with real camera on flat ground
python deploy.py --command_vx 0.3

# 4. Test with small obstacles (5cm)
python deploy.py --command_vx 0.5

# 5. Full deployment
python deploy.py  # Default: 0.5 m/s, 2.5m mission
```

---

## 📞 Support

If depth image orientation looks wrong after following this guide:
1. Save snapshot with 's' key in verification tool
2. Check against visual references in this guide
3. Adjust camera mounting physically
4. Re-run verification until all checks pass

**Do NOT proceed to deployment if verification fails!**
