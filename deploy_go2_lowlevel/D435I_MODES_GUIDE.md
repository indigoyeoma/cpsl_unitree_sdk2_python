# Intel RealSense D435i Visual Presets Guide

## 🎯 What Are Visual Presets?

Visual presets are **pre-configured depth processing settings** that optimize the D435i for different scenarios. They affect:
- Depth accuracy
- Noise levels
- Hole filling
- Processing speed
- Range

---

## 📊 Available Visual Presets

### **Preset IDs (rs.option.visual_preset)**

| ID | Name | Best For | Accuracy | Speed | Noise |
|----|------|----------|----------|-------|-------|
| 0 | **Custom** | User-defined settings | Variable | Variable | Variable |
| 1 | **Default** | General use | Medium | Fast | Medium |
| 2 | **Hand** | Hand tracking | Medium | Fast | Low |
| 3 | **High Accuracy** | Precise measurements | **High** | Slow | Very Low |
| 4 | **High Density** | Dense point clouds | Medium | Medium | Medium |
| 5 | **Medium Density** | Balanced performance | Medium | Fast | Medium |

---

## 🔍 Detailed Preset Descriptions

### **0. Custom**
```python
depth_sensor.set_option(rs.option.visual_preset, 0)
```
**What it does:**
- Uses manually configured settings
- No automatic adjustments

**Use when:**
- You want full control over all parameters
- Experimenting with custom settings

**For Go2 walking:** ❌ Not recommended (too much manual tuning)

---

### **1. Default**
```python
depth_sensor.set_option(rs.option.visual_preset, 1)
```
**What it does:**
- Balanced settings for general use
- Good compromise between speed and accuracy
- Moderate hole filling

**Characteristics:**
- Depth range: 0.3m - 3m
- Medium noise filtering
- Fast processing

**Use when:**
- General robotics applications
- Indoor navigation
- Balanced performance needed

**For Go2 walking:** ✓ OK, but not optimal

---

### **2. Hand**
```python
depth_sensor.set_option(rs.option.visual_preset, 2)
```
**What it does:**
- Optimized for tracking hands/gestures
- Prioritizes close-range accuracy (0.3-1m)
- High temporal filtering (reduces noise in motion)

**Characteristics:**
- Best range: 0.3m - 1.5m
- High noise filtering for moving objects
- Fast processing

**Use when:**
- Hand tracking
- Gesture recognition
- Close-range object manipulation

**For Go2 walking:** ⚠️ Suboptimal (range too short, over-filtered)

---

### **3. High Accuracy** ⭐ **CURRENT SETTING**
```python
depth_sensor.set_option(rs.option.visual_preset, 3)
```
**What it does:**
- **Maximizes depth accuracy**
- Minimal hole filling (preserves edges)
- Strong noise reduction
- Best for precise measurements

**Characteristics:**
- Depth range: 0.3m - 3m (full range)
- Very low noise
- Clear obstacle edges
- Slower processing (but still real-time)

**Use when:**
- **Obstacle navigation** ← Our use case!
- Precise distance measurements
- Edge detection important
- Accuracy > speed

**For Go2 walking:** ✅ **BEST CHOICE**
- Clear step/obstacle edges
- Accurate depth for timing steps
- Low noise for stable policy input

---

### **4. High Density**
```python
depth_sensor.set_option(rs.option.visual_preset, 4)
```
**What it does:**
- Fills more holes in depth image
- Creates denser point clouds
- More depth pixels, but noisier

**Characteristics:**
- Dense depth coverage
- Aggressive hole filling
- Higher noise in edges
- Medium processing speed

**Use when:**
- 3D reconstruction
- Need complete surface coverage
- Some noise acceptable

**For Go2 walking:** ⚠️ Not ideal
- Too much hole filling (blurs edges)
- Noise could confuse policy
- Obstacles might appear less distinct

---

### **5. Medium Density**
```python
depth_sensor.set_option(rs.option.visual_preset, 5)
```
**What it does:**
- Moderate hole filling
- Balanced density vs. noise
- Faster than High Accuracy

**Characteristics:**
- Good depth coverage
- Moderate hole filling
- Fast processing

**Use when:**
- Real-time applications
- Some holes acceptable
- Speed matters

**For Go2 walking:** ✓ Alternative if performance issues
- Faster than High Accuracy
- Still reasonable quality
- Slightly noisier edges

---

## 🎯 Recommendation for Go2 Parkour Walking

### **Primary Choice: High Accuracy (Preset 3)** ⭐

**Why:**
```
✓ Clear obstacle edges (critical for step detection)
✓ Low noise (stable policy inputs)
✓ Full depth range (0.3-3m matches training)
✓ Accurate distances (proper step timing)
✓ Still real-time (30fps depth)
```

**Trade-off:**
- Slightly slower processing (negligible at 30fps)
- Some holes in textureless surfaces (OK for terrain)

### **Backup Choice: Medium Density (Preset 5)**

**Use if:**
- Performance issues on robot's onboard computer
- Need faster processing
- Policy handles noisier input well (test first!)

---

## 🔧 How to Change Presets

### **In Deployment Code**

Edit `depth_camera.py` line 93:

```python
# Current (High Accuracy):
depth_sensor.set_option(rs.option.visual_preset, 3)

# To try Medium Density:
depth_sensor.set_option(rs.option.visual_preset, 5)

# To try Default:
depth_sensor.set_option(rs.option.visual_preset, 1)
```

### **In Verification Tool**

Edit `verify_depth_camera.py` line 93:

```python
depth_sensor.set_option(rs.option.visual_preset, 3)  # Change this number
```

---

## 🧪 Testing Different Presets

### **Comparison Script**

Create `test_presets.py`:
```python
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()

presets = {
    1: "Default",
    2: "Hand",
    3: "High Accuracy",
    4: "High Density",
    5: "Medium Density"
}

for preset_id, preset_name in presets.items():
    print(f"\n{preset_name} (ID: {preset_id})")
    print("=" * 50)

    try:
        depth_sensor.set_option(rs.option.visual_preset, preset_id)
    except:
        print(f"Could not set preset {preset_id}")
        continue

    # Capture a few frames
    for _ in range(10):
        pipeline.wait_for_frames()

    # Capture test frame
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()

    if depth:
        depth_image = np.asanyarray(depth.get_data())
        print(f"  Non-zero pixels: {np.count_nonzero(depth_image)}")
        print(f"  Mean depth: {depth_image[depth_image > 0].mean() * 0.001:.2f}m")
        print(f"  Holes (zeros): {np.count_nonzero(depth_image == 0)}")

    input("  Press Enter for next preset...")

pipeline.stop()
```

### **Visual Comparison**

1. Run verification tool with each preset
2. Place objects at known distances (0.5m, 1m, 2m)
3. Compare:
   - Edge sharpness (obstacles)
   - Noise level (flat surfaces)
   - Hole filling (texture-less areas)
   - Processing speed

---

## 📊 Preset Comparison for Go2 Walking

| Aspect | Default | Hand | High Accuracy | High Density | Medium Density |
|--------|---------|------|---------------|--------------|----------------|
| **Edge clarity** | Medium | Medium | ⭐ **Excellent** | Poor | Good |
| **Noise level** | Medium | Low | ⭐ **Very Low** | High | Medium |
| **Processing speed** | ⭐ Fast | ⭐ Fast | Medium | Medium | Fast |
| **Hole filling** | Medium | High | ⭐ **Minimal** | Excessive | Medium |
| **Range (0.3-3m)** | ✓ | Partial | ⭐ **Full** | ✓ | ✓ |
| **Obstacle detection** | Good | OK | ⭐ **Excellent** | Fair | Good |
| **For Go2 walking** | ✓ OK | ⚠️ No | ⭐ **Best** | ⚠️ No | ✓ Backup |

---

## 🎛️ Advanced: Manual Parameter Tuning

If you want even more control, you can manually set parameters:

```python
depth_sensor.set_option(rs.option.visual_preset, 0)  # Custom mode

# Then manually set:
depth_sensor.set_option(rs.option.laser_power, 360)  # 0-360 mW
depth_sensor.set_option(rs.option.confidence_threshold, 3)  # 0-3
depth_sensor.set_option(rs.option.min_distance, 0)  # Minimum depth
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)  # Auto exposure
# ... many more options available
```

**For Go2:** Not recommended - High Accuracy preset is well-tuned

---

## 🔍 Preset Selection Flowchart

```
Start
  ↓
Need maximum accuracy? → YES → High Accuracy (3) ⭐
  ↓ NO
  ↓
Performance issues? → YES → Medium Density (5)
  ↓ NO
  ↓
Balanced use? → YES → Default (1)
  ↓
Done!
```

**For Go2 Parkour:** → **High Accuracy (3)** ⭐

---

## ⚙️ Current Configuration

Your deployment uses:

```python
# depth_camera.py line 93
depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
```

**This is optimal for:**
- Detecting steps (5-15cm height)
- Clear obstacle edges
- Stable depth measurements
- Policy trained on clean simulation data

**Don't change unless:**
- Experiencing performance issues
- Testing different presets for comparison

---

## 📝 Testing Checklist

If you want to test a different preset:

- [ ] Edit preset ID in `depth_camera.py`
- [ ] Run verification tool: `python verify_depth_camera.py`
- [ ] Check edge clarity (place small box 1m away)
- [ ] Check noise level (point at flat wall)
- [ ] Check range (test 0.5m, 1.5m, 2.5m distances)
- [ ] Test with obstacles (5-10cm steps)
- [ ] Compare to High Accuracy results
- [ ] Only switch if **clearly better** for your use case

---

## 🚨 Warning

**Changing presets affects depth quality!**

- Noisier depth → Policy gets confused
- Blurred edges → Misses obstacles
- Different characteristics → Sim2real mismatch

**Recommendation:** Stick with **High Accuracy (3)** unless you have a specific reason to change.

---

## ✅ Summary

**For Go2 Vision-Based Parkour Walking:**

```
Best Choice: High Accuracy (Preset 3)
   ✓ Clear obstacle edges
   ✓ Low noise
   ✓ Accurate distances
   ✓ Matches training expectations

Backup: Medium Density (Preset 5)
   ✓ Faster processing
   ⚠ Slightly noisier

Avoid: Hand, High Density
   ✗ Wrong characteristics for walking
```

**Current setting in your code:** ✅ **High Accuracy (3)** - Perfect! Don't change.
