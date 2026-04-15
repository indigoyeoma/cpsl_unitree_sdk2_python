# SkillFFN: Shared Base + Task-Specific MLP Decomposition for Efficient Multi-Skill Diffusion Locomotion

**Project:** go2_lora_diffuseloco
**Date:** 2026-04-09
**Target Venues:** CoRL, RSS, ICRA, RAL
**Target Platform:** Unitree Go2 (Jetson Orin)

---

## 1. Problem Statement

DiffuseLoco uses a Diffusion Transformer (DiT) to generate multi-skill locomotion actions (trot, pace, hop, bounce, bipedal walk) for quadruped robots. The architecture consists of 6 decoder layers with `n_emb=256` and `dim_feedforward=1024` (the standard 4x expansion), operating over 16 action tokens and ~13 conditioning tokens.

At this token count, the compute profile is inverted compared to image-generation DiTs:

| Component | Cost per layer | Relative |
|-----------|---------------|----------|
| Self-attention | O(T^2 * d) ~ O(16^2 * 256) ~ **65K ops** | 1x |
| FFN (4x MLP) | O(T * d * 4d) ~ O(16 * 256 * 1024) ~ **4.2M ops** | **~64x** |

The FFN dominates compute by a factor of 64x. Across 6 decoder layers, the FFN accounts for ~25M of the ~25.4M total ops -- over **98% of decoder compute**.

The core tension: DiffuseLoco's multi-skill capacity lives primarily in the FFN. The 4x expansion MLP is where the model encodes the distinct motor patterns for each locomotion skill. Naively shrinking the FFN (e.g., reducing from 4x to 2x expansion) destroys multimodality -- the model collapses into an averaged single-gait behavior that fails on specialized skills like bipedal walking.

**The problem is:** How do we reduce FFN compute for real-time edge deployment while preserving the full multi-skill capacity that makes diffusion locomotion policies valuable?

---

## 2. Why We Need to Solve This

### 2.1 Edge Deployment is the Bottleneck

DiffuseLoco deploys on the Unitree Go2 with a Jetson Orin. Real-time locomotion control requires action generation within tight latency budgets (typically 5-20ms per control step). The Jetson Orin is constrained in both raw compute (FLOPs/s) and memory bandwidth (GB/s). The FFN's large weight matrices (W1 in R^{256x1024}, W2 in R^{1024x256} per layer, x6 layers) must be loaded from memory every forward pass, making memory bandwidth the practical bottleneck on edge hardware -- not just FLOPs.

### 2.2 Naive Solutions Fail

**Uniform width reduction** (e.g., 4x -> 2x): Reduces compute but all 5 locomotion skills compete for the same reduced capacity. Skills that are most different from the average (bipedal walk, hopping) degrade catastrophically.

**Masking-based approaches** (E-DiT, DyDiT): These dynamically mask neurons within the full-width MLP, which saves FLOPs but **not memory bandwidth**. The full weight matrices still must be stored and loaded on the device. On Jetson Orin, memory bandwidth is often the true bottleneck, so masking provides limited real-world speedup.

**Full MoE** (MoDE, MoE-Loco): Uses multiple full-sized expert MLPs with learned routers. This adds parameters, router overhead, and training instability. Overkill for a setting with 5 known discrete skills.

### 2.3 The Opportunity

Locomotion skills share significant common structure -- gravity compensation, contact timing, basic balance, joint coordination -- while differing in specific gait patterns, phase offsets, foot clearance profiles, and swing dynamics. This shared-plus-specialized structure is a natural fit for decomposition. Furthermore, DiffuseLoco already conditions on a skill/task label, enabling deterministic routing at inference with zero overhead.

---

## 3. Proposed Solution: SkillFFN

### 3.1 Core Architecture

Replace the standard 4x FFN in each DiT decoder layer with a **dual-pathway SkillFFN**:

```
FFN_k(z) = Base_MLP(z) + Task_MLP_k(z)
```

where:
- `Base_MLP`: A shared MLP with reduced hidden dimension (e.g., 256), always computed. Captures common locomotion dynamics.
- `Task_MLP_k`: A small skill-specific MLP (e.g., hidden dim 256), computed **only** for the active skill k. Captures gait-specific patterns.
- The other (K-1) task MLPs are never loaded or computed at inference.

```
Input z (dim 256)
       |
       +---------------+
       |               |
       v               v
  Base_MLP          Task_MLP_k     <-- only the active skill's MLP
  (256->256->256)   (256->256->256)    is loaded and computed
       |               |
       +-------+-------+
               |
               v
         z_out = base_out + task_out
```

### 3.2 Compute Analysis

**Original FFN per layer (16 tokens):**
- W1 + W2: 256x1024 + 1024x256 -> ~8.4M ops (both matmuls)
- Parameters: 524,288

**SkillFFN per layer (base_dim=256, task_dim=256):**
- Base: 256x256 + 256x256 -> ~2.1M ops
- Task: 256x256 + 256x256 -> ~2.1M ops
- **Active total: ~4.2M ops -> 2x compute reduction**
- Active parameters: 262,144

**More aggressive (base=256, task=128):**
- Active total: ~3.1M ops -> **~2.7x reduction**

**Most aggressive (base=256, task=64):**
- Active total: ~2.6M ops -> **~3.2x reduction**

### 3.3 Relationship to MoE

SkillFFN is structurally an MoE with a shared expert, but with critical simplifications:

| Aspect | Standard MoE (Mixtral) | SkillFFN (Ours) |
|--------|------------------------|-----------------|
| Experts | N full-sized MLPs | 1 small base + K small task MLPs |
| Router | Learned (per-token) | Skill label (deterministic at inference) |
| Active experts | top-k per token | base + 1 task MLP |
| Routing cost | Router forward + softmax | Zero at inference |
| Training | Load-balancing loss needed | Soft masking, no aux losses |
| Memory at inference | All N experts loaded | Base + 1 task MLP only |

---

## 4. Training: Elastic Soft-Masked Training

### 4.1 Core Idea: Train Soft, Deploy Hard

Inspired by E-DiT's elastic training philosophy, we train with all K task MLPs instantiated and use **soft skill-conditioned masking** during training, then harden to deterministic single-task routing at deployment.

### 4.2 Training Forward Pass

For a training sample with skill label k:

```python
# All task MLPs compute in parallel
base_out = Base_MLP(z)                              # always computed
task_outs = [Task_MLP_i(z) for i in range(K)]       # all K computed

# Soft mask: mostly skill k, slight leakage to others
mask = soft_skill_mask(k, K, temperature=tau)
# mask[k] ~ 0.9, mask[others] ~ 0.025 each

task_out = sum(mask[i] * task_outs[i] for i in range(K))
output = base_out + task_out
```

The soft mask is constructed from the one-hot skill label, softened by a temperature parameter:

```
mask = softmax(one_hot(k, K) / tau)
```

At high temperature (early training), the mask is nearly uniform -- all task MLPs receive gradients, encouraging exploration. As temperature anneals toward zero (late training), the mask sharpens toward one-hot, pushing each task MLP to specialize.

### 4.3 Why Soft Masking is Better Than Hard Routing During Training

**Hard routing** (just forward through Base + Task_MLP_k):
- Each Task_MLP only ever sees its own skill's gradients
- No signal pushing the model to cleanly separate shared vs. specific
- Base_MLP may accidentally learn skill-specific features
- Task MLPs may redundantly learn the same shared features

**Soft masking** (slight gradient leakage to inactive Task_MLPs):
- Inactive Task_MLPs receive a small negative gradient from other skills' samples
- This pushes Task_MLPs to differentiate from each other
- Base_MLP receives gradients informed by the masking structure, learning to absorb what's truly shared
- Annealing from soft to hard over training: early epochs encourage sharing, late epochs encourage specialization

### 4.4 Temperature Annealing Schedule

```
tau(t) = tau_start * (tau_end / tau_start) ^ (t / T_total)
```

- `tau_start = 1.0` (nearly uniform mask, all Task_MLPs get ~equal gradient)
- `tau_end = 0.05` (nearly one-hot, each Task_MLP sees mostly its own skill)
- Exponential decay over training steps T_total

At tau=1.0: mask ~ [0.24, 0.24, 0.04, 0.24, 0.24] (for skill k=2 with K=5)
At tau=0.05: mask ~ [0.00, 0.00, 0.98, 0.01, 0.01]

### 4.5 Training Loss

Standard DiffuseLoco diffusion loss (flow matching / DDPM). **No auxiliary routing or load-balancing losses needed** -- the masking structure is entirely determined by the known skill label, not learned.

```
L = L_diffusion(epsilon, epsilon_hat)   # standard denoising loss
```

Stratified batch sampling ensures each mini-batch contains samples from all K skills, so Base_MLP receives balanced gradients every update step.

### 4.6 Initialization

E-DiT's ablation showed that full-capacity initialization dramatically outperforms random init (DPG 85.4 vs 78.6). For SkillFFN:

1. **Base_MLP**: Initialize from the pretrained DiffuseLoco FFN via truncated SVD. Take the top-256 singular value components of the original W1 and W2 matrices.
2. **Task_MLPs**: Initialize as small random noise (scaled by 0.01), or from the SVD residual projected per-skill.

This gives a warm start where the base already captures the dominant locomotion dynamics, and task MLPs start near zero and gradually learn skill-specific corrections.

### 4.7 Training Compute Cost

Per forward pass during training: Base_MLP + K * Task_MLP = (256 dim) + 5 * (256 dim). This totals ~12.6M ops per layer -- more than the inference cost but still comparable to the original 8.4M ops per layer with 4x FFN. Training is on GPU clusters where this overhead is negligible.

---

## 5. Deployment: Deterministic Routing on Jetson Orin

### 5.1 Runtime Memory Layout

```
GPU Memory (always resident):         CPU RAM (cold storage):
+-----------------------------+       +---------------------------+
| Encoder weights             |       | Task_MLP_trot    (~780KB) |
| Attention weights (6 layers)|       | Task_MLP_pace    (~780KB) |
| Base_MLP weights (6 layers) |  swap | Task_MLP_hop     (~780KB) |
| Task_MLP_slot (6 layers)  <-+--<1ms-| Task_MLP_bounce  (~780KB) |
|  currently = active skill   |       | Task_MLP_bipedal (~780KB) |
+-----------------------------+       +---------------------------+
```

### 5.2 Inference Forward Pass

For active skill k:
```
z_out = Base_MLP(z) + Task_MLP_slot(z)
```

No soft mask. No router evaluation. Just two small MLP forward passes summed. The mask, temperature, and other training machinery are completely absent at inference.

### 5.3 Skill Switching

When the operator changes the skill command (e.g., trot -> pace):

```python
# Just copy weights into the pre-allocated slot -- <1ms on Orin
task_mlp_slot.load_state_dict(task_bank['pace'])
```

No model reinitialization, no graph rebuilding, no recompilation. The architecture is identical; only the numbers in the task slot change. Each task MLP is ~780KB in FP16 across 6 layers -- copying this from CPU to GPU takes microseconds.

### 5.4 Skill Transitions (Optional)

For smooth gait transitions, briefly run both outgoing and incoming task MLPs:
```
task_out = alpha * Task_MLP_old(z) + (1-alpha) * Task_MLP_new(z)
```
Or use weight-space interpolation (same compute as single task MLP):
```
W_slot = alpha * W_old + (1-alpha) * W_new
```
In practice, start with hard switching. The diffusion model's stochastic denoising should smooth over transitions naturally.

---

## 6. Related Work Positioning

| Method | Domain | MLP Treatment | Skill Decomposition | Key Gap |
|--------|--------|--------------|---------------------|---------|
| E-DiT (2026) | Image/3D | Width masking | None (input difficulty) | Full weights in memory, not skill-conditioned |
| DyDiT (ICLR 25) | Image | Dynamic masks | None (timestep) | No skill structure, image-only |
| MoDE (2024) | Manipulation | Full experts | Noise-level routing | Full-sized experts, router overhead, manipulation not locomotion |
| MoE-Loco (IROS 25) | Locomotion (RL) | Full MoE experts | Expert specialization | RL not diffusion, not compute-motivated |
| EC-DiT (ICLR 25) | Image | Full experts | Expert-choice routing | Scaling focus, not inference efficiency |
| **SkillFFN (Ours)** | **Locomotion (Diffusion)** | **Structural decomposition** | **Skill-conditioned soft masking** | -- |

**Novelty claim:** No existing work combines structural FFN decomposition + skill-conditioned elastic training + diffusion locomotion + edge deployment.

**Key differentiators:**
1. **Structural, not masking** -- physically smaller forward pass, saves FLOPs AND memory bandwidth
2. **Train soft, deploy hard** -- elastic masking during training for clean skill separation, deterministic routing at inference for zero overhead
3. **Skill-conditioned, not input-conditioned** -- exploits known discrete task structure rather than learning per-sample difficulty routing
4. **Physical AI domain** -- targeting real-time quadruped locomotion on edge hardware, not image/3D generation

---

## 7. Implementation Plan

### Phase 1: Architecture
1. Locate FFN in DiffuseLoco codebase
2. Implement `SkillFFN` module as drop-in FFN replacement
3. Implement soft mask construction with temperature parameter
4. Implement weight-swap inference mode for deployment

```python
class SkillFFN(nn.Module):
    def __init__(self, d_model=256, base_dim=256, task_dim=256,
                 skills=['trot','pace','hop','bounce','bipedal']):
        super().__init__()
        self.base_mlp = MLP(d_model, base_dim, d_model)
        self.task_mlps = nn.ModuleDict({
            s: MLP(d_model, task_dim, d_model) for s in skills
        })
        self.skills = skills

    def forward(self, z, skill_label, tau=None):
        base_out = self.base_mlp(z)

        if tau is not None and tau > 0:
            # Training: soft-masked elastic forward
            logits = torch.zeros(len(self.skills), device=z.device)
            logits[self.skills.index(skill_label)] = 1.0 / tau
            mask = F.softmax(logits, dim=0)

            task_out = sum(
                mask[i] * self.task_mlps[s](z)
                for i, s in enumerate(self.skills)
            )
        else:
            # Inference: deterministic hard routing
            task_out = self.task_mlps[skill_label](z)

        return base_out + task_out
```

### Phase 2: Training
1. Initialize Base_MLP from pretrained FFN via truncated SVD
2. Initialize Task_MLPs with small random noise
3. Train with soft masking + temperature annealing (tau: 1.0 -> 0.05)
4. Stratified batch sampling across all 5 skills
5. Standard diffusion loss, no auxiliary losses

### Phase 3: Ablation Studies

| Axis | Values |
|------|--------|
| Base MLP hidden dim | 128, 256, 384, 512 |
| Task MLP hidden dim | 64, 128, 256 |
| Temperature schedule | linear, exponential, cosine annealing |
| Initial tau | 0.5, 1.0, 2.0 |
| Final tau | 0.01, 0.05, 0.1 |
| Which layers get task MLPs | all 6, last 4, last 2, alternating |
| Initialization | SVD, distillation, random |
| Training: soft mask vs hard routing | compare final quality |

### Phase 4: Baselines

| Baseline | Description |
|----------|-------------|
| DiffuseLoco (original) | Full 4x FFN -- upper bound on quality |
| Uniform reduction | Single MLP with reduced width (2x, 1x) |
| E-DiT-style masking | Neuron masking within full 4x FFN |
| Full MoE | K full-width experts + learned router |
| SkillFFN (hard training) | Same architecture, but hard routing during training |
| **SkillFFN (elastic training)** | **Full method: soft mask training + hard deploy** |

The comparison between SkillFFN with hard vs elastic training directly validates the soft masking contribution.

### Phase 5: Deployment & Real-Robot Evaluation
1. Latency benchmarking on Jetson Orin (ms/step)
2. Memory profiling (GPU footprint, bandwidth utilization)
3. Skill switching latency measurement
4. Real-robot locomotion quality across all 5 gaits
5. Gait transition smoothness evaluation

---

## 8. Open Questions

1. **Optimal base vs task dim ratio:** Is 256/256 the right split? The ablation will answer this.

2. **Per-layer adapter granularity:** Are early layers more skill-agnostic? If so, early layers might not need task MLPs at all.

3. **Temperature annealing schedule:** Linear vs exponential vs cosine? How sensitive is the result to tau_start and tau_end?

4. **Soft mask vs hard routing impact:** How much does elastic training actually help over simple hard routing? This is the key ablation for justifying the training approach.

5. **Skill transition handling:** Is hard switching sufficient, or do we need weight interpolation?

6. **Combination with denoising acceleration:** SkillFFN reduces per-step cost. This is orthogonal to reducing NFE (flow matching, consistency distillation). Can both combine for multiplicative speedup?

---

## 9. Expected Contributions

1. **SkillFFN architecture** -- Structurally decomposed FFN for multi-skill diffusion locomotion: shared Base_MLP + hot-swappable Task_MLPs with deterministic skill-conditioned routing at inference
2. **Elastic training procedure** -- Soft skill-conditioned masking during training with temperature annealing for clean shared/specific separation, hardened to deterministic routing at deployment
3. **Compute and memory efficiency** -- 2-3x FFN compute reduction with reduced memory footprint on Jetson Orin
4. **Preserved multimodality** -- Dedicated task capacity per skill prevents multimodal collapse
5. **Real-robot validation** -- Deployment on Unitree Go2 across 5 locomotion skills

---

## 10. Key References

- [1] MoDE: Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers (2024)
- [2] MoE-Loco: Mixture of Experts for Multitask Locomotion (IROS 2025)
- [3] DyDiT: Dynamic Diffusion Transformer (ICLR 2025)
- [4] E-DiT: Elastic Diffusion Transformer (arXiv 2602.13993, 2026)
- [5] DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion (CoRL 2024)
- [6] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023)
- [10] LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
- [12] EC-DiT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing (ICLR 2025)
- [18] VersatileFFN: Adaptive Wide-and-Deep Reuse (2026)
- [20] Low-rank FFN training (ICML 2024 Workshop)