"""
Standalone comparison of diffusion-transformer policies (baseline vs multiskill).

Self-contained: all model code and configs are inlined. No external repo
or yaml files required. Only depends on: torch, diffusers, (optional) fvcore.

Three variants are measured and compared:
  1. baseline_dp                  — standard FFN, dim_feedforward = 4 * n_emb = 1536
  2. multiskill_dp (task=384)     — SkillFFN with base=384, task=384 (5 skills)
  3. multiskill_dp (task=128)     — SkillFFN with base=384, task=128 (5 skills)

Metrics per model:
  * Parameter counts (total / trainable-without-depth / decoder-FFN-only)
  * FLOPs per single transformer forward and per full 10-step DDPM loop
  * Latency per single forward and per full 10-step DDPM loop (CUDA events or perf_counter)

Usage:  python testing/compare_dp_models.py
"""

import gc
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Config dicts (copied from original yamls, every key inlined)
# ============================================================================
BASELINE_CFG = dict(
    input_dim=12,
    output_dim=12,
    horizon=12,
    n_obs_steps=8,
    cond_dim=45,
    depth_latent_dim=32,
    num_tasks=5,
    n_layer=8,
    n_head=8,
    n_emb=384,
    p_drop_emb=0.0,
    p_drop_attn=0.3,
    causal_attn=True,
    n_cond_layers=0,
    use_recurrent_depth=False,
    separate_goal_conditioning=True,
)

# Multiskill adds 3 extra keys for the SkillFFN decoder.
MULTISKILL_CFG = dict(**BASELINE_CFG, base_dim=384, task_dim=384, num_skills=5)

NOISE_SCHEDULER_CFG = dict(
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    beta_start=0.0001,
    clip_sample=True,
    num_train_timesteps=10,
    prediction_type="epsilon",
    variance_type="fixed_small",
)

NUM_INFERENCE_STEPS = 10
BATCH = 1


# ============================================================================
# Inlined building blocks
# ============================================================================
class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DepthOnlyFCBackbone58x87(nn.Module):
    """CNN backbone: 58x87 depth image -> scandots_output_dim latent."""
    def __init__(self, scandots_output_dim=32, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(62400, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        return self.output_activation(images_compressed)


class SkillFFN(nn.Module):
    """Dual-pathway FFN: shared base_mlp + per-skill task_mlps."""
    def __init__(self, d_model, base_dim, task_dim, num_skills=5, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.base_mlp = nn.Sequential(
            nn.Linear(d_model, base_dim), act, nn.Linear(base_dim, d_model))
        self.task_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, task_dim), act, nn.Linear(task_dim, d_model))
            for _ in range(num_skills)])
        self.num_skills = num_skills

    def forward(self, z, skill_label, tau=None):
        base_out = self.base_mlp(z)
        if tau is not None and tau > 0:
            task_outs = torch.stack([mlp(z) for mlp in self.task_mlps], dim=0)
            logits = torch.zeros(z.shape[0], self.num_skills,
                                 device=z.device, dtype=z.dtype)
            logits.scatter_(1, skill_label.unsqueeze(1), 1.0 / tau)
            mask = F.softmax(logits, dim=1)
            mask = mask.t().unsqueeze(-1).unsqueeze(-1)
            task_out = (mask * task_outs).sum(dim=0)
        else:
            # Hard routing (inference): only the active task_mlp runs per sample.
            unique_skills = skill_label.unique()
            task_out = torch.zeros_like(base_out)
            for k in unique_skills:
                mask_k = (skill_label == k)
                task_out[mask_k] = self.task_mlps[k.item()](z[mask_k])
        return base_out + task_out


class SkillFFNDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, base_dim, task_dim, num_skills=5,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.skill_ffn = SkillFFN(d_model, base_dim, task_dim, num_skills, activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, skill_label, tau=None,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.multihead_attn(x2, memory, memory, attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask)[0]
        x = x + self.dropout2(x2)
        x2 = self.norm3(x)
        x2 = self.skill_ffn(x2, skill_label=skill_label, tau=tau)
        x = x + self.dropout3(x2)
        return x


# ============================================================================
# Inlined baseline transformer (mirrors TransformerForDiffusionWithDepth)
# ============================================================================
class TransformerForDiffusionWithDepth(ModuleAttrMixin):
    def __init__(self, input_dim, output_dim, horizon, n_obs_steps=8, cond_dim=45,
                 depth_latent_dim=32, num_tasks=5, n_layer=6, n_head=8, n_emb=256,
                 p_drop_emb=0.0, p_drop_attn=0.1, causal_attn=True, n_cond_layers=0,
                 use_recurrent_depth=False, separate_goal_conditioning=True):
        super().__init__()
        self.separate_goal_conditioning = separate_goal_conditioning
        T = horizon
        To = n_obs_steps
        T_cond = 2 + To * 3  # time(1) + task(1) + obs(To) + depth(To) + goal(To)

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.task_emb = nn.Embedding(num_tasks, n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim - 3, n_emb)
        self.cond_depth_emb = nn.Linear(depth_latent_dim, n_emb)
        self.cond_obs_emb_2 = nn.Linear(3, n_emb)

        self.use_recurrent_depth = use_recurrent_depth
        assert not use_recurrent_depth, "Only non-recurrent depth backbone is inlined."
        self.depth_backbone = DepthOnlyFCBackbone58x87(scandots_output_dim=depth_latent_dim)

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                dropout=p_drop_attn, activation="gelu",
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb), nn.Mish(),
                nn.Linear(4 * n_emb, n_emb))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
            dropout=p_drop_attn, activation="gelu",
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            self.register_buffer("mask", mask)

            S = To + 2
            t_idx, s_idx = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
            base = t_idx >= (s_idx - 2)
            memory_mask = torch.zeros((T, T_cond), dtype=torch.bool)
            memory_mask[:, :S] = base
            memory_mask[:, S:S + To] = base[:, 2:]
            memory_mask[:, S + To:S + 2 * To] = base[:, 2:]
            memory_mask = memory_mask.float().masked_fill(
                memory_mask == 0, float("-inf")).masked_fill(memory_mask == 1, 0.0)
            self.register_buffer("memory_mask", memory_mask)
        else:
            self.mask = None
            self.memory_mask = None

        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps

        # Freeze depth backbone (mirror original behavior — no pretrained load here).
        for p in self.depth_backbone.parameters():
            p.requires_grad = False
        self.depth_backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.depth_backbone.eval()
        return self

    def forward(self, sample, timestep, cond=None, depth=None, task_id=None, **kwargs):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_token = self.time_emb(timesteps).unsqueeze(1)

        B, To = depth.shape[0], depth.shape[1]
        depth_flat = depth.reshape(B * To, depth.shape[2], depth.shape[3])
        with torch.no_grad():
            depth_latent = self.depth_backbone(depth_flat)
        depth_latent = depth_latent.reshape(B, To, -1)

        obs_no_goal = torch.cat([cond[..., :6], cond[..., 9:]], dim=-1)
        cond_obs_emb = self.cond_obs_emb(obs_no_goal)
        cond_depth_emb = self.cond_depth_emb(depth_latent)
        cond_goal_emb = self.cond_obs_emb_2(cond[..., 6:9])
        task_token = self.task_emb(task_id).unsqueeze(1)

        cond_embeddings = torch.cat(
            [time_token, task_token, cond_obs_emb, cond_depth_emb, cond_goal_emb], dim=1)
        tc = cond_embeddings.shape[1]
        x = self.drop(cond_embeddings + self.cond_pos_emb[:, :tc, :])
        memory = self.encoder(x)

        t = sample.shape[1]
        token_embeddings = self.input_emb(sample)
        x = self.drop(token_embeddings + self.pos_emb[:, :t, :])
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        return self.head(x)


class TransformerForDiffusionWithDepthSkillFFN(TransformerForDiffusionWithDepth):
    def __init__(self, base_dim=384, task_dim=384, num_skills=5, **kwargs):
        super().__init__(**kwargs)
        n_emb = kwargs["n_emb"]
        n_head = kwargs["n_head"]
        n_layer = kwargs["n_layer"]
        p_drop_attn = kwargs.get("p_drop_attn", 0.1)
        self.decoder = nn.ModuleList([
            SkillFFNDecoderLayer(d_model=n_emb, nhead=n_head,
                                 base_dim=base_dim, task_dim=task_dim,
                                 num_skills=num_skills, dropout=p_drop_attn,
                                 activation="gelu")
            for _ in range(n_layer)])
        self.num_skills = num_skills
        self.base_dim = base_dim
        self.task_dim = task_dim

    def forward(self, sample, timestep, cond=None, depth=None, task_id=None,
                tau=None, **kwargs):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_token = self.time_emb(timesteps).unsqueeze(1)

        B, To = depth.shape[0], depth.shape[1]
        depth_flat = depth.reshape(B * To, depth.shape[2], depth.shape[3])
        with torch.no_grad():
            depth_latent = self.depth_backbone(depth_flat)
        depth_latent = depth_latent.reshape(B, To, -1)

        obs_no_goal = torch.cat([cond[..., :6], cond[..., 9:]], dim=-1)
        cond_obs_emb = self.cond_obs_emb(obs_no_goal)
        cond_depth_emb = self.cond_depth_emb(depth_latent)
        cond_goal_emb = self.cond_obs_emb_2(cond[..., 6:9])
        task_token = self.task_emb(task_id).unsqueeze(1)

        cond_embeddings = torch.cat(
            [time_token, cond_obs_emb, cond_depth_emb, cond_goal_emb, task_token], dim=1)
        tc = cond_embeddings.shape[1]
        x = self.drop(cond_embeddings + self.cond_pos_emb[:, :tc, :])
        memory = self.encoder(x)

        t = sample.shape[1]
        token_embeddings = self.input_emb(sample)
        x = self.drop(token_embeddings + self.pos_emb[:, :t, :])
        for layer in self.decoder:
            x = layer(x, memory, skill_label=task_id, tau=tau,
                      tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        return self.head(x)


# ============================================================================
# Measurement utilities
# ============================================================================
def make_dummy_inputs(device, dtype=torch.float32):
    cfg = BASELINE_CFG
    return dict(
        sample=torch.randn(BATCH, cfg["horizon"], cfg["input_dim"], device=device, dtype=dtype),
        timestep=torch.zeros(BATCH, dtype=torch.long, device=device),
        cond=torch.randn(BATCH, cfg["n_obs_steps"], cfg["cond_dim"], device=device, dtype=dtype),
        depth=torch.randn(BATCH, cfg["n_obs_steps"], 58, 87, device=device, dtype=dtype),
        task_id=torch.zeros(BATCH, dtype=torch.long, device=device),
    )


class BaselineForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, cond, depth, task_id):
        return self.model(sample=sample, timestep=timestep, cond=cond,
                          depth=depth, task_id=task_id)


class MultiskillForward(nn.Module):
    """tau=None → hard routing at inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, cond, depth, task_id):
        return self.model(sample=sample, timestep=timestep, cond=cond,
                          depth=depth, task_id=task_id, tau=None)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable_no_depth = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("depth_backbone."))
    return total, trainable_no_depth


def count_decoder_ffn_params(model, kind):
    n = 0
    if kind == "baseline":
        for layer in model.decoder.layers:
            n += sum(p.numel() for p in layer.linear1.parameters())
            n += sum(p.numel() for p in layer.linear2.parameters())
    else:
        for layer in model.decoder:
            n += sum(p.numel() for p in layer.skill_ffn.parameters())
    return n


def measure_flops(forward_module, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        t_in = (inputs["sample"], inputs["timestep"], inputs["cond"],
                inputs["depth"], inputs["task_id"])
        fca = FlopCountAnalysis(forward_module, t_in)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total()), "fvcore"
    except ImportError:
        pass
    try:
        from thop import profile
        t_in = (inputs["sample"], inputs["timestep"], inputs["cond"],
                inputs["depth"], inputs["task_id"])
        macs, _ = profile(forward_module, inputs=t_in, verbose=False)
        return int(macs * 2), "thop (MACs*2)"
    except ImportError:
        return None, "none"


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_single_forward(forward_module, inputs, device, warmup=10, iters=50):
    forward_module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward_module(**inputs)
        _cuda_sync(device)
        times_ms = []
        for _ in range(iters):
            _cuda_sync(device)
            t0 = time.perf_counter()
            forward_module(**inputs)
            _cuda_sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times_ms), (statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0)


def time_ddpm_loop(forward_module, inputs, scheduler, device, warmup=5, iters=20):
    forward_module.eval()
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    sample_shape = inputs["sample"].shape
    sample_dtype = inputs["sample"].dtype

    def run_loop():
        trajectory = torch.randn(sample_shape, dtype=sample_dtype, device=device)
        for t in scheduler.timesteps:
            out = forward_module(
                sample=trajectory, timestep=t,
                cond=inputs["cond"], depth=inputs["depth"], task_id=inputs["task_id"])
            trajectory = scheduler.step(out, t, trajectory).prev_sample
        return trajectory

    with torch.no_grad():
        for _ in range(warmup):
            run_loop()
        _cuda_sync(device)
        times_ms = []
        for _ in range(iters):
            _cuda_sync(device)
            t0 = time.perf_counter()
            run_loop()
            _cuda_sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times_ms), (statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0)


# ============================================================================
# Per-model pipeline
# ============================================================================
def measure_variant(label, model, kind, device):
    print(f"\n{'=' * 72}")
    print(f"  Measuring {label}")
    print(f"{'=' * 72}")

    wrapper_cls = BaselineForward if kind == "baseline" else MultiskillForward
    wrapped = wrapper_cls(model).to(device).eval()

    inputs = make_dummy_inputs(device)
    with torch.no_grad():
        y = wrapped(**inputs)
    assert y.shape == (BATCH, BASELINE_CFG["horizon"], BASELINE_CFG["input_dim"])

    total, trainable_no_depth = count_params(model)
    ffn_params = count_decoder_ffn_params(model, kind)
    flops_single, backend = measure_flops(wrapped, inputs)
    flops_loop = flops_single * NUM_INFERENCE_STEPS if flops_single is not None else None

    single_mean, single_std = time_single_forward(wrapped, inputs, device)

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler = DDPMScheduler(**NOISE_SCHEDULER_CFG)
    loop_mean, loop_std = time_ddpm_loop(wrapped, inputs, scheduler, device)
    throughput_hz = 1000.0 / loop_mean if loop_mean > 0 else float("nan")

    del wrapped, model, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dict(
        label=label,
        total_params=total,
        trainable_no_depth=trainable_no_depth,
        ffn_params=ffn_params,
        flops_single=flops_single,
        flops_loop=flops_loop,
        flops_backend=backend,
        single_mean_ms=single_mean,
        single_std_ms=single_std,
        loop_mean_ms=loop_mean,
        loop_std_ms=loop_std,
        throughput_hz=throughput_hz,
    )


# ============================================================================
# Pretty printing
# ============================================================================
def fmt_params(n):
    return f"{n / 1e6:.3f} M"


def fmt_flops(n):
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    return f"{n}"


def fmt_ratio(new, base):
    if base is None or new is None or base == 0:
        return "N/A"
    delta = (new - base) / base * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def print_comparison(results):
    print(f"\n{'=' * 100}")
    print("  SUMMARY (ratio vs baseline_dp)")
    print(f"{'=' * 100}")

    base = results[0]

    def ratios(key):
        return [""] + [fmt_ratio(r[key], base[key]) for r in results[1:]]

    rows = [
        ("Total params",
         [fmt_params(r["total_params"]) for r in results], ratios("total_params")),
        ("Trainable (no depth)",
         [fmt_params(r["trainable_no_depth"]) for r in results], ratios("trainable_no_depth")),
        ("Decoder FFN params",
         [fmt_params(r["ffn_params"]) for r in results], ratios("ffn_params")),
        ("FLOPs / single fwd",
         [fmt_flops(r["flops_single"]) for r in results], ratios("flops_single")),
        ("FLOPs / 10-step loop",
         [fmt_flops(r["flops_loop"]) for r in results], ratios("flops_loop")),
        ("Latency single fwd (ms)",
         [f"{r['single_mean_ms']:.2f} ± {r['single_std_ms']:.2f}" for r in results],
         ratios("single_mean_ms")),
        ("Latency 10-step loop (ms)",
         [f"{r['loop_mean_ms']:.2f} ± {r['loop_std_ms']:.2f}" for r in results],
         ratios("loop_mean_ms")),
        ("Throughput (Hz, full loop)",
         [f"{r['throughput_hz']:.1f}" for r in results], ratios("throughput_hz")),
    ]

    col_w = 22
    header = f"{'':30s}"
    for r in results:
        header += f" {r['label']:>{col_w}s}"
    print(header)
    print("-" * len(header))
    for name, cells, rs in rows:
        line = f"{name:30s}"
        for c, r in zip(cells, rs):
            line += f" {c:>{col_w}s}"
            if r:
                line = line[:-2] + f" ({r})"
                line = f"{line:<{len(header)}s}"
        print(line)
    print()
    print("  FLOP backend: " + "  |  ".join(
        f"{r['label']}: {r['flops_backend']}" for r in results))


# ============================================================================
# Main
# ============================================================================
def main():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu (CUDA not available)")

    # 1. Baseline
    baseline = TransformerForDiffusionWithDepth(**BASELINE_CFG).to(device).eval()
    r_base = measure_variant("baseline_dp", baseline, "baseline", device)

    # 2. Multiskill, task_dim = 384
    mcfg_384 = dict(MULTISKILL_CFG)
    msk_384 = TransformerForDiffusionWithDepthSkillFFN(**mcfg_384).to(device).eval()
    r_msk_384 = measure_variant("multiskill (t=384)", msk_384, "multiskill", device)

    # 3. Multiskill, task_dim = 128
    mcfg_128 = dict(MULTISKILL_CFG, task_dim=128)
    msk_128 = TransformerForDiffusionWithDepthSkillFFN(**mcfg_128).to(device).eval()
    r_msk_128 = measure_variant("multiskill (t=128)", msk_128, "multiskill", device)

    print_comparison([r_base, r_msk_384, r_msk_128])

    # Summary of FFN deltas
    print("\n  Active FFN hidden (per decoder layer, at inference):")
    print(f"    baseline_dp        : 4 × 384 = 1536")
    print(f"    multiskill (t=384) : base(384) + task(384) =  768   (half of baseline)")
    print(f"    multiskill (t=128) : base(384) + task(128) =  512   (~1/3 of baseline)")


if __name__ == "__main__":
    main()
