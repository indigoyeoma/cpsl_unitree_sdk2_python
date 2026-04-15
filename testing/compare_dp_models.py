"""
Compare baseline_dp vs multiskill_dp diffusion-transformer policies.

Both models share every architectural knob except the decoder FFN:
  - baseline_dp : standard TransformerDecoderLayer with dim_feedforward = 4 * n_emb (= 1536)
  - multiskill_dp : SkillFFNDecoderLayer with base_mlp(hidden=384) + 5x task_mlp(hidden=384)
                    (at inference, only 1 task_mlp is active via hard routing)

Measures, per model:
  * Parameter counts (total / trainable-without-depth / decoder-FFN-only)
  * FLOPs per single transformer forward + per full 10-step DDPM loop (fvcore)
  * Latency per single transformer forward + per full 10-step DDPM loop (CUDA events)

Target: Jetson Orin NX onboard the Go2 EDU (falls back to CPU if CUDA unavailable).

Usage:
  python testing/compare_dp_models.py
  python testing/compare_dp_models.py \\
      --baseline-root /path/to/baseline_dp \\
      --multiskill-root /path/to/multiskill_dp
"""

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


# ----------------------------------------------------------------------------
# Dimensions (must match both yaml configs — verified identical)
# ----------------------------------------------------------------------------
BATCH = 1
HORIZON = 12
N_OBS_STEPS = 8
INPUT_DIM = 12      # action_dim
COND_DIM = 45       # obs_dim
DEPTH_H, DEPTH_W = 58, 87
NUM_INFERENCE_STEPS = 10


# ----------------------------------------------------------------------------
# sys.path / sys.modules juggling — both repos ship a package named
# `diffusion_policy`, so we must purge between loads.
# ----------------------------------------------------------------------------
def _purge_diffusion_policy():
    for mod_name in list(sys.modules):
        if mod_name == "diffusion_policy" or mod_name.startswith("diffusion_policy."):
            del sys.modules[mod_name]


def _prepend_syspath(*paths):
    for p in paths:
        p = str(p)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _remove_syspath(*paths):
    for p in paths:
        p = str(p)
        while p in sys.path:
            sys.path.remove(p)


# ----------------------------------------------------------------------------
# Dummy input builder
# ----------------------------------------------------------------------------
def make_dummy_inputs(device, dtype=torch.float32):
    return dict(
        sample=torch.randn(BATCH, HORIZON, INPUT_DIM, device=device, dtype=dtype),
        timestep=torch.zeros(BATCH, dtype=torch.long, device=device),
        cond=torch.randn(BATCH, N_OBS_STEPS, COND_DIM, device=device, dtype=dtype),
        depth=torch.randn(BATCH, N_OBS_STEPS, DEPTH_H, DEPTH_W, device=device, dtype=dtype),
        task_id=torch.zeros(BATCH, dtype=torch.long, device=device),
    )


# ----------------------------------------------------------------------------
# Forward wrappers — fix non-tensor kwargs so FLOP counters and timing loops
# see a tensor-only signature.
# ----------------------------------------------------------------------------
class BaselineForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, cond, depth, task_id):
        return self.model(sample=sample, timestep=timestep, cond=cond,
                          depth=depth, task_id=task_id)


class MultiskillForward(nn.Module):
    """tau=None → hard routing at inference (single active task MLP)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, cond, depth, task_id):
        return self.model(sample=sample, timestep=timestep, cond=cond,
                          depth=depth, task_id=task_id, tau=None)


# ----------------------------------------------------------------------------
# Param counting
# ----------------------------------------------------------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable_no_depth = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("depth_backbone.")
    )
    return total, trainable_no_depth


def count_decoder_ffn_params(model, kind):
    """kind in {'baseline', 'multiskill'}"""
    n = 0
    if kind == "baseline":
        for layer in model.decoder.layers:
            n += sum(p.numel() for p in layer.linear1.parameters())
            n += sum(p.numel() for p in layer.linear2.parameters())
    else:
        for layer in model.decoder:
            n += sum(p.numel() for p in layer.skill_ffn.parameters())
    return n


# ----------------------------------------------------------------------------
# FLOPs (fvcore → thop → skip)
# ----------------------------------------------------------------------------
def measure_flops(forward_module, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        tensor_inputs = (inputs["sample"], inputs["timestep"], inputs["cond"],
                         inputs["depth"], inputs["task_id"])
        fca = FlopCountAnalysis(forward_module, tensor_inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total()), "fvcore"
    except ImportError:
        pass
    try:
        from thop import profile
        tensor_inputs = (inputs["sample"], inputs["timestep"], inputs["cond"],
                         inputs["depth"], inputs["task_id"])
        macs, _ = profile(forward_module, inputs=tensor_inputs, verbose=False)
        return int(macs * 2), "thop (MACs*2)"
    except ImportError:
        return None, "none"


# ----------------------------------------------------------------------------
# Latency — single forward and full 10-step DDPM loop
# ----------------------------------------------------------------------------
def _cuda_sync_if(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_single_forward(forward_module, inputs, device,
                        warmup=10, iters=50):
    forward_module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward_module(**inputs)
        _cuda_sync_if(device)

        times_ms = []
        for _ in range(iters):
            _cuda_sync_if(device)
            t0 = time.perf_counter()
            forward_module(**inputs)
            _cuda_sync_if(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times_ms), statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def time_ddpm_loop(forward_module, inputs, scheduler, device,
                   warmup=5, iters=20):
    """Full DDPM denoising loop: NUM_INFERENCE_STEPS forward passes + scheduler steps."""
    forward_module.eval()
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    sample_shape = inputs["sample"].shape
    sample_dtype = inputs["sample"].dtype

    def run_loop():
        trajectory = torch.randn(sample_shape, dtype=sample_dtype, device=device)
        for t in scheduler.timesteps:
            model_out = forward_module(
                sample=trajectory,
                timestep=t,
                cond=inputs["cond"],
                depth=inputs["depth"],
                task_id=inputs["task_id"],
            )
            trajectory = scheduler.step(model_out, t, trajectory).prev_sample
        return trajectory

    with torch.no_grad():
        for _ in range(warmup):
            run_loop()
        _cuda_sync_if(device)

        times_ms = []
        for _ in range(iters):
            _cuda_sync_if(device)
            t0 = time.perf_counter()
            run_loop()
            _cuda_sync_if(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times_ms), statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


# ----------------------------------------------------------------------------
# Per-model measurement pipeline
# ----------------------------------------------------------------------------
def measure_one(label, repo_root, yaml_rel, model_class_name, kind, device,
                cfg_overrides=None):
    print(f"\n{'=' * 72}")
    print(f"  Loading {label}  ({repo_root})")
    print(f"{'=' * 72}")

    diff_pkg_path = Path(repo_root) / "diffusion_policy"
    _purge_diffusion_policy()
    _prepend_syspath(diff_pkg_path)

    cfg = OmegaConf.load(Path(repo_root) / yaml_rel)
    model_cfg = OmegaConf.to_container(cfg.policy.model, resolve=True)
    model_cfg.pop("_target_", None)
    # Skip checkpoint load — only architecture matters for this comparison.
    model_cfg["pretrained_depth_backbone_path"] = None
    if cfg_overrides:
        model_cfg.update(cfg_overrides)

    import importlib
    tfd_module = importlib.import_module(
        "diffusion_policy.model.diffusion.transformer_for_diffusion")
    ModelCls = getattr(tfd_module, model_class_name)

    model = ModelCls(**model_cfg).to(device).eval()

    # FFN-wise config echo
    n_emb = model_cfg["n_emb"]
    if kind == "baseline":
        ffn_hidden = f"{4 * n_emb} (=4 * n_emb)"
    else:
        ffn_hidden = (f"base={model_cfg['base_dim']}, task={model_cfg['task_dim']} "
                      f"(x{model_cfg['num_skills']} skills)")
    print(f"  n_layer={model_cfg['n_layer']}  n_head={model_cfg['n_head']}  "
          f"n_emb={n_emb}  horizon={model_cfg['horizon']}  "
          f"n_obs_steps={model_cfg['n_obs_steps']}")
    print(f"  decoder FFN: {ffn_hidden}")

    # Wrap for FLOPs / timing
    if kind == "baseline":
        wrapped = BaselineForward(model).to(device).eval()
    else:
        wrapped = MultiskillForward(model).to(device).eval()

    # Dummy inputs
    inputs = make_dummy_inputs(device)

    # Sanity: forward shape
    with torch.no_grad():
        y = wrapped(**inputs)
    assert y.shape == (BATCH, HORIZON, INPUT_DIM), f"unexpected output shape {y.shape}"

    # Params
    total, trainable_no_depth = count_params(model)
    ffn_params = count_decoder_ffn_params(model, kind)

    # FLOPs
    flops_single, flops_backend = measure_flops(wrapped, inputs)
    flops_loop = flops_single * NUM_INFERENCE_STEPS if flops_single is not None else None

    # Latency — single forward
    single_mean_ms, single_std_ms = time_single_forward(wrapped, inputs, device)

    # Latency — full 10-step DDPM loop
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    ns_cfg = OmegaConf.to_container(cfg.policy.noise_scheduler, resolve=True)
    ns_cfg.pop("_target_", None)
    scheduler = DDPMScheduler(**ns_cfg)

    loop_mean_ms, loop_std_ms = time_ddpm_loop(wrapped, inputs, scheduler, device)
    throughput_hz = 1000.0 / loop_mean_ms if loop_mean_ms > 0 else float("nan")

    # Cleanup
    del wrapped, model, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _remove_syspath(diff_pkg_path)
    _purge_diffusion_policy()

    return dict(
        label=label,
        total_params=total,
        trainable_no_depth=trainable_no_depth,
        ffn_params=ffn_params,
        flops_single=flops_single,
        flops_loop=flops_loop,
        flops_backend=flops_backend,
        single_mean_ms=single_mean_ms,
        single_std_ms=single_std_ms,
        loop_mean_ms=loop_mean_ms,
        loop_std_ms=loop_std_ms,
        throughput_hz=throughput_hz,
    )


# ----------------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------------
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
    """results: list of measurement dicts. First entry is the baseline for ratios."""
    print(f"\n{'=' * 96}")
    print("  SUMMARY — all variants (ratio = delta vs. first column / baseline_dp)")
    print(f"{'=' * 96}")

    base = results[0]

    def row_vals(metric_key, fmt_fn, is_str=False, lo_is_good=True):
        cells = []
        for r in results:
            if is_str:
                cells.append(fmt_fn(r))
            else:
                cells.append(fmt_fn(r[metric_key]))
        return cells

    def ratio_cells(metric_key):
        cells = [""]  # baseline vs itself is trivial
        for r in results[1:]:
            cells.append(fmt_ratio(r[metric_key], base[metric_key]))
        return cells

    rows = []
    rows.append(("Total params",
                 row_vals("total_params", fmt_params),
                 ratio_cells("total_params")))
    rows.append(("Trainable (no depth)",
                 row_vals("trainable_no_depth", fmt_params),
                 ratio_cells("trainable_no_depth")))
    rows.append(("Decoder FFN params",
                 row_vals("ffn_params", fmt_params),
                 ratio_cells("ffn_params")))
    rows.append(("FLOPs / single fwd",
                 row_vals("flops_single", fmt_flops),
                 ratio_cells("flops_single")))
    rows.append(("FLOPs / 10-step loop",
                 row_vals("flops_loop", fmt_flops),
                 ratio_cells("flops_loop")))
    rows.append(("Latency single fwd (ms)",
                 [f"{r['single_mean_ms']:.2f} ± {r['single_std_ms']:.2f}" for r in results],
                 ratio_cells("single_mean_ms")))
    rows.append(("Latency 10-step loop (ms)",
                 [f"{r['loop_mean_ms']:.2f} ± {r['loop_std_ms']:.2f}" for r in results],
                 ratio_cells("loop_mean_ms")))
    rows.append(("Throughput (Hz, full loop)",
                 [f"{r['throughput_hz']:.1f}" for r in results],
                 ratio_cells("throughput_hz")))

    col_w = 20
    header = f"{'':30s}"
    for r in results:
        header += f" {r['label']:>{col_w}s}"
    header += f" {'ratio vs base':>16s}"
    print(header)
    print("-" * len(header))
    for name, cells, ratios in rows:
        line = f"{name:30s}"
        for c in cells:
            line += f" {c:>{col_w}s}"
        # compose ratio column(s) — join all non-empty ratios
        combined = " / ".join(x for x in ratios if x) or ""
        line += f" {combined:>16s}"
        print(line)
    print()
    print("  FLOP backend: " + "  |  ".join(
        f"{r['label']}: {r['flops_backend']}" for r in results))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root",
                    default="/home/jk_edge/research_otmamba/baseline_dp")
    ap.add_argument("--multiskill-root",
                    default="/home/jk_edge/research_otmamba/multiskill_dp")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu (CUDA not available)")

    torch.manual_seed(0)

    base = measure_one(
        label="baseline_dp",
        repo_root=args.baseline_root,
        yaml_rel="config_files/go2_difloco_depth_diffusion_policy.yaml",
        model_class_name="TransformerForDiffusionWithDepth",
        kind="baseline",
        device=device,
    )
    msk = measure_one(
        label="multiskill_dp (task=384)",
        repo_root=args.multiskill_root,
        yaml_rel="config_files/go2_skillffn_depth_policy.yaml",
        model_class_name="TransformerForDiffusionWithDepthSkillFFN",
        kind="multiskill",
        device=device,
    )
    msk_small = measure_one(
        label="multiskill_dp (task=128)",
        repo_root=args.multiskill_root,
        yaml_rel="config_files/go2_skillffn_depth_policy.yaml",
        model_class_name="TransformerForDiffusionWithDepthSkillFFN",
        kind="multiskill",
        device=device,
        cfg_overrides={"task_dim": 128},
    )

    print_comparison([base, msk, msk_small])


if __name__ == "__main__":
    main()
