import torch
import time
import numpy as np
import sys
import os

# Add local directory to path to find ppuda if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import itertools
from depth_encoder_ghn import DepthEncoderConfig, build_depth_backbone, estimate_parameters

def generate_all_configs():
    configs = []
    
    # Common settings
    pool_positions = [0, 1]
    fc_hidden = 128
    activation = 'elu'
    
    possible_channels = [32, 64]
    possible_kernels = [3, 5]
    
    # 1. 4 and 5 Layers (Full Diversity)
    for depth in [4, 5]:
        all_channel_combs = list(itertools.product(possible_channels, repeat=depth))
        all_kernel_combs = list(itertools.product(possible_kernels, repeat=depth))
        
        for ch in all_channel_combs:
            for k in all_kernel_combs:
                cfg = DepthEncoderConfig(
                    num_layers=depth,
                    channels=list(ch),
                    kernel_sizes=list(k),
                    strides=[1]*depth,
                    pool_positions=pool_positions,
                    fc_hidden=fc_hidden,
                    activation=activation
                )
                configs.append(cfg)

    # 2. 6 Layers (Full Diversity now allowed!)
    # As per user request: "if we use the 6 layres with kerenel 5 as well"
    depth = 6
    all_channel_combs = list(itertools.product(possible_channels, repeat=depth))
    all_kernel_combs = list(itertools.product(possible_kernels, repeat=depth))
    
    for ch in all_channel_combs:
        for k in all_kernel_combs:
            cfg = DepthEncoderConfig(
                num_layers=depth,
                channels=list(ch),
                kernel_sizes=list(k),
                strides=[1]*depth,
                pool_positions=pool_positions,
                fc_hidden=fc_hidden,
                activation=activation
            )
            configs.append(cfg)
            
    print(f"Generated {len(configs)} total unique architectures.")
    return configs

def measure_latency(model, device='cuda', input_shape=(128, 96), repeats=50):
    dummy_input = torch.randn(1, *input_shape).to(device)
    model = model.to(device)
    model.eval()
    
    # Warmup
    try:
        for _ in range(10):
            _ = model(dummy_input)
    except Exception as e:
        print(f"Error during warmup: {e}")
        return None
        
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)
            
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / repeats) * 1000
    return avg_latency_ms

def main():
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Profiling GHN Architectures on device: {device}")
    
    # Exhaustive generation
    configs = generate_all_configs()
    n_samples = len(configs)
    
    latencies = []
    params = []
    
    print("Starting profiling loop...")
    for i, cfg in enumerate(configs):
        model = build_depth_backbone(cfg)
        lat = measure_latency(model, device)
        
        if lat is not None:
            par = estimate_parameters(cfg)
            latencies.append(lat)
            params.append(par)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{n_samples}...")

    latencies = np.array(latencies)
    params = np.array(params)
    
    print("\n" + "="*40)
    print("       GHN LATENCY PROFILE (ON-DEVICE)       ")
    print("="*40)
    print(f"Device: {device}")
    print(f"Total Unique Architectures: {len(latencies)}")
    print("-" * 40)
    print(f"Latency (ms):")
    if len(latencies) > 0:
        print(f"  Min:    {latencies.min():.4f} ms")
        print(f"  Mean:   {latencies.mean():.4f} ms")
        print(f"  Median: {np.median(latencies):.4f} ms")
        print(f"  Max:    {latencies.max():.4f} ms")
    print("-" * 40)
    print(f"Parameters (M):")
    if len(params) > 0:
        print(f"  Min:    {params.min()/1e6:.2f} M")
        print(f"  Max:    {params.max()/1e6:.2f} M")
    print("="*40)

    # Plotting (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        print("Generating Latency vs Params plot...")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(params/1e6, latencies, alpha=0.5, c='blue', s=2)
        plt.title(f'GHN Latency vs Parameters ({device}) - All {n_samples} Combos')
        plt.xlabel('Parameters (Millions)')
        plt.ylabel('Latency (ms)')
        plt.grid(True)
        
        # Add timestamp to filename to avoid overwrites
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'ghn_latency_vs_params_{device}_{timestamp}.png'
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        
    except ImportError:
        print("matplotlib not found. Skipping plot generation.")
        print("Install matplotlib to generate visualization: pip install matplotlib")

if __name__ == '__main__':
    main()
