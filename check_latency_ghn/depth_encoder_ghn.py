# Depth Encoder Architecture with Trainable GHN
#
# This module provides tools to:
# 1. Sample diverse simple CNN architectures for depth encoding
# 2. Build depth encoder networks from configs
# 3. Train a GHN to predict good initial weights for depth encoders
#
# Two modes of operation:
# A) GHN Training Mode: Train a GHN that learns to predict weights
#    - GHN is updated via backprop through weight prediction
#    - After training, GHN can initialize any new architecture instantly
#
# B) Direct Training Mode: Train encoders directly without GHN
#    - Multiple architectures trained in parallel
#    - Each architecture's weights are directly updated
#
# Usage:
#     from rsl_rl.modules.depth_encoder_ghn import (
#         sample_depth_encoder_config,
#         build_depth_backbone,
#         TrainableGHN,
#     )
#
#     # Create trainable GHN
#     ghn = TrainableGHN(device='cuda')
#
#     # Sample architectures and predict weights
#     configs = [sample_depth_encoder_config() for _ in range(8)]
#     backbones = [build_depth_backbone(cfg) for cfg in configs]
#     backbones = ghn.predict_weights(backbones)  # GHN predicts weights
#
#     # Forward pass through backbones, compute loss, backprop
#     # Gradients flow through backbone weights to GHN

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random


@dataclass
class DepthEncoderConfig:
    """Configuration for a simple CNN depth encoder backbone.

    The backbone processes depth images and outputs a latent vector.
    Architecture: Conv → BN → Act → Pool → Conv → ... → Flatten → FC → output_dim
    """
    # Input (128x96 - compromise between memory and architecture flexibility)
    # Larger than 48x64 to avoid 1x1 spatial, smaller than 160x120 to save memory
    input_height: int = 96
    input_width: int = 128
    input_channels: int = 1
    output_dim: int = 128

    # Architecture
    num_layers: int = 2
    channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1])

    # Pooling (applied after each conv)
    pool_type: str = 'max'  # 'max', 'avg', or 'none'
    pool_size: int = 2
    pool_positions: List[int] = field(default_factory=lambda: [0])  # Which layers to pool after

    # Activation
    activation: str = 'elu'  # 'elu', 'relu', 'gelu'

    # FC layers
    fc_hidden: int = 128  # Hidden dim before output

    def __post_init__(self):
        """Validate config."""
        assert len(self.channels) == self.num_layers
        assert len(self.kernel_sizes) == self.num_layers
        assert len(self.strides) == self.num_layers

    def to_dict(self):
        return {
            'num_layers': self.num_layers,
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
            'pool_positions': self.pool_positions,
            'fc_hidden': self.fc_hidden,
        }

    def __repr__(self):
        return (f"DepthEncoderConfig(layers={self.num_layers}, "
                f"ch={self.channels}, k={self.kernel_sizes}, s={self.strides}, "
                f"pool@{self.pool_positions})")





# Preset configurations
# Preset configurations

def sample_depth_encoder_config(
    num_layers_range: Tuple[int, int] = (3, 5),
    channel_options: List[int] = None,
    kernel_options: List[int] = None,
    stride_options: List[int] = None,
    seed: int = None,
) -> DepthEncoderConfig:
    """Sample a depth encoder config guaranteed to be < 10M parameters.
    
    Strategy: Constrain max channel width based on depth (pooling factor) to
    ensure the fully connected layer doesn't explode.
    
    Args:
        num_layers_range: (min, max) number of conv layers (3, 4, or 5)
        channel_options: Ignored (overridden by safe constructive logic)
        kernel_options: Choices for kernel sizes (default: [3, 5, 7])
        stride_options: Ignored (always 1, pooling handles downsampling)
        seed: Random seed for reproducibility
    
    Returns:
        DepthEncoderConfig with valid architecture < 10M params
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    kernel_options = kernel_options or [3, 5, 7]
    
    # 1. Variable Depth (4, 5, or 6 Layers)
    # Deeper networks = more sequential compute = higher latency.
    num_layers = random.choice([4, 5, 6])
    
    # 2. Reduced Pooling for Higher Resolution (32x24)
    # We pool early to get to 32x24, then stay there for remaining layers.
    pool_positions = [0, 1]
    
    # 3. Diverse Channels (Heavy Compute Allowed)
    # To reach ~1ms latency, we need wider layers on high-resolution maps.
    possible_channels = [32, 64, 128]
    channels = [random.choice(possible_channels) for _ in range(num_layers)]

    # 4. Diverse Kernels
    kernel_sizes = [random.choice([3, 5, 7]) for _ in range(num_layers)]
    
    # 5. Strides (Always 1)
    strides = [1] * num_layers
    
    # 6. FC Hidden (Fixed)
    fc_hidden = 128
    
    config = DepthEncoderConfig(
        num_layers=num_layers,
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pool_positions=pool_positions,
        fc_hidden=fc_hidden,
        activation='elu'
    )
    return config


def estimate_parameters(config: DepthEncoderConfig) -> int:
    """Estimate number of parameters for a given config."""
    # 1. Conv Layers
    params = 0
    in_ch = config.input_channels
    for i in range(config.num_layers):
        out_ch = config.channels[i]
        k = config.kernel_sizes[i]
        # Conv2d weights: out * in * k * k
        params += out_ch * in_ch * k * k
        # Bias
        params += out_ch
        # BatchNorm (scale + shift)
        params += 2 * out_ch
        in_ch = out_ch

    # 2. FC Layers
    # Compute flattened size
    h, w = compute_output_size(config.input_height, config.input_width, config)
    flat_size = config.channels[-1] * h * w
    
    # Linear 1
    params += flat_size * config.fc_hidden + config.fc_hidden
    # Linear 2
    params += config.fc_hidden * config.output_dim + config.output_dim
    
    return params


def sample_depth_encoder_configs(n: int, unique: bool = True, **kwargs) -> List[DepthEncoderConfig]:
    """Sample multiple depth encoder configurations.

    Args:
        n: Number of configs to sample
        unique: If True, try to ensure configs are different (best effort)
        **kwargs: Passed to sample_depth_encoder_config

    Returns:
        List of DepthEncoderConfig
    """
    configs = []
    seen = set()
    max_attempts = n * 50
    attempts = 0

    while len(configs) < n and attempts < max_attempts:
        cfg = sample_depth_encoder_config(**kwargs)
        key = str(cfg.to_dict()) if unique else None

        if not unique or key not in seen:
            configs.append(cfg)
            if unique:
                seen.add(key)
        attempts += 1

    return configs


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        'elu': nn.ELU(),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.1),
    }
    return activations.get(name.lower(), nn.ELU())


def compute_output_size(h: int, w: int, config: DepthEncoderConfig) -> Tuple[int, int]:
    """Compute the spatial size after all conv/pool layers."""
    for i in range(config.num_layers):
        k = config.kernel_sizes[i]
        s = config.strides[i]
        p = k // 2  # Same padding

        # Conv
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1

        # Pool
        if i in config.pool_positions:
            h = h // config.pool_size
            w = w // config.pool_size

    return h, w


class DepthBackboneWrapper(nn.Module):
    """Wrapper that adds unsqueeze to match DepthOnlyFCBackbone58x87 interface.

    Takes [B, H, W] input and adds channel dim internally.
    """
    def __init__(self, sequential: nn.Sequential, config: DepthEncoderConfig):
        super().__init__()
        self.sequential = sequential
        self.config = config
        # (H, W) only - wrapper adds channel dim in forward()
        # GHN Graph will create [1, H, W] input, then unsqueeze makes [1, 1, H, W]
        self.expected_input_sz = (config.input_height, config.input_width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: [B, H, W] -> [B, 1, H, W]
        # Matches DepthOnlyFCBackbone58x87.forward() behavior
        images = images.unsqueeze(1)
        return self.sequential(images)


def build_depth_backbone(config: DepthEncoderConfig) -> nn.Module:
    """Build a depth encoder backbone from config.

    Args:
        config: DepthEncoderConfig specifying the architecture

    Returns:
        nn.Module that takes [B, H, W] depth image and outputs [B, output_dim]
        (matches DepthOnlyFCBackbone58x87 interface)
    """
    layers = []
    in_ch = config.input_channels
    h, w = config.input_height, config.input_width

    activation = get_activation(config.activation)

    for i in range(config.num_layers):
        out_ch = config.channels[i]
        k = config.kernel_sizes[i]
        s = config.strides[i]
        p = k // 2

        # Conv + BN + Activation
        layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation)

        # Update spatial size
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1

        # Pool
        if i in config.pool_positions:
            if config.pool_type == 'max':
                layers.append(nn.MaxPool2d(config.pool_size))
            elif config.pool_type == 'avg':
                layers.append(nn.AvgPool2d(config.pool_size))
            h = h // config.pool_size
            w = w // config.pool_size

        in_ch = out_ch

    # Flatten
    layers.append(nn.Flatten())
    flat_size = config.channels[-1] * h * w

    # FC layers
    layers.append(nn.Linear(flat_size, config.fc_hidden))
    layers.append(activation)
    layers.append(nn.Linear(config.fc_hidden, config.output_dim))

    sequential = nn.Sequential(*layers)

    # Wrap with unsqueeze handler (matches DepthOnlyFCBackbone58x87 interface)
    backbone = DepthBackboneWrapper(sequential, config)

    return backbone


def build_depth_backbones(configs: List[DepthEncoderConfig]) -> List[nn.Module]:
    """Build multiple depth encoder backbones from configs."""
    return [build_depth_backbone(cfg) for cfg in configs]


class TrainableGHN(nn.Module):
    """Trainable GHN for learning to predict depth encoder weights.

    Unlike using a pre-trained GHN for initialization, this class is designed
    for training the GHN from scratch. The GHN learns to predict good weights
    for depth encoders by backpropagating through the weight prediction.

    Training loop:
        1. Sample diverse architectures
        2. GHN predicts weights for each architecture
        3. Forward pass through depth encoders with predicted weights
        4. Compute loss (DAgger from teacher)
        5. Backprop through encoders AND GHN
        6. Update GHN parameters

    After training, the GHN can instantly predict good weights for any new
    depth encoder architecture.

    Usage:
        ghn = TrainableGHN(device='cuda')
        optimizer = Adam(ghn.parameters(), lr=1e-3)

        # Training loop
        backbones = [build_depth_backbone(cfg) for cfg in configs]
        backbones = ghn.predict_weights(backbones)  # Differentiable!
        outputs = [backbone(depth_images) for backbone in backbones]
        loss = compute_loss(outputs, targets)
        loss.backward()  # Gradients flow to GHN
        optimizer.step()
    """

    def __init__(
        self,
        max_shape: Tuple[int, int, int, int] = (128, 128, 7, 7),
        num_classes: int = 32,  # Output dim of depth encoder
        hid: int = 64,  # Larger hidden dim for better capacity
        hypernet: str = 'gatedgnn',
        decoder: str = 'conv',
        weight_norm: bool = True,
        ve: bool = True,  # Virtual edges
        device: str = 'cuda',
    ):
        """Initialize the trainable GHN.

        Args:
            max_shape: Maximum shape of conv kernels (out, in, h, w)
            num_classes: Output dimension (should match encoder output_dim)
            hid: Hidden dimension for GNN (larger = more capacity)
            hypernet: Type of hypernet ('gatedgnn' or 'mlp')
            decoder: Type of decoder ('conv' or 'mlp')
            weight_norm: Whether to normalize predicted weights
            ve: Whether to use virtual edges
            device: Device to place the GHN on
        """
        super().__init__()

        from ppuda.ppuda.ghn.nn import GHN

        self.ghn = GHN(
            max_shape=max_shape,
            num_classes=num_classes,
            hypernet=hypernet,
            decoder=decoder,
            weight_norm=weight_norm,
            ve=ve,
            hid=hid,
        )
        self.device = device
        self.ve = ve
        self.to(device)

    def predict_weights(
        self,
        models: List[nn.Module],
        return_graphs: bool = False,
    ) -> List[nn.Module]:
        """Predict weights for multiple depth encoder backbones.

        This is differentiable - gradients will flow back to GHN parameters.

        Args:
            models: List of nn.Module backbones (created by build_depth_backbone)
            return_graphs: If True, also return the GraphBatch for reuse

        Returns:
            List of models with predicted weights (same objects, modified in-place)
            If return_graphs=True, also returns (models, graphs)
        """
        from ppuda.ppuda.deepnets1m.graph import Graph, GraphBatch
        from ppuda.ppuda.deepnets1m.net import named_layered_modules

        # Build computation graphs for all models
        graphs = GraphBatch([Graph(m, ve_cutoff=50 if self.ve else 1) for m in models])
        graphs.to_device(self.device)

        # Move models to device
        models = [m.to(self.device) for m in models]

        # Pre-compute _layered_modules for training mode
        # This is required by PPUDA GHN when self.training=True
        for m in models:
            m._layered_modules = named_layered_modules(m)

        # GHN in training mode for gradient flow
        self.ghn.train()

        # Predict weights - this modifies models in-place
        result = self.ghn(models, graphs=graphs)

        if return_graphs:
            return result, graphs
        return result

    def predict_weights_with_graphs(
        self,
        models: List[nn.Module],
        graphs,
    ) -> List[nn.Module]:
        """Predict weights using pre-computed graphs (faster for repeated calls).

        Args:
            models: List of nn.Module backbones
            graphs: Pre-computed GraphBatch from predict_weights(return_graphs=True)

        Returns:
            List of models with predicted weights
        """
        from ppuda.ppuda.deepnets1m.net import named_layered_modules

        # Pre-compute _layered_modules for training mode
        for m in models:
            m._layered_modules = named_layered_modules(m)

        self.ghn.train()
        return self.ghn(models, graphs=graphs)

    def forward(self, models: List[nn.Module]) -> List[nn.Module]:
        """Alias for predict_weights for nn.Module compatibility."""
        return self.predict_weights(models)

    def save(self, path: str):
        """Save GHN checkpoint."""
        torch.save({
            'ghn_state_dict': self.ghn.state_dict(),
            'config': {
                'max_shape': self.ghn.max_shape,
                'num_classes': self.ghn.num_classes,
                've': self.ve,
            }
        }, path)
        print(f"Saved TrainableGHN to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'TrainableGHN':
        """Load GHN from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        ghn = cls(
            max_shape=config['max_shape'],
            num_classes=config['num_classes'],
            ve=config['ve'],
            device=device,
        )
        ghn.ghn.load_state_dict(checkpoint['ghn_state_dict'])
        print(f"Loaded TrainableGHN from {path}")
        return ghn


# Keep old name for backwards compatibility
DepthEncoderGHN = TrainableGHN





if __name__ == '__main__':
    # Test the module
    print("Testing depth encoder GHN module...")

    # Sample configs
    configs = sample_depth_encoder_configs(20, unique=False)
    print(f"Sampled {len(configs)} configurations")

    for i, config in enumerate(configs):
        params = estimate_parameters(config)
        print(f"Config {i}: {config}")
        print(f"  Estimated Params: {params:,}")
        if params > 10_000_000:
            print(f"  WARNING: Params > 10M!")

    # Build backbones
    backbones = build_depth_backbones(configs[:4])

    # Test forward pass
    x = torch.randn(2, 96, 128)
    for i, backbone in enumerate(backbones):
        y = backbone(x)
        print(f"Backbone {i}: input {x.shape} -> output {y.shape}")
        print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")
