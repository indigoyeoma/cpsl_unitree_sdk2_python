#!/usr/bin/env python3
"""
Simple deployment script for Go2 vision policy.

Usage:
    # Use models from policy/ directory
    python deploy.py --command_vx 0.5

    # Or specify custom paths
    python deploy.py --vision_weight policy/student_depth-15000-vision_weight.pt \\
                     --base_jit policy/student_depth-15000-base_jit.pt \\
                     --command_vx 0.5
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
from collections import deque

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from depth_camera import create_camera
from config import DeployConfig, JointLimits, LegID, PosStopF, VelStopF
from policy_jit import JITPolicyRunner


# Import the controller from deploy_jit
import importlib.util
spec = importlib.util.spec_from_file_location("deploy_jit",
                                               os.path.join(os.path.dirname(__file__), "deploy_jit.py"))
deploy_jit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_jit)
Go2JITController = deploy_jit.Go2JITController


def find_latest_models(policy_dir):
    """Find the latest JIT models in policy directory."""
    if not os.path.exists(policy_dir):
        return None, None

    files = os.listdir(policy_dir)
    vision_weights = [f for f in files if 'vision_weight.pt' in f]
    base_jits = [f for f in files if 'base_jit.pt' in f]

    if not vision_weights or not base_jits:
        return None, None

    # Sort by checkpoint number (assumes format: exptid-checkpoint-type.pt)
    vision_weights.sort(key=lambda x: int(x.split('-')[1]))
    base_jits.sort(key=lambda x: int(x.split('-')[1]))

    return (os.path.join(policy_dir, vision_weights[-1]),
            os.path.join(policy_dir, base_jits[-1]))


def main():
    parser = argparse.ArgumentParser(description='Go2 Vision Policy Deployment')
    parser.add_argument('--vision_weight', type=str, default=None,
                        help='Path to vision_weight.pt file')
    parser.add_argument('--base_jit', type=str, default=None,
                        help='Path to base_jit.pt file')
    parser.add_argument('--policy_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'policy'),
                        help='Directory containing policy files (default: ./policy/)')
    parser.add_argument('--command_vx', type=float, default=0.5,
                        help='Forward velocity command in m/s (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    parser.add_argument('--use_dummy_camera', action='store_true',
                        help='Use dummy camera for testing')
    parser.add_argument('--network_interface', type=str, default=None,
                        help='Network interface for DDS (e.g., eth0)')
    args = parser.parse_args()

    # Determine model paths
    if args.vision_weight and args.base_jit:
        vision_weight_path = args.vision_weight
        base_jit_path = args.base_jit
    else:
        print(f"Looking for models in: {args.policy_dir}")
        vision_weight_path, base_jit_path = find_latest_models(args.policy_dir)

        if not vision_weight_path or not base_jit_path:
            print(f"ERROR: No models found in {args.policy_dir}")
            print("\nPlease either:")
            print("1. Place JIT models in deploy_go2_lowlevel/policy/")
            print("2. Specify paths with --vision_weight and --base_jit")
            return

    print("=" * 70)
    print("Go2 Vision Policy Deployment")
    print("=" * 70)
    print(f"Vision weights: {vision_weight_path}")
    print(f"Base JIT:       {base_jit_path}")
    print(f"Command velocity: {args.command_vx} m/s")
    print(f"Device:         {args.device}")
    print(f"Dummy camera:   {args.use_dummy_camera}")
    print("=" * 70)

    # Verify files exist
    if not os.path.exists(vision_weight_path):
        print(f"ERROR: Vision weights not found: {vision_weight_path}")
        return
    if not os.path.exists(base_jit_path):
        print(f"ERROR: Base JIT not found: {base_jit_path}")
        return

    if not args.use_dummy_camera:
        print("\n⚠️  WARNING: Ensure the robot is in a safe area with no obstacles!")
        print(f"⚠️  The robot will walk forward at {args.command_vx} m/s after standup.")
        print("\nPress Ctrl+C at any time to stop.")
        response = input("\nReady to deploy? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Deployment cancelled.")
            return

    # Initialize DDS
    print("\nInitializing DDS...")
    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    # Load JIT policy
    print("Loading policy...")
    policy = JITPolicyRunner(vision_weight_path, base_jit_path, args.device)

    # Create camera
    print("Initializing camera...")
    camera = create_camera(
        use_real=not args.use_dummy_camera,
        target_width=DeployConfig.depth_width,
        target_height=DeployConfig.depth_height,
        near_clip=DeployConfig.depth_near,
        far_clip=DeployConfig.depth_far,
    )

    # Update config with command velocity
    config = DeployConfig()
    config.command_vx = args.command_vx

    # Create controller
    controller = Go2JITController(policy, camera, config)

    try:
        # Initialize
        print("\nInitializing robot controller...")
        controller.init()

        if not args.use_dummy_camera:
            # Release built-in control
            controller.release_control()

            # Stand up
            controller.standup_sequence()

            print("\n" + "=" * 70)
            print("Starting vision control in 3 seconds...")
            print(f"Target velocity: {args.command_vx} m/s")
            print("Press Ctrl+C to stop")
            print("=" * 70)
            time.sleep(3)

        # Start vision control
        controller.start()

        # Run until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping robot...")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()
        print("\n✓ Shutdown complete.")


if __name__ == '__main__':
    main()
