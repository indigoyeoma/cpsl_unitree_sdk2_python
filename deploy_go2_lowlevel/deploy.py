#!/usr/bin/env python3
"""
Go2 Vision Policy Deployment with Safe Sit→Stand→Walk→Sit Sequence

Sequence:
1. Camera warmup (3s)
2. Sit → Stand (smooth transition)
3. Wait 5 seconds in standing pose
4. Run vision policy (walk forward)
5. On Ctrl+C: Stand → Sit (safe shutdown)

Usage:
    python deploy.py --command_vx 0.5
"""

import sys
import os
import time
import argparse
import signal
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


class Go2VisionController:
    """
    Go2 controller with safe sit→stand→walk→sit sequence.

    Phases:
    - Phase 1 (sit→stand): Smooth transition from sitting to standing
    - Phase 2 (hold): Hold standing position for 5 seconds
    - Phase 3 (walk): Run vision policy
    - Phase 4 (stand→sit): Safe sit down on shutdown
    """

    def __init__(self, policy: JITPolicyRunner, camera, config: DeployConfig):
        self.policy = policy
        self.camera = camera
        self.config = config
        self.crc = CRC()

        # Control state
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.running = False
        self.shutdown_requested = False

        # Observation buffers
        self.obs_history = deque(maxlen=config.history_len)
        self.action_history = deque(maxlen=2)
        self.last_action = np.zeros(12, dtype=np.float32)

        # Phase control (matching go2_stand_example.py pattern)
        self.phase = 0  # 0=sit→stand, 1=hold, 2=walk, 3=stand→sit
        self.dt = 0.002  # 500Hz control

        # Target positions (SDK joint order: FR, FL, RR, RL)
        # Sitting pose (relaxed, motors can be soft)
        self._sit_pos = np.array([
            0.0, 1.4, -2.7,   # FR: hip, thigh, calf
            0.0, 1.4, -2.7,   # FL
            0.0, 1.4, -2.7,   # RR
            0.0, 1.4, -2.7,   # RL
        ])

        # Standing pose (from training config, SDK order)
        self._stand_pos = self._training_to_sdk_order(config.default_joint_angles)

        # Start position (captured from robot)
        self.start_pos = np.zeros(12)
        self.first_run = True

        # Phase durations and progress
        self.duration_sit_to_stand = 1000  # 2 seconds @ 500Hz
        self.duration_hold = 2500          # 5 seconds @ 500Hz
        self.duration_stand_to_sit = 1000  # 2 seconds @ 500Hz

        self.percent_sit_to_stand = 0.0
        self.percent_hold = 0.0
        self.percent_stand_to_sit = 0.0

        # Walking phase control
        self.walk_startup_duration = 500  # 10 seconds @ 50Hz policy
        self.walk_startup_counter = 0

        # Timing
        self.last_policy_time = 0
        self.policy_dt = config.control_dt
        self.control_step = 0

        print(f"Controller initialized")
        print(f"  Command velocity: {self.config.command_vx} m/s")
        print(f"  Sit→Stand: {self.duration_sit_to_stand * self.dt:.1f}s")
        print(f"  Hold: {self.duration_hold * self.dt:.1f}s")
        print(f"  Walk startup ramp: {self.walk_startup_duration * self.policy_dt:.1f}s")
        print(f"  Stand→Sit: {self.duration_stand_to_sit * self.dt:.1f}s")

    def init(self):
        """Initialize SDK channels."""
        self._init_low_cmd()

        # Create publisher
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        # Sport client
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()

        # Motion switcher
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(5.0)
        self.motion_switcher.Init()

        print("\nWaiting for robot state...")
        while self.low_state is None:
            time.sleep(0.1)
        print("✓ Robot state received")

    def _init_low_cmd(self):
        """Initialize low-level command structure."""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _low_state_handler(self, msg: LowState_):
        """Callback for low-level state messages."""
        self.low_state = msg

    def release_control(self):
        """Release built-in motion control."""
        print("\nReleasing built-in motion control...")
        status, result = self.motion_switcher.CheckMode()
        while result['name']:
            self.sport_client.StandDown()
            time.sleep(0.5)
            self.motion_switcher.ReleaseMode()
            status, result = self.motion_switcher.CheckMode()
            time.sleep(1)
        print("✓ Control released")

    def start(self):
        """Start the control loop."""
        self.running = True
        self.phase = 0  # Start with sit→stand

        # Start camera
        print("\n" + "=" * 70)
        print("Starting Camera")
        print("=" * 70)
        self.camera.start()
        print("✓ Camera started")

        # Camera warmup
        print("\nWarming up camera for 3 seconds...")
        for i in range(3):
            print(f"  {i+1}/3...", end='\r')
            time.sleep(1.0)
        print("  ✓ Camera ready" + " " * 20)

        # Start control thread at 500Hz
        print("\nStarting control loop at 500Hz...")
        self.control_thread = RecurrentThread(
            interval=self.dt,
            target=self._control_loop,
            name="vision_control"
        )
        self.control_thread.Start()
        print("✓ Control loop started")

    def request_shutdown(self):
        """Request safe shutdown (triggers stand→sit)."""
        if not self.shutdown_requested:
            print("\n\n⚠️  Shutdown requested - returning to sit position...")
            self.shutdown_requested = True
            self.phase = 3  # Jump to stand→sit phase
            self.percent_stand_to_sit = 0.0

    def stop(self):
        """Stop the control loop."""
        print("\nStopping control...")
        self.running = False
        time.sleep(0.1)

        # Send damping mode
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

        self.camera.stop()
        print("✓ Control stopped")

    def _control_loop(self):
        """Main control loop running at 500Hz."""
        if not self.running or self.low_state is None:
            return

        # Capture start position on first run
        if self.first_run:
            self.start_pos = self._get_joint_positions()
            self.first_run = False
            print(f"\n✓ Captured start position")

        # Execute current phase
        if self.phase == 0:
            self._phase_sit_to_stand()
        elif self.phase == 1:
            self._phase_hold_stand()
        elif self.phase == 2:
            self._phase_walk()
        elif self.phase == 3:
            self._phase_stand_to_sit()

        # Send motor commands
        self._send_command()

        self.control_step += 1

    def _phase_sit_to_stand(self):
        """Phase 0: Smooth transition from sitting to standing."""
        self.percent_sit_to_stand += 1.0 / self.duration_sit_to_stand
        self.percent_sit_to_stand = min(self.percent_sit_to_stand, 1.0)

        # Interpolate from start to standing
        target_pos = (1 - self.percent_sit_to_stand) * self.start_pos + \
                     self.percent_sit_to_stand * self._stand_pos

        self.target_positions = target_pos

        # Print progress
        if self.control_step % 250 == 0:  # Every 0.5s
            progress = int(self.percent_sit_to_stand * 100)
            print(f"  Sit→Stand: {progress}%", end='\r')

        # Move to next phase when complete
        if self.percent_sit_to_stand >= 1.0:
            print(f"  Sit→Stand: 100% ✓" + " " * 20)
            self.phase = 1
            self.percent_hold = 0.0
            print("\n✓ Standing pose reached")
            print(f"\nHolding position for {self.duration_hold * self.dt:.0f} seconds...")

    def _phase_hold_stand(self):
        """Phase 1: Hold standing position."""
        self.percent_hold += 1.0 / self.duration_hold
        self.percent_hold = min(self.percent_hold, 1.0)

        # Hold standing position
        self.target_positions = self._stand_pos.copy()

        # Print countdown
        if self.control_step % 500 == 0:  # Every 1s
            remaining = (1.0 - self.percent_hold) * self.duration_hold * self.dt
            print(f"  Holding: {remaining:.1f}s remaining", end='\r')

        # Move to walk phase when complete
        if self.percent_hold >= 1.0:
            print(f"  Holding: Complete ✓" + " " * 30)
            self.phase = 2
            self.walk_startup_counter = 0
            self.last_policy_time = time.time()
            print("\n" + "=" * 70)
            print(f"Starting Vision Policy (target: {self.config.command_vx} m/s)")
            print("=" * 70)
            print("Press Ctrl+C to stop and sit down safely\n")

    def _phase_walk(self):
        """Phase 2: Run vision policy for walking."""
        current_time = time.time()

        # Run policy at 50Hz
        if current_time - self.last_policy_time >= self.policy_dt:
            self.last_policy_time = current_time

            # Get depth image and build observation
            depth_image = self.camera.get_depth()
            obs = self._build_observation()

            # Run policy inference
            action = self.policy.get_action(depth_image, obs)

            # Scale action and add to default pose (in training order)
            target_delta = action * self.config.action_scale
            target_pos_train = self.config.default_joint_angles + target_delta

            # Apply joint limits
            target_pos_train = JointLimits.clip_joints(target_pos_train)

            # Smooth startup ramp (10 seconds)
            if self.walk_startup_counter < self.walk_startup_duration:
                alpha = self.walk_startup_counter / self.walk_startup_duration
                target_pos_train = (1 - alpha) * self.config.default_joint_angles + \
                                 alpha * target_pos_train
                self.walk_startup_counter += 1

            # Convert to SDK order
            self.target_positions = self._training_to_sdk_order(target_pos_train)

            # Update history
            self.last_action = action
            self.action_history.append(action)

            # Print status every 1 second
            if self.walk_startup_counter % 50 == 0:
                if self.walk_startup_counter < self.walk_startup_duration:
                    progress = 100 * self.walk_startup_counter / self.walk_startup_duration
                    print(f"  Startup ramp: {progress:.0f}%", end='\r')
                elif self.walk_startup_counter == self.walk_startup_duration:
                    print(f"  Startup ramp: 100% ✓" + " " * 20)
                    print("  Walking at full speed...")

    def _phase_stand_to_sit(self):
        """Phase 3: Safe sit down from standing."""
        self.percent_stand_to_sit += 1.0 / self.duration_stand_to_sit
        self.percent_stand_to_sit = min(self.percent_stand_to_sit, 1.0)

        # Interpolate from standing to sitting
        target_pos = (1 - self.percent_stand_to_sit) * self._stand_pos + \
                     self.percent_stand_to_sit * self._sit_pos

        self.target_positions = target_pos

        # Print progress
        if self.control_step % 250 == 0:  # Every 0.5s
            progress = int(self.percent_stand_to_sit * 100)
            print(f"  Stand→Sit: {progress}%", end='\r')

        # Stop when complete
        if self.percent_stand_to_sit >= 1.0:
            print(f"  Stand→Sit: 100% ✓" + " " * 20)
            print("\n✓ Sit position reached - safe to power off")
            self.running = False

    def _send_command(self):
        """Send low-level motor commands."""
        if not hasattr(self, 'target_positions'):
            return

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(self.target_positions[i])
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.config.kp
            self.low_cmd.motor_cmd[i].kd = self.config.kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from real sensors."""
        # Get IMU data
        imu = self.low_state.imu_state
        roll = np.arctan2(
            2 * (imu.quaternion[0] * imu.quaternion[1] + imu.quaternion[2] * imu.quaternion[3]),
            1 - 2 * (imu.quaternion[1]**2 + imu.quaternion[2]**2)
        )
        pitch = np.arcsin(np.clip(
            2 * (imu.quaternion[0] * imu.quaternion[2] - imu.quaternion[3] * imu.quaternion[1]),
            -1, 1
        ))

        # Angular velocity
        ang_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])

        # Joint states
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()
        joint_pos_train = self._sdk_to_training_order(joint_pos)
        joint_vel_train = self._sdk_to_training_order(joint_vel)

        dof_pos = joint_pos_train - self.config.default_joint_angles
        last_action = self.last_action if len(self.action_history) > 0 else np.zeros(12)

        # Contacts
        contacts = np.array([
            self.low_state.foot_force[1] > 20,  # FL
            self.low_state.foot_force[0] > 20,  # FR
            self.low_state.foot_force[3] > 20,  # RL
            self.low_state.foot_force[2] > 20,  # RR
        ]).astype(np.float32) - 0.5

        # Scales
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05

        # Build proprio (53 dims)
        proprio = np.concatenate([
            ang_vel * ang_vel_scale,       # 3
            [roll, pitch],                  # 2
            [0.0],                          # delta_yaw (masked)
            [0.0],                          # delta_yaw
            [0.0],                          # delta_next_yaw
            [0.0, 0.0],                     # commands (masked)
            [self.config.command_vx],       # command_vx (GOAL)
            [1.0, 0.0],                     # env_class flags
            dof_pos * dof_pos_scale,        # 12
            joint_vel_train * dof_vel_scale, # 12
            last_action,                    # 12
            contacts,                       # 4
        ]).astype(np.float32)

        # Update history
        self.obs_history.append(proprio.copy())
        while len(self.obs_history) < self.config.history_len:
            self.obs_history.appendleft(proprio.copy())

        history = np.concatenate(list(self.obs_history))

        # Full observation with placeholders
        full_obs = np.concatenate([
            proprio,                                            # 53
            np.zeros(self.config.n_scan, dtype=np.float32),   # 132 (zeros!)
            np.zeros(self.config.n_priv_explicit, dtype=np.float32), # 9
            np.zeros(self.config.n_priv_latent, dtype=np.float32),   # 29
            history                                             # 530
        ])

        return full_obs.astype(np.float32)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([self.low_state.motor_state[i].q for i in range(12)])

    def _get_joint_velocities(self) -> np.ndarray:
        return np.array([self.low_state.motor_state[i].dq for i in range(12)])

    def _training_to_sdk_order(self, training_joints: np.ndarray) -> np.ndarray:
        """FL,FR,RL,RR → FR,FL,RR,RL"""
        sdk_joints = np.zeros(12)
        sdk_joints[3:6] = training_joints[0:3]    # FL
        sdk_joints[0:3] = training_joints[3:6]    # FR
        sdk_joints[9:12] = training_joints[6:9]   # RL
        sdk_joints[6:9] = training_joints[9:12]   # RR
        return sdk_joints

    def _sdk_to_training_order(self, sdk_joints: np.ndarray) -> np.ndarray:
        """FR,FL,RR,RL → FL,FR,RL,RR"""
        training_joints = np.zeros(12)
        training_joints[0:3] = sdk_joints[3:6]    # FL
        training_joints[3:6] = sdk_joints[0:3]    # FR
        training_joints[6:9] = sdk_joints[9:12]   # RL
        training_joints[9:12] = sdk_joints[6:9]   # RR
        return training_joints


def find_latest_models(policy_dir):
    """Find the latest JIT models in policy directory."""
    if not os.path.exists(policy_dir):
        return None, None

    files = os.listdir(policy_dir)
    vision_weights = [f for f in files if 'vision_weight.pt' in f]
    base_jits = [f for f in files if 'base_jit.pt' in f]

    if not vision_weights or not base_jits:
        return None, None

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
                        help='Directory containing policy files')
    parser.add_argument('--command_vx', type=float, default=0.5,
                        help='Forward velocity goal in m/s (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    parser.add_argument('--use_dummy_camera', action='store_true',
                        help='Use dummy camera for testing')
    parser.add_argument('--network_interface', type=str, default=None,
                        help='Network interface for DDS')
    args = parser.parse_args()

    # Find models
    if args.vision_weight and args.base_jit:
        vision_weight_path = args.vision_weight
        base_jit_path = args.base_jit
    else:
        print(f"Looking for models in: {args.policy_dir}")
        vision_weight_path, base_jit_path = find_latest_models(args.policy_dir)

        if not vision_weight_path or not base_jit_path:
            print(f"ERROR: No models found in {args.policy_dir}")
            return 1

    print("=" * 70)
    print("Go2 Vision Policy Deployment")
    print("=" * 70)
    print(f"Vision weights: {os.path.basename(vision_weight_path)}")
    print(f"Base JIT:       {os.path.basename(base_jit_path)}")
    print(f"Command velocity: {args.command_vx} m/s")
    print(f"Device:         {args.device}")
    print("=" * 70)

    if not args.use_dummy_camera:
        print("\n⚠️  SAFETY WARNING")
        print("  - Clear 3-5 meters in front of robot")
        print("  - Press Ctrl+C anytime for safe shutdown")
        print("  - Robot will: Sit→Stand→Walk→Sit")
        response = input("\nReady? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            return 0

    # Initialize DDS
    print("\nInitializing DDS...")
    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    # Load policy
    print("Loading policy...")
    policy = JITPolicyRunner(vision_weight_path, base_jit_path, args.device)

    # Create camera
    camera = create_camera(
        use_real=not args.use_dummy_camera,
        target_width=DeployConfig.depth_width,
        target_height=DeployConfig.depth_height,
        near_clip=DeployConfig.depth_near,
        far_clip=DeployConfig.depth_far,
    )

    # Setup config
    config = DeployConfig()
    config.command_vx = args.command_vx

    # Create controller
    controller = Go2VisionController(policy, camera, config)

    # Setup signal handler for safe shutdown
    def signal_handler(sig, frame):
        controller.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize
        print("\n" + "=" * 70)
        print("Initializing Robot")
        print("=" * 70)
        controller.init()

        if not args.use_dummy_camera:
            controller.release_control()

        # Start control (camera warmup + sit→stand→hold→walk)
        controller.start()

        # Run until shutdown
        while controller.running:
            time.sleep(0.1)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()
        print("\n✓ Shutdown complete")

    return 0


if __name__ == '__main__':
    sys.exit(main())
