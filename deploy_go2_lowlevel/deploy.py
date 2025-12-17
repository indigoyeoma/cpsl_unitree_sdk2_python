#!/usr/bin/env python3
"""
Go2 Vision Policy Deployment Script

Controls:
    Y:  Stand up (from any state)
    L1: Enable walking policy (only when standing)
    R2: EMERGENCY STOP - cuts all motor power

Usage:
    python deploy.py                    # Normal mode
    python deploy.py --dryrun           # Test without motor commands
    python deploy.py --no-camera        # Test without depth camera
"""
import sys
import os
import time
import struct
import argparse
import numpy as np
from enum import IntEnum
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from config import (
    MOTOR_DT, POLICY_DECIMATION,
    KP_WALK, KD_WALK, KP_STAND, KD_STAND,
    DEFAULT_STAND_ANGLES_SDK, JOINT_POS_MIN, JOINT_POS_MAX,
    STAND_UP_DURATION, SIT_DOWN_DURATION,
    FIXED_VEL_X, SDK_TO_TRAIN_JOINTS,
    DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH
)
from depth_camera import DepthCamera
from policy_runner import PolicyRunner

# Motor control constants
PosStopF = 2.146e9
VelStopF = 16000.0


class State(IntEnum):
    """Robot state machine states."""
    IDLE = 0           # Damping mode, waiting for commands
    STANDING_UP = 1    # Interpolating to stand pose
    STANDING = 2       # Holding stand pose
    WALKING = 3        # Running vision policy
    EMERGENCY = 4      # Emergency stop - motors off


class Go2Deployment:
    """
    Main deployment class for Go2 vision policy.
    """

    def __init__(self, dryrun: bool = False, no_camera: bool = False):
        """
        Initialize deployment.

        Args:
            dryrun: If True, don't send motor commands
            no_camera: If True, use dummy depth frames
        """
        self.dryrun = dryrun
        self.no_camera = no_camera

        # State machine
        self.state = State.IDLE
        self.state_start_time = time.time()

        # Low-level command/state
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.crc = CRC()

        # Control loop counter
        self.tick = 0
        self.policy_tick = 0

        # Interpolation targets
        self.start_pos = np.zeros(12, dtype=np.float32)
        self.target_pos = DEFAULT_STAND_ANGLES_SDK.copy()
        self.current_target = DEFAULT_STAND_ANGLES_SDK.copy()

        # Button state (for edge detection)
        self.prev_buttons = 0
        self.button_pressed = {}

        # Components
        self.camera: Optional[DepthCamera] = None
        self.policy: Optional[PolicyRunner] = None

        # Publishers/subscribers
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None

        # Thread control
        self.running = False
        self.control_thread = None

    def init_low_cmd(self):
        """Initialize the low-level command structure."""
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

    def init(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if successful
        """
        print("=" * 60)
        print("Go2 Vision Policy Deployment")
        print("=" * 60)
        print(f"Mode: {'DRYRUN' if self.dryrun else 'LIVE'}")
        print(f"Camera: {'DISABLED' if self.no_camera else 'ENABLED'}")
        print()

        # Initialize command structure
        self.init_low_cmd()

        # Initialize DDS communication
        print("Initializing DDS communication...")
        try:
            ChannelFactoryInitialize(0)
        except Exception as e:
            print(f"Failed to initialize DDS: {e}")
            return False

        # Create publisher/subscriber
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_callback, 10)

        # Wait for first state message
        print("Waiting for robot state...")
        timeout = 5.0
        start = time.time()
        while self.low_state is None and time.time() - start < timeout:
            time.sleep(0.1)

        if self.low_state is None:
            print("ERROR: No robot state received. Is the robot connected?")
            return False
        print("Robot state received")

        # Release high-level control
        print("Releasing high-level control...")
        try:
            sc = SportClient()
            sc.SetTimeout(5.0)
            sc.Init()

            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()

            status, result = msc.CheckMode()
            while result.get('name'):
                sc.StandDown()
                msc.ReleaseMode()
                time.sleep(1)
                status, result = msc.CheckMode()
            print("High-level control released")
        except Exception as e:
            print(f"Warning: Could not release high-level control: {e}")

        # Verify joint mapping
        if not self._verify_joint_mapping():
            return False

        # Initialize depth camera
        if not self.no_camera:
            print("Initializing depth camera...")
            self.camera = DepthCamera(enable_filters=True)
            if not self.camera.start():
                print("ERROR: Failed to start depth camera")
                return False
        else:
            print("Camera disabled - using dummy frames")
            self.camera = DepthCamera(enable_filters=False)

        # Initialize policy
        print("Loading policy...")
        self.policy = PolicyRunner(device="cuda")
        if not self.policy.load_models():
            print("ERROR: Failed to load policy models")
            return False

        print()
        print("=" * 60)
        print("Initialization complete!")
        print()
        print("Controls:")
        print("  Y:  Stand up")
        print("  L1: Enable walking policy (when standing)")
        print("  R2: EMERGENCY STOP")
        print("=" * 60)
        print()

        return True

    def _verify_joint_mapping(self) -> bool:
        """
        Verify joint ordering by printing current positions.

        Returns:
            True if user confirms mapping
        """
        if self.low_state is None:
            return False

        # Get current joint positions
        joints_sdk = np.array([
            self.low_state.motor_state[i].q for i in range(12)
        ], dtype=np.float32)

        joints_train = joints_sdk[SDK_TO_TRAIN_JOINTS]

        print()
        print("=" * 60)
        print("JOINT VERIFICATION")
        print("=" * 60)
        print()
        print("SDK Order [FR, FL, RR, RL]:")
        print(f"  FR: hip={joints_sdk[0]:+.3f}, thigh={joints_sdk[1]:+.3f}, calf={joints_sdk[2]:+.3f}")
        print(f"  FL: hip={joints_sdk[3]:+.3f}, thigh={joints_sdk[4]:+.3f}, calf={joints_sdk[5]:+.3f}")
        print(f"  RR: hip={joints_sdk[6]:+.3f}, thigh={joints_sdk[7]:+.3f}, calf={joints_sdk[8]:+.3f}")
        print(f"  RL: hip={joints_sdk[9]:+.3f}, thigh={joints_sdk[10]:+.3f}, calf={joints_sdk[11]:+.3f}")
        print()
        print("Training Order [FL, FR, RL, RR]:")
        print(f"  FL: hip={joints_train[0]:+.3f}, thigh={joints_train[1]:+.3f}, calf={joints_train[2]:+.3f}")
        print(f"  FR: hip={joints_train[3]:+.3f}, thigh={joints_train[4]:+.3f}, calf={joints_train[5]:+.3f}")
        print(f"  RL: hip={joints_train[6]:+.3f}, thigh={joints_train[7]:+.3f}, calf={joints_train[8]:+.3f}")
        print(f"  RR: hip={joints_train[9]:+.3f}, thigh={joints_train[10]:+.3f}, calf={joints_train[11]:+.3f}")
        print()
        print("Expected: FL_hip should be ~+0.1, FR_hip should be ~-0.1")
        print("          Rear thighs (~1.0) should differ from front thighs (~0.8)")
        print()

        try:
            input("Press Enter to confirm joint mapping is correct, or Ctrl+C to abort...")
            return True
        except KeyboardInterrupt:
            print("\nAborted by user")
            return False

    def _low_state_callback(self, msg: LowState_):
        """Callback for low-level state messages."""
        self.low_state = msg

    def _parse_buttons(self) -> dict:
        """
        Parse button states from wireless remote.

        Returns:
            Dict of button states with edge detection
        """
        if self.low_state is None:
            return {}

        remote = self.low_state.wireless_remote
        data1 = remote[2]
        data2 = remote[3]
        buttons = data1 | (data2 << 8)

        # Button masks
        BUTTON_R1 = 1 << 0
        BUTTON_L1 = 1 << 1
        BUTTON_R2 = 1 << 4
        BUTTON_L2 = 1 << 5
        BUTTON_A = 1 << 8
        BUTTON_B = 1 << 9
        BUTTON_X = 1 << 10
        BUTTON_Y = 1 << 11

        # Detect rising edges
        self.button_pressed = {
            'Y': bool(buttons & BUTTON_Y) and not bool(self.prev_buttons & BUTTON_Y),
            'L1': bool(buttons & BUTTON_L1) and not bool(self.prev_buttons & BUTTON_L1),
            'R2': bool(buttons & BUTTON_R2),  # No edge detection for emergency
            'A': bool(buttons & BUTTON_A) and not bool(self.prev_buttons & BUTTON_A),
            'B': bool(buttons & BUTTON_B) and not bool(self.prev_buttons & BUTTON_B),
        }

        self.prev_buttons = buttons
        return self.button_pressed

    def _enter_state(self, new_state: State):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()

        # State entry actions
        if new_state == State.STANDING_UP:
            # Record current position for interpolation
            if self.low_state is not None:
                self.start_pos = np.array([
                    self.low_state.motor_state[i].q for i in range(12)
                ], dtype=np.float32)
            self.target_pos = DEFAULT_STAND_ANGLES_SDK.copy()
            print("STATE: Standing up...")

        elif new_state == State.STANDING:
            print("STATE: Standing - press L1 to start walking")

        elif new_state == State.WALKING:
            self.policy.reset()
            self.policy_tick = 0
            print("STATE: Walking - policy active")

        elif new_state == State.IDLE:
            print("STATE: Idle")

        elif new_state == State.EMERGENCY:
            print("!!! EMERGENCY STOP - Motors disabled !!!")

    def _get_state_duration(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_start_time

    def _send_motor_commands(
        self,
        targets: np.ndarray,
        kp: float,
        kd: float
    ):
        """
        Send motor position commands.

        Args:
            targets: Joint position targets in SDK order [12]
            kp: Position gain
            kd: Damping gain
        """
        # Clip to joint limits
        targets = np.clip(targets, JOINT_POS_MIN, JOINT_POS_MAX)

        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = float(targets[i])
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)

        if not self.dryrun:
            self.lowcmd_publisher.Write(self.low_cmd)

    def _send_emergency_stop(self):
        """Send emergency stop - cut all motor power."""
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x00  # Disable motor
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)

        if not self.dryrun:
            self.lowcmd_publisher.Write(self.low_cmd)

    def _control_loop(self):
        """Main control loop - runs at 500Hz."""
        if self.low_state is None:
            return

        self.tick += 1

        # Parse button inputs
        buttons = self._parse_buttons()

        # Emergency stop takes priority
        if buttons.get('R2', False):
            if self.state != State.EMERGENCY:
                self._enter_state(State.EMERGENCY)
            self._send_emergency_stop()
            return

        # State machine
        if self.state == State.IDLE:
            # Damping mode
            self._send_motor_commands(
                self.current_target, kp=0.0, kd=KD_STAND
            )
            if buttons.get('Y', False):
                self._enter_state(State.STANDING_UP)

        elif self.state == State.STANDING_UP:
            duration = self._get_state_duration()
            progress = min(1.0, duration / STAND_UP_DURATION)

            # Smooth interpolation (ease-in-out)
            t = progress * progress * (3 - 2 * progress)
            self.current_target = (1 - t) * self.start_pos + t * self.target_pos

            self._send_motor_commands(self.current_target, KP_STAND, KD_STAND)

            if progress >= 1.0:
                self._enter_state(State.STANDING)

        elif self.state == State.STANDING:
            self._send_motor_commands(DEFAULT_STAND_ANGLES_SDK, KP_STAND, KD_STAND)

            if buttons.get('L1', False):
                self._enter_state(State.WALKING)
            elif buttons.get('Y', False):
                self._enter_state(State.IDLE)

        elif self.state == State.WALKING:
            # Run policy at 50Hz
            if self.tick % POLICY_DECIMATION == 0:
                self.policy_tick += 1
                self.current_target = self._run_policy()

            self._send_motor_commands(self.current_target, KP_WALK, KD_WALK)

            if buttons.get('Y', False):
                self._enter_state(State.STANDING)

        elif self.state == State.EMERGENCY:
            self._send_emergency_stop()
            # Can only exit emergency with restart

    def _run_policy(self) -> np.ndarray:
        """
        Run one step of the vision policy.

        Returns:
            Joint position targets in SDK order [12]
        """
        # Get depth frame
        if self.camera is not None:
            depth = self.camera.get_frame()
            if depth is None:
                depth = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)
        else:
            depth = np.zeros((DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH), dtype=np.float32)

        # Get robot state
        state = self.low_state
        ang_vel = np.array([
            state.imu_state.gyroscope[0],
            state.imu_state.gyroscope[1],
            state.imu_state.gyroscope[2],
        ], dtype=np.float32)

        roll = state.imu_state.rpy[0]
        pitch = state.imu_state.rpy[1]

        dof_pos = np.array([state.motor_state[i].q for i in range(12)], dtype=np.float32)
        dof_vel = np.array([state.motor_state[i].dq for i in range(12)], dtype=np.float32)

        # Foot contacts (threshold force)
        foot_contacts = np.array([
            1.0 if state.foot_force[i] > 20 else 0.0 for i in range(4)
        ], dtype=np.float32)

        # Run inference
        targets = self.policy.run_inference(
            depth_image=depth,
            ang_vel=ang_vel,
            roll=roll,
            pitch=pitch,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            foot_contacts=foot_contacts,
            cmd_vel_x=FIXED_VEL_X,
        )

        return targets

    def start(self):
        """Start the control loop."""
        self.running = True
        self.control_thread = RecurrentThread(
            interval=MOTOR_DT,
            target=self._control_loop,
            name="control"
        )
        self.control_thread.Start()
        print("Control loop started")

    def stop(self):
        """Stop everything."""
        print("Stopping...")
        self.running = False

        if self.control_thread is not None:
            self.control_thread.Stop()

        if self.camera is not None:
            self.camera.stop()

        print("Stopped")


def main():
    parser = argparse.ArgumentParser(description="Go2 Vision Policy Deployment")
    parser.add_argument("--dryrun", action="store_true",
                        help="Don't send motor commands")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable depth camera (use dummy frames)")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface (e.g., eth0)")
    args = parser.parse_args()

    # Initialize DDS with network interface if specified
    if args.interface:
        ChannelFactoryInitialize(0, args.interface)

    deployment = Go2Deployment(
        dryrun=args.dryrun,
        no_camera=args.no_camera
    )

    if not deployment.init():
        print("Initialization failed")
        sys.exit(1)

    deployment.start()

    print("\nRunning... Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        deployment.stop()


if __name__ == "__main__":
    main()
