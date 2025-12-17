#!/usr/bin/env python3
"""
Go2 Vision Policy Deployment Script

Controls:
    Y:  Stand up (from any state)
    L1: Enable walking policy (only when standing)
    R2: EMERGENCY STOP - cuts all motor power

Usage:
    python deploy.py                    # Normal mode
    python deploy.py --dryrun           # Test without motor commands (saves logs)
    python deploy.py --no-camera        # Test without depth camera

Dryrun mode saves sensor logs to: deploy_log_YYYYMMDD_HHMMSS.txt
"""
import sys
import os
import time
import struct
import argparse
import numpy as np
from enum import IntEnum
from typing import Optional
from datetime import datetime

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
    ACTION_SCALE,
    DEFAULT_STAND_ANGLES_SDK, JOINT_POS_MIN, JOINT_POS_MAX,
    STAND_UP_DURATION, SIT_DOWN_DURATION,
    FIXED_VEL_X, SDK_TO_TRAIN_JOINTS,
    DEPTH_OUTPUT_HEIGHT, DEPTH_OUTPUT_WIDTH,
    DEPTH_NEAR, DEPTH_FAR
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
        self.button_hold_time = {}  # Track how long buttons are held

        # Components
        self.camera: Optional[DepthCamera] = None
        self.policy: Optional[PolicyRunner] = None

        # Publishers/subscribers
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None

        # Thread control
        self.running = False
        self.control_thread = None

        # Logging (enabled for both dryrun and normal mode)
        self.log_file = None
        self.log_interval = 50  # Log every 50 ticks (10Hz at 500Hz loop)
        self.last_depth_stats = {}
        self.last_policy_output = None
        self.log_start_time = time.time()

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

        # Initialize logging (always enabled for debugging)
        self._init_logging()

        return True

    def _init_logging(self):
        """Initialize log file for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "DRYRUN" if self.dryrun else "LIVE"
        log_filename = f"deploy_log_{mode_str}_{timestamp}.txt"
        log_path = os.path.join(os.path.dirname(__file__), log_filename)

        self.log_file = open(log_path, 'w')
        self.log_start_time = time.time()

        # Write header
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Go2 Deployment Log - {mode_str} MODE\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 80 + "\n\n")

        # Write configuration
        self.log_file.write("## Configuration\n")
        self.log_file.write(f"- Mode: {mode_str}\n")
        self.log_file.write(f"- Camera enabled: {not self.no_camera}\n")
        self.log_file.write(f"- Fixed velocity: {FIXED_VEL_X} m/s\n")
        self.log_file.write(f"- KP_WALK: {KP_WALK}, KD_WALK: {KD_WALK}\n")
        self.log_file.write(f"- KP_STAND: {KP_STAND}, KD_STAND: {KD_STAND}\n")
        self.log_file.write(f"- ACTION_SCALE: {ACTION_SCALE}\n")
        self.log_file.write(f"- Depth range: {DEPTH_NEAR}m - {DEPTH_FAR}m\n")
        self.log_file.write("\n")

        print(f"Logging to: {log_path}")

    def _log_sensor_data(self, depth_stats: dict = None, policy_output: np.ndarray = None, extra_debug: dict = None):
        """Log sensor data to file."""
        if self.log_file is None or self.low_state is None:
            return

        t = time.time() - self.log_start_time
        state = self.low_state

        # Get sensor data
        ang_vel = np.array([state.imu_state.gyroscope[i] for i in range(3)])
        rpy = np.array([state.imu_state.rpy[i] for i in range(3)])
        dof_pos = np.array([state.motor_state[i].q for i in range(12)])
        dof_vel = np.array([state.motor_state[i].dq for i in range(12)])
        foot_force = np.array([state.foot_force[i] for i in range(4)])

        # Write log entry
        self.log_file.write(f"{'='*80}\n")
        self.log_file.write(f"--- Tick {self.policy_tick} | t={t:.3f}s | State: {self.state.name} ---\n")
        self.log_file.write(f"{'='*80}\n")

        # IMU
        self.log_file.write(f"\n[IMU]\n")
        self.log_file.write(f"  roll={rpy[0]:+.4f} rad ({np.degrees(rpy[0]):+.2f} deg)\n")
        self.log_file.write(f"  pitch={rpy[1]:+.4f} rad ({np.degrees(rpy[1]):+.2f} deg)\n")
        self.log_file.write(f"  yaw={rpy[2]:+.4f} rad ({np.degrees(rpy[2]):+.2f} deg)\n")
        self.log_file.write(f"  ang_vel=[{ang_vel[0]:+.4f}, {ang_vel[1]:+.4f}, {ang_vel[2]:+.4f}] rad/s\n")

        # Joints (SDK order: FR, FL, RR, RL)
        self.log_file.write(f"\n[Joint Positions - ACTUAL (SDK order: FR,FL,RR,RL)]\n")
        self.log_file.write(f"  FR (hip,thigh,calf): [{dof_pos[0]:+.4f}, {dof_pos[1]:+.4f}, {dof_pos[2]:+.4f}]\n")
        self.log_file.write(f"  FL (hip,thigh,calf): [{dof_pos[3]:+.4f}, {dof_pos[4]:+.4f}, {dof_pos[5]:+.4f}]\n")
        self.log_file.write(f"  RR (hip,thigh,calf): [{dof_pos[6]:+.4f}, {dof_pos[7]:+.4f}, {dof_pos[8]:+.4f}]\n")
        self.log_file.write(f"  RL (hip,thigh,calf): [{dof_pos[9]:+.4f}, {dof_pos[10]:+.4f}, {dof_pos[11]:+.4f}]\n")

        # Default stand for comparison
        default = DEFAULT_STAND_ANGLES_SDK
        self.log_file.write(f"\n[Default Stand Angles (SDK order)]\n")
        self.log_file.write(f"  FR: [{default[0]:+.4f}, {default[1]:+.4f}, {default[2]:+.4f}]\n")
        self.log_file.write(f"  FL: [{default[3]:+.4f}, {default[4]:+.4f}, {default[5]:+.4f}]\n")
        self.log_file.write(f"  RR: [{default[6]:+.4f}, {default[7]:+.4f}, {default[8]:+.4f}]\n")
        self.log_file.write(f"  RL: [{default[9]:+.4f}, {default[10]:+.4f}, {default[11]:+.4f}]\n")

        # Policy output (commanded positions)
        if policy_output is not None:
            self.log_file.write(f"\n[Policy Output - COMMANDED (SDK order)]\n")
            self.log_file.write(f"  FR: [{policy_output[0]:+.4f}, {policy_output[1]:+.4f}, {policy_output[2]:+.4f}]\n")
            self.log_file.write(f"  FL: [{policy_output[3]:+.4f}, {policy_output[4]:+.4f}, {policy_output[5]:+.4f}]\n")
            self.log_file.write(f"  RR: [{policy_output[6]:+.4f}, {policy_output[7]:+.4f}, {policy_output[8]:+.4f}]\n")
            self.log_file.write(f"  RL: [{policy_output[9]:+.4f}, {policy_output[10]:+.4f}, {policy_output[11]:+.4f}]\n")

            # Error between commanded and actual
            error = policy_output - dof_pos
            self.log_file.write(f"\n[Position Error (commanded - actual)]\n")
            self.log_file.write(f"  FR: [{error[0]:+.4f}, {error[1]:+.4f}, {error[2]:+.4f}]\n")
            self.log_file.write(f"  FL: [{error[3]:+.4f}, {error[4]:+.4f}, {error[5]:+.4f}]\n")
            self.log_file.write(f"  RR: [{error[6]:+.4f}, {error[7]:+.4f}, {error[8]:+.4f}]\n")
            self.log_file.write(f"  RL: [{error[9]:+.4f}, {error[10]:+.4f}, {error[11]:+.4f}]\n")

        # Joint velocities
        self.log_file.write(f"\n[Joint Velocities (SDK order)]\n")
        self.log_file.write(f"  FR: [{dof_vel[0]:+.3f}, {dof_vel[1]:+.3f}, {dof_vel[2]:+.3f}]\n")
        self.log_file.write(f"  FL: [{dof_vel[3]:+.3f}, {dof_vel[4]:+.3f}, {dof_vel[5]:+.3f}]\n")
        self.log_file.write(f"  RR: [{dof_vel[6]:+.3f}, {dof_vel[7]:+.3f}, {dof_vel[8]:+.3f}]\n")
        self.log_file.write(f"  RL: [{dof_vel[9]:+.3f}, {dof_vel[10]:+.3f}, {dof_vel[11]:+.3f}]\n")

        # Foot forces and contacts
        self.log_file.write(f"\n[Foot Forces & Contacts]\n")
        self.log_file.write(f"  Forces: FR={foot_force[0]:.1f}, FL={foot_force[1]:.1f}, RR={foot_force[2]:.1f}, RL={foot_force[3]:.1f}\n")
        if extra_debug and 'foot_contacts' in extra_debug:
            fc = extra_debug['foot_contacts']
            self.log_file.write(f"  Contacts: FR={fc[0]:.0f}, FL={fc[1]:.0f}, RR={fc[2]:.0f}, RL={fc[3]:.0f}\n")

        # Button states
        remote = state.wireless_remote
        buttons_raw = remote[2] | (remote[3] << 8)
        self.log_file.write(f"\n[Buttons]\n")
        self.log_file.write(f"  raw=0x{buttons_raw:04X}, pressed={self.button_pressed}\n")

        # Depth stats
        if depth_stats:
            self.log_file.write(f"\n[Depth Image]\n")
            self.log_file.write(f"  min={depth_stats['min']:.4f}, max={depth_stats['max']:.4f}, mean={depth_stats['mean']:.4f}\n")
            self.log_file.write(f"  (Expected range: [-0.5, 0.5])\n")

        # Extra debug info (raw actions, etc.)
        if extra_debug:
            if extra_debug.get('last_actions') is not None:
                actions = extra_debug['last_actions']
                self.log_file.write(f"\n[Last Actions (Training order, raw from policy)]\n")
                self.log_file.write(f"  FL: [{actions[0]:+.4f}, {actions[1]:+.4f}, {actions[2]:+.4f}]\n")
                self.log_file.write(f"  FR: [{actions[3]:+.4f}, {actions[4]:+.4f}, {actions[5]:+.4f}]\n")
                self.log_file.write(f"  RL: [{actions[6]:+.4f}, {actions[7]:+.4f}, {actions[8]:+.4f}]\n")
                self.log_file.write(f"  RR: [{actions[9]:+.4f}, {actions[10]:+.4f}, {actions[11]:+.4f}]\n")

        self.log_file.write("\n")
        self.log_file.flush()  # Ensure data is written

    def _close_logging(self):
        """Close the log file."""
        if self.log_file is not None:
            self.log_file.write("=" * 80 + "\n")
            self.log_file.write(f"Log ended: {datetime.now().isoformat()}\n")
            self.log_file.write(f"Total policy ticks: {self.policy_tick}\n")
            self.log_file.write("=" * 80 + "\n")
            self.log_file.close()
            print(f"Log saved with {self.policy_tick} entries")

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

        # Current button states (level detection for responsiveness)
        y_pressed = bool(buttons & BUTTON_Y)
        l1_pressed = bool(buttons & BUTTON_L1)
        r2_pressed = bool(buttons & BUTTON_R2)
        a_pressed = bool(buttons & BUTTON_A)
        b_pressed = bool(buttons & BUTTON_B)

        # Previous states
        y_was_pressed = bool(self.prev_buttons & BUTTON_Y)
        l1_was_pressed = bool(self.prev_buttons & BUTTON_L1)
        a_was_pressed = bool(self.prev_buttons & BUTTON_A)
        b_was_pressed = bool(self.prev_buttons & BUTTON_B)

        # Track hold times for responsive button detection
        current_time = time.time()
        HOLD_THRESHOLD = 0.1  # Trigger after holding 100ms

        for btn, pressed in [('Y', y_pressed), ('L1', l1_pressed)]:
            if pressed:
                if btn not in self.button_hold_time:
                    self.button_hold_time[btn] = current_time
            else:
                self.button_hold_time.pop(btn, None)

        # Rising edge OR held long enough triggers action
        y_trigger = (y_pressed and not y_was_pressed) or \
                    (y_pressed and self.button_hold_time.get('Y', current_time) <= current_time - HOLD_THRESHOLD)
        l1_trigger = (l1_pressed and not l1_was_pressed) or \
                     (l1_pressed and self.button_hold_time.get('L1', current_time) <= current_time - HOLD_THRESHOLD)

        # Clear hold time after triggering to prevent repeated triggers
        if y_trigger and 'Y' in self.button_hold_time:
            self.button_hold_time['Y'] = current_time + 1.0  # Prevent re-trigger for 1 second
        if l1_trigger and 'L1' in self.button_hold_time:
            self.button_hold_time['L1'] = current_time + 1.0

        self.button_pressed = {
            'Y': y_trigger,
            'L1': l1_trigger,
            'R2': r2_pressed,  # Level detection for emergency (always active when held)
            'A': a_pressed and not a_was_pressed,
            'B': b_pressed and not b_was_pressed,
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

        # Log comprehensive data for debugging
        depth_stats = {
            'min': float(depth.min()),
            'max': float(depth.max()),
            'mean': float(depth.mean())
        }
        extra_debug = {
            'ang_vel': ang_vel.tolist(),
            'roll': float(roll),
            'pitch': float(pitch),
            'dof_pos_sdk': dof_pos.tolist(),
            'dof_vel_sdk': dof_vel.tolist(),
            'foot_contacts': foot_contacts.tolist(),
            'last_actions': self.policy.last_actions.tolist() if self.policy else None,
        }
        self._log_sensor_data(depth_stats=depth_stats, policy_output=targets, extra_debug=extra_debug)

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

        # Close log file
        self._close_logging()

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
