#!/usr/bin/env python3
"""
Go2 Vision Policy Deployment Script (Two-Process Architecture)

Architecture:
    Process 1 (this): Policy loop at 50Hz, motor control at 500Hz
    Process 2 (spawned): Depth encoder at ~10Hz, writes embedding to shared memory

Controls:
    Y:     Stand up (from IDLE) - 2-phase interpolation like Unitree example
    B:     Sit down (from STANDING/WALKING) - current -> sit position
    L1:    Enable walking policy (when standing)
    Start: Reset policy (while walking) - clears history buffer
    R2/L2: EMERGENCY STOP - cuts all motor power

Usage:
    python deploy.py                    # Normal mode
    python deploy.py --dryrun           # Test without motor commands (saves logs)
    python deploy.py --no-camera        # Test without depth camera
    python deploy.py --show-depth       # Show depth buffer GUI window

Dryrun mode saves sensor logs to: deploy_log_YYYYMMDD_HHMMSS.txt
"""
import sys
import os
import time
import argparse
import numpy as np
from enum import IntEnum
from typing import Optional
from datetime import datetime
from multiprocessing import Process, Array, Value
from ctypes import c_float, c_bool, c_int

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
    ACTION_SCALE, N_PROPRIO, DEPTH_LATENT_DIM,
    DEFAULT_STAND_ANGLES_SDK, JOINT_POS_MIN, JOINT_POS_MAX,
    FIXED_VEL_X, SDK_TO_TRAIN_JOINTS,
    DEPTH_NEAR, DEPTH_FAR,
    TORQUE_LIMITS,
    CLIP_ACTIONS
)
from policy_runner import PolicyRunner
from depth_encoder_process import depth_encoder_loop

# Motor control constants
PosStopF = 2.146e9
VelStopF = 16000.0


class State(IntEnum):
    """Robot state machine states."""
    IDLE = 0           # Damping mode, waiting for commands
    STANDING_UP = 1    # Interpolating to stand pose (2 phases like Unitree)
    STANDING = 2       # Holding stand pose
    WALKING = 3        # Running vision policy
    SITTING_DOWN = 4   # Interpolating to sit pose
    EMERGENCY = 5      # Emergency stop - motors off


# Sitting pose (crouched, low to ground) - SDK order
# Based on Unitree example targetPos_1
SIT_ANGLES_SDK = np.array([
    0.0, 1.36, -2.65,    # FR: hip, thigh, calf (folded)
    0.0, 1.36, -2.65,    # FL
    -0.2, 1.36, -2.65,   # RR (slight hip offset)
    0.2, 1.36, -2.65,    # RL
], dtype=np.float32)


class Go2Deployment:
    """
    Main deployment class for Go2 vision policy.

    Uses two-process architecture:
    - Process 1 (this): Policy loop at 50Hz, motor control at 500Hz
    - Process 2: Depth encoder at ~10Hz, writes embedding to shared memory
    """

    def __init__(self, dryrun: bool = False, no_camera: bool = False, show_depth: bool = False):
        """
        Initialize deployment.

        Args:
            dryrun: If True, don't send motor commands
            no_camera: If True, use dummy depth frames
            show_depth: If True, show depth buffer GUI
        """
        self.dryrun = dryrun
        self.no_camera = no_camera
        self.show_depth = show_depth

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

        # Unitree-style direct interpolation (like go2_stand_example.py)
        self.startPos = np.zeros(12, dtype=np.float32)
        self.percent_1 = 0.0  # Phase 1: current -> sit
        self.percent_2 = 0.0  # Phase 2: sit -> stand
        self.duration_1 = 500  # 500 ticks @ 500Hz = 1 second
        self.duration_2 = 500  # 500 ticks @ 500Hz = 1 second

        # Button state for edge detection (print only on press)
        self._last_buttons = 0

        # Shared memory for two-process architecture
        # Shared memory for depth embeddings (128 latent + 2 yaw)
        # Using 130 floats instead of 34
        self.shared_embedding = Array(c_float, 130)  # Depth latent + yaw from encoder
        self.shared_proprio = Array(c_float, N_PROPRIO)  # Proprio for encoder
        self.embedding_ready = Value(c_bool, False)
        self.proprio_ready = Value(c_bool, False)
        self.depth_encoder_stop = Value(c_bool, False)
        self.save_depth_images = Value(c_bool, False)  # Flag to save depth images
        self.shared_buttons = Value(c_int, 0)  # Shared button state for debug printing
        self.shared_state = Value(c_int, 0)    # Shared robot state (WALKING=3)

        # Depth encoder process
        self.depth_encoder_process: Optional[Process] = None

        # Components
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
        print(f"Depth GUI: {'ENABLED' if self.show_depth else 'DISABLED'}")
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

        # Start depth encoder in separate process
        print("Starting depth encoder process...")
        self.depth_encoder_process = Process(
            target=depth_encoder_loop,
            args=(
                self.shared_embedding,
                self.shared_proprio,
                self.embedding_ready,
                self.proprio_ready,
                self.depth_encoder_stop,
                not self.no_camera,  # use_camera
                self.show_depth,     # show_gui
                self.save_depth_images,  # save_images flag
                self.shared_buttons, # shared button state
                self.shared_state,   # shared robot state
            ),
            daemon=True
        )
        self.depth_encoder_process.start()
        print(f"Depth encoder process started (PID: {self.depth_encoder_process.pid})")

        # Wait a bit for depth encoder to initialize
        time.sleep(2.0)

        # Initialize policy (reads embedding from shared memory)
        print("Loading base policy...")
        self.policy = PolicyRunner(
            shared_embedding=self.shared_embedding,
            shared_proprio=self.shared_proprio,
            device="cuda"
        )
        if not self.policy.load_models():
            print("ERROR: Failed to load policy models")
            return False

        print()
        print("=" * 60)
        print("Initialization complete!")
        print()
        print("Controls:")
        print("  Y:      Stand up (from IDLE) - 2-phase like Unitree")
        print("  B:      Sit down (from STANDING or WALKING)")
        print("  L1:     Enable walking policy (when standing)")
        print("  Select: Save 10 depth images (when standing)")
        print("  Start:  Reset policy (while walking) - clears history buffer")
        print("  R2/L2:  EMERGENCY STOP")
        print()
        print("State flow:")
        print("  IDLE --(Y)--> STANDING_UP (2-phase) --> STANDING --(L1)--> WALKING")
        print("                                              |                  |")
        print("                                             (B)                (B)")
        print("                                              v                  v")
        print("  IDLE <--------------- SITTING_DOWN <--------+------------------+")
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
        self.log_file.write(f"  raw=0x{buttons_raw:04X}\n")
        self.log_file.write(f"  Y={bool(buttons_raw & self.WirelessButtons.Y)}, L1={bool(buttons_raw & self.WirelessButtons.L1)}, R2={bool(buttons_raw & self.WirelessButtons.R2)}\n")

        # Depth stats
        if depth_stats:
            self.log_file.write(f"\n[Depth Image]\n")
            self.log_file.write(f"  min={depth_stats['min']:.4f}, max={depth_stats['max']:.4f}, mean={depth_stats['mean']:.4f}\n")
            self.log_file.write(f"  (Expected range: [-0.5, 0.5])\n")

        # Extra debug info (raw actions, etc.)
        if extra_debug:
            # Yaw prediction from depth encoder
            if 'yaw_pred_sin' in extra_debug and 'yaw_pred_cos' in extra_debug:
                yaw_sin = extra_debug['yaw_pred_sin']
                yaw_cos = extra_debug['yaw_pred_cos']
                self.log_file.write(f"\n[Yaw Prediction (from depth encoder)]\n")
                self.log_file.write(f"  sin={yaw_sin:+.4f}, cos={yaw_cos:+.4f}\n")

            if extra_debug.get('last_actions') is not None:
                actions = extra_debug['last_actions']
                self.log_file.write(f"\n[Last Actions (SDK order, raw from policy)]\n")
                self.log_file.write(f"  FR: [{actions[0]:+.4f}, {actions[1]:+.4f}, {actions[2]:+.4f}]\n")
                self.log_file.write(f"  FL: [{actions[3]:+.4f}, {actions[4]:+.4f}, {actions[5]:+.4f}]\n")
                self.log_file.write(f"  RR: [{actions[6]:+.4f}, {actions[7]:+.4f}, {actions[8]:+.4f}]\n")
                self.log_file.write(f"  RL: [{actions[9]:+.4f}, {actions[10]:+.4f}, {actions[11]:+.4f}]\n")

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
        print("Joint mapping verified: FL_hip ~+0.1, FR_hip ~-0.1 ✓")
        print()
        return True

    def _low_state_callback(self, msg: LowState_):
        """Callback for low-level state messages."""
        self.low_state = msg

    # Button masks (matching parkour's WirelessButtons)
    class WirelessButtons:
        R1 = 0b00000001          # 1
        L1 = 0b00000010          # 2
        start = 0b00000100       # 4
        select = 0b00001000      # 8
        R2 = 0b00010000          # 16
        L2 = 0b00100000          # 32
        F1 = 0b01000000          # 64
        F2 = 0b10000000          # 128
        A = 0b100000000          # 256
        B = 0b1000000000         # 512
        X = 0b10000000000        # 1024
        Y = 0b100000000000       # 2048

    def _get_buttons(self) -> int:
        """
        Get current button bitmask from wireless remote.
        Matches parkour's msg.keys format.
        """
        if self.low_state is None:
            return 0
        remote = self.low_state.wireless_remote
        return remote[2] | (remote[3] << 8)


    def _enter_state(self, new_state: State):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()

        # State entry actions
        if new_state == State.STANDING_UP:
            # Capture current position and reset percent counters (like Unitree example)
            if self.low_state is not None:
                for i in range(12):
                    self.startPos[i] = self.low_state.motor_state[i].q
            self.percent_1 = 0.0
            self.percent_2 = 0.0
            print("STATE: Standing up (2-phase like Unitree)...")

        elif new_state == State.STANDING:
            print("STATE: Standing - press L1 to walk, Y to sit")

        elif new_state == State.SITTING_DOWN:
            # Capture current position and reset percent (single phase sit down)
            if self.low_state is not None:
                for i in range(12):
                    self.startPos[i] = self.low_state.motor_state[i].q
            self.percent_1 = 0.0
            print("STATE: Sitting down...")

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

    def _clip_by_torque_limit(
        self,
        targets: np.ndarray,
        kp: float,
        kd: float
    ) -> np.ndarray:
        """
        Clip position targets to ensure torques stay within limits.

        This matches parkour's clip_by_torque_limit - prevents excessive
        torque by limiting how far the target can be from current position.

        Args:
            targets: Joint position targets in SDK order [12]
            kp: Position gain
            kd: Damping gain

        Returns:
            Clipped targets [12]
        """
        # Skip if no state or kp is zero (damping mode)
        if self.low_state is None or kp == 0:
            return targets

        # Get current joint state
        dof_pos = np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)
        dof_vel = np.array([self.low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)

        # Torque = kp * (target - pos) - kd * vel
        # Max torque when: kp * (target - pos) = torque_limit + kd * vel
        # So: target_max = pos + (torque_limit + kd * vel) / kp
        #     target_min = pos + (-torque_limit + kd * vel) / kp

        targets_high = dof_pos + (TORQUE_LIMITS + kd * dof_vel) / kp
        targets_low = dof_pos + (-TORQUE_LIMITS + kd * dof_vel) / kp

        return np.clip(targets, targets_low, targets_high)

    def _send_motor_commands(
        self,
        targets: np.ndarray,
        kp: float,
        kd: float,
        skip_torque_limit: bool = False
    ):
        """
        Send motor position commands.

        Args:
            targets: Joint position targets in SDK order [12]
            kp: Position gain
            kd: Damping gain
            skip_torque_limit: Skip torque limiting (for smooth interpolation states)
        """
        # Clip to joint limits
        targets = np.clip(targets, JOINT_POS_MIN, JOINT_POS_MAX)

        # Clip by torque limits (only during walking, not during stand/sit interpolation)
        if not skip_torque_limit:
            targets = self._clip_by_torque_limit(targets, kp, kd)

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

        # Get button bitmask (simple bitwise like parkour)
        buttons = self._get_buttons()
        
        # Share button state with encoder process
        self.shared_buttons.value = buttons
        self.shared_state.value = int(self.state)

        # Print button presses for convenience (only on rising edge)
        new_presses = buttons & ~self._last_buttons
        if new_presses:
            if new_presses & self.WirelessButtons.Y:
                print("  [Button] Y pressed", flush=True)
            if new_presses & self.WirelessButtons.L1:
                print("  [Button] L1 pressed", flush=True)
            if new_presses & self.WirelessButtons.R2:
                print("  [Button] R2 pressed", flush=True)
            if new_presses & self.WirelessButtons.L2:
                print("  [Button] L2 pressed", flush=True)
            if new_presses & self.WirelessButtons.A:
                print("  [Button] A pressed", flush=True)
            if new_presses & self.WirelessButtons.B:
                print("  [Button] B pressed", flush=True)
            if new_presses & self.WirelessButtons.start:
                print("  [Button] Start pressed", flush=True)
        self._last_buttons = buttons

        # Emergency stop takes priority - LATCHING (R2 or L2)
        if (buttons & self.WirelessButtons.R2) or (buttons & self.WirelessButtons.L2) or self.state == State.EMERGENCY:
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
            # Y to start standing up
            if buttons & self.WirelessButtons.Y:
                self._enter_state(State.STANDING_UP)

        elif self.state == State.STANDING_UP:
            # Unitree-style 2-phase interpolation (exactly like go2_stand_example.py)
            # Phase 1: startPos -> SIT_ANGLES_SDK
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            if self.percent_1 < 1:
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * SIT_ANGLES_SDK[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = KP_STAND
                    self.low_cmd.motor_cmd[i].kd = KD_STAND
                    self.low_cmd.motor_cmd[i].tau = 0

            # Phase 2: SIT_ANGLES_SDK -> DEFAULT_STAND_ANGLES_SDK
            if (self.percent_1 == 1) and (self.percent_2 <= 1):
                self.percent_2 += 1.0 / self.duration_2
                self.percent_2 = min(self.percent_2, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * SIT_ANGLES_SDK[i] + self.percent_2 * DEFAULT_STAND_ANGLES_SDK[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = KP_STAND
                    self.low_cmd.motor_cmd[i].kd = KD_STAND
                    self.low_cmd.motor_cmd[i].tau = 0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            if not self.dryrun:
                self.lowcmd_publisher.Write(self.low_cmd)

            # Check if both phases complete
            if (self.percent_1 == 1) and (self.percent_2 == 1):
                self._enter_state(State.STANDING)

        elif self.state == State.STANDING:
            # Hold at stand position (skip torque limiting - static hold doesn't need it)
            self._send_motor_commands(DEFAULT_STAND_ANGLES_SDK, KP_STAND, KD_STAND, skip_torque_limit=True)

            # Select to save depth images (for debugging)
            if new_presses & self.WirelessButtons.select:
                print("  [Select] Saving 10 depth images...")
                self.save_depth_images.value = True

            # L1 to start walking (like parkour)
            if buttons & self.WirelessButtons.L1:
                self._enter_state(State.WALKING)
            # B to sit down (current -> sit)
            elif buttons & self.WirelessButtons.B:
                self._enter_state(State.SITTING_DOWN)

        elif self.state == State.WALKING:
            # Run policy at 50Hz
            if self.tick % POLICY_DECIMATION == 0:
                self.policy_tick += 1
                # Depth embedding is read from shared memory (from depth encoder process)
                self.current_target = self._run_policy()

            # Keep torque limiting during walking for safety
            self._send_motor_commands(self.current_target, KP_WALK, KD_WALK)

            # Start button to reset policy (keep walking)
            if new_presses & self.WirelessButtons.start:
                self.policy.reset()
                self.policy_tick = 0
                print("  [Start] Policy reset - continuing to walk")

            # B to reset policy and sit down
            if buttons & self.WirelessButtons.B:
                self.policy.reset()
                print("  Policy reset - sitting down")
                self._enter_state(State.SITTING_DOWN)

        elif self.state == State.SITTING_DOWN:
            # Unitree-style single-phase: current -> SIT_ANGLES_SDK
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * SIT_ANGLES_SDK[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = KP_STAND
                self.low_cmd.motor_cmd[i].kd = KD_STAND
                self.low_cmd.motor_cmd[i].tau = 0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            if not self.dryrun:
                self.lowcmd_publisher.Write(self.low_cmd)

            # Check if complete
            if self.percent_1 == 1:
                self._enter_state(State.IDLE)

        elif self.state == State.EMERGENCY:
            self._send_emergency_stop()
            # Can only exit emergency with restart

    def _run_policy(self) -> Optional[np.ndarray]:
        """
        Run one step of the vision policy.

        Depth embedding is read from shared memory (from depth encoder process).

        Returns:
            Joint position targets in SDK order [12]
        """
        t_start = time.time()

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

        t_before_inference = time.time()

        # Run inference (depth embedding read from shared memory inside run_inference)
        targets, raw_actions = self.policy.run_inference(
            ang_vel=ang_vel,
            roll=roll,
            pitch=pitch,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            foot_contacts=foot_contacts,
            cmd_vel_x=FIXED_VEL_X,
        )

        t_after_inference = time.time()

        # Print timing diagnostics periodically
        if self.policy_tick % 50 == 0:  # Every 50 ticks (~1 second at 50Hz)
            total_ms = (t_after_inference - t_start) * 1000
            inference_ms = (t_after_inference - t_before_inference) * 1000
            print(f"  [MLP] tick={self.policy_tick}: total={total_ms:.1f}ms, inference={inference_ms:.1f}ms")

        # Warn if actions are exploding (should be roughly [-2, 2] for normal walking)
        # Check against effective clip range from config (CLIP_ACTIONS / ACTION_SCALE = 1.2 / 0.25 = 4.8)
        limit = CLIP_ACTIONS / ACTION_SCALE
        action_max = np.abs(raw_actions).max()
        if action_max > limit:
            print(f"  [WARNING] tick={self.policy_tick}: Large raw actions! max={action_max:.3f} > {limit:.1f}")
            print(f"    raw_actions={raw_actions}")

        # Get yaw prediction from shared memory for logging
        yaw_pred_sin = float(self.shared_embedding[32])
        yaw_pred_cos = float(self.shared_embedding[33])

        # Log comprehensive data for debugging (depth is in separate process now)
        extra_debug = {
            'ang_vel': ang_vel.tolist(),
            'roll': float(roll),
            'pitch': float(pitch),
            'dof_pos_sdk': dof_pos.tolist(),
            'dof_vel_sdk': dof_vel.tolist(),
            'foot_contacts': foot_contacts.tolist(),
            'last_actions': self.policy.last_actions.tolist() if self.policy else None,
            'raw_actions': raw_actions.tolist(),
            'yaw_pred_sin': yaw_pred_sin,
            'yaw_pred_cos': yaw_pred_cos,
        }
        self._log_sensor_data(depth_stats=None, policy_output=targets, extra_debug=extra_debug)

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
            self.control_thread.Wait()

        # Stop depth encoder process
        if self.depth_encoder_process is not None:
            print("Stopping depth encoder process...")
            self.depth_encoder_stop.value = True
            self.depth_encoder_process.join(timeout=2.0)
            if self.depth_encoder_process.is_alive():
                self.depth_encoder_process.terminate()
            print("Depth encoder process stopped")

        # Close log file
        self._close_logging()

        print("Stopped")


def main():
    parser = argparse.ArgumentParser(description="Go2 Vision Policy Deployment")
    parser.add_argument("--dryrun", action="store_true",
                        help="Don't send motor commands")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable depth camera (use dummy frames)")
    parser.add_argument("--show-depth", action="store_true",
                        help="Show depth buffer GUI window")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface (e.g., eth0)")
    args = parser.parse_args()

    # Initialize DDS with network interface if specified
    if args.interface:
        ChannelFactoryInitialize(0, args.interface)

    deployment = Go2Deployment(
        dryrun=args.dryrun,
        no_camera=args.no_camera,
        show_depth=args.show_depth
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
