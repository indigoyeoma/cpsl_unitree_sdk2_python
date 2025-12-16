#!/usr/bin/env python3
"""
Go2 Vision Policy Deployment with Button Controls (Parkour-style)

Button Controls:
    Y      - Stand up (from idle) or exit policy and stand (from walking)
    L1     - Start walking policy (from standing)
    L2/R2  - EMERGENCY STOP (from any state)

State Machine:
    IDLE (damping) --[Y]--> STANDING --[L1]--> WALKING --[Y]--> STANDING
                                                      |
                                             [L2/R2] EMERGENCY STOP

Usage:
    python deploy.py                    # Default: 0.3 m/s
    python deploy.py --command_vx 0.5   # Faster walking
    python deploy.py --use_dummy_camera # Test without real camera
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
    Go2 controller with button-triggered state machine (Parkour-style).

    States:
    - Phase 0 (IDLE): Damping mode, waiting for Y button
    - Phase 1 (SIT_TO_STAND): Smooth transition from sitting to standing
    - Phase 2 (STANDING): Hold standing position, waiting for L1
    - Phase 3 (WALKING): Run vision policy, Y exits to standing
    - Phase 4 (STAND_TO_SIT): Smooth transition back to sitting
    - Phase 5 (EMERGENCY): Emergency stop (L2/R2)

    Button Controls:
    - Y: Stand up (from IDLE) or exit policy (from WALKING)
    - L1: Start walking policy (from STANDING)
    - L2/R2: Emergency stop (from any state)
    """

    # Phase constants
    PHASE_IDLE = 0
    PHASE_SIT_TO_STAND = 1
    PHASE_STANDING = 2
    PHASE_WALKING = 3
    PHASE_STAND_TO_SIT = 4
    PHASE_EMERGENCY = 5
    PHASE_DEBUG = 6  # Debug mode - prints data, no motor control

    # Button constants (from SDK wireless_controller.py)
    # Byte 2 (data1): R1, L1, Start, Select, R2, L2, F1, F3
    # Byte 3 (data2): A, B, X, Y, Up, Right, Down, Left
    # We combine as: data1 | (data2 << 8)
    BUTTON_R1 = 1 << 0       # data1 bit 0
    BUTTON_L1 = 1 << 1       # data1 bit 1
    BUTTON_START = 1 << 2   # data1 bit 2
    BUTTON_SELECT = 1 << 3  # data1 bit 3
    BUTTON_R2 = 1 << 4       # data1 bit 4
    BUTTON_L2 = 1 << 5       # data1 bit 5
    BUTTON_F1 = 1 << 6       # data1 bit 6
    BUTTON_F3 = 1 << 7       # data1 bit 7
    BUTTON_A = 1 << 8        # data2 bit 0
    BUTTON_B = 1 << 9        # data2 bit 1
    BUTTON_X = 1 << 10       # data2 bit 2
    BUTTON_Y = 1 << 11       # data2 bit 3
    BUTTON_UP = 1 << 12      # data2 bit 4
    BUTTON_RIGHT = 1 << 13   # data2 bit 5
    BUTTON_DOWN = 1 << 14    # data2 bit 6
    BUTTON_LEFT = 1 << 15    # data2 bit 7

    def __init__(self, policy: JITPolicyRunner, camera, config: DeployConfig,
                 skip_standup: bool = False):
        self.policy = policy
        self.camera = camera
        self.config = config
        self.crc = CRC()
        self.skip_standup = skip_standup

        # Control state
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.running = False
        self.shutdown_requested = False
        self.emergency_stop = False  # L2/R2 emergency stop flag
        self.torque_clip_enabled = True  # Enable torque clipping for safety

        # Button state (for edge detection)
        self.prev_buttons = 0
        self.button_y_pressed = False
        self.button_l1_pressed = False

        # Observation buffers
        self.obs_history = deque(maxlen=config.history_len)
        self.action_history = deque(maxlen=2)
        self.last_action = np.zeros(12, dtype=np.float32)

        # Goal-based navigation (matching training)
        self.current_goal = np.array([config.goal_distance, 0.0], dtype=np.float32)  # [x, y] in world frame
        self.next_goal = np.array([config.next_goal_distance, 0.0], dtype=np.float32)
        self.robot_position = np.zeros(2, dtype=np.float32)  # [x, y] in world frame
        self.robot_yaw = 0.0  # Current heading in radians
        self.delta_yaw = 0.0  # Yaw error to current goal
        self.delta_next_yaw = 0.0  # Yaw error to next goal

        # Distance tracking
        self.start_position = None
        self.distance_traveled = 0.0

        # Phase control (button-triggered state machine)
        self.phase = self.PHASE_IDLE
        self.dt = 0.002  # 500Hz control

        # Target positions (SDK joint order: FR, FL, RR, RL)
        # Sitting pose (relaxed, motors can be soft)
        self._sit_pos = np.array([
            0.0, 1.4, -2.7,   # FR: hip, thigh, calf
            0.0, 1.4, -2.7,   # FL
            0.0, 1.4, -2.7,   # RR
            0.0, 1.4, -2.7,   # RL
        ])

        # Standing pose (already in SDK order)
        self._stand_pos = config.default_joint_angles.copy()

        # Start position (captured from robot)
        self.start_pos = np.zeros(12)
        self.first_run = True

        # Phase durations and progress
        self.duration_sit_to_stand = 1000  # 2 seconds @ 500Hz
        self.duration_stand_to_sit = 1000  # 2 seconds @ 500Hz

        self.percent_sit_to_stand = 0.0
        self.percent_stand_to_sit = 0.0

        # Walking phase control
        self.walk_startup_counter = 0

        # Timing
        self.last_policy_time = 0
        self.policy_dt = config.control_dt
        self.control_step = 0

        print(f"Controller initialized")
        print(f"  Command velocity: {self.config.command_vx} m/s")
        print(f"  Sit→Stand duration: {self.duration_sit_to_stand * self.dt:.1f}s")
        print(f"  Stand→Sit duration: {self.duration_stand_to_sit * self.dt:.1f}s")
        print(f"\nButton Controls:")
        print(f"  Y     - Stand up / Exit policy")
        print(f"  L1    - Start walking policy")
        print(f"  A     - DEBUG MODE (print sensors, no motor control)")
        print(f"  L2/R2 - EMERGENCY STOP")

        # Debug mode counter
        self.debug_counter = 0

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

        # Start phase depends on skip_standup flag
        if self.skip_standup:
            self.phase = self.PHASE_STANDING  # Skip to standing
            self.start_pos = self._stand_pos.copy()
            self.first_run = False
            print("\n✓ Starting in standing pose (like simulator)")
        else:
            self.phase = self.PHASE_IDLE  # Start in idle, waiting for Y

        # Start camera
        print("\n" + "=" * 60)
        print("Starting Camera")
        print("=" * 60)
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
            if self.phase in [self.PHASE_STANDING, self.PHASE_WALKING]:
                self.phase = self.PHASE_STAND_TO_SIT
                self.percent_stand_to_sit = 0.0
            else:
                self.running = False

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

    def _get_button_state(self):
        """Get current button state from wireless remote (SDK format)."""
        if self.low_state is None:
            return 0

        # wireless_remote format (from SDK wireless_controller.py):
        # Byte 2 (data1): R1, L1, Start, Select, R2, L2, F1, F3
        # Byte 3 (data2): A, B, X, Y, Up, Right, Down, Left
        wireless_remote = self.low_state.wireless_remote
        data1 = wireless_remote[2]  # Buttons byte 1
        data2 = wireless_remote[3]  # Buttons byte 2
        buttons = data1 | (data2 << 8)
        return buttons

    def _check_button_pressed(self, button_mask):
        """Check if button was just pressed (edge detection)."""
        buttons = self._get_button_state()
        # Rising edge: button is pressed now but wasn't before
        pressed = (buttons & button_mask) and not (self.prev_buttons & button_mask)
        return pressed

    def _update_button_state(self):
        """Update previous button state for edge detection."""
        self.prev_buttons = self._get_button_state()

    def _check_emergency_stop(self):
        """Check L2/R2 buttons for emergency stop (from parkour)."""
        buttons = self._get_button_state()
        return (buttons & self.BUTTON_L2) or (buttons & self.BUTTON_R2)

    def _check_y_pressed(self):
        """Check if Y button was just pressed."""
        return self._check_button_pressed(self.BUTTON_Y)

    def _check_l1_pressed(self):
        """Check if L1 button was just pressed."""
        return self._check_button_pressed(self.BUTTON_L1)

    def _check_a_pressed(self):
        """Check if A button was just pressed."""
        return self._check_button_pressed(self.BUTTON_A)

    def _emergency_motor_shutdown(self):
        """Immediately disable all motors (from parkour)."""
        print("\n" + "!" * 70)
        print("!!! EMERGENCY STOP - L2/R2 PRESSED !!!")
        print("!" * 70)

        # Send damping mode command to all motors
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x00  # Disable motor
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 3.0  # Some damping for soft landing
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

        self.emergency_stop = True
        self.phase = self.PHASE_EMERGENCY
        self.running = False
        print("Motors disabled. Robot should go limp.")
        print("CATCH THE ROBOT if needed!")

    def _control_loop(self):
        """Main control loop running at 500Hz."""
        if not self.running or self.low_state is None:
            return

        # Check for emergency stop (L2/R2) - highest priority
        if self._check_emergency_stop():
            self._emergency_motor_shutdown()
            return

        # Capture start position on first run
        if self.first_run:
            self.start_pos = self._get_joint_positions()
            self.first_run = False
            print(f"\n✓ Captured start position")
            self._print_controls()

        # Handle button inputs based on current phase
        self._handle_button_inputs()

        # Execute current phase
        if self.phase == self.PHASE_IDLE:
            self._phase_idle()
        elif self.phase == self.PHASE_SIT_TO_STAND:
            self._phase_sit_to_stand()
        elif self.phase == self.PHASE_STANDING:
            self._phase_standing()
        elif self.phase == self.PHASE_WALKING:
            self._phase_walk()
        elif self.phase == self.PHASE_STAND_TO_SIT:
            self._phase_stand_to_sit()
        elif self.phase == self.PHASE_DEBUG:
            self._phase_debug()
        elif self.phase == self.PHASE_EMERGENCY:
            pass  # Motors already disabled

        # Update button state for edge detection
        self._update_button_state()

        # Send motor commands (except in IDLE and EMERGENCY)
        if self.phase not in [self.PHASE_IDLE, self.PHASE_EMERGENCY]:
            self._send_command()

        self.control_step += 1

    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("BUTTON CONTROLS:")
        print("  Y     - Stand up / Exit policy and stand")
        print("  L1    - Start walking policy (from standing)")
        print("  L2/R2 - EMERGENCY STOP")
        print("=" * 60)
        print("\n>>> Press Y to STAND UP <<<\n")

    def _handle_button_inputs(self):
        """Handle button inputs for state transitions."""
        # Y button: Stand up (from IDLE) or exit policy (from WALKING)
        if self._check_y_pressed():
            if self.phase == self.PHASE_IDLE:
                print("\n[Y] Standing up...")
                self.phase = self.PHASE_SIT_TO_STAND
                self.percent_sit_to_stand = 0.0
            elif self.phase == self.PHASE_WALKING:
                print("\n[Y] Exiting policy, returning to standing...")
                self.phase = self.PHASE_STANDING
                self._reset_walk_state()
                print("\n>>> Press L1 to WALK or Y to SIT DOWN <<<\n")
            elif self.phase == self.PHASE_STANDING:
                print("\n[Y] Sitting down...")
                self.phase = self.PHASE_STAND_TO_SIT
                self.percent_stand_to_sit = 0.0

        # L1 button: Start walking policy (from STANDING)
        if self._check_l1_pressed():
            if self.phase == self.PHASE_STANDING:
                print("\n[L1] Starting walking policy...")
                self.phase = self.PHASE_WALKING
                self.walk_startup_counter = 0
                self._reset_walk_state()
                print("\n" + "=" * 60)
                print("WALKING - Vision policy active")
                print("  Press Y to exit to standing")
                print("  Press L2/R2 for EMERGENCY STOP")
                print("=" * 60 + "\n")

        # A button: Enter/exit debug mode (from STANDING or WALKING)
        if self._check_a_pressed():
            if self.phase == self.PHASE_STANDING:
                print("\n[A] Entering DEBUG MODE (from standing)...")
                self.phase = self.PHASE_DEBUG
                self.debug_counter = 0
                open("debug_output.txt", "w").close()  # Clear file
                print("\n" + "=" * 60)
                print("DEBUG MODE - Printing sensor data, holding standing pose")
                print("  Output saved to: debug_output.txt")
                print("  Press A to exit, Y to sit down")
                print("=" * 60 + "\n")
            elif self.phase == self.PHASE_WALKING:
                print("\n[A] Entering DEBUG MODE (from walking)...")
                self.phase = self.PHASE_DEBUG
                self.debug_counter = 0
                open("debug_output.txt", "w").close()  # Clear file
                print("\n" + "=" * 60)
                print("DEBUG MODE - Preserving walk state for inspection")
                print("  Output saved to: debug_output.txt")
                print("  last_action will show actual values from walking")
                print("  Press A to exit, Y to sit down")
                print("=" * 60 + "\n")
            elif self.phase == self.PHASE_DEBUG:
                print("\n[A] Exiting debug mode...")
                self.phase = self.PHASE_STANDING
                self._reset_walk_state()  # Reset when EXITING debug
                print("\n>>> Press L1 for DEBUG or Y to SIT DOWN <<<\n")

    def _reset_walk_state(self):
        """Reset walking state variables."""
        self.obs_history.clear()
        self.action_history.clear()
        self.last_action = np.zeros(12, dtype=np.float32)
        self.start_position = None
        self.distance_traveled = 0.0
        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        # Reset initial heading so it gets set fresh at start of walking
        if hasattr(self, 'initial_heading'):
            delattr(self, 'initial_heading')

    def _phase_idle(self):
        """Phase IDLE: Damping mode, waiting for Y button."""
        # Just hold in damping - don't send active commands
        # Print reminder occasionally
        if self.control_step % 2500 == 0:  # Every 5 seconds
            print(">>> Press Y to STAND UP <<<", end='\r')

    def _phase_sit_to_stand(self):
        """Phase SIT_TO_STAND: Smooth transition from sitting to standing."""
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

        # Move to STANDING phase when complete
        if self.percent_sit_to_stand >= 1.0:
            print(f"  Sit→Stand: 100% ✓" + " " * 20)
            self.phase = self.PHASE_STANDING
            print("\n✓ Standing pose reached")
            print("\n>>> Press L1 to WALK or Y to SIT DOWN <<<\n")

    def _phase_standing(self):
        """Phase STANDING: Hold standing position, waiting for L1."""
        # Hold standing position
        self.target_positions = self._stand_pos.copy()

        # Print reminder occasionally
        if self.control_step % 2500 == 0:  # Every 5 seconds
            print(">>> Press L1 to WALK, A for DEBUG, or Y to SIT DOWN <<<", end='\r')

    def _phase_debug(self):
        """Phase DEBUG: Print all sensor data while holding standing pose."""
        # Always hold standing position in debug mode
        self.target_positions = self._stand_pos.copy()

        self.debug_counter += 1

        # Only print every 0.5 seconds (250 iterations at 500Hz)
        if self.debug_counter % 250 != 0:
            return

        # Open file for appending debug output
        debug_file = open("debug_output.txt", "a")

        def debug_print(msg=""):
            print(msg)
            debug_file.write(msg + "\n")

        # Get IMU data
        imu = self.low_state.imu_state
        quat = imu.quaternion  # [w, x, y, z]

        # Compute roll, pitch, yaw
        roll = np.arctan2(
            2 * (quat[0] * quat[1] + quat[2] * quat[3]),
            1 - 2 * (quat[1]**2 + quat[2]**2)
        )
        pitch = np.arcsin(np.clip(
            2 * (quat[0] * quat[2] - quat[3] * quat[1]),
            -1, 1
        ))
        yaw = np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2]**2 + quat[3]**2)
        )

        # Angular velocity
        ang_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])

        # Joint positions (SDK order: FR, FL, RR, RL)
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()

        # Foot forces
        foot_forces = [
            self.low_state.foot_force[0],  # FR
            self.low_state.foot_force[1],  # FL
            self.low_state.foot_force[2],  # RR
            self.low_state.foot_force[3],  # RL
        ]

        # Get depth image
        depth_image = self.camera.get_depth()

        # Build observation and run policy (but don't apply)
        obs = self._build_observation()

        # Get proprio part (first 53 elements)
        proprio = obs[:self.config.n_proprio]

        # Run policy to get action (but don't apply)
        # NO CONVERSION NEEDED - policy outputs in SDK order
        action = self.policy.get_action(depth_image, obs)

        # Print everything (and save to file)
        import datetime
        debug_print("\n" + "=" * 70)
        debug_print(f"DEBUG OUTPUT - {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        debug_print("=" * 70)

        debug_print("\n--- IMU ---")
        debug_print(f"  Quaternion [w,x,y,z]: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        debug_print(f"  Roll/Pitch/Yaw (deg): [{np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f}]")
        debug_print(f"  Angular Vel (rad/s):  [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")

        debug_print("\n--- Joint Positions (SDK order: FR, FL, RR, RL) ---")
        debug_print(f"  FR: [{joint_pos[0]:.3f}, {joint_pos[1]:.3f}, {joint_pos[2]:.3f}]")
        debug_print(f"  FL: [{joint_pos[3]:.3f}, {joint_pos[4]:.3f}, {joint_pos[5]:.3f}]")
        debug_print(f"  RR: [{joint_pos[6]:.3f}, {joint_pos[7]:.3f}, {joint_pos[8]:.3f}]")
        debug_print(f"  RL: [{joint_pos[9]:.3f}, {joint_pos[10]:.3f}, {joint_pos[11]:.3f}]")

        debug_print("\n--- Default Joint Angles (SDK order) ---")
        d = self.config.default_joint_angles
        debug_print(f"  FR: [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]")
        debug_print(f"  FL: [{d[3]:.3f}, {d[4]:.3f}, {d[5]:.3f}]")
        debug_print(f"  RR: [{d[6]:.3f}, {d[7]:.3f}, {d[8]:.3f}]")
        debug_print(f"  RL: [{d[9]:.3f}, {d[10]:.3f}, {d[11]:.3f}]")

        debug_print("\n--- Joint Position Error (pos - default) ---")
        err = joint_pos - self.config.default_joint_angles
        debug_print(f"  FR: [{err[0]:.3f}, {err[1]:.3f}, {err[2]:.3f}]")
        debug_print(f"  FL: [{err[3]:.3f}, {err[4]:.3f}, {err[5]:.3f}]")
        debug_print(f"  RR: [{err[6]:.3f}, {err[7]:.3f}, {err[8]:.3f}]")
        debug_print(f"  RL: [{err[9]:.3f}, {err[10]:.3f}, {err[11]:.3f}]")

        debug_print("\n--- Foot Forces (FR, FL, RR, RL) ---")
        debug_print(f"  [{foot_forces[0]:.1f}, {foot_forces[1]:.1f}, {foot_forces[2]:.1f}, {foot_forces[3]:.1f}]")

        debug_print("\n--- Depth Image ---")
        debug_print(f"  Shape: {depth_image.shape} (expected: 58x87)")
        debug_print(f"  Min/Max/Mean: [{depth_image.min():.3f}, {depth_image.max():.3f}, {depth_image.mean():.3f}]")
        debug_print(f"  Expected: [-0.5 (near/0.3m), +0.5 (far/3.0m)]")
        # Sample pixel values
        debug_print(f"  Sample pixels (row 29, cols 0,20,40,60,80):")
        mid_row = depth_image[29, :]
        debug_print(f"    [{mid_row[0]:+.2f}, {mid_row[20]:+.2f}, {mid_row[40]:+.2f}, {mid_row[60]:+.2f}, {mid_row[80]:+.2f}]")
        # Save depth image
        np.save("debug_depth.npy", depth_image)
        debug_print(f"  [Saved to: debug_depth.npy]")

        debug_print("\n--- Policy Action (SDK order: FR,FL,RR,RL - no conversion needed) ---")
        debug_print(f"  FR: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
        debug_print(f"  FL: [{action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}]")
        debug_print(f"  RR: [{action[6]:.3f}, {action[7]:.3f}, {action[8]:.3f}]")
        debug_print(f"  RL: [{action[9]:.3f}, {action[10]:.3f}, {action[11]:.3f}]")
        debug_print(f"  Range: [{action.min():.3f}, {action.max():.3f}]")

        debug_print("\n--- Proprio Observation (first 53 dims) ---")
        debug_print(f"  [0:3]   ang_vel*0.25:     [{proprio[0]:.3f}, {proprio[1]:.3f}, {proprio[2]:.3f}]")
        debug_print(f"  [3:5]   roll,pitch:       [{proprio[3]:.3f}, {proprio[4]:.3f}]")
        debug_print(f"  [5:8]   yaw (mask,d,dn):  [{proprio[5]:.3f}, {proprio[6]:.3f}, {proprio[7]:.3f}]")
        debug_print(f"  [8:10]  cmd (masked):     [{proprio[8]:.3f}, {proprio[9]:.3f}] (should be 0,0)")
        debug_print(f"  [10]    cmd (vx):         {proprio[10]:.3f} (should be {self.config.command_vx})")
        debug_print(f"  [11:13] env_class:        [{proprio[11]:.3f}, {proprio[12]:.3f}]")
        debug_print(f"  [13:25] dof_pos:          min={proprio[13:25].min():.3f}, max={proprio[13:25].max():.3f}")
        debug_print(f"  [25:37] dof_vel*0.05:     min={proprio[25:37].min():.3f}, max={proprio[25:37].max():.3f}")
        debug_print(f"  [49:53] contacts:         [{proprio[49]:.1f}, {proprio[50]:.1f}, {proprio[51]:.1f}, {proprio[52]:.1f}]")

        debug_print("\n--- Last Action (SDK order: FR,FL,RR,RL - policy output stored) ---")
        la = self.last_action
        debug_print(f"  FR: [{la[0]:.3f}, {la[1]:.3f}, {la[2]:.3f}]")
        debug_print(f"  FL: [{la[3]:.3f}, {la[4]:.3f}, {la[5]:.3f}]")
        debug_print(f"  RR: [{la[6]:.3f}, {la[7]:.3f}, {la[8]:.3f}]")
        debug_print(f"  RL: [{la[9]:.3f}, {la[10]:.3f}, {la[11]:.3f}]")
        debug_print(f"  (zeros = entered from standing, non-zero = entered from walking)")

        debug_print("\n--- Target Position (SDK order, no conversion needed) ---")
        target = self.config.default_joint_angles + action * self.config.action_scale
        debug_print(f"  FR: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        debug_print(f"  FL: [{target[3]:.3f}, {target[4]:.3f}, {target[5]:.3f}]")
        debug_print(f"  RR: [{target[6]:.3f}, {target[7]:.3f}, {target[8]:.3f}]")
        debug_print(f"  RL: [{target[9]:.3f}, {target[10]:.3f}, {target[11]:.3f}]")

        debug_print("\n>>> Press A to exit DEBUG, Y to sit down <<<")
        debug_print("=" * 70)

        # Close debug file
        debug_file.close()

    def _phase_walk(self):
        """Phase WALKING: Run vision policy for walking."""
        current_time = time.time()

        # Set start position on first walk iteration
        if self.start_position is None:
            self.start_position = self.robot_position.copy()
            # Clear walk log file at start
            open("walk_log.txt", "w").close()
            print("  Logging to: walk_log.txt")

        # Update distance traveled
        self.distance_traveled = np.linalg.norm(self.robot_position - self.start_position)

        # Run policy at 50Hz
        if current_time - self.last_policy_time >= self.policy_dt:
            self.last_policy_time = current_time

            # Get depth image and build observation
            depth_image = self.camera.get_depth()
            obs = self._build_observation()

            # Run policy inference
            action = self.policy.get_action(depth_image, obs)

            # Clip actions (must match training: clip to clip_actions/action_scale before scaling)
            # Training clips raw policy output to ±(1.2/0.25) = ±4.8, then scales by 0.25
            clip_limit = self.config.clip_actions / self.config.action_scale  # 4.8
            action = np.clip(action, -clip_limit, clip_limit)

            # NO CONVERSION NEEDED!
            # Training reindexes BOTH observations AND actions to SDK order before applying.
            # So policy outputs in SDK order (FR, FL, RR, RL), same as our SDK motors.

            # Scale action and add to default pose (both in SDK order)
            target_delta = action * self.config.action_scale
            target_pos = self.config.default_joint_angles + target_delta

            # Apply joint limits
            target_pos = JointLimits.clip_joints(target_pos)

            self.target_positions = target_pos

            # Update history - store action for observation (already in SDK order)
            self.last_action = action
            self.action_history.append(action)

            # Log data every 10 policy steps (5 times per second)
            self.walk_startup_counter += 1
            if self.walk_startup_counter % 10 == 0:
                self._log_walk_data(depth_image, obs, action, target_pos)

            # Print summary every second
            if self.walk_startup_counter % 50 == 0:
                print(f"  Action min/max: {action.min():.2f}/{action.max():.2f} | "
                      f"Depth min/max: {depth_image.min():.2f}/{depth_image.max():.2f} | "
                      f"{self.distance_traveled:.2f}m")

    def _log_walk_data(self, depth_image, obs, action, target_pos):
        """Log comprehensive walking data to file for analysis."""
        import datetime

        with open("walk_log.txt", "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"WALK LOG - {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Step {self.walk_startup_counter}\n")
            f.write(f"{'='*70}\n")

            # ===== IMU DATA =====
            imu = self.low_state.imu_state
            quat = imu.quaternion
            roll = np.arctan2(
                2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                1 - 2 * (quat[1]**2 + quat[2]**2)
            )
            pitch = np.arcsin(np.clip(
                2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1
            ))
            yaw = np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2]**2 + quat[3]**2)
            )
            ang_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]])
            accel = np.array([imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2]])

            f.write(f"\n--- IMU ---\n")
            f.write(f"  Quaternion [w,x,y,z]: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n")
            f.write(f"  Roll/Pitch/Yaw (deg): [{np.degrees(roll):.2f}, {np.degrees(pitch):.2f}, {np.degrees(yaw):.2f}]\n")
            f.write(f"  Roll/Pitch/Yaw (rad): [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]\n")
            f.write(f"  Angular Vel (rad/s): [{ang_vel[0]:.4f}, {ang_vel[1]:.4f}, {ang_vel[2]:.4f}]\n")
            f.write(f"  Accelerometer (m/s2): [{accel[0]:.4f}, {accel[1]:.4f}, {accel[2]:.4f}]\n")

            # ===== JOINT POSITIONS (SDK order) =====
            joint_pos = self._get_joint_positions()
            joint_vel = self._get_joint_velocities()
            default_pos = self.config.default_joint_angles

            f.write(f"\n--- Joint Positions (SDK: FR,FL,RR,RL) ---\n")
            f.write(f"  FR: [{joint_pos[0]:.4f}, {joint_pos[1]:.4f}, {joint_pos[2]:.4f}]\n")
            f.write(f"  FL: [{joint_pos[3]:.4f}, {joint_pos[4]:.4f}, {joint_pos[5]:.4f}]\n")
            f.write(f"  RR: [{joint_pos[6]:.4f}, {joint_pos[7]:.4f}, {joint_pos[8]:.4f}]\n")
            f.write(f"  RL: [{joint_pos[9]:.4f}, {joint_pos[10]:.4f}, {joint_pos[11]:.4f}]\n")

            f.write(f"\n--- Joint Velocities (SDK: FR,FL,RR,RL) ---\n")
            f.write(f"  FR: [{joint_vel[0]:.4f}, {joint_vel[1]:.4f}, {joint_vel[2]:.4f}]\n")
            f.write(f"  FL: [{joint_vel[3]:.4f}, {joint_vel[4]:.4f}, {joint_vel[5]:.4f}]\n")
            f.write(f"  RR: [{joint_vel[6]:.4f}, {joint_vel[7]:.4f}, {joint_vel[8]:.4f}]\n")
            f.write(f"  RL: [{joint_vel[9]:.4f}, {joint_vel[10]:.4f}, {joint_vel[11]:.4f}]\n")

            # Position error from default
            pos_err = joint_pos - default_pos
            f.write(f"\n--- Position Error (pos - default) ---\n")
            f.write(f"  FR: [{pos_err[0]:.4f}, {pos_err[1]:.4f}, {pos_err[2]:.4f}]\n")
            f.write(f"  FL: [{pos_err[3]:.4f}, {pos_err[4]:.4f}, {pos_err[5]:.4f}]\n")
            f.write(f"  RR: [{pos_err[6]:.4f}, {pos_err[7]:.4f}, {pos_err[8]:.4f}]\n")
            f.write(f"  RL: [{pos_err[9]:.4f}, {pos_err[10]:.4f}, {pos_err[11]:.4f}]\n")

            # Tracking error (target - actual)
            track_err = target_pos - joint_pos
            f.write(f"\n--- Tracking Error (target - actual) ---\n")
            f.write(f"  FR: [{track_err[0]:.4f}, {track_err[1]:.4f}, {track_err[2]:.4f}]\n")
            f.write(f"  FL: [{track_err[3]:.4f}, {track_err[4]:.4f}, {track_err[5]:.4f}]\n")
            f.write(f"  RR: [{track_err[6]:.4f}, {track_err[7]:.4f}, {track_err[8]:.4f}]\n")
            f.write(f"  RL: [{track_err[9]:.4f}, {track_err[10]:.4f}, {track_err[11]:.4f}]\n")
            f.write(f"  Max tracking error: {np.abs(track_err).max():.4f}\n")

            # ===== FOOT FORCES =====
            foot_forces = [self.low_state.foot_force[i] for i in range(4)]
            f.write(f"\n--- Foot Forces (FR,FL,RR,RL) ---\n")
            f.write(f"  Raw: [{foot_forces[0]:.1f}, {foot_forces[1]:.1f}, {foot_forces[2]:.1f}, {foot_forces[3]:.1f}]\n")
            contacts = [1 if f > 20 else 0 for f in foot_forces]
            f.write(f"  Contact (>20): [{contacts[0]}, {contacts[1]}, {contacts[2]}, {contacts[3]}]\n")

            # ===== DEPTH IMAGE =====
            f.write(f"\n--- Depth Image ---\n")
            f.write(f"  Shape: {depth_image.shape} (expected: 58x87)\n")
            f.write(f"  Min/Max/Mean/Std: [{depth_image.min():.4f}, {depth_image.max():.4f}, {depth_image.mean():.4f}, {depth_image.std():.4f}]\n")
            f.write(f"  Expected range: [-0.5 (near/0.3m), +0.5 (far/3.0m)]\n")
            f.write(f"  Interpretation: -0.5=CLOSE (floor), +0.5=FAR (nothing ahead)\n")

            # ===== TOP vs BOTTOM ANALYSIS (Key sanity check!) =====
            # Image is 58 rows: row 0 = top (should be FAR), row 57 = bottom (should be CLOSE/floor)
            h = depth_image.shape[0]  # 58
            top_rows = depth_image[:10, :]      # Top 10 rows (sky/far)
            mid_rows = depth_image[20:38, :]    # Middle rows (ahead)
            bottom_rows = depth_image[-10:, :]  # Bottom 10 rows (floor)

            f.write(f"\n  === TOP vs BOTTOM ANALYSIS (sanity check) ===\n")
            f.write(f"  TOP 10 rows (row 0-9, should be FAR/+0.5 if nothing ahead):\n")
            f.write(f"    Min/Max/Mean: [{top_rows.min():.3f}, {top_rows.max():.3f}, {top_rows.mean():.3f}]\n")
            f.write(f"  MIDDLE rows (row 20-37, what's directly ahead):\n")
            f.write(f"    Min/Max/Mean: [{mid_rows.min():.3f}, {mid_rows.max():.3f}, {mid_rows.mean():.3f}]\n")
            f.write(f"  BOTTOM 10 rows (row 48-57, should be CLOSE/-0.5 = floor):\n")
            f.write(f"    Min/Max/Mean: [{bottom_rows.min():.3f}, {bottom_rows.max():.3f}, {bottom_rows.mean():.3f}]\n")

            # Check if floor detection makes sense
            if bottom_rows.mean() < top_rows.mean():
                f.write(f"  ✓ GOOD: Bottom is closer than top (floor detected)\n")
            else:
                f.write(f"  ✗ WARNING: Bottom is NOT closer than top! Camera may be inverted or broken\n")

            # Estimate floor distance from bottom row mean
            # Normalized value to meters: depth_m = (normalized + 0.5) * (3.0 - 0.3) + 0.3
            bottom_mean_m = (bottom_rows.mean() + 0.5) * (3.0 - 0.3) + 0.3
            top_mean_m = (top_rows.mean() + 0.5) * (3.0 - 0.3) + 0.3
            f.write(f"  Estimated distances:\n")
            f.write(f"    Bottom (floor): ~{bottom_mean_m:.2f}m (expected: 0.3-0.8m for standing robot)\n")
            f.write(f"    Top (ahead):    ~{top_mean_m:.2f}m (expected: 2-3m if nothing ahead)\n")

            # Histogram bins
            hist, _ = np.histogram(depth_image.flatten(), bins=5, range=(-0.5, 0.5))
            f.write(f"\n  Histogram [-0.5→0.5]: {hist.tolist()}\n")

            # Center region stats (what robot sees directly ahead)
            center_h, center_w = depth_image.shape[0]//2, depth_image.shape[1]//2
            center_region = depth_image[center_h-5:center_h+5, center_w-10:center_w+10]
            f.write(f"  Center region (10x20) mean: {center_region.mean():.4f}\n")

            # Sample actual pixel values - downsampled view of the depth image
            f.write(f"\n  Depth image sample (every 10th row, every 15th col):\n")
            f.write(f"  (row 0=TOP/far, row 50=BOTTOM/floor)\n")
            for row_idx in range(0, depth_image.shape[0], 10):
                row_vals = depth_image[row_idx, ::15]  # Every 15th column
                row_str = " ".join([f"{v:+.2f}" for v in row_vals])
                f.write(f"    Row {row_idx:2d}: [{row_str}]\n")

            # Row-wise statistics
            f.write(f"\n  Row-wise means (top→bottom, should increase from - to +... wait no):\n")
            f.write(f"  (Actually: top=far=+, bottom=close=-, so should DECREASE top→bottom)\n")
            row_means = [depth_image[i, :].mean() for i in range(0, depth_image.shape[0], 10)]
            f.write(f"    {[f'{m:+.3f}' for m in row_means]}\n")

            # Check for potential issues
            num_at_min = np.sum(depth_image <= -0.49)
            num_at_max = np.sum(depth_image >= 0.49)
            num_zero = np.sum(np.abs(depth_image) < 0.01)
            f.write(f"\n  Potential issues:\n")
            f.write(f"    Pixels at min (-0.5/close): {num_at_min} ({100*num_at_min/depth_image.size:.1f}%)\n")
            f.write(f"    Pixels at max (+0.5/far):   {num_at_max} ({100*num_at_max/depth_image.size:.1f}%)\n")
            f.write(f"    Pixels near zero (1.65m):   {num_zero} ({100*num_zero/depth_image.size:.1f}%)\n")

            # Save full depth image to file on first log
            if self.walk_startup_counter == 10:
                np.save("depth_image_sample.npy", depth_image)
                f.write(f"  [Saved full depth to: depth_image_sample.npy]\n")

            # ===== ACTION (SDK order - no conversion needed) =====
            f.write(f"\n--- Action (SDK order: FR,FL,RR,RL) ---\n")
            f.write(f"  FR: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]\n")
            f.write(f"  FL: [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]\n")
            f.write(f"  RR: [{action[6]:.4f}, {action[7]:.4f}, {action[8]:.4f}]\n")
            f.write(f"  RL: [{action[9]:.4f}, {action[10]:.4f}, {action[11]:.4f}]\n")
            f.write(f"  Range: [{action.min():.4f}, {action.max():.4f}]\n")

            # ===== TARGET POSITIONS =====
            f.write(f"\n--- Target Position (SDK order) ---\n")
            f.write(f"  FR: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]\n")
            f.write(f"  FL: [{target_pos[3]:.4f}, {target_pos[4]:.4f}, {target_pos[5]:.4f}]\n")
            f.write(f"  RR: [{target_pos[6]:.4f}, {target_pos[7]:.4f}, {target_pos[8]:.4f}]\n")
            f.write(f"  RL: [{target_pos[9]:.4f}, {target_pos[10]:.4f}, {target_pos[11]:.4f}]\n")

            # ===== DEFAULT POSITIONS (for reference) =====
            f.write(f"\n--- Default Position (SDK order) ---\n")
            f.write(f"  FR: [{default_pos[0]:.4f}, {default_pos[1]:.4f}, {default_pos[2]:.4f}]\n")
            f.write(f"  FL: [{default_pos[3]:.4f}, {default_pos[4]:.4f}, {default_pos[5]:.4f}]\n")
            f.write(f"  RR: [{default_pos[6]:.4f}, {default_pos[7]:.4f}, {default_pos[8]:.4f}]\n")
            f.write(f"  RL: [{default_pos[9]:.4f}, {default_pos[10]:.4f}, {default_pos[11]:.4f}]\n")

            # ===== FULL PROPRIO OBSERVATION (53 dims) =====
            proprio = obs[:self.config.n_proprio]
            f.write(f"\n--- Full Proprio Observation (53 dims) ---\n")
            f.write(f"  [0:3]   ang_vel*0.25:    [{proprio[0]:.4f}, {proprio[1]:.4f}, {proprio[2]:.4f}]\n")
            f.write(f"  [3:5]   roll,pitch:      [{proprio[3]:.4f}, {proprio[4]:.4f}]\n")
            f.write(f"  [5]     delta_yaw_mask:  {proprio[5]:.4f}\n")
            f.write(f"  [6]     delta_yaw:       {proprio[6]:.4f}\n")
            f.write(f"  [7]     delta_next_yaw:  {proprio[7]:.4f}\n")
            f.write(f"  [8:10]  cmd (masked):    [{proprio[8]:.4f}, {proprio[9]:.4f}] (should be 0,0)\n")
            f.write(f"  [10]    cmd (vx):        {proprio[10]:.4f} (should be {self.config.command_vx})\n")
            f.write(f"  [11:13] env_class:       [{proprio[11]:.4f}, {proprio[12]:.4f}]\n")
            f.write(f"  [13:25] dof_pos (12):    [{', '.join([f'{x:.3f}' for x in proprio[13:25]])}]\n")
            f.write(f"  [25:37] dof_vel*0.05:    [{', '.join([f'{x:.3f}' for x in proprio[25:37]])}]\n")
            f.write(f"  [37:49] last_action:     [{', '.join([f'{x:.3f}' for x in proprio[37:49]])}]\n")
            f.write(f"  [49:53] contacts:        [{proprio[49]:.2f}, {proprio[50]:.2f}, {proprio[51]:.2f}, {proprio[52]:.2f}]\n")

            # ===== LAST ACTION (SDK order - same as policy output) =====
            la = self.last_action
            f.write(f"\n--- Last Action (SDK order: FR,FL,RR,RL - policy output) ---\n")
            f.write(f"  FR: [{la[0]:.4f}, {la[1]:.4f}, {la[2]:.4f}]\n")
            f.write(f"  FL: [{la[3]:.4f}, {la[4]:.4f}, {la[5]:.4f}]\n")
            f.write(f"  RR: [{la[6]:.4f}, {la[7]:.4f}, {la[8]:.4f}]\n")
            f.write(f"  RL: [{la[9]:.4f}, {la[10]:.4f}, {la[11]:.4f}]\n")

            # ===== NAVIGATION/GOAL DATA =====
            f.write(f"\n--- Navigation ---\n")
            f.write(f"  Robot position: [{self.robot_position[0]:.4f}, {self.robot_position[1]:.4f}]\n")
            f.write(f"  Robot yaw (rad): {self.robot_yaw:.4f}\n")
            f.write(f"  Current goal: [{self.current_goal[0]:.4f}, {self.current_goal[1]:.4f}]\n")
            f.write(f"  Delta yaw: {self.delta_yaw:.4f}\n")
            f.write(f"  Delta next yaw: {self.delta_next_yaw:.4f}\n")
            f.write(f"  Distance traveled: {self.distance_traveled:.4f}\n")

            # ===== CONFIG VALUES (for reference) =====
            f.write(f"\n--- Config ---\n")
            f.write(f"  action_scale: {self.config.action_scale}\n")
            f.write(f"  clip_actions: {self.config.clip_actions}\n")
            f.write(f"  kp_walk: {self.config.kp_walk}\n")
            f.write(f"  kd_walk: {self.config.kd_walk}\n")
            f.write(f"  command_vx: {self.config.command_vx}\n")

    def _phase_stand_to_sit(self):
        """Phase STAND_TO_SIT: Safe sit down from standing."""
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

        # Return to IDLE when complete
        if self.percent_stand_to_sit >= 1.0:
            print(f"  Stand→Sit: 100% ✓" + " " * 20)
            print("\n✓ Sit position reached")
            self.phase = self.PHASE_IDLE
            # Update start position for next stand-up
            self.start_pos = self._get_joint_positions()
            print("\n>>> Press Y to STAND UP again <<<\n")

    def _send_command(self):
        """Send low-level motor commands with torque clipping (from parkour)."""
        if not hasattr(self, 'target_positions'):
            return

        # Use training gains ONLY for walking (must match sim2real)
        # Use strong gains for standing/transitions (need stiffness to hold position)
        if self.phase == self.PHASE_WALKING:
            kp = self.config.kp_walk   # 25.0 (training gains)
            kd = self.config.kd_walk   # 0.6
        else:
            kp = self.config.kp_stand  # 70.0 (strong for standing)
            kd = self.config.kd_stand  # 3.0

        # Get current state for torque clipping
        target_pos = self.target_positions.copy()

        # Apply torque clipping during walking phase to prevent motor damage
        if self.torque_clip_enabled and self.phase == self.PHASE_WALKING:
            current_pos = self._get_joint_positions()
            current_vel = self._get_joint_velocities()
            target_pos = JointLimits.clip_by_torque_limit(
                target_pos, current_pos, current_vel, kp, kd
            )

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(target_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _update_goals_and_yaw(self):
        """Update goals and compute delta_yaw for straight-line walking.

        Goal direction is FIXED at initial heading (yaw at start of walking).
        This gives the policy a yaw error signal when the robot drifts.
        """
        # Get current robot yaw from IMU
        imu = self.low_state.imu_state
        quat = imu.quaternion  # [w, x, y, z]

        # Compute yaw from quaternion (heading angle)
        self.robot_yaw = np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2]**2 + quat[3]**2)
        )

        # Set initial heading on first call (reference direction for walking)
        if not hasattr(self, 'initial_heading'):
            self.initial_heading = self.robot_yaw
            print(f"[NAV] Initial heading set to {np.degrees(self.initial_heading):.1f}°")

        # Update robot position by integrating velocity in world frame
        dt = self.policy_dt
        vel_forward = self.config.command_vx * 0.8

        dx = vel_forward * np.cos(self.robot_yaw) * dt
        dy = vel_forward * np.sin(self.robot_yaw) * dt
        self.robot_position += np.array([dx, dy], dtype=np.float32)

        # Goal direction is FIXED along initial heading (not rotating with robot!)
        fixed_forward_dir = np.array([np.cos(self.initial_heading), np.sin(self.initial_heading)])

        # Project robot position onto initial heading axis to get forward progress
        forward_progress = np.dot(self.robot_position, fixed_forward_dir)

        # Place goal ahead along fixed direction
        self.current_goal = fixed_forward_dir * (forward_progress + self.config.goal_distance)
        self.next_goal = fixed_forward_dir * (forward_progress + self.config.next_goal_distance)

        # Compute target yaw (direction from robot to goal)
        target_pos_rel = self.current_goal - self.robot_position
        next_target_pos_rel = self.next_goal - self.robot_position

        norm = np.linalg.norm(target_pos_rel) + 1e-5
        target_yaw = np.arctan2(target_pos_rel[1] / norm, target_pos_rel[0] / norm)

        norm_next = np.linalg.norm(next_target_pos_rel) + 1e-5
        next_target_yaw = np.arctan2(next_target_pos_rel[1] / norm_next, next_target_pos_rel[0] / norm_next)

        # Compute delta yaw (yaw error to goal) - now NON-ZERO when robot drifts!
        self.delta_yaw = target_yaw - self.robot_yaw
        self.delta_next_yaw = next_target_yaw - self.robot_yaw

        # Wrap angles to [-pi, pi]
        self.delta_yaw = np.arctan2(np.sin(self.delta_yaw), np.cos(self.delta_yaw))
        self.delta_next_yaw = np.arctan2(np.sin(self.delta_next_yaw), np.cos(self.delta_next_yaw))

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from real sensors."""
        # Update goals and compute delta_yaw (matching training)
        self._update_goals_and_yaw()

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

        # Joint states (SDK order = FR,FL,RR,RL = training observation order after reindex)
        joint_pos = self._get_joint_positions()  # SDK order
        joint_vel = self._get_joint_velocities()  # SDK order

        # Use SDK order directly - it matches training observation order!
        dof_pos = joint_pos - self.config.default_joint_angles  # Both in SDK order
        last_action = self.last_action if len(self.action_history) > 0 else np.zeros(12)

        # Contacts (SDK: FR=0, FL=1, RR=2, RL=3)
        # Training observation order (after reindex_feet): FR, FL, RR, RL
        contacts = np.array([
            self.low_state.foot_force[0] > 20,  # FR
            self.low_state.foot_force[1] > 20,  # FL
            self.low_state.foot_force[2] > 20,  # RR
            self.low_state.foot_force[3] > 20,  # RL
        ]).astype(np.float32) - 0.5

        # Scales
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05

        # Build proprio (53 dims)
        # Training observation order from legged_robot.py lines 393-407:
        # ang_vel(3), imu_obs(2), 0*delta_yaw(1), delta_yaw(1), delta_next_yaw(1),
        # 0*commands[:2](2), commands[0:1](1), env_class(2), dof_pos(12), dof_vel(12), last_action(12), contacts(4)
        proprio = np.concatenate([
            ang_vel * ang_vel_scale,       # 3  [0:3]
            [roll, pitch],                  # 2  [3:5]
            [0.0],                          # 0*delta_yaw (masked) [5]
            [self.delta_yaw],               # delta_yaw [6]
            [self.delta_next_yaw],          # delta_next_yaw [7]
            [0.0, 0.0],                     # 0*commands[:2] (MASKED vx,vy) [8:10]
            [self.config.command_vx],       # commands[0:1] = vx [10] - THIS IS THE ACTUAL COMMAND!
            [1.0, 0.0],                     # env_class flags [11:13]
            dof_pos * dof_pos_scale,        # 12 [13:25]
            joint_vel * dof_vel_scale,      # 12 [25:37] (SDK order = observation order)
            last_action,                    # 12 [37:49]
            contacts,                       # 4  [49:53]
        ]).astype(np.float32)

        # Update history (mask yaw before appending, matching training!)
        proprio_for_history = proprio.copy()
        proprio_for_history[6:8] = 0  # Mask delta_yaw and delta_next_yaw in history
        self.obs_history.append(proprio_for_history)
        while len(self.obs_history) < self.config.history_len:
            self.obs_history.appendleft(proprio_for_history.copy())

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
    parser.add_argument('--command_vx', type=float, default=0.3,
                        help='Forward velocity goal in m/s (default: 0.3)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    parser.add_argument('--use_dummy_camera', action='store_true',
                        help='Use dummy camera for testing')
    parser.add_argument('--network_interface', type=str, default=None,
                        help='Network interface for DDS')
    parser.add_argument('--skip_standup', action='store_true',
                        help='Skip sit-to-stand transition, start in standing pose (like simulator)')
    parser.add_argument('--rotate_camera', action='store_true',
                        help='Rotate camera 180 degrees (for inverted mounting like parkour)')
    parser.add_argument('--no_torque_clip', action='store_true',
                        help='Disable torque clipping (not recommended)')
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
        rotate_180=args.rotate_camera,  # For inverted camera mounting
    )
    if args.rotate_camera:
        print("✓ Camera rotation enabled (180°)")

    # Setup config
    config = DeployConfig()
    config.command_vx = args.command_vx

    # Create controller
    controller = Go2VisionController(policy, camera, config, skip_standup=args.skip_standup)

    # Disable torque clipping if requested (not recommended!)
    if args.no_torque_clip:
        controller.torque_clip_enabled = False
        print("⚠️  WARNING: Torque clipping disabled - motor damage possible!")

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
