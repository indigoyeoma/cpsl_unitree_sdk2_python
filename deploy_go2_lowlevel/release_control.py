#!/usr/bin/env python3
"""
Re-enable high-level controller after low-level deployment.

This script returns control to the original Go2 controller (remote/app control)
after running deploy.py, allowing you to walk the dog back comfortably.

Usage:
    python release_control.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


def main():
    print("=" * 60)
    print("Go2 Control Release - Return to Normal Controller")
    print("=" * 60)
    print()

    # Initialize DDS
    print("Initializing DDS communication...")
    try:
        ChannelFactoryInitialize(0)
    except Exception as e:
        print(f"Failed to initialize DDS: {e}")
        sys.exit(1)

    time.sleep(0.5)

    # Initialize clients
    print("Connecting to robot...")
    try:
        sc = SportClient()
        sc.SetTimeout(5.0)
        sc.Init()

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
    except Exception as e:
        print(f"Failed to initialize clients: {e}")
        sys.exit(1)

    # Check current mode
    print("Checking current mode...")
    status, result = msc.CheckMode()
    current_mode = result.get('name', 'none')
    print(f"  Current mode: {current_mode if current_mode else 'none (low-level)'}")

    # If already in a high-level mode, we're done
    if current_mode:
        print(f"\nAlready in high-level mode '{current_mode}'.")
        print("Remote/app control should be working.")
        return

    # Select normal mode to return control
    print("\nEnabling high-level control mode...")
    try:
        # First, try to get the robot into a safe standing position
        # by selecting the normal mode
        status, result = msc.SelectMode("normal")
        if status == 0:
            print("  Successfully switched to 'normal' mode")
        else:
            print(f"  SelectMode returned status {status}: {result}")
            # Try alternative approach - just release and let default take over
            print("  Trying StandUp command...")
            sc.StandUp()

        time.sleep(1.0)

        # Verify
        status, result = msc.CheckMode()
        new_mode = result.get('name', 'none')
        print(f"  Current mode now: {new_mode if new_mode else 'none'}")

    except Exception as e:
        print(f"  Error enabling mode: {e}")
        print("  The robot may need to be restarted if this fails.")

    print()
    print("=" * 60)
    print("Done! Remote/app control should now be restored.")
    print("You can use the remote controller to walk the dog.")
    print("=" * 60)


if __name__ == "__main__":
    main()
