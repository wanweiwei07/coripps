"""Open the XH430 gripper (on the picker) to its working open angle.

Headless: no viewer, nothing to press. Run it before taking a tube out of the
gripper or moving the arm by hand -- forcing the blades while they still grip
is what trips the servo's overload protection.

    py -3.12 coripps/tool/xh430_gripper_open.py
"""

import sys

from coripps.end_effectors.xh430_gripper.exec import (XH430GripperExec,
                                                      DynamixelError)


def open_gripper():
    """Open the gripper. True when it arrived."""
    try:
        with XH430GripperExec() as gripper:
            gripper.open()
            gripper.status()
            return abs(gripper.angle_deg() - gripper.cfg.open_deg) <= 1.0
    except DynamixelError as exc:
        print(f'gripper: {exc}')
        return False


if __name__ == '__main__':
    sys.exit(0 if open_gripper() else 1)
