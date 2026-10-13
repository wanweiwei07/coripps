"""Send the capper (the XH430 that opens / presses shut the tube cap) home.

Home is 180 deg on the servo dial: parked, waiting for the picker to place a
tube. Headless: no viewer, nothing to press.

    py -3.12 coripps/tool/home_capper.py
"""

import sys

from coripps.end_effectors.xh430_capper.exec import (XH430CapperExec,
                                                     DynamixelError)


def home():
    """Run the capper to its home angle. True when it arrived."""
    try:
        with XH430CapperExec() as capper:
            capper.set_opening('home')
            capper.status()
            return abs(capper.servo_deg() - capper.cfg.home_deg) <= 1.0
    except DynamixelError as exc:
        print(f'capper: {exc}')
        return False


if __name__ == '__main__':
    sys.exit(0 if home() else 1)
