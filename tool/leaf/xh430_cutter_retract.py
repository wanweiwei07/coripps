"""Retract the XH430 cutter's blade (on the sampler): pull it back in.

Headless: no viewer, nothing to press. Motion only -- the ultrasonic stays
off (``coripps/tool/contec/ultrasonic_cutter.py`` switches it).

    py -3.12 coripps/tool/leaf/xh430_cutter_retract.py
"""

import sys

from coripps.drivers.contec_dio import ContecDIOError
from coripps.end_effectors.xh430_cutter.exec import (XH430CutterExec,
                                                     DynamixelError)


def retract_cutter():
    """Pull the blade back in. True when it arrived."""
    try:
        with XH430CutterExec() as cutter:
            cutter.retract()
            cutter.status()
            return abs(cutter.angle_deg()) <= 1.0
    except (DynamixelError, ContecDIOError) as exc:
        print(f'cutter: {exc}')
        return False


if __name__ == '__main__':
    sys.exit(0 if retract_cutter() else 1)
