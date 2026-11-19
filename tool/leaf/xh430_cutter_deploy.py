"""Deploy the XH430 cutter's blade (on the sampler): swing it fully out.

Headless: no viewer, nothing to press. Motion only -- the ultrasonic stays
off (``coripps/tool/contec/ultrasonic_cutter.py`` switches it). The cut the
take makes (slow, a step at a time) is ``xh430_cutter_cut.py``.

    py -3.12 coripps/tool/leaf/xh430_cutter_deploy.py
"""

import sys

from coripps.drivers.contec_dio import ContecDIOError
from coripps.end_effectors.xh430_cutter.exec import (XH430CutterExec,
                                                     DynamixelError)


def deploy_cutter():
    """Swing the blade out. True when it arrived."""
    try:
        with XH430CutterExec() as cutter:
            cutter.deploy()
            cutter.status()
            return abs(cutter.angle_deg() - cutter.cfg.travel_deg) <= 1.0
    except (DynamixelError, ContecDIOError) as exc:
        print(f'cutter: {exc}')
        return False


if __name__ == '__main__':
    sys.exit(0 if deploy_cutter() else 1)
