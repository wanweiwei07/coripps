"""The XH430 cutter's cut (on the sampler), as the take makes it:
``XH430CutterExec.cut`` -- the last stretch slow, a step at a time (its
CUT_* in the cutter's exec.py), so a bench test is the robot's cut. Ends
deployed (``xh430_cutter_retract.py`` pulls it back in).

Headless: no viewer, nothing to press. Motion only -- the ultrasonic stays
off (``coripps/tool/contec/ultrasonic_cutter.py`` switches it).

    py -3.12 coripps/tool/leaf/xh430_cutter_cut.py
"""

import sys

from coripps.drivers.contec_dio import ContecDIOError
from coripps.end_effectors.xh430_cutter.exec import (XH430CutterExec,
                                                     DynamixelError)


def cut():
    """The cut. True when every pass reached the end of the travel."""
    try:
        with XH430CutterExec() as cutter:
            reached = cutter.cut()
            cutter.status()
            return all(abs(a - cutter.cfg.travel_deg) <= 1.0 for a in reached)
    except (DynamixelError, ContecDIOError) as exc:
        print(f'cutter: {exc}')
        return False


if __name__ == '__main__':
    sys.exit(0 if cut() else 1)
