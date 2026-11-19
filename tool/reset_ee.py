"""Reset the end effectors after a run stopped half way: whatever state the
error left them in, back to where a run starts.

    1   ultrasonic off
    2   vacuum: blow BLOW_TIME (a leaf stuck on the nozzle), then off
    3   cutter: the blade retracted
    4   gripper (picker): open
    5   capper: home (where the picker places the tube)

Each step on its own: one failing (a servo not answering, ...) is reported
and the rest still run. The arms are not moved (``reset_robot.py``).

Headless, nothing to press.

    py -3.12 coripps/tool/reset_ee.py
"""

import sys
import time

from coripps.drivers.contec_dio import ContecDIO
from coripps.tool.leaf.xh430_cutter_retract import retract_cutter
from coripps.tool.tube.xh430_gripper_open import open_gripper
from coripps.end_effectors.xh430_capper.exec import XH430CapperExec

BLOW_TIME = 1.0                 # s


def ultrasonic_vacuum():
    """Ultrasonic off, blow BLOW_TIME, vacuum off."""
    with ContecDIO('DIO000') as dio:
        dio.ultrasonic_cutter(False)
        dio.vacuum_generator('blow')
        try:
            time.sleep(BLOW_TIME)
        finally:
            dio.vacuum_generator('off')
        state = dio.state()
    print('  board:', state)
    return state == {'vacuum_generator': 'off', 'ultrasonic_cutter': False}


def capper_home():
    with XH430CapperExec() as capper:
        capper.set_opening('home')
        capper.status()
    return True


STEPS = (
    ('ultrasonic off, vacuum blow then off', ultrasonic_vacuum),
    ('cutter retract', retract_cutter),
    ('gripper open', open_gripper),
    ('capper home', capper_home),
)


def reset():
    """Run every step; True when all of them succeeded."""
    failed = []
    for name, step in STEPS:
        print(f'== {name}', flush=True)
        try:
            ok = step()
        except Exception as exc:            # one step's fault: the rest run
            print(f'  {type(exc).__name__}: {exc}')
            ok = False
        if not ok:
            failed.append(name)
    print('reset: all done' if not failed else
          f"reset: FAILED -- {', '.join(failed)}")
    return not failed


if __name__ == '__main__':
    sys.exit(0 if reset() else 1)
