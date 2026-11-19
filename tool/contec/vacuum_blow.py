"""Blow with the vacuum generator (CONTEC DIO-1616LN-USB) for BLOW_TIME,
then off -- e.g. a leaf stuck on the nozzle.

Headless, nothing to press. The ultrasonic is left as it is.

    py -3.12 coripps/tool/contec/vacuum_blow.py
"""

import time

from coripps.drivers.contec_dio import ContecDIO

BLOW_TIME = 1.0                 # s


def vacuum_blow(seconds=BLOW_TIME):
    """Blow ``seconds``, then off; the board's outputs after it."""
    with ContecDIO('DIO000') as dio:
        dio.vacuum_generator('blow')
        try:
            time.sleep(seconds)
        finally:
            dio.vacuum_generator('off')
        state = dio.state()
    print('state:', state)
    return state


if __name__ == '__main__':
    vacuum_blow()
